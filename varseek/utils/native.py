"""Locate, build and cache the compiled helpers that ship as C++ source with varseek.

varseek is distributed as a pure-Python package, so the C++ tools under ``varseek/cpp``
are compiled on demand on the machine that runs them and cached in a user cache
directory. Nothing here is required at install time: callers that cannot build fall back
to their Python implementation.

htslib is discovered in this order, first hit wins:

1. ``VARSEEK_HTSLIB_INCLUDE`` / ``VARSEEK_HTSLIB_LIB`` environment variables
2. ``pkg-config --cflags --libs htslib``
3. the active conda/virtualenv prefix (``$CONDA_PREFIX``, ``sys.prefix``)
4. common system prefixes (``/usr``, ``/usr/local``, Homebrew)

Note that the htslib bundled inside pysam is deliberately *not* used. It is a CPython
extension module (``libchtslib.cpython-*.so``) that references ``Py*`` symbols and links
vendored dependencies through a private rpath, so a standalone executable cannot be linked
against it. When no real htslib is present, callers fall back to their Python
implementation instead (``bam_to_vcf(..., engine="python")``).
"""

import hashlib
import logging
import os
import shutil
import subprocess
import sys
import sysconfig
import tempfile
from pathlib import Path

logger = logging.getLogger(__name__)

CPP_DIR = Path(__file__).resolve().parent.parent / "cpp"

#: Programs that can be built from ``varseek/cpp``, mapped to their source file.
NATIVE_PROGRAMS = {"bam2vcf": "bam2vcf.cpp"}


class NativeBuildError(RuntimeError):
    """Raised when a bundled C++ helper cannot be located or compiled."""


# ---------------------------------------------------------------------------
# cache location
# ---------------------------------------------------------------------------
def cache_dir():
    """Directory holding compiled helpers, honouring VARSEEK_CACHE_DIR and XDG_CACHE_HOME."""
    override = os.environ.get("VARSEEK_CACHE_DIR")
    if override:
        return Path(override).expanduser() / "bin"
    xdg = os.environ.get("XDG_CACHE_HOME")
    base = Path(xdg).expanduser() if xdg else Path.home() / ".cache"
    return base / "varseek" / "bin"


# ---------------------------------------------------------------------------
# htslib discovery
# ---------------------------------------------------------------------------
def _lib_names(libdir):
    """htslib shared/static library candidates inside libdir, best first."""
    candidates = ["libhts.so", "libhts.dylib", "libhts.so.3", "libhts.3.dylib", "libhts.a"]
    return [Path(libdir) / name for name in candidates]


def _probe_prefix(prefix):
    """Return (include_dir, lib_dir) if prefix holds both htslib headers and a library."""
    prefix = Path(prefix)
    inc = prefix / "include"
    if not (inc / "htslib" / "sam.h").is_file():
        return None
    for libdir in (prefix / "lib", prefix / "lib64"):
        if any(p.is_file() for p in _lib_names(libdir)):
            return (str(inc), str(libdir))
    return None


def _pkg_config_htslib():
    """Ask pkg-config for htslib flags. Returns (cflags, libs) or None."""
    if not shutil.which("pkg-config"):
        return None
    try:
        cflags = subprocess.run(
            ["pkg-config", "--cflags", "htslib"], capture_output=True, text=True, check=True
        ).stdout.split()
        libs = subprocess.run(
            ["pkg-config", "--libs", "htslib"], capture_output=True, text=True, check=True
        ).stdout.split()
    except (subprocess.CalledProcessError, OSError):
        return None
    return (cflags, libs)


def find_htslib():
    """Locate htslib and return the compiler/linker flags needed to use it.

    Returns a dict with ``cflags``, ``libs`` and ``source`` (which mechanism matched).
    Raises NativeBuildError when nothing usable is found.
    """
    env_inc = os.environ.get("VARSEEK_HTSLIB_INCLUDE")
    env_lib = os.environ.get("VARSEEK_HTSLIB_LIB")
    if env_inc or env_lib:
        cflags = [f"-I{env_inc}"] if env_inc else []
        libs = []
        if env_lib:
            libs += [f"-L{env_lib}", f"-Wl,-rpath,{env_lib}"]
        libs += ["-lhts"]
        return {"cflags": cflags, "libs": libs, "source": "VARSEEK_HTSLIB_* environment"}

    pc = _pkg_config_htslib()
    if pc:
        cflags, libs = pc
        # pkg-config rarely emits an rpath, so a non-standard prefix would link but not run.
        for flag in list(libs):
            if flag.startswith("-L"):
                libs.append(f"-Wl,-rpath,{flag[2:]}")
        return {"cflags": cflags, "libs": libs, "source": "pkg-config"}

    prefixes = []
    for var in ("CONDA_PREFIX", "VIRTUAL_ENV"):
        if os.environ.get(var):
            prefixes.append(os.environ[var])
    prefixes += [sys.prefix, sys.base_prefix, "/usr/local", "/opt/homebrew", "/usr"]
    seen = set()
    for prefix in prefixes:
        if not prefix or prefix in seen:
            continue
        seen.add(prefix)
        hit = _probe_prefix(prefix)
        if hit:
            inc, libdir = hit
            return {
                "cflags": [f"-I{inc}"],
                "libs": [f"-L{libdir}", f"-Wl,-rpath,{libdir}", "-lhts"],
                "source": f"prefix {prefix}",
            }

    raise NativeBuildError(
        "could not find htslib (headers plus a linkable libhts). Install it with "
        "`conda install -c bioconda htslib`, `apt install libhts-dev` or `brew install htslib`, "
        "or point varseek at an existing copy with VARSEEK_HTSLIB_INCLUDE and VARSEEK_HTSLIB_LIB. "
        "The htslib inside pysam cannot be used: it is a CPython extension module, not a "
        "linkable library. Without htslib, the Python implementation is used instead."
    )


def find_compiler():
    """Return a C++ compiler command, preferring CXX then the interpreter's own compiler."""
    candidates = []
    if os.environ.get("CXX"):
        candidates.append(os.environ["CXX"])
    cxx = sysconfig.get_config_var("CXX")
    if cxx:
        candidates.append(cxx.split()[0])
    candidates += ["g++", "clang++", "c++"]
    for cand in candidates:
        exe = shutil.which(cand)
        if exe:
            return exe
    raise NativeBuildError("no C++ compiler found. Install g++ or clang++, or set CXX.")


# ---------------------------------------------------------------------------
# build
# ---------------------------------------------------------------------------
def _extra_flags():
    flags = ["-O3", "-std=c++17", "-fno-plt"] if sys.platform != "darwin" else ["-O3", "-std=c++17"]
    if os.environ.get("VARSEEK_NATIVE_MARCH"):
        # Off by default: build hosts and compute nodes are often different CPUs.
        flags.append(f"-march={os.environ['VARSEEK_NATIVE_MARCH']}")
    extra = os.environ.get("VARSEEK_NATIVE_CXXFLAGS")
    if extra:
        flags += extra.split()
    return flags


def _fingerprint(program, source, compiler, cflags, libs):
    h = hashlib.sha256()
    h.update(source.read_bytes())
    for part in [program, compiler, sys.platform, *cflags, *libs, *_extra_flags()]:
        h.update(str(part).encode())
    try:
        ver = subprocess.run([compiler, "--version"], capture_output=True, text=True, check=False).stdout
        h.update(ver.encode())
    except OSError:
        pass
    return h.hexdigest()[:16]


def build(program="bam2vcf", force=False, quiet=True):
    """Compile `program` from varseek/cpp and return the path to the cached binary.

    The binary is keyed by a fingerprint of the source, compiler and htslib flags, so an
    edited source or a changed environment produces a fresh build rather than a stale hit.
    """
    if program not in NATIVE_PROGRAMS:
        raise NativeBuildError(f"unknown native program '{program}'")
    source = CPP_DIR / NATIVE_PROGRAMS[program]
    if not source.is_file():
        raise NativeBuildError(f"source file '{source}' is missing from the installed package")

    compiler = find_compiler()
    hts = find_htslib()
    fp = _fingerprint(program, source, compiler, hts["cflags"], hts["libs"])
    outdir = cache_dir()
    binary = outdir / f"{program}-{fp}"
    if binary.is_file() and os.access(binary, os.X_OK) and not force:
        return str(binary)

    outdir.mkdir(parents=True, exist_ok=True)
    cmd = [compiler, *_extra_flags(), *hts["cflags"], str(source), *hts["libs"], "-lz", "-lpthread", "-lm", "-o"]

    # Compile to a private temp path and rename, so concurrent builds cannot observe a
    # half-written binary.
    with tempfile.TemporaryDirectory(dir=str(outdir)) as tmp:
        tmp_out = Path(tmp) / program
        proc = subprocess.run(cmd + [str(tmp_out)], capture_output=True, text=True)
        if proc.returncode != 0:
            raise NativeBuildError(
                f"compiling {source.name} failed (htslib from {hts['source']}).\n"
                f"command: {' '.join(cmd + [str(tmp_out)])}\n"
                f"{proc.stderr.strip()}"
            )
        if proc.stderr.strip() and not quiet:
            logger.debug("compiler warnings for %s:\n%s", source.name, proc.stderr.strip())
        os.chmod(tmp_out, 0o755)
        os.replace(tmp_out, binary)

    logger.info("built %s at %s (htslib from %s)", program, binary, hts["source"])
    return str(binary)


def program_path(program="bam2vcf", build_if_missing=True):
    """Path to a usable `program` binary.

    Checks the ``VARSEEK_<PROGRAM>`` environment override, then ``PATH``, then the
    build cache, compiling on demand unless build_if_missing is False.
    """
    override = os.environ.get(f"VARSEEK_{program.upper()}")
    if override:
        if not (os.path.isfile(override) and os.access(override, os.X_OK)):
            raise NativeBuildError(f"VARSEEK_{program.upper()} is set to '{override}', which is not executable")
        return override
    on_path = shutil.which(program)
    if on_path:
        return on_path
    if not build_if_missing:
        raise NativeBuildError(f"'{program}' is not built yet")
    return build(program)


def native_available(program="bam2vcf", build_if_missing=False):
    """True if `program` can be used right now (optionally building it first)."""
    try:
        program_path(program, build_if_missing=build_if_missing)
        return True
    except NativeBuildError:
        return False


def diagnostics():
    """Human-readable report of what the native build machinery can see."""
    lines = [f"varseek native helpers  (cache: {cache_dir()})"]
    try:
        lines.append(f"  compiler: {find_compiler()}")
    except NativeBuildError as exc:
        lines.append(f"  compiler: NOT FOUND -- {exc}")
    try:
        hts = find_htslib()
        lines.append(f"  htslib:   {hts['source']}")
        lines.append(f"    cflags: {' '.join(hts['cflags'])}")
        lines.append(f"    libs:   {' '.join(hts['libs'])}")
    except NativeBuildError as exc:
        lines.append(f"  htslib:   NOT FOUND -- {exc}")
    for program in NATIVE_PROGRAMS:
        try:
            lines.append(f"  {program}: {program_path(program, build_if_missing=True)}")
        except NativeBuildError as exc:
            lines.append(f"  {program}: unavailable -- {exc}")
    return "\n".join(lines)


if __name__ == "__main__":
    print(diagnostics())
