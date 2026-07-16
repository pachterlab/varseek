"""Reusable pydantic types and validation helpers for varseek.

This module centralizes the parameter validation that used to live in the
per-function ``validate_input_*`` helpers. The building blocks here are meant to
be attached directly to function signatures and validated with pydantic's
``@validate_call`` decorator, e.g.::

    from varseek.utils import validate_call, vk_config, FastaFile, Odd1to63

    @validate_call(config=vk_config)
    def build(sequences: FastaFile, k: Odd1to63 = 51, ...):
        ...

Design notes
------------
* The path types validate *structure* (the value is a string/Path with an
  accepted extension), NOT filesystem state. Existence / non-existence and any
  cross-field logic (``w < k``, "if sequences is a list then variants must be a
  list of equal length", etc.) belong in a per-function pydantic model with a
  ``@model_validator`` — see the ``*Params`` models in the ``varseek_*`` modules.
  Keeping existence out of the annotations is deliberate: varseek supports
  overwriting outputs, re-running partially-completed workflows, and passing
  sentinels such as the literal string ``"None"``, all of which a strict
  "must exist" / "must not exist" annotation would break.
* Extension checks accept an optional trailing ``.gz`` / ``.zip`` where relevant,
  matching the historical behavior of
  ``check_file_path_is_string_with_valid_extension``.
"""

import os
from pathlib import Path
from typing import Annotated, Any, Literal, Optional, Union

from pydantic import (
    AfterValidator,
    BeforeValidator,
    ConfigDict,
    Field,
    validate_call,
)

from ..constants import species_to_url, technology_valid_values

__all__ = [
    # config + decorator re-export
    "validate_call",
    "vk_config",
    # extension vocabulary
    "EXTENSIONS",
    # generic path helpers / factories
    "path_type",
    "existing_path_type",
    "coerce_pathlike",
    # structural (extension-only) file types
    "FastaFile",
    "GtfFile",
    "IndexFile",
    "TxtFile",
    "T2GFile",
    "CsvFile",
    "TsvFile",
    "VcfFile",
    "H5adFile",
    "FastqFile",
    "BamFile",
    "BedFile",
    "JsonFile",
    "YamlFile",
    # existence-checking file types
    "ExistingFasta",
    "ExistingGtf",
    "ExistingIndex",
    "ExistingTxt",
    "ExistingT2G",
    "ExistingCsv",
    "ExistingVcf",
    "ExistingH5ad",
    "ExistingFastq",
    "ExistingFile",
    "ExistingExecutable",
    # directories
    "Directory",
    "ExistingDirectory",
    # integers
    "PositiveInt",
    "NonNegativeInt",
    "OddInt",
    "Odd1to63",
    "OddInt3To63",
    "int_range",
    # floats
    "Ratio",
    # polymorphic domain args
    "GtfArg",
    # domain literals
    "Parity",
    "Strand",
    "StrandBiasEnd",
    "Workflow",
    "ReferenceType",
    "Species",
    "LoggingLevel",
    "Technology",
]


# --------------------------------------------------------------------------- #
# pydantic config shared by every varseek @validate_call site
# --------------------------------------------------------------------------- #
# arbitrary_types_allowed -> lets params accept pandas DataFrames / AnnData objs.
# validate_default=False   -> defaults (already valid python objects) are trusted.
vk_config = ConfigDict(arbitrary_types_allowed=True, validate_default=False)


# --------------------------------------------------------------------------- #
# extension vocabulary (mirrors check_file_path_is_string_with_valid_extension)
# --------------------------------------------------------------------------- #
EXTENSIONS: dict[str, tuple[str, ...]] = {
    "fasta": (".fasta", ".fa", ".fna", ".ffn"),
    "gtf": (".gtf",),
    "index": (".idx",),
    "txt": (".txt",),
    "t2g": (".txt",),
    "csv": (".csv",),
    "tsv": (".tsv",),
    "vcf": (".vcf",),
    "h5ad": (".h5ad", ".h5"),
    "fastq": (".fastq", ".fq"),
    "bam": (".bam",),
    "bed": (".bed",),
    "json": (".json",),
    "yaml": (".yaml", ".yml"),
}


def coerce_pathlike(value: Any) -> Any:
    """Turn ``os.PathLike`` inputs into plain strings; pass everything else through."""
    if isinstance(value, os.PathLike):
        return os.fspath(value)
    return value


def _validate_extension(
    value: str,
    *,
    extensions: tuple[str, ...],
    zip_allowed: bool,
) -> str:
    """Return ``value`` if its final extension is one of ``extensions`` (case-insensitive)."""
    lowered = value.lower()
    accepted = set(extensions)
    if zip_allowed:
        for ext in extensions:
            accepted.add(f"{ext}.gz")
            accepted.add(f"{ext}.zip")
    if not lowered.endswith(tuple(accepted)):
        pretty = ", ".join(f"{e}[.gz/.zip]" if zip_allowed else e for e in extensions)
        raise ValueError(f"'{value}' must have one of the following extensions: {pretty}")
    return value


def _validate_exists(value: str) -> str:
    if not Path(value).is_file():
        raise ValueError(f"'{value}' is not an existing file")
    return value


def path_type(file_kind: str, *, zip_allowed: bool = True):
    """Build an ``Annotated[str, ...]`` type that checks type + extension (not existence).

    ``file_kind`` is a key of :data:`EXTENSIONS`.
    """
    extensions = EXTENSIONS[file_kind]

    def _v(value: str) -> str:
        return _validate_extension(value, extensions=extensions, zip_allowed=zip_allowed)

    return Annotated[str, BeforeValidator(coerce_pathlike), AfterValidator(_v)]


def existing_path_type(file_kind: str, *, zip_allowed: bool = True):
    """Like :func:`path_type` but additionally requires the file to already exist."""
    extensions = EXTENSIONS[file_kind]

    def _v(value: str) -> str:
        value = _validate_extension(value, extensions=extensions, zip_allowed=zip_allowed)
        return _validate_exists(value)

    return Annotated[str, BeforeValidator(coerce_pathlike), AfterValidator(_v)]


# --------------------------------------------------------------------------- #
# structural (extension-only) file types — safe for outputs and sentinels
# --------------------------------------------------------------------------- #
FastaFile = path_type("fasta")
GtfFile = path_type("gtf")
IndexFile = path_type("index", zip_allowed=False)
TxtFile = path_type("txt", zip_allowed=False)
T2GFile = path_type("t2g", zip_allowed=False)
CsvFile = path_type("csv")
TsvFile = path_type("tsv")
VcfFile = path_type("vcf")
H5adFile = path_type("h5ad", zip_allowed=False)
FastqFile = path_type("fastq")
BamFile = path_type("bam", zip_allowed=False)
BedFile = path_type("bed")
JsonFile = path_type("json", zip_allowed=False)
YamlFile = path_type("yaml", zip_allowed=False)

# --------------------------------------------------------------------------- #
# existence-checking file types — for inputs that must already be on disk
# --------------------------------------------------------------------------- #
ExistingFasta = existing_path_type("fasta")
ExistingGtf = existing_path_type("gtf")
ExistingIndex = existing_path_type("index", zip_allowed=False)
ExistingTxt = existing_path_type("txt", zip_allowed=False)
ExistingT2G = existing_path_type("t2g", zip_allowed=False)
ExistingCsv = existing_path_type("csv")
ExistingVcf = existing_path_type("vcf")
ExistingH5ad = existing_path_type("h5ad", zip_allowed=False)
ExistingFastq = existing_path_type("fastq")

ExistingFile = Annotated[str, BeforeValidator(coerce_pathlike), AfterValidator(_validate_exists)]


def _validate_executable(value: str) -> str:
    value = _validate_exists(value)
    if not os.access(value, os.X_OK):
        raise ValueError(f"'{value}' is not executable")
    return value


ExistingExecutable = Annotated[str, BeforeValidator(coerce_pathlike), AfterValidator(_validate_executable)]


# --------------------------------------------------------------------------- #
# directories
# --------------------------------------------------------------------------- #
def _resolve_dir(value: str) -> str:
    return str(Path(value).resolve())


def _resolve_existing_dir(value: str) -> str:
    if not Path(value).is_dir():
        raise ValueError(f"'{value}' is not an existing directory")
    return str(Path(value).resolve())


Directory = Annotated[str, BeforeValidator(coerce_pathlike), AfterValidator(_resolve_dir)]
ExistingDirectory = Annotated[str, BeforeValidator(coerce_pathlike), AfterValidator(_resolve_existing_dir)]


# --------------------------------------------------------------------------- #
# integers
# --------------------------------------------------------------------------- #
PositiveInt = Annotated[int, Field(ge=1)]
NonNegativeInt = Annotated[int, Field(ge=0)]


def _must_be_odd(v: int) -> int:
    if v % 2 == 0:
        raise ValueError("value must be odd")
    return v


OddInt = Annotated[int, AfterValidator(_must_be_odd)]
Odd1to63 = Annotated[int, Field(ge=1, le=63), AfterValidator(_must_be_odd)]
OddInt3To63 = Annotated[int, Field(ge=3, le=63), AfterValidator(_must_be_odd)]


def int_range(min_value: Optional[int] = None, max_value: Optional[int] = None):
    """Return an ``Annotated[int, ...]`` bounded inclusively by ``min_value``/``max_value``."""
    constraints: dict[str, int] = {}
    if min_value is not None:
        constraints["ge"] = min_value
    if max_value is not None:
        constraints["le"] = max_value
    return Annotated[int, Field(**constraints)]


# --------------------------------------------------------------------------- #
# floats
# --------------------------------------------------------------------------- #
# A proportion in the half-open unit interval (0, 1] — e.g. min_triplet_complexity.
Ratio = Annotated[float, Field(gt=0, le=1)]


# --------------------------------------------------------------------------- #
# polymorphic domain args
# --------------------------------------------------------------------------- #
def _validate_gtf_arg(value: Any) -> Any:
    """Validate build's polymorphic ``gtf`` argument, returning it unchanged.

    Accepts ``None``, a ``bool`` (or the CLI-passed strings ``"True"``/``"False"``),
    or a path with a ``.gtf`` / ``.gtf.gz`` / ``.gtf.zip`` extension. The value is
    returned as-is (no coercion) so downstream code sees exactly what was passed.
    """
    if value is None or isinstance(value, bool):
        return value
    candidate = os.fspath(value) if isinstance(value, os.PathLike) else value
    if candidate in ("True", "False"):
        return value
    if isinstance(candidate, str) and candidate.lower().endswith(("gtf", "gtf.zip", "gtf.gz")):
        return value
    raise ValueError(f"Invalid value for gtf: {value!r}. Expected a .gtf[.gz/.zip] path, a bool, 'True'/'False', or None.")


# Based on ``Any`` (not ``Union[bool, str, Path]``) on purpose: it keeps pydantic
# from coercing across the union — e.g. the string "True" into a bool — so the
# checked value reaches the function body with its original type intact.
GtfArg = Annotated[Any, AfterValidator(_validate_gtf_arg)]


# --------------------------------------------------------------------------- #
# domain literals
# --------------------------------------------------------------------------- #
Parity = Literal["single", "paired"]
Strand = Literal["unstranded", "forward", "reverse"]
StrandBiasEnd = Literal["5p", "3p"]
Workflow = Literal["standard", "nac", "custom"]
# Also what build's alignment_to_reference_type is checked against.
ReferenceType = Literal["genome", "cdna", "transcriptome", "genome_or_transcriptome"]
# Derived from the keys of constants.species_to_url so there is one source of truth.
Species = Literal[tuple(species_to_url)]
LoggingLevel = Literal[
    "NOTSET", "DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL",
    "0", "10", "20", "30", "40", "50", "60",
]


def _validate_technology(value: str) -> str:
    """Accept any casing of a supported sequencing technology; return it unchanged."""
    if value.lower() not in {t.lower() for t in technology_valid_values}:
        raise ValueError(f"technology must be one of {sorted(technology_valid_values)} (case-insensitive)")
    return value


Technology = Annotated[str, AfterValidator(_validate_technology)]
