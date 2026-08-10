"""Helper functions for varseek denovo."""

import bisect
import gzip
import json
import os
import re
import shutil
import subprocess
import logging
from collections import Counter

import numpy as np
import pysam
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

logger = logging.getLogger(__name__)


def infer_hgvs_prefix(seq_id, reference_type="auto"):
    """Return an HGVS nucleotide prefix for a VCF sequence ID."""
    reference_type = (reference_type or "auto").lower()
    if reference_type in {"dna", "genome", "genomic", "g"}:
        return "g."
    if reference_type in {"cdna", "transcriptome", "transcript", "c"}:
        return "c."
    if reference_type != "auto":
        raise ValueError("--tsv-reference-type must be one of auto, dna, genome, cdna, transcriptome")

    seq_id = str(seq_id)
    transcript_prefixes = ("ENST", "NM_", "NR_", "XM_", "XR_")
    return "c." if seq_id.startswith(transcript_prefixes) else "g."


def infer_reference_type_from_fasta(sequences):
    """Guess whether a reference FASTA is a genome or transcriptome from its first header.

    Returns "genome" or "transcriptome" using the same transcript-ID prefixes as
    infer_hgvs_prefix (ENST/NM_/NR_/XM_/XR_). Falls back to "genome" if no header is found."""
    transcript_prefixes = ("ENST", "NM_", "NR_", "XM_", "XR_")
    opener = gzip.open if str(sequences).endswith(".gz") else open
    with opener(sequences, "rt") as handle:
        for line in handle:
            if line.startswith(">"):
                seq_id = line[1:].strip().split()[0] if line[1:].strip() else ""
                return "transcriptome" if seq_id.startswith(transcript_prefixes) else "genome"
    return "genome"


def vcf_allele_to_hgvs(pos, ref, alt, prefix):
    """Convert one VCF REF/ALT allele into a simple HGVS-like nucleotide variant."""
    if alt in {None, "", "."}:
        return None
    alt = str(alt)
    ref = str(ref)
    if any(token in alt for token in ("<", ">", "[", "]")) or alt == "*":
        logger.warning("Skipping unsupported symbolic VCF ALT allele: %s", alt)
        return None

    start = int(pos)
    ref_trimmed = ref
    alt_trimmed = alt

    while ref_trimmed and alt_trimmed and ref_trimmed[0] == alt_trimmed[0]:
        ref_trimmed = ref_trimmed[1:]
        alt_trimmed = alt_trimmed[1:]
        start += 1

    while ref_trimmed and alt_trimmed and ref_trimmed[-1] == alt_trimmed[-1]:
        ref_trimmed = ref_trimmed[:-1]
        alt_trimmed = alt_trimmed[:-1]

    if len(ref_trimmed) == 1 and len(alt_trimmed) == 1:
        return f"{prefix}{start}{ref_trimmed}>{alt_trimmed}"

    if ref_trimmed and not alt_trimmed:
        end = start + len(ref_trimmed) - 1
        return f"{prefix}{start}del" if start == end else f"{prefix}{start}_{end}del"

    if alt_trimmed and not ref_trimmed:
        left = start - 1
        right = start
        return f"{prefix}{left}_{right}ins{alt_trimmed}"

    if ref_trimmed and alt_trimmed:
        end = start + len(ref_trimmed) - 1
        return f"{prefix}{start}_{end}delins{alt_trimmed}"

    return None


def vcf_to_tsv(vcf_path, tsv_path, reference_type="auto", overwrite=False, logger=logger):
    """Write a two-column seq_id/variant TSV from a VCF file."""
    if os.path.exists(tsv_path) and not overwrite:
        raise ValueError(f"TSV output file '{tsv_path}' already exists. Use --overwrite to overwrite.")
    os.makedirs(os.path.dirname(tsv_path) or ".", exist_ok=True)

    rows = []
    with pysam.VariantFile(vcf_path) as vcf:
        for record in vcf:
            seq_id = record.chrom
            prefix = infer_hgvs_prefix(seq_id, reference_type=reference_type)
            for alt in record.alts or []:
                variant = vcf_allele_to_hgvs(record.pos, record.ref, alt, prefix)
                if variant:
                    rows.append({"seq_id": seq_id, "variant": variant})

    pd.DataFrame(rows, columns=["seq_id", "variant"]).to_csv(tsv_path, sep="\t", index=False)
    logger.info("TSV written to %s with %d variants", tsv_path, len(rows))

def configure_logger(verbose_level, quiet):
    """Configure the logger based on verbosity and quiet flags."""
    if quiet:
        level = logging.CRITICAL
    elif verbose_level >= 2:
        level = logging.DEBUG
    elif verbose_level == 1:
        level = logging.INFO
    else:
        level = logging.WARNING

    logging.basicConfig(level=level, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")

def run(cmd, check=True, shell=True, logger=logger):
    """Run a shell command and log it before execution."""
    logger.debug(cmd)
    subprocess.run(cmd, shell=shell, check=check)

def check_tool(tool):
    """Ensure that a required command-line tool is available."""
    from varseek.utils.logger_utils import is_program_installed  # shared availability check (reused across modules)
    if not is_program_installed(tool):
        raise ValueError(f"required tool '{tool}' is not installed or not in PATH.")


def tokenize_md(md):
    """Tokenize an MD tag into a list of ('match', n) / ('mismatch', base) / ('del', bases)."""
    tokens = []
    for num, deletion, mismatch in re.findall(r"(\d+)|\^([A-Za-z]+)|([A-Za-z])", md):
        if num:
            n = int(num)
            if n:  # MD emits 0-length runs between adjacent mismatches
                tokens.append(("match", n))
        elif deletion:
            tokens.append(("del", deletion.upper()))
        elif mismatch:
            tokens.append(("mismatch", mismatch.upper()))
    return tokens


def extract_variants_from_read(read, chrom):
    """Return HGVS-style variant strings for one aligned read.

    Walks the CIGAR and the MD tag together. MD describes only the reference bases covered
    by M/=/X and D operations -- it encodes nothing about soft clips (S), insertions (I), or
    spliced gaps (N), so it cannot be walked on its own to derive coordinates. Doing so
    shifts every variant downstream of a junction by the intron length, which is why spliced
    (e.g. STAR) alignments in particular need the joint walk.
    """
    seq = read.query_sequence
    cigar = read.cigartuples
    if seq is None or not cigar or not read.has_tag("MD"):
        return []

    md_tokens = tokenize_md(read.get_tag("MD"))
    md_idx = 0
    md_match_left = 0  # unconsumed length of the current ('match', n) token

    variants = []
    ref_pos = read.reference_start + 1  # pysam is 0-based, HGVS is 1-based
    read_pos = 0

    for op, length in cigar:
        if op in (0, 7, 8):  # M / = / X -- consume both; MD says which of these mismatch
            consumed = 0
            while consumed < length:
                if md_match_left == 0:
                    if md_idx >= len(md_tokens):
                        break  # MD exhausted early (malformed); treat the rest as matches
                    kind, value = md_tokens[md_idx]
                    md_idx += 1
                    if kind == "match":
                        md_match_left = value
                    elif kind == "mismatch":
                        alt_base = seq[read_pos + consumed]
                        variants.append(f"{chrom}:g.{ref_pos + consumed}{value}>{alt_base}")
                        consumed += 1
                    # a 'del' token cannot occur inside an M run; skip it defensively
                    continue
                step = min(md_match_left, length - consumed)
                md_match_left -= step
                consumed += step
            ref_pos += length
            read_pos += length
        elif op == 1:  # I -- consumes read only; invisible to MD. ref_pos is the base *after* the insertion.
            variants.append(f"{chrom}:g.{ref_pos - 1}_{ref_pos}ins{seq[read_pos:read_pos + length]}")
            read_pos += length
        elif op == 2:  # D -- consumes reference; MD carries a matching ^ token
            if md_match_left == 0 and md_idx < len(md_tokens) and md_tokens[md_idx][0] == "del":
                md_idx += 1
            if length == 1:
                variants.append(f"{chrom}:g.{ref_pos}del")
            else:
                variants.append(f"{chrom}:g.{ref_pos}_{ref_pos + length - 1}del")
            ref_pos += length
        elif op == 3:  # N -- spliced gap: a reference skip, not a deletion, and absent from MD
            ref_pos += length
        elif op == 4:  # S -- soft clip: present in query_sequence, so it advances read_pos
            read_pos += length
        # 5 (H) and 6 (P) consume neither the read nor the reference

    return variants


def parse_cigars(bam_path=None, total=None, out_plot=None, do_baq=False, regions=None, min_threshold=3, strip_version_numbers=False, out_dataframe=None, logger=logger):
    # TODO: accept multiple BAMs
    # TODO: change df column name from Chromosome to Transcript if genome vs cdna (need to detect which I'm using)
    # TODO: implement BAQ adjustment if do_baq=True
    # TODO: distinguish delins from multiple substitutions
    # TODO: try parsing STAR-aligned (I need to run STAR with --outSAMattributes NH HI AS nM MD)
    bam = pysam.AlignmentFile(bam_path, "rb")
    variant_counter = Counter()

    if total is True:
        total = sum(1 for _ in bam)
        bam.reset()

    reads_considered = 0
    reads_with_md = 0
    for read in tqdm(bam, total=total):
        if read.is_unmapped:
            continue
        if all(op in {3, 4, 5, 6, 7} for op, _ in read.cigartuples):  # skip reads without difference from reference
            continue

        reads_considered += 1
        if not read.has_tag("MD"):
            continue
        reads_with_md += 1

        chrom = bam.get_reference_name(read.reference_id)
        for hgvs in extract_variants_from_read(read, chrom):
            variant_counter[hgvs] += 1

    # Without MD there is nothing to read mismatches and deletions out of, and extraction
    # would quietly return an empty (or insertion-only) result rather than failing.
    if reads_considered and not reads_with_md:
        raise ValueError(
            f"no read in '{bam_path}' carries an MD tag, so the 'cigar' variant caller cannot "
            "identify mismatches or deletions. Re-align emitting MD (for STAR: "
            "--outSAMattributes NH HI AS nM MD), or add it with "
            f"`samtools calmd -b {bam_path} <reference.fa> > with_md.bam`."
        )
    if reads_considered and reads_with_md < reads_considered:
        logger.warning(f"{reads_considered - reads_with_md} of {reads_considered} reads lack an MD tag and were skipped.")

    # make histogram of filtered_variants counts
    if out_plot:
        counts = list(variant_counter.values())
        plt.figure(figsize=(8,5))
        plt.hist(counts, bins=50, color="skyblue", edgecolor="black")
        plt.yscale("log")
        plt.xlabel("Variant Count")
        plt.ylabel("Frequency (log scale)")
        plt.title("Histogram of Variant Counts (Filtered)")
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(out_plot)
        plt.close()

    df = pd.DataFrame(variant_counter.items(), columns=["key", "Count"])
    logger.info(f"Total unique variants found: {len(df)}")

    df[["Chromosome", "Variant"]] = df["key"].str.split(":", n=1, expand=True)  # Split key like "1:g.17746delA" into two columns

    if regions:
        # Imported here rather than at module scope: pyranges is not a declared dependency and is
        # only needed for region filtering, so requiring it up front would block the caller entirely.
        try:
            import pyranges as pr
        except ImportError as exc:
            raise ImportError("region filtering (regions=...) requires pyranges. Install it with `pip install pyranges`, or omit regions.") from exc

        df["pos"] = pd.to_numeric(df["Variant"].str.extract(r"g\.(\d+)")[0])
        df["Start"] = df["pos"] - 1  # PyRanges is half-open: [start, end)
        df["End"] = df["pos"]

        # Convert both to PyRanges
        variants_pr = pr.PyRanges(df[["Chromosome", "Variant", "Start", "End", "key", "Count"]])
        bed_pr = pr.read_bed(regions)

        # Interval intersection (fast!)
        df = variants_pr.join(bed_pr).as_df()
        df = df.drop_duplicates(subset=["Chromosome", "Variant"]).reset_index(drop=True)
        logger.info(f"Total unique variants after region filtering: {len(df)}")

    df = df[["Chromosome", "Variant", "Count"]]

    if strip_version_numbers:
        df["Chromosome"] = df["Chromosome"].str.replace(r"\.[0-9]+$", "", regex=True)

    if min_threshold:
        df = df.loc[df["Count"] >= min_threshold].reset_index(drop=True)
        logger.info(f"Total unique variants after applying min_threshold: {len(df)}")

    if out_dataframe:
        df.to_csv(out_dataframe, index=False)

    logger.info(f"Final unique variants: {len(df)}")


# ---------------------------------------------------------------------------
# bam_to_vcf: every variant recorded in the alignments, straight to VCF
#
# parse_cigars above does the same walk in Python and writes a CSV of HGVS strings; it is
# bounded by the per-read CPython loop. bam_to_vcf emits a standard sites-only VCF and
# pushes the walk into varseek/cpp/bam2vcf.cpp, falling back to an equivalent pysam
# implementation when that cannot be compiled.
# ---------------------------------------------------------------------------

#: Reads carrying any of these flags are ignored: unmapped, secondary, QC fail, duplicate,
#: supplementary. Same default as `bcftools mpileup`.
DEFAULT_EXCLUDE_FLAGS = 0xF04

_MD_TOKEN = re.compile(r"(\d+)|\^([A-Za-z]+)|([A-Za-z])")


class _VcfWriter:
    """Text or BGZF VCF output behind one ``write(str)`` interface."""

    def __init__(self, path, bgzip):
        self.bgzip = bgzip
        if bgzip:
            self.handle = pysam.BGZFile(path, "wb")
        else:
            self.handle = open(path, "w")
        self.buf = []
        self.buflen = 0

    def write(self, text):
        self.buf.append(text)
        self.buflen += len(text)
        if self.buflen >= (1 << 20):
            self.flush()

    def flush(self):
        if not self.buf:
            return
        chunk = "".join(self.buf)
        self.handle.write(chunk.encode() if self.bgzip else chunk)
        self.buf = []
        self.buflen = 0

    def close(self):
        self.flush()
        self.handle.close()


class _RefWindow:
    """Chunked reference FASTA reader.

    Reads arrive in ascending coordinate order, so a single sliding chunk keeps the hit
    rate near 1. The left margin lets indel left-alignment look backwards without
    forcing a refetch.
    """

    CHUNK = 1 << 20
    MARGIN = 1 << 16

    def __init__(self, fasta):
        self.fa = fasta
        self.chrom = None
        self.start = 1
        self.seq = ""
        self._known = set(fasta.references) if fasta is not None else set()

    def span(self, chrom, beg, length):
        """Reference bases for 1-based inclusive [beg, beg+length-1], or None."""
        if self.fa is None or length <= 0 or beg < 1:
            return None
        end = beg + length - 1
        if chrom != self.chrom or beg < self.start or end > self.start + len(self.seq) - 1:
            if chrom not in self._known:
                return None
            contig_len = self.fa.get_reference_length(chrom)
            if end > contig_len:
                return None
            win_start = max(1, beg - self.MARGIN)
            win_end = min(contig_len, max(end, win_start + self.CHUNK - 1))
            self.seq = self.fa.fetch(chrom, win_start - 1, win_end).upper()
            self.chrom = chrom
            self.start = win_start
            if beg < self.start or end > self.start + len(self.seq) - 1:
                return None
        off = beg - self.start
        return self.seq[off : off + length]

    def base(self, chrom, pos):
        return self.span(chrom, pos, 1) or None


class _DepthRing:
    """Sliding per-position read depth as a power-of-two numpy ring buffer.

    Mirrors the ring in varseek/cpp/bam2vcf.cpp. Because the BAM is coordinate sorted,
    every position below the current read start is final, so the buffer only ever has to
    span the longest read reference span rather than the contig. Increments go through
    numpy slice arithmetic, which keeps the per-read cost at a couple of operations
    instead of one per covered base.
    """

    def __init__(self, cap=4096):
        self.cap = cap
        self.mask = cap - 1
        self.buf = np.zeros(cap, dtype=np.int32)
        self.head = 0
        self.win_start = 1
        self.win_end = 0

    def reset(self, win_start):
        self.buf[:] = 0
        self.head = 0
        self.win_start = win_start
        self.win_end = win_start - 1

    def _slices(self, i, length):
        """One or two (start, stop) index pairs covering `length` entries from ring index i."""
        end = i + length
        if end <= self.cap:
            return ((i, end),)
        return ((i, self.cap), (0, end - self.cap))

    def _grow(self, need):
        cap = self.cap
        while cap < need:
            cap <<= 1
        buf = np.zeros(cap, dtype=np.int32)
        span = self.win_end - self.win_start + 1
        if span > 0:
            off = 0
            for a, b in self._slices(self.head, span):
                buf[off : off + (b - a)] = self.buf[a:b]
                off += b - a
        self.buf = buf
        self.cap = cap
        self.mask = cap - 1
        self.head = 0

    def add(self, pos1, length):
        """Count one read as covering `length` bases starting at 1-based pos1."""
        if length <= 0:
            return
        if pos1 < self.win_start:  # already flushed; nothing to record
            length -= self.win_start - pos1
            pos1 = self.win_start
            if length <= 0:
                return
        last = pos1 + length - 1
        if last - self.win_start + 1 > self.cap:
            self._grow(last - self.win_start + 1)
        if last > self.win_end:
            self.win_end = last
        i = (self.head + (pos1 - self.win_start)) & self.mask
        for a, b in self._slices(i, length):
            self.buf[a:b] += 1

    def at(self, pos1):
        if pos1 < self.win_start or pos1 > self.win_end:
            return 0
        return int(self.buf[(self.head + (pos1 - self.win_start)) & self.mask])

    def advance(self, upto):
        """Drop every position below `upto`."""
        if upto <= self.win_start:
            return
        n = min(upto, self.win_end + 1) - self.win_start
        if n > 0:
            for a, b in self._slices(self.head, n):
                self.buf[a:b] = 0
            self.head = (self.head + n) & self.mask
            self.win_start += n
        if upto > self.win_start:  # jumped past everything buffered
            self.win_start = upto
            self.win_end = upto - 1
            self.head = 0


class _BedFilter:
    """Merged-interval BED membership test."""

    def __init__(self, path):
        by_chrom = {}
        opener = gzip.open if str(path).endswith(".gz") else open
        with opener(path, "rt") as handle:
            for line in handle:
                if not line.strip() or line.startswith(("#", "track", "browser")):
                    continue
                fields = line.split()
                if len(fields) < 3:
                    continue
                by_chrom.setdefault(fields[0], []).append([int(fields[1]), int(fields[2])])
        self.by_chrom = {}
        for chrom, intervals in by_chrom.items():
            intervals.sort()
            merged = []
            for start, end in intervals:
                if merged and start <= merged[-1][1]:
                    merged[-1][1] = max(merged[-1][1], end)
                else:
                    merged.append([start, end])
            self.by_chrom[chrom] = ([m[0] for m in merged], [m[1] for m in merged])

    def overlaps(self, chrom, beg0, end0):
        """True if 0-based half-open [beg0, end0) touches any region."""
        entry = self.by_chrom.get(chrom)
        if not entry:
            return False
        starts, ends = entry
        i = bisect.bisect_right(starts, end0 - 1) - 1
        return i >= 0 and beg0 < ends[i]


def _refspan_from_md(md, cigartuples, query_bytes, span_len):
    """Reference bases over a read's aligned span, reconstructed from its MD tag.

    Returns a bytearray of length span_len, with 0 wherever the base is unknown (spliced
    gaps), or None if the MD tag is malformed. MD describes only the reference bases under
    M/=/X and D operations -- it encodes nothing about soft clips, insertions or spliced
    gaps -- so it is consumed here in lockstep with the CIGAR rather than on its own. Note
    that an MD match run may therefore span an I or N operation, so `run` deliberately
    carries across CIGAR operations.

    Matched stretches are copied from the read in bulk rather than base by base, which is
    what keeps this within reach of the compiled path.
    """
    tokens = _MD_TOKEN.findall(md)
    ntok = len(tokens)
    out = bytearray(span_len)
    ti = 0
    run = 0  # unconsumed length of the current MD match run
    off = 0
    qpos = 0
    for op, length in cigartuples:
        if op in (0, 7, 8):  # M / = / X
            remaining = length
            while remaining > 0:
                if run > 0:
                    n = run if run < remaining else remaining
                    out[off : off + n] = query_bytes[qpos : qpos + n]  # a match: ref base == read base
                    run -= n
                    off += n
                    qpos += n
                    remaining -= n
                    continue
                advanced = False
                while ti < ntok:
                    num, deletion, mismatch = tokens[ti]
                    ti += 1
                    if num:
                        n = int(num)
                        if n == 0:  # MD emits 0-length runs between adjacent mismatches
                            continue
                        run = n
                        advanced = True
                        break
                    if deletion:
                        return None  # a deletion token cannot occur inside an M run
                    out[off] = ord(mismatch.upper())
                    off += 1
                    qpos += 1
                    remaining -= 1
                    advanced = True
                    break
                if not advanced:
                    return None  # MD exhausted before the CIGAR
        elif op == 2:  # D
            if run != 0:
                return None
            while ti < ntok and tokens[ti][0] and int(tokens[ti][0]) == 0:
                ti += 1
            if ti >= ntok or not tokens[ti][1]:
                return None
            bases = tokens[ti][1].upper()
            ti += 1
            if len(bases) != length:
                return None
            out[off : off + length] = bases.encode()
            off += length
        elif op == 3:  # N -- reference skip; left unknown, and never needed
            off += length
        elif op in (1, 4):  # I / S consume the read only
            qpos += length
    return out


def _vcf_header(bam_file, bam_path, reference, min_count, min_vaf, max_vaf, min_mapq, min_baseq, normalize,
                strip_version_numbers, emit_type):
    """VCF header text matching what varseek/cpp/bam2vcf.cpp writes."""
    lines = [
        "##fileformat=VCFv4.2",
        "##source=varseek-bam2vcf-python",
        f'##bam2vcfCommandLine=<bam="{bam_path}",minCount={min_count},'
        f"minVAF={min_vaf:g},maxVAF={max_vaf:g},minMapQ={min_mapq},"
        f"minBaseQ={min_baseq},normalize={1 if normalize else 0}>",
    ]
    if reference:
        lines.append(f"##reference=file://{reference}")
    for name, length in zip(bam_file.references, bam_file.lengths):
        lines.append(f"##contig=<ID={_display_contig(name, strip_version_numbers)},length={length}>")
    lines += [
        '##INFO=<ID=AO,Number=A,Type=Integer,Description="Reads supporting the alternate allele">',
        '##INFO=<ID=DP,Number=1,Type=Integer,Description="Reads passing filters that span this position">',
        '##INFO=<ID=VAF,Number=A,Type=Float,Description="Alternate allele fraction, AO/DP">',
    ]
    if emit_type:
        lines.append('##INFO=<ID=TYPE,Number=A,Type=String,Description="Variant class: snv, ins or del">')
    lines.append("#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO")
    return "\n".join(lines) + "\n"


def _display_contig(name, strip_version_numbers):
    if strip_version_numbers:
        return re.sub(r"\.[0-9]+$", "", str(name))
    return str(name)


def _header_is_coordinate_sorted(bam_file):
    hd = bam_file.header.to_dict().get("HD", {})
    return hd.get("SO") == "coordinate"


def bam_to_vcf(
    bam,
    output="out.vcf.gz",
    reference=None,
    regions=None,
    min_count=1,
    min_vaf=0.0,
    max_vaf=1.0,
    min_mapq=0,
    min_baseq=0,
    exclude_flags=DEFAULT_EXCLUDE_FLAGS,
    require_flags=0,
    max_reads=None,
    threads=1,
    normalize=True,
    skip_indels=False,
    index=False,
    strip_version_numbers=False,
    assume_sorted=False,
    emit_type=True,
    overwrite=False,
    engine="auto",
    progress=False,
    logger=logger,
):
    """Write a sites-only VCF of every variant recorded in a BAM's alignment records.

    Walks each read's CIGAR to collect mismatches (from the MD tag, or by comparison
    against `reference` when given) plus insertions and deletions, aggregates identical
    alleles across reads, and writes one VCF record per allele with INFO fields AO
    (supporting reads), DP (reads spanning the position) and VAF (AO/DP).

    This is the fast counterpart to parse_cigars, which writes HGVS strings to CSV. The
    work is done by varseek/cpp/bam2vcf.cpp, compiled and cached on first use; if it
    cannot be built, an equivalent pysam implementation runs instead.

    Arguments:
    - bam                     (str) Coordinate-sorted BAM/CRAM. Required.
    - output                  (str) Output VCF path. A .gz/.bgz suffix produces BGZF. Default: "out.vcf.gz"
    - reference               (str) Reference FASTA (indexed). Enables indel left-alignment
                                   (matching `bcftools norm`) and removes the MD-tag
                                   requirement. Without it, indels are reported as aligned
                                   and reads need MD. Default: None
    - regions                 (str) BED file; only variants overlapping these regions are kept. Default: None
    - min_count               (int) Minimum supporting reads per allele, i.e. a filter on AO. Default: 1 (report everything)
    - min_vaf               (float) Minimum alternate allele fraction AO/DP, inclusive. Sites with DP of 0
                                   cannot satisfy a VAF bound and are dropped when either bound is active. Default: 0.0
    - max_vaf               (float) Maximum alternate allele fraction AO/DP, inclusive. Lower it to drop
                                   likely-germline homozygous sites. Default: 1.0
    - min_mapq                (int) Minimum read mapping quality. Default: 0
    - min_baseq               (int) Minimum base quality at the variant; applies to
                                   substitutions and inserted bases. Default: 0
    - exclude_flags           (int) Skip reads carrying any of these SAM flags. Default: 0xF04
    - require_flags           (int) Skip reads missing any of these SAM flags. Default: 0
    - max_reads               (int) Stop after this many reads, for quick checks. Default: None
    - threads                 (int) BGZF worker threads. Default: 1
    - normalize              (bool) Left-align indels; requires `reference`. Default: True
    - skip_indels            (bool) Report substitutions only. Default: False
    - index                  (bool) Tabix-index the output; requires BGZF. Default: False
    - strip_version_numbers  (bool) Drop trailing .N version suffixes from contig names. Default: False
    - assume_sorted          (bool) Skip the @HD SO:coordinate check. Default: False
    - emit_type              (bool) Include the TYPE INFO field. Default: True
    - overwrite              (bool) Overwrite an existing output file. Default: False
    - engine                  (str) "auto", "native" or "python". Default: "auto"
    - progress               (bool) Report progress while running. Default: False

    Returns a dict of run statistics (reads seen and used, records written, and so on).
    """
    if engine not in {"auto", "native", "python"}:
        raise ValueError("engine must be one of 'auto', 'native' or 'python'")
    if not os.path.isfile(bam):
        raise ValueError(f"BAM file '{bam}' does not exist")
    if reference is not None and not os.path.isfile(reference):
        raise ValueError(f"reference FASTA '{reference}' does not exist")
    if regions is not None and not os.path.isfile(regions):
        raise ValueError(f"regions BED '{regions}' does not exist")
    if not 0.0 <= min_vaf <= 1.0 or not 0.0 <= max_vaf <= 1.0:
        raise ValueError(f"min_vaf and max_vaf must lie between 0 and 1, got min_vaf={min_vaf}, max_vaf={max_vaf}")
    if min_vaf > max_vaf:
        raise ValueError(f"min_vaf ({min_vaf}) exceeds max_vaf ({max_vaf}), so no variant could pass")
    if os.path.exists(output) and not overwrite:
        raise ValueError(f"output file '{output}' already exists. Use overwrite=True to overwrite.")
    os.makedirs(os.path.dirname(os.path.abspath(output)) or ".", exist_ok=True)

    bgzip = str(output).endswith((".gz", ".bgz"))
    if index and not bgzip:
        raise ValueError("index=True requires a BGZF output path (use a .gz suffix)")
    if normalize and not reference:
        # Left-alignment needs reference context beyond the read, so it is silently a no-op
        # without a FASTA; say so rather than implying the output is normalized.
        logger.info("no reference FASTA given, so indels will be reported as aligned rather than left-aligned")
        normalize = False

    common = dict(
        bam=bam,
        output=output,
        reference=reference,
        regions=regions,
        min_count=min_count,
        min_vaf=min_vaf,
        max_vaf=max_vaf,
        min_mapq=min_mapq,
        min_baseq=min_baseq,
        exclude_flags=exclude_flags,
        require_flags=require_flags,
        max_reads=max_reads,
        threads=threads,
        normalize=normalize,
        skip_indels=skip_indels,
        index=index,
        strip_version_numbers=strip_version_numbers,
        assume_sorted=assume_sorted,
        emit_type=emit_type,
        bgzip=bgzip,
        progress=progress,
        logger=logger,
    )

    if engine in {"auto", "native"}:
        from varseek.utils.native import NativeBuildError, program_path  # imported lazily: only the native path needs it

        try:
            binary = program_path("bam2vcf")
        except NativeBuildError as exc:
            if engine == "native":
                raise
            logger.warning("falling back to the Python implementation of bam_to_vcf: %s", exc)
        else:
            return _bam_to_vcf_native(binary, **common)

    return _bam_to_vcf_python(**common)


def _bam_to_vcf_native(binary, bam, output, reference, regions, min_count, min_vaf, max_vaf, min_mapq, min_baseq, exclude_flags,
                       require_flags, max_reads, threads, normalize, skip_indels, index, strip_version_numbers,
                       assume_sorted, emit_type, bgzip, progress, logger):
    """Run the compiled bam2vcf and return its statistics."""
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        stats_path = os.path.join(tmp, "stats.json")
        cmd = [
            binary,
            "--bam", str(bam),
            "--out", str(output),
            "--min-count", str(min_count),
            "--min-vaf", repr(float(min_vaf)),
            "--max-vaf", repr(float(max_vaf)),
            "--min-mapq", str(min_mapq),
            "--min-baseq", str(min_baseq),
            "--exclude-flags", str(int(exclude_flags)),
            "--require-flags", str(int(require_flags)),
            "--threads", str(max(1, int(threads))),
            "--stats-json", stats_path,
        ]
        if reference:
            cmd += ["--reference", str(reference)]
        if regions:
            cmd += ["--regions", str(regions)]
        if max_reads:
            cmd += ["--max-reads", str(int(max_reads))]
        if not normalize:
            cmd.append("--no-normalize")
        if skip_indels:
            cmd.append("--skip-indels")
        if not emit_type:
            cmd.append("--no-type")
        if bgzip:
            cmd.append("--bgzip")
        if index:
            cmd.append("--index")
        if strip_version_numbers:
            cmd.append("--strip-version")
        if assume_sorted:
            cmd.append("--assume-sorted")
        if progress:
            cmd.append("--progress")

        logger.debug("%s", " ".join(cmd))
        proc = subprocess.run(cmd, capture_output=True, text=True)
        for line in (proc.stderr or "").splitlines():
            text = line.strip()
            if not text:
                continue
            text = text[len("bam2vcf:") :].strip() if text.startswith("bam2vcf:") else text
            (logger.info if proc.returncode == 0 else logger.error)("bam2vcf: %s", text)
        if proc.returncode != 0:
            # Carry the reason into the exception: callers that catch it (and tracebacks in
            # scripts) would otherwise only see an exit code, with the cause stranded in the log.
            detail = [line.strip() for line in (proc.stderr or "").strip().splitlines() if line.strip()]
            reason = " ".join(detail[-6:]) if detail else "no error output"
            raise RuntimeError(f"bam2vcf failed with exit code {proc.returncode} for '{bam}': {reason}")

        stats = {}
        if os.path.isfile(stats_path):
            with open(stats_path) as handle:
                stats = json.load(handle)

    stats["engine"] = "native"
    logger.info(
        "bam_to_vcf wrote %s records from %s reads to %s",
        stats.get("records_emitted", "?"),
        stats.get("reads_used", "?"),
        output,
    )
    return stats


def _bam_to_vcf_python(bam, output, reference, regions, min_count, min_vaf, max_vaf, min_mapq, min_baseq, exclude_flags, require_flags,
                       max_reads, threads, normalize, skip_indels, index, strip_version_numbers, assume_sorted,
                       emit_type, bgzip, progress, logger):
    """Pure-pysam equivalent of varseek/cpp/bam2vcf.cpp.

    Same algorithm and same output, including the sliding window that makes depth
    accounting single-pass: because the input is coordinate sorted, no read starting at S
    can carry a variant at, or contribute coverage to, any position below S, so positions
    below the current read start are final and can be emitted and dropped.
    """
    stats = {
        "reads_total": 0,
        "reads_used": 0,
        "reads_skipped_flag": 0,
        "reads_skipped_mapq": 0,
        "reads_skipped_region": 0,
        "reads_fast_path": 0,
        "reads_with_md": 0,
        "reads_bad_md": 0,
        "reads_no_ref": 0,
        "reads_without_ref_bases": 0,
        "alleles_seen": 0,
        "records_emitted": 0,
        "sites_emitted": 0,
        "dropped_min_count": 0,
        "dropped_vaf": 0,
        "dropped_region": 0,
        "skipped_no_anchor": 0,
        "shifts_clamped": 0,
        "engine": "python",
    }

    bed = _BedFilter(regions) if regions else None
    fasta = pysam.FastaFile(reference) if reference else None
    refwin = _RefWindow(fasta) if fasta is not None else None
    flush_margin = 1000 if normalize else 0
    vaf_filter = min_vaf > 0.0 or max_vaf < 1.0

    bam_file = pysam.AlignmentFile(bam, "rb", threads=max(1, int(threads)))
    try:
        if not assume_sorted and not _header_is_coordinate_sorted(bam_file):
            raise ValueError(
                f"'{bam}' is not marked coordinate sorted (@HD SO:coordinate). Depth accounting and "
                f"sorted VCF output require coordinate order. Run `samtools sort -o sorted.bam {bam}`, "
                "or pass assume_sorted=True if the header is merely missing the tag."
            )

        writer = _VcfWriter(output, bgzip)
        try:
            writer.write(
                _vcf_header(bam_file, bam, reference, min_count, min_vaf, max_vaf, min_mapq, min_baseq, normalize,
                            strip_version_numbers, emit_type)
            )

            depth = _DepthRing()
            alleles = {}      # pos1 -> {(ref, alt, type): supporting reads}
            state = {"tid": None, "chrom": None, "out_chrom": None, "win_start": 1}

            def emit_upto(upto):
                """Emit and drop every position below `upto`."""
                if upto <= state["win_start"]:
                    return
                for pos in sorted(p for p in alleles if p < upto):
                    site_written = False
                    dp = depth.at(pos)
                    for (ref_allele, alt_allele, vtype), count in alleles[pos].items():
                        if count < min_count:
                            stats["dropped_min_count"] += 1
                            continue
                        # VAF is undefined without depth, so a site with DP==0 cannot satisfy a
                        # VAF bound and is dropped rather than silently passed through.
                        vaf_value = (count / dp) if dp else None
                        if vaf_filter and (vaf_value is None or vaf_value < min_vaf or vaf_value > max_vaf):
                            stats["dropped_vaf"] += 1
                            continue
                        if bed is not None and not bed.overlaps(state["chrom"], pos - 1, pos - 1 + len(ref_allele)):
                            stats["dropped_region"] += 1
                            continue
                        vaf = f"{vaf_value:.6g}" if vaf_value is not None else "."
                        info = f"AO={count};DP={dp};VAF={vaf}"
                        if emit_type:
                            info += f";TYPE={vtype}"
                        writer.write(f"{state['out_chrom']}\t{pos}\t.\t{ref_allele}\t{alt_allele}\t.\t.\t{info}\n")
                        stats["records_emitted"] += 1
                        site_written = True
                    if site_written:
                        stats["sites_emitted"] += 1
                    del alleles[pos]
                depth.advance(upto)
                state["win_start"] = upto

            def drain():
                """Emit everything still buffered, e.g. at a contig change or end of file."""
                highest = max(list(alleles.keys()) + [depth.win_end], default=None)
                if highest is not None:
                    emit_upto(highest + 1)

            def add_allele(pos, ref_allele, alt_allele, vtype):
                if pos < state["win_start"]:
                    stats["shifts_clamped"] += 1
                    return
                key = (ref_allele, alt_allele, vtype)
                slot = alleles.setdefault(pos, {})
                if key in slot:
                    slot[key] += 1
                else:
                    slot[key] = 1
                    stats["alleles_seen"] += 1

            def left_shift(seq, first1):
                """Rotate `seq` left while the preceding reference base equals its last base.

                This is the standard left-shift for an indel inside a repeat, matching
                `bcftools norm`. `first1` is the 1-based position of the first base of
                `seq` (for an insertion, the position it would occupy). The shift is
                clamped so the anchor never falls behind the flush frontier, which keeps
                the output coordinate sorted.
                """
                if not normalize or refwin is None or not seq:
                    return seq, first1
                budget = flush_margin
                while budget > 0 and first1 > 1 and (first1 - 1) > state["win_start"]:
                    prev = refwin.base(state["chrom"], first1 - 1)
                    if not prev or prev != seq[-1]:
                        break
                    seq = prev + seq[:-1]
                    first1 -= 1
                    budget -= 1
                return seq, first1

            def anchor_base(anchor1, refspan, span_start1):
                if refwin is not None:
                    base = refwin.base(state["chrom"], anchor1)
                    if base:
                        return base
                off = anchor1 - span_start1
                if refspan is not None and 0 <= off < len(refspan):
                    value = refspan[off]
                    if value:
                        return chr(value) if isinstance(value, int) else value
                return None

            iterator = bam_file.fetch(until_eof=True)
            if progress:
                iterator = tqdm(iterator, desc="bam_to_vcf")

            for read in iterator:
                stats["reads_total"] += 1
                if max_reads and stats["reads_total"] > max_reads:
                    stats["reads_total"] -= 1
                    break
                flag = read.flag
                if flag & exclude_flags:
                    stats["reads_skipped_flag"] += 1
                    continue
                if require_flags and (flag & require_flags) != require_flags:
                    stats["reads_skipped_flag"] += 1
                    continue
                cigartuples = read.cigartuples
                if read.reference_id < 0 or not cigartuples:
                    stats["reads_skipped_flag"] += 1
                    continue
                if read.mapping_quality < min_mapq:
                    stats["reads_skipped_mapq"] += 1
                    continue

                chrom = bam_file.get_reference_name(read.reference_id)
                start1 = read.reference_start + 1
                end0 = read.reference_end if read.reference_end is not None else read.reference_start + 1
                if bed is not None and not bed.overlaps(chrom, read.reference_start, end0):
                    stats["reads_skipped_region"] += 1
                    continue
                stats["reads_used"] += 1

                if read.reference_id != state["tid"]:
                    drain()
                    alleles.clear()
                    depth.reset(start1)
                    state["tid"] = read.reference_id
                    state["chrom"] = chrom
                    state["out_chrom"] = _display_contig(chrom, strip_version_numbers)
                    state["win_start"] = start1
                else:
                    emit_upto(start1 - flush_margin)

                span_len = sum(length for op, length in cigartuples if op in (0, 2, 3, 7, 8))
                if span_len <= 0:
                    continue
                has_indel = any(op in (1, 2) for op, _ in cigartuples)

                # Fast path: NM==0 with no indels means the read matches the reference
                # exactly, so it contributes depth only.
                depth_only = False
                if not has_indel:
                    try:
                        depth_only = read.get_tag("NM") == 0
                    except KeyError:
                        depth_only = False

                query = read.query_sequence
                quals = read.query_qualities
                # refspan is kept as bytes throughout so the mismatch scan can be vectorised.
                query_bytes = query.encode() if query is not None else None
                refspan = None
                if depth_only:
                    stats["reads_fast_path"] += 1
                elif query is None:
                    depth_only = True
                    stats["reads_without_ref_bases"] += 1
                else:
                    if refwin is not None:
                        got = refwin.span(chrom, start1, span_len)
                        if got is not None:
                            refspan = got.encode()
                        else:
                            stats["reads_no_ref"] += 1
                    if refspan is None:
                        try:
                            md = read.get_tag("MD")
                        except KeyError:
                            md = None
                        if md is None:
                            stats["reads_without_ref_bases"] += 1
                            depth_only = True
                        else:
                            stats["reads_with_md"] += 1
                            refspan = _refspan_from_md(md, cigartuples, query_bytes, span_len)
                            if refspan is None:
                                stats["reads_bad_md"] += 1
                                depth_only = True

                rpos1 = start1
                qpos = 0
                for op, length in cigartuples:
                    if op in (0, 7, 8):  # M / = / X
                        depth.add(rpos1, length)
                        if not depth_only:
                            base_off = rpos1 - start1
                            # Compare the whole block at once; only differing offsets reach
                            # Python, so a perfectly matching block costs one numpy call.
                            ref_arr = np.frombuffer(refspan, dtype=np.uint8, count=length, offset=base_off)
                            qry_arr = np.frombuffer(query_bytes, dtype=np.uint8, count=length, offset=qpos)
                            for k in np.nonzero(ref_arr != qry_arr)[0].tolist():
                                ref_base = chr(ref_arr[k])
                                if ref_base == "N" or ref_base == "\x00":
                                    continue
                                alt_base = chr(qry_arr[k])
                                if alt_base == "N" or alt_base == "=":
                                    continue
                                if min_baseq and quals is not None and quals[qpos + k] < min_baseq:
                                    continue
                                add_allele(rpos1 + k, ref_base, alt_base, "snv")
                        rpos1 += length
                        qpos += length
                    elif op == 1:  # I
                        low_quality = (
                            min_baseq and quals is not None and min(quals[qpos : qpos + length]) < min_baseq
                        )
                        if not depth_only and not low_quality and not skip_indels:
                            inserted = query[qpos : qpos + length].upper()
                            inserted, first1 = left_shift(inserted, rpos1)
                            anchor1 = first1 - 1
                            anchor = anchor_base(anchor1, refspan, start1) if anchor1 >= 1 else None
                            if anchor is None:
                                stats["skipped_no_anchor"] += 1
                            else:
                                add_allele(anchor1, anchor, anchor + inserted, "ins")
                        qpos += length
                    elif op == 2:  # D
                        depth.add(rpos1, length)  # the read spans deleted bases
                        if not depth_only and not skip_indels:
                            off = rpos1 - start1
                            deleted = refspan[off : off + length].decode("ascii", "replace")
                            if len(deleted) == length and "\x00" not in deleted and "N" not in deleted:
                                deleted, first1 = left_shift(deleted, rpos1)
                                anchor1 = first1 - 1
                                anchor = anchor_base(anchor1, refspan, start1) if anchor1 >= 1 else None
                                if anchor is None:
                                    stats["skipped_no_anchor"] += 1
                                else:
                                    add_allele(anchor1, anchor + deleted, anchor, "del")
                        rpos1 += length
                    elif op == 3:  # N -- spliced gap: a reference skip, not a deletion
                        rpos1 += length
                    elif op == 4:  # S -- soft clip, present in query_sequence
                        qpos += length
                    # 5 (H) and 6 (P) consume neither the read nor the reference

            drain()  # the final contig
        finally:
            writer.close()
    finally:
        bam_file.close()
        if fasta is not None:
            fasta.close()

    if stats["reads_used"] and stats["reads_without_ref_bases"] == stats["reads_used"] - stats["reads_fast_path"] \
            and not stats["reads_with_md"] and reference is None:
        raise ValueError(
            f"no read in '{bam}' carries an MD tag, so mismatches and deletions cannot be identified. "
            "Pass reference=<ref.fa>, re-align emitting MD (for STAR: --outSAMattributes NH HI AS nM MD), "
            f"or add it with `samtools calmd -b {bam} <reference.fa> > with_md.bam`."
        )
    if stats["reads_without_ref_bases"]:
        logger.warning(
            "%d of %d reads lacked usable reference bases and were skipped",
            stats["reads_without_ref_bases"], stats["reads_used"],
        )
    if stats["reads_bad_md"]:
        logger.warning("%d reads had a malformed MD tag and were skipped", stats["reads_bad_md"])
    if stats["reads_no_ref"]:
        logger.warning("%d reads mapped to contigs absent from the reference FASTA", stats["reads_no_ref"])

    if index:
        pysam.tabix_index(output, preset="vcf", force=True)

    logger.info("bam_to_vcf wrote %d records from %d reads to %s", stats["records_emitted"], stats["reads_used"], output)
    return stats
