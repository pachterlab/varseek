"""Helper functions for varseek denovo."""

import gzip
import os
import re
import shutil
import subprocess
import logging
from collections import Counter

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


def parse_cigars(bam_path=None, total=None, out_plot=None, do_baq=False, regions=None, min_threshold=3, strip_version_numbers=False, out_dataframe=None, logger=logger):
    import pyranges as pr
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

    for read in tqdm(bam, total=total):
        if read.is_unmapped:
            continue
        if all(op in {3, 4, 5, 6, 7} for op, _ in read.cigartuples):  # skip reads without difference from reference
            continue

        chrom = bam.get_reference_name(read.reference_id)
        seq = read.query_sequence
        ref_pos = read.reference_start + 1
        read_pos = 0
        md = read.get_tag("MD") if read.has_tag("MD") else None

        # --- Mismatches and deletions (from MD) ---
        if md:
            tokens = re.findall(r'(\d+)|(\^[A-Z]+)|([A-Z])', md)
            for num, deletion, mismatch in tokens:
                if num:
                    n = int(num)
                    ref_pos += n
                    read_pos += n
                elif deletion:
                    # Deletion from reference
                    deleted_bases = deletion[1:]
                    if len(deleted_bases) == 1:
                        hgvs = f"{chrom}:g.{ref_pos}del"  # {deleted_bases}"
                    else:
                        hgvs = f"{chrom}:g.{ref_pos}_{ref_pos+len(deleted_bases)-1}del"  # {deleted_bases}"
                    variant_counter[hgvs] += 1
                    ref_pos += len(deleted_bases)
                elif mismatch:
                    # SNV
                    ref_base = mismatch
                    alt_base = seq[read_pos]
                    hgvs = f"{chrom}:g.{ref_pos}{ref_base}>{alt_base}"
                    variant_counter[hgvs] += 1
                    ref_pos += 1
                    read_pos += 1

        # --- Insertions (from CIGAR) ---
        for op, length in read.cigartuples:
            if op in (0, 7, 8):  # M, =, X
                ref_pos += length
                read_pos += length
            elif op == 1:  # Insertion
                ins_seq = seq[read_pos:read_pos + length]
                hgvs = f"{chrom}:g.{ref_pos}_{ref_pos+1}ins{ins_seq}"
                variant_counter[hgvs] += 1
                read_pos += length
            elif op in (2, 3):  # Deletion (already handled by MD) or skipped region (N - e.g., intron, functionally similar to deletion)
                ref_pos += length
            elif op == 4:  # soft clipping
                read_pos += length
            # elif op in (5, 6):  # hard clipping or padding
                # pass  # do nothing

        # if len(variant_counter) >= limit:
        #     break

    # for variant, count in variant_counter.most_common(10):
    #     print(variant, count)

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
    logger.info("Total unique variants found:", len(df))

    df[["Chromosome", "Variant"]] = df["key"].str.split(":", n=1, expand=True)  # Split key like "1:g.17746delA" into two columns

    if regions:
        df["pos"] = pd.to_numeric(df["Variant"].str.extract(r"g\.(\d+)")[0])
        df["Start"] = df["pos"] - 1  # PyRanges is half-open: [start, end)
        df["End"] = df["pos"]

        # Convert both to PyRanges
        variants_pr = pr.PyRanges(df[["Chromosome", "Variant", "Start", "End", "key", "Count"]])
        bed_pr = pr.read_bed(regions)

        # Interval intersection (fast!)
        df = variants_pr.join(bed_pr).as_df()
        df = df.drop_duplicates(subset=["Chromosome", "Variant"]).reset_index(drop=True)
        logger.info("Total unique variants after region filtering:", len(df))

    df = df[["Chromosome", "Variant", "Count"]]

    if strip_version_numbers:
        df["Chromosome"] = df["Chromosome"].str.replace(r"\.[0-9]+$", "", regex=True)

    if min_threshold:
        df = df.loc[df["Count"] >= min_threshold].reset_index(drop=True)
        logger.info("Total unique variants after applying min_threshold:", len(df))

    if out_dataframe:
        df.to_csv(out_dataframe, index=False)

    logger.info(f"Final unique variants: {len(df)}")
