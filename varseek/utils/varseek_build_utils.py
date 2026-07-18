import os
import re
import bisect
import gzip
import json
import shutil
import tarfile
import tempfile
import time
from pathlib import Path
from typing import get_args
from collections import OrderedDict, defaultdict

import subprocess
import numpy as np
import gget

import pandas as pd
import pyfastx
from tqdm import tqdm

from varseek.constants import codon_to_amino_acid, mutation_pattern, supported_databases_and_corresponding_reference_sequence_type, complement, fasta_extensions, varseek_ref_only_allowable_kb_ref_arguments, species_to_url
from varseek.utils.logger_utils import set_up_logger, download_box_url, is_program_installed
from varseek.utils.seq_utils import fasta_to_fastq, make_intergenic_fasta, make_transcriptome_fasta, make_cdna_fasta, make_nascent_fasta, create_identity_t2g, reverse_complement, gene_name_lookups_from_gtf
from varseek.utils.type_utils import ReferenceType

import logging

logger = logging.getLogger(__name__)
logger = set_up_logger(logger, logging_level="INFO", save_logs=False, log_dir=None)

tqdm.pandas()


def convert_chromosome_value_to_int_when_possible(val):
    try:
        # Try to convert the value to a float, then to an int, and finally to a string
        return str(int(float(val)))
    except ValueError:
        # If conversion fails, keep the value as it is
        return str(val)


def _gene_name_for_position(chrom_index, pos):
    """Return the gene name whose interval contains ``pos`` (1-based), or "" if none.

    ``chrom_index`` is the ``(starts, ends, names, max_end)`` tuple produced by
    :func:`_build_chrom_gene_index` for a single chromosome. Uses a binary search on the
    (start-sorted) intervals plus a running max-end so the common "position falls in no gene"
    case returns in O(log G), while overlapping/nested genes are still resolved correctly.
    """
    starts, ends, names, max_end = chrom_index
    hi = bisect.bisect_right(starts, pos)  # intervals [0:hi] have start <= pos
    if hi == 0 or max_end[hi - 1] < pos:
        return ""
    for i in range(hi - 1, -1, -1):
        if max_end[i] < pos:  # max_end is non-decreasing: nothing earlier can contain pos
            break
        if ends[i] >= pos:
            return names[i]
    return ""


def _build_chrom_gene_index(intervals):
    """Turn a start-sorted list of ``(start, end, name)`` into a fast per-chromosome lookup."""
    starts = [interval[0] for interval in intervals]
    ends = [interval[1] for interval in intervals]
    names = [interval[2] for interval in intervals]
    max_end = []
    running = 0
    for end in ends:
        running = end if end > running else running
        max_end.append(running)
    return starts, ends, names, max_end


def compute_gene_name_series_for_headers(mutations, seq_id_column, var_column, gtf, lookups=None):
    """Map each variant to its gene name (symbol) using a GTF, for HGVS-header annotation.

    Transcript/cDNA/CDS variants (seq_ID like ``ENST...``) are mapped ENST -> gene_name; genome
    variants (chromosome seq_ID with a ``g.`` mutation) are mapped chromosome+position -> gene_name
    via the GTF gene intervals. Variants with no gene match get an empty string (they are written
    without a gene annotation).

    ``lookups`` optionally supplies an already-parsed ``(transcript_to_gene_name, gene_intervals)``
    pair (from :func:`gene_name_lookups_from_gtf`), so callers that annotate in several passes can
    parse the (potentially multi-GB) GTF once instead of on every call. When None it is parsed here.

    Returns a pandas Series of gene names aligned to ``mutations.index``.
    """
    transcript_to_gene_name, gene_intervals = lookups if lookups is not None else gene_name_lookups_from_gtf(gtf)

    seq_ids = mutations[seq_id_column].astype(str).str.split(".").str[0]  # strip any version suffix
    variants = mutations[var_column].astype(str)
    gene_names = pd.Series("", index=mutations.index, dtype=object)

    # Transcript/cDNA variants: ENST -> gene_name via the transcript->gene map.
    is_transcript = seq_ids.str.startswith("ENST")
    if is_transcript.any():
        gene_names.loc[is_transcript] = seq_ids[is_transcript].map(transcript_to_gene_name).fillna("")

    # Genome variants: chromosome + position -> gene_name via interval lookup.
    is_genome = (~is_transcript) & variants.str.startswith("g.")
    if is_genome.any() and gene_intervals:
        chrom_indexes = {chrom: _build_chrom_gene_index(intervals) for chrom, intervals in gene_intervals.items()}
        positions = variants[is_genome].str.extract(r"g\.(\d+)")[0]
        for idx, chrom, pos in zip(positions.index, seq_ids[is_genome], positions):
            if pos is None or (isinstance(pos, float) and np.isnan(pos)):
                continue
            chrom_index = chrom_indexes.get(str(chrom))
            if chrom_index is None:
                continue
            gene_name = _gene_name_for_position(chrom_index, int(pos))
            if gene_name:
                gene_names.at[idx] = gene_name

    return gene_names


# Function to ensure unique IDs
def generate_unique_ids(num_ids, start=1, total_rows=None):
    if total_rows is None:
        total_rows = num_ids
    num_digits = len(str(total_rows + start - 1))
    generated_ids = [f"vcrs_{i:0{num_digits}}" for i in range(start, start + num_ids)]
    return list(generated_ids)

def translate_sequence(sequence, start=0, end=None):
    if end is None:  # If end is not provided, set it to the length of the sequence
        end = len(sequence)
    
    amino_acid_sequence = ""
    for i in range(start, end, 3):
        codon = sequence[i : (i + 3)].upper()
        amino_acid = codon_to_amino_acid.get(codon, "X")  # Use 'X' for unknown codons
        amino_acid_sequence += amino_acid

    amino_acid_sequence = amino_acid_sequence.strip("X")  # Remove leading and trailing 'X's
    return amino_acid_sequence


def wt_fragment_and_mutant_fragment_share_kmer(mutated_fragment: str, wildtype_fragment: str, k: int) -> bool:
    if len(mutated_fragment) <= k:
        return bool(mutated_fragment in wildtype_fragment)

    # else:
    for mutant_position in range(len(mutated_fragment) - k):
        mutant_kmer = mutated_fragment[mutant_position : (mutant_position + k)]
        if mutant_kmer in wildtype_fragment:
            # wt_position = wildtype_fragment.find(mutant_kmer)
            return True
    return False


def return_pyfastx_index_object_with_header_versions_removed(fasta_path):
    logger.info(f"Removing version numbers in fasta headers for {fasta_path}")
    fa_read_only = pyfastx.Fastx(fasta_path)
    with tempfile.NamedTemporaryFile(mode="w", suffix=".fa", encoding="utf-8", delete=True) as temp_fasta:
        temp_fasta_path = temp_fasta.name
        temp_fasta_index_path = temp_fasta_path + ".fxi"
        for name, seq in fa_read_only:
            name_without_version = name.split(".")[0]
            temp_fasta.write(f">{name_without_version}\n{seq}\n")
        temp_fasta.flush()
        logger.info(f"Building pyfastx index for {fasta_path}")
        fa = pyfastx.Fasta(temp_fasta_path, build_index=True)
    return fa, temp_fasta_index_path


# Helper function to find starting position of CDS in cDNA
def find_cds_position(cdna_seq, cds_seq):
    pos = cdna_seq.find(cds_seq)
    return pos if pos != -1 else None


def count_leading_Ns(seq):
    return len(seq) - len(seq.lstrip("N"))


def convert_mutation_cds_locations_to_cdna(input_csv_path, cdna_fasta_path, cds_fasta_path, output_csv_path=None, verbose=True, strip_leading_Ns_cds=False):
    # Load the CSV
    if isinstance(input_csv_path, str):
        logger.info(f"Loading CSV from {input_csv_path}")
        df = pd.read_csv(input_csv_path)
    elif isinstance(input_csv_path, pd.DataFrame):
        df = input_csv_path.copy()
    else:
        raise ValueError("input_csv_path must be a string or a pandas DataFrame")

    logger.info("Copying df internally to avoid in-place modifications")
    df_original = df.copy()

    bad_mutations_dict = {}

    logger.info("Removing unknown mutations")
    # get rids of mutations that are uncertain, ambiguous, intronic, posttranslational
    uncertain_mutations = df["mutation"].str.contains(r"\?").sum()

    ambiguous_position_mutations = df["mutation"].str.contains(r"\(|\)").sum()

    intronic_mutations = df["mutation"].str.contains(r"\+|\-").sum()

    posttranslational_region_mutations = df["mutation"].str.contains(r"\*").sum()

    logger.info("Removing unsupported mutation types")
    logger.info(f"Uncertain mutations: {uncertain_mutations}")
    logger.info(f"Ambiguous position mutations: {ambiguous_position_mutations}")
    logger.info(f"Intronic mutations: {intronic_mutations}")
    logger.info(f"Posttranslational region mutations: {posttranslational_region_mutations}")

    bad_mutations_dict["uncertain_mutations"] = uncertain_mutations
    bad_mutations_dict["ambiguous_position_mutations"] = ambiguous_position_mutations
    bad_mutations_dict["intronic_mutations"] = intronic_mutations
    bad_mutations_dict["posttranslational_region_mutations"] = posttranslational_region_mutations

    # Filter out bad mutations
    combined_pattern = re.compile(r"(\?|\(|\)|\+|\-|\*)")  # gets rids of mutations that are uncertain, ambiguous, intronic, posttranslational
    mask = df["mutation"].str.contains(combined_pattern)
    df = df[~mask]

    logger.info("Sorting df")
    df = df.sort_values(by="seq_ID")  # to make iterrows more efficient

    logger.info("Determining mutation positions")
    df[["nucleotide_positions", "actual_variant"]] = df["mutation"].str.extract(mutation_pattern)

    split_positions = df["nucleotide_positions"].str.split("_", expand=True)

    df["start_variant_position"] = split_positions[0]
    if split_positions.shape[1] > 1:
        df["end_variant_position"] = split_positions[1].fillna(split_positions[0])
    else:
        df["end_variant_position"] = df["start_variant_position"]

    df.loc[df["end_variant_position"].isna(), "end_variant_position"] = df["start_variant_position"]

    df[["start_variant_position", "end_variant_position"]] = df[["start_variant_position", "end_variant_position"]].astype(int)

    # # Rename the mutation column
    # df.rename(columns={"mutation": "mutation_cds"}, inplace=True)

    temp_fasta_index_path_cdna, temp_fasta_index_path_cds = None, None  # in case the block fails

    # put in try-except-finally block to ensure that the temp index files are erased no matter what
    try:
        # Load the FASTA files
        fa_cdna, temp_fasta_index_path_cdna = return_pyfastx_index_object_with_header_versions_removed(cdna_fasta_path)
        fa_cds, temp_fasta_index_path_cds = return_pyfastx_index_object_with_header_versions_removed(cds_fasta_path)

        number_bad = 0
        seq_id_previous = None

        iterator = tqdm(df.iterrows(), total=len(df), desc="Processing rows") if verbose else df.iterrows()

        # Process each row
        for index, row in iterator:
            seq_id = row["seq_ID"]

            if seq_id != seq_id_previous:
                if seq_id in fa_cdna and seq_id in fa_cds:
                    cdna_seq = fa_cdna[seq_id].seq
                    cds_seq = fa_cds[seq_id].seq
                    number_of_leading_ns = count_leading_Ns(cds_seq)  # modified in April 2025 - used to be count_leading_Ns(cdna_seq)
                    cds_seq = cds_seq.strip("N")
                    cds_start_pos = find_cds_position(cdna_seq, cds_seq)
                    seq_id_found_in_cdna_and_cds = True
                else:
                    seq_id_found_in_cdna_and_cds = False

            if (not seq_id_found_in_cdna_and_cds) or (cds_start_pos is None):
                df.at[index, "mutation_cdna"] = None
                number_bad += 1
            else:
                if strip_leading_Ns_cds:
                    df.at[index, "start_variant_position"] += cds_start_pos
                    df.at[index, "end_variant_position"] += cds_start_pos
                else:
                    df.at[index, "start_variant_position"] += cds_start_pos - number_of_leading_ns
                    df.at[index, "end_variant_position"] += cds_start_pos - number_of_leading_ns

                start = df.at[index, "start_variant_position"]
                end = df.at[index, "end_variant_position"]
                actual_variant = row["actual_variant"]

                if start == end:
                    df.at[index, "mutation_cdna"] = f"c.{start}{actual_variant}"
                else:
                    df.at[index, "mutation_cdna"] = f"c.{start}_{end}{actual_variant}"

            seq_id_previous = seq_id

        logger.info(f"Number of bad mutations: {number_bad}")
        logger.info("Merging dfs")

        if (df_original.duplicated(subset=["seq_ID", "mutation"]).sum() == 0) and (df.duplicated(subset=["seq_ID", "mutation"]).sum() == 0):  # this condition should be True if downloading with default gget cosmic, but in case the user wants duplicate rows then I'll give both options
            df_merged = df_original.set_index(["seq_ID", "mutation"]).join(df.set_index(["seq_ID", "mutation"])[["mutation_cdna"]], how="left").reset_index()
        else:
            df_merged = df_original.merge(df[["seq_ID", "mutation", "mutation_cdna"]], on=["seq_ID", "mutation"], how="left")  # new as of Feb 2025

        # Write to new CSV
        # if not output_csv_path:
        #     output_csv_path = input_csv_path
        if output_csv_path:
            logger.info(f"Saving output to {output_csv_path}")
            df_merged.to_csv(output_csv_path, index=False)  # new as of Feb 2025 (replaced df.to_csv with df_merged.to_csv)

        return df_merged, bad_mutations_dict

    except Exception as e:
        raise RuntimeError(f"Error converting CDS to cDNA: {e}") from e
    finally:
        logger.info("Cleaning up temporary files...")
        for temp_path in [temp_fasta_index_path_cdna, temp_fasta_index_path_cds]:
            if temp_path and os.path.exists(temp_path):
                os.remove(temp_path)


def convert_mutation_cds_locations_to_cdna_old(input_csv_path, cdna_fasta_path, cds_fasta_path, output_csv_path=None, verbose=True):
    # Load the CSV
    if isinstance(input_csv_path, str):
        logger.info(f"Loading CSV from {input_csv_path}")
        df = pd.read_csv(input_csv_path)
    elif isinstance(input_csv_path, pd.DataFrame):
        df = input_csv_path.copy()
    else:
        raise ValueError("input_csv_path must be a string or a pandas DataFrame")

    logger.info("Copying df internally to avoid in-place modifications")
    df_original = df.copy()

    bad_mutations_dict = {}

    logger.info("Removing unknown mutations")
    # get rids of mutations that are uncertain, ambiguous, intronic, posttranslational
    uncertain_mutations = df["mutation"].str.contains(r"\?").sum()

    ambiguous_position_mutations = df["mutation"].str.contains(r"\(|\)").sum()

    intronic_mutations = df["mutation"].str.contains(r"\+|\-").sum()

    posttranslational_region_mutations = df["mutation"].str.contains(r"\*").sum()

    logger.info("Removing unsupported mutation types")
    logger.info(f"Uncertain mutations: {uncertain_mutations}")
    logger.info(f"Ambiguous position mutations: {ambiguous_position_mutations}")
    logger.info(f"Intronic mutations: {intronic_mutations}")
    logger.info(f"Posttranslational region mutations: {posttranslational_region_mutations}")

    bad_mutations_dict["uncertain_mutations"] = uncertain_mutations
    bad_mutations_dict["ambiguous_position_mutations"] = ambiguous_position_mutations
    bad_mutations_dict["intronic_mutations"] = intronic_mutations
    bad_mutations_dict["posttranslational_region_mutations"] = posttranslational_region_mutations

    # Filter out bad mutations
    combined_pattern = re.compile(r"(\?|\(|\)|\+|\-|\*)")  # gets rids of mutations that are uncertain, ambiguous, intronic, posttranslational
    mask = df["mutation"].str.contains(combined_pattern)
    df = df[~mask]

    logger.info("Sorting df")
    df = df.sort_values(by="seq_ID")  # to make iterrows more efficient

    logger.info("Determining mutation positions")
    df[["nucleotide_positions", "actual_variant"]] = df["mutation"].str.extract(mutation_pattern)

    split_positions = df["nucleotide_positions"].str.split("_", expand=True)

    df["start_variant_position"] = split_positions[0]
    if split_positions.shape[1] > 1:
        df["end_variant_position"] = split_positions[1].fillna(split_positions[0])
    else:
        df["end_variant_position"] = df["start_variant_position"]

    df.loc[df["end_variant_position"].isna(), "end_variant_position"] = df["start_variant_position"]

    df[["start_variant_position", "end_variant_position"]] = df[["start_variant_position", "end_variant_position"]].astype(int)

    # # Rename the mutation column
    # df.rename(columns={"mutation": "mutation_cds"}, inplace=True)

    temp_fasta_index_path_cdna, temp_fasta_index_path_cds = None, None  # in case the block fails

    # put in try-except-finally block to ensure that the temp index files are erased no matter what
    try:
        # Load the FASTA files
        fa_cdna, temp_fasta_index_path_cdna = return_pyfastx_index_object_with_header_versions_removed(cdna_fasta_path)
        fa_cds, temp_fasta_index_path_cds = return_pyfastx_index_object_with_header_versions_removed(cds_fasta_path)

        number_bad = 0
        seq_id_previous = None

        iterator = tqdm(df.iterrows(), total=len(df), desc="Processing rows") if verbose else df.iterrows()

        # Process each row
        for index, row in iterator:
            seq_id = row["seq_ID"]

            if seq_id != seq_id_previous:
                if seq_id in fa_cdna and seq_id in fa_cds:
                    cdna_seq = fa_cdna[seq_id].seq
                    cds_seq = fa_cds[seq_id].seq
                    cds_seq = cds_seq.strip("N")
                    cds_start_pos = find_cds_position(cdna_seq, cds_seq)
                    seq_id_found_in_cdna_and_cds = True
                else:
                    seq_id_found_in_cdna_and_cds = False

            if (not seq_id_found_in_cdna_and_cds) or (cds_start_pos is None):
                df.at[index, "mutation_cdna"] = None
                number_bad += 1
            else:
                df.at[index, "start_variant_position"] += cds_start_pos
                df.at[index, "end_variant_position"] += cds_start_pos

                start = df.at[index, "start_variant_position"]
                end = df.at[index, "end_variant_position"]
                actual_variant = row["actual_variant"]

                if start == end:
                    df.at[index, "mutation_cdna"] = f"c.{start}{actual_variant}"
                else:
                    df.at[index, "mutation_cdna"] = f"c.{start}_{end}{actual_variant}"

            seq_id_previous = seq_id

        logger.info(f"Number of bad mutations: {number_bad}")
        logger.info("Merging dfs")

        if (df_original.duplicated(subset=["seq_ID", "mutation"]).sum() == 0) and (df.duplicated(subset=["seq_ID", "mutation"]).sum() == 0):  # this condition should be True if downloading with default gget cosmic, but in case the user wants duplicate rows then I'll give both options
            df_merged = df_original.set_index(["seq_ID", "mutation"]).join(df.set_index(["seq_ID", "mutation"])[["mutation_cdna"]], how="left").reset_index()
        else:
            df_merged = df_original.merge(df[["seq_ID", "mutation", "mutation_cdna"]], on=["seq_ID", "mutation"], how="left")  # new as of Feb 2025

        # Write to new CSV
        # if not output_csv_path:
        #     output_csv_path = input_csv_path
        if output_csv_path:
            logger.info(f"Saving output to {output_csv_path}")
            df_merged.to_csv(output_csv_path, index=False)  # new as of Feb 2025 (replaced df.to_csv with df_merged.to_csv)

        return df_merged, bad_mutations_dict

    except Exception as e:
        raise RuntimeError(f"Error converting CDS to cDNA: {e}") from e
    finally:
        logger.info("Cleaning up temporary files...")
        for temp_path in [temp_fasta_index_path_cdna, temp_fasta_index_path_cds]:
            if temp_path and os.path.exists(temp_path):
                os.remove(temp_path)


# --- cDNA-vs-genome detection ------------------------------------------------------
# Description-line biotype tokens. cDNA is checked before genome because Ensembl cDNA
# headers also carry a `chromosome:` token (e.g. ">ENST... cdna chromosome:GRCh38:1:...").
_CDNA_TOKEN_RE = re.compile(r"\b(cdna|ncrna|transcript:)", re.IGNORECASE)
_GENOME_TOKEN_RE = re.compile(r"\bdna(_sm|_rm)?:(chromosome|scaffold|supercontig|primary_assembly)|\bchromosome:", re.IGNORECASE)
# First-token accession / ID patterns (most decisive when present).
_ENSEMBL_TRANSCRIPT_RE = re.compile(r"^ENS[A-Z]{0,5}T\d{6,}", re.IGNORECASE)  # e.g. ENST..., ENSMUST...
_REFSEQ_TRANSCRIPT_RE = re.compile(r"^(NM|NR|XM|XR)_\d+", re.IGNORECASE)
_REFSEQ_GENOMIC_RE = re.compile(r"^(NC|NT|NW|NG|NZ)_\d+", re.IGNORECASE)
_CHROM_NAME_RE = re.compile(r"^(chr)?([0-9]{1,2}|[XYWZ]|MT?)$", re.IGNORECASE)  # 1, chr1, X, MT, ...


def _iter_fasta_headers(fasta_path):
    """Yield fasta header lines (without the leading '>'), reading only header lines.

    Avoids parsing sequence bodies entirely; transparently handles gzipped fastas.
    """
    open_func = gzip.open if str(fasta_path).endswith(".gz") else open
    with open_func(fasta_path, "rt") as fh:
        for line in fh:
            if line.startswith(">"):
                yield line[1:].strip()


def _classify_fasta_header(header):
    """Classify a single header as 'cdna', 'genome', or None (undetermined)."""
    tokens = header.split()
    first = tokens[0] if tokens else header
    if _ENSEMBL_TRANSCRIPT_RE.match(first) or _REFSEQ_TRANSCRIPT_RE.match(first):
        return "cdna"
    if _REFSEQ_GENOMIC_RE.match(first) or _CHROM_NAME_RE.match(first):
        return "genome"
    if _CDNA_TOKEN_RE.search(header):  # check cdna before genome (see note above)
        return "cdna"
    if _GENOME_TOKEN_RE.search(header):
        return "genome"
    return None


def _gtf_seqnames_and_transcript_ids(gtf):
    """Return (seqname set, transcript_id set) from a GTF, versions stripped, read in chunks."""
    seqnames, transcript_ids = set(), set()
    for chunk in pd.read_csv(
        gtf,
        sep="\t",
        comment="#",
        header=None,
        usecols=[0, 8],
        names=["seqname", "attribute"],
        dtype=str,
        chunksize=200_000,
    ):
        seqnames.update(chunk["seqname"].dropna().astype(str))
        tids = chunk["attribute"].str.extract(r'transcript_id "([^"]+)"')[0].dropna()
        transcript_ids.update(tids)
    strip_version = lambda s: s.split(".")[0]
    return {strip_version(s) for s in seqnames}, {strip_version(t) for t in transcript_ids}


def fasta_looks_like_cdna(fasta_path, gtf=None, genome_min_seq_len=1_000_000, header_sample_size=200):
    """Heuristically decide whether a fasta is cDNA/transcriptome rather than genomic DNA.

    Detection is layered from most to least reliable, returning as soon as a layer decides:

    1. **Header tokens.** Standard references self-identify in their description lines --
       Ensembl ``dna:chromosome`` vs ``cdna``/``transcript:``; RefSeq ``NC_``/``NW_``
       genomic accessions vs ``NM_``/``NR_``/``XM_`` transcript accessions; Ensembl
       ``ENS...T`` transcript IDs; bare chromosome names (``1``, ``chr1``, ``MT``). The
       first ``header_sample_size`` headers are sampled and a majority vote is taken.
    2. **GTF cross-check.** If ``gtf`` is given, a genome fasta's sequence names match the
       gtf ``seqname`` (chromosome) column, whereas a cDNA fasta's names match its
       ``transcript_id``s. Whichever set the fasta's names overlap more wins.
    3. **Sequence length.** Last resort: stream sequences; a chromosome-scale sequence
       (>= ``genome_min_seq_len``) means genomic DNA, otherwise cDNA. This reads sequence
       bodies, so it only runs when the cheaper header/gtf layers are inconclusive.

    Returns:
        True  if the fasta looks like cDNA/transcriptome,
        False if it looks like genomic DNA,
        None  if the file is empty or could not be inspected (classification skipped).
    """
    # --- Layer 1: header tokens --------------------------------------------------------
    try:
        genome_votes = cdna_votes = 0
        for n, header in enumerate(_iter_fasta_headers(fasta_path)):
            label = _classify_fasta_header(header)
            if label == "genome":
                genome_votes += 1
            elif label == "cdna":
                cdna_votes += 1
            if n + 1 >= header_sample_size:
                break
        if cdna_votes > genome_votes:
            return True
        if genome_votes > cdna_votes:
            return False
    except Exception as exc:  # pylint: disable=broad-except
        logger.debug("Could not read headers of %s to detect cDNA vs genome: %s", fasta_path, exc)

    # --- Layer 2: gtf cross-check ------------------------------------------------------
    if gtf is not None and isinstance(gtf, str) and os.path.isfile(gtf):
        try:
            fasta_ids = set()
            for header in _iter_fasta_headers(fasta_path):
                tokens = header.split()
                if tokens:
                    fasta_ids.add(tokens[0].split(".")[0])  # first token, version stripped
                if len(fasta_ids) >= 5000:
                    break
            seqnames, transcript_ids = _gtf_seqnames_and_transcript_ids(gtf)
            genome_overlap = len(fasta_ids & seqnames)
            cdna_overlap = len(fasta_ids & transcript_ids)
            if genome_overlap or cdna_overlap:
                return cdna_overlap > genome_overlap
        except Exception as exc:  # pylint: disable=broad-except
            logger.debug("Could not cross-check %s against gtf %s: %s", fasta_path, gtf, exc)

    # --- Layer 3: sequence length ------------------------------------------------------
    try:
        saw_sequence = False
        for _, sequence in pyfastx.Fastx(fasta_path):
            saw_sequence = True
            if len(sequence) >= genome_min_seq_len:
                return False  # chromosome-scale sequence -> genomic DNA
        return True if saw_sequence else None
    except Exception as exc:  # pylint: disable=broad-except
        logger.debug("Could not inspect %s to detect cDNA vs genome: %s", fasta_path, exc)
        return None


def find_matching_sequences_through_fasta(file_path, ref_sequence):
    headers_of_matching_sequences = []
    for header, sequence in pyfastx.Fastx(file_path):
        if sequence == ref_sequence:
            headers_of_matching_sequences.append(header)

    return headers_of_matching_sequences


def create_header_to_sequence_ordered_dict_from_fasta_after_semicolon_splitting(
    input_fasta,
):
    mutant_reference = OrderedDict()
    for mutant_reference_header, mutant_reference_sequence in pyfastx.Fastx(input_fasta):
        mutant_reference_header_individual_list = mutant_reference_header.split(";")
        for mutant_reference_header_individual in mutant_reference_header_individual_list:
            mutant_reference[mutant_reference_header_individual] = mutant_reference_sequence
    return mutant_reference


# def merge_genome_into_transcriptome_fasta(
#     mutation_reference_file_fasta_transcriptome,
#     mutation_reference_file_fasta_genome,
#     mutation_reference_file_fasta_combined,
#     cosmic_reference_file_mutation_dataframe,
# ):

#     # TODO: make header fasta from id fasta with id:header dict

#     mutant_reference_transcriptome = create_header_to_sequence_ordered_dict_from_fasta_after_semicolon_splitting(mutation_reference_file_fasta_transcriptome)
#     mutant_reference_genome = create_header_to_sequence_ordered_dict_from_fasta_after_semicolon_splitting(mutation_reference_file_fasta_genome)

#     cosmic_df = pd.read_csv(
#         cosmic_reference_file_mutation_dataframe,
#         usecols=["seq_ID", "mutation_cdna", "chromosome", "mutation_genome"],  # TODO: remove column hard-coding
#     )
#     cosmic_df["chromosome"] = cosmic_df["chromosome"].apply(convert_chromosome_value_to_int_when_possible)

#     mutant_reference_genome_to_keep = OrderedDict()

#     for header_genome, sequence_genome in mutant_reference_genome.items():
#         seq_id_genome, mutation_id_genome = header_genome.split(":", 1)
#         row_corresponding_to_genome = cosmic_df[(cosmic_df["chromosome"] == seq_id_genome) & (cosmic_df["mutation_genome"] == mutation_id_genome)]
#         seq_id_transcriptome_corresponding_to_genome = row_corresponding_to_genome["seq_ID"].iloc[0]
#         mutation_id_transcriptome_corresponding_to_genome = row_corresponding_to_genome["mutation_cdna"].iloc[0]
#         header_transcriptome_corresponding_to_genome = f"{seq_id_transcriptome_corresponding_to_genome}:{mutation_id_transcriptome_corresponding_to_genome}"

#         if header_transcriptome_corresponding_to_genome in mutant_reference_transcriptome:
#             if mutant_reference_transcriptome[header_transcriptome_corresponding_to_genome] != sequence_genome:
#                 header_genome_transcriptome_style = f"unspliced{header_transcriptome_corresponding_to_genome}"  # TODO: change when I change unspliced notation
#                 mutant_reference_genome_to_keep[header_genome_transcriptome_style] = sequence_genome
#         else:
#             header_genome_transcriptome_style = f"unspliced{header_transcriptome_corresponding_to_genome}"  # TODO: change when I change unspliced notation
#             mutant_reference_genome_to_keep[header_genome_transcriptome_style] = sequence_genome

#     mutant_reference_combined = OrderedDict(mutant_reference_transcriptome)
#     mutant_reference_combined.update(mutant_reference_genome_to_keep)

#     mutant_reference_combined = join_keys_with_same_values(mutant_reference_combined)

#     # initialize combined fasta file with transcriptome fasta
#     with open(mutation_reference_file_fasta_combined, "w", encoding="utf-8") as fasta_file:
#         for (
#             header_transcriptome,
#             sequence_transcriptome,
#         ) in mutant_reference_combined.items():
#             # write the header followed by the sequence
#             fasta_file.write(f">{header_transcriptome}\n{sequence_transcriptome}\n")

#     # TODO: make id fasta from header fasta with id:header dict

#     print(f"Combined fasta file created at {mutation_reference_file_fasta_combined}")


def join_keys_with_same_values(original_dict):
    # Step 1: Group keys by their values
    grouped_dict = defaultdict(list)
    for key, value in original_dict.items():
        grouped_dict[value].append(key)

    # Step 2: Create the new OrderedDict with concatenated keys
    concatenated_dict = OrderedDict((";".join(keys), value) for value, keys in grouped_dict.items())

    return concatenated_dict


def download_cosmic_mutations(gtf, gtf_transcript_id_column, reference_out_dir, cosmic_version, cosmic_email, cosmic_password, columns_to_keep, grch, mutations_original, sequences, cds_file, cdna_file, var_id_column, verbose):
    reference_out_cosmic = f"{reference_out_dir}/cosmic"
    if int(cosmic_version) == 100:
        mutations = f"{reference_out_cosmic}/CancerMutationCensus_AllData_Tsv_v{cosmic_version}_GRCh{grch}_v2/CancerMutationCensus_AllData_v{cosmic_version}_GRCh{grch}_mutation_workflow.csv"
    else:
        mutations = f"{reference_out_cosmic}/CancerMutationCensus_AllData_Tsv_v{cosmic_version}_GRCh{grch}/CancerMutationCensus_AllData_v{cosmic_version}_GRCh{grch}_mutation_workflow.csv"
    mutations_path = mutations

    if not os.path.isfile(mutations):  # DO NOT specify column names in gget cosmic - I instead code them in later
        gget.cosmic(
                    None,
                    grch_version=grch,
                    cosmic_version=cosmic_version,
                    out=reference_out_cosmic,
                    cosmic_project="cancer",
                    download_cosmic=True,
                    gget_mutate=True,
                    keep_genome_info=True,
                    remove_duplicates=True,
                    email=cosmic_email,
                    password=cosmic_password,
                )

        mutations = pd.read_csv(mutations_path)

        if gtf:
            if os.path.isfile(gtf):
                mutations = merge_gtf_transcript_locations_into_cosmic_csv(mutations, gtf, gtf_transcript_id_column=gtf_transcript_id_column, output_mutations_path=mutations_path)
                columns_to_keep.extend(
                            [
                                "start_transcript_position",
                                "end_transcript_position",
                                "strand",
                            ]
                        )

                if "CancerMutationCensus" in mutations_original or mutations_original == "cosmic_cmc":
                    logger.info("COSMIC CMC genome strand information is not fully accurate. Improving with gtf information.")
                    mutations = improve_genome_strand_information(mutations, mutation_genome_column_name="mutation_genome", output_mutations_path=mutations_path)
            else:
                raise ValueError(f"gtf file '{gtf}' does not exist.")
    else:
        mutations = pd.read_csv(mutations_path)

    if sequences == "cdna" or sequences.endswith(supported_databases_and_corresponding_reference_sequence_type[mutations_original]["sequence_file_names"]["cdna"].replace("GRCH_NUMBER", grch)):  # covers whether sequences == "cdna" or sequences == "PATH/TO/Homo_sapiens.GRCh37.cdna.all.fa"
        if "mutation_cdna" not in mutations.columns:
            logger.info("Adding in cdna information into COSMIC. Note that the 'mutation_cdna' column will refer to cDNA mutation notation, and the 'mutation' column will refer to CDS mutation notation.")
            mutations, bad_cosmic_mutations_dict = convert_mutation_cds_locations_to_cdna(input_csv_path=mutations, output_csv_path=mutations_path, cds_fasta_path=cds_file, cdna_fasta_path=cdna_file, verbose=verbose, strip_leading_Ns_cds=True)

            # uncertain_mutations += bad_cosmic_mutations_dict["uncertain_mutations"]
            # ambiguous_position_mutations += bad_cosmic_mutations_dict["ambiguous_position_mutations"]
            # intronic_mutations += bad_cosmic_mutations_dict["intronic_mutations"]
            # posttranslational_region_mutations += bad_cosmic_mutations_dict["posttranslational_region_mutations"]

        seq_id_column = supported_databases_and_corresponding_reference_sequence_type[mutations_original]["column_names"]["seq_id_cdna_column"]  # "seq_ID"
        var_column = supported_databases_and_corresponding_reference_sequence_type[mutations_original]["column_names"]["var_cdna_column"]  # "mutation_cdna"

    elif sequences == "genome" or sequences.endswith(supported_databases_and_corresponding_reference_sequence_type[mutations_original]["sequence_file_names"]["genome"].replace("GRCH_NUMBER", grch)):  # covers whether sequences == "genome" or sequences == "PATH/TO/Homo_sapiens.GRCh37.dna.primary_assembly.fa"
        mutations_path_no_duplications = mutations_path.replace(".csv", "_no_duplications.csv")
        if not os.path.isfile(mutations_path_no_duplications):
            logger.info("COSMIC genome location is not accurate for duplications. Dropping duplications in a copy of the csv file.")
            mutations = drop_duplication_mutations(mutations, mutations_path_no_duplications, logger)  # COSMIC incorrectly records genome positions of duplications
        else:
            mutations = pd.read_csv(mutations_path_no_duplications)

        seq_id_column = supported_databases_and_corresponding_reference_sequence_type[mutations_original]["column_names"]["seq_id_genome_column"]  # "chromosome"
        var_column = supported_databases_and_corresponding_reference_sequence_type[mutations_original]["column_names"]["var_genome_column"]  # "mutation_genome"

    elif sequences == "cds" or sequences.endswith(supported_databases_and_corresponding_reference_sequence_type[mutations_original]["sequence_file_names"]["cds"].replace("GRCH_NUMBER", grch)):  # covers whether sequences == "cds" or sequences == "PATH/TO/Homo_sapiens.GRCh37.cds.all.fa"
        seq_id_column = supported_databases_and_corresponding_reference_sequence_type[mutations_original]["column_names"]["seq_id_cds_column"]  # "seq_ID"
        var_column = supported_databases_and_corresponding_reference_sequence_type[mutations_original]["column_names"]["var_cds_column"]  # "mutation"

    var_id_column = supported_databases_and_corresponding_reference_sequence_type[mutations_original]["column_names"]["var_id_column"] if var_id_column is not None else None  # use the id column if the user wanted to; otherwise keep as default
    return mutations,mutations_path,seq_id_column,var_column,var_id_column,columns_to_keep


def download_cosmic_sequences(sequences, seq_id_column, gtf, gtf_transcript_id_column, reference_out_dir, cosmic_version, mutations, grch, logger):
    if grch == "37":
        gget_ref_species = "human_grch37"
    elif grch == "38":
        gget_ref_species = "human"
    else:
        gget_ref_species = grch
    
    ensembl_version = supported_databases_and_corresponding_reference_sequence_type[mutations]["database_version_to_reference_release"][cosmic_version]
    reference_out_sequences = f"{reference_out_dir}/ensembl_grch{grch}_release{ensembl_version}"  # matches vk info

    sequences_download_command = supported_databases_and_corresponding_reference_sequence_type[mutations]["sequence_download_commands"][sequences]
    sequences_download_command = sequences_download_command.replace("OUT_DIR", reference_out_sequences)
    sequences_download_command = sequences_download_command.replace("ENSEMBL_VERSION", ensembl_version)
    sequences_download_command = sequences_download_command.replace("SPECIES", gget_ref_species)

    genome_file = supported_databases_and_corresponding_reference_sequence_type[mutations]["sequence_file_names"]["genome"]
    genome_file = genome_file.replace("GRCH_NUMBER", grch)
    genome_file = f"{reference_out_sequences}/{genome_file}"

    gtf_file = supported_databases_and_corresponding_reference_sequence_type[mutations]["sequence_file_names"]["gtf"]
    gtf_file = gtf_file.replace("GRCH_NUMBER", grch)
    gtf_file = f"{reference_out_sequences}/{gtf_file}"

    cds_file = supported_databases_and_corresponding_reference_sequence_type[mutations]["sequence_file_names"]["cds"]
    cds_file = cds_file.replace("GRCH_NUMBER", grch)
    cds_file = f"{reference_out_sequences}/{cds_file}"

    cdna_file = supported_databases_and_corresponding_reference_sequence_type[mutations]["sequence_file_names"]["cdna"]
    cdna_file = cdna_file.replace("GRCH_NUMBER", grch)
    cdna_file = f"{reference_out_sequences}/{cdna_file}"

    files_to_download_list = []
    if sequences == "genome" and not os.path.isfile(genome_file):
        files_to_download_list.append("dna")
    if gtf and not os.path.isfile(gtf):
        gtf = gtf_file
        gtf_transcript_id_column = seq_id_column
        if not os.path.isfile(gtf):  # now that I have overridden the user-provided gtf with my gtf
            files_to_download_list.append("gtf")
    if (sequences == "cdna" or sequences == "cds") and not os.path.isfile(cds_file):
        files_to_download_list.append("cds")
    if sequences == "cdna" and not os.path.isfile(cdna_file):
        files_to_download_list.append("cdna")

    files_to_download = ",".join(files_to_download_list)
    sequences_download_command = sequences_download_command.replace("FILES_TO_DOWNLOAD", files_to_download)

    sequences_download_command_list = sequences_download_command.split(" ")

    if files_to_download_list:  # means that at least 1 of the necessary files must be downloaded
        logger.warning("Downloading reference sequences with %s. Note that this requires curl >=7.73.0", " ".join(sequences_download_command_list))
        subprocess.run(sequences_download_command_list, check=True)
        if "dna" in files_to_download_list:
            subprocess.run(["gunzip", f"{genome_file}.gz"], check=True)
        if "gtf" in files_to_download_list:
            subprocess.run(["gunzip", f"{gtf_file}.gz"], check=True)
        if "cds" in files_to_download_list:
            subprocess.run(["gunzip", f"{cds_file}.gz"], check=True)
        if "cdna" in files_to_download_list:
            subprocess.run(["gunzip", f"{cdna_file}.gz"], check=True)

    if sequences == "genome":
        sequences = genome_file
    elif sequences == "cds":
        sequences = cds_file
    elif sequences == "cdna":
        sequences = cdna_file
    return sequences,gtf,gtf_transcript_id_column,genome_file,cds_file,cdna_file


def merge_gtf_transcript_locations_into_cosmic_csv(mutations, gtf_path, gtf_transcript_id_column, output_mutations_path=None):
    # mutations = mutations.copy()  # commented out to save time, but be careful not to make unwanted changes to the original df (in general, if I set df = myfunc(df), then I likely welcome any modifications to df within the function; if not, be especially careful)

    gtf_df = pd.read_csv(
        gtf_path,
        sep="\t",
        comment="#",
        header=None,
        names=[
            "seqname",
            "source",
            "feature",
            "start",
            "end",
            "score",
            "strand",
            "frame",
            "attribute",
        ],
    )

    if "strand" in mutations.columns:
        mutations.rename(columns={"strand": "strand_original"}, inplace=True)

    gtf_df = gtf_df[gtf_df["feature"] == "transcript"]

    gtf_df["transcript_id"] = gtf_df["attribute"].str.extract('transcript_id "([^"]+)"')

    if not len(gtf_df["transcript_id"]) == len(set(gtf_df["transcript_id"])):
        raise ValueError("Duplicate transcript_id values found!")

    # Filter out rows where transcript_id is NaN
    gtf_df = gtf_df.dropna(subset=["transcript_id"])

    gtf_df = gtf_df[["transcript_id", "start", "end", "strand"]].rename(
        columns={
            "transcript_id": gtf_transcript_id_column,
            "start": "start_transcript_position",
            "end": "end_transcript_position",
        }
    )

    merged_df = pd.merge(mutations, gtf_df, on=gtf_transcript_id_column, how="left")

    # Fill NaN values
    merged_df["start_transcript_position"] = merged_df["start_transcript_position"].fillna(0)
    merged_df["end_transcript_position"] = merged_df["end_transcript_position"].fillna(9999999)
    merged_df["strand"] = merged_df["strand"].fillna(".")

    if output_mutations_path is not None:
        merged_df.to_csv(output_mutations_path, index=False)

    return merged_df


def drop_duplication_mutations(input_mutations, output, mutation_column="mutation_genome"):
    if isinstance(input_mutations, str):
        df = pd.read_csv(input_mutations)
    elif isinstance(input_mutations, pd.DataFrame):
        df = input_mutations
    else:
        raise ValueError("input_mutations must be a string or a DataFrame.")

    # count number of duplications
    num_dup = df[mutation_column].str.contains("dup", na=False).sum()
    logger.info(f"Number of duplication mutations that have been dropped: {num_dup}")

    df_no_dup_mutations = df.loc[~(df[mutation_column].str.contains("dup"))]

    df_no_dup_mutations.to_csv(output, index=False)

    return df_no_dup_mutations


def improve_genome_strand_information(cosmic_reference_file_mutation_dataframe, mutation_genome_column_name="mutation_genome", output_mutations_path=None):
    if isinstance(cosmic_reference_file_mutation_dataframe, str):
        df = pd.read_csv(cosmic_reference_file_mutation_dataframe)
    elif isinstance(cosmic_reference_file_mutation_dataframe, pd.DataFrame):
        df = cosmic_reference_file_mutation_dataframe
    else:
        raise ValueError("cosmic_reference_file_mutation_dataframe must be a string or a DataFrame.")

    df["strand_modified"] = df["strand"].replace(".", "+")

    genome_nucleotide_position_pattern = r"g\.(\d+)(?:_(\d+))?[A-Za-z]*"

    extracted_numbers = df[mutation_genome_column_name].str.extract(genome_nucleotide_position_pattern)
    extracted_numbers[1] = extracted_numbers[1].fillna(extracted_numbers[0])
    df["GENOME_START"] = extracted_numbers[0]
    df["GENOME_STOP"] = extracted_numbers[1]

    def complement_substitution(actual_variant):
        return "".join(complement.get(nucleotide, "N") for nucleotide in actual_variant[:])

    def reverse_complement_insertion(actual_variant):
        return "".join(complement.get(nucleotide, "N") for nucleotide in actual_variant[::-1])

    df[["nucleotide_positions", "actual_variant"]] = df["mutation"].str.extract(mutation_pattern)

    minus_sub_mask = (df["strand_modified"] == "-") & (df["actual_variant"].str.contains(">"))
    ins_delins_mask = (df["strand_modified"] == "-") & (df["actual_variant"].str.contains("ins"))

    df["actual_variant_rc"] = df["actual_variant"]

    df.loc[minus_sub_mask, "actual_variant_rc"] = df.loc[minus_sub_mask, "actual_variant"].apply(complement_substitution)

    df.loc[ins_delins_mask, ["variant_type", "mut_nucleotides"]] = df.loc[ins_delins_mask, "actual_variant"].str.extract(r"(delins|ins)([A-Z]+)").values

    df.loc[ins_delins_mask, "mut_nucleotides_rc"] = df.loc[ins_delins_mask, "mut_nucleotides"].apply(reverse_complement_insertion)

    df.loc[ins_delins_mask, "actual_variant_rc"] = df.loc[ins_delins_mask, "variant_type"] + df.loc[ins_delins_mask, "mut_nucleotides_rc"]

    df["actual_variant_final"] = np.where(df["strand_modified"] == "+", df["actual_variant"], df["actual_variant_rc"])

    df[mutation_genome_column_name] = np.where(
        df["GENOME_START"] != df["GENOME_STOP"],
        "g." + df["GENOME_START"].astype(str) + "_" + df["GENOME_STOP"].astype(str) + df["actual_variant_final"],
        "g." + df["GENOME_START"].astype(str) + df["actual_variant_final"],
    )

    df.drop(
        columns=["GENOME_START", "GENOME_STOP", "nucleotide_positions", "actual_variant", "actual_variant_rc", "variant_type", "mut_nucleotides", "mut_nucleotides_rc", "actual_variant_final", "strand_modified"],
        inplace=True,
    )  # drop all columns except mutation_genome_column_name (and the original ones)

    if output_mutations_path:
        df.to_csv(output_mutations_path, index=False)

    return df


def get_last_vcrs_number(filename):
    import csv
    if not os.path.isfile(filename):
        return 1
    with open(filename, "r") as f:
        last_line = list(csv.reader(f))[-1][0]  # Read last row, first column

    # Extract number using regex
    match = re.search(r"vcrs_(\d+)", last_line)
    number = int(match.group(1)) if match else None
    return number



def find_subsequence_merge_targets(sequences, merge_identical_rc=False, seed_len=31):
    """Map each VCRS to the longest VCRS that fully contains it.

    Given a list of *unique* ``sequences`` (i.e. after identical VCRSs have already
    been merged), return a list ``target`` where ``target[i]`` is the index of the
    longest sequence that fully contains ``sequences[i]`` as a substring (also
    checking the reverse complement of ``sequences[i]`` when ``merge_identical_rc``
    is True), or ``i`` itself when ``sequences[i]`` is maximal (not contained in any
    other sequence).

    Sequences are processed longest-first, so every potential container is indexed
    before any shorter sequence is queried against it; this guarantees each returned
    target is itself a maximal ("kept") sequence and that groups are flat (no chains).

    A k-mer seed index over the kept sequences prunes the containment checks: because
    ``seed_len`` is clamped to be no longer than the shortest sequence, every candidate
    has a prefix seed, and if the candidate is contained in a kept sequence then that
    kept sequence was indexed under the candidate's prefix seed - so the prune has no
    false negatives.
    """
    n = len(sequences)
    target = list(range(n))
    if n < 2:
        return target

    lengths = [len(s) for s in sequences]
    if min(lengths) == max(lengths):
        return target  # unique sequences of equal length cannot strictly contain one another

    seed_len = max(1, min(seed_len, min(lengths)))  # every candidate must have a prefix seed of this length

    # longest-first (ties broken by original index for determinism)
    order = sorted(range(n), key=lambda i: (-lengths[i], i))

    seed_index = {}  # seed (str) -> list of kept indices whose sequence contains that seed
    for i in order:
        s = sequences[i]
        queries = (s, reverse_complement(s)) if merge_identical_rc else (s,)
        container = None
        for query in queries:
            for j in seed_index.get(query[:seed_len], ()):
                t = sequences[j]
                if len(t) > len(s) and query in t:  # strictly longer container that contains the candidate
                    container = j
                    break
            if container is not None:
                break
        if container is not None:
            target[i] = container
        else:
            # keep as maximal, and index all of its overlapping seeds for future lookups
            seen_seeds = set()
            for p in range(len(s) - seed_len + 1):
                seed = s[p : p + seed_len]
                if seed not in seen_seeds:
                    seen_seeds.add(seed)
                    seed_index.setdefault(seed, []).append(i)
    return target


def _concatenate_lists(series):
    merged = []
    for value in series:
        merged.extend(value)
    return merged


def merge_subsequence_vcrss(mutations, merge_identical_rc=False):
    """Merge VCRSs whose sequence is a subsequence of another VCRS's sequence.

    Operates on ``mutations`` *after* identical VCRSs have already been merged, so
    every row has a unique ``vcrs_sequence``. When VCRS A's ``vcrs_sequence`` is fully
    contained in VCRS B's (B strictly longer; reverse complements are also considered
    when ``merge_identical_rc`` is True), A is merged into B: B keeps the longer
    sequence, A's ``header`` is semicolon-joined into B's, and A's row is dropped.

    Columns other than ``header``/``vcrs_sequence`` are combined the same way the
    identical-merge does: list-valued columns are concatenated, ``original_order`` takes
    the minimum, and any remaining scalar column keeps the surviving (longest) VCRS's
    value. Returns ``(merged_mutations, num_merged)`` where ``num_merged`` is the number
    of rows that were folded into a supersequence VCRS.
    """
    if len(mutations) < 2:
        return mutations, 0

    sequences = mutations["vcrs_sequence"].tolist()
    target = find_subsequence_merge_targets(sequences, merge_identical_rc=merge_identical_rc)
    num_merged = sum(1 for i, t in enumerate(target) if t != i)
    if num_merged == 0:
        return mutations, 0

    mutations = mutations.reset_index(drop=True).copy()
    mutations["merge_target"] = target
    mutations["seq_len"] = [len(s) for s in sequences]
    # order so the longest (the container = merge target) is first within each group -> "first" keeps its value
    mutations = mutations.sort_values(by=["merge_target", "seq_len"], ascending=[True, False], kind="stable")

    agg_dict = {}
    for col in mutations.columns:
        if col in ("merge_target", "seq_len"):
            continue
        if col == "header":
            agg_dict[col] = lambda values: ";".join(sorted(values))  # alphabetical, matching the identical-merge ordering
        elif col == "original_order":
            agg_dict[col] = "min"
        elif isinstance(mutations[col].iloc[0], list):
            agg_dict[col] = _concatenate_lists
        else:
            agg_dict[col] = "first"

    mutations = mutations.groupby("merge_target", sort=False).agg(agg_dict).reset_index(drop=True)
    return mutations, num_merged


def merge_fasta_file_headers(input_fasta, use_IDs=False, id_to_header_csv_out=None, merge_subsequences=True, merge_identical_rc=True):
    output_fasta = input_fasta.replace(".fa", "_merged.fa")
    
    # Create two temporary files for the intermediate steps.
    with tempfile.NamedTemporaryFile(delete=False, mode='w+', suffix='.tsv') as temp_tsv:
        temp_tsv_name = temp_tsv.name
    with tempfile.NamedTemporaryFile(delete=False, mode='w+', suffix='.tsv') as sorted_tsv:
        sorted_tsv_name = sorted_tsv.name
    with tempfile.NamedTemporaryFile(delete=False, suffix='.fasta') as merged_fasta:
        merged_fasta_name = merged_fasta.name

    try:
        # Step 1: Convert FASTA to a single-line, tab-separated file (sequence<tab>header)
        awk_cmd = f"""awk 'BEGIN {{FS="\\n"; RS=">"; ORS=""}} 
            NR > 1 {{
                split($0, lines, "\\n");
                header = lines[1];
                seq = "";
                for (i = 2; i <= length(lines); i++) {{
                    seq = seq lines[i];
                }}
                print seq "\\t" header "\\n";
            }}' {input_fasta} > {temp_tsv_name}"""
        subprocess.run(awk_cmd, shell=True, check=True, executable="/bin/bash")

        # Step 2: Sort the temporary file by sequence (first column)
        sort_cmd = f"sort -k1,1 -k2,2 {temp_tsv_name} > {sorted_tsv_name}"
        subprocess.run(sort_cmd, shell=True, check=True, executable="/bin/bash")

        
        step_3_output_fasta = merged_fasta_name if use_IDs else output_fasta
        # Step 3: Merge headers for identical sequences
        awk_merge_cmd = f"""awk -F'\\t' 'BEGIN {{ prevSeq = ""; mergedHeader = "" }}
            {{
                if ($1 == prevSeq) {{
                    mergedHeader = mergedHeader ";" $2;
                }} else {{
                    if (prevSeq != "") {{
                        print ">" mergedHeader "\\n" prevSeq;
                    }}
                    prevSeq = $1;
                    mergedHeader = $2;
                }}
            }}
            END {{
                if (prevSeq != "") {{
                    print ">" mergedHeader "\\n" prevSeq;
                }}
            }}' {sorted_tsv_name} > {step_3_output_fasta}"""
        subprocess.run(awk_merge_cmd, shell=True, check=True, executable="/bin/bash")

        # Step 3.5: fold any VCRS whose sequence is a subsequence of a longer VCRS into that longer VCRS
        if merge_subsequences:
            headers, sequences = [], []
            with open(step_3_output_fasta, "r", encoding="utf-8") as merged_handle:
                current_header = None
                current_seq = []
                for line in merged_handle:
                    line = line.rstrip("\n")
                    if line.startswith(">"):
                        if current_header is not None:
                            headers.append(current_header)
                            sequences.append("".join(current_seq))
                        current_header = line[1:]
                        current_seq = []
                    else:
                        current_seq.append(line)
                if current_header is not None:
                    headers.append(current_header)
                    sequences.append("".join(current_seq))

            merged_df = pd.DataFrame({"header": headers, "vcrs_sequence": sequences})
            merged_df, _ = merge_subsequence_vcrss(merged_df, merge_identical_rc=merge_identical_rc)
            with open(step_3_output_fasta, "w", encoding="utf-8") as merged_handle:
                merged_handle.write("".join(f">{h}\n{s}\n" for h, s in zip(merged_df["header"], merged_df["vcrs_sequence"])))

        if use_IDs:
            # Step 4: Count the number of merged FASTA records.
            count_cmd = f"grep -c '^>' {merged_fasta_name}"
            count_output = subprocess.check_output(count_cmd, shell=True, executable="/bin/bash").strip()
            total_records = int(count_output)
            num_digits = len(str(total_records))

            # Step 5: Replace headers with generated IDs and write a mapping CSV.
            # This AWK command processes the merged FASTA, replacing headers with new IDs,
            # and outputs a mapping (new ID, original merged header) to a CSV file.
            awk_id_cmd = f"""awk -v num_digits={num_digits} -v mapping_file="{id_to_header_csv_out}" 'BEGIN {{
                    count = 1;
                    print "vcrs_id,vcrs_header" > mapping_file;
                }}
                /^>/ {{
                    original = substr($0, 2);
                    new_id = sprintf("vcrs_%0*d", num_digits, count);
                    print ">" new_id;
                    print new_id "," original >> mapping_file;
                    count++;
                    next;
                }}
                {{
                    print;
                }}' {merged_fasta_name} > {output_fasta}"""
            subprocess.run(awk_id_cmd, shell=True, check=True, executable="/bin/bash")

        os.remove(input_fasta)
        os.rename(output_fasta, input_fasta)

    finally:
        # Delete the temporary files.
        os.remove(temp_tsv_name)
        os.remove(sorted_tsv_name)
        if os.path.exists(merged_fasta_name):
            os.remove(merged_fasta_name)


# --------------------------------------------------------------------------- #
# Sequence-complexity helpers (used by vk build quality filters)
# --------------------------------------------------------------------------- #
def longest_homopolymer(sequence):
    """Return (length, sequence(s)) of the longest homopolymer run in ``sequence``."""
    homopolymers = re.findall(r"(A+|C+|G+|T+)", sequence)

    if homopolymers:
        max_length = len(max(homopolymers, key=len))
        longest_homopolymers = [h for h in homopolymers if len(h) == max_length]
        if len(longest_homopolymers) == 1:
            return max_length, longest_homopolymers[0]
        return max_length, sorted(list(set(longest_homopolymers)))
    return 0, None  # If no homopolymer is found


def triplet_stats(sequence):
    """Return (num_distinct_triplets, num_total_triplets, triplet_complexity) for ``sequence``."""
    triplets = [sequence[i : (i + 3)] for i in range(len(sequence) - 2)]
    distinct_triplets = set(triplets)
    total_triplets = len(triplets)
    triplet_complexity = len(distinct_triplets) / total_triplets if total_triplets > 0 else 0
    return len(distinct_triplets), total_triplets, triplet_complexity


# --------------------------------------------------------------------------- #
# Pseudoalignment-based reference filtering (alignment_to_reference filter)
# --------------------------------------------------------------------------- #
# Maps to the reference index built (or downloaded) for the pseudoalignment filter.
# Single source of truth for these vocabularies is the Literal types in type_utils.
REFERENCE_TYPES = get_args(ReferenceType)


def run_kb_ref(index, workflow, dna_fasta, t2g=None, f1=None, f2=None, c1=None, c2=None, k=None, threads=None, kallisto=None, bustools=None, gtf=None, tmp=None, skip_index=False):
    """Build a kallisto index for a normal reference by wrapping ``kb ref``."""
    kb_ref_command = ["kb", "ref", "--workflow", workflow, "--make-unique", "-i", index]
    if workflow != "custom":
        kb_ref_command += ["-g", t2g, "-f1", f1, "--d-list=None"]

    if workflow == "nac":
        if f2 is None or c1 is None or c2 is None:
            raise ValueError("f2, c1, and c2 are required for 'nac' workflow")
        kb_ref_command += ["-f2", f2, "-c1", c1, "-c2", c2]
    else:
        for arg in [f2, c1, c2]:
            if arg is not None:
                logger.warning(f"{arg} is ignored for '{workflow}' workflow")

    if k:
        kb_ref_command += ["-k", str(k)]
    if threads:
        kb_ref_command += ["-t", str(threads)]
    if kallisto:
        kb_ref_command += ["--kallisto", kallisto]
    if bustools:
        kb_ref_command += ["--bustools", bustools]
    if tmp:
        # kb ref requires its --tmp directory to NOT already exist
        if os.path.exists(tmp):
            shutil.rmtree(tmp, ignore_errors=True)
        kb_ref_command += ["--tmp", tmp]
    logging_level = logging.getLevelName(logger.getEffectiveLevel())
    if logging_level == "DEBUG":
        kb_ref_command += ["--verbose"]
    kb_ref_command += [dna_fasta]

    if workflow == "custom":
        if gtf is not None:
            logger.warning("gtf file is ignored for 'custom' workflow")
    else:
        if gtf is None:
            raise ValueError(f"gtf file is required for '{workflow}' workflow")
        kb_ref_command += [gtf]

    if skip_index:
        if not os.path.exists(index):
            open(index, "w").close()  # empty file at index lets kb ref create f1 without building the index

    logger.debug(f"Running kb ref command: {' '.join(kb_ref_command)}")
    subprocess.run(kb_ref_command, check=True)


def make_reference_index(dna_fasta, reference_type, out_dir="reference_index", index=None, t2g=None, gtf=None, k=None, threads=None, kallisto=None, bustools=None, overwrite=False, species=None, tmp=None):
    """Build (or download) a kallisto index for a normal reference of the given ``reference_type``."""
    os.makedirs(out_dir, exist_ok=True)
    if index is None:
        index = os.path.join(out_dir, f"{reference_type}.idx")
    if t2g is None:
        t2g = os.path.join(out_dir, f"{reference_type}_t2g.txt")
    if tmp is None:
        # give kb ref a scoped temp dir so it does not collide with a `tmp/` in the CWD
        tmp = os.path.join(out_dir, "kb_ref_tmp")
    f1 = os.path.join(out_dir, "cdna.fasta")

    if os.path.exists(index) or os.path.exists(t2g):
        if overwrite:
            logger.warning(f"Overwriting existing index/t2g files at {index} and/or {t2g}")
        else:
            raise FileExistsError(f"Index or t2g files already exist at {index} and/or {t2g}. Use overwrite=True to overwrite.")

    # Download a prebuilt index if a supported species is given.
    if species is not None:
        if k is None:
            raise ValueError("k must be provided to download a prebuilt index via `species`.")
        try:
            url = species_to_url[species]["kallisto"][reference_type][str(k)]
        except KeyError as exc:
            raise ValueError(
                f"No prebuilt kallisto index for species={species}, reference_type={reference_type}, k={k}. "
                f"Valid options for this species: {species_to_url.get(species, {}).get('kallisto', {})}"
            ) from exc
        logger.info(f"Downloading prebuilt kallisto index for species={species}, reference_type={reference_type}, k={k}")
        download_box_url(url, output_folder=out_dir, output_file_name=index, verbose=True)
        logger.info(f"Downloaded reference index to {index}")
        return  # only the .idx is needed for `kallisto bus`

    if reference_type == "genome":  # custom-dna: map to dna directly
        run_kb_ref(index=index, t2g=t2g, f1=f1, workflow="custom", dna_fasta=dna_fasta, gtf=gtf, k=k, threads=threads, kallisto=kallisto, bustools=bustools, tmp=tmp)
    elif reference_type == "cdna":  # standard: map to cdna
        run_kb_ref(index=index, t2g=t2g, f1=f1, workflow="standard", dna_fasta=dna_fasta, gtf=gtf, k=k, threads=threads, kallisto=kallisto, bustools=bustools, tmp=tmp)
    elif reference_type == "transcriptome":  # nac: map to spliced + unspliced
        f2 = os.path.join(out_dir, "nascent.fasta")
        c1 = os.path.join(out_dir, "cdna.txt")
        c2 = os.path.join(out_dir, "nascent.txt")
        run_kb_ref(index=index, t2g=t2g, f1=f1, workflow="nac", dna_fasta=dna_fasta, gtf=gtf, k=k, threads=threads, kallisto=kallisto, bustools=bustools, f2=f2, c1=c1, c2=c2, tmp=tmp)
    else:
        raise ValueError(f"Invalid reference_type: {reference_type}. Must be one of {REFERENCE_TYPES}")

    logger.info(f"Reference index created at {index}")


def pseudoalign(vcrs_fasta, reference_type, out_dir="kallisto_bus_out", dna_fasta=None, index_dir=None, index=None, t2g=None, gtf=None, k=None, threads=None, kallisto=None, bustools=None, overwrite=False, species=None, tmp=None):
    """Pseudoalign ``vcrs_fasta`` against a normal ``reference_type`` index and return the list of aligned read ids."""
    if reference_type == "genome_or_transcriptome":
        # Rather than building a single combined genome+cdna index, pseudoalign against the genome
        # and cdna references independently (each gets its own index + kallisto bus) and return the
        # union of aligned read ids. The cdna reference is created from the genome via `kb ref`
        # (standard workflow) as part of building its index if it does not already exist.
        aligned = set()
        for sub_type in ("genome", "cdna"):
            sub_aligned = pseudoalign(
                vcrs_fasta=vcrs_fasta,
                reference_type=sub_type,
                out_dir=os.path.join(out_dir, sub_type),
                dna_fasta=dna_fasta,
                index_dir=index_dir,
                index=None,  # derive per-sub-type index/t2g from index_dir
                t2g=None,
                gtf=gtf,
                k=k,
                threads=threads,
                kallisto=kallisto,
                bustools=bustools,
                overwrite=overwrite,
                species=species,
                tmp=tmp,
            )
            aligned.update(sub_aligned)
        return sorted(aligned)

    os.makedirs(out_dir, exist_ok=True)
    if index is None:
        index = os.path.join(index_dir, f"{reference_type}.idx")
    if t2g is None:
        t2g = os.path.join(index_dir, f"{reference_type}.txt")

    if not os.path.exists(index) or overwrite:
        if not os.path.exists(index):
            logger.info(f"Index file not found at {index}. Will create file.")
        else:
            logger.info(f"Overwriting existing index file at {index}.")
        make_reference_index(dna_fasta=dna_fasta, reference_type=reference_type, out_dir=index_dir, index=index, t2g=t2g, gtf=gtf, k=k, threads=threads, kallisto=kallisto, bustools=bustools, overwrite=overwrite, species=species, tmp=tmp)

    vcrs_fastq = os.path.splitext(vcrs_fasta)[0] + ".fastq"
    fasta_to_fastq(vcrs_fasta, vcrs_fastq)

    import kb_python  # bundled kallisto/bustools binaries live alongside kb_python
    if kallisto is None:
        kb_dir = Path(kb_python.__file__).parent
        kallisto_exec = "kallisto_k64" if (k and k > 32) else "kallisto"
        kallisto = str(kb_dir / "bins" / "linux" / "kallisto" / kallisto_exec)
    if bustools is None:
        kb_dir = Path(kb_python.__file__).parent
        bustools = str(kb_dir / "bins" / "linux" / "bustools" / "bustools")

    kallisto_bus_command = [kallisto, "bus", "-i", index, "-o", out_dir, "-x", "BULK", "--num", "--unstranded", "--union"]
    if threads:
        kallisto_bus_command += ["-t", str(threads)]
    logging_level = logging.getLevelName(logger.getEffectiveLevel())
    if logging_level == "DEBUG":
        kallisto_bus_command += ["--verbose"]
    kallisto_bus_command += [vcrs_fastq]

    bus_file = os.path.join(out_dir, "output.bus")
    bus_txt = os.path.join(out_dir, "output.txt")
    run_info_path = os.path.join(out_dir, "run_info.json")
    if not os.path.exists(bus_file) or overwrite:
        logger.debug(f"Running kallisto bus command: {' '.join(kallisto_bus_command)}")
        result = subprocess.run(kallisto_bus_command, check=False)
        # kallisto bus returns a non-zero exit code when 0 reads pseudoalign; treat that as a valid empty result
        if result.returncode != 0:
            if os.path.exists(run_info_path):
                with open(run_info_path, encoding="utf-8") as fh:
                    run_info = json.load(fh)
                if run_info.get("n_pseudoaligned", 0) == 0:
                    logger.info(f"reference_type: {reference_type}, pseudoaligned: 0 / {run_info.get('n_processed', 0)} reads (no VCRSs aligned to the reference).")
                    return []
            raise subprocess.CalledProcessError(result.returncode, kallisto_bus_command)

    bustools_text_command = [bustools, "text", "-f", "-o", bus_txt, bus_file]
    if not os.path.exists(bus_txt) or overwrite:
        logger.debug(f"Running bustools text command: {' '.join(bustools_text_command)}")
        subprocess.run(bustools_text_command, check=True)

    # an empty bus text file means nothing pseudoaligned
    if not os.path.exists(bus_txt) or os.path.getsize(bus_txt) == 0:
        aligned_headers = []
    else:
        bus_df = pd.read_csv(bus_txt, sep="\t", header=None, names=["barcode", "umi", "gene", "count", "read"], usecols=["read"])
        aligned_headers = sorted(set(bus_df["read"].unique()))
    n_pseudoaligned_set = len(aligned_headers)

    # Cross-check against kallisto's own tally in run_info.json.
    with open(run_info_path, encoding="utf-8") as fh:
        run_info = json.load(fh)
    n_pseudoaligned = run_info["n_pseudoaligned"]
    n_processed = run_info["n_processed"]
    if n_pseudoaligned != n_pseudoaligned_set:
        logger.warning(f"Mismatch between run_info.json ({n_pseudoaligned}) and BUS ({n_pseudoaligned_set}) pseudoaligned read counts")
    logger.info(f"reference_type: {reference_type}, pseudoaligned: {n_pseudoaligned} / {n_processed} reads ({(n_pseudoaligned / n_processed) if n_processed else 0:.2%})")
    return aligned_headers


def _parse_fasta_records(path):
    """Yield (header, sequence) tuples from a FASTA file."""
    header, seq = None, []
    with open(path) as f:
        for line in f:
            line = line.rstrip("\n")
            if line.startswith(">"):
                if header is not None:
                    yield header, "".join(seq)
                header, seq = line[1:], []
            else:
                seq.append(line)
    if header is not None:
        yield header, "".join(seq)


def _read_name_to_positions(read_fa):
    """Map each read's SAM QNAME (header up to first whitespace) to the list of
    0-based positions at which it appears in ``read_fa``.

    A list is used because two reads could in principle share a truncated header.
    """
    name_to_positions = {}
    for pos, (header, _) in enumerate(_parse_fasta_records(read_fa)):
        name = header.split()[0] if header else header
        name_to_positions.setdefault(name, []).append(pos)
    return name_to_positions


def kmer_match_bowtie2(
    ref_fa,
    read_fa,
    k=31,
    threads=2,
    bowtie2="bowtie2",
    bowtie2_build="bowtie2-build",
    strandedness=False,
    N_penalty=1,
    max_ambiguous=0,
    work_dir=None,
    index_prefix=None,
):
    """Align every k-mer of every read in ``read_fa`` to ``ref_fa`` with bowtie2
    and return ``(matched_positions, read_match_counts)``:

    - ``matched_positions``: sorted 0-based indices of reads with >= 1 exact
      k-mer hit to ``ref_fa``.
    - ``read_match_counts``: a list (indexed by read position) giving the total
      number of k-mer matches for each read, i.e. the sum over all of the read's
      k-mers of the number of exact reference hits each k-mer had.

    If ``index_prefix`` is given, that prebuilt bowtie2 index is used directly and
    ``ref_fa`` is ignored (nothing is built). Otherwise a bowtie2 index is built
    over ``ref_fa`` inside ``work_dir``. A read matches if any of its k-mers aligns
    with an exact, full-length (score 0) hit. Modeled on
    ``align_to_normal_genome_and_build_dlist`` in ``varseek/utils/varseek_info_utils.py``.
    """
    # Set up a working directory for the index and SAM output.
    cleanup = False
    if work_dir is None:
        work_dir = tempfile.mkdtemp(prefix="kmer_match_bowtie2_")
        cleanup = True
    else:
        os.makedirs(work_dir, exist_ok=True)

    output_sam_file = os.path.join(work_dir, "alignment.sam")

    runtimes = {}

    try:
        # --- Build the bowtie2 index over the reference FASTA (unless a prebuilt one was given). ---
        if index_prefix is None:
            index_dir = os.path.join(work_dir, "bowtie_index")
            index_prefix = os.path.join(index_dir, "ref")
            if not os.path.exists(index_dir) or not os.listdir(index_dir):
                os.makedirs(index_dir, exist_ok=True)
                logger.info("Building bowtie2 index over reference")
                index_start = time.perf_counter()
                subprocess.run(
                    [
                        bowtie2_build,
                        "--threads",
                        str(threads),
                        ref_fa,
                        index_prefix,
                    ],
                    check=True,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
                runtimes["index_build_seconds"] = time.perf_counter() - index_start
                logger.debug(f"Index build runtime: {runtimes['index_build_seconds']:.2f} s")

        # --- Align every k-mer of every read to the reference index. ---
        # -F k,1 extracts each length-k substring (stride 1) of every read and
        # aligns it as its own query, named "{read_header}_{offset}".
        # --score-min C,0,0 with -N 0 / --no-1mm-upfront forces exact matches.
        logger.info("Running bowtie2 alignment")
        bowtie2_alignment_command = [
            bowtie2,
            # "-a",  # report all alignments
            "-k", "1",  # stop after the first valid alignment (we only need boolean occurrence)
            "-f",  # input reads are FASTA
            "--threads",
            str(threads),
            "--xeq",
            "--score-min",
            "C,0,0",  # only perfect-score (exact) alignments
            "--np",
            str(N_penalty),
            "--n-ceil",
            f"C,0,{max_ambiguous}",
            "-F",
            f"{k},1",  # extract every k-mer (stride 1) from each read
            "-R",
            "1",
            "-N",
            "0",
            "-L",
            "31",
            "-i",
            "C,1,0",
            "--no-1mm-upfront",
            "--no-unal",  # do not write unaligned k-mers
            "--no-hd",  # suppress SAM header lines
            "-x",
            index_prefix,
            "-U",
            read_fa,
            "-S",
            output_sam_file,
        ]
        if strandedness:
            bowtie2_alignment_command.insert(3, "--norc")

        alignment_start = time.perf_counter()
        subprocess.run(
            bowtie2_alignment_command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=True,
        )
        runtimes["alignment_seconds"] = time.perf_counter() - alignment_start
        logger.debug(f"Alignment runtime: {runtimes['alignment_seconds']:.2f} s")

        # --- Recover, per read, the total number of aligning k-mers. ---
        # With ``-a`` bowtie2 emits one SAM record per (k-mer, reference hit),
        # so summing the SAM records that belong to a read gives the total
        # number of k-mer matches for that read (summed across all its k-mers).
        name_to_positions = _read_name_to_positions(read_fa)
        n_reads = sum(len(positions) for positions in name_to_positions.values())
        read_match_counts = [0] * n_reads
        with open(output_sam_file) as sam:
            for line in tqdm(sam, desc="Scanning bowtie2 alignments"):
                if not line or line.startswith("@"):
                    continue
                qname = line.split("\t", 1)[0]
                # -F names each k-mer "{read_header}_{offset}"; strip the offset.
                read_name = qname.rsplit("_", 1)[0]
                for pos in name_to_positions.get(read_name, ()):
                    read_match_counts[pos] += 1

        matched_positions = [pos for pos, c in enumerate(read_match_counts) if c > 0]
        return sorted(matched_positions), read_match_counts
    finally:
        if cleanup:
            shutil.rmtree(work_dir, ignore_errors=True)


# How each user-facing ``reference_type`` decomposes into the individual bowtie2
# reference "components". Unlike kallisto's combined nac index, bowtie2 has no single
# "transcriptome" index, so the transcriptome is covered by iterating over the cdna and
# nascent references separately. This is the single source of truth shared by the
# local-build (:func:`_reference_fastas_for_bowtie2`) and prebuilt-download
# (:func:`_download_bowtie2_indices`) paths.
_BOWTIE2_COMPONENTS = {
    "genome": ["genome"],
    "cdna": ["cdna"],
    "transcriptome": ["cdna", "nascent"],
    "genome_or_transcriptome": ["genome", "cdna"],
}

# The fasta filename + builder used to construct each non-genome bowtie2 component from
# the genomic DNA fasta (plus a gtf). ``genome`` uses the provided dna_fasta directly.
_BOWTIE2_COMPONENT_BUILDERS = {
    "cdna": ("cdna.fasta", make_cdna_fasta),
    "nascent": ("nascent.fasta", make_nascent_fasta),
}


def _reference_fastas_for_bowtie2(reference_type, dna_fasta, gtf, out_dir, overwrite=False):
    """Resolve the reference fasta(s) to align a VCRS set against for ``reference_type``.

    A genomic DNA fasta is assumed to always be provided; cDNA / nascent references are
    generated from it (plus a gtf) via the sequence helpers as needed. See
    :data:`_BOWTIE2_COMPONENTS` for how each ``reference_type`` decomposes, e.g.:

      - ``genome``                  -> [genomic dna]
      - ``cdna``                    -> [cdna]                (cdna built from dna + gtf)
      - ``transcriptome``           -> [cdna, nascent]       (both built from dna + gtf; unioned)
      - ``genome_or_transcriptome`` -> [genomic dna, cdna]   (unioned)
    """
    if dna_fasta is None or not os.path.isfile(dna_fasta):
        raise ValueError(f"A genomic DNA fasta is required for bowtie2 pseudoalignment (got dna_fasta={dna_fasta!r}).")

    components = _BOWTIE2_COMPONENTS.get(reference_type)
    if components is None:
        raise ValueError(f"Invalid reference_type: {reference_type}. Must be one of {REFERENCE_TYPES}")

    def _build(name, builder):
        path = os.path.join(out_dir, name)
        if overwrite or not os.path.isfile(path):
            if gtf is None or not os.path.isfile(gtf):
                raise ValueError(f"reference_type={reference_type} requires a gtf to build {name} from the genomic DNA fasta.")
            logger.info(f"Building {name} from {dna_fasta} + {gtf}")
            builder(dna_fasta, gtf, path)
        return path

    fastas = []
    for component in components:
        if component == "genome":
            fastas.append(dna_fasta)
        else:
            name, builder = _BOWTIE2_COMPONENT_BUILDERS[component]
            fastas.append(_build(name, builder))
    return fastas


def _download_and_extract_bowtie2_index(species, component, url, out_dir):
    """Download + unpack one prebuilt bowtie2 component index (``genome``/``cdna``/``nascent``).

    The index is distributed as a ``.tar.gz`` archive that unpacks to ``FOLDER/ref.*``
    bowtie2 index files. Returns the shared ``.../ref`` prefix of the ``ref.1.bt2`` ...
    files that bowtie2 expects (no suffix). The index is independent of the k-mer size
    used downstream, so no ``k`` is needed to select it.
    """
    extract_dir = os.path.join(out_dir, f"{component}_bowtie2_index")

    def _find_index_prefix():
        # bowtie2 wants the prefix shared by <prefix>.1.bt2[l], <prefix>.2.bt2[l], ...
        # The .rev.*.bt2[l] files also end in ".1.bt2[l]", so exclude them.
        for suffix in (".1.bt2l", ".1.bt2"):
            for path in sorted(Path(extract_dir).rglob(f"*{suffix}")):
                if ".rev." not in path.name:
                    return str(path)[: -len(suffix)]
        return None

    index_prefix = _find_index_prefix()
    if index_prefix is None:  # not already downloaded/unpacked
        os.makedirs(extract_dir, exist_ok=True)
        tarball = os.path.join(extract_dir, f"{component}_bowtie2_index.tar.gz")
        logger.info(f"Downloading prebuilt bowtie2 index for species={species}, component={component}")
        download_box_url(url, output_folder=extract_dir, output_file_name=tarball, verbose=True)
        with tarfile.open(tarball, "r:gz") as tar:
            try:
                tar.extractall(extract_dir, filter="data")  # filter added in py3.12
            except TypeError:
                tar.extractall(extract_dir)
        os.remove(tarball)
        index_prefix = _find_index_prefix()

    if index_prefix is None:
        raise ValueError(
            f"Downloaded bowtie2 index archive for species={species}, component={component} "
            f"contained no bowtie2 index files (expected *.1.bt2[l]) under {extract_dir}."
        )
    return index_prefix


def _download_bowtie2_indices(species, reference_type, out_dir):
    """Download and unpack the prebuilt bowtie2 index/indices for ``species``/``reference_type``.

    bowtie2 has no single combined "transcriptome" index (unlike kallisto's nac index),
    so a ``reference_type`` that spans the transcriptome is served by downloading and
    iterating over its individual components (e.g. ``transcriptome`` -> cdna + nascent);
    see :data:`_BOWTIE2_COMPONENTS`.

    Returns a list of ``(index_prefix, is_prebuilt=True)`` tuples to hand to
    :func:`kmer_match_bowtie2` -- one per component, in decomposition order.
    """
    components = _BOWTIE2_COMPONENTS.get(reference_type)
    if components is None:
        raise ValueError(f"Invalid reference_type: {reference_type}. Must be one of {REFERENCE_TYPES}")

    species_indices = species_to_url.get(species, {}).get("bowtie2", {})
    references = []
    for component in components:
        try:
            url = species_indices[component]
        except KeyError as exc:
            raise ValueError(
                f"No prebuilt bowtie2 index for species={species}, reference_type={reference_type} "
                f"(missing component '{component}'). Available components for this species: {sorted(species_indices)}"
            ) from exc
        references.append((_download_and_extract_bowtie2_index(species, component, url, out_dir), True))
    return references


def pseudoalign_bowtie2(vcrs_fasta, reference_type, out_dir="bowtie2_out", index_dir=None, dna_fasta=None, gtf=None, k=None, threads=None, bowtie2="bowtie2", bowtie2_build="bowtie2-build", overwrite=False, species=None, vcrs_strandedness=False):
    """Bowtie2 analogue of :func:`pseudoalign`.

    Returns the positional read-ids (as strings ``"0"``, ``"1"``, ...) in ``vcrs_fasta`` that share
    an exact k-mer with the normal ``reference_type`` reference. The reference is built from the
    genomic DNA fasta (plus a gtf for cdna/nascent) via :func:`_reference_fastas_for_bowtie2`, or
    downloaded for a supported ``species``. When ``reference_type`` maps to multiple references
    (e.g. ``transcriptome`` -> cdna + nascent), the aligned read-ids are unioned across them.

    Prebuilt indices (downloaded for a supported ``species``) are stored in ``index_dir`` so they
    persist alongside the other reference files; the transient per-run alignment work (SAM output,
    locally built cdna/nascent fastas) stays under ``out_dir``. ``index_dir`` defaults to ``out_dir``.
    """
    os.makedirs(out_dir, exist_ok=True)
    if index_dir is None:
        index_dir = out_dir
    else:
        os.makedirs(index_dir, exist_ok=True)
    k = k or 31

    if species is not None:
        references = _download_bowtie2_indices(species, reference_type, index_dir)
    else:
        references = [(ref, False) for ref in _reference_fastas_for_bowtie2(reference_type, dna_fasta, gtf, out_dir, overwrite=overwrite)]

    # Name each per-reference work dir after its component (e.g. "ref_genome", "ref_cdna") so the
    # scratch layout is self-describing. `references` is built in `_BOWTIE2_COMPONENTS` order.
    components = _BOWTIE2_COMPONENTS.get(reference_type, [])

    aligned_positions = set()
    n_reads = None
    for i, (ref, is_prebuilt) in enumerate(references):
        component = components[i] if i < len(components) else str(i)
        work_dir = os.path.join(out_dir, f"ref_{component}")
        if overwrite and os.path.isdir(work_dir):
            shutil.rmtree(work_dir, ignore_errors=True)
        matched, read_match_counts = kmer_match_bowtie2(
            ref_fa=None if is_prebuilt else ref,
            read_fa=vcrs_fasta,
            k=k,
            threads=(threads or 2),
            bowtie2=bowtie2,
            bowtie2_build=bowtie2_build,
            strandedness=vcrs_strandedness,
            work_dir=work_dir,
            index_prefix=ref if is_prebuilt else None,
        )
        aligned_positions.update(matched)
        n_reads = len(read_match_counts)
        logger.info(f"reference_type: {reference_type} (bowtie2, ref {i + 1}/{len(references)} = {component}: {ref}): {len(matched)} / {n_reads} VCRSs matched a k-mer.")

    aligned_headers = [str(p) for p in sorted(aligned_positions)]
    if n_reads:
        logger.info(f"reference_type: {reference_type} (bowtie2), union across references: {len(aligned_headers)} / {n_reads} reads matched.")
    return aligned_headers


def run_pseudoalign_on_vcrs_df(df, reference_type, index_dir, out_dir, dna_fasta=None, gtf=None, k=None, threads=None, seq_col="vcrs_sequence", species=None, aligner="bowtie2", bowtie2="bowtie2", bowtie2_build="bowtie2-build", vcrs_strandedness=False):
    """Drop rows of ``df`` whose VCRS sequence pseudoaligns to the normal ``reference_type`` reference.

    A temporary fasta is written with contiguous positional headers (0..n-1 over the written,
    non-empty records) so that the read ids the aligner reports map unambiguously back to the
    originating rows of ``df``. Rows that pseudoalign are removed (they would be false positives).
    Returns the filtered DataFrame with its original index labels preserved.

    ``aligner`` selects the alignment backend: ``"bowtie2"`` (default) aligns the VCRS sequences'
    k-mers against reference fasta(s) built from the genomic DNA (plus a gtf for cdna/nascent);
    ``"kallisto"`` uses kallisto bus against a kb-ref index. When ``aligner="bowtie2"`` but bowtie2
    is not installed, a warning is emitted and it falls back to kallisto.
    """
    if seq_col not in df.columns:
        raise ValueError(f"Column '{seq_col}' not found in DataFrame. Available columns: {df.columns.tolist()}")

    if aligner not in ("bowtie2", "kallisto"):
        raise ValueError(f"Invalid aligner: {aligner!r}. Must be 'bowtie2' or 'kallisto'.")
    # bowtie2 is the default, but fall back to kallisto when bowtie2 is unavailable.
    if aligner == "bowtie2" and not (is_program_installed(bowtie2) and is_program_installed(bowtie2_build)):
        logger.warning(f"aligner='bowtie2' requested but bowtie2 is not installed/available (checked '{bowtie2}' and '{bowtie2_build}'); falling back to kallisto pseudoalignment.")
        aligner = "kallisto"

    fd, vcrs_fasta = tempfile.mkstemp(suffix=".fasta")
    os.close(fd)
    vcrs_fastq = os.path.splitext(vcrs_fasta)[0] + ".fastq"
    try:
        written_positions = []  # maps written fastq record index -> positional row index in df
        with open(vcrs_fasta, "w", encoding="utf-8") as fasta_file:
            for i, seq in enumerate(df[seq_col].values):
                if seq:  # empty sequences can never pseudoalign
                    fasta_file.write(f">{len(written_positions)}\n{seq}\n")
                    written_positions.append(i)

        len_df = len(df)
        mask = np.ones(len_df, dtype=bool)

        if written_positions:
            if aligner == "bowtie2":
                aligned_headers = pseudoalign_bowtie2(
                    reference_type=reference_type,
                    out_dir=out_dir,
                    index_dir=index_dir,
                    vcrs_fasta=vcrs_fasta,
                    dna_fasta=dna_fasta,
                    gtf=gtf,
                    threads=threads,
                    k=k,
                    bowtie2=bowtie2,
                    bowtie2_build=bowtie2_build,
                    species=species,
                    vcrs_strandedness=vcrs_strandedness,
                )
            else:
                aligned_headers = pseudoalign(
                    reference_type=reference_type,
                    index_dir=index_dir,
                    out_dir=out_dir,
                    vcrs_fasta=vcrs_fasta,
                    dna_fasta=dna_fasta,
                    gtf=gtf,
                    threads=threads,
                    k=k,
                    species=species,
                )
            rows_to_remove = [written_positions[int(h)] for h in aligned_headers]
            if rows_to_remove:
                mask[rows_to_remove] = False

        kept = df.iloc[mask]  # positional; preserves original index labels
        logger.info(f"After pseudoalignment ({reference_type}), {len(kept)} / {len_df} VCRSs remain ({(len(kept) / len_df) if len_df else 0:.2%}); removed {len_df - len(kept)} that aligned to the reference.")
        return kept
    finally:
        for tmp_path in (vcrs_fasta, vcrs_fastq):
            try:
                os.remove(tmp_path)
            except OSError:
                pass


def create_vcrs_index(vcrs_fasta_for_index, index_out, k, threads=2, overwrite=False, dry_run=False, kb_ref_kwargs=None, dlist="None"):
    """Build the kallisto index from a VCRS fasta by wrapping `kb ref` (previously done by vk ref).

    Runs `kb ref` for the VCRS fasta. Existing index files are left untouched unless overwrite=True,
    and dry_run prints the command instead of running it.

    `dlist` is the already-resolved value passed through to `kb ref --d-list`: either the literal
    "None" (no d-list) or a path to a d-list fasta (see `resolve_dlist`).
    """
    # nothing to index if no (non-empty) VCRS fasta was produced (e.g. all variants were dropped) - skip gracefully (but still allow dry_run to print the command)
    if not dry_run and (not vcrs_fasta_for_index or not os.path.isfile(vcrs_fasta_for_index) or os.path.getsize(vcrs_fasta_for_index) == 0):
        logger.warning(f"Skipping index creation because no non-empty VCRS fasta was found at {vcrs_fasta_for_index}")
        return

    dlist = dlist if dlist is not None else "None"
    kb_ref_command = ["kb", "ref", "--workflow", "custom", "-t", str(threads), "-i", index_out, "--d-list", dlist, "-k", str(k), "--overwrite"]

    # assumes any allowable kb ref pass-through argument matches kb ref identically (matched to python kwargs by replacing dashes with underscores)
    kb_ref_kwargs = kb_ref_kwargs or {}
    for dict_key, arguments in varseek_ref_only_allowable_kb_ref_arguments.items():
        for argument in list(arguments):
            argument_py = argument.lstrip("-").replace("-", "_")
            if argument_py in kb_ref_kwargs:
                value = kb_ref_kwargs[argument_py]
                if dict_key == "zero_arguments":
                    if value:  # only add if value is True
                        kb_ref_command.append(argument)
                elif dict_key == "one_argument":
                    kb_ref_command.extend([argument, value])
                else:  # multiple_arguments or something else
                    pass

    # Give kb ref a scoped --tmp dir (unless the caller already supplied one) so it does not collide
    # with a `tmp/` directory in the current working directory.
    default_tmp = None
    if "--tmp" not in kb_ref_command:
        default_tmp = os.path.join(os.path.dirname(index_out) or ".", "kb_ref_tmp")
        kb_ref_command.extend(["--tmp", default_tmp])

    kb_ref_command.append(vcrs_fasta_for_index)

    if not os.path.exists(index_out) or overwrite:
        if dry_run:
            print(" ".join(kb_ref_command))
        else:
            if default_tmp and os.path.exists(default_tmp):
                shutil.rmtree(default_tmp, ignore_errors=True)
            logger.info(f"Running kb ref with command: {' '.join(kb_ref_command)}")
            subprocess.run(kb_ref_command, check=True)
    else:
        logger.warning(f"Skipping kb ref because {index_out} already exists and overwrite=False")


def resolve_reference_dna_and_gtf(reference_dna, reference_gtf, sequences, gtf, need_gtf=True):
    """Resolve the reference genome DNA fasta and GTF for downstream steps (pseudoalignment
    filter, d-list construction), falling back to the build `sequences` (if it looks like a
    genome fasta) and `gtf` respectively when not explicitly provided.

    Returns (reference_dna, reference_gtf); either element may still be None if unresolvable.
    """
    if reference_dna is None and isinstance(sequences, str) and os.path.isfile(sequences) and sequences.endswith(fasta_extensions):
        logger.warning("Reference DNA fasta not provided; falling back to `sequences`. Ensure `sequences` is a genome (DNA) fasta, not cDNA.")
        reference_dna = sequences
    if need_gtf and reference_gtf is None and isinstance(gtf, str) and os.path.isfile(gtf):
        logger.warning("Reference GTF not provided; falling back to the build `gtf`.")
        reference_gtf = gtf
    return reference_dna, reference_gtf


def resolve_dlist(dlist, reference_dna, reference_gtf, sequences, gtf, out_dir, k, overwrite=False, dry_run=False):
    """Resolve the `dlist` argument into a value for `kb ref --d-list`.

    Returns the literal "None" (no d-list) or a path to a d-list fasta:
      - None (or "None")  -> "None"
      - "intergenic_dna"  -> build an intergenic-region fasta from the reference DNA + GTF
      - "cdna"            -> build a spliced-transcript (cDNA) fasta from the reference DNA + GTF
      - path to a fasta   -> used directly

    For "intergenic_dna"/"cdna", the reference DNA fasta and GTF are resolved via
    `resolve_reference_dna_and_gtf` (falling back to the build `sequences`/`gtf`).
    """
    if dlist is None or dlist == "None":
        return "None"
    dlist = str(dlist)

    if dlist not in ("intergenic_dna", "cdna"):
        # anything else must be a path to an existing fasta file
        if dlist.endswith(fasta_extensions) and os.path.isfile(dlist):
            return dlist
        raise ValueError(f"Invalid dlist value: {dlist!r}. Must be None, 'intergenic_dna', 'cdna', or a path to a fasta file.")

    reference_dna, reference_gtf = resolve_reference_dna_and_gtf(reference_dna, reference_gtf, sequences, gtf, need_gtf=True)
    if not reference_dna or not reference_gtf:
        raise ValueError(f"dlist='{dlist}' requires a reference DNA (genome) fasta and a GTF (via alignment_to_reference_dna/alignment_to_reference_gtf, or the build `sequences`/`gtf`).")

    os.makedirs(out_dir, exist_ok=True)
    if dlist == "intergenic_dna":
        dlist_fasta = os.path.join(out_dir, "dlist_intergenic.fa")
        builder = lambda: make_intergenic_fasta(reference_dna, reference_gtf, dlist_fasta, min_length=k)
    else:  # cdna
        dlist_fasta = os.path.join(out_dir, "dlist_cdna.fa")
        builder = lambda: make_transcriptome_fasta(reference_dna, reference_gtf, dlist_fasta)

    if dry_run:
        return dlist_fasta
    if not os.path.isfile(dlist_fasta) or overwrite:
        logger.info(f"Building d-list fasta ('{dlist}') from {reference_dna} + {reference_gtf} at {dlist_fasta}")
        builder()
    else:
        logger.info(f"Using existing d-list fasta at {dlist_fasta}")
    return dlist_fasta


def print_valid_values_for_variants_and_sequences_in_varseek_build():
    mydict = supported_databases_and_corresponding_reference_sequence_type

    # mydict.keys() has mutations, and mydict[mutation]["sequence_download_commands"].keys() has sequences
    message = "vk build internally supported values for 'variants' and 'sequences' are as follows:\n"
    for mutation, mutation_data in mydict.items():
        sequences = list(mutation_data["sequence_download_commands"].keys())
        message += f"'variants': {mutation}\n"
        message += f"  'sequences': {', '.join(sequences)}\n"
    print(message)


def get_sequence_length(seq_id, seq_dict):
    return len(seq_dict.get(seq_id, ""))


def get_nucleotide_at_position(seq_id, pos, seq_dict):
    full_seq = seq_dict.get(seq_id, "")
    if pos < len(full_seq):
        return full_seq[pos]
    return None


def remove_gt_after_semicolon(line):
    parts = line.split(";")
    # Remove '>' from the beginning of each part except the first part
    parts = [parts[0]] + [part.lstrip(">") for part in parts[1:]]
    return ";".join(parts)


def extract_sequence(row, seq_dict, seq_id_column="seq_ID"):
    if pd.isna(row["start_variant_position"]) or pd.isna(row["end_variant_position"]):
        return None
    seq = seq_dict[row[seq_id_column]][int(row["start_variant_position"]) : int(row["end_variant_position"]) + 1]
    return seq


def common_prefix_length(s1, s2):
    min_len = min(len(s1), len(s2))
    for i in range(min_len):
        if s1[i] != s2[i]:
            return i
    return min_len


# Function to find the length of the common suffix with the prefix
def common_suffix_length(s1, s2):
    min_len = min(len(s1), len(s2))
    for i in range(min_len):
        if s1[-(i + 1)] != s2[-(i + 1)]:
            return i
    return min_len


def count_repeat_right_flank(mut_nucleotides, right_flank_region):
    total_overlap_len = 0
    while right_flank_region.startswith(mut_nucleotides):
        total_overlap_len += len(mut_nucleotides)
        right_flank_region = right_flank_region[len(mut_nucleotides) :]
    total_overlap_len += common_prefix_length(mut_nucleotides, right_flank_region)
    return total_overlap_len


def count_repeat_left_flank(mut_nucleotides, left_flank_region):
    total_overlap_len = 0
    while left_flank_region.endswith(mut_nucleotides):
        total_overlap_len += len(mut_nucleotides)
        left_flank_region = left_flank_region[: -len(mut_nucleotides)]
    total_overlap_len += common_suffix_length(mut_nucleotides, left_flank_region)
    return total_overlap_len


def beginning_mut_nucleotides_with_right_flank(mut_nucleotides, right_flank_region):
    if mut_nucleotides == right_flank_region[: len(mut_nucleotides)]:
        return count_repeat_right_flank(mut_nucleotides, right_flank_region)
    else:
        return common_prefix_length(mut_nucleotides, right_flank_region)


# Comparing end of mut_nucleotides to the end of left_flank_region
def end_mut_nucleotides_with_left_flank(mut_nucleotides, left_flank_region):
    if mut_nucleotides == left_flank_region[-len(mut_nucleotides) :]:
        return count_repeat_left_flank(mut_nucleotides, left_flank_region)
    else:
        return common_suffix_length(mut_nucleotides, left_flank_region)


def calculate_beginning_mutation_overlap_with_right_flank(row):
    if row["variant_type"] == "deletion":
        sequence_to_check = row["wt_nucleotides_ensembl"]
    else:
        sequence_to_check = row["mut_nucleotides"]

    if row["variant_type"] == "delins" or row["variant_type"] == "inversion":
        original_sequence = row["wt_nucleotides_ensembl"] + row["right_flank_region"]
    else:
        original_sequence = row["right_flank_region"]

    return beginning_mut_nucleotides_with_right_flank(sequence_to_check, original_sequence)


def calculate_end_mutation_overlap_with_left_flank(row):
    if row["variant_type"] == "deletion":
        sequence_to_check = row["wt_nucleotides_ensembl"]
    else:
        sequence_to_check = row["mut_nucleotides"]

    if row["variant_type"] == "delins" or row["variant_type"] == "inversion":
        original_sequence = row["left_flank_region"] + row["wt_nucleotides_ensembl"]
    else:
        original_sequence = row["left_flank_region"]

    return end_mut_nucleotides_with_left_flank(sequence_to_check, original_sequence)

def iterate_through_vcf_in_chunks(variants, params_dict, chunksize, merge_identical=True):
    from varseek.varseek_build import build
    import pysam
    tmp_file = variants.replace(".vcf", "_chunked.vcf")
    with pysam.VariantFile(variants, "r") as vcf:
        header = str(vcf.header)  # Store the header lines

        chunk = []
        chunk_number = 1

        for i, record in enumerate(vcf):
            chunk.append(str(record))

            # Process the chunk once it reaches the desired size
            if (i + 1) % chunksize == 0:
                with open(tmp_file, "w") as f:
                    f.write(header + "".join(chunk))
                build(variants=tmp_file, chunksize=None, chunk_number=chunk_number, running_within_chunk_iteration=True, merge_identical=False, **params_dict)  # running_within_chunk_iteration here for logger setup and report_time_elapsed decorator
                chunk = []  # Reset the chunk
                chunk_number += 1

        # Process any remaining variants
        if chunk:
            with open(tmp_file, "w") as f:
                f.write(header + "".join(chunk))
            build(variants=tmp_file, chunksize=None, chunk_number=chunk_number, running_within_chunk_iteration=True, **params_dict)  # running_within_chunk_iteration here for logger setup and report_time_elapsed decorator

        if isinstance(tmp_file, str) and os.path.exists(tmp_file):
            os.remove(tmp_file)

        if merge_identical:
            # the unfiltered vcrs was appended (unmerged) per-chunk; the merge + t2g happen on the filtered file
            out, vcrs_fasta_out, vcrs_t2g_out, id_to_header_csv_out = params_dict["out"], params_dict.get("vcrs_fasta_out", None), params_dict.get("vcrs_t2g_out", None), params_dict.get("id_to_header_csv_out", None)
            vcrs_fasta_out = os.path.join(out, "vcrs.fa") if not vcrs_fasta_out else vcrs_fasta_out  # copy-paste from below
            id_to_header_csv_out = os.path.join(out, "id_to_header_mapping.csv") if not id_to_header_csv_out else id_to_header_csv_out  # copy-paste from below
            vcrs_t2g_out = os.path.join(out, "vcrs_t2g.txt") if not vcrs_t2g_out else vcrs_t2g_out  # copy-paste from below
            merge_fasta_file_headers(vcrs_fasta_out, use_IDs=params_dict.get("use_IDs", True), id_to_header_csv_out=id_to_header_csv_out, merge_subsequences=params_dict.get("merge_subsequences", True), merge_identical_rc=not params_dict.get("vcrs_strandedness", False))
            create_identity_t2g(vcrs_fasta_out, vcrs_t2g_out, mode="w")
        return