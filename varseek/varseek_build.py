import os
import subprocess
from typing import Union, List, Optional
import re
import time
import numpy as np
import pandas as pd
from tqdm import tqdm
import gget

from .constants import (
    complement,
    supported_databases_and_corresponding_reference_sequence_type,
    mutation_pattern,
)
from .utils import (
    set_up_logger,
    read_fasta,
    convert_mutation_cds_locations_to_cdna,
    convert_chromosome_value_to_int_when_possible,
    generate_unique_ids,
    translate_sequence,
    wt_fragment_and_mutant_fragment_share_kmer,
    create_mutant_t2g,
    add_mutation_type,
    report_time_and_memory,
    save_params_to_config_file,
    make_function_parameter_to_value_dict,
    check_file_path_is_string_with_valid_extension,
    print_varseek_dry_run
)

# from gget.utils import read_fasta

tqdm.pandas()
logger = set_up_logger()

# Define global variables to count occurences of weird mutations
intronic_mutations = 0
posttranslational_region_mutations = 0
unknown_mutations = 0
uncertain_mutations = 0
ambiguous_position_mutations = 0
cosmic_incorrect_wt_base = 0
mut_idx_outside_seq = 0


def print_valid_values_for_mutations_and_sequences_in_varseek_build(return_message=False):
    mydict = supported_databases_and_corresponding_reference_sequence_type
    
    # mydict.keys() has mutations, and mydict[mutation]["sequence_download_commands"].keys() has sequences
    message = "vk build internally supported values for 'mutations' and 'sequences' are as follows:\n"
    for mutation, mutation_data in mydict.items():
        sequences = list(mutation_data["sequence_download_commands"].keys())
        message += f"'mutations': {mutation}\n"
        message += f"  'sequences': {', '.join(sequences)}\n"
    if return_message:
        return message
    else:
        print(message)
    

def reverse_complement(seq):
    if pd.isna(seq):  # Check if the sequence is NaN
        return np.nan
    complement = str.maketrans("ATCGNatcgn.*", "TAGCNtagcn.*")
    return seq.translate(complement)[::-1]


def merge_gtf_transcript_locations_into_cosmic_csv(mutations, gtf_path, gtf_transcript_id_column, output_mutations_path=None):
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

    assert len(gtf_df["transcript_id"]) == len(set(gtf_df["transcript_id"])), "Duplicate transcript_id values found!"

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


def drop_duplication_mutations(input, output, mutation_column="mutation_genome"):
    df = pd.read_csv(input)

    # count number of duplications
    num_dup = df[mutation_column].str.contains("dup", na=False).sum()
    print(f"Number of duplication mutations that have been dropped: {num_dup}")

    df_no_dup_mutations = df.loc[~(df[mutation_column].str.contains("dup"))]

    df_no_dup_mutations.to_csv(output, index=False)


def improve_genome_strand_information(cosmic_reference_file_mutation_csv, mutation_genome_column_name="mutation_genome"):
    df = pd.read_csv(cosmic_reference_file_mutation_csv)

    df["strand"] = df["strand"].replace(".", "+")

    genome_nucleotide_position_pattern = r"g\.(\d+)(?:_(\d+))?[A-Za-z]*"

    extracted_numbers = df[mutation_genome_column_name].str.extract(genome_nucleotide_position_pattern)
    extracted_numbers[1] = extracted_numbers[1].fillna(extracted_numbers[0])
    df["GENOME_START"] = extracted_numbers[0]
    df["GENOME_STOP"] = extracted_numbers[1]

    def complement_substitution(actual_mutation):
        return "".join(complement.get(nucleotide, "N") for nucleotide in actual_mutation[:])

    def reverse_complement_insertion(actual_mutation):
        return "".join(complement.get(nucleotide, "N") for nucleotide in actual_mutation[::-1])

    df[["nucleotide_positions", "actual_mutation"]] = df["mutation"].str.extract(mutation_pattern)

    minus_sub_mask = (df["strand"] == "-") & (df["actual_mutation"].str.contains(">"))
    ins_delins_mask = (df["strand"] == "-") & (df["actual_mutation"].str.contains("ins"))

    df["actual_mutation_rc"] = df["actual_mutation"]

    df.loc[minus_sub_mask, "actual_mutation_rc"] = df.loc[minus_sub_mask, "actual_mutation"].apply(complement_substitution)

    df.loc[ins_delins_mask, ["mutation_type", "mut_nucleotides"]] = df.loc[ins_delins_mask, "actual_mutation"].str.extract(r"(delins|ins)([A-Z]+)").values

    df.loc[ins_delins_mask, "mut_nucleotides_rc"] = df.loc[ins_delins_mask, "mut_nucleotides"].apply(reverse_complement_insertion)

    df.loc[ins_delins_mask, "actual_mutation_rc"] = df.loc[ins_delins_mask, "mutation_type"] + df.loc[ins_delins_mask, "mut_nucleotides_rc"]

    df["actual_mutation_final"] = np.where(df["strand"] == "+", df["actual_mutation"], df["actual_mutation_rc"])

    df["mutation_genome"] = np.where(
        df["GENOME_START"] != df["GENOME_STOP"],
        "g." + df["GENOME_START"].astype(str) + "_" + df["GENOME_STOP"].astype(str) + df["actual_mutation_final"],
        "g." + df["GENOME_START"].astype(str) + df["actual_mutation_final"],
    )

    df.drop(
        columns=[
            "GENOME_START",
            "GENOME_STOP",
            "nucleotide_positions",
            "actual_mutation",
            "actual_mutation_rc",
            "mutation_type",
            "mut_nucleotides",
            "mut_nucleotides_rc",
            "actual_mutation_final",
        ],
        inplace=True,
    )

    df.to_csv(cosmic_reference_file_mutation_csv, index=False)


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
    if pd.isna(row["start_mutation_position"]) or pd.isna(row["end_mutation_position"]):
        return None
    seq = seq_dict[row[seq_id_column]][int(row["start_mutation_position"]) : int(row["end_mutation_position"]) + 1]
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
    if row["mutation_type"] == "deletion":
        sequence_to_check = row["wt_nucleotides_ensembl"]
    else:
        sequence_to_check = row["mut_nucleotides"]

    if row["mutation_type"] == "delins" or row["mutation_type"] == "inversion":
        original_sequence = row["wt_nucleotides_ensembl"] + row["right_flank_region"]
    else:
        original_sequence = row["right_flank_region"]

    return beginning_mut_nucleotides_with_right_flank(sequence_to_check, original_sequence)


def calculate_end_mutation_overlap_with_left_flank(row):
    if row["mutation_type"] == "deletion":
        sequence_to_check = row["wt_nucleotides_ensembl"]
    else:
        sequence_to_check = row["mut_nucleotides"]

    if row["mutation_type"] == "delins" or row["mutation_type"] == "inversion":
        original_sequence = row["left_flank_region"] + row["wt_nucleotides_ensembl"]
    else:
        original_sequence = row["left_flank_region"]

    return end_mut_nucleotides_with_left_flank(sequence_to_check, original_sequence)

def validate_input_build(sequences, mutations, mut_column, seq_id_column, mut_id_column, gtf, gtf_transcript_id_column, w, k, insertion_size_limit, min_seq_len, optimize_flanking_regions, remove_seqs_with_wt_kmers, max_ambiguous, required_insertion_overlap_length, merge_identical, strandedness, keep_original_headers, save_wt_mcrs_fasta_and_t2g, save_mutations_updated_csv, store_full_sequences, translate, translate_start, translate_end, out, reference_out_dir, mcrs_fasta_out, mutations_updated_csv_out, id_to_header_csv_out, mcrs_t2g_out, wt_mcrs_fasta_out, wt_mcrs_t2g_out, return_mutation_output, verbose, **kwargs,):
    # Validate sequences
    if not (isinstance(sequences, str) or isinstance(sequences, list)):
        raise ValueError(f"sequences must be a nucleotide string, a path, or a list of nucleotide strings. Got {type(sequences)}.")
    if isinstance(sequences, list) and not all(isinstance(seq, str) for seq in sequences):
        raise ValueError("All elements in sequences must be nucleotide strings.")
    if isinstance(sequences, str) and (not os.path.isfile(sequences) or not all(c in "ACGTNU-.*" for c in sequences.upper())):
        raise ValueError("If sequences is a string, it must be a valid file path or a nucleotide string.")

    # Validate mutations
    if not (isinstance(mutations, str) or isinstance(mutations, list)):
        raise ValueError(f"mutations must be a string, a path, or a list of strings. Got {type(mutations)}.")
    if isinstance(mutations, list) and not all(isinstance(mut, str) for mut in mutations):
        raise ValueError("All elements in mutations must be strings.")
    if isinstance(mutations, str) and not (mutations.startswith("c.") or mutations.startswith("g.")):  # mutations refers to an internally supported value, eg cosmic_cmc
        if mutations not in supported_databases_and_corresponding_reference_sequence_type:
            vk_build_end_help_message = print_valid_values_for_mutations_and_sequences_in_varseek_build(return_message=True)
            raise ValueError(f"mutations {mutations} not internally supported.\n{vk_build_end_help_message}")
        else:
            if sequences not in supported_databases_and_corresponding_reference_sequence_type[mutations]['sequence_download_commands']:
                vk_build_end_help_message = print_valid_values_for_mutations_and_sequences_in_varseek_build(return_message=True)
                raise ValueError(f"sequences {sequences} not internally supported.\n{vk_build_end_help_message}")
    
    # Directories
    if not isinstance(out, str) or not os.path.isdir(out):
        raise ValueError(f"Invalid input directory: {out}")
    if reference_out_dir and (not isinstance(reference_out_dir, str) or not os.path.isdir(reference_out_dir)):
        raise ValueError(f"Invalid reference output directory: {reference_out_dir}")
    
    check_file_path_is_string_with_valid_extension(gtf, "gtf", "gtf")
    check_file_path_is_string_with_valid_extension(mcrs_fasta_out, "mcrs_fasta_out", "fasta")
    check_file_path_is_string_with_valid_extension(mutations_updated_csv_out, "mutations_updated_csv_out", "csv")
    check_file_path_is_string_with_valid_extension(id_to_header_csv_out, "id_to_header_csv_out", "csv")
    check_file_path_is_string_with_valid_extension(mcrs_t2g_out, "mcrs_t2g_out", "t2g")
    check_file_path_is_string_with_valid_extension(wt_mcrs_fasta_out, "wt_mcrs_fasta_out", "fasta")
    check_file_path_is_string_with_valid_extension(wt_mcrs_t2g_out, "wt_mcrs_t2g_out", "t2g")

    # Validate string parameters
    for param_name, param_value in {
        "mut_column": mut_column,
        "seq_id_column": seq_id_column,
        "mut_id_column": mut_id_column,
        "gtf_transcript_id_column": gtf_transcript_id_column,
    }.items():
        if param_value is not None and not isinstance(param_value, str):
            raise ValueError(f"{param_name} must be a string or None. Got {type(param_value)}.")

    # Validate numeric parameters
    # Required
    for param_name, param_value, min_value in [
        ("w", w, 1),
    ]:
        if not isinstance(param_value, int) or param_value < min_value:
            raise ValueError(f"{param_name} must be an integer >= {min_value} or None. Got {param_value}.")
    
    # Optional
    for param_name, param_value, min_value in [
        ("k", k, 1),
        ("insertion_size_limit", insertion_size_limit, 1),
        ("min_seq_len", min_seq_len, 1),
        ("max_ambiguous", max_ambiguous, 0),
    ]:
        if param_value is not None and (not isinstance(param_value, int) or param_value < min_value):
            raise ValueError(f"{param_name} must be an integer >= {min_value} or None. Got {param_value}.")

    # Validate required_insertion_overlap_length
    if required_insertion_overlap_length is not None and not (
        isinstance(required_insertion_overlap_length, int)
        or isinstance(required_insertion_overlap_length, str)
    ):
        raise ValueError(
            f"required_insertion_overlap_length must be an int, a string, or None. Got {type(required_insertion_overlap_length)}."
        )

    # Validate boolean parameters
    for param_name, param_value in {
        "optimize_flanking_regions": optimize_flanking_regions,
        "remove_seqs_with_wt_kmers": remove_seqs_with_wt_kmers,
        "merge_identical": merge_identical,
        "strandedness": strandedness,
        "keep_original_headers": keep_original_headers,
        "save_wt_mcrs_fasta_and_t2g": save_wt_mcrs_fasta_and_t2g,
        "save_mutations_updated_csv": save_mutations_updated_csv,
        "store_full_sequences": store_full_sequences,
        "translate": translate,
        "return_mutation_output": return_mutation_output,
        "verbose": verbose,
    }.items():
        if not isinstance(param_value, bool):
            raise ValueError(f"{param_name} must be a boolean. Got {type(param_value)}.")

    # Validate output directory
    if not isinstance(out, str) or not os.path.isdir(out):
        raise ValueError(f"Output directory (out) must be a valid directory path. Got {out}.")

    # Validate translation parameters
    for param_name, param_value in {
        "translate_start": translate_start,
        "translate_end": translate_end,
    }.items():
        if param_value is not None and not (isinstance(param_value, int) or isinstance(param_value, str)):
            raise ValueError(f"{param_name} must be an int, a string, or None. Got {type(param_value)}.")


def build(
    sequences: Union[str, List[str]],
    mutations: Union[str, List[str]],
    mut_column: str = "mutation",
    seq_id_column: str = "seq_ID",
    mut_id_column: Optional[str] = None,
    gtf: Optional[str] = None,
    gtf_transcript_id_column: Optional[str] = None,
    w: int = 30,
    k: Optional[int] = None,
    insertion_size_limit: Optional[int] = None,
    min_seq_len: Optional[int] = None,
    optimize_flanking_regions: bool = False,
    remove_seqs_with_wt_kmers: bool = False,
    max_ambiguous: Optional[int] = None,
    required_insertion_overlap_length: Union[int, str, None] = None,
    merge_identical: bool = False,
    strandedness: bool = False,
    keep_original_headers: bool = False,
    save_wt_mcrs_fasta_and_t2g: bool = False,
    save_mutations_updated_csv: bool = False,
    store_full_sequences: bool = False,
    translate: bool = False,
    translate_start: Union[int, str, None] = None,
    translate_end: Union[int, str, None] = None,
    out: str = ".",
    reference_out_dir: Optional[str] = None,
    mcrs_fasta_out: Optional[str] = None,
    mutations_updated_csv_out: Optional[str] = None,
    id_to_header_csv_out: Optional[str] = None,
    mcrs_t2g_out: Optional[str] = None,
    wt_mcrs_fasta_out: Optional[str] = None,
    wt_mcrs_t2g_out: Optional[str] = None,
    return_mutation_output: bool = False,
    overwrite: bool = False,
    dry_run: bool = False,
    verbose: bool = True,
    **kwargs,
):
    """
    Takes in nucleotide sequences and mutations (in standard mutation annotation - see below)
    and returns mutated versions of the input sequences according to the provided mutations.

    Required input argument:
    - sequences     (str) Path to the fasta file containing the sequences to be mutated, e.g., 'seqs.fa'.
                    Sequence identifiers following the '>' character must correspond to the identifiers
                    in the seq_ID column of 'mutations'.

                    Example:
                    >seq1 (or ENSG00000106443)
                    ACTGCGATAGACT
                    >seq2
                    AGATCGCTAG

                    Alternatively: Input sequence(s) as a string or a list of strings,
                    e.g. 'AGCTAGCT' or ['ACTGCTAGCT', 'AGCTAGCT'].

                    NOTE: Only the letters until the first space or dot will be used as sequence identifiers
                    - Version numbers of Ensembl IDs will be ignored.
                    NOTE: When 'sequences' input is a genome, also see 'gtf' argument below.

                    Alternatively, if 'mutations' is a string specifying a supported database, 
                    sequences can be a string indicating the source upon which to apply the mutations.
                    See below for supported databases and sequences options.
                    To see the supported combinations of mutations and sequences, either
                    1) run `vk build --help` from the command line, or
                    2) run varseek.varseek_build.print_valid_values_for_mutations_and_sequences_in_varseek_build() in python

    - mutations     (str or DataFrame object) Path to csv or tsv file (str) (e.g., 'mutations.csv'), or DataFrame (DataFrame object),
                    containing information about the mutations in the following format:

                    | mutation         | mut_ID | seq_ID |
                    | c.2C>T           | mut1   | seq1   | -> Apply mutation 1 to sequence 1
                    | c.9_13inv        | mut2   | seq2   | -> Apply mutation 2 to sequence 2
                    | c.9_13inv        | mut2   | seq3   | -> Apply mutation 2 to sequence 3
                    | c.9_13delinsAAT  | mut3   | seq3   | -> Apply mutation 3 to sequence 3
                    | ...              | ...    | ...    |

                    'mutation' = Column containing the mutations to be performed written in standard mutation annotation (see below)
                    'seq_ID' = Column containing the identifiers of the sequences to be mutated (must correspond to the string following
                    the > character in the 'sequences' fasta file; do NOT include spaces or dots)
                    'mut_ID' = Column containing an identifier for each mutation (optional).

                    Alternatively: Input mutation(s) as a string or list, e.g., 'c.2C>T' or ['c.2C>T', 'c.1A>C'].
                    If a list is provided, the number of mutations must equal the number of input sequences.

                    For more information on the standard mutation annotation, see https://www.ncbi.nlm.nih.gov/pmc/articles/PMC1867422/.

                    Alternatively, 'mutations' can be a string specifying a supported database, which will automatically download
                    both the mutation database and corresponding reference sequence (if the 'sequences' is not a path).
                    To see the supported combinations of mutations and sequences, either
                    1) run `vk build --help` from the command line, or
                    2) run varseek.varseek_build.print_valid_values_for_mutations_and_sequences_in_varseek_build() in python

    Additional input arguments:
    - mut_column                         (str) Name of the column containing the mutations to be performed in 'mutations'. Default: 'mutation'.
    - seq_id_column                      (str) Name of the column containing the IDs of the sequences to be mutated in 'mutations'. Default: 'seq_ID'.
    - mut_id_column                      (str) Name of the column containing the IDs of each mutation in 'mutations'. Optional. Default: use <seq_ID>_<mutation> for each row.
    - gtf                                (str) Path to .gtf file. When providing a genome fasta file as input for 'sequences', you can provide a .gtf file here
                                         and the input sequences will be defined according to the transcript boundaries. Default: None
    - gtf_transcript_id_column           (str) Column name in the input 'mutations' file containing the transcript ID. 
                                         In this case, column seq_id_column should contain the chromosome number.
                                         Required when 'gtf' is provided. Default: None

    Mutant sequence generation/filtering options:
    - w                                  (int) Length of sequence windows flanking the mutation. Default: 30.
                                         If w > total length of the sequence, the entire sequence will be kept.
    - k                                  (int) Length of the k-mers to be considered when removed remove_seqs_with_wt_kmers.
                                         If using kallisto in a later workflow, then this should correspond to kallisto k).
                                         Must be greater than the value passed in for w. Default: w+1.
    - insertion_size_limit               (int) Maximum number of nucleotides allowed in an insertion-type mutation. Mutations with insertions larger than this will be dropped.
                                         Default: None (no insertion size limit will be applied)
    - min_seq_len                        (int) Minimum length of the mutant output sequence. Mutant sequences smaller than this will be dropped. Default: None (No length filter will be applied)
    - optimize_flanking_regions          (True/False) Whether to remove nucleotides from either end of the mutant sequence to ensure (when possible)
                                         that the mutant sequence does not contain any w-mers (where a w-mer is a subsequence of length w) also found in the wildtype/input sequence. Default: False
    - remove_seqs_with_wt_kmers          (True/False) Removes output sequences where at least one (w+1)-mer (where a w-mer is a subsequence of length w) is also present in the wildtype/input sequence in the same region.
                                         If optimize_flanking_regions=True, only sequences for which a wildtype w-mer is still present after optimization will be removed.
                                         Default: False
    - max_ambiguous                      (int) Maximum number of 'N' (or 'n') characters allowed in the output sequence. Default: None (no 'N' filter will be applied)
    - required_insertion_overlap_length  (int | str | None) Sets the Minimum number of nucleotides that must overlap between the inserted sequence and the flanking regions after flank optimization. Only effective when optimize_flanking_regions is also True.
                                         Default: None (No checking). If "all", then require the entire insertion and the following nucleotide
    - merge_identical                    (True/False) Whether to merge identical mutant sequences in the output (identical sequences will be merged by concatenating the sequence
                                         headers for all identical sequences with semicolons). Default: False
    - strandedness                       (True/False) Whether to consider the forward and reverse-complement mutant sequences as distinct if merging identical sequences. Only effective when merge_identical is also True.
                                         Default: False (ie do not consider forward and reverse-complement sequences to be equivalent)
    - keep_original_headers              (True/False) Whether to keep the original sequence headers in the output fasta file, or to replace them with unique IDs of the form 'vcrs_<int>.
                                         If False, then an additional file at the path <id_to_header_csv_out> will be formed that maps sequence IDs from the fasta file to the <mut_id_column>. Default: False.

    # Optional arguments to generate additional output stored in a copy of the 'mutations' DataFrame
    - save_mutations_updated_csv         (True/False) Whether to update the input 'mutations' DataFrame to include additional columns with the mutation type,
                                         wildtype nucleotide sequence, and mutant nucleotide sequence (only valid if 'mutations' is a csv or tsv file). Default: False
    - store_full_sequences               (True/False) Whether to also include the complete wildtype and mutant sequences in the updated 'mutations' DataFrame (not just the sub-sequence with
                                         w-length flanks). Only valid if save_mutations_updated_csv=True. Default: False
    - translate                          (True/False) Add additional columns to the 'mutations' DataFrame containing the wildtype and mutant amino acid sequences.
                                         Only valid if store_full_sequences=True. Default: False
    - translate_start                    (int | str | None) The position in the input nucleotide sequence to start translating. If a string is provided, it should correspond
                                         to a column name in 'mutations' containing the open reading frame start positions for each sequence/mutation.
                                         Only valid if translate=True. Default: None (translate from the beginning of the sequence)
    - translate_end                      (int | str | None) The position in the input nucleotide sequence to end translating. If a string is provided, it should correspond
                                         to a column name in 'mutations' containing the open reading frame end positions for each sequence/mutation.
                                         Only valid if translate=True. Default: None (translate from to the end of the sequence)

    # Additional arguments affecting output:
    - save_wt_mcrs_fasta_and_t2g         (True/False) Whether to create a fasta file containing the wildtype sequence counterparts of the mutation-containing reference sequences (MCRSs)
                                         and the corresponding t2g. Default: False.
    - return_mutation_output             (True/False) Whether to return the mutation output saved in the fasta file. Default: False.

    # General arguments:
    - out                                (str) Path to default output directory to containing created files. Any individual output file path can be overriden if the specific file path is provided
                                         as an argument. Default: "." (current directory).
    - reference_out_dir                  (str) Path to reference file directory to be downloaded if 'mutations' is a supported database and the file corresponding to 'sequences' does not exist.
                                         Default: <out>/reference directory.
    - mcrs_fasta_out                     (str) Path to output fasta file containing the mutation-containing reference sequences (MCRSs). 
                                         If keep_original_headers=True, then the fasta headers will be the values in the column 'mut_ID' (semicolon-jooined if merge_identical=True).
                                         Otherwise, if keep_original_headers=False (default), then the fasta headers will be of the form 'vcrs_<int>' where <int> is a unique integer. Default: "<out>/mcrs.fa"
    - mutations_updated_csv_out          (str) Path to output csv file containing the updated DataFrame. Only valid if save_mutations_updated_csv=True. Default: "<out>/mutation_metadata_df.csv"
    - id_to_header_csv_out               (str) File name of csv file containing the mapping of unique IDs to the original sequence headers if keep_original_headers=False. Default: "<out>/id_to_header_mapping.csv"
    - mcrs_t2g_out                       (str) Path to output t2g file containing the transcript-to-gene mapping for the MCRSs. Used in kallisto | bustools workflow. Default: "<out>/mcrs_t2g.txt"
    - wt_mcrs_fasta_out                  (str) Path to output fasta file containing the wildtype sequence counterparts of the mutation-containing reference sequences (MCRSs). Default: "<out>/wt_mcrs.fa"
    - wt_mcrs_t2g_out                    (str) Path to output t2g file containing the transcript-to-gene mapping for the wildtype MCRSs. Default: "<out>/wt_mcrs_t2g.txt"
    - dry_run                            (True/False) Whether to simulate the function call without executing it. Default: False.
    - overwrite                          (True/False) Whether to overwrite existing output files. Will return if any output file already exists. Default: False.
    - verbose                            (True/False) whether to print progress information. Default: True

    # kwargs options (related to specific databases or meant primarily for debugging purposes):
    - cosmic_release                     (str) COSMIC release version to download. Default: "100".
    - cosmic_grch                        (str) COSMIC genome reference version to download. Default: "37".
    - cosmic_email                       (str) Email address for COSMIC download. Default: None.
    - cosmic_password                    (str) Password for COSMIC download. Default: None.
    - do_not_save_files                  (True/False) Whether to save the output files. Default: False.


    Saves mutated sequences in fasta format (or returns a list containing the mutated sequences if out=None).
    """

    global intronic_mutations, posttranslational_region_mutations, unknown_mutations, uncertain_mutations, ambiguous_position_mutations, cosmic_incorrect_wt_base, mut_idx_outside_seq

    # enforce type-checking of parameters
    params_dict = make_function_parameter_to_value_dict(1)
    if dry_run:
        print_varseek_dry_run(params_dict, function_name="build")
        return None
    validate_input_build(**params_dict)

    # begin tracking time and memory
    start_overall, peaks_list = report_time_and_memory(logger=logger, report=True)
    start = start_overall

    config_file = os.path.join(out, "config", "vk_build_config.json")
    save_params_to_config_file(config_file)

    if not reference_out_dir:
        reference_out_dir = os.path.join(out, "reference")

    os.makedirs(out, exist_ok=True)
    os.makedirs(reference_out_dir, exist_ok=True)

    if not mcrs_fasta_out:
        mcrs_fasta_out = os.path.join(out, "mcrs.fa")
    if not mutations_updated_csv_out:
        mutations_updated_csv_out = os.path.join(out, "mutation_metadata_df.csv")
    if not id_to_header_csv_out:
        id_to_header_csv_out = os.path.join(out, "id_to_header_mapping.csv")
    if not mcrs_t2g_out:
        mcrs_t2g_out = os.path.join(out, "mcrs_t2g.txt")
    if not wt_mcrs_fasta_out:
        wt_mcrs_fasta_out = os.path.join(out, "wt_mcrs.fa")
    if not wt_mcrs_t2g_out:
        wt_mcrs_t2g_out = os.path.join(out, "wt_mcrs_t2g.txt")
    

    # make sure directories of all output files exist
    output_files = [mcrs_fasta_out, mutations_updated_csv_out, id_to_header_csv_out, mcrs_t2g_out, wt_mcrs_fasta_out, wt_mcrs_t2g_out]
    for output_file in output_files:
        if os.path.isfile(output_file) and not overwrite:
            raise ValueError(f"Output file '{output_file}' already exists. Set 'overwrite=True' to overwrite it.")
        if os.path.dirname(output_file):
            os.makedirs(os.path.dirname(output_file), exist_ok=True)

    merge_identical_rc = not strandedness

    do_not_save_files = kwargs.get("do_not_save_files", False)
    cosmic_email = kwargs.get("cosmic_email", None)
    cosmic_password = kwargs.get("cosmic_password", None)
    cosmic_release = kwargs.get("cosmic_release", None)
    cosmic_grch = kwargs.get("cosmic_grch", None)
    
    if not k:
        k = w + 1

    columns_to_keep = [
        "header",
        seq_id_column,
        mut_column,
        "mutation_type",
        "wt_sequence",
        "mutant_sequence",
        "nucleotide_positions",
        "start_mutation_position",
        "end_mutation_position",
        "actual_mutation",
    ]

    sequences_original = ""

    if isinstance(mutations, str):
        if mutations in supported_databases_and_corresponding_reference_sequence_type and "cosmic" in mutations:
            if not cosmic_release:
                cosmic_release = "100"
            if not cosmic_grch:
                grch_dict = supported_databases_and_corresponding_reference_sequence_type[mutations]["database_version_to_reference_assembly_build"]
                largest_key = max(int(k) for k in grch_dict.keys())
                grch = grch_dict[str(largest_key)]
            else:
                grch = cosmic_grch
                assert grch in supported_databases_and_corresponding_reference_sequence_type[mutations]["database_version_to_reference_assembly_build"], "The 'cosmic_grch' argument must be supported by the corresponding database."
            if grch == "37":
                gget_cosmic_grch = "human_grch37"
            elif grch == "38":
                gget_cosmic_grch = "human"
            else:
                gget_cosmic_grch = grch

    # Load input sequences and their identifiers from fasta file
    if isinstance(sequences, str) and ("." in sequences or (mutations in supported_databases_and_corresponding_reference_sequence_type and sequences in supported_databases_and_corresponding_reference_sequence_type[mutations]["sequence_download_commands"])):
        if mutations in supported_databases_and_corresponding_reference_sequence_type and sequences in supported_databases_and_corresponding_reference_sequence_type[mutations]["sequence_download_commands"]:
            start, peaks_list = report_time_and_memory(report=False, peaks_list=peaks_list)

            # TODO: expand beyond COSMIC
            sequences_original = sequences
            if "cosmic" in mutations:
                ensembl_version = supported_databases_and_corresponding_reference_sequence_type[mutations]["database_version_to_reference_release"][cosmic_release]
                reference_out_sequences = f"{reference_out_dir}/ensembl_grch{grch}_release{ensembl_version}"

                sequences_download_command = supported_databases_and_corresponding_reference_sequence_type[mutations]["sequence_download_commands"][sequences]
                sequences_download_command = sequences_download_command.replace("OUT_DIR", reference_out_sequences)
                sequences_download_command = sequences_download_command.replace(
                    "ENSEMBL_VERSION",
                    ensembl_version,
                )
                sequences_download_command = sequences_download_command.replace("GRCH_NUMBER", gget_cosmic_grch)
                sequences_download_command_list = sequences_download_command.split(" ")

                if sequences == "genome":
                    genome_file = supported_databases_and_corresponding_reference_sequence_type[mutations]["sequence_file_names"]["genome"]
                    genome_file = genome_file.replace("GRCH_NUMBER", grch)
                    genome_file = f"{reference_out_sequences}/{genome_file}"
                    gtf_file = supported_databases_and_corresponding_reference_sequence_type[mutations]["sequence_file_names"]["gtf"]
                    gtf_file = gtf_file.replace("GRCH_NUMBER", grch)
                    gtf_file = f"{reference_out_sequences}/{gtf_file}"
                    gtf_transcript_id_column = "seq_ID"
                    gtf = gtf_file

                    if not os.path.isfile(genome_file) or not os.path.isfile(gtf_file):
                        logger.warning(f"Downloading reference sequences with {' '.join(sequences_download_command_list)}. Note that this requires curl >=7.73.0")
                        subprocess.run(sequences_download_command_list, check=True)

                        subprocess.run(["gunzip", f"{genome_file}.gz"], check=True)
                        subprocess.run(["gunzip", f"{gtf_file}.gz"], check=True)

                    sequences = genome_file

                elif sequences == "cdna" or sequences == "cds":
                    cds_file = supported_databases_and_corresponding_reference_sequence_type[mutations]["sequence_file_names"]["cds"]
                    cds_file = cds_file.replace("GRCH_NUMBER", grch)
                    cds_file = f"{reference_out_sequences}/{cds_file}"
                    if not os.path.isfile(cds_file) and sequences == "cds":
                        logger.warning(f"Downloading reference sequences with {' '.join(sequences_download_command_list)}. Note that this requires curl >=7.73.0")
                        subprocess.run(sequences_download_command_list, check=True)

                        subprocess.run(["gunzip", f"{cds_file}.gz"], check=True)
                    if sequences == "cdna":
                        cdna_file = supported_databases_and_corresponding_reference_sequence_type[mutations]["sequence_file_names"]["cdna"]
                        cdna_file = cdna_file.replace("GRCH_NUMBER", grch)
                        cdna_file = f"{reference_out_sequences}/{cdna_file}"
                        if not os.path.isfile(cdna_file):
                            logger.warning(f"Downloading reference sequences with {' '.join(sequences_download_command_list)}. Note that this requires curl >=7.73.0")
                            subprocess.run(sequences_download_command_list, check=True)

                            subprocess.run(["gunzip", f"{cds_file}.gz"], check=True)
                            subprocess.run(["gunzip", f"{cdna_file}.gz"], check=True)
                        sequences = cdna_file
                    else:
                        sequences = cds_file

                start, peaks_list = report_time_and_memory(process_name="Downloaded reference genome", start=start, peaks_list=peaks_list, logger=logger, report=True)

        titles, seqs = [], []
        for title, seq in read_fasta(sequences):
            titles.append(title)
            seqs.append(seq)
        # titles, seqs = read_fasta(sequences)  # when using gget.utils.read_fasta()

        start, peaks_list = report_time_and_memory(process_name="Loaded in reference sequence", start=start, peaks_list=peaks_list, logger=logger, report=True)

    # Handle input sequences passed as a list
    elif isinstance(sequences, list):
        titles = [f"seq{i+1}" for i in range(len(sequences))]
        seqs = sequences

        start, peaks_list = report_time_and_memory(process_name="Loaded in reference sequence", start=start, peaks_list=peaks_list, logger=logger, report=True)

    # Handle a single sequence passed as a string
    elif isinstance(sequences, str) and "." not in sequences:
        titles = ["seq1"]
        seqs = [sequences]

    else:
        raise ValueError(
            """
            Format of the input to the 'sequences' argument not recognized. 
            'sequences' must be one of the following:
            - Path to the fasta file containing the sequences to be mutated (e.g. 'seqs.fa')
            - A list of sequences to be mutated (e.g. ['ACTGCTAGCT', 'AGCTAGCT'])
            - A single sequence to be mutated passed as a string (e.g. 'AGCTAGCT')
            """
        )

    mutations_path = None

    # logger.warning("Always ensure that the 'sequences' and 'mutations' are compatible with each other. This generally requires the correct source (e.g., Ensembl, RefSeq), version (e.g., GRCh37, GRCh38), and release (e.g., Ensembl release 112 - for transcript locations in particular).")

    start, peaks_list = report_time_and_memory(report=False, peaks_list=peaks_list)

    if isinstance(mutations, str) and mutations in supported_databases_and_corresponding_reference_sequence_type:
        # TODO: expand beyond COSMIC
        if "cosmic" in mutations:
            reference_out_cosmic = f"{reference_out_dir}/cosmic"
            mutations = f"{reference_out_cosmic}/CancerMutationCensus_AllData_Tsv_v{cosmic_release}_GRCh{grch}/CancerMutationCensus_AllData_v{cosmic_release}_GRCh{grch}_mutation_workflow.csv"

            if not os.path.isfile(mutations):
                gget.cosmic(
                    None,
                    grch_version=grch,
                    cosmic_version=cosmic_release,
                    out=reference_out_cosmic,
                    mutation_class="cancer",
                    download_cosmic=True,
                    keep_genome_info=True,
                    remove_duplicates=True,
                    email=cosmic_email,
                    password=cosmic_password,
                )

                if gtf is not None:
                    mutations = merge_gtf_transcript_locations_into_cosmic_csv(
                        mutations,
                        gtf,
                        gtf_transcript_id_column=gtf_transcript_id_column,
                    )
                    columns_to_keep.extend(
                        [
                            "start_transcript_position",
                            "end_transcript_position",
                            "strand",
                        ]
                    )

                if "CancerMutationCensus" in mutations or mutations == "cosmic_cmc":
                    logger.info("COSMIC CMC genome strand information is not fully accurate. Improving with gtf information.")
                    improve_genome_strand_information(mutations, mutation_genome_column_name="mutation_genome")

            if sequences_original == "cdna":
                mutations_with_cdna = mutations.replace(".csv", "_with_cdna.csv")
                if not os.path.isfile(mutations_with_cdna):
                    convert_mutation_cds_locations_to_cdna(
                        input_csv_path=mutations,
                        output_csv_path=mutations_with_cdna,
                        cds_fasta_path=cds_file,
                        cdna_fasta_path=cdna_file,
                    )

                mutations = mutations_with_cdna

            elif sequences_original == "genome":
                mutations_no_duplications = mutations.replace(".csv", "_no_duplications.csv")
                if not os.path.isfile(mutations_no_duplications):
                    logger.info("COSMIC genome location is not accurate for duplications. Dropping duplications in a copy of the csv file.")
                    drop_duplication_mutations(mutations, mutations_no_duplications)  # COSMIC incorrectly records genome positions of duplications

                mutations = mutations_no_duplications

            start, peaks_list = report_time_and_memory(process_name="Download and preprocess COSMIC", start=start, peaks_list=peaks_list, logger=logger, report=True)

    # Read in 'mutations' if passed as filepath to comma-separated csv
    if isinstance(mutations, str) and mutations.endswith(".csv"):
        mutations_path = mutations
        mutations = pd.read_csv(mutations)
        for col in mutations.columns:
            if col not in columns_to_keep:
                columns_to_keep.append(col)  # append "mutation_aa", "gene_name", "mutation_id"

    elif isinstance(mutations, str) and mutations.endswith(".tsv"):
        mutations_path = mutations
        mutations = pd.read_csv(mutations, sep="\t")
        for col in mutations.columns:
            if col not in columns_to_keep:
                columns_to_keep.append(col)  # append "mutation_aa", "gene_name", "mutation_id"

    # Handle mutations passed as a list
    elif isinstance(mutations, list):
        if len(mutations) > 1:
            if len(mutations) != len(seqs):
                raise ValueError("If a list is passed, the number of mutations must equal the number of input sequences.")

            temp = pd.DataFrame()
            temp["mutation"] = mutations
            temp["mut_ID"] = [f"mut{i+1}" for i in range(len(mutations))]
            temp["seq_ID"] = [f"seq{i+1}" for i in range(len(mutations))]
            mutations = temp
        else:
            temp = pd.DataFrame()
            temp["mutation"] = [mutations[0]] * len(seqs)
            temp["mut_ID"] = [f"mut{i+1}" for i in range(len(seqs))]
            temp["seq_ID"] = [f"seq{i+1}" for i in range(len(seqs))]
            mutations = temp

    # Handle single mutation passed as a string
    elif isinstance(mutations, str) and not mutations in supported_databases_and_corresponding_reference_sequence_type:
        # This will work for one mutation for one sequence as well as one mutation for multiple sequences
        mutations_path = mutations
        temp = pd.DataFrame()
        temp["mutation"] = [mutations] * len(seqs)
        temp["mut_ID"] = [f"mut{i+1}" for i in range(len(seqs))]
        temp["seq_ID"] = [f"seq{i+1}" for i in range(len(seqs))]
        mutations = temp

    elif isinstance(mutations, pd.DataFrame):
        pass

    else:
        raise ValueError(
            """
            Format of the input to the 'mutations' argument not recognized. 
            'mutations' must be one of the following:
            - Path to comma-separated csv file (e.g. 'mutations.csv')
            - A pandas DataFrame object
            - A single mutation to be applied to all input sequences (e.g. 'c.2C>T')
            - A list of mutations (the number of mutations must equal the number of input sequences) (e.g. ['c.2C>T', 'c.1A>C'])
            """
        )
    
    start, peaks_list = report_time_and_memory(process_name="Loaded in mutations dataframe", start=start, peaks_list=peaks_list, logger=logger, report=True, dfs={"mutations": mutations}, cols=True)

    # Set of possible nucleotides (- and . are gap annotations)
    nucleotides = set("ATGCUNatgcun.-")

    seq_dict = {}
    non_nuc_seqs = 0
    for title, seq in zip(titles, seqs):
        # Check that sequences are nucleotide sequences
        if not set(seq) <= nucleotides:
            non_nuc_seqs += 1

        # Keep text following the > until the first space/dot as the sequence identifier
        # Dots are removed so Ensembl version numbers are removed
        seq_dict[title.split(" ")[0].split(".")[0]] = seq

    if non_nuc_seqs > 0:
        logger.warning(
            f"""
            Non-nucleotide characters detected in {non_nuc_seqs} input sequences. gget mutate is currently only optimized for mutating nucleotide sequences.
            Specifically inversion mutations might not be performed correctly. 
            """
        )

    number_of_missing_seq_ids = mutations[seq_id_column].isna().sum()

    if number_of_missing_seq_ids > 0:
        logger.warning(
            f"""
            {number_of_missing_seq_ids} rows in 'mutations' are missing sequence IDs. These rows will be dropped from the analysis.
            """
        )

        # Drop rows with missing sequence IDs
        mutations = mutations.dropna(subset=[seq_id_column])

    # ensure seq_ID column is string type, and chromosome numbers don't have decimals
    mutations[seq_id_column] = mutations[seq_id_column].apply(convert_chromosome_value_to_int_when_possible)

    mutations = add_mutation_type(mutations, mut_column)

    start, peaks_list = report_time_and_memory(process_name="Added in mutation types", start=start, peaks_list=peaks_list, logger=logger, report=True, dfs={"mutations": mutations}, cols=True)

    # Link sequences to their mutations using the sequence identifiers
    if store_full_sequences:
        mutations["wt_sequence_full"] = mutations[seq_id_column].map(seq_dict)
        start, peaks_list = report_time_and_memory(process_name="Stored WT full sequences in df", start=start, peaks_list=peaks_list, logger=logger, report=True, dfs={"mutations": mutations}, cols=True)

    # Handle sequences that were not found based on their sequence IDs
    seqs_not_found = mutations[~mutations[seq_id_column].isin(seq_dict.keys())]
    if 0 < len(seqs_not_found) < 20:
        logger.warning(
            f"""
            The sequences with the following {len(seqs_not_found)} sequence ID(s) were not found: {", ".join(seqs_not_found[seq_id_column].values)}  
            These sequences and their corresponding mutations will not be included in the output.  
            Ensure that the sequence IDs correspond to the string following the > character in the 'sequences' fasta file (do NOT include spaces or dots).
            """
        )
    elif len(seqs_not_found) > 0:
        logger.warning(
            f"""
            The sequences corresponding to {len(seqs_not_found)} sequence IDs were not found.  
            These sequences and their corresponding mutations will not be included in the output.  
            Ensure that the sequence IDs correspond to the string following the > character in the 'sequences' fasta file (do NOT include spaces or dots).
            """
        )

    # Drop inputs for sequences that were not found
    mutations = mutations.dropna(subset=[seq_id_column, mut_column])
    if len(mutations) < 1:
        raise ValueError(
            """
            None of the input sequences match the sequence IDs provided in 'mutations'. 
            Ensure that the sequence IDs correspond to the string following the > character in the 'sequences' fasta file (do NOT include spaces or dots).
            """
        )

    total_mutations = mutations.shape[0]

    mutations["mutant_sequence"] = ""

    if mut_id_column is not None:
        mutations["header"] = ">" + mutations[mut_id_column]
    else:
        mutations["header"] = ">" + mutations[seq_id_column] + ":" + mutations[mut_column]

    # Calculate number of bad mutations
    uncertain_mutations = mutations[mut_column].str.contains(r"\?").sum()

    ambiguous_position_mutations = mutations[mut_column].str.contains(r"\(|\)").sum()

    intronic_mutations = mutations[mut_column].str.contains(r"\+|\-").sum()

    posttranslational_region_mutations = mutations[mut_column].str.contains(r"\*").sum()

    # Filter out bad mutations
    combined_pattern = re.compile(r"(\?|\(|\)|\+|\-|\*)")
    mask = mutations[mut_column].str.contains(combined_pattern)
    mutations = mutations[~mask]

    # Extract nucleotide positions and mutation info from Mutation CDS
    mutations[["nucleotide_positions", "actual_mutation"]] = mutations[mut_column].str.extract(mutation_pattern)

    # Filter out mutations that did not match the re
    unknown_mutations = mutations["nucleotide_positions"].isna().sum()
    mutations = mutations.dropna(subset=["nucleotide_positions", "actual_mutation"])

    if mutations.empty:
        logger.warning("No valid mutations found in the input.")
        return []

    # Split nucleotide positions into start and end positions
    split_positions = mutations["nucleotide_positions"].str.split("_", expand=True)

    mutations["start_mutation_position"] = split_positions[0]
    if split_positions.shape[1] > 1:
        mutations["end_mutation_position"] = split_positions[1].fillna(split_positions[0])
    else:
        mutations["end_mutation_position"] = mutations["start_mutation_position"]

    mutations.loc[mutations["end_mutation_position"].isna(), "end_mutation_position"] = mutations["start_mutation_position"]

    mutations[["start_mutation_position", "end_mutation_position"]] = mutations[["start_mutation_position", "end_mutation_position"]].astype(int)

    # Adjust positions to 0-based indexing
    mutations["start_mutation_position"] -= 1
    mutations["end_mutation_position"] -= 1  # don't forget to increment by 1 later

    # Calculate sequence length
    mutations["sequence_length"] = mutations[seq_id_column].apply(lambda x: get_sequence_length(x, seq_dict))

    # Filter out mutations with positions outside the sequence
    index_error_mask = (mutations["start_mutation_position"] > mutations["sequence_length"]) | (mutations["end_mutation_position"] > mutations["sequence_length"])

    mut_idx_outside_seq = index_error_mask.sum()

    mutations = mutations[~index_error_mask]

    if mutations.empty:
        logger.warning("No valid mutations found in the input.")
        return []

    # Create masks for each type of mutation
    mutations["wt_nucleotides_ensembl"] = None
    substitution_mask = mutations["mutation_type"] == "substitution"
    deletion_mask = mutations["mutation_type"] == "deletion"
    delins_mask = mutations["mutation_type"] == "delins"
    insertion_mask = mutations["mutation_type"] == "insertion"
    duplication_mask = mutations["mutation_type"] == "duplication"
    inversion_mask = mutations["mutation_type"] == "inversion"

    if remove_seqs_with_wt_kmers:
        long_duplications = ((duplication_mask) & ((mutations["end_mutation_position"] - mutations["start_mutation_position"]) >= w)).sum()
        logger.info(f"Removing {long_duplications} duplications > w")
        mutations = mutations[~((duplication_mask) & ((mutations["end_mutation_position"] - mutations["start_mutation_position"]) >= w))]

    # Create a mask for all non-substitution mutations
    non_substitution_mask = deletion_mask | delins_mask | insertion_mask | duplication_mask | inversion_mask
    insertion_and_delins_and_dup_and_inversion_mask = insertion_mask | delins_mask | duplication_mask | inversion_mask

    # Extract the WT nucleotides for the substitution rows from reference fasta (i.e., Ensembl)
    start_positions = mutations.loc[substitution_mask, "start_mutation_position"].values

    # Get the nucleotides at the start positions
    wt_nucleotides_substitution = np.array([get_nucleotide_at_position(seq_id, pos, seq_dict) for seq_id, pos in zip(mutations.loc[substitution_mask, seq_id_column], start_positions)])

    mutations.loc[substitution_mask, "wt_nucleotides_ensembl"] = wt_nucleotides_substitution

    # Extract the WT nucleotides for the substitution rows from the Mutation CDS (i.e., COSMIC)
    mutations["wt_nucleotides_cosmic"] = None
    mutations.loc[substitution_mask, "wt_nucleotides_cosmic"] = mutations["actual_mutation"].str[0]

    congruent_wt_bases_mask = (mutations["wt_nucleotides_cosmic"] == mutations["wt_nucleotides_ensembl"]) | mutations[["wt_nucleotides_cosmic", "wt_nucleotides_ensembl"]].isna().any(axis=1)

    cosmic_incorrect_wt_base = (~congruent_wt_bases_mask).sum()

    mutations = mutations[congruent_wt_bases_mask]

    if mutations.empty:
        logger.warning("No valid mutations found in the input.")
        return []

    # Adjust the start and end positions for insertions
    mutations.loc[insertion_mask, "start_mutation_position"] += 1  # in other cases, we want left flank to exclude the start of mutation site; but with insertion, the start of mutation site as it is denoted still belongs in the flank region
    mutations.loc[insertion_mask, "end_mutation_position"] -= 1  # in this notation, the end position is one before the start position

    # Extract the WT nucleotides for the non-substitution rows from the Mutation CDS (i.e., COSMIC)
    mutations.loc[non_substitution_mask, "wt_nucleotides_ensembl"] = mutations.loc[non_substitution_mask].apply(lambda row: extract_sequence(row, seq_dict, seq_id_column), axis=1)

    start, peaks_list = report_time_and_memory(process_name="Various string extractions", start=start, peaks_list=peaks_list, logger=logger, report=True, dfs={"mutations": mutations}, cols=True)

    # Apply mutations to the sequences
    mutations["mut_nucleotides"] = None
    mutations.loc[substitution_mask, "mut_nucleotides"] = mutations.loc[substitution_mask, "actual_mutation"].str[-1]
    mutations.loc[deletion_mask, "mut_nucleotides"] = ""
    mutations.loc[delins_mask, "mut_nucleotides"] = mutations.loc[delins_mask, "actual_mutation"].str.extract(r"delins([A-Z]+)")[0]
    mutations.loc[insertion_mask, "mut_nucleotides"] = mutations.loc[insertion_mask, "actual_mutation"].str.extract(r"ins([A-Z]+)")[0]
    mutations.loc[duplication_mask, "mut_nucleotides"] = mutations.loc[duplication_mask].apply(lambda row: row["wt_nucleotides_ensembl"], axis=1)
    if inversion_mask.any():
        mutations.loc[inversion_mask, "mut_nucleotides"] = mutations.loc[inversion_mask].apply(
            lambda row: "".join(complement.get(nucleotide, "N") for nucleotide in row["wt_nucleotides_ensembl"][::-1]),
            axis=1,
        )

    # Adjust the nucleotide positions of duplication mutations to mimic that of insertions (since duplications are essentially just insertions)
    mutations.loc[duplication_mask, "start_mutation_position"] = mutations.loc[duplication_mask, "end_mutation_position"] + 1  # in the case of duplication, the "mutant" site is still in the left flank as well

    mutations.loc[duplication_mask, "wt_nucleotides_ensembl"] = ""

    # Calculate the kmer bounds
    mutations["start_kmer_position_min"] = mutations["start_mutation_position"] - w
    mutations["start_kmer_position"] = mutations["start_kmer_position_min"].combine(0, max)

    mutations["end_kmer_position_max"] = mutations["end_mutation_position"] + w
    mutations["end_kmer_position"] = mutations[["end_kmer_position_max", "sequence_length"]].min(axis=1)  # don't forget to increment by 1 later on

    start, peaks_list = report_time_and_memory(process_name="Extracting mutational info", start=start, peaks_list=peaks_list, logger=logger, report=True, dfs={"mutations": mutations}, cols=True)

    if gtf is not None:
        assert mutations_path.endswith(".csv") or mutations_path.endswith(".tsv"), "Mutations must be a CSV or TSV file"
        if "start_transcript_position" not in mutations.columns and "end_transcript_position" not in mutations.columns:  # * currently hard-coded column names, but optionally can be changed to arguments later
            mutations = merge_gtf_transcript_locations_into_cosmic_csv(mutations, gtf, gtf_transcript_id_column=gtf_transcript_id_column)

            columns_to_keep.extend(["start_transcript_position", "end_transcript_position", "strand"])
            start, peaks_list = report_time_and_memory(process_name="Merged gtf", start=start, peaks_list=peaks_list, logger=logger, report=True, dfs={"mutations": mutations}, cols=True)
        else:
            logger.warning("Transcript positions already present in the input mutations file. Skipping GTF file merging.")

        # adjust start_transcript_position to be 0-index
        mutations["start_transcript_position"] -= 1

        mutations["start_kmer_position"] = mutations[["start_kmer_position", "start_transcript_position"]].max(axis=1)
        mutations["end_kmer_position"] = mutations[["end_kmer_position", "end_transcript_position"]].min(axis=1)

    mut_apply = (lambda *args, **kwargs: mutations.progress_apply(*args, **kwargs)) if verbose else mutations.apply

    if save_mutations_updated_csv and store_full_sequences:
        # Extract flank sequences
        if verbose:
            tqdm.pandas(desc="Extracting full left flank sequences")

        mutations["left_flank_region_full"] = mut_apply(
            lambda row: seq_dict[row[seq_id_column]][0 : row["start_mutation_position"]],
            axis=1,
        )  # ? vectorize

        if verbose:
            tqdm.pandas(desc="Extracting full right flank sequences")

        mutations["right_flank_region_full"] = mut_apply(
            lambda row: seq_dict[row[seq_id_column]][row["end_mutation_position"] + 1 : row["sequence_length"]],
            axis=1,
        )  # ? vectorize

    if verbose:
        tqdm.pandas(desc="Extracting MCRS left flank sequences")

    mutations["left_flank_region"] = mut_apply(
        lambda row: seq_dict[row[seq_id_column]][row["start_kmer_position"] : row["start_mutation_position"]],
        axis=1,
    )  # ? vectorize

    if verbose:
        tqdm.pandas(desc="Extracting MCRS right flank sequences")

    mutations["right_flank_region"] = mut_apply(
        lambda row: seq_dict[row[seq_id_column]][row["end_mutation_position"] + 1 : row["end_kmer_position"] + 1],
        axis=1,
    )  # ? vectorize

    mutations["inserted_nucleotide_length"] = None

    if insertion_and_delins_and_dup_and_inversion_mask.any():
        mutations.loc[insertion_and_delins_and_dup_and_inversion_mask, "inserted_nucleotide_length"] = mutations.loc[insertion_and_delins_and_dup_and_inversion_mask, "mut_nucleotides"].str.len()

        if insertion_size_limit is not None:
            mutations = mutations[
                (mutations["inserted_nucleotide_length"].isna()) |  # Keep rows where it is None/NaN
                (mutations["inserted_nucleotide_length"] <= insertion_size_limit)     # Keep rows where it's <= insertion_size_limit
            ]

    mutations["beginning_mutation_overlap_with_right_flank"] = 0
    mutations["end_mutation_overlap_with_left_flank"] = 0

    start, peaks_list = report_time_and_memory(process_name="Extracted flank regions", start=start, peaks_list=peaks_list, logger=logger, report=True, dfs={"mutations": mutations}, cols=True)

    # Rules for shaving off kmer ends - r1 = left flank, r2 = right flank, d = deleted portion, i = inserted portion
    # Substitution: N/A
    # Deletion:
    # To what extend the beginning of d overlaps with the beginning of r2 --> shave up to that many nucleotides off the beginning of r1 until w - len(r1) ≥ extent of overlap
    # To what extend the end of d overlaps with the beginning of r1 --> shave up to that many nucleotides off the end of r2 until w - len(r2) ≥ extent of overlap
    # Insertion, Duplication:
    # To what extend the beginning of i overlaps with the beginning of r2 --> shave up to that many nucleotides off the beginning of r1 until w - len(r1) ≥ extent of overlap
    # To what extend the end of i overlaps with the beginning of r1 --> shave up to that many nucleotides off the end of r2 until w - len(r2) ≥ extent of overlap
    # Delins, inversion:
    # To what extend the beginning of i overlaps with the beginning of d --> shave up to that many nucleotides off the beginning of r1 until w - len(r1) ≥ extent of overlap
    # To what extend the end of i overlaps with the beginning of d --> shave up to that many nucleotides off the end of r2 until w - len(r2) ≥ extent of overlap
    if optimize_flanking_regions and non_substitution_mask.any():
        # Apply the function for beginning of mut_nucleotides with right_flank_region
        mutations.loc[non_substitution_mask, "beginning_mutation_overlap_with_right_flank"] = mutations.loc[non_substitution_mask].apply(calculate_beginning_mutation_overlap_with_right_flank, axis=1)

        # Apply the function for end of mut_nucleotides with left_flank_region
        mutations.loc[non_substitution_mask, "end_mutation_overlap_with_left_flank"] = mutations.loc[non_substitution_mask].apply(calculate_end_mutation_overlap_with_left_flank, axis=1)

        # for insertions and delins, make sure I see at bare minimum the full insertion context and the subseqeuent nucleotide - eg if I have c.2_3insA to become ACGTT to ACAGTT, if I only check for ACAG, then I can't distinguosh between ACAGTT, ACAGGTT, ACAGGGTT, etc. (and there are more complex examples)
        if required_insertion_overlap_length and insertion_and_delins_and_dup_and_inversion_mask.any():  #* new as of 11/20/24
            if required_insertion_overlap_length == "all":
                required_insertion_overlap_length = np.inf
            
            if required_insertion_overlap_length >= 2*w:
                mutations = mutations[
                    (mutations["inserted_nucleotide_length"].isna()) |  # Keep rows where it is None/NaN
                    (mutations["inserted_nucleotide_length"] < 2*w)     # Keep rows where it's < 2*w
                ]
            
            mutations.loc[insertion_and_delins_and_dup_and_inversion_mask, "beginning_mutation_overlap_with_right_flank"] = np.maximum(
                mutations.loc[insertion_and_delins_and_dup_and_inversion_mask, "beginning_mutation_overlap_with_right_flank"], np.minimum(mutations.loc[insertion_and_delins_and_dup_and_inversion_mask, "inserted_nucleotide_length"], required_insertion_overlap_length),
            )

            mutations.loc[insertion_and_delins_and_dup_and_inversion_mask, "end_mutation_overlap_with_left_flank"] = np.maximum(
                mutations.loc[insertion_and_delins_and_dup_and_inversion_mask, "end_mutation_overlap_with_left_flank"], np.minimum(mutations.loc[insertion_and_delins_and_dup_and_inversion_mask, "inserted_nucleotide_length"], required_insertion_overlap_length),
            )
        
        # Calculate w-len(flank) (see above instructions)
        mutations.loc[non_substitution_mask, "k_minus_left_flank_length"] = w - mutations.loc[non_substitution_mask, "left_flank_region"].apply(len)
        mutations.loc[non_substitution_mask, "k_minus_right_flank_length"] = w - mutations.loc[non_substitution_mask, "right_flank_region"].apply(len)

        mutations.loc[non_substitution_mask, "updated_left_flank_start"] = np.maximum(
            mutations.loc[non_substitution_mask, "beginning_mutation_overlap_with_right_flank"] - mutations.loc[non_substitution_mask, "k_minus_left_flank_length"],
            0,
        )
        mutations.loc[non_substitution_mask, "updated_right_flank_end"] = np.maximum(
            mutations.loc[non_substitution_mask, "end_mutation_overlap_with_left_flank"] - mutations.loc[non_substitution_mask, "k_minus_right_flank_length"],
            0,
        )

        mutations["updated_left_flank_start"] = mutations["updated_left_flank_start"].fillna(0).astype(int)
        mutations["updated_right_flank_end"] = mutations["updated_right_flank_end"].fillna(0).astype(int)

        start, peaks_list = report_time_and_memory(process_name="Optimized flank regions", start=start, peaks_list=peaks_list, logger=logger, report=True, dfs={"mutations": mutations}, cols=True)

    else:
        mutations["updated_left_flank_start"] = 0
        mutations["updated_right_flank_end"] = 0

    # Create WT substitution w-mer sequences
    if substitution_mask.any():
        mutations.loc[substitution_mask, "wt_sequence"] = mutations.loc[substitution_mask, "left_flank_region"] + mutations.loc[substitution_mask, "wt_nucleotides_ensembl"] + mutations.loc[substitution_mask, "right_flank_region"]

    # Create WT non-substitution w-mer sequences
    if non_substitution_mask.any():
        mutations.loc[non_substitution_mask, "wt_sequence"] = mutations.loc[non_substitution_mask].apply(
            lambda row: row["left_flank_region"][row["updated_left_flank_start"] :] + row["wt_nucleotides_ensembl"] + row["right_flank_region"][: len(row["right_flank_region"]) - row["updated_right_flank_end"]],
            axis=1,
        )

    # Create mutant substitution w-mer sequences
    if substitution_mask.any():
        mutations.loc[substitution_mask, "mutant_sequence"] = mutations.loc[substitution_mask, "left_flank_region"] + mutations.loc[substitution_mask, "mut_nucleotides"] + mutations.loc[substitution_mask, "right_flank_region"]

    # Create mutant non-substitution w-mer sequences
    if non_substitution_mask.any():
        mutations.loc[non_substitution_mask, "mutant_sequence"] = mutations.loc[non_substitution_mask].apply(
            lambda row: row["left_flank_region"][row["updated_left_flank_start"] :] + row["mut_nucleotides"] + row["right_flank_region"][: len(row["right_flank_region"]) - row["updated_right_flank_end"]],
            axis=1,
        )

    start, peaks_list = report_time_and_memory(process_name="Created wt/mutant mutation-containing reference sequences", start=start, peaks_list=peaks_list, logger=logger, report=True, dfs={"mutations": mutations}, cols=True)

    if remove_seqs_with_wt_kmers:
        if verbose:
            tqdm.pandas(desc="Removing mutant fragments that share a kmer with wt fragments")

        mutations["wt_fragment_and_mutant_fragment_share_kmer"] = mut_apply(
            lambda row: wt_fragment_and_mutant_fragment_share_kmer(
                mutated_fragment=row["mutant_sequence"],
                wildtype_fragment=row["wt_sequence"],
                k=k,
            ),
            axis=1,
        )

        mutations_overlapping_with_wt = mutations["wt_fragment_and_mutant_fragment_share_kmer"].sum()

        mutations = mutations[~mutations["wt_fragment_and_mutant_fragment_share_kmer"]]

        start, peaks_list = report_time_and_memory(process_name="Removed MCRSs with WT k-mers (even after flank optimization, if enabled)", start=start, peaks_list=peaks_list, logger=logger, report=True, dfs={"mutations": mutations}, cols=True)

    if save_mutations_updated_csv and store_full_sequences:
        columns_to_keep.extend(["wt_sequence_full", "mutant_sequence_full"])

        # Create full sequences (substitution and non-substitution)
        mutations["mutant_sequence_full"] = mutations["left_flank_region_full"] + mutations["mut_nucleotides"] + mutations["right_flank_region_full"]

        start, peaks_list = report_time_and_memory(process_name="Stored mutant full sequences in df", start=start, peaks_list=peaks_list, logger=logger, report=True, dfs={"mutations": mutations}, cols=True)

    if min_seq_len:
        # Calculate k-mer lengths (where k=w) and report the distribution
        mutations["mutant_sequence_kmer_length"] = mutations["mutant_sequence"].apply(lambda x: len(x) if pd.notna(x) else 0)

        rows_less_than_minimum = (mutations["mutant_sequence_kmer_length"] < min_seq_len).sum()

        mutations = mutations[mutations["mutant_sequence_kmer_length"] >= min_seq_len]

        if verbose:
            logger.info(f"Removed {rows_less_than_minimum} mutant kmers with length less than {min_seq_len}...")

        start, peaks_list = report_time_and_memory(process_name="Removed short sequences", start=start, peaks_list=peaks_list, logger=logger, report=True, dfs={"mutations": mutations}, cols=True)

    if max_ambiguous is not None:
        # Get number of 'N' or 'n' occuring in the sequence
        mutations["num_N"] = mutations["mutant_sequence"].str.lower().str.count("n")
        num_rows_with_N = (mutations["num_N"] > max_ambiguous).sum()
        mutations = mutations[mutations["num_N"] <= max_ambiguous]

        if verbose:
            logger.info(f"Removed {num_rows_with_N} mutant kmers containing more than {max_ambiguous} 'N's...")

        # Drop the 'num_N' column after filtering
        mutations = mutations.drop(columns=["num_N"])

    # Report status of mutations back to user
    good_mutations = mutations.shape[0]

    report = f"""
        {good_mutations} mutations correctly recorded ({good_mutations/total_mutations*100:.2f}%)
        {intronic_mutations} intronic mutations found ({intronic_mutations/total_mutations*100:.2f}%)
        {posttranslational_region_mutations} posttranslational region mutations found ({posttranslational_region_mutations/total_mutations*100:.2f}%)
        {unknown_mutations} unknown mutations found ({unknown_mutations/total_mutations*100:.2f}%)
        {uncertain_mutations} mutations with uncertain mutation found ({uncertain_mutations/total_mutations*100:.2f}%)
        {ambiguous_position_mutations} mutations with ambiguous position found ({ambiguous_position_mutations/total_mutations*100:.2f}%)
        {cosmic_incorrect_wt_base} mutations with incorrect wildtype base found ({cosmic_incorrect_wt_base/total_mutations*100:.2f}%)
        {mut_idx_outside_seq} mutations with indices outside of the sequence length found ({mut_idx_outside_seq/total_mutations*100:.2f}%)
        """

    if remove_seqs_with_wt_kmers:
        report += f"""{long_duplications} duplications longer than w found ({long_duplications/total_mutations*100:.2f}%)
        {mutations_overlapping_with_wt} mutations with overlapping kmers found ({mutations_overlapping_with_wt/total_mutations*100:.2f}%)
        """

    if min_seq_len:
        report += f"""{rows_less_than_minimum} mutations with fragment length < w found ({rows_less_than_minimum/total_mutations*100:.2f}%)
        """

    if max_ambiguous is not None:
        report += f"""{num_rows_with_N} mutations with Ns found ({num_rows_with_N/total_mutations*100:.2f}%)
        """

    if good_mutations != total_mutations:
        logger.warning(report)
    else:
        logger.info("All mutations correctly recorded")

    if translate and save_mutations_updated_csv and store_full_sequences:
        columns_to_keep.extend(["wt_sequence_aa_full", "mutant_sequence_aa_full"])

        if not mutations_path:
            assert type(translate_start) != str and type(translate_end) != str, "translate_start and translate_end must be integers when translating sequences (or default None)."
            if translate_start is None:
                translate_start = 0
            if translate_end is None:
                translate_end = mutations["sequence_length"][0]

            # combined_df['ORF'] = combined_df[translate_start] % 3

            if verbose:
                tqdm.pandas(desc="Translating WT amino acid sequences")
                mutations["wt_sequence_aa_full"] = mutations["wt_sequence_full"].progress_apply(lambda x: translate_sequence(x, start=translate_start, end=translate_end))
            else:
                mutations["wt_sequence_aa_full"] = mutations["wt_sequence_full"].apply(lambda x: translate_sequence(x, start=translate_start, end=translate_end))

            if verbose:
                tqdm.pandas(desc="Translating mutant amino acid sequences")

                mutations["mutant_sequence_aa_full"] = mutations["mutant_sequence_full"].progress_apply(lambda x: translate_sequence(x, start=translate_start, end=translate_end))

            else:
                mutations["mutant_sequence_aa_full"] = mutations["mutant_sequence_full"].apply(lambda x: translate_sequence(x, start=translate_start, end=translate_end))

            print(f"Translated mutated sequences: {mutations['wt_sequence_aa_full']}")
        else:
            if not translate_start:
                translate_start = "translate_start"

            if not translate_end:
                translate_end = "translate_end"

            if translate_start not in mutations.columns:
                mutations["translate_start"] = 0

            if translate_end not in mutations.columns:
                mutations["translate_end"] = mutations["sequence_length"]

            if verbose:
                tqdm.pandas(desc="Translating WT amino acid sequences")

            mutations["wt_sequence_aa_full"] = mut_apply(
                lambda row: translate_sequence(row["wt_sequence_full"], row[translate_start], row[translate_end]),
                axis=1,
            )

            if verbose:
                tqdm.pandas(desc="Translating mutant amino acid sequences")

            mutations["mutant_sequence_aa_full"] = mut_apply(
                lambda row: translate_sequence(
                    row["mutant_sequence_full"],
                    row[translate_start],
                    row[translate_end],
                ),
                axis=1,
            )
        
        start, peaks_list = report_time_and_memory(process_name="Translated mutations", start=start, peaks_list=peaks_list, logger=logger, report=True, dfs={"mutations": mutations}, cols=True)

    mutations = mutations[columns_to_keep]

    if save_mutations_updated_csv:
        # recalculate start_mutation_position and end_mutation_position due to messing with it above
        mutations.drop(
            columns=["start_mutation_position", "end_mutation_position"],
            inplace=True,
            errors="ignore",
        )
        mutations["start_mutation_position"] = split_positions[0]
        if split_positions.shape[1] > 1:
            mutations["end_mutation_position"] = split_positions[1].fillna(split_positions[0])
        else:
            mutations["end_mutation_position"] = mutations["start_mutation_position"]

        mutations[["start_mutation_position", "end_mutation_position"]] = mutations[["start_mutation_position", "end_mutation_position"]].astype(int)

    if merge_identical:
        logger.info("Merging identical mutated sequences")

        if merge_identical_rc:
            mutations["mutant_sequence_rc"] = mutations["mutant_sequence"].apply(reverse_complement)

            # Create a column that stores a sorted tuple of (mutant_sequence, mutant_sequence_rc)
            mutations["mutant_sequence_and_rc_tuple"] = mutations.apply(
                lambda row: tuple(sorted([row["mutant_sequence"], row["mutant_sequence_rc"]])),
                axis=1,
            )

            # mutations = mutations.drop(columns=['mutant_sequence_rc'])

            group_key = "mutant_sequence_and_rc_tuple"
            columns_not_to_semicolon_join = [
                "mutant_sequence",
                "mutant_sequence_rc",
                "mutant_sequence_and_rc_tuple",
            ]
            agg_columns = mutations.columns

        else:
            group_key = "mutant_sequence"
            columns_not_to_semicolon_join = []
            agg_columns = [col for col in mutations.columns if col != "mutant_sequence"]

        if save_mutations_updated_csv:
            logger.warning("Merging identical mutated sequences can take a while if save_mutations_updated_csv=True since it will concatenate all MCRSs too)")
            mutations = (
                mutations.groupby(group_key, sort=False).agg({col: ("first" if col in columns_not_to_semicolon_join else (";".join if col == "header" else lambda x: list(x.fillna(np.nan)))) for col in agg_columns}).reset_index(drop=merge_identical_rc)
            )  # lambda x: list(x) will make simple list, but lengths will be inconsistent with NaN values  # concatenate values with semicolons: lambda x: `";".join(x.astype(str))`   # drop if merging by mutant_sequence_and_rc_tuple, but not if merging by mutant_sequence

        else:
            mutations_temp = mutations.groupby(group_key, sort=False, group_keys=False)["header"].apply(";".join).reset_index()

            if merge_identical_rc:
                mutations_temp = mutations_temp.merge(mutations[["mutant_sequence", group_key]], on=group_key, how="left")
                mutations_temp = mutations_temp.drop_duplicates(subset="header")
                mutations_temp.drop(columns=[group_key], inplace=True)

            mutations = mutations_temp

        start, peaks_list = report_time_and_memory(process_name="Merged by identical MCRSs", start=start, peaks_list=peaks_list, logger=logger, report=True, dfs={"mutations": mutations}, cols=True)

        if "mutant_sequence_and_rc_tuple" in mutations.columns:
            mutations = mutations.drop(columns=["mutant_sequence_and_rc_tuple"])

        # apply remove_gt_after_semicolon to mutant_sequence
        mutations["header"] = mutations["header"].apply(remove_gt_after_semicolon)

        # Calculate the number of semicolons in each entry
        mutations["semicolon_count"] = mutations["header"].str.count(";")

        mutations["semicolon_count"] += 1

        # Convert all 1 values to NaN
        mutations["semicolon_count"] = mutations["semicolon_count"].replace(1, np.nan)

        # Take the sum across all rows of the new column
        total_semicolons = int(mutations["semicolon_count"].sum())

        mutations = mutations.drop(columns=["semicolon_count"])

        if verbose:
            logger.info(f"{total_semicolons} identical mutated sequences were merged (headers were combined and separated using a semicolon (;). Occurences of identical mutated sequences may be reduced by increasing w.")

    empty_kmer_count = (mutations["mutant_sequence"] == "").sum()

    if empty_kmer_count > 0 and verbose:
        logger.warning(f"{empty_kmer_count} mutated sequences were empty and were not included in the output.")

    mutations = mutations[mutations["mutant_sequence"] != ""]

    mutations["header"] = mutations["header"].str[1:]  # remove the > character

    if not keep_original_headers:  # or (mut_id_column in mutations.columns and not merge_identical):
        mutations["mcrs_id"] = generate_unique_ids(len(mutations))
        if not do_not_save_files:
            mutations[["mcrs_id", "header"]].to_csv(id_to_header_csv_out, index=False)  # make the mapping csv
            start, peaks_list = report_time_and_memory(process_name="Saved ID to header file", start=start, peaks_list=peaks_list, logger=logger, report=True, dfs={"mutations": mutations}, cols=True)
    else:
        mutations["mcrs_id"] = mutations["header"]

    if save_mutations_updated_csv:  # use mutations_updated_csv_out if present,
        logger.info("Saving dataframe with updated mutation info...")
        logger.warning("File size can be very large if the number of mutations is large.")
        mutations.to_csv(mutations_updated_csv_out, index=False)
        print(f"Updated mutation info has been saved to {mutations_updated_csv_out}")
        start, peaks_list = report_time_and_memory(process_name="Saved updated df", start=start, peaks_list=peaks_list, logger=logger, report=True, dfs={"mutations": mutations}, cols=True)

    if len(mutations) > 0:
        mutations["fasta_format"] = ">" + mutations["mcrs_id"] + "\n" + mutations["mutant_sequence"] + "\n"

        if save_wt_mcrs_fasta_and_t2g:
            assert save_mutations_updated_csv, "save_mutations_updated_csv must be True to create wt_mcrs_counterpart_fa"

            mutations_with_exactly_1_wt_sequence_per_row = mutations[["mcrs_id", "wt_sequence"]].copy()

            if merge_identical:  # remove the rows with multiple WT counterparts for 1 MCRS, and convert the list of strings to string
                # Step 1: Filter rows where the length of the set of the list in `wt_sequence` is 1
                mutations_with_exactly_1_wt_sequence_per_row = mutations_with_exactly_1_wt_sequence_per_row[mutations_with_exactly_1_wt_sequence_per_row["wt_sequence"].apply(lambda x: len(set(x)) == 1)]

                # Step 2: Convert the list to a string
                mutations_with_exactly_1_wt_sequence_per_row["wt_sequence"] = mutations_with_exactly_1_wt_sequence_per_row["wt_sequence"].apply(lambda x: x[0])

            mutations_with_exactly_1_wt_sequence_per_row["fasta_format_wt"] = ">" + mutations_with_exactly_1_wt_sequence_per_row["mcrs_id"] + "\n" + mutations_with_exactly_1_wt_sequence_per_row["wt_sequence"] + "\n"

    # Save mutated sequences in new fasta file
    if not do_not_save_files:
        with open(mcrs_fasta_out, "w") as fasta_file:
            fasta_file.write("".join(mutations["fasta_format"].values))

        create_mutant_t2g(mcrs_fasta_out, mcrs_t2g_out)

    if verbose:
        logger.info(f"FASTA file containing mutated sequences created at {mcrs_fasta_out}.")
        logger.info(f"t2g file containing mutated sequences created at {mcrs_t2g_out}.")

    if save_wt_mcrs_fasta_and_t2g:
        with open(wt_mcrs_fasta_out, "w") as fasta_file:
            fasta_file.write("".join(mutations_with_exactly_1_wt_sequence_per_row["fasta_format_wt"].values))
        create_mutant_t2g(wt_mcrs_fasta_out, wt_mcrs_t2g_out)  # separate t2g is needed because it may have a subset of the rows of mutant (because it doesn't contain any MCRSs with merged mutations and 2+ originating WT sequences)

    start, peaks_list = report_time_and_memory(process_name="Wrote fasta file(s) and t2g(s)", start=start, peaks_list=peaks_list, logger=logger, report=True, dfs={"mutations": mutations}, cols=True)

    # When stream_output is True, return list of mutated seqs
    if return_mutation_output:
        all_mut_seqs = []
        all_mut_seqs.extend(mutations["mutant_sequence"].values)

        # Remove empty strings from final list of mutated sequences
        # (these are introduced when unknown mutations are encountered)
        while "" in all_mut_seqs:
            all_mut_seqs.remove("")

        if len(all_mut_seqs) > 0:
            return all_mut_seqs
        
    start, peaks_list = report_time_and_memory(start=start_overall, peaks_list=peaks_list, logger=logger, report=True, final_call=True, dfs={"mutations": mutations}, cols=True)
