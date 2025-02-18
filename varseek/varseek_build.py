"""varseek build and specific helper functions."""

import os
from pathlib import Path
import re
import subprocess
import time
import logging

import gget
import numpy as np
import pandas as pd
import pyfastx
from tqdm import tqdm

from .constants import (
    complement,
    fasta_extensions,
    mutation_pattern,
    supported_databases_and_corresponding_reference_sequence_type,
)
from .utils import (
    add_mutation_type,
    check_file_path_is_string_with_valid_extension,
    convert_chromosome_value_to_int_when_possible,
    convert_mutation_cds_locations_to_cdna,
    create_identity_t2g,
    generate_mutation_notation_from_vcf_columns,
    generate_unique_ids,
    is_valid_int,
    make_function_parameter_to_value_dict,
    print_varseek_dry_run,
    report_time_elapsed,
    reverse_complement,
    save_params_to_config_file,
    save_run_info,
    set_up_logger,
    translate_sequence,
    vcf_to_dataframe,
    wt_fragment_and_mutant_fragment_share_kmer,
)

tqdm.pandas()
logger = logging.getLogger(__name__)

# Define global variables to count occurences of weird mutations
intronic_mutations = 0
posttranslational_region_mutations = 0
unknown_mutations = 0
uncertain_mutations = 0
ambiguous_position_mutations = 0
variants_incorrect_wt_base = 0
mut_idx_outside_seq = 0


def print_valid_values_for_variants_and_sequences_in_varseek_build():
    mydict = supported_databases_and_corresponding_reference_sequence_type

    # mydict.keys() has mutations, and mydict[mutation]["sequence_download_commands"].keys() has sequences
    message = "vk build internally supported values for 'variants' and 'sequences' are as follows:\n"
    for mutation, mutation_data in mydict.items():
        sequences = list(mutation_data["sequence_download_commands"].keys())
        message += f"'variants': {mutation}\n"
        message += f"  'sequences': {', '.join(sequences)}\n"
    print(message)


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
    print(f"Number of duplication mutations that have been dropped: {num_dup}")

    df_no_dup_mutations = df.loc[~(df[mutation_column].str.contains("dup"))]

    df_no_dup_mutations.to_csv(output, index=False)

    return df_no_dup_mutations


def improve_genome_strand_information(cosmic_reference_file_mutation_csv, mutation_genome_column_name="mutation_genome", output_mutations_path=None):
    if isinstance(cosmic_reference_file_mutation_csv, str):
        df = pd.read_csv(cosmic_reference_file_mutation_csv)
    elif isinstance(cosmic_reference_file_mutation_csv, pd.DataFrame):
        df = cosmic_reference_file_mutation_csv
    else:
        raise ValueError("cosmic_reference_file_mutation_csv must be a string or a DataFrame.")

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

    df.loc[ins_delins_mask, ["mutation_type", "mut_nucleotides"]] = df.loc[ins_delins_mask, "actual_variant"].str.extract(r"(delins|ins)([A-Z]+)").values

    df.loc[ins_delins_mask, "mut_nucleotides_rc"] = df.loc[ins_delins_mask, "mut_nucleotides"].apply(reverse_complement_insertion)

    df.loc[ins_delins_mask, "actual_variant_rc"] = df.loc[ins_delins_mask, "mutation_type"] + df.loc[ins_delins_mask, "mut_nucleotides_rc"]

    df["actual_variant_final"] = np.where(df["strand_modified"] == "+", df["actual_variant"], df["actual_variant_rc"])

    df["mutation_genome"] = np.where(
        df["GENOME_START"] != df["GENOME_STOP"],
        "g." + df["GENOME_START"].astype(str) + "_" + df["GENOME_STOP"].astype(str) + df["actual_variant_final"],
        "g." + df["GENOME_START"].astype(str) + df["actual_variant_final"],
    )

    df.drop(
        columns=["GENOME_START", "GENOME_STOP", "nucleotide_positions", "actual_variant", "actual_variant_rc", "mutation_type", "mut_nucleotides", "mut_nucleotides_rc", "actual_variant_final", "strand_modified"],
        inplace=True,
    )  # drop all columns except mutation_genome (and the original ones)

    if output_mutations_path:
        df.to_csv(output_mutations_path, index=False)

    return df


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


def validate_input_build(params_dict):
    # Required parameters
    # sequences
    sequences = params_dict.get("sequences")
    mutations = params_dict.get("variants")  # apologies for the naming confusion

    if not isinstance(sequences, (list, str, Path)):
        raise ValueError(f"sequences must be a nucleotide string, a list of nucleotide strings, a path to a reference genome, or a string specifying a reference genome supported by varseek. Got {type(sequences)}\nTo see a list of supported variant databases and reference genomes, please use the 'list_supported_databases' flag/argument.")
    if isinstance(sequences, list) and not all(isinstance(seq, str) for seq in sequences):
        raise ValueError("All elements in sequences must be nucleotide strings.")
    if isinstance(sequences, str):
        if all(c in "ACGTNU-.*" for c in sequences.upper()):  # a single reference sequence
            pass
        elif supported_databases_and_corresponding_reference_sequence_type.get(mutations, {}).get("sequence_file_names", {}).get(sequences, None):  # a supported reference genome
            pass
        elif os.path.isfile(sequences) and sequences.endswith(fasta_extensions):  # a path to a reference genome with a valid extension
            pass
        else:
            raise ValueError(f"sequences must be a nucleotide string, a list of nucleotide strings, a path to a reference genome, or a string specifying a reference genome supported by varseek. Got {type(sequences)}.\nTo see a list of supported variant databases and reference genomes, please use the 'list_supported_databases' flag/argument.")

    # mutations
    if not isinstance(mutations, (list, str, Path)):
        raise ValueError(f"variants must be a string, a list of strings, a path to a variant database, or a string specifying a variant database supported by varseek. Got {type(mutations)}\nTo see a list of supported variant databases and reference genomes, please use the 'list_supported_databases' flag/argument.")
    if isinstance(mutations, list) and not all((isinstance(mut, str) and mut.startswith(("c.", "g."))) for mut in mutations):
        raise ValueError("All elements in variants must be strings that start with 'c.' or 'g.'.")
    if isinstance(mutations, str):
        if mutations.startswith("c.") or mutations.startswith("g."):  # a single mutation
            pass
        elif mutations in supported_databases_and_corresponding_reference_sequence_type:  # a supported mutation database
            if sequences not in supported_databases_and_corresponding_reference_sequence_type[mutations]["sequence_download_commands"]:
                raise ValueError(f"sequences {sequences} not internally supported.\nTo see a list of supported variant databases and reference genomes, please use the 'list_supported_databases' flag/argument.")
        elif os.path.isfile(mutations) and mutations.endswith(accepted_build_file_types):  # a path to a mutation database with a valid extension
            pass
        else:
            raise ValueError(f"variants must be a string, a list of strings, a path to a variant database, or a string specifying a variant database supported by varseek. Got {type(mutations)}.\nTo see a list of supported variant databases and reference genomes, please use the 'list_supported_databases' flag/argument.")

    # Directories
    if not isinstance(params_dict.get("out", None), (str, Path)):
        raise ValueError(f"Invalid value for out: {params_dict.get('out', None)}")
    if params_dict.get("reference_out_dir", None) and not isinstance(params_dict.get("reference_out_dir", None), (str, Path)):
        raise ValueError(f"Invalid value for reference_out_dir: {params_dict.get('reference_out_dir', None)}")
    
    gtf = params_dict.get("gtf")  # gtf gets special treatment because it can be a bool
    if gtf is not None and not isinstance(gtf, bool) and not gtf.lower().endswith(("gtf", "gtf.zip", "gtf.gz")):
        raise ValueError(f"Invalid value for gtf: {gtf}. Expected gtf filepath string, bool, or None.")

    # file paths
    for param_name, file_type in {
        "vcrs_fasta_out": "fasta",
        "variants_updated_csv_out": "csv",
        "id_to_header_csv_out": "csv",
        "vcrs_t2g_out": "t2g",
        "wt_vcrs_fasta_out": "fasta",
        "wt_vcrs_t2g_out": "t2g",
        "removed_variants_text_out": "txt",
    }.items():
        check_file_path_is_string_with_valid_extension(params_dict.get(param_name), param_name, file_type)

    # column names
    for column, optional_status in [("var_column", False), ("seq_id_column", False), ("var_id_column", True), ("gtf_transcript_id_column", True)]:
        if not (isinstance(params_dict.get(column), str) or (optional_status and params_dict.get(column) is None)):
            raise ValueError(f"Invalid column name for {column}: {params_dict.get(column)}")

    # integers - optional just means that it's in kwargs
    for param_name, min_value, optional_value in [
        ("w", 1, False),
        ("k", 1, False),
        ("max_ambiguous", 0, False),
        ("insertion_size_limit", 1, True),
        ("min_seq_len", 1, True),
    ]:
        param_value = params_dict.get(param_name)
        if not is_valid_int(param_value, ">=", min_value, optional=optional_value):
            raise ValueError(f"{param_name} must be an integer >= {min_value}. Got {params_dict.get(param_name)}.")

    k = params_dict.get("k", None)
    w = params_dict.get("w", None)
    if int(k) % 2 == 0 or int(k) > 63:
        logger.warning("If running a workflow with vk ref or kb ref, k should be an odd number between 1 and 63. Got k=%s.", k)
    if int(w) >= int(k):
        raise ValueError(f"w should be less than k. Got w={w}, k={k}.")

    # required_insertion_overlap_length
    if params_dict.get("required_insertion_overlap_length") is not None and not is_valid_int(params_dict.get("required_insertion_overlap_length"), ">=", 1) and params_dict.get("required_insertion_overlap_length") != "all":
        raise ValueError(f"required_insertion_overlap_length must be an int >= 1, the string 'all', or None. Got {type(params_dict.get('required_insertion_overlap_length'))}.")

    # Boolean - optional_status means that None is also valid (because the kwargs is defined later)
    for param_name, optional_status in [
        ("optimize_flanking_regions", True),
        ("remove_seqs_with_wt_kmers", True),
        ("merge_identical", True),
        ("vcrs_strandedness", True),
        ("use_IDs", True),
        ("save_wt_vcrs_fasta_and_t2g", False),
        ("save_variants_updated_csv", False),
        ("store_full_sequences", False),
        ("translate", False),
        ("return_variant_output", True),
        ("save_removed_variants_text", False),
        ("save_filtering_report_text", False),
        ("dry_run", False),
        ("verbose", False),
        ("list_supported_databases", False),
        ("overwrite", False),
    ]:
        if not (isinstance(params_dict.get(param_name), bool) or (optional_status and params_dict.get(param_name) is None)):
            raise ValueError(f"{param_name} must be a boolean. Got {param_name} of type {type(params_dict.get(param_name))}.")

    # Validate translation parameters
    for param_name in ["translate_start", "translate_end"]:
        param_value = params_dict.get(param_name)
        if param_value is not None and not isinstance(param_value, (int, str)):
            raise ValueError(f"{param_name} must be an int, a string, or None. Got param_name of type {type(param_value)}.")


accepted_build_file_types = (".csv", ".tsv", ".vcf")


def build(
    sequences,
    variants,
    w=54,
    k=59,
    max_ambiguous=0,
    var_column="mutation",
    seq_id_column="seq_ID",
    var_id_column=None,
    gtf=None,
    gtf_transcript_id_column=None,
    transcript_boundaries=False,
    identify_all_spliced_from_genome=False,
    out=".",
    reference_out_dir=None,
    vcrs_fasta_out=None,
    variants_updated_csv_out=None,
    id_to_header_csv_out=None,
    vcrs_t2g_out=None,
    wt_vcrs_fasta_out=None,
    wt_vcrs_t2g_out=None,
    removed_variants_text_out=None,
    filtering_report_text_out=None,
    return_variant_output=False,
    save_variants_updated_csv=False,
    save_wt_vcrs_fasta_and_t2g=False,
    save_removed_variants_text=True,
    save_filtering_report_text=True,
    store_full_sequences=False,
    translate=False,
    translate_start=None,
    translate_end=None,
    dry_run=False,
    list_supported_databases=False,
    overwrite=False,
    logging_level=None,
    save_logs=False,
    log_out_dir=None,
    verbose=False,
    **kwargs,
):
    """
    Takes in nucleotide sequences and variants (in standard mutation/variant annotation - see below)
    and returns sequences containing the variants and the surrounding local context, dubbed variant-containing reference sequences (VCRSs),
    compatible with k-mer-based methods (i.e., kallisto | bustools) for variant detection.

    # Required input argument:
    - sequences     (str) Path to the fasta file containing the sequences to have the variants added, e.g., 'seqs.fa'.
                    Sequence identifiers following the '>' character must correspond to the identifiers
                    in the seq_ID column of 'variants'.

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

                    Alternatively, if 'variants' is a string specifying a supported database,
                    sequences can be a string indicating the source upon which to apply the variants.
                    See below for supported databases and sequences options.
                    To see the supported combinations of variants and sequences, either
                    1) run `vk build --list_supported_databases` from the command line, or
                    2) run varseek.build(list_supported_databases=True) in python

    - variants     (str or list[str] or DataFrame object) Path to csv or tsv file (str) (e.g., 'variants.csv'), or DataFrame (DataFrame object),
                    containing information about the variants in the following format:

                    | var_column         | var_id_column | seq_id_column |
                    | c.2C>T             | var1          | seq1          | -> Apply variant 1 to sequence 1
                    | c.9_13inv          | var2          | seq2          | -> Apply variant 2 to sequence 2
                    | c.9_13inv          | var2          | seq3          | -> Apply variant 2 to sequence 3
                    | c.9_13delinsAAT    | var3          | seq3          | -> Apply variant 3 to sequence 3
                    | ...                | ...           | ...           |

                    'var_column' = Column containing the variants to be performed written in standard mutation/variant annotation (see below)
                    'seq_id_column' = Column containing the identifiers of the sequences to be mutated (must correspond to the string following
                    the > character in the 'sequences' fasta file; do NOT include spaces or dots)
                    'var_id_column' = Column containing an identifier for each variant (optional).

                    Alternatively: Input variant(s) as a string or list, e.g., 'c.2C>T' or ['c.2C>T', 'c.1A>C'].
                    If a list is provided, the number of variants must equal the number of input sequences.

                    For more information on the standard mutation/variant annotation, see https://www.ncbi.nlm.nih.gov/pmc/articles/PMC1867422/.

                    Alternatively, 'variants' can be a string specifying a supported database, which will automatically download
                    both the variant database and corresponding reference sequence (if the 'sequences' is not a path).
                    To see the supported combinations of variants and sequences, either
                    1) run `vk build --list_supported_databases` from the command line, or
                    2) run varseek.build(list_supported_databases=True) in python

    # Parameters affecting VCRS creation
    - w                                  (int) Length of sequence windows flanking the variant. Default: 30.
                                         If w > total length of the sequence, the entire sequence will be kept.
    - k                                  (int) Length of the k-mers to be considered in remove_seqs_with_wt_kmers, and the default minimum value for the minimum sequence length (which can be changed with 'min_seq_len').
                                         If using kallisto in a later workflow, then this should correspond to kallisto k.
                                         Must be greater than the value passed in for w. Default: 59.
    - max_ambiguous                      (int) Maximum number of 'N' (or 'n') characters allowed in a VCRS. None means no 'N' filter will be applied. Default: 0.

    # Additional input files and associated parameters
    - var_column                         (str) Name of the column containing the variants to be introduced in 'variants'. Default: 'mutation'.
    - seq_id_column                      (str) Name of the column containing the IDs of the sequences to be mutated in 'variants'. Default: 'seq_ID'.
    - var_id_column                      (str) Name of the column containing the IDs of each variant in 'variants'. Optional. Default: use <seq_id_column>_<var_column> for each row.
    - gtf                                (str) Path to .gtf file. Only used in conjunction with the arguments `transcript_boundaries` and `identify_all_spliced_from_genome`, as well as to add some information to the downloaded database when variants='cosmic_cmc'. If downloading sequence information, then setting gtf=True will automatically include it in the download. Default: None
    - gtf_transcript_id_column           (str) Column name in the input 'variants' file containing the transcript ID.
                                         In this case, column seq_id_column should contain the chromosome number.
                                         Required when 'gtf' is provided. Default: None
    - transcript_boundaries              (True/False) Whether to use the transcript boundaries in the input 'gtf' file to define the boundaries of the VCRSs. Only used when the `sequences` and `variants` information is in terms of the genome, and when `gtf` is specified. Default: False.
    - identify_all_spliced_from_genome   (True/False) Whether to identify all spliced VCRSs from the genome. Currently not implemented. Default: False.

    # Output paths and associated parameters
    - out                                (str) Path to default output directory to containing created files. Any individual output file path can be overriden if the specific file path is provided
                                         as an argument. Default: "." (current directory).
    - reference_out_dir                  (str) Path to reference file directory to be downloaded if 'variants' is a supported database and the file corresponding to 'sequences' does not exist.
                                         Default: <out>/reference.
    - vcrs_fasta_out                     (str) Path to output fasta file containing the VCRSs.
                                         If use_IDs=False, then the fasta headers will be the values in the column 'mut_ID' (semicolon-jooined if merge_identical=True).
                                         Otherwise, if use_IDs=True (default), then the fasta headers will be of the form 'vcrs_<int>' where <int> is a unique integer. Default: "<out>/vcrs.fa"
    - variants_updated_csv_out          (str) Path to output csv file containing the updated DataFrame. Only valid if save_variants_updated_csv=True. Default: "<out>/variants_updated.csv"
    - id_to_header_csv_out               (str) File name of csv file containing the mapping of unique IDs to the original sequence headers if use_IDs=True. Default: "<out>/id_to_header_mapping.csv"
    - vcrs_t2g_out                       (str) Path to output t2g file containing the transcript-to-gene mapping for the VCRSs. Used in kallisto | bustools workflow. Default: "<out>/vcrs_t2g.txt"
    - wt_vcrs_fasta_out                  (str) Path to output fasta file containing the wildtype sequence counterparts of the variant-containing reference sequences (VCRSs). Default: "<out>/wt_vcrs.fa"
    - wt_vcrs_t2g_out                    (str) Path to output t2g file containing the transcript-to-gene mapping for the wildtype VCRSs. Default: "<out>/wt_vcrs_t2g.txt"
    - removed_variants_text_out          (str) Path to output text file containing the removed variants. Default: "<out>/removed_variants.txt"
    - filtering_report_text_out          (str) Path to output text file containing the filtering report. Default: "<out>/filtering_report.txt"

    # Returning and saving of optional output
    - return_variant_output             (True/False) Whether to return the variant output saved in the fasta file. Default: False.
    - save_variants_updated_csv         (True/False) Whether to update the input 'variants' DataFrame to include additional columns with the variant type,
                                         wildtype nucleotide sequence, and mutant nucleotide sequence (only valid if 'variants' is a csv or tsv file). Default: False
    - save_wt_vcrs_fasta_and_t2g         (True/False) Whether to create a fasta file containing the wildtype sequence counterparts of the variant-containing reference sequences (VCRSs)
                                         and the corresponding t2g. Default: False.
    - save_removed_variants_text         (True/False) Whether to save a text file containing the removed variants. Default: True.
    - save_filtering_report_text         (True/False) Whether to save a text file containing the filtering report. Default: True.
    - store_full_sequences               (True/False) Whether to also include the complete wildtype and mutant sequences in the updated 'variants' DataFrame (not just the sub-sequence with
                                         w-length flanks). Only valid if save_variants_updated_csv=True. Default: False
    - translate                          (True/False) Add additional columns to the 'variants' DataFrame containing the wildtype and mutant amino acid sequences.
                                         Only valid if store_full_sequences=True. Default: False
    - translate_start                    (int | str | None) The position in the input nucleotide sequence to start translating. If a string is provided, it should correspond
                                         to a column name in 'variants' containing the open reading frame start positions for each sequence/variant.
                                         Only valid if translate=True. Default: None (translate from the beginning of the sequence)
    - translate_end                      (int | str | None) The position in the input nucleotide sequence to end translating. If a string is provided, it should correspond
                                         to a column name in 'variants' containing the open reading frame end positions for each sequence/variant.
                                         Only valid if translate=True. Default: None (translate from to the end of the sequence)

    # General arguments:
    - dry_run                            (True/False) Whether to simulate the function call without executing it. Default: False.
    - list_supported_databases           (True/False) Whether to print the supported databases and sequences. Default: False.
    - overwrite                          (True/False) Whether to overwrite existing output files. Will return if any output file already exists. Default: False.
    - logging_level                      (str) Logging level. Can also be set with the environment variable VARSEEK_LOGGING_LEVEL. Default: INFO.
    - save_logs                          (True/False) Whether to save logs to a file. Default: False.
    - log_out_dir                        (str) Directory to save logs. Default: `out`/logs
    - verbose                            (True/False) Whether to print additional information e.g., progress bars. Default: False.

    # # Hidden arguments (part of kwargs) - for niche use cases, specific databases, or debugging:
    # # niche use cases
    - insertion_size_limit               (int) Maximum number of nucleotides allowed in an insertion-type variant. Variants with insertions larger than this will be dropped.
                                         Default: None (no insertion size limit will be applied)
    - min_seq_len                        (int) Minimum length of the variant output sequence. Mutant sequences smaller than this will be dropped. None means no length filter will be applied. Default: k (from the "k" parameter)
    - optimize_flanking_regions          (True/False) Whether to remove nucleotides from either end of the mutant sequence to ensure (when possible)
                                         that the mutant sequence does not contain any (w+1)-mers (where a (w+1)-mer is a subsequence of length w+1, with w defined by the 'w' argument) also found in the wildtype/input sequence. Default: True
    - remove_seqs_with_wt_kmers          (True/False) Removes output sequences where at least one k-mer is also present in the wildtype/input sequence in the same region.
                                         If optimize_flanking_regions=True, only sequences for which a wildtype k-mer (with k defined by the 'k' argument) is still present after optimization will be removed.
                                         Default: True

    - required_insertion_overlap_length  (int | str | None) Enforces the minimum number of bases included in the inserted region for all (w+1)-mers (where a (w+1)-mer is a subsequence of length w+1, with w defined by the 'w' argument),
                                         or that all (w+1)-mers contain the entire inserted sequence (whatever is smaller). Only effective when optimize_flanking_regions is also True. None or 1 (minimum value) means that flank optimization occurs only until there is no shared k-mer in the VCRS and the reference sequence (i.e., as little as 1 base from the insertion could be required). If "all", then require the entire insertion and the following nucleotide (and filter out insertions of length >= 2*w). Experimental - does not work quite properly with values > 1 when there is overlap between the mutated regions and flanks. Default: None
    - merge_identical                    (True/False) Whether to merge sequence-identical VCRSs in the output (identical VCRSs will be merged by concatenating the sequence
                                         headers for all identical sequences with semicolons). Default: True
    - vcrs_strandedness                  (True/False) Whether to consider the forward and reverse-complement mutant sequences as distinct if merging identical sequences. Only effective when merge_identical is also True. Default: False (ie consider forward and reverse-complement sequences to be equivalent).
    - use_IDs                            (True/False) Whether to keep the original sequence headers in the output fasta file, or to replace them with unique IDs of the form 'vcrs_<int>.
                                         If False, then an additional file at the path <id_to_header_csv_out> will be formed that maps sequence IDs from the fasta file to the <var_id_column>. Default: True.
    - original_order                     (True/False) Whether to keep the original order of the sequences in the output fasta file. Default: True.

    # # specific databases
    - cosmic_version                     (str) COSMIC release version to download. Default: "100".
    - cosmic_grch                        (str) COSMIC genome reference version to download. Default: "37".
    - cosmic_email                       (str) Email address for COSMIC download. Default: None.
    - cosmic_password                    (str) Password for COSMIC download. Default: None.

    # other
    - logger                             (logging.Logger) Logger object. Default: None.


    Saves mutated sequences in fasta format (or returns a list containing the mutated sequences if out=None).
    """

    global intronic_mutations, posttranslational_region_mutations, unknown_mutations, uncertain_mutations, ambiguous_position_mutations, variants_incorrect_wt_base, mut_idx_outside_seq

    # * 0. Informational arguments that exit early
    if list_supported_databases:
        print_valid_values_for_variants_and_sequences_in_varseek_build()
        return None

    # * 1. Start timer
    start_time = time.perf_counter()

    # * 1.5. logger
    global logger
    if kwargs.get("logger") and isinstance(kwargs.get("logger"), logging.Logger):
        logger = kwargs.get("logger")
    else:
        if save_logs and not log_out_dir:
            log_out_dir = os.path.join(out, "logs")
        logger = set_up_logger(logger, logging_level=logging_level, save_logs=save_logs, log_dir=log_out_dir)

    # * 2. Type-checking
    params_dict = make_function_parameter_to_value_dict(1)
    validate_input_build(params_dict)

    # * 3. Dry-run
    if dry_run:
        print_varseek_dry_run(params_dict, function_name="build")
        return None

    # * 4. Save params to config file and run info file
    config_file = os.path.join(out, "config", "vk_build_config.json")
    save_params_to_config_file(params_dict, config_file)

    run_info_file = os.path.join(out, "config", "vk_build_run_info.txt")
    save_run_info(run_info_file)

    # * 5. Set up default folder/file input paths, and make sure the necessary ones exist
    # all input files for vk build are required in the varseek workflow, so this is skipped

    # * 6. Set up default folder/file output paths, and make sure they don't exist unless overwrite=True
    if not reference_out_dir:
        reference_out_dir = os.path.join(out, "reference")

    os.makedirs(out, exist_ok=True)
    os.makedirs(reference_out_dir, exist_ok=True)
    
    # if someone specifies an output path, then it should be saved - could technically incorporate this logic in else statements below, but this feels cleaner
    if variants_updated_csv_out:
        save_variants_updated_csv = True
    if wt_vcrs_fasta_out or wt_vcrs_t2g_out:
        save_wt_vcrs_fasta_and_t2g = True
    if removed_variants_text_out:
        save_removed_variants_text = True
    if filtering_report_text_out:
        save_filtering_report_text = True

    if not vcrs_fasta_out:
        vcrs_fasta_out = os.path.join(out, "vcrs.fa")
    if not variants_updated_csv_out:
        variants_updated_csv_out = os.path.join(out, "variants_updated.csv")
    if not id_to_header_csv_out:
        id_to_header_csv_out = os.path.join(out, "id_to_header_mapping.csv")
    if not vcrs_t2g_out:
        vcrs_t2g_out = os.path.join(out, "vcrs_t2g.txt")
    if not wt_vcrs_fasta_out:
        wt_vcrs_fasta_out = os.path.join(out, "wt_vcrs.fa")
    if not wt_vcrs_t2g_out:
        wt_vcrs_t2g_out = os.path.join(out, "wt_vcrs_t2g.txt")
    if not removed_variants_text_out:
        removed_variants_text_out = os.path.join(out, "removed_variants.txt")
    if not filtering_report_text_out:
        filtering_report_text_out = os.path.join(out, "filtering_report.txt")

    # make sure directories of all output files exist
    output_files = [vcrs_fasta_out, variants_updated_csv_out, id_to_header_csv_out, vcrs_t2g_out, wt_vcrs_fasta_out, wt_vcrs_t2g_out, removed_variants_text_out, filtering_report_text_out]
    for output_file in output_files:
        if os.path.isfile(output_file) and not overwrite:
            raise ValueError(f"Output file '{output_file}' already exists. Set 'overwrite=True' to overwrite it.")
        if os.path.dirname(output_file):
            os.makedirs(os.path.dirname(output_file), exist_ok=True)

    # * 7. Define kwargs defaults
    cosmic_version = kwargs.get("cosmic_version", None)
    cosmic_grch = kwargs.get("cosmic_grch", None)
    cosmic_email = kwargs.get("cosmic_email", None)
    cosmic_password = kwargs.get("cosmic_password", None)
    insertion_size_limit = kwargs.get("insertion_size_limit", None)
    min_seq_len = kwargs.get("min_seq_len", k)
    optimize_flanking_regions = kwargs.get("optimize_flanking_regions", True)
    remove_seqs_with_wt_kmers = kwargs.get("remove_seqs_with_wt_kmers", True)
    required_insertion_overlap_length = kwargs.get("required_insertion_overlap_length", None)  # set to None for original vk build
    merge_identical = kwargs.get("merge_identical", True)
    vcrs_strandedness = kwargs.get("vcrs_strandedness", False)
    use_IDs = kwargs.get("use_IDs", True)
    original_order = kwargs.get("original_order", True)

    mutations = variants
    del variants

    # * 8. Start the actual function
    if not k:
        k = w + 1

    merge_identical_rc = not vcrs_strandedness

    columns_to_keep = [
        "header",
        seq_id_column,
        var_column,
        "mutation_type",
        "wt_sequence",
        "variant_sequence",
        "nucleotide_positions",
        "start_variant_position",
        "end_variant_position",
        "actual_variant",
    ]

    if isinstance(mutations, str):
        if mutations in supported_databases_and_corresponding_reference_sequence_type and "cosmic" in mutations:
            if not cosmic_version:
                cosmic_version = "100"
            if not cosmic_grch:
                grch_dict = supported_databases_and_corresponding_reference_sequence_type[mutations]["database_version_to_reference_assembly_build"]
                largest_key = max(int(k) for k in grch_dict.keys())
                grch = grch_dict[str(largest_key)]
            else:
                grch = cosmic_grch
                if grch not in supported_databases_and_corresponding_reference_sequence_type[mutations]["database_version_to_reference_assembly_build"]:
                    raise ValueError(f"Invalid value for cosmic_grch: {cosmic_grch}. Supported values are {', '.join(supported_databases_and_corresponding_reference_sequence_type[mutations]['database_version_to_reference_assembly_build'].keys())}.")
            if grch == "37":
                gget_ref_grch = "human_grch37"
            elif grch == "38":
                gget_ref_grch = "human"
            else:
                gget_ref_grch = grch

    # Load input sequences and their identifiers from fasta file
    if isinstance(sequences, str) and ("." in sequences or (mutations in supported_databases_and_corresponding_reference_sequence_type and sequences in supported_databases_and_corresponding_reference_sequence_type[mutations]["sequence_download_commands"])):
        if mutations in supported_databases_and_corresponding_reference_sequence_type and sequences in supported_databases_and_corresponding_reference_sequence_type[mutations]["sequence_download_commands"]:
            # TODO: expand beyond COSMIC
            if "cosmic" in mutations:
                ensembl_version = supported_databases_and_corresponding_reference_sequence_type[mutations]["database_version_to_reference_release"][cosmic_version]
                reference_out_sequences = f"{reference_out_dir}/ensembl_grch{grch}_release{ensembl_version}"  # matches vk info

                sequences_download_command = supported_databases_and_corresponding_reference_sequence_type[mutations]["sequence_download_commands"][sequences]
                sequences_download_command = sequences_download_command.replace("OUT_DIR", reference_out_sequences)
                sequences_download_command = sequences_download_command.replace("ENSEMBL_VERSION", ensembl_version)
                sequences_download_command = sequences_download_command.replace("GRCH_NUMBER", gget_ref_grch)
                
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

        titles, seqs = [], []
        for title, seq in pyfastx.Fastx(sequences):
            titles.append(title)
            seqs.append(seq)
        # titles, seqs = read_fasta(sequences)  # when using gget.utils.read_fasta()

    # Handle input sequences passed as a list
    elif isinstance(sequences, list):
        titles = [f"seq{i+1}" for i in range(len(sequences))]
        seqs = sequences

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

    if isinstance(mutations, str) and os.path.isfile(mutations):
        mutations_path = mutations  # will account for mutations in supported_databases_and_corresponding_reference_sequence_type once the file is defined in the conditional
    else:
        mutations_path = None

    if isinstance(mutations, str) and mutations in supported_databases_and_corresponding_reference_sequence_type:
        # TODO: expand beyond COSMIC (utilize the variant_file_name key in supported_databases_and_corresponding_reference_sequence_type)
        if "cosmic" in mutations:
            mutations_original = mutations
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
                    mutation_class="cancer",
                    download_cosmic=True,
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
                    mutations, bad_cosmic_mutations_dict = convert_mutation_cds_locations_to_cdna(
                        input_csv_path=mutations,
                        output_csv_path=mutations_path,
                        cds_fasta_path=cds_file,
                        cdna_fasta_path=cdna_file,
                        logger=logger,
                        verbose=verbose
                    )

                    # uncertain_mutations += bad_cosmic_mutations_dict["uncertain_mutations"]
                    # ambiguous_position_mutations += bad_cosmic_mutations_dict["ambiguous_position_mutations"]
                    # intronic_mutations += bad_cosmic_mutations_dict["intronic_mutations"]
                    # posttranslational_region_mutations += bad_cosmic_mutations_dict["posttranslational_region_mutations"]
                    
                seq_id_column = "seq_ID"
                var_column = "mutation_cdna"

            elif sequences == "genome" or sequences.endswith(supported_databases_and_corresponding_reference_sequence_type[mutations_original]["sequence_file_names"]["genome"].replace("GRCH_NUMBER", grch)):  # covers whether sequences == "genome" or sequences == "PATH/TO/Homo_sapiens.GRCh37.dna.primary_assembly.fa"
                mutations_path_no_duplications = mutations_path.replace(".csv", "_no_duplications.csv")
                if not os.path.isfile(mutations_path_no_duplications):
                    logger.info("COSMIC genome location is not accurate for duplications. Dropping duplications in a copy of the csv file.")
                    mutations = drop_duplication_mutations(mutations, mutations_path_no_duplications)  # COSMIC incorrectly records genome positions of duplications
                else:
                    mutations = pd.read_csv(mutations_path_no_duplications)

                seq_id_column = "chromosome"
                var_column = "mutation_genome"

            elif sequences == "cds" or sequences.endswith(supported_databases_and_corresponding_reference_sequence_type[mutations_original]["sequence_file_names"]["cds"].replace("GRCH_NUMBER", grch)):  # covers whether sequences == "cds" or sequences == "PATH/TO/Homo_sapiens.GRCh37.cds.all.fa"
                seq_id_column = "seq_ID"
                var_column = "mutation"  # if "mutation_cds" not in mutations.columns else "mutation_cds"  # checks if CDS mutation column was renamed by convert_mutation_cds_locations_to_cdna

            var_id_column = "mutation_id" if var_id_column is not None else None  # use the id column if the user wanted to; otherwise keep as default

    # Read in 'mutations' if passed as filepath to comma-separated csv
    if isinstance(mutations, str) and mutations.endswith(".csv"):
        mutations = pd.read_csv(mutations)
        for col in mutations.columns:
            if col not in columns_to_keep:
                columns_to_keep.append(col)  # append "mutation_aa", "gene_name", "mutation_id"

    elif isinstance(mutations, str) and mutations.endswith(".tsv"):
        mutations = pd.read_csv(mutations, sep="\t")
        for col in mutations.columns:
            if col not in columns_to_keep:
                columns_to_keep.append(col)  # append "mutation_aa", "gene_name", "mutation_id"

    elif isinstance(mutations, str) and (mutations.endswith(".vcf") or mutations.endswith(".vcf.gz")):
        mutations = vcf_to_dataframe(mutations, additional_columns=save_variants_updated_csv, explode_alt=True, filter_empty_alt=True)  # only load in additional columns if I plan to later save this updated csv
        mutations.rename(columns={"CHROM": seq_id_column, "ID": var_id_column}, inplace=True)
        mutations[var_column] = mutations.apply(generate_mutation_notation_from_vcf_columns, axis=1)  #!! untested

    # Handle mutations passed as a list
    elif isinstance(mutations, list):
        if len(mutations) > 1:
            if len(mutations) != len(seqs):
                raise ValueError("If a list is passed, the number of mutations must equal the number of input sequences.")

            temp = pd.DataFrame()
            temp[var_column] = mutations
            temp[var_id_column] = [f"var{i+1}" for i in range(len(mutations))]
            temp[seq_id_column] = [f"seq{i+1}" for i in range(len(mutations))]
            mutations = temp
        else:
            temp = pd.DataFrame()
            temp[var_column] = [mutations[0]] * len(seqs)
            temp[var_id_column] = [f"var{i+1}" for i in range(len(seqs))]
            temp[seq_id_column] = [f"seq{i+1}" for i in range(len(seqs))]
            mutations = temp

    # Handle single mutation passed as a string
    elif isinstance(mutations, str) and mutations not in supported_databases_and_corresponding_reference_sequence_type:
        # This will work for one mutation for one sequence as well as one mutation for multiple sequences
        temp = pd.DataFrame()
        mutations = [mutations]
        temp[var_column] = [mutations[0]] * len(seqs)
        temp[var_id_column] = [f"var{i+1}" for i in range(len(seqs))]
        temp[seq_id_column] = [f"seq{i+1}" for i in range(len(seqs))]
        mutations = temp

    elif isinstance(mutations, pd.DataFrame):
        for col in mutations.columns:
            if col not in columns_to_keep:
                columns_to_keep.append(col)  # append "mutation_aa", "gene_name", "mutation_id"

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

    del titles
    del seqs

    if non_nuc_seqs > 0:
        logger.warning("Non-nucleotide characters detected in %s input sequences. vk build is only optimized for mutating nucleotide sequences.", non_nuc_seqs)

    if original_order:
        mutations['original_order'] = range(len(mutations))  # ensure that original order can be restored at the end
        columns_to_keep.append("original_order")  # just so it doesn't get removed automatically (but I remove it manually later)

    total_mutations = mutations.shape[0]

    # Drop inputs for sequences or variants that were not found
    mutations = mutations.dropna(subset=[seq_id_column, var_column])
    missing_seq_id_or_var = total_mutations - mutations.shape[0]
    total_mutations_updated = mutations.shape[0]
    if len(mutations) < 1:
        raise ValueError(
            """
            None of the input sequences match the sequence IDs provided in 'mutations'.
            Ensure that the sequence IDs correspond to the string following the > character in the 'sequences' fasta file (do NOT include spaces or dots).
            """
        )

    # remove duplicate entries from the mutations dataframe, keeping the ones with the most information
    mutations["non_na_count"] = mutations.notna().sum(axis=1)
    mutations = mutations.sort_values(by="non_na_count", ascending=False)
    mutations = mutations.drop_duplicates(subset=[seq_id_column, var_column], keep="first")
    mutations = mutations.drop(columns=["non_na_count"])

    duplicate_count = total_mutations_updated - mutations.shape[0]
    total_mutations_updated = mutations.shape[0]

    # ensure seq_ID column is string type, and chromosome numbers don't have decimals
    mutations[seq_id_column] = mutations[seq_id_column].apply(convert_chromosome_value_to_int_when_possible)

    mutations = add_mutation_type(mutations, var_column)

    # Link sequences to their mutations using the sequence identifiers
    if store_full_sequences:
        mutations["wt_sequence_full"] = mutations[seq_id_column].map(seq_dict)

    # Handle sequences that were not found based on their sequence IDs
    seqs_not_found_count = len(mutations[~mutations[seq_id_column].isin(seq_dict.keys())])
    if seqs_not_found_count > 0:
        logger.warning(
            """
            The sequences corresponding to %d sequence IDs were not found.
            These sequences and their corresponding mutations will not be included in the output.
            Ensure that the sequence IDs correspond to the string following the > character in the 'sequences' FASTA file (do NOT include spaces or dots).
            """,
            seqs_not_found_count,
        )

        mutations = mutations[mutations[seq_id_column].isin(seq_dict.keys())]

    mutations["variant_sequence"] = ""

    if var_id_column is not None:
        mutations["header"] = mutations[var_id_column]
        logger.info("Using var_id_column '%s' as the variant header column.", var_id_column)
    else:
        mutations["header"] = mutations[seq_id_column] + ":" + mutations[var_column]
        logger.info("Using the seq_id_column:var_column '%s' columns as the variant header column.", f"{seq_id_column}:{var_column}")

    # make a set of all initial mutation IDs
    initial_mutation_id_set = set(mutations["header"].dropna())

    # Calculate number of bad mutations
    uncertain_mutations = mutations[var_column].str.contains(r"\?").sum()  # I originally tried doing a += thing here to account for the cDNA to CDS thing, but then it is hard to track with the double counting and becomes not worth it

    ambiguous_position_mutations = mutations[var_column].str.contains(r"\(|\)").sum()

    intronic_mutations = mutations[var_column].str.contains(r"\+|\-").sum()

    posttranslational_region_mutations = mutations[var_column].str.contains(r"\*").sum()

    # Filter out bad mutations
    combined_pattern = re.compile(r"(\?|\(|\)|\+|\-|\*)")
    mask = mutations[var_column].str.contains(combined_pattern)
    mutations = mutations[~mask]

    # Extract nucleotide positions and mutation info from Mutation CDS
    mutations[["nucleotide_positions", "actual_variant"]] = mutations[var_column].str.extract(mutation_pattern)

    # Filter out mutations that did not match the re
    unknown_mutations = mutations["nucleotide_positions"].isna().sum()
    mutations = mutations.dropna(subset=["nucleotide_positions", "actual_variant"])

    if mutations.empty:
        logger.warning("No valid variants found in the input.")
        return [] if return_variant_output else None

    # Split nucleotide positions into start and end positions
    split_positions = mutations["nucleotide_positions"].str.split("_", expand=True)

    mutations["start_variant_position"] = split_positions[0]
    if split_positions.shape[1] > 1:
        mutations["end_variant_position"] = split_positions[1].fillna(split_positions[0])
    else:
        mutations["end_variant_position"] = mutations["start_variant_position"]

    mutations.loc[mutations["end_variant_position"].isna(), "end_variant_position"] = mutations["start_variant_position"]

    mutations[["start_variant_position", "end_variant_position"]] = mutations[["start_variant_position", "end_variant_position"]].astype(int)

    # Adjust positions to 0-based indexing
    mutations["start_variant_position"] -= 1
    mutations["end_variant_position"] -= 1  # don't forget to increment by 1 later

    # Calculate sequence length
    mutations["sequence_length"] = mutations[seq_id_column].apply(lambda x: get_sequence_length(x, seq_dict))

    # Filter out mutations with positions outside the sequence
    index_error_mask = (mutations["start_variant_position"] > mutations["sequence_length"]) | (mutations["end_variant_position"] > mutations["sequence_length"])

    mut_idx_outside_seq = index_error_mask.sum()

    mutations = mutations[~index_error_mask]

    if mutations.empty:
        logger.warning("No valid variants found in the input.")
        return [] if return_variant_output else None

    # Create masks for each type of mutation
    mutations["wt_nucleotides_ensembl"] = None
    substitution_mask = mutations["mutation_type"] == "substitution"
    deletion_mask = mutations["mutation_type"] == "deletion"
    delins_mask = mutations["mutation_type"] == "delins"
    insertion_mask = mutations["mutation_type"] == "insertion"
    duplication_mask = mutations["mutation_type"] == "duplication"
    inversion_mask = mutations["mutation_type"] == "inversion"

    if remove_seqs_with_wt_kmers:
        long_duplications = ((duplication_mask) & ((mutations["end_variant_position"] - mutations["start_variant_position"]) >= k)).sum()
        logger.info("Removing %d duplications > k", long_duplications)
        mutations = mutations[~((duplication_mask) & ((mutations["end_variant_position"] - mutations["start_variant_position"]) >= k))]
    else:
        long_duplications = 0

    # Create a mask for all non-substitution mutations
    non_substitution_mask = deletion_mask | delins_mask | insertion_mask | duplication_mask | inversion_mask
    insertion_and_delins_and_dup_and_inversion_mask = insertion_mask | delins_mask | duplication_mask | inversion_mask

    # Extract the WT nucleotides for the substitution rows from reference fasta (i.e., Ensembl)
    start_positions = mutations.loc[substitution_mask, "start_variant_position"].values

    # Get the nucleotides at the start positions
    wt_nucleotides_substitution = np.array([get_nucleotide_at_position(seq_id, pos, seq_dict) for seq_id, pos in zip(mutations.loc[substitution_mask, seq_id_column], start_positions)])

    mutations.loc[substitution_mask, "wt_nucleotides_ensembl"] = wt_nucleotides_substitution

    # Extract the WT nucleotides for the substitution rows from the Mutation CDS (i.e., COSMIC)
    mutations["wt_nucleotides_cosmic"] = None
    mutations.loc[substitution_mask, "wt_nucleotides_cosmic"] = mutations["actual_variant"].str[0]

    congruent_wt_bases_mask = (mutations["wt_nucleotides_cosmic"] == mutations["wt_nucleotides_ensembl"]) | mutations[["wt_nucleotides_cosmic", "wt_nucleotides_ensembl"]].isna().any(axis=1)

    variants_incorrect_wt_base = (~congruent_wt_bases_mask).sum()

    mutations = mutations[congruent_wt_bases_mask]

    if mutations.empty:
        logger.warning("No valid variants found in the input.")
        return [] if return_variant_output else None

    # Adjust the start and end positions for insertions
    mutations.loc[insertion_mask, "start_variant_position"] += 1  # in other cases, we want left flank to exclude the start of mutation site; but with insertion, the start of mutation site as it is denoted still belongs in the flank region
    mutations.loc[insertion_mask, "end_variant_position"] -= 1  # in this notation, the end position is one before the start position

    # Extract the WT nucleotides for the non-substitution rows from the Mutation CDS (i.e., COSMIC)
    mutations.loc[non_substitution_mask, "wt_nucleotides_ensembl"] = mutations.loc[non_substitution_mask].apply(lambda row: extract_sequence(row, seq_dict, seq_id_column), axis=1)

    # Apply mutations to the sequences
    mutations["mut_nucleotides"] = None
    mutations.loc[substitution_mask, "mut_nucleotides"] = mutations.loc[substitution_mask, "actual_variant"].str[-1]
    mutations.loc[deletion_mask, "mut_nucleotides"] = ""
    mutations.loc[delins_mask, "mut_nucleotides"] = mutations.loc[delins_mask, "actual_variant"].str.extract(r"delins([A-Z]+)")[0]
    mutations.loc[insertion_mask, "mut_nucleotides"] = mutations.loc[insertion_mask, "actual_variant"].str.extract(r"ins([A-Z]+)")[0]
    mutations.loc[duplication_mask, "mut_nucleotides"] = mutations.loc[duplication_mask].apply(lambda row: row["wt_nucleotides_ensembl"], axis=1)
    if inversion_mask.any():
        mutations.loc[inversion_mask, "mut_nucleotides"] = mutations.loc[inversion_mask].apply(
            lambda row: "".join(complement.get(nucleotide, "N") for nucleotide in row["wt_nucleotides_ensembl"][::-1]),
            axis=1,
        )

    # Adjust the nucleotide positions of duplication mutations to mimic that of insertions (since duplications are essentially just insertions)
    mutations.loc[duplication_mask, "start_variant_position"] = mutations.loc[duplication_mask, "end_variant_position"] + 1  # in the case of duplication, the "mutant" site is still in the left flank as well

    mutations.loc[duplication_mask, "wt_nucleotides_ensembl"] = ""

    # Calculate the kmer bounds
    mutations["start_kmer_position_min"] = mutations["start_variant_position"] - w
    mutations["start_kmer_position"] = mutations["start_kmer_position_min"].combine(0, max)

    mutations["end_kmer_position_max"] = mutations["end_variant_position"] + w
    mutations["end_kmer_position"] = mutations[["end_kmer_position_max", "sequence_length"]].min(axis=1)  # don't forget to increment by 1 later on

    if gtf is not None and transcript_boundaries:
        if "start_transcript_position" not in mutations.columns and "end_transcript_position" not in mutations.columns:  # * currently hard-coded column names, but optionally can be changed to arguments later
            mutations = merge_gtf_transcript_locations_into_cosmic_csv(mutations, gtf, gtf_transcript_id_column=gtf_transcript_id_column, output_mutations_path=mutations_path)

            columns_to_keep.extend(["start_transcript_position", "end_transcript_position", "strand"])
        else:
            logger.warning("Transcript positions already present in the input variants file. Skipping GTF file merging.")

        # adjust start_transcript_position to be 0-index
        mutations["start_transcript_position"] -= 1

        mutations["start_kmer_position"] = mutations[["start_kmer_position", "start_transcript_position"]].max(axis=1)
        mutations["end_kmer_position"] = mutations[["end_kmer_position", "end_transcript_position"]].min(axis=1)

    mut_apply = (lambda *args, **kwargs: mutations.progress_apply(*args, **kwargs)) if verbose else mutations.apply

    if save_variants_updated_csv and store_full_sequences:
        # Extract flank sequences
        if verbose:
            tqdm.pandas(desc="Extracting full left flank sequences")

        mutations["left_flank_region_full"] = mut_apply(
            lambda row: seq_dict[row[seq_id_column]][0 : row["start_variant_position"]],
            axis=1,
        )  # ? vectorize

        if verbose:
            tqdm.pandas(desc="Extracting full right flank sequences")

        mutations["right_flank_region_full"] = mut_apply(
            lambda row: seq_dict[row[seq_id_column]][row["end_variant_position"] + 1 : row["sequence_length"]],
            axis=1,
        )  # ? vectorize

    if verbose:
        tqdm.pandas(desc="Extracting VCRS left flank sequences")

    mutations["left_flank_region"] = mut_apply(
        lambda row: seq_dict[row[seq_id_column]][row["start_kmer_position"] : row["start_variant_position"]],
        axis=1,
    )  # ? vectorize

    if verbose:
        tqdm.pandas(desc="Extracting VCRS right flank sequences")

    mutations["right_flank_region"] = mut_apply(
        lambda row: seq_dict[row[seq_id_column]][row["end_variant_position"] + 1 : row["end_kmer_position"] + 1],
        axis=1,
    )  # ? vectorize

    mutations["inserted_nucleotide_length"] = None

    if insertion_and_delins_and_dup_and_inversion_mask.any():
        mutations.loc[insertion_and_delins_and_dup_and_inversion_mask, "inserted_nucleotide_length"] = mutations.loc[insertion_and_delins_and_dup_and_inversion_mask, "mut_nucleotides"].str.len()

        if insertion_size_limit is not None:
            mutations = mutations[(mutations["inserted_nucleotide_length"].isna()) | (mutations["inserted_nucleotide_length"] <= insertion_size_limit)]  # Keep rows where it is None/NaN  # Keep rows where it's <= insertion_size_limit

    mutations["beginning_mutation_overlap_with_right_flank"] = 0
    mutations["end_mutation_overlap_with_left_flank"] = 0

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
        # TODO: for duplications, required_insertion_overlap_length=None works fine; but required_insertion_overlap_length="all" causes issues (ruins symmetry)
        if required_insertion_overlap_length and required_insertion_overlap_length != 1 and insertion_and_delins_and_dup_and_inversion_mask.any():  # * new as of 11/20/24
            if required_insertion_overlap_length == "all":
                required_insertion_overlap_length = np.inf

            if required_insertion_overlap_length >= 2 * w:
                mutations = mutations[(mutations["inserted_nucleotide_length"].isna()) | (mutations["inserted_nucleotide_length"] < 2 * w)]  # Keep rows where it is None/NaN  # Keep rows where it's < 2*w

            mutations.loc[insertion_and_delins_and_dup_and_inversion_mask, "beginning_mutation_overlap_with_right_flank"] = np.maximum(
                mutations.loc[insertion_and_delins_and_dup_and_inversion_mask, "beginning_mutation_overlap_with_right_flank"],
                np.minimum(mutations.loc[insertion_and_delins_and_dup_and_inversion_mask, "inserted_nucleotide_length"], required_insertion_overlap_length-1),  # Feb 2025: the -1 was added empirically
            )

            mutations.loc[insertion_and_delins_and_dup_and_inversion_mask, "end_mutation_overlap_with_left_flank"] = np.maximum(
                mutations.loc[insertion_and_delins_and_dup_and_inversion_mask, "end_mutation_overlap_with_left_flank"],
                np.minimum(mutations.loc[insertion_and_delins_and_dup_and_inversion_mask, "inserted_nucleotide_length"], required_insertion_overlap_length-1),
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
        mutations.loc[substitution_mask, "variant_sequence"] = mutations.loc[substitution_mask, "left_flank_region"] + mutations.loc[substitution_mask, "mut_nucleotides"] + mutations.loc[substitution_mask, "right_flank_region"]

    # Create mutant non-substitution w-mer sequences
    if non_substitution_mask.any():
        mutations.loc[non_substitution_mask, "variant_sequence"] = mutations.loc[non_substitution_mask].apply(
            lambda row: row["left_flank_region"][row["updated_left_flank_start"] :] + row["mut_nucleotides"] + row["right_flank_region"][: len(row["right_flank_region"]) - row["updated_right_flank_end"]],
            axis=1,
        )

    if remove_seqs_with_wt_kmers:
        if verbose:
            tqdm.pandas(desc="Removing mutant fragments that share a kmer with wt fragments")

        mutations["wt_fragment_and_mutant_fragment_share_kmer"] = mut_apply(
            lambda row: wt_fragment_and_mutant_fragment_share_kmer(
                mutated_fragment=row["variant_sequence"],
                wildtype_fragment=row["wt_sequence"],
                k=k,
            ),
            axis=1,
        )

        mutations_overlapping_with_wt = mutations["wt_fragment_and_mutant_fragment_share_kmer"].sum()

        mutations = mutations[~mutations["wt_fragment_and_mutant_fragment_share_kmer"]]
    else:
        mutations_overlapping_with_wt = 0

    if save_variants_updated_csv and store_full_sequences:
        columns_to_keep.extend(["wt_sequence_full", "variant_sequence_full"])

        # Create full sequences (substitution and non-substitution)
        mutations["variant_sequence_full"] = mutations["left_flank_region_full"] + mutations["mut_nucleotides"] + mutations["right_flank_region_full"]

    if min_seq_len:
        # Calculate k-mer lengths (where k=w) and report the distribution
        mutations["variant_sequence_kmer_length"] = mutations["variant_sequence"].apply(lambda x: len(x) if pd.notna(x) else 0)

        rows_less_than_minimum = (mutations["variant_sequence_kmer_length"] < min_seq_len).sum()

        mutations = mutations[mutations["variant_sequence_kmer_length"] >= min_seq_len]

        logger.info("Removed %d variant kmers with length less than %d...", rows_less_than_minimum, min_seq_len)
    else:
        rows_less_than_minimum = 0

    if max_ambiguous is not None:
        # Get number of 'N' or 'n' occuring in the sequence
        mutations["num_N"] = mutations["variant_sequence"].str.lower().str.count("n")
        num_rows_with_N = (mutations["num_N"] > max_ambiguous).sum()
        mutations = mutations[mutations["num_N"] <= max_ambiguous]

        logger.info("Removed %d variant kmers containing more than %d 'N's...", num_rows_with_N, max_ambiguous)
    else:
        num_rows_with_N = 0

        # Drop the 'num_N' column after filtering
        mutations = mutations.drop(columns=["num_N"])

    # Report status of mutations back to user
    good_mutations = mutations.shape[0]
    total_removed_mutations = total_mutations - good_mutations

    report = f"""
        {good_mutations} variants correctly recorded ({good_mutations/total_mutations*100:.2f}%)
        {total_removed_mutations} variants removed ({total_removed_mutations/total_mutations*100:.2f}%)
          {missing_seq_id_or_var} variants missing seq_id or var_column ({missing_seq_id_or_var/total_mutations*100:.3f}%)
          {duplicate_count} entries removed due to having a duplicate entry ({duplicate_count/total_mutations*100:.3f}%)
          {seqs_not_found_count} variants with seq_ID not found in sequences ({seqs_not_found_count/total_mutations*100:.3f}%)
          {intronic_mutations} intronic variants found ({intronic_mutations/total_mutations*100:.3f}%)
          {posttranslational_region_mutations} posttranslational region variants found ({posttranslational_region_mutations/total_mutations*100:.3f}%)
          {unknown_mutations} unknown variants found ({unknown_mutations/total_mutations*100:.3f}%)
          {uncertain_mutations} variants with uncertain mutation found ({uncertain_mutations/total_mutations*100:.3f}%)
          {ambiguous_position_mutations} variants with ambiguous position found ({ambiguous_position_mutations/total_mutations*100:.3f}%)
          {variants_incorrect_wt_base} variants with incorrect wildtype base found ({variants_incorrect_wt_base/total_mutations*100:.3f}%)
          {mut_idx_outside_seq} variants with indices outside of the sequence length found ({mut_idx_outside_seq/total_mutations*100:.3f}%)
        """

    if remove_seqs_with_wt_kmers:
        report += f"""  {long_duplications} duplications longer than k found ({long_duplications/total_mutations*100:.3f}%)
          {mutations_overlapping_with_wt} variants with overlapping kmers found ({mutations_overlapping_with_wt/total_mutations*100:.3f}%)
        """

    if min_seq_len:
        report += f"""  {rows_less_than_minimum} variants with fragment length < min_seq_len found ({rows_less_than_minimum/total_mutations*100:.3f}%)
        """

    if max_ambiguous is not None:
        report += f"""  {num_rows_with_N} variants with Ns found ({num_rows_with_N/total_mutations*100:.3f}%)
        """

    if good_mutations != total_mutations:
        logger.warning(report)
    else:
        logger.info("All variants correctly recorded")

    # Save the report string to the specified path
    if save_filtering_report_text:
        with open(filtering_report_text_out, "w", encoding="utf-8") as file:
            file.write(report)

    if translate and save_variants_updated_csv and store_full_sequences:
        columns_to_keep.extend(["wt_sequence_aa_full", "variant_sequence_aa_full"])

        if not mutations_path:
            if not isinstance(translate_start, int) and not isinstance(translate_end, int):
                raise ValueError("translate_start and translate_end must be integers when translating sequences (or default None).")
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

                mutations["variant_sequence_aa_full"] = mutations["variant_sequence_full"].progress_apply(lambda x: translate_sequence(x, start=translate_start, end=translate_end))

            else:
                mutations["variant_sequence_aa_full"] = mutations["variant_sequence_full"].apply(lambda x: translate_sequence(x, start=translate_start, end=translate_end))

            logger.info(f"Translated variant sequences: {mutations['wt_sequence_aa_full']}")
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

            mutations["variant_sequence_aa_full"] = mut_apply(
                lambda row: translate_sequence(
                    row["variant_sequence_full"],
                    row[translate_start],
                    row[translate_end],
                ),
                axis=1,
            )

    mutations = mutations[columns_to_keep]

    # save text files of mutations filtered out
    final_mutation_id_set = set(mutations["header"].dropna())

    removed_mutation_set = initial_mutation_id_set - final_mutation_id_set

    # Save as a newline-separated text file
    if save_removed_variants_text:
        with open(removed_variants_text_out, "w", encoding="utf-8") as file:
            for mutation in removed_mutation_set:
                file.write(f"{mutation}\n")

    if save_variants_updated_csv:
        # recalculate start_variant_position and end_variant_position due to messing with it above
        mutations.drop(
            columns=["start_variant_position", "end_variant_position"],
            inplace=True,
            errors="ignore",
        )
        mutations["start_variant_position"] = split_positions[0]
        if split_positions.shape[1] > 1:
            mutations["end_variant_position"] = split_positions[1].fillna(split_positions[0])
        else:
            mutations["end_variant_position"] = mutations["start_variant_position"]

        mutations[["start_variant_position", "end_variant_position"]] = mutations[["start_variant_position", "end_variant_position"]].astype(int)

    if merge_identical:
        logger.info("Merging rows of identical VCRSs")

        mutations = mutations.sort_values(by='header', ascending=True)  # so that the headers are merged in alphabetical order

        # total mutations
        number_of_mutations_total = len(mutations)

        if merge_identical_rc:
            mutations["variant_sequence_rc"] = mutations["variant_sequence"].apply(reverse_complement)

            # Create a column that stores a sorted tuple of (variant_sequence, variant_sequence_rc)
            mutations["variant_sequence_and_rc_tuple"] = mutations.apply(
                lambda row: tuple(sorted([row["variant_sequence"], row["variant_sequence_rc"]])),
                axis=1,
            )

            # mutations = mutations.drop(columns=['variant_sequence_rc'])

            group_key = "variant_sequence_and_rc_tuple"
            columns_not_to_semicolon_join = [
                "variant_sequence",
                "variant_sequence_rc",
                "variant_sequence_and_rc_tuple",
            ]
            agg_columns = mutations.columns

        else:
            group_key = "variant_sequence"
            columns_not_to_semicolon_join = []
            agg_columns = [col for col in mutations.columns if col != "variant_sequence"]

        if save_variants_updated_csv:
            logger.warning("Merging rows of identical VCRSs can take a while if save_variants_updated_csv=True since it will concatenate all VCRSs too)")
            mutations = mutations.groupby(group_key, sort=False).agg({col: ("first" if col in columns_not_to_semicolon_join else (";".join if col == "header" else lambda x: list(x.fillna(np.nan)))) for col in agg_columns}).reset_index(drop=merge_identical_rc)  # lambda x: list(x) will make simple list, but lengths will be inconsistent with NaN values  # concatenate values with semicolons: lambda x: `";".join(x.astype(str))`   # drop if merging by variant_sequence_and_rc_tuple, but not if merging by variant_sequence
            if original_order:
                mutations['original_order'] = mutations['original_order'].apply(min)  # get the minimum original order for each group
        else:
            if original_order:
                mutations_temp = mutations.groupby(group_key, sort=False, group_keys=False).agg({
                    "header": ";".join,
                    "original_order": lambda x: min(x)  # Take the minimum order value
                }).reset_index()
            else:
                mutations_temp = mutations.groupby(group_key, sort=False, group_keys=False)["header"].apply(";".join).reset_index()  # ignores original_order
            
            if merge_identical_rc:
                mutations_temp = mutations_temp.merge(mutations[["variant_sequence", group_key]], on=group_key, how="left")
                mutations_temp = mutations_temp.drop_duplicates(subset="header")
                mutations_temp.drop(columns=[group_key], inplace=True)

            mutations = mutations_temp

        if "variant_sequence_and_rc_tuple" in mutations.columns:
            mutations = mutations.drop(columns=["variant_sequence_and_rc_tuple"])

        # Calculate the number of semicolons in each entry
        mutations["semicolon_count"] = mutations["header"].str.count(";")

        # number of VCRSs
        number_of_vcrss = len(mutations)

        # number_of_unique_mutations
        number_of_unique_mutations = (mutations["semicolon_count"] == 0).sum()

        number_of_merged_mutations = number_of_mutations_total - number_of_unique_mutations

        # equivalent code to calculate number_of_merged_mutations
        # mutations["semicolon_count"] += 1

        # # Convert all 1 values to NaN
        # mutations["semicolon_count"] = mutations["semicolon_count"].replace(1, np.nan)

        # # Take the sum across all rows of the new column
        # number_of_merged_mutations = int(mutations["semicolon_count"].sum())

        mutations = mutations.drop(columns=["semicolon_count"])

        merging_report = f"""
        Number of variants total: {number_of_mutations_total}
        Number of variants merged: {number_of_merged_mutations}
        Number of unique variants: {number_of_unique_mutations}
        Number of VCRSs: {number_of_vcrss}
        """

        # Save the report string to the specified path
        if save_filtering_report_text:
            filtering_report_write_mode = "a" if os.path.exists(filtering_report_text_out) else "w"
            with open(filtering_report_text_out, filtering_report_write_mode, encoding="utf-8") as file:
                file.write(merging_report)

        logger.info(merging_report)
        logger.info("Merged headers were combined and separated using a semicolon (;). Occurences of identical VCRSs may be reduced by increasing w.")

    empty_kmer_count = (mutations["variant_sequence"] == "").sum()

    if empty_kmer_count > 0:
        logger.warning(f"{empty_kmer_count} VCRSs were empty and were not included in the output.")

    mutations = mutations[mutations["variant_sequence"] != ""]

    # Restore the original order (minus any dropped rows)
    if original_order:
        mutations = mutations.sort_values(by="original_order").drop(columns="original_order")

    if use_IDs:  # or (var_id_column in mutations.columns and not merge_identical):
        mutations["vcrs_id"] = generate_unique_ids(len(mutations))
        mutations[["vcrs_id", "header"]].to_csv(id_to_header_csv_out, index=False)  # make the mapping csv
    else:
        mutations["vcrs_id"] = mutations["header"]

    if save_variants_updated_csv:  # use variants_updated_csv_out if present,
        logger.info("Saving dataframe with updated variant info...")
        logger.warning("File size can be very large if the number of variants is large.")
        mutations.to_csv(variants_updated_csv_out, index=False)
        print(f"Updated variant info has been saved to {variants_updated_csv_out}")

    if len(mutations) > 0:
        mutations["fasta_format"] = ">" + mutations["vcrs_id"] + "\n" + mutations["variant_sequence"] + "\n"

        if save_wt_vcrs_fasta_and_t2g:
            if not save_variants_updated_csv:
                raise ValueError("save_variants_updated_csv must be True to create wt_vcrs_fasta_and_t2g")

            mutations_with_exactly_1_wt_sequence_per_row = mutations[["vcrs_id", "wt_sequence"]].copy()

            if merge_identical:  # remove the rows with multiple WT counterparts for 1 VCRS, and convert the list of strings to string
                # Step 1: Filter rows where the length of the set of the list in `wt_sequence` is 1
                mutations_with_exactly_1_wt_sequence_per_row = mutations_with_exactly_1_wt_sequence_per_row[mutations_with_exactly_1_wt_sequence_per_row["wt_sequence"].apply(lambda x: len(set(x)) == 1)]

                # Step 2: Convert the list to a string
                mutations_with_exactly_1_wt_sequence_per_row["wt_sequence"] = mutations_with_exactly_1_wt_sequence_per_row["wt_sequence"].apply(lambda x: x[0])

            mutations_with_exactly_1_wt_sequence_per_row["fasta_format_wt"] = ">" + mutations_with_exactly_1_wt_sequence_per_row["vcrs_id"] + "\n" + mutations_with_exactly_1_wt_sequence_per_row["wt_sequence"] + "\n"

    # Save mutated sequences in new fasta file
    if not mutations.empty:
        with open(vcrs_fasta_out, "w", encoding="utf-8") as fasta_file:
            fasta_file.write("".join(mutations["fasta_format"].values))

        create_identity_t2g(vcrs_fasta_out, vcrs_t2g_out)

    logger.info("FASTA file containing VCRSs created at %s.", vcrs_fasta_out)
    logger.info("t2g file containing VCRSs created at %s.", vcrs_t2g_out)

    if save_wt_vcrs_fasta_and_t2g:
        with open(wt_vcrs_fasta_out, "w", encoding="utf-8") as fasta_file:
            fasta_file.write("".join(mutations_with_exactly_1_wt_sequence_per_row["fasta_format_wt"].values))
        create_identity_t2g(wt_vcrs_fasta_out, wt_vcrs_t2g_out)  # separate t2g is needed because it may have a subset of the rows of mutant (because it doesn't contain any VCRSs with merged mutations and 2+ originating WT sequences)

    # When stream_output is True, return list of mutated seqs
    if return_variant_output:
        all_mut_seqs = []
        all_mut_seqs.extend(mutations["variant_sequence"].values)

        # Remove empty strings from final list of mutated sequences (these are introduced when unknown mutations are encountered)
        while "" in all_mut_seqs:
            all_mut_seqs.remove("")

        if len(all_mut_seqs) > 0:
            report_time_elapsed(start_time, logger=logger, function_name="build")
            return all_mut_seqs

    report_time_elapsed(start_time, logger=logger, function_name="build")
