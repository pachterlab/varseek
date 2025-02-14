"""varseek info and specific helper functions."""
# CELL
import os
import subprocess
import time
from collections import OrderedDict

import numpy as np
import pandas as pd
import pyfastx
from tqdm import tqdm

from varseek.constants import mutation_pattern
from varseek.utils import (
    add_mcrs_mutation_type,
    align_to_normal_genome_and_build_dlist,
    calculate_nearby_mutations,
    calculate_total_gene_info,
    check_file_path_is_string_with_valid_extension,
    collapse_df,
    compare_cdna_and_genome,
    compute_distance_to_closest_splice_junction,
    create_df_of_mcrs_to_self_headers,
    download_ensembl_reference_files,
    download_t2t_reference_files,
    explode_df,
    fasta_summary_stats,
    get_df_overlap,
    get_mcrss_that_pseudoalign_but_arent_dlisted,
    is_program_installed,
    is_valid_int,
    longest_homopolymer,
    make_function_parameter_to_value_dict,
    make_mapping_dict,
    plot_histogram_of_nearby_mutations_7_5,
    plot_kat_histogram,
    print_varseek_dry_run,
    report_time_elapsed,
    reverse_complement,
    safe_literal_eval,
    save_params_to_config_file,
    save_run_info,
    set_up_logger,
    splitext_custom,
    swap_ids_for_headers_in_fasta,
    triplet_stats,
)

tqdm.pandas()
logger = set_up_logger()
pd.set_option("display.max_columns", None)


def add_some_mutation_information_when_cdna_and_genome_combined(df, columns_to_change):
    for column in columns_to_change:
        # Create new columns
        df[f"{column}_cdna"] = None
        df[f"{column}_genome"] = None
        df.loc[df["source"] == "cdna", f"{column}_cdna"] = df.loc[df["source"] == "cdna", column]
        df.loc[df["source"] == "genome", f"{column}_genome"] = df.loc[df["source"] == "genome", column]

    # Create a helper DataFrame by grouping based on 'header_cdna'
    grouped = df.groupby("header_cdna")

    for id_val, group in grouped:
        for column in columns_to_change:
            # Find the cdna_info from the 'cdna' row for this group
            cdna_info_value = group.loc[group["source"] == "cdna", f"{column}_cdna"].values
            genome_info_value = group.loc[group["source"] == "genome", f"{column}_genome"].values

            # If there's a cdna_info, update the genome row with it
            if len(cdna_info_value) > 0 and len(genome_info_value) > 0:
                df.loc[
                    (df["header_cdna"] == id_val) & (df["source"] == "genome"),
                    f"{column}_cdna",
                ] = cdna_info_value[0]
                df.loc[
                    (df["header_cdna"] == id_val) & (df["source"] == "cdna"),
                    f"{column}_genome",
                ] = genome_info_value[0]

    return df


def add_mutation_information(mutation_metadata_df, mutation_column="mutation", mcrs_source="cdna"):
    mutation_metadata_df[["nucleotide_positions", "actual_variant"]] = mutation_metadata_df[mutation_column].str.extract(mutation_pattern)

    split_positions = mutation_metadata_df["nucleotide_positions"].str.split("_", expand=True)
    mutation_metadata_df["start_variant_position"] = split_positions[0]

    if split_positions.shape[1] > 1:
        mutation_metadata_df["end_variant_position"] = split_positions[1].fillna(split_positions[0])
    else:
        mutation_metadata_df["end_variant_position"] = mutation_metadata_df["start_variant_position"]

    mutation_metadata_df[["start_variant_position", "end_variant_position"]] = mutation_metadata_df[["start_variant_position", "end_variant_position"]].astype("Int64")

    if mcrs_source is not None:
        mutation_metadata_df[f"nucleotide_positions_{mcrs_source}"] = mutation_metadata_df["nucleotide_positions"]
        mutation_metadata_df[f"actual_variant_{mcrs_source}"] = mutation_metadata_df["actual_variant"]
        mutation_metadata_df[f"start_variant_position_{mcrs_source}"] = mutation_metadata_df["start_variant_position"]
        mutation_metadata_df[f"end_variant_position_{mcrs_source}"] = mutation_metadata_df["end_variant_position"]

    return mutation_metadata_df


def print_list_columns():
    print("Available values for `columns_to_include`:")
    for col, description_and_utilized_parameters_tuple in columns_to_include_possible_values.items():
        print(f"- {col}:\n    {description_and_utilized_parameters_tuple[0]}\n    {description_and_utilized_parameters_tuple[1]}\n")


def validate_input_info(params_dict):
    # Directories
    if not isinstance(params_dict.get("input_dir"), str) or not os.path.isdir(params_dict.get("input_dir")):  # only use os.path.isdir when I require that a directory already exists
        raise ValueError(f"Invalid value for input_dir: {params_dict.get('input_dir')}")
    if not isinstance(params_dict.get("out"), str):
        raise ValueError(f"Invalid value for out: {params_dict.get('out')}")
    if params_dict.get("reference_out_dir") and (not isinstance(params_dict.get("reference_out_dir"), str) or not os.path.isdir(params_dict.get("reference_out_dir"))):
        raise ValueError(f"Invalid value for reference_out_dir: {params_dict.get('reference_out_dir')}")

    # file paths
    for param_name, file_type in {
        "vcrs_fasta": "fasta",
        "variants_updated_csv": "csv",
        "id_to_header_csv": "csv",
        "gtf": "gtf",
        "variants_updated_vk_info_csv_out": "csv",
        "variants_updated_exploded_vk_info_csv_out": "csv",
        "dlist_genome_fasta_out": "fasta",
        "dlist_cdna_fasta_out": "fasta",
        "dlist_combined_fasta_out": "fasta",
        "reference_cdna_fasta": "fasta",
        "reference_genome_fasta": "fasta",
        "variants": "csv",
    }:
        check_file_path_is_string_with_valid_extension(params_dict.get(param_name), param_name, file_type)

    # dlist reference files
    for dlist_reference_file in ["dlist_reference_source", "dlist_reference_genome_fasta", "dlist_reference_cdna_fasta", "dlist_reference_gtf"]:
        if params_dict.get(dlist_reference_file):
            if not isinstance(params_dict.get(dlist_reference_file), str):
                raise ValueError(f"{dlist_reference_file} must be a string, got {type(params_dict.get(dlist_reference_file))}")
            if params_dict.get(dlist_reference_file) not in supported_dlist_reference_values and not os.path.isfile(params_dict.get(dlist_reference_file)):
                raise ValueError(f"Invalid value for {dlist_reference_file}: {params_dict.get(dlist_reference_file)}")
    if params_dict.get("dlist_reference_genome_fasta") in supported_dlist_reference_values and params_dict.get("dlist_reference_gtf") in supported_dlist_reference_values:
        if not params_dict.get("dlist_reference_genome_fasta") == params_dict.get("dlist_reference_gtf"):
            raise ValueError(f"dlist_reference_genome_fasta and dlist_reference_gtf must be the same value when using a supported dlist reference. Got {params_dict.get('dlist_reference_genome_fasta')} and {params_dict.get('dlist_reference_gtf')}.")
    # check if dlist_reference_source is not a valid value and the 3 dlist file parameters are also not valid values (i.e., a real file or a supported dlist reference)
    if not params_dict.get("dlist_reference_source") in supported_dlist_reference_values and not (
        (os.path.isfile(params_dict.get("dlist_reference_genome_fasta")) or params_dict.get("dlist_reference_genome_fasta") in supported_dlist_reference_values) and
        (os.path.isfile(params_dict.get("dlist_reference_cdna_fasta")) or params_dict.get("dlist_reference_cdna_fasta") in supported_dlist_reference_values) and
        (os.path.isfile(params_dict.get("dlist_reference_gtf")) or params_dict.get("dlist_reference_gtf") in supported_dlist_reference_values)
    ):
        raise ValueError(
            f"Invalid value for dlist_reference_source: {params_dict.get('dlist_reference_source')} without specifying dlist_reference_genome_fasta, dlist_reference_cdna_fasta, and dlist_reference_gtf. dlist_reference_source must be one of {supported_dlist_reference_values}, or the other arguments must be provided (each as valid file paths or one of {supported_dlist_reference_values})."
        )

    # column names
    for column in ["vcrs_id_column", "vcrs_sequence_column", "vcrs_source_column", "var_column", "seq_id_column", "variant_cdna_column", "seq_id_cdna_column", "variant_genome_column", "seq_id_genome_column"]:
        if not isinstance(params_dict.get(column), str):
            raise ValueError(f"Invalid column name: {params_dict.get(column)}")

    # columns_to_include
    columns_to_include = params_dict.get("columns_to_include")
    if not isinstance(columns_to_include, (list, set, str, tuple)):
        raise ValueError(f"columns_to_include must be a string or list of strings, got {type(columns_to_include)}")
    if isinstance(columns_to_include, (list, set, tuple)):
        if not all(isinstance(col, str) for col in columns_to_include):
            raise ValueError("All elements in columns_to_include must be strings.")
        if not all(col in columns_to_include_possible_values for col in columns_to_include):
            raise ValueError(f"columns_to_include must be a subset of {columns_to_include_possible_values}. Got {columns_to_include}. Use 'all' to include all columns.")

    # integers - optional just means that it's in kwargs
    for param_name, min_value, optional_value in [
        ("w", 1, False),
        ("max_ambiguous_vcrs", 0, False),
        ("max_ambiguous_reference", 0, False),
        ("dlist_reference_ensembl_release", 50, False),
        ("threads", 1, False),
        ("near_splice_junction_threshold", 1, True),
    ]:
        param_value = params_dict.get(param_name)
        if not is_valid_int(param_value, ">=", min_value, optional=optional_value):
            raise ValueError(f"{param_name} must be an integer >= {min_value}. Got {param_value} of type {type(param_value)}.")

    k = params_dict.get("k")
    w = params_dict.get("w")
    if w and k:
        if not int(k) > int(w):
            raise ValueError(f"k must be an integer > w. Got k={k}, w={w}.")
    if int(k) % 2 != 0 or int(k) > 63:
        logger.warning(f"If running a workflow with vk ref or kb ref, k should be an odd number between 1 and 63. Got k={k}.")

    # boolean
    for param_name in ["vcrs_strandedness", "verbose", "save_variants_updated_exploded_vk_info_csv", "make_pyfastx_summary_file", "make_kat_histogram", "dry_run", "list_columns", "overwrite", "threads"]:
        param_value = params_dict.get(param_name)
        if not isinstance(param_value, bool):
            raise ValueError(f"{param_name} must be a boolean. Got {param_value} of type {type(param_value)}.")


supported_dlist_reference_values = {"T2T", "grch37", "grch38"}

# {column_name, (description, list_of_utilized_parameters)}
columns_to_include_possible_values = OrderedDict(
    [
        ("all", ("Include all possible columns", ["all parameters"])),
        ("cdna_and_genome_same", ("Whether the cDNA-derived and genome-derived VCRSs are the same", ["w", "reference_cdna_fasta", "reference_genome_fasta", "variants"])),
        ("distance_to_nearest_splice_junction", ("Distance to the nearest splice junction (bases) based on the GTF file", ["gtf", "near_splice_junction_threshold"])),
        ("number_of_variants_in_this_gene_total", ("Number of variants per gene", [])),
        ("header_with_gene_name", ("Header with gene name (e.g., ENST00004156 (BRCA1):c.123A>T)", [])),
        ("nearby_variants", ("The list of nearby variants (i.e., within `k` bases) for each variant", ["k"])),
        ("nearby_variants_count", ("Nearby variants count", ["k"])),
        ("has_a_nearby_variant", ("Has a nearby variant (a boolean of `nearby_variants_count`)", ["k"])),
        ("vcrs_header_length", ("VCRS header length", ["k"])),
        ("vcrs_sequence_length", ("VCRS sequence length", [])),
        ("alignment_to_reference", ("States whether a VCRS contains any k-mer that aligns anywhere in the reference genome ('genome'), transcriptome ('cdna'), genome and transcriptome ('cdna_and_genome'), or neither the genome nor transcriptome ('none') (requires bowtie2)", ["k", "max_ambiguous_vcrs", "max_ambiguous_reference", "dlist_reference_genome_fasta", "dlist_reference_cdna_fasta", "dlist_reference_gtf", "dlist_genome_fasta_out", "dlist_cdna_fasta_out", "dlist_combined_fasta_out", "threads", "vcrs_strandedness"])),
        ("alignment_to_reference_count_total", ("The total number of times that all k-mers within a VCRS match up to any k-mer in the genome and transcriptome specified by `dlist_reference_source` (requires bowtie2)", ["k", "max_ambiguous_vcrs", "max_ambiguous_reference", "dlist_reference_genome_fasta", "dlist_reference_cdna_fasta", "dlist_reference_gtf", "dlist_genome_fasta_out", "dlist_cdna_fasta_out", "dlist_combined_fasta_out", "threads", "vcrs_strandedness"])),
        ("alignment_to_reference_count_cdna", ("The total number of times that all k-mers within a VCRS match up to any k-mer in the cDNA specified by `dlist_reference_source` (requires bowtie2)", ["k", "max_ambiguous_vcrs", "max_ambiguous_reference", "dlist_reference_genome_fasta", "dlist_reference_cdna_fasta", "dlist_reference_gtf", "dlist_genome_fasta_out", "dlist_cdna_fasta_out", "dlist_combined_fasta_out", "threads", "vcrs_strandedness"])),
        ("alignment_to_reference_count_genome", ("The total number of times that all k-mers within a VCRS match up to any k-mer in the genome specified by `dlist_reference_source` (requires bowtie2)", ["k", "max_ambiguous_vcrs", "max_ambiguous_reference", "dlist_reference_genome_fasta", "dlist_reference_cdna_fasta", "dlist_reference_gtf", "dlist_genome_fasta_out", "dlist_cdna_fasta_out", "dlist_combined_fasta_out", "threads", "vcrs_strandedness"])),
        ("substring_alignment_to_reference", ("States whether a VCRS contains any k-mer that is a substring alignment to anywhere in the reference genome ('genome'), transcriptome ('cdna'), genome and transcriptome ('cdna_and_genome'), or neither the genome nor transcriptome ('none') (requires bowtie2)", ["k", "max_ambiguous_vcrs", "max_ambiguous_reference", "dlist_reference_genome_fasta", "dlist_reference_cdna_fasta", "dlist_reference_gtf", "dlist_genome_fasta_out", "dlist_cdna_fasta_out", "dlist_combined_fasta_out", "threads", "vcrs_strandedness"])),
        ("substring_alignment_to_reference_count_total", ("Number of substring matches to normal human reference (requires bowtie2)", ["k", "max_ambiguous_vcrs", "max_ambiguous_reference", "dlist_reference_genome_fasta", "dlist_reference_cdna_fasta", "dlist_reference_gtf", "dlist_genome_fasta_out", "dlist_cdna_fasta_out", "dlist_combined_fasta_out", "threads", "vcrs_strandedness"])),
        ("substring_alignment_to_reference_count_cdna", ("Number of substring matches to normal human reference cDNA (requires bowtie2)", ["k", "max_ambiguous_vcrs", "max_ambiguous_reference", "dlist_reference_genome_fasta", "dlist_reference_cdna_fasta", "dlist_reference_gtf", "dlist_genome_fasta_out", "dlist_cdna_fasta_out", "dlist_combined_fasta_out", "threads", "vcrs_strandedness"])),
        ("substring_alignment_to_reference_count_genome", ("Number of substring matches to normal human reference genome (requires bowtie2)", ["k", "max_ambiguous_vcrs", "max_ambiguous_reference", "dlist_reference_genome_fasta", "dlist_reference_cdna_fasta", "dlist_reference_gtf", "dlist_genome_fasta_out", "dlist_cdna_fasta_out", "dlist_combined_fasta_out", "threads", "vcrs_strandedness"])),
        ("pseudoaligned_to_reference", ("Pseudoaligned to human reference", ["k", "dlist_reference_genome_fasta", "dlist_reference_gtf", "threads", "vcrs_strandedness"])),
        ("pseudoaligned_to_reference_despite_not_truly_aligning", ("Pseudoaligned to human reference despite not truly aligning", ["k", "dlist_reference_genome_fasta", "dlist_reference_gtf", "threads", "vcrs_strandedness"])),
        ("number_of_kmers_with_overlap_to_other_VCRSs", ("Number of k-mers with overlap to other VCRS items in VCRS reference", ["k", "vcrs_strandedness"])),
        ("number_of_other_VCRSs_with_overlapping_kmers", ("Number of VCRS items with overlapping k-mers in VCRS reference", ["k", "vcrs_strandedness"])),
        ("VCRSs_with_overlapping_kmers", ("VCRS items with overlapping k-mers in VCRS reference (requires bowtie2)", ["k", "threads"])),
        ("kmer_overlap_with_other_VCRSs", ("K-mer overlap in VCRS reference (a boolean of `number_of_kmers_with_overlap_to_other_VCRSs`)", [])),
        ("longest_homopolymer_length", ("Longest homopolymer length", [])),
        ("longest_homopolymer", ("Longest homopolymer", [])),
        ("num_distinct_triplets", ("Number of distinct triplets", [])),
        ("num_total_triplets", ("Number of total triplets", [])),
        ("triplet_complexity", ("Triplet complexity", [])),
        ("vcrs_mutation_type", ("VCRS mutation type", [])),
        ("merged_variants_in_VCRS_entry", ("Concatenated headers in VCRS", [])),
        ("number_of_variants_in_VCRS_entry", ("Number of variants in VCRS header", [])),
        ("vcrs_sequence_rc", ("VCRS sequence reverse complement", [])),
        ("VCRSs_for_which_this_VCRS_is_a_substring", ("Entries for which this VCRS is substring (requires bowtie2)", ["threads"])),
        ("VCRSs_for_which_this_VCRS_is_a_superstring", ("Entries for which this VCRS is superstring (requires bowtie2)", ["threads"])),
        ("VCRS_is_a_substring_of_another_VCRS", ("VCRS is substring (requires bowtie2)", ["threads"])),
        ("VCRS_is_a_superstring_of_another_VCRS", ("VCRS is superstring (requires bowtie2)", ["threads"])),
    ]
)


# TODO: finish implementing the cdna/genome column stuff, and remove hard-coding of some column names
def info(
    input_dir,
    columns_to_include=("number_of_variants_in_this_gene_total", "alignment_to_reference_count_total", "pseudoaligned_to_reference_despite_not_truly_aligning", "longest_homopolymer_length", "triplet_complexity"),
    k=59,
    max_ambiguous_vcrs=0,
    max_ambiguous_reference=0,
    dlist_reference_source="T2T",
    dlist_reference_ensembl_release=111,
    vcrs_fasta=None,
    variants_updated_csv=None,
    id_to_header_csv=None,  # if none then assume no swapping occurred
    gtf=None,
    dlist_reference_genome_fasta=None,
    dlist_reference_cdna_fasta=None,
    dlist_reference_gtf=None,
    vcrs_id_column="vcrs_id",
    vcrs_sequence_column="variant_sequence",
    vcrs_source_column="vcrs_source",  # if input df has concatenated cdna and header VCRS's, then I want to know whether it came from cdna or genome
    var_column="mutation_cdna",
    seq_id_column="seq_ID",
    variant_cdna_column="mutation_cdna",
    seq_id_cdna_column="seq_ID",
    variant_genome_column="mutation_genome",
    seq_id_genome_column="chromosome",
    out=None,
    reference_out_dir=None,
    variants_updated_vk_info_csv_out=None,
    variants_updated_exploded_vk_info_csv_out=None,
    dlist_genome_fasta_out=None,
    dlist_cdna_fasta_out=None,
    dlist_combined_fasta_out=None,
    save_variants_updated_exploded_vk_info_csv=False,
    make_pyfastx_summary_file=False,
    make_kat_histogram=False,
    dry_run=False,
    list_columns=False,
    list_d_list_values=False,
    overwrite=False,
    threads=2,
    verbose=True,
    **kwargs,
):
    """
    Takes in the input directory containing with the VCRS fasta file generated from varseek build, and returns a dataframe with additional columns containing information about the variants.

    # Required input arguments:
    - input_dir     (str) Path to the directory containing the input files. Corresponds to `out` in the varseek build function.

    # Additional Parameters
    - columns_to_include                 (str or list[str]) List of columns to include in the output dataframe. Default: ("number_of_variants_in_this_gene_total", "alignment_to_reference_count_total", "pseudoaligned_to_reference_despite_not_truly_aligning", "longest_homopolymer_length", "triplet_complexity"). See all possible values and their description by setting list_columns=True (python) or --list_columns (command line).
    - k                                  (int) Length of the k-mers utilized by kallisto | bustools. Only used by the following columns: 'nearby_variants', 'nearby_variants_count', 'has_a_nearby_variant', 'alignment_to_reference', 'alignment_to_reference_count_total', 'alignment_to_reference_count_cdna', 'alignment_to_reference_count_genome', 'substring_alignment_to_reference', 'substring_alignment_to_reference_count_total', 'substring_alignment_to_reference_count_cdna',  'substring_alignment_to_reference_count_genome', 'pseudoaligned_to_reference', 'pseudoaligned_to_reference_despite_not_truly_aligning', 'number_of_kmers_with_overlap_to_other_VCRSs', 'number_of_other_VCRSs_with_overlapping_kmers', 'VCRSs_with_overlapping_kmers', 'kmer_overlap_with_other_VCRSs'; and when make_kat_histogram==True. Default: 59.
    - max_ambiguous_vcrs                 (int) Maximum number of 'N' characters allowed in the VCRS when considering alignment to the reference genome/transcriptome. Only used by the following columns: 'alignment_to_reference', 'alignment_to_reference_count_total', 'alignment_to_reference_count_cdna', 'alignment_to_reference_count_genome', 'substring_alignment_to_reference', 'substring_alignment_to_reference_count_total', 'substring_alignment_to_reference_count_cdna',  'substring_alignment_to_reference_count_genome'. Default: 0.
    - max_ambiguous_reference            (int) Maximum number of 'N' characters allowed in the aligned reference genome portion when considering alignment to the reference genome/transcriptome. Only used by the following columns: 'alignment_to_reference', 'alignment_to_reference_count_total', 'alignment_to_reference_count_cdna', 'alignment_to_reference_count_genome', 'substring_alignment_to_reference', 'substring_alignment_to_reference_count_total', 'substring_alignment_to_reference_count_cdna',  'substring_alignment_to_reference_count_genome'. Default: 0.
    - dlist_reference_source             (str) Source of the d-list reference genome and transcriptome if files are not provided by `dlist_reference_genome_fasta`, `dlist_reference_cdna_fasta`, and `dlist_reference_gtf`. Only used by the following columns: 'alignment_to_reference', 'alignment_to_reference_count_total', 'alignment_to_reference_count_cdna', 'alignment_to_reference_count_genome', 'substring_alignment_to_reference', 'substring_alignment_to_reference_count_total', 'substring_alignment_to_reference_count_cdna',  'substring_alignment_to_reference_count_genome', 'pseudoaligned_to_reference', 'pseudoaligned_to_reference_despite_not_truly_aligning'. Possible values are {supported_dlist_reference_values}. Ignored if values for `dlist_reference_genome_fasta`, `dlist_reference_cdna_fasta`, and `dlist_reference_gtf` are provided. Default: "T2T". (will automatically download the T2T reference genome files to `reference_out_dir`)
    - dlist_reference_ensembl_release    (int) Ensembl release number for the d-list reference genome and transcriptome if files are not provided by `dlist_reference_genome_fasta`, `dlist_reference_cdna_fasta`, and `dlist_reference_gtf`. Only used by the following columns: 'alignment_to_reference', 'alignment_to_reference_count_total', 'alignment_to_reference_count_cdna', 'alignment_to_reference_count_genome', 'substring_alignment_to_reference', 'substring_alignment_to_reference_count_total', 'substring_alignment_to_reference_count_cdna',  'substring_alignment_to_reference_count_genome', 'pseudoaligned_to_reference', 'pseudoaligned_to_reference_despite_not_truly_aligning'. Only used if `dlist_reference_source`, `dlist_reference_genome_fasta`, `dlist_reference_cdna_fasta`, `dlist_reference_gtf` is grch37 or grch38. Default: 111. (will automatically download the Ensembl reference genome files to `reference_out_dir`)

    # Optional input file paths: (only needed if changing/customizing file names or locations):
    - vcrs_fasta                         (str) Path to the VCRS fasta file generated from varseek build. Corresponds to `vcrs_fasta_out` in the varseek build function. Only needed if the original file was changed or renamed. Default: None (will find it in `input_dir`).
    - variants_updated_csv              (str) Path to the updated dataframe containing the VCRS headers and sequences. Corresponds to `variants_updated_csv_out` in the varseek build function. Only needed if the original file was changed or renamed. Default: None (will find it in `input_dir` if it exists).
    - id_to_header_csv                   (str) Path to the csv file containing the mapping of IDs to headers generated from varseek build corresponding to vcrs_fasta. Corresponds to `id_to_header_csv_out` in the varseek build function. Only needed if the original file was changed or renamed. Default: None (will find it in `input_dir` if it exists).
    - gtf                                (str) Path to the GTF file containing the gene annotations for the reference genome. Corresponds to `gtf` in the varseek build function. Must align to genome coordinates used in the annotation of variants. Only used by the following columns: 'distance_to_nearest_splice_junction'. Default: None.
    - dlist_reference_genome_fasta       (str) Path to the reference genome fasta file for the d-list. Only used by the following columns: 'alignment_to_reference', 'alignment_to_reference_count_total', 'alignment_to_reference_count_cdna', 'alignment_to_reference_count_genome', 'substring_alignment_to_reference', 'substring_alignment_to_reference_count_total', 'substring_alignment_to_reference_count_cdna',  'substring_alignment_to_reference_count_genome', 'pseudoaligned_to_reference', 'pseudoaligned_to_reference_despite_not_truly_aligning'. Default: `dlist_reference_source`.
    - dlist_reference_cdna_fasta         (str) Path to the reference cDNA fasta file for the d-list. Only used by the following columns: 'alignment_to_reference', 'alignment_to_reference_count_total', 'alignment_to_reference_count_cdna', 'alignment_to_reference_count_genome', 'substring_alignment_to_reference', 'substring_alignment_to_reference_count_total', 'substring_alignment_to_reference_count_cdna',  'substring_alignment_to_reference_count_genome'. Default: `dlist_reference_source`.
    - dlist_reference_gtf                (str) Path to the GTF file containing the gene annotations for the reference genome. Only used by the following columns: 'pseudoaligned_to_reference', 'pseudoaligned_to_reference_despite_not_truly_aligning'. Default: `dlist_reference_source`.

    # Column names in variants_updated_csv:
    - vcrs_id_column                     (str) Name of the column containing the VCRS IDs in `variants_updated_csv`. Only used if `variants_updated_csv` exists (i.e., was generated from varseek build). Default: 'vcrs_id'.
    - vcrs_sequence_column               (str) Name of the column containing the VCRS sequences in `variants_updated_csv`. Only used if `variants_updated_csv` exists (i.e., was generated from varseek build). Default: 'mutant_sequence'.
    - vcrs_source_column                 (str) Name of the column containing the source of the VCRS (cdna or genome) in `variants_updated_csv`. Only used if `variants_updated_csv` exists (i.e., was generated from varseek build). Default: 'mcrs_source'.
    - var_column                         (str) Name of the column containing the variants in `variants_updated_csv`. Only used if `variants_updated_csv` exists (i.e., was generated from varseek build). Default: 'mutation_cdna'.
    - seq_id_column                      (str) Name of the column containing the sequence IDs in `variants_updated_csv`. Only used if `variants_updated_csv` exists (i.e., was generated from varseek build). Default: 'seq_ID'.
    - variant_cdna_column               (str) Name of the column containing the cDNA variants in `variants_updated_csv`. Only used if `variants_updated_csv` exists (i.e., was generated from varseek build) and contains information regarding both genome and cDNA notation (essential if running spliced + unspliced workflow, optional otherwise). Default: 'mutation_cdna'.
    - seq_id_cdna_column                 (str) Name of the column containing the cDNA sequence IDs in `variants_updated_csv`. Only used if `variants_updated_csv` exists (i.e., was generated from varseek build) and contains information regarding both genome and cDNA notation (essential if running spliced + unspliced workflow, optional otherwise). Default: 'seq_ID'.
    - variant_genome_column             (str) Name of the column containing the genome variants in `variants_updated_csv`. Only used if `variants_updated_csv` exists (i.e., was generated from varseek build) and contains information regarding both genome and cDNA notation (essential if running spliced + unspliced workflow, optional otherwise). Default: 'mutation_genome'.
    - seq_id_genome_column               (str) Name of the column containing the genome sequence IDs in `variants_updated_csv`. Only used if `variants_updated_csv` exists (i.e., was generated from varseek build) and contains information regarding both genome and cDNA notation (essential if running spliced + unspliced workflow, optional otherwise). Default: 'chromosome'.

    # Output file paths:
    - out                                (str) Path to the directory where the output files will be saved. Default: `input_dir`.
    - reference_out_dir                  (str) Path to the directory where the reference files will be saved. Default: `out`/reference.
    - variants_updated_vk_info_csv_out  (str) Path to the output csv file containing the updated dataframe with the additional columns. Default: `out`/variants_updated_vk_info.csv.
    - variants_updated_exploded_vk_info_csv_out (str) Path to the output csv file containing the exploded dataframe with the additional columns. Default: `out`/variants_updated_exploded_vk_info.csv.
    - dlist_genome_fasta_out             (str) Path to the output fasta file containing the d-list sequences for the genome-based alignmed. Only used by the following columns: 'alignment_to_reference', 'alignment_to_reference_count_total', 'alignment_to_reference_count_cdna', 'alignment_to_reference_count_genome', 'substring_alignment_to_reference', 'substring_alignment_to_reference_count_total', 'substring_alignment_to_reference_count_cdna',  'substring_alignment_to_reference_count_genome'. Default: `out`/dlist_genome.fa.
    - dlist_cdna_fasta_out               (str) Path to the output fasta file containing the d-list sequences for the cDNA-based alignmed. Only used by the following columns: 'alignment_to_reference', 'alignment_to_reference_count_total', 'alignment_to_reference_count_cdna', 'alignment_to_reference_count_genome', 'substring_alignment_to_reference', 'substring_alignment_to_reference_count_total', 'substring_alignment_to_reference_count_cdna',  'substring_alignment_to_reference_count_genome'.Default: `out`/dlist_cdna.fa.
    - dlist_combined_fasta_out           (str) Path to the output fasta file containing the d-list sequences  combined genome-based and cDNA-based alignment. Only used by the following columns: 'alignment_to_reference', 'alignment_to_reference_count_total', 'alignment_to_reference_count_cdna', 'alignment_to_reference_count_genome', 'substring_alignment_to_reference', 'substring_alignment_to_reference_count_total', 'substring_alignment_to_reference_count_cdna',  'substring_alignment_to_reference_count_genome'.Default: `out`/dlist.fa.

    # Returning and saving of optional output
    - save_variants_updated_exploded_vk_info_csv (bool) Whether to save the exploded dataframe. Default: False.
    - make_pyfastx_summary_file          (bool) Whether to make a summary file of the VCRS fasta file using pyfastx. Default: False.
    - make_kat_histogram                 (bool) Whether to make a histogram of the k-mer abundances using kat. Default: False.

    # General arguments:
    - dry_run                            (bool) Whether to do a dry run (i.e., print the parameters and return without running the function). Default: False.
    - list_columns                       (bool) Whether to list the possible values for `columns_to_include` and their descriptions and immediately exit. Default: False.
    - list_d_list_values                 (bool) Whether to list the possible values for `dlist_reference_source` and immediately exit. Default: False.
    - overwrite                          (bool) Whether to overwrite the output files if they already exist. Default: False.
    - threads                            (int) Number of threads to use for bowtie2 and bowtie2-build. Only used by the following columns: 'alignment_to_reference', 'alignment_to_reference_count_total', 'alignment_to_reference_count_cdna', 'alignment_to_reference_count_genome', 'substring_alignment_to_reference', 'substring_alignment_to_reference_count_total', 'substring_alignment_to_reference_count_cdna',  'substring_alignment_to_reference_count_genome', 'pseudoaligned_to_reference', 'pseudoaligned_to_reference_despite_not_truly_aligning', 'VCRSs_for_which_this_VCRS_is_a_substring', 'VCRSs_for_which_this_VCRS_is_a_superstring', 'VCRS_is_a_substring_of_another_VCRS', 'VCRS_is_a_superstring_of_another_VCRS'; and when 'VCRSs_for_which_this_VCRS_is_a_superstring', 'VCRS_is_a_substring_of_another_VCRS', 'VCRS_is_a_superstring_of_another_VCRS', make_kat_histogram==True. Default: 2.
    - verbose                            (bool) Whether to print verbose output. Default: True.

    # Hidden arguments (part of kwargs):
    - w                                  (int) Maximum length of the VCRS flanking regions. Must be an integer between [1, k-1]. Only utilized for the column 'cdna_and_genome_same'. Corresponds to `w` in the varseek build function. Default: 54.
    - bowtie2_path                        (str) Path to the directory containing the bowtie2 and bowtie2-build executables. Default: None.
    - vcrs_strandedness                  (bool) Whether to consider VCRSs as stranded when aligning to the human reference and comparing VCRS k-mers to each other. vcrs_strandedness True corresponds to treating forward and reverse-complement as distinct; False corresponds to treating them as the same. Corresponds to `vcrs_strandedness` in the varseek build function. Only used by the following columns: 'alignment_to_reference', 'alignment_to_reference_count_total', 'alignment_to_reference_count_cdna', 'alignment_to_reference_count_genome', 'substring_alignment_to_reference', 'substring_alignment_to_reference_count_total', 'substring_alignment_to_reference_count_cdna',  'substring_alignment_to_reference_count_genome', 'pseudoaligned_to_reference', 'pseudoaligned_to_reference_despite_not_truly_aligning', 'number_of_kmers_with_overlap_to_other_VCRSs', 'number_of_other_VCRSs_with_overlapping_kmers', 'VCRSs_with_overlapping_kmers, 'kmer_overlap_with_other_VCRSs'; and if make_kat_histogram==True. Default: False.
    - near_splice_junction_threshold     (int) Maximum distance from a splice junction to be considered "near" a splice junction. Only utilized for the column 'distance_to_nearest_splice_junction'. Default: 10.
    - reference_cdna_fasta               (str) Path to the reference cDNA fasta file. Only utilized for the column 'cdna_and_genome_same'. Default: None.
    - reference_genome_fasta             (str) Path to the reference genome fasta file. Only utilized for the column 'cdna_and_genome_same'. Default: None.
    - variants                      (str) Path to the variants csv file. Only utilized for the column 'cdna_and_genome_same'. Corresponds to `variants` in the varseek build function. Default: None.
    """
    # CELL
    # * 0. Informational arguments that exit early

    if list_columns:
        print_list_columns()
        return
    if list_d_list_values:
        print(f"Available values for `dlist_reference_source`: {supported_dlist_reference_values}")
        return

    # * 1. Start timer
    start_time = time.perf_counter()

    # * 2. Type-checking
    params_dict = make_function_parameter_to_value_dict(1)
    validate_input_info(params_dict)

    # * 3. Dry-run and set out folder (must to it up here or else config will save in the wrong place)
    if dry_run:
        print_varseek_dry_run(params_dict, function_name="info")
        return
    if out is None:
        out = input_dir if input_dir else "."

    # * 4. Save params to config file and run info file
    config_file = os.path.join(out, "config", "vk_info_config.json")
    save_params_to_config_file(params_dict, config_file)

    run_info_file = os.path.join(out, "config", "vk_info_run_info.txt")
    save_run_info(run_info_file)

    # * 5. Set up default folder/file input paths, and make sure the necessary ones exist
    if not vcrs_fasta:
        vcrs_fasta = os.path.join(input_dir, "vcrs.fa")
    if not os.path.isfile(vcrs_fasta):
        raise FileNotFoundError(f"File not found: {vcrs_fasta}")

    if not variants_updated_csv:
        variants_updated_csv = os.path.join(input_dir, "variants_updated.csv")
    if not os.path.isfile(variants_updated_csv):
        logger.warning(f"File not found: {variants_updated_csv}")
        variants_updated_csv = None

    if not id_to_header_csv:
        id_to_header_csv = os.path.join(input_dir, "id_to_header_mapping.csv")
    if not os.path.isfile(id_to_header_csv):
        logger.warning(f"File not found: {id_to_header_csv}")
        id_to_header_csv = None

    # * 6. Set up default folder/file output paths, and make sure they don't exist unless overwrite=True
    if not reference_out_dir:
        reference_out_dir = os.path.join(out, "reference")

    os.makedirs(out, exist_ok=True)
    os.makedirs(reference_out_dir, exist_ok=True)

    if not variants_updated_vk_info_csv_out:
        variants_updated_vk_info_csv_out = os.path.join(out, "variants_updated_vk_info.csv")
    if not variants_updated_exploded_vk_info_csv_out:
        variants_updated_exploded_vk_info_csv_out = os.path.join(out, "variants_updated_exploded_vk_info.csv")
    if not dlist_genome_fasta_out:  #! these 3 dlist paths are copied in vk ref
        dlist_genome_fasta_out = os.path.join(out, "dlist_genome.fa")
    if not dlist_cdna_fasta_out:
        dlist_cdna_fasta_out = os.path.join(out, "dlist_cdna.fa")
    if not dlist_combined_fasta_out:
        dlist_combined_fasta_out = os.path.join(out, "dlist.fa")

    # make sure directories of all output files exist
    output_files = [variants_updated_vk_info_csv_out, variants_updated_exploded_vk_info_csv_out, dlist_genome_fasta_out, dlist_cdna_fasta_out, dlist_combined_fasta_out]
    for output_file in output_files:
        if os.path.isfile(output_file) and not overwrite:
            raise ValueError(f"Output file '{output_file}' already exists. Set 'overwrite=True' to overwrite it.")
        if os.path.dirname(output_file):
            os.makedirs(os.path.dirname(output_file), exist_ok=True)

    # * 7. Define kwargs defaults
    w = kwargs.get("w", 54)
    bowtie_path = kwargs.get("bowtie2_path", None)
    vcrs_strandedness = kwargs.get("vcrs_strandedness", False)
    near_splice_junction_threshold = kwargs.get("near_splice_junction_threshold", 10)
    reference_cdna_fasta = kwargs.get("reference_cdna_fasta", None)
    reference_genome_fasta = kwargs.get("reference_genome_fasta", None)
    mutations_csv = kwargs.get("variants", None)

    # * 8. Start the actual function
    if columns_to_include == "all":
        make_pyfastx_summary_file = True
        make_kat_histogram = True

    if dlist_reference_source:
        if not dlist_reference_genome_fasta:
            dlist_reference_genome_fasta = dlist_reference_source
        if not dlist_reference_cdna_fasta:
            dlist_reference_cdna_fasta = dlist_reference_source
        if not dlist_reference_gtf:
            dlist_reference_gtf = dlist_reference_source

    if dlist_reference_genome_fasta == "T2T" or dlist_reference_cdna_fasta == "T2T" or dlist_reference_gtf == "T2T":
        t2t_reference_dir = os.path.join(reference_out_dir, "t2t")
        dlist_reference_genome_fasta, dlist_reference_cdna_fasta, dlist_reference_gtf = download_t2t_reference_files(t2t_reference_dir)
    elif dlist_reference_genome_fasta == "grch37" or dlist_reference_cdna_fasta == "grch37" or dlist_reference_gtf == "grch37":
        grch37_reference_dir = os.path.join(reference_out_dir, f"ensembl_grch37_release{dlist_reference_ensembl_release}")
        dlist_reference_genome_fasta, dlist_reference_cdna_fasta, dlist_reference_gtf = download_ensembl_reference_files(grch37_reference_dir, grch=37, ensembl_release=dlist_reference_ensembl_release)
    elif dlist_reference_genome_fasta == "grch38" or dlist_reference_cdna_fasta == "grch38" or dlist_reference_gtf == "grch38":
        grch38_reference_dir = os.path.join(reference_out_dir, f"ensembl_grch38_release{dlist_reference_ensembl_release}")
        dlist_reference_genome_fasta, dlist_reference_cdna_fasta, dlist_reference_gtf = download_ensembl_reference_files(grch38_reference_dir, grch=38, ensembl_release=dlist_reference_ensembl_release)

    columns_to_explode = ["header", "order"]
    columns_not_successfully_added = []

    # --np (N penalty) caps number of Ns in read (VCRS), reference (human reference genome/transcriptome), or both
    # --n-ceil (max_ambiguous_vcrs) caps number of Ns in read (VCRS) only
    # I have my remove_Ns_fasta function which caps number of Ns in reference (human reference genome/transcriptome) only
    if max_ambiguous_vcrs is None:  # no N-penalty for VCRS during d-listing
        max_ambiguous_vcrs = 99999  #! be careful of changing this number - it must be an int for bowtie2
    if max_ambiguous_reference is None:  # no N-penalty for reference during d-listing
        max_ambiguous_reference = 99999  #! be careful of changing this number - it is related to the condition in 'align_to_normal_genome_and_build_dlist' - max_ambiguous_reference < 9999

    if max_ambiguous_vcrs == 0 and max_ambiguous_reference == 0:  # probably redundant with the filters above but still nice to have
        N_penalty = 1
    else:
        N_penalty = 0

    output_stat_folder = f"{out}/stats"
    output_plot_folder = f"{out}/plots"

    os.makedirs(output_stat_folder, exist_ok=True)
    os.makedirs(output_plot_folder, exist_ok=True)

    # CELL

    if id_to_header_csv is not None:
        id_to_header_dict = make_mapping_dict(id_to_header_csv, dict_key="id")
        # header_to_id_dict = {value: key for key, value in id_to_header_dict.items()}
        mcrs_fasta_base, mcrs_fasta_ext = splitext_custom(vcrs_fasta)
        temp_header_fa = f"{mcrs_fasta_base}_with_headers{mcrs_fasta_ext}"
        swap_ids_for_headers_in_fasta(vcrs_fasta, id_to_header_csv, out_fasta=temp_header_fa)
    else:
        id_to_header_dict = None
        # header_to_id_dict = None
        temp_header_fa = vcrs_fasta

    # CELL
    # # Calculate lengths of lists in each column to explode
    # lengths_df = mutation_metadata_df[columns_to_explode].applymap(lambda x: len(x) if isinstance(x, list) else 0)

    # # Identify rows where list lengths differ across columns to explode
    # inconsistent_rows = lengths_df[lengths_df.nunique(axis=1) > 1]

    # # Display these problematic rows
    # print("Rows with inconsistent list lengths across columns to explode:")
    # inconsistent_rows

    # CELL
    if make_pyfastx_summary_file:
        output_pyfastx_stat_file = f"{output_stat_folder}/pyfastx_stats.txt"
        fasta_summary_stats(vcrs_fasta, output_file=output_pyfastx_stat_file)

    # CELL
    # columns_to_change = ['nucleotide_positions', 'start_variant_position', 'end_variant_position', 'actual_variant']

    if variants_updated_csv is None:  # does not support concatenated cdna and genome
        columns_original = []
        data = list(pyfastx.Fastx(vcrs_fasta))
        mutation_metadata_df = pd.DataFrame(data, columns=[vcrs_id_column, "vcrs_sequence"])

        if id_to_header_dict is not None:
            mutation_metadata_df["vcrs_header"] = mutation_metadata_df[vcrs_id_column].map(id_to_header_dict)
        else:
            mutation_metadata_df["vcrs_header"] = mutation_metadata_df[vcrs_id_column]

        contains_enst = mutation_metadata_df["vcrs_header"].iloc[0].find("ENST") != -1  # TODO: this just differentiates cdna from genome based on searching for ENST, but there may be other ways to discern cDNA from genome

        if contains_enst:
            mcrs_source = "cdna"
        else:
            mcrs_source = "genome"

        mutation_metadata_df[vcrs_source_column] = mcrs_source

        mutation_metadata_df["header_list"] = mutation_metadata_df["vcrs_header"].str.split(";")
        mutation_metadata_df["order_list"] = mutation_metadata_df["header_list"].apply(lambda x: list(range(len(x))))

        mcrs_header_has_merged_values = mutation_metadata_df["vcrs_header"].apply(lambda x: isinstance(x, str) and ";" in x).any()

        if mcrs_header_has_merged_values:
            mutation_metadata_df_exploded = explode_df(mutation_metadata_df, columns_to_explode)
        else:
            mutation_metadata_df_exploded = mutation_metadata_df

        mutation_metadata_df_exploded[["seq_ID", "mutation"]] = mutation_metadata_df_exploded["vcrs_header"].str.split(":", expand=True)

        mutation_metadata_df_exploded["seq_ID"] = mutation_metadata_df_exploded["seq_ID"].astype(str)

        mutation_metadata_df_exploded = add_mutation_information(mutation_metadata_df_exploded, mcrs_source=mcrs_source)

        if mcrs_source == "genome":
            mutation_metadata_df_exploded.rename(
                columns={"seq_ID": "chromosome", "mutation": "mutation_genome"},
                inplace=True,
            )

        columns_to_explode_extend_values = [col for col in mutation_metadata_df_exploded.columns if col not in [vcrs_id_column, "header", "vcrs_sequence"] and col not in columns_to_explode]

    else:
        mutation_metadata_df = pd.read_csv(variants_updated_csv)
        mutation_metadata_df.rename(
            columns={"header": "vcrs_header", vcrs_sequence_column: "vcrs_sequence"},
            inplace=True,
        )

        columns_original = mutation_metadata_df.columns.tolist()

        for column in mutation_metadata_df.columns:
            if column not in columns_to_explode + [
                vcrs_id_column,
                "vcrs_header",
                "vcrs_sequence",
                "variant_sequence_rc",
            ]:  # alternative: check if the first and last characters are '[' and ']', respectively
                mutation_metadata_df[column] = mutation_metadata_df[column].apply(lambda x: (safe_literal_eval(x) if isinstance(x, str) and x.startswith("[") and x.endswith("]") else x))

        columns_to_explode.extend(
            [
                col
                for col in mutation_metadata_df.columns
                if col
                not in [
                    vcrs_id_column,
                    "vcrs_header",
                    "vcrs_sequence",
                    "variant_sequence_rc",
                ]
            ]
        )
        mutation_metadata_df["header_list"] = mutation_metadata_df["vcrs_header"].str.split(";")
        mutation_metadata_df["order_list"] = mutation_metadata_df["header_list"].apply(lambda x: list(range(len(x))))

        if vcrs_source_column in mutation_metadata_df.columns:
            mcrs_source = mutation_metadata_df[vcrs_source_column].unique()
            if len(mcrs_source) > 1:
                mcrs_source = "combined"
            else:
                mcrs_source = mcrs_source[0]
        else:
            contains_enst = mutation_metadata_df["vcrs_header"].iloc[0].find("ENST") != -1  # TODO: this just differentiates cdna from genome based on searching for ENST, but there may be other ways to discern cDNA from genome

            if contains_enst:
                mcrs_source = "cdna"
            else:
                mcrs_source = "genome"

            mutation_metadata_df[vcrs_source_column] = mcrs_source

        if mcrs_source == "combined":
            mutation_metadata_df = add_some_mutation_information_when_cdna_and_genome_combined(mutation_metadata_df, columns_to_explode)
        else:
            mutation_metadata_df[f"nucleotide_positions_{mcrs_source}"] = mutation_metadata_df["nucleotide_positions"]
            mutation_metadata_df[f"actual_variant_{mcrs_source}"] = mutation_metadata_df["actual_variant"]
            mutation_metadata_df[f"start_variant_position_{mcrs_source}"] = mutation_metadata_df["start_variant_position"]
            mutation_metadata_df[f"end_variant_position_{mcrs_source}"] = mutation_metadata_df["end_variant_position"]

        columns_to_explode.extend(
            [
                f"nucleotide_positions_{mcrs_source}",
                f"actual_variant_{mcrs_source}",
                f"start_variant_position_{mcrs_source}",
                f"end_variant_position_{mcrs_source}",
            ]
        )

        mcrs_header_has_merged_values = mutation_metadata_df["vcrs_header"].apply(lambda x: isinstance(x, str) and ";" in x).any()

        if mcrs_header_has_merged_values:
            mutation_metadata_df_exploded = explode_df(mutation_metadata_df, columns_to_explode)
        else:
            mutation_metadata_df_exploded = mutation_metadata_df

        if "chromosome" in mutation_metadata_df_exploded.columns and "mutation_genome" in mutation_metadata_df_exploded.columns:
            mutation_metadata_df_exploded["header_genome"] = mutation_metadata_df_exploded["chromosome"].astype(str) + ":" + mutation_metadata_df_exploded["mutation_genome"].astype(str)

        if "seq_ID" in mutation_metadata_df_exploded.columns and "mutation_cdna" in mutation_metadata_df_exploded.columns:
            mutation_metadata_df_exploded["header_cdna"] = mutation_metadata_df_exploded["seq_ID"].astype(str) + ":" + mutation_metadata_df_exploded["mutation_cdna"].astype(str)

        if "seq_ID" in mutation_metadata_df_exploded.columns and "mutation" in mutation_metadata_df_exploded.columns:
            mutation_metadata_df_exploded["header_cds"] = mutation_metadata_df_exploded["seq_ID"].astype(str) + ":" + mutation_metadata_df_exploded["mutation"].astype(str)

        columns_to_explode_extend_values = [
            "header_genome",
            "header_cdna",
            "header_cds",
        ]

        if mutation_metadata_df_exploded[vcrs_source_column].unique()[0] == "cdna" and variant_genome_column in mutation_metadata_df_exploded:
            mutation_metadata_df_exploded = add_mutation_information(
                mutation_metadata_df_exploded,
                mutation_column=variant_genome_column,
                mcrs_source="genome",
            )
            columns_to_explode_extend_values.extend(
                [
                    "nucleotide_positions_genome",
                    "actual_variant_genome",
                    "start_variant_position_genome",
                    "end_variant_position_genome",
                ]
            )
            # TODO: this is a little hacky (I set these values in the function and then reset them now)
            mutation_metadata_df_exploded["nucleotide_positions"] = mutation_metadata_df_exploded[f"nucleotide_positions_{mcrs_source}"]
            mutation_metadata_df_exploded["actual_variant"] = mutation_metadata_df_exploded[f"actual_variant_{mcrs_source}"]
            mutation_metadata_df_exploded["start_variant_position"] = mutation_metadata_df_exploded[f"start_variant_position_{mcrs_source}"]
            mutation_metadata_df_exploded["end_variant_position"] = mutation_metadata_df_exploded[f"end_variant_position_{mcrs_source}"]

        if mutation_metadata_df_exploded[vcrs_source_column].unique()[0] == "genome" and variant_cdna_column in mutation_metadata_df_exploded:
            mutation_metadata_df_exploded = add_mutation_information(
                mutation_metadata_df_exploded,
                mutation_column=variant_cdna_column,
                mcrs_source="cdna",
            )
            columns_to_explode_extend_values.extend(
                [
                    "nucleotide_positions_cdna",
                    "actual_variant_cdna",
                    "start_variant_position_cdna",
                    "end_variant_position_cdna",
                ]
            )
            mutation_metadata_df_exploded["nucleotide_positions"] = mutation_metadata_df_exploded[f"nucleotide_positions_{mcrs_source}"]
            mutation_metadata_df_exploded["actual_variant"] = mutation_metadata_df_exploded[f"actual_variant_{mcrs_source}"]
            mutation_metadata_df_exploded["start_variant_position"] = mutation_metadata_df_exploded[f"start_variant_position_{mcrs_source}"]
            mutation_metadata_df_exploded["end_variant_position"] = mutation_metadata_df_exploded[f"end_variant_position_{mcrs_source}"]

    # CELL
    if mcrs_source in {"genome", "combined"}:
        mutation_metadata_df_exploded = mutation_metadata_df_exploded.loc[~((mutation_metadata_df_exploded[vcrs_source_column] == "genome") & ((pd.isna(mutation_metadata_df_exploded["chromosome"])) | (mutation_metadata_df_exploded["mutation_genome"].str.contains("g.nan", na=True))))]

    columns_to_explode.extend(columns_to_explode_extend_values)

    # CELL
    if columns_to_include == "all" or "cdna_and_genome_same" in columns_to_include:
        if "cdna_and_genome_same" in mutation_metadata_df_exploded.columns:  #! "cdna_and_genome_same" corresponds to column name in vk build
            columns_to_explode.append("cdna_and_genome_same")
        else:
            try:
                logger.info("Comparing cDNA and genome")
                mutation_metadata_df_exploded, columns_to_explode = compare_cdna_and_genome(
                    mutation_metadata_df_exploded,
                    reference_cdna_fasta=reference_cdna_fasta,
                    reference_genome_fasta=reference_genome_fasta,
                    mutations_csv=mutations_csv,
                    w=w,
                    mcrs_source=mcrs_source,
                    columns_to_explode=columns_to_explode,
                    seq_id_column_cdna=seq_id_cdna_column,
                    var_column_cdna=variant_cdna_column,
                    seq_id_column_genome=seq_id_genome_column,
                    var_column_genome=variant_genome_column,
                )
            except Exception as e:
                logger.error(f"Error comparing cDNA and genome: {e}")
                columns_not_successfully_added.append("cdna_and_genome_same")

    # CELL

    if columns_to_include == "all" or "distance_to_nearest_splice_junction" in columns_to_include:
        # Add metadata: distance to nearest splice junction
        try:
            logger.info("Computing distance to nearest splice junction")
            mutation_metadata_df_exploded, columns_to_explode = compute_distance_to_closest_splice_junction(
                mutation_metadata_df_exploded,
                gtf,
                columns_to_explode=columns_to_explode,
                near_splice_junction_threshold=near_splice_junction_threshold,
            )
        except Exception as e:
            logger.error(f"Error computing distance to nearest splice junction: {e}")
            columns_not_successfully_added.append("distance_to_nearest_splice_junction")

    # CELL
    if columns_to_include == "all" or "number_of_variants_in_this_gene_total" in columns_to_include or "header_with_gene_name" in columns_to_include:
        total_genes_output_stat_file = f"{output_stat_folder}/total_genes_and_transcripts.txt"
        try:
            logger.info("Calculating total gene info")
            mutation_metadata_df_exploded, columns_to_explode = calculate_total_gene_info(
                mutation_metadata_df_exploded,
                vcrs_id_column=vcrs_id_column,
                output_stat_file=total_genes_output_stat_file,
                output_plot_folder=output_plot_folder,
                columns_to_include=columns_to_include,
                columns_to_explode=columns_to_explode,
            )
        except Exception as e:
            logger.error(f"Error calculating total gene info: {e}")
            columns_not_successfully_added.extend(["number_of_variants_in_this_gene_total", "header_with_gene_name"])

    # CELL
    # Calculate variants within (k-1) of each mutation
    # compare transcript location for spliced only with cDNA header;
    # filter out genome rows where cdna and genome are the same (because I don't want to count spliced and unspliced as 2 separate things when they are the same - but maybe I do?) and compare genome location for all (both spliced and unspliced) with regular header (will be the sole way to add information for unspliced rows, and will add unspliced info for cdna comparisons);
    # take union of sets

    if columns_to_include == "all" or ("nearby_variants" in columns_to_include or "nearby_variants_count" in columns_to_include or "has_a_nearby_variant" in columns_to_include):
        try:
            logger.info("Calculating nearby variants")
            mutation_metadata_df_exploded, columns_to_explode = calculate_nearby_mutations(
                vcrs_source_column=vcrs_source_column,
                k=k,
                output_plot_folder=output_plot_folder,
                mcrs_source=mcrs_source,
                mutation_metadata_df_exploded=mutation_metadata_df_exploded,
                columns_to_explode=columns_to_explode,
            )
        except Exception as e:
            logger.error(f"Error calculating nearby variants: {e}")
            columns_not_successfully_added.extend(["nearby_variants", "nearby_variants_count", "has_a_nearby_variant"])

    # CELL
    if mcrs_header_has_merged_values:
        logger.info("Collapsing dataframe")
        mutation_metadata_df, columns_to_explode = collapse_df(
            mutation_metadata_df_exploded,
            columns_to_explode,
            columns_to_explode_extend_values=columns_to_explode_extend_values,
        )
    else:
        mutation_metadata_df = mutation_metadata_df_exploded

    # CELL

    mutation_metadata_df[vcrs_id_column] = mutation_metadata_df[vcrs_id_column].astype(str)

    if columns_to_include == "all" or "vcrs_header_length" in columns_to_include:
        try:
            logger.info("Calculating VCRS header length")
            mutation_metadata_df["vcrs_header_length"] = mutation_metadata_df["vcrs_header"].str.len()
        except Exception as e:
            logger.error(f"Error calculating VCRS header length: {e}")
            columns_not_successfully_added.append("vcrs_header_length")
    if columns_to_include == "all" or "vcrs_sequence_length" in columns_to_include:
        try:
            logger.info("Calculating VCRS sequence length")
            mutation_metadata_df["vcrs_sequence_length"] = mutation_metadata_df["vcrs_sequence"].str.len()
        except Exception as e:
            logger.error(f"Error calculating VCRS sequence length: {e}")
            columns_not_successfully_added.append("vcrs_sequence_length")

    # CELL

    # TODO: calculate if VCRS was optimized - compare MCRS_length to length of unoptimized - exclude subs, and calculate max([2*w + length(added) - length(removed)], [2*w - 1])

    if bowtie_path is not None:
        bowtie2_build = f"{bowtie_path}/bowtie2-build"
        bowtie2 = f"{bowtie_path}/bowtie2"
    else:
        bowtie2_build = "bowtie2-build"
        bowtie2 = "bowtie2"


    # TODO: have more columns_to_include options that allows me to do cdna alone, genome alone, or both combined - currently it is either cdna+genome or nothing
    if columns_to_include == "all" or ("alignment_to_reference" in columns_to_include or "alignment_to_reference_count_total" in columns_to_include or 'alignment_to_reference_count_cdna' in columns_to_include or 'alignment_to_reference_count_genome' in columns_to_include or "substring_alignment_to_reference" in columns_to_include or "substring_alignment_to_reference_count_total" in columns_to_include or "substring_alignment_to_reference_count_cdna" in columns_to_include or "substring_alignment_to_reference_count_genome" in columns_to_include):
        if not is_program_installed(bowtie2):
            bowtie_columns = ["alignment_to_reference", "alignment_to_reference_count_total", "alignment_to_reference_count_cdna", "alignment_to_reference_count_genome", "substring_alignment_to_reference", "substring_alignment_to_reference_count_total", "substring_alignment_to_reference_count_cdna", "substring_alignment_to_reference_count_genome"]
            raise ValueError(f"bowtie2 must be installed to run for the following columns: {bowtie_columns}. Please install bowtie2 or omit these columns")
        try:
            logger.info("Aligning to normal genome and building dlist")
            mutation_metadata_df, sequence_names_set_union_genome_and_cdna = align_to_normal_genome_and_build_dlist(
                mutations=vcrs_fasta,
                vcrs_id_column=vcrs_id_column,
                out_dir_notebook=out,
                reference_out=reference_out_dir,
                dlist_fasta_file_genome_full=dlist_genome_fasta_out,
                dlist_fasta_file_cdna_full=dlist_cdna_fasta_out,
                dlist_fasta_file=dlist_combined_fasta_out,
                dlist_reference_genome_fasta=dlist_reference_genome_fasta,
                dlist_reference_cdna_fasta=dlist_reference_cdna_fasta,
                ref_prefix="index",
                strandedness=vcrs_strandedness,
                threads=threads,
                N_penalty=N_penalty,
                max_ambiguous_vcrs=max_ambiguous_vcrs,
                max_ambiguous_reference=max_ambiguous_reference,
                k=k,
                output_stat_folder=output_stat_folder,
                mutation_metadata_df=mutation_metadata_df,
                bowtie2_build=bowtie2_build,
                bowtie2=bowtie2,
                logger=logger,
            )
        except Exception as e:
            logger.error("Error aligning to normal genome and building alignment_to_reference: %s", e)
            columns_not_successfully_added.extend(
                [
                    "alignment_to_reference",
                    "alignment_to_reference_count_total",
                    "alignment_to_reference_count_cdna",
                    "alignment_to_reference_count_genome",
                    "substring_alignment_to_reference",
                    "substring_alignment_to_reference_count_total",
                    "substring_alignment_to_reference_count_cdna",
                    "substring_alignment_to_reference_count_genome",
                ]
            )

    # CELL
    if make_kat_histogram:
        if not is_program_installed("kat"):
            logger.warning(f"kat must be installed to run make_kat_histogram. Skipping make_kat_histogram")
        else:
            kat_output = f"{out}/kat_output/kat.hist"
            try:
                kat_hist_command = [
                    "kat",
                    "hist",
                    "-m",
                    str(k),
                    "--threads",
                    str(threads),
                    "-o",
                    kat_output,
                    vcrs_fasta,
                ]
                if vcrs_strandedness:
                    # insert as the second element
                    kat_hist_command.insert(2, "--stranded")
                logger.info("Running KAT")
                subprocess.run(kat_hist_command, check=True)
            except Exception as e:
                logger.error(f"Error running KAT: {e}")

            if os.path.exists(kat_output):
                plot_kat_histogram(kat_output)

    # CELL

    if columns_to_include == "all" or ("pseudoaligned_to_reference" in columns_to_include or "pseudoaligned_to_reference_despite_not_truly_aligning" in columns_to_include):
        if not sequence_names_set_union_genome_and_cdna:
            sequence_names_set_union_genome_and_cdna = set()
            column_name = "pseudoaligned_to_reference"
        else:
            column_name = "pseudoaligned_to_reference_despite_not_truly_aligning"

        ref_folder_kb = f"{reference_out_dir}/kb_index_for_mcrs_pseudoalignment_to_reference_genome"

        try:
            logger.info("Getting VCRSs that pseudoalign but aren't dlisted")
            mutation_metadata_df = get_mcrss_that_pseudoalign_but_arent_dlisted(
                mutation_metadata_df=mutation_metadata_df,
                vcrs_id_column=vcrs_id_column,
                mcrs_fa=vcrs_fasta,
                sequence_names_set=sequence_names_set_union_genome_and_cdna,
                human_reference_genome_fa=dlist_reference_genome_fasta,
                human_reference_gtf=dlist_reference_gtf,
                out_dir_notebook=out,
                ref_folder_kb=ref_folder_kb,
                header_column_name=vcrs_id_column,
                additional_kb_extract_filtering_workflow="nac",
                k=k,
                threads=threads,
                strandedness=vcrs_strandedness,
                column_name=column_name,
            )
        except Exception as e:
            logger.error(f"Error getting VCRSs that pseudoalign but aren't dlisted: {e}")
            columns_not_successfully_added.append(column_name)

    # CELL

    if columns_to_include == "all" or ("number_of_kmers_with_overlap_to_other_VCRSs" in columns_to_include or "number_of_other_VCRSs_with_overlapping_kmers" in columns_to_include or 'VCRSs_with_overlapping_kmers' in columns_to_include or "kmer_overlap_with_other_VCRSs" in columns_to_include):
        try:
            logger.info("Calculating overlap between VCRS items")
            df_overlap_stat_file = f"{output_stat_folder}/df_overlap_stat.txt"
            df_overlap = get_df_overlap(
                vcrs_fasta,
                out_dir_notebook=out,
                k=k,
                strandedness=vcrs_strandedness,
                vcrs_id_column=vcrs_id_column,
                output_text_file=df_overlap_stat_file,
                output_plot_folder=output_plot_folder,
            )

            mutation_metadata_df = mutation_metadata_df.merge(df_overlap, on=vcrs_id_column, how="left")
            mutation_metadata_df["kmer_overlap_with_other_VCRSs"] = mutation_metadata_df["number_of_kmers_with_overlap_to_other_VCRSs"].astype(bool)
            mutation_metadata_df["kmer_overlap_with_other_VCRSs"] = mutation_metadata_df["number_of_kmers_with_overlap_to_other_VCRSs"].notna() & mutation_metadata_df["number_of_kmers_with_overlap_to_other_VCRSs"].astype(bool)
        except Exception as e:
            logger.error(f"Error calculating overlap between VCRS items: {e}")
            columns_not_successfully_added.extend(
                [
                    "number_of_kmers_with_overlap_to_other_VCRSs",
                    "number_of_other_VCRSs_with_overlapping_kmers",
                    "VCRSs_with_overlapping_kmers",
                    "kmer_overlap_with_other_VCRSs",
                ]
            )

    # CELL

    # Applying the function to the DataFrame
    if columns_to_include == "all" or ("longest_homopolymer_length" in columns_to_include or "longest_homopolymer" in columns_to_include):
        try:
            logger.info("Calculating longest homopolymer")
            (
                mutation_metadata_df["longest_homopolymer_length"],
                mutation_metadata_df["longest_homopolymer"],
            ) = zip(*mutation_metadata_df["vcrs_sequence"].apply(lambda x: (longest_homopolymer(x) if pd.notna(x) else (np.nan, np.nan))))
        except Exception as e:
            logger.error(f"Error calculating longest homopolymer: {e}")
            columns_not_successfully_added.extend(["longest_homopolymer_length", "longest_homopolymer"])

    # CELL

    if columns_to_include == "all" or ("num_distinct_triplets" in columns_to_include or "num_total_triplets" in columns_to_include or "triplet_complexity" in columns_to_include):
        logger.info("Calculating triplet stats")
        try:
            (
                mutation_metadata_df["num_distinct_triplets"],
                mutation_metadata_df["num_total_triplets"],
                mutation_metadata_df["triplet_complexity"],
            ) = zip(*mutation_metadata_df["vcrs_sequence"].apply(lambda x: (triplet_stats(x) if pd.notna(x) else (np.nan, np.nan, np.nan))))

            output_file_longest_homopolymer = f"{output_plot_folder}/longest_homopolymer.png"
            plot_histogram_of_nearby_mutations_7_5(
                mutation_metadata_df,
                "longest_homopolymer_length",
                bins=20,
                output_file=output_file_longest_homopolymer,
            )

            output_file_triplet_complexity = f"{output_plot_folder}/triplet_complexity.png"
            plot_histogram_of_nearby_mutations_7_5(
                mutation_metadata_df,
                "triplet_complexity",
                bins=20,
                output_file=output_file_triplet_complexity,
            )

        except Exception as e:
            logger.error(f"Error calculating triplet stats: {e}")
            columns_not_successfully_added.extend(["num_distinct_triplets", "num_total_triplets", "triplet_complexity"])

    # CELL
    # add metadata: VCRS mutation type
    if columns_to_include == "all" or "vcrs_mutation_type" in columns_to_include:
        try:
            logger.info("Adding VCRS mutation type")
            mutation_metadata_df = add_mcrs_mutation_type(mutation_metadata_df, var_column="vcrs_header")
        except Exception as e:
            logger.error(f"Error adding VCRS mutation type: {e}")
            columns_not_successfully_added.append("vcrs_mutation_type")

    # CELL

    # Add metadata: ';' in vcrs_header
    if columns_to_include == "all" or ("merged_variants_in_VCRS_entry" in columns_to_include or "number_of_variants_in_VCRS_entry" in columns_to_include):
        try:
            logger.info("Adding concatenated header info")
            mutation_metadata_df["merged_variants_in_VCRS_entry"] = mutation_metadata_df["vcrs_header"].str.contains(";")
            mutation_metadata_df["number_of_variants_in_VCRS_entry"] = mutation_metadata_df["vcrs_header"].str.count(";") + 1
        except Exception as e:
            logger.error(f"Error adding concatenated headers in VCRS: {e}")
            columns_not_successfully_added.extend(["merged_variants_in_VCRS_entry", "number_of_variants_in_VCRS_entry"])

    # CELL

    # Add metadata: vcrs_sequence_rc
    if columns_to_include == "all" or "vcrs_sequence_rc" in columns_to_include:
        try:
            logger.info("Adding VCRS reverse complement")
            mutation_metadata_df["vcrs_sequence_rc"] = mutation_metadata_df["vcrs_sequence"].apply(reverse_complement)
        except Exception as e:
            logger.error(f"Error adding VCRS reverse complement: {e}")
            columns_not_successfully_added.append("vcrs_sequence_rc")

    # CELL

    # Add metadata: VCRS substring and superstring (forward and rc)
    if columns_to_include == "all" or ("VCRSs_for_which_this_VCRS_is_a_substring" in columns_to_include or "VCRSs_for_which_this_VCRS_is_a_superstring" in columns_to_include or "VCRS_is_a_substring_of_another_VCRS" in columns_to_include or "VCRS_is_a_superstring_of_another_VCRS" in columns_to_include):
        if not is_program_installed(bowtie2):
            bowtie_columns = ["VCRSs_for_which_this_VCRS_is_a_substring", "VCRSs_for_which_this_VCRS_is_a_superstring", "VCRS_is_a_substring_of_another_VCRS", "VCRS_is_a_superstring_of_another_VCRS"]
            raise ValueError(f"bowtie2 must be installed to run for the following columns: {bowtie_columns}. Please install bowtie2 or omit these columns")
        mcrs_to_mcrs_bowtie_folder = f"{out}/bowtie_vcrs_to_vcrs"
        mcrs_sam_file = f"{mcrs_to_mcrs_bowtie_folder}/mutant_reads_to_vcrs_index.sam"
        substring_output_stat_file = f"{output_stat_folder}/substring_output_stat.txt"

        try:
            logger.info("Creating VCRS to self headers")
            substring_to_superstring_df, superstring_to_substring_df = create_df_of_mcrs_to_self_headers(
                mcrs_sam_file=mcrs_sam_file,
                mcrs_fa=vcrs_fasta,
                bowtie_mcrs_reference_folder=mcrs_to_mcrs_bowtie_folder,
                bowtie_path=bowtie_path,
                threads=threads,
                strandedness=vcrs_strandedness,
                vcrs_id_column=vcrs_id_column,
                output_stat_file=substring_output_stat_file,
            )

            mutation_metadata_df[vcrs_id_column] = mutation_metadata_df[vcrs_id_column].astype(str)
            mutation_metadata_df = mutation_metadata_df.merge(substring_to_superstring_df, on=vcrs_id_column, how="left")
            mutation_metadata_df = mutation_metadata_df.merge(superstring_to_substring_df, on=vcrs_id_column, how="left")

            mutation_metadata_df["VCRS_is_a_substring_of_another_VCRS"] = mutation_metadata_df["VCRS_is_a_substring_of_another_VCRS"].fillna(False).astype(bool)
            mutation_metadata_df["VCRS_is_a_superstring_of_another_VCRS"] = mutation_metadata_df["VCRS_is_a_superstring_of_another_VCRS"].fillna(False).astype(bool)

        except Exception as e:
            logger.error(f"Error creating VCRS to self headers: {e}")
            columns_not_successfully_added.extend(
                [
                    "VCRSs_for_which_this_VCRS_is_a_substring",
                    "VCRSs_for_which_this_VCRS_is_a_superstring",
                    "VCRS_is_a_substring_of_another_VCRS",
                    "VCRS_is_a_superstring_of_another_VCRS",
                ]
            )

    # CELL
    logger.info("sorting variant metadata by VCRS id")
    mutation_metadata_df = mutation_metadata_df.sort_values(by=vcrs_id_column).reset_index(drop=True)

    logger.info("Saving variant metadata")
    mutation_metadata_df.to_csv(variants_updated_vk_info_csv_out, index=False)

    # CELL

    if save_variants_updated_exploded_vk_info_csv:
        logger.info("Saving exploded variant metadata")
        mutation_metadata_df_exploded = explode_df(mutation_metadata_df, columns_to_explode)
        mutation_metadata_df_exploded.to_csv(variants_updated_exploded_vk_info_csv_out, index=False)

    if verbose:
        logger.info(f"Saved variant metadata to {variants_updated_vk_info_csv_out}")
        logger.info(f"Columns: {mutation_metadata_df.columns}")
        logger.info(f"Columns successfully added: {set(mutation_metadata_df.columns.tolist()) - set(columns_original)}")
        logger.info(f"Columns not successfully added: {columns_not_successfully_added}")

    # Report time
    report_time_elapsed(start_time, logger=logger, verbose=verbose, function_name="info")
