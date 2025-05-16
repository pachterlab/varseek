"""varseek info and specific helper functions."""

# CELL
import logging
import shutil
import os
import subprocess
import time
from collections import OrderedDict
from pathlib import Path

import numpy as np
import pandas as pd
import pyfastx
from tqdm import tqdm

from varseek.constants import HGVS_pattern, mutation_pattern
from varseek.utils import (
    add_mutation_information,
    add_variant_type,
    add_vcrs_variant_type,
    align_to_normal_genome_and_build_dlist,
    calculate_nearby_mutations,
    calculate_total_gene_info,
    check_file_path_is_string_with_valid_extension,
    collapse_df,
    compare_cdna_and_genome,
    compute_distance_to_closest_splice_junction,
    create_df_of_vcrs_to_self_headers,
    download_ensembl_reference_files,
    download_t2t_reference_files,
    explode_df,
    fasta_summary_stats,
    get_df_overlap,
    get_vcrss_that_pseudoalign_but_arent_dlisted,
    identify_variant_source,
    is_program_installed,
    is_valid_int,
    longest_homopolymer,
    make_function_parameter_to_value_dict,
    make_mapping_dict,
    plot_histogram_of_nearby_mutations_7_5,
    plot_kat_histogram,
    get_varseek_dry_run,
    report_time_elapsed,
    reverse_complement,
    safe_literal_eval,
    save_params_to_config_file,
    save_run_info,
    set_varseek_logging_level_and_filehandler,
    set_up_logger,
    splitext_custom,
    swap_ids_for_headers_in_fasta,
    triplet_stats,
    count_chunks,
    save_fasta_chunk,
    save_csv_chunk,
    determine_write_mode
)

tqdm.pandas()
logger = logging.getLogger(__name__)
logger = set_up_logger(logger, logging_level="INFO", save_logs=False, log_dir=None)
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


def print_list_columns():
    print("Available values for `columns_to_include`:")
    for col, description_and_utilized_parameters_tuple in columns_to_include_possible_values.items():
        print(f"- {col}:\n    {description_and_utilized_parameters_tuple[0]}\n    {description_and_utilized_parameters_tuple[1]}\n")


def validate_input_info(params_dict):
    # Directories
    if not isinstance(params_dict.get("input_dir"), (str, Path)):  # I will enforce that input_dir exists later, as otherwise it will throw an error when I call this through vk ref before vk build's out exists
        raise ValueError(f"Invalid value for input_dir: {params_dict.get('input_dir')}")
    if params_dict.get("out") is not None and not isinstance(params_dict.get("out"), (str, Path)):
        raise ValueError(f"Invalid value for out: {params_dict.get('out')}")
    if params_dict.get("reference_out_dir") and not isinstance(params_dict.get("reference_out_dir"), (str, Path)):
        raise ValueError(f"Invalid value for reference_out_dir: {params_dict.get('reference_out_dir')}")

    gtf = params_dict.get("gtf")  # gtf gets special treatment because it can be a bool - and check for {"True", "False"} because of CLI passing
    if gtf is not None and not isinstance(gtf, bool) and gtf not in {"True", "False"} and not gtf.lower().endswith(("gtf", "gtf.zip", "gtf.gz")):
        raise ValueError(f"Invalid value for gtf: {gtf}. Expected gtf filepath string, bool, or None.")

    # file paths
    for param_name, file_type in {
        "vcrs_fasta": "fasta",
        "variants_updated_csv": "csv",
        "id_to_header_csv": "csv",
        "variants_updated_vk_info_csv_out": "csv",
        "variants_updated_exploded_vk_info_csv_out": "csv",
        "dlist_genome_fasta_out": "fasta",
        "dlist_cdna_fasta_out": "fasta",
        "dlist_combined_fasta_out": "fasta",
        "reference_cdna_fasta": "fasta",
        "reference_genome_fasta": "fasta",
    }.items():
        check_file_path_is_string_with_valid_extension(params_dict.get(param_name), param_name, file_type)
    
    # column names
    for column in ["gene_name_column", "variant_source_column", "var_cdna_column", "seq_id_cdna_column", "var_genome_column", "seq_id_genome_column", "var_id_column"]:
        if not isinstance(params_dict.get(column), str) and params_dict.get(column) is not None:
            raise ValueError(f"Invalid column name: {params_dict.get(column)}")
    if params_dict.get("variant_source_column") is None:
        raise ValueError("variant_source_column must be provided.")

    if params_dict.get("var_id_column") is not None and (params_dict.get("variants_updated_csv") is None or params_dict.get("variants") is None):
        raise ValueError("variants_updated_csv or variants must be provided when var_id_column is not None.")

    # columns_to_include
    columns_to_include = params_dict.get("columns_to_include")
    if not isinstance(columns_to_include, (list, set, str, tuple)):
        raise ValueError(f"columns_to_include must be a string or list of strings, got {type(columns_to_include)}")
    if isinstance(columns_to_include, (list, set, tuple)):
        if not all(isinstance(col, str) for col in columns_to_include):
            raise ValueError("All elements in columns_to_include must be strings.")
        if not all(col in columns_to_include_possible_values for col in columns_to_include):
            raise ValueError(f"columns_to_include must be a subset of {columns_to_include_possible_values}. Got {columns_to_include}. Use 'all' to include all columns.")
        
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

    # check if any column requiring dlist_reference_genome_fasta is present, and dlist_reference_genome_fasta is not available
    if any(column in params_dict.get("columns_to_include") for column in columns_that_require_dlist_genome) and not params_dict.get("dlist_reference_source") and not params_dict.get("dlist_reference_genome_fasta"):  # above, I checked that if it was not None, that it was either a valid string (ie in supported_dlist_reference_values) or an existing path - so now, I just need to ensure that it's not None
        raise ValueError(f"Missing dlist_reference_source and dlist_reference_genome_fasta. At least one of these is required for columns: {columns_that_require_dlist_genome}")
    if any(column in params_dict.get("columns_to_include") for column in columns_that_require_dlist_transcriptome) and not params_dict.get("dlist_reference_source") and not params_dict.get("dlist_reference_cdna_fasta"):  # above, I checked that if it was not None, that it was either a valid string (ie in supported_dlist_reference_values) or an existing path - so now, I just need to ensure that it's not None
        raise ValueError(f"Missing dlist_reference_source/dlist_reference_cdna_fasta. At least one of these is required for columns: {columns_that_require_dlist_genome}")
    if any(column in params_dict.get("columns_to_include") for column in columns_that_require_dlist_gtf) and not params_dict.get("dlist_reference_source") and not params_dict.get("dlist_reference_gtf"):  # above, I checked that if it was not None, that it was either a valid string (ie in supported_dlist_reference_values) or an existing path - so now, I just need to ensure that it's not None
        raise ValueError(f"Missing dlist_reference_source/dlist_reference_gtf. At least one of these is required for columns: {columns_that_require_dlist_genome}")

    # integers - optional just means that it's in kwargs
    for param_name, min_value, optional_value in [
        ("w", 1, True),
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
    if int(k) % 2 == 0 or int(k) > 63:
        logger.warning("If running a workflow with vk ref or kb ref, k should be an odd number between 1 and 63. Got k=%s.", k)

    # boolean
    for param_name in ["save_variants_updated_exploded_vk_info_csv", "make_pyfastx_summary_file", "make_kat_histogram", "dry_run", "list_columns", "overwrite"]:
        param_value = params_dict.get(param_name)
        if not isinstance(param_value, bool):
            raise ValueError(f"{param_name} must be a boolean. Got {param_value} of type {type(param_value)}.")

    # Optional boolean
    if params_dict.get("vcrs_strandedness") is not None and not isinstance(params_dict.get("vcrs_strandedness"), bool):
        raise ValueError(f"vcrs_strandedness must be a boolean. Got {params_dict.get('vcrs_strandedness')} of type {type(params_dict.get('vcrs_strandedness'))}.")

    if params_dict.get("pseudoalignment_workflow") and params_dict.get("pseudoalignment_workflow") not in {"standard", "nac"}:
        raise ValueError(f"Invalid value for pseudoalignment_workflow: {params_dict.get('pseudoalignment_workflow')}. Expected 'standard' or 'nac'.")


supported_dlist_reference_values = {"t2t", "grch37", "grch38"}
columns_that_require_dlist_genome = ("alignment_to_reference", "alignment_to_reference_genome", "alignment_to_reference_count_total", "alignment_to_reference_count_genome", "substring_alignment_to_reference", "substring_alignment_to_reference_genome", "substring_alignment_to_reference_count_total", "substring_alignment_to_reference_count_genome", "pseudoaligned_to_reference", "pseudoaligned_to_reference_despite_not_truly_aligning")
columns_that_require_dlist_transcriptome = ("alignment_to_reference", "alignment_to_reference_cdna", "alignment_to_reference_count_total", "alignment_to_reference_count_cdna", "substring_alignment_to_reference", "substring_alignment_to_reference_cdna", "substring_alignment_to_reference_count_total", "substring_alignment_to_reference_count_cdna")
columns_that_require_dlist_gtf = ("pseudoaligned_to_reference", "pseudoaligned_to_reference_despite_not_truly_aligning")

bowtie_columns_dlist = ["alignment_to_reference", "alignment_to_reference_cdna", "alignment_to_reference_genome", "alignment_to_reference_count_total", "alignment_to_reference_count_cdna", "alignment_to_reference_count_genome", "substring_alignment_to_reference", "substring_alignment_to_reference_cdna", "substring_alignment_to_reference_genome", "substring_alignment_to_reference_count_total", "substring_alignment_to_reference_count_cdna", "substring_alignment_to_reference_count_genome"]
bowtie_columns_vcrs_to_vcrs = ["VCRSs_for_which_this_VCRS_is_a_substring", "VCRSs_for_which_this_VCRS_is_a_superstring", "VCRS_is_a_substring_of_another_VCRS", "VCRS_is_a_superstring_of_another_VCRS"]

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
        ("alignment_to_reference", ("States whether a VCRS contains any k-mer that aligns anywhere in the reference genome or transcriptome (requires bowtie2)", ["k", "max_ambiguous_vcrs", "max_ambiguous_reference", "dlist_reference_genome_fasta", "dlist_reference_cdna_fasta", "dlist_genome_fasta_out", "dlist_cdna_fasta_out", "dlist_combined_fasta_out", "threads", "vcrs_strandedness"])),
        ("alignment_to_reference_cdna", ("States whether a VCRS contains any k-mer that aligns anywhere in the reference transcriptome (requires bowtie2)", ["k", "max_ambiguous_vcrs", "max_ambiguous_reference", "dlist_reference_cdna_fasta", "dlist_cdna_fasta_out", "dlist_combined_fasta_out", "threads", "vcrs_strandedness"])),
        ("alignment_to_reference_genome", ("States whether a VCRS contains any k-mer that aligns anywhere in the reference genome (requires bowtie2)", ["k", "max_ambiguous_vcrs", "max_ambiguous_reference", "dlist_reference_genome_fasta", "dlist_genome_fasta_out", "dlist_combined_fasta_out", "threads", "vcrs_strandedness"])),
        ("alignment_to_reference_count_total", ("The total number of times that all k-mers within a VCRS match up to any k-mer in the genome and transcriptome specified by `dlist_reference_source` (requires bowtie2)", ["k", "max_ambiguous_vcrs", "max_ambiguous_reference", "dlist_reference_genome_fasta", "dlist_reference_cdna_fasta", "dlist_genome_fasta_out", "dlist_cdna_fasta_out", "dlist_combined_fasta_out", "threads", "vcrs_strandedness"])),
        ("alignment_to_reference_count_cdna", ("The total number of times that all k-mers within a VCRS match up to any k-mer in the cDNA specified by `dlist_reference_source` (requires bowtie2)", ["k", "max_ambiguous_vcrs", "max_ambiguous_reference", "dlist_reference_cdna_fasta", "dlist_cdna_fasta_out", "dlist_combined_fasta_out", "threads", "vcrs_strandedness"])),
        ("alignment_to_reference_count_genome", ("The total number of times that all k-mers within a VCRS match up to any k-mer in the genome specified by `dlist_reference_source` (requires bowtie2)", ["k", "max_ambiguous_vcrs", "max_ambiguous_reference", "dlist_reference_genome_fasta", "dlist_genome_fasta_out", "dlist_combined_fasta_out", "threads", "vcrs_strandedness"])),
        ("substring_alignment_to_reference", ("States whether a VCRS contains any k-mer that is a substring alignment to anywhere in the reference genome or transcriptome (requires bowtie2)", ["k", "max_ambiguous_vcrs", "max_ambiguous_reference", "dlist_reference_genome_fasta", "dlist_reference_cdna_fasta", "dlist_genome_fasta_out", "dlist_cdna_fasta_out", "dlist_combined_fasta_out", "threads", "vcrs_strandedness"])),
        ("substring_alignment_to_reference_cdna", ("States whether a VCRS contains any k-mer that is a substring alignment to anywhere in the reference transcriptome (requires bowtie2)", ["k", "max_ambiguous_vcrs", "max_ambiguous_reference", "dlist_reference_cdna_fasta", "dlist_cdna_fasta_out", "dlist_combined_fasta_out", "threads", "vcrs_strandedness"])),
        ("substring_alignment_to_reference_genome", ("States whether a VCRS contains any k-mer that is a substring alignment to anywhere in the reference genome (requires bowtie2)", ["k", "max_ambiguous_vcrs", "max_ambiguous_reference", "dlist_reference_genome_fasta", "dlist_genome_fasta_out", "dlist_combined_fasta_out", "threads", "vcrs_strandedness"])),
        ("substring_alignment_to_reference_count_total", ("Number of substring matches to normal human reference (requires bowtie2)", ["k", "max_ambiguous_vcrs", "max_ambiguous_reference", "dlist_reference_genome_fasta", "dlist_reference_cdna_fasta", "dlist_genome_fasta_out", "dlist_cdna_fasta_out", "dlist_combined_fasta_out", "threads", "vcrs_strandedness"])),
        ("substring_alignment_to_reference_count_cdna", ("Number of substring matches to normal human reference cDNA (requires bowtie2)", ["k", "max_ambiguous_vcrs", "max_ambiguous_reference", "dlist_reference_cdna_fasta", "dlist_cdna_fasta_out", "dlist_combined_fasta_out", "threads", "vcrs_strandedness"])),
        ("substring_alignment_to_reference_count_genome", ("Number of substring matches to normal human reference genome (requires bowtie2)", ["k", "max_ambiguous_vcrs", "max_ambiguous_reference", "dlist_reference_genome_fasta", "dlist_genome_fasta_out", "dlist_combined_fasta_out", "threads", "vcrs_strandedness"])),
        ("pseudoaligned_to_reference", ("Pseudoaligned to human reference", ["k", "dlist_reference_genome_fasta", "dlist_reference_gtf", "threads", "vcrs_strandedness"])),
        ("pseudoaligned_to_reference_despite_not_truly_aligning", ("Pseudoaligned to human reference despite not truly aligning", ["k", "dlist_reference_genome_fasta", "dlist_reference_gtf", "threads", "vcrs_strandedness"])),
        ("number_of_kmers_with_overlap_to_other_VCRSs", ("Number of k-mers with overlap to other VCRS items in VCRS reference (MEMORY-INTENSIVE)", ["k", "vcrs_strandedness"])),
        ("number_of_other_VCRSs_with_overlapping_kmers", ("Number of VCRS items with overlapping k-mers in VCRS reference (MEMORY-INTENSIVE)", ["k", "vcrs_strandedness"])),
        ("VCRSs_with_overlapping_kmers", ("VCRS items with overlapping k-mers in VCRS reference (requires bowtie2) (MEMORY-INTENSIVE)", ["k", "threads"])),
        ("kmer_overlap_with_other_VCRSs", ("K-mer overlap in VCRS reference (a boolean of `number_of_kmers_with_overlap_to_other_VCRSs`) (MEMORY-INTENSIVE)", [])),
        ("longest_homopolymer_length", ("Longest homopolymer length", [])),
        ("longest_homopolymer", ("Longest homopolymer", [])),
        ("num_distinct_triplets", ("Number of distinct triplets", [])),
        ("num_total_triplets", ("Number of total triplets", [])),
        ("triplet_complexity", ("Triplet complexity", [])),
        ("vcrs_variant_type", ("VCRS mutation type", [])),
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
@report_time_elapsed
def info(
    input_dir,
    columns_to_include=("number_of_variants_in_this_gene_total", "alignment_to_reference", "pseudoaligned_to_reference_despite_not_truly_aligning", "triplet_complexity"),
    k=59,
    max_ambiguous_vcrs=0,
    max_ambiguous_reference=0,
    dlist_reference_source=None,
    dlist_reference_ensembl_release=111,
    dlist_reference_type=None,
    vcrs_fasta=None,
    variants_updated_csv=None,
    id_to_header_csv=None,  # if none then assume no swapping occurred
    gtf=None,
    dlist_reference_genome_fasta=None,
    dlist_reference_cdna_fasta=None,
    dlist_reference_gtf=None,
    var_id_column=None,
    gene_name_column=None,
    variant_source_column="variant_source",  # if input df has concatenated cdna and header VCRS's, then I want to know whether it came from cdna or genome
    var_cdna_column=None,
    seq_id_cdna_column=None,
    var_genome_column=None,
    seq_id_genome_column=None,
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
    chunksize=None,
    dry_run=False,
    list_columns=False,
    overwrite=False,
    threads=2,
    logging_level=None,
    save_logs=False,
    log_out_dir=None,
    verbose=False,
    **kwargs,
):
    """
    Takes in the input directory containing with the VCRS fasta file generated from varseek build, and returns a dataframe with additional columns containing information about the variants.

    # Required input arguments:
    - input_dir     (str) Path to the directory containing the input files. Corresponds to `out` in the varseek build function.

    # Additional Parameters
    - columns_to_include                 (str or list[str]) List of columns to include in the output dataframe. Default: ("number_of_variants_in_this_gene_total", "alignment_to_reference", "pseudoaligned_to_reference_despite_not_truly_aligning", "num_distinct_triplets"). See all possible values and their description by setting list_columns=True (python) or --list_columns (command line).
    - k                                  (int) Length of the k-mers utilized by kallisto | bustools. Only used by the following columns: 'nearby_variants', 'nearby_variants_count', 'has_a_nearby_variant', 'alignment_to_reference', 'alignment_to_reference_cdna', 'alignment_to_reference_genome', 'alignment_to_reference_count_total', 'alignment_to_reference_count_cdna', 'alignment_to_reference_count_genome', 'substring_alignment_to_reference', 'substring_alignment_to_reference_cdna', 'substring_alignment_to_reference_genome', 'substring_alignment_to_reference_count_total', 'substring_alignment_to_reference_count_cdna',  'substring_alignment_to_reference_count_genome', 'pseudoaligned_to_reference', 'pseudoaligned_to_reference_despite_not_truly_aligning', 'number_of_kmers_with_overlap_to_other_VCRSs', 'number_of_other_VCRSs_with_overlapping_kmers', 'VCRSs_with_overlapping_kmers', 'kmer_overlap_with_other_VCRSs'; and when make_kat_histogram==True. Default: 59.
    - max_ambiguous_vcrs                 (int) Maximum number of 'N' characters allowed in the VCRS when considering alignment to the reference genome/transcriptome. Only used by the following columns: 'alignment_to_reference', 'alignment_to_reference_count_total', 'alignment_to_reference_count_cdna', 'alignment_to_reference_count_genome', 'substring_alignment_to_reference', 'substring_alignment_to_reference_count_total', 'substring_alignment_to_reference_count_cdna',  'substring_alignment_to_reference_count_genome'. Default: 0.
    - max_ambiguous_reference            (int) Maximum number of 'N' characters allowed in the aligned reference genome portion when considering alignment to the reference genome/transcriptome. Only used by the following columns: 'alignment_to_reference', 'alignment_to_reference_cdna', 'alignment_to_reference_genome', 'alignment_to_reference_count_total', 'alignment_to_reference_count_cdna', 'alignment_to_reference_count_genome', 'substring_alignment_to_reference', 'substring_alignment_to_reference_cdna', 'substring_alignment_to_reference_genome', 'substring_alignment_to_reference_count_total', 'substring_alignment_to_reference_count_cdna',  'substring_alignment_to_reference_count_genome'. Default: 0.
    - dlist_reference_source (str or None) Specifies which reference to use during alignment of VCRS k-mers to the reference genome/transcriptome and any possible d-list construction. However, no d-list is used during the creation of the VCRS reference index unless `dlist` is not None. This can refer to the same genome version as used by the "sequences" argument, but need not be. The purpose of this genome is simply to provide an accurate and comprehensive reference genome/transcriptome to determine which k-mers from the VCRSs overlap with the reference. Will look for files in `reference_out_dir`, and will download in this directory if necessary files do not exist. Only used by the following columns: 'alignment_to_reference', 'alignment_to_reference_cdna', 'alignment_to_reference_genome', 'alignment_to_reference_count_total', 'alignment_to_reference_count_cdna', 'alignment_to_reference_count_genome', 'substring_alignment_to_reference', 'substring_alignment_to_reference_cdna', 'substring_alignment_to_reference_genome', 'substring_alignment_to_reference_count_total', 'substring_alignment_to_reference_count_cdna',  'substring_alignment_to_reference_count_genome', 'pseudoaligned_to_reference', 'pseudoaligned_to_reference_despite_not_truly_aligning'. Possible values are {supported_dlist_reference_values}. Ignored if values for `dlist_reference_genome_fasta`, `dlist_reference_cdna_fasta`, and `dlist_reference_gtf` are provided. Default: None. Possible values:
        - "t2t" - Telomere to telomere: https://www.ncbi.nlm.nih.gov/datasets/genome/GCF_009914755.1/. Directory: {reference_out_dir}/t2t.
        - "grch38" - Ensembl GRCh38: https://useast.ensembl.org/Homo_sapiens/Info/Annotation. Directory: {reference_out_dir}/ensembl_grch38_release{dlist_reference_ensembl_release}.
        - "grch37" - Ensembl GRCh37: http://useast.ensembl.org/info/website/tutorials/grch37.html. Directory: {reference_out_dir}/ensembl_grch37_release{dlist_reference_ensembl_release}.
        If wanting to provide a reference genome outside of those above supported for automatic download, then please provide existing file paths for the parameters `dlist_reference_genome_fasta`, `dlist_reference_cdna_fasta`, and/or `dlist_reference_gtf`.
    - dlist_reference_ensembl_release    (int) Ensembl release number for the d-list reference genome and transcriptome if dlist_reference_source in {"grch37", "grch38"}. Only used by the following columns: 'alignment_to_reference', 'alignment_to_reference_cdna', 'alignment_to_reference_genome', 'alignment_to_reference_count_total', 'alignment_to_reference_count_cdna', 'alignment_to_reference_count_genome', 'substring_alignment_to_reference', 'substring_alignment_to_reference_cdna', 'substring_alignment_to_reference_genome', 'substring_alignment_to_reference_count_total', 'substring_alignment_to_reference_count_cdna',  'substring_alignment_to_reference_count_genome', 'pseudoaligned_to_reference', 'pseudoaligned_to_reference_despite_not_truly_aligning'. Possible values are {supported_dlist_reference_values}. Ignored if values for `dlist_reference_genome_fasta`, `dlist_reference_cdna_fasta`, and `dlist_reference_gtf` are provided. Default: 111.
    - dlist_reference_type              (str) Type of d-list reference to use. Only used by the following columns: 'alignment_to_reference', 'alignment_to_reference_cdna', 'alignment_to_reference_genome', 'alignment_to_reference_count_total', 'alignment_to_reference_count_cdna', 'alignment_to_reference_count_genome', 'substring_alignment_to_reference', 'substring_alignment_to_reference_cdna', 'substring_alignment_to_reference_genome', 'substring_alignment_to_reference_count_total', 'substring_alignment_to_reference_count_cdna',  'substring_alignment_to_reference_count_genome', 'pseudoaligned_to_reference', 'pseudoaligned_to_reference_despite_not_truly_aligning'. Options are "genome", "transcriptome", "combined" or None. Default: None (will utilize dlist_reference_genome_fasta and dlist_reference_cdna_fasta; if dlist_reference_source is provided, then will use combined).

    # Optional input file paths: (only needed if changing/customizing file names or locations):
    - vcrs_fasta                         (str) Path to the VCRS fasta file generated from varseek build. Corresponds to `vcrs_fasta_out` in the varseek build function. Only needed if the original file was changed or renamed. Default: None (will find it in `input_dir`).
    - variants_updated_csv               (str) Path to the updated dataframe containing the VCRS headers and sequences. Corresponds to `variants_updated_csv_out` in the varseek build function. Only needed if the original file was changed or renamed. Default: None (will find it in `input_dir` if it exists).
    - id_to_header_csv                   (str) Path to the csv file containing the mapping of IDs to headers generated from varseek build corresponding to vcrs_fasta. Corresponds to `id_to_header_csv_out` in the varseek build function. Only needed if the original file was changed or renamed. Default: None (will find it in `input_dir` if it exists).
    - gtf                                (str) Path to the GTF file containing the gene annotations for the reference genome. Corresponds to `gtf` in the varseek build function. Must align to genome coordinates used in the annotation of variants. Only used by the following columns: 'distance_to_nearest_splice_junction'. Default: None.
    - dlist_reference_genome_fasta       (str) Path to the reference genome fasta file for the d-list. Only used by the following columns: 'alignment_to_reference', 'alignment_to_reference_cdna', 'alignment_to_reference_genome', 'alignment_to_reference_count_total', 'alignment_to_reference_count_cdna', 'alignment_to_reference_count_genome', 'substring_alignment_to_reference', 'substring_alignment_to_reference_cdna', 'substring_alignment_to_reference_genome', 'substring_alignment_to_reference_count_total', 'substring_alignment_to_reference_count_cdna',  'substring_alignment_to_reference_count_genome', 'pseudoaligned_to_reference', 'pseudoaligned_to_reference_despite_not_truly_aligning'. Default: `dlist_reference_source`.
    - dlist_reference_cdna_fasta         (str) Path to the reference cDNA fasta file for the d-list. Only used by the following columns: 'alignment_to_reference', 'alignment_to_reference_cdna', 'alignment_to_reference_genome', 'alignment_to_reference_count_total', 'alignment_to_reference_count_cdna', 'alignment_to_reference_count_genome', 'substring_alignment_to_reference', 'substring_alignment_to_reference_cdna', 'substring_alignment_to_reference_genome', 'substring_alignment_to_reference_count_total', 'substring_alignment_to_reference_count_cdna',  'substring_alignment_to_reference_count_genome'. Default: `dlist_reference_source`.
    - dlist_reference_gtf                (str) Path to the GTF file containing the gene annotations for the reference genome. Only used by the following columns: 'pseudoaligned_to_reference', 'pseudoaligned_to_reference_despite_not_truly_aligning'. Default: `dlist_reference_source`.

    # Column names in variants_updated_csv:
    - var_id_column                      (str) Name of the column containing the IDs of each variant in 'variants'. Matches `var_id_column` from vk build. Only provide if also provided in vk build. If var_id_column is provided, must also provide `variants_updated_csv` (or it must exist in `input_dir` from vk build) or `variants` (same as in vk build). Default: None.
    - gene_name_column                   (str) Name of the column containing the gene names in `variants_updated_csv`. Only used if `variants_updated_csv` exists (i.e., was generated from varseek build). Default: None.
    - vcrs_sequence_column               (str) Name of the column containing the VCRS sequences in `variants_updated_csv`. Only used if `variants_updated_csv` exists (i.e., was generated from varseek build). Default: 'mutant_sequence'.
    - variant_source_column              (str) Name of the column containing the source of the VCRS (cdna or genome) in `variants_updated_csv`. Only used if `variants_updated_csv` exists (i.e., was generated from varseek build). Default: 'variant_source'.
    - var_cdna_column                    (str) Name of the column containing the cDNA variants in `variants_updated_csv`. Only used if `variants_updated_csv` exists (i.e., was generated from varseek build) and contains information regarding both genome and cDNA notation (essential if running spliced + unspliced workflow, optional otherwise). Default: None.
    - seq_id_cdna_column                 (str) Name of the column containing the cDNA sequence IDs in `variants_updated_csv`. Only used if `variants_updated_csv` exists (i.e., was generated from varseek build) and contains information regarding both genome and cDNA notation (essential if running spliced + unspliced workflow, optional otherwise). Default: None.
    - var_genome_column                  (str) Name of the column containing the genome variants in `variants_updated_csv`. Only used if `variants_updated_csv` exists (i.e., was generated from varseek build) and contains information regarding both genome and cDNA notation (essential if running spliced + unspliced workflow, optional otherwise). Default: None.
    - seq_id_genome_column               (str) Name of the column containing the genome sequence IDs in `variants_updated_csv`. Only used if `variants_updated_csv` exists (i.e., was generated from varseek build) and contains information regarding both genome and cDNA notation (essential if running spliced + unspliced workflow, optional otherwise). Default: None.

    # Output file paths:
    - out                                (str) Path to the directory where the output files will be saved. Default: `input_dir`.
    - reference_out_dir                  (str) Path to the directory where the reference files will be saved. Default: `out`/reference.
    - variants_updated_vk_info_csv_out  (str) Path to the output csv file containing the updated dataframe with the additional columns. Default: `out`/variants_updated_vk_info.csv.
    - variants_updated_exploded_vk_info_csv_out (str) Path to the output csv file containing the exploded dataframe with the additional columns. Default: `out`/variants_updated_exploded_vk_info.csv.
    - dlist_genome_fasta_out             (str) Path to the output fasta file containing the d-list sequences for the genome-based alignmed. Only used by the following columns: 'alignment_to_reference', 'alignment_to_reference_cdna', 'alignment_to_reference_genome', 'alignment_to_reference_count_total', 'alignment_to_reference_count_cdna', 'alignment_to_reference_count_genome', 'substring_alignment_to_reference', 'substring_alignment_to_reference_cdna', 'substring_alignment_to_reference_genome', 'substring_alignment_to_reference_count_total', 'substring_alignment_to_reference_count_cdna',  'substring_alignment_to_reference_count_genome'. Default: `out`/dlist_genome.fa.
    - dlist_cdna_fasta_out               (str) Path to the output fasta file containing the d-list sequences for the cDNA-based alignmed. Only used by the following columns: 'alignment_to_reference', 'alignment_to_reference_cdna', 'alignment_to_reference_genome', 'alignment_to_reference_count_total', 'alignment_to_reference_count_cdna', 'alignment_to_reference_count_genome', 'substring_alignment_to_reference', 'substring_alignment_to_reference_cdna', 'substring_alignment_to_reference_genome', 'substring_alignment_to_reference_count_total', 'substring_alignment_to_reference_count_cdna',  'substring_alignment_to_reference_count_genome'.Default: `out`/dlist_cdna.fa.
    - dlist_combined_fasta_out           (str) Path to the output fasta file containing the d-list sequences  combined genome-based and cDNA-based alignment. Only used by the following columns: 'alignment_to_reference', 'alignment_to_reference_cdna', 'alignment_to_reference_genome', 'alignment_to_reference_count_total', 'alignment_to_reference_count_cdna', 'alignment_to_reference_count_genome', 'substring_alignment_to_reference', 'substring_alignment_to_reference_cdna', 'substring_alignment_to_reference_genome', 'substring_alignment_to_reference_count_total', 'substring_alignment_to_reference_count_cdna',  'substring_alignment_to_reference_count_genome'.Default: `out`/dlist.fa.

    # Returning and saving of optional output
    - save_variants_updated_exploded_vk_info_csv (bool) Whether to save the exploded dataframe. Default: False.
    - make_pyfastx_summary_file          (bool) Whether to make a summary file of the VCRS fasta file using pyfastx. Default: False.
    - make_kat_histogram                 (bool) Whether to make a histogram of the k-mer abundances using kat. Default: False.

    # General arguments:
    - chunksize                          (int) Number of variants to process at a time. If None, then all variants will be processed at once. Default: None.
    - dry_run                            (bool) Whether to do a dry run (i.e., print the parameters and return without running the function). Default: False.
    - list_columns                       (bool) Whether to list the possible values for `columns_to_include` and their descriptions and immediately exit. Default: False.
    - overwrite                          (bool) Whether to overwrite the output files if they already exist. Default: False.
    - threads                            (int) Number of threads to use for bowtie2 and bowtie2-build. Only used by the following columns: 'alignment_to_reference', 'alignment_to_reference_cdna', 'alignment_to_reference_genome', 'alignment_to_reference_count_total', 'alignment_to_reference_count_cdna', 'alignment_to_reference_count_genome', 'substring_alignment_to_reference', 'substring_alignment_to_reference_cdna', 'substring_alignment_to_reference_genome', 'substring_alignment_to_reference_count_total', 'substring_alignment_to_reference_count_cdna',  'substring_alignment_to_reference_count_genome', 'pseudoaligned_to_reference', 'pseudoaligned_to_reference_despite_not_truly_aligning', 'VCRSs_for_which_this_VCRS_is_a_substring', 'VCRSs_for_which_this_VCRS_is_a_superstring', 'VCRS_is_a_substring_of_another_VCRS', 'VCRS_is_a_superstring_of_another_VCRS'; and when 'VCRSs_for_which_this_VCRS_is_a_superstring', 'VCRS_is_a_substring_of_another_VCRS', 'VCRS_is_a_superstring_of_another_VCRS', make_kat_histogram==True. Default: 2.
    - logging_level                      (str) Logging level. Can also be set with the environment variable VARSEEK_LOGGING_LEVEL. Default: INFO.
    - save_logs                          (True/False) Whether to save logs to a file. Default: False.
    - log_out_dir                        (str) Directory to save logs. Default: `out`/logs
    - verbose                            (True/False) Whether to print additional information e.g., progress bars. Does not affect logging. Default: False.

    # Hidden arguments (part of kwargs):
    - w                                  (int) Maximum length of the VCRS flanking regions. Must be an integer between [1, k-1]. Only utilized for the column 'cdna_and_genome_same'. Corresponds to `w` in the varseek build function. Default: 54.
    - bowtie2_path                       (str) Path to the directory containing the bowtie2 and bowtie2-build executables. Default: None.
    - vcrs_strandedness                  (bool) Whether to consider VCRSs as stranded when aligning to the human reference and comparing VCRS k-mers to each other. vcrs_strandedness True corresponds to treating forward and reverse-complement as distinct; False corresponds to treating them as the same. Corresponds to `vcrs_strandedness` in the varseek build function. Only used by the following columns: 'alignment_to_reference', 'alignment_to_reference_cdna', 'alignment_to_reference_genome', 'alignment_to_reference_count_total', 'alignment_to_reference_count_cdna', 'alignment_to_reference_count_genome', 'substring_alignment_to_reference', 'substring_alignment_to_reference_cdna', 'substring_alignment_to_reference_genome', 'substring_alignment_to_reference_count_total', 'substring_alignment_to_reference_count_cdna',  'substring_alignment_to_reference_count_genome', 'pseudoaligned_to_reference', 'pseudoaligned_to_reference_despite_not_truly_aligning', 'number_of_kmers_with_overlap_to_other_VCRSs', 'number_of_other_VCRSs_with_overlapping_kmers', 'VCRSs_with_overlapping_kmers, 'kmer_overlap_with_other_VCRSs'; and if make_kat_histogram==True. Default: False.
    - near_splice_junction_threshold     (int) Maximum distance from a splice junction to be considered "near" a splice junction. Only utilized for the column 'distance_to_nearest_splice_junction'. Default: 10.
    - reference_cdna_fasta               (str) Path to the reference cDNA fasta file. Only utilized for the column 'cdna_and_genome_same'. Default: "cdna".
    - reference_genome_fasta             (str) Path to the reference genome fasta file. Only utilized for the column 'cdna_and_genome_same'. Default: "genome".
    - variants                           (str) Path to the variants csv file. Only utilized for the column 'cdna_and_genome_same', and when `var_id_column` provided. Corresponds to `variants` in the varseek build function. Default: None.
    - sequences                          (str) Path to the sequences fasta file. Only utilized for the column 'cdna_and_genome_same'. Corresponds to `sequences` in the varseek build function. Default: None.
    - kallisto                           (str) Path to the directory containing the kallisto executable. Only utilized for the columns `pseudoaligned_to_reference`, `pseudoaligned_to_reference_despite_not_truly_aligning`. Default: None.
    - bustools                           (str) Path to the directory containing the bustools executable. Only utilized for the columns `pseudoaligned_to_reference`, `pseudoaligned_to_reference_despite_not_truly_aligning`. Default: None.
    - pseudoalignment_workflow           (str) Pseudoalignment workflow to use. Only utilized for the columns `pseudoaligned_to_reference`, `pseudoaligned_to_reference_despite_not_truly_aligning`. Options: {"standard", "nac"}. Default: "nac".
    """
    # CELL
    # * 0. Informational arguments that exit early
    if list_columns:
        print_list_columns()
        return

    # * 1. logger and set out folder (must to it up here or else logger and config will save in the wrong place)
    if out is None:
        out = input_dir if input_dir else "."
    
    if save_logs and not log_out_dir:
        log_out_dir = os.path.join(out, "logs")
    if not kwargs.get("running_within_chunk_iteration", False):
        set_varseek_logging_level_and_filehandler(logging_level=logging_level, save_logs=save_logs, log_dir=log_out_dir)

    if isinstance(columns_to_include, str):
        columns_to_include = [columns_to_include]

    # * 1.5 Chunk iteration
    if chunksize is not None:
        params_dict = make_function_parameter_to_value_dict(1)
        for key in ["vcrs_fasta", "variants_updated_csv", "id_to_header_csv", "chunksize"]:
            params_dict.pop(key, None)
        
        vcrs_fasta = os.path.join(input_dir, "vcrs.fa") if not vcrs_fasta else vcrs_fasta  # copy-paste from below (sorry)
        variants_updated_csv = os.path.join(input_dir, "variants_updated.csv") if (not variants_updated_csv and os.path.isfile(os.path.join(input_dir, "variants_updated.csv"))) else None  # copy-paste from below
        id_to_header_csv = os.path.join(input_dir, "id_to_header_mapping.csv") if (not id_to_header_csv and os.path.isfile(os.path.join(input_dir, "id_to_header_mapping.csv"))) else None  # copy-paste from below
        
        total_chunks = count_chunks(vcrs_fasta, chunksize)
        for i in range(0, total_chunks):
            chunk_number = i + 1  # start at 1
            logger.info(f"Processing chunk {chunk_number}/{total_chunks}")
            vcrs_fasta_chunk = save_fasta_chunk(fasta_path=vcrs_fasta, chunk_number=chunk_number, chunksize=chunksize)
            variants_updated_csv_chunk = save_csv_chunk(csv_path=variants_updated_csv, chunk_number=chunk_number, chunksize=chunksize)
            id_to_header_csv_chunk = save_csv_chunk(csv_path=id_to_header_csv, chunk_number=chunk_number, chunksize=chunksize)
            info(vcrs_fasta=vcrs_fasta_chunk, variants_updated_csv=variants_updated_csv_chunk, id_to_header_csv=id_to_header_csv_chunk, chunksize=None, chunk_number=chunk_number, running_within_chunk_iteration=True, **params_dict)  # running_within_chunk_iteration here for logger setup and report_time_elapsed decorator
            if chunk_number == total_chunks:
                for tmp_file in [vcrs_fasta_chunk, variants_updated_csv_chunk, id_to_header_csv_chunk]:
                    if isinstance(tmp_file, str) and os.path.exists(tmp_file):
                        os.remove(tmp_file)
                return
    
    chunk_number = kwargs.get("chunk_number", 1)
    first_chunk = (chunk_number == 1)

    # * 2. Type-checking
    params_dict = make_function_parameter_to_value_dict(1)
    validate_input_info(params_dict)

    if not os.path.isdir(input_dir):  # only use os.path.isdir when I require that a directory already exists; checked outside validate_input_info to avoid raising issue when type-checking within vk ref
        raise ValueError(f"Input directory '{input_dir}' does not exist. Please provide a valid directory.")

    # * 3. Dry-run
    if dry_run:
        print(get_varseek_dry_run(params_dict, function_name="info"))
        return

    # * 4. Save params to config file and run info file
    config_file = os.path.join(out, "config", "vk_info_config.json")
    save_params_to_config_file(params_dict, config_file)

    run_info_file = os.path.join(out, "config", "vk_info_run_info.txt")
    save_run_info(run_info_file, params_dict=params_dict, function_name="info")

    # * 5. Set up default folder/file input paths, and make sure the necessary ones exist
    if not vcrs_fasta:
        vcrs_fasta = os.path.join(input_dir, "vcrs.fa")
    if not os.path.isfile(vcrs_fasta):
        raise FileNotFoundError(f"File not found: {vcrs_fasta}")

    if not variants_updated_csv:
        variants_updated_csv = os.path.join(input_dir, "variants_updated.csv")
    if not os.path.isfile(variants_updated_csv):
        # logger.info(f"File not found: {variants_updated_csv}. Proceeding without it.")
        variants_updated_csv = None

    if not id_to_header_csv:
        id_to_header_csv = os.path.join(input_dir, "id_to_header_mapping.csv")
    if not os.path.isfile(id_to_header_csv):
        # logger.info(f"File not found: {id_to_header_csv}. Proceeding without it.")
        id_to_header_csv = None

    # * 6. Set up default folder/file output paths, and make sure they don't exist unless overwrite=True
    if not reference_out_dir:
        reference_out_dir = os.path.join(out, "reference")

    os.makedirs(out, exist_ok=True)
    os.makedirs(reference_out_dir, exist_ok=True)

    # if someone specifies an output path, then it should be saved
    if variants_updated_exploded_vk_info_csv_out:
        save_variants_updated_exploded_vk_info_csv = True

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
        if os.path.isfile(output_file) and not overwrite and first_chunk:
            raise ValueError(f"Output file '{output_file}' already exists. Set 'overwrite=True' to overwrite it.")
        if os.path.dirname(output_file):
            os.makedirs(os.path.dirname(output_file), exist_ok=True)

    # * 7. Define kwargs defaults
    w = kwargs.get("w", 54)
    bowtie_path = kwargs.get("bowtie2_path", None)
    vcrs_strandedness = kwargs.get("vcrs_strandedness", False)
    near_splice_junction_threshold = kwargs.get("near_splice_junction_threshold", 10)
    reference_cdna_fasta = kwargs.get("reference_cdna_fasta", "cdna")
    reference_genome_fasta = kwargs.get("reference_genome_fasta", "genome")
    variants = kwargs.get("variants", None)
    sequences = kwargs.get("sequences", None)
    kallisto = kwargs.get("kallisto", None)
    bustools = kwargs.get("bustools", None)
    pseudoalignment_workflow = kwargs.get("pseudoalignment_workflow", "nac")

    # * 7.5 make sure ints are ints
    k, max_ambiguous_vcrs, max_ambiguous_reference, dlist_reference_ensembl_release, threads = int(k), int(max_ambiguous_vcrs), int(max_ambiguous_reference), int(dlist_reference_ensembl_release), int(threads)

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
        if dlist_reference_type != "combined":
            logger.warning("Setting dlist_reference_type to 'combined' because dlist_reference_source is provided. If you want to use a different type, please set dlist_reference_type explicitly.")
            dlist_reference_type = "combined"

    if dlist_reference_genome_fasta == "t2t" or dlist_reference_cdna_fasta == "t2t" or dlist_reference_gtf == "t2t":
        dlist_reference_dir = os.path.join(reference_out_dir, "t2t")
        dlist_reference_genome_fasta, dlist_reference_cdna_fasta, dlist_reference_gtf = download_t2t_reference_files(dlist_reference_dir)
    elif dlist_reference_genome_fasta == "grch37" or dlist_reference_cdna_fasta == "grch37" or dlist_reference_gtf == "grch37":
        dlist_reference_dir = os.path.join(reference_out_dir, f"ensembl_grch37_release{dlist_reference_ensembl_release}")  # matches vk build
        dlist_reference_genome_fasta, dlist_reference_cdna_fasta, dlist_reference_gtf = download_ensembl_reference_files(dlist_reference_dir, grch=37, ensembl_release=dlist_reference_ensembl_release)
    elif dlist_reference_genome_fasta == "grch38" or dlist_reference_cdna_fasta == "grch38" or dlist_reference_gtf == "grch38":
        dlist_reference_dir = os.path.join(reference_out_dir, f"ensembl_grch38_release{dlist_reference_ensembl_release}")
        dlist_reference_genome_fasta, dlist_reference_cdna_fasta, dlist_reference_gtf = download_ensembl_reference_files(dlist_reference_dir, grch=38, ensembl_release=dlist_reference_ensembl_release)
    else:
        if dlist_reference_source:
            if dlist_reference_ensembl_release:
                dlist_reference_dir = os.path.join(reference_out_dir, f"{dlist_reference_source}_release{dlist_reference_ensembl_release}")
            else:
                dlist_reference_dir = os.path.join(reference_out_dir, dlist_reference_source)
        else:
            dlist_reference_dir = os.path.join(reference_out_dir, "dlist_reference_dir")

    #* normally, I barrel through the function if I cannot successfully add a column - but in these cases, because they are defaults of vk ref and have rather unintuitive parameters, I return error early on rather than barreling through
    if dlist_reference_type == "genome":
        if not dlist_reference_genome_fasta:
            raise ValueError("For alignment to reference and d-list construction, you must provide specify the arguments dlist_reference_genome_fasta and/or dlist_reference_cdna_fasta.")
        if dlist_reference_cdna_fasta is not None:
            logger.warning("dlist_reference_type is set to 'genome', but dlist_reference_cdna_fasta is provided. The d-list will be constructed only for the genome-based alignment. If you want to include cDNA as well, please set dlist_reference_type='combined' or None.")
            dlist_reference_cdna_fasta = None
    elif dlist_reference_type == "transcriptome":
        if not dlist_reference_cdna_fasta:
            raise ValueError("For alignment to reference and d-list construction, you must provide specify the arguments dlist_reference_genome_fasta and/or dlist_reference_cdna_fasta.")
        if dlist_reference_genome_fasta is not None:
            logger.warning("dlist_reference_type is set to 'transcriptome', but dlist_reference_genome_fasta is provided. The d-list will be constructed only for the cDNA-based alignment. If you want to include genome as well, please set dlist_reference_type='combined' or None.")
            dlist_reference_genome_fasta = None
    elif dlist_reference_type == "combined":
        if not dlist_reference_genome_fasta or not dlist_reference_cdna_fasta:
            raise ValueError("For alignment to reference and d-list construction, you must provide specify the arguments dlist_reference_genome_fasta and dlist_reference_cdna_fasta.")
    
    if any(column in columns_to_include for column in bowtie_columns_dlist) or columns_to_include == "all":
        if not dlist_reference_genome_fasta and not dlist_reference_cdna_fasta:
            raise ValueError("For alignment to reference and d-list construction, you must provide specify the arguments dlist_reference_genome_fasta and/or dlist_reference_cdna_fasta.")
        elif dlist_reference_genome_fasta and not dlist_reference_cdna_fasta:
            logger.warning("Only dlist_reference_genome_fasta is provided. The d-list will be constructed only for the genome-based alignment. If you want to include cDNA, please provide dlist_reference_cdna_fasta as well.")
        elif dlist_reference_cdna_fasta and not dlist_reference_genome_fasta:
            logger.warning("Only dlist_reference_cdna_fasta is provided. The d-list will be constructed only for the cDNA-based alignment. If you want to include genome, please provide dlist_reference_genome_fasta as well.")
    
    if any(column in columns_to_include for column in ["pseudoaligned_to_reference", "pseudoaligned_to_reference_despite_not_truly_aligning"]) or columns_to_include == "all":
        if not dlist_reference_genome_fasta:
            raise ValueError("For pseudoalignment to reference, you must provide the dlist_reference_genome_fasta argument. Please provide it.")
        if not os.path.isfile(dlist_reference_genome_fasta):
            raise FileNotFoundError(f"File not found: {dlist_reference_genome_fasta}")
        if not dlist_reference_gtf:
            raise ValueError("For pseudoalignment to reference, you must provide the dlist_reference_gtf argument. Please provide it.")
        if not os.path.isfile(dlist_reference_gtf):
            raise FileNotFoundError(f"File not found: {dlist_reference_gtf}")
        

    columns_to_explode = ["header", "order"]
    columns_NOT_to_explode = ["vcrs_id", "vcrs_header", "vcrs_sequence", "vcrs_sequence_rc"]
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
        vcrs_fasta_base, vcrs_fasta_ext = splitext_custom(vcrs_fasta)
        temp_header_fa = f"{vcrs_fasta_base}_with_headers{vcrs_fasta_ext}"
        if not os.path.exists(temp_header_fa):
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
    # load in data
    if variants_updated_csv is None:
        columns_original = []
        mutation_metadata_df = pd.DataFrame(list(pyfastx.Fastx(vcrs_fasta)), columns=["vcrs_id", "vcrs_sequence"])
        if id_to_header_dict is not None:
            mutation_metadata_df["vcrs_header"] = mutation_metadata_df["vcrs_id"].map(id_to_header_dict)
        else:
            mutation_metadata_df["vcrs_header"] = mutation_metadata_df["vcrs_id"]
            if mutation_metadata_df["vcrs_header"].iloc[0].startswith("vcrs"):  # use_IDs was False in vk build and header column is "vcrs_id"
                raise ValueError("Header column in vcrs fasta file is 'vcrs_id'. Please provide a mapping file to swap IDs for headers.")
    else:
        mutation_metadata_df = pd.read_csv(variants_updated_csv)
        columns_original = mutation_metadata_df.columns.tolist()

        columns_to_include = set(columns_to_include_possible_values.keys()) if columns_to_include == "all" else set(columns_to_include)
        columns_to_include = list(set(columns_to_include) - set(columns_original))  # ensure that I don't try to add columns that already exist in the dataframe

        for column in mutation_metadata_df.columns:
            if column not in columns_to_explode + columns_NOT_to_explode:  # alternative: check if the first and last characters are '[' and ']', respectively
                mutation_metadata_df[column] = mutation_metadata_df[column].apply(lambda x: (safe_literal_eval(x) if isinstance(x, str) and x.startswith("[") and x.endswith("]") else x))
        columns_to_explode.extend([col for col in mutation_metadata_df.columns if col not in columns_NOT_to_explode])

    mutation_metadata_df["header_list"] = mutation_metadata_df["vcrs_header"].str.split(";")
    mutation_metadata_df["order_list"] = mutation_metadata_df["header_list"].apply(lambda x: list(range(len(x))))

    vcrs_header_has_merged_values = mutation_metadata_df["vcrs_header"].apply(lambda x: isinstance(x, str) and ";" in x).any()

    if vcrs_header_has_merged_values:
        mutation_metadata_df_exploded = explode_df(mutation_metadata_df, columns_to_explode, verbose=verbose)  # will add columns 'header' and 'order'
    else:
        mutation_metadata_df_exploded = mutation_metadata_df
        mutation_metadata_df_exploded["header"] = mutation_metadata_df_exploded["vcrs_header"]
        mutation_metadata_df_exploded["order"] = [[0]] * len(mutation_metadata_df_exploded)

    if var_id_column is not None:
        mutation_metadata_df_exploded.rename(columns={"header": var_id_column}, inplace=True)
        mutation_metadata_df_exploded["header"] = mutation_metadata_df_exploded["hgvs"]

    first_few_headers_follow_HGVS_pattern = mutation_metadata_df_exploded["header"].head(100).str.match(HGVS_pattern, na=False)
    if not first_few_headers_follow_HGVS_pattern.all():
        logger.warning("Some headers do not follow the HGVS pattern. Please check the input data.")

    mutation_metadata_df_exploded[["seq_ID_used_for_vcrs", "variant_used_for_vcrs"]] = mutation_metadata_df_exploded["header"].str.split(":", expand=True)
    mutation_metadata_df_exploded["seq_ID_used_for_vcrs"] = mutation_metadata_df_exploded["seq_ID_used_for_vcrs"].astype(str)

    if variant_source_column not in mutation_metadata_df_exploded.columns:
        identify_variant_source(mutation_metadata_df_exploded, variant_source_column=variant_source_column)
    unique_vcrs_sources = mutation_metadata_df_exploded[variant_source_column].unique()
    variant_source = "combined" if len(unique_vcrs_sources) > 1 else unique_vcrs_sources[0]

    # ensures proper handling if someone passes in seq_id_column and var_column to vk ref, but not cdna/genome columns in particular; as well as if they passed sequences but not reference_cdna_fasta
    if variant_source == "cdna":
        if kwargs.get("seq_id_column") and not seq_id_cdna_column:
            seq_id_cdna_column = kwargs.get("seq_id_column")
        if kwargs.get("var_column") and not var_cdna_column:
            var_cdna_column = kwargs.get("var_column")
        if (not reference_cdna_fasta or not os.path.isfile(reference_cdna_fasta)) and (sequences and os.path.isfile(sequences)):
            reference_cdna_fasta = sequences
    elif variant_source == "genome":
        if kwargs.get("seq_id_column") and not seq_id_genome_column:
            seq_id_genome_column = kwargs.get("seq_id_column")
        if kwargs.get("var_column") and not var_genome_column:
            var_genome_column = kwargs.get("var_column")
        if (not reference_genome_fasta or not os.path.isfile(reference_genome_fasta)) and (sequences and os.path.isfile(sequences)):
            reference_genome_fasta = sequences

    add_mutation_information(mutation_metadata_df_exploded, mutation_column="variant_used_for_vcrs")

    if seq_id_cdna_column and var_cdna_column in mutation_metadata_df_exploded.columns:
        add_mutation_information(mutation_metadata_df_exploded, mutation_column=var_cdna_column, variant_source="cdna")
    if seq_id_genome_column and var_genome_column in mutation_metadata_df_exploded.columns:
        add_mutation_information(mutation_metadata_df_exploded, mutation_column=var_genome_column, variant_source="genome")

    if "variant_type" not in mutation_metadata_df_exploded.columns:
        add_variant_type(mutation_metadata_df_exploded, var_column="variant_used_for_vcrs")

    columns_to_explode.extend([col for col in mutation_metadata_df_exploded.columns if col not in columns_NOT_to_explode and col not in columns_to_explode])

    # CELL
    if variant_source in {"genome", "combined"}:
        mutation_metadata_df_exploded = mutation_metadata_df_exploded.loc[~((mutation_metadata_df_exploded[variant_source_column] == "genome") & ((pd.isna(mutation_metadata_df_exploded[seq_id_genome_column])) | (mutation_metadata_df_exploded[var_genome_column].str.contains("g.nan", na=True))))]

    # CELL
    if columns_to_include == "all" or "cdna_and_genome_same" in columns_to_include:
        if "cdna_and_genome_same" in mutation_metadata_df_exploded.columns:
            columns_to_explode.append("cdna_and_genome_same")
        else:
            varseek_build_temp_folder="vk_build_tmp"
            delete_temp_dir=True
            try:
                logger.info("Comparing cDNA and genome")
                mutation_metadata_df_exploded, columns_to_explode = compare_cdna_and_genome(
                    mutation_metadata_df_exploded,
                    reference_cdna_fasta=reference_cdna_fasta,
                    reference_genome_fasta=reference_genome_fasta,
                    mutations_csv=variants,
                    w=w,
                    variant_source=variant_source,
                    columns_to_explode=columns_to_explode,
                    seq_id_column_cdna=seq_id_cdna_column,
                    var_column_cdna=var_cdna_column,
                    seq_id_column_genome=seq_id_genome_column,
                    var_column_genome=var_genome_column,
                    reference_out_dir=reference_out_dir,
                    varseek_build_temp_folder=varseek_build_temp_folder,
                    delete_temp_dir=delete_temp_dir
                )
            except Exception as e:
                logger.error(f"Error comparing cDNA and genome: {e}")
                columns_not_successfully_added.append("cdna_and_genome_same")
            if delete_temp_dir and os.path.exists(varseek_build_temp_folder):
                shutil.rmtree(varseek_build_temp_folder)

    # CELL
    if columns_to_include == "all" or "distance_to_nearest_splice_junction" in columns_to_include:
        # Add metadata: distance to nearest splice junction
        if seq_id_genome_column not in mutation_metadata_df_exploded.columns or "start_variant_position_genome" not in mutation_metadata_df_exploded.columns or "end_variant_position_genome" not in mutation_metadata_df_exploded.columns:
            logger.warning("Missing 'start_variant_position_genome' or 'end_variant_position_genome' columns. Cannot compute distance to nearest splice junction. Make these by adding seq_id_genome_column and var_genome_column to the input dataframe.")
            columns_not_successfully_added.append("distance_to_nearest_splice_junction")
        else:
            try:
                logger.info("Computing distance to nearest splice junction")
                mutation_metadata_df_exploded, columns_to_explode = compute_distance_to_closest_splice_junction(
                    mutation_metadata_df_exploded,
                    gtf,
                    columns_to_explode=columns_to_explode,
                    near_splice_junction_threshold=near_splice_junction_threshold,
                    seq_id_genome_column=seq_id_genome_column,
                )
            except Exception as e:
                logger.error(f"Error computing distance to nearest splice junction: {e}")
                columns_not_successfully_added.append("distance_to_nearest_splice_junction")

    # CELL
    if columns_to_include == "all" or "number_of_variants_in_this_gene_total" in columns_to_include or "header_with_gene_name" in columns_to_include:
        if gene_name_column in mutation_metadata_df_exploded.columns:
            total_genes_output_stat_file = f"{output_stat_folder}/total_genes_and_transcripts.txt"
            try:
                logger.info("Calculating total gene info")
                mutation_metadata_df_exploded, columns_to_explode = calculate_total_gene_info(
                    mutation_metadata_df_exploded,
                    vcrs_id_column="vcrs_id",
                    gene_name_column=gene_name_column,
                    output_stat_file=total_genes_output_stat_file,
                    output_plot_folder=output_plot_folder,
                    columns_to_include=columns_to_include,
                    columns_to_explode=columns_to_explode,
                    overwrite=overwrite,
                    first_chunk=first_chunk,
                )
            except Exception as e:
                logger.error(f"Error calculating total gene info: {e}")
                columns_not_successfully_added.extend(["number_of_variants_in_this_gene_total", "header_with_gene_name"])
        else:
            logger.warning(f"Gene name column '{gene_name_column}' not found in dataframe. Skipping total gene info calculation.")
            columns_not_successfully_added.extend(["number_of_variants_in_this_gene_total", "header_with_gene_name"])

    # CELL
    # Calculate variants within (k-1) of each mutation
    # compare transcript location for spliced only with cDNA header;
    # filter out genome rows where cdna and genome are the same (because I don't want to count spliced and unspliced as 2 separate things when they are the same - but maybe I do?) and compare genome location for all (both spliced and unspliced) with regular header (will be the sole way to add information for unspliced rows, and will add unspliced info for cdna comparisons);
    # take union of sets

    if columns_to_include == "all" or ("nearby_variants" in columns_to_include or "nearby_variants_count" in columns_to_include or "has_a_nearby_variant" in columns_to_include):
        try:
            logger.info("Calculating nearby variants")
            mutation_metadata_df_exploded, columns_to_explode = calculate_nearby_mutations(variant_source_column=variant_source_column, k=k, output_plot_folder=output_plot_folder, variant_source=variant_source, mutation_metadata_df_exploded=mutation_metadata_df_exploded, columns_to_explode=columns_to_explode, seq_id_cdna_column=seq_id_cdna_column, seq_id_genome_column=seq_id_genome_column)
        except Exception as e:
            logger.error(f"Error calculating nearby variants: {e}")
            columns_not_successfully_added.extend(["nearby_variants", "nearby_variants_count", "has_a_nearby_variant"])

    # CELL
    if vcrs_header_has_merged_values:
        logger.info("Collapsing dataframe")
        mutation_metadata_df, columns_to_explode = collapse_df(
            mutation_metadata_df_exploded,
            columns_to_explode,
        )
    else:
        mutation_metadata_df = mutation_metadata_df_exploded

    # CELL
    mutation_metadata_df["vcrs_id"] = mutation_metadata_df["vcrs_id"].astype(str)

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

    # TODO: calculate if VCRS was optimized - compare VCRS_length to length of unoptimized - exclude subs, and calculate max([2*w + length(added) - length(removed)], [2*w - 1])

    if bowtie_path is not None:
        bowtie2_build = f"{bowtie_path}/bowtie2-build"
        bowtie2 = f"{bowtie_path}/bowtie2"
    else:
        bowtie2_build = "bowtie2-build"
        bowtie2 = "bowtie2"

    # TODO: have more columns_to_include options that allows me to do cdna alone, genome alone, or both combined - currently it is either cdna+genome or nothing
    if columns_to_include == "all" or any(column in columns_to_include for column in bowtie_columns_dlist):
        if not is_program_installed(bowtie2):
            logger.error(f"bowtie2 must be installed to run for the following columns: {bowtie_columns_dlist}. Please install bowtie2 or omit these columns")

        try:
            logger.info("Aligning to normal genome and building dlist")
            mutation_metadata_df, sequence_names_set_union_genome_and_cdna = align_to_normal_genome_and_build_dlist(
                mutations=vcrs_fasta,
                vcrs_id_column="vcrs_id",
                out_dir_notebook=out,
                reference_out=dlist_reference_dir,
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
                overwrite=overwrite,
                first_chunk=first_chunk,
                chunk_number=chunk_number,
            )
        except Exception as e:
            logger.error("Error aligning to normal genome and building alignment_to_reference: %s", e)
            columns_not_successfully_added.extend(bowtie_columns_dlist)  # list(column for column in bowtie_columns_dlist if column in columns_to_include)

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
    if columns_to_include == "all" or ("pseudoaligned_to_reference" in columns_to_include):
        try:
            logger.info("Getting VCRSs that pseudoalign")
            mutation_metadata_df = get_vcrss_that_pseudoalign_but_arent_dlisted(
                mutation_metadata_df=mutation_metadata_df,
                vcrs_id_column="vcrs_id",
                vcrs_fa=vcrs_fasta,
                sequence_names_set=set(),
                human_reference_genome_fa=dlist_reference_genome_fasta,
                human_reference_gtf=dlist_reference_gtf,
                out_dir_notebook=out,
                ref_folder_kb=dlist_reference_dir,
                header_column_name="vcrs_id",
                additional_kb_extract_filtering_workflow=pseudoalignment_workflow,
                k=k,
                threads=threads,
                strandedness=vcrs_strandedness,
                column_name="pseudoaligned_to_reference",
                kallisto=kallisto,
                bustools=bustools,
            )
        except Exception as e:
            logger.error(f"Error getting VCRSs that pseudoalign: {e}")
            columns_not_successfully_added.append("pseudoaligned_to_reference")

    if columns_to_include == "all" or ("pseudoaligned_to_reference_despite_not_truly_aligning" in columns_to_include):
        if "alignment_to_reference" not in mutation_metadata_df.columns:
            logger.warning("alignment_to_reference not found in dataframe. Skipping pseudoalignment to reference.")
            columns_not_successfully_added.append("pseudoaligned_to_reference_despite_not_truly_aligning")
        else:
            if "pseudoaligned_to_reference" in mutation_metadata_df.columns:  #!! untested
                mutation_metadata_df["pseudoaligned_to_reference_despite_not_truly_aligning"] = (~mutation_metadata_df["alignment_to_reference"]) & (mutation_metadata_df["pseudoaligned_to_reference"])
            else:
                try:
                    logger.info("Getting VCRSs that pseudoalign but aren't dlisted")
                    mutation_metadata_df = get_vcrss_that_pseudoalign_but_arent_dlisted(
                        mutation_metadata_df=mutation_metadata_df,
                        vcrs_id_column="vcrs_id",
                        vcrs_fa=vcrs_fasta,
                        sequence_names_set=sequence_names_set_union_genome_and_cdna,
                        human_reference_genome_fa=dlist_reference_genome_fasta,
                        human_reference_gtf=dlist_reference_gtf,
                        out_dir_notebook=out,
                        ref_folder_kb=dlist_reference_dir,
                        header_column_name="vcrs_id",
                        additional_kb_extract_filtering_workflow=pseudoalignment_workflow,
                        k=k,
                        threads=threads,
                        strandedness=vcrs_strandedness,
                        column_name="pseudoaligned_to_reference_despite_not_truly_aligning",
                        kallisto=kallisto,
                        bustools=bustools,
                    )
                except Exception as e:
                    logger.error(f"Error getting VCRSs that pseudoalign but aren't dlisted: {e}")
                    columns_not_successfully_added.append("pseudoaligned_to_reference_despite_not_truly_aligning")

    # CELL
    if columns_to_include == "all" or ("number_of_kmers_with_overlap_to_other_VCRSs" in columns_to_include or "number_of_other_VCRSs_with_overlapping_kmers" in columns_to_include or "VCRSs_with_overlapping_kmers" in columns_to_include or "kmer_overlap_with_other_VCRSs" in columns_to_include):
        try:
            logger.info("Calculating overlap between VCRS items")
            df_overlap_stat_file = f"{output_stat_folder}/df_overlap_stat.txt"
            df_overlap = get_df_overlap(
                vcrs_fasta,
                out_dir_notebook=out,
                k=k,
                strandedness=vcrs_strandedness,
                vcrs_id_column="vcrs_id",
                output_text_file=df_overlap_stat_file,
                output_plot_folder=output_plot_folder,
            )

            mutation_metadata_df = mutation_metadata_df.merge(df_overlap, on="vcrs_id", how="left")
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

            output_file_longest_homopolymer = f"{output_plot_folder}/longest_homopolymer.png"
            plot_histogram_of_nearby_mutations_7_5(
                mutation_metadata_df,
                "longest_homopolymer_length",
                bins=20,
                output_file=output_file_longest_homopolymer,
            )
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
    if columns_to_include == "all" or "vcrs_variant_type" in columns_to_include:
        try:
            logger.info("Adding VCRS mutation type")
            mutation_metadata_df = add_vcrs_variant_type(mutation_metadata_df, var_column="vcrs_header")
        except Exception as e:
            logger.error(f"Error adding VCRS mutation type: {e}")
            columns_not_successfully_added.append("vcrs_variant_type")

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
    if columns_to_include == "all" or any(column in columns_to_include for column in bowtie_columns_vcrs_to_vcrs):
        if not is_program_installed(bowtie2):
            logger.error(f"bowtie2 must be installed to run for the following columns: {bowtie_columns_vcrs_to_vcrs}. Please install bowtie2 or omit these columns")
        vcrs_to_vcrs_bowtie_folder = f"{out}/bowtie_vcrs_to_vcrs"
        vcrs_sam_file = f"{vcrs_to_vcrs_bowtie_folder}/mutant_reads_to_vcrs_index.sam"
        substring_output_stat_file = f"{output_stat_folder}/substring_output_stat.txt"

        try:
            logger.info("Creating VCRS to self headers")
            substring_to_superstring_df, superstring_to_substring_df = create_df_of_vcrs_to_self_headers(
                vcrs_sam_file=vcrs_sam_file,
                vcrs_fa=vcrs_fasta,
                bowtie_vcrs_reference_folder=vcrs_to_vcrs_bowtie_folder,
                bowtie_path=bowtie_path,
                threads=threads,
                strandedness=vcrs_strandedness,
                vcrs_id_column="vcrs_id",
                output_stat_file=substring_output_stat_file,
            )

            mutation_metadata_df["vcrs_id"] = mutation_metadata_df["vcrs_id"].astype(str)
            mutation_metadata_df = mutation_metadata_df.merge(substring_to_superstring_df, on="vcrs_id", how="left")
            mutation_metadata_df = mutation_metadata_df.merge(superstring_to_substring_df, on="vcrs_id", how="left")

            mutation_metadata_df["VCRS_is_a_substring_of_another_VCRS"] = mutation_metadata_df["VCRS_is_a_substring_of_another_VCRS"].fillna(False).astype(bool)
            mutation_metadata_df["VCRS_is_a_superstring_of_another_VCRS"] = mutation_metadata_df["VCRS_is_a_superstring_of_another_VCRS"].fillna(False).astype(bool)

        except Exception as e:
            logger.error(f"Error creating VCRS to self headers: {e}")
            columns_not_successfully_added.extend(bowtie_columns_vcrs_to_vcrs)   # list(column for column in bowtie_columns_vcrs_to_vcrs if column in columns_to_include)

    # CELL
    logger.info("sorting variant metadata by VCRS id")
    mutation_metadata_df = mutation_metadata_df.sort_values(by="vcrs_id").reset_index(drop=True)

    mutation_metadata_df.drop(columns=["header_list", "order_list"], inplace=True, errors="ignore")

    logger.info("Saving variant metadata")
    mutation_metadata_df.to_csv(variants_updated_vk_info_csv_out, index=False, header=first_chunk, mode=determine_write_mode(variants_updated_vk_info_csv_out, overwrite=overwrite, first_chunk=first_chunk))

    # CELL
    if save_variants_updated_exploded_vk_info_csv:
        if vcrs_header_has_merged_values:
            logger.info("Saving exploded variant metadata")
            mutation_metadata_df_exploded = explode_df(mutation_metadata_df, columns_to_explode, verbose=verbose)
            mutation_metadata_df_exploded.to_csv(variants_updated_exploded_vk_info_csv_out, index=False, header=first_chunk, mode=determine_write_mode(variants_updated_vk_info_csv_out, overwrite=overwrite, first_chunk=first_chunk))
        else:
            logger.info("Variant data has no merged values, so skipping exploding")

    logger.info(f"Saved variant metadata to {variants_updated_vk_info_csv_out}")
    logger.info(f"Columns: {mutation_metadata_df.columns}")
    logger.info(f"Columns successfully added: {set(mutation_metadata_df.columns.tolist()) - set(columns_original)}")
    logger.info(f"Columns not successfully added: {set(columns_not_successfully_added)}")
