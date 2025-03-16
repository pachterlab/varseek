"""varseek clean and specific helper functions."""

import json
import logging
import os
import subprocess
import time
from pathlib import Path

import anndata
import anndata as ad
import numpy as np
import pandas as pd
from packaging import version

from varseek.utils import (
    adjust_variant_adata_by_normal_gene_matrix,
    check_file_path_is_string_with_valid_extension,
    decrement_adata_matrix_when_split_by_Ns_or_running_paired_end_in_single_end_mode,
    increment_adata_based_on_dlist_fns,
    is_valid_int,
    load_in_fastqs,
    make_function_parameter_to_value_dict,
    match_adata_orders,
    plot_knee_plot,
    print_varseek_dry_run,
    remove_adata_columns,
    report_time_elapsed,
    save_params_to_config_file,
    save_run_info,
    set_up_logger,
    sort_fastq_files_for_kb_count,
    make_good_barcodes_and_file_index_tuples,
    write_to_vcf,
    write_vcfs_for_rows,
    set_varseek_logging_level_and_filehandler
)

from .constants import non_single_cell_technologies, technology_valid_values

logger = logging.getLogger(__name__)
logger = set_up_logger(logger, logging_level="INFO", save_logs=False, log_dir=None)


def make_vcf():
    pass


def prepare_set(vcrs_id_set_to_exclusively_keep):
    if vcrs_id_set_to_exclusively_keep is None:
        return None  # Keep None as None

    elif isinstance(vcrs_id_set_to_exclusively_keep, (list, tuple)):
        return set(vcrs_id_set_to_exclusively_keep)  # Convert list/tuple to set

    elif isinstance(vcrs_id_set_to_exclusively_keep, str) and os.path.isfile(vcrs_id_set_to_exclusively_keep) and vcrs_id_set_to_exclusively_keep.endswith(".txt"):
        # Load lines from text file, stripping whitespace
        with open(vcrs_id_set_to_exclusively_keep, "r", encoding="utf-8") as f:
            return set(line.strip() for line in f if line.strip())  # Ignore empty lines

    elif isinstance(vcrs_id_set_to_exclusively_keep, set):
        return vcrs_id_set_to_exclusively_keep  # Already a set, return as is

    else:
        raise ValueError("Invalid input: must be None, a list, tuple, set, or a path to a text file")


scanpy_conditions = ["filter_cells_by_min_counts", "filter_cells_by_min_genes", "filter_genes_by_min_cells", "filter_cells_by_max_mt_content", "doublet_detection"]


def validate_input_clean(params_dict):
    # check if adata_vcrs is of type Anndata
    adata_vcrs = params_dict["adata_vcrs"]
    if not isinstance(adata_vcrs, (str, Path, anndata.AnnData)):  # I will enforce that adata_vcrs exists later, as otherwise it will throw an error when I call this through vk count before kb count can run
        raise ValueError("adata_vcrs must be an AnnData object or a file path to an h5ad object.")
    if isinstance(adata_vcrs, (str, Path)):
        check_file_path_is_string_with_valid_extension(adata_vcrs, "adata_vcrs", "h5ad")

    # technology
    technology = params_dict.get("technology", None)
    technology_valid_values_lower = {x.lower() for x in technology_valid_values}
    if technology is None or technology.lower() not in technology_valid_values_lower:
        raise ValueError(f"Technology must be one of {technology_valid_values_lower}")

    for param_name, min_value, optional_value in [
        ("min_counts", 1, False),
        ("filter_cells_by_min_genes", 0, True),  # optional True means that it can be None
        ("filter_genes_by_min_cells", 0, True),
        ("threads", 1, False),
    ]:
        param_value = params_dict.get(param_name)
        if not is_valid_int(param_value, ">=", min_value, optional=optional_value):
            must_be_value = f"an integer >= {min_value}" if not optional_value else f"an integer >= {min_value} or None"
            raise ValueError(f"{param_name} must be {must_be_value}. Got {param_value} of type {type(param_value)}.")

    if params_dict.get("cpm_normalization") and (not params_dict.get("adata_reference_genome") and not params_dict.get("kb_count_reference_genome_dir")):
        raise ValueError("adata_reference_genome or kb_count_reference_genome_dir must be provided if cpm_normalization=True.")

    if technology not in non_single_cell_technologies:
        if params_dict.get("filter_cells_by_min_counts") is True:
            try:
                from kneed import KneeLocator
            except ImportError:
                raise ImportError("kneed is required for filter_cells_by_min_counts=True. See pyproject.toml project.optional-dependencies for version recommendation. Install it with:\n" "  pip install kneed")
        for condition in scanpy_conditions:
            if params_dict.get(condition):
                if not params_dict.get("adata_reference_genome") and not params_dict.get("kb_count_reference_genome_dir"):
                    raise ValueError(f"adata_reference_genome or kb_count_reference_genome_dir must be provided if {condition}=True.")
                try:
                    import scanpy as sc
                except ImportError:
                    raise ImportError("Scanpy is required for this function. See pyproject.toml project.optional-dependencies for version recommendation. Install it with:\n" "  pip install scanpy")

    # filter_cells_by_min_counts - gets special treatment because it can also be True for automatic calculation
    filter_cells_by_min_counts = params_dict.get("filter_cells_by_min_counts", None)
    if filter_cells_by_min_counts is not None and not is_valid_int(filter_cells_by_min_counts, ">=", 1):
        raise ValueError(f"filter_cells_by_min_counts must be an integer >= 1 (for manual threshold), True (for automatic threshold calculation with kneed's KneeLocator), or None (for no threshold). Got {filter_cells_by_min_counts} of type {type(filter_cells_by_min_counts)}.")

    # filter_cells_by_max_mt_content - special treatment because it is between rather than lower-bounded only
    if not is_valid_int(params_dict.get("filter_cells_by_max_mt_content"), "between", min_value_inclusive=0, max_value_inclusive=100, optional=True):
        raise ValueError(f"filter_cells_by_max_mt_content must be an integer between 0 and 100, or None. Got {params_dict.get('filter_cells_by_max_mt_content')}.")

    # boolean
    for param_name in ["use_binary_matrix", "drop_empty_columns", "apply_single_end_mode_on_paired_end_data_correction", "split_reads_by_Ns_and_low_quality_bases", "apply_dlist_correction", "qc_against_gene_matrix", "doublet_detection", "remove_doublets", "cpm_normalization", "sum_rows", "mm", "union", "save_vcf", "dry_run", "overwrite"]:
        if not isinstance(params_dict.get(param_name), bool):
            raise ValueError(f"{param_name} must be a boolean. Got {param_name} of type {type(params_dict.get(param_name))}.")
    if not isinstance(params_dict.get("multiplexed"), bool) and params_dict.get("multiplexed") is not None:
        raise ValueError(f"multiplexed must be a boolean pr None. Got {params_dict.get('multiplexed')} of type {type(params_dict.get('multiplexed'))}.")

    # sets
    for param_name in ["vcrs_id_set_to_exclusively_keep", "vcrs_id_set_to_exclude", "transcript_set_to_exclusively_keep", "transcript_set_to_exclude", "gene_set_to_exclusively_keep", "gene_set_to_exclude"]:
        param_value = params_dict.get(param_name, None)
        if param_value is not None and not isinstance(param_value, (set, list, tuple) and not (isinstance(param_value, str) and param_value.endswith(".txt") and os.path.isfile(param_value))):  # checks if it is (1) None, (2) a set/list/tuple, or (3) a string path to a txt file that exists
            raise ValueError(f"{param_name} must be a set. Got {param_name} of type {type(param_value)}.")

    # k
    k = params_dict.get("k", None)
    if not isinstance(k, (int, str)) or int(k) < 1:
        raise ValueError(f"k must be a positive integer. Got {k} of type {type(k)}.")
    if int(k) % 2 == 0 or int(k) > 63:
        logger.warning("If running a workflow with vk ref or kb ref, k should be an odd number between 1 and 63. Got k=%s.", k)

    # fastqs
    fastqs = params_dict["fastqs"]  # tuple
    if len(fastqs) == 0:
        raise ValueError("No fastq files provided")

    parity_valid_values = {"single", "paired"}
    if params_dict["parity"] not in parity_valid_values:
        raise ValueError(f"Parity must be one of {parity_valid_values}")

    # $ type checking of the directory and text file performed earlier by load_in_fastqs

    for fastq in fastqs:
        check_file_path_is_string_with_valid_extension(fastq, variable_name=fastq, file_type="fastq")  # ensure that all fastq files have valid extension
        if not os.path.isfile(fastq):  # ensure that all fastq files exist
            raise ValueError(f"File {fastq} does not exist")

    # file paths
    for param_name, file_type in {
        "config": ["json", "yaml"],
        "vcrs_index": "index",
        "vcrs_t2g": "t2g",
        "vcrs_fasta": "fasta",
        "dlist_fasta": "fasta",
        "adata_reference_genome": "h5ad",
        "adata_vcrs_clean_out": "h5ad",
        "adata_reference_genome_clean_out": "h5ad",
        "vcf_out": "vcf",
    }.items():
        check_file_path_is_string_with_valid_extension(params_dict.get(param_name), param_name, file_type)

    # directories
    for param_name in ["vk_ref_dir", "kb_count_vcrs_dir", "kb_count_reference_genome_dir"]:
        if not isinstance(params_dict.get(param_name), (str, Path)) and params_dict.get(param_name) is not None:
            raise ValueError(f"Directory {param_name} {params_dict.get(param_name)} is not a string or None")
        if params_dict.get(param_name) and not params_dict.get("dry_run") and (not os.path.isdir(params_dict.get(param_name)) or len(os.listdir(params_dict.get(param_name))) == 0):  # including the dry_run condition so that vk count dry run does not throw an error
            raise ValueError(f"Directory {params_dict.get(param_name)} does not exist")
    if not isinstance(params_dict.get("out"), (str, Path)):
        raise ValueError(f"Out directory {params_dict.get('out')} is not a string")

    if params_dict.get("qc_against_gene_matrix"):
        for arg in ["kb_count_vcrs_dir", "kb_count_reference_genome_dir"]:
            kb_count_normal_dir = params_dict.get(arg)
            if kb_count_normal_dir and os.path.exists(kb_count_normal_dir):
                run_info_json = os.path.join(kb_count_normal_dir, "run_info.json")
                with open(run_info_json, "r") as f:
                    data = json.load(f)
                if "--num" not in data["call"]:
                    raise ValueError(f"--num must be included in the provided value for {arg}. Please run kb count on the normal genome again, or provide a new path for {arg} to allow varseek count to make this file for you.")
        logger.warning("For the best results with qc_against_gene_matrix=True, try to ensure the reference assembly and release of the genome used with kb_count_reference_genome_dir is as similar as possible to the one used with kb_count_vcrs_dir. This helps ensure that transcript/gene IDs are as stable as possible.")


needs_for_normal_genome_matrix = ["filter_cells_by_min_counts", "filter_cells_by_min_genes", "filter_genes_by_min_cells", "filter_cells_by_max_mt_content", "doublet_detection", "cpm_normalization"]

@report_time_elapsed
def clean(
    adata_vcrs,  # required inputs
    technology,
    min_counts=2,  # parameters
    use_binary_matrix=False,
    drop_empty_columns=False,
    apply_single_end_mode_on_paired_end_data_correction=False,
    split_reads_by_Ns_and_low_quality_bases=False,
    apply_dlist_correction=False,
    qc_against_gene_matrix=False,
    filter_cells_by_min_counts=None,
    filter_cells_by_min_genes=None,
    filter_genes_by_min_cells=None,
    filter_cells_by_max_mt_content=None,
    doublet_detection=False,
    remove_doublets=False,
    cpm_normalization=False,
    sum_rows=False,
    vcrs_id_set_to_exclusively_keep=None,
    vcrs_id_set_to_exclude=None,
    transcript_set_to_exclusively_keep=None,
    transcript_set_to_exclude=None,
    gene_set_to_exclusively_keep=None,
    gene_set_to_exclude=None,
    k=None,
    mm=False,
    union=False,
    parity="single",
    multiplexed=None,
    sort_fastqs=True,
    adata_reference_genome=None,  # optional inputs
    fastqs=None,
    vk_ref_dir=None,
    vcrs_index=None,
    vcrs_t2g=None,
    vcrs_fasta=None,
    dlist_fasta=None,
    variants_updated_csv=None,
    kb_count_vcrs_dir=None,
    kb_count_reference_genome_dir=None,
    variants_updated_csv_columns_to_merge=None,
    seq_id_column="seq_ID",
    gene_id_column="gene_id",
    out=".",  # output paths
    adata_vcrs_clean_out=None,
    adata_reference_genome_clean_out=None,
    vcf_out=None,
    save_vcf=False,  # optional saves
    dry_run=False,  # general
    overwrite=False,
    threads=2,
    logging_level=None,
    save_logs=False,
    log_out_dir=None,
    **kwargs,
):
    """
    Apply quality control to the VCRS count matrix (cell/sample x variant) and save the cleaned AnnData object.

    # Required input arguments:
    - adata_vcrs                            (str or anndata.AnnData): The input AnnData object containing the VCRS data.
    - technology                            (str)  Technology used to generate the data. To see list of spported technologies, run `kb --list`.

    # Additional parameters
    - min_counts                            (int): Minimum counts to consider valid in the VCRS count matrix - everything below this number gets set to 0. Default: 2.
    - use_binary_matrix                     (bool): Whether to binarize the matrix (i.e., set all values >=1 to 1). Default: False.
    - drop_empty_columns                    (bool): Whether to drop columns (variants) that are empty across all samples. Default: False.
    - apply_single_end_mode_on_paired_end_data_correction    (bool): Whether to apply correction for when paired end data was run in single end mode (which happens when (concatenate_paired_fastqs = True or split_reads_by_Ns_and_low_quality_bases = True) and parity == paired in vk count/fastqpp). Will correct for double-counting when each pair maps to the same VCRS, which can happen when the fragment length is <2*read length, or when the VCRS contains a long insertion that spans both paired ends. Default: False.
    - split_reads_by_Ns_and_low_quality_bases                     (bool): Whether split_reads_by_Ns_and_low_quality_bases was run in vk fastqpp/count. Only used when apply_single_end_mode_on_paired_end_data_correction=True (helps with parsing the fastq header names). Default: False.
    - apply_dlist_correction                (bool): Whether to apply correction for dlist. If a read mapping(s) is tossed due to a d-list entry that is derived from an unrelated gene, then count this entry in the count matrix. Only relevant if a d-list was used during the reference construction (the default: no d-list unless otherwise specified in vk ref/kb ref - see vk ref --list_downloadable_references). Requires `dlist_fasta` to be a valid path as generated by vk ref/info/filter. Default: False.
    - qc_against_gene_matrix                (bool): Whether to apply correction for qc against gene matrix. If a read maps to 2+ VCRSs that belong to different genes, then cross-reference with the reference genome to determine which gene the read belongs to, and set all VCRSs that do not correspond to this gene to 0 for that read. Also, cross-reference all reads that map to 1 VCRS and ensure that the reads maps to the gene corresponding to this VCRS, or else set this value to 0 in the count matrix. Default: True.
    - filter_cells_by_min_counts            (int or bool or None): Part of the QC performed on the **gene** count matrix with which to adjust the **variant** count matrix. Minimum number of gene counts per cell to keep the cell. If True, will use kneed's KneeLocator to determine the cutoff. Default: None.
    - filter_cells_by_min_genes             (int or None): Part of the QC performed on the **gene** count matrix with which to adjust the **variant** count matrix. Minimum number of genes per cell to keep the cell. Default: None.
    - filter_genes_by_min_cells             (int or None): Part of the QC performed on the **gene** count matrix with which to adjust the **variant** count matrix. Minimum number of cells per gene to keep the gene. Default: None.
    - filter_cells_by_max_mt_content        (int or None): Part of the QC performed on the **gene** count matrix with which to adjust the **variant** count matrix. Maximum percentage of mitochondrial content per cell to keep the cell. Default: None.
    - doublet_detection                     (bool): Part of the QC performed on the **gene** count matrix with which to adjust the **variant** count matrix. Whether to run doublet detection. Default: False.
    - remove_doublets                       (bool): Part of the QC performed on the **gene** count matrix with which to adjust the **variant** count matrix. Whether to remove doublets. Default: False.
    - cpm_normalization                     (bool): Part of the QC performed on the **gene** count matrix with which to adjust the **variant** count matrix. Whether to run cpm normalization. Default: False.
    - sum_rows                              (bool): Whether to sum across barcodes (rows) in the VCRS count matrix. Default: False.
    - vcrs_id_set_to_exclusively_keep       (str or Set(str) or None): If a set, will keep only the VCRSs in this set. If a list/tuple, will convert to a set and then keep only the VCRSs in this set. If a string, will load the text file and keep only the VCRSs in this set. Default: None.
    - vcrs_id_set_to_exclude                (str or Set(str) or None): If a set, will exclude the VCRSs in this set. If a list/tuple, will convert to a set and then exclude the VCRSs in this set. If a string, will load the text file and exclude the VCRSs in this set. Default: None.
    - transcript_set_to_exclusively_keep    (str or Set(str) or None): If a set, will keep only the transcripts in this set. If a list/tuple, will convert to a set and then keep only the transcripts in this set. If a string, will load the text file and keep only the transcripts in this set. Default: None.
    - transcript_set_to_exclude             (str or Set(str) or None): If a set, will exclude the transcripts in this set. If a list/tuple, will convert to a set and then exclude the transcripts in this set. If a string, will load the text file and exclude the transcripts in this set. Default: None.
    - gene_set_to_exclusively_keep          (str or Set(str) or None): If a set, will keep only the genes in this set. If a list/tuple, will convert to a set and then keep only the genes in this set. If a string, will load the text file and keep only the genes in this set. Default: None.
    - gene_set_to_exclude                   (str or Set(str) or None): If a set, will exclude the genes in this set. If a list/tuple, will convert to a set and then exclude the genes in this set. If a string, will load the text file and exclude the genes in this set. Default: None.
    - k                                     (int): K-mer length used for the k-mer index. Used only when apply_dlist_correction=True. Default: None.
    - mm                                    (bool): Whether to count multimapped reads in the adata count matrix. Only used when apply_single_end_mode_on_paired_end_data_correction, apply_dlist_correction, or qc_against_gene_matrix is True. Default: False.
    - union                                 (bool): Whether to count unioned reads in the adata count matrix. Only used when apply_single_end_mode_on_paired_end_data_correction, apply_dlist_correction, or qc_against_gene_matrix is True. Default: False.
    - parity                                (str) "single" or "paired". Only relevant if technology is bulk or a smart-seq. Default: "single"
    - multiplexed                           (bool) Indicates that the fastq files are multiplexed. Only used if sort_fastqs=True and technology is a smartseq technology. Default: None
    - sort_fastqs                           (bool): Whether to sort the fastqs. Default: True.

    # Optional input arguments:
    - adata_reference_genome                (str): Path to the reference genome AnnData object. Default: `kb_count_reference_genome_dir`/counts_unfiltered/adata.h5ad.
    - fastqs                                (str or list[str]) List of fastq files to be processed. If paired end, the list should contains paths such as [file1_R1, file1_R2, file2_R1, file2_R2, ...]. Only used when `apply_single_end_mode_on_paired_end_data_correction`, `apply_dlist_correction`, `qc_against_gene_matrix` is True. Default: None
    - vk_ref_dir                            (str): Directory containing the VCRS reference files. Same as `out` as specified in vk ref. Default: None.
    - vcrs_index                            (str): Path to the VCRS index file. Default: None.
    - vcrs_t2g                              (str): Path to the VCRS t2g file. Default: None.
    - vcrs_fasta                            (str): Path to the VCRS fasta file. Default: None.
    - dlist_fasta                           (str): Path to the dlist fasta file. Default: None.
    - variants_updated_csv                  (str): Path to the variants updated csv file generated by vk build, info, or filter (i.e., something where each row corresponds to a VCRS). Default: None.
    - kb_count_vcrs_dir                     (str): Path to the kb count output directory for the VCRS reference. Default: None.
    - kb_count_reference_genome_dir         (str): Path to the kb count output directory for the reference genome. Default: None.

    # Optional column names variants_updated_csv
    - variants_updated_csv_columns_to_merge (str or set): Columns in the variants_updated_csv to merge with the adata var. Default: None.
    - seq_id_column                         (str): Column name in the adata var that contains the transcript ID. Default: "seq_ID".
    - gene_id_column                        (str): Column name in the adata var that contains the gene ID. Default: "gene_id".

    # Output paths:
    - out                                   (str): Output directory. Default: ".".
    - adata_vcrs_clean_out                  (str): Path to save the cleaned VCRS AnnData object. Default: `out`/adata_cleaned.h5ad.
    - adata_reference_genome_clean_out      (str): Path to save the cleaned reference genome AnnData object. Default: `out`/adata_reference_genome_cleaned.h5ad.
    - save_vcf                              (bool): Whether to save the VCF file. Default: True.
    - vcf_out                               (str): Path to save the VCF file. Default: `out`/vcrs.vcf.

    # General:
    - dry_run                               (bool): Whether to run in dry run mode. Default: False.
    - overwrite                             (bool): Whether to overwrite existing files. Default: False.
    - threads                               (int): Number of threads to use. Default: 2.
    - logging_level                         (str) Logging level. Can also be set with the environment variable VARSEEK_LOGGING_LEVEL. Default: INFO.
    - save_logs                             (True/False) Whether to save logs to a file. Default: False.
    - log_out_dir                           (str) Directory to save logs. Default: `out`/logs

    # Hidden arguments
    - id_to_header_csv                      (str): Path to the VCRS id to header csv file. Default: None.
    - kallisto                              (str): Path to the kallisto binary. Default: None.
    - bustools                              (str): Path to the bustools binary. Default: None.
    - parity_kb_count                       (str) The parity of the reads used in kb count when generating adata_vcrs. Default: `parity`.
    """
    # * 1. logger
    if save_logs and not log_out_dir:
        log_out_dir = os.path.join(out, "logs")
    set_varseek_logging_level_and_filehandler(logging_level=logging_level, save_logs=save_logs, log_dir=log_out_dir)


    # * 1.5 load in fastqs
    fastqs_original = fastqs
    fastqs = load_in_fastqs(fastqs)  # this will make it in params_dict

    # * 2. Type-checking
    params_dict = make_function_parameter_to_value_dict(1)
    validate_input_clean(params_dict)
    params_dict["fastqs"] = fastqs_original  # change back for dry run and config_file

    if isinstance(adata_vcrs, (str, Path)) and not os.path.isfile(adata_vcrs) and not dry_run:  # only use os.path.isfile when I require that a directory already exists; checked outside validate_input_clean to avoid raising issue when type-checking within vk count
        raise ValueError(f"adata_vcrs file path {adata_vcrs} does not exist.")

    # * 3. Dry-run and set out folder (must to it up here or else config will save in the wrong place)
    if dry_run:
        print_varseek_dry_run(params_dict, function_name="clean")
        return None

    # * 4. Save params to config file and run info file
    config_file = os.path.join(out, "config", "vk_info_config.json")
    save_params_to_config_file(params_dict, config_file)

    run_info_file = os.path.join(out, "config", "vk_info_run_info.txt")
    save_run_info(run_info_file)

    # * 5. Set up default folder/file input paths, and make sure the necessary ones exist
    if kb_count_reference_genome_dir and not adata_reference_genome:
        adata_reference_genome = os.path.join(kb_count_reference_genome_dir, "counts_unfiltered", "adata.h5ad")

    id_to_header_csv = kwargs.get("id_to_header_csv", None)

    if vk_ref_dir and os.path.exists(vk_ref_dir):  # make sure all of the defaults below match vk info/filter
        vcrs_index = os.path.join(vk_ref_dir, "vcrs_index.idx") if not vcrs_index else vcrs_index
        if not vcrs_t2g:
            vcrs_t2g = os.path.join(vk_ref_dir, "vcrs_t2g_filtered.txt") if os.path.isfile(os.path.join(vk_ref_dir, "vcrs_t2g_filtered.txt")) else os.path.join(vk_ref_dir, "vcrs_t2g.txt")
        if not vcrs_fasta:
            vcrs_fasta = os.path.join(vk_ref_dir, "vcrs_filtered.fa") if os.path.isfile(os.path.join(vk_ref_dir, "vcrs_filtered.fa")) else os.path.join(vk_ref_dir, "vcrs.fa")
        if not id_to_header_csv:
            id_to_header_csv = os.path.join(vk_ref_dir, "id_to_header_mapping_filtered.csv") if os.path.isfile(os.path.join(vk_ref_dir, "id_to_header_mapping_filtered.csv")) else os.path.join(vk_ref_dir, "id_to_header_mapping.csv")
        if not variants_updated_csv:
            variants_updated_csv = os.path.join(vk_ref_dir, "variants_updated_filtered.csv") if os.path.isfile(os.path.join(vk_ref_dir, "variants_updated_filtered.csv")) else os.path.join(vk_ref_dir, "variants_updated.csv")
        if not dlist_fasta:
            dlist_fasta = os.path.join(vk_ref_dir, "dlist_filtered.fa") if os.path.isfile(os.path.join(vk_ref_dir, "dlist_filtered.fa")) else os.path.join(vk_ref_dir, "dlist.fa")

    if (apply_single_end_mode_on_paired_end_data_correction or apply_dlist_correction or qc_against_gene_matrix) and (not kb_count_vcrs_dir or not os.path.exists(kb_count_vcrs_dir) or len(os.listdir(kb_count_vcrs_dir)) == 0):
        raise ValueError("kb_count_vcrs_dir must be provided as the output from kb count out to the VCRS reference if apply_single_end_mode_on_paired_end_data_correction, apply_dlist_correction, or qc_against_gene_matrix is True.")
    if qc_against_gene_matrix and (not kb_count_reference_genome_dir or not os.path.exists(kb_count_reference_genome_dir) or len(os.listdir(kb_count_reference_genome_dir)) == 0):
        raise ValueError("kb_count_reference_genome_dir must be provided as the output from kb count out to the reference genome if qc_against_gene_matrix is True.")

    # * 6. Set up default folder/file output paths, and make sure they don't exist unless overwrite=True
    # if someone specifies an output path, then it should be saved
    if vcf_out:
        save_vcf = True

    output_figures_dir = os.path.join(out, "vk_clean_figures")

    adata_vcrs_clean_out = os.path.join(out, "adata_cleaned.h5ad") if not adata_vcrs_clean_out else adata_vcrs_clean_out
    adata_reference_genome_clean_out = os.path.join(out, "adata_reference_genome_cleaned.h5ad") if not adata_reference_genome_clean_out else adata_reference_genome_clean_out
    vcf_out = os.path.join(out, "vcrs.vcf") if not vcf_out else vcf_out

    for output_path in [output_figures_dir, adata_vcrs_clean_out, adata_reference_genome_clean_out]:
        if os.path.exists(output_path) and not overwrite:
            raise ValueError(f"Output path {output_path} already exists. Please set overwrite=True to overwrite it.")
        if os.path.dirname(output_path):
            os.makedirs(os.path.dirname(output_path), exist_ok=True)

    os.makedirs(out, exist_ok=True)
    os.makedirs(output_figures_dir, exist_ok=True)

    # * 7. Define kwargs defaults
    # id_to_header_csv was defined in step 5
    kallisto = kwargs.get("kallisto", "kallisto")
    bustools = kwargs.get("bustools", "bustools")
    parity_kb_count = kwargs.get("parity_kb_count", parity)  # TODO: implement this check (compare parity_kb_count to parity and make the mapping dict)

    # * 7.5 make sure ints are ints
    min_counts, threads = int(min_counts), int(threads)

    # * 8. Start the actual function
    if remove_doublets and not doublet_detection:
        logger.warning("remove_doublets is True, but doublet_detection is False. Setting doublet_detection to True.")
        doublet_detection = True

    if technology.lower() != "bulk" and "smartseq" not in technology.lower():
        parity = "single"

    if apply_dlist_correction and not dlist_fasta:
        raise ValueError("dlist_fasta must be provided if apply_dlist_correction is True.")

    if fastqs:
        try:
            fastqs = sort_fastq_files_for_kb_count(fastqs, technology=technology, multiplexed=multiplexed, check_only=(not sort_fastqs))
        except Exception:
            if sort_fastqs:
                logger.warning(f"Automatic FASTQ argument order sorting for kb count could not recognize FASTQ file name format. Skipping argument order sorting.")

    if not kallisto:
        kallisto_binary_path_command = "kb info | grep 'kallisto:' | awk '{print $3}' | sed 's/[()]//g'"
        kallisto = subprocess.run(kallisto_binary_path_command, shell=True, executable="/bin/bash", stdout=subprocess.PIPE, text=True, check=True).stdout.strip()
    if not bustools:
        bustools_binary_path_command = "kb info | grep 'bustools:' | awk '{print $3}' | sed 's/[()]//g'"
        bustools = subprocess.run(bustools_binary_path_command, shell=True, executable="/bin/bash", stdout=subprocess.PIPE, text=True, check=True).stdout.strip()

    if apply_dlist_correction:  # and anything else that requires new kallisto
        kallisto_version_command = rf"{kallisto} 2>&1 | grep -oP 'kallisto \K[0-9]+\.[0-9]+\.[0-9]+'"
        kallisto_version_installed = subprocess.run(kallisto_version_command, shell=True, executable="/bin/bash", stdout=subprocess.PIPE, text=True, check=True).stdout.strip()
        kallisto_version_required = "0.51.1"

        if version.parse(kallisto_version_installed) < kallisto_version_required:
            raise ValueError(f"Please install kallisto version {kallisto_version_required} or higher.")

    vcrs_id_set_to_exclusively_keep = prepare_set(vcrs_id_set_to_exclusively_keep)
    vcrs_id_set_to_exclude = prepare_set(vcrs_id_set_to_exclude)
    transcript_set_to_exclusively_keep = prepare_set(transcript_set_to_exclusively_keep)
    transcript_set_to_exclude = prepare_set(transcript_set_to_exclude)
    gene_set_to_exclusively_keep = prepare_set(gene_set_to_exclusively_keep)
    gene_set_to_exclude = prepare_set(gene_set_to_exclude)

    adata = adata_vcrs

    if isinstance(adata, str) and os.path.exists(adata) and adata.endswith(".h5ad"):
        adata = ad.read_h5ad(adata)

    if adata.var.index[0].startswith("vcrs_"):
        adata.var["vcrs_id"] = adata.var.index
        if id_to_header_csv and isinstance(id_to_header_csv, str) and os.path.exists(id_to_header_csv):
            id_to_header_df = pd.read_csv(id_to_header_csv, index_col=0)
            adata.var = adata.var.merge(id_to_header_df, on="vcrs_id", how="left")  # will add vcrs_header
        else:
            adata.var["vcrs_header"] = adata.var["vcrs_id"]
    else:
        adata.var["vcrs_header"] = adata.var.index

    adata.var.index.name = "variant"
    original_var_names = adata.var_names.copy()

    if parity == "paired" and parity_kb_count == "single":
        barcodes_file = os.path.join(kb_count_vcrs_dir, "matrix.sample.barcodes")
        bad_to_good_barcode_dict = make_good_barcodes_and_file_index_tuples(barcodes_file)
        adata.obs.index = adata.obs.index.map(lambda x: bad_to_good_barcode_dict.get(x, x))  # map from old (incorrect) barcodes to new (correct) barcodes  #!!! ensure the old barcodes don't linger anywhere else

        # Convert to DataFrame for easy manipulation
        df = pd.DataFrame(adata.X.toarray() if hasattr(adata.X, "toarray") else adata.X, 
                  index=adata.obs.index, 
                  columns=adata.var.index)
        df_grouped = df.groupby(df.index).sum()

        # Create new AnnData object
        adata = ad.AnnData(X=df_grouped.values, obs=pd.DataFrame(index=df_grouped.index), var=adata.var)  #!!! inspect this

    variants_updated_csv_columns_to_merge = kwargs.get("variants_updated_csv_columns_to_merge", None)
    for id_column, corresponding_argument in [(seq_id_column, transcript_set_to_exclusively_keep), (seq_id_column, transcript_set_to_exclude), (gene_id_column, gene_set_to_exclusively_keep), (gene_id_column, gene_set_to_exclude)]:
        if corresponding_argument is not None:
            if variants_updated_csv_columns_to_merge is None:
                variants_updated_csv_columns_to_merge = {id_column}
            elif variants_updated_csv_columns_to_merge is not None and id_column not in variants_updated_csv_columns_to_merge:
                variants_updated_csv_columns_to_merge.add(id_column)
            else:
                pass  # variants_updated_csv_columns_to_merge is a set and id_column is already in it

    if variants_updated_csv_columns_to_merge is not None and not os.path.isfile(variants_updated_csv):
        raise ValueError(f"variants_updated_csv_columns_to_merge is not None, but variants_updated_csv does not exist.")

    if variants_updated_csv and os.path.isfile(variants_updated_csv) and variants_updated_csv_columns_to_merge is not None:
        variants_updated_df = pd.read_csv(variants_updated_csv, index_col=0, usecols=list(variants_updated_csv_columns_to_merge))
        merging_column = "vcrs_header"
        if merging_column not in variants_updated_csv_columns_to_merge:
            variants_updated_csv_columns_to_merge.add(merging_column)
        adata.var = adata.var.merge(variants_updated_df, left_on="vcrs_header", right_on=merging_column, how="left")

    if adata_reference_genome:
        if isinstance(adata_reference_genome, str) and os.path.exists(adata_reference_genome) and adata_reference_genome.endswith(".h5ad"):
            adata_reference_genome = ad.read_h5ad(adata_reference_genome)

    adata.var_names = original_var_names

    if apply_single_end_mode_on_paired_end_data_correction:
        # TODO: test this; also add union
        # TODO; also enforce that when each pair has a pseudoalignment, that the variants belong to the same gene
        adata = decrement_adata_matrix_when_split_by_Ns_or_running_paired_end_in_single_end_mode(
            adata,
            fastq=fastqs,
            kb_count_out=kb_count_vcrs_dir,
            t2g=vcrs_t2g,
            mm=mm,
            bustools=bustools,
            split_Ns=split_reads_by_Ns_and_low_quality_bases,
            paired_end_fastqs=False,
            paired_end_suffix_length=2,
            technology=technology,
            keep_only_insertions=True,
        )

    if dlist_fasta and apply_dlist_correction:
        # TODO: test this; also add union
        adata = increment_adata_based_on_dlist_fns(
            adata=adata,
            vcrs_fasta=vcrs_fasta,
            dlist_fasta=dlist_fasta,
            kb_count_out=kb_count_vcrs_dir,
            index=vcrs_index,
            t2g=vcrs_t2g,
            fastq=fastqs,
            newer_kallisto=kallisto,
            k=k,
            mm=mm,
            bustools=bustools,
        )

    if qc_against_gene_matrix:
        # TODO: test this
        adata = adjust_variant_adata_by_normal_gene_matrix(adata, kb_count_vcrs_dir=kb_count_vcrs_dir, kb_count_reference_genome_dir=kb_count_reference_genome_dir, id_to_header_csv=id_to_header_csv, vcrs_t2g=vcrs_t2g, t2g_standard=None, fastq_file_list=fastqs, mm=mm, union=union, technology=technology, parity=parity, bustools=bustools)

    if sum_rows and adata.shape[0] > 1:
        # Sum across barcodes (rows)
        summed_data = adata.X.sum(axis=0)

        # Retain the first barcode
        first_barcode = adata.obs_names[0]

        # Create a new AnnData object
        new_adata = anndata.AnnData(X=summed_data.reshape(1, -1), obs=adata.obs.iloc[[0]].copy(), var=adata.var.copy())  # Reshape to (1, n_features)  # Copy the first barcode's metadata  # Copy the original feature metadata

        # Update the obs_names to reflect the first barcode
        new_adata.obs_names = [first_barcode]
        adata = new_adata.copy()

    # set all count values below min_counts to 0
    if min_counts is not None:
        adata.X = adata.X.multiply(adata.X >= min_counts)

    # # remove 0s for memory purposes
    # adata.X.eliminate_zeros()

    if use_binary_matrix:
        adata.X = (adata.X > 0).astype(int)

    # TODO: make sure the adata objects are in the same order (relevant for both bulk and sc) - possibly with match_adata_orders
    if isinstance(adata_reference_genome, anndata.AnnData):
        if technology not in non_single_cell_technologies:  # pardon the double negative - this is just a way to say "if technology is single cell"
            for condition in scanpy_conditions:
                if params_dict.get(condition):
                    import scanpy as sc
            if filter_cells_by_min_counts:
                if not isinstance(filter_cells_by_min_counts, int):  # ie True for automatic
                    from kneed import KneeLocator

                    umi_counts = np.array(adata_reference_genome.X.sum(axis=1)).flatten()
                    umi_counts_sorted = np.sort(umi_counts)[::-1]  # Sort in descending order for the knee plot

                    # Step 2: Use KneeLocator to find the cutoff
                    knee_locator = KneeLocator(
                        range(len(umi_counts_sorted)),
                        umi_counts_sorted,
                        curve="convex",
                        direction="decreasing",
                    )
                    filter_cells_by_min_counts = umi_counts_sorted[knee_locator.knee]
                    plot_knee_plot(
                        umi_counts_sorted=umi_counts_sorted,
                        knee_locator=knee_locator,
                        min_counts_assessed_by_knee_plot=filter_cells_by_min_counts,
                        output_file=f"{output_figures_dir}/knee_plot.png",
                    )
                sc.pp.filter_cells(adata_reference_genome, min_counts=filter_cells_by_min_counts)  # filter cells by min counts
            if filter_cells_by_min_genes:
                sc.pp.filter_cells(adata_reference_genome, min_genes=filter_cells_by_min_genes)  # filter cells by min genes
            if filter_genes_by_min_cells:
                sc.pp.filter_genes(adata_reference_genome, min_cells=filter_genes_by_min_cells)  # filter genes by min cells
            if filter_cells_by_max_mt_content:
                has_mt_genes = adata_reference_genome.var_names.str.startswith("MT-").any()
                if has_mt_genes:
                    adata_reference_genome.var["mt"] = adata_reference_genome.var_names.str.startswith("MT-")
                else:
                    mito_ensembl_ids = sc.queries.mitochondrial_genes("hsapiens", attrname="ensembl_gene_id")
                    mito_genes = set(mito_ensembl_ids["ensembl_gene_id"].values)

                    adata_base_var_names = adata_reference_genome.var_names.str.split(".").str[0]  # Removes minor version from var names
                    mito_genes_base = {gene.split(".")[0] for gene in mito_genes}  # Removes minor version from mito_genes

                    # Identify mitochondrial genes in adata.var using the stripped version of gene IDs
                    adata_reference_genome.var["mt"] = adata_base_var_names.isin(mito_genes_base)

                mito_counts = adata_reference_genome[:, adata_reference_genome.var["mt"]].X.sum(axis=1)

                # Calculate total counts per cell
                total_counts = adata_reference_genome.X.sum(axis=1)

                # Calculate percent mitochondrial gene expression per cell
                adata_reference_genome.obs["percent_mito"] = np.array(mito_counts / total_counts * 100).flatten()

                adata_reference_genome.obs["total_counts"] = adata_reference_genome.X.sum(axis=1).A1
                sc.pp.calculate_qc_metrics(
                    adata_reference_genome,
                    qc_vars=["mt"],
                    percent_top=None,
                    log1p=False,
                    inplace=True,
                )

                sc.pl.violin(
                    adata_reference_genome,
                    ["n_genes_by_counts", "total_counts", "pct_counts_mt"],
                    jitter=0.4,
                    multi_panel=True,
                    save=True,
                )

                # * TODO: move violin plot file path
                violin_plot_path = f"{output_figures_dir}/qc_violin_plot.png"

                adata_reference_genome = adata_reference_genome[adata_reference_genome.obs.pct_counts_mt < filter_cells_by_max_mt_content, :].copy()  # filter cells by high MT content

                adata.obs["percent_mito"] = adata_reference_genome.obs["percent_mito"]
                adata.obs["total_counts"] = adata_reference_genome.obs["total_counts"]
            if doublet_detection:
                sc.pp.scrublet(adata_reference_genome, batch_key="sample")  # filter doublets
                adata.obs["predicted_doublet"] = adata_reference_genome.obs["predicted_doublet"]
                if remove_doublets:
                    adata_reference_genome = adata_reference_genome[~adata_reference_genome.obs["predicted_doublet"], :].copy()
                    adata = adata[~adata.obs["predicted_doublet"], :].copy()

            common_cells = adata.obs_names.intersection(adata_reference_genome.obs_names)
            adata = adata[common_cells, :].copy()

        # do cpm
        if cpm_normalization and not use_binary_matrix:  # normalization not needed for binary matrix
            total_counts = adata_reference_genome.X.sum(axis=1)
            cpm_factor = total_counts / 1e6

            adata.X = adata.X / cpm_factor[:, None]  # Reshape to make cpm_factor compatible with adata.X
            adata.uns["cpm_factor"] = cpm_factor

    if drop_empty_columns:
        # Identify columns (genes) with non-zero counts across samples
        nonzero_gene_mask = np.array((adata.X != 0).sum(axis=0)).flatten() > 0

        # Filter the AnnData object to keep only genes with non-zero counts across samples
        adata = adata[:, nonzero_gene_mask]

    # include or exclude certain genes
    if vcrs_id_set_to_exclusively_keep:
        adata = remove_adata_columns(
            adata,
            values_of_interest=vcrs_id_set_to_exclusively_keep,
            operation="keep",
            var_column_name="vcrs_header",
        )

    if vcrs_id_set_to_exclude:
        adata = remove_adata_columns(
            adata,
            values_of_interest=vcrs_id_set_to_exclude,
            operation="exclude",
            var_column_name="vcrs_header",
        )

    if transcript_set_to_exclusively_keep:
        adata = remove_adata_columns(
            adata,
            values_of_interest=transcript_set_to_exclusively_keep,
            operation="keep",
            var_column_name=seq_id_column,
        )

    if transcript_set_to_exclude:
        adata = remove_adata_columns(
            adata,
            values_of_interest=transcript_set_to_exclude,
            operation="exclude",
            var_column_name=seq_id_column,
        )

    if gene_set_to_exclusively_keep:
        adata = remove_adata_columns(
            adata,
            values_of_interest=gene_set_to_exclusively_keep,
            operation="keep",
            var_column_name=gene_id_column,
        )

    if gene_set_to_exclude:
        adata = remove_adata_columns(
            adata,
            values_of_interest=gene_set_to_exclude,
            operation="exclude",
            var_column_name=gene_id_column,
        )

    adata.var["vcrs_count"] = adata.X.sum(axis=0).A1 if hasattr(adata.X, "A1") else np.asarray(adata.X.sum(axis=0)).flatten()

    if save_vcf:
        make_vcf(vcf_out)  # TODO: write this

    if isinstance(adata_reference_genome, anndata.AnnData):
        adata_reference_genome.write(adata_reference_genome_clean_out)

    adata.write(adata_vcrs_clean_out)

    return adata
