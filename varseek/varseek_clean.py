"""varseek clean and specific helper functions."""

import json
import logging
import os
import scipy.sparse as sp
import subprocess
import pyfastx
import time
from pathlib import Path
import re

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
    add_vcf_info_to_cosmic_tsv,
    make_function_parameter_to_value_dict,
    plot_knee_plot,
    get_varseek_dry_run,
    remove_adata_columns,
    report_time_elapsed,
    save_params_to_config_file,
    save_run_info,
    set_up_logger,
    sort_fastq_files_for_kb_count,
    make_good_barcodes_and_file_index_tuples,
    write_to_vcf,
    cleaned_adata_to_vcf,
    set_varseek_logging_level_and_filehandler,
    remove_variants_from_adata_for_stranded_technologies,
    merge_variants_with_adata,
    identify_variant_source,
    add_information_from_variant_header_to_adata_var_exploded,
    plot_cdna_locations
)

from .constants import non_single_cell_technologies, technology_valid_values, technology_to_strand_bias_mapping, technology_to_file_index_with_transcripts_mapping, HGVS_pattern_general, mutation_pattern

logger = logging.getLogger(__name__)
logger = set_up_logger(logger, logging_level="INFO", save_logs=False, log_dir=None)

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
        ("read_length", 1, True),
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
    for param_name in ["use_binary_matrix", "drop_empty_columns", "apply_single_end_mode_on_paired_end_data_correction", "split_reads_by_Ns_and_low_quality_bases", "apply_dlist_correction", "qc_against_gene_matrix", "doublet_detection", "remove_doublets", "cpm_normalization", "sum_rows", "mm", "save_vcf", "dry_run", "overwrite", "account_for_strand_bias"]:
        if not isinstance(params_dict.get(param_name), bool):
            raise ValueError(f"{param_name} must be a boolean. Got {param_name} of type {type(params_dict.get(param_name))}.")
    if not isinstance(params_dict.get("multiplexed"), bool) and params_dict.get("multiplexed") is not None:
        raise ValueError(f"multiplexed must be a boolean or None. Got {params_dict.get('multiplexed')} of type {type(params_dict.get('multiplexed'))}.")

    # sets
    for param_name in ["vcrs_id_set_to_exclusively_keep", "vcrs_id_set_to_exclude", "gene_set_to_exclusively_keep", "gene_set_to_exclude"]:
        param_value = params_dict.get(param_name, None)
        if param_value is not None and not isinstance(param_value, (set, list, tuple) and not (isinstance(param_value, str) and param_value.endswith(".txt") and os.path.isfile(param_value))):  # checks if it is (1) None, (2) a set/list/tuple, or (3) a string path to a txt file that exists
            raise ValueError(f"{param_name} must be a set. Got {param_name} of type {type(param_value)}.")

    # k
    k = params_dict.get("k", None)
    if k:
        if not isinstance(k, (int, str)) or int(k) < 1:
            raise ValueError(f"k must be a positive integer. Got {k} of type {type(k)}.")
        if int(k) % 2 == 0 or int(k) > 63:
            logger.warning("If running a workflow with vk ref or kb ref, k should be an odd number between 1 and 63. Got k=%s.", k)

    parity_valid_values = {"single", "paired"}
    if params_dict["parity"] not in parity_valid_values:
        raise ValueError(f"Parity must be one of {parity_valid_values}")
    
    strand_bias_end_valid_values = {"5p", "3p", None}
    if params_dict["strand_bias_end"] not in strand_bias_end_valid_values:
        raise ValueError(f"strand_bias_end must be one of {strand_bias_end_valid_values}")
    
    if params_dict.get("account_for_strand_bias") and not params_dict.get("strand_bias_end"):
        raise ValueError("strand_bias_end must be provided if account_for_strand_bias=True.")

    # $ type checking of the directory and text file performed earlier by load_in_fastqs
    fastqs = params_dict["fastqs"]  # tuple
    if fastqs:
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

# @profile
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
    count_reads_that_dont_pseudoalign_to_reference_genome=True,
    account_for_strand_bias=False,
    strand_bias_end=None,
    read_length=None,
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
    gene_set_to_exclusively_keep=None,
    gene_set_to_exclude=None,
    k=None,
    mm=False,
    parity="single",
    multiplexed=None,
    sort_fastqs=True,
    adata_reference_genome=None,  # optional inputs
    fastqs=None,
    vk_ref_dir=None,
    vcrs_index=None,
    vcrs_t2g=None,
    gtf=None,
    kb_count_vcrs_dir=None,
    kb_count_reference_genome_dir=None,
    reference_genome_t2g=None,
    vcf_data_csv=None,
    variants=None,
    sequences=None,
    variant_source=None,
    vcrs_metadata_df=None,
    variants_usecols=None,
    seq_id_column="seq_ID",
    var_column="mutation",
    var_id_column=None,
    gene_id_column="gene_id",
    out=".",  # output paths
    adata_vcrs_clean_out=None,
    adata_reference_genome_clean_out=None,
    vcf_out=None,
    save_vcf=False,  # optional saves
    save_vcf_samples=False,
    chunksize=None,
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
    - count_reads_that_dont_pseudoalign_to_reference_genome (bool): Whether to count reads that don't pseudoalign to the reference genome. Only used when qc_against_gene_matrix=True. Default: True.
    - account_for_strand_bias               (bool): Whether to account for strand bias from stranded single-cell technologies. Default: False.
    - strand_bias_end                       (str): The end of the read to use for strand bias correction. Either "5p" or "3p". Must be provided if and only if account_for_strand_bias=True. Default: None.
    - read_length                           (int): The read length used in the experiment. Must be provided if and only if account_for_strand_bias=True. Default: None.
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
    - gene_set_to_exclusively_keep          (str or Set(str) or None): If a set, will keep only the genes in this set. If a list/tuple, will convert to a set and then keep only the genes in this set. If a string, will load the text file and keep only the genes in this set. Default: None.
    - gene_set_to_exclude                   (str or Set(str) or None): If a set, will exclude the genes in this set. If a list/tuple, will convert to a set and then exclude the genes in this set. If a string, will load the text file and exclude the genes in this set. Default: None.
    - k                                     (int): K-mer length used for the k-mer index. Used only when apply_dlist_correction=True. Default: None.
    - mm                                    (bool): Whether to count multimapped reads in the adata count matrix. Only used when apply_single_end_mode_on_paired_end_data_correction, apply_dlist_correction, or qc_against_gene_matrix is True. Default: False.
    - parity                                (str) "single" or "paired". Only relevant if technology is bulk or a smart-seq. Default: "single"
    - multiplexed                           (bool) Indicates that the fastq files are multiplexed. Only used if sort_fastqs=True and technology is a smartseq technology. Default: None
    - sort_fastqs                           (bool): Whether to sort the fastqs. Default: True.

    # Optional input arguments:
    - adata_reference_genome                (str): Path to the reference genome AnnData object. Default: `kb_count_reference_genome_dir`/counts_unfiltered/adata.h5ad.
    - fastqs                                (str or list[str]) List of fastq files to be processed. If paired end, the list should contains paths such as [file1_R1, file1_R2, file2_R1, file2_R2, ...]. Only used when `apply_single_end_mode_on_paired_end_data_correction`, `apply_dlist_correction`, `qc_against_gene_matrix` is True. Default: None
    - vk_ref_dir                            (str): Directory containing the VCRS reference files. Same as `out` as specified in vk ref. Default: None.
    - vcrs_index                            (str): Path to the VCRS index file. Default: None.
    - vcrs_t2g                              (str): Path to the VCRS t2g file. Default: None.
    - gtf                                   (str): Path to the GTF file. Only used when account_for_strand_bias=True and either (1) strand_bias_end='3p' and/or (2) some VCRSs are derived from genome sequences. Default: None.
    - kb_count_vcrs_dir                     (str): Path to the kb count output directory for the VCRS reference. Default: None.
    - kb_count_reference_genome_dir         (str): Path to the kb count output directory for the reference genome. Default: None.
    - reference_genome_t2g                  (str): Path to the t2g file for the standard reference genome. Only used when qc_against_gene_matrix=True. Default: None.
    - vcf_data_csv                          (str): Path to the VCF data csv file. It needs columns ID (corresponding to vcrs_header from adata.var), CHROM, POS, REF, ALT. If using downloaded reference files from vk ref, then simply provide the `variants` and `sequences` arguments entered at vk ref for this file to be created internally. Default: None.
    - variants                              (str): The variants parameter from vk ref/build. Merged into adata.var with columns variants_usecols. Only strictly needed if using downloaded reference files from vk ref and wanting to save an output VCF file. Default: None.
    - sequences                             (str): The sequences parameter from vk ref/build. Only needed if using downloaded reference files from vk ref and wanting to save an output VCF file. Default: None.
    - variant_source                        (str): The source of the variants. If not provided, it will be inferred from the `variants` parameter. Can be "genome" or "transcriptome". Default: None
    - vcrs_metadata_df                      (str): Path to the VCRS metadata dataframe. Will be merged into adata.var. If a path is provided and the file does not exist, it will be created. Default: None.

    # Optional column names in variants
    - variants_usecols (str or list): Columns in the variants to merge with the adata var. Default: None (use all columns).
    - seq_id_column                         (str): Column name in the variants that contains the sequence ID. The same parameter as in vk ref/build. This is used to merge the sequences with the variants. Default: "seq_ID".
    - var_column                            (str): Column name in the variants that contains the variant information. The same parameter as in vk ref/build. This is used to merge the sequences with the variants. Default: "mutation".
    - var_id_column                         (str): Column name in the variants that contains the variant ID. The same parameter as in vk ref/build. This is used to merge the sequences with the variants. Default: None (will be created as seq_id_column:var_column).
    - gene_id_column                        (str): Column name in the adata var that contains the gene ID. Default: "gene_id".

    # Output paths:
    - out                                   (str): Output directory. Default: ".".
    - adata_vcrs_clean_out                  (str): Path to save the cleaned VCRS AnnData object. Default: `out`/adata_cleaned.h5ad.
    - adata_reference_genome_clean_out      (str): Path to save the cleaned reference genome AnnData object. Default: `out`/adata_reference_genome_cleaned.h5ad.
    - save_vcf                              (bool): Whether to save the VCF file. Default: True.
    - save_vcf_samples                      (bool): Whether to save samples in the VCF. If many samples (including single-cell), can lead to very large files. Default: False.
    - vcf_out                               (str): Path to save the VCF file. Default: `out`/variants.vcf.

    # General:
    - chunksize                             (int) Number of BUS file lines to process at a time. If None, then all lines will be processed at once. Default: None.
    - dry_run                               (bool): Whether to run in dry run mode. Default: False.
    - overwrite                             (bool): Whether to overwrite existing files. Default: False.
    - threads                               (int): Number of threads to use. Default: 2.
    - logging_level                         (str) Logging level. Can also be set with the environment variable VARSEEK_LOGGING_LEVEL. Default: INFO.
    - save_logs                             (True/False) Whether to save logs to a file. Default: False.
    - log_out_dir                           (str) Directory to save logs. Default: `out`/logs

    # Hidden arguments
    - vcrs_fasta                            (str): Path to the VCRS fasta file. Default: None.
    - id_to_header_csv                      (str): Path to the VCRS id to header csv file. Default: None.
    - variants_updated_csv                  (str): Path to the updated variants csv file. Only used if using downloaded reference files from vk ref and vcf_data_csv does not exist. Default: None.
    - dlist_fasta                           (str): Path to the dlist fasta file. Default: None.
    - kallisto                              (str): Path to the kallisto binary. Default: None.
    - bustools                              (str): Path to the bustools binary. Default: None.
    - parity_kb_count                       (str) The parity of the reads used in kb count when generating adata_vcrs. Default: `parity`.
    - cosmic_tsv                            (str): Path to the cosmic tsv file. Only used if creating a VCF file and using a downloaded reference from vk ref and vcf_data_csv does not exist. Default: None.
    - cosmic_reference_genome_fasta         (str): Path to the reference genome fasta file. Only used if creating a VCF file and using a downloaded reference from vk ref and vcf_data_csv does not exist. Default: None.
    - cosmic_version                        (str): Version of the cosmic tsv file. Only used if creating a VCF file and using a downloaded reference from vk ref and vcf_data_csv does not exist. Default: 101.
    - cosmic_email                          (str): Email address for cosmic. Only used if creating a VCF file and using a downloaded reference from vk ref and vcf_data_csv does not exist. Default: None.
    - cosmic_password                       (str): Password for cosmic. Only used if creating a VCF file and using a downloaded reference from vk ref and vcf_data_csv does not exist. Default: None.
    - forgiveness                           (int): Number of bases allowed to be off when account_for_strand_bias=True. E.g., if I am using a 5' technology with a read length of 91 and a forgiveness of 100, then I will keep only variants that fall within the last 191 bases of each respective transcript. Default: 100.
    - add_hgvs_breakdown_to_adata_var       (bool): Whether to add the HGVS breakdown of the variants in separate columns of adata.var including nucleotide positions, variant, variant start position, variant end position, and gene name. Default: True.
    """
    # * 1. logger
    if save_logs and not log_out_dir:
        log_out_dir = os.path.join(out, "logs")
    set_varseek_logging_level_and_filehandler(logging_level=logging_level, save_logs=save_logs, log_dir=log_out_dir)


    # * 1.5 load in fastqs
    fastqs_original = fastqs
    if fastqs:
        fastqs = load_in_fastqs(fastqs)  # this will make it in params_dict

    # * 2. Type-checking
    params_dict = make_function_parameter_to_value_dict(1)
    validate_input_clean(params_dict)
    params_dict["fastqs"] = fastqs_original  # change back for dry run and config_file

    if isinstance(adata_vcrs, (str, Path)) and not os.path.isfile(adata_vcrs) and not dry_run:  # only use os.path.isfile when I require that a directory already exists; checked outside validate_input_clean to avoid raising issue when type-checking within vk count
        raise ValueError(f"adata_vcrs file path {adata_vcrs} does not exist.")

    # * 3. Dry-run and set out folder (must to it up here or else config will save in the wrong place)
    if dry_run:
        print(get_varseek_dry_run(params_dict, function_name="clean"))
        return None

    # * 4. Save params to config file and run info file
    config_file = os.path.join(out, "config", "vk_clean_config.json")
    save_params_to_config_file(params_dict, config_file)

    run_info_file = os.path.join(out, "config", "vk_clean_run_info.txt")
    save_run_info(run_info_file, params_dict=params_dict, function_name="clean")

    # * 5. Set up default folder/file input paths, and make sure the necessary ones exist
    if kb_count_reference_genome_dir and not adata_reference_genome:
        adata_reference_genome = os.path.join(kb_count_reference_genome_dir, "counts_unfiltered", "adata.h5ad")

    vcrs_fasta = kwargs.get("vcrs_fasta", None)
    id_to_header_csv = kwargs.get("id_to_header_csv", None)
    variants_updated_csv = kwargs.get("variants_updated_csv", None)
    dlist_fasta = kwargs.get("dlist_fasta", None)

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
    vcf_out = os.path.join(out, "variants.vcf") if not vcf_out else vcf_out

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
    cosmic_tsv = kwargs.get("cosmic_tsv", None)
    cosmic_reference_genome_fasta = kwargs.get("cosmic_reference_genome_fasta", None)
    cosmic_version = kwargs.get("cosmic_version", 101)  #!! must match the downloadable version in vk ref
    cosmic_email = kwargs.get("cosmic_email", None)
    cosmic_password = kwargs.get("cosmic_password", None)
    forgiveness = kwargs.get("forgiveness", 100)
    add_hgvs_breakdown_to_adata_var = kwargs.get("add_hgvs_breakdown_to_adata_var", True)

    # * 7.5 make sure ints are ints
    if min_counts is None:
        min_counts = 0
    min_counts, threads = int(min_counts), int(threads)

    # * 8. Start the actual function
    if fastqs:
        try:
            fastqs = sort_fastq_files_for_kb_count(fastqs, technology=technology, multiplexed=kwargs.get("multiplexed"), check_only=(not sort_fastqs))
        except ValueError as e:
            if sort_fastqs:
                logger.warning(f"Automatic FASTQ argument order sorting for kb count could not recognize FASTQ file name format. Skipping argument order sorting.")

    if remove_doublets and not doublet_detection:
        logger.warning("remove_doublets is True, but doublet_detection is False. Setting doublet_detection to True.")
        doublet_detection = True

    if technology.lower() != "bulk" and "smartseq" not in technology.lower():
        parity = "single"
    else:  # bulk or SMARTSEQ
        if account_for_strand_bias:
            logger.warning("account_for_strand_bias is True, but technology is not a single-cell technology. Setting account_for_strand_bias to False.")
            account_for_strand_bias = False

    if account_for_strand_bias:
        if not strand_bias_end:
            strand_bias_end_possible_values = technology_to_strand_bias_mapping[technology.upper()]
            if strand_bias_end_possible_values is None or len(strand_bias_end_possible_values) != 1:
                raise ValueError(f"strand_bias_end must be provided if account_for_strand_bias=True and technology is {technology}. Possible values are {strand_bias_end_possible_values}.")
            strand_bias_end = strand_bias_end_possible_values[0]
            logger.info(f"Determined strand_bias_end to be {strand_bias_end} based on technology {technology}.")
        if not read_length:
            if not fastqs:
                raise ValueError("read_length must be provided if account_for_strand_bias=True and fastqs is not provided.")
            file_index_with_transcripts = technology_to_file_index_with_transcripts_mapping[technology.upper()]
            first_fastq_file_with_transcripts = fastqs[file_index_with_transcripts]
            first_fastq_file_with_transcripts_pyfastx = pyfastx.Fastx(first_fastq_file_with_transcripts)
            for _, seq, _ in first_fastq_file_with_transcripts_pyfastx:
                read_length = len(seq)
                break
            logger.info(f"Determined read_length to be {read_length} based on the fastq file {first_fastq_file_with_transcripts}.")

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
    gene_set_to_exclusively_keep = prepare_set(gene_set_to_exclusively_keep)
    gene_set_to_exclude = prepare_set(gene_set_to_exclude)

    adata = adata_vcrs

    if isinstance(adata, str) and os.path.exists(adata) and adata.endswith(".h5ad"):
        adata = ad.read_h5ad(adata)

    #$ As far as naming convention goes: (1) vcrs_header is the HGVS format name, (2) vcrs_id is the name in the actual index file (and thus in the BUS file and custom t2g etc), (3) vcrs_id_from_vk_ref is the ID if I used my custom ID_to_header_csv
    if adata.var.index[0].startswith("vcrs_"):
        adata.var["vcrs_id_from_vk_ref"] = adata.var.index
        if id_to_header_csv and isinstance(id_to_header_csv, str) and os.path.exists(id_to_header_csv):
            id_to_header_df = pd.read_csv(id_to_header_csv, index_col=0)
            adata.var = adata.var.merge(id_to_header_df, left_on="vcrs_id_from_vk_ref", right_on="vcrs_id", how="left")  # will add vcrs_header
            adata.var.set_index("vcrs_header", inplace=True)
        else:
            adata.var["vcrs_header"] = adata.var.index
    else:
        adata.var["vcrs_header"] = adata.var.index

    adata.var.index.name = "variant"
    original_var_names = adata.var_names.copy()

    adata.var.rename(columns={"vcrs_header": "vcrs_id"}, inplace=True)
    first_vcrs_header = adata.var["vcrs_id"].iloc[0].split(';')[0]
    if not re.fullmatch(HGVS_pattern_general, first_vcrs_header):
        pass  # deal with this later
    else:
        adata.var["vcrs_header"] = adata.var["vcrs_id"]

    if add_hgvs_breakdown_to_adata_var or variants is not None:
        #* work with column names
        if variants_usecols is not None:
            if not variants_usecols:  # eg False, [], set(), tuple()
                variants_usecols = []
            if isinstance(variants_usecols, str):
                variants_usecols = [variants_usecols]
            # ensure these important columns are present
            for column_to_add in [gene_id_column, seq_id_column, var_column, var_id_column]:
                if isinstance(column_to_add, str) and isinstance(variants_usecols, list) and column_to_add not in variants_usecols:
                    variants_usecols.append(column_to_add)

        #* explode the adata.var df and do some work
        adata_var_original_columns = adata.var.columns.tolist()
        
        adata.var['vcrs_id_individual'] = adata.var['vcrs_id'].str.split(';')  # don't let the naming be confusing - it will only be original AFTER exploding; before, it will be a list
        if not re.fullmatch(HGVS_pattern_general, first_vcrs_header):
            if variants is None:
                raise ValueError(f"The first vcrs_header '{first_vcrs_header}' does not match the expected HGVS format. Please provide variants as an input.")
        else:
            adata.var["vcrs_header_individual"] = adata.var["vcrs_id_individual"]
        adata_var_exploded = adata.var.copy().explode('vcrs_id_individual', ignore_index=True)
        
        if variants is not None:
            adata_var_exploded = merge_variants_with_adata(variants, adata_var_exploded, seq_id_column, var_column, var_id_column, variants_usecols=variants_usecols)
            first_vcrs_header = adata.var["vcrs_header"].iloc[0].split(';')[0]

        # TODO: tldr: ALL I NEED TO DO TO FIX THIS IS JUST SET variant_source = None AND ERASE THIS CHECK (I set up everything else below to work - the checks will take extra time, which is why I don't do it now, but it will be easy later) - I am assuming that the first vcrs_header_individual is representative of the variant source - but in cases where I combine genome and transcriptome, I should keep variant_source = None, do the identify_variant_source check below, and then evaluate my strand bias function and normal genome function to ensure they work correctly
        if ":c." in first_vcrs_header:
            variant_source = "transcriptome"
        elif ":g." in first_vcrs_header:
            variant_source = "genome"
        else:
            variant_source = None

        if add_hgvs_breakdown_to_adata_var:
            adata_var_exploded = add_information_from_variant_header_to_adata_var_exploded(adata_var_exploded, seq_id_column=seq_id_column, var_column=var_column, gene_id_column=gene_id_column, variant_source=variant_source, t2g_file=reference_genome_t2g)

        #* for unmerged instances of vcrs_header (which should be most of them), merge these into adata.var directly
        exclude_cols = {"vcrs_id_individual", "vcrs_header_individual"}
        columns_to_list = [col for col in adata_var_exploded.columns if (col not in adata_var_original_columns + ["vcrs_header", "vcrs_id"]) and col not in exclude_cols]
        columns_to_first = [col for col in adata_var_exploded.columns if (col in adata_var_original_columns + ["vcrs_header", "vcrs_id"]) and col not in exclude_cols]

        nonmerged_headers_mask = adata_var_exploded["vcrs_id"] == adata_var_exploded["vcrs_id_individual"]
        cols_to_use = [col for col in adata_var_exploded.columns if col not in exclude_cols]

        adata.var = adata.var.merge(adata_var_exploded.loc[nonmerged_headers_mask, cols_to_use], on='vcrs_id', how='left', suffixes=('', '_merged_tmp'))
        for col in adata.var.columns:
            if col in columns_to_list:
                mask = adata.var[col].notna()
                adata.var.loc[mask, col] = adata.var.loc[mask, col].map(lambda x: [x] if not isinstance(x, list) else x)

        #* collapse and re-merge the adata.var df
        grouped_var = (
                adata_var_exploded.loc[~nonmerged_headers_mask, cols_to_use].groupby("vcrs_id", as_index=False)
                .agg(
                    {
                        **{col: list for col in columns_to_list},  # merge variants columns into lists
                        **{col: "first" for col in columns_to_first},  # keep adata.var columns as-is
                    })
                .reset_index(drop=True)
            )
        adata.var = adata.var.merge(grouped_var, on='vcrs_id', how='left', suffixes=('', '_merged_tmp'))
        for col in adata.var.columns:
            if col in adata.var.columns and f"{col}_merged_tmp" in adata.var.columns and col != "vcrs_header":
                adata.var[col] = adata.var[col].combine_first(adata.var[f"{col}_merged_tmp"])  # fill na of that column with the merged values
                adata.var.drop(columns=[f"{col}_merged_tmp"], inplace=True)  # remove the merged column
        adata.var.drop(columns=[col for col in adata.var.columns if col.endswith('_merged_tmp')], inplace=True)  # ensure these columns are dropped
        del adata_var_exploded, grouped_var
    adata.var.drop(columns=["vcrs_header_individual", "vcrs_id_individual"], inplace=True, errors='ignore')

    if vcrs_metadata_df is not None:
        if isinstance(vcrs_metadata_df, pd.DataFrame) or isinstance(vcrs_metadata_df, (str, Path) and os.path.exists(vcrs_metadata_df)):
            if isinstance(vcrs_metadata_df, (str, Path)) and str(vcrs_metadata_df).endswith(".csv"):
                vcrs_metadata_df = pd.read_csv(vcrs_metadata_df, index_col=0)
            elif isinstance(vcrs_metadata_df, (str, Path)) and str(vcrs_metadata_df).endswith(".h5ad"):
                adata_tmp = ad.read_h5ad(vcrs_metadata_df, backed='r')
                vcrs_metadata_df = adata_tmp.var.to_dataframe()
            logger.info(f"vcrs_metadata_df file path {vcrs_metadata_df} exists. Merging with adata.var.")
            adata.var = adata.var.merge(vcrs_metadata_df, _on="vcrs_id", how="left")
        elif isinstance(vcrs_metadata_df, (str, Path) and not os.path.exists(vcrs_metadata_df)):
            logger.info(f"vcrs_metadata_df file path {vcrs_metadata_df} does not exist. Skipping merging with adata.var, and saving the current vcrs_metadata_df to csv (good if running this function in a loop, where it will be created upon the first iteration and then loaded in upon subsequent iterations).")
            adata.var.to_csv(vcrs_metadata_df, index=False)
        else:
            raise ValueError(f"vcrs_metadata_df must be a pandas DataFrame or a file path to a csv or h5ad file. Got {type(vcrs_metadata_df)} instead.")
        
    if gene_id_column in adata.var.columns and "vcrs_header" in adata.var.columns:
        # adata.var['vcrs_header_with_gene_name'] = ("(" + adata.var[gene_id_column].astype(str) + ")" + adata.var['vcrs_header'].astype(str))
        adata.var['vcrs_header_with_gene_name'] = (
            adata.var[gene_id_column].astype(str) + "(" +
            adata.var['vcrs_header'].str.split(":").str[0] + "):" +
            adata.var['vcrs_header'].str.split(":").str[1]
        )
    
    #* fixed the parity stuff
    if parity == "paired" and parity_kb_count == "single" and adata.uns.get("corrected_barcodes") is None:  # the last part is to ensure I didn't make the correction already
        barcodes_file = os.path.join(kb_count_vcrs_dir, "matrix.sample.barcodes")
        bad_to_good_barcode_dict = make_good_barcodes_and_file_index_tuples(barcodes_file)
        adata.obs.index = adata.obs.index.map(lambda x: bad_to_good_barcode_dict.get(x, x))  # map from old (incorrect) barcodes to new (correct) barcodes  #!!! ensure the old barcodes don't linger anywhere else

        # Convert to DataFrame for easy manipulation
        df = pd.DataFrame(adata.X.toarray() if hasattr(adata.X, "toarray") else adata.X, 
                  index=adata.obs.index, 
                  columns=adata.var.index)
        
        # sum across pairs
        df_grouped = df.groupby(df.index).sum()

        # Create new AnnData object
        adata = ad.AnnData(X=df_grouped.values, obs=pd.DataFrame(index=df_grouped.index), var=adata.var)  #!!! inspect this
        adata.uns["corrected_barcodes"] = True

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
        adata = adjust_variant_adata_by_normal_gene_matrix(adata, kb_count_vcrs_dir=kb_count_vcrs_dir, kb_count_reference_genome_dir=kb_count_reference_genome_dir, fastq_file_list=fastqs, technology=technology, t2g_standard=reference_genome_t2g, adata_output_path=None, mm=mm, parity=parity, bustools=bustools, check_only=(not sort_fastqs), save_type="parquet", count_reads_that_dont_pseudoalign_to_reference_genome=count_reads_that_dont_pseudoalign_to_reference_genome, variant_source=variant_source, gtf=gtf)
    
    # set all count values below min_counts to 0
    if min_counts is not None:
        adata.X = adata.X.multiply(adata.X >= min_counts)

    if account_for_strand_bias:
        adata = remove_variants_from_adata_for_stranded_technologies(adata=adata, strand_bias_end=strand_bias_end, read_length=read_length, header_column="vcrs_header", variant_source=variant_source, gtf=gtf, forgiveness=forgiveness, out=out)

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

    # # remove 0s for memory purposes
    # adata.X.eliminate_zeros()

    if use_binary_matrix:
        adata.X = (adata.X > 0).astype(int)

    if isinstance(adata_reference_genome, anndata.AnnData):
        adata_reference_genome = adata_reference_genome[adata.obs_names].copy()  # ensures adata_reference_genome rows/obs match order of adata
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

                violin_plot_path = os.path.join(output_figures_dir, "qc_violin_plot.pdf")
                os.rename(
                    os.path.join("figures", "violin.pdf"),
                    violin_plot_path,
                )

                if os.path.isdir("figures") and len(os.listdir("figures")) == 0:
                    os.rmdir("figures")

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
            var_column_name="vcrs_id",
        )

    if vcrs_id_set_to_exclude:
        adata = remove_adata_columns(
            adata,
            values_of_interest=vcrs_id_set_to_exclude,
            operation="exclude",
            var_column_name="vcrs_id",
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
    adata.var["vcrs_count"] = adata.var["vcrs_count"].fillna(0).astype("Int32")
    adata.var["vcrs_detected"] = adata.var["vcrs_count"] > 0

    if gene_id_column in adata.var:
        filtered_var = adata.var[adata.var["vcrs_count"] > 0]
        gene_counts = filtered_var[gene_id_column].explode().value_counts()
        gene_counts_dict = gene_counts.to_dict()
        adata.var["gene_count"] = adata.var[gene_id_column].apply(lambda gene_list: [gene_counts_dict.get(g, 0) for g in gene_list])
        del filtered_var

    if sp.issparse(adata.X):
        # Sparse matrix handling
        adata.var["number_obs"] = np.array((adata.X != 0).sum(axis=0)).flatten()
    else:
        # Dense matrix handling
        adata.var["number_obs"] = (adata.X != 0).sum(axis=0)

    adata.var["number_obs"] = adata.var["number_obs"].astype("Int32")

    if save_vcf:
        if not vcf_data_csv or not os.path.exists(vcf_data_csv):
            if variants == "cosmic_cmc":
                vcf_data_csv = add_vcf_info_to_cosmic_tsv(cosmic_tsv=cosmic_tsv, reference_genome_fasta=cosmic_reference_genome_fasta, cosmic_df_out=vcf_data_csv, sequences=sequences, cosmic_version=cosmic_version, cosmic_email=cosmic_email, cosmic_password=cosmic_password)
            # insert other supported databases here
            else:
                raise ValueError("vcf_data_csv must be provided if variants is supported. Supported databases can be viewed with `vk ref --list_prebuilt_indices`.")
        cleaned_adata_to_vcf(adata.var, vcf_data_df=vcf_data_csv, output_vcf=vcf_out, save_vcf_samples=save_vcf_samples, adata=adata)

    #* adata can't save lists in var columns, so we need to convert them to semicolon-separated strings - later get back with something like adata.var[col] = adata.var[col].apply(lambda x: x.split(";") if isinstance(x, str) else x)
    for adata_object in [adata, adata_reference_genome]:
        if isinstance(adata_object, anndata.AnnData):
            for col in adata_object.var.columns:
                adata_object.var[col] = adata_object.var[col].apply(lambda x: ";".join(map(str, x)) if isinstance(x, list) else x).astype(str)
    
    if isinstance(adata_reference_genome, anndata.AnnData):
        adata_reference_genome.write(adata_reference_genome_clean_out)

    adata.write(adata_vcrs_clean_out)

    return adata