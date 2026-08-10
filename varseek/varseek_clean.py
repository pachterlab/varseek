"""varseek clean and specific helper functions."""

import json
import logging
import os
import scipy.sparse as sp
from scipy.io import mmread
import subprocess
import pyfastx
import time
from pathlib import Path
from typing import Optional, Union
import re

import anndata
import anndata as ad
import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, model_validator

from varseek.utils import (
    validate_call,
    vk_config,
    PositiveInt,
    NonNegativeInt,
    Parity,
    StrandBiasEnd,
    Technology,
    adjust_variant_adata_by_pseudobam,
    check_file_path_is_string_with_valid_extension,
    decrement_adata_matrix_when_split_by_Ns_or_running_paired_end_in_single_end_mode,
    is_valid_int,
    load_in_fastqs,
    load_adata_from_mtx,
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
    plot_cdna_locations,
    save_df_types_adata,
    correct_adata_barcodes_for_running_paired_data_in_single_mode,
    safe_literal_eval,
    load_adata_from_mtx,
    join_list_columns_for_h5ad
)

from .constants import non_single_cell_technologies, technology_valid_values, technology_info, HGVS_pattern_general, mutation_pattern

from .utils.varseek_clean_utils import prepare_set, _validate_clean_params

logger = logging.getLogger(__name__)
logger = set_up_logger(logger, logging_level="INFO", save_logs=False, log_dir=None)


class CleanParams(BaseModel):
    """Cross-field and filesystem validation for :func:`clean`.

    Per-parameter type/extension checks are enforced on the ``clean`` signature by
    ``@validate_call``; the detailed cross-field, file-path, directory, and
    kwargs-only checks run here via ``_validate_clean_params`` (fed the full params
    dict, including kwargs, via ``extra="allow"``).
    """

    model_config = ConfigDict(extra="allow", arbitrary_types_allowed=True)

    @model_validator(mode="after")
    def _validate(self):
        _validate_clean_params(dict(self.model_extra or {}))
        return self


needs_for_normal_genome_matrix = ["filter_cells_by_min_counts", "filter_cells_by_min_genes", "filter_genes_by_min_cells", "filter_cells_by_max_mt_content", "doublet_detection", "cpm_normalization"]

# Advanced `clean` parameters: still accepted on the command line (and in the Python signature,
# where @validate_call validates them), but hidden from `vk clean --help` to keep it uncluttered.
# Consumed by main.py, which flips each matching argparse action's help to SUPPRESS. Matched by dest.
vk_clean_hidden_from_help = {
    "variants_updated_dataframe",
    "bustools",
    "parity_kb_count",
    "cosmic_tsv",
    "cosmic_reference_genome_fasta",
    "cosmic_version",
    "forgiveness",
    "add_hgvs_breakdown_to_adata_var",
}

# Documentation for the advanced `clean` parameters. These are real, fully typed
# arguments on the `clean` signature (validated by @validate_call like every other
# parameter), but they are deliberately kept out of the public `clean` docstring
# (and therefore out of help(vk.clean) / the Sphinx site) and out of `vk clean --help`
# (via vk_clean_hidden_from_help) to keep the common interface uncluttered. The text
# lives here so maintainers still have it close to the definition.
CLEAN_ADVANCED_PARAMS_DOC = """
# # Advanced parameters (real `clean` arguments, but hidden from the `vk clean --help` CLI):
- variants_updated_dataframe                  (str) Path to the updated variants csv file. Only used if using downloaded reference files from vk ref and vcf_data_dataframe does not exist. Default: None.
- bustools                              (str) Path to the bustools binary. Default: None.
- parity_kb_count                       (str) The parity of the reads used in kb count when generating adata_vcrs. Default: `parity`.
- cosmic_tsv                            (str) Path to the cosmic tsv file. Only used if creating a VCF file and using a downloaded reference from vk ref and vcf_data_dataframe does not exist. Default: None.
- cosmic_reference_genome_fasta         (str) Path to the reference genome fasta file. Only used if creating a VCF file and using a downloaded reference from vk ref and vcf_data_dataframe does not exist. Default: None.
- cosmic_version                        (str) Version of the cosmic tsv file. Only used if creating a VCF file and using a downloaded reference from vk ref and vcf_data_dataframe does not exist. Default: 101.
                                        NOTE: COSMIC download credentials are read only from the COSMIC_EMAIL and COSMIC_PASSWORD environment variables (never written to the on-disk run config); there are no cosmic_email/cosmic_password arguments.
- forgiveness                           (int) Number of bases allowed to be off when account_for_strand_bias=True. E.g., if I am using a 5' technology with a read length of 91 and a forgiveness of 100, then I will keep only variants that fall within the last 191 bases of each respective transcript. Default: 100.
- add_hgvs_breakdown_to_adata_var       (bool) Whether to add the HGVS breakdown of the variants in separate columns of adata.var including nucleotide positions, variant, variant start position, variant end position, and gene name. Default: True.
"""

# @profile
@validate_call(config=vk_config)
def clean(
    adata_vcrs: object,  # required inputs
    technology: Optional[Technology],
    min_counts: PositiveInt = 2,  # parameters
    use_binary_matrix: bool = False,
    drop_empty_columns: bool = False,
    pseudobam_validation: bool = False,
    pseudobam_validation_aligner: str = "pseudo",
    reference_genome_index: Optional[Union[str, Path]] = None,
    reference_bam: object = None,
    check_alignment_position: bool = False,
    alignment_position_tolerance: int = 50,
    reference_sequences_type: Optional[str] = None,
    reads_type: Optional[str] = None,
    kallisto: Optional[str] = None,
    threads: int = 2,
    count_reads_that_dont_pseudoalign_to_reference_genome: bool = True,
    avoid_paired_double_counting: bool = False,
    account_for_strand_bias: bool = False,
    strand_bias_end: Optional[StrandBiasEnd] = None,
    read_length: Optional[PositiveInt] = None,
    filter_cells_by_min_counts: Optional[Union[bool, int]] = None,
    filter_cells_by_min_genes: Optional[NonNegativeInt] = None,
    filter_genes_by_min_cells: Optional[NonNegativeInt] = None,
    filter_cells_by_max_mt_content: Optional[int] = None,
    doublet_detection: bool = False,
    remove_doublets: bool = False,
    cpm_normalization: bool = False,
    sum_rows: bool = False,
    drop_multi_variant_vcrs: bool = False,
    rename_vcrs_to_variant: bool = True,
    vcrs_id_set_to_exclusively_keep: object = None,
    vcrs_id_set_to_exclude: object = None,
    gene_set_to_exclusively_keep: object = None,
    gene_set_to_exclude: object = None,
    mm: bool = True,
    parity: Parity = "single",
    multiplexed: Optional[bool] = None,
    sort_fastqs: bool = True,
    adata_reference_genome: object = None,  # optional inputs
    fastqs: object = None,
    vk_ref_dir: Optional[Union[str, Path]] = None,
    vcrs_index: Optional[Union[str, Path]] = None,
    vcrs_t2g: Optional[Union[str, Path]] = None,
    gtf: Optional[Union[bool, str, Path]] = None,
    kb_count_vcrs_dir: Optional[Union[str, Path]] = None,
    kallisto_quant_reference_genome_dir: Optional[Union[str, Path]] = None,
    reference_genome_t2g: Optional[Union[str, Path]] = None,
    vcf_data_dataframe: Optional[Union[str, Path]] = None,
    variants: object = None,
    sequences: object = None,
    variant_source: Optional[str] = None,
    vcrs_metadata_df: object = None,
    variants_usecols: Optional[Union[str, list]] = None,
    seq_id_column: str = "seq_ID",
    var_column: str = "mutation",
    var_id_column: Optional[str] = None,
    gene_id_column: str = "gene_id",
    out: Union[str, Path] = ".",  # output paths
    adata_vcrs_clean_out: Optional[Union[str, Path]] = None,
    adata_reference_genome_clean_out: Optional[Union[str, Path]] = None,
    vcf_out: Optional[Union[str, Path]] = None,
    save_vcf: bool = False,  # optional saves
    save_vcf_samples: bool = False,
    chunksize: Optional[int] = None,
    dry_run: bool = False,  # general
    overwrite: bool = False,
    logging_level: Optional[Union[str, int]] = None,
    save_logs: bool = False,
    log_out_dir: Optional[Union[str, Path]] = None,
    # --- Advanced parameters: part of the Python signature (validated by @validate_call) but hidden from `vk clean --help`. See the "Advanced parameters" docstring section. ---
    variants_updated_dataframe: Optional[Union[str, Path]] = None,
    bustools: Optional[str] = None,
    parity_kb_count: Optional[Parity] = None,
    cosmic_tsv: Optional[Union[str, Path]] = None,
    cosmic_reference_genome_fasta: Optional[Union[str, Path]] = None,
    cosmic_version: Union[str, int] = 101,
    forgiveness: int = 100,
    add_hgvs_breakdown_to_adata_var: bool = True,
):
    """
    Apply quality control to the VCRS count matrix (cell/sample x variant) and save the cleaned AnnData object.

    # Required input arguments:
    - adata_vcrs                            (str or anndata.AnnData) The input AnnData object containing the VCRS data.
    - technology                            (str)  Technology used to generate the data. To see list of supported technologies, run `kb --list`.

    # Additional parameters
    - min_counts                            (int) Minimum counts to consider valid in the VCRS count matrix - everything below this number gets set to 0. Default: 2.
    - use_binary_matrix                     (bool) Whether to binarize the matrix (i.e., set all values >=1 to 1). Default: False.
    - drop_empty_columns                    (bool) Whether to drop columns (variants) that are empty across all samples. Default: False.
    - pseudobam_validation                  (bool) Whether to validate the VCRS count matrix against the reference genome by re-aligning the reads to it. Each read's true alignment coordinate is compared against the HGVS locus of its assigned VCRS; reads that aligned somewhere inconsistent with the variant are dropped and the VCRS count matrix is regenerated. Requires `fastqs`, plus either `reference_genome_index` (or `sequences` to build it) or a pre-made `reference_bam`. Default: False.
    - pseudobam_validation_aligner          (str) How the validation alignments are produced when pseudobam_validation=True. "pseudo" uses `kallisto quant --pseudobam` (or `--genomebam` when RNA reads are aligned against a DNA/genome reference) - fast, and reuses a kallisto index. "true" uses a real aligner: splice-aware STAR when RNA reads are aligned against a genome reference (the reads span exon-exon junctions), bowtie2 otherwise - slower and heavier, but a genuine alignment. "STAR"/"bowtie2" request one explicitly and raise if it disagrees with the read/reference combination. Ignored when `reference_bam` is given. Default: "pseudo".
    - reference_genome_index                (str) Path to the standard reference index used for pseudobam validation: a kallisto index for pseudobam_validation_aligner="pseudo", the STAR genome directory for STAR, or the bowtie2 index prefix for bowtie2. Passing an index of the wrong kind for the chosen aligner (or for the read/reference combination) raises. If the path does not exist, it is built from `sequences`: kallisto builds a cDNA index from the genome + `gtf` with `kb ref --workflow standard` when RNA reads are aligned against a DNA (genome) reference and indexes `sequences` directly (`kb ref --workflow custom`) otherwise; STAR builds a genome index from `sequences` + `gtf`; bowtie2 builds one from `sequences`. Not needed when `reference_bam` is given. Default: None.
    - reference_bam                         (str or list[str]) Path to a BAM (or list of BAMs) of the same reads already aligned against the standard reference - e.g. the BAM produced by vk denovo. When given, no re-alignment is run and `reference_genome_index`/`pseudobam_validation_aligner` are ignored. Pass either one BAM covering all reads, or one BAM per sample in the same order as `fastqs`. Only used when pseudobam_validation=True. Default: None.
    - check_alignment_position              (bool) When pseudobam_validation=True, also require each read's alignment position to fall within `alignment_position_tolerance` of its assigned variant's position (in addition to matching the sequence/reference name). Heavier but more accurate. Default: False.
    - alignment_position_tolerance          (int) Position tolerance (bp) used when check_alignment_position=True. Default: 50.
    - reference_sequences_type              (str) "rna" or "dna": whether the VCRS reference was built from transcriptome (RNA) or genome (DNA) sequences. Used (with reads_type) to pick the pseudobam alignment strategy; inferred from `sequences` when not provided. Default: None.
    - reads_type                            (str) "rna" or "dna": the sequencing molecule of the reads. Normally derived from `technology` (the "dna" technology => "dna", any RNA technology => "rna"); pass explicitly to override. RNA reads against a DNA/genome reference trigger the `kb ref --workflow standard` + `kallisto quant --genomebam` path. When None, it defaults to `reference_sequences_type` (reads assumed to match the reference). Default: None.
    - kallisto                              (str) Path to the kallisto binary (must support --pseudobam/--genomebam). Resolved from `kb info` when not provided. Only used for pseudobam_validation=True. Default: None.
    - threads                               (int) Number of threads for kallisto quant / bustools. Only used for pseudobam_validation=True. Default: 2.
    - count_reads_that_dont_pseudoalign_to_reference_genome (bool) Whether to count reads that don't pseudoalign to the reference genome. Only used when pseudobam_validation=True. Default: True.
    - avoid_paired_double_counting          (bool) For paired-end BULK/SMARTSEQ2 data run through kb count in single-end mode, count each VCRS at most once per fragment instead of once per mate (the first mate to carry a variant keeps it; the second mate has it removed). Only used when pseudobam_validation=True and the data is paired-end run in single mode. Default: False.
    - account_for_strand_bias               (bool) Whether to account for strand bias from stranded single-cell technologies. Default: False.
    - strand_bias_end                       (str) The end of the read to use for strand bias correction. Either "5p" or "3p". Must be provided if and only if account_for_strand_bias=True. Default: None.
    - read_length                           (int) The read length used in the experiment. Must be provided if and only if account_for_strand_bias=True. Default: None.
    - filter_cells_by_min_counts            (int or bool or None) Part of the QC performed on the **gene** count matrix with which to adjust the **variant** count matrix. Minimum number of gene counts per cell to keep the cell. If True, will use kneed's KneeLocator to determine the cutoff. Default: None.
    - filter_cells_by_min_genes             (int or None) Part of the QC performed on the **gene** count matrix with which to adjust the **variant** count matrix. Minimum number of genes per cell to keep the cell. Default: None.
    - filter_genes_by_min_cells             (int or None) Part of the QC performed on the **gene** count matrix with which to adjust the **variant** count matrix. Minimum number of cells per gene to keep the gene. Default: None.
    - filter_cells_by_max_mt_content        (int or None) Part of the QC performed on the **gene** count matrix with which to adjust the **variant** count matrix. Maximum percentage of mitochondrial content per cell to keep the cell. Default: None.
    - doublet_detection                     (bool) Part of the QC performed on the **gene** count matrix with which to adjust the **variant** count matrix. Whether to run doublet detection. Default: False.
    - remove_doublets                       (bool) Part of the QC performed on the **gene** count matrix with which to adjust the **variant** count matrix. Whether to remove doublets. Default: False.
    - cpm_normalization                     (bool) Part of the QC performed on the **gene** count matrix with which to adjust the **variant** count matrix. Whether to run cpm normalization. Default: False.
    - sum_rows                              (bool) Whether to sum across barcodes (rows) in the VCRS count matrix. Default: False.
    - drop_multi_variant_vcrs               (bool) Whether to drop merged VCRS entries that represent multiple variants (i.e., those whose vcrs_id contains a semicolon) from the cleaned AnnData object. Default: False.
    - rename_vcrs_to_variant                (bool) Whether to rename all "vcrs" var columns to "variant" (e.g. vcrs_count -> variant_count, vcrs_detected -> variant_detected) in the cleaned AnnData object. The raw vcrs_id column is left unchanged. Default: True.
    - vcrs_id_set_to_exclusively_keep       (str or Set(str) or None) If a set, will keep only the VCRSs in this set. If a list/tuple, will convert to a set and then keep only the VCRSs in this set. If a string, will load the text file and keep only the VCRSs in this set. Default: None.
    - vcrs_id_set_to_exclude                (str or Set(str) or None) If a set, will exclude the VCRSs in this set. If a list/tuple, will convert to a set and then exclude the VCRSs in this set. If a string, will load the text file and exclude the VCRSs in this set. Default: None.
    - gene_set_to_exclusively_keep          (str or Set(str) or None) If a set, will keep only the genes in this set. If a list/tuple, will convert to a set and then keep only the genes in this set. If a string, will load the text file and keep only the genes in this set. Default: None.
    - gene_set_to_exclude                   (str or Set(str) or None) If a set, will exclude the genes in this set. If a list/tuple, will convert to a set and then exclude the genes in this set. If a string, will load the text file and exclude the genes in this set. Default: None.
    - mm                                    (bool) Whether to count multimapped reads in the adata count matrix. Only used when pseudobam_validation is True. Default: True.
    - parity                                (str) "single" or "paired". Only relevant if technology is bulk or a smart-seq. Default: "single"
    - multiplexed                           (bool) Indicates that the fastq files are multiplexed. Only used if sort_fastqs=True and technology is a smartseq technology. Default: None
    - sort_fastqs                           (bool) Whether to sort the fastqs. Default: True.

    # Optional input arguments:
    - adata_reference_genome                (str) Path to the reference genome AnnData object. Default: `kallisto_quant_reference_genome_dir`/counts_unfiltered/adata.h5ad.
    - fastqs                                (str or list[str]) List of fastq files to be processed. If paired end, the list should contains paths such as [file1_R1, file1_R2, file2_R1, file2_R2, ...]. Only used when `pseudobam_validation` is True. Default: None
    - vk_ref_dir                            (str) Directory containing the VCRS reference files. Same as `out` as specified in vk ref. Default: None.
    - vcrs_index                            (str) Path to the VCRS index file. Default: None.
    - vcrs_t2g                              (str) Path to the VCRS t2g file. Default: None.
    - gtf                                   (str) Path to the GTF file. Only used when account_for_strand_bias=True and either (1) strand_bias_end='3p' and/or (2) some VCRSs are derived from genome sequences. Default: None.
    - kb_count_vcrs_dir                     (str) Path to the kb count output directory for the VCRS reference. Default: None.
    - kallisto_quant_reference_genome_dir         (str) Path to the kb count output directory for the reference genome. Default: None.
    - reference_genome_t2g                  (str) Path to the t2g file for the standard reference genome. Only used for the optional HGVS breakdown of variant headers (add_hgvs_breakdown_to_adata_var=True). Default: None.
    - vcf_data_dataframe                          (str) Path to the VCF data csv file. It needs columns ID (corresponding to vcrs_id from adata.var), CHROM, POS, REF, ALT. If using downloaded reference files from vk ref, then simply provide the `variants` and `sequences` arguments entered at vk ref for this file to be created internally. Default: None.
    - variants                              (str) The variants parameter from vk ref/build. Merged into adata.var with columns variants_usecols. Only strictly needed if using downloaded reference files from vk ref and wanting to save an output VCF file. Default: None.
    - sequences                             (str) The sequences parameter from vk ref/build. Only needed if using downloaded reference files from vk ref and wanting to save an output VCF file. Default: None.
    - variant_source                        (str) The source of the variants. If not provided, it will be inferred from the `variants` parameter. Can be "genome" or "transcriptome". Default: None
    - vcrs_metadata_df                      (str) Path to the VCRS metadata dataframe. Will be merged into adata.var. If a path is provided and the file does not exist, it will be created. Default: None.

    # Optional column names in variants
    - variants_usecols                      (str or list) Columns in the variants to merge with the adata var. Default: None (use all columns).
    - seq_id_column                         (str) Column name in the variants that contains the sequence ID. The same parameter as in vk ref/build. This is used to merge the sequences with the variants. Default: "seq_ID".
    - var_column                            (str) Column name in the variants that contains the variant information. The same parameter as in vk ref/build. This is used to merge the sequences with the variants. Default: "mutation".
    - var_id_column                         (str) Column name in the variants that contains the variant ID. The same parameter as in vk ref/build. This is used to merge the sequences with the variants. Default: None (will be created as seq_id_column:var_column).
    - gene_id_column                        (str) Column name in the adata var that contains the gene ID. Default: "gene_id".

    # Output paths:
    - out                                   (str) Output directory. Default: ".".
    - adata_vcrs_clean_out                  (str) Path to save the cleaned VCRS AnnData object. Default: `out`/adata_cleaned.h5ad.
    - adata_reference_genome_clean_out      (str) Path to save the cleaned reference genome AnnData object. Default: `out`/adata_reference_genome_cleaned.h5ad.
    - save_vcf                              (bool) Whether to save the VCF file. Default: False.
    - save_vcf_samples                      (bool) Whether to save samples in the VCF. If many samples (including single-cell), can lead to very large files. Default: False.
    - vcf_out                               (str) Path to save the VCF file. Default: `out`/variants.vcf.

    # General:
    - chunksize                             (int) Number of BUS file lines to process at a time. If None, then all lines will be processed at once. Default: None.
    - dry_run                               (bool) Whether to run in dry run mode. Default: False.
    - overwrite                             (bool) Whether to overwrite existing files. Default: False.
    - logging_level                         (str) Logging level. Can also be set with the environment variable VARSEEK_LOGGING_LEVEL. Default: INFO.
    - save_logs                             (True/False) Whether to save logs to a file. Default: False.
    - log_out_dir                           (str) Directory to save logs. Default: `out`/logs

    NOTE: `clean` also accepts several advanced/niche parameters that are intentionally omitted here (and from
    `vk clean --help`). They are fully typed and validated like any other argument; see CLEAN_ADVANCED_PARAMS_DOC
    in varseek/varseek_clean.py for their documentation.
    """
    start_time = time.perf_counter()

    # * 1. logger
    if save_logs and not log_out_dir:
        log_out_dir = os.path.join(out, "logs")
    set_varseek_logging_level_and_filehandler(logging_level=logging_level, save_logs=save_logs, log_dir=log_out_dir)


    # * 1.5 load in fastqs
    fastqs_original = fastqs
    if fastqs:
        fastqs = load_in_fastqs(fastqs)  # this will make it in params_dict

    # * 2. Type-checking (per-parameter types enforced by @validate_call; cross-field + filesystem by CleanParams)
    params_dict = make_function_parameter_to_value_dict(1)
    CleanParams(**params_dict)
    params_dict["fastqs"] = fastqs_original  # change back for dry run and config_file

    if isinstance(adata_vcrs, (str, Path)) and not os.path.isfile(adata_vcrs) and not dry_run:  # only use os.path.isfile when I require that a directory already exists; checked outside validate_input_clean to avoid raising issue when type-checking within vk count
        raise ValueError(f"adata_vcrs file path {adata_vcrs} does not exist.")

    # * 3. Dry-run and set out folder (must to it up here or else config will save in the wrong place)
    if dry_run:
        print(get_varseek_dry_run(params_dict, function_name="clean"))
        report_time_elapsed(start_time, "clean")
        return None

    # * 4. Save params to config file and run info file
    config_file = os.path.join(out, "config", "vk_clean_config.json")
    save_params_to_config_file(params_dict, config_file)

    run_info_file = os.path.join(out, "config", "vk_clean_run_info.txt")
    save_run_info(run_info_file, params_dict=params_dict, function_name="clean")

    # The virtual "dna" technology behaves exactly like bulk for kb count / barcode handling, but
    # additionally declares that the reads are genomic DNA. Derive the reads-type signal (used only by
    # the pseudobam QC to pick --pseudobam vs --genomebam) and normalize the technology to bulk so all
    # downstream bulk logic applies unchanged. When reads_type is not given (e.g. vk count already
    # derived it, or a caller left it unset), infer it from the technology: any RNA technology => "rna".
    if reads_type is None and technology is not None:
        reads_type = "dna" if str(technology).upper() == "DNA" else "rna"
    if technology is not None and str(technology).upper() == "DNA":
        technology = "bulk"

    # * 5. Set up default folder/file input paths, and make sure the necessary ones exist
    if kallisto_quant_reference_genome_dir and not adata_reference_genome:
        adata_reference_genome = os.path.join(kallisto_quant_reference_genome_dir, "counts_unfiltered", "adata.h5ad")

    if vk_ref_dir and os.path.exists(vk_ref_dir):  # make sure all of the defaults below match vk info/filter
        vcrs_index = os.path.join(vk_ref_dir, "vcrs_index.idx") if not vcrs_index else vcrs_index
        if not vcrs_t2g:
            vcrs_t2g = os.path.join(vk_ref_dir, "vcrs_t2g_filtered.txt") if os.path.isfile(os.path.join(vk_ref_dir, "vcrs_t2g_filtered.txt")) else os.path.join(vk_ref_dir, "vcrs_t2g.txt")
        if not variants_updated_dataframe:
            variants_updated_dataframe = os.path.join(vk_ref_dir, "variants_updated_filtered.csv") if os.path.isfile(os.path.join(vk_ref_dir, "variants_updated_filtered.csv")) else os.path.join(vk_ref_dir, "variants_updated.csv")

    if pseudobam_validation:
        if not kb_count_vcrs_dir or not os.path.exists(kb_count_vcrs_dir) or len(os.listdir(kb_count_vcrs_dir)) == 0:
            raise ValueError("kb_count_vcrs_dir must be provided as the output from kb count out to the VCRS reference if pseudobam_validation is True.")
        if reference_bam:  # the alignments are already made -- no index needed
            if reference_genome_index:
                logger.warning("Both reference_bam and reference_genome_index were provided; reference_bam takes precedence and no re-alignment is run.")
        else:
            if not reference_genome_index:
                # Only the kallisto path has a conventional default location/name; the STAR/bowtie2
                # indexes are named and defaulted inside adjust_variant_adata_by_pseudobam, which is
                # where the STAR-vs-bowtie2 decision is actually made.
                if pseudobam_validation_aligner.lower() in {"pseudo", "kallisto", "pseudobam", "genomebam"} and kallisto_quant_reference_genome_dir:
                    logger.warning("reference_genome_index was not provided. Attempting to use the default index from kallisto_quant_reference_genome_dir.")
                    reference_genome_index = os.path.join(kallisto_quant_reference_genome_dir, "index.idx")
            if (not reference_genome_index or not os.path.exists(reference_genome_index)) and (not sequences or not os.path.exists(str(sequences))):
                raise ValueError(f"reference_genome_index '{reference_genome_index}' does not exist and cannot be built because `sequences` was not provided (or does not exist). Provide `sequences` (the reference fasta the VCRSs were built from: the transcriptome cDNA for HGVSc variants or the genome for HGVSg variants), point reference_genome_index at a prebuilt index, or pass a pre-made reference_bam.")
        if not fastqs:
            raise ValueError("fastqs must be provided when pseudobam_validation=True (the reads must be matched to the alignments by FASTQ header).")

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

    # * 7. Resolve advanced-parameter defaults
    if parity_kb_count is None:
        parity_kb_count = parity  # default parity_kb_count to the value of parity

    try:
        with open(f"{kb_count_vcrs_dir}/kb_info.json", 'r') as f:
            kb_info_data_vcrs = json.load(f)
        parity_kb_count = "paired" if kb_info_data_vcrs.get("call", "").endswith("paired") else "single"
    except Exception as e:
        pass

    # * 7.5 make sure ints are ints
    if min_counts is None:
        min_counts = 0
    min_counts = int(min_counts)

    # * 8. Start the actual function
    if fastqs:
        try:
            fastqs = sort_fastq_files_for_kb_count(fastqs, technology=technology, multiplexed=multiplexed, check_only=(not sort_fastqs))
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
            strand_bias_end_possible_values = technology_info[technology.upper()]["strand_bias"]
            if strand_bias_end_possible_values is None or len(strand_bias_end_possible_values) != 1:
                raise ValueError(f"strand_bias_end must be provided if account_for_strand_bias=True and technology is {technology}. Possible values are {strand_bias_end_possible_values}.")
            strand_bias_end = strand_bias_end_possible_values[0]
            logger.info(f"Determined strand_bias_end to be {strand_bias_end} based on technology {technology}.")
        if not read_length:
            if not fastqs:
                raise ValueError("read_length must be provided if account_for_strand_bias=True and fastqs is not provided.")
            file_index_with_transcripts = technology_info[technology.upper()]["transcript_file_index"]
            first_fastq_file_with_transcripts = fastqs[file_index_with_transcripts]
            first_fastq_file_with_transcripts_pyfastx = pyfastx.Fastx(first_fastq_file_with_transcripts)
            for _, seq, _ in first_fastq_file_with_transcripts_pyfastx:
                read_length = len(seq)
                break
            logger.info(f"Determined read_length to be {read_length} based on the fastq file {first_fastq_file_with_transcripts}.")

    if not bustools:
        bustools_binary_path_command = "kb info | grep 'bustools:' | awk '{print $3}' | sed 's/[()]//g'"
        bustools = subprocess.run(bustools_binary_path_command, shell=True, executable="/bin/bash", stdout=subprocess.PIPE, text=True, check=True).stdout.strip()

    vcrs_id_set_to_exclusively_keep = prepare_set(vcrs_id_set_to_exclusively_keep)
    vcrs_id_set_to_exclude = prepare_set(vcrs_id_set_to_exclude)
    gene_set_to_exclusively_keep = prepare_set(gene_set_to_exclusively_keep)
    gene_set_to_exclude = prepare_set(gene_set_to_exclude)

    adata = adata_vcrs

    if isinstance(adata, str) and os.path.exists(adata):
        if adata.endswith(".h5ad"):
            adata = ad.read_h5ad(adata)
        elif adata.endswith(".mtx"):                
            adata = load_adata_from_mtx(adata)

    #$ vcrs_id is the concatenated HGVS-format header (semicolon-separated for merged VCRSs), taken directly from the adata var index. There is no longer a separate vcrs_header / vcrs_id_from_vk_ref column - the index IS both the id and the header.
    adata.var["vcrs_id"] = adata.var.index
    adata.var.index.name = "variant"
    original_var_names = adata.var_names.copy()
    first_vcrs_header = adata.var["vcrs_id"].iloc[0].split(';')[0]

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
        adata_var_exploded = adata.var.copy().explode('vcrs_id_individual', ignore_index=True)
        if not re.fullmatch(HGVS_pattern_general, first_vcrs_header) and variants is None:
            raise ValueError(f"The first vcrs_id '{first_vcrs_header}' does not match the expected HGVS format. Please provide variants as an input.")

        if variants is not None:
            adata_var_exploded = merge_variants_with_adata(variants, adata_var_exploded, seq_id_column, var_column, var_id_column, variants_usecols=variants_usecols)
            adata_var_exploded.drop(columns=["vcrs_header"], inplace=True, errors='ignore')  # merge_variants_with_adata joins on a synthesized vcrs_header - it is redundant with vcrs_id_individual now that the id is the HGVS header

        # TODO: tldr: ALL I NEED TO DO TO FIX THIS IS JUST SET variant_source = None AND ERASE THIS CHECK (I set up everything else below to work - the checks will take extra time, which is why I don't do it now, but it will be easy later) - I am assuming that the first vcrs_id_individual is representative of the variant source - but in cases where I combine genome and transcriptome, I should keep variant_source = None, do the identify_variant_source check below, and then evaluate my strand bias function and normal genome function to ensure they work correctly
        if ":c." in first_vcrs_header:
            variant_source = "transcriptome"
        elif ":g." in first_vcrs_header:
            variant_source = "genome"
        else:
            variant_source = None

        if add_hgvs_breakdown_to_adata_var:
            adata_var_exploded = add_information_from_variant_header_to_adata_var_exploded(adata_var_exploded, vcrs_header_individual_column="vcrs_id_individual", seq_id_column=seq_id_column, var_column=var_column, gene_id_column=gene_id_column, variant_source=variant_source, t2g_file=reference_genome_t2g)

        #* drop duplicates (will mess up merges below otherwise)
        adata_var_exploded = adata_var_exploded.drop_duplicates(subset=["vcrs_id_individual"], keep="first")

        #* for unmerged instances of vcrs_id (which should be most of them), merge these into adata.var directly
        exclude_cols = {"vcrs_id_individual"}
        columns_to_list = [col for col in adata_var_exploded.columns if (col not in adata_var_original_columns + ["vcrs_id"]) and col not in exclude_cols]
        columns_to_first = [col for col in adata_var_exploded.columns if (col in adata_var_original_columns + ["vcrs_id"]) and col not in exclude_cols]

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
            if col in adata.var.columns and f"{col}_merged_tmp" in adata.var.columns:
                adata.var[col] = adata.var[col].combine_first(adata.var[f"{col}_merged_tmp"])  # fill na of that column with the merged values
                adata.var.drop(columns=[f"{col}_merged_tmp"], inplace=True)  # remove the merged column
        adata.var.drop(columns=[col for col in adata.var.columns if col.endswith('_merged_tmp')], inplace=True)  # ensure these columns are dropped
        del adata_var_exploded, grouped_var
    adata.var.drop(columns=["vcrs_id_individual"], inplace=True, errors='ignore')

    if vcrs_metadata_df is not None:
        if isinstance(vcrs_metadata_df, pd.DataFrame) or (isinstance(vcrs_metadata_df, (str, Path)) and os.path.exists(vcrs_metadata_df)):
            if isinstance(vcrs_metadata_df, (str, Path)) and str(vcrs_metadata_df).endswith(".csv"):
                vcrs_metadata_df = pd.read_csv(vcrs_metadata_df, index_col=0)
            elif isinstance(vcrs_metadata_df, (str, Path)) and str(vcrs_metadata_df).endswith(".h5ad"):
                adata_tmp = ad.read_h5ad(vcrs_metadata_df, backed='r')
                vcrs_metadata_df = adata_tmp.var.to_dataframe()
            elif isinstance(vcrs_metadata_df, pd.DataFrame):
                vcrs_metadata_df = vcrs_metadata_df.copy()

            # drop repeat columns, and convert to lists if needed
            for col in vcrs_metadata_df.columns:
                if col in adata.var.columns:
                    vcrs_metadata_df.drop(columns=[col], inplace=True, errors='ignore')
                    continue
                first_val = vcrs_metadata_df.iloc[0][col]
                if isinstance(first_val, str) and first_val[0] == "[" and first_val[-1] == "]":
                    try:
                        vcrs_metadata_df[col] = vcrs_metadata_df[col].apply(safe_literal_eval)
                    except (ValueError, SyntaxError):
                        pass

            logger.info(f"vcrs_metadata_df exists. Merging with adata.var.")
            adata.var = adata.var.merge(vcrs_metadata_df, on="vcrs_id", how="left")
        elif isinstance(vcrs_metadata_df, (str, Path)) and not os.path.exists(vcrs_metadata_df):
            logger.info(f"vcrs_metadata_df does not exist. Skipping merging with adata.var, and saving the current vcrs_metadata_df to csv (good if running this function in a loop, where it will be created upon the first iteration and then loaded in upon subsequent iterations).")
            adata.var.to_csv(vcrs_metadata_df, index=False)
        else:
            raise ValueError(f"vcrs_metadata_df must be a pandas DataFrame or a file path to a csv or h5ad file. Got {type(vcrs_metadata_df)} instead.")

    #* fixed the parity stuff
    if parity == "paired" and parity_kb_count == "single" and technology in {"BULK", "SMARTSEQ2"} and adata.uns.get("corrected_barcodes") is None:  # the last part is to ensure I didn't make the correction already
        adata = correct_adata_barcodes_for_running_paired_data_in_single_mode(kb_count_vcrs_dir, adata=adata)

    if adata_reference_genome:
        if isinstance(adata_reference_genome, str) and os.path.exists(adata_reference_genome):
            if adata_reference_genome.endswith(".h5ad"):
                adata_reference_genome = ad.read_h5ad(adata_reference_genome)
            elif adata_reference_genome.endswith(".mtx"):                
                adata_reference_genome = load_adata_from_mtx(adata_reference_genome)

    adata.var_names = original_var_names

    # if apply_single_end_mode_on_paired_end_data_correction:
    #     # TODO: test this; also add union
    #     adata = decrement_adata_matrix_when_split_by_Ns_or_running_paired_end_in_single_end_mode(
    #         adata,
    #         fastq=fastqs,
    #         kb_count_out=kb_count_vcrs_dir,
    #         t2g=vcrs_t2g,
    #         mm=mm,
    #         bustools=bustools,
    #         split_Ns=split_reads_by_Ns_and_low_quality_bases,
    #         paired_end_fastqs=False,
    #         paired_end_suffix_length=2,
    #         technology=technology,
    #         keep_only_insertions=True,
    #     )

    if pseudobam_validation:
        adata = adjust_variant_adata_by_pseudobam(kb_count_vcrs_dir=kb_count_vcrs_dir, reference_genome_index=reference_genome_index, technology=technology, kallisto_quant_reference_genome_dir=kallisto_quant_reference_genome_dir, adata=adata, fastq_file_list=fastqs, adata_output_path=None, parity=parity, mm=mm, bustools=bustools, kallisto=kallisto, threads=threads, variant_source=variant_source, reference_sequences_type=reference_sequences_type, reads_type=reads_type, sequences=sequences, gtf=gtf, vcrs_t2g=vcrs_t2g, check_alignment_position=check_alignment_position, alignment_position_tolerance=alignment_position_tolerance, count_reads_that_dont_pseudoalign_to_reference_genome=count_reads_that_dont_pseudoalign_to_reference_genome, avoid_paired_double_counting=avoid_paired_double_counting, seq_id_column=seq_id_column, var_column=var_column, fastq_sorting_check_only=(not sort_fastqs), save_type="parquet", overwrite=overwrite, pseudobam_validation_aligner=pseudobam_validation_aligner, reference_bam=reference_bam, read_length=read_length)
    
    # set all count values below min_counts to 0
    if min_counts is not None:  # important I do BEFORE CPM normalization
        adata.X = adata.X.multiply(adata.X >= min_counts)

    if account_for_strand_bias:
        adata = remove_variants_from_adata_for_stranded_technologies(adata=adata, strand_bias_end=strand_bias_end, read_length=read_length, header_column="vcrs_id", seq_id_column=seq_id_column, var_column=var_column, variant_source=variant_source, gtf=gtf, forgiveness=forgiveness, out=out)

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

    if isinstance(adata_reference_genome, anndata.AnnData):  # and any([filter_cells_by_min_counts, filter_cells_by_min_genes, filter_genes_by_min_cells, filter_cells_by_max_mt_content, doublet_detection, cpm_normalization]):  # the list of reasons to enter here
        adata_reference_genome = adata_reference_genome[adata.obs_names].copy()  # ensures adata_reference_genome rows/obs match order of adata
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

        # do cpm
        if cpm_normalization and not use_binary_matrix:  # normalization not needed for binary matrix
            common_cells = adata.obs_names.intersection(adata_reference_genome.obs_names)
            adata = adata[common_cells, :].copy()
            
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
    adata.var["vcrs_count"] = adata.var["vcrs_count"].fillna(0).astype("float32")
    adata.var["vcrs_detected"] = adata.var["vcrs_count"] > 0

    if gene_id_column in adata.var:
        # Step 1: Filter rows where the list has exactly one gene name (no semicolon) and with a VCRS mapping
        adata.var["gene_id_set"] = adata.var[gene_id_column].apply(lambda x: list(set(x)))
        single_entry_mask = adata.var['gene_id_set'].apply(lambda x: isinstance(x, list) and len(x) == 1)  # ~adata.var['gene_id_set'].str.contains(";")
        filtered_var = adata.var[(single_entry_mask) & (adata.var["vcrs_count"] > 0)].copy()
        filtered_var['gene_id_single'] = filtered_var['gene_id_set'].apply(lambda x: x[0] if isinstance(x, list) else x)
        adata.var.drop(columns=["gene_id_set"], inplace=True, errors='ignore')

        # Step 2: Count occurrences of each gene name
        gene_counts_df = (filtered_var['gene_id_single'].value_counts().rename_axis('gene_name').reset_index(name='count'))
        gene_counts_dict = dict(zip(gene_counts_df["gene_name"], gene_counts_df["count"]))
        adata.var["gene_count"] = adata.var[gene_id_column].apply(lambda gene_list: [gene_counts_dict.get(g, 0) for g in gene_list])
        del filtered_var

    if sp.issparse(adata.X):
        # Sparse matrix handling
        adata.var["number_obs"] = np.array((adata.X != 0).sum(axis=0)).flatten()
    else:
        # Dense matrix handling
        adata.var["number_obs"] = (adata.X != 0).sum(axis=0)

    adata.var["number_obs"] = adata.var["number_obs"].astype("Int32")

    #* adata can't save lists in var columns, so we need to convert them to semicolon-separated strings - later get back with something like adata.var[col] = adata.var[col].apply(lambda x: x.split(";") if isinstance(x, str) else x)
    adata.var = join_list_columns_for_h5ad(adata.var)

    # Optionally drop merged VCRSs that represent multiple variants (i.e., those whose vcrs_id contains a semicolon)
    if drop_multi_variant_vcrs:
        multi_variant_mask = adata.var["vcrs_id"].astype(str).str.contains(";", na=False)
        number_of_multi_variant_vcrs = int(multi_variant_mask.sum())
        if number_of_multi_variant_vcrs > 0:
            logger.info(f"Dropping {number_of_multi_variant_vcrs} multi-variant VCRS entries (vcrs_id containing ';') from adata.")
            adata = adata[:, ~multi_variant_mask.to_numpy()].copy()

    if isinstance(adata_reference_genome, anndata.AnnData):
        logger.info(f"Saving adata_reference_genome to {adata_reference_genome_clean_out}.")
        adata_reference_genome.write(adata_reference_genome_clean_out)
        # adata_reference_genome_dtypes_file_base = adata_reference_genome_clean_out.replace(".h5ad", "_dtypes")
        # save_df_types_adata(adata_reference_genome, out_file_base=adata_reference_genome_dtypes_file_base)

    if save_vcf:
        if not vcf_data_dataframe or not os.path.exists(vcf_data_dataframe):
            if variants == "cosmic_cmc":
                # COSMIC credentials come only from the environment (never from arguments/config on disk)
                cosmic_email = os.getenv("COSMIC_EMAIL")
                cosmic_password = os.getenv("COSMIC_PASSWORD")
                vcf_data_dataframe = add_vcf_info_to_cosmic_tsv(cosmic_tsv=cosmic_tsv, reference_genome_fasta=cosmic_reference_genome_fasta, cosmic_df_out=vcf_data_dataframe, sequences=sequences, cosmic_version=cosmic_version, cosmic_email=cosmic_email, cosmic_password=cosmic_password)
            # insert other supported databases here
            else:
                raise ValueError("vcf_data_dataframe must be provided if variants is supported. Supported databases can be viewed with `vk ref --list_internally_supported_indices`.")
        logger.info(f"Saving VCF file to {vcf_out}.")
        cleaned_adata_to_vcf(adata.var, vcf_data_df=vcf_data_dataframe, output_vcf=vcf_out, save_vcf_samples=save_vcf_samples, adata=adata)

    # Optionally rename all "vcrs" var columns (and the var index name) to "variant", keeping the raw vcrs_id column as-is. Done after the VCF step, which relies on the vcrs_* column names.
    if rename_vcrs_to_variant:
        vcrs_to_variant_rename_map = {col: col.replace("vcrs", "variant") for col in adata.var.columns if "vcrs" in col and col != "vcrs_id"}
        adata.var.rename(columns=vcrs_to_variant_rename_map, inplace=True)
        if adata.var.index.name and "vcrs" in adata.var.index.name:
            adata.var.index.name = adata.var.index.name.replace("vcrs", "variant")

    logger.info(f"Saving adata to {adata_vcrs_clean_out}.")
    adata.write(adata_vcrs_clean_out)
    # adata_dtypes_file_base = adata_vcrs_clean_out.replace(".h5ad", "_dtypes")
    # save_df_types_adata(adata, out_file_base=adata_dtypes_file_base)

    report_time_elapsed(start_time, "clean")
    return adata

