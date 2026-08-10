"""varseek count and specific helper functions."""

import copy
import anndata as ad
import inspect
import json
import logging
import os
import subprocess
import time
from pathlib import Path
from typing import Optional, Union
import pandas as pd
from pydantic import BaseModel, ConfigDict, model_validator

import varseek as vk
from varseek.utils import (
    check_file_path_is_string_with_valid_extension,
    check_that_two_paths_are_the_same_if_both_provided_otherwise_set_them_equal,
    load_in_fastqs,
    correct_adata_barcodes_for_running_paired_data_in_single_mode,
    make_function_parameter_to_value_dict,
    report_time_elapsed,
    save_params_to_config_file,
    save_run_info,
    set_up_logger,
    sort_fastq_files_for_kb_count,
    set_varseek_logging_level_and_filehandler,
    load_adata_from_mtx,
    validate_call,
    vk_config,
    Technology,
    Parity,
    Strand,
    StrandBiasEnd,
    PositiveInt,
    NonNegativeInt,
    int_range,
)

from .constants import (
    non_single_cell_technologies,
    varseek_count_only_allowable_kb_count_arguments,
)

logger = logging.getLogger(__name__)
logger = set_up_logger(logger, logging_level="INFO", save_logs=False, log_dir=None)

mode_parameters = {
    "very_sensitive": {},
    "sensitive": {},
    "balanced": {},
    "specific": {},
    "very_specific": {},
}

varseek_count_unallowable_arguments = {
    "varseek_fastqpp": set(),
    "kb_count": {"aa", "workflow"},
    "varseek_clean": set(),
    "varseek_summarize": set(),
}

# count's own (native) arguments. Every OTHER parameter in count()'s signature is a verbatim passthrough
# copied from vk fastqpp / clean / summarize (see the "passthrough to vk ..." blocks in the signature) and
# is forwarded to the relevant child using the child's own default. Native args, by contrast, are either
# count-only or are forwarded to the children explicitly (by name) in the vk.fastqpp/clean/summarize calls,
# so they must NOT be broad-forwarded again. KEEP IN SYNC with count()'s signature: when you add a
# NON-passthrough argument to count(), add it here too. (To (re)derive the passthrough block itself, copy
# each child's parameter lines whose name is not already native and not in {vcrs_index, vcrs_t2g, adata,
# adata_vcrs}.)
count_native_parameters = frozenset({
    "fastqs", "index", "technology", "t2g", "k", "pseudobam_validation", "account_for_strand_bias",
    "strand_bias_end", "read_length", "strand", "mm", "union", "parity", "gtf", "out",
    "kb_count_vcrs_out_dir", "vk_summarize_out_dir", "disable_fastqpp", "disable_clean", "summarize",
    "chunksize", "dry_run", "overwrite", "sort_fastqs", "threads", "logging_level", "save_logs",
    "log_out_dir", "num", "parity_kb_count",
})

# Advanced `count` parameters: still accepted on the command line (and in the Python signature,
# where @validate_call validates them), but hidden from `vk count --help` to keep it uncluttered.
# Consumed by main.py, which flips each matching argparse action's help to SUPPRESS. Matched by dest,
# so the --disable_num flag maps to its underlying parameter name.
vk_count_hidden_from_help = {
    "num",
    "parity_kb_count",
}

# Documentation for the advanced `count` parameters. These are real, fully typed
# arguments on the `count` signature (validated by @validate_call like every other
# parameter), but they are deliberately kept out of the public `count` docstring
# (and therefore out of help(vk.count) / the Sphinx site) and out of `vk count --help`
# (via vk_count_hidden_from_help) to keep the common interface uncluttered. The text
# lives here so maintainers still have it close to the definition.
COUNT_ADVANCED_PARAMS_DOC = """
# # Advanced parameters (real `count` arguments, but hidden from the `vk count --help` CLI):
- num                                  (bool) If True, use the --num argument in kb count. Default: True.
- parity_kb_count                      (str) The parity of the reads for kb count. Always recommended to run in single. Default: "single".
"""


class CountParams(BaseModel):
    """Cross-field / kwargs-only validation for :func:`count`.

    Per-parameter type/extension checks live on the ``count`` signature (validated
    by ``@validate_call``). ``extra="allow"`` retains the kb-count pass-through
    kwargs so their types can be checked via ``self.model_extra`` (mirroring
    ``RefParams``).
    """

    model_config = ConfigDict(extra="allow", arbitrary_types_allowed=True)

    index: object = None
    t2g: object = None
    adata_reference_genome: object = None
    parity: object = None
    technology: object = None
    parity_kb_count: object = None
    strand: object = None
    out: object = "."
    kallisto_quant_reference_genome_dir: object = None

    @model_validator(mode="after")
    def _validate(self):
        # $ fastqs, technology will get checked through fastqpp

        # other required files
        for param_name, file_type in {
            "index": "index",
            "t2g": "t2g",
        }.items():
            value = getattr(self, param_name)
            check_file_path_is_string_with_valid_extension(value, param_name, file_type=file_type, required=False)
            if not os.path.isfile(value) and value != "None":  # ensure that all files exist
                raise ValueError(f"File {value} does not exist")

        # adata_reference_genome extension
        check_file_path_is_string_with_valid_extension(self.adata_reference_genome, "adata_reference_genome", "adata")

        # strings
        parity_valid_values = {"single", "paired", None}
        if self.parity not in parity_valid_values:
            raise ValueError(f"Parity must be one of {parity_valid_values}")
        if self.technology in {"BULK", "SMARTSEQ2"} and self.parity is None:
            raise ValueError("Parity must be set to 'single' or 'paired' for bulk or smartseq2 data")

        if self.parity_kb_count is not None and self.parity_kb_count not in parity_valid_values:
            raise ValueError(f"parity_kb_count must be one of {parity_valid_values}")

        if self.parity_kb_count == "paired" and self.parity == "single":
            raise ValueError("If parity_kb_count is 'paired', then parity must be 'paired' as well")

        if self.parity_kb_count == "single" and self.parity == "paired":
            logger.info("parity='paired' and parity_kb_count='single'. This means that kb count will run in single-end mode on this paired-end data, which will enable different pairs to be processed independently and thus potentially detect different variants. To turn this feature off, set parity_kb_count='paired'.")

        strand_valid_values = {"unstranded", "forward", "reverse", None}
        if self.strand not in strand_valid_values:
            raise ValueError(f"Strand must be one of {strand_valid_values}")

        # kb count stuff
        extra = self.model_extra or {}
        for argument_type, argument_set in varseek_count_only_allowable_kb_count_arguments.items():
            for argument in argument_set:
                argument = argument[2:]
                if argument in extra:
                    argument_value = extra[argument]
                    if argument_value is None:  # None => not provided (e.g. clean's kallisto/bustools defaults); skip type check
                        continue
                    if argument_type == "zero_arguments":
                        if not isinstance(argument_value, bool):  # all zero-arguments are bool
                            raise ValueError(f"{argument} must be a boolean. Got {type(argument_value)}.")
                    elif argument_type == "one_argument":
                        if not isinstance(argument_value, str):  # all one-arguments are string
                            raise ValueError(f"{argument} must be a string. Got {type(argument_value)}.")
                    elif argument_type == "multiple_arguments":
                        pass

        return self


# don't worry if it says an argument is unused, as they will all get put in params_dict for each respective function and passed to the child functions
@validate_call(config=vk_config)
def count(
    fastqs: object,
    index: str,
    technology: Optional[Technology],
    t2g: str = "None",
    k: PositiveInt = 59,  # params
    pseudobam_validation: bool = False,
    account_for_strand_bias: bool = False,
    strand_bias_end: Optional[StrandBiasEnd] = None,
    read_length: Optional[Union[int, str]] = None,
    strand: Optional[Strand] = None,
    mm: bool = True,
    union: bool = True,
    parity: Optional[Parity] = "single",
    gtf: Optional[Union[bool, str, Path]] = None,
    out: Union[str, Path] = ".",  # optional outputs
    kb_count_vcrs_out_dir: Optional[Union[str, Path]] = None,
    vk_summarize_out_dir: Optional[Union[str, Path]] = None,
    disable_fastqpp: bool = False,  # general
    disable_clean: bool = False,
    summarize: bool = False,
    chunksize: Optional[int] = None,
    dry_run: bool = False,
    overwrite: bool = False,
    sort_fastqs: bool = True,
    threads: PositiveInt = 2,
    logging_level: Optional[Union[str, int]] = None,
    save_logs: bool = False,
    log_out_dir: Optional[Union[str, Path]] = None,
    # === Passthrough parameters ===
    # Every argument of vk fastqpp / clean / summarize is copied here verbatim (name, type, and default)
    # so that it is a first-class `count` argument rather than an anonymous **kwargs entry. They are hidden
    # from `vk count --help` (see main.py) but fully validated by @validate_call. Each is forwarded to the
    # relevant child using the child's own default (see count_native_parameters). Regenerate with the helper
    # noted next to count_native_parameters if a child signature changes.
    # ---- passthrough to vk fastqpp ----
    multiplexed: Optional[bool] = None,
    quality_control_fastqs: bool = False,
    cut_front: bool = False,
    cut_tail: bool = False,
    cut_window_size: int_range(1, 1000) = 4,
    cut_mean_quality: int_range(1, 36) = 15,
    disable_adapter_trimming: bool = False,
    qualified_quality_phred: int_range(1, 40) = 15,
    unqualified_percent_limit: int_range(0, 100) = 40,
    average_qual: int_range(0, 40) = 15,
    n_base_limit: int_range(1, 50) = 10,
    disable_quality_filtering: bool = False,
    length_required: Optional[int_range(0, 9999)] = None,  # None => default to k (see body); vk fastqpp's own default is 31
    disable_length_filtering: bool = False,
    dont_eval_duplication: bool = False,
    disable_trim_poly_g: bool = False,
    failed_out: Union[bool, str] = False,
    split_reads_by_Ns_and_low_quality_bases: bool = False,
    min_base_quality_for_splitting: int_range(0, 93) = 5,
    concatenate_paired_fastqs: bool = False,
    seqtk_path: str = "seqtk",
    quality_control_fastqs_out_dir: Optional[Union[str, Path]] = None,
    replace_low_quality_bases_with_N_out_dir: Optional[Union[str, Path]] = None,
    split_by_Ns_and_low_quality_bases_out_dir: Optional[Union[str, Path]] = None,
    concatenate_paired_fastqs_out_dir: Optional[Union[str, Path]] = None,
    delete_intermediate_files: bool = True,
    # ---- passthrough to vk clean ----
    min_counts: PositiveInt = 2,
    use_binary_matrix: bool = False,
    drop_empty_columns: bool = False,
    pseudobam_validation_aligner: str = "pseudo",
    reference_genome_index: Optional[Union[str, Path]] = None,
    reference_bam: object = None,
    check_alignment_position: bool = False,
    alignment_position_tolerance: int = 50,
    reference_sequences_type: Optional[str] = None,
    kallisto: Optional[str] = None,
    count_reads_that_dont_pseudoalign_to_reference_genome: bool = True,
    avoid_paired_double_counting: bool = False,
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
    adata_reference_genome: object = None,
    vk_ref_dir: Optional[Union[str, Path]] = None,
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
    adata_vcrs_clean_out: Optional[Union[str, Path]] = None,
    adata_reference_genome_clean_out: Optional[Union[str, Path]] = None,
    vcf_out: Optional[Union[str, Path]] = None,
    save_vcf: bool = False,
    save_vcf_samples: bool = False,
    variants_updated_dataframe: Optional[Union[str, Path]] = None,
    bustools: Optional[str] = None,
    cosmic_tsv: Optional[Union[str, Path]] = None,
    cosmic_reference_genome_fasta: Optional[Union[str, Path]] = None,
    cosmic_version: Union[str, int] = 101,
    forgiveness: int = 100,
    add_hgvs_breakdown_to_adata_var: bool = True,
    # ---- passthrough to vk summarize ----
    top_values: PositiveInt = 10,
    gene_name_column: Optional[str] = None,
    plot_strand_bias: bool = False,
    cdna_fasta: Optional[str] = None,
    seq_id_cdna_column: str = "seq_ID",
    start_variant_position_cdna_column: str = "start_variant_position",
    end_variant_position_cdna_column: str = "end_variant_position",
    stats_file: Optional[Union[str, Path]] = None,
    specific_stats_folder: Optional[Union[str, Path]] = None,
    plots_folder: Optional[Union[str, Path]] = None,
    # --- Advanced parameters: part of the Python signature (validated by @validate_call) but hidden from `vk count --help`. See the "Advanced parameters" docstring section. ---
    num: bool = True,
    parity_kb_count: Optional[Parity] = "single",
    **kwargs,  # * kb count pass-through args + forward-compat only (every fastqpp/clean/summarize arg is now an explicit parameter above)
):
    """
    Perform variant screening on sequencing data. Wraps around varseek fastqpp, kb count, varseek clean, and varseek summarize.

    # Required input arguments:
    - fastqs                                (str or list[str]) List of fastq files to be processed. If paired end, the list should contain paths such as [file1_R1, file1_R2, file2_R1, file2_R2, ...]
    - index                                 (str)  Path to variant index generated by varseek ref
    - technology                            (str)  Technology used to generate the data. To see the list of supported technologies, run `kb --list`. varseek additionally accepts "dna", a bulk-like technology that declares the reads are genomic DNA (rather than RNA); this only affects the pseudobam validation (RNA reads against a DNA reference use `kallisto quant --genomebam`, DNA reads use `--pseudobam`).

    # Additional parameters
    - k                                     (int) The length of each k-mer in the kallisto reference index construction. Corresponds to `k` used in the earlier varseek commands (i.e., varseek ref). If using a downloaded index from varseek ref -d, then check the description for the k value used to construct this index with varseek ref --list_downloadable_references. Default: 59.
    - pseudobam_validation                  (bool) Whether to validate the VCRS count matrix against the reference genome in vk clean by re-aligning the reads to it. Each read's true alignment coordinate is compared to the HGVS locus of its assigned VCRS; reads that aligned somewhere inconsistent with the variant are dropped. Requires a standard reference-genome index (pass `reference_genome_index`) or a pre-made `reference_bam`; both are forwarded to vk clean, as is `pseudobam_validation_aligner`. Default: False.
    - account_for_strand_bias               (bool) Whether to account for strand bias from stranded single-cell technologies. Default: False.
    - strand_bias_end                       (str) The end of the read to use for strand bias correction. Either "5p" or "3p". Must be provided if and only if account_for_strand_bias=True. Default: None.
    - read_length                           (int) The read length used in the experiment. Must be provided if and only if account_for_strand_bias=True. Default: None.
    - strand                                (str)  The strandedness of the data. Either "unstranded", "forward", or "reverse". Default: None.
    - mm                                    (bool)  If True, use the multi-mapping reads. Default: True.
    - union                                 (bool)  If True, use the union of the read mappings. Default: True.
    - parity                                (str)  The parity of the reads for vk fastqpp. Either "single" or "paired". Only used if technology is bulk or a smart-seq. For parity in kb count, see parity_kb_count. Default: "single".

    # Optional input arguments
    - t2g                                   (str) Path to t2g file generated by varseek ref. Default: "None" (will be created if not provided).
    - gtf                                   (str) Path to the GTF file. Only used when account_for_strand_bias=True and either (1) strand_bias_end='3p' and/or (2) some VCRSs are derived from genome sequences. Default: None.

    # Optional output file paths: (only needed if changing/customizing file names or locations):
    - out                                   (str) Output directory. Default: ".".
    - kb_count_vcrs_out_dir                 (str) Output directory for kb count. Default: `out`/kb_count_out_vcrs
    - vk_summarize_out_dir                  (str) Output directory for vk summarize. Default: `out`/vk_summarize

    # General arguments:
    - disable_fastqpp                       (bool) If True, skip fastqpp step. Default: False.
    - disable_clean                         (bool) If True, skip clean step. Default: False.
    - summarize                             (bool) If True, perform summarize step. Default: False.
    - chunksize                             (int) Number of BUS file lines to process at a time. If None, then all lines will be processed at once. Default: None.
    - dry_run                               (bool) If True, print the commands that would be run without actually running them. Default: False.
    - overwrite                             (bool) If True, overwrite existing files. Default: False.
    - sort_fastqs                           (bool) If True, sort fastq files by kb count. If False, then still check the order but do not change anything. Default: True
    - threads                               (int) Number of threads to use. Default: 2.
    - logging_level                         (str) Logging level. Can also be set with the environment variable VARSEEK_LOGGING_LEVEL. Default: INFO.
    - save_logs                             (True/False) Whether to save logs to a file. Default: False.
    - log_out_dir                           (str) Directory to save logs. Default: None (do not save logs).

    NOTE: `count` also accepts several advanced/niche parameters that are intentionally omitted here (and from
    `vk count --help`). They are fully typed and validated like any other argument; see COUNT_ADVANCED_PARAMS_DOC
    in varseek/varseek_count.py for their documentation.

    For a complete list of supported parameters, see the documentation for varseek fastqpp, kb count, varseek clean, and varseek summarize. Note that any shared parameter names between functions are meant to have identical purposes.
    """
    start_time = time.perf_counter()

    # * 0. Informational arguments that exit early
    # Nothing here

    # * 1. logger
    if save_logs and not log_out_dir:
        log_out_dir = os.path.join(out, "logs")
    set_varseek_logging_level_and_filehandler(logging_level=logging_level, save_logs=save_logs, log_dir=log_out_dir)
    
    # * 1.5. For the nargs="+" arguments, convert any list of length 1 to a string
    if isinstance(fastqs, (list, tuple)) and len(fastqs) == 1:
        fastqs = fastqs[0]

    # * 1.75 load in fastqs
    fastqs_original = fastqs
    fastqs = load_in_fastqs(fastqs)

    # * 2. Type-checking
    params_dict = make_function_parameter_to_value_dict(1)
    params_dict["adata_vcrs"] = "placeholder/adata.h5ad"  # placeholder for type checking (adata_vcrs is derived internally, not a count parameter)
    params_dict["adata"] = "placeholder/adata_cleaned.h5ad"  # placeholder for type checking (adata is derived internally, not a count parameter)

    # Set params_dict_for_type_checking to default values of children functions - important so type checking works properly
    count_signature = inspect.signature(count)
    for function in (vk.varseek_fastqpp.fastqpp, vk.varseek_clean.clean, vk.varseek_summarize.summarize):
        signature = inspect.signature(function)
        for key in signature.parameters.keys():
            if key not in params_dict and key not in count_signature.parameters.keys():
                params_dict[key] = signature.parameters[key].default

    vk.varseek_fastqpp.FastqppParams(**params_dict)  # this passes all vk count parameters to the function - I could only pass in the vk fastqpp parameters here if desired (and likewise below), but there should be no naming conflicts anyways
    vk.varseek_clean.CleanParams(**params_dict)
    vk.varseek_summarize.SummarizeParams(**params_dict)
    CountParams(**params_dict)

    # * 3. Dry-run
    # handled within child functions

    # * 4. Save params to config file and run info file
    if not dry_run:
        # Save parameters to config file
        params_dict["fastqs"] = fastqs_original  # for config file
        config_file = os.path.join(out, "config", "vk_count_config.json")
        save_params_to_config_file(params_dict, config_file)  # $ Now I am done with params_dict

        run_info_file = os.path.join(out, "config", "vk_count_run_info.txt")
        save_run_info(run_info_file, params_dict=params_dict, function_name="count")

    # * 4.5 Pop out any unallowable arguments
    for key, unallowable_set in varseek_count_unallowable_arguments.items():
        for unallowable_key in unallowable_set:
            kwargs.pop(unallowable_key, None)

    # * 4.8 Set kwargs to default values of children functions (not strictly necessary, as if these arguments are not in kwargs then it will use the default values anyways, but important if I need to rely on these default values within vk count)
    # count_signature = inspect.signature(count)
    # for function in (vk.varseek_fastqpp.fastqpp, vk.varseek_clean.clean, vk.varseek_summarize.summarize):
    #     signature = inspect.signature(function)
    #     for key in signature.parameters.keys():
    #         if key not in kwargs and key not in count_signature.parameters.keys():
    #             kwargs[key] = signature.parameters[key].default

    # * 5. Set up default folder/file input paths, and make sure the necessary ones exist
    # all input files for vk count are required in the varseek workflow, so this is skipped

    # * 6. Set up default folder/file output paths, and make sure they don't exist unless overwrite=True
    if not kb_count_vcrs_out_dir:
        kb_count_vcrs_out_dir = os.path.join(out, "kb_count_out_vcrs") if not kb_count_vcrs_dir else kb_count_vcrs_dir
    if not vk_summarize_out_dir:
        vk_summarize_out_dir = os.path.join(out, "vk_summarize")

    if not dry_run:
        os.makedirs(out, exist_ok=True)
        os.makedirs(kb_count_vcrs_out_dir, exist_ok=True)

    # for kb count --> vk clean
    kb_count_vcrs_out_dir, kb_count_vcrs_dir = check_that_two_paths_are_the_same_if_both_provided_otherwise_set_them_equal(kb_count_vcrs_out_dir, kb_count_vcrs_dir)  # check that, if kb_count_vcrs_dir and kb_count_vcrs_out_dir are both provided, they are the same directory; otherwise, if only one is provided, then make them equal to each other

    adata_vcrs = f"{kb_count_vcrs_out_dir}/counts_unfiltered/adata.h5ad" if not kwargs.get("adata_vcrs") else kwargs.get("adata_vcrs")  # derived internally; overridable via **kwargs
    # adata_reference_genome is forwarded to vk clean as-is (raw, may be None). adata_reference_genome_resolved is
    # only for vk count's own bookkeeping (existence check / returned output dict): if the user did not pass a
    # matrix directly but did pass kallisto_quant_reference_genome_dir, point at the adata inside that directory.
    adata_reference_genome_resolved = adata_reference_genome
    if not adata_reference_genome_resolved and kallisto_quant_reference_genome_dir:
        adata_reference_genome_resolved = f"{kallisto_quant_reference_genome_dir}/counts_unfiltered/adata.h5ad"
    adata_vcrs_clean_out = f"{out}/adata_cleaned.h5ad" if not adata_vcrs_clean_out else adata_vcrs_clean_out  # from vk clean
    adata_reference_genome_clean_out = f"{out}/adata_cleaned.h5ad" if not adata_reference_genome_clean_out else adata_reference_genome_clean_out  # from vk clean
    if save_vcf and not vcf_out:
        vcf_out = os.path.join(out, "variants.vcf")  # from vk clean; only default when a VCF is actually being saved (a truthy vcf_out forces save_vcf=True inside vk clean)
    stats_file = os.path.join(vk_summarize_out_dir, "varseek_summarize_stats.txt") if not stats_file else stats_file  # from vk summarize

    for file in [stats_file]:  # purposely excluded adata_reference_genome because it is fine if someone provides this as input even if overwrite=False; and purposely excluded adata_vcrs, adata_vcrs_clean_out, adata_reference_genome_clean_out, kb_count_vcrs_out_dir for the reasons provided in vk ref
        if os.path.isfile(file) and not overwrite:
            raise FileExistsError(f"Output file {file} already exists. Please delete it or specify a different output directory or set overwrite=True.")

    # no need for file_signifying_successful_vk_fastqpp_completion because overwrite=False just gives warning rather than error in fastqpp
    file_signifying_successful_kb_count_vcrs_completion = adata_vcrs
    file_signifying_successful_vk_clean_completion = adata_vcrs_clean_out
    file_signifying_successful_vk_summarize_completion = stats_file

    # * 7.5 make sure ints are ints
    k, threads = int(k), int(threads)

    # * 8. Start the actual function
    technology = technology.upper()

    # The virtual "DNA" technology is processed exactly like BULK (kb count, barcodes, parity), but
    # additionally declares that the reads are genomic DNA. Capture that reads-type signal (forwarded to
    # vk clean for the pseudobam QC: RNA reads against a DNA reference use --genomebam, DNA reads use
    # --pseudobam) and normalize the technology to BULK so kb count and all downstream bulk logic apply.
    reads_type = "dna" if technology == "DNA" else "rna"
    if technology == "DNA":
        technology = "BULK"

    if isinstance(fastqs, list):
        fastqs = tuple(fastqs)

    fastqs_unsorted = fastqs
    try:
        fastqs = sort_fastq_files_for_kb_count(fastqs, technology=technology, multiplexed=multiplexed, check_only=(not sort_fastqs))
    except ValueError as e:
        if sort_fastqs:
            logger.warning(f"Automatic FASTQ argument order sorting for kb count could not recognize FASTQ file name format. Skipping argument order sorting.")

    # so parity_vcrs is set correctly - copied from fastqpp (concatenate_paired_fastqs / split_reads_by_Ns_and_low_quality_bases are now count parameters)
    if (concatenate_paired_fastqs or split_reads_by_Ns_and_low_quality_bases) and parity == "paired":
        if not concatenate_paired_fastqs:
            logger.info("Setting concatenate_paired_fastqs=True")
        concatenate_paired_fastqs = True
    else:
        if concatenate_paired_fastqs:
            logger.info("Setting concatenate_paired_fastqs=False")
        concatenate_paired_fastqs = False

    if disable_clean:
        pseudobam_validation = False  # disable_clean gets priority (pseudobam validation happens inside vk clean)

    if not length_required:
        logger.info("Setting length_required to %s if fastqpp is run", k)
        length_required = k

    # define the vk fastqpp, clean, and summarize arguments (explicit arguments and allowable kwargs)
    explicit_parameters_vk_fastqpp = vk.utils.get_set_of_parameters_from_function_signature(vk.varseek_fastqpp.fastqpp)  # originally did not include *fastqs due to asterisk, but I removed the asterisk so now it's fine
    allowable_kwargs_vk_fastqpp = vk.utils.get_set_of_allowable_kwargs(vk.varseek_fastqpp.fastqpp)

    explicit_parameters_vk_clean = vk.utils.get_set_of_parameters_from_function_signature(vk.varseek_clean.clean)
    allowable_kwargs_vk_clean = vk.utils.get_set_of_allowable_kwargs(vk.varseek_clean.clean)

    explicit_parameters_vk_summarize = vk.utils.get_set_of_parameters_from_function_signature(vk.varseek_summarize.summarize)
    allowable_kwargs_vk_summarize = vk.utils.get_set_of_allowable_kwargs(vk.varseek_summarize.summarize)

    all_parameter_names_set_vk_fastqpp = explicit_parameters_vk_fastqpp | allowable_kwargs_vk_fastqpp
    all_parameter_names_set_vk_clean = explicit_parameters_vk_clean | allowable_kwargs_vk_clean
    all_parameter_names_set_vk_summarize = explicit_parameters_vk_summarize | allowable_kwargs_vk_summarize

    # Every passthrough argument is now forwarded to its child explicitly by name in the vk.fastqpp / vk.clean /
    # vk.summarize calls below (the child param sets above are used only to forward any leftover **kwargs for
    # forward-compat if a child signature ever gains an argument that count does not yet mirror).

    # vk fastqpp
    if not any([quality_control_fastqs, split_reads_by_Ns_and_low_quality_bases, concatenate_paired_fastqs]):
        disable_fastqpp_original = disable_fastqpp
        disable_fastqpp = True

    if not disable_fastqpp:  # don't do the whole overwrite thing here because it is the first function, and a user should know if they are overwriting their fastqs
        kwargs_vk_fastqpp = {key: value for key, value in kwargs.items() if key in all_parameter_names_set_vk_fastqpp}  # forward-compat: leftover **kwargs that vk fastqpp accepts (all mirrored args are passed explicitly below)

        logger.info("Running vk fastqpp")
        if dry_run:
            logger.warning("Note: during dry run, the fastq's passed into kb count will be the default fastqs, not any updated fastqs from fastqpp")
        fastqpp_dict = vk.fastqpp(
            fastqs, technology=technology, parity=parity, out=out, dry_run=dry_run, overwrite=overwrite, sort_fastqs=sort_fastqs, threads=threads, logging_level=logging_level, save_logs=save_logs, log_out_dir=log_out_dir,  # native count args
            multiplexed=multiplexed, quality_control_fastqs=quality_control_fastqs, cut_front=cut_front, cut_tail=cut_tail, cut_window_size=cut_window_size, cut_mean_quality=cut_mean_quality, disable_adapter_trimming=disable_adapter_trimming, qualified_quality_phred=qualified_quality_phred, unqualified_percent_limit=unqualified_percent_limit, average_qual=average_qual, n_base_limit=n_base_limit, disable_quality_filtering=disable_quality_filtering, length_required=length_required, disable_length_filtering=disable_length_filtering, dont_eval_duplication=dont_eval_duplication, disable_trim_poly_g=disable_trim_poly_g, failed_out=failed_out, split_reads_by_Ns_and_low_quality_bases=split_reads_by_Ns_and_low_quality_bases, min_base_quality_for_splitting=min_base_quality_for_splitting, concatenate_paired_fastqs=concatenate_paired_fastqs, seqtk_path=seqtk_path, quality_control_fastqs_out_dir=quality_control_fastqs_out_dir, replace_low_quality_bases_with_N_out_dir=replace_low_quality_bases_with_N_out_dir, split_by_Ns_and_low_quality_bases_out_dir=split_by_Ns_and_low_quality_bases_out_dir, concatenate_paired_fastqs_out_dir=concatenate_paired_fastqs_out_dir, delete_intermediate_files=delete_intermediate_files,  # passthrough to vk fastqpp
            **kwargs_vk_fastqpp,
        )  # intentionally do not set overwrite = True here because it is the first function, and a user should know if they are overwriting their fastqs

        fastqs_vcrs = fastqpp_dict["final"]
    else:
        if disable_fastqpp_original:
            logger.info("Skipping vk fastqpp because disable_fastqpp=True")
        else:
            logger.info("Skipping vk fastqpp because there was no use for it")
        fastqs_vcrs = fastqs
    fastqs = fastqs_vcrs  # so that the correct fastqs get passed into vk clean

    # # kb count, VCRS
    if not os.path.exists(file_signifying_successful_kb_count_vcrs_completion) or overwrite:
        kb_count_command = [
            "kb",
            "count",
            "-t",
            str(threads),
            "-k",
            str(k),
            "-i",
            index,
            "-g",
            t2g,
            "-x",
            technology,
            "--h5ad",
            "-o",
            kb_count_vcrs_out_dir,
            "--overwrite",  # set overwrite here regardless of the overwrite argument because I would only even enter this block if kb count was only partially run (as seen by the lack of existing of file_signifying_successful_kb_count_vcrs_completion), in which case I should overwrite anyways
        ]

        if strand:
            kb_count_command.extend(["--strand", strand])
        if technology in {"BULK", "SMARTSEQ2"}:
            parity_vcrs = "single" if concatenate_paired_fastqs else parity_kb_count
            kb_count_command.extend(["--parity", parity_vcrs])

        if pseudobam_validation:
            kb_count_command.extend(["--union",])  # don't need mm here, as mm does not affect the BUS file (only the count matrix)
        if pseudobam_validation or kwargs.get("apply_split_reads_by_Ns_correction") or num:
            kb_count_command.extend(["--num"])

        if mm and "--mm" not in kb_count_command:
            kb_count_command.append("--mm")
        if union and "--union" not in kb_count_command:
            kb_count_command.append("--union")

        # assumes any argument in varseek count matches kb count identically, except dashes replaced with underscores
        params_dict_kb_count_vcrs = make_function_parameter_to_value_dict(1)  # will reflect any updated values to variables found in vk count signature and anything in kwargs
        for dict_key, arguments in varseek_count_only_allowable_kb_count_arguments.items():
            for argument in list(arguments):
                dash_count = len(argument) - len(argument.lstrip("-"))
                leading_dashes = "-" * dash_count
                argument = argument.lstrip("-").replace("-", "_")
                if argument in params_dict_kb_count_vcrs:
                    value = params_dict_kb_count_vcrs[argument]
                    if dict_key == "zero_arguments":
                        if value:  # only add if value is True
                            kb_count_command.append(f"{leading_dashes}{argument}")
                    elif dict_key == "one_argument":
                        if value is not None:  # skip when not provided (many one-arg kb flags are now count params defaulting to None)
                            kb_count_command.extend([f"{leading_dashes}{argument}", str(value)])
                    else:  # multiple_arguments or something else
                        pass

        kb_count_command += fastqs_vcrs

        if dry_run:
            print(" ".join(kb_count_command))
        else:
            logger.info(f"Running kb count with command: {' '.join(kb_count_command)}")
            subprocess.run(kb_count_command, check=True)

            if not os.path.isfile(adata_vcrs):
                mtx_file = os.path.join(kb_count_vcrs_out_dir, "counts_unfiltered", "cells_x_genes.mtx")
                if os.path.isfile(mtx_file):
                    _ = load_adata_from_mtx(mtx_file, adata_out = adata_vcrs)

            if os.path.exists(adata_vcrs) and parity == "paired" and parity_kb_count == "single" and technology in {"BULK", "SMARTSEQ2"}:
                _ = correct_adata_barcodes_for_running_paired_data_in_single_mode(kb_count_vcrs_out_dir, adata_out=adata_vcrs)  # will check if the correction has already occurred internally
    else:
        logger.info(f"Skipping kb count because file {file_signifying_successful_kb_count_vcrs_completion} already exists and overwrite=False")

    # note: vk count no longer runs kb count against a reference genome. The pseudobam validation (pseudobam_validation=True)
    # re-aligns the reads with `kallisto quant` inside vk clean, so no separate reference-genome kb count is needed. If a user
    # wants the gene-matrix-based scanpy QC (filter_cells_by_min_counts, doublet_detection, cpm_normalization, ...), they should
    # run kb count against the reference genome themselves and pass kallisto_quant_reference_genome_dir / adata_reference_genome to vk clean.
    if kallisto_quant_reference_genome_dir and (not os.path.exists(kallisto_quant_reference_genome_dir) or len(os.listdir(kallisto_quant_reference_genome_dir)) == 0):
        kallisto_quant_reference_genome_dir = None  # don't pass anything into clean if it's empty

    # vk clean
    if not disable_clean:
        if not os.path.exists(file_signifying_successful_vk_clean_completion) or overwrite:
            # adata_vcrs/vcrs_index/vcrs_t2g are derived internally and passed explicitly below
            kwargs_vk_clean = {key: value for key, value in kwargs.items() if key in all_parameter_names_set_vk_clean and key not in {"adata_vcrs", "vcrs_index", "vcrs_t2g"}}  # forward-compat: leftover **kwargs that vk clean accepts (all mirrored args are passed explicitly below)

            logger.info("Running vk clean")
            _ = vk.clean(
                adata_vcrs=adata_vcrs, vcrs_index=index, vcrs_t2g=t2g, technology=technology, fastqs=fastqs, threads=threads, pseudobam_validation=pseudobam_validation, account_for_strand_bias=account_for_strand_bias, strand_bias_end=strand_bias_end, read_length=read_length, gtf=gtf, mm=mm, parity=parity, parity_kb_count=parity_kb_count, out=out, chunksize=chunksize, dry_run=dry_run, overwrite=True, sort_fastqs=sort_fastqs, logging_level=logging_level, save_logs=save_logs, log_out_dir=log_out_dir,  # native count args (adata_vcrs/vcrs_index/vcrs_t2g derived internally)
                min_counts=min_counts, use_binary_matrix=use_binary_matrix, drop_empty_columns=drop_empty_columns, pseudobam_validation_aligner=pseudobam_validation_aligner, reference_genome_index=reference_genome_index, reference_bam=reference_bam, check_alignment_position=check_alignment_position, alignment_position_tolerance=alignment_position_tolerance, reference_sequences_type=reference_sequences_type, reads_type=reads_type, kallisto=kallisto, count_reads_that_dont_pseudoalign_to_reference_genome=count_reads_that_dont_pseudoalign_to_reference_genome, avoid_paired_double_counting=avoid_paired_double_counting, filter_cells_by_min_counts=filter_cells_by_min_counts, filter_cells_by_min_genes=filter_cells_by_min_genes, filter_genes_by_min_cells=filter_genes_by_min_cells, filter_cells_by_max_mt_content=filter_cells_by_max_mt_content, doublet_detection=doublet_detection, remove_doublets=remove_doublets, cpm_normalization=cpm_normalization, sum_rows=sum_rows, drop_multi_variant_vcrs=drop_multi_variant_vcrs, rename_vcrs_to_variant=rename_vcrs_to_variant, vcrs_id_set_to_exclusively_keep=vcrs_id_set_to_exclusively_keep, vcrs_id_set_to_exclude=vcrs_id_set_to_exclude, gene_set_to_exclusively_keep=gene_set_to_exclusively_keep, gene_set_to_exclude=gene_set_to_exclude, multiplexed=multiplexed, adata_reference_genome=adata_reference_genome, vk_ref_dir=vk_ref_dir, kb_count_vcrs_dir=kb_count_vcrs_dir, kallisto_quant_reference_genome_dir=kallisto_quant_reference_genome_dir, reference_genome_t2g=reference_genome_t2g, vcf_data_dataframe=vcf_data_dataframe, variants=variants, sequences=sequences, variant_source=variant_source, vcrs_metadata_df=vcrs_metadata_df, variants_usecols=variants_usecols, seq_id_column=seq_id_column, var_column=var_column, var_id_column=var_id_column, gene_id_column=gene_id_column, adata_vcrs_clean_out=adata_vcrs_clean_out, adata_reference_genome_clean_out=adata_reference_genome_clean_out, vcf_out=vcf_out, save_vcf=save_vcf, save_vcf_samples=save_vcf_samples, variants_updated_dataframe=variants_updated_dataframe, bustools=bustools, cosmic_tsv=cosmic_tsv, cosmic_reference_genome_fasta=cosmic_reference_genome_fasta, cosmic_version=cosmic_version, forgiveness=forgiveness, add_hgvs_breakdown_to_adata_var=add_hgvs_breakdown_to_adata_var,  # passthrough to vk clean
                **kwargs_vk_clean,
            )
        else:
            logger.info(f"Skipping vk clean because file {file_signifying_successful_vk_clean_completion} already exists and overwrite=False")
        adata = adata_vcrs_clean_out  # for vk summarize
    else:
        logger.info("Skipping vk clean because disable_clean=True")
        adata = adata_vcrs  # for vk summarize

    # # vk summarize
    if summarize:
        if not os.path.exists(file_signifying_successful_vk_summarize_completion) or overwrite:
            # adata is derived internally and passed explicitly; strand_bias_end/read_length are native (count owns them) but forwarded here so vk summarize's strand-bias plotting matches the clean path
            kwargs_vk_summarize = {key: value for key, value in kwargs.items() if key in all_parameter_names_set_vk_summarize and key != "adata"}  # forward-compat: leftover **kwargs that vk summarize accepts (all mirrored args are passed explicitly below)

            logger.info("Running vk summarize")
            try:
                _ = vk.summarize(
                    adata=adata, technology=technology, out=vk_summarize_out_dir, dry_run=dry_run, overwrite=True, strand_bias_end=strand_bias_end, read_length=read_length, logging_level=logging_level, save_logs=save_logs, log_out_dir=log_out_dir,  # native count args (adata derived internally)
                    top_values=top_values, gene_name_column=gene_name_column, plot_strand_bias=plot_strand_bias, cdna_fasta=cdna_fasta, seq_id_cdna_column=seq_id_cdna_column, start_variant_position_cdna_column=start_variant_position_cdna_column, end_variant_position_cdna_column=end_variant_position_cdna_column, stats_file=stats_file, specific_stats_folder=specific_stats_folder, plots_folder=plots_folder,  # passthrough to vk summarize
                    **kwargs_vk_summarize,
                )
            except Exception as e:
                if os.path.isfile(file_signifying_successful_vk_summarize_completion):
                    os.remove(file_signifying_successful_vk_summarize_completion)  # remove the file vk summarize stats file so that the vk count can be rerun
                logger.error(f"Error in vk summarize: {e}")
                raise e
        else:
            logger.info(f"Skipping vk summarize because file {file_signifying_successful_vk_summarize_completion} already exists and overwrite=False")
    else:
        logger.info("Skipping vk summarize because summarize=False")

    vk_count_output_dict = {}
    vk_count_output_dict["adata_path_unprocessed"] = os.path.abspath(adata_vcrs) if os.path.isfile(os.path.abspath(adata_vcrs)) else None
    vk_count_output_dict["adata_path_reference_genome_unprocessed"] = os.path.abspath(adata_reference_genome_resolved) if (adata_reference_genome_resolved and os.path.isfile(os.path.abspath(adata_reference_genome_resolved))) else None
    vk_count_output_dict["adata_path"] = os.path.abspath(adata_vcrs_clean_out) if os.path.isfile(os.path.abspath(adata_vcrs_clean_out)) else None
    vk_count_output_dict["adata_path_reference_genome"] = os.path.abspath(adata_reference_genome_clean_out) if os.path.isfile(os.path.abspath(adata_reference_genome_clean_out)) else None

    vk_count_output_dict["vcf"] = os.path.abspath(vcf_out) if (vcf_out and os.path.isfile(os.path.abspath(vcf_out))) else None
    vk_count_output_dict["vk_summarize_output_dir"] = os.path.abspath(vk_summarize_out_dir) if os.path.exists(os.path.abspath(vk_summarize_out_dir)) else None

    report_time_elapsed(start_time, "count")
    return vk_count_output_dict
