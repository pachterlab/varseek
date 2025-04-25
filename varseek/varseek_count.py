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
import pandas as pd

import varseek as vk
from varseek.utils import (
    check_file_path_is_string_with_valid_extension,
    check_that_two_paths_are_the_same_if_both_provided_otherwise_set_them_equal,
    is_valid_int,
    load_in_fastqs,
    correct_adata_barcodes_for_running_paired_data_in_single_mode,
    make_function_parameter_to_value_dict,
    report_time_elapsed,
    save_params_to_config_file,
    save_run_info,
    set_up_logger,
    sort_fastq_files_for_kb_count,
    set_varseek_logging_level_and_filehandler
)
from varseek.varseek_clean import needs_for_normal_genome_matrix

from .constants import (
    non_single_cell_technologies,
    supported_downloadable_normal_reference_genomes_with_kb_ref,
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


def validate_input_count(params_dict):
    # $ fastqs, technology will get checked through fastqpp

    # other required files
    for param_name, file_type in {
        "index": "index",
        "t2g": "t2g",
    }.items():
        check_file_path_is_string_with_valid_extension(params_dict[param_name], param_name, file_type=file_type, required=False)
        if not os.path.isfile(params_dict[param_name]) or params_dict[param_name] != "None":  # ensure that all files exist
            raise ValueError(f"File {params_dict[param_name]} does not exist")

    # file paths
    check_file_path_is_string_with_valid_extension(params_dict.get("reference_genome_index", None), "reference_genome_index", "index")
    check_file_path_is_string_with_valid_extension(params_dict.get("reference_genome_t2g", None), "reference_genome_t2g", "t2g")
    check_file_path_is_string_with_valid_extension(params_dict.get("adata_reference_genome", None), "adata_reference_genome", "adata")

    if not is_valid_int(params_dict.get("threads", None), threshold_type=">=", threshold_value=1):
        raise ValueError(f"Threads must be a positive integer, got {params_dict.get('threads')}")

    # out dirs
    for param_name in ["out", "kb_count_vcrs_out_dir", "kb_count_reference_genome_out_dir", "vk_summarize_out_dir"]:
        if not isinstance(params_dict.get(param_name, None), (str, Path)) and params_dict.get(param_name) is not None:
            raise ValueError(f"Invalid value for {param_name}: {params_dict.get(param_name, None)}")

    # booleans
    for param_name in ["dry_run", "overwrite", "sort_fastqs", "disable_fastqpp", "disable_clean", "disable_summarize"]:
        if not isinstance(params_dict.get(param_name), bool):
            raise ValueError(f"{param_name} must be a boolean. Got {param_name} of type {type(params_dict.get(param_name))}.")

    # strings
    parity_valid_values = {"single", "paired", None}
    if params_dict["parity"] not in parity_valid_values:
        raise ValueError(f"Parity must be one of {parity_valid_values}")
    if params_dict["technology"] in {"BULK", "SMARTSEQ2"} and params_dict["parity"] is None:
        raise ValueError("Parity must be set to 'single' or 'paired' for bulk or smartseq2 data")

    if params_dict.get("parity_kb_count") is not None and params_dict.get("parity_kb_count") not in parity_valid_values:
        raise ValueError(f"parity_kb_count must be one of {parity_valid_values}")

    if params_dict.get("parity_kb_count") == "paired" and params_dict["parity"] == "single":
        raise ValueError("If parity_kb_count is 'paired', then parity must be 'paired' as well")

    if params_dict.get("parity_kb_count") == "single" and params_dict["parity"] == "paired":
        logger.info("parity='paired' and parity_kb_count='single'. This means that kb count will run in single-end mode on this paired-end data, which will enable different pairs to be processed independently and thus potentially detect different variants. To turn this feature off, set parity_kb_count='paired'.")

    strand_valid_values = {"unstranded", "forward", "reverse", None}
    if params_dict["strand"] not in strand_valid_values:
        raise ValueError(f"Strand must be one of {strand_valid_values}")

    out = params_dict.get("out", ".")
    kb_count_reference_genome_out_dir = params_dict.get("kb_count_reference_genome_out_dir") if params_dict.get("kb_count_reference_genome_out_dir") else f"{out}/kb_count_out_reference_genome"
    if params_dict.get("qc_against_gene_matrix") and not os.path.exists(kb_count_reference_genome_out_dir):  # align to this genome if (1) adata doesn't exist and (2) qc_against_gene_matrix=True (because I need the BUS file for this)  # purposely omitted overwrite because it is reasonable to expect that someone has pre-computed this matrix and doesn't want it recomputed under any circumstances (and if they did, then simply point to a different directory)
        if not os.path.exists(params_dict.get("reference_genome_index")) or not os.path.exists(params_dict.get("reference_genome_t2g")):
            raise ValueError(f"Reference genome index {params_dict.get('reference_genome_index')} or t2g {params_dict.get('reference_genome_t2g')} does not exist. Please provide a valid reference genome index and t2g file created with the `kb ref` command (a standard reference genome index/t2g, *not* a variant reference).")

    # kb count stuff
    for argument_type, argument_set in varseek_count_only_allowable_kb_count_arguments.items():
        for argument in argument_set:
            argument = argument[2:]
            if argument in params_dict:
                argument_value = params_dict[argument]
                if argument_type == "zero_arguments":
                    if not isinstance(argument_value, bool):  # all zero-arguments are bool
                        raise ValueError(f"{argument} must be a boolean. Got {type(argument_value)}.")
                elif argument_type == "one_argument":
                    if not isinstance(argument_value, str):  # all one-arguments are string
                        raise ValueError(f"{argument} must be a string. Got {type(argument_value)}.")
                elif argument_type == "multiple_arguments":
                    pass

    # --nums required when qc_against_gene_matrix=True
    if params_dict.get("qc_against_gene_matrix"):
        for arg in ["kb_count_reference_genome_dir", "kb_count_reference_genome_out_dir"]:
            kb_count_normal_dir = params_dict.get(arg)
            if kb_count_normal_dir and os.path.exists(kb_count_normal_dir):
                run_info_json = os.path.join(kb_count_normal_dir, "run_info.json")
                with open(run_info_json, "r") as f:
                    data = json.load(f)
                if "--num" not in data["call"]:
                    raise ValueError(f"--num must be included in the provided value for {arg}. Please run kb count on the normal genome again, or provide a new path for {arg} to allow varseek count to make this file for you.")


# don't worry if it says an argument is unused, as they will all get put in params_dict for each respective function and passed to the child functions
@report_time_elapsed
def count(
    fastqs,
    index,
    technology,
    t2g="None",
    k=59,  # params
    run_kb_count_against_reference_genome=False,
    qc_against_gene_matrix=False,
    account_for_strand_bias=False,
    strand_bias_end=None,
    read_length=None,
    strand=None,
    mm=True,
    union=True,
    parity="single",
    reference_genome_index=None,  # optional inputs
    reference_genome_t2g=None,
    gtf=None,
    out=".",  # optional outputs
    kb_count_vcrs_out_dir=None,
    kb_count_reference_genome_out_dir=None,
    vk_summarize_out_dir=None,
    disable_fastqpp=False,  # general
    disable_clean=False,
    disable_summarize=False,
    chunksize=None,
    dry_run=False,
    overwrite=False,
    sort_fastqs=True,
    threads=2,
    logging_level=None,
    save_logs=False,
    log_out_dir=None,
    **kwargs,  # * including all arguments for vk fastqpp, clean, summarize, and kb count
):
    """
    Perform variant screening on sequencing data. Wraps around varseek fastqpp, kb count, varseek clean, and kb summarize.

    # Required input arguments:
    - fastqs                                (str or list[str]) List of fastq files to be processed. If paired end, the list should contains paths such as [file1_R1, file1_R2, file2_R1, file2_R2, ...]
    - index                                 (str)  Path to variant index generated by varseek ref
    - technology                            (str)  Technology used to generate the data. To see list of spported technologies, run `kb --list`.

    # Additional parameters
    - k                                     (int) The length of each k-mer in the kallisto reference index construction. Corresponds to `k` used in the earlier varseek commands (i.e., varseek ref). If using a downloaded index from varseek ref -d, then check the description for the k value used to construct this index with varseek ref --list_downloadable_references. Default: 59.
    - run_kb_count_against_reference_genome (bool) Whether to run kb count against the normal reference genome. This is only needed if you want to use the normal reference genome for downstream analysis. Default: False.
    - qc_against_gene_matrix                (bool) Whether to apply correction for qc against gene matrix. If a read maps to 2+ VCRSs that belong to different genes, then cross-reference with the reference genome to determine which gene the read belongs to, and set all VCRSs that do not correspond to this gene to 0 for that read. Also, cross-reference all reads that map to 1 VCRS and ensure that the reads maps to the gene corresponding to this VCRS, or else set this value to 0 in the count matrix. Default: False.
    - account_for_strand_bias               (bool) Whether to account for strand bias from stranded single-cell technologies. Default: False.
    - strand_bias_end                       (str) The end of the read to use for strand bias correction. Either "5p" or "3p". Must be provided if and only if account_for_strand_bias=True. Default: None.
    - read_length                           (int) The read length used in the experiment. Must be provided if and only if account_for_strand_bias=True. Default: None.
    - strand                                (str)  The strandedness of the data. Either "unstranded", "forward", or "reverse". Default: None.
    - mm                                    (bool)  If True, use the multi-mapping reads. Default: True.
    - union                                 (bool)  If True, use the union of the read mappings. Default: True.
    - parity                                (str)  The parity of the reads for vk fastqpp. Either "single" or "paired". Only used if technology is bulk or a smart-seq. For parity in kb count, see parity_kb_count. Default: "single".

    # Optional input arguments
    - t2g                                   (str) Path to t2g file generated by varseek ref. Default: "None" (will be created if not provided).
    - reference_genome_index                (str) Path to index file for the "normal" reference genome. Created if not provided. Only used if qc_against_gene_matrix=True (see vk clean --help). Default: None.
    - reference_genome_t2g                  (str) Path to t2g file for the "normal" reference genome. Created if not provided. Only used if qc_against_gene_matrix=True (see vk clean --help). Default: None.
    - gtf                                   (str) Path to the GTF file. Only used when account_for_strand_bias=True and either (1) strand_bias_end='3p' and/or (2) some VCRSs are derived from genome sequences. Default: None.

    # Optional output file paths: (only needed if changing/customizing file names or locations):
    - out                                   (str) Output directory. Default: ".".
    - kb_count_vcrs_out_dir                 (str) Output directory for kb count. Default: `out`/kb_count_out_vcrs
    - kb_count_reference_genome_out_dir     (str) Output directory for kb count on "normal" reference genome. Default: `out`/kb_count_out_reference_genome
    - vk_summarize_out_dir                  (str) Output directory for vk summarize. Default: `out`/vk_summarize

    # General arguments:
    - disable_fastqpp                       (bool) If True, skip fastqpp step. Default: False.
    - disable_clean                         (bool) If True, skip clean step. Default: False.
    - disable_summarize                     (bool) If True, skip summarize step. Default: False.
    - chunksize                             (int) Number of BUS file lines to process at a time. If None, then all lines will be processed at once. Default: None.
    - dry_run                               (bool) If True, print the commands that would be run without actually running them. Default: False.
    - overwrite                             (bool) If True, overwrite existing files. Default: False.
    - sort_fastqs                           (bool) If True, sort fastq files by kb count. If False, then still check the order but do not change anything. Default: True
    - threads                               (int) Number of threads to use. Default: 2.
    - logging_level                         (str) Logging level. Can also be set with the environment variable VARSEEK_LOGGING_LEVEL. Default: INFO.
    - save_logs                             (True/False) Whether to save logs to a file. Default: False.
    - log_out_dir                           (str) Directory to save logs. Default: None (do not save logs).

    # Hidden arguments (part of kwargs):
    - num                                  (bool) If True, use the --num argument in kb count. Default: True.
    - parity_kb_count                      (str) The parity of the reads for kb count. Always recommended to run in single. Default: "single".

    For a complete list of supported parameters, see the documentation for varseek fastqpp, kb count, varseek clean, and varseek summarize. Note that any shared parameter names between functions are meant to have identical purposes.
    """

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
    params_dict["adata_vcrs"] = "placeholder/adata.h5ad"  # this is just a placeholder, but it is needed for type checking
    params_dict["adata"] = "placeholder/adata_cleaned.h5ad"  # this is just a placeholder, but it is needed for type checking
    params_dict["kb_count_reference_genome_dir"] = None  # this is just a placeholder, but it is needed for type checking

    # Set params_dict_for_type_checking to default values of children functions - important so type checking works properly
    count_signature = inspect.signature(count)
    for function in (vk.varseek_fastqpp.fastqpp, vk.varseek_clean.clean, vk.varseek_summarize.summarize):
        signature = inspect.signature(function)
        for key in signature.parameters.keys():
            if key not in params_dict and key not in count_signature.parameters.keys():
                params_dict[key] = signature.parameters[key].default

    vk.varseek_fastqpp.validate_input_fastqpp(params_dict)  # this passes all vk count parameters to the function - I could only pass in the vk fastqpp parameters here if desired (and likewise below), but there should be no naming conflicts anyways
    vk.varseek_clean.validate_input_clean(params_dict)
    vk.varseek_summarize.validate_input_summarize(params_dict)
    validate_input_count(params_dict)

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
        kb_count_vcrs_out_dir = os.path.join(out, "kb_count_out_vcrs") if not kwargs.get("kb_count_vcrs_dir") else kwargs["kb_count_vcrs_dir"]
    if not kb_count_reference_genome_out_dir:
        kb_count_reference_genome_out_dir = os.path.join(out, "kb_count_out_reference_genome") if not kwargs.get("kb_count_reference_genome_dir") else kwargs["kb_count_reference_genome_dir"]
    if not vk_summarize_out_dir:
        vk_summarize_out_dir = os.path.join(out, "vk_summarize")

    if not dry_run:
        os.makedirs(out, exist_ok=True)
        os.makedirs(kb_count_vcrs_out_dir, exist_ok=True)

    # for kb count --> vk clean
    kb_count_vcrs_out_dir, kwargs["kb_count_vcrs_dir"] = check_that_two_paths_are_the_same_if_both_provided_otherwise_set_them_equal(kb_count_vcrs_out_dir, kwargs.get("kb_count_vcrs_dir"))  # check that, if kb_count_vcrs_dir and kb_count_vcrs_out_dir are both provided, they are the same directory; otherwise, if only one is provided, then make them equal to each other
    kb_count_reference_genome_out_dir, kwargs["kb_count_reference_genome_dir"] = check_that_two_paths_are_the_same_if_both_provided_otherwise_set_them_equal(kb_count_reference_genome_out_dir, kwargs.get("kb_count_reference_genome_dir"))  # same story as above but for kb_count_reference_genome and kb_count_reference_genome_out_dir

    adata_vcrs = f"{kb_count_vcrs_out_dir}/counts_unfiltered/adata.h5ad" if not kwargs.get("adata_vcrs") else kwargs.get("adata_vcrs")  # from vk clean
    adata_reference_genome = f"{kb_count_reference_genome_out_dir}/counts_unfiltered/adata.h5ad" if not kwargs.get("adata_reference_genome") else kwargs.get("adata_reference_genome")  # from vk clean
    adata_vcrs_clean_out = f"{out}/adata_cleaned.h5ad" if not kwargs.get("adata_vcrs_clean_out") else kwargs.get("adata_vcrs_clean_out")  # from vk clean
    adata_reference_genome_clean_out = f"{out}/adata_cleaned.h5ad" if not kwargs.get("adata_reference_genome_clean_out") else kwargs.get("adata_reference_genome_clean_out")  # from vk clean
    vcf_out = os.path.join(out, "variants.vcf") if not kwargs.get("vcf_out") else kwargs["vcf_out"]  # from vk clean
    stats_file = os.path.join(vk_summarize_out_dir, "varseek_summarize_stats.txt") if not kwargs.get("stats_file") else kwargs["stats_file"]  # from vk summarize

    for file in [stats_file]:  # purposely excluded adata_reference_genome because it is fine if someone provides this as input even if overwrite=False; and purposely excluded adata_vcrs, adata_vcrs_clean_out, adata_reference_genome_clean_out, kb_count_vcrs_out_dir, kb_count_reference_genome_out_dir for the reasons provided in vk ref
        if os.path.isfile(file) and not overwrite:
            raise FileExistsError(f"Output file {file} already exists. Please delete it or specify a different output directory or set overwrite=True.")

    # no need for file_signifying_successful_vk_fastqpp_completion because overwrite=False just gives warning rather than error in fastqpp
    file_signifying_successful_kb_count_vcrs_completion = adata_vcrs
    file_signifying_successful_kb_count_reference_genome_completion = adata_reference_genome
    file_signifying_successful_vk_clean_completion = adata_vcrs_clean_out
    file_signifying_successful_vk_summarize_completion = stats_file

    # * 7. Define kwargs defaults
    kwargs["parity_kb_count"] = kwargs.get("parity_kb_count", "single")
    kwargs["num"] = kwargs.get("num", True)

    # * 7.5 make sure ints are ints
    k, threads = int(k), int(threads)

    # * 8. Start the actual function
    technology = technology.upper()
    
    if isinstance(fastqs, list):
        fastqs = tuple(fastqs)

    fastqs_unsorted = fastqs
    try:
        fastqs = sort_fastq_files_for_kb_count(fastqs, technology=technology, multiplexed=kwargs.get("multiplexed"), check_only=(not sort_fastqs))
    except ValueError as e:
        if sort_fastqs:
            logger.warning(f"Automatic FASTQ argument order sorting for kb count could not recognize FASTQ file name format. Skipping argument order sorting.")

    # so parity_vcrs is set correctly - copied from fastqpp
    concatenate_paired_fastqs = kwargs.get("concatenate_paired_fastqs", False)
    split_reads_by_Ns_and_low_quality_bases = kwargs.get("split_reads_by_Ns_and_low_quality_bases", False)
    if (concatenate_paired_fastqs or split_reads_by_Ns_and_low_quality_bases) and parity == "paired":
        if not concatenate_paired_fastqs:
            logger.info("Setting concatenate_paired_fastqs=True")
        concatenate_paired_fastqs = True
    else:
        if concatenate_paired_fastqs:
            logger.info("Setting concatenate_paired_fastqs=False")
        concatenate_paired_fastqs = False
    kwargs["concatenate_paired_fastqs"] = concatenate_paired_fastqs

    if disable_clean:
        qc_against_gene_matrix = False  # disable_clean gets priority

    if not kwargs.get("length_required"):
        logger.info("Setting length_required to %s if fastqpp is run", k)
        kwargs["length_required"] = k

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

    # vk fastqpp
    if not any([kwargs.get(fastqpp_param) for fastqpp_param in ["quality_control_fastqs", "split_reads_by_Ns_and_low_quality_bases", "concatenate_paired_fastqs"]]):
        disable_fastqpp_original = disable_fastqpp
        disable_fastqpp = True

    if not disable_fastqpp:  # don't do the whole overwrite thing here because it is the first function, and a user should know if they are overwriting their fastqs
        kwargs_vk_fastqpp = {key: value for key, value in kwargs.items() if ((key in all_parameter_names_set_vk_fastqpp) and (key not in count_signature.parameters.keys()))}
        # update anything in kwargs_vk_fastqpp that is not fully updated in (vk count's) kwargs (should be nothing or very close to it, as I try to avoid these double-assignments by always keeping kwargs in kwargs)
        # eg kwargs_vk_summarize['mykwarg'] = mykwarg

        logger.info("Running vk fastqpp")
        if dry_run:
            logger.warning("Note: during dry run, the fastq's passed into kb count will be the default fastqs, not any updated fastqs from fastqpp")
        fastqpp_dict = vk.fastqpp(fastqs, technology=technology, parity=parity, out=out, dry_run=dry_run, overwrite=overwrite, sort_fastqs=sort_fastqs, logging_level=logging_level, save_logs=save_logs, log_out_dir=log_out_dir, **kwargs_vk_fastqpp)  # intentionally do not set overwrite = True here because it is the first function, and a user should know if they are overwriting their fastqs

        fastqs_vcrs = fastqpp_dict["final"]
        fastqs_reference_genome = fastqpp_dict["quality_controlled"] if "quality_controlled" in fastqpp_dict else fastqs
    else:
        if disable_fastqpp_original:
            logger.info("Skipping vk fastqpp because disable_fastqpp=True")
        else:
            logger.info("Skipping vk fastqpp because there was no use for it")
        fastqs_vcrs = fastqs
        fastqs_reference_genome = fastqs
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
            parity_vcrs = "single" if kwargs.get("concatenate_paired_fastqs") else kwargs["parity_kb_count"]  # I set the default value earlier, so I don't need to use the .get method
            kb_count_command.extend(["--parity", parity_vcrs])

        if qc_against_gene_matrix:
            kb_count_command.extend(["--union",])  # don't need mm here, as mm does not affect the BUS file (only the count matrix)
        if qc_against_gene_matrix or kwargs.get("apply_split_reads_by_Ns_correction") or kwargs.get("apply_dlist_correction") or kwargs.get("num"):
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
                        kb_count_command.extend([f"{leading_dashes}{argument}", value])
                    else:  # multiple_arguments or something else
                        pass

        kb_count_command += fastqs_vcrs

        if dry_run:
            print(" ".join(kb_count_command))
        else:
            logger.info(f"Running kb count with command: {' '.join(kb_count_command)}")
            subprocess.run(kb_count_command, check=True)

            if os.path.exists(adata_vcrs) and parity == "paired" and kwargs["parity_kb_count"] == "single" and technology in {"BULK", "SMARTSEQ2"}:
                _ = correct_adata_barcodes_for_running_paired_data_in_single_mode(kb_count_vcrs_out_dir, adata_out=adata_vcrs)  # will check if the correction has already occurred internally
    else:
        logger.info(f"Skipping kb count because file {file_signifying_successful_kb_count_vcrs_completion} already exists and overwrite=False")

    # # kb count, reference genome
    if ((not os.path.exists(file_signifying_successful_kb_count_reference_genome_completion)) and any(kwargs.get(value, False) for value in needs_for_normal_genome_matrix)) or (qc_against_gene_matrix and (not os.path.exists(kb_count_reference_genome_out_dir) or len(os.listdir(kb_count_reference_genome_out_dir)) == 0)):  # align to this genome if either (1) adata doesn't exist and I do downstream analysis with the normal gene count matrix for scRNA-seq data (ie not bulk) or (2) [qc_against_gene_matrix=True and kb_count_reference_genome_out_dir is nonexistent/empty (because I need the BUS file for this)]  # purposely omitted overwrite because it is reasonable to expect that someone has pre-computed this matrix and doesn't want it recomputed under any circumstances (and if they did, then simply point to a different directory)
        run_kb_count_against_reference_genome = True
    
    if run_kb_count_against_reference_genome:
        reference_genome_index = reference_genome_index if reference_genome_index else os.path.join(out, "reference_genome_index.idx")
        reference_genome_t2g = reference_genome_t2g if reference_genome_t2g else os.path.join(out, "reference_genome_t2g.t2g")

        if not os.path.exists(reference_genome_index) or not os.path.exists(reference_genome_t2g):  # download reference if does not exist
            raise ValueError(f"Reference genome index {reference_genome_index} or t2g {reference_genome_t2g} does not exist. Please provide a valid reference genome index and t2g file created with the `kb ref` command (a standard reference genome index/t2g, *not* a variant reference).")

        os.makedirs(kb_count_reference_genome_out_dir, exist_ok=True)

        #!!! WT vcrs alignment, copied from previous notebook 1_2 (still not implemented in here correctly)
        # if os.path.exists(wt_vcrs_index) and (not os.path.exists(kb_count_out_wt_vcrs) or len(os.listdir(kb_count_out_wt_vcrs)) == 0):
        #     kb_count_command = ["kb", "count", "-t", str(threads), "-k", str(k), "-i", wt_vcrs_index, "-g", wt_vcrs_t2g, "-x", technology, "--num", "--h5ad", "--parity", "single", "--strand", strand, "-o", kb_count_out_wt_vcrs] + rnaseq_fastq_files_final
        #     subprocess.run(kb_count_command, check=True)

        # kb count, reference genome
        kb_count_standard_index_command = [
            "kb",
            "count",
            "-t",
            str(threads),
            "-i",
            reference_genome_index,
            "-g",
            reference_genome_t2g,
            "-x",
            technology,
            "--h5ad",
            "-o",
            kb_count_reference_genome_out_dir,
        ]

        if strand:
            kb_count_standard_index_command.extend(["--strand", strand])
        if qc_against_gene_matrix or kwargs.get("num"):
            kb_count_standard_index_command.extend(["--num"])
        if technology in {"BULK", "SMARTSEQ2"}:
            kb_count_standard_index_command.extend(["--parity", parity])

        # assumes any argument in varseek count matches kb count identically, except dashes replaced with underscores
        params_dict_kb_count_standard = make_function_parameter_to_value_dict(1)  # will reflect any updated values to variables found in vk count signature and anything in kwargs
        for dict_key, arguments in varseek_count_only_allowable_kb_count_arguments.items():
            for argument in list(arguments):
                dash_count = len(argument) - len(argument.lstrip("-"))
                leading_dashes = "-" * dash_count
                argument = argument.lstrip("-").replace("-", "_")
                if argument in params_dict_kb_count_standard:
                    value = params_dict_kb_count_standard[argument]
                    if dict_key == "zero_arguments":
                        if value:  # only add if value is True
                            kb_count_standard_index_command.append(f"{leading_dashes}{argument}")
                    elif dict_key == "one_argument":
                        kb_count_standard_index_command.extend([f"{leading_dashes}{argument}", value])
                    else:  # multiple_arguments or something else
                        pass

        kb_count_standard_index_command += fastqs_reference_genome  # the ones unprocessed by fastqpp

        if dry_run:
            print(" ".join(kb_count_standard_index_command))
        else:
            logger.info(f"Running kb count for reference genome with command: {' '.join(kb_count_standard_index_command)}")
            subprocess.run(kb_count_standard_index_command, check=True)

    else:
        logger.info(f"Skipping kb count for reference genome because the reference genome adata object was not needed and/or the file '{file_signifying_successful_kb_count_reference_genome_completion}' already exists. Note that even setting overwrite=True will still not overwrite this particular directory")

    if not os.path.exists(kb_count_reference_genome_out_dir) or len(os.listdir(kb_count_reference_genome_out_dir)) == 0:
        kb_count_reference_genome_out_dir, kwargs["kb_count_reference_genome_dir"] = None, None  # don't pass anything into clean if they're empty

    # vk clean
    if not disable_clean:
        if not os.path.exists(file_signifying_successful_vk_clean_completion) or overwrite:
            kwargs_vk_clean = {key: value for key, value in kwargs.items() if ((key in all_parameter_names_set_vk_clean) and (key not in count_signature.parameters.keys()))}
            # update anything in kwargs_vk_clean that is not fully updated in (vk count's) kwargs (should be nothing or very close to it, as I try to avoid these double-assignments by always keeping kwargs in kwargs)
            # eg kwargs_vk_clean['mykwarg'] = mykwarg

            logger.info("Running vk clean")
            _ = vk.clean(adata_vcrs=adata_vcrs, vcrs_index=index, vcrs_t2g=t2g, technology=technology, fastqs=fastqs, reference_genome_t2g=reference_genome_t2g, k=k, qc_against_gene_matrix=qc_against_gene_matrix, account_for_strand_bias=account_for_strand_bias, strand_bias_end=strand_bias_end, read_length=read_length, gtf=gtf, mm=mm, union=union, parity=parity, out=out, chunksize=chunksize, dry_run=dry_run, overwrite=True, sort_fastqs=sort_fastqs, logging_level=logging_level, save_logs=save_logs, log_out_dir=log_out_dir, **kwargs_vk_clean)  # kb_count_reference_genome_dir is passed in via kwargs, as is adata_reference_genome
        else:
            logger.info(f"Skipping vk clean because file {file_signifying_successful_vk_clean_completion} already exists and overwrite=False")
        adata = adata_vcrs_clean_out  # for vk summarize
    else:
        logger.info("Skipping vk clean because disable_clean=True")
        adata = adata_vcrs  # for vk summarize

    # # vk summarize
    if not disable_summarize:
        if not os.path.exists(file_signifying_successful_vk_summarize_completion) or overwrite:
            kwargs_vk_summarize = {key: value for key, value in kwargs.items() if ((key in all_parameter_names_set_vk_summarize) and (key not in count_signature.parameters.keys()))}
            # update anything in kwargs_vk_summarize that is not fully updated in (vk count's) kwargs (should be nothing or very close to it, as I try to avoid these double-assignments by always keeping kwargs in kwargs)
            # eg kwargs_vk_summarize['mykwarg'] = mykwarg

            logger.info("Running vk summarize")
            try:
                _ = vk.summarize(adata=adata, technology=technology, out=vk_summarize_out_dir, dry_run=dry_run, overwrite=True, logging_level=logging_level, save_logs=save_logs, log_out_dir=log_out_dir, **kwargs_vk_summarize)
            except Exception as e:
                if os.path.isfile(file_signifying_successful_vk_summarize_completion):
                    os.remove(file_signifying_successful_vk_summarize_completion)  # remove the file vk summarize stats file so that the vk count can be rerun
                logger.error(f"Error in vk summarize: {e}")
                raise e
        else:
            logger.info(f"Skipping vk summarize because file {file_signifying_successful_vk_summarize_completion} already exists and overwrite=False")
    else:
        logger.info("Skipping vk summarize because disable_summarize=True")

    vk_count_output_dict = {}
    vk_count_output_dict["adata_path_unprocessed"] = os.path.abspath(adata_vcrs) if os.path.isfile(os.path.abspath(adata_vcrs)) else None
    vk_count_output_dict["adata_path_reference_genome_unprocessed"] = os.path.abspath(adata_reference_genome) if os.path.isfile(os.path.abspath(adata_reference_genome)) else None
    vk_count_output_dict["adata_path"] = os.path.abspath(adata_vcrs_clean_out) if os.path.isfile(os.path.abspath(adata_vcrs_clean_out)) else None
    vk_count_output_dict["adata_path_reference_genome"] = os.path.abspath(adata_reference_genome_clean_out) if os.path.isfile(os.path.abspath(adata_reference_genome_clean_out)) else None

    vk_count_output_dict["vcf"] = os.path.abspath(vcf_out) if os.path.isfile(os.path.abspath(vcf_out)) else None
    vk_count_output_dict["vk_summarize_output_dir"] = os.path.abspath(vk_summarize_out_dir) if os.path.exists(os.path.abspath(vk_summarize_out_dir)) else None

    return vk_count_output_dict
