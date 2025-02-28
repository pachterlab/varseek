"""varseek fastqpp and specific helper functions."""

import os
import time
from pathlib import Path
import logging

from .constants import technology_valid_values
from .utils import (
    check_file_path_is_string_with_valid_extension,
    concatenate_fastqs,
    get_printlog,
    is_program_installed,
    is_valid_int,
    load_in_fastqs,
    make_function_parameter_to_value_dict,
    print_varseek_dry_run,
    replace_low_quality_bases_with_N_list,
    report_time_elapsed,
    run_fastqc_and_multiqc,
    save_params_to_config_file,
    save_run_info,
    set_up_logger,
    sort_fastq_files_for_kb_count,
    split_reads_by_N_list,
    trim_edges_off_reads_fastq_list,
)

logger = logging.getLogger(__name__)


def validate_input_fastqpp(params_dict):
    fastqs = params_dict["fastqs"]  # tuple
    parity = params_dict["parity"]  # str

    # fastqs
    if len(fastqs) == 0:
        raise ValueError("No fastq files provided")

    # $ type checking of the directory and text file performed earlier by load_in_fastqs

    if parity == "paired" and len(fastqs) % 2 != 0:  # if fastqs parity is paired, then ensure an even number of files
        raise ValueError("Number of fastq files must be even when parity == paired")
    for fastq in fastqs:
        check_file_path_is_string_with_valid_extension(fastq, variable_name=fastq, file_type="fastq")  # ensure that all fastq files have valid extension
        if not os.path.isfile(fastq):  # ensure that all fastq files exist
            raise ValueError(f"File {fastq} does not exist")

    # technology
    technology = params_dict.get("technology", None)
    technology_valid_values_lower = {x.lower() for x in technology_valid_values}
    if technology is not None:
        if technology.lower() not in technology_valid_values_lower:
            raise ValueError(f"Technology must be None or one of {technology_valid_values_lower}")

    parity_valid_values = {"single", "paired"}
    if params_dict["parity"] not in parity_valid_values:
        raise ValueError(f"Parity must be one of {parity_valid_values}")

    # directories
    if not isinstance(params_dict.get("out", None), (str, Path)):
        raise ValueError(f"Invalid value for out: {params_dict.get('out', None)}")

    # optional str
    for file_name_suffix in ["split_by_Ns_and_low_quality_bases_out_suffix", "concatenate_paired_fastqs_out_suffix"]:
        if params_dict.get(file_name_suffix) is not None and not isinstance(params_dict.get(file_name_suffix), str):
            raise ValueError(f"Invalid suffix: {params_dict.get(file_name_suffix)}")

    # integers - optional just means that it's in kwargs
    for param_name, min_value, max_value, optional_value in [
        ("min_base_quality_for_splitting", 0, 93, False),
    ]:
        param_value = params_dict.get(param_name)
        if not is_valid_int(param_value, "between", min_value_inclusive=min_value, max_value_inclusive=max_value, optional=optional_value):
            raise ValueError(f"{param_name} must be an integer between {min_value} and {max_value}. Got {params_dict.get(param_name)}.")

    if not is_valid_int(params_dict["min_read_len"], ">=", 1, optional=False) and params_dict["min_read_len"] is not None:
        raise ValueError(f"min_read_len must be an integer >= 1 or None. Got {params_dict.get('min_read_len')}.")

    # boolean
    for param_name in ["split_reads_by_Ns_and_low_quality_bases", "concatenate_paired_fastqs", "dry_run", "overwrite", "sort_fastqs"]:
        if not isinstance(params_dict.get(param_name), bool):
            raise ValueError(f"{param_name} must be a boolean. Got {param_name} of type {type(params_dict.get(param_name))}.")

    if parity == "paired" and params_dict["split_reads_by_Ns_and_low_quality_bases"] and not params_dict["concatenate_paired_fastqs"]:
        raise ValueError("When parity==paired, if split_reads_by_Ns_and_low_quality_bases==True, then concatenate_paired_fastqs must also be True (split_reads_by_Ns_and_low_quality_bases messes up the paired nature of the fastqs).")

    if not isinstance(params_dict.get("multiplexed"), bool) and params_dict.get("multiplexed") is not None:
        raise ValueError(f"multiplexed must be a boolean or None. Got {params_dict.get('multiplexed')} of type {type(params_dict.get('multiplexed'))}.")


def fastqpp(
    *fastqs,
    technology=None,
    multiplexed=None,
    parity="single",
    split_reads_by_Ns_and_low_quality_bases=False,
    min_base_quality_for_splitting=5,
    min_read_len=63,
    concatenate_paired_fastqs=False,
    out=".",
    dry_run=False,
    overwrite=False,
    sort_fastqs=True,
    logging_level=None,
    save_logs=False,
    log_out_dir=None,
    **kwargs,
):
    """
    Apply quality control to fastq files. This includes trimming edges off reads, running FastQC and MultiQC, replacing low quality bases with N, splitting reads by Ns, and concatenating paired fastq files.

    # Required input arguments:
    - fastqs                            (str or list[str]) List of fastq files to be processed. If paired end, the list should contains paths such as [file1_R1, file1_R2, file2_R1, file2_R2, ...]

    # Optional input arguments:
    - technology                        (str) Technology used to generate the data. Only used if sort_fastqs=True. To see list of spported technologies, run `kb --list`. Default: None
    - multiplexed                       (bool) Indicates that the fastq files are multiplexed. Only used if sort_fastqs=True and technology is a smartseq technology. Default: None
    - parity                            (str) "single" or "paired". Only relevant if technology is bulk or a smart-seq. Default: "single"
    - split_reads_by_Ns_and_low_quality_bases   (bool) If True, split reads by Ns and low quality bases (lower than min_base_quality_for_splitting) into multiple smaller reads. If min_base_quality_for_splitting > 0, then requires seqtk to be installed. If technology == "bulk", then seqtk will speed this up significantly. Default: False
    - min_base_quality_for_splitting    (int) The minimum acceptable base quality for split_reads_by_Ns_and_low_quality_bases. Bases below this quality will split. Only used if split_reads_by_Ns_and_low_quality_bases=True. Range: 0-93. Default: 13
    - concatenate_paired_fastqs         (bool) If True, concatenate paired fastq files. Default: False
    - out                               (str) Output directory. Default: "."
    - dry_run                           (bool) If True, print the commands that would be run without actually running them. Default: False
    - overwrite                         (True/False) Whether to overwrite existing output files. Will return if any output file already exists. Default: False.
    - sort_fastqs                       (bool) If True, sort fastq files by kb count. If False, then still check the order but do not change anything. Default: True
    - logging_level                      (str) Logging level. Can also be set with the environment variable VARSEEK_LOGGING_LEVEL. Default: INFO.
    - save_logs                          (True/False) Whether to save logs to a file. Default: False.
    - log_out_dir                        (str) Directory to save logs. Default: `out`/logs

    # Hidden arguments (part of kwargs)
    - seqtk_path                       (str) Path to seqtk. Default: "seqtk"
    - split_by_Ns_and_low_quality_bases_out_suffix            (str) Suffix to add to fastq files after splitting by Ns (preceded by underscore). Default: "splitNs"
    - concatenate_paired_fastqs_out_suffix (str) Suffix to add to fastq files after concatenating paired fastq files (preceded by underscore). Default: "concatenated"
    - delete_intermediate_files        (bool) If True, delete intermediate files. Default: True
    - logger                           (logging.Logger) Logger to use. Default: None
    """

    # * 0. Informational arguments that exit early
    # Not in this function

    # * 1. Start timer
    start_time = time.perf_counter()

    # * 1.25. logger
    global logger
    if kwargs.get("logger") and isinstance(kwargs.get("logger"), logging.Logger):
        logger = kwargs.get("logger")
    else:
        if save_logs and not log_out_dir:
            log_out_dir = os.path.join(out, "logs")
        logger = set_up_logger(logger, logging_level=logging_level, save_logs=save_logs, log_dir=log_out_dir)

    # * 1.5. For the nargs="+" arguments, convert any list of length 1 to a string
    if isinstance(fastqs, (list, tuple)) and len(fastqs) == 1:
        fastqs = fastqs[0]

    # * 1.75 load in fastqs
    fastqs_original = fastqs
    fastqs = load_in_fastqs(fastqs)  # this will make it in params_dict

    # * 2. Type-checking
    params_dict = make_function_parameter_to_value_dict(1)
    validate_input_fastqpp(params_dict)
    params_dict["fastqs"] = fastqs_original  # change back for dry run and config_file

    # * 3. Dry-run
    if dry_run:
        print_varseek_dry_run(params_dict, function_name="fastqpp")
        fastqpp_dict = {"original": fastqs, "final": fastqs}
        return fastqpp_dict

    # * 4. Save params to config file and run info file
    config_file = os.path.join(out, "config", "vk_fastqpp_config.json")
    save_params_to_config_file(params_dict, config_file)

    run_info_file = os.path.join(out, "config", "vk_fastqpp_run_info.txt")
    save_run_info(run_info_file)

    # * 5. Set up default folder/file input paths, and make sure the necessary ones exist
    # all input files for vk fastqpp are required in the varseek workflow, so this is skipped

    # * 6. Set up default folder/file output paths, and make sure they don't exist unless overwrite=True
    replace_low_quality_bases_with_N_out_suffix = kwargs.get("replace_low_quality_bases_with_N_out_suffix", "addedNs")
    split_by_Ns_and_low_quality_bases_out_suffix = kwargs.get("split_by_Ns_and_low_quality_bases_out_suffix", "split")
    concatenate_paired_fastqs_out_suffix = kwargs.get("concatenate_paired_fastqs_out_suffix", "concatenatedPairs")

    os.makedirs(out, exist_ok=True)

    fastq_more_Ns_all_files = []
    split_by_Ns_and_low_quality_bases_all_files = []
    fastq_concatenated_all_files = []

    if not overwrite:
        for fastq in fastqs:
            parts_filename = fastq.split(".", 1)
            if split_reads_by_Ns_and_low_quality_bases:
                fastq_more_Ns = os.path.join(out, f"{parts_filename[0]}_{replace_low_quality_bases_with_N_out_suffix}.{parts_filename[1]}")
                fastq_more_Ns_all_files.append(fastq_more_Ns)
                fastq_split_by_N_and_low_quality_bases = os.path.join(out, f"{parts_filename[0]}_{split_by_Ns_and_low_quality_bases_out_suffix}.{parts_filename[1]}")
                split_by_Ns_and_low_quality_bases_all_files.append(fastq_split_by_N_and_low_quality_bases)
            if (concatenate_paired_fastqs or split_reads_by_Ns_and_low_quality_bases) and parity == "paired":
                fastq_concatenated = os.path.join(out, f"{parts_filename[0]}_{concatenate_paired_fastqs_out_suffix}.{parts_filename[1]}")
                fastq_concatenated_all_files.append(fastq_concatenated)

    # * 7. Define kwargs defaults
    seqtk = kwargs.get("seqtk_path", "seqtk")
    delete_intermediate_files = kwargs.get("delete_intermediate_files", True)

    # * 7.5 make sure ints are ints
    min_read_len, threads = int(min_read_len), int(threads)

    # * 8. Start the actual function
    try:
        fastqs = sort_fastq_files_for_kb_count(fastqs, technology=technology, multiplexed=multiplexed, logger=logger, check_only=(not sort_fastqs))
    except ValueError as e:
        if sort_fastqs:
            logger.warning(f"Automatic FASTQ argument order sorting for kb count could not recognize FASTQ file name format. Skipping argument order sorting.")

    if technology.lower() != "bulk" and "smartseq" not in technology.lower():
        parity = "single"

    if (concatenate_paired_fastqs or split_reads_by_Ns_and_low_quality_bases) and parity == "paired":
        if not concatenate_paired_fastqs:
            logger.info("Setting concatenate_paired_fastqs=True")
        concatenate_paired_fastqs = True
    else:
        if concatenate_paired_fastqs:
            logger.info("Setting concatenate_paired_fastqs=False")
        concatenate_paired_fastqs = False

    fastqpp_dict = {}
    fastqpp_dict["original"] = fastqs

    # see trim_edges_off_reads_fastq_list for the fastp code
    # see run_fastqc_and_multiqc for the fastqc/multiqc code

    # TODO: only process the file with sequencing data, and make sure any barcode/UMI is not processed and gets duplicated for each split
    if split_reads_by_Ns_and_low_quality_bases:  # seqtk install is checked internally, as it only applies to the bulk condition and the code can still run with the same output without it (albeit slower)
        delete_intermediate_files_original = delete_intermediate_files
        if min_base_quality_for_splitting > 0:
            if not is_program_installed(seqtk):
                raise ValueError(f"seqtk must be installed to run split_reads_by_Ns_and_low_quality_bases with min_base_quality_for_splitting > 0. Please install it and try again, set split_reads_by_Ns_and_low_quality_bases=False, or set min_base_quality_for_splitting>0.")

            # check if any file in fastq_more_Ns_all_files does not exist
            if not all(os.path.exists(f) for f in fastq_more_Ns_all_files) or overwrite:
                logger.info("Replacing low quality bases with N")
                fastqs = replace_low_quality_bases_with_N_list(rnaseq_fastq_files=fastqs, minimum_base_quality=min_base_quality_for_splitting, seqtk=seqtk, out_dir=out, logger=logger, suffix=replace_low_quality_bases_with_N_out_suffix)
            else:
                logger.warning("Fastq files with low quality bases replaced with N already exist. Skipping this step. Use overwrite=True to overwrite existing files.")
            if not delete_intermediate_files:
                fastqpp_dict["replaced_loq_quality_with_N"] = fastqs
        else:
            delete_intermediate_files = False  # don't delete intermediate file if I don't do the step above, as the intermediate file would be either the fastp output file or the original fastq, either of which I don't want to delete

        # check if any file in split_by_Ns_and_low_quality_bases_all_files does not exist
        if not all(os.path.exists(f) for f in split_by_Ns_and_low_quality_bases_all_files) or overwrite:
            logger.info("Splitting reads by Ns")
            fastqs = split_reads_by_N_list(fastqs, minimum_sequence_length=min_read_len, delete_original_files=delete_intermediate_files, out_dir=out, logger=logger, suffix=split_by_Ns_and_low_quality_bases_out_suffix, seqtk=seqtk)
        else:
            logger.warning("Fastq files with reads split by N already exist. Skipping this step. Use overwrite=True to overwrite existing files.")
        if not delete_intermediate_files:
            fastqpp_dict["split_by_N_and_low_quality"] = fastqs

        delete_intermediate_files = delete_intermediate_files_original
    else:
        delete_intermediate_files = False  # if I skip this step, then I don't want to delete the intermediate files during the concatenation step

    if concatenate_paired_fastqs:
        # check if any file in fastq_concatenated_all_files does not exist
        if not all(os.path.exists(f) for f in fastq_concatenated_all_files) or overwrite:
            logger.info("Concatenating paired fastq files")
            rnaseq_fastq_files_list_copy = []
            for i in range(0, len(fastqs), 2):
                file1 = fastqs[i]
                file2 = fastqs[i + 1]
                logger.info(f"Concatenating {file1} and {file2}")
                file_concatenated = concatenate_fastqs(file1, file2, out_dir=out, delete_original_files=delete_intermediate_files, suffix=concatenate_paired_fastqs_out_suffix)
                rnaseq_fastq_files_list_copy.append(file_concatenated)
            fastqs = rnaseq_fastq_files_list_copy
            fastqpp_dict["concatenated"] = fastqs
        else:
            logger.warning("Concatenated fastq files already exist. Skipping this step. Use overwrite=True to overwrite existing files.")

    fastqpp_dict["final"] = fastqs

    logger.info("Returning a dictionary with keys describing the fastq files and values pointing to their file paths")
    report_time_elapsed(start_time, logger=logger, function_name="fastqpp")

    return fastqpp_dict
