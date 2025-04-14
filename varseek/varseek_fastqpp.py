"""varseek fastqpp and specific helper functions."""

import inspect
import logging
import os
import time
from pathlib import Path

from .constants import technology_valid_values
from .utils import (
    check_file_path_is_string_with_valid_extension,
    concatenate_fastqs,
    is_program_installed,
    is_valid_int,
    load_in_fastqs,
    make_function_parameter_to_value_dict,
    perform_fastp_trimming_and_filtering,
    get_varseek_dry_run,
    replace_low_quality_bases_with_N_list,
    report_time_elapsed,
    run_fastqc_and_multiqc,
    save_params_to_config_file,
    save_run_info,
    set_up_logger,
    sort_fastq_files_for_kb_count,
    set_varseek_logging_level_and_filehandler,
    split_reads_by_N_list,
    trim_edges_off_reads_fastq_list,
)

logger = logging.getLogger(__name__)
logger = set_up_logger(logger, logging_level="INFO", save_logs=False, log_dir=None)


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
        ("cut_window_size", 1, 1000, False),
        ("cut_mean_quality", 1, 36, False),
        ("qualified_quality_phred", 1, 40, False),
        ("unqualified_percent_limit", 0, 100, False),
        ("average_qual", 0, 40, False),
        ("n_base_limit", 1, 50, False),
        ("length_required", 0, 9999, False),
        ("threads", 1, 100, False),
        ("min_base_quality_for_splitting", 0, 93, False),
    ]:
        param_value = params_dict.get(param_name)
        if not is_valid_int(param_value, "between", min_value_inclusive=min_value, max_value_inclusive=max_value, optional=optional_value):
            raise ValueError(f"{param_name} must be an integer between {min_value} and {max_value}. Got {params_dict.get(param_name)}.")

    if not is_valid_int(params_dict["length_required"], ">=", 1, optional=False) and params_dict["length_required"] is not None:
        raise ValueError(f"length_required must be an integer >= 1 or None. Got {params_dict.get('length_required')}.")

    # boolean
    for param_name in ["quality_control_fastqs", "split_reads_by_Ns_and_low_quality_bases", "concatenate_paired_fastqs", "cut_front", "cut_tail", "disable_adapter_trimming", "disable_quality_filtering", "disable_length_filtering", "dont_eval_duplication", "disable_trim_poly_g", "dry_run", "overwrite", "sort_fastqs"]:
        if not isinstance(params_dict.get(param_name), bool):
            raise ValueError(f"{param_name} must be a boolean. Got {param_name} of type {type(params_dict.get(param_name))}.")

    if parity == "paired" and params_dict["split_reads_by_Ns_and_low_quality_bases"] and not params_dict["concatenate_paired_fastqs"]:
        raise ValueError("When parity==paired, if split_reads_by_Ns_and_low_quality_bases==True, then concatenate_paired_fastqs must also be True (split_reads_by_Ns_and_low_quality_bases messes up the paired nature of the fastqs).")

    if not isinstance(params_dict.get("multiplexed"), bool) and params_dict.get("multiplexed") is not None:
        raise ValueError(f"multiplexed must be a boolean or None. Got {params_dict.get('multiplexed')} of type {type(params_dict.get('multiplexed'))}.")

    if not isinstance(params_dict.get("failed_out"), (bool, str)):
        raise ValueError(f"failed_out must be a boolean or string. Got {params_dict.get('failed_out')} of type {type(params_dict.get('failed_out'))}.")

@report_time_elapsed
def fastqpp(
    fastqs,
    technology,
    multiplexed=None,
    parity="single",
    quality_control_fastqs=False,
    cut_front=False,
    cut_tail=False,
    cut_window_size=4,
    cut_mean_quality=15,
    disable_adapter_trimming=False,
    qualified_quality_phred=15,
    unqualified_percent_limit=40,
    average_qual=15,
    n_base_limit=10,
    disable_quality_filtering=False,
    length_required=31,
    disable_length_filtering=False,
    dont_eval_duplication=False,
    disable_trim_poly_g=False,
    failed_out=False,
    split_reads_by_Ns_and_low_quality_bases=False,
    min_base_quality_for_splitting=5,
    concatenate_paired_fastqs=False,
    out=".",
    dry_run=False,
    overwrite=False,
    sort_fastqs=True,
    threads=2,
    logging_level=None,
    save_logs=False,
    log_out_dir=None,
    **kwargs,
):
    """
    Apply quality control to fastq files. This includes trimming edges off reads, running FastQC and MultiQC, replacing low quality bases with N, splitting reads by Ns, and concatenating paired fastq files.

    # Required input arguments:
    - fastqs                            (str or list[str]) List of fastq files to be processed. If paired end, the list should contains paths such as [file1_R1, file1_R2, file2_R1, file2_R2, ...]
    - technology                        (str) Technology used to generate the data. To see list of spported technologies, run `kb --list`. Default: None

    # Optional input arguments:
    - multiplexed                       (bool) Indicates that the fastq files are multiplexed. Only used if sort_fastqs=True and technology is a smartseq technology. Default: None
    - parity                            (str) "single" or "paired". Only relevant if technology is bulk or a smart-seq. Default: "single"
    - quality_control_fastqs  (bool) If True, trim edges off reads and filter out low quality reads using fastp. Default: False
    - cut_front                          (bool) If True, trim bases from the front of the read. See fastp for more details. Default: False
    - cut_tail                           (bool) If True, trim bases from the tail of the read. See fastp for more details. Default: False
    - cut_window_size                    (int) Window size for sliding window trimming. See fastp for more details. Default: 4
    - cut_mean_quality                   (int) Mean quality for sliding window trimming. See fastp for more details. Default: 15
    - disable_adapter_trimming           (bool) If True, disable adapter trimming. See fastp for more details. Default: True
    - qualified_quality_phred            (int) Minimum quality for a base to be considered qualified. See fastp for more details. Default: 15
    - unqualified_percent_limit          (int) Maximum percentage of unqualified bases in a read. See fastp for more details. Default: 40
    - average_qual                       (int) Minimum average quality for a read to be considered qualified. See fastp for more details. Default: 15
    - n_base_limit                       (int) Maximum number of N bases in a read. See fastp for more details. Default: 10
    - disable_quality_filtering          (bool) If True, disable quality filtering. See fastp for more details. Default: True
    - length_required                    (int) Reads shorter than length_required will be discarded. Also used by split_reads_by_Ns_and_low_quality_bases. See fastp for more details. Default: 31
    - disable_length_filtering           (bool) If True, disable length filtering. See fastp for more details. Default: True
    - dont_eval_duplication              (bool) If True, do not evaluate duplication. See fastp for more details. Default: True
    - disable_trim_poly_g                (bool) If True, disable trimming of poly-G tails. See fastp for more details. Default: True
    - failed_out                         (bool) If True, output reads that fail filtering. See fastp for more details. Default: False
    - split_reads_by_Ns_and_low_quality_bases   (bool) If True, split reads by Ns and low quality bases (lower than min_base_quality_for_splitting) into multiple smaller reads. If min_base_quality_for_splitting > 0, then requires seqtk to be installed. If technology == "bulk", then seqtk will speed this up significantly. Default: False
    - min_base_quality_for_splitting    (int) The minimum acceptable base quality for split_reads_by_Ns_and_low_quality_bases. Bases below this quality will split. Only used if split_reads_by_Ns_and_low_quality_bases=True. Range: 0-93. Default: 13
    - concatenate_paired_fastqs         (bool) If True, concatenate paired fastq files. Default: False
    - out                               (str) Output directory. Default: "."
    - dry_run                           (bool) If True, print the commands that would be run without actually running them. Default: False
    - overwrite                         (True/False) Whether to overwrite existing output files. Will return if any output file already exists. Default: False.
    - sort_fastqs                       (bool) If True, sort fastq files by kb count. If False, then still check the order but do not change anything. Default: True
    - threads                           (int) Number of threads to use during do_sequence_trimming_or_filtering. Default: 2
    - logging_level                     (str) Logging level. Can also be set with the environment variable VARSEEK_LOGGING_LEVEL. Default: INFO.
    - save_logs                         (True/False) Whether to save logs to a file. Default: False.
    - log_out_dir                       (str) Directory to save logs. Default: `out`/logs

    # Hidden arguments (part of kwargs)
    - seqtk_path                       (str) Path to seqtk. Default: "seqtk"
    - quality_control_fastqs_out_dir       (str) Directory to save quality controlled fastq files. Default: `out`/fastqs_quality_controlled
    - replace_low_quality_bases_with_N_out_dir   (str) Directory to save fastq files with low quality bases replaced with N. Default: `out`/fastqs_replaced_low_quality_with_N
    - split_by_Ns_and_low_quality_bases_out_dir   (str) Directory to save fastq files with reads split by Ns and low quality bases. Default: `out`/fastqs_split_by_Ns_and_low_quality_bases
    - concatenate_paired_fastqs_out_dir    (str) Directory to save concatenated paired fastq files. Default: `out`/fastqs_concatenated_paired
    - delete_intermediate_files        (bool) If True, delete intermediate files. Default: True
    """

    # * 0. Informational arguments that exit early
    # Not in this function

    # * 1. logger
    if save_logs and not log_out_dir:
        log_out_dir = os.path.join(out, "logs")
    set_varseek_logging_level_and_filehandler(logging_level=logging_level, save_logs=save_logs, log_dir=log_out_dir)

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

    sig = inspect.signature(fastqpp)
    defaults = {k: v.default for k, v in sig.parameters.items()}
    for param in ["cut_front", "cut_tail", "cut_window_size", "cut_mean_quality", "disable_adapter_trimming", "qualified_quality_phred", "unqualified_percent_limit", "average_qual", "n_base_limit", "disable_quality_filtering", "length_required", "disable_length_filtering", "dont_eval_duplication", "disable_trim_poly_g"]:
        if params_dict[param] != defaults[param] and not quality_control_fastqs:
            logger.warning(f"quality_control_fastqs is False but {param} is set to {params_dict[param]}. {param} will not have any effect unless quality_control_fastqs=True.")

    # * 3. Dry-run
    if dry_run:
        print(get_varseek_dry_run(params_dict, function_name="fastqpp"))
        fastqpp_dict = {"original": fastqs, "final": fastqs}
        return fastqpp_dict

    # * 4. Save params to config file and run info file
    config_file = os.path.join(out, "config", "vk_fastqpp_config.json")
    save_params_to_config_file(params_dict, config_file)

    run_info_file = os.path.join(out, "config", "vk_fastqpp_run_info.txt")
    save_run_info(run_info_file, params_dict=params_dict, function_name="fastqpp")

    # * 5. Set up default folder/file input paths, and make sure the necessary ones exist
    # all input files for vk fastqpp are required in the varseek workflow, so this is skipped

    # * 6. Set up default folder/file output paths, and make sure they don't exist unless overwrite=True
    quality_control_fastqs_out_dir = kwargs.get("quality_control_fastqs_out_dir", os.path.join(out, "fastqs_quality_controlled"))
    replace_low_quality_bases_with_N_out_dir = kwargs.get("replace_low_quality_bases_with_N_out_dir", os.path.join(out, "fastqs_replaced_low_quality_with_N"))
    split_by_Ns_and_low_quality_bases_out_dir = kwargs.get("split_by_Ns_and_low_quality_bases_out_dir", os.path.join(out, "fastqs_split_by_Ns_and_low_quality_bases"))
    concatenate_paired_fastqs_out_dir = kwargs.get("concatenate_paired_fastqs_out_dir", os.path.join(out, "fastqs_concatenated_paired"))

    if len({quality_control_fastqs_out_dir, replace_low_quality_bases_with_N_out_dir, split_by_Ns_and_low_quality_bases_out_dir, concatenate_paired_fastqs_out_dir}) < 4:
        raise ValueError("Output directories must be unique.")
    for directory in [quality_control_fastqs_out_dir, replace_low_quality_bases_with_N_out_dir, split_by_Ns_and_low_quality_bases_out_dir, concatenate_paired_fastqs_out_dir]:
        if directory == os.path.dirname(fastqs[0]):
            raise ValueError(f"Output directory {directory} cannot be the same as the input directory {os.path.dirname(fastqs[0])}.")
        # if (os.path.exists(directory) and len(os.listdir(directory)) != 0) and not overwrite:
        #     raise ValueError(f"Output directory {directory} already exists. Use overwrite=True to overwrite existing files.")

    os.makedirs(out, exist_ok=True)

    # * 7. Define kwargs defaults
    seqtk = kwargs.get("seqtk_path", "seqtk")
    delete_intermediate_files = kwargs.get("delete_intermediate_files", True)

    # * 7.5 make sure ints are ints
    length_required, threads = int(length_required), int(threads)

    # * 8. Start the actual function
    try:
        fastqs = sort_fastq_files_for_kb_count(fastqs, technology=technology, multiplexed=multiplexed, check_only=(not sort_fastqs))
    except ValueError as e:
        if sort_fastqs:
            logger.warning(f"Automatic FASTQ argument order sorting for kb count could not recognize FASTQ file name format. Skipping argument order sorting.")

    fastq_quality_controlled_all_files = [os.path.join(quality_control_fastqs_out_dir, os.path.basename(fastq)) for fastq in fastqs]
    fastq_more_Ns_all_files = [os.path.join(replace_low_quality_bases_with_N_out_dir, os.path.basename(fastq)) for fastq in fastqs]
    split_by_Ns_and_low_quality_bases_all_files = [os.path.join(split_by_Ns_and_low_quality_bases_out_dir, os.path.basename(fastq)) for fastq in fastqs]
    # rather than determine file names of fastq_concatenated_all_files, simply use list (too complicated otherwise)

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

    if quality_control_fastqs:
        if not is_program_installed("fastp"):
            raise ValueError(f"fastp must be installed to run quality_control_fastqs. Please install and try again, or set quality_control_fastqs=False.")  # opting for an exception rather than a warning because I think the user would be more frustrated with incorrect results with a log message that could be easy to miss than having to restart their run

        # check if any file in fastq_quality_controlled_all_files does not exist
        if not all(os.path.exists(f) for f in fastq_quality_controlled_all_files) or overwrite:
            logger.info("Quality controlling fastq files (trimming adaptors, trimming low-quality read edges, filtering low quality reads)")
            os.makedirs(quality_control_fastqs_out_dir, exist_ok=True)
            if technology.upper() in {"10XV1", "INDROPSV3"}:
                for i in range(0, len(fastqs), 3):  # assumes I1, R1, R2
                    logger.info(f"Processing {fastqs[i]} {fastqs[i+1]} {fastqs[i+2]}")
                    _ = perform_fastp_trimming_and_filtering(technology=technology, r1_fastq_path=fastqs[i + 1], r2_fastq_path=fastqs[i + 2], i1_fastq_path=fastqs[i], out_dir=quality_control_fastqs_out_dir, parity=parity, cut_front=cut_front, cut_tail=cut_tail, cut_window_size=cut_window_size, cut_mean_quality=cut_mean_quality, disable_adapter_trimming=disable_adapter_trimming, qualified_quality_phred=qualified_quality_phred, unqualified_percent_limit=unqualified_percent_limit, average_qual=average_qual, n_base_limit=n_base_limit, disable_quality_filtering=disable_quality_filtering, length_required=length_required, disable_length_filtering=disable_length_filtering, dont_eval_duplication=dont_eval_duplication, disable_trim_poly_g=disable_trim_poly_g, threads=threads, failed_out=failed_out)
            elif technology.upper() == "BULK" and parity == "single":
                for i in range(len(fastqs)):  # I could just iterate through fastqs, but this parallels the other branches
                    logger.info(f"Processing {fastqs[i]}")
                    _ = perform_fastp_trimming_and_filtering(technology=technology, r1_fastq_path=fastqs[i], out_dir=quality_control_fastqs_out_dir, parity=parity, cut_front=cut_front, cut_tail=cut_tail, cut_window_size=cut_window_size, cut_mean_quality=cut_mean_quality, disable_adapter_trimming=disable_adapter_trimming, qualified_quality_phred=qualified_quality_phred, unqualified_percent_limit=unqualified_percent_limit, average_qual=average_qual, n_base_limit=n_base_limit, disable_quality_filtering=disable_quality_filtering, length_required=length_required, disable_length_filtering=disable_length_filtering, dont_eval_duplication=dont_eval_duplication, disable_trim_poly_g=disable_trim_poly_g, threads=threads, failed_out=failed_out)
            else:
                for i in range(0, len(fastqs), 2):
                    logger.info(f"Processing {fastqs[i]} {fastqs[i+1]}")
                    _ = perform_fastp_trimming_and_filtering(technology=technology, r1_fastq_path=fastqs[i], r2_fastq_path=fastqs[i + 1], out_dir=quality_control_fastqs_out_dir, parity=parity, cut_front=cut_front, cut_tail=cut_tail, cut_window_size=cut_window_size, cut_mean_quality=cut_mean_quality, disable_adapter_trimming=disable_adapter_trimming, qualified_quality_phred=qualified_quality_phred, unqualified_percent_limit=unqualified_percent_limit, average_qual=average_qual, n_base_limit=n_base_limit, disable_quality_filtering=disable_quality_filtering, length_required=length_required, disable_length_filtering=disable_length_filtering, dont_eval_duplication=dont_eval_duplication, disable_trim_poly_g=disable_trim_poly_g, threads=threads, failed_out=failed_out)
        else:
            logger.warning("Quality controlled fastq files already exist. Skipping quality control step. Use overwrite=True to overwrite existing files.")

        fastqs = fastq_quality_controlled_all_files
        fastqpp_dict["quality_controlled"] = fastqs

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
                os.makedirs(replace_low_quality_bases_with_N_out_dir, exist_ok=True)
                fastqs = replace_low_quality_bases_with_N_list(rnaseq_fastq_files=fastqs, minimum_base_quality=min_base_quality_for_splitting, seqtk=seqtk, out_dir=out)
            else:
                logger.warning("Fastq files with low quality bases replaced with N already exist. Skipping this step. Use overwrite=True to overwrite existing files.")
            if not delete_intermediate_files:
                fastqpp_dict["replaced_loq_quality_with_N"] = fastqs
        else:
            delete_intermediate_files = False  # don't delete intermediate file if I don't do the step above, as the intermediate file would be either the fastp output file or the original fastq, either of which I don't want to delete

        # check if any file in split_by_Ns_and_low_quality_bases_all_files does not exist
        if not all(os.path.exists(f) for f in split_by_Ns_and_low_quality_bases_all_files) or overwrite:
            logger.info("Splitting reads by Ns")
            os.makedirs(split_by_Ns_and_low_quality_bases_out_dir, exist_ok=True)
            fastqs = split_reads_by_N_list(fastqs, minimum_sequence_length=length_required, delete_original_files=delete_intermediate_files, out_dir=split_by_Ns_and_low_quality_bases_out_dir, seqtk=seqtk)
        else:
            logger.warning("Fastq files with reads split by N already exist. Skipping this step. Use overwrite=True to overwrite existing files.")
        if not delete_intermediate_files:
            fastqpp_dict["split_by_N_and_low_quality"] = fastqs

        delete_intermediate_files = delete_intermediate_files_original
    else:
        delete_intermediate_files = False  # if I skip this step, then I don't want to delete the intermediate files during the concatenation step

    if concatenate_paired_fastqs:
        # check if any file in fastq_concatenated_all_files does not exist
        if len(concatenate_paired_fastqs_out_dir) < (len(fastqs) // 2) or overwrite:  # if not all(os.path.exists(f) for f in fastq_concatenated_all_files) or overwrite:
            logger.info("Concatenating paired fastq files")
            os.makedirs(concatenate_paired_fastqs_out_dir, exist_ok=True)
            rnaseq_fastq_files_list_copy = []
            for i in range(0, len(fastqs), 2):
                file1 = fastqs[i]
                file2 = fastqs[i + 1]
                logger.info(f"Concatenating {file1} and {file2}")
                file_concatenated = concatenate_fastqs(file1, file2, out_dir=concatenate_paired_fastqs_out_dir, delete_original_files=delete_intermediate_files)
                rnaseq_fastq_files_list_copy.append(file_concatenated)
            fastqs = rnaseq_fastq_files_list_copy
            fastqpp_dict["concatenated"] = fastqs
        else:
            logger.warning("Concatenated fastq files already exist. Skipping this step. Use overwrite=True to overwrite existing files.")

    fastqpp_dict["final"] = fastqs

    logger.info("Returning a dictionary with keys describing the fastq files and values pointing to their file paths")

    return fastqpp_dict
