import os
import re
import time

from .utils import (
    set_up_logger,
    trim_edges_off_reads_fastq_list,
    run_fastqc_and_multiqc,
    replace_low_quality_bases_with_N_list,
    split_reads_by_N_list,
    concatenate_fastqs,
    save_params_to_config_file,
    make_function_parameter_to_value_dict,
    check_file_path_is_string_with_valid_extension,
    print_varseek_dry_run,
    report_time_elapsed,
    is_valid_int,
    save_run_info,
    sort_fastq_files_for_kb_count,
    rnaseq_fastq_filename_pattern
)

from .constants import fastq_extensions, technology_valid_values

logger = set_up_logger()


def load_in_fastqs(fastqs):
    if len(fastqs) != 1:
        return fastqs
    fastqs = fastqs[0]
    if not os.path.exists(fastqs):
        raise ValueError(f"File/folder {fastqs} does not exist")
    if os.path.isdir(fastqs):
        files = []
        for file in os.listdir(fastqs):  # make fastqs list from fastq files in immediate child directory
            if (os.path.isfile(os.path.join(fastqs, file))) and (any(file.lower().endswith((ext, f"{ext}.zip", f"{ext}.gz")) for ext in fastq_extensions)):
                files.append(file)
        if len(files) == 0:
            raise ValueError(f"No fastq files found in {fastqs}")  # redundant with type-checking below, but prints a different error message (informs that the directory has no fastqs, rather than simply telling the user that no fastqs were provided)
    elif os.path.isfile(fastqs):
        if file.lower().endswith("txt"):  # make fastqs list from items in txt file
            with open(fastqs, "r") as f:
                files = [line.strip() for line in f.readlines()]
            if len(files) == 0:
                raise ValueError(f"No fastq files found in {fastqs}")  # redundant with type-checking below, but prints a different error message (informs that the text file has no fastqs, rather than simply telling the user that no fastqs were provided)
        elif any(fastqs.lower().endswith((ext, f"{ext}.zip", f"{ext}.gz")) for ext in fastq_extensions):
            files = [fastqs]
        else:
            raise ValueError(f"File {fastqs} is not a fastq file, text file, or directory")
    return files
    

def validate_input_fastqpp(params_dict):
    fastqs = params_dict["fastqs"]  # tuple
    parity = params_dict["parity"]  # str

    # fastqs
    if len(fastqs) == 0:
        raise ValueError("No fastq files provided")
    
    #$ type checking of the directory and text file performed earlier by load_in_fastqs 

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
    if not isinstance(params_dict.get("out", None), str):
        raise ValueError(f"Invalid value for out: {params_dict.get('out', None)}")
    
    # optional str
    for file_name_suffix in ["quality_control_fastqs_out_suffix", "replace_low_quality_bases_with_N_out_suffix", "split_by_N_out_suffix", "concatenate_paired_fastqs_out_suffix"]:
        if params_dict.get(file_name_suffix) is not None and not isinstance(params_dict.get(file_name_suffix), str):
            raise ValueError(f"Invalid suffix: {params_dict.get(file_name_suffix)}")
    
    # integers - optional just means that it's in kwargs
    for param_name, min_value, max_value, optional_value in [
        ("cut_mean_quality", 1, 36, False),
        ("cut_window_size", 1, 1000, False),
        ("qualified_quality_phred", 0, 93, False),
        ("unqualified_percent_limit", 1, 100, False),
        ("max_ambiguous", 1, 50, False),
        ("min_base_quality", 0, 93, False),
    ]:
        param_value = params_dict.get(param_name)
        if not is_valid_int(param_value, "between", min_value_inclusive=min_value, max_value_inclusive=max_value, optional=optional_value):
            raise ValueError(f"{param_name} must be an integer between {min_value} and {max_value}. Got {params_dict.get(param_name)}.")
    
    if not is_valid_int(params_dict['threads'], ">=", 1, optional=False):
        raise ValueError(f"threads must be an integer >= 1. Got {params_dict.get('threads')}.")
    
    if not is_valid_int(params_dict['min_read_len'], ">=", 1, optional=False) and params_dict['min_read_len'] is not None:
        raise ValueError(f"min_read_len must be an integer >= 1 or None. Got {params_dict.get('threads')}.")
    
    # boolean
    for param_name in ["quality_control_fastqs", "fastqc_and_multiqc", "replace_low_quality_bases_with_N", "split_reads_by_Ns", "concatenate_paired_fastqs", "delete_intermediate_files", "dry_run", "overwrite", "sort_fastqs"]:
        if not isinstance(params_dict.get(param_name), bool):
            raise ValueError(f"{param_name} must be a boolean. Got {param_name} of type {type(params_dict.get(param_name))}.")
        
    if parity == "paired" and params_dict['split_reads_by_Ns'] and not params_dict['concatenate_paired_fastqs']:
        raise ValueError("When parity==paired, if split_reads_by_Ns==True, then concatenate_paired_fastqs must also be True (split_reads_by_Ns messes up the paired nature of the fastqs).")
        
    if not isinstance(params_dict.get("multiplexed"), bool) and params_dict.get("multiplexed") is not None:
        raise ValueError(f"multiplexed must be a boolean or None. Got {params_dict.get('multiplexed')} of type {type(params_dict.get('multiplexed'))}.")

def fastqpp(
    *fastqs,
    technology=None,
    multiplexed=None,
    parity="single",
    quality_control_fastqs=False,
    cut_mean_quality=13,
    cut_window_size=4,  # new
    qualified_quality_phred=0,
    unqualified_percent_limit=100,
    max_ambiguous=50,
    min_read_len=None,
    fastqc_and_multiqc=False,
    replace_low_quality_bases_with_N=False,
    min_base_quality=13,
    split_reads_by_Ns=False,
    concatenate_paired_fastqs=False,
    out=".",
    delete_intermediate_files=False,
    dry_run=False,
    overwrite=False,
    sort_fastqs=True,
    threads=2,
    verbose=True,
    **kwargs
):
    """
    Apply quality control to fastq files. This includes trimming edges off reads, running FastQC and MultiQC, replacing low quality bases with N, splitting reads by Ns, and concatenating paired fastq files.

    # Required input arguments:
    - fastqs                            (str or list[str]) List of fastq files to be processed. If paired end, the list should contains paths such as [file1_R1, file1_R2, file2_R1, file2_R2, ...]

    # Optional input arguments:
    - technology                        (str) Technology used to generate the data. Only used if sort_fastqs=True. To see list of spported technologies, run `kb --list`. Default: None
    - multiplexed                       (bool) Indicates that the fastq files are multiplexed. Only used if sort_fastqs=True and technology is a smartseq technology. Default: None
    - parity                            (str) "single" or "paired". Default: "single"
    - quality_control_fastqs            (bool) If True, run fastp to trim and filter reads. Default: False
    - cut_mean_quality                  (int) The mean quality requirement option in cut_window_size when trimming edges. Only used if quality_control_fastqs=True. if See details with `fastp --help`. Range: 1-36. Default: 20
    - cut_window_size                   (int) The window size with which to calculate cut_mean_quality when trimming edges. Only used if quality_control_fastqs=True. See details with `fastp --help`. Range: 1-1000. Default: 4
    - qualified_quality_phred           (int) The phred quality score for a base to be considered qualified. Only used if quality_control_fastqs=True. See details with `fastp --help`. Range: 0-93. Default: 0 (no average quality filtering)
    - unqualified_percent_limit         (int) The percent of unqualified bases allowed in a read. Only used if quality_control_fastqs=True. See details with `fastp --help`. Range: 1-100. Default: 100 (no average quality filtering)
    - max_ambiguous                     (int) The maximum number of ambiguous bases allowed in a read. Only used if quality_control_fastqs=True. See details with `fastp --help`. Range: 1-50. Default: 50
    - min_read_len                      (int) The minimum length of a read. Only used if quality_control_fastqs=True or replace_low_quality_bases_with_N=True. Default: None (no minimum length)
    - fastqc_and_multiqc            (bool) If True, run FastQC and MultiQC. Default: False
    - replace_low_quality_bases_with_N  (bool) If True, replace low quality bases with N. Default: False
    - min_base_quality                  (int) The minimum acceptable base quality. Bases below this quality will be masked with 'N'. Only used if replace_low_quality_bases_with_N=True. Range: 0-93. Default: 13
    - split_reads_by_Ns                 (bool) If True, split reads by Ns into multiple smaller reads. Default: False
    - concatenate_paired_fastqs         (bool) If True, concatenate paired fastq files. Only used when parity=paired. Default: False
    - out                               (str) Output directory. Default: "."
    - delete_intermediate_files         (bool) If True, delete intermediate files. Default: False
    - dry_run                           (bool) If True, print the commands that would be run without actually running them. Default: False
    - overwrite                          (True/False) Whether to overwrite existing output files. Will return if any output file already exists. Default: False.
    - sort_fastqs                       (bool) If True, sort fastq files by kb count. If False, then still check the order but do not change anything. Default: True
    - threads                           (int) Number of threads to use. Default: 2
    - verbose                           (bool) If True, print progress messages. Default: True

    # Hidden arguments (part of kwargs)
    - fastp_path                       (str) Path to fastp. Default: "fastp"
    - seqtk_path                       (str) Path to seqtk. Default: "seqtk"
    - quality_control_fastqs_out_suffix (str) Suffix to add to fastq files after quality control (preceded by underscore). Default: "qc"
    - replace_low_quality_bases_with_N_out_suffix (str) Suffix to add to fastq files after replacing low quality bases with N (preceded by underscore). Default: "addedNs"
    - split_by_N_out_suffix            (str) Suffix to add to fastq files after splitting by Ns (preceded by underscore). Default: "splitNs"
    - concatenate_paired_fastqs_out_suffix (str) Suffix to add to fastq files after concatenating paired fastq files (preceded by underscore). Default: "concatenated"
    """

    #* 0. Informational arguments that exit early
    # Not in this function

    #* 1. Start timer
    start_time = time.perf_counter()

    #* 1.5 load in fastqs
    fastqs = load_in_fastqs(fastqs)

    #* 2. Type-checking
    params_dict = make_function_parameter_to_value_dict(1)
    fastqs_original = params_dict["fastqs"].copy()
    params_dict["fastqs"] = fastqs
    validate_input_fastqpp(params_dict)
    params_dict["fastqs"] = fastqs_original  #* 3. Type-checking

    #* 3. Dry-run
    if dry_run:
        print_varseek_dry_run(params_dict, function_name="fastqpp")
        return None
    
    #* 4. Save params to config file and run info file
    config_file = os.path.join(out, "config", "vk_fastqpp_config.json")
    save_params_to_config_file(params_dict, config_file)

    run_info_file = os.path.join(out, "config", "vk_fastqpp_run_info.txt")
    save_run_info(run_info_file)

    #* 5. Set up default folder/file input paths, and make sure the necessary ones exist
    # all input files for vk fastqpp are required in the varseek workflow, so this is skipped

    #* 6. Set up default folder/file output paths, and make sure they don't exist unless overwrite=True
    quality_control_fastqs_out_suffix = kwargs.get("quality_control_fastqs_out_suffix", "qc")
    replace_low_quality_bases_with_N_out_suffix = kwargs.get("replace_low_quality_bases_with_N_out_suffix", "addedNs")
    split_by_N_out_suffix = kwargs.get("split_by_N_out_suffix", "splitNs")
    concatenate_paired_fastqs_out_suffix = kwargs.get("concatenate_paired_fastqs_out_suffix", "concatenatedPairs")

    os.makedirs(out, exist_ok=True)

    if not overwrite:
        for fastq in fastqs:
            parts_filename = fastq.split(".", 1)
            if quality_control_fastqs:
                fastq_quality_controlled = os.path.join(out, f"{parts_filename[0]}_{quality_control_fastqs_out_suffix}.{parts_filename[1]}")
                if os.path.exists(fastq_quality_controlled):
                    raise ValueError(f"Output file {fastq_quality_controlled} already exists. Use overwrite=True to overwrite existing files, or set quality_control_fastqs=False to skip this step.")
            if fastqc_and_multiqc:
                fastq_fastqc_html = os.path.join(out, f"{parts_filename[0]}_fastqc.html")
                fastq_fastqc_zip = os.path.join(out, f"{parts_filename[0]}_fastqc.zip")
                if (os.path.exists(fastq_fastqc_html) or os.path.exists(fastq_fastqc_zip)):
                    raise ValueError(f"Output file {fastq_fastqc_html} or {fastq_fastqc_zip} already exists. Use overwrite=True to overwrite existing files, or set fastqc_and_multiqc=False to skip this step.")
            if replace_low_quality_bases_with_N:
                fastq_more_Ns = os.path.join(out, f"{parts_filename[0]}_{replace_low_quality_bases_with_N_out_suffix}.{parts_filename[1]}")
                if os.path.exists(fastq_more_Ns):
                    raise ValueError(f"Output file {fastq_more_Ns} already exists. Use overwrite=True to overwrite existing files, or set replace_low_quality_bases_with_N=False to skip this step.")
            if split_reads_by_Ns:
                fastq_split_by_N = os.path.join(out, f"{parts_filename[0]}_{split_by_N_out_suffix}.{parts_filename[1]}")
                if os.path.exists(fastq_split_by_N):
                    raise ValueError(f"Output file {fastq_split_by_N} already exists. Use overwrite=True to overwrite existing files, or set split_reads_by_Ns=False to skip this step.")
            if concatenate_paired_fastqs and parity == "paired":
                fastq_concatenated = os.path.join(out, f"{parts_filename[0]}_{concatenate_paired_fastqs_out_suffix}.{parts_filename[1]}")
                if os.path.exists(fastq_concatenated):
                    raise ValueError(f"Output file {fastq_concatenated} already exists. Use overwrite=True to overwrite existing files, or set concatenate_paired_fastqs=False to skip this step.")

        multiqc_html = os.path.join(out, "multiqc_report.html")
        multiqc_dir = os.path.join(out, "multiqc_data")
        if (os.path.exists(multiqc_html) or os.path.exists(multiqc_dir)) and fastqc_and_multiqc:
            raise ValueError(f"Output file {multiqc_html} or {multiqc_dir} already exists. Use overwrite=True to overwrite existing files, or set fastqc_and_multiqc=False to skip this step.")

    #* 7. Define kwargs defaults
    fastp = kwargs.get("fastp_path", "fastp")
    seqtk = kwargs.get("seqtk_path", "seqtk")
    
    #* 8. Start the actual function
    fastqs = sort_fastq_files_for_kb_count(fastqs, technology=technology, multiplexed=multiplexed, logger=logger, check_only=(not sort_fastqs), verbose=verbose)

    rnaseq_fastq_files_list_dict = {}
    rnaseq_fastq_files_list_dict["original"] = fastqs

    if quality_control_fastqs:
        logger.info("Quality controlling fastq files (trimming adaptors, trimming low-quality read edges, filtering low quality reads)")
        fastqs = trim_edges_off_reads_fastq_list(
            rnaseq_fastq_files=fastqs,
            parity=parity,
            minimum_base_quality_trim_reads=cut_mean_quality,
            cut_window_size=cut_window_size,
            qualified_quality_phred=qualified_quality_phred,
            unqualified_percent_limit=unqualified_percent_limit,
            n_base_limit=max_ambiguous,
            length_required=min_read_len,
            fastp=fastp,
            out_dir=out,
            threads=threads,
            logger=logger,
            verbose=verbose,
            suffix=quality_control_fastqs_out_suffix
        )
        rnaseq_fastq_files_list_dict["quality_controlled"] = fastqs

    if fastqc_and_multiqc:
        run_fastqc_and_multiqc(fastqs, out)

    if replace_low_quality_bases_with_N:
        logger.info("Replacing low quality bases with N")
        fastqs = replace_low_quality_bases_with_N_list(
            rnaseq_fastq_files_quality_controlled=fastqs,
            minimum_base_quality_replace_with_N=min_base_quality,
            seqtk=seqtk,
            out_dir=out,
            logger=logger,
            verbose=verbose,
            suffix=replace_low_quality_bases_with_N_out_suffix
        )
        rnaseq_fastq_files_list_dict["replaced_with_N"] = fastqs

    if split_reads_by_Ns:
        logger.info("Splitting reads by Ns")
        fastqs = split_reads_by_N_list(
            fastqs,
            minimum_sequence_length=min_read_len,
            delete_original_files=delete_intermediate_files,
            out_dir=out,
            logger=logger,
            verbose=verbose,
            suffix=split_by_N_out_suffix
        )
        rnaseq_fastq_files_list_dict["split_by_N"] = fastqs

    if concatenate_paired_fastqs and parity == "paired":
        logger.info("Concatenating paired fastq files")
        rnaseq_fastq_files_list_copy = []
        for i in range(0, len(fastqs), 2):
            file1 = fastqs[i]
            file2 = fastqs[i + 1]
            logger.info(f"Concatenating {file1} and {file2}")
            file_concatenated = concatenate_fastqs(file1, file2, out_dir=out, delete_original_files=delete_intermediate_files, suffix=concatenate_paired_fastqs_out_suffix)
            rnaseq_fastq_files_list_copy.append(file_concatenated)
        fastqs = rnaseq_fastq_files_list_copy
        rnaseq_fastq_files_list_dict["concatenated"] = fastqs

    rnaseq_fastq_files_list_dict["final"] = fastqs

    report_time_elapsed(start_time, logger=logger, verbose=verbose)

    return fastqs  # rnaseq_fastq_files_list_dict


