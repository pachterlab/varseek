"""varseek count and specific helper functions."""

import inspect
from pathlib import Path
import os
import subprocess
import time

import varseek as vk
from varseek.utils import (
    check_file_path_is_string_with_valid_extension,
    check_that_two_directories_in_params_dict_are_the_same_if_both_provided_otherwise_set_them_equal,
    is_valid_int,
    load_in_fastqs,
    make_function_parameter_to_value_dict,
    report_time_elapsed,
    save_params_to_config_file,
    save_run_info,
    set_up_logger,
    sort_fastq_files_for_kb_count,
)
from varseek.varseek_clean import needs_for_normal_genome_matrix

from .constants import (
    non_single_cell_technologies,
    supported_downloadable_normal_reference_genomes_with_kb_ref,
)

logger = set_up_logger()

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


varseek_count_only_allowable_kb_count_arguments = {
    "zero_arguments": {"--keep-tmp", "--verbose", "--tcc", "--cellranger", "--gene-names", "--report", "--long", "--opt-off", "--matrix-to-files", "--matrix-to-directories"},
    "one_argument": {"--tmp", "--kallisto", "--bustools", "-w", "-r", "-m", "--inleaved", "--filter", "filter-threshold", "-N", "--threshold", "--platform"},
    "multiple_arguments": set(),
}  # don't include d-list, t, i, k, workflow here because I do it myself later


def validate_input_count(params_dict):
    # $ fastqs, technology will get checked through fastqpp

    # other required files
    for param_name, file_type in {
        "index": "index",
        "t2g": "t2g",
    }.items():
        check_file_path_is_string_with_valid_extension(params_dict[param_name], param_name, file_type=file_type, required=True)
        if not os.path.isfile(params_dict[param_name]):  # ensure that all fastq files exist
            raise ValueError(f"File {params_dict[param_name]} does not exist")

    # file paths
    check_file_path_is_string_with_valid_extension(params_dict.get("config", None), "config", ["json", "yaml"])
    check_file_path_is_string_with_valid_extension(params_dict.get("reference_genome_index", None), "reference_genome_index", "index")
    check_file_path_is_string_with_valid_extension(params_dict.get("reference_genome_t2g", None), "reference_genome_t2g", "t2g")
    check_file_path_is_string_with_valid_extension(params_dict.get("adata_reference_genome", None), "adata_reference_genome", "adata")

    if not is_valid_int(params_dict.get("threads", None), threshold_type=">=", threshold_value=1):
        raise ValueError(f"Threads must be a positive integer, got {params_dict.get('threads')}")

    # out dirs
    for param_name in ["out", "kb_count_vcrs_out_dir", "kb_count_reference_genome_out_dir", "vk_summarize_out_dir"]:
        if not isinstance(params_dict.get(param_name, None), (str, Path)):
            raise ValueError(f"Invalid value for {param_name}: {params_dict.get(param_name, None)}")

    # booleans
    for param_name in ["dry_run", "overwrite", "sort_fastqs", "verbose"]:
        if not isinstance(params_dict.get(param_name), bool):
            raise ValueError(f"{param_name} must be a boolean. Got {param_name} of type {type(params_dict.get(param_name))}.")

    # strings
    parity_valid_values = {"single", "paired"}
    if params_dict["parity"] not in parity_valid_values:
        raise ValueError(f"Parity must be one of {parity_valid_values}")

    strand_valid_values = {"unstranded", "forward", "reverse"}
    if params_dict["strand"] not in strand_valid_values:
        raise ValueError(f"Strand must be one of {strand_valid_values}")

    out = params_dict.get("out", ".")
    kb_count_reference_genome_out_dir = params_dict.get("kb_count_reference_genome_out_dir", f"{out}/kb_count_out_reference_genome")
    if params_dict.get("qc_against_gene_matrix") and not os.path.exists(kb_count_reference_genome_out_dir):  # align to this genome if (1) adata doesn't exist and (2) qc_against_gene_matrix=True (because I need the BUS file for this)  # purposely omitted overwrite because it is reasonable to expect that someone has pre-computed this matrix and doesn't want it recomputed under any circumstances (and if they did, then simply point to a different directory)
        species = params_dict.get("species", None)
        if not isinstance(species, str) and species not in supported_downloadable_normal_reference_genomes_with_kb_ref:
            raise ValueError(f"Species {species} is not supported. Supported values are {supported_downloadable_normal_reference_genomes_with_kb_ref}. See more details at https://github.com/pachterlab/kallisto-transcriptome-indices/")

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


# don't worry if it says an argument is unused - they will all get put in params_dict
def count(
    *fastqs,
    index,
    t2g,
    technology,  # params
    k=59,
    min_read_len=None,
    strand="unstranded",
    mm=False,
    union=False,
    parity="single",
    species=None,
    config=None,  # optional inputs
    reference_genome_index=None,
    reference_genome_t2g=None,
    adata_reference_genome=None,
    out=".",  # optional outputs
    kb_count_vcrs_out_dir=None,
    kb_count_reference_genome_out_dir=None,
    vk_summarize_out_dir=None,
    disable_fastqpp=False,  # general
    disable_clean=False,
    disable_summarize=False,
    dry_run=False,
    overwrite=False,
    sort_fastqs=True,
    threads=2,
    verbose=True,
    **kwargs,  # * including all arguments for vk build, info, and filter
):
    """
    Perform variant screening on sequencing data. Wraps around varseek fastqpp, kb count, varseek clean, and kb summarize.

    # Required input arguments:
    - fastqs                                (str or list[str]) List of fastq files to be processed. If paired end, the list should contains paths such as [file1_R1, file1_R2, file2_R1, file2_R2, ...]
    - index                                 (str)  Path to variant index generated by varseek ref
    - t2g                                   (str)  Path to t2g file generated by varseek ref
    - technology                            (str)  Technology used to generate the data. To see list of spported technologies, run `kb --list`.

    # Additional parameters
    - k                                     (int) The length of each k-mer in the kallisto reference index construction. Corresponds to `k` used in the earlier varseek commands (i.e., varseek ref). If using a downloaded index from varseek ref -d, then check the description for the k value used to construct this index with varseek ref --list_downloadable_references. Default: 59.
    - strand                                (str)  The strandedness of the data. Either "unstranded", "forward", or "reverse". Default: "unstranded".
    - mm                                    (bool)  If True, use the multi-mapping reads. Default: False.
    - union                                 (bool)  If True, use the union of the read mappings. Default: False.
    - parity                                (str)  The parity of the reads. Either "single" or "paired". Only used if technology is bulk or a smart-seq. Default: "single".
    - species                               (str)  The species of the reference genome. Only used if qc_against_gene_matrix=True (see vk clean --help). Default: None.

    # Optional input arguments
    - config                                (str) Path to config file. Default: None.
    - reference_genome_index                (str) Path to index file for the "normal" reference genome. Created if not provided. Only used if qc_against_gene_matrix=True (see vk clean --help). Default: None.
    - reference_genome_t2g                  (str) Path to t2g file for the "normal" reference genome. Created if not provided. Only used if qc_against_gene_matrix=True (see vk clean --help). Default: None.
    - adata_reference_genome                (str) Path to adata file for the "normal" reference genome. Created if not provided. Only used if qc_against_gene_matrix=True or performing gene-level normalization on the VCRS count matrix (see vk clean --help). Default: None.

    # Optional output file paths: (only needed if changing/customizing file names or locations):
    - out                                   (str) Output directory. Default: ".".
    - kb_count_vcrs_out_dir                 (str) Output directory for kb count. Default: `out`/kb_count_out_vcrs
    - kb_count_reference_genome_out_dir     (str) Output directory for kb count on "normal" reference genome. Default: `out`/kb_count_out_reference_genome
    - vk_summarize_out_dir                  (str) Output directory for vk summarize. Default: `out`/vk_summarize

    # General arguments:
    - disable_fastqpp                       (bool) If True, skip fastqpp step. Default: False.
    - disable_clean                         (bool) If True, skip clean step. Default: False.
    - disable_summarize                     (bool) If True, skip summarize step. Default: False.
    - dry_run                               (bool) If True, print the commands that would be run without actually running them. Default: False.
    - overwrite                             (bool) If True, overwrite existing files. Default: False.
    - sort_fastqs                           (bool) If True, sort fastq files by kb count. If False, then still check the order but do not change anything. Default: True
    - threads                               (int) Number of threads to use. Default: 2.
    - verbose                               (bool) If True, print progress messages. Default: True.

    # Hidden arguments (part of kwargs):
    - use_num                              (bool) If True, use the --num argument in kb count. Default: False.

    For a complete list of supported parameters, see the documentation for varseek fastqpp, kb count, varseek clean, and varseek summarize. Note that any shared parameter names between functions are meant to have identical purposes.
    """

    # * 0. Informational arguments that exit early
    # Nothing here

    # * 1. Start timer
    start_time = time.perf_counter()

    # * 1.5. Load in parameters from a config file if provided
    if isinstance(config, str) and os.path.isfile(config):
        vk_count_config_file_input = vk.utils.load_params(config)

        # overwrite any parameters passed in with those from the config file
        count_signature = inspect.signature(count)
        for key, value in vk_count_config_file_input.items():
            if key in count_signature.parameters.keys():
                exec("%s = %s" % (key, value))  # assign the value to the variable name
            else:
                kwargs[key] = value  # if the variable is not in the function signature, then add it to kwargs

    # * 1.75 load in fastqs
    fastqs_original = fastqs
    fastqs = load_in_fastqs(fastqs)

    # * 2. Type-checking
    params_dict = make_function_parameter_to_value_dict(1)
    vk.varseek_fastqpp.validate_input_fastqpp(params_dict)  # this passes all vk ref parameters to the function - I could only pass in the vk build parameters here if desired (and likewise below), but there should be no naming conflicts anyways
    vk.varseek_clean.validate_input_clean(params_dict)
    vk.varseek_summarize.validate_input_summarize(params_dict)
    validate_input_count(params_dict)
    params_dict["fastqs"] = fastqs_original  # for config file - reversed later

    # * 3. Dry-run
    # handled within child functions

    # * 3.75 Pop out any unallowable arguments
    for key, unallowable_set in varseek_count_unallowable_arguments.items():
        for unallowable_key in unallowable_set:
            params_dict.pop(unallowable_key, None)

    # * 3.8 Set params_dict to default values of children functions (not strictly necessary, as if these arguments are not in params_dict then it will use the default values anyways, but important if I need to rely on these default values within vk count)
    # count_signature = inspect.signature(count)
    # for function in (vk.varseek_fastqpp.fastqpp, vk.varseek_clean.clean, vk.varseek_summarize.summarize):
    #     signature = inspect.signature(function)
    #     for key in signature.parameters.keys():
    #         if key not in params_dict and key not in count_signature.parameters.keys():
    #             params_dict[key] = signature.parameters[key].default

    # * 4. Save params to config file and run info file
    # Save parameters to config file
    config_file = os.path.join(out, "config", "vk_count_config.json")
    save_params_to_config_file(params_dict, config_file)
    params_dict["fastqs"] = fastqs

    run_info_file = os.path.join(out, "config", "vk_count_run_info.txt")
    save_run_info(run_info_file)

    # * 5. Set up default folder/file input paths, and make sure the necessary ones exist
    # all input files for vk count are required in the varseek workflow, so this is skipped

    # * 5.5 Setting up modes
    # if mode:  #* uncomment once I have modes
    #     for key in mode_parameters[mode]:
    #         params_dict[key] = mode_parameters[mode][key]

    # * 6. Set up default folder/file output paths, and make sure they don't exist unless overwrite=True
    if not kb_count_vcrs_out_dir:
        kb_count_vcrs_out_dir = f"{out}/kb_count_out_vcrs"
    if not kb_count_reference_genome_out_dir:
        kb_count_reference_genome_out_dir = f"{out}/kb_count_out_reference_genome"
    if not vk_summarize_out_dir:
        vk_summarize_out_dir = f"{out}/vk_summarize"

    os.makedirs(out, exist_ok=True)
    os.makedirs(kb_count_vcrs_out_dir, exist_ok=True)
    os.makedirs(kb_count_reference_genome_out_dir, exist_ok=True)
    os.makedirs(vk_summarize_out_dir, exist_ok=True)

    # for vk clean arguments - generalizes the params_dict["kb_count_vcrs_dir"] = kb_count_vcrs_out_dir and params_dict["kb_count_reference_genome_dir"] = kb_count_reference_genome_out_dir calls
    params_dict = check_that_two_directories_in_params_dict_are_the_same_if_both_provided_otherwise_set_them_equal(params_dict, "kb_count_vcrs_dir", "kb_count_vcrs_out_dir")  # check that, if kb_count_vcrs_dir and kb_count_vcrs_out_dir are both provided, they are the same directory; otherwise, if only one is provided, then make them equal to each other
    params_dict = check_that_two_directories_in_params_dict_are_the_same_if_both_provided_otherwise_set_them_equal(params_dict, "kb_count_reference_genome_dir", "kb_count_reference_genome_out_dir")  # same story as above but for kb_count_reference_genome and kb_count_reference_genome_out_dir

    adata_vcrs = f"{kb_count_vcrs_out_dir}/counts_unfiltered/adata.h5ad"
    adata_reference_genome = f"{kb_count_reference_genome_out_dir}/counts_unfiltered/adata.h5ad" if not params_dict.get("adata_reference_genome") else params_dict.get("adata_reference_genome")
    adata_vcrs_clean_out = f"{out}/adata_cleaned.h5ad" if not params_dict.get("adata_vcrs_clean_out") else params_dict.get("adata_vcrs_clean_out")  # from vk clean
    adata_reference_genome_clean_out = f"{out}/adata_cleaned.h5ad" if not params_dict.get("adata_reference_genome_clean_out") else params_dict.get("adata_reference_genome_clean_out")  # from vk clean
    vcf_out = os.path.join(out, "vcf") if not params_dict.get("vcf_out") else params_dict["vcf_out"]
    stats_file = os.path.join(vk_summarize_out_dir, "varseek_summarize_stats.txt") if not params_dict.get("stats_file") else params_dict["stats_file"]  # from vk summarize

    for file in [stats_file]:  # purposely excluded adata_reference_genome because it is fine if someone provides this as input even if overwrite=False; and purposely excluded adata_vcrs, adata_vcrs_clean_out, adata_reference_genome_clean_out, kb_count_vcrs_out_dir, kb_count_reference_genome_out_dir for the reasons provided in vk ref
        if os.path.isfile(file) and not overwrite:
            raise FileExistsError(f"Output file {file} already exists. Please delete it or specify a different output directory or set overwrite=True.")

    # no need for file_signifying_successful_vk_fastqpp_completion because overwrite=False just gives warning rather than error in fastqpp
    file_signifying_successful_kb_count_vcrs_completion = adata_vcrs
    file_signifying_successful_kb_count_reference_genome_completion = adata_reference_genome
    file_signifying_successful_vk_clean_completion = adata_vcrs_clean_out
    file_signifying_successful_vk_summarize_completion = stats_file

    overwrite_original = params_dict.get("overwrite_original", False)

    # * 6.5 Just to make the unused parameter coloration go away in VSCode
    min_read_len = min_read_len
    mm = mm
    union = union

    # * 7. Define kwargs defaults
    use_num = params_dict.get("use_num", False)

    # * 8. Start the actual function
    fastqs_unsorted = fastqs.copy()
    fastqs = sort_fastq_files_for_kb_count(fastqs, technology=technology, multiplexed=params_dict.get("multiplexed"), logger=logger, check_only=(not sort_fastqs), verbose=verbose)
    # params_dict["fastqs"] = fastqs  # no need because I do this later

    parity = params_dict.get("parity", "single")
    if technology.lower() != "bulk" and "smartseq" not in technology.lower():
        parity = "single"
        params_dict["parity"] = parity

    # so parity_vcrs is set correctly - copied from fastqpp
    concatenate_paired_fastqs = params_dict.get("concatenate_paired_fastqs", False)
    split_reads_by_Ns = params_dict.get("split_reads_by_Ns", False)
    if (concatenate_paired_fastqs or split_reads_by_Ns) and parity == "paired":
        if not concatenate_paired_fastqs:
            logger.info("Setting concatenate_paired_fastqs=True")
        concatenate_paired_fastqs = True
    else:
        if concatenate_paired_fastqs:
            logger.info("Setting concatenate_paired_fastqs=False")
        concatenate_paired_fastqs = False
    params_dict["concatenate_paired_fastqs"] = concatenate_paired_fastqs

    if not params_dict.get("min_read_len"):
        min_read_len = k
        params_dict["min_read_len"] = min_read_len

    # define the vk fastqpp, clean, and summarize arguments (explicit arguments and allowable kwargs)
    explicit_parameters_vk_fastqpp = vk.utils.get_set_of_parameters_from_function_signature(vk.varseek_fastqpp.fastqpp)
    allowable_kwargs_vk_fastqpp = vk.utils.get_set_of_allowable_kwargs(vk.varseek_fastqpp.fastqpp)

    explicit_parameters_vk_clean = vk.utils.get_set_of_parameters_from_function_signature(vk.varseek_clean.clean)
    allowable_kwargs_vk_clean = vk.utils.get_set_of_allowable_kwargs(vk.varseek_clean.clean)

    explicit_parameters_vk_summarize = vk.utils.get_set_of_parameters_from_function_signature(vk.varseek_summarize.summarize)
    allowable_kwargs_vk_summarize = vk.utils.get_set_of_allowable_kwargs(vk.varseek_summarize.summarize)

    all_parameter_names_set_vk_fastqpp = explicit_parameters_vk_fastqpp | allowable_kwargs_vk_fastqpp
    all_parameter_names_set_vk_clean = explicit_parameters_vk_clean | allowable_kwargs_vk_clean
    all_parameter_names_set_vk_summarize = explicit_parameters_vk_summarize | allowable_kwargs_vk_summarize

    params_dict_vk_fastqpp = {key: value for key, value in params_dict.items() if key in all_parameter_names_set_vk_fastqpp}
    params_dict_vk_clean = {key: value for key, value in params_dict.items() if key in all_parameter_names_set_vk_clean}
    params_dict_vk_summarize = {key: value for key, value in params_dict.items() if key in all_parameter_names_set_vk_summarize}

    # vk fastqpp
    if not disable_fastqpp:
        logger.info("Running vk fastqpp")
        fastqpp_dict = vk.fastqpp(**params_dict) if not params_dict.get("dry_run", False) else vk.fastqpp(**params_dict_vk_fastqpp)  # best of both worlds - will only pass in defined arguments if dry run is True (which is good so that I don't show each function with a bunch of args it never uses), but will pass in all arguments if dry run is False (which is good if I run vk ref with a new parameter that I have not included in docstrings yet, as I only get usable kwargs list from docstrings)
        fastqs_vcrs = fastqpp_dict["final"]
        fastqs_reference_genome = fastqpp_dict["quality_controlled"] if "quality_controlled" in fastqpp_dict else fastqs
    else:
        logger.warning("Skipping vk fastqpp because disable_fastqpp=True")
        fastqs_vcrs = fastqs
        fastqs_reference_genome = fastqs
    params_dict["fastqs"] = fastqs_vcrs  # so that the correct fastqs get passed into vk clean

    # # kb count, VCRS
    if not os.path.exists(file_signifying_successful_kb_count_vcrs_completion) or overwrite:
        if params_dict.get("concatenate_paired_fastqs"):
            parity_vcrs = "single"
        else:
            parity_vcrs = parity

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
            "--parity",
            parity_vcrs,
            "--strand",
            strand,
            "-o",
            kb_count_vcrs_out_dir,
            "--overwrite",
            True,  # set to True here regardless of the overwrite argument because I would only even enter this block if kb count was only partially run (as seen by the lack of existing of file_signifying_successful_kb_count_vcrs_completion), in which case I should overwrite anyways
        ]

        if params_dict.get("qc_against_gene_matrix"):
            kb_count_command.extend(["--union", "--mm"])
        if params_dict.get("qc_against_gene_matrix") or params_dict.get("apply_split_reads_by_Ns_correction") or params_dict.get("apply_dlist_correction") or use_num:
            kb_count_command.extend(["--num"])

        # assumes any argument in varseek count matches kb count identically, except dashes replaced with underscores
        for dict_key, arguments in varseek_count_only_allowable_kb_count_arguments.items():
            for argument in list(arguments):
                dash_count = len(argument) - len(argument.lstrip("-"))
                leading_dashes = "-" * dash_count
                argument = argument.lstrip("-").replace("-", "_")
                if argument in params_dict:
                    value = params_dict[argument]
                    if dict_key == "zero_arguments":
                        if value:  # only add if value is True
                            kb_count_command.append(f"{leading_dashes}{argument}")
                    elif dict_key == "one_argument":
                        kb_count_command.extend([f"{leading_dashes}{argument}", value])
                    else:  # multiple_arguments or something else
                        pass

        kb_count_command += fastqs_vcrs

        if dry_run:
            print(kb_count_command)
        else:
            logger.info(f"Running kb count with command: {' '.join(kb_count_command)}")
            subprocess.run(kb_count_command, check=True)
    else:
        logger.warning(f"Skipping kb count because file {file_signifying_successful_kb_count_vcrs_completion} already exists and overwrite=False")

    if ((not os.path.exists(file_signifying_successful_kb_count_reference_genome_completion)) and (technology not in non_single_cell_technologies) and any(params_dict.get(value, False) for value in needs_for_normal_genome_matrix)) or (params_dict.get("qc_against_gene_matrix") and len(os.listdir(kb_count_reference_genome_out_dir)) == 0):  # align to this genome if either (1) adata doesn't exist and I do downstream analysis with the normal gene count matrix for scRNA-seq data (ie not bulk) or (2) [qc_against_gene_matrix=True and kb_count_reference_genome_out_dir is empty (because I need the BUS file for this)]  # purposely omitted overwrite because it is reasonable to expect that someone has pre-computed this matrix and doesn't want it recomputed under any circumstances (and if they did, then simply point to a different directory)
        if not isinstance(species, str) and species not in supported_downloadable_normal_reference_genomes_with_kb_ref:
            raise ValueError(f"Species {species} is not supported. Supported values are {supported_downloadable_normal_reference_genomes_with_kb_ref}. See more details at https://github.com/pachterlab/kallisto-transcriptome-indices/")

        reference_genome_index = params_dict.get("reference_genome_index", os.path.join(out, "reference_genome_index.idx"))
        reference_genome_t2g = params_dict.get("reference_genome_t2g", os.path.join(out, "reference_genome_t2g.t2g"))

        if not os.path.exists(reference_genome_index) or not os.path.exists(reference_genome_t2g):  # download reference if does not exist
            # kb ref, reference genome
            kb_ref_command = [
                "kb",
                "ref",
                "-t",
                str(threads),
                "-i",
                reference_genome_index,
                "-g",
                reference_genome_t2g,
                "-d",
                species,
            ]

            if dry_run:
                print(kb_ref_command)
            else:
                logger.info(f"Running kb ref for reference genome with command: {' '.join(kb_ref_command)}")
                subprocess.run(kb_ref_command, check=True)

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
            "-k",
            str(k),
            "-i",
            reference_genome_index,
            "-g",
            reference_genome_t2g,
            "-x",
            technology,
            "--h5ad",
            "--parity",
            parity,
            "--strand",
            strand,
            "-o",
            kb_count_reference_genome_out_dir,
        ]

        # assumes any argument in varseek count matches kb count identically, except dashes replaced with underscores
        for dict_key, arguments in varseek_count_only_allowable_kb_count_arguments.items():
            for argument in list(arguments):
                dash_count = len(argument) - len(argument.lstrip("-"))
                leading_dashes = "-" * dash_count
                argument = argument.lstrip("-").replace("-", "_")
                if argument in params_dict:
                    value = params_dict[argument]
                    if dict_key == "zero_arguments":
                        if value:  # only add if value is True
                            kb_count_standard_index_command.append(f"{leading_dashes}{argument}")
                    elif dict_key == "one_argument":
                        kb_count_standard_index_command.extend([f"{leading_dashes}{argument}", value])
                    else:  # multiple_arguments or something else
                        pass

        kb_count_standard_index_command += fastqs_reference_genome

        if dry_run:
            print(kb_count_standard_index_command)
        else:
            logger.info(f"Running kb count for reference genome with command: {' '.join(kb_count_standard_index_command)}")
            subprocess.run(kb_count_standard_index_command, check=True)

    elif os.path.exists(file_signifying_successful_kb_count_reference_genome_completion) and params_dict.get("qc_against_gene_matrix"):
        logger.warning(f"Skipping kb count for reference genome because file {file_signifying_successful_kb_count_reference_genome_completion} already exists. Note that even setting overwrite=True will still not overwrite this particular directory")

    params_dict["adata_vcrs"] = adata_vcrs if not params_dict.get("adata_vcrs") else params_dict.get("adata_vcrs")  # for vk clean
    # kb_count_vcrs_dir already set to kb_count_vcrs_out_dir by check_that_two_directories_in_params_dict_are_the_same_if_both_provided_otherwise_set_them_equal
    params_dict["adata_reference_genome"] = adata_reference_genome  # for vk clean - the conditional part already handled in section 6

    # vk clean
    if not disable_clean:
        if not os.path.exists(file_signifying_successful_vk_clean_completion) or overwrite:
            params_dict["overwrite"], params_dict_vk_clean["overwrite"] = True, True
            logger.info("Running vk clean")
            _ = vk.clean(**params_dict) if not params_dict.get("dry_run", False) else vk.clean(**params_dict_vk_clean)
            params_dict["overwrite"], params_dict_vk_clean["overwrite"] = overwrite_original, overwrite_original
        else:
            logger.warning(f"Skipping vk clean because file {file_signifying_successful_vk_clean_completion} already exists and overwrite=False")
        adata_for_summarize = adata_vcrs_clean_out
    else:
        logger.warning("Skipping vk clean because disable_clean=True")
        adata_for_summarize = adata_vcrs
    params_dict["adata"] = adata_for_summarize  # so that the correct adata gets passed into vk summarize

    # # vk summarize
    if not disable_summarize:
        out_original = params_dict.get("out")
        if not os.path.exists(file_signifying_successful_vk_summarize_completion) or overwrite:
            params_dict["out"] = vk_summarize_out_dir
            params_dict["overwrite"], params_dict_vk_summarize["overwrite"] = True, True
            logger.info("Running vk summarize")
            _ = vk.summarize(**params_dict) if not params_dict.get("dry_run", False) else vk.summarize(**params_dict_vk_summarize)
            params_dict["overwrite"], params_dict_vk_summarize["overwrite"] = overwrite_original, overwrite_original
            params_dict["out"] = out_original
        else:
            logger.warning(f"Skipping vk summarize because file {file_signifying_successful_vk_summarize_completion} already exists and overwrite=False")
    else:
        logger.warning("Skipping vk summarize because disable_summarize=True")

    vk_count_output_dict = {}
    vk_count_output_dict["adata_path_unprocessed"] = adata_vcrs
    vk_count_output_dict["adata_path_reference_genome_unprocessed"] = adata_reference_genome
    vk_count_output_dict["adata_path"] = adata_vcrs_clean_out
    vk_count_output_dict["adata_path_reference_genome"] = adata_reference_genome_clean_out

    vk_count_output_dict["vcf"] = vcf_out
    vk_count_output_dict["vk_summarize_output_dir"] = vk_summarize_out_dir

    report_time_elapsed(start_time, logger=logger, verbose=verbose, function_name="count")

    return vk_count_output_dict
