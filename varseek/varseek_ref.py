import os
import subprocess
import time
import varseek as vk
from varseek.utils import set_up_logger, save_params_to_config_file, make_function_parameter_to_value_dict, download_varseek_files, report_time_elapsed, is_valid_int, check_file_path_is_string_with_valid_extension
from .constants import prebuilt_vk_ref_files
import inspect

logger = set_up_logger()

mode_parameters = {
    "very_sensitive": {},
    "sensitive": {},
    "balanced": {},
    "specific": {},
    "very_specific": {},
}

varseek_ref_unallowable_arguments = {
    "varseek_build": {"return_mutation_output"},
    "varseek_info": set(),
    "varseek_filter": set(),
    "kb_ref": set(),
}

varseek_ref_only_allowable_kb_ref_arguments = {
    "zero_arguments": {"--keep-tmp", "--verbose", "--aa", "--overwrite"},
    "one_argument": {"--tmp", "--kallisto", "--bustools"},
    "multiple_arguments": set()
}  # don't include d-list, t, i, k, workflow here because I do it myself later

# covers both varseek ref AND kb ref, but nothing else (i.e., all of the arguments that are not contained in varseek build, info, or filter)
def validate_input_ref(dlist, mode, threads, mcrs_index_out, config, minimum_info_columns, download, dry_run, **kwargs):
    build_signature = inspect.signature(vk.varseek_build.build)
    k = kwargs.get("k", None)
    w = kwargs.get("w", None)
    
    if k is None:
        k = build_signature.parameters["k"].default
    if w is None:
        w = build_signature.parameters["w"].default
    
    # make sure k is odd and <= 63
    if k is not None:
        if not isinstance(k, int) or k % 2 != 0 or k < 1 or k > 63:
            raise ValueError("k must be odd, positive, integer, and less than or equal to 63")
    
        if (w is not None) and (k < w + 1):
            raise ValueError("k must be greater than or equal to w + 1")
    
    dlist_valid_values = {"genome", "transcriptome", "genome_and_transcriptome", 'None', None}
    if dlist not in dlist_valid_values:
        raise ValueError(f"dlist must be one of {dlist_valid_values}")
    
    if mode is not None and mode not in mode_parameters:
        raise ValueError(f"mode must be one of {mode_parameters.keys()}")
    
    # sequences, mutations, out handled by vk build

    check_file_path_is_string_with_valid_extension(mcrs_index_out, "mcrs_index_out", "index")
    check_file_path_is_string_with_valid_extension(config, "config", ["json", "yaml"])
    
    if not is_valid_int(threads, threshold_type=">=", threshold_value=1):
        raise ValueError(f"Threads must be a positive integer, got {threads}")
    
    if not isinstance(minimum_info_columns, bool):
        raise ValueError(f"minimum_info_columns must be a boolean. Got {type(minimum_info_columns)}.")
    if not isinstance(download, bool):
        raise ValueError(f"download must be a boolean. Got {type(download)}.")
    if not isinstance(dry_run, bool):
        raise ValueError(f"dry_run must be a boolean. Got {type(dry_run)}.")
    
    # kb ref stuff
    for argument_type, argument_set in varseek_ref_only_allowable_kb_ref_arguments.items():
        for argument in argument_set:
            argument = argument[2:]
            if argument in kwargs:
                argument_value = kwargs[argument]
                if argument_type == "zero_arguments":
                    if not isinstance(argument_value, bool):  # all zero-arguments are bool
                        raise ValueError(f"{argument} must be a boolean. Got {type(argument_value)}.")
                elif argument_type == "one_argument":
                    if not isinstance(argument_value, str):  # all one-arguments are string
                        raise ValueError(f"{argument} must be a string. Got {type(argument_value)}.")
                elif argument_type == "multiple_arguments":
                    pass

def ref(
    sequences,
    mutations,
    out = ".",
    mcrs_index_out = None,
    dlist=False,  # path to dlist fasta file or "None" (including the quotes)
    config=None,
    minimum_info_columns=True,
    download=False,
    mode=None,
    threads=2,
    dry_run=False,
    **kwargs  #* including all arguments for vk build, info, and filter
):
    #* 1. Start timer
    start_time = time.perf_counter()

    #* 2. Type-checking
    params_dict = make_function_parameter_to_value_dict(1)
    vk.varseek_build.validate_input_build(**params_dict)  # this passes all vk ref parameters to the function - I could only pass in the vk build parameters here if desired (and likewise below), but there should be no naming conflicts anyways
    vk.varseek_info.validate_input_info(**params_dict)
    vk.varseek_filter.validate_input_filter(**params_dict)
    validate_input_ref(**params_dict)

    #* 3. Dry-run
    # handled within child functions

    #* 3.5. Load in parameters from a config file if provided
    if isinstance(config, str) and os.path.isfile(config):
        vk_ref_config_file_input = vk.utils.load_params(config)

        # overwrite any parameters passed in with those from the config file
        ref_signature = inspect.signature(ref)
        for k, v in vk_ref_config_file_input.items():
            if k in ref_signature.parameters.keys():
                exec("%s = %s" % (k, v))  # assign the value to the variable name
            else:
                kwargs[k] = v  # if the variable is not in the function signature, then add it to kwargs

    #* 3.75 Pop out any unallowable arguments
    for key, unallowable_set in varseek_ref_unallowable_arguments.items():
        for unallowable_key in unallowable_set:
            kwargs.pop(unallowable_key, None)

    #* 3.8 Set kwargs to default values of children functions (not strictly necessary, as if these arguments are not in kwargs then it will use the default values anyways, but important if I need to rely on these default values within vk ref)
    # ref_signature = inspect.signature(ref)
    # for function in (vk.varseek_build.build, vk.varseek_info.info, vk.varseek_filter.filter):
    #     signature = inspect.signature(function)
    #     for key in signature.parameters.keys():
    #         if key not in kwargs and key not in ref_signature.parameters.keys():
    #             kwargs[key] = signature.parameters[key].default

    #* 4. Save params to config file
    # Save parameters to config file
    config_file = os.path.join(out, "config", "vk_ref_config.json")
    save_params_to_config_file(config_file)

    #* 5. Set up default folder/file input paths, and make sure the necessary ones exist
    # all input files for vk build are required in the varseek workflow, so this is skipped
    
    #* 5.5 Setting up modes
    if mode:
        for key in mode_parameters[mode]:
            kwargs[key] = mode_parameters[mode][key]
    
    #* 6. Set up default folder/file output paths, and make sure they don't exist unless overwrite=True
    # Make directories
    if not reference_out_dir:
        reference_out_dir = os.path.join(out, "reference")

    os.makedirs(out, exist_ok=True)
    os.makedirs(reference_out_dir, exist_ok=True)        

    # define some more file paths
    if not mcrs_index_out:
        mcrs_index_out = os.path.join(out, "mcrs_index.idx")
    os.makedirs(os.path.dirname(mcrs_index_out), exist_ok=True)

    if not mcrs_fasta_out:  # make sure this matches vk build
        mcrs_fasta_out = os.path.join(out, "mcrs.fa")
    if not mcrs_filtered_fasta_out:  # make sure this matches vk filter
        mcrs_filtered_fasta_out = os.path.join(out, "mcrs_filtered.fa")
    if not mcrs_t2g_out:  # make sure this matches vk build
        mcrs_t2g_out = os.path.join(out, "mcrs_t2g.txt")
    if not mcrs_t2g_filtered_out:  # make sure this matches vk filter
        mcrs_t2g_filtered_out = os.path.join(out, "mcrs_t2g_filtered.txt")

    #* 7. Start the actual function
    # get COSMIC info
    cosmic_email = kwargs.get("cosmic_email", None)
    if not cosmic_email:
        cosmic_email = os.getenv("COSMIC_EMAIL")
    if cosmic_email:
        logger.info(f"Using COSMIC email from COSMIC_EMAIL environment variable: {cosmic_email}")
        kwargs["cosmic_email"] = cosmic_email

    cosmic_password = kwargs.get("cosmic_password", None)
    if not cosmic_password:
        cosmic_password = os.getenv("COSMIC_PASSWORD")
    if cosmic_password:
        logger.info("Using COSMIC password from COSMIC_PASSWORD environment variable")
        kwargs["cosmic_password"] = cosmic_password
    
    # decide whether to skip vk info and vk filter
    # filters_column_names = list({filter.split('-')[0] for filter in filters})
    skip_filter = not bool(kwargs.get("filters"))  # skip filtering if no filters provided
    skip_info = (minimum_info_columns and skip_filter)  # skip vk info if no filtering will be performed and one specifies minimum info columns

    if skip_filter:
        mcrs_fasta_for_index = mcrs_fasta_out
        mcrs_t2g_for_alignment = mcrs_t2g_out
    else:
        mcrs_fasta_for_index = mcrs_filtered_fasta_out
        mcrs_t2g_for_alignment = mcrs_t2g_filtered_out

    # download if download argument is True
    if download:
        file_dict = prebuilt_vk_ref_files.get(mutations, {}).get(sequences, {})  # when I add mode: file_dict = prebuilt_vk_ref_files.get(mutations, {}).get(sequences, {}).get(mode, {})
        if file_dict:
            vk_ref_output_dict = download_varseek_files(file_dict, out=out)  # TODO: replace with DOI download (will need to replace prebuilt_vk_ref_files urls with DOIs)
            if mcrs_index_out and vk_ref_output_dict['index'] != mcrs_index_out:
                os.rename(vk_ref_output_dict['index'], mcrs_index_out)
                vk_ref_output_dict['index'] = mcrs_index_out
            if mcrs_t2g_out and vk_ref_output_dict['t2g'] != mcrs_t2g_out:
                os.rename(vk_ref_output_dict['t2g'], mcrs_t2g_out)
                vk_ref_output_dict['t2g'] = mcrs_t2g_out
        
            return vk_ref_output_dict
        else:
            raise ValueError(f"No prebuilt files found for the given arguments:\nmutations: {mutations}\nsequences: {sequences}")  # \nmode: {mode}"
    

    
    # set d-list argument
    if not dlist_genome_fasta_out:
        dlist_genome_fasta_out = os.path.join(out, "dlist_genome.fa")
    if not dlist_cdna_fasta_out:
        dlist_cdna_fasta_out = os.path.join(out, "dlist_cdna.fa")
    if not dlist_combined_fasta_out:
        dlist_combined_fasta_out = os.path.join(out, "dlist.fa")

    if dlist == "genome":
        dlist_kb_argument = dlist_genome_fasta_out
    elif dlist == "transcriptome":
        dlist_kb_argument = dlist_cdna_fasta_out
    elif dlist == "genome_and_transcriptome":
        dlist_kb_argument = dlist_combined_fasta_out
    elif dlist == "None" or dlist is None:
        dlist_kb_argument = "None"
    else:
        raise ValueError("dlist must be 'genome', 'transcriptome', 'genome_and_transcriptome', or 'None'")
    
    # define the vk build, info, and filter arguments (explicit arguments and allowable kwargs)
    explicit_parameters_vk_build = vk.utils.get_set_of_parameters_from_function_signature(vk.varseek_build.build)
    allowable_kwargs_vk_build = vk.utils.get_set_of_allowable_kwargs(vk.varseek_build.build)

    explicit_parameters_vk_info = vk.utils.get_set_of_parameters_from_function_signature(vk.varseek_info.info)
    allowable_kwargs_vk_info = vk.utils.get_set_of_allowable_kwargs(vk.varseek_info.info)

    explicit_parameters_vk_filter = vk.utils.get_set_of_parameters_from_function_signature(vk.varseek_filter.filter)
    allowable_kwargs_vk_filter = vk.utils.get_set_of_allowable_kwargs(vk.varseek_filter.filter)

    function_name_to_dict_of_all_args = {}
    function_name_to_dict_of_all_args['varseek_build'] = explicit_parameters_vk_build | allowable_kwargs_vk_build
    function_name_to_dict_of_all_args['varseek_info'] = explicit_parameters_vk_info | allowable_kwargs_vk_info
    function_name_to_dict_of_all_args['varseek_filter'] = explicit_parameters_vk_filter | allowable_kwargs_vk_filter

    # vk build
    vk.build(**kwargs) if not kwargs.get("dry_run", False) else vk.build(**function_name_to_dict_of_all_args['varseek_build'])  # best of both worlds - will only pass in defined arguments if dry run is True (which is good so that I don't show each function with a bunch of args it never uses), but will pass in all arguments if dry run is False (which is good if I run vk ref with a new parameter that I have not included in docstrings yet, as I only get usable kwargs list from docstrings)

    # vk info
    if not skip_info:
        vk.info(**kwargs) if not kwargs.get("dry_run", False) else vk.info(**function_name_to_dict_of_all_args['varseek_info'])

    # vk filter
    if not skip_filter:
        vk.filter(**kwargs) if not kwargs.get("dry_run", False) else vk.filter(**function_name_to_dict_of_all_args['varseek_filter'])

    # kb ref
    kb_ref_command = [
        "kb",
        "ref",
        "--workflow",
        "custom",
        "-t",
        str(threads),
        "-i",
        mcrs_index_out,
        "--d-list",
        dlist_kb_argument,
        "-k",
        str(k)
    ]

    # assumes any argument in varseek ref matches kb ref identically, except dashes replaced with underscores
    for dict_key in varseek_ref_only_allowable_kb_ref_arguments:
        for argument in list(varseek_ref_only_allowable_kb_ref_arguments[dict_key]):
            dash_count = len(argument) - len(argument.lstrip('-'))
            leading_dashes = "-"*dash_count
            argument = argument.lstrip('-').replace('-', '_')
            if argument in kwargs:
                value = kwargs[argument]
                if dict_key == 'zero_arguments':
                    if value:  # only add if value is True
                        kb_ref_command.append(f"{leading_dashes}{argument}")
                elif dict_key == 'one_argument':
                    kb_ref_command.extend([f"{leading_dashes}{argument}", value])
                else:  # multiple_arguments or something else
                    pass

    kb_ref_command.append(mcrs_fasta_for_index)

    if dry_run:
        print(kb_ref_command)
        mcrs_index_out = None
        mcrs_t2g_for_alignment = None
    else:
        subprocess.run(kb_ref_command, check=True)

    vk_ref_output_dict = {}
    vk_ref_output_dict["index"] = mcrs_index_out
    vk_ref_output_dict["t2g"] = mcrs_t2g_for_alignment

    # Report time
    report_time_elapsed(start_time)

    return vk_ref_output_dict

# ref.__doc__ = ref.__doc__ + vk.varseek_build.build.__doc__ + vk.varseek_info.info.__doc__ + vk.varseek_filter.filter.__doc__
