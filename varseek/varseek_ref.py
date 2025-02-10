"""varseek ref and specific helper functions."""
import inspect
import os
import subprocess
import time

import varseek as vk
from varseek.utils import (
    authenticate_cosmic_credentials,
    authenticate_cosmic_credentials_via_server,
    check_file_path_is_string_with_valid_extension,
    download_varseek_files,
    encode_cosmic_credentials,
    get_python_or_cli_function_call,
    is_valid_int,
    make_function_parameter_to_value_dict,
    report_time_elapsed,
    save_params_to_config_file,
    save_run_info,
    set_up_logger,
    check_that_two_directories_in_params_dict_are_the_same_if_both_provided_otherwise_set_them_equal
)

from .constants import prebuilt_vk_ref_files

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

varseek_ref_only_allowable_kb_ref_arguments = {"zero_arguments": {"--keep-tmp", "--verbose", "--aa"}, "one_argument": {"--tmp", "--kallisto", "--bustools"}, "multiple_arguments": set()}  # don't include d-list, t, i, k, workflow, overwrite here because I do it myself later


# covers both varseek ref AND kb ref, but nothing else (i.e., all of the arguments that are not contained in varseek build, info, or filter)
def validate_input_ref(params_dict):
    dlist = params_dict.get("dlist", None)
    mode = params_dict.get("mode", None)
    threads = params_dict.get("threads", None)
    index_out = params_dict.get("index_out", None)
    config = params_dict.get("config", None)

    build_signature = inspect.signature(vk.varseek_build.build)
    k = params_dict.get("k", None)
    w = params_dict.get("w", None)

    if k is None:
        k = build_signature.parameters["k"].default
    if w is None:
        w = build_signature.parameters["w"].default

    # make sure k is odd and <= 63
    if not isinstance(k, int) or k % 2 == 0 or k < 1 or k > 63:
        raise ValueError("k must be odd, positive, integer, and less than or equal to 63")

    if k < w + 1:
        raise ValueError("k must be greater than or equal to w + 1")

    dlist_valid_values = {"genome", "transcriptome", "genome_and_transcriptome", "None", None}
    if dlist not in dlist_valid_values:
        raise ValueError(f"dlist must be one of {dlist_valid_values}")

    if mode is not None and mode not in mode_parameters:
        raise ValueError(f"mode must be one of {mode_parameters.keys()}")

    # sequences, mutations, out handled by vk build

    check_file_path_is_string_with_valid_extension(index_out, "index_out", "index")
    check_file_path_is_string_with_valid_extension(config, "config", ["json", "yaml"])

    if not is_valid_int(threads, threshold_type=">=", threshold_value=1):
        raise ValueError(f"Threads must be a positive integer, got {threads}")

    for param_name in ["minimum_info_columns", "download", "dry_run"]:
        if not isinstance(params_dict.get(param_name), bool):
            raise ValueError(f"{param_name} must be a boolean. Got {param_name} of type {type(params_dict.get(param_name))}.")

    mutations = params_dict["mutations"]
    sequences = params_dict["sequences"]
    # more on download
    if mutations not in prebuilt_vk_ref_files:
        raise ValueError(f"When downloading prebuilt reference, `mutations` must be one of {prebuilt_vk_ref_files.keys()}. mutation={mutations} not recognized.")
    if sequences not in prebuilt_vk_ref_files[mutations]:
        raise ValueError(f"When downloading prebuilt reference, `sequences` must be one of {prebuilt_vk_ref_files[mutations].keys()}. sequences={sequences} not recognized.")

    # kb ref stuff
    for argument_type, argument_set in varseek_ref_only_allowable_kb_ref_arguments.items():
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


# for determining when to check for cosmic credentials
mutation_values_for_cosmic = {"cosmic_cmc"}

# a list of dictionaries with keys "mutations", "sequences", and "description"
downloadable_references = [
    {"mutations": "cosmic_cmc", "sequences": "cdna", "description": "COSMIC Cancer Mutation Census version 100 - Ensembl GRCh37 release 93 cDNA reference annotations. All default arguments of varseek ref (k=59, w=54, filters, no d-list, etc.)."},
    {"mutations": "cosmic_cmc", "sequences": "genome", "description": "COSMIC Cancer Mutation Census version 100 - Ensembl GRCh37 release 93 genome reference annotations. All default arguments of varseek ref (k=59, w=54, filters, no d-list, etc.)."},
]


# don't worry if it says an argument is unused - they will all get put in params_dict
def ref(
    sequences,
    mutations,
    w=54,
    k=59,
    filters=(
        "dlist_substring:equal=none",  # filter out mutations which are a substring of the reference genome
        "pseudoaligned_to_human_reference_despite_not_truly_aligning:is_not_true",  # filter out mutations which pseudoaligned to human genome despite not truly aligning
        "dlist:equal=none",  # *** erase eventually when I want to d-list  # filter out mutations which are capable of being d-listed (given that I filter out the substrings above)
        "longest_homopolymer_length:bottom_percent=99.99",  # filters out MCRSs with repeating single nucleotide - 99.99 keeps the bottom 99.99% (fraction 0.9999) ie filters out the top 0.01%
        "triplet_complexity:top_percent=99.9",  # filters out MCRSs with repeating triplets - 99.9 keeps the top 99.9% (fraction 0.999) ie filters out the bottom 0.1%
    ),
    mode=None,
    dlist=None,
    dlist_reference_source="T2T",
    config=None,
    out=".",
    index_out=None,
    t2g_out=None,  # intentionally avoid having this name clash with the t2g from vk build and vk filter, as it could refer to either (depending on whether or not filtering will occur)
    download=False,
    dry_run=False,
    list_downloadable_references=False,
    minimum_info_columns=True,
    overwrite=False,
    threads=2,
    verbose=True,
    **kwargs,  # * including all arguments for vk build, info, and filter
):
    """
    Create a reference index and t2g file for variant screening with varseek count. Wraps around varseek build, varseek info, varseek filter, and kb ref.

    # Required input argument:
    - sequences     (str) Path to the fasta file containing the sequences to have the mutations added, e.g., 'seqs.fa'.
                    Sequence identifiers following the '>' character must correspond to the identifiers
                    in the seq_ID column of 'mutations'.

                    Example:
                    >seq1 (or ENSG00000106443)
                    ACTGCGATAGACT
                    >seq2
                    AGATCGCTAG

                    Alternatively: Input sequence(s) as a string or a list of strings,
                    e.g. 'AGCTAGCT' or ['ACTGCTAGCT', 'AGCTAGCT'].

                    NOTE: Only the letters until the first space or dot will be used as sequence identifiers
                    - Version numbers of Ensembl IDs will be ignored.
                    NOTE: When 'sequences' input is a genome, also see 'gtf' argument below.

                    Alternatively, if 'mutations' is a string specifying a supported database,
                    sequences can be a string indicating the source upon which to apply the mutations.
                    See below for supported databases and sequences options.
                    To see the supported combinations of mutations and sequences, either
                    1) run `vk build --list_supported_databases` from the command line, or
                    2) run varseek.build(list_supported_databases=True) in python

    - mutations     (str or DataFrame object) Path to csv or tsv file (str) (e.g., 'mutations.csv'), or DataFrame (DataFrame object),
                    containing information about the mutations in the following format:

                    | mutation         | mut_ID | seq_ID |
                    | c.2C>T           | mut1   | seq1   | -> Apply mutation 1 to sequence 1
                    | c.9_13inv        | mut2   | seq2   | -> Apply mutation 2 to sequence 2
                    | c.9_13inv        | mut2   | seq3   | -> Apply mutation 2 to sequence 3
                    | c.9_13delinsAAT  | mut3   | seq3   | -> Apply mutation 3 to sequence 3
                    | ...              | ...    | ...    |

                    'mutation' = Column containing the mutations to be performed written in standard mutation annotation (see below)
                    'seq_ID' = Column containing the identifiers of the sequences to be mutated (must correspond to the string following
                    the > character in the 'sequences' fasta file; do NOT include spaces or dots)
                    'mut_ID' = Column containing an identifier for each mutation (optional).

                    Alternatively: Input mutation(s) as a string or list, e.g., 'c.2C>T' or ['c.2C>T', 'c.1A>C'].
                    If a list is provided, the number of mutations must equal the number of input sequences.

                    For more information on the standard mutation annotation, see https://www.ncbi.nlm.nih.gov/pmc/articles/PMC1867422/.

                    Alternatively, 'mutations' can be a string specifying a supported database, which will automatically download
                    both the mutation database and corresponding reference sequence (if the 'sequences' is not a path).
                    To see the supported combinations of mutations and sequences, either
                    1) run `vk build --list_supported_databases` from the command line, or
                    2) run varseek.build(list_supported_databases=True) in python

    # Additional parameters
    - w             (int) Length of sequence windows flanking the variant. Default: 30. If w > total length of the sequence, the entire sequence will be kept.
    - k             (int) The length of each k-mer in the kallisto reference index construction. Accordingly corresponds to the length of the k-mers to be considered in vk build's remove_seqs_with_wt_kmers, and the default minimum value for vk build's minimum sequence length (which can be changed with 'min_seq_len'). Must be greater than the value passed in for w. Default: 59.
    - filters       (str or list[str]) List of filters to apply to the mutations. See varseek filter documentation for more information.
    - mode          (str) Mode to use for kb ref. Currently not implemented. Default: None.
    - dlist         (str) Specifies whether ones wants to d-list against the genome, transcriptome, or both. Possible values are "genome", "transcriptome", "genome_and_transcriptome", or None. Default: None.
    - dlist_reference_source (str) Specifies whether to use the T2T, grch37, or grch38 reference genome during alignment of VCRS k-mers to the reference genome/transcriptome and any possible d-list construction. However, no d-list is used during the creation of the VCRS reference index unless `dlist` is not None. Default: "T2T".
    - config        (str) Path to config file. Default: None.

    # Optional output file paths: (only needed if changing/customizing file names or locations):
    - out           (str) Output directory. Default: ".".
    - index_out (str) Path to output mcrs index file. Default: `out`/mcrs_index.idx.
    - t2g_out (str) Path to output mcrs t2g file to be used in alignment. Default: `out`/mcrs_t2g.txt.

    # General arguments:
    - download      (bool) If True, download prebuilt reference files. Default: False.
    - dry_run       (bool) If True, print the commands that would be run without actually running them. Default: False.
    - list_downloadable_references (bool) If True, list the available downloadable references. Default: False.
    - minimum_info_columns (bool) If True, run vk info with minimum columns. Default: True.
    - overwrite     (bool) If True, overwrite existing files. Default: False.
    - threads       (int) Number of threads to use. Default: 2.
    - verbose       (bool) If True, print progress messages. Default: True.

    For a complete list of supported parameters, see the documentation for varseek build, varseek info, varseek filter, and kb ref. Note that any shared parameter names between functions are meant to have identical purposes.
    """

    # * 0. Informational arguments that exit early
    if list_downloadable_references:  # for vk ref
        for downloadable_reference in downloadable_references:
            print(f"mutations: {downloadable_reference['mutations']}, sequences: {downloadable_reference['sequences']}, description: {downloadable_reference['description']}")
        return None

    if kwargs.get("list_supported_databases"):  # from vk build
        vk.varseek_build.print_valid_values_for_mutations_and_sequences_in_varseek_build()
        return None
    if kwargs.get("list_columns"):  # from vk info
        vk.varseek_info.print_list_columns()
        return None
    if kwargs.get("list_filter_rules"):  # from vk filter
        vk.varseek_filter.print_list_filter_rules()
        return None

    # * 1. Start timer
    start_time = time.perf_counter()

    # * 1.5. Load in parameters from a config file if provided
    if isinstance(config, str) and os.path.isfile(config):
        vk_ref_config_file_input = vk.utils.load_params(config)

        # overwrite any parameters passed in with those from the config file
        ref_signature = inspect.signature(ref)
        for key, value in vk_ref_config_file_input.items():
            if key in ref_signature.parameters.keys():
                exec("%s = %s" % (key, value))  # assign the value to the variable name
            else:
                kwargs[key] = value  # if the variable is not in the function signature, then add it to kwargs

    # * 2. Type-checking
    params_dict = make_function_parameter_to_value_dict(1)
    vk.varseek_build.validate_input_build(params_dict)  # this passes all vk ref parameters to the function - I could only pass in the vk build parameters here if desired (and likewise below), but there should be no naming conflicts anyways
    vk.varseek_info.validate_input_info(params_dict)
    vk.varseek_filter.validate_input_filter(params_dict)
    validate_input_ref(params_dict)

    # * 3. Dry-run
    # handled within child functions

    # * 3.75 Pop out any unallowable arguments
    for key, unallowable_set in varseek_ref_unallowable_arguments.items():
        for unallowable_key in unallowable_set:
            params_dict.pop(unallowable_key, None)

    # * 3.8 Set params_dict to default values of children functions (not strictly necessary, as if these arguments are not in params_dict then it will use the default values anyways, but important if I need to rely on these default values within vk ref)
    # ref_signature = inspect.signature(ref)
    # for function in (vk.varseek_build.build, vk.varseek_info.info, vk.varseek_filter.filter):
    #     signature = inspect.signature(function)
    #     for key in signature.parameters.keys():
    #         if key not in params_dict and key not in ref_signature.parameters.keys():
    #             params_dict[key] = signature.parameters[key].default

    # * 4. Save params to config file and run info file
    # Save parameters to config file
    config_file = os.path.join(out, "config", "vk_ref_config.json")
    save_params_to_config_file(params_dict, config_file)

    run_info_file = os.path.join(out, "config", "vk_ref_run_info.txt")
    save_run_info(run_info_file)

    # * 5. Set up default folder/file input paths, and make sure the necessary ones exist
    # all input files for vk ref are required in the varseek workflow, so this is skipped

    # * 5.5 Setting up modes
    # if mode:  #* uncomment once I have modes
    #     for key in mode_parameters[mode]:
    #         params_dict[key] = mode_parameters[mode][key]

    # * 6. Set up default folder/file output paths, and make sure they don't exist unless overwrite=True
    # Make directories
    os.makedirs(out, exist_ok=True)

    # define some more file paths
    if not index_out:
        index_out = os.path.join(out, "mcrs_index.idx")
    os.makedirs(os.path.dirname(index_out), exist_ok=True)

    mcrs_fasta_out = params_dict.get("mcrs_fasta_out", os.path.join(out, "mcrs.fa"))  # make sure this matches vk build
    mcrs_filtered_fasta_out = params_dict.get("mcrs_filtered_fasta_out", os.path.join(out, "mcrs_filtered.fa"))  # make sure this matches vk filter
    mcrs_t2g_out = params_dict.get("mcrs_t2g_out", os.path.join(out, "mcrs_t2g.txt"))  # make sure this matches vk build
    mcrs_t2g_filtered_out = params_dict.get("mcrs_t2g_filtered_out", os.path.join(out, "mcrs_t2g_filtered.txt"))  # make sure this matches vk filter
    dlist_genome_fasta_out = params_dict.get("dlist_genome_fasta_out", os.path.join(out, "dlist_genome.fa"))  # make sure this matches vk info
    dlist_cdna_fasta_out = params_dict.get("dlist_cdna_fasta_out", os.path.join(out, "dlist_cdna.fa"))  # make sure this matches vk info
    dlist_combined_fasta_out = params_dict.get("dlist_combined_fasta_out", os.path.join(out, "dlist.fa"))  # make sure this matches vk info

    for file in [index_out]:  # purposely exluding mcrs_fasta_out, mcrs_filtered_fasta_out, mcrs_t2g_out, mcrs_t2g_filtered_out because - let's say someone runs vk ref and they get an error write in the kb ref step because of a bad argument that doesn't affect the prior steps - it would be nice for someone to be able to rerun the command with the changed argument without having to rerun vk build, info, filter from scratch when overwrite=False - and if they do want to rerun those steps, they can just delete the files or set overwrite=True
        if os.path.isfile(file) and not overwrite:
            raise FileExistsError(f"Output file {file} already exists. Please delete it or specify a different output directory or set overwrite=True.")

    mutations_updated_vk_info_csv_out = params_dict.get("mutations_updated_vk_info_csv_out", None)
    if not mutations_updated_vk_info_csv_out:
        mutations_updated_vk_info_csv_out = os.path.join(out, "mutation_metadata_df_updated_vk_info.csv")

    file_signifying_successful_vk_build_completion = mcrs_fasta_out
    file_signifying_successful_vk_info_completion = mutations_updated_vk_info_csv_out
    files_signifying_successful_vk_filter_completion = (mcrs_filtered_fasta_out, mcrs_t2g_filtered_out)
    file_signifying_successful_kb_ref_completion = index_out

    overwrite_original = params_dict.get("overwrite_original", False)

    params_dict = check_that_two_directories_in_params_dict_are_the_same_if_both_provided_otherwise_set_them_equal(params_dict, "out", "input_dir")  # check that, if out and input_dir are both provided, they are the same directory; otherwise, if only one is provided, then make them equal to each other
    params_dict = check_that_two_directories_in_params_dict_are_the_same_if_both_provided_otherwise_set_them_equal(params_dict, "mcrs_fasta_out", "mcrs_fasta")  # build --> info
    params_dict = check_that_two_directories_in_params_dict_are_the_same_if_both_provided_otherwise_set_them_equal(params_dict, "id_to_header_csv_out", "id_to_header_csv")  # build --> info/filter
    params_dict = check_that_two_directories_in_params_dict_are_the_same_if_both_provided_otherwise_set_them_equal(params_dict, "mutations_updated_csv_out", "mutations_updated_csv")  # build --> info
    params_dict = check_that_two_directories_in_params_dict_are_the_same_if_both_provided_otherwise_set_them_equal(params_dict, "mutations_updated_vk_info_csv_out", "mutations_updated_vk_info_csv")  # info --> filter
    params_dict = check_that_two_directories_in_params_dict_are_the_same_if_both_provided_otherwise_set_them_equal(params_dict, "mutations_updated_exploded_vk_info_csv_out", "mutations_updated_exploded_vk_info_csv")  # info --> filter
    # dlist handled below - see the comment "set d-list argument"

    # * 6.5 Just to make the unused parameter coloration go away in VSCode
    filters = filters
    w = w
    mode = mode
    verbose = verbose
    dlist_reference_source = dlist_reference_source

    # * 7. Define kwargs defaults
    # Nothing to see here

    # * 8. Start the actual function
    # get COSMIC info
    cosmic_email = params_dict.get("cosmic_email", None)
    if not cosmic_email:
        cosmic_email = os.getenv("COSMIC_EMAIL")
    if cosmic_email:
        logger.info(f"Using COSMIC email from COSMIC_EMAIL environment variable: {cosmic_email}")
        params_dict["cosmic_email"] = cosmic_email

    cosmic_password = params_dict.get("cosmic_password", None)
    if not cosmic_password:
        cosmic_password = os.getenv("COSMIC_PASSWORD")
    if cosmic_password:
        logger.info("Using COSMIC password from COSMIC_PASSWORD environment variable")
        params_dict["cosmic_password"] = cosmic_password

    # decide whether to skip vk info and vk filter
    # filters_column_names = list({filter.split('-')[0] for filter in filters})
    skip_filter = not bool(params_dict.get("filters"))  # skip filtering if no filters provided
    skip_info = minimum_info_columns and skip_filter  # skip vk info if no filtering will be performed and one specifies minimum info columns

    if skip_filter:
        mcrs_fasta_for_index = mcrs_fasta_out
        if t2g_out:
            params_dict["mcrs_t2g_out"] = t2g_out  # pass this custom path into vk build
            mcrs_t2g_out = t2g_out  # pass this custom path into the output dict of vk ref
        mcrs_t2g_for_alignment = mcrs_t2g_out
    else:
        mcrs_fasta_for_index = mcrs_filtered_fasta_out
        if t2g_out:
            params_dict["mcrs_t2g_filtered_out"] = t2g_out
            mcrs_t2g_filtered_out = t2g_out
        mcrs_t2g_for_alignment = mcrs_t2g_filtered_out

    # download if download argument is True
    if download:
        # $ I opt to keep it like this rather than converting the keys of prebuilt_vk_ref_files to a tuple of many arguments for user simplicity - simply document the uploaded references, but no need to differentiate - but if I do end up having multiple reference documents with the same values for mutations and sequences, then switch over to this approach where the dict keys are tuples
        file_dict = prebuilt_vk_ref_files.get(mutations, {}).get(sequences, {})  # when I add mode: file_dict = prebuilt_vk_ref_files.get(mutations, {}).get(sequences, {}).get(mode, {})
        if file_dict:
            if mutations in mutation_values_for_cosmic:
                # encoded_cosmic_credentials = encode_cosmic_credentials(cosmic_email, cosmic_password)  #!! uncomment once I'm ready to try out server
                # if not authenticate_cosmic_credentials_via_server(encoded_cosmic_credentials):  #!! uncomment once I'm ready to try out server
                if not authenticate_cosmic_credentials(cosmic_email, cosmic_password):  #!! erase once I'm ready to try out server
                    raise ValueError(f"Trying to download a COSMIC-based index (mutations={mutations}) with invalid COSMIC credentials")
            print(f"Downloading reference files with mutations={mutations}, sequences={sequences}")
            vk_ref_output_dict = download_varseek_files(file_dict, out=out)  # TODO: replace with DOI download (will need to replace prebuilt_vk_ref_files urls with DOIs)
            if index_out and vk_ref_output_dict["index"] != index_out:
                os.rename(vk_ref_output_dict["index"], index_out)
                vk_ref_output_dict["index"] = index_out
            if t2g_out and vk_ref_output_dict["t2g"] != t2g_out:
                os.rename(vk_ref_output_dict["t2g"], t2g_out)
                vk_ref_output_dict["t2g"] = t2g_out

            return vk_ref_output_dict
        else:
            raise ValueError(f"No prebuilt files found for the given arguments:\nmutations: {mutations}\nsequences: {sequences}")  # \nmode: {mode}"

    # set d-list argument
    if dlist == "genome":
        dlist_kb_argument = dlist_genome_fasta_out  # for kb ref
        params_dict["dlist_fasta"] = dlist_genome_fasta_out  # for vk filter
    elif dlist == "transcriptome":
        dlist_kb_argument = dlist_cdna_fasta_out
        params_dict["dlist_fasta"] = dlist_cdna_fasta_out
    elif dlist == "genome_and_transcriptome":
        dlist_kb_argument = dlist_combined_fasta_out
        params_dict["dlist_fasta"] = dlist_combined_fasta_out
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

    all_parameter_names_set_vk_build = explicit_parameters_vk_build | allowable_kwargs_vk_build
    all_parameter_names_set_vk_info = explicit_parameters_vk_info | allowable_kwargs_vk_info
    all_parameter_names_set_vk_filter = explicit_parameters_vk_filter | allowable_kwargs_vk_filter

    params_dict_vk_build = {key: value for key, value in params_dict.items() if key in all_parameter_names_set_vk_build}
    params_dict_vk_info = {key: value for key, value in params_dict.items() if key in all_parameter_names_set_vk_info}
    params_dict_vk_filter = {key: value for key, value in params_dict.items() if key in all_parameter_names_set_vk_filter}

    # vk build
    if not os.path.exists(file_signifying_successful_vk_build_completion) or overwrite:  # the reason I do it like this, rather than if overwrite or not os.path.exists(MYPATH), is because I would like vk ref/count to automatically overwrite partially-completed function outputs even when overwrite=False; but when overwrite=True, then run from scratch regardless
        params_dict["overwrite"], params_dict_vk_build["overwrite"] = True, True
        logger.info("Running vk build")
        _ = vk.build(**params_dict) if not params_dict.get("dry_run", False) else vk.build(**params_dict_vk_build)  # best of both worlds - will only pass in defined arguments if dry run is True (which is good so that I don't show each function with a bunch of args it never uses), but will pass in all arguments if dry run is False (which is good if I run vk ref with a new parameter that I have not included in docstrings yet, as I only get usable kwargs list from docstrings)
        params_dict["overwrite"], params_dict_vk_build["overwrite"] = overwrite_original, overwrite_original
    else:
        logger.warning(f"Skipping vk build because {file_signifying_successful_vk_build_completion} already exists and overwrite=False")

    # vk info
    if not skip_info:
        if not os.path.exists(file_signifying_successful_vk_info_completion) or overwrite:
            params_dict["overwrite"], params_dict_vk_info["overwrite"] = True, True
            logger.info("Running vk info")
            _ = vk.info(**params_dict) if not params_dict.get("dry_run", False) else vk.info(**params_dict_vk_info)
            params_dict["overwrite"], params_dict_vk_info["overwrite"] = overwrite_original, overwrite_original
        else:
            logger.warning(f"Skipping vk info because {file_signifying_successful_vk_info_completion} already exists and overwrite=False")

    # vk filter
    if not skip_filter:
        if not all(os.path.exists(f) for f in files_signifying_successful_vk_filter_completion) or overwrite:
            params_dict["overwrite"], params_dict_vk_filter["overwrite"] = True, True
            logger.info("Running vk filter")
            _ = vk.filter(**params_dict) if not params_dict.get("dry_run", False) else vk.filter(**params_dict_vk_filter)
            params_dict["overwrite"], params_dict_vk_filter["overwrite"] = overwrite_original, overwrite_original
        else:
            logger.warning(f"Skipping vk filter because {files_signifying_successful_vk_filter_completion} already exist and overwrite=False")

    # kb ref
    kb_ref_command = [
        "kb",
        "ref",
        "--workflow",
        "custom",
        "-t",
        str(threads),
        "-i",
        index_out,
        "--d-list",
        dlist_kb_argument,
        "-k",
        str(k),
        "--overwrite",
        True,  # set to True here regardless of the overwrite argument because I would only even enter this block if kb count was only partially run (as seen by the lack of existing of file_signifying_successful_kb_ref_completion), in which case I should overwrite anyways
    ]

    # assumes any argument in varseek ref matches kb ref identically, except dashes replaced with underscores
    for dict_key, arguments in varseek_ref_only_allowable_kb_ref_arguments.items():
        for argument in list(arguments):
            dash_count = len(argument) - len(argument.lstrip("-"))
            leading_dashes = "-" * dash_count
            argument = argument.lstrip("-").replace("-", "_")
            if argument in params_dict:
                value = params_dict[argument]
                if dict_key == "zero_arguments":
                    if value:  # only add if value is True
                        kb_ref_command.append(f"{leading_dashes}{argument}")
                elif dict_key == "one_argument":
                    kb_ref_command.extend([f"{leading_dashes}{argument}", value])
                else:  # multiple_arguments or something else
                    pass

    kb_ref_command.append(mcrs_fasta_for_index)

    if dry_run:
        print(kb_ref_command)
        index_out = None
        mcrs_t2g_for_alignment = None
    else:
        if not os.path.exists(file_signifying_successful_kb_ref_completion) or overwrite:
            logger.info(f"Running kb ref with command: {' '.join(kb_ref_command)}")
            subprocess.run(kb_ref_command, check=True)
        else:
            logger.warning(f"Skipping kb ref because {file_signifying_successful_kb_ref_completion} already exists and overwrite=False")


    #!!! erase if removing wt mcrs feature
    wt_mcrs_fasta_out = params_dict.get("mcrs_fasta_out", os.path.join(out, "wt_mcrs.fa"))  # make sure this matches vk build
    wt_mcrs_filtered_fasta_out = params_dict.get("mcrs_filtered_fasta_out", os.path.join(out, "wt_mcrs_filtered.fa"))  # make sure this matches vk filter
    mcrs_wt_fasta_for_index = wt_mcrs_fasta_out if skip_filter else wt_mcrs_filtered_fasta_out
    wt_mcrs_index_out = params_dict.get("wt_mcrs_index_out", os.path.join(out, "wt_mcrs_index.idx"))
    file_signifying_successful_wt_mcrs_kb_ref_completion = wt_mcrs_index_out
    
    kb_ref_wt_mcrs_command = [
        "kb",
        "ref",
        "--workflow",
        "custom",
        "-t",
        str(threads),
        "-i",
        wt_mcrs_index_out,
        "--d-list",
        "None",
        "-k",
        str(k),
        "--overwrite",
        True,  # set to True here regardless of the overwrite argument because I would only even enter this block if kb count was only partially run (as seen by the lack of existing of file_signifying_successful_wt_mcrs_kb_ref_completion), in which case I should overwrite anyways
        mcrs_wt_fasta_for_index
    ]

    if dry_run:
        print(kb_ref_wt_mcrs_command)
    else:
        if not os.path.exists(file_signifying_successful_wt_mcrs_kb_ref_completion) or overwrite:
            logger.info(f"Running kb ref for wt mcrs index with command: {' '.join(kb_ref_wt_mcrs_command)}")
            subprocess.run(kb_ref_wt_mcrs_command, check=True)
        else:
            logger.warning(f"Skipping kb ref for wt mcrs because {file_signifying_successful_wt_mcrs_kb_ref_completion} already exists and overwrite=False")
    #!!! erase if removing wt mcrs feature

    vk_ref_output_dict = {}
    vk_ref_output_dict["index"] = index_out
    vk_ref_output_dict["t2g"] = mcrs_t2g_for_alignment

    # Report time
    report_time_elapsed(start_time, logger=logger, verbose=verbose, function_name="ref")

    return vk_ref_output_dict


# ref.__doc__ = ref.__doc__ + "\n================================\n\n" + "varseek build parameters" + vk.varseek_build.build.__doc__ + "\n================================\n\n" + vk.varseek_info.info.__doc__ + "\n================================\n\n" + vk.varseek_filter.filter.__doc__ + "\n================================\n\n" + "Run `kb ref --help` for more information on kb ref arguments"
