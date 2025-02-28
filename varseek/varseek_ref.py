"""varseek ref and specific helper functions."""

import inspect
import os
import subprocess
import time
import json
import logging

import requests

import varseek as vk
from varseek.utils import (
    check_file_path_is_string_with_valid_extension,
    check_that_two_paths_are_the_same_if_both_provided_otherwise_set_them_equal,
    download_varseek_files,
    get_python_or_cli_function_call,
    is_valid_int,
    make_function_parameter_to_value_dict,
    report_time_elapsed,
    save_params_to_config_file,
    save_run_info,
    set_up_logger,
)

from .constants import prebuilt_vk_ref_files, supported_databases_and_corresponding_reference_sequence_type, varseek_ref_only_allowable_kb_ref_arguments

logger = logging.getLogger(__name__)
COSMIC_CREDENTIAL_VALIDATION_URL = "https://varseek-server-3relpk35fa-wl.a.run.app"

varseek_ref_unallowable_arguments = {
    "varseek_build": {"return_variant_output"},
    "varseek_info": set(),
    "varseek_filter": set(),
    "kb_ref": set(),
}


# covers both varseek ref AND kb ref, but nothing else (i.e., all of the arguments that are not contained in varseek build, info, or filter)
def validate_input_ref(params_dict):
    dlist = params_dict.get("dlist", None)
    threads = params_dict.get("threads", None)
    index_out = params_dict.get("index_out", None)

    k = params_dict.get("k", None)
    w = params_dict.get("w", None)

    k, w = int(k), int(w)

    # make sure k is odd and <= 63
    if not isinstance(k, int) or k % 2 == 0 or k < 1 or k > 63:
        raise ValueError("k must be odd, positive, integer, and less than or equal to 63")

    if k < w + 1:
        raise ValueError("k must be greater than or equal to w + 1")
    
    if k > 2*w:
        raise ValueError("k must be less than or equal to 2*w")

    dlist_valid_values = {"genome", "transcriptome", "genome_and_transcriptome", "None", None}
    if dlist not in dlist_valid_values:
        raise ValueError(f"dlist must be one of {dlist_valid_values}")

    # sequences, variants, out handled by vk build

    check_file_path_is_string_with_valid_extension(index_out, "index_out", "index")

    if not is_valid_int(threads, threshold_type=">=", threshold_value=1):
        raise ValueError(f"Threads must be a positive integer, got {threads}")

    for param_name in ["minimum_info_columns", "download", "dry_run"]:
        if not isinstance(params_dict.get(param_name), bool):
            raise ValueError(f"{param_name} must be a boolean. Got {param_name} of type {type(params_dict.get(param_name))}.")

    variants = params_dict["variants"]
    sequences = params_dict["sequences"]
    # more on download

    if params_dict.get("download"):
        if variants not in prebuilt_vk_ref_files:
            raise ValueError(f"When downloading prebuilt reference, `variants` must be one of {prebuilt_vk_ref_files.keys()}. variants={variants} not recognized.")
        if sequences not in prebuilt_vk_ref_files[variants]:
            raise ValueError(f"When downloading prebuilt reference, `sequences` must be one of {prebuilt_vk_ref_files[variants].keys()}. sequences={sequences} not recognized.")

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


# a list of dictionaries with keys "variants", "sequences", and "description"
downloadable_references = [
    {"variants": "cosmic_cmc", "sequences": "cdna", "description": "COSMIC Cancer Mutation Census version 100 - Ensembl GRCh37 release 93 cDNA reference annotations. All default arguments of varseek ref (k=59, w=54, filters, no d-list, etc.). Header format (showing the column(s) from the original database used): 'seq_ID':'mutation_cdna'"},
    {"variants": "cosmic_cmc", "sequences": "genome", "description": "COSMIC Cancer Mutation Census version 100 - Ensembl GRCh37 release 93 genome reference annotations. All default arguments of varseek ref (k=59, w=54, filters, no d-list, etc.). Header format (showing the column(s) from the original database used): 'chromosome':'mutation_genome'"},
]


# don't worry if it says an argument is unused, as they will all get put in params_dict for each respective function and passed to the child functions
def ref(
    sequences,
    variants,
    w=54,
    k=59,
    filters=(
        "alignment_to_reference:is_not_true",
        # "substring_alignment_to_reference:is_not_true",  # filter out variants that are a substring of the reference genome  #* uncomment this and erase the line above when implementing d-list
        "pseudoaligned_to_reference_despite_not_truly_aligning:is_not_true",  # filter out variants that pseudoaligned to human genome despite not truly aligning
        "num_distinct_triplets:greater_than=2",  # filters out VCRSs with <= 2 unique triplets
    ),
    dlist=None,
    dlist_reference_source="T2T",
    var_column="mutation",
    seq_id_column="seq_ID",
    var_id_column=None,
    out=".",
    reference_out_dir=None,
    index_out=None,
    t2g_out=None,  # intentionally avoid having this name clash with the t2g from vk build and vk filter, as it could refer to either (depending on whether or not filtering will occur)
    download=False,
    dry_run=False,
    list_downloadable_references=False,
    minimum_info_columns=True,
    overwrite=False,
    threads=2,
    logging_level=None,
    save_logs=False,
    log_out_dir=None,
    verbose=False,
    **kwargs,  # * including all arguments for vk build, info, filter, and kb ref
):
    """
    Create a reference index and t2g file for variant screening with varseek count. Wraps around varseek build, varseek info, varseek filter, and kb ref.

    # Required input argument:
    - sequences     (str) Path to the fasta file containing the sequences to have the variants added, e.g., 'seqs.fa'.
                    Sequence identifiers following the '>' character must correspond to the identifiers
                    in the seq_ID column of 'variants'.

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

                    Alternatively, if 'variants' is a string specifying a supported database,
                    sequences can be a string indicating the source upon which to apply the variants.
                    See below for supported databases and sequences options.
                    To see the supported combinations of variants and sequences, either
                    1) run `vk build --list_supported_databases` from the command line, or
                    2) run varseek.build(list_supported_databases=True) in python

    - variants     (str or list[str] or DataFrame object) Path to csv or tsv file (str) (e.g., 'variants.csv'), or DataFrame (DataFrame object),
                    containing information about the variants in the following format:

                    | var_column         | var_id_column | seq_id_column |
                    | c.2C>T             | var1          | seq1          | -> Apply varation 1 to sequence 1
                    | c.9_13inv          | var2          | seq2          | -> Apply varation 2 to sequence 2
                    | c.9_13inv          | var2          | seq3          | -> Apply varation 2 to sequence 3
                    | c.9_13delinsAAT    | var3          | seq3          | -> Apply varation 3 to sequence 3
                    | ...                | ...           | ...           |

                    'var_column' = Column containing the variants to be performed written in standard mutation/variant annotation (see below)
                    'seq_id_column' = Column containing the identifiers of the sequences to be mutated (must correspond to the string following
                    the > character in the 'sequences' fasta file; do NOT include spaces or dots)
                    'var_id_column' = Column containing an identifier for each variant (optional).

                    Alternatively: Input variant(s) as a string or list, e.g., 'c.2C>T' or ['c.2C>T', 'c.1A>C'].
                    If a list is provided, the number of variants must equal the number of input sequences.

                    For more information on the standard mutation/variant annotation, see https://www.ncbi.nlm.nih.gov/pmc/articles/PMC1867422/.

                    Alternatively, 'variants' can be a string specifying a supported database, which will automatically download
                    both the variant database and corresponding reference sequence (if the 'sequences' is not a path).
                    To see the supported combinations of variants and sequences, either
                    1) run `vk build --list_supported_databases` from the command line, or
                    2) run varseek.build(list_supported_databases=True) in python

    # Additional parameters
    - w             (int) Length of sequence windows flanking the variant. Default: 30. If w > total length of the sequence, the entire sequence will be kept.
    - k             (int) The length of each k-mer in the kallisto reference index construction. Accordingly corresponds to the length of the k-mers to be considered in vk build's remove_seqs_with_wt_kmers, and the default minimum value for vk build's minimum sequence length (which can be changed with 'min_seq_len'). Must be greater than the value passed in for w. Default: 59.
    - filters       (str or list[str]) List of filters to apply to the variants. See varseek filter documentation for more information.
    - dlist         (str) Specifies whether ones wants to d-list against the genome, transcriptome, or both. Possible values are "genome", "transcriptome", "genome_and_transcriptome", or None. Default: None.
    - dlist_reference_source (str) Specifies whether to use the T2T, grch37, or grch38 reference genome during alignment of VCRS k-mers to the reference genome/transcriptome and any possible d-list construction. However, no d-list is used during the creation of the VCRS reference index unless `dlist` is not None. Default: "T2T".

    # Optional output file paths: (only needed if changing/customizing file names or locations):
    - out           (str) Output directory. Default: ".".
    - reference_out_dir  (str) Path to the directory where the reference files will be saved. Default: `out`/reference.
    - index_out (str) Path to output VCRS index file. Default: `out`/vcrs_index.idx.
    - t2g_out (str) Path to output VCRS t2g file to be used in alignment. Default: `out`/vcrs_t2g.txt.

    # General arguments:
    - download      (bool) If True, download prebuilt reference files. Default: False.
    - dry_run       (bool) If True, print the commands that would be run without actually running them. Default: False.
    - list_downloadable_references (bool) If True, list the available downloadable references. Default: False.
    - minimum_info_columns (bool) If True, run vk info with minimum columns. Default: True.
    - overwrite     (bool) If True, overwrite existing files. Default: False.
    - threads       (int) Number of threads to use. Default: 2.
    - logging_level (str) Logging level. Can also be set with the environment variable VARSEEK_LOGGING_LEVEL. Default: INFO.
    - save_logs     (True/False) Whether to save logs to a file. Default: False.
    - log_out_dir   (str) Directory to save logs. Default: None (do not save logs).
    - verbose       (True/False) Whether to print additional information e.g., progress bars. Default: False.

    For a complete list of supported parameters, see the documentation for varseek build, varseek info, varseek filter, and kb ref. Note that any shared parameter names between functions are meant to have identical purposes.
    """

    # * 0. Informational arguments that exit early
    if list_downloadable_references:  # for vk ref
        for downloadable_reference in downloadable_references:
            print(f"variants: {downloadable_reference['variants']}, sequences: {downloadable_reference['sequences']}, description: {downloadable_reference['description']}")
        return None

    if kwargs.get("list_supported_databases"):  # from vk build
        vk.varseek_build.print_valid_values_for_variants_and_sequences_in_varseek_build()
        return None
    if kwargs.get("list_columns"):  # from vk info
        vk.varseek_info.print_list_columns()
        return None
    if kwargs.get("list_d_list_values"):  # from vk info
        print(f"Available values for `dlist_reference_source`: {vk.varseek_info.supported_dlist_reference_values}")
        return None
    if kwargs.get("list_filter_rules"):  # from vk filter
        vk.varseek_filter.print_list_filter_rules()
        return None

    # * 1. Start timer
    start_time = time.perf_counter()

    # * 1.25 logger
    global logger
    if kwargs.get("logger") and isinstance(kwargs.get("logger"), logging.Logger):
        logger = kwargs.get("logger")
    else:
        if save_logs and not log_out_dir:
            log_out_dir = os.path.join(out, "logs")
        logger = set_up_logger(logger, logging_level=logging_level, save_logs=save_logs, log_dir=log_out_dir)
    kwargs["logger"] = logger

    # * 1.5. For the nargs="+" arguments, convert any list of length 1 to a string
    if isinstance(sequences, (list, tuple)) and len(sequences) == 1:
        sequences = sequences[0]
    if isinstance(variants, (list, tuple)) and len(variants) == 1:
        variants = variants[0]

    # * 2. Type-checking
    params_dict = make_function_parameter_to_value_dict(1)
    params_dict["out"], params_dict["input_dir"] = check_that_two_paths_are_the_same_if_both_provided_otherwise_set_them_equal(params_dict.get("out"), params_dict.get("input_dir"))  # because input_dir is a required argument, it does not have a default, and so I should enforce this default manually

    # Set params_dict to default values of children functions - important so type checking works properly
    ref_signature = inspect.signature(ref)
    for function in (vk.varseek_build.build, vk.varseek_info.info, vk.varseek_filter.filter):
        signature = inspect.signature(function)
        for key in signature.parameters.keys():
            if key not in params_dict and key not in ref_signature.parameters.keys():
                params_dict[key] = signature.parameters[key].default
    
    vk.varseek_build.validate_input_build(params_dict)  # this passes all vk ref parameters to the function - I could only pass in the vk build parameters here if desired (and likewise below), but there should be no naming conflicts anyways
    vk.varseek_info.validate_input_info(params_dict)
    vk.varseek_filter.validate_input_filter(params_dict)
    validate_input_ref(params_dict)

    # * 3. Dry-run
    # handled within child functions   

    # * 4. Save params to config file and run info file
    if not dry_run:
        # Save parameters to config file
        config_file = os.path.join(out, "config", "vk_ref_config.json")
        save_params_to_config_file(params_dict, config_file)  #$ Now I am done with params_dict 

        run_info_file = os.path.join(out, "config", "vk_ref_run_info.txt")
        save_run_info(run_info_file)

    # * 4.5. Pop out any unallowable arguments
    for key, unallowable_set in varseek_ref_unallowable_arguments.items():
        for unallowable_key in unallowable_set:
            kwargs.pop(unallowable_key, None)

    # * 4.8 Set kwargs to default values of children functions (not strictly necessary, as if these arguments are not in kwargs then it will use the default values anyways, but important if I need to rely on these default values within vk ref)
    # ref_signature = inspect.signature(ref)
    # for function in (vk.varseek_build.build, vk.varseek_info.info, vk.varseek_filter.filter):
    #     signature = inspect.signature(function)
    #     for key in signature.parameters.keys():
    #         if key not in kwargs and key not in ref_signature.parameters.keys():
    #             kwargs[key] = signature.parameters[key].default

    # * 5. Set up default folder/file input paths, and make sure the necessary ones exist
    # all input files for vk ref are required in the varseek workflow, so this is skipped

    # * 6. Set up default folder/file output paths, and make sure they don't exist unless overwrite=True    
    # Make directories
    os.makedirs(out, exist_ok=True)

    # define some more file paths
    if not index_out:
        index_out = os.path.join(out, "vcrs_index.idx")
    os.makedirs(os.path.dirname(index_out), exist_ok=True)

    vcrs_fasta_out = kwargs.get("vcrs_fasta_out", os.path.join(out, "vcrs.fa"))  # make sure this matches vk build
    vcrs_filtered_fasta_out = kwargs.get("vcrs_filtered_fasta_out", os.path.join(out, "vcrs_filtered.fa"))  # make sure this matches vk filter
    vcrs_t2g_out = kwargs.get("vcrs_t2g_out", os.path.join(out, "vcrs_t2g.txt"))  # make sure this matches vk build
    vcrs_t2g_filtered_out = kwargs.get("vcrs_t2g_filtered_out", os.path.join(out, "vcrs_t2g_filtered.txt"))  # make sure this matches vk filter
    dlist_genome_fasta_out = kwargs.get("dlist_genome_fasta_out", os.path.join(out, "dlist_genome.fa"))  # make sure this matches vk info
    dlist_cdna_fasta_out = kwargs.get("dlist_cdna_fasta_out", os.path.join(out, "dlist_cdna.fa"))  # make sure this matches vk info
    dlist_combined_fasta_out = kwargs.get("dlist_combined_fasta_out", os.path.join(out, "dlist.fa"))  # make sure this matches vk info

    for file in [index_out]:  # purposely exluding vcrs_fasta_out, vcrs_filtered_fasta_out, vcrs_t2g_out, vcrs_t2g_filtered_out because - let's say someone runs vk ref and they get an error write in the kb ref step because of a bad argument that doesn't affect the prior steps - it would be nice for someone to be able to rerun the command with the changed argument without having to rerun vk build, info, filter from scratch when overwrite=False - and if they do want to rerun those steps, they can just delete the files or set overwrite=True
        if os.path.isfile(file) and not overwrite:
            raise FileExistsError(f"Output file {file} already exists. Please delete it or specify a different output directory or set overwrite=True.")

    variants_updated_vk_info_csv_out = kwargs.get("variants_updated_vk_info_csv_out", None)
    if not variants_updated_vk_info_csv_out:
        variants_updated_vk_info_csv_out = os.path.join(out, "variants_updated_vk_info.csv")  # make sure this matches vk info

    file_signifying_successful_vk_build_completion = vcrs_fasta_out
    file_signifying_successful_vk_info_completion = variants_updated_vk_info_csv_out
    files_signifying_successful_vk_filter_completion = (vcrs_filtered_fasta_out, vcrs_t2g_filtered_out)
    file_signifying_successful_kb_ref_completion = index_out
    
    out, kwargs["input_dir"] = check_that_two_paths_are_the_same_if_both_provided_otherwise_set_them_equal(out, kwargs.get("input_dir"))  # check that, if out and input_dir are both provided, they are the same directory; otherwise, if only one is provided, then make them equal to each other
    kwargs["vcrs_fasta_out"], kwargs["vcrs_fasta"] = check_that_two_paths_are_the_same_if_both_provided_otherwise_set_them_equal(kwargs.get("vcrs_fasta_out"), kwargs.get("vcrs_fasta"))  # build --> info
    kwargs["id_to_header_csv_out"], kwargs["id_to_header_csv"] = check_that_two_paths_are_the_same_if_both_provided_otherwise_set_them_equal(kwargs.get("id_to_header_csv_out"), kwargs.get("id_to_header_csv"))  # build --> info/filter
    kwargs["variants_updated_csv_out"], kwargs["variants_updated_csv"] = check_that_two_paths_are_the_same_if_both_provided_otherwise_set_them_equal(kwargs.get("variants_updated_csv_out"), kwargs.get("variants_updated_csv"))  # build --> info
    kwargs["variants_updated_vk_info_csv_out"], kwargs["variants_updated_vk_info_csv"] = check_that_two_paths_are_the_same_if_both_provided_otherwise_set_them_equal(kwargs.get("variants_updated_vk_info_csv_out"), kwargs.get("variants_updated_vk_info_csv"))  # info --> filter
    kwargs["variants_updated_exploded_vk_info_csv_out"], kwargs["variants_updated_exploded_vk_info_csv"] = check_that_two_paths_are_the_same_if_both_provided_otherwise_set_them_equal(kwargs.get("variants_updated_exploded_vk_info_csv_out"), kwargs.get("variants_updated_exploded_vk_info_csv"))  # info --> filter
    # dlist handled below - see the comment "set d-list argument"

    # * 7. Define kwargs defaults
    # Nothing to see here

    # * 7.5. make sure ints are ints
    w, k, threads = int(w), int(k), int(threads)

    # * 8. Start the actual function
    # get COSMIC info
    cosmic_email = kwargs.get("cosmic_email", None)
    if cosmic_email:
        logger.info(f"Using COSMIC email from arguments: {cosmic_email}")
    else:
        cosmic_email = os.getenv("COSMIC_EMAIL")
        if cosmic_email:
            logger.info(f"Using COSMIC email from COSMIC_EMAIL environment variable: {cosmic_email}")
            kwargs["cosmic_email"] = cosmic_email
    
    cosmic_password = kwargs.get("cosmic_password", None)
    if cosmic_password:
        logger.info("Using COSMIC password from arguments")
    else:
        cosmic_password = os.getenv("COSMIC_PASSWORD")
        if cosmic_password:
            logger.info("Using COSMIC password from COSMIC_PASSWORD environment variable")
            kwargs["cosmic_password"] = cosmic_password

    # ensure that max_ambiguous (build) and max_ambiguous_vcrs (info) are the same if only one is provided
    if kwargs.get("max_ambiguous") and not kwargs.get("max_ambiguous_vcrs"):
        kwargs['max_ambiguous_vcrs'] = kwargs['max_ambiguous']
    if kwargs.get("max_ambiguous_vcrs") and not kwargs.get("max_ambiguous"):
        kwargs['max_ambiguous'] = kwargs['max_ambiguous_vcrs']

    if kwargs.get("columns_to_include") is not None:
        logger.info("columns_to_include is not None, so minimum_info_columns will be set to False")
        minimum_info_columns = False
    else:
        if minimum_info_columns:
            if isinstance(filters, str):
                columns_to_include = filters.split(":")[0]
            else:
                columns_to_include = tuple([item.split(":")[0] for item in filters])
        else:
            columns_to_include = ("number_of_variants_in_this_gene_total", "alignment_to_reference", "pseudoaligned_to_reference_despite_not_truly_aligning", "triplet_complexity")  #!! matches vk info default
        kwargs["columns_to_include"] = columns_to_include


    # decide whether to skip vk info and vk filter
    # filters_column_names = list({filter.split('-')[0] for filter in filters})
    skip_filter = not bool(filters)  # skip filtering if no filters provided
    skip_info = minimum_info_columns and skip_filter  # skip vk info if no filtering will be performed and one specifies minimum info columns

    if skip_filter:
        vcrs_fasta_for_index = vcrs_fasta_out
        if t2g_out:
            kwargs["vcrs_t2g_out"] = t2g_out  # pass this custom path into vk build
            vcrs_t2g_out = t2g_out  # pass this custom path into the output dict of vk ref
        vcrs_t2g_for_alignment = vcrs_t2g_out
        if kwargs.get("use_IDs") is None:  # if someone has a strong preference, then who am I to tell them otherwise - but otherwise, I will want to override the default to False for vk build
            kwargs["use_IDs"] = False
    else:
        vcrs_fasta_for_index = vcrs_filtered_fasta_out
        if t2g_out:
            kwargs["vcrs_t2g_filtered_out"] = t2g_out
            vcrs_t2g_filtered_out = t2g_out
        vcrs_t2g_for_alignment = vcrs_t2g_filtered_out
        # don't touch use_IDs - if not provided, then will resort to defaults (True for vk build, False for vk filter); if provided, then will be passed into vk build and vk filter

    # download if download argument is True
    if download:
        # $ I opt to keep it like this rather than converting the keys of prebuilt_vk_ref_files to a tuple of many arguments for user simplicity - simply document the uploaded references, but no need to differentiate - but if I do end up having multiple reference documents with the same values for variants and sequences, then switch over to this approach where the dict keys are tuples
        file_dict = prebuilt_vk_ref_files.get(variants, {}).get(sequences, {})
        if file_dict:
            if file_dict["index"] == "COSMIC":
                response = requests.post(COSMIC_CREDENTIAL_VALIDATION_URL, json={"email": cosmic_email, "password": cosmic_password, "variants": variants, "sequences": sequences})
                if response.status_code == 200:
                    file_dict = response.json()  # Converts JSON to dict
                    file_dict = file_dict.get("download_links")
                    logger.info("Successfully verified COSMIC credentials.")
                    logger.warning("According to COSMIC regulations, please do not share any data that utilizes the COSMIC database. See more here: https://cancer.sanger.ac.uk/cosmic/help/terms")
                else:
                    raise ValueError(f"Failed to verify COSMIC credentials. Status code: {response.status_code}")
            logger.info(f"Downloading reference files with variants={variants}, sequences={sequences}")
            vk_ref_output_dict = download_varseek_files(file_dict, out=out)  # TODO: replace with DOI download (will need to replace prebuilt_vk_ref_files urls with DOIs) - ensure if this is allowed with COSMIC
            if index_out and vk_ref_output_dict["index"] != index_out:
                os.rename(vk_ref_output_dict["index"], index_out)
                vk_ref_output_dict["index"] = index_out
            if t2g_out and vk_ref_output_dict["t2g"] != t2g_out:
                os.rename(vk_ref_output_dict["t2g"], t2g_out)
                vk_ref_output_dict["t2g"] = t2g_out

            return vk_ref_output_dict
        else:
            raise ValueError(f"No prebuilt files found for the given arguments:\nvariants: {variants}\nsequences: {sequences}")

    # set d-list argument
    if dlist == "genome":
        dlist_kb_argument = dlist_genome_fasta_out  # for kb ref
        kwargs["dlist_fasta"] = dlist_genome_fasta_out  # for vk filter
    elif dlist == "transcriptome":
        dlist_kb_argument = dlist_cdna_fasta_out
        kwargs["dlist_fasta"] = dlist_cdna_fasta_out
    elif dlist == "genome_and_transcriptome":
        dlist_kb_argument = dlist_combined_fasta_out
        kwargs["dlist_fasta"] = dlist_combined_fasta_out
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

    #* vk build
    if not os.path.exists(file_signifying_successful_vk_build_completion) or overwrite:  # the reason I do it like this, rather than if overwrite or not os.path.exists(MYPATH), is because I would like vk ref/count to automatically overwrite partially-completed function outputs even when overwrite=False; but when overwrite=True, then run from scratch regardless
        kwargs_vk_build = {key: value for key, value in kwargs.items() if ((key in all_parameter_names_set_vk_build) and (key not in ref_signature.parameters.keys()))}
        # update anything in kwargs_vk_build that is not fully updated in (vk ref's) kwargs (should be nothing or very close to it, as I try to avoid these double-assignments by always keeping kwargs in kwargs)
        # eg kwargs_vk_build['mykwarg'] = mykwarg
        # just to be extra clear, I must explicitly pass arguments that are in the signature of vk ref; anything not in vk ref's signature should go in kwargs_vk_build (it is irrelevant what is in vk build's signature); and in the line above, I should update any values that are (1) not in vk ref's signature (so therefore they're in vk ref's kwargs), (2) I want to pass to vk build, and (3) have been updated outside of vk ref's kwargs somewhere in the function

        save_column_names_json_path=f"{out}/column_names_tmp.json"

        logger.info("Running vk build")
        _ = vk.build(
                sequences=sequences,
                variants=variants,
                seq_id_column=seq_id_column,
                var_column=var_column,
                var_id_column=var_id_column,
                w=w,
                k=k,
                out=out,
                reference_out_dir=reference_out_dir,
                dry_run=dry_run,
                overwrite=True,  # overwrite=True rather than overwrite=overwrite because I only enter this condition if the file signifying success does not exist and/or overwrite is True anyways - this allows me to overwrite half-completed functions
                logging_level=logging_level,
                save_logs=save_logs,
                log_out_dir=log_out_dir,
                verbose=verbose,
                save_column_names_json_path=save_column_names_json_path,  # saves the temp json
                **kwargs_vk_build
        )

        # use values for columns and file paths as provided in vk build
        if os.path.exists(save_column_names_json_path):  # will only exist if variants in supported_databases_and_corresponding_reference_sequence_type
            with open(save_column_names_json_path, "r") as f:
                column_names_and_file_names_dict = json.load(f)
            os.remove(save_column_names_json_path)
            if column_names_and_file_names_dict['seq_id_column']:
                seq_id_column = column_names_and_file_names_dict['seq_id_column']
            if column_names_and_file_names_dict['var_column']:
                var_column = column_names_and_file_names_dict['var_column']
            if column_names_and_file_names_dict['var_id_column']:
                var_id_column = column_names_and_file_names_dict['var_id_column']
            for column in ("seq_id_genome_column", "var_genome_column", "seq_id_cdna_column", "var_cdna_column", "gene_name_column"):
                kwargs[column] = kwargs.get(column, supported_databases_and_corresponding_reference_sequence_type[variants]["column_names"][column])
            for file in ("gtf", "reference_genome_fasta", "reference_cdna_fasta"):
                if column_names_and_file_names_dict[file]:
                    kwargs[file] = kwargs.get(file, column_names_and_file_names_dict[file])

    else:
        logger.warning(f"Skipping vk build because {file_signifying_successful_vk_build_completion} already exists and overwrite=False")

    #* vk info
    if not skip_info:
        if kwargs.get("use_IDs", None) is False:
            logger.warning("use_IDs=False is not recommended for vk info, as the headers output by vk build can break some programs that read fasta files due to the inclusion of '>' symbols in substitutions and the potentially long length of the headers (with multiple combined headers and/or long insertions). Consider setting use_IDs=True (use IDs throughout the workflow) or leaving this parameter blank (will use IDs in vk build so that vk info runs properly [unless vk info/filter will not be run, in which case it will use headers], and will use headers in vk filter so that the output is more readable).")
        if not os.path.exists(file_signifying_successful_vk_info_completion) or overwrite:
            kwargs_vk_info = {key: value for key, value in kwargs.items() if ((key in all_parameter_names_set_vk_info) and (key not in ref_signature.parameters.keys()))}
            # update anything in kwargs_vk_info that is not fully updated in (vk ref's) kwargs (should be nothing or very close to it, as I try to avoid these double-assignments by always keeping kwargs in kwargs)
            # eg kwargs_vk_info['mykwarg'] = mykwarg
            
            logger.info("Running vk info")
            _ = vk.info(
                k=k,
                dlist_reference_source=dlist_reference_source,
                seq_id_column=seq_id_column,
                var_column=var_column,
                out=out,
                reference_out_dir=reference_out_dir,
                dry_run=dry_run,
                overwrite=True,  # overwrite=True rather than overwrite=overwrite because I only enter this condition if the file signifying success does not exist and/or overwrite is True anyways - this allows me to overwrite half-completed functions
                threads=threads,
                logging_level=logging_level,
                save_logs=save_logs,
                log_out_dir=log_out_dir,
                verbose=verbose,
                variants=variants,  # a kwargs of vk info but explicit in vk ref
                w=w,  # a kwargs of vk info but explicit in vk ref
                **kwargs_vk_info  # including input_dir
        )
        else:
            logger.warning(f"Skipping vk info because {file_signifying_successful_vk_info_completion} already exists and overwrite=False")

    # vk filter
    if not skip_filter:
        if not all(os.path.exists(f) for f in files_signifying_successful_vk_filter_completion) or overwrite:
            kwargs_vk_filter = {key: value for key, value in kwargs.items() if ((key in all_parameter_names_set_vk_filter) and (key not in ref_signature.parameters.keys()))}
            # update anything in kwargs_vk_filter that is not fully updated in (vk ref's) kwargs (should be nothing or very close to it, as I try to avoid these double-assignments by always keeping kwargs in kwargs)
            # eg kwargs_vk_filter['mykwarg'] = mykwarg
            
            logger.info("Running vk filter")
            _ = vk.filter(
                filters=filters,
                out=out,
                dry_run=dry_run,
                overwrite=True,  # overwrite=True rather than overwrite=overwrite because I only enter this condition if the file signifying success does not exist and/or overwrite is True anyways - this allows me to overwrite half-completed functions
                logging_level=logging_level,
                save_logs=save_logs,
                log_out_dir=log_out_dir,
                **kwargs_vk_filter
        )
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
        "--overwrite",  # set overwrite here regardless of the overwrite argument because I would only even enter this block if kb count was only partially run (as seen by the lack of existing of file_signifying_successful_kb_ref_completion), in which case I should overwrite anyways
    ]

    # assumes any argument in varseek ref matches kb ref identically, except dashes replaced with underscores
    params_dict_kb_ref = make_function_parameter_to_value_dict(1)  # will reflect any updated values to variables found in vk ref signature and anything in kwargs
    for dict_key, arguments in varseek_ref_only_allowable_kb_ref_arguments.items():
        for argument in list(arguments):
            dash_count = len(argument) - len(argument.lstrip("-"))
            leading_dashes = "-" * dash_count
            argument = argument.lstrip("-").replace("-", "_")
            if argument in params_dict_kb_ref:
                value = params_dict_kb_ref[argument]
                if dict_key == "zero_arguments":
                    if value:  # only add if value is True
                        kb_ref_command.append(f"{leading_dashes}{argument}")
                elif dict_key == "one_argument":
                    kb_ref_command.extend([f"{leading_dashes}{argument}", value])
                else:  # multiple_arguments or something else
                    pass

    kb_ref_command.append(vcrs_fasta_for_index)

    if not os.path.exists(file_signifying_successful_kb_ref_completion) or overwrite:
        if dry_run:
            print(" ".join(kb_ref_command))
        else:
            logger.info(f"Running kb ref with command: {' '.join(kb_ref_command)}")
            subprocess.run(kb_ref_command, check=True)
    else:
        logger.warning(f"Skipping kb ref because {file_signifying_successful_kb_ref_completion} already exists and overwrite=False")

    #!!! erase if removing wt vcrs feature
    wt_vcrs_fasta_out = kwargs.get("wt_vcrs_fasta_out", os.path.join(out, "wt_vcrs.fa"))  # make sure this matches vk build
    wt_vcrs_filtered_fasta_out = kwargs.get("wt_vcrs_filtered_fasta_out", os.path.join(out, "wt_vcrs_filtered.fa"))  # make sure this matches vk filter
    vcrs_wt_fasta_for_index = wt_vcrs_fasta_out if skip_filter else wt_vcrs_filtered_fasta_out
    wt_vcrs_index_out = kwargs.get("wt_vcrs_index_out", os.path.join(out, "wt_vcrs_index.idx"))
    file_signifying_successful_wt_vcrs_kb_ref_completion = wt_vcrs_index_out

    if os.path.exists(vcrs_wt_fasta_for_index):
        if (not os.path.exists(file_signifying_successful_wt_vcrs_kb_ref_completion) or overwrite):
            kb_ref_wt_vcrs_command = ["kb", "ref", "--workflow", "custom", "-t", str(threads), "-i", wt_vcrs_index_out, "--d-list", "None", "-k", str(k), "--overwrite", True, vcrs_wt_fasta_for_index]  # set to True here regardless of the overwrite argument because I would only even enter this block if kb count was only partially run (as seen by the lack of existing of file_signifying_successful_wt_vcrs_kb_ref_completion), in which case I should overwrite anyways
            if dry_run:
                print(" ".join(kb_ref_wt_vcrs_command))
            else:
                logger.info(f"Running kb ref for wt vcrs index with command: {' '.join(kb_ref_wt_vcrs_command)}")
                subprocess.run(kb_ref_wt_vcrs_command, check=True)
        else:
            logger.warning(f"Skipping kb ref for wt vcrs because {file_signifying_successful_wt_vcrs_kb_ref_completion} already exists and overwrite=False")
    #!!! erase if removing wt vcrs feature

    if dry_run:
        index_out = None
        vcrs_t2g_for_alignment = None

    vk_ref_output_dict = {}
    vk_ref_output_dict["index"] = os.path.abspath(index_out)
    vk_ref_output_dict["t2g"] = os.path.abspath(vcrs_t2g_for_alignment)

    # Report time
    if not dry_run:
        report_time_elapsed(start_time, logger=logger, function_name="ref")

    return vk_ref_output_dict

