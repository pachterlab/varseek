"""varseek ref and specific helper functions."""

import getpass
import inspect
import json
import logging
import os
import subprocess
import time
import pyfastx

import requests

import varseek as vk
from varseek.utils import (
    check_file_path_is_string_with_valid_extension,
    check_that_two_paths_are_the_same_if_both_provided_otherwise_set_them_equal,
    download_varseek_files,
    authenticate_cosmic_credentials,
    get_python_or_cli_function_call,
    is_valid_int,
    make_function_parameter_to_value_dict,
    report_time_elapsed,
    save_params_to_config_file,
    save_run_info,
    set_up_logger,
    set_varseek_logging_level_and_filehandler
)

from .constants import (
    prebuilt_vk_ref_files,
    supported_databases_and_corresponding_reference_sequence_type,
    varseek_ref_only_allowable_kb_ref_arguments,
)

logger = logging.getLogger(__name__)
logger = set_up_logger(logger, logging_level="INFO", save_logs=False, log_dir=None)
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

    if k > 2 * w:
        raise ValueError("k must be less than or equal to 2*w")

    dlist_valid_values = {"genome", "transcriptome", "genome_and_transcriptome", "None", None}
    if dlist not in dlist_valid_values:
        raise ValueError(f"dlist must be one of {dlist_valid_values}")

    # sequences, variants, out handled by vk build

    check_file_path_is_string_with_valid_extension(index_out, "index_out", "index")

    if not is_valid_int(threads, threshold_type=">=", threshold_value=1):
        raise ValueError(f"Threads must be a positive integer, got {threads}")

    for param_name in ["download", "dry_run"]:
        if not isinstance(params_dict.get(param_name), bool):
            raise ValueError(f"{param_name} must be a boolean. Got {param_name} of type {type(params_dict.get(param_name))}.")

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
    {"description": "COSMIC Cancer Mutation Census version 101 - Ensembl GRCh37 release 93 cDNA reference annotations. w=47, k=51, dlist_reference_source=t2t. Header format (showing the column(s) from the original database used): 'seq_ID':'mutation_cdna'.", "download_command": "vk ref -v cosmic_cmc -s cdna -d"},
    {"description": "Geuvadis dataset - Ensembl GRCh37 release 113 cDNA reference annotations. w=37, k=41, dlist_reference_source=t2t. Header format (showing the column(s) from the original database used): 'seq_ID':'mutation_cdna'.", "download_command": "vk ref -v geuvadis -s cdna -d"},
    # {"variants": "cosmic_cmc", "sequences": "genome", "description": "COSMIC Cancer Mutation Census version 101 - Ensembl GRCh37 release 93 genome reference annotations. w=47,k=51. Header format (showing the column(s) from the original database used): 'chromosome':'mutation_genome'"},
]


# don't worry if it says an argument is unused, as they will all get put in params_dict for each respective function and passed to the child functions
@report_time_elapsed
def ref(
    variants,
    sequences,
    w=47,
    k=51,
    filters=(
        "alignment_to_reference:is_not_true",
        # "substring_alignment_to_reference:is_not_true",  # filter out variants that are a substring of the reference genome  #* uncomment this and erase the line above when implementing d-list
        "pseudoaligned_to_reference_despite_not_truly_aligning:is_not_true",  # filter out variants that pseudoaligned to human genome despite not truly aligning
        "num_distinct_triplets:greater_than=5",  # filters out VCRSs in with 5 or fewer distinct triplets
        "longest_homopolymer_length:less_or_equal=10",  # filters out VCRSs with a homopolymer length greater than 10 bp
    ),
    dlist=None,
    dlist_reference_source=None,
    dlist_reference_ensembl_release=111,
    var_column="mutation",
    seq_id_column="seq_ID",
    var_id_column=None,
    out=".",
    reference_out_dir=None,
    index_out=None,
    t2g_out=None,  # intentionally avoid having this name clash with the t2g from vk build and vk filter, as it could refer to either (depending on whether or not filtering will occur)
    fasta_out=None,  # intentionally avoid having this name clash with the fasta from vk build and vk filter, as it could refer to either (depending on whether or not filtering will occur)
    download=False,
    chunksize=None,
    dry_run=False,
    list_downloadable_references=False,
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
    - variants     (str or list[str] or DataFrame object) Variants to apply to the sequences. Input formats options include the following:
                    1) Single variant (str), along with a single sequence for `sequences` (str). E.g., variants='c.2G>T' and sequences='AGCTAGCT'.
                    2) List of variants (list[str]), along with a list of sequences for `sequences` (list[str]). E.g., variants=['c.2G>T', 'c.1A>C'] and sequences=['AGCTAGCT', 'AGCTAGCT'].
                    NOTE: The number of variants must equal the number of input sequences.
                    3) Path to CSV/TSV file (str) (e.g., 'variants.csv') or DataFrame (DataFrame object), along with a fasta file for `sequences`.
                    NOTE: The `sequences` reference genome assembly (e.g., GRCh37 vs. GRCh38), source (e.g., genome vs. cDNA vs. CDS), and release (if source is cDNA or CDS, e.g., Ensembl release 111) must match the source used to annotate the variants.
                    The CSV/TSV/DataFrame must be structured in the following way:

                    | var_column         | seq_id_column | var_id_column |
                    | c.2C>T             | seq1          | var1          | -> Apply varation 1 to sequence 1
                    | c.9_13inv          | seq2          | var2          | -> Apply varation 2 to sequence 2
                    | c.9_13inv          | seq3          | var2          | -> Apply varation 2 to sequence 3
                    | c.9_13delinsAAT    | seq3          | var3          | -> Apply varation 3 to sequence 3
                    | ...                | ...           | ...           |

                    'var_column' = Column containing the variants to be performed written in standard mutation/variant annotation matching HGVS variant format (see below).
                    'seq_id_column' = Column containing the identifiers of the sequences to be mutated (must correspond to the string following the > character in the 'sequences' fasta file; do NOT include spaces or dots).
                    'var_id_column' = Column containing an identifier for each variant (optional).

                    For more information on the standard mutation/variant annotation, see https://www.ncbi.nlm.nih.gov/pmc/articles/PMC1867422/.

                    4) Path to VCF file (str) (e.g., 'variants.vcf'), along with a fasta file for `sequences`.
                    NOTE: The `sequences` reference genome assembly (e.g., GRCh37 vs. GRCh38) and release (if source is cDNA or CDS, e.g., Ensembl release 111) must match the source used to annotate the variants.
                    NOTE: For VCF input, the reference source is always the genome (i.e., never the cDNA or CDS). The arguments `var_column` and `seq_id_column` are not needed for VCF input (will be automatically set).
                    The `var_id_column` ID column can be provided if wanting to use the value from the ID column in the VCF as the variant ID instead of the default HGVS ID.
                    5) A value supported internally by vk ref (str), along with a value internally supported by vk ref corresponding to this variants value (str). See vk ref --list_internally_supported_indices for more information.

    - sequences     (str) Sequences to which to apply the variants from `variants`. See the 'variants' argument for more information on the input formats for `sequences` and their corresponding `variants` formats.
                    NOTE: Only the letters until the first space or dot will be used as sequence identifiers
                    NOTE: When 'sequences' input is a genome, also see the arguments `gtf`, `gtf_transcript_id_column`, and `transcript_boundaries` in varseek build.

    # Additional parameters
    - w             (int) Length of sequence windows flanking the variant. Default: 47. If w > total length of the sequence, the entire sequence will be kept.
    - k             (int) The length of each k-mer in the kallisto reference index construction. Accordingly corresponds to the length of the k-mers to be considered in vk build's remove_seqs_with_wt_kmers, and the default minimum value for vk build's minimum sequence length (which can be changed with 'min_seq_len'). Must be greater than the value passed in for w. Default: 51.
    - filters       (str or list[str]) Filter or list of filters to apply to the variant reference fasta. Each filter should be in the format COLUMN-RULE=VALUE or COLUMN-RULE (for boolean evaluation). For details, run vk filter --list_filter_rules, or see the documentation at https://github.com/pachterlab/varseek/blob/main/docs/filter.md
    - dlist         (str) Specifies whether ones wants to d-list against the genome, transcriptome, or both. Possible values are "genome", "transcriptome", "genome_and_transcriptome", or None. Default: None.
    - dlist_reference_source (str or None) Specifies which reference to use during alignment of VCRS k-mers to the reference genome/transcriptome and any possible d-list construction. However, no d-list is used during the creation of the VCRS reference index unless `dlist` is not None. This can refer to the same genome version as used by the "sequences" argument, but need not be. The purpose of this genome is simply to provide an accurate and comprehensive reference genome/transcriptome to determine which k-mers from the VCRSs overlap with the reference. Will look for files in `reference_out_dir`, and will download in this directory if necessary files do not exist. Ignored if values for `dlist_reference_genome_fasta`, `dlist_reference_cdna_fasta`, and `dlist_reference_gtf` are provided. Default: None. Possible values:
        - "t2t" - Telomere to telomere: https://www.ncbi.nlm.nih.gov/datasets/genome/GCF_009914755.1/. Directory: {reference_out_dir}/t2t.
        - "grch38" - Ensembl GRCh38: https://useast.ensembl.org/Homo_sapiens/Info/Annotation. Directory: {reference_out_dir}/ensembl_grch38_release{dlist_reference_ensembl_release}.
        - "grch37" - Ensembl GRCh37: http://useast.ensembl.org/info/website/tutorials/grch37.html. Directory: {reference_out_dir}/ensembl_grch37_release{dlist_reference_ensembl_release}.
        If wanting to provide a reference genome outside of those above supported for automatic download, then please provide existing file paths for the parameters `dlist_reference_genome_fasta`, `dlist_reference_cdna_fasta`, and/or `dlist_reference_gtf`.
    - dlist_reference_ensembl_release    (int) Ensembl release number for the d-list reference genome and transcriptome if dlist_reference_source in {"grch37", "grch38"}. Default: 111.
    - var_column                         (str) Name of the column containing the variants to be introduced in 'variants'. Important for CSV/TSV/DataFrame input with pre-defined columns. Default: 'mutation'.
    - seq_id_column                      (str) Name of the column containing the IDs of the sequences to be mutated in 'variants'. Important for CSV/TSV/DataFrame input with pre-defined columns. Default: 'seq_ID'.
    - var_id_column                      (str) Name of the column containing the IDs of each variant in 'variants'. Optional. Default: use <seq_id_column>_<var_column> for each row.

    # Optional output file paths: (only needed if changing/customizing file names or locations):
    - out           (str) Output directory. Default: ".".
    - reference_out_dir  (str) Path to the directory where the reference files will be saved. Default: `out`/reference.
    - index_out (str) Path to output VCRS index file. Default: `out`/vcrs_index.idx.
    - t2g_out (str) Path to output VCRS t2g file to be used in alignment. Default: None (default file names in vk build/info, as well as for download).
    - fasta_out (str) Path to output VCRS fasta file. Default: Default: None (default file names in vk build/info, as well as for download).

    # General arguments:
    - download      (bool) If True, download prebuilt reference files (index, t2g, vcrs fasta) into `out` (or paths specified by index_out, t2g_out, and fasta_out, respectively). Default: False.
    - chunksize     (int) Number of variants to process at a time. If None, then all variants will be processed at once. Default: None.
    - dry_run       (bool) If True, print the commands that would be run without actually running them. Default: False.
    - list_downloadable_references (bool) If True, list the available downloadable references. Default: False.
    - overwrite     (bool) If True, overwrite existing files. Default: False.
    - threads       (int) Number of threads to use. Default: 2.
    - logging_level (str) Logging level. Can also be set with the environment variable VARSEEK_LOGGING_LEVEL. Default: INFO.
    - save_logs     (True/False) Whether to save logs to a file. Default: False.
    - log_out_dir   (str) Directory to save logs. Default: None (do not save logs).
    - verbose       (True/False) Whether to print additional information e.g., progress bars. Default: False.

    # kwargs
    - minimum_info_columns (bool) If True, run vk info with minimum columns. Default: True.

    For a complete list of supported parameters, see the documentation for varseek build, varseek info, varseek filter, and kb ref. Note that any shared parameter names between functions are meant to have identical purposes.
    """

    # * 0. Informational arguments that exit early
    if list_downloadable_references:  # for vk ref
        print("All varseek ref arguments are defaults unless otherwise specified.\n")
        for downloadable_reference in downloadable_references:
            print(f"Description: {downloadable_reference['description']}\nDownload command: {downloadable_reference['download_command']}\n")
        return None

    if kwargs.get("list_internally_supported_indices"):  # from vk build
        vk.varseek_build.print_valid_values_for_variants_and_sequences_in_varseek_build()
        return None
    if kwargs.get("list_columns"):  # from vk info
        vk.varseek_info.print_list_columns()
        return None
    if kwargs.get("list_filter_rules"):  # from vk filter
        vk.varseek_filter.print_list_filter_rules()
        return None

    # * 1. logger
    if save_logs and not log_out_dir:
        log_out_dir = os.path.join(out, "logs")
    set_varseek_logging_level_and_filehandler(logging_level=logging_level, save_logs=save_logs, log_dir=log_out_dir)


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

    if not download:  # skip when downloading, as I don't need all of these functions to run normally
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
        save_params_to_config_file(params_dict, config_file)  # $ Now I am done with params_dict

        run_info_file = os.path.join(out, "config", "vk_ref_run_info.txt")
        save_run_info(run_info_file, params_dict=params_dict, function_name="ref")

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

    # ensure equivalent parameter file paths agree
    out, kwargs["input_dir"] = check_that_two_paths_are_the_same_if_both_provided_otherwise_set_them_equal(out, kwargs.get("input_dir"))  # check that, if out and input_dir are both provided, they are the same directory; otherwise, if only one is provided, then make them equal to each other
    kwargs["vcrs_fasta_out"], kwargs["vcrs_fasta"] = check_that_two_paths_are_the_same_if_both_provided_otherwise_set_them_equal(kwargs.get("vcrs_fasta_out"), kwargs.get("vcrs_fasta"))  # build --> info
    kwargs["id_to_header_csv_out"], kwargs["id_to_header_csv"] = check_that_two_paths_are_the_same_if_both_provided_otherwise_set_them_equal(kwargs.get("id_to_header_csv_out"), kwargs.get("id_to_header_csv"))  # build --> info/filter
    kwargs["variants_updated_csv_out"], kwargs["variants_updated_csv"] = check_that_two_paths_are_the_same_if_both_provided_otherwise_set_them_equal(kwargs.get("variants_updated_csv_out"), kwargs.get("variants_updated_csv"))  # build --> info
    kwargs["variants_updated_vk_info_csv_out"], kwargs["variants_updated_vk_info_csv"] = check_that_two_paths_are_the_same_if_both_provided_otherwise_set_them_equal(kwargs.get("variants_updated_vk_info_csv_out"), kwargs.get("variants_updated_vk_info_csv"))  # info --> filter
    kwargs["variants_updated_exploded_vk_info_csv_out"], kwargs["variants_updated_exploded_vk_info_csv"] = check_that_two_paths_are_the_same_if_both_provided_otherwise_set_them_equal(kwargs.get("variants_updated_exploded_vk_info_csv_out"), kwargs.get("variants_updated_exploded_vk_info_csv"))  # info --> filter
    # dlist handled below - see the comment "set d-list argument"

    # define some more file paths
    if not index_out:
        index_out = os.path.join(out, "vcrs_index.idx")
    os.makedirs(os.path.dirname(index_out), exist_ok=True)

    vcrs_fasta_out = kwargs.get("vcrs_fasta_out") if kwargs.get("vcrs_fasta_out") else os.path.join(out, "vcrs.fa")  # make sure this matches vk build - I use this if else rather than simply the kwargs.get as below, because if someone provides the path None here then I don't want an error
    vcrs_filtered_fasta_out = kwargs.get("vcrs_filtered_fasta_out") if kwargs.get("vcrs_filtered_fasta_out") else os.path.join(out, "vcrs_filtered.fa")  # make sure this matches vk filter
    vcrs_t2g_out = kwargs.get("vcrs_t2g_out") if kwargs.get("vcrs_t2g_out") else os.path.join(out, "vcrs_t2g.txt")  # make sure this matches vk build
    vcrs_t2g_filtered_out = kwargs.get("vcrs_t2g_filtered_out") if kwargs.get("vcrs_t2g_filtered_out") else os.path.join(out, "vcrs_t2g_filtered.txt")  # make sure this matches vk filter
    variants_updated_vk_info_csv_out = kwargs.get("variants_updated_vk_info_csv_out") if kwargs.get("variants_updated_vk_info_csv_out") else os.path.join(out, "variants_updated_vk_info.csv")  # make sure this matches vk info
    dlist_genome_fasta_out = kwargs.get("dlist_genome_fasta_out", os.path.join(out, "dlist_genome.fa"))  # make sure this matches vk info
    dlist_cdna_fasta_out = kwargs.get("dlist_cdna_fasta_out", os.path.join(out, "dlist_cdna.fa"))  # make sure this matches vk info
    dlist_combined_fasta_out = kwargs.get("dlist_combined_fasta_out", os.path.join(out, "dlist.fa"))  # make sure this matches vk info

    for file in [index_out]:  # purposely exluding vcrs_fasta_out, vcrs_filtered_fasta_out, vcrs_t2g_out, vcrs_t2g_filtered_out because - let's say someone runs vk ref and they get an error write in the kb ref step because of a bad argument that doesn't affect the prior steps - it would be nice for someone to be able to rerun the command with the changed argument without having to rerun vk build, info, filter from scratch when overwrite=False - and if they do want to rerun those steps, they can just delete the files or set overwrite=True
        if os.path.isfile(file) and not overwrite:
            raise FileExistsError(f"Output file {file} already exists. Please delete it or specify a different output directory or set overwrite=True.")

    file_signifying_successful_vk_build_completion = vcrs_fasta_out
    file_signifying_successful_vk_info_completion = variants_updated_vk_info_csv_out
    files_signifying_successful_vk_filter_completion = (vcrs_filtered_fasta_out, vcrs_t2g_filtered_out)
    file_signifying_successful_kb_ref_completion = index_out

    # * 7. Define kwargs defaults
    minimum_info_columns = kwargs.get("minimum_info_columns", True)

    # * 7.5. make sure ints are ints
    w, k, threads = int(w), int(k), int(threads)

    # * 8. Start the actual function
    # ensure that max_ambiguous (build) and max_ambiguous_vcrs (info) are the same if only one is provided
    if kwargs.get("max_ambiguous") and not kwargs.get("max_ambiguous_vcrs"):
        kwargs["max_ambiguous_vcrs"] = kwargs["max_ambiguous"]
    if kwargs.get("max_ambiguous_vcrs") and not kwargs.get("max_ambiguous"):
        kwargs["max_ambiguous"] = kwargs["max_ambiguous_vcrs"]

    cosmic_email = kwargs.get("cosmic_email", None)
    if not cosmic_email and os.getenv("COSMIC_EMAIL"):
        logger.info(f"Using COSMIC email from COSMIC_EMAIL environment variable")
        cosmic_email = os.getenv("COSMIC_EMAIL")
    
    cosmic_password = kwargs.get("cosmic_password", None)
    if not cosmic_password and os.getenv("COSMIC_PASSWORD"):
        logger.info("Using COSMIC password from COSMIC_PASSWORD environment variable")
        cosmic_password = os.getenv("COSMIC_PASSWORD")

    if isinstance(variants, str) and ".vcf" in variants.lower():
        var_id_column = None

    if kwargs.get("columns_to_include") is not None:
        logger.info("columns_to_include is not None, so minimum_info_columns will be set to False")
        minimum_info_columns = False
    else:
        if minimum_info_columns:  # just use what I need from filter
            if isinstance(filters, str):
                columns_to_include = filters.split(":")[0]
            else:
                columns_to_include = tuple([item.split(":")[0] for item in filters])
            kwargs["columns_to_include"] = columns_to_include
        else:  # use the default from vk info - make sure kwargs has no value for columns_to_include so that nothing gets passed in to vk info
            if "columns_to_include" in kwargs:
                del kwargs["columns_to_include"]

    # decide whether to skip vk info and vk filter
    # filters_column_names = list({filter.split('-')[0] for filter in filters})
    skip_filter = not bool(filters)  # skip filtering if no filters provided
    skip_info = minimum_info_columns and skip_filter  # skip vk info if no filtering will be performed and one specifies minimum info columns

    if skip_filter:
        if fasta_out:
            kwargs["vcrs_fasta_out"] = fasta_out
            vcrs_fasta_out = fasta_out
        vcrs_fasta_for_index = vcrs_fasta_out
        if t2g_out:
            kwargs["vcrs_t2g_out"] = t2g_out  # pass this custom path into vk build
            vcrs_t2g_out = t2g_out  # pass this custom path into the output dict of vk ref
        vcrs_t2g_for_alignment = vcrs_t2g_out
        if kwargs.get("use_IDs") is None:  # if someone has a strong preference, then who am I to tell them otherwise - but otherwise, I will want to override the default to False for vk build
            kwargs["use_IDs"] = False
    else:
        if fasta_out:
            kwargs["vcrs_filtered_fasta_out"] = fasta_out
            vcrs_filtered_fasta_out = fasta_out
        vcrs_fasta_for_index = vcrs_filtered_fasta_out
        if t2g_out:
            kwargs["vcrs_t2g_filtered_out"] = t2g_out
            vcrs_t2g_filtered_out = t2g_out
        vcrs_t2g_for_alignment = vcrs_t2g_filtered_out
        # don't touch use_IDs - if not provided, then will resort to defaults (True for vk build, False for vk filter); if provided, then will be passed into vk build and vk filter

    # download if download argument is True
    if download:
        prebuilt_vk_ref_files_key = f"variants={variants},sequences={sequences}"  # matches constants.py (all) and server (COSMIC only)  #* if I need more keys (eg w, k), then add onto this string conditionally based on the database
        if prebuilt_vk_ref_files_key not in prebuilt_vk_ref_files:
            raise ValueError(f"Invalid combination of parameters for downloading prebuilt reference files. Supported combinations are: {list(prebuilt_vk_ref_files.keys())}")
        file_dict = prebuilt_vk_ref_files[prebuilt_vk_ref_files_key]
        if file_dict:
            if file_dict["index"] == "COSMIC":
                raise NotImplementedError("Downloading COSMIC files is not currently supported. Please select another option or build a custom index.")
                # authenticate_cosmic_credentials(cosmic_email, cosmic_password)  # will raise an error if credentials are not valid
                # response = requests.post(COSMIC_CREDENTIAL_VALIDATION_URL, json={"email": cosmic_email, "password": cosmic_password, "prebuilt_vk_ref_files_key": prebuilt_vk_ref_files_key})
                # if response.status_code == 200:
                #     file_dict = response.json()  # Converts JSON to dict
                #     file_dict = file_dict.get("download_links")
                #     logger.info("Successfully verified COSMIC credentials.")
                #     logger.warning("According to COSMIC regulations, please do not share any data that utilizes the COSMIC database. See more here: https://cancer.sanger.ac.uk/cosmic/help/terms")
                # else:
                #     raise ValueError(f"Failed to verify COSMIC credentials. Status code: {response.status_code}")
            
            fasta_file_previously_existed = os.path.isfile(vcrs_fasta_for_index)
            logger.info(f"Downloading reference files with variants={variants}, sequences={sequences}")
            vk_ref_output_dict = download_varseek_files(file_dict, out=out, verbose=False)
            if index_out and vk_ref_output_dict["index"] != index_out:
                os.rename(vk_ref_output_dict["index"], index_out)
                vk_ref_output_dict["index"] = index_out
            if t2g_out and vk_ref_output_dict["t2g"] != t2g_out:
                os.rename(vk_ref_output_dict["t2g"], t2g_out)
                vk_ref_output_dict["t2g"] = t2g_out
            if fasta_out and vk_ref_output_dict["fasta"] != fasta_out:
                os.rename(vk_ref_output_dict["fasta"], fasta_out)
                vk_ref_output_dict["fasta"] = fasta_out
            # elif not fasta_out and not fasta_file_previously_existed:
            #     # delete vk_ref_output_dict["fasta"] if a path was not provided and this file did not previously exist
            #     os.remove(vk_ref_output_dict["fasta"])
            #     vk_ref_output_dict["fasta"] = None

            logger.info(f"Downloaded files: {vk_ref_output_dict}")

            return vk_ref_output_dict
        else:
            raise ValueError(f"No prebuilt files found for the given arguments:\nvariants: {variants}\nsequences: {sequences}")

    # use the value of sequences as of one of the dlist reference files if it is not provided
    if any(column in kwargs.get("columns_to_include", []) for column in vk.varseek_info.bowtie_columns_dlist) or kwargs.get("columns_to_include") == "all":
        if not dlist_reference_source and (not kwargs.get("dlist_reference_genome_fasta") or not kwargs.get("dlist_reference_cdna_fasta")):
            # determine if sequences is genome or transcriptome
            sequences_pyfastx = pyfastx.Fastx(sequences)

            max_seq_length = 0
            for name, seq in sequences_pyfastx:
                if len(seq) > max_seq_length:
                    max_seq_length = len(seq)
            if max_seq_length > 1_000_000 or (isinstance(variants, str) and variants.endswith(".vcf")):  # a heuristic to differentiate genome from transcriptome
                if not kwargs.get("dlist_reference_genome_fasta"):
                    logger.info("Assuming sequences is a genome based on its length (> 1,000,000 bp) or because variants parameter is a vcf file. Setting dlist_reference_genome_fasta to sequences.")
                    kwargs["dlist_reference_genome_fasta"] = sequences
            else:
                if not kwargs.get("dlist_reference_cdna_fasta"):
                    logger.info("Assuming sequences is a transcriptome based on its length (<= 1,000,000 bp). Setting dlist_reference_cdna_fasta to sequences.")
                    kwargs["dlist_reference_cdna_fasta"] = sequences
        if not dlist_reference_source and not kwargs.get("dlist_reference_genome_fasta"):
            logger.warning("Please provide a value to dlist_reference_source or dlist_reference_genome_fasta to cross-check VCRS k-mers to the reference genome. With the current setup, only the transcriptome will be checked.")
        if not dlist_reference_source and not kwargs.get("dlist_reference_cdna_fasta"):
            logger.warning("Please provide a value to dlist_reference_source or dlist_reference_cdna_fasta to cross-check VCRS k-mers to the reference transcriptome. With the current setup, only the genome will be checked.")
    
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

    # * vk build
    if not os.path.exists(file_signifying_successful_vk_build_completion) or overwrite:  # the reason I do it like this, rather than if overwrite or not os.path.exists(MYPATH), is because I would like vk ref/count to automatically overwrite partially-completed function outputs even when overwrite=False; but when overwrite=True, then run from scratch regardless
        kwargs_vk_build = {key: value for key, value in kwargs.items() if ((key in all_parameter_names_set_vk_build) and (key not in ref_signature.parameters.keys()))}
        # update anything in kwargs_vk_build that is not fully updated in (vk ref's) kwargs (should be nothing or very close to it, as I try to avoid these double-assignments by always keeping kwargs in kwargs)
        # eg kwargs_vk_build['mykwarg'] = mykwarg
        # just to be extra clear, I must explicitly pass arguments that are in the signature of vk ref; anything not in vk ref's signature should go in kwargs_vk_build (it is irrelevant what is in vk build's signature); and in the line above, I should update any values that are (1) not in vk ref's signature (so therefore they're in vk ref's kwargs), (2) I want to pass to vk build, and (3) have been updated outside of vk ref's kwargs somewhere in the function

        save_column_names_json_path = f"{out}/column_names_tmp.json"

        logger.info("Running vk build")
        _ = vk.build(sequences=sequences, variants=variants, seq_id_column=seq_id_column, var_column=var_column, var_id_column=var_id_column, w=w, k=k, out=out, reference_out_dir=reference_out_dir, chunksize=chunksize, dry_run=dry_run, overwrite=True, logging_level=logging_level, save_logs=save_logs, log_out_dir=log_out_dir, verbose=verbose, save_column_names_json_path=save_column_names_json_path, **kwargs_vk_build)  # overwrite=True rather than overwrite=overwrite because I only enter this condition if the file signifying success does not exist and/or overwrite is True anyways - this allows me to overwrite half-completed functions  # saves the temp json

        # use values for columns and file paths as provided in vk build
        if os.path.exists(save_column_names_json_path):  # will only exist if variants in supported_databases_and_corresponding_reference_sequence_type
            with open(save_column_names_json_path, "r") as f:
                column_names_and_file_names_dict = json.load(f)
            os.remove(save_column_names_json_path)
            if column_names_and_file_names_dict["seq_id_column"]:
                seq_id_column = column_names_and_file_names_dict["seq_id_column"]
            if column_names_and_file_names_dict["var_column"]:
                var_column = column_names_and_file_names_dict["var_column"]
            if column_names_and_file_names_dict["var_id_column"]:
                var_id_column = column_names_and_file_names_dict["var_id_column"]
            for column in ("seq_id_genome_column", "var_genome_column", "seq_id_cdna_column", "var_cdna_column", "gene_name_column"):
                kwargs[column] = kwargs.get(column, supported_databases_and_corresponding_reference_sequence_type[variants]["column_names"][column])
            for file in ("gtf", "reference_genome_fasta", "reference_cdna_fasta"):
                if column_names_and_file_names_dict[file]:
                    kwargs[file] = kwargs.get(file, column_names_and_file_names_dict[file])
            
            # if kwargs.get("reference_cds_fasta") and not kwargs.get("reference_cdna_fasta"):  # updated thought: this does in fact need to be cDNA (cannot be cds), as the column above refers to cDNA  # original thought: for the purposes of vk info, reference_cdna_fasta really just refers to the transcriptome, whether it is cDNA or CDS
            #     kwargs["reference_cdna_fasta"] = kwargs["reference_cds_fasta"]

    else:
        logger.warning(f"Skipping vk build because {file_signifying_successful_vk_build_completion} already exists and overwrite=False")

    # * vk info
    if not skip_info:
        if kwargs.get("use_IDs", None) is False:
            logger.warning("use_IDs=False is not recommended for vk info, as the headers output by vk build can break some programs that read fasta files due to the inclusion of '>' symbols in substitutions and the potentially long length of the headers (with multiple combined headers and/or long insertions). Consider setting use_IDs=True (use IDs throughout the workflow) or leaving this parameter blank (will use IDs in vk build so that vk info runs properly [unless vk info/filter will not be run, in which case it will use headers], and will use headers in vk filter so that the output is more readable).")
        if not os.path.exists(file_signifying_successful_vk_info_completion) or overwrite:
            kwargs_vk_info = {key: value for key, value in kwargs.items() if ((key in all_parameter_names_set_vk_info) and (key not in ref_signature.parameters.keys()))}
            # update anything in kwargs_vk_info that is not fully updated in (vk ref's) kwargs (should be nothing or very close to it, as I try to avoid these double-assignments by always keeping kwargs in kwargs)
            # eg kwargs_vk_info['mykwarg'] = mykwarg

            logger.info("Running vk info")
            _ = vk.info(k=k, dlist_reference_source=dlist_reference_source, dlist_reference_ensembl_release=dlist_reference_ensembl_release, seq_id_column=seq_id_column, var_column=var_column, out=out, reference_out_dir=reference_out_dir, chunksize=chunksize, dry_run=dry_run, overwrite=True, threads=threads, logging_level=logging_level, save_logs=save_logs, log_out_dir=log_out_dir, verbose=verbose, variants=variants, sequences=sequences, w=w, **kwargs_vk_info)  # overwrite=True rather than overwrite=overwrite because I only enter this condition if the file signifying success does not exist and/or overwrite is True anyways - this allows me to overwrite half-completed functions  # a kwargs of vk info but explicit in vk ref  # a kwargs of vk info but explicit in vk ref  # including input_dir
        else:
            logger.warning(f"Skipping vk info because {file_signifying_successful_vk_info_completion} already exists and overwrite=False")

    # vk filter
    if not skip_filter:
        if not all(os.path.exists(f) for f in files_signifying_successful_vk_filter_completion) or overwrite:
            kwargs_vk_filter = {key: value for key, value in kwargs.items() if ((key in all_parameter_names_set_vk_filter) and (key not in ref_signature.parameters.keys()))}
            # update anything in kwargs_vk_filter that is not fully updated in (vk ref's) kwargs (should be nothing or very close to it, as I try to avoid these double-assignments by always keeping kwargs in kwargs)
            # eg kwargs_vk_filter['mykwarg'] = mykwarg

            logger.info("Running vk filter")
            _ = vk.filter(filters=filters, out=out, chunksize=chunksize, dry_run=dry_run, overwrite=True, logging_level=logging_level, save_logs=save_logs, log_out_dir=log_out_dir, **kwargs_vk_filter)  # overwrite=True rather than overwrite=overwrite because I only enter this condition if the file signifying success does not exist and/or overwrite is True anyways - this allows me to overwrite half-completed functions
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
            print(' '.join(kb_ref_command))
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
        if not os.path.exists(file_signifying_successful_wt_vcrs_kb_ref_completion) or overwrite:
            kb_ref_wt_vcrs_command = ["kb", "ref", "--workflow", "custom", "-t", str(threads), "-i", wt_vcrs_index_out, "--d-list", "None", "-k", str(k), "--overwrite", vcrs_wt_fasta_for_index]  # set to True here regardless of the overwrite argument because I would only even enter this block if kb count was only partially run (as seen by the lack of existing of file_signifying_successful_wt_vcrs_kb_ref_completion), in which case I should overwrite anyways
            if dry_run:
                print(' '.join(kb_ref_wt_vcrs_command))
            else:
                logger.info(f"Running kb ref for wt vcrs index with command: {' '.join(kb_ref_wt_vcrs_command)}")
                subprocess.run(kb_ref_wt_vcrs_command, check=True)
        else:
            logger.warning(f"Skipping kb ref for wt vcrs because {file_signifying_successful_wt_vcrs_kb_ref_completion} already exists and overwrite=False")
    #!!! erase if removing wt vcrs feature
    
    vk_ref_output_dict = {}
    vk_ref_output_dict["index"] = os.path.abspath(index_out) if (isinstance(index_out, str) and os.path.isfile(index_out) and not dry_run) else None
    vk_ref_output_dict["t2g"] = os.path.abspath(vcrs_t2g_for_alignment) if (isinstance(vcrs_t2g_for_alignment, str) and os.path.isfile(vcrs_t2g_for_alignment) and not dry_run) else None
    vk_ref_output_dict["fasta"] = os.path.abspath(vcrs_fasta_for_index) if (isinstance(vcrs_fasta_for_index, str) and os.path.isfile(vcrs_fasta_for_index) and not dry_run) else None

    logger.info(f"Produced files: {vk_ref_output_dict}")

    return vk_ref_output_dict
