"""main function for argparse."""

import argparse
import inspect
import math
import os
import re
import sys
from datetime import datetime
from pdb import set_trace as st

import pandas as pd

from .__init__ import __version__
from .constants import (
    varseek_count_only_allowable_kb_count_arguments,
    varseek_ref_only_allowable_kb_ref_arguments,
)
from .utils import set_up_logger
from .varseek_build import build, vk_build_hidden_from_help
from .varseek_clean import clean, vk_clean_hidden_from_help
from .varseek_count import count, vk_count_hidden_from_help
from .varseek_denovo import denovo
from .varseek_fastqpp import fastqpp, vk_fastqpp_hidden_from_help
from .varseek_ref import ref
from .varseek_summarize import summarize, vk_summarize_hidden_from_help

# Get current date and time for alphafold default foldername
dt_string = datetime.now().strftime("%Y_%m_%d-%H_%M")


def extract_help_from_doc(module, arg_name, disable=False):
    """
    Extracts the help message for a given argument from the module's docstring, handling multi-line descriptions.
    Requires a docstring line of the following format:
    - ARGUMENT1     (TYPE1 or TYPE2 or ...) DESCRIPTION
    OPTIONAL EXTRA DESCRIPTION LINE 1
    OPTIONAL EXTRA DESCRIPTION LINE 2
    ...
    - ARGUMENT2    (TYPE1 or TYPE2 or ...) DESCRIPTION
    ...
    # Another block of arguments
    - ARGUMENT3    (TYPE1 or TYPE2 or ...) DESCRIPTION
    ...
    """
    docstring = inspect.getdoc(module)
    help_message = []

    # Regular expression to match the argument line with flexible type matching
    arg_pattern = rf"-\s*{arg_name}\s*\((.*?)\)\s*(.*)"

    # Regular expression to match the start of a new argument or 'Additional input arguments:'
    new_arg_pattern = r"-\s*[a-zA-Z_]\w*\s*\(.*?\)|\n\n# "

    capturing = False  # Flag to check if we are reading the target argument's help message

    for line in docstring.splitlines():
        # Stop capturing if a new argument or 'Additional input arguments:' is found after starting
        if re.match(new_arg_pattern, line.strip()) and capturing:
            break

        if capturing:
            # Continue capturing the help message if the line is indented or blank (continuation of help message)
            if line.strip() == "" or line.startswith(" ") or line.startswith("\t"):
                help_message.append(line.strip())
            else:
                break  # Stop if we encounter an unindented line that does not belong to the current argument
        elif re.match(arg_pattern, line.strip()):
            # Start capturing when the argument is found
            capturing = True
            # Extract the help message part after the argument's type annotation
            match = re.search(arg_pattern, line.strip())
            if match:
                # Append the help message (ignoring the type in parentheses)
                if match.group(2).strip():
                    help_message.append(match.group(2).strip())

    if help_message:
        if disable:
            help_message = [f"Disable {arg_name}, described below:"] + help_message
        return "\n".join(help_message).strip()
    return "Help message not found in docstring."
    # raise ValueError(f"Argument '{arg_name}' not found in the docstring of the module '{module}'.")


def replace_old_arg_names_with_new_arg_names_in_params_dict_and_combine_with_kwargs(params_dict, python_arg_to_cli_arg_dict, kwargs=None):
    if python_arg_to_cli_arg_dict:
        for python_arg, cli_arg in python_arg_to_cli_arg_dict.items():
            if cli_arg in params_dict:
                params_dict[python_arg] = params_dict.pop(cli_arg)  # removes the old arg and adds in the new arg

    # combine params_dict and kwargs - params_dict takes precedence
    if kwargs:
        params_dict = {**kwargs, **params_dict}

    return params_dict


# * important rules: make sure to specify all positional arguments and all requirement arguments within dest_parser (e.g., vk ref/count) - this function will look at any parameters not in dest_parser and convert positional --> keyword (eg adata --> --adata), and will make required=False
def copy_arguments(src_parser_list, dest_parser):
    if not isinstance(src_parser_list, (list, tuple)):
        src_parser_list = [src_parser_list]

    for src_parser in src_parser_list:
        for action in src_parser._actions:
            if not any(a.dest == action.dest for a in dest_parser._actions):
                # Start with common arguments
                kwargs = {
                    "dest": action.dest,
                    "default": argparse.SUPPRESS,
                    "required": False,  # anything required will already be included in dest_parser, so make sure it's all optional
                    "help": argparse.SUPPRESS,  # Hide in help
                }

                # Add optional arguments based on action type
                if isinstance(action, argparse._StoreAction):  # Normal --arg val
                    kwargs.update(
                        {
                            "type": action.type,
                            "choices": action.choices,
                            "nargs": action.nargs,
                            "metavar": action.metavar,
                        }
                    )
                elif isinstance(action, argparse._StoreConstAction) and not isinstance(action, (argparse._StoreTrueAction, argparse._StoreFalseAction)):
                    kwargs["const"] = action.const
                elif isinstance(action, (argparse._StoreTrueAction, argparse._StoreFalseAction)):
                    pass  # No extra kwargs needed for store_true/store_false

                if not action.option_strings:  # convert positional --> keyword argument
                    option_strings = ["--" + action.dest]
                else:  # keep keyword argument as-is
                    option_strings = action.option_strings

                    # Remove conflicting short flags
                    for flag in option_strings.copy():
                        # Identify short flags: e.g. "-x" (starts with a single dash and is 2 characters long)
                        if flag.startswith("-") and not flag.startswith("--"):  # and len(flag) == 2:
                            if flag in dest_parser._option_string_actions:
                                option_strings.remove(flag)

                # Add the argument to the destination parser
                dest_parser.add_argument(*option_strings, action=action.__class__, **kwargs)


def add_build_arguments(parser, required):
    """Add the full set of `vk build` command-line arguments to `parser`.

    `vk ref` is literally an alias/wrapper for `vk build` (see varseek_ref.ref), so both
    subparsers expose the exact same arguments by calling this. `required` controls whether
    the core `--variants`/`--sequences` arguments are required (they are not when a
    list-and-exit flag such as --list_downloadable_references is present).
    """
    parser.add_argument(
        "-v",
        "--variants",
        # type=strpath_or_str_or_list_or_df,
        nargs="+",
        required=required,
        help=extract_help_from_doc(build, "variants"),
    )
    parser.add_argument(
        "-s",
        "--sequences",
        type=str,
        nargs="+",
        required=required,
        help=extract_help_from_doc(build, "sequences"),
    )
    parser.add_argument(
        "-w",
        "--w",
        type=int,
        required=False,
        default=argparse.SUPPRESS,  # Remove from args if not provided
        help=extract_help_from_doc(build, "w"),
    )
    parser.add_argument(
        "-k",
        "--k",
        type=int,
        required=False,
        default=argparse.SUPPRESS,  # Remove from args if not provided
        help=extract_help_from_doc(build, "k"),
    )
    parser.add_argument(
        "-ma",
        "--max_ambiguous",
        type=int,
        required=False,
        default=argparse.SUPPRESS,  # Remove from args if not provided
        help=extract_help_from_doc(build, "max_ambiguous"),
    )
    # disabled for now (see the commented-out `technology` parameter on vk build)
    # parser.add_argument(
    #     "-x",
    #     "--technology",
    #     type=str,
    #     required=False,
    #     default=argparse.SUPPRESS,  # Remove from args if not provided
    #     help=extract_help_from_doc(build, "technology"),
    # )
    parser.add_argument(
        "-vc",
        "--var_column",
        type=str,
        required=False,
        default=argparse.SUPPRESS,  # Remove from args if not provided
        help=extract_help_from_doc(build, "var_column"),
    )
    parser.add_argument(
        "-sic",
        "--seq_id_column",
        type=str,
        required=False,
        default=argparse.SUPPRESS,  # Remove from args if not provided
        help=extract_help_from_doc(build, "seq_id_column"),
    )
    parser.add_argument(
        "-vic",
        "--var_id_column",
        required=False,
        default=argparse.SUPPRESS,  # Remove from args if not provided
        help=extract_help_from_doc(build, "var_id_column"),
    )
    parser.add_argument(
        "-gtf",
        "--gtf",
        required=False,
        default=argparse.SUPPRESS,  # Remove from args if not provided
        help=extract_help_from_doc(build, "gtf"),
    )
    parser.add_argument(
        "-gtic",
        "--gtf_transcript_id_column",
        required=False,
        default=argparse.SUPPRESS,  # Remove from args if not provided
        help=extract_help_from_doc(build, "gtf_transcript_id_column"),
    )
    parser.add_argument(
        "--transcript_boundaries",
        action="store_true",
        default=argparse.SUPPRESS,  # Remove from args if not provided
        help=extract_help_from_doc(build, "transcript_boundaries"),
    )
    parser.add_argument(
        "--convert_variant_coordinates",
        choices=["genome_to_transcript", "transcript_to_genome"],
        default=argparse.SUPPRESS,  # Remove from args if not provided
        help=extract_help_from_doc(build, "convert_variant_coordinates"),
    )
    parser.add_argument(
        "-o",
        "--out",
        type=str,
        required=False,
        default=argparse.SUPPRESS,  # Remove from args if not provided
        help=extract_help_from_doc(build, "out"),
    )
    parser.add_argument(
        "-r",
        "--reference_out_dir",
        required=False,
        default=argparse.SUPPRESS,  # Remove from args if not provided
        help=extract_help_from_doc(build, "reference_out_dir"),
    )
    parser.add_argument(
        "--vcrs_unfiltered_fasta_out",
        required=False,
        default=argparse.SUPPRESS,  # Remove from args if not provided
        help=extract_help_from_doc(build, "vcrs_unfiltered_fasta_out"),
    )
    parser.add_argument(
        "--variants_updated_csv_out",
        required=False,
        default=argparse.SUPPRESS,  # Remove from args if not provided
        help=extract_help_from_doc(build, "variants_updated_csv_out"),
    )
    parser.add_argument(
        "--id_to_header_csv_out",
        required=False,
        default=argparse.SUPPRESS,  # Remove from args if not provided
        help=extract_help_from_doc(build, "id_to_header_csv_out"),
    )
    parser.add_argument(
        "--id_to_header_unfiltered_csv_out",
        required=False,
        default=argparse.SUPPRESS,  # Remove from args if not provided
        help=extract_help_from_doc(build, "id_to_header_unfiltered_csv_out"),
    )
    parser.add_argument(
        "-g",
        "--vcrs_t2g_out",
        required=False,
        default=argparse.SUPPRESS,  # Remove from args if not provided
        help=extract_help_from_doc(build, "vcrs_t2g_out"),
    )
    parser.add_argument(
        "--removed_variants_text_out",
        required=False,
        default=argparse.SUPPRESS,  # Remove from args if not provided
        help=extract_help_from_doc(build, "removed_variants_text_out"),
    )
    parser.add_argument(
        "--filtering_report_text_out",
        required=False,
        default=argparse.SUPPRESS,  # Remove from args if not provided
        help=extract_help_from_doc(build, "filtering_report_text_out"),
    )
    parser.add_argument(
        "--return_variant_output",
        action="store_true",
        default=argparse.SUPPRESS,  # Remove from args if not provided
        help=extract_help_from_doc(build, "return_variant_output"),
    )
    parser.add_argument(
        "--save_variants_updated_dataframe",
        action="store_true",
        default=argparse.SUPPRESS,  # Remove from args if not provided
        help=extract_help_from_doc(build, "save_variants_updated_dataframe"),
    )
    parser.add_argument(
        "--chunksize",
        type=int,
        default=argparse.SUPPRESS,  # Remove from args if not provided
        help=extract_help_from_doc(build, "chunksize"),
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        default=argparse.SUPPRESS,  # Remove from args if not provided
        help=extract_help_from_doc(build, "dry_run"),
    )
    parser.add_argument(
        "--list_internally_supported_indices",
        action="store_true",
        default=argparse.SUPPRESS,  # Remove from args if not provided
        help=extract_help_from_doc(build, "list_internally_supported_indices"),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        default=argparse.SUPPRESS,  # Remove from args if not provided
        help=extract_help_from_doc(build, "overwrite"),
    )
    parser.add_argument(
        "--logging_level",
        choices=["NOTSET", "DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL", "0", "10", "20", "30", "40", "50", "60", None],
        required=False,
        default=argparse.SUPPRESS,  # Remove from args if not provided
        help=extract_help_from_doc(build, "logging_level"),
    )
    parser.add_argument(
        "--save_logs",
        action="store_true",
        default=argparse.SUPPRESS,  # Remove from args if not provided
        help=extract_help_from_doc(build, "save_logs"),
    )
    parser.add_argument(
        "--log_out_dir",
        default=argparse.SUPPRESS,  # Remove from args if not provided
        help=extract_help_from_doc(build, "log_out_dir"),
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        default=argparse.SUPPRESS,  # Remove from args if not provided
        help=extract_help_from_doc(build, "verbose"),
    )

    # Additional kwargs arguments that I still want as command-line options
    parser.add_argument(
        "--insertion_size_limit",
        required=False,
        default=argparse.SUPPRESS,  # Remove from args if not provided
        help=extract_help_from_doc(build, "insertion_size_limit"),
    )
    parser.add_argument(
        "--min_seq_len",
        type=int,
        required=False,
        default=argparse.SUPPRESS,  # Remove from args if not provided
        help=extract_help_from_doc(build, "min_seq_len"),
    )
    parser.add_argument(
        "--disable_optimize_flanking_regions",
        dest="optimize_flanking_regions",
        action="store_false",
        default=argparse.SUPPRESS,  # Remove from args if not provided
        help=extract_help_from_doc(build, "optimize_flanking_regions", disable=True),
    )
    parser.add_argument(
        "--disable_remove_seqs_with_wt_kmers",
        dest="remove_seqs_with_wt_kmers",
        action="store_false",
        default=argparse.SUPPRESS,  # Remove from args if not provided
        help=extract_help_from_doc(build, "remove_seqs_with_wt_kmers", disable=True),
    )
    parser.add_argument(
        "--disable_merge_identical",
        dest="merge_identical",
        action="store_false",
        default=argparse.SUPPRESS,  # Remove from args if not provided
        help=extract_help_from_doc(build, "merge_identical", disable=True),
    )
    parser.add_argument(
        "--merge_reference_equivalent_headers",
        action="store_true",
        default=argparse.SUPPRESS,  # Remove from args if not provided
        help=extract_help_from_doc(build, "merge_reference_equivalent_headers"),
    )
    parser.add_argument(
        "--disable_merge_subsequences",
        dest="merge_subsequences",
        action="store_false",
        default=argparse.SUPPRESS,  # Remove from args if not provided
        help=extract_help_from_doc(build, "merge_subsequences", disable=True),
    )
    parser.add_argument(
        "--disable_add_gene_name_to_header",
        dest="add_gene_name_to_header",
        action="store_false",
        default=argparse.SUPPRESS,  # Remove from args if not provided
        help=extract_help_from_doc(build, "add_gene_name_to_header", disable=True),
    )
    parser.add_argument(
        "-vs",
        "--vcrs_strandedness",
        action="store_true",
        default=argparse.SUPPRESS,  # Remove from args if not provided
        help=extract_help_from_doc(build, "vcrs_strandedness"),
    )
    parser.add_argument(
        "--use_IDs",
        dest="use_IDs",
        action="store_true",
        default=argparse.SUPPRESS,  # Remove from args if not provided
        help=extract_help_from_doc(build, "use_IDs"),
    )
    parser.add_argument(
        "--cosmic_version",
        required=False,
        default=argparse.SUPPRESS,  # Remove from args if not provided
        help=extract_help_from_doc(build, "cosmic_version"),
    )
    parser.add_argument(
        "--cosmic_grch",
        required=False,
        default=argparse.SUPPRESS,  # Remove from args if not provided
        help=extract_help_from_doc(build, "cosmic_grch"),
    )
    parser.add_argument(
        "--disable_original_order",
        dest="original_order",
        action="store_false",
        default=argparse.SUPPRESS,  # Remove from args if not provided
        help=extract_help_from_doc(build, "original_order", disable=True),
    )
    parser.add_argument(
        "--cdna_derived_vcf",
        action="store_true",
        default=argparse.SUPPRESS,  # Remove from args if not provided
        help=extract_help_from_doc(build, "cdna_derived_vcf"),
    )

    # index-creation / reference arguments (kb ref is now run by vk build; formerly vk ref)
    parser.add_argument(
        "-i",
        "--index_out",
        required=False,
        default=argparse.SUPPRESS,  # Remove from args if not provided
        help=extract_help_from_doc(build, "index_out"),
    )
    parser.add_argument(
        "--vcrs_fasta_out",
        required=False,
        default=argparse.SUPPRESS,  # Remove from args if not provided
        help=extract_help_from_doc(build, "vcrs_fasta_out"),
    )
    parser.add_argument(
        "--max_homopolymer_length",
        type=int,
        required=False,
        default=argparse.SUPPRESS,  # Remove from args if not provided
        help=extract_help_from_doc(build, "max_homopolymer_length"),
    )
    parser.add_argument(
        "--min_triplet_complexity",
        type=float,
        required=False,
        default=argparse.SUPPRESS,  # Remove from args if not provided
        help=extract_help_from_doc(build, "min_triplet_complexity"),
    )
    parser.add_argument(
        "--keep_alignment_to_reference",
        dest="remove_alignment_to_reference",
        action="store_false",
        default=argparse.SUPPRESS,  # Remove from args if not provided
        help=extract_help_from_doc(build, "remove_alignment_to_reference"),
    )
    parser.add_argument(
        "--alignment_to_reference_type",
        required=False,
        default=argparse.SUPPRESS,  # Remove from args if not provided
        help=extract_help_from_doc(build, "alignment_to_reference_type"),
    )
    parser.add_argument(
        "--alignment_to_reference_aligner",
        required=False,
        choices=["bowtie2", "kallisto"],
        default=argparse.SUPPRESS,  # Remove from args if not provided
        help=extract_help_from_doc(build, "alignment_to_reference_aligner"),
    )
    parser.add_argument(
        "--species",
        required=False,
        default=argparse.SUPPRESS,  # Remove from args if not provided
        help=extract_help_from_doc(build, "species"),
    )
    parser.add_argument(
        "--alignment_to_reference_dna",
        required=False,
        default=argparse.SUPPRESS,  # Remove from args if not provided
        help=extract_help_from_doc(build, "alignment_to_reference_dna"),
    )
    parser.add_argument(
        "--alignment_to_reference_gtf",
        required=False,
        default=argparse.SUPPRESS,  # Remove from args if not provided
        help=extract_help_from_doc(build, "alignment_to_reference_gtf"),
    )
    parser.add_argument(
        "--dlist",
        required=False,
        default=argparse.SUPPRESS,  # Remove from args if not provided
        help=extract_help_from_doc(build, "dlist"),
    )
    parser.add_argument(
        "-t",
        "--threads",
        type=int,
        required=False,
        default=argparse.SUPPRESS,  # Remove from args if not provided
        help=extract_help_from_doc(build, "threads"),
    )
    parser.add_argument(
        "--dont_create_index",
        action="store_true",
        default=argparse.SUPPRESS,  # Remove from args if not provided
        help="Skip the kb ref index-creation step and only produce the VCRS fasta/t2g (the classic vk build behavior). By default the index is created.",
    )
    parser.add_argument(
        "-d",
        "--download",
        action="store_true",
        default=argparse.SUPPRESS,  # Remove from args if not provided
        help=extract_help_from_doc(build, "download"),
    )
    parser.add_argument(
        "--list_downloadable_references",
        action="store_true",
        default=argparse.SUPPRESS,  # Remove from args if not provided
        help=extract_help_from_doc(build, "list_downloadable_references"),
    )

    # Hide the advanced `build` parameters (defined in varseek_build.vk_build_hidden_from_help) from
    # help. They remain fully parseable — only their help text is suppressed. Matched by
    # dest, so the --disable_* flags map to their underlying parameter name.
    for action in parser._actions:
        if action.dest in vk_build_hidden_from_help:
            action.help = argparse.SUPPRESS


# Custom formatter for help messages that preserved the text formatting and adds the default value to the end of the help message
class CustomHelpFormatter(argparse.RawTextHelpFormatter):
    def _get_help_string(self, action):
        help_str = action.help if action.help else ""
        if (
            "%(default)" not in help_str
            and action.default is not argparse.SUPPRESS
            and action.default is not None
            # default information can be deceptive or confusing for boolean flags.
            # For example, `--quiet` says "Does not print progress information. (default: True)" even though
            # the default action is to NOT be quiet (to the user, the default is False).
            and not isinstance(action, argparse._StoreTrueAction)
            and not isinstance(action, argparse._StoreFalseAction)
        ):
            help_str += " (default: %(default)s)"
        return help_str


def convert_to_list(*args):
    args_list = list(args)
    return args_list


def int_or_float(value):
    # Check if the value is an int or a float (including infinity)
    return isinstance(value, (int, float)) and not isinstance(value, bool)  # Excludes boolean values


def is_int_or_float_or_inf(value):
    return int_or_float(value) or (isinstance(value, float) and math.isinf(value))


valid_df_file_extensions = [".csv", ".tsv", ".xls", ".xlsx", ".parquet", ".h5"]


def strpath_or_df(value):
    # Check if the input is a pandas DataFrame
    if isinstance(value, pd.DataFrame):
        return value

    # Check if the input is a string (potential file path) and has a valid extension
    if isinstance(value, str) and os.path.isfile(value):
        if any(value.endswith(extension) for extension in valid_df_file_extensions):
            return value
        else:
            raise ValueError(f"File has an unsupported extension: {value}")

    # If neither condition is satisfied, raise an error
    raise ValueError("Input must be either a valid file path or a pandas DataFrame.")


def strpath_or_strnonpath_or_df(value):
    # Check if the input is a pandas DataFrame
    if isinstance(value, pd.DataFrame):
        return value

    if isinstance(value, str) and not os.path.isfile(value):
        return value

    # Check if the input is a string (potential file path) and has a valid extension
    if isinstance(value, str) and os.path.isfile(value):
        if any(value.endswith(extension) for extension in valid_df_file_extensions):
            return value
        else:
            raise ValueError(f"File has an unsupported extension: {value}")

    # If neither condition is satisfied, raise an error
    raise ValueError("Input must be either a valid file path or a pandas DataFrame.")


def int_or_str(value):
    try:
        return int(value)
    except ValueError:
        return value


def strpath_or_list_like_of_strings(value):
    if isinstance(value, str):
        return value

    if isinstance(value, list) or isinstance(value, set) or isinstance(value, tuple):
        for v in value:
            if not isinstance(v, str):
                raise ValueError(f"All elements in the list must be strings. Found: {v}")
        return value

    raise TypeError(f"Expected a string or list-like of strings, but got {type(value).__name__}")


def strpath_or_str_or_list_or_df(value):
    # Check if the input is a DataFrame
    if isinstance(value, pd.DataFrame):
        return value

    # Check if the input is a list
    if isinstance(value, list):
        # return [strpath_or_str_or_list_or_df(v) for v in value]
        return value

    # Check if the input is a string (non-path)
    if isinstance(value, str) and not os.path.isfile(value):
        return value

    # Check if the input is a string that is also a valid file path
    if isinstance(value, str) and os.path.isfile(value):
        if any(value.endswith(extension) for extension in valid_df_file_extensions):
            return value
        else:
            raise ValueError(f"File has an unsupported extension: {value}")

    # If none of the conditions match, raise an error
    raise ValueError("Input must be a non-path string, a valid file path, a list, or a pandas DataFrame.")


def main():  # noqa: C901
    """
    Function containing argparse parsers and arguments to allow the use of varseek from the terminal (as varseek).
    """

    # Define parent parser
    parent_parser = argparse.ArgumentParser(description=f"varseek v{__version__}", add_help=False)
    # Initiate subparsers.
    # `build` is intentionally omitted from this metavar so it stays hidden from the top-level
    # `vk --help` command listing. It remains fully invocable (`vk build ...`) and is the argument
    # source for `vk ref`; `vk ref` is the public-facing name for the same functionality (mirroring
    # kb ref / kb count). If you add or rename a *public* subcommand, update this list to match.
    parent_subparsers = parent_parser.add_subparsers(dest="command", metavar="{denovo,fastqpp,clean,summarize,ref,count}")
    # Define parent (not sure why I need both parent parser and parent, but otherwise it does not work)
    parent = argparse.ArgumentParser(add_help=False)

    # Add custom help argument to parent parser
    parent_parser.add_argument("-h", "--help", action="store_true", help="Print manual.")
    # Add custom version argument to parent parser
    parent_parser.add_argument("-v", "--version", action="store_true", help="Print version.")


    # Check if a flag is passed that causes a script to exist early, thus making normally required arguments optional (eg I can call vk build --list_internally_supported_indices without providing -s and -v)
    list_information_and_exit_flag_dict = {}
    for list_information_and_exit_flag in ("list_internally_supported_indices", "list_downloadable_references"):
        list_information_and_exit_flag_dict[list_information_and_exit_flag] = False
        for i, arg in enumerate(sys.argv):
            if arg == f"--{list_information_and_exit_flag}":
                list_information_and_exit_flag_dict[list_information_and_exit_flag] = True
                break

    vk_build_list_information_and_exit_flag_present = list_information_and_exit_flag_dict["list_internally_supported_indices"]
    vk_ref_list_information_and_exit_flag_present = any(list_information_and_exit_flag_dict.values())


    # denovo parser arguments
    denovo_desc = "Call de novo variants from FASTQ or BAM inputs."

    parser_denovo = parent_subparsers.add_parser(
        "denovo",
        parents=[parent],
        description=denovo_desc,
        help=denovo_desc,
        add_help=True,
        formatter_class=CustomHelpFormatter,
    )
    parser_denovo.add_argument(
        "inputs",
        nargs="+",
        help=extract_help_from_doc(denovo, "inputs"),
    )
    parser_denovo.add_argument(
        "-s",
        "--sequences",
        required=True,
        help=extract_help_from_doc(denovo, "sequences"),
    )
    parser_denovo.add_argument(
        "-g",
        "--gtf",
        default=argparse.SUPPRESS,
        help=extract_help_from_doc(denovo, "gtf"),
    )
    parser_denovo.add_argument(
        "-p",
        "--parity",
        choices=["single", "paired"],
        default=argparse.SUPPRESS,
        help=extract_help_from_doc(denovo, "parity"),
    )
    parser_denovo.add_argument(
        "-xs",
        "--star-genome-index-dir",
        "--star_genome_index_dir",
        default=argparse.SUPPRESS,
        help=extract_help_from_doc(denovo, "star_genome_index_dir"),
    )
    parser_denovo.add_argument(
        "-xb",
        "--bowtie2-genome-index-prefix",
        "--bowtie2_genome_index_prefix",
        default=argparse.SUPPRESS,
        help=extract_help_from_doc(denovo, "bowtie2_genome_index_prefix"),
    )
    parser_denovo.add_argument(
        "--star-alignment-prefix",
        "--star_alignment_prefix",
        default=argparse.SUPPRESS,
        help=extract_help_from_doc(denovo, "star_alignment_prefix"),
    )
    parser_denovo.add_argument(
        "--bowtie2-alignment-dir",
        "--bowtie2_alignment_dir",
        default=argparse.SUPPRESS,
        help=extract_help_from_doc(denovo, "bowtie2_alignment_dir"),
    )
    parser_denovo.add_argument(
        "-R",
        "--regions",
        default=argparse.SUPPRESS,
        help=extract_help_from_doc(denovo, "regions"),
    )
    parser_denovo.add_argument(
        "-obd",
        "--out-bam-dir",
        "--out_bam_dir",
        default=argparse.SUPPRESS,
        help=extract_help_from_doc(denovo, "out_bam_dir"),
    )
    parser_denovo.add_argument(
        "-o",
        "--output",
        default=argparse.SUPPRESS,
        help=extract_help_from_doc(denovo, "output"),
    )
    parser_denovo.add_argument(
        "--output-tsv",
        "--output_tsv",
        default=argparse.SUPPRESS,
        help=extract_help_from_doc(denovo, "output_tsv"),
    )
    parser_denovo.add_argument(
        "--tsv-reference-type",
        "--tsv_reference_type",
        choices=["auto", "dna", "genome", "cdna", "transcriptome"],
        default=argparse.SUPPRESS,
        help=extract_help_from_doc(denovo, "tsv_reference_type"),
    )
    parser_denovo.add_argument(
        "-r",
        "--read-length",
        "--read_length",
        type=int,
        default=argparse.SUPPRESS,
        help=extract_help_from_doc(denovo, "read_length"),
    )
    parser_denovo.add_argument(
        "-m",
        "--min-counts",
        "--min_counts",
        type=int,
        default=argparse.SUPPRESS,
        help=extract_help_from_doc(denovo, "min_counts"),
    )
    parser_denovo.add_argument(
        "-a",
        "--aligner",
        choices=["STAR", "bowtie2"],
        default=argparse.SUPPRESS,
        help=extract_help_from_doc(denovo, "aligner"),
    )
    parser_denovo.add_argument(
        "--technology",
        default=argparse.SUPPRESS,
        help=extract_help_from_doc(denovo, "technology"),
    )
    parser_denovo.add_argument(
        "--reference-type",
        "--reference_type",
        choices=["auto", "genome", "dna", "transcriptome", "cdna"],
        default=argparse.SUPPRESS,
        help=extract_help_from_doc(denovo, "reference_type"),
    )
    parser_denovo.add_argument(
        "--variant-caller",
        "--variant_caller",
        choices=["bcftools", "cigar"],
        default=argparse.SUPPRESS,
        help=extract_help_from_doc(denovo, "variant_caller"),
    )
    parser_denovo.add_argument(
        "--bowtie2-seed-length",
        "--bowtie2_seed_length",
        type=int,
        default=argparse.SUPPRESS,
        help=extract_help_from_doc(denovo, "bowtie2_seed_length"),
    )
    parser_denovo.add_argument(
        "--bowtie2-score-min",
        "--bowtie2_score_min",
        default=argparse.SUPPRESS,
        help=extract_help_from_doc(denovo, "bowtie2_score_min"),
    )
    parser_denovo.add_argument(
        "-i",
        "--include",
        default=argparse.SUPPRESS,
        help=extract_help_from_doc(denovo, "include"),
    )
    parser_denovo.add_argument(
        "-I",
        "--skip-indels",
        "--skip_indels",
        action="store_true",
        default=argparse.SUPPRESS,
        help=extract_help_from_doc(denovo, "skip_indels"),
    )
    parser_denovo.add_argument(
        "--disable-baq",
        "--disable_baq",
        action="store_true",
        default=argparse.SUPPRESS,
        help=extract_help_from_doc(denovo, "disable_baq"),
    )
    parser_denovo.add_argument(
        "--split-bam-by-n",
        "--split_bam_by_n",
        action="store_true",
        default=argparse.SUPPRESS,
        help=extract_help_from_doc(denovo, "split_bam_by_n"),
    )
    parser_denovo.add_argument(
        "--merge-bam-files",
        "--merge_bam_files",
        action="store_true",
        default=argparse.SUPPRESS,
        help=extract_help_from_doc(denovo, "merge_bam_files"),
    )
    parser_denovo.add_argument(
        "--strip-version-numbers",
        "--strip_version_numbers",
        action="store_true",
        default=argparse.SUPPRESS,
        help=extract_help_from_doc(denovo, "strip_version_numbers"),
    )
    parser_denovo.add_argument(
        "--disable-bcftools-norm",
        "--disable_bcftools_norm",
        action="store_true",
        default=argparse.SUPPRESS,
        help=extract_help_from_doc(denovo, "disable_bcftools_norm"),
    )
    parser_denovo.add_argument(
        "--bcftools-call-prior",
        "--bcftools_call_prior",
        default=argparse.SUPPRESS,
        help=extract_help_from_doc(denovo, "bcftools_call_prior"),
    )
    parser_denovo.add_argument(
        "--tmp-dir",
        "--tmp_dir",
        default=argparse.SUPPRESS,
        help=extract_help_from_doc(denovo, "tmp_dir"),
    )
    parser_denovo.add_argument(
        "-t",
        "--threads",
        type=int,
        default=argparse.SUPPRESS,
        help=extract_help_from_doc(denovo, "threads"),
    )
    parser_denovo.add_argument(
        "--overwrite",
        action="store_true",
        default=argparse.SUPPRESS,
        help=extract_help_from_doc(denovo, "overwrite"),
    )
    parser_denovo.add_argument(
        "--verbose",
        action="count",
        default=argparse.SUPPRESS,
        help=extract_help_from_doc(denovo, "verbose"),
    )
    parser_denovo.add_argument(
        "-q",
        "--quiet",
        action="store_true",
        default=argparse.SUPPRESS,
        help=extract_help_from_doc(denovo, "quiet"),
    )


    # NEW PARSER
    # build parser arguments
    build_desc = "Build a variant-containing reference sequence (VCRS) file."

    parser_build = parent_subparsers.add_parser(
        "build",
        parents=[parent],
        description=build_desc,
        # No `help=` on purpose: omitting it keeps `build` out of the top-level `vk --help` listing
        # while leaving it fully invocable (`vk build ...`, `vk build --help`) and usable as the
        # argument source for `vk ref`. `vk ref` is the public-facing name for this functionality.
        # epilog=vk_build_end_help_message,
        add_help=True,
        formatter_class=CustomHelpFormatter,
    )
    add_build_arguments(parser_build, required=not vk_ref_list_information_and_exit_flag_present)

    # NEW PARSER
    fastqpp_desc = "Preprocess the fastq files."
    parser_fastqpp = parent_subparsers.add_parser(
        "fastqpp",
        parents=[parent],
        description=fastqpp_desc,
        help=fastqpp_desc,
        add_help=True,
        formatter_class=CustomHelpFormatter,
    )
    parser_fastqpp.add_argument(
        "fastqs",
        nargs="+",
        help=extract_help_from_doc(fastqpp, "fastqs"),
    )
    parser_fastqpp.add_argument(
        "-x",
        "--technology",
        required=True,
        default=argparse.SUPPRESS,  # Remove from args if not provided
        help=extract_help_from_doc(fastqpp, "technology"),
    )
    parser_fastqpp.add_argument(
        "--multiplexed",
        action="store_true",
        default=argparse.SUPPRESS,  # Remove from args if not provided
        help=extract_help_from_doc(fastqpp, "multiplexed"),
    )
    parser_fastqpp.add_argument(
        "--parity",
        type=str,
        required=False,
        choices=["single", "paired"],
        default=argparse.SUPPRESS,  # Remove from args if not provided
        help=extract_help_from_doc(fastqpp, "parity"),
    )
    parser_fastqpp.add_argument(
        "--quality_control_fastqs",
        action="store_true",
        default=argparse.SUPPRESS,  # Remove from args if not provided
        help=extract_help_from_doc(fastqpp, "quality_control_fastqs"),
    )
    parser_fastqpp.add_argument(
        "--cut_front",
        action="store_true",
        default=argparse.SUPPRESS,  # Remove from args if not provided
        help=extract_help_from_doc(fastqpp, "cut_front"),
    )
    parser_fastqpp.add_argument(
        "--cut_tail",
        action="store_true",
        default=argparse.SUPPRESS,  # Remove from args if not provided
        help=extract_help_from_doc(fastqpp, "cut_tail"),
    )
    parser_fastqpp.add_argument(
        "--cut_window_size",
        type=int,
        default=argparse.SUPPRESS,  # Remove from args if not provided
        help=extract_help_from_doc(fastqpp, "cut_window_size"),
    )
    parser_fastqpp.add_argument(
        "--cut_mean_quality",
        type=int,
        default=argparse.SUPPRESS,  # Remove from args if not provided
        help=extract_help_from_doc(fastqpp, "cut_mean_quality"),
    )
    parser_fastqpp.add_argument(
        "--disable_adapter_trimming",
        action="store_true",
        default=argparse.SUPPRESS,  # Remove from args if not provided
        help=extract_help_from_doc(fastqpp, "disable_adapter_trimming"),
    )
    parser_fastqpp.add_argument(
        "--qualified_quality_phred",
        type=int,
        default=argparse.SUPPRESS,  # Remove from args if not provided
        help=extract_help_from_doc(fastqpp, "qualified_quality_phred"),
    )
    parser_fastqpp.add_argument(
        "--unqualified_percent_limit",
        type=int,
        default=argparse.SUPPRESS,  # Remove from args if not provided
        help=extract_help_from_doc(fastqpp, "unqualified_percent_limit"),
    )
    parser_fastqpp.add_argument(
        "--average_qual",
        type=int,
        default=argparse.SUPPRESS,  # Remove from args if not provided
        help=extract_help_from_doc(fastqpp, "average_qual"),
    )
    parser_fastqpp.add_argument(
        "--n_base_limit",
        type=int,
        default=argparse.SUPPRESS,  # Remove from args if not provided
        help=extract_help_from_doc(fastqpp, "n_base_limit"),
    )
    parser_fastqpp.add_argument(
        "--disable_quality_filtering",
        action="store_true",
        default=argparse.SUPPRESS,  # Remove from args if not provided
        help=extract_help_from_doc(fastqpp, "disable_quality_filtering"),
    )
    parser_fastqpp.add_argument(
        "--length_required",
        required=False,
        default=argparse.SUPPRESS,  # Remove from args if not provided
        help=extract_help_from_doc(fastqpp, "length_required"),
    )
    parser_fastqpp.add_argument(
        "--disable_length_filtering",
        action="store_true",
        default=argparse.SUPPRESS,  # Remove from args if not provided
        help=extract_help_from_doc(fastqpp, "disable_length_filtering"),
    )
    parser_fastqpp.add_argument(
        "--dont_eval_duplication",
        action="store_true",
        default=argparse.SUPPRESS,  # Remove from args if not provided
        help=extract_help_from_doc(fastqpp, "dont_eval_duplication"),
    )
    parser_fastqpp.add_argument(
        "--disable_trim_poly_g",
        action="store_true",
        default=argparse.SUPPRESS,  # Remove from args if not provided
        help=extract_help_from_doc(fastqpp, "disable_trim_poly_g"),
    )
    parser_fastqpp.add_argument(
        "--failed_out",
        default=argparse.SUPPRESS,  # Remove from args if not provided
        help=extract_help_from_doc(fastqpp, "failed_out"),
    )
    parser_fastqpp.add_argument(
        "--split_reads_by_Ns_and_low_quality_bases",
        action="store_true",
        default=argparse.SUPPRESS,  # Remove from args if not provided
        help=extract_help_from_doc(fastqpp, "split_reads_by_Ns_and_low_quality_bases"),
    )
    parser_fastqpp.add_argument(
        "--min_base_quality_for_splitting",
        type=int,
        required=False,
        default=argparse.SUPPRESS,  # Remove from args if not provided
        help=extract_help_from_doc(fastqpp, "min_base_quality_for_splitting"),
    )
    parser_fastqpp.add_argument(
        "--concatenate_paired_fastqs",
        action="store_true",
        default=argparse.SUPPRESS,  # Remove from args if not provided
        help=extract_help_from_doc(fastqpp, "concatenate_paired_fastqs"),
    )
    parser_fastqpp.add_argument(
        "-o",
        "--out",
        type=str,
        required=False,
        default=argparse.SUPPRESS,  # Remove from args if not provided
        help=extract_help_from_doc(fastqpp, "out"),
    )
    parser_fastqpp.add_argument(
        "--dry_run",
        action="store_true",
        default=argparse.SUPPRESS,  # Remove from args if not provided
        help=extract_help_from_doc(fastqpp, "dry_run"),
    )
    parser_fastqpp.add_argument(
        "--overwrite",
        action="store_true",
        default=argparse.SUPPRESS,  # Remove from args if not provided
        help=extract_help_from_doc(fastqpp, "overwrite"),
    )
    parser_fastqpp.add_argument(
        "--disable_sort_fastqs",
        dest="sort_fastqs",
        action="store_false",
        default=argparse.SUPPRESS,  # Remove from args if not provided
        help=extract_help_from_doc(fastqpp, "sort_fastqs", disable=True),
    )
    parser_fastqpp.add_argument(
        "--threads",
        type=int,
        default=argparse.SUPPRESS,  # Remove from args if not provided
        help=extract_help_from_doc(fastqpp, "threads"),
    )
    parser_fastqpp.add_argument(
        "--logging_level",
        choices=["NOTSET", "DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL", "0", "10", "20", "30", "40", "50", "60", None],
        required=False,
        default=argparse.SUPPRESS,  # Remove from args if not provided
        help=extract_help_from_doc(fastqpp, "logging_level"),
    )
    parser_fastqpp.add_argument(
        "--save_logs",
        action="store_true",
        default=argparse.SUPPRESS,  # Remove from args if not provided
        help=extract_help_from_doc(fastqpp, "save_logs"),
    )
    parser_fastqpp.add_argument(
        "--log_out_dir",
        required=False,
        default=argparse.SUPPRESS,  # Remove from args if not provided
        help=extract_help_from_doc(fastqpp, "log_out_dir"),
    )

    # kwargs
    parser_fastqpp.add_argument(
        "--seqtk_path",
        required=False,
        default=argparse.SUPPRESS,  # Remove from args if not provided
        help=extract_help_from_doc(fastqpp, "seqtk_path"),
    )
    parser_fastqpp.add_argument(
        "--quality_control_fastqs_out_dir",
        required=False,
        default=argparse.SUPPRESS,  # Remove from args if not provided
        help=extract_help_from_doc(fastqpp, "quality_control_fastqs_out_dir"),
    )
    parser_fastqpp.add_argument(
        "--replace_low_quality_bases_with_N_out_dir",
        required=False,
        default=argparse.SUPPRESS,  # Remove from args if not provided
        help=extract_help_from_doc(fastqpp, "replace_low_quality_bases_with_N_out_dir"),
    )
    parser_fastqpp.add_argument(
        "--split_by_Ns_and_low_quality_bases_out_dir",
        required=False,
        default=argparse.SUPPRESS,  # Remove from args if not provided
        help=extract_help_from_doc(fastqpp, "split_by_Ns_and_low_quality_bases_out_dir"),
    )
    parser_fastqpp.add_argument(
        "--concatenate_paired_fastqs_out_dir",
        required=False,
        default=argparse.SUPPRESS,  # Remove from args if not provided
        help=extract_help_from_doc(fastqpp, "concatenate_paired_fastqs_out_dir"),
    )
    parser_fastqpp.add_argument(
        "--delete_intermediate_files",
        action="store_true",
        default=argparse.SUPPRESS,  # Remove from args if not provided
        help=extract_help_from_doc(fastqpp, "delete_intermediate_files"),
    )

    # Hide the advanced `fastqpp` parameters (defined in varseek_fastqpp.vk_fastqpp_hidden_from_help)
    # from `vk fastqpp --help`. They remain fully parseable — only their help text is suppressed.
    for action in parser_fastqpp._actions:
        if action.dest in vk_fastqpp_hidden_from_help:
            action.help = argparse.SUPPRESS

    # NEW PARSER
    clean_desc = "Run standard processing on the VCRS count matrix."
    parser_clean = parent_subparsers.add_parser(
        "clean",
        parents=[parent],
        description=clean_desc,
        help=clean_desc,
        add_help=True,
        formatter_class=CustomHelpFormatter,
    )
    parser_clean.add_argument(
        "adata_vcrs",
        type=str,
        help=extract_help_from_doc(clean, "adata_vcrs"),
    )
    parser_clean.add_argument(
        "-x",
        "--technology",
        type=str,
        required=True,
        help=extract_help_from_doc(clean, "technology"),
    )
    parser_clean.add_argument(
        "--min_counts",
        type=int,
        required=False,
        default=argparse.SUPPRESS,  # Remove from args if not provided
        help=extract_help_from_doc(clean, "min_counts"),
    )
    parser_clean.add_argument(
        "--use_binary_matrix",
        action="store_true",
        default=argparse.SUPPRESS,  # Remove from args if not provided
        help=extract_help_from_doc(clean, "use_binary_matrix"),
    )
    parser_clean.add_argument(
        "--drop_empty_columns",
        action="store_true",
        default=argparse.SUPPRESS,  # Remove from args if not provided
        help=extract_help_from_doc(clean, "drop_empty_columns"),
    )
    parser_clean.add_argument(
        "--pseudobam_validation",
        action="store_true",
        default=argparse.SUPPRESS,  # Remove from args if not provided
        help=extract_help_from_doc(clean, "pseudobam_validation"),
    )
    parser_clean.add_argument(
        "--reference_genome_index",
        default=argparse.SUPPRESS,  # Remove from args if not provided
        help=extract_help_from_doc(clean, "reference_genome_index"),
    )
    parser_clean.add_argument(
        "--check_alignment_position",
        action="store_true",
        default=argparse.SUPPRESS,  # Remove from args if not provided
        help=extract_help_from_doc(clean, "check_alignment_position"),
    )
    parser_clean.add_argument(
        "--alignment_position_tolerance",
        type=int,
        default=argparse.SUPPRESS,  # Remove from args if not provided
        help=extract_help_from_doc(clean, "alignment_position_tolerance"),
    )
    parser_clean.add_argument(
        "--reference_sequences_type",
        choices=["rna", "dna"],
        default=argparse.SUPPRESS,  # Remove from args if not provided
        help=extract_help_from_doc(clean, "reference_sequences_type"),
    )
    parser_clean.add_argument(
        "--kallisto",
        default=argparse.SUPPRESS,  # Remove from args if not provided
        help=extract_help_from_doc(clean, "kallisto"),
    )
    parser_clean.add_argument(
        "--disable_count_reads_that_dont_pseudoalign_to_reference_genome",
        dest="count_reads_that_dont_pseudoalign_to_reference_genome",
        action="store_false",
        default=argparse.SUPPRESS,  # Remove from args if not provided
        help=extract_help_from_doc(clean, "count_reads_that_dont_pseudoalign_to_reference_genome", disable=True),
    )
    parser_clean.add_argument(
        "--avoid_paired_double_counting",
        action="store_true",
        default=argparse.SUPPRESS,  # Remove from args if not provided
        help=extract_help_from_doc(clean, "avoid_paired_double_counting"),
    )
    parser_clean.add_argument(
        "--account_for_strand_bias",
        action="store_true",
        default=argparse.SUPPRESS,  # Remove from args if not provided
        help=extract_help_from_doc(clean, "account_for_strand_bias"),
    )
    parser_clean.add_argument(
        "--strand_bias_end",
        type=str,
        choices=["5p", "3p"],
        required=False,
        default=argparse.SUPPRESS,  # Remove from args if not provided
        help=extract_help_from_doc(clean, "strand_bias_end"),
    )
    parser_clean.add_argument(
        "--read_length",
        required=False,
        default=argparse.SUPPRESS,  # Remove from args if not provided
        help=extract_help_from_doc(clean, "read_length"),
    )
    parser_clean.add_argument(
        "--filter_cells_by_min_counts",
        required=False,
        default=argparse.SUPPRESS,  # Remove from args if not provided
        help=extract_help_from_doc(clean, "filter_cells_by_min_counts"),
    )
    parser_clean.add_argument(
        "--filter_cells_by_min_genes",
        required=False,
        default=argparse.SUPPRESS,  # Remove from args if not provided
        help=extract_help_from_doc(clean, "filter_cells_by_min_genes"),
    )
    parser_clean.add_argument(
        "--filter_genes_by_min_cells",
        required=False,
        default=argparse.SUPPRESS,  # Remove from args if not provided
        help=extract_help_from_doc(clean, "filter_genes_by_min_cells"),
    )
    parser_clean.add_argument(
        "--filter_cells_by_max_mt_content",
        required=False,
        default=argparse.SUPPRESS,  # Remove from args if not provided
        help=extract_help_from_doc(clean, "filter_cells_by_max_mt_content"),
    )
    parser_clean.add_argument(
        "--doublet_detection",
        action="store_true",
        default=argparse.SUPPRESS,  # Remove from args if not provided
        help=extract_help_from_doc(clean, "doublet_detection"),
    )
    parser_clean.add_argument(
        "--remove_doublets",
        action="store_true",
        default=argparse.SUPPRESS,  # Remove from args if not provided
        help=extract_help_from_doc(clean, "remove_doublets"),
    )
    parser_clean.add_argument(
        "--cpm_normalization",
        action="store_true",
        default=argparse.SUPPRESS,  # Remove from args if not provided
        help=extract_help_from_doc(clean, "cpm_normalization"),
    )
    parser_clean.add_argument(
        "--sum_rows",
        action="store_true",
        default=argparse.SUPPRESS,  # Remove from args if not provided
        help=extract_help_from_doc(clean, "sum_rows"),
    )
    parser_clean.add_argument(
        "--drop_multi_variant_vcrs",
        action="store_true",
        default=argparse.SUPPRESS,  # Remove from args if not provided
        help=extract_help_from_doc(clean, "drop_multi_variant_vcrs"),
    )
    parser_clean.add_argument(
        "--disable_rename_vcrs_to_variant",
        dest="rename_vcrs_to_variant",
        action="store_false",
        default=argparse.SUPPRESS,  # Remove from args if not provided
        help=extract_help_from_doc(clean, "rename_vcrs_to_variant", disable=True),
    )
    parser_clean.add_argument(
        "--vcrs_id_set_to_exclusively_keep",
        nargs="+",
        type=str,
        required=False,
        help=extract_help_from_doc(clean, "vcrs_id_set_to_exclusively_keep"),
    )
    parser_clean.add_argument(
        "--vcrs_id_set_to_exclude",
        nargs="+",
        type=str,
        required=False,
        help=extract_help_from_doc(clean, "vcrs_id_set_to_exclude"),
    )
    parser_clean.add_argument(
        "--gene_set_to_exclusively_keep",
        nargs="+",
        type=str,
        required=False,
        help=extract_help_from_doc(clean, "gene_set_to_exclusively_keep"),
    )
    parser_clean.add_argument(
        "--gene_set_to_exclude",
        nargs="+",
        type=str,
        required=False,
        help=extract_help_from_doc(clean, "gene_set_to_exclude"),
    )
    parser_clean.add_argument(
        "--disable_mm",
        dest="mm",
        action="store_false",
        default=argparse.SUPPRESS,  # Remove from args if not provided
        help=extract_help_from_doc(clean, "mm", disable=True),
    )
    parser_clean.add_argument(
        "--parity",
        type=str,
        choices=["single", "paired"],
        required=False,
        default=argparse.SUPPRESS,  # Remove from args if not provided
        help=extract_help_from_doc(clean, "parity"),
    )
    parser_clean.add_argument(
        "--multiplexed",
        action="store_true",
        default=argparse.SUPPRESS,  # Remove from args if not provided
        help=extract_help_from_doc(clean, "multiplexed"),
    )
    parser_clean.add_argument(
        "--disable_sort_fastqs",
        dest="sort_fastqs",
        action="store_false",
        default=argparse.SUPPRESS,  # Remove from args if not provided
        help=extract_help_from_doc(clean, "sort_fastqs", disable=True),
    )
    parser_clean.add_argument(
        "--adata_reference_genome",
        required=False,
        default=argparse.SUPPRESS,  # Remove from args if not provided
        help=extract_help_from_doc(clean, "adata_reference_genome"),
    )
    parser_clean.add_argument(
        "--fastqs",
        nargs="+",
        required=False,
        help=extract_help_from_doc(clean, "fastqs"),
    )
    parser_clean.add_argument(
        "--vk_ref_dir",
        required=False,
        default=argparse.SUPPRESS,  # Remove from args if not provided
        help=extract_help_from_doc(clean, "vk_ref_dir"),
    )
    parser_clean.add_argument(
        "--vcrs_index",
        required=False,
        default=argparse.SUPPRESS,  # Remove from args if not provided
        help=extract_help_from_doc(clean, "vcrs_index"),
    )
    parser_clean.add_argument(
        "--vcrs_t2g",
        required=False,
        default=argparse.SUPPRESS,  # Remove from args if not provided
        help=extract_help_from_doc(clean, "vcrs_t2g"),
    )
    parser_clean.add_argument(
        "--gtf",
        required=False,
        default=argparse.SUPPRESS,  # Remove from args if not provided
        help=extract_help_from_doc(clean, "gtf"),
    )
    parser_clean.add_argument(
        "--kb_count_vcrs_dir",
        required=False,
        default=argparse.SUPPRESS,  # Remove from args if not provided
        help=extract_help_from_doc(clean, "kb_count_vcrs_dir"),
    )
    parser_clean.add_argument(
        "--kallisto_quant_reference_genome_dir",
        required=False,
        default=argparse.SUPPRESS,  # Remove from args if not provided
        help=extract_help_from_doc(clean, "kallisto_quant_reference_genome_dir"),
    )
    parser_clean.add_argument(
        "--reference_genome_t2g",
        required=False,
        default=argparse.SUPPRESS,  # Remove from args if not provided
        help=extract_help_from_doc(clean, "reference_genome_t2g"),
    )
    parser_clean.add_argument(
        "--vcf_data_dataframe",
        required=False,
        default=argparse.SUPPRESS,  # Remove from args if not provided
        help=extract_help_from_doc(clean, "vcf_data_dataframe"),
    )
    parser_clean.add_argument(
        "--variants",
        required=False,
        default=argparse.SUPPRESS,  # Remove from args if not provided
        help=extract_help_from_doc(clean, "variants"),
    )
    parser_clean.add_argument(
        "--sequences",
        required=False,
        default=argparse.SUPPRESS,  # Remove from args if not provided
        help=extract_help_from_doc(clean, "sequences"),
    )
    parser_clean.add_argument(
        "--variant_source",
        required=False,
        default=argparse.SUPPRESS,  # Remove from args if not provided
        help=extract_help_from_doc(clean, "variant_source"),
    )
    parser_clean.add_argument(
        "--vcrs_metadata_df",
        required=False,
        default=argparse.SUPPRESS,  # Remove from args if not provided
        help=extract_help_from_doc(clean, "vcrs_metadata_df"),
    )
    parser_clean.add_argument(
        "--variants_usecols",
        nargs="*",
        type=str,
        required=False,
        help=extract_help_from_doc(clean, "variants_usecols"),
    )
    parser_clean.add_argument(
        "--seq_id_column",
        required=False,
        default=argparse.SUPPRESS,  # Remove from args if not provided
        help=extract_help_from_doc(clean, "seq_id_column"),
    )
    parser_clean.add_argument(
        "--var_column",
        required=False,
        default=argparse.SUPPRESS,  # Remove from args if not provided
        help=extract_help_from_doc(clean, "var_column"),
    )
    parser_clean.add_argument(
        "--var_id_column",
        required=False,
        default=argparse.SUPPRESS,  # Remove from args if not provided
        help=extract_help_from_doc(clean, "var_id_column"),
    )
    parser_clean.add_argument(
        "--gene_id_column",
        required=False,
        default=argparse.SUPPRESS,  # Remove from args if not provided
        help=extract_help_from_doc(clean, "gene_id_column"),
    )
    parser_clean.add_argument(
        "--out",
        required=False,
        default=argparse.SUPPRESS,  # Remove from args if not provided
        help=extract_help_from_doc(clean, "out"),
    )
    parser_clean.add_argument(
        "--adata_vcrs_clean_out",
        required=False,
        default=argparse.SUPPRESS,  # Remove from args if not provided
        help=extract_help_from_doc(clean, "adata_vcrs_clean_out"),
    )
    parser_clean.add_argument(
        "--adata_reference_genome_clean_out",
        required=False,
        default=argparse.SUPPRESS,  # Remove from args if not provided
        help=extract_help_from_doc(clean, "adata_reference_genome_clean_out"),
    )
    parser_clean.add_argument(
        "--vcf_out",
        required=False,
        default=argparse.SUPPRESS,  # Remove from args if not provided
        help=extract_help_from_doc(clean, "vcf_out"),
    )
    parser_clean.add_argument(
        "--save_vcf",
        action="store_true",
        default=argparse.SUPPRESS,  # Remove from args if not provided
        help=extract_help_from_doc(clean, "save_vcf"),
    )
    parser_clean.add_argument(
        "--save_vcf_samples",
        action="store_true",
        default=argparse.SUPPRESS,  # Remove from args if not provided
        help=extract_help_from_doc(clean, "save_vcf_samples"),
    )
    parser_clean.add_argument(
        "--chunksize",
        type=int,
        default=argparse.SUPPRESS,  # Remove from args if not provided
        help=extract_help_from_doc(clean, "chunksize"),
    )
    parser_clean.add_argument(
        "--dry_run",
        action="store_true",
        default=argparse.SUPPRESS,  # Remove from args if not provided
        help=extract_help_from_doc(clean, "dry_run"),
    )
    parser_clean.add_argument(
        "--overwrite",
        action="store_true",
        default=argparse.SUPPRESS,  # Remove from args if not provided
        help=extract_help_from_doc(clean, "overwrite"),
    )
    parser_clean.add_argument(
        "--logging_level",
        choices=["NOTSET", "DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL", "0", "10", "20", "30", "40", "50", "60", None],
        required=False,
        default=argparse.SUPPRESS,  # Remove from args if not provided
        help=extract_help_from_doc(clean, "logging_level"),
    )
    parser_clean.add_argument(
        "--save_logs",
        action="store_true",
        default=argparse.SUPPRESS,  # Remove from args if not provided
        help=extract_help_from_doc(clean, "save_logs"),
    )
    parser_clean.add_argument(
        "--log_out_dir",
        required=False,
        default=argparse.SUPPRESS,  # Remove from args if not provideds
        help=extract_help_from_doc(clean, "log_out_dir"),
    )
    # kwargs
    parser_clean.add_argument(
        "--variants_updated_dataframe",
        required=False,
        default=argparse.SUPPRESS,  # Remove from args if not provided
        help=extract_help_from_doc(clean, "variants_updated_dataframe"),
    )
    parser_clean.add_argument(
        "--bustools",
        required=False,
        default=argparse.SUPPRESS,  # Remove from args if not provided
        help=extract_help_from_doc(clean, "bustools"),
    )
    parser_clean.add_argument(
        "--parity_kb_count",
        type=str,
        choices=["single", "paired"],
        required=False,
        default=argparse.SUPPRESS,  # Remove from args if not provided
        help=extract_help_from_doc(clean, "parity_kb_count"),
    )
    parser_clean.add_argument(
        "--cosmic_tsv",
        required=False,
        default=argparse.SUPPRESS,  # Remove from args if not provided
        help=extract_help_from_doc(clean, "cosmic_tsv"),
    )
    parser_clean.add_argument(
        "--cosmic_reference_genome_fasta",
        required=False,
        default=argparse.SUPPRESS,  # Remove from args if not provided
        help=extract_help_from_doc(clean, "cosmic_reference_genome_fasta"),
    )
    parser_clean.add_argument(
        "--cosmic_version",
        required=False,
        default=argparse.SUPPRESS,  # Remove from args if not provided
        help=extract_help_from_doc(clean, "cosmic_version"),
    )
    parser_clean.add_argument(
        "--forgiveness",
        required=False,
        default=argparse.SUPPRESS,  # Remove from args if not provided
        help=extract_help_from_doc(clean, "forgiveness"),
    )
    parser_clean.add_argument(
        "--disable_add_hgvs_breakdown_to_adata_var",
        dest="add_hgvs_breakdown_to_adata_var",
        action="store_false",
        default=argparse.SUPPRESS,  # Remove from args if not provided
        help=extract_help_from_doc(clean, "add_hgvs_breakdown_to_adata_var", disable=True),
    )

    # Hide the advanced `clean` parameters (defined in varseek_clean.vk_clean_hidden_from_help) from
    # `vk clean --help`. They remain fully parseable — only their help text is suppressed. Matched by
    # dest, so the --disable_* flags map to their underlying parameter name.
    for action in parser_clean._actions:
        if action.dest in vk_clean_hidden_from_help:
            action.help = argparse.SUPPRESS

    # NEW PARSER
    summarize_desc = "Analyze the VCRS count matrix results."
    parser_summarize = parent_subparsers.add_parser(
        "summarize",
        parents=[parent],
        description=summarize_desc,
        help=summarize_desc,
        add_help=True,
        formatter_class=CustomHelpFormatter,
    )
    parser_summarize.add_argument(
        "adata",
        type=str,
        help=extract_help_from_doc(summarize, "adata"),
    )
    parser_summarize.add_argument(
        "-t",
        "--top_values",
        type=int,
        required=False,
        default=argparse.SUPPRESS,  # Remove from args if not provided
        help=extract_help_from_doc(summarize, "top_values"),
    )
    parser_summarize.add_argument(
        "-x",
        "--technology",
        required=False,
        default=argparse.SUPPRESS,  # Remove from args if not provided
        help=extract_help_from_doc(summarize, "technology"),
    )
    parser_summarize.add_argument(
        "--gene_name_column",
        required=False,
        default=argparse.SUPPRESS,  # Remove from args if not provided
        help=extract_help_from_doc(summarize, "gene_name_column"),
    )
    parser_summarize.add_argument(
        "-o",
        "--out",
        required=False,
        default=argparse.SUPPRESS,  # Remove from args if not provided
        help=extract_help_from_doc(summarize, "out"),
    )
    parser_summarize.add_argument(
        "--dry_run",
        action="store_true",
        default=argparse.SUPPRESS,  # Remove from args if not provided
        help=extract_help_from_doc(summarize, "dry_run"),
    )
    parser_summarize.add_argument(
        "--overwrite",
        action="store_true",
        default=argparse.SUPPRESS,  # Remove from args if not provided
        help=extract_help_from_doc(summarize, "overwrite"),
    )
    parser_summarize.add_argument(
        "--logging_level",
        choices=["NOTSET", "DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL", "0", "10", "20", "30", "40", "50", "60", None],
        required=False,
        default=argparse.SUPPRESS,  # Remove from args if not provided
        help=extract_help_from_doc(summarize, "logging_level"),
    )
    parser_summarize.add_argument(
        "--save_logs",
        action="store_true",
        default=argparse.SUPPRESS,  # Remove from args if not provided
        help=extract_help_from_doc(summarize, "save_logs"),
    )
    parser_summarize.add_argument(
        "--log_out_dir",
        required=False,
        default=argparse.SUPPRESS,  # Remove from args if not provided
        help=extract_help_from_doc(summarize, "log_out_dir"),
    )
    parser_summarize.add_argument(
        "--plot_strand_bias",
        action="store_true",
        default=argparse.SUPPRESS,  # Remove from args if not provided
        help=extract_help_from_doc(summarize, "plot_strand_bias"),
    )
    parser_summarize.add_argument(
        "--strand_bias_end",
        required=False,
        choices=["auto", "both", "5p", "3p"],
        default=argparse.SUPPRESS,  # Remove from args if not provided
        help=extract_help_from_doc(summarize, "strand_bias_end"),
    )
    parser_summarize.add_argument(
        "--cdna_fasta",
        required=False,
        default=argparse.SUPPRESS,  # Remove from args if not provided
        help=extract_help_from_doc(summarize, "cdna_fasta"),
    )
    parser_summarize.add_argument(
        "--seq_id_cdna_column",
        required=False,
        default=argparse.SUPPRESS,  # Remove from args if not provided
        help=extract_help_from_doc(summarize, "seq_id_cdna_column"),
    )
    parser_summarize.add_argument(
        "--start_variant_position_cdna_column",
        required=False,
        default=argparse.SUPPRESS,  # Remove from args if not provided
        help=extract_help_from_doc(summarize, "start_variant_position_cdna_column"),
    )
    parser_summarize.add_argument(
        "--end_variant_position_cdna_column",
        required=False,
        default=argparse.SUPPRESS,  # Remove from args if not provided
        help=extract_help_from_doc(summarize, "end_variant_position_cdna_column"),
    )
    parser_summarize.add_argument(
        "--read_length",
        required=False,
        default=argparse.SUPPRESS,  # Remove from args if not provided
        help=extract_help_from_doc(summarize, "read_length"),
    )

    # kwargs
    parser_summarize.add_argument(
        "--stats_file",
        required=False,
        default=argparse.SUPPRESS,  # Remove from args if not provided
        help=extract_help_from_doc(summarize, "stats_file"),
    )
    parser_summarize.add_argument(
        "--specific_stats_folder",
        required=False,
        default=argparse.SUPPRESS,  # Remove from args if not provided
        help=extract_help_from_doc(summarize, "specific_stats_folder"),
    )
    parser_summarize.add_argument(
        "--plots_folder",
        required=False,
        default=argparse.SUPPRESS,  # Remove from args if not provided
        help=extract_help_from_doc(summarize, "plots_folder"),
    )

    # Hide the advanced `summarize` parameters (defined in varseek_summarize.vk_summarize_hidden_from_help)
    # from `vk summarize --help`. They remain fully parseable — only their help text is suppressed.
    for action in parser_summarize._actions:
        if action.dest in vk_summarize_hidden_from_help:
            action.help = argparse.SUPPRESS

    # NEW PARSER
    ref_desc = "Create a reference index and t2g file for variant screening with varseek count. Wraps around varseek build and kb ref."
    parser_ref = parent_subparsers.add_parser("ref", parents=[parent], description=ref_desc, help=ref_desc, add_help=True, formatter_class=CustomHelpFormatter, epilog="To see the full list of allowable arguments, please explore vk build and (kallisto-bustools') kb ref")

    add_build_arguments(parser_ref, required=not vk_ref_list_information_and_exit_flag_present)

    # NEW PARSER
    count_desc = "Perform variant screening on sequencing data. Wraps around varseek fastqpp, kb count, varseek clean, and vk summarize."
    parser_count = parent_subparsers.add_parser("count", parents=[parent], description=count_desc, help=count_desc, add_help=True, formatter_class=CustomHelpFormatter, epilog="To see the full list of allowable arguments, please explore vk fastqpp, vk clean, vk summarize, and (kallisto-bustools') kb count")
    parser_count.add_argument(
        "fastqs",
        nargs="+",
        help=extract_help_from_doc(count, "fastqs"),
    )
    parser_count.add_argument(
        "-i",
        "--index",
        required=True,
        help=extract_help_from_doc(count, "index"),
    )
    parser_count.add_argument(
        "-g",
        "--t2g",
        required=True,
        help=extract_help_from_doc(count, "t2g"),
    )
    parser_count.add_argument(
        "-x",
        "--technology",
        required=True,
        help=extract_help_from_doc(count, "technology"),
    )
    parser_count.add_argument(
        "-k",
        "--k",
        type=int,
        required=False,
        default=argparse.SUPPRESS,  # Remove from args if not provided
        help=extract_help_from_doc(count, "k"),
    )
    parser_count.add_argument(
        "--pseudobam_validation",
        action="store_true",
        default=argparse.SUPPRESS,  # Remove from args if not provided
        help=extract_help_from_doc(count, "pseudobam_validation"),
    )
    parser_count.add_argument(
        "--reference_sequences_type",
        choices=["rna", "dna"],
        default=argparse.SUPPRESS,  # Remove from args if not provided
        help=extract_help_from_doc(clean, "reference_sequences_type"),
    )
    parser_count.add_argument(
        "--alignment_position_tolerance",
        type=int,
        default=argparse.SUPPRESS,  # Remove from args if not provided
        help=extract_help_from_doc(clean, "alignment_position_tolerance"),
    )
    parser_count.add_argument(
        "--check_alignment_position",
        action="store_true",
        default=argparse.SUPPRESS,  # Remove from args if not provided
        help=extract_help_from_doc(clean, "check_alignment_position"),
    )
    parser_count.add_argument(
        "--kallisto",
        default=argparse.SUPPRESS,  # Remove from args if not provided
        help=extract_help_from_doc(clean, "kallisto"),
    )
    parser_count.add_argument(
        "--account_for_strand_bias",
        action="store_true",
        default=argparse.SUPPRESS,  # Remove from args if not provided
        help=extract_help_from_doc(count, "account_for_strand_bias"),
    )
    parser_count.add_argument(
        "--strand_bias_end",
        type=str,
        choices=["5p", "3p"],
        required=False,
        default=argparse.SUPPRESS,  # Remove from args if not provided
        help=extract_help_from_doc(count, "strand_bias_end"),
    )
    parser_count.add_argument(
        "--read_length",
        required=False,
        default=argparse.SUPPRESS,  # Remove from args if not provided
        help=extract_help_from_doc(count, "read_length"),
    )
    parser_count.add_argument(
        "--strand",
        type=str,
        required=False,
        default=argparse.SUPPRESS,  # Remove from args if not provided
        help=extract_help_from_doc(count, "strand"),
    )
    parser_count.add_argument(
        "--disable_mm",
        dest="mm",
        action="store_false",
        default=argparse.SUPPRESS,  # Remove from args if not provided
        help=extract_help_from_doc(count, "mm", disable=True),
    )
    parser_count.add_argument(
        "--disable_union",
        dest="union",
        action="store_false",
        default=argparse.SUPPRESS,  # Remove from args if not provided
        help=extract_help_from_doc(count, "union", disable=True),
    )
    parser_count.add_argument(
        "--parity",
        type=str,
        choices=["single", "paired"],
        required=False,
        default=argparse.SUPPRESS,  # Remove from args if not provided
        help=extract_help_from_doc(count, "parity"),
    )
    parser_count.add_argument(
        "--reference_genome_index",
        required=False,
        default=argparse.SUPPRESS,  # Remove from args if not provided
        help=extract_help_from_doc(clean, "reference_genome_index"),  # forwarded to vk clean (used by pseudobam_validation)
    )
    parser_count.add_argument(
        "--gtf",
        required=False,
        default=argparse.SUPPRESS,  # Remove from args if not provided
        help=extract_help_from_doc(count, "gtf"),
    )
    parser_count.add_argument(
        "-o",
        "--out",
        type=str,
        required=False,
        default=argparse.SUPPRESS,  # Remove from args if not provided
        help=extract_help_from_doc(count, "out"),
    )
    parser_count.add_argument(
        "--kb_count_vcrs_out_dir",
        required=False,
        default=argparse.SUPPRESS,  # Remove from args if not provided
        help=extract_help_from_doc(count, "kb_count_vcrs_out_dir"),
    )
    parser_count.add_argument(
        "--vk_summarize_out_dir",
        required=False,
        default=argparse.SUPPRESS,  # Remove from args if not provided
        help=extract_help_from_doc(count, "vk_summarize_out_dir"),
    )
    parser_count.add_argument(
        "--disable_fastqpp",
        action="store_true",
        default=argparse.SUPPRESS,  # Remove from args if not provided
        help=extract_help_from_doc(count, "disable_fastqpp"),  # not disable=True because the acual python argument is named disable_fastqpp
    )
    parser_count.add_argument(
        "--disable_clean",
        action="store_true",
        default=argparse.SUPPRESS,  # Remove from args if not provided
        help=extract_help_from_doc(count, "disable_clean"),
    )
    parser_count.add_argument(
        "--summarize",
        action="store_true",
        default=argparse.SUPPRESS,  # Remove from args if not provided
        help=extract_help_from_doc(count, "summarize"),
    )
    parser_count.add_argument(
        "--chunksize",
        type=int,
        default=argparse.SUPPRESS,  # Remove from args if not provided
        help=extract_help_from_doc(count, "chunksize"),
    )
    parser_count.add_argument(
        "--dry_run",
        action="store_true",
        default=argparse.SUPPRESS,  # Remove from args if not provided
        help=extract_help_from_doc(count, "dry_run"),
    )
    parser_count.add_argument(
        "--overwrite",
        action="store_true",
        default=argparse.SUPPRESS,  # Remove from args if not provided
        help=extract_help_from_doc(count, "overwrite"),
    )
    parser_count.add_argument(
        "--disable_sort_fastqs",
        dest="sort_fastqs",
        action="store_false",
        default=argparse.SUPPRESS,  # Remove from args if not provided
        help=extract_help_from_doc(count, "sort_fastqs", disable=True),
    )
    parser_count.add_argument(
        "-t",
        "--threads",
        type=int,
        required=False,
        default=argparse.SUPPRESS,  # Remove from args if not provided
        help=extract_help_from_doc(count, "threads"),
    )
    parser_count.add_argument(
        "--logging_level",
        choices=["NOTSET", "DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL", "0", "10", "20", "30", "40", "50", "60", None],
        required=False,
        default=argparse.SUPPRESS,  # Remove from args if not provided
        help=extract_help_from_doc(count, "logging_level"),
    )
    parser_count.add_argument(
        "--save_logs",
        action="store_true",
        default=argparse.SUPPRESS,  # Remove from args if not provided
        help=extract_help_from_doc(count, "save_logs"),
    )
    parser_count.add_argument(
        "--log_out_dir",
        required=False,
        default=argparse.SUPPRESS,  # Remove from args if not provided
        help=extract_help_from_doc(count, "log_out_dir"),
    )

    # kwargs
    parser_count.add_argument(
        "--disable_num",
        action="store_false",
        dest="num",
        default=argparse.SUPPRESS,  # Remove from args if not provided
        help=extract_help_from_doc(count, "num", disable=True),
    )
    parser_count.add_argument(
        "--parity_kb_count",
        type=str,
        choices=["single", "paired"],
        required=False,
        default=argparse.SUPPRESS,  # Remove from args if not provided
        help=extract_help_from_doc(count, "parity_kb_count"),
    )

    copy_arguments([parser_build], parser_ref)  # allows parser_ref to accept all arguments from parser_build - if parser_ref does not already contain the argument, then it will inherent from vk build, notably with positional args converted into keyword args, required False (since vk ref will necessarily handle this internally), and no help message displayed
    copy_arguments([parser_fastqpp, parser_clean, parser_summarize], parser_count)

    # kb count args with 1 value - note that because varseek count has the positional fastqs with nargs="+", unknown_args only works with a flag with no following value (if an unknown flag has a following value, then it assumes this refers to the fastqs) - so for varseek ref (all kb args), and for varseek count with kb's store_true args, I don't need to add anything to main (kwargs will handle it)
    for flag in varseek_count_only_allowable_kb_count_arguments["one_argument"]:
        if flag not in parser_count._option_string_actions:  # Skip if already present
            parser_count.add_argument(
                flag,
                required=False,
                default=argparse.SUPPRESS,  # Remove from args if not provided
                help=argparse.SUPPRESS,  # don't show help
            )

    # $ to continue adding support for future kb ref/count args, simply update the dicts varseek_ref_only_allowable_kb_ref_arguments and varseek_count_only_allowable_kb_count_arguments, and ensure that they follow the current rules (0 args is store_true as handled by kwargs, 1 arg is a single string, 2+ args is a list of strings) - and if anything differs from here, then in addition to adding to the dict, then also add the logic custom here
    # manually add any non-standard flags from kb ref/count to varseek ref/count here (e.g., store_false)

    # Hide the advanced `count` parameters (defined in varseek_count.vk_count_hidden_from_help) from
    # `vk count --help`. They remain fully parseable — only their help text is suppressed. Matched by
    # dest, so the --disable_num flag maps to its underlying parameter name.
    for action in parser_count._actions:
        if action.dest in vk_count_hidden_from_help:
            action.help = argparse.SUPPRESS

    # * Define return values
    args, unknown_args = parent_parser.parse_known_args()

    # assumes everything either has one or more values or is a store_true - this can be confirmed with reviewing the args in https://github.com/pachterlab/kb_python/blob/786b2e1772dfa675b8165a35cfb06b67951f7960/kb_python/main.py#L1791, but is subject to change - either way, when a new release of kb python comes out with new args, I should review the flags to ensure that they either take 1+ args or are store_true (otherwise write support for these other types), and I must add them as being supported in vk ref/count
    kwargs = {}
    i = 0
    while i < len(unknown_args):
        # Remove leading '--'
        arg = unknown_args[i].lstrip("--")
        values = []

        # Look ahead to gather consecutive non-flag values
        while i + 1 < len(unknown_args) and not unknown_args[i + 1].startswith("--"):
            values.append(unknown_args[i + 1])
            i += 1  # Move forward

        # Assign the value:
        # - If no values were found, store as True (store_true behavior)
        # - If one value, store as a single string
        # - If multiple values, store as a list
        if len(values) == 0:
            kwargs[arg] = True  # store_true
        elif len(values) == 1:
            kwargs[arg] = values[0]  # single string
        else:  # len(values) > 1
            kwargs[arg] = values  # list

        i += 1  # Move to the next flag

    # Help return
    if args.help:
        parent_parser.print_help(sys.stderr)
        sys.exit(1)

    # Version return
    if args.version:
        print(f"varseek version: {__version__}")
        sys.exit(1)

    # Show help when no arguments are given
    if len(sys.argv) == 1:
        parent_parser.print_help(sys.stderr)
        sys.exit(1)

    # Show  module specific help if only module but no further arguments are given
    command_to_parser = {
        "build": parser_build,
        "denovo": parser_denovo,
        "fastqpp": parser_fastqpp,
        "clean": parser_clean,
        "summarize": parser_summarize,
        "ref": parser_ref,
        "count": parser_count,
    }

    if len(sys.argv) == 2:
        if sys.argv[1] in command_to_parser:
            command_to_parser[sys.argv[1]].print_help(sys.stderr)
        else:
            parent_parser.print_help(sys.stderr)
        sys.exit(1)

    params_dict = vars(args).copy()
    # remove special keys i.e., command, help, version
    for special_key in ["command", "help", "version"]:
        params_dict.pop(special_key, None)

    # st()

    # * build return
    if args.command == "build":
        if isinstance(args.sequences, list) and len(args.sequences) == 1:
            sequences = args.sequences[0]
        else:
            sequences = args.sequences

        if isinstance(args.variants, list) and len(args.variants) == 1:
            variants = args.variants[0]
        else:
            variants = args.variants

        # * ensure that all keys in params_dict correspond to the python parameters, and the values correspond to the command line values - the default is to pull from args
        # * I must override cases where I modify a variable outside of args (e.g., how I set sequences and variants above):
        # * I used to call replace_old_arg_names_with_new_arg_names_in_params_dict_and_combine_with_kwargs to update command line name with python name, but now I just have dest handle this for each argument (dest is the python argument name)
        # * I also use default=argparse.SUPPRESS to remove the argument from the args if not provided

        # modify variable outside of args
        params_dict["sequences"] = sequences
        params_dict["variants"] = variants

        # combine with kwargs (if both params_dict and kwargs have the same key, params_dict takes precedence)
        params_dict = {**kwargs, **params_dict}

        # for pytest
        if os.getenv("TESTING") == "true":
            return params_dict

        build_results = build(**params_dict)

        # Print list of mutated sequences if any are returned (this should only happen when return_variant_output=True).
        # Note: build now returns a dict of produced reference file paths in the normal case, which we do not print here.
        if isinstance(build_results, list):
            for mut_seq in build_results:
                print(mut_seq)

    # * denovo return
    if args.command == "denovo":
        params_dict = {**kwargs, **params_dict}

        # for pytest
        if os.getenv("TESTING") == "true":
            return params_dict

        denovo_results = denovo(**params_dict)

    # * fastqpp return
    if args.command == "fastqpp":
        # * ensure that all keys in params_dict correspond to the python parameters, and the values correspond to the command line values - see the vk build section in main for more details

        # (1) modify variable outside of args

        # combine with kwargs (if both params_dict and kwargs have the same key, params_dict takes precedence)
        params_dict = {**kwargs, **params_dict}

        # for pytest
        if os.getenv("TESTING") == "true":
            return params_dict

        fastqpp_results = fastqpp(**params_dict)

        # * optionally do something with fastqpp_results (e.g., save, or print to console)

    # * clean return
    if args.command == "clean":
        # * ensure that all keys in params_dict correspond to the python parameters, and the values correspond to the command line values - see the vk build section in main for more details

        # (1) modify variable outside of args

        # combine with kwargs (if both params_dict and kwargs have the same key, params_dict takes precedence)
        params_dict = {**kwargs, **params_dict}

        # for pytest
        if os.getenv("TESTING") == "true":
            return params_dict

        clean_results = clean(**params_dict)

        # * optionally do something with clean_results (e.g., save, or print to console)

    # * summarize return
    if args.command == "summarize":
        # * ensure that all keys in params_dict correspond to the python parameters, and the values correspond to the command line values - see the vk build section in main for more details

        # (1) modify variable outside of args

        # combine with kwargs (if both params_dict and kwargs have the same key, params_dict takes precedence)
        params_dict = {**kwargs, **params_dict}

        # for pytest
        if os.getenv("TESTING") == "true":
            return params_dict

        summarize_results = summarize(**params_dict)

        # * optionally do something with summarize_results (e.g., save, or print to console)

    # * ref return
    if args.command == "ref":
        if isinstance(args.sequences, list) and len(args.sequences) == 1:
            sequences = args.sequences[0]
        else:
            sequences = args.sequences

        if isinstance(args.variants, list) and len(args.variants) == 1:
            variants = args.variants[0]
        else:
            variants = args.variants

        # * ensure that all keys in params_dict correspond to the python parameters, and the values correspond to the command line values - see the vk build section in main for more details

        # (1) modify variable outside of args
        params_dict["sequences"] = sequences
        params_dict["variants"] = variants

        # combine with kwargs (if both params_dict and kwargs have the same key, params_dict takes precedence)
        params_dict = {**kwargs, **params_dict}

        # for pytest
        if os.getenv("TESTING") == "true":
            return params_dict

        ref_results = ref(**params_dict)

        # * optionally do something with ref_results (e.g., save, or print to console)

    # * count return
    if args.command == "count":
        # * ensure that all keys in params_dict correspond to the python parameters, and the values correspond to the command line values - see the vk build section in main for more details

        # combine with kwargs (if both params_dict and kwargs have the same key, params_dict takes precedence)
        params_dict = {**kwargs, **params_dict}

        # (1) fastqs is the positional argument, so pull it out and pass it positionally
        fastqs = params_dict.pop("fastqs")

        # for pytest
        if os.getenv("TESTING") == "true":
            return fastqs, params_dict

        count_results = count(fastqs, **params_dict)

        # * optionally do something with count_results (e.g., save, or print to console)


# * new Feb 2025 for pytest
if __name__ == "__main__":
    main()
