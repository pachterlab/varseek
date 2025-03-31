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
from .varseek_build import build
from .varseek_clean import clean
from .varseek_count import count
from .varseek_fastqpp import fastqpp
from .varseek_filter import filter
from .varseek_info import info
from .varseek_ref import ref
from .varseek_sim import sim
from .varseek_summarize import summarize

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
    # Initiate subparsers
    parent_subparsers = parent_parser.add_subparsers(dest="command")
    # Define parent (not sure why I need both parent parser and parent, but otherwise it does not work)
    parent = argparse.ArgumentParser(add_help=False)

    # Add custom help argument to parent parser
    parent_parser.add_argument("-h", "--help", action="store_true", help="Print manual.")
    # Add custom version argument to parent parser
    parent_parser.add_argument("-v", "--version", action="store_true", help="Print version.")


    # Check if a flag is passed that causes a script to exist early, thus making normally required arguments optiona (eg I can call vk build --list_prebuilt_indices without providing -s and -v)
    list_information_and_exit_flag_dict = {}
    for list_information_and_exit_flag in ("list_prebuilt_indices", "list_columns", "list_filter_rules", "list_downloadable_references"):
        list_information_and_exit_flag_dict[list_information_and_exit_flag] = False
        for i, arg in enumerate(sys.argv):
            if arg == f"--{list_information_and_exit_flag}":
                list_information_and_exit_flag_dict[list_information_and_exit_flag] = True
                break
    
    vk_build_list_information_and_exit_flag_present = list_information_and_exit_flag_dict["list_prebuilt_indices"] 
    vk_info_list_information_and_exit_flag_present = list_information_and_exit_flag_dict["list_columns"]
    vk_filter_list_information_and_exit_flag_present = list_information_and_exit_flag_dict["list_filter_rules"] 
    vk_ref_list_information_and_exit_flag_present = any(list_information_and_exit_flag_dict.values())


    # NEW PARSER
    # build parser arguments
    build_desc = "Build a variant-containing reference sequence (VCRS) file."

    parser_build = parent_subparsers.add_parser(
        "build",
        parents=[parent],
        description=build_desc,
        help=build_desc,
        # epilog=vk_build_end_help_message,
        add_help=True,
        formatter_class=CustomHelpFormatter,
    )
    parser_build.add_argument(
        "-v",
        "--variants",
        # type=strpath_or_str_or_list_or_df,
        nargs="+",
        required=not vk_build_list_information_and_exit_flag_present,
        help=extract_help_from_doc(build, "variants"),
    )
    parser_build.add_argument(
        "-s",
        "--sequences",
        type=str,
        nargs="+",
        required=not vk_build_list_information_and_exit_flag_present,
        help=extract_help_from_doc(build, "sequences"),
    )
    parser_build.add_argument(
        "-w",
        "--w",
        type=int,
        required=False,
        default=argparse.SUPPRESS,  # Remove from args if not provided
        help=extract_help_from_doc(build, "w"),
    )
    parser_build.add_argument(
        "-k",
        "--k",
        type=int,
        required=False,
        default=argparse.SUPPRESS,  # Remove from args if not provided
        help=extract_help_from_doc(build, "k"),
    )
    parser_build.add_argument(
        "-ma",
        "--max_ambiguous",
        type=int,
        required=False,
        default=argparse.SUPPRESS,  # Remove from args if not provided
        help=extract_help_from_doc(build, "max_ambiguous"),
    )
    parser_build.add_argument(
        "-vc",
        "--var_column",
        type=str,
        required=False,
        default=argparse.SUPPRESS,  # Remove from args if not provided
        help=extract_help_from_doc(build, "var_column"),
    )
    parser_build.add_argument(
        "-sic",
        "--seq_id_column",
        type=str,
        required=False,
        default=argparse.SUPPRESS,  # Remove from args if not provided
        help=extract_help_from_doc(build, "seq_id_column"),
    )
    parser_build.add_argument(
        "-vic",
        "--var_id_column",
        required=False,
        default=argparse.SUPPRESS,  # Remove from args if not provided
        help=extract_help_from_doc(build, "var_id_column"),
    )
    parser_build.add_argument(
        "-gtf",
        "--gtf",
        required=False,
        default=argparse.SUPPRESS,  # Remove from args if not provided
        help=extract_help_from_doc(build, "gtf"),
    )
    parser_build.add_argument(
        "-gtic",
        "--gtf_transcript_id_column",
        required=False,
        default=argparse.SUPPRESS,  # Remove from args if not provided
        help=extract_help_from_doc(build, "gtf_transcript_id_column"),
    )
    parser_build.add_argument(
        "--transcript_boundaries",
        action="store_true",
        default=argparse.SUPPRESS,  # Remove from args if not provided
        help=extract_help_from_doc(build, "transcript_boundaries"),
    )
    parser_build.add_argument(
        "--identify_all_spliced_from_genome",
        action="store_true",
        default=argparse.SUPPRESS,  # Remove from args if not provided
        help=extract_help_from_doc(build, "identify_all_spliced_from_genome"),
    )
    parser_build.add_argument(
        "-o",
        "--out",
        type=str,
        required=False,
        default=argparse.SUPPRESS,  # Remove from args if not provided
        help=extract_help_from_doc(build, "out"),
    )
    parser_build.add_argument(
        "-r",
        "--reference_out_dir",
        required=False,
        default=argparse.SUPPRESS,  # Remove from args if not provided
        help=extract_help_from_doc(build, "reference_out_dir"),
    )
    parser_build.add_argument(
        "--vcrs_fasta_out",
        required=False,
        default=argparse.SUPPRESS,  # Remove from args if not provided
        help=extract_help_from_doc(build, "vcrs_fasta_out"),
    )
    parser_build.add_argument(
        "--variants_updated_csv_out",
        required=False,
        default=argparse.SUPPRESS,  # Remove from args if not provided
        help=extract_help_from_doc(build, "variants_updated_csv_out"),
    )
    parser_build.add_argument(
        "--id_to_header_csv_out",
        required=False,
        default=argparse.SUPPRESS,  # Remove from args if not provided
        help=extract_help_from_doc(build, "id_to_header_csv_out"),
    )
    parser_build.add_argument(
        "--vcrs_t2g_out",
        required=False,
        default=argparse.SUPPRESS,  # Remove from args if not provided
        help=extract_help_from_doc(build, "vcrs_t2g_out"),
    )
    parser_build.add_argument(
        "--wt_vcrs_fasta_out",
        required=False,
        default=argparse.SUPPRESS,  # Remove from args if not provided
        help=extract_help_from_doc(build, "wt_vcrs_fasta_out"),
    )
    parser_build.add_argument(
        "--wt_vcrs_t2g_out",
        required=False,
        default=argparse.SUPPRESS,  # Remove from args if not provided
        help=extract_help_from_doc(build, "wt_vcrs_t2g_out"),
    )
    parser_build.add_argument(
        "--removed_variants_text_out",
        required=False,
        default=argparse.SUPPRESS,  # Remove from args if not provided
        help=extract_help_from_doc(build, "removed_variants_text_out"),
    )
    parser_build.add_argument(
        "--filtering_report_text_out",
        required=False,
        default=argparse.SUPPRESS,  # Remove from args if not provided
        help=extract_help_from_doc(build, "filtering_report_text_out"),
    )
    parser_build.add_argument(
        "--return_variant_output",
        action="store_true",
        default=argparse.SUPPRESS,  # Remove from args if not provided
        help=extract_help_from_doc(build, "return_variant_output"),
    )
    parser_build.add_argument(
        "--save_variants_updated_csv",
        action="store_true",
        default=argparse.SUPPRESS,  # Remove from args if not provided
        help=extract_help_from_doc(build, "save_variants_updated_csv"),
    )
    parser_build.add_argument(
        "--save_wt_vcrs_fasta_and_t2g",
        action="store_true",
        default=argparse.SUPPRESS,  # Remove from args if not provided
        help=extract_help_from_doc(build, "save_wt_vcrs_fasta_and_t2g"),
    )
    parser_build.add_argument(
        "--disable_save_removed_variants_text",
        dest="save_removed_variants_text",
        action="store_false",
        default=argparse.SUPPRESS,  # Remove from args if not provided
        help=extract_help_from_doc(build, "save_removed_variants_text"),
    )
    parser_build.add_argument(
        "--disable_save_filtering_report_text",
        dest="save_filtering_report_text",
        action="store_false",
        default=argparse.SUPPRESS,  # Remove from args if not provided
        help=extract_help_from_doc(build, "save_filtering_report_text"),
    )
    parser_build.add_argument(
        "--store_full_sequences",
        action="store_true",
        default=argparse.SUPPRESS,  # Remove from args if not provided
        help=extract_help_from_doc(build, "store_full_sequences"),
    )
    parser_build.add_argument(
        "--translate",
        action="store_true",
        default=argparse.SUPPRESS,  # Remove from args if not provided
        help=extract_help_from_doc(build, "translate"),
    )
    parser_build.add_argument(
        "-ts",
        "--translate_start",
        type=int_or_str,
        required=False,
        default=argparse.SUPPRESS,  # Remove from args if not provided
        help=extract_help_from_doc(build, "translate_start"),
    )
    parser_build.add_argument(
        "-te",
        "--translate_end",
        type=int_or_str,
        required=False,
        default=argparse.SUPPRESS,  # Remove from args if not provided
        help=extract_help_from_doc(build, "translate_end"),
    )
    parser_build.add_argument(
        "--chunksize",
        type=int,
        default=argparse.SUPPRESS,  # Remove from args if not provided
        help=extract_help_from_doc(build, "chunksize"),
    )
    parser_build.add_argument(
        "--dry_run",
        action="store_true",
        default=argparse.SUPPRESS,  # Remove from args if not provided
        help=extract_help_from_doc(build, "dry_run"),
    )
    parser_build.add_argument(
        "--list_prebuilt_indices",
        action="store_true",
        default=argparse.SUPPRESS,  # Remove from args if not provided
        help=extract_help_from_doc(build, "list_prebuilt_indices"),
    )
    parser_build.add_argument(
        "--overwrite",
        action="store_true",
        default=argparse.SUPPRESS,  # Remove from args if not provided
        help=extract_help_from_doc(build, "overwrite"),
    )
    parser_build.add_argument(
        "--logging_level",
        choices=["NOTSET", "DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL", "0", "10", "20", "30", "40", "50", "60", None],
        required=False,
        default=argparse.SUPPRESS,  # Remove from args if not provided
        help=extract_help_from_doc(build, "logging_level"),
    )
    parser_build.add_argument(
        "--save_logs",
        action="store_true",
        default=argparse.SUPPRESS,  # Remove from args if not provided
        help=extract_help_from_doc(build, "save_logs"),
    )
    parser_build.add_argument(
        "--log_out_dir",
        default=argparse.SUPPRESS,  # Remove from args if not provided
        help=extract_help_from_doc(build, "log_out_dir"),
    )
    parser_build.add_argument(
        "--verbose",
        action="store_true",
        default=argparse.SUPPRESS,  # Remove from args if not provided
        help=extract_help_from_doc(build, "verbose"),
    )

    # Additional kwargs arguments that I still want as command-line options
    parser_build.add_argument(
        "--insertion_size_limit",
        required=False,
        default=argparse.SUPPRESS,  # Remove from args if not provided
        help=extract_help_from_doc(build, "insertion_size_limit"),
    )
    parser_build.add_argument(
        "--min_seq_len",
        type=int,
        required=False,
        default=argparse.SUPPRESS,  # Remove from args if not provided
        help=extract_help_from_doc(build, "min_seq_len"),
    )
    parser_build.add_argument(
        "--disable_optimize_flanking_regions",
        dest="optimize_flanking_regions",
        action="store_false",
        default=argparse.SUPPRESS,  # Remove from args if not provided
        help=extract_help_from_doc(build, "optimize_flanking_regions", disable=True),
    )
    parser_build.add_argument(
        "--disable_remove_seqs_with_wt_kmers",
        dest="remove_seqs_with_wt_kmers",
        action="store_false",
        default=argparse.SUPPRESS,  # Remove from args if not provided
        help=extract_help_from_doc(build, "remove_seqs_with_wt_kmers", disable=True),
    )
    parser_build.add_argument(
        "--required_insertion_overlap_length",
        type=int_or_str,
        required=False,
        default=argparse.SUPPRESS,  # Remove from args if not provided
        help=extract_help_from_doc(build, "required_insertion_overlap_length"),
    )
    parser_build.add_argument(
        "--disable_merge_identical",
        dest="merge_identical",
        action="store_false",
        default=argparse.SUPPRESS,  # Remove from args if not provided
        help=extract_help_from_doc(build, "merge_identical", disable=True),
    )
    parser_build.add_argument(
        "-vs",
        "--vcrs_strandedness",
        action="store_true",
        default=argparse.SUPPRESS,  # Remove from args if not provided
        help=extract_help_from_doc(build, "vcrs_strandedness"),
    )
    parser_build.add_argument(
        "--disable_use_IDs",
        dest="use_IDs",
        action="store_false",
        default=argparse.SUPPRESS,  # Remove from args if not provided
        help=extract_help_from_doc(build, "use_IDs", disable=True),
    )
    parser_build.add_argument(
        "--cosmic_version",
        required=False,
        default=argparse.SUPPRESS,  # Remove from args if not provided
        help=extract_help_from_doc(build, "cosmic_version"),
    )
    parser_build.add_argument(
        "--cosmic_grch",
        required=False,
        default=argparse.SUPPRESS,  # Remove from args if not provided
        help=extract_help_from_doc(build, "cosmic_grch"),
    )
    parser_build.add_argument(
        "--cosmic_email",
        required=False,
        default=argparse.SUPPRESS,  # Remove from args if not provided
        help=extract_help_from_doc(build, "cosmic_email"),
    )
    parser_build.add_argument(
        "--cosmic_password",
        required=False,
        default=argparse.SUPPRESS,  # Remove from args if not provided
        help=extract_help_from_doc(build, "cosmic_password"),
    )

    # NEW PARSER
    info_desc = "Takes in the input directory containing with the VCRS fasta file generated from varseek build, and returns a dataframe with additional columns containing information about the variants."
    parser_info = parent_subparsers.add_parser(
        "info",
        parents=[parent],
        description=info_desc,
        help=info_desc,
        add_help=True,
        formatter_class=CustomHelpFormatter,
    )

    parser_info.add_argument(
        "-i",
        "--input_dir",
        type=str,
        required=not vk_info_list_information_and_exit_flag_present,
        help=extract_help_from_doc(info, "input_dir"),
    )
    parser_info.add_argument(
        "-c",
        "--columns_to_include",
        type=strpath_or_list_like_of_strings,
        nargs="+",
        required=False,
        default=argparse.SUPPRESS,  # Remove from args if not provided
        help=extract_help_from_doc(info, "columns_to_include"),
    )
    parser_info.add_argument(
        "-k",
        "--k",
        type=int,
        required=False,
        default=argparse.SUPPRESS,  # Remove from args if not provided
        help=extract_help_from_doc(info, "k"),
    )
    parser_info.add_argument(
        "--max_ambiguous_vcrs",
        type=int,
        required=False,
        default=argparse.SUPPRESS,  # Remove from args if not provided
        help=extract_help_from_doc(info, "max_ambiguous_vcrs"),
    )
    parser_info.add_argument(
        "--max_ambiguous_reference",
        type=int,
        required=False,
        default=argparse.SUPPRESS,  # Remove from args if not provided
        help=extract_help_from_doc(info, "max_ambiguous_reference"),
    )
    parser_info.add_argument(
        "--vcrs_fasta",
        type=str,
        required=False,
        default=argparse.SUPPRESS,  # Remove from args if not provided
        help=extract_help_from_doc(info, "vcrs_fasta"),
    )
    parser_info.add_argument(
        "--variants_updated_csv",
        required=False,
        default=argparse.SUPPRESS,  # Remove from args if not provided
        help=extract_help_from_doc(info, "variants_updated_csv"),
    )
    parser_info.add_argument(
        "--id_to_header_csv",
        required=False,
        default=argparse.SUPPRESS,  # Remove from args if not provided
        help=extract_help_from_doc(info, "id_to_header_csv"),
    )
    parser_info.add_argument(
        "--gtf",
        required=False,
        default=argparse.SUPPRESS,  # Remove from args if not provided
        help=extract_help_from_doc(info, "gtf"),
    )
    parser_info.add_argument(
        "--dlist_reference_source",
        required=False,
        choices=["grch37", "grch38", "t2t"],
        default=argparse.SUPPRESS,  # Remove from args if not provided
        help=extract_help_from_doc(info, "dlist_reference_source"),
    )
    parser_info.add_argument(
        "--dlist_reference_genome_fasta",
        required=False,
        default=argparse.SUPPRESS,  # Remove from args if not provided
        help=extract_help_from_doc(info, "dlist_reference_genome_fasta"),
    )
    parser_info.add_argument(
        "--dlist_reference_cdna_fasta",
        type=str,
        required=False,
        default=argparse.SUPPRESS,  # Remove from args if not provided
        help=extract_help_from_doc(info, "dlist_reference_cdna_fasta"),
    )
    parser_info.add_argument(
        "--dlist_reference_gtf",
        type=str,
        required=False,
        default=argparse.SUPPRESS,  # Remove from args if not provided
        help=extract_help_from_doc(info, "dlist_reference_gtf"),
    )
    parser_info.add_argument(
        "--dlist_reference_ensembl_release",
        type=int,
        required=False,
        default=argparse.SUPPRESS,  # Remove from args if not provided
        help=extract_help_from_doc(info, "dlist_reference_ensembl_release"),
    )
    parser_info.add_argument(
        "--var_id_column",
        type=str,
        required=False,
        default=argparse.SUPPRESS,  # Remove from args if not provided
        help=extract_help_from_doc(info, "gene_name_column"),
    )
    parser_info.add_argument(
        "--gene_name_column",
        type=str,
        required=False,
        default=argparse.SUPPRESS,  # Remove from args if not provided
        help=extract_help_from_doc(info, "gene_name_column"),
    )
    parser_info.add_argument(
        "--variant_source_column",
        type=str,
        required=False,
        default=argparse.SUPPRESS,  # Remove from args if not provided
        help=extract_help_from_doc(info, "variant_source_column"),
    )
    parser_info.add_argument(
        "--var_cdna_column",
        type=str,
        required=False,
        default=argparse.SUPPRESS,  # Remove from args if not provided
        help=extract_help_from_doc(info, "var_cdna_column"),
    )
    parser_info.add_argument(
        "--seq_id_cdna_column",
        type=str,
        required=False,
        default=argparse.SUPPRESS,  # Remove from args if not provided
        help=extract_help_from_doc(info, "seq_id_cdna_column"),
    )
    parser_info.add_argument(
        "--var_genome_column",
        type=str,
        required=False,
        default=argparse.SUPPRESS,  # Remove from args if not provided
        help=extract_help_from_doc(info, "var_genome_column"),
    )
    parser_info.add_argument(
        "--seq_id_genome_column",
        type=str,
        required=False,
        default=argparse.SUPPRESS,  # Remove from args if not provided
        help=extract_help_from_doc(info, "seq_id_genome_column"),
    )
    parser_info.add_argument(
        "-o",
        "--out",
        required=False,
        default=argparse.SUPPRESS,  # Remove from args if not provided
        help=extract_help_from_doc(info, "out"),
    )
    parser_info.add_argument(
        "-r",
        "--reference_out_dir",
        required=False,
        default=argparse.SUPPRESS,  # Remove from args if not provided
        help=extract_help_from_doc(info, "reference_out_dir"),
    )
    parser_info.add_argument(
        "--variants_updated_vk_info_csv_out",
        required=False,
        default=argparse.SUPPRESS,  # Remove from args if not provided
        help=extract_help_from_doc(info, "variants_updated_vk_info_csv_out"),
    )
    parser_info.add_argument(
        "--variants_updated_exploded_vk_info_csv_out",
        required=False,
        default=argparse.SUPPRESS,  # Remove from args if not provided
        help=extract_help_from_doc(info, "variants_updated_exploded_vk_info_csv_out"),
    )
    parser_info.add_argument(
        "--dlist_genome_fasta_out",
        required=False,
        default=argparse.SUPPRESS,  # Remove from args if not provided
        help=extract_help_from_doc(info, "dlist_genome_fasta_out"),
    )
    parser_info.add_argument(
        "--dlist_cdna_fasta_out",
        required=False,
        default=argparse.SUPPRESS,  # Remove from args if not provided
        help=extract_help_from_doc(info, "dlist_cdna_fasta_out"),
    )
    parser_info.add_argument(
        "--dlist_combined_fasta_out",
        required=False,
        default=argparse.SUPPRESS,  # Remove from args if not provided
        help=extract_help_from_doc(info, "dlist_combined_fasta_out"),
    )
    parser_info.add_argument(
        "--save_variants_updated_exploded_vk_info_csv",
        action="store_true",
        default=argparse.SUPPRESS,  # Remove from args if not provided
        help=extract_help_from_doc(info, "save_variants_updated_exploded_vk_info_csv"),
    )
    parser_info.add_argument(
        "--make_pyfastx_summary_file",
        action="store_true",
        default=argparse.SUPPRESS,  # Remove from args if not provided
        help=extract_help_from_doc(info, "make_pyfastx_summary_file"),
    )
    parser_info.add_argument(
        "--make_kat_histogram",
        action="store_true",
        default=argparse.SUPPRESS,  # Remove from args if not provided
        help=extract_help_from_doc(info, "make_kat_histogram"),
    )
    parser_info.add_argument(
        "--chunksize",
        type=int,
        default=argparse.SUPPRESS,  # Remove from args if not provided
        help=extract_help_from_doc(info, "chunksize"),
    )
    parser_info.add_argument(
        "--dry_run",
        action="store_true",
        default=argparse.SUPPRESS,  # Remove from args if not provided
        help=extract_help_from_doc(info, "dry_run"),
    )
    parser_info.add_argument(
        "--list_columns",
        action="store_true",
        default=argparse.SUPPRESS,  # Remove from args if not provided
        help=extract_help_from_doc(info, "list_columns"),
    )
    parser_info.add_argument(
        "--overwrite",
        action="store_true",
        default=argparse.SUPPRESS,  # Remove from args if not provided
        help=extract_help_from_doc(info, "overwrite"),
    )
    parser_info.add_argument(
        "--threads",
        type=int,
        required=False,
        default=argparse.SUPPRESS,  # Remove from args if not provided
        help=extract_help_from_doc(info, "threads"),
    )
    parser_info.add_argument(
        "--logging_level",
        choices=["NOTSET", "DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL", "0", "10", "20", "30", "40", "50", "60", None],
        required=False,
        default=argparse.SUPPRESS,  # Remove from args if not provided
        help=extract_help_from_doc(info, "logging_level"),
    )
    parser_info.add_argument(
        "--save_logs",
        action="store_true",
        default=argparse.SUPPRESS,  # Remove from args if not provided
        help=extract_help_from_doc(info, "save_logs"),
    )
    parser_info.add_argument(
        "--log_out_dir",
        required=False,
        default=argparse.SUPPRESS,  # Remove from args if not provided
        help=extract_help_from_doc(info, "log_out_dir"),
    )
    parser_info.add_argument(
        "--verbose",
        action="store_true",
        default=argparse.SUPPRESS,  # Remove from args if not provided
        help=extract_help_from_doc(info, "verbose"),
    )

    # kwargs
    parser_info.add_argument(
        "-w",
        "--w",
        type=int,
        required=False,
        default=argparse.SUPPRESS,  # Remove from args if not provided
        help=extract_help_from_doc(info, "w"),
    )
    parser_info.add_argument(
        "--bowtie2_path",
        required=False,
        default=argparse.SUPPRESS,  # Remove from args if not provided
        help=extract_help_from_doc(info, "bowtie2_path"),
    )
    parser_info.add_argument(
        "-vs",
        "--vcrs_strandedness",
        action="store_true",
        default=argparse.SUPPRESS,  # Remove from args if not provided
        help=extract_help_from_doc(info, "vcrs_strandedness"),
    )
    parser_info.add_argument(
        "--near_splice_junction_threshold",
        type=int,
        required=False,
        default=argparse.SUPPRESS,  # Remove from args if not provided
        help=extract_help_from_doc(info, "near_splice_junction_threshold"),
    )
    parser_info.add_argument(
        "--reference_cdna_fasta",
        required=False,
        default=argparse.SUPPRESS,  # Remove from args if not provided
        help=extract_help_from_doc(info, "reference_cdna_fasta"),
    )
    parser_info.add_argument(
        "--reference_genome_fasta",
        required=False,
        default=argparse.SUPPRESS,  # Remove from args if not provided
        help=extract_help_from_doc(info, "reference_genome_fasta"),
    )
    parser_info.add_argument(
        "--variants",
        required=False,
        default=argparse.SUPPRESS,  # Remove from args if not provided
        help=extract_help_from_doc(info, "variants"),
    )
    parser_info.add_argument(
        "--seq_id_column",
        type=str,
        required=False,
        default=argparse.SUPPRESS,  # Remove from args if not provided
        help=extract_help_from_doc(info, "seq_id_column"),
    )
    parser_info.add_argument(
        "--var_column",
        type=str,
        required=False,
        default=argparse.SUPPRESS,  # Remove from args if not provided
        help=extract_help_from_doc(info, "var_column"),
    )
    parser_info.add_argument(
        "--kallisto",
        type=str,
        required=False,
        default=argparse.SUPPRESS,  # Remove from args if not provided
        help=extract_help_from_doc(info, "kallisto"),
    )
    parser_info.add_argument(
        "--bustools",
        type=str,
        required=False,
        default=argparse.SUPPRESS,  # Remove from args if not provided
        help=extract_help_from_doc(info, "bustools"),
    )

    # NEW PARSER
    filter_desc = "Filter variants based on the provided filters and save the filtered variants to a fasta file."
    parser_filter = parent_subparsers.add_parser(
        "filter",
        parents=[parent],
        description=filter_desc,
        help=filter_desc,
        add_help=True,
        formatter_class=CustomHelpFormatter,
    )
    parser_filter.add_argument(
        "-i",
        "--input_dir",
        type=str,
        required=not vk_filter_list_information_and_exit_flag_present,
        help=extract_help_from_doc(filter, "input_dir"),
    )
    parser_filter.add_argument(
        "-f",
        "--filters",
        type=strpath_or_list_like_of_strings,
        nargs="+",
        required=not vk_filter_list_information_and_exit_flag_present,
        help=extract_help_from_doc(filter, "filters"),
    )
    parser_filter.add_argument(
        "--variants_updated_vk_info_csv",
        required=False,
        default=argparse.SUPPRESS,  # Remove from args if not provided
        help=extract_help_from_doc(filter, "variants_updated_vk_info_csv"),
    )
    parser_filter.add_argument(
        "--variants_updated_exploded_vk_info_csv",
        required=False,
        default=argparse.SUPPRESS,  # Remove from args if not provided
        help=extract_help_from_doc(filter, "variants_updated_exploded_vk_info_csv"),
    )
    parser_filter.add_argument(
        "--id_to_header_csv",
        required=False,
        default=argparse.SUPPRESS,  # Remove from args if not provided
        help=extract_help_from_doc(filter, "id_to_header_csv"),
    )
    parser_filter.add_argument(
        "--dlist_fasta",
        required=False,
        default=argparse.SUPPRESS,  # Remove from args if not provided
        help=extract_help_from_doc(filter, "dlist_fasta"),
    )
    parser_filter.add_argument(
        "--vcrs_id_column",
        required=False,
        default=argparse.SUPPRESS,  # Remove from args if not provided
        help=extract_help_from_doc(filter, "vcrs_id_column"),
    )
    parser_filter.add_argument(
        "--vcrs_sequence_column",
        required=False,
        default=argparse.SUPPRESS,  # Remove from args if not provided
        help=extract_help_from_doc(filter, "vcrs_sequence_column"),
    )
    parser_filter.add_argument(
        "-o",
        "--out",
        required=False,
        default=argparse.SUPPRESS,  # Remove from args if not provided
        help=extract_help_from_doc(filter, "out"),
    )
    parser_filter.add_argument(
        "--variants_updated_filtered_csv_out",
        required=False,
        default=argparse.SUPPRESS,  # Remove from args if not provided
        help=extract_help_from_doc(filter, "variants_updated_filtered_csv_out"),
    )
    parser_filter.add_argument(
        "--variants_updated_exploded_filtered_csv_out",
        required=False,
        default=argparse.SUPPRESS,  # Remove from args if not provided
        help=extract_help_from_doc(filter, "variants_updated_exploded_filtered_csv_out"),
    )
    parser_filter.add_argument(
        "--id_to_header_filtered_csv_out",
        required=False,
        default=argparse.SUPPRESS,  # Remove from args if not provided
        help=extract_help_from_doc(filter, "id_to_header_filtered_csv_out"),
    )
    parser_filter.add_argument(
        "--dlist_filtered_fasta_out",
        required=False,
        default=argparse.SUPPRESS,  # Remove from args if not provided
        help=extract_help_from_doc(filter, "dlist_filtered_fasta_out"),
    )
    parser_filter.add_argument(
        "--vcrs_filtered_fasta_out",
        required=False,
        default=argparse.SUPPRESS,  # Remove from args if not provided
        help=extract_help_from_doc(filter, "vcrs_filtered_fasta_out"),
    )
    parser_filter.add_argument(
        "--vcrs_t2g_filtered_out",
        required=False,
        default=argparse.SUPPRESS,  # Remove from args if not provided
        help=extract_help_from_doc(filter, "vcrs_t2g_filtered_out"),
    )
    parser_filter.add_argument(
        "--wt_vcrs_filtered_fasta_out",
        required=False,
        default=argparse.SUPPRESS,  # Remove from args if not provided
        help=extract_help_from_doc(filter, "wt_vcrs_filtered_fasta_out"),
    )
    parser_filter.add_argument(
        "--wt_vcrs_t2g_filtered_out",
        required=False,
        default=argparse.SUPPRESS,  # Remove from args if not provided
        help=extract_help_from_doc(filter, "wt_vcrs_t2g_filtered_out"),
    )
    parser_filter.add_argument(
        "--save_wt_vcrs_fasta_and_t2g",
        action="store_true",
        default=argparse.SUPPRESS,  # Remove from args if not provided
        help=extract_help_from_doc(filter, "save_wt_vcrs_fasta_and_t2g"),
    )
    parser_filter.add_argument(
        "--save_variants_updated_filtered_csvs",
        action="store_true",
        default=argparse.SUPPRESS,  # Remove from args if not provided
        help=extract_help_from_doc(filter, "save_variants_updated_filtered_csvs"),
    )
    parser_filter.add_argument(
        "--return_variants_updated_filtered_csv_df",
        action="store_true",
        default=argparse.SUPPRESS,  # Remove from args if not provided
        help=extract_help_from_doc(filter, "return_variants_updated_filtered_csv_df"),
    )
    parser_filter.add_argument(
        "--chunksize",
        type=int,
        default=argparse.SUPPRESS,  # Remove from args if not provided
        help=extract_help_from_doc(filter, "chunksize"),
    )
    parser_filter.add_argument(
        "--dry_run",
        action="store_true",
        default=argparse.SUPPRESS,  # Remove from args if not provided
        help=extract_help_from_doc(filter, "dry_run"),
    )
    parser_filter.add_argument(
        "--list_filter_rules",
        action="store_true",
        default=argparse.SUPPRESS,  # Remove from args if not provided
        help=extract_help_from_doc(filter, "list_filter_rules"),
    )
    parser_filter.add_argument(
        "--overwrite",
        action="store_true",
        default=argparse.SUPPRESS,  # Remove from args if not provided
        help=extract_help_from_doc(filter, "overwrite"),
    )
    parser_filter.add_argument(
        "--logging_level",
        choices=["NOTSET", "DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL", "0", "10", "20", "30", "40", "50", "60", None],
        required=False,
        default=argparse.SUPPRESS,  # Remove from args if not provided
        help=extract_help_from_doc(filter, "logging_level"),
    )
    parser_filter.add_argument(
        "--save_logs",
        action="store_true",
        default=argparse.SUPPRESS,  # Remove from args if not provided
        help=extract_help_from_doc(filter, "save_logs"),
    )
    parser_filter.add_argument(
        "--log_out_dir",
        required=False,
        default=argparse.SUPPRESS,  # Remove from args if not provided
        help=extract_help_from_doc(filter, "log_out_dir"),
    )
    parser_filter.add_argument(
        "--disable_make_internal_copies",
        action="store_false",
        default=argparse.SUPPRESS,  # Remove from args if not provided
        help=extract_help_from_doc(filter, "make_internal_copies", disable=True),
    )

    # kwargs
    parser_filter.add_argument(
        "--filter_all_dlists",
        action="store_true",
        default=argparse.SUPPRESS,  # Remove from args if not provided
        help=extract_help_from_doc(filter, "filter_all_dlists"),
    )
    parser_filter.add_argument(
        "--dlist_genome_fasta",
        required=False,
        default=argparse.SUPPRESS,  # Remove from args if not provided
        help=extract_help_from_doc(filter, "dlist_genome_fasta"),
    )
    parser_filter.add_argument(
        "--dlist_cdna_fasta",
        required=False,
        default=argparse.SUPPRESS,  # Remove from args if not provided
        help=extract_help_from_doc(filter, "dlist_cdna_fasta"),
    )
    parser_filter.add_argument(
        "--dlist_genome_filtered_fasta_out",
        required=False,
        default=argparse.SUPPRESS,  # Remove from args if not provided
        help=extract_help_from_doc(filter, "dlist_genome_filtered_fasta_out"),
    )
    parser_filter.add_argument(
        "--dlist_cdna_filtered_fasta_out",
        required=False,
        default=argparse.SUPPRESS,  # Remove from args if not provided
        help=extract_help_from_doc(filter, "dlist_cdna_filtered_fasta_out"),
    )
    parser_filter.add_argument(
        "--disable_save_vcrs_filtered_fasta_and_t2g",
        dest="save_vcrs_filtered_fasta_and_t2g",
        action="store_false",
        default=argparse.SUPPRESS,  # Remove from args if not provided
        help=extract_help_from_doc(filter, "save_vcrs_filtered_fasta_and_t2g", disable=True),
    )
    parser_filter.add_argument(
        "--disable_use_IDs",
        dest="use_IDs",
        action="store_false",
        default=argparse.SUPPRESS,  # Remove from args if not provided
        help=extract_help_from_doc(filter, "use_IDs", disable=True),
    )

    # NEW PARSER
    sim_desc = "Create synthetic RNA-seq dataset with variant-containing reads."
    parser_sim = parent_subparsers.add_parser(
        "sim",
        parents=[parent],
        description=sim_desc,
        help=sim_desc,
        add_help=True,
        formatter_class=CustomHelpFormatter,
    )
    parser_sim.add_argument(
        "-v",
        "--variants",
        default=None,
        type=strpath_or_strnonpath_or_df,
        required=True,
        help=extract_help_from_doc(sim, "variants"),
    )
    parser_sim.add_argument(
        "--number_of_variants_to_sample",
        type=int,
        required=False,
        default=argparse.SUPPRESS,  # Remove from args if not provided
        help=extract_help_from_doc(sim, "number_of_variants_to_sample"),
    )
    parser_sim.add_argument(
        "--number_of_reads_per_variant_alt",
        required=False,
        default=argparse.SUPPRESS,  # Remove from args if not provided
        help=extract_help_from_doc(sim, "number_of_reads_per_variant_alt"),
    )
    parser_sim.add_argument(
        "--number_of_reads_per_variant_ref",
        required=False,
        default=argparse.SUPPRESS,  # Remove from args if not provided
        help=extract_help_from_doc(sim, "number_of_reads_per_variant_ref"),
    )
    parser_sim.add_argument(
        "--sample_ref_and_alt_reads_from_same_locations",
        action="store_true",
        default=argparse.SUPPRESS,  # Remove from args if not provided
        help=extract_help_from_doc(sim, "sample_ref_and_alt_reads_from_same_locations"),
    )
    parser_sim.add_argument(
        "--with_replacement",
        action="store_true",
        default=argparse.SUPPRESS,  # Remove from args if not provided
        help=extract_help_from_doc(sim, "with_replacement"),
    )
    parser_sim.add_argument(
        "--strand",
        choices=["f", "r", "both", "random", None],
        required=False,
        default=argparse.SUPPRESS,  # Remove from args if not provided
        help=extract_help_from_doc(sim, "strand"),
    )
    parser_sim.add_argument(
        "--read_length",
        type=int,
        required=False,
        default=argparse.SUPPRESS,  # Remove from args if not provided
        help=extract_help_from_doc(sim, "read_length"),
    )
    parser_sim.add_argument(
        "-f",
        "--filters",
        nargs="*",  # Accept multiple sequential filters or a single JSON file
        type=str,
        required=False,
        default=argparse.SUPPRESS,  # Remove from args if not provided
        help=extract_help_from_doc(sim, "filters"),
    )
    parser_sim.add_argument(
        "--add_noise_sequencing_error",
        action="store_true",
        default=argparse.SUPPRESS,  # Remove from args if not provided
        help=extract_help_from_doc(sim, "add_noise_sequencing_error"),
    )
    parser_sim.add_argument(
        "--add_noise_base_quality",
        action="store_true",
        default=argparse.SUPPRESS,  # Remove from args if not provided
        help=extract_help_from_doc(sim, "add_noise_base_quality"),
    )
    parser_sim.add_argument(
        "--error_rate",
        required=False,
        default=argparse.SUPPRESS,  # Remove from args if not provided
        help=extract_help_from_doc(sim, "error_rate"),
    )
    parser_sim.add_argument(
        "--error_distribution",
        required=False,
        default=argparse.SUPPRESS,  # Remove from args if not provided
        help=extract_help_from_doc(sim, "error_distribution"),
    )
    parser_sim.add_argument(
        "--max_errors",
        required=False,
        default=argparse.SUPPRESS,  # Remove from args if not provided
        help=extract_help_from_doc(sim, "max_errors"),
    )
    parser_sim.add_argument(
        "--variant_sequence_read_parent_column",
        required=False,
        default="mutant_sequence_read_parent",
        help=extract_help_from_doc(sim, "variant_sequence_read_parent_column"),
    )
    parser_sim.add_argument(
        "--ref_sequence_read_parent_column",
        required=False,
        default=argparse.SUPPRESS,  # Remove from args if not provided
        help=extract_help_from_doc(sim, "ref_sequence_read_parent_column"),
    )
    parser_sim.add_argument(
        "--variant_sequence_read_parent_rc_column",
        required=False,
        default=argparse.SUPPRESS,  # Remove from args if not provided
        help=extract_help_from_doc(sim, "variant_sequence_read_parent_rc_column"),
    )
    parser_sim.add_argument(
        "--ref_sequence_read_parent_rc_column",
        required=False,
        default=argparse.SUPPRESS,  # Remove from args if not provided
        help=extract_help_from_doc(sim, "ref_sequence_read_parent_rc_column"),
    )
    parser_sim.add_argument(
        "--reads_fastq_parent",
        required=False,
        default=argparse.SUPPRESS,  # Remove from args if not provided
        help=extract_help_from_doc(sim, "reads_fastq_parent"),
    )
    parser_sim.add_argument(
        "--reads_csv_parent",
        required=False,
        default=argparse.SUPPRESS,  # Remove from args if not provided
        help=extract_help_from_doc(sim, "reads_csv_parent"),
    )
    parser_sim.add_argument(
        "--out",
        type=str,
        required=False,
        default=argparse.SUPPRESS,  # Remove from args if not provided
        help=extract_help_from_doc(sim, "out"),
    )
    parser_sim.add_argument(
        "--reads_fastq_out",
        required=False,
        default=argparse.SUPPRESS,  # Remove from args if not provided
        help=extract_help_from_doc(sim, "reads_fastq_out"),
    )
    parser_sim.add_argument(
        "--variants_updated_csv_out",
        required=False,
        default=argparse.SUPPRESS,  # Remove from args if not provided
        help=extract_help_from_doc(sim, "variants_updated_csv_out"),
    )
    parser_sim.add_argument(
        "--reads_csv_out",
        required=False,
        default=argparse.SUPPRESS,  # Remove from args if not provided
        help=extract_help_from_doc(sim, "reads_csv_out"),
    )
    parser_sim.add_argument(
        "--disable_save_variants_updated_csv",
        dest="save_variants_updated_csv",
        action="store_false",
        default=argparse.SUPPRESS,  # Remove from args if not provided
        help=extract_help_from_doc(sim, "save_variants_updated_csv", disable=True),
    )
    parser_sim.add_argument(
        "--disable_save_reads_csv",
        dest="save_reads_csv",
        action="store_false",
        default=argparse.SUPPRESS,  # Remove from args if not provided
        help=extract_help_from_doc(sim, "save_reads_csv", disable=True),
    )
    parser_sim.add_argument(
        "--vk_build_out_dir",
        type=str,
        required=False,
        default=argparse.SUPPRESS,  # Remove from args if not provided
        help=extract_help_from_doc(sim, "vk_build_out_dir"),
    )
    parser_sim.add_argument(
        "--sequences",
        required=False,
        default=argparse.SUPPRESS,  # Remove from args if not provided
        help=extract_help_from_doc(sim, "sequences"),
    )
    parser_sim.add_argument(
        "--seq_id_column",
        type=str,
        required=False,
        default=argparse.SUPPRESS,  # Remove from args if not provided
        help=extract_help_from_doc(sim, "seq_id_column"),
    )
    parser_sim.add_argument(
        "--var_column",
        type=str,
        required=False,
        default=argparse.SUPPRESS,  # Remove from args if not provided
        help=extract_help_from_doc(sim, "var_column"),
    )
    parser_sim.add_argument(
        "--var_id_column",
        type=str,
        required=False,
        default=argparse.SUPPRESS,  # Remove from args if not provided
        help=extract_help_from_doc(sim, "var_id_column"),
    )
    parser_sim.add_argument(
        "--k",
        type=int,
        required=False,
        default=argparse.SUPPRESS,  # Remove from args if not provided
        help=extract_help_from_doc(sim, "k"),
    )
    parser_sim.add_argument(
        "-w",
        "--w",
        type=int,
        required=False,
        default=argparse.SUPPRESS,  # Remove from args if not provided
        help=extract_help_from_doc(sim, "w"),
    )
    parser_sim.add_argument(
        "--sequences_cdna",
        required=False,
        default=argparse.SUPPRESS,  # Remove from args if not provided
        help=extract_help_from_doc(sim, "sequences_cdna"),
    )
    parser_sim.add_argument(
        "--seq_id_column_cdna",
        required=False,
        default=argparse.SUPPRESS,  # Remove from args if not provided
        help=extract_help_from_doc(sim, "seq_id_column_cdna"),
    )
    parser_sim.add_argument(
        "--var_column_cdna",
        required=False,
        default=argparse.SUPPRESS,  # Remove from args if not provided
        help=extract_help_from_doc(sim, "var_column_cdna"),
    )
    parser_sim.add_argument(
        "--sequences_genome",
        required=False,
        help=extract_help_from_doc(sim, "sequences_genome"),
    )
    parser_sim.add_argument(
        "--seq_id_column_genome",
        required=False,
        default=argparse.SUPPRESS,  # Remove from args if not provided
        help=extract_help_from_doc(sim, "seq_id_column_genome"),
    )
    parser_sim.add_argument(
        "--var_column_genome",
        required=False,
        default=argparse.SUPPRESS,  # Remove from args if not provided
        help=extract_help_from_doc(sim, "var_column_genome"),
    )
    parser_sim.add_argument(
        "--seed",
        required=False,
        default=argparse.SUPPRESS,  # Remove from args if not provided
        help=extract_help_from_doc(sim, "seed"),
    )
    parser_sim.add_argument(
        "--gzip_reads_fastq_out",
        action="store_true",
        default=argparse.SUPPRESS,  # Remove from args if not provided
        help=extract_help_from_doc(sim, "gzip_reads_fastq_out"),
    )
    parser_sim.add_argument(
        "--dry_run",
        action="store_true",
        default=argparse.SUPPRESS,  # Remove from args if not provided
        help=extract_help_from_doc(sim, "dry_run"),
    )
    parser_sim.add_argument(
        "--logging_level",
        choices=["NOTSET", "DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL", "0", "10", "20", "30", "40", "50", "60", None],
        required=False,
        default=argparse.SUPPRESS,  # Remove from args if not provided
        help=extract_help_from_doc(sim, "logging_level"),
    )
    parser_sim.add_argument(
        "--save_logs",
        action="store_true",
        default=argparse.SUPPRESS,  # Remove from args if not provided
        help=extract_help_from_doc(sim, "save_logs"),
    )
    parser_sim.add_argument(
        "--log_out_dir",
        required=False,
        default=argparse.SUPPRESS,  # Remove from args if not provided
        help=extract_help_from_doc(sim, "log_out_dir"),
    )
    # kwargs
    parser_sim.add_argument(
        "--disable_make_internal_copies",
        action="store_false",
        default=argparse.SUPPRESS,  # Remove from args if not provided
        help=extract_help_from_doc(sim, "make_internal_copies", disable=True),
    )
    parser_sim.add_argument(
        "--disable_filter_null_rows_from_important_cols",
        action="store_false",
        default=argparse.SUPPRESS,  # Remove from args if not provided
        help=extract_help_from_doc(sim, "filter_null_rows_from_important_cols", disable=True),
    )

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
        "--apply_single_end_mode_on_paired_end_data_correction",
        action="store_true",
        default=argparse.SUPPRESS,  # Remove from args if not provided
        help=extract_help_from_doc(clean, "apply_single_end_mode_on_paired_end_data_correction"),
    )
    parser_clean.add_argument(
        "--split_reads_by_Ns_and_low_quality_bases",
        action="store_true",
        default=argparse.SUPPRESS,  # Remove from args if not provided
        help=extract_help_from_doc(clean, "split_reads_by_Ns_and_low_quality_bases"),
    )
    parser_clean.add_argument(
        "--apply_dlist_correction",
        action="store_true",
        default=argparse.SUPPRESS,  # Remove from args if not provided
        help=extract_help_from_doc(clean, "apply_dlist_correction"),
    )
    parser_clean.add_argument(
        "--qc_against_gene_matrix",
        action="store_true",
        default=argparse.SUPPRESS,  # Remove from args if not provided
        help=extract_help_from_doc(clean, "qc_against_gene_matrix"),
    )
    # parser_clean.add_argument(
    #     "--disable_qc_against_gene_matrix",
    #     dest="qc_against_gene_matrix",
    #     action="store_false",
    #     default=argparse.SUPPRESS,  # Remove from args if not provided
    #     help=extract_help_from_doc(clean, "disable_qc_against_gene_matrix", disable=True),
    # )
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
        "--gtf",
        required=False,
        default=argparse.SUPPRESS,  # Remove from args if not provided
        help=extract_help_from_doc(clean, "gtf"),
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
        "--transcript_set_to_exclusively_keep",
        nargs="+",
        type=str,
        required=False,
        help=extract_help_from_doc(clean, "transcript_set_to_exclusively_keep"),
    )
    parser_clean.add_argument(
        "--transcript_set_to_exclude",
        nargs="+",
        type=str,
        required=False,
        help=extract_help_from_doc(clean, "transcript_set_to_exclude"),
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
        "-k",
        "--k",
        type=int,
        required=False,
        default=argparse.SUPPRESS,  # Remove from args if not provided
        help=extract_help_from_doc(clean, "k"),
    )
    parser_clean.add_argument(
        "--mm",
        action="store_true",
        default=argparse.SUPPRESS,  # Remove from args if not provided
        help=extract_help_from_doc(clean, "mm"),
    )
    parser_clean.add_argument(
        "--union",
        action="store_true",
        default=argparse.SUPPRESS,  # Remove from args if not provided
        help=extract_help_from_doc(clean, "union"),
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
        "--vcrs_fasta",
        required=False,
        default=argparse.SUPPRESS,  # Remove from args if not provided
        help=extract_help_from_doc(clean, "vcrs_fasta"),
    )
    parser_clean.add_argument(
        "--dlist_fasta",
        required=False,
        default=argparse.SUPPRESS,  # Remove from args if not provided
        help=extract_help_from_doc(clean, "dlist_fasta"),
    )
    parser_clean.add_argument(
        "--variants_updated_csv",
        required=False,
        default=argparse.SUPPRESS,  # Remove from args if not provided
        help=extract_help_from_doc(clean, "variants_updated_csv"),
    )
    parser_clean.add_argument(
        "--adata_reference_genome",
        required=False,
        default=argparse.SUPPRESS,  # Remove from args if not provided
        help=extract_help_from_doc(clean, "adata_reference_genome"),
    )
    parser_clean.add_argument(
        "--kb_count_vcrs_dir",
        required=False,
        default=argparse.SUPPRESS,  # Remove from args if not provided
        help=extract_help_from_doc(clean, "kb_count_vcrs_dir"),
    )
    parser_clean.add_argument(
        "--kb_count_reference_genome_dir",
        required=False,
        default=argparse.SUPPRESS,  # Remove from args if not provided
        help=extract_help_from_doc(clean, "kb_count_reference_genome_dir"),
    )
    parser_clean.add_argument(
        "--variants_updated_csv_columns_to_merge",
        nargs="*",
        type=str,
        required=False,
        help=extract_help_from_doc(clean, "variants_updated_csv_columns_to_merge"),
    )
    parser_clean.add_argument(
        "--seq_id_column",
        required=False,
        default=argparse.SUPPRESS,  # Remove from args if not provided
        help=extract_help_from_doc(clean, "seq_id_column"),
    )
    parser_clean.add_argument(
        "--gene_id_column",
        required=False,
        default=argparse.SUPPRESS,  # Remove from args if not provided
        help=extract_help_from_doc(clean, "gene_id_column"),
    )
    parser_clean.add_argument(
        "--vcf_data_csv",
        required=False,
        default=argparse.SUPPRESS,  # Remove from args if not provided
        help=extract_help_from_doc(clean, "vcf_data_csv"),
    )
    parser_clean.add_argument(
        "--variants",
        required=False,
        default=argparse.SUPPRESS,  # Remove from args if not provided
        help=extract_help_from_doc(clean, "variants"),
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
        "--threads",
        type=int,
        required=False,
        default=argparse.SUPPRESS,  # Remove from args if not provided
        help=extract_help_from_doc(clean, "threads"),
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
        "--id_to_header_csv",
        required=False,
        default=argparse.SUPPRESS,  # Remove from args if not provided
        help=extract_help_from_doc(clean, "id_to_header_csv"),
    )
    parser_clean.add_argument(
        "--kallisto",
        required=False,
        default=argparse.SUPPRESS,  # Remove from args if not provided
        help=extract_help_from_doc(clean, "kallisto"),
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
        "--vcrs_header_column",
        required=False,
        default=argparse.SUPPRESS,  # Remove from args if not provided
        help=extract_help_from_doc(summarize, "vcrs_header_column"),
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

    # NEW PARSER
    ref_desc = "Create a reference index and t2g file for variant screening with varseek count. Wraps around varseek build, varseek info, varseek filter, and kb ref."
    parser_ref = parent_subparsers.add_parser("ref", parents=[parent], description=ref_desc, help=ref_desc, add_help=True, formatter_class=CustomHelpFormatter, epilog="To see the full list of allowable arguments, please explore vk build, vk info, vk filter, and (kallisto-bustools') kb ref")

    parser_ref.add_argument(
        "-v",
        "--variants",
        # type=strpath_or_str_or_list_or_df,
        nargs="+",
        required=not vk_ref_list_information_and_exit_flag_present,  # generally True
        help=extract_help_from_doc(ref, "variants"),
    )
    parser_ref.add_argument(
        "-s",
        "--sequences",
        type=str,
        nargs="+",
        required=not vk_ref_list_information_and_exit_flag_present,  # generally True
        help=extract_help_from_doc(ref, "sequences"),
    )
    parser_ref.add_argument(
        "-w",
        "--w",
        type=int,
        required=False,
        default=argparse.SUPPRESS,  # Remove from args if not provided
        help=extract_help_from_doc(ref, "w"),
    )
    parser_ref.add_argument(
        "-k",
        "--k",
        type=int,
        required=False,
        default=argparse.SUPPRESS,  # Remove from args if not provided
        help=extract_help_from_doc(ref, "k"),
    )
    parser_ref.add_argument(
        "-f",
        "--filters",
        type=strpath_or_list_like_of_strings,
        nargs="*",
        required=False,
        default=argparse.SUPPRESS,  # Remove from args if not provided
        help=extract_help_from_doc(ref, "filters"),
    )
    parser_ref.add_argument(
        "--dlist",
        required=False,
        default=argparse.SUPPRESS,  # Remove from args if not provided
        help=extract_help_from_doc(ref, "dlist"),
    )
    parser_ref.add_argument(
        "--dlist_reference_source",
        required=False,
        choices=["grch37", "grch38", "t2t"],
        default=argparse.SUPPRESS,  # Remove from args if not provided
        help=extract_help_from_doc(ref, "dlist_reference_source"),
    )
    parser_ref.add_argument(
        "--dlist_reference_ensembl_release",
        required=False,
        default=argparse.SUPPRESS,  # Remove from args if not provided
        help=extract_help_from_doc(ref, "dlist_reference_ensembl_release"),
    )
    parser_ref.add_argument(
        "--var_column",
        required=False,
        default=argparse.SUPPRESS,  # Remove from args if not provided
        help=extract_help_from_doc(ref, "var_column"),
    )
    parser_ref.add_argument(
        "--seq_id_column",
        required=False,
        default=argparse.SUPPRESS,  # Remove from args if not provided
        help=extract_help_from_doc(ref, "seq_id_column"),
    )
    parser_ref.add_argument(
        "--var_id_column",
        required=False,
        default=argparse.SUPPRESS,  # Remove from args if not provided
        help=extract_help_from_doc(ref, "var_id_column"),
    )
    parser_ref.add_argument(
        "-o",
        "--out",
        type=str,
        required=False,
        default=argparse.SUPPRESS,  # Remove from args if not provided
        help=extract_help_from_doc(ref, "out"),
    )
    parser_ref.add_argument(
        "--reference_out_dir",
        type=str,
        required=False,
        default=argparse.SUPPRESS,  # Remove from args if not provided
        help=extract_help_from_doc(ref, "reference_out_dir"),
    )
    parser_ref.add_argument(
        "-i",
        "--index_out",
        required=False,
        default=argparse.SUPPRESS,  # Remove from args if not provided
        help=extract_help_from_doc(ref, "index_out"),
    )
    parser_ref.add_argument(
        "-g",
        "--t2g_out",
        required=False,
        default=argparse.SUPPRESS,  # Remove from args if not provided
        help=extract_help_from_doc(ref, "t2g_out"),
    )
    parser_ref.add_argument(
        "--fasta_out",
        required=False,
        default=argparse.SUPPRESS,  # Remove from args if not provided
        help=extract_help_from_doc(ref, "fasta_out"),
    )
    parser_ref.add_argument(
        "-d",
        "--download",
        action="store_true",
        default=argparse.SUPPRESS,  # Remove from args if not provided
        help=extract_help_from_doc(ref, "download"),
    )
    parser_ref.add_argument(
        "--chunksize",
        type=int,
        default=argparse.SUPPRESS,  # Remove from args if not provided
        help=extract_help_from_doc(ref, "chunksize"),
    )
    parser_ref.add_argument(
        "--dry_run",
        action="store_true",
        default=argparse.SUPPRESS,  # Remove from args if not provided
        help=extract_help_from_doc(ref, "dry_run"),
    )
    parser_ref.add_argument(
        "--list_downloadable_references",
        action="store_true",
        default=argparse.SUPPRESS,  # Remove from args if not provided
        help=extract_help_from_doc(ref, "list_downloadable_references"),
    )
    parser_ref.add_argument(
        "-dmic",
        "--disable_minimum_info_columns",
        dest="minimum_info_columns",
        action="store_false",
        default=argparse.SUPPRESS,  # Remove from args if not provided
        help=extract_help_from_doc(ref, "minimum_info_columns", disable=True),
    )
    parser_ref.add_argument(
        "--overwrite",
        action="store_true",
        default=argparse.SUPPRESS,  # Remove from args if not provided
        help=extract_help_from_doc(ref, "overwrite"),
    )
    parser_ref.add_argument(
        "-t",
        "--threads",
        type=int,
        required=False,
        default=argparse.SUPPRESS,  # Remove from args if not provided
        help=extract_help_from_doc(ref, "threads"),
    )
    parser_ref.add_argument(
        "--logging_level",
        choices=["NOTSET", "DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL", "0", "10", "20", "30", "40", "50", "60", None],
        required=False,
        default=argparse.SUPPRESS,  # Remove from args if not provided
        help=extract_help_from_doc(ref, "logging_level"),
    )
    parser_ref.add_argument(
        "--save_logs",
        action="store_true",
        default=argparse.SUPPRESS,  # Remove from args if not provided
        help=extract_help_from_doc(ref, "save_logs"),
    )
    parser_ref.add_argument(
        "--log_out_dir",
        required=False,
        default=argparse.SUPPRESS,  # Remove from args if not provided
        help=extract_help_from_doc(ref, "log_out_dir"),
    )
    parser_ref.add_argument(
        "--verbose",
        action="store_true",
        default=argparse.SUPPRESS,  # Remove from args if not provided
        help=extract_help_from_doc(ref, "verbose"),
    )

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
        "--disable_qc_against_gene_matrix",
        dest="qc_against_gene_matrix",
        action="store_false",
        default=argparse.SUPPRESS,  # Remove from args if not provided
        help=extract_help_from_doc(count, "disable_qc_against_gene_matrix", disable=True),
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
        "--mm",
        action="store_true",
        default=argparse.SUPPRESS,  # Remove from args if not provided
        help=extract_help_from_doc(count, "mm"),
    )
    parser_count.add_argument(
        "--union",
        action="store_true",
        default=argparse.SUPPRESS,  # Remove from args if not provided
        help=extract_help_from_doc(count, "union"),
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
        help=extract_help_from_doc(count, "reference_genome_index"),
    )
    parser_count.add_argument(
        "--reference_genome_t2g",
        required=False,
        default=argparse.SUPPRESS,  # Remove from args if not provided
        help=extract_help_from_doc(count, "reference_genome_t2g"),
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
        "--kb_count_reference_genome_out_dir",
        required=False,
        default=argparse.SUPPRESS,  # Remove from args if not provided
        help=extract_help_from_doc(count, "kb_count_reference_genome_out_dir"),
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
        "--disable_summarize",
        action="store_true",
        default=argparse.SUPPRESS,  # Remove from args if not provided
        help=extract_help_from_doc(count, "disable_summarize"),
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
        "--num",
        action="store_true",
        default=argparse.SUPPRESS,  # Remove from args if not provided
        help=extract_help_from_doc(count, "num"),
    )
    parser_count.add_argument(
        "--parity_kb_count",
        type=str,
        choices=["single", "paired"],
        required=False,
        default=argparse.SUPPRESS,  # Remove from args if not provided
        help=extract_help_from_doc(count, "parity_kb_count"),
    )

    copy_arguments([parser_build, parser_info, parser_filter], parser_ref)  # allows parser_ref to accept all arguments from parser_build, parser_info, and parser_filter - if parser_ref does not already contain the argument, then it will inherent from the other functions, notably with positional args converted into keyword args, required False (since vk ref will necessarily handle this internally), and no help message displayed
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
        # Retrieve all subparsers from the parent parser
        subparsers_actions = [action for action in parent_parser._actions if isinstance(action, argparse._SubParsersAction)]
        for subparsers_action in subparsers_actions:
            # Get all subparsers and print help
            for choice, subparser in subparsers_action.choices.items():
                print("Subparser '{}'".format(choice))
                print(subparser.format_help())
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
        "info": parser_info,
        "filter": parser_filter,
        "sim": parser_sim,
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

        # Print list of mutated sequences if any are returned (this should only happen when out=None)
        if build_results:
            for mut_seq in build_results:
                print(mut_seq)

    # * info return
    if args.command == "info":
        # * ensure that all keys in params_dict correspond to the python parameters, and the values correspond to the command line values - see the vk build section in main for more details

        # (1) modify variable outside of args

        # combine with kwargs (if both params_dict and kwargs have the same key, params_dict takes precedence)
        params_dict = {**kwargs, **params_dict}

        # for pytest
        if os.getenv("TESTING") == "true":
            return params_dict

        info_results = info(**params_dict)

        # * optionally do something with info_results (e.g., save, or print to console)

    # * filter return
    if args.command == "filter":
        # * ensure that all keys in params_dict correspond to the python parameters, and the values correspond to the command line values - see the vk build section in main for more details

        # (1) modify variable outside of args

        # combine with kwargs (if both params_dict and kwargs have the same key, params_dict takes precedence)
        params_dict = {**kwargs, **params_dict}

        # for pytest
        if os.getenv("TESTING") == "true":
            return params_dict

        filter_results = filter(**params_dict)

        # * optionally do something with filter_results (e.g., save, or print to console)

    # * sim return
    if args.command == "sim":
        # * ensure that all keys in params_dict correspond to the python parameters, and the values correspond to the command line values - see the vk build section in main for more details

        # (1) modify variable outside of args

        # combine with kwargs (if both params_dict and kwargs have the same key, params_dict takes precedence)
        params_dict = {**kwargs, **params_dict}

        # for pytest
        if os.getenv("TESTING") == "true":
            return params_dict

        simulated_df_dict = sim(**params_dict)

        # * optionally do something with simulated_df_dict (e.g., save, or print to console)

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

        # (1) modify variable outside of args
        fastqs = params_dict.pop("fastqs")

        # combine with kwargs (if both params_dict and kwargs have the same key, params_dict takes precedence)
        params_dict = {**kwargs, **params_dict}

        # for pytest
        if os.getenv("TESTING") == "true":
            return fastqs, params_dict

        count_results = count(*fastqs, **params_dict)

        # * optionally do something with count_results (e.g., save, or print to console)


# * new Feb 2025 for pytest
if __name__ == "__main__":
    main()
