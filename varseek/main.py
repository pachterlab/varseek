"""main function for argparse."""
import argparse
import inspect
import math
import os
import re
import sys
from datetime import datetime

import pandas as pd

from .__init__ import __version__
from .utils import load_params, set_up_logger
from .varseek_build import build
from .varseek_clean import clean
from .varseek_count import count
from .varseek_fastqpp import fastqpp
from .varseek_filter import filter, prepare_filters_list
from .varseek_info import info
from .varseek_ref import ref
from .varseek_sim import sim
from .varseek_summarize import summarize

# Get current date and time for alphafold default foldername
dt_string = datetime.now().strftime("%Y_%m_%d-%H_%M")

logger = set_up_logger(logging_level_name=None, save_logs=False, log_dir="logs")


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


# Check that if `--config` is passed, no other arguments are set
def assert_only_config(args, parser):
    if args.config:
        vk_commands = {"build", "info", "filter", "sim", "clean", "summarize", "fastqpp", "ref", "count"}
        additional_acceptable_commands = {"dry_run"}
        additional_acceptable_commands_final = vk_commands.union(additional_acceptable_commands)
        for key, value in vars(args).items():
            # Ignore `config` itself
            if key == "config" or key in additional_acceptable_commands_final:
                continue
            # Check if the argument value differs from its default
            if value != parser.get_default(key):
                raise ValueError(f"If '--config' is passed, no other arguments (like '{key}') can be set.")


def copy_parser_arguments(parser_list, parser_target):
    for parser in parser_list:
        # Dynamically copy arguments from parser_build to parser_target
        for action in parser._actions:
            # Skip help action (default argparse behavior)
            if isinstance(action, argparse._HelpAction):
                continue

            # Check if any option string already exists in parser_target (eg already contains --technology or -x)
            if any(opt in parser_target._option_string_actions for opt in action.option_strings):
                continue  # Skip adding this argument if there's an overlap

            # Extract all option strings and argument attributes
            option_strings = action.option_strings
            kwargs = {
                "default": action.default,
                "type": action.type,
                "required": action.required,
                "help": action.help,
                "choices": action.choices,
                "metavar": action.metavar,
                "dest": action.dest,
                "action": action.__class__,  # Retain the action type
            }

            if not isinstance(action, (argparse._StoreTrueAction, argparse._StoreFalseAction)):
                kwargs["nargs"] = action.nargs  # Only include nargs for compatible actions

            # Filter out None values to avoid redundant keyword arguments
            kwargs = {key: value for key, value in kwargs.items() if value is not None}

            # Add the argument to the target parser
            parser_target.add_argument(*option_strings, **kwargs)

            return parser_target


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

    # NEW PARSER
    # build parser arguments
    build_desc = "Build a mutation-containing reference sequence (MCRS) file."

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
        "-s",
        "--sequences",
        type=str,
        nargs="+",
        required=True,
        help=extract_help_from_doc(build, "sequences"),
    )
    parser_build.add_argument(
        "-m",
        "--mutations",
        type=strpath_or_str_or_list_or_df,
        nargs="+",
        required=True,
        help=extract_help_from_doc(build, "mutations"),
    )
    parser_build.add_argument(
        "-w",
        "--w",
        default=54,
        type=int,
        required=False,
        help=extract_help_from_doc(build, "w"),
    )
    parser_build.add_argument(
        "-k",
        "--k",
        type=int,
        required=False,
        default=59,
        help=extract_help_from_doc(build, "k"),
    )
    parser_build.add_argument(
        "-ma",
        "--max_ambiguous",
        default=0,
        type=int,
        required=False,
        help=extract_help_from_doc(build, "max_ambiguous"),
    )
    parser_build.add_argument(
        "-mc",
        "--mut_column",
        default="mutation",
        type=str,
        required=False,
        help=extract_help_from_doc(build, "mut_column"),
    )
    parser_build.add_argument(
        "-sic",
        "--seq_id_column",
        default="seq_ID",
        type=str,
        required=False,
        help=extract_help_from_doc(build, "seq_id_column"),
    )
    parser_build.add_argument(
        "-mic",
        "--mut_id_column",
        default=None,
        required=False,
        help=extract_help_from_doc(build, "mut_id_column"),
    )
    parser_build.add_argument(
        "-gtf",
        "--gtf",
        default=None,
        required=False,
        help=extract_help_from_doc(build, "gtf"),
    )
    parser_build.add_argument(
        "-gtic",
        "--gtf_transcript_id_column",
        default=None,
        required=False,
        help=extract_help_from_doc(build, "gtf_transcript_id_column"),
    )
    parser_build.add_argument(
        "-tb",
        "--transcript_boundaries",
        default=False,
        action="store_true",
        help=extract_help_from_doc(build, "transcript_boundaries"),
    )
    parser_build.add_argument(
        "--identify_all_spliced_from_genome",
        default=False,
        action="store_true",
        help=extract_help_from_doc(build, "identify_all_spliced_from_genome"),
    )
    parser_build.add_argument(
        "-o",
        "--out",
        type=str,
        required=False,
        default=".",
        help=extract_help_from_doc(build, "out"),
    )
    parser_build.add_argument(
        "-r",
        "--reference_out_dir",
        default=None,
        required=False,
        help=extract_help_from_doc(build, "reference_out_dir"),
    )
    parser_build.add_argument(
        "-mfo",
        "--mcrs_fasta_out",
        default=None,
        required=False,
        help=extract_help_from_doc(build, "mcrs_fasta_out"),
    )
    parser_build.add_argument(
        "--mutations_updated_csv_out",
        default=None,
        required=False,
        help=extract_help_from_doc(build, "mutations_updated_csv_out"),
    )
    parser_build.add_argument(
        "--id_to_header_csv_out",
        default=None,
        required=False,
        help=extract_help_from_doc(build, "id_to_header_csv_out"),
    )
    parser_build.add_argument(
        "--mcrs_t2g_out",
        default=None,
        required=False,
        help=extract_help_from_doc(build, "mcrs_t2g_out"),
    )
    parser_build.add_argument(
        "--wt_mcrs_fasta_out",
        default=None,
        required=False,
        help=extract_help_from_doc(build, "wt_mcrs_fasta_out"),
    )
    parser_build.add_argument(
        "--wt_mcrs_t2g_out",
        default=None,
        required=False,
        help=extract_help_from_doc(build, "wt_mcrs_t2g_out"),
    )
    parser_build.add_argument(
        "--removed_variants_text_out",
        default=None,
        required=False,
        help=extract_help_from_doc(build, "removed_variants_text_out"),
    )
    parser_build.add_argument(
        "--filtering_report_text_out",
        default=None,
        required=False,
        help=extract_help_from_doc(build, "filtering_report_text_out"),
    )
    parser_build.add_argument(
        "-rmo",
        "--return_mutation_output",
        action="store_true",
        required=False,
        help=extract_help_from_doc(build, "return_mutation_output"),
    )
    parser_build.add_argument(
        "-smuc",
        "--save_mutations_updated_csv",
        action="store_true",
        required=False,
        help=extract_help_from_doc(build, "save_mutations_updated_csv"),
    )
    parser_build.add_argument(
        "-swmfat",
        "--save_wt_mcrs_fasta_and_t2g",
        action="store_true",
        required=False,
        help=extract_help_from_doc(build, "save_wt_mcrs_fasta_and_t2g"),
    )
    parser_build.add_argument(
        "-dsrvt",
        "--disable_save_removed_variants_text",
        action="store_false",
        required=False,
        help=extract_help_from_doc(build, "save_removed_variants_text"),
    )
    parser_build.add_argument(
        "-dsfrt",
        "--disable_save_filtering_report_text",
        action="store_false",
        required=False,
        help=extract_help_from_doc(build, "save_filtering_report_text"),
    )
    parser_build.add_argument(
        "-sfs",
        "--store_full_sequences",
        action="store_true",
        required=False,
        help=extract_help_from_doc(build, "store_full_sequences"),
    )
    parser_build.add_argument(
        "--translate",
        action="store_true",
        required=False,
        help=extract_help_from_doc(build, "translate"),
    )
    parser_build.add_argument(
        "-ts",
        "--translate_start",
        default=None,
        type=int_or_str,
        required=False,
        help=extract_help_from_doc(build, "translate_start"),
    )
    parser_build.add_argument(
        "-te",
        "--translate_end",
        default=None,
        type=int_or_str,
        required=False,
        help=extract_help_from_doc(build, "translate_end"),
    )
    parser_build.add_argument(
        "--dry_run",
        action="store_true",
        required=False,
        help=extract_help_from_doc(build, "dry_run"),
    )
    parser_build.add_argument(
        "--list_supported_databases",
        action="store_true",
        required=False,
        help=extract_help_from_doc(build, "list_supported_databases"),
    )
    parser_build.add_argument(
        "--overwrite",
        action="store_true",
        required=False,
        help=extract_help_from_doc(build, "overwrite"),
    )
    parser_build.add_argument(
        "-q",
        "--quiet",
        action="store_false",
        required=False,
        help="Do not print progress information.",
    )

    # Additional kwargs arguments that I still want as command-line options
    parser_build.add_argument(
        "--insertion_size_limit",
        default=lambda x: int(x) if x is not None else None,
        type=int,
        required=False,
        help=extract_help_from_doc(build, "insertion_size_limit"),
    )
    parser_build.add_argument(
        "--min_seq_len",
        default=0,
        type=int,
        required=False,
        help=extract_help_from_doc(build, "min_seq_len"),
    )
    parser_build.add_argument(
        "-dofr",
        "--disable_optimize_flanking_regions",
        action="store_false",
        required=False,
        help=extract_help_from_doc(build, "optimize_flanking_regions", disable=True),
    )
    parser_build.add_argument(
        "-drswk",
        "--disable_remove_seqs_with_wt_kmers",
        action="store_false",
        required=False,
        help=extract_help_from_doc(build, "remove_seqs_with_wt_kmers", disable=True),
    )
    parser_build.add_argument(
        "-riol",
        "--required_insertion_overlap_length",
        default=6,
        type=int_or_str,
        required=False,
        help=extract_help_from_doc(build, "required_insertion_overlap_length"),
    )
    parser_build.add_argument(
        "-dmi",
        "--disable_merge_identical",
        action="store_false",
        required=False,
        help=extract_help_from_doc(build, "merge_identical", disable=True),
    )
    parser_build.add_argument(
        "-vs",
        "--vcrs_strandedness",
        action="store_true",
        required=False,
        help=extract_help_from_doc(build, "vcrs_strandedness"),
    )
    parser_build.add_argument(
        "-droh",
        "--disable_use_IDs",
        action="store_false",
        required=False,
        help=extract_help_from_doc(build, "use_IDs", disable=True),
    )
    parser_build.add_argument(
        "--cosmic_release",
        type=lambda x: int(x) if x is not None else None,
        required=False,
        help=extract_help_from_doc(build, "cosmic_release"),
    )
    parser_build.add_argument(
        "--cosmic_grch",
        type=lambda x: int(x) if x is not None else None,
        required=False,
        help=extract_help_from_doc(build, "cosmic_grch"),
    )
    parser_build.add_argument(
        "--cosmic_email",
        required=False,
        help=extract_help_from_doc(build, "cosmic_email"),
    )
    parser_build.add_argument(
        "--cosmic_password",
        required=False,
        help=extract_help_from_doc(build, "cosmic_password"),
    )
    parser_build.add_argument(
        "-dsf",
        "--disable_save_files",
        action="store_false",
        required=False,
        help=extract_help_from_doc(build, "save_files", disable=True),
    )

    # NEW PARSER
    info_desc = "Takes in the input directory containing with the MCRS fasta file generated from varseek build, and returns a dataframe with additional columns containing information about the mutations."
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
        required=True,
        help=extract_help_from_doc(info, "input_dir"),
    )
    parser_info.add_argument(
        "-c",
        "--columns_to_include",
        type=strpath_or_list_like_of_strings,
        nargs="+",
        required=False,
        default=("number_of_mutations_in_this_gene_total", "number_of_alignments_to_normal_human_reference", "pseudoaligned_to_human_reference_despite_not_truly_aligning", "longest_homopolymer_length", "triplet_complexity"),
        help=extract_help_from_doc(info, "columns_to_include"),
    )
    parser_info.add_argument(
        "-k",
        "--k",
        type=int,
        required=False,
        default=59,
        help=extract_help_from_doc(info, "k"),
    )
    parser_info.add_argument(
        "--max_ambiguous_mcrs",
        type=int,
        required=False,
        default=0,
        help=extract_help_from_doc(info, "max_ambiguous_mcrs"),
    )
    parser_info.add_argument(
        "--max_ambiguous_reference",
        type=int,
        required=False,
        default=0,
        help=extract_help_from_doc(info, "max_ambiguous_reference"),
    )
    parser_info.add_argument(
        "--mcrs_fasta",
        required=False,
        help=extract_help_from_doc(info, "mcrs_fasta"),
    )
    parser_info.add_argument(
        "--mutations_updated_csv",
        required=False,
        help=extract_help_from_doc(info, "mutations_updated_csv"),
    )
    parser_info.add_argument(
        "--id_to_header_csv",
        required=False,
        help=extract_help_from_doc(info, "id_to_header_csv"),
    )
    parser_info.add_argument(
        "--gtf",
        required=False,
        help=extract_help_from_doc(info, "gtf"),
    )
    parser_info.add_argument(
        "--dlist_reference_source",
        required=False,
        default="T2T",
        help=extract_help_from_doc(info, "dlist_reference_source"),
    )
    parser_info.add_argument(
        "--dlist_reference_genome_fasta",
        required=False,
        help=extract_help_from_doc(info, "dlist_reference_genome_fasta"),
    )
    parser_info.add_argument(
        "--dlist_reference_cdna_fasta",
        type=str,
        required=False,
        help=extract_help_from_doc(info, "dlist_reference_cdna_fasta"),
    )
    parser_info.add_argument(
        "--dlist_reference_gtf",
        type=str,
        required=False,
        help=extract_help_from_doc(info, "dlist_reference_gtf"),
    )
    parser_info.add_argument(
        "--dlist_reference_ensembl_release",
        type=int,
        required=False,
        default=111,
        help=extract_help_from_doc(info, "dlist_reference_ensembl_release"),
    )
    parser_info.add_argument(
        "--mcrs_id_column",
        type=str,
        required=False,
        default="mcrs_id",
        help=extract_help_from_doc(info, "mcrs_id_column"),
    )
    parser_info.add_argument(
        "--mcrs_sequence_column",
        type=str,
        required=False,
        default="mutant_sequence",
        help=extract_help_from_doc(info, "mcrs_sequence_column"),
    )
    parser_info.add_argument(
        "--mcrs_source_column",
        type=str,
        required=False,
        default="mcrs_source",
        help=extract_help_from_doc(info, "mcrs_source_column"),
    )
    parser_info.add_argument(
        "--mut_column",
        type=str,
        required=False,
        default="mutation",
        help=extract_help_from_doc(info, "mut_column"),
    )
    parser_info.add_argument(
        "--seq_id_column",
        type=str,
        required=False,
        default="seq_ID",
        help=extract_help_from_doc(info, "seq_id_column"),
    )
    parser_info.add_argument(
        "--mutation_cdna_column",
        type=str,
        required=False,
        default="mutation",
        help=extract_help_from_doc(info, "mutation_cdna_column"),
    )
    parser_info.add_argument(
        "--seq_id_cdna_column",
        type=str,
        required=False,
        default="seq_ID",
        help=extract_help_from_doc(info, "seq_id_cdna_column"),
    )
    parser_info.add_argument(
        "--mutation_genome_column",
        type=str,
        required=False,
        default="mutation_genome",
        help=extract_help_from_doc(info, "mutation_genome_column"),
    )
    parser_info.add_argument(
        "--seq_id_genome_column",
        type=str,
        required=False,
        default="chromosome",
        help=extract_help_from_doc(info, "seq_id_genome_column"),
    )
    parser_info.add_argument(
        "-o",
        "--out",
        required=False,
        help=extract_help_from_doc(info, "out"),
    )
    parser_info.add_argument(
        "-r",
        "--reference_out_dir",
        required=False,
        help=extract_help_from_doc(info, "reference_out_dir"),
    )
    parser_info.add_argument(
        "--mutations_updated_vk_info_csv_out",
        required=False,
        help=extract_help_from_doc(info, "mutations_updated_vk_info_csv_out"),
    )
    parser_info.add_argument(
        "--mutations_updated_exploded_vk_info_csv_out",
        required=False,
        help=extract_help_from_doc(info, "mutations_updated_exploded_vk_info_csv_out"),
    )
    parser_info.add_argument(
        "--dlist_genome_fasta_out",
        required=False,
        help=extract_help_from_doc(info, "dlist_genome_fasta_out"),
    )
    parser_info.add_argument(
        "--dlist_cdna_fasta_out",
        required=False,
        help=extract_help_from_doc(info, "dlist_cdna_fasta_out"),
    )
    parser_info.add_argument(
        "--dlist_combined_fasta_out",
        required=False,
        help=extract_help_from_doc(info, "dlist_combined_fasta_out"),
    )
    parser_info.add_argument(
        "--save_mutations_updated_exploded_vk_info_csv",
        action="store_true",
        required=False,
        help=extract_help_from_doc(info, "save_mutations_updated_exploded_vk_info_csv"),
    )
    parser_info.add_argument(
        "--make_pyfastx_summary_file",
        action="store_true",
        required=False,
        help=extract_help_from_doc(info, "make_pyfastx_summary_file"),
    )
    parser_info.add_argument(
        "--make_kat_histogram",
        action="store_true",
        required=False,
        help=extract_help_from_doc(info, "make_kat_histogram"),
    )
    parser_info.add_argument(
        "--dry_run",
        action="store_true",
        required=False,
        help=extract_help_from_doc(info, "dry_run"),
    )
    parser_info.add_argument(
        "--list_columns",
        action="store_true",
        required=False,
        help=extract_help_from_doc(info, "list_columns"),
    )
    parser_info.add_argument(
        "--list_d_list_values",
        action="store_true",
        required=False,
        help=extract_help_from_doc(info, "list_d_list_values"),
    )
    parser_info.add_argument(
        "--overwrite",
        action="store_true",
        required=False,
        help=extract_help_from_doc(info, "overwrite"),
    )
    parser_info.add_argument(
        "--threads",
        type=int,
        default=2,
        required=False,
        help=extract_help_from_doc(info, "threads"),
    )
    parser_info.add_argument(
        "-q",
        "--quiet",
        action="store_false",
        required=False,
        help="Do not print progress information.",
    )
    # kwargs
    parser_info.add_argument(
        "-w",
        type=int,
        required=False,
        default=54,
        help=extract_help_from_doc(info, "w"),
    )
    parser_info.add_argument(
        "--bowtie2_path",
        required=False,
        help=extract_help_from_doc(info, "bowtie2_path"),
    )
    parser_info.add_argument(
        "-vs",
        "--vcrs_strandedness",
        action="store_true",
        required=False,
        help=extract_help_from_doc(info, "vcrs_strandedness"),
    )
    parser_info.add_argument(
        "--near_splice_junction_threshold",
        type=int,
        required=False,
        default=10,
        help=extract_help_from_doc(info, "near_splice_junction_threshold"),
    )
    parser_info.add_argument(
        "--reference_cdna_fasta",
        required=False,
        help=extract_help_from_doc(info, "reference_cdna_fasta"),
    )
    parser_info.add_argument(
        "--reference_genome_fasta",
        required=False,
        help=extract_help_from_doc(info, "reference_genome_fasta"),
    )
    parser_info.add_argument(
        "--mutations_csv",
        required=False,
        help=extract_help_from_doc(info, "mutations_csv"),
    )

    # NEW PARSER
    filter_desc = "Filter mutations based on the provided filters and save the filtered mutations to a fasta file."
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
        required=True,
        help=extract_help_from_doc(filter, "input_dir"),
    )
    parser_filter.add_argument(
        "-f",
        "--filters",
        type=strpath_or_list_like_of_strings,
        nargs="+",
        required=True,
        help=extract_help_from_doc(filter, "filters"),
    )
    parser_filter.add_argument(
        "--mutations_updated_vk_info_csv",
        required=False,
        help=extract_help_from_doc(filter, "mutations_updated_vk_info_csv"),
    )
    parser_filter.add_argument(
        "--mutations_updated_exploded_vk_info_csv",
        required=False,
        help=extract_help_from_doc(filter, "mutations_updated_exploded_vk_info_csv"),
    )
    parser_filter.add_argument(
        "--id_to_header_csv",
        required=False,
        help=extract_help_from_doc(filter, "id_to_header_csv"),
    )
    parser_filter.add_argument(
        "--dlist_fasta",
        required=False,
        help=extract_help_from_doc(filter, "dlist_fasta"),
    )
    parser_filter.add_argument(
        "-o",
        "--out",
        required=False,
        help=extract_help_from_doc(filter, "out"),
    )
    parser_filter.add_argument(
        "--mutations_updated_filtered_csv_out",
        required=False,
        help=extract_help_from_doc(filter, "mutations_updated_filtered_csv_out"),
    )
    parser_filter.add_argument(
        "--mutations_updated_exploded_filtered_csv_out",
        required=False,
        help=extract_help_from_doc(filter, "mutations_updated_exploded_filtered_csv_out"),
    )
    parser_filter.add_argument(
        "--id_to_header_filtered_csv_out",
        required=False,
        help=extract_help_from_doc(filter, "id_to_header_filtered_csv_out"),
    )
    parser_filter.add_argument(
        "--dlist_filtered_fasta_out",
        required=False,
        help=extract_help_from_doc(filter, "dlist_filtered_fasta_out"),
    )
    parser_filter.add_argument(
        "--mcrs_filtered_fasta_out",
        required=False,
        help=extract_help_from_doc(filter, "mcrs_filtered_fasta_out"),
    )
    parser_filter.add_argument(
        "--mcrs_t2g_filtered_out",
        required=False,
        help=extract_help_from_doc(filter, "mcrs_t2g_filtered_out"),
    )
    parser_filter.add_argument(
        "--wt_mcrs_filtered_fasta_out",
        required=False,
        help=extract_help_from_doc(filter, "wt_mcrs_filtered_fasta_out"),
    )
    parser_filter.add_argument(
        "--wt_mcrs_t2g_filtered_out",
        required=False,
        help=extract_help_from_doc(filter, "wt_mcrs_t2g_filtered_out"),
    )
    parser_filter.add_argument(
        "--save_wt_mcrs_fasta_and_t2g",
        action="store_true",
        required=False,
        help=extract_help_from_doc(filter, "save_wt_mcrs_fasta_and_t2g"),
    )
    parser_filter.add_argument(
        "--save_mutations_updated_filtered_csvs",
        action="store_true",
        required=False,
        help=extract_help_from_doc(filter, "save_mutations_updated_filtered_csvs"),
    )
    parser_filter.add_argument(
        "--return_mutations_updated_filtered_csv_df",
        action="store_true",
        required=False,
        help=extract_help_from_doc(filter, "return_mutations_updated_filtered_csv_df"),
    )
    parser_filter.add_argument(
        "--dry_run",
        action="store_true",
        required=False,
        help=extract_help_from_doc(filter, "dry_run"),
    )
    parser_filter.add_argument(
        "--list_filter_rules",
        action="store_true",
        required=False,
        help=extract_help_from_doc(filter, "list_filter_rules"),
    )
    parser_filter.add_argument(
        "--overwrite",
        action="store_true",
        required=False,
        help=extract_help_from_doc(filter, "overwrite"),
    )
    parser_filter.add_argument(
        "-q",
        "--quiet",
        action="store_false",
        required=False,
        help="Do not print progress information.",
    )
    # kwargs
    parser_filter.add_argument(
        "--filter_all_dlists",
        action="store_true",
        required=False,
        help=extract_help_from_doc(filter, "filter_all_dlists"),
    )
    parser_filter.add_argument(
        "--dlist_genome_fasta",
        required=False,
        help=extract_help_from_doc(filter, "dlist_genome_fasta"),
    )
    parser_filter.add_argument(
        "--dlist_cdna_fasta",
        required=False,
        help=extract_help_from_doc(filter, "dlist_cdna_fasta"),
    )
    parser_filter.add_argument(
        "--dlist_genome_filtered_fasta_out",
        required=False,
        help=extract_help_from_doc(filter, "dlist_genome_filtered_fasta_out"),
    )
    parser_filter.add_argument(
        "--dlist_cdna_filtered_fasta_out",
        required=False,
        help=extract_help_from_doc(filter, "dlist_cdna_filtered_fasta_out"),
    )
    parser_filter.add_argument(
        "--disable_save_mcrs_filtered_fasta_and_t2g",
        action="store_false",
        required=False,
        help=extract_help_from_doc(filter, "save_mcrs_filtered_fasta_and_t2g", disable=True),
    )
    parser_filter.add_argument(
        "--disable_use_IDs",
        action="store_false",
        required=False,
        help=extract_help_from_doc(filter, "use_IDs", disable=True),
    )

    # NEW PARSER
    sim_desc = "Create synthetic RNA-seq dataset with mutation reads."
    parser_sim = parent_subparsers.add_parser(
        "sim",
        parents=[parent],
        description=sim_desc,
        help=sim_desc,
        add_help=True,
        formatter_class=CustomHelpFormatter,
    )
    parser_sim.add_argument(
        "-m",
        "--mutations",
        default=None,
        type=strpath_or_strnonpath_or_df,
        required=True,
        help=extract_help_from_doc(sim, "mutations"),
    )
    parser_sim.add_argument(
        "--number_of_mutations_to_sample",
        default=1500,
        type=int,
        required=False,
        help=extract_help_from_doc(sim, "number_of_mutations_to_sample"),
    )
    parser_sim.add_argument(
        "--number_of_reads_per_mutation_m",
        default="all",
        required=False,
        help=extract_help_from_doc(sim, "number_of_reads_per_mutation_m"),
    )
    parser_sim.add_argument(
        "--number_of_reads_per_mutation_w",
        default="all",
        required=False,
        help=extract_help_from_doc(sim, "number_of_reads_per_mutation_w"),
    )
    parser_sim.add_argument(
        "--sample_m_and_w_from_same_locations",
        action="store_true",
        help=extract_help_from_doc(sim, "sample_m_and_w_from_same_locations"),
    )
    parser_sim.add_argument(
        "--with_replacement",
        action="store_true",
        help=extract_help_from_doc(sim, "with_replacement"),
    )
    parser_sim.add_argument(
        "--strand",
        choices=["f", "r", "both", "random", None],
        help=extract_help_from_doc(sim, "strand"),
    )
    parser_sim.add_argument(
        "--read_length",
        default=150,
        type=int,
        required=False,
        help=extract_help_from_doc(sim, "read_length"),
    )
    parser_sim.add_argument(
        "-f",
        "--filters",
        nargs="*",  # Accept multiple sequential filters or a single JSON file
        type=str,
        required=False,
        help=extract_help_from_doc(sim, "filters"),
    )
    parser_sim.add_argument(
        "--add_noise_sequencing_error",
        action="store_true",
        help=extract_help_from_doc(sim, "add_noise_sequencing_error"),
    )
    parser_sim.add_argument(
        "--add_noise_base_quality",
        action="store_true",
        help=extract_help_from_doc(sim, "add_noise_base_quality"),
    )
    parser_sim.add_argument(
        "--error_rate",
        required=False,
        default=0.0001,
        help=extract_help_from_doc(sim, "error_rate"),
    )
    parser_sim.add_argument(
        "--error_distribution",
        required=False,
        default=(0.85, 0.1, 0.05),
        help=extract_help_from_doc(sim, "error_distribution"),
    )
    parser_sim.add_argument(
        "--max_errors",
        required=False,
        default=float("inf"),
        help=extract_help_from_doc(sim, "max_errors"),
    )
    parser_sim.add_argument(
        "--mutant_sequence_read_parent_column",
        required=False,
        default="mutant_sequence_read_parent",
        help=extract_help_from_doc(sim, "mutant_sequence_read_parent_column"),
    )
    parser_sim.add_argument(
        "--wt_sequence_read_parent_column",
        required=False,
        default="wt_sequence_read_parent",
        help=extract_help_from_doc(sim, "wt_sequence_read_parent_column"),
    )
    parser_sim.add_argument(
        "--mutant_sequence_read_parent_rc_column",
        required=False,
        default="mutant_sequence_read_parent_rc",
        help=extract_help_from_doc(sim, "mutant_sequence_read_parent_rc_column"),
    )
    parser_sim.add_argument(
        "--wt_sequence_read_parent_rc_column",
        required=False,
        default="wt_sequence_read_parent_rc",
        help=extract_help_from_doc(sim, "wt_sequence_read_parent_rc_column"),
    )
    parser_sim.add_argument(
        "--reads_fastq_parent",
        required=False,
        help=extract_help_from_doc(sim, "reads_fastq_parent"),
    )
    parser_sim.add_argument(
        "--reads_csv_parent",
        required=False,
        help=extract_help_from_doc(sim, "reads_csv_parent"),
    )
    parser_sim.add_argument(
        "--out",
        required=False,
        type=str,
        default=".",
        help=extract_help_from_doc(sim, "out"),
    )
    parser_sim.add_argument(
        "--reads_fastq_out",
        required=False,
        help=extract_help_from_doc(sim, "reads_fastq_out"),
    )
    parser_sim.add_argument(
        "--mutations_updated_csv_out",
        required=False,
        help=extract_help_from_doc(sim, "mutations_updated_csv_out"),
    )
    parser_sim.add_argument(
        "--reads_csv_out",
        required=False,
        help=extract_help_from_doc(sim, "reads_csv_out"),
    )
    parser_sim.add_argument(
        "--disable_save_mutations_updated_csv",
        action="store_true",
        help=extract_help_from_doc(sim, "save_mutations_updated_csv", disable=True),
    )
    parser_sim.add_argument(
        "--disable_save_read_csv",
        action="store_true",
        help=extract_help_from_doc(sim, "save_read_csv", disable=True),
    )
    parser_sim.add_argument(
        "--vk_build_out_dir",
        required=False,
        type=str,
        default=".",
        help=extract_help_from_doc(sim, "vk_build_out_dir"),
    )
    parser_sim.add_argument(
        "--sequences",
        required=False,
        help=extract_help_from_doc(sim, "sequences"),
    )
    parser_sim.add_argument(
        "--seq_id_column",
        required=False,
        type=str,
        default="seq_ID",
        help=extract_help_from_doc(sim, "seq_id_column"),
    )
    parser_sim.add_argument(
        "--mut_column",
        required=False,
        type=str,
        default="mutation",
        help=extract_help_from_doc(sim, "mut_column"),
    )
    parser_sim.add_argument(
        "--k",
        required=False,
        type=int,
        default=59,
        help=extract_help_from_doc(sim, "k"),
    )
    parser_sim.add_argument(
        "--w",
        required=False,
        type=int,
        default=54,
        help=extract_help_from_doc(sim, "w"),
    )
    parser_sim.add_argument(
        "--sequences_cdna",
        required=False,
        help=extract_help_from_doc(sim, "sequences_cdna"),
    )
    parser_sim.add_argument(
        "--seq_id_column_cdna",
        required=False,
        help=extract_help_from_doc(sim, "seq_id_column_cdna"),
    )
    parser_sim.add_argument(
        "--mut_column_cdna",
        required=False,
        help=extract_help_from_doc(sim, "mut_column_cdna"),
    )
    parser_sim.add_argument(
        "--sequences_genome",
        required=False,
        help=extract_help_from_doc(sim, "sequences_genome"),
    )
    parser_sim.add_argument(
        "--seq_id_column_genome",
        required=False,
        help=extract_help_from_doc(sim, "seq_id_column_genome"),
    )
    parser_sim.add_argument(
        "--mut_column_genome",
        required=False,
        help=extract_help_from_doc(sim, "mut_column_genome"),
    )
    parser_sim.add_argument(
        "--seed",
        required=False,
        help=extract_help_from_doc(sim, "seed"),
    )
    parser_sim.add_argument(
        "--gzip_output_fastq",
        action="store_true",
        help=extract_help_from_doc(sim, "gzip_output_fastq"),
    )
    parser_sim.add_argument(
        "--dry_run",
        action="store_true",
        help=extract_help_from_doc(sim, "dry_run"),
    )
    parser_sim.add_argument(
        "--verbose",
        action="store_true",
        help=extract_help_from_doc(sim, "verbose"),
    )
    parser_sim.add_argument(
        "-q",
        "--quiet",
        action="store_false",
        required=False,
        help="Do not print progress information.",
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
        required=False,
        help=extract_help_from_doc(fastqpp, "technology"),
    )
    parser_fastqpp.add_argument(
        "--multiplexed",
        required=False,
        action="store_true",
        help=extract_help_from_doc(fastqpp, "multiplexed"),
    )
    parser_fastqpp.add_argument(
        "--parity",
        required=False,
        type=str,
        choices=["single", "paired"],
        default="single",
        help=extract_help_from_doc(fastqpp, "parity"),
    )
    parser_fastqpp.add_argument(
        "--quality_control_fastqs",
        required=False,
        action="store_true",
        help=extract_help_from_doc(fastqpp, "quality_control_fastqs"),
    )
    parser_fastqpp.add_argument(
        "--cut_mean_quality",
        required=False,
        type=int,
        default=13,
        help=extract_help_from_doc(fastqpp, "cut_mean_quality"),
    )
    parser_fastqpp.add_argument(
        "--cut_window_size",
        required=False,
        type=int,
        default=4,
        help=extract_help_from_doc(fastqpp, "cut_window_size"),
    )
    parser_fastqpp.add_argument(
        "--qualified_quality_phred",
        required=False,
        type=int,
        default=0,
        help=extract_help_from_doc(fastqpp, "qualified_quality_phred"),
    )
    parser_fastqpp.add_argument(
        "--unqualified_percent_limit",
        required=False,
        type=int,
        default=100,
        help=extract_help_from_doc(fastqpp, "unqualified_percent_limit"),
    )
    parser_fastqpp.add_argument(
        "--max_ambiguous",
        required=False,
        type=int,
        default=50,
        help=extract_help_from_doc(fastqpp, "max_ambiguous"),
    )
    parser_fastqpp.add_argument(
        "--min_read_len",
        required=False,
        type=lambda x: int(x) if x is not None else None,
        default=None,
        help=extract_help_from_doc(fastqpp, "min_read_len"),
    )
    parser_fastqpp.add_argument(
        "--fastqc_and_multiqc",
        required=False,
        action="store_true",
        help=extract_help_from_doc(fastqpp, "fastqc_and_multiqc"),
    )
    parser_fastqpp.add_argument(
        "--replace_low_quality_bases_with_N",
        required=False,
        action="store_true",
        help=extract_help_from_doc(fastqpp, "replace_low_quality_bases_with_N"),
    )
    parser_fastqpp.add_argument(
        "--min_base_quality",
        required=False,
        type=int,
        default=13,
        help=extract_help_from_doc(fastqpp, "min_base_quality"),
    )
    parser_fastqpp.add_argument(
        "--split_reads_by_Ns",
        required=False,
        action="store_true",
        help=extract_help_from_doc(fastqpp, "split_reads_by_Ns"),
    )
    parser_fastqpp.add_argument(
        "--concatenate_paired_fastqs",
        required=False,
        action="store_true",
        help=extract_help_from_doc(fastqpp, "concatenate_paired_fastqs"),
    )
    parser_fastqpp.add_argument(
        "-o",
        "--out",
        type=str,
        required=False,
        default=".",
        help=extract_help_from_doc(fastqpp, "out"),
    )
    parser_fastqpp.add_argument(
        "--delete_intermediate_files",
        required=False,
        action="store_true",
        help=extract_help_from_doc(fastqpp, "delete_intermediate_files"),
    )
    parser_fastqpp.add_argument(
        "--dry_run",
        required=False,
        action="store_true",
        help=extract_help_from_doc(fastqpp, "dry_run"),
    )
    parser_fastqpp.add_argument(
        "--overwrite",
        required=False,
        action="store_true",
        help=extract_help_from_doc(fastqpp, "overwrite"),
    )
    parser_fastqpp.add_argument(
        "--disable_sort_fastqs",
        action="store_false",
        required=False,
        help=extract_help_from_doc(fastqpp, "dry_run", disable=True),
    )
    parser_fastqpp.add_argument(
        "-t",
        "--threads",
        type=int,
        required=False,
        default=2,
        help=extract_help_from_doc(fastqpp, "threads"),
    )
    parser_fastqpp.add_argument(
        "-q",
        "--quiet",
        action="store_false",
        required=False,
        help="Do not print progress information.",
    )
    # kwargs
    parser_fastqpp.add_argument(
        "--fastp_path",
        required=False,
        default=None,
        help=extract_help_from_doc(fastqpp, "fastp_path"),
    )
    parser_fastqpp.add_argument(
        "--seqtk_path",
        required=False,
        default=None,
        help=extract_help_from_doc(fastqpp, "seqtk_path"),
    )
    parser_fastqpp.add_argument(
        "--fastqc_path",
        required=False,
        default=None,
        help=extract_help_from_doc(fastqpp, "fastqc_path"),
    )
    parser_fastqpp.add_argument(
        "--multiqc_path",
        required=False,
        default=None,
        help=extract_help_from_doc(fastqpp, "multiqc_path"),
    )
    parser_fastqpp.add_argument(
        "--quality_control_fastqs_out_suffix",
        required=False,
        default=None,
        help=extract_help_from_doc(fastqpp, "quality_control_fastqs_out_suffix"),
    )
    parser_fastqpp.add_argument(
        "--replace_low_quality_bases_with_N_out_suffix",
        required=False,
        default=None,
        help=extract_help_from_doc(fastqpp, "replace_low_quality_bases_with_N_out_suffix"),
    )
    parser_fastqpp.add_argument(
        "--split_by_N_out_suffix",
        required=False,
        default=None,
        help=extract_help_from_doc(fastqpp, "split_by_N_out_suffix"),
    )
    parser_fastqpp.add_argument(
        "--concatenate_paired_fastqs_out_suffix",
        required=False,
        default=None,
        help=extract_help_from_doc(fastqpp, "concatenate_paired_fastqs_out_suffix"),
    )

    # NEW PARSER
    clean_desc = "Run standard processing on the mutation count matrix."
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
        help=extract_help_from_doc(clean, "min_counts"),
    )
    parser_clean.add_argument(
        "--use_binary_matrix",
        action="store_true",
        required=False,
        help=extract_help_from_doc(clean, "use_binary_matrix"),
    )
    parser_clean.add_argument(
        "--drop_empty_columns",
        action="store_true",
        required=False,
        help=extract_help_from_doc(clean, "drop_empty_columns"),
    )
    parser_clean.add_argument(
        "--apply_single_end_mode_on_paired_end_data_correction",
        action="store_true",
        required=False,
        help=extract_help_from_doc(clean, "apply_single_end_mode_on_paired_end_data_correction"),
    )
    parser_clean.add_argument(
        "--split_reads_by_Ns",
        action="store_true",
        required=False,
        help=extract_help_from_doc(clean, "split_reads_by_Ns"),
    )
    parser_clean.add_argument(
        "--apply_dlist_correction",
        action="store_true",
        required=False,
        help=extract_help_from_doc(clean, "apply_dlist_correction"),
    )
    parser_clean.add_argument(
        "--qc_against_gene_matrix",
        action="store_true",
        required=False,
        help=extract_help_from_doc(clean, "qc_against_gene_matrix"),
    )
    parser_clean.add_argument(
        "--filter_cells_by_min_counts",
        required=False,
        help=extract_help_from_doc(clean, "filter_cells_by_min_counts"),
    )
    parser_clean.add_argument(
        "--filter_cells_by_min_genes",
        required=False,
        help=extract_help_from_doc(clean, "filter_cells_by_min_genes"),
    )
    parser_clean.add_argument(
        "--filter_genes_by_min_cells",
        required=False,
        help=extract_help_from_doc(clean, "filter_genes_by_min_cells"),
    )
    parser_clean.add_argument(
        "--filter_cells_by_max_mt_content",
        required=False,
        help=extract_help_from_doc(clean, "filter_cells_by_max_mt_content"),
    )
    parser_clean.add_argument(
        "--doublet_detection",
        action="store_true",
        required=False,
        help=extract_help_from_doc(clean, "doublet_detection"),
    )
    parser_clean.add_argument(
        "--remove_doublets",
        action="store_true",
        required=False,
        help=extract_help_from_doc(clean, "remove_doublets"),
    )
    parser_clean.add_argument(
        "--cpm_normalization",
        action="store_true",
        required=False,
        help=extract_help_from_doc(clean, "cpm_normalization"),
    )
    parser_clean.add_argument(
        "--sum_rows",
        action="store_true",
        required=False,
        help=extract_help_from_doc(clean, "sum_rows"),
    )
    parser_clean.add_argument(
        "--mcrs_id_set_to_exclusively_keep",
        nargs="+",
        type=str,
        required=False,
        help=extract_help_from_doc(clean, "mcrs_id_set_to_exclusively_keep"),
    )
    parser_clean.add_argument(
        "--mcrs_id_set_to_exclude",
        nargs="+",
        type=str,
        required=False,
        help=extract_help_from_doc(clean, "mcrs_id_set_to_exclude"),
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
        required=False,
        help=extract_help_from_doc(clean, "k"),
    )
    parser_clean.add_argument(
        "--mm",
        action="store_true",
        required=False,
        help=extract_help_from_doc(clean, "mm"),
    )
    parser_clean.add_argument(
        "--union",
        action="store_true",
        required=False,
        help=extract_help_from_doc(clean, "union"),
    )
    parser_clean.add_argument(
        "--parity",
        required=False,
        type=str,
        choices=["single", "paired"],
        default="single",
        help=extract_help_from_doc(clean, "parity"),
    )
    parser_clean.add_argument(
        "--multiplexed",
        action="store_true",
        required=False,
        help=extract_help_from_doc(clean, "multiplexed"),
    )
    parser_clean.add_argument(
        "--sort_fastqs",
        action="store_true",
        required=False,
        help=extract_help_from_doc(clean, "sort_fastqs"),
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
        help=extract_help_from_doc(clean, "vk_ref_dir"),
    )
    parser_clean.add_argument(
        "--vcrs_index",
        required=False,
        help=extract_help_from_doc(clean, "vcrs_index"),
    )
    parser_clean.add_argument(
        "--vcrs_t2g",
        required=False,
        help=extract_help_from_doc(clean, "vcrs_t2g"),
    )
    parser_clean.add_argument(
        "--mcrs_fasta",
        required=False,
        help=extract_help_from_doc(clean, "mcrs_fasta"),
    )
    parser_clean.add_argument(
        "--id_to_header_csv",
        required=False,
        help=extract_help_from_doc(clean, "id_to_header_csv"),
    )
    parser_clean.add_argument(
        "--dlist_fasta",
        required=False,
        help=extract_help_from_doc(clean, "dlist_fasta"),
    )
    parser_clean.add_argument(
        "--mutations_updated_csv",
        required=False,
        help=extract_help_from_doc(clean, "mutations_updated_csv"),
    )
    parser_clean.add_argument(
        "--mcrs_id_column",
        required=False,
        help=extract_help_from_doc(clean, "mcrs_id_column"),
    )
    parser_clean.add_argument(
        "--mutations_updated_csv_columns",
        nargs="+",
        type=str,
        required=False,
        help=extract_help_from_doc(clean, "mutations_updated_csv_columns"),
    )
    parser_clean.add_argument(
        "--adata_reference_genome",
        required=False,
        help=extract_help_from_doc(clean, "adata_reference_genome"),
    )
    parser_clean.add_argument(
        "--kb_count_vcrs_dir",
        required=False,
        help=extract_help_from_doc(clean, "kb_count_vcrs_dir"),
    )
    parser_clean.add_argument(
        "--kb_count_reference_genome_dir",
        required=False,
        help=extract_help_from_doc(clean, "kb_count_reference_genome_dir"),
    )
    parser_clean.add_argument(
        "--out",
        required=False,
        help=extract_help_from_doc(clean, "out"),
    )
    parser_clean.add_argument(
        "--adata_vcrs_clean_out",
        required=False,
        help=extract_help_from_doc(clean, "adata_vcrs_clean_out"),
    )
    parser_clean.add_argument(
        "--adata_reference_genome_clean_out",
        required=False,
        help=extract_help_from_doc(clean, "adata_reference_genome_clean_out"),
    )
    parser_clean.add_argument(
        "--vcf_out",
        required=False,
        help=extract_help_from_doc(clean, "vcf_out"),
    )
    parser_clean.add_argument(
        "--save_vcf",
        action="store_true",
        required=False,
        help=extract_help_from_doc(clean, "save_vcf"),
    )
    parser_clean.add_argument(
        "--dry_run",
        action="store_true",
        required=False,
        help=extract_help_from_doc(clean, "dry_run"),
    )
    parser_clean.add_argument(
        "--overwrite",
        action="store_true",
        required=False,
        help=extract_help_from_doc(clean, "overwrite"),
    )
    parser_clean.add_argument(
        "--threads",
        type=int,
        required=False,
        help=extract_help_from_doc(clean, "threads"),
    )
    parser_clean.add_argument(
        "--kallisto",
        required=False,
        help=extract_help_from_doc(clean, "kallisto"),
    )
    parser_clean.add_argument(
        "--bustools",
        required=False,
        help=extract_help_from_doc(clean, "bustools"),
    )
    parser_clean.add_argument(
        "-q",
        "--quiet",
        default=True,
        action="store_false",
        required=False,
        help="Do not print progress information.",
    )

    # NEW PARSER
    summarize_desc = "Analyze the mutation count matrix results."
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
        default=10,
        help=extract_help_from_doc(summarize, "top_values"),
    )
    parser_summarize.add_argument(
        "-x",
        "--technology",
        required=False,
        help=extract_help_from_doc(summarize, "technology"),
    )
    parser_summarize.add_argument(
        "-o",
        "--out",
        required=False,
        default=".",
        help=extract_help_from_doc(summarize, "out"),
    )
    parser_summarize.add_argument(
        "--dry_run",
        required=False,
        action="store_true",
        help=extract_help_from_doc(summarize, "dry_run"),
    )
    parser_summarize.add_argument(
        "--overwrite",
        required=False,
        action="store_true",
        help=extract_help_from_doc(summarize, "overwrite"),
    )
    parser_summarize.add_argument(
        "-q",
        "--quiet",
        default=True,
        action="store_false",
        required=False,
        help="Do not print progress information.",
    )
    # kwargs
    parser_summarize.add_argument(
        "--stats_file",
        required=False,
        help=extract_help_from_doc(summarize, "stats_file"),
    )
    parser_summarize.add_argument(
        "--specific_stats_folder",
        required=False,
        help=extract_help_from_doc(summarize, "specific_stats_folder"),
    )
    parser_summarize.add_argument(
        "--plots_folder",
        required=False,
        help=extract_help_from_doc(summarize, "plots_folder"),
    )

    # NEW PARSER
    ref_desc = "Create a reference index and t2g file for variant screening with varseek count. Wraps around varseek build, varseek info, varseek filter, and kb ref."
    parser_ref = parent_subparsers.add_parser(
        "ref",
        parents=[parent],
        description=ref_desc,
        help=ref_desc,
        add_help=True,
        formatter_class=CustomHelpFormatter,
    )
    parser_ref.add_argument(
        "-s",
        "--sequences",
        type=str,
        nargs="+",
        required=True,
        help=extract_help_from_doc(ref, "sequences"),
    )
    parser_ref.add_argument(
        "-m",
        "--mutations",
        type=strpath_or_str_or_list_or_df,
        nargs="+",
        required=True,
        help=extract_help_from_doc(ref, "mutations"),
    )
    parser_ref.add_argument(
        "-w",
        "--w",
        default=54,
        type=int,
        required=False,
        help=extract_help_from_doc(ref, "w"),
    )
    parser_ref.add_argument(
        "-k",
        "--k",
        default=59,
        type=int,
        required=False,
        help=extract_help_from_doc(ref, "k"),
    )
    parser_ref.add_argument(
        "-f",
        "--filters",
        type=strpath_or_list_like_of_strings,
        nargs="*",
        required=False,
        default=(
            "dlist_substring:equal=none",  # filter out mutations which are a substring of the reference genome
            "pseudoaligned_to_human_reference_despite_not_truly_aligning:is_not_true",  # filter out mutations which pseudoaligned to human genome despite not truly aligning
            "dlist:equal=none",  # *** erase eventually when I want to d-list  # filter out mutations which are capable of being d-listed (given that I filter out the substrings above)
            "number_of_kmers_with_overlap_to_other_mcrs_items_in_mcrs_reference:less_than=999999",  # filter out mutations which overlap with other MCRSs in the reference
            "number_of_mcrs_items_with_overlapping_kmers_in_mcrs_reference:less_than=999999",  # filter out mutations which overlap with other MCRSs in the reference
            "longest_homopolymer_length:bottom_percent=99.99",  # filters out MCRSs with repeating single nucleotide - 99.99 keeps the bottom 99.99% (fraction 0.9999) ie filters out the top 0.01%
            "triplet_complexity:top_percent=99.9",  # filters out MCRSs with repeating triplets - 99.9 keeps the top 99.9% (fraction 0.999) ie filters out the bottom 0.1%
        ),
        help=extract_help_from_doc(ref, "filters"),
    )
    parser_ref.add_argument(
        "--mode",
        default=None,
        required=False,
        help=extract_help_from_doc(ref, "mode"),
    )
    parser_ref.add_argument(
        "--dlist",
        default=None,
        required=False,
        help=extract_help_from_doc(ref, "dlist"),
    )
    parser_ref.add_argument(
        "-c",
        "--config",
        required=False,
        help=extract_help_from_doc(ref, "config"),
    )
    parser_ref.add_argument(
        "-o",
        "--out",
        type=str,
        required=False,
        default=".",
        help=extract_help_from_doc(ref, "out"),
    )
    parser_ref.add_argument(
        "--reference_out_dir",
        type=str,
        required=False,
        default=None,
        help=extract_help_from_doc(ref, "reference_out_dir"),
    )
    parser_ref.add_argument(
        "-i",
        "--index_out",
        required=False,
        help=extract_help_from_doc(ref, "index_out"),
    )
    parser_ref.add_argument(
        "-g",
        "--t2g_out",
        required=False,
        help=extract_help_from_doc(ref, "t2g_out"),
    )
    parser_ref.add_argument(
        "-d",
        "--download",
        action="store_true",
        required=False,
        help=extract_help_from_doc(ref, "download"),
    )
    parser_ref.add_argument(
        "--dry_run",
        action="store_true",
        required=False,
        help=extract_help_from_doc(ref, "dry_run"),
    )
    parser_ref.add_argument(
        "--list_downloadable_references",
        action="store_true",
        required=False,
        help=extract_help_from_doc(ref, "list_downloadable_references"),
    )
    parser_ref.add_argument(
        "-dmic",
        "--disable_minimum_info_columns",
        action="store_false",
        help=extract_help_from_doc(ref, "minimum_info_columns", disable=True),
    )
    parser_ref.add_argument(
        "--overwrite",
        action="store_true",
        required=False,
        help=extract_help_from_doc(ref, "overwrite"),
    )
    parser_ref.add_argument(
        "-t",
        "--threads",
        type=int,
        required=False,
        default=2,
        help=extract_help_from_doc(ref, "threads"),
    )
    parser_ref.add_argument(
        "-q",
        "--quiet",
        action="store_false",
        required=False,
        help="Do not print progress information.",
    )

    # NEW PARSER
    count_desc = "Perform variant screening on sequencing data. Wraps around varseek fastqpp, kb count, varseek clean, and kb summarize.."
    parser_count = parent_subparsers.add_parser(
        "count",
        parents=[parent],
        description=count_desc,
        help=count_desc,
        add_help=True,
        formatter_class=CustomHelpFormatter,
    )
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
        required=False,
        type=int,
        default=59,
        help=extract_help_from_doc(count, "k"),
    )
    parser_count.add_argument(
        "--strand",
        required=False,
        type=str,
        default="unstranded",
        help=extract_help_from_doc(count, "strand"),
    )
    parser_count.add_argument(
        "--mm",
        required=False,
        action="store_true",
        help=extract_help_from_doc(count, "mm"),
    )
    parser_count.add_argument(
        "--union",
        required=False,
        action="store_true",
        help=extract_help_from_doc(count, "union"),
    )
    parser_count.add_argument(
        "--parity",
        required=False,
        type=str,
        choices=["single", "paired"],
        default="single",
        help=extract_help_from_doc(count, "parity"),
    )
    parser_count.add_argument(
        "--species",
        required=False,
        help=extract_help_from_doc(count, "species"),
    )
    parser_count.add_argument(
        "--config",
        required=False,
        help=extract_help_from_doc(count, "config"),
    )
    parser_count.add_argument(
        "--reference_genome_index",
        required=False,
        help=extract_help_from_doc(count, "reference_genome_index"),
    )
    parser_count.add_argument(
        "--reference_genome_t2g",
        required=False,
        help=extract_help_from_doc(count, "reference_genome_t2g"),
    )
    parser_count.add_argument(
        "--adata_reference_genome",
        required=False,
        help=extract_help_from_doc(count, "adata_reference_genome"),
    )
    parser_count.add_argument(
        "-o",
        "--out",
        required=False,
        type=str,
        default=".",
        help=extract_help_from_doc(count, "out"),
    )
    parser_count.add_argument(
        "--kb_count_vcrs_out_dir",
        required=False,
        help=extract_help_from_doc(count, "kb_count_vcrs_out_dir"),
    )
    parser_count.add_argument(
        "--kb_count_reference_genome_out_dir",
        required=False,
        help=extract_help_from_doc(count, "kb_count_reference_genome_out_dir"),
    )
    parser_count.add_argument(
        "--vk_summarize_out_dir",
        required=False,
        help=extract_help_from_doc(count, "vk_summarize_out_dir"),
    )
    parser_count.add_argument(
        "--disable_fastqpp",
        action="store_true",
        required=False,
        help=extract_help_from_doc(count, "disable_fastqpp"), # not disable=True because the acual python argument is named disable_fastqpp
    )
    parser_count.add_argument(
        "--disable_clean",
        action="store_true",
        required=False,
        help=extract_help_from_doc(count, "disable_clean"),
    )
    parser_count.add_argument(
        "--disable_summarize",
        action="store_true",
        required=False,
        help=extract_help_from_doc(count, "disable_summarize"),
    )
    parser_count.add_argument(
        "--dry_run",
        action="store_true",
        required=False,
        help=extract_help_from_doc(count, "dry_run"),
    )
    parser_count.add_argument(
        "--overwrite",
        action="store_true",
        required=False,
        help=extract_help_from_doc(count, "overwrite"),
    )
    parser_count.add_argument(
        "--sort_fastqs",
        action="store_true",
        required=False,
        help=extract_help_from_doc(count, "sort_fastqs"),
    )
    parser_count.add_argument(
        "-t",
        "--threads",
        type=int,
        required=False,
        default=2,
        help=extract_help_from_doc(count, "threads"),
    )
    parser_count.add_argument(
        "-q",
        "--quiet",
        action="store_false",
        required=False,
        help="Do not print progress information.",
    )
    # kwargs
    parser_count.add_argument(
        "--use_num",
        action="store_true",
        required=False,
        help=extract_help_from_doc(count, "use_num"),
    )

    #* Define return values
    args, unknown_args = parent_parser.parse_known_args()

    kwargs = {unknown_args[i].lstrip("--"): unknown_args[i + 1] for i in range(0, len(unknown_args), 2)}

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

    # Load params from config if provided
    if args.config:
        # Assert that, if `--config` is passed, that no other arguments are passed
        assert_only_config(args, parent_parser)

        params = load_params(file=args.config)

        # Update args with params if not already set
        for key, value in params.items():
            setattr(args, key, value)

    #* build return
    if args.command == "build":
        if isinstance(args.sequences, list) and len(args.sequences) == 1:
            seqs = args.sequences[0]
        else:
            seqs = args.sequences

        if isinstance(args.mutations, list) and len(args.mutations) == 1:
            muts = args.mutations[0]
        else:
            muts = args.mutations

        # Run build_desc function (automatically saves output)
        build_results = build(
            sequences=seqs,
            mutations=muts,
            w=args.w,
            k=args.k,
            max_ambiguous=args.max_ambiguous,
            mut_column=args.mut_column,
            seq_id_column=args.seq_id_column,
            mut_id_column=args.mut_id_column,
            gtf=args.gtf,
            gtf_transcript_id_column=args.gtf_transcript_id_column,
            transcript_boundaries=args.transcript_boundaries,
            identify_all_spliced_from_genome=args.identify_all_spliced_from_genome,
            out=args.out,
            reference_out_dir=args.reference_out_dir,
            mcrs_fasta_out=args.mcrs_fasta_out,
            mutations_updated_csv_out=args.mutations_updated_csv_out,
            id_to_header_csv_out=args.id_to_header_csv_out,
            mcrs_t2g_out=args.mcrs_t2g_out,
            wt_mcrs_fasta_out=args.wt_mcrs_fasta_out,
            wt_mcrs_t2g_out=args.wt_mcrs_t2g_out,
            removed_variants_text_out=args.removed_variants_text_out,
            filtering_report_text_out=args.filtering_report_text_out,
            return_mutation_output=args.return_mutation_output,
            save_mutations_updated_csv=args.save_mutations_updated_csv,
            save_wt_mcrs_fasta_and_t2g=args.save_wt_mcrs_fasta_and_t2g,
            store_full_sequences=args.store_full_sequences,
            translate=args.translate,
            translate_start=args.translate_start,
            translate_end=args.translate_end,
            dry_run=args.dry_run,
            list_supported_databases=args.list_supported_databases,
            overwrite=args.overwrite,
            verbose=args.quiet,
            insertion_size_limit=args.insertion_size_limit,
            min_seq_len=args.min_seq_len,
            optimize_flanking_regions=args.disable_optimize_flanking_regions,
            remove_seqs_with_wt_kmers=args.disable_remove_seqs_with_wt_kmers,
            required_insertion_overlap_length=args.required_insertion_overlap_length,
            merge_identical=args.disable_merge_identical,
            vcrs_strandedness=args.vcrs_strandedness,
            use_IDs=args.disable_use_IDs,
            cosmic_release=args.cosmic_release,
            cosmic_grch=args.cosmic_grch,
            cosmic_email=args.cosmic_email,
            cosmic_password=args.cosmic_password,
            save_files=args.disable_save_files,
            **kwargs,
        )

        # Print list of mutated sequences if any are returned (this should only happen when out=None)
        if build_results:
            for mut_seq in build_results:
                print(mut_seq)

    #* info return
    if args.command == "info":
        info_results = info(
            input_dir=args.input_dir,
            columns_to_include=args.columns_to_include,
            k=args.k,
            max_ambiguous_mcrs=args.max_ambiguous_mcrs,
            max_ambiguous_reference=args.max_ambiguous_reference,
            mcrs_fasta=args.mcrs_fasta,
            mutations_updated_csv=args.mutations_updated_csv,
            id_to_header_csv=args.id_to_header_csv,
            gtf=args.gtf,
            dlist_reference_source=args.dlist_reference_source,
            dlist_reference_genome_fasta=args.dlist_reference_genome_fasta,
            dlist_reference_cdna_fasta=args.dlist_reference_cdna_fasta,
            dlist_reference_gtf=args.dlist_reference_gtf,
            dlist_reference_ensembl_release=args.dlist_reference_ensembl_release,
            mcrs_id_column=args.mcrs_id_column,
            mcrs_sequence_column=args.mcrs_sequence_column,
            mcrs_source_column=args.mcrs_source_column,
            mut_column=args.mut_column,
            seq_id_column=args.seq_id_column,
            mutation_cdna_column=args.mutation_cdna_column,
            seq_id_cdna_column=args.seq_id_cdna_column,
            mutation_genome_column=args.mutation_genome_column,
            seq_id_genome_column=args.seq_id_genome_column,
            out=args.out,
            reference_out_dir=args.reference_out_dir,
            mutations_updated_vk_info_csv_out=args.mutations_updated_vk_info_csv_out,
            mutations_updated_exploded_vk_info_csv_out=args.mutations_updated_exploded_vk_info_csv_out,
            dlist_genome_fasta_out=args.dlist_genome_fasta_out,
            dlist_cdna_fasta_out=args.dlist_cdna_fasta_out,
            dlist_combined_fasta_out=args.dlist_combined_fasta_out,
            save_mutations_updated_exploded_vk_info_csv=args.save_mutations_updated_exploded_vk_info_csv,
            make_pyfastx_summary_file=args.make_pyfastx_summary_file,
            make_kat_histogram=args.make_kat_histogram,
            dry_run=args.dry_run,
            list_columns=args.list_columns,
            list_dlist_references=args.list_dlist_references,
            overwrite=args.overwrite,
            threads=args.threads,
            verbose=args.quiet,
            bowtie2_path=args.bowtie2_path,
            vcrs_strandedness=args.vcrs_strandedness,
            near_splice_junction_threshold=args.near_splice_junction_threshold,
            reference_cdna_fasta=args.reference_cdna_fasta,
            reference_genome_fasta=args.reference_genome_fasta,
            mutations_csv=args.mutations_csv,
            **kwargs,
        )

        # * optionally do something with info_results (e.g., save, or print to console)

    #* filter return
    if args.command == "filter":
        filter_rules = prepare_filters_list(args.filters)

        filter_results = filter(
            input_dir=args.input_dir,
            filters=filter_rules,
            mutations_updated_vk_info_csv=args.mutations_updated_vk_info_csv,
            mutations_updated_exploded_vk_info_csv=args.mutations_updated_exploded_vk_info_csv,
            id_to_header_csv=args.id_to_header_csv,
            dlist_fasta=args.dlist_fasta,
            out=args.out,
            mutations_updated_filtered_csv_out=args.mutations_updated_filtered_csv_out,
            mutations_updated_exploded_filtered_csv_out=args.mutations_updated_exploded_filtered_csv_out,
            id_to_header_filtered_csv_out=args.id_to_header_filtered_csv_out,
            dlist_filtered_fasta_out=args.dlist_filtered_fasta_out,
            mcrs_filtered_fasta_out=args.mcrs_filtered_fasta_out,
            mcrs_t2g_filtered_out=args.mcrs_t2g_filtered_out,
            wt_mcrs_filtered_fasta_out=args.wt_mcrs_filtered_fasta_out,
            wt_mcrs_t2g_filtered_out=args.wt_mcrs_t2g_filtered_out,
            save_wt_mcrs_fasta_and_t2g=args.save_wt_mcrs_fasta_and_t2g,
            save_mutations_updated_filtered_csvs=args.save_mutations_updated_filtered_csvs,
            return_mutations_updated_filtered_csv_df=args.return_mutations_updated_filtered_csv_df,
            dry_run=args.dry_run,
            list_filter_rules=args.list_filter_rules,
            overwrite=args.overwrite,
            verbose=args.quiet,
            filter_all_dlists=args.filter_all_dlists,
            dlist_genome_fasta=args.dlist_genome_fasta,
            dlist_cdna_fasta=args.dlist_cdna_fasta,
            dlist_genome_filtered_fasta_out=args.dlist_genome_filtered_fasta_out,
            dlist_cdna_filtered_fasta_out=args.dlist_cdna_filtered_fasta_out,
            save_mcrs_filtered_fasta_and_t2g=args.disable_save_mcrs_filtered_fasta_and_t2g,
            use_IDs=args.disable_use_IDs,
            **kwargs,
        )

        # * optionally do something with filter_results (e.g., save, or print to console)

    #* sim return
    if args.command == "sim":
        filter_rules = prepare_filters_list(args.filters)

        simulated_df_dict = sim(
            mutations=args.mutations,
            number_of_mutations_to_sample=args.number_of_mutations_to_sample,
            number_of_reads_per_mutation_m=args.number_of_reads_per_mutation_m,
            number_of_reads_per_mutation_w=args.number_of_reads_per_mutation_w,
            sample_m_and_w_from_same_locations=args.sample_m_and_w_from_same_locations,
            with_replacement=args.with_replacement,
            strand=args.strand,
            read_length=args.read_length,
            filters=filter_rules,
            add_noise_sequencing_error=args.add_noise_sequencing_error,
            add_noise_base_quality=args.add_noise_base_quality,
            error_rate=args.error_rate,
            error_distribution=args.error_distribution,
            max_errors=args.max_errors,
            mutant_sequence_read_parent_column=args.mutant_sequence_read_parent_column,
            wt_sequence_read_parent_column=args.wt_sequence_read_parent_column,
            mutant_sequence_read_parent_rc_column=args.mutant_sequence_read_parent_rc_column,
            wt_sequence_read_parent_rc_column=args.wt_sequence_read_parent_rc_column,
            reads_fastq_parent=args.reads_fastq_parent,
            reads_csv_parent=args.reads_csv_parent,
            out=args.out,
            reads_fastq_out=args.reads_fastq_out,
            mutations_updated_csv_out=args.mutations_updated_csv_out,
            reads_csv_out=args.reads_csv_out,
            save_mutations_updated_csv=args.disable_save_mutations_updated_csv,
            save_read_csv=args.disable_save_read_csv,
            vk_build_out_dir=args.vk_build_out_dir,
            sequences=args.sequences,
            seq_id_column=args.seq_id_column,
            mut_column=args.mut_column,
            k=args.k,
            w=args.w,
            sequences_cdna=args.sequences_cdna,
            seq_id_column_cdna=args.seq_id_column_cdna,
            mut_column_cdna=args.mut_column_cdna,
            sequences_genome=args.sequences_genome,
            seq_id_column_genome=args.seq_id_column_genome,
            mut_column_genome=args.mut_column_genome,
            seed=args.seed,
            gzip_output_fastq=args.gzip_output_fastq,
            dry_run=args.dry_run,
            verbose=args.quiet,
            **kwargs

        )

        # * optionally do something with simulated_df_dict (e.g., save, or print to console)

    #* fastqpp return
    if args.command == "fastqpp":
        fastqpp_results = filter(
            fastqs=args.fastqs,
            technology=args.technology,
            multiplexed=args.multiplexed,
            parity=args.parity,
            quality_control_fastqs=args.quality_control_fastqs,
            cut_mean_quality=args.cut_mean_quality,
            cut_window_size=args.cut_window_size,
            qualified_quality_phred=args.qualified_quality_phred,
            unqualified_percent_limit=args.unqualified_percent_limit,
            max_ambiguous=args.max_ambiguous,
            min_read_len=args.min_read_len,
            fastqc_and_multiqc=args.fastqc_and_multiqc,
            replace_low_quality_bases_with_N=args.replace_low_quality_bases_with_N,
            min_base_quality=args.min_base_quality,
            split_reads_by_Ns=args.split_reads_by_Ns,
            concatenate_paired_fastqs=args.concatenate_paired_fastqs,
            out=args.out,
            delete_intermediate_files=args.delete_intermediate_files,
            dry_run=args.dry_run,
            overwrite=args.overwrite,
            sort_fastqs=args.disable_sort_fastqs,
            threads=args.threads,
            verbose=args.quiet,
            fastp_path=args.fastp_path,
            seqtk_path=args.seqtk_path,
            fastqc_path=args.fastqc_path,
            multiqc_path=args.multiqc_path,
            quality_control_fastqs_out_suffix=args.quality_control_fastqs_out_suffix,
            replace_low_quality_bases_with_N_out_suffix=args.replace_low_quality_bases_with_N_out_suffix,
            split_by_N_out_suffix=args.split_by_N_out_suffix,
            concatenate_paired_fastqs_out_suffix=args.concatenate_paired_fastqs_out_suffix,
            **kwargs,
        )

        # * optionally do something with fastqpp_results (e.g., save, or print to console)

    #* clean return
    if args.command == "clean":
        clean_results = clean(
            adata_vcrs=args.adata_vcrs,
            technology=args.technology,
            min_counts=args.min_counts,
            use_binary_matrix=args.use_binary_matrix,
            drop_empty_columns=args.drop_empty_columns,
            apply_single_end_mode_on_paired_end_data_correction=args.apply_single_end_mode_on_paired_end_data_correction,
            split_reads_by_Ns=args.split_reads_by_Ns,
            apply_dlist_correction=args.apply_dlist_correction,
            qc_against_gene_matrix=args.qc_against_gene_matrix,
            filter_cells_by_min_counts=args.filter_cells_by_min_counts,
            filter_cells_by_min_genes=args.filter_cells_by_min_genes,
            filter_genes_by_min_cells=args.filter_genes_by_min_cells,
            filter_cells_by_max_mt_content=args.filter_cells_by_max_mt_content,
            doublet_detection=args.doublet_detection,
            remove_doublets=args.remove_doublets,
            cpm_normalization=args.cpm_normalization,
            sum_rows=args.sum_rows,
            mcrs_id_set_to_exclusively_keep=args.mcrs_id_set_to_exclusively_keep,
            mcrs_id_set_to_exclude=args.mcrs_id_set_to_exclude,
            transcript_set_to_exclusively_keep=args.transcript_set_to_exclusively_keep,
            transcript_set_to_exclude=args.transcript_set_to_exclude,
            gene_set_to_exclusively_keep=args.gene_set_to_exclusively_keep,
            gene_set_to_exclude=args.gene_set_to_exclude,
            k=args.k,
            mm=args.mm,
            union=args.union,
            parity=args.parity,
            multiplexed=args.multiplexed,
            sort_fastqs=args.sort_fastqs,
            fastqs=args.fastqs,
            vk_ref_dir=args.vk_ref_dir,
            vcrs_index=args.vcrs_index,
            vcrs_t2g=args.vcrs_t2g,
            mcrs_fasta=args.mcrs_fasta,
            id_to_header_csv=args.id_to_header_csv,
            dlist_fasta=args.dlist_fasta,
            mutations_updated_csv=args.mutations_updated_csv,
            mcrs_id_column=args.mcrs_id_column,
            mutations_updated_csv_columns=args.mutations_updated_csv_columns,
            adata_reference_genome=args.adata_reference_genome,
            kb_count_vcrs_dir=args.kb_count_vcrs_dir,
            kb_count_reference_genome_dir=args.kb_count_reference_genome_dir,
            out=args.out,
            adata_vcrs_clean_out=args.adata_vcrs_clean_out,
            adata_reference_genome_clean_out=args.adata_reference_genome_clean_out,
            vcf_out=args.vcf_out,
            save_vcf=args.save_vcf,
            dry_run=args.dry_run,
            overwrite=args.overwrite,
            threads=args.threads,
            kallisto=args.kallisto,
            bustools=args.bustools,
            verbose=args.quiet,
            **kwargs,
        )

        # * optionally do something with clean_results (e.g., save, or print to console)

    #* summarize return
    if args.command == "summarize":
        summarize_results = summarize(
            top_values=args.top_values,
            technology=args.technology,
            out=args.out,
            dry_run=args.dry_run,
            overwrite=args.overwrite,
            verbose=args.quiet,
            stats_file=args.stats_file,
            specific_stats_folder=args.specific_stats_folder,
            plots_folder=args.plots_folder,
            **kwargs,
        )

        # * optionally do something with summarize_results (e.g., save, or print to console)

    #* ref return
    if args.command == "ref":
        ref_results = ref(
            sequences=args.sequences,
            mutations=args.mutations,
            filters=args.filters,
            mode=args.mode,
            dlist=args.dlist,
            out=args.out,
            reference_out_dir=args.reference_out_dir,
            index_out=args.index_out,
            t2g_out=args.t2g_out,
            download=args.download,
            dry_run=args.dry_run,
            list_downloadable_references=args.list_downloadable_references,
            disable_minimum_info_columns=args.disable_minimum_info_columns,
            overwrite=args.overwrite,
            verbose=args.quiet,
            **kwargs,
        )

        # * optionally do something with ref_results (e.g., save, or print to console)

    #* count return
    if args.command == "count":
        count_results = count(
            fastqs=args.fastqs,
            index=args.index,
            t2g=args.t2g,
            technology=args.technology,
            k=args.k,
            strand=args.strand,
            mm=args.mm,
            union=args.union,
            parity=args.parity,
            species=args.species,
            config=args.config,
            reference_genome_index=args.reference_genome_index,
            reference_genome_t2g=args.reference_genome_t2g,
            adata_reference_genome=args.adata_reference_genome,
            out=args.out,
            kb_count_vcrs_out_dir=args.kb_count_vcrs_out_dir,
            kb_count_reference_genome_out_dir=args.kb_count_reference_genome_out_dir,
            vk_summarize_out_dir=args.vk_summarize_out_dir,
            disable_fastqpp=args.disable_fastqpp,
            disable_clean=args.disable_clean,
            disable_summarize=args.disable_summarize,
            dry_run=args.dry_run,
            overwrite=args.overwrite,
            sort_fastqs=args.sort_fastqs,
            threads=args.threads,
            verbose=args.quiet,
            use_num=args.use_num,
            **kwargs,
        )

        # * optionally do something with count_results (e.g., save, or print to console)
