import os
import inspect
import re
import math
import argparse
import sys
import pandas as pd
from datetime import datetime

from .__init__ import __version__
from .varseek_build import build
from .varseek_summarize import summarize
from .varseek_filter import filter
from .varseek_clean import clean
from .varseek_info import info
from .varseek_sim import sim
from .utils import set_up_logger, prepare_filters_json, prepare_filters_list

# Get current date and time for alphafold default foldername
dt_string = datetime.now().strftime("%Y_%m_%d-%H_%M")

logger = set_up_logger(logging_level_name=None, save_logs=False, log_dir="logs")


def process_filters(filters):
    if len(filters) == 1 and os.path.isfile(filters[0]) and filters[0].endswith(".json"):
        return prepare_filters_json(filters[0])
    else:
        return prepare_filters_list(filters)


def extract_help_from_doc(module, arg_name):
    """
    Extracts the help message for a given argument from the module's docstring, handling multi-line descriptions.
    Requires a docstring line of the following format:
    - ARGUMENT1     (TYPE1 or TYPE2 or ...) DESCRIPTION
    OPTIONAL EXTRA DESCRIPTION LINE 1
    OPTIONAL EXTRA DESCRIPTION LINE 2
    ...
    - ARGUMENT2    (TYPE1 or TYPE2 or ...) DESCRIPTION
    ...
    """
    docstring = inspect.getdoc(module)
    help_message = []

    # Regular expression to match the argument line with flexible type matching
    arg_pattern = rf"-\s*{arg_name}\s*\((.*?)\)\s*(.*)"

    # Regular expression to match the start of a new argument or 'Additional input arguments:'
    new_arg_pattern = r"-\s*[a-zA-Z_]\w*\s*\(.*?\)|Additional input arguments:"

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
        return "\n".join(help_message).strip()
    else:
        return "Help message not found in docstring."
        # raise ValueError(f"Argument '{arg_name}' not found in the docstring of the module '{module}'.")


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


def strpath_or_df(value):
    # List of valid file extensions
    valid_file_extensions = [".csv", ".tsv", ".xls", ".xlsx", ".parquet", ".h5"]

    # Check if the input is a pandas DataFrame
    if isinstance(value, pd.DataFrame):
        return value

    # Check if the input is a string (potential file path) and has a valid extension
    if isinstance(value, str) and os.path.isfile(value):
        if any(value.endswith(extension) for extension in valid_file_extensions):
            return value
        else:
            raise ValueError(f"File has an unsupported extension: {value}")

    # If neither condition is satisfied, raise an error
    raise ValueError("Input must be either a valid file path or a pandas DataFrame.")


def strpath_or_strnonpath_or_df(value):
    # List of valid file extensions
    valid_file_extensions = [".csv", ".tsv", ".xls", ".xlsx", ".parquet", ".h5"]

    # Check if the input is a pandas DataFrame
    if isinstance(value, pd.DataFrame):
        return value

    if isinstance(value, str) and not os.path.isfile(value):
        return value

    # Check if the input is a string (potential file path) and has a valid extension
    if isinstance(value, str) and os.path.isfile(value):
        if any(value.endswith(extension) for extension in valid_file_extensions):
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


def strpath_or_str_or_list_or_df(value):
    # List of valid file extensions
    valid_file_extensions = [".csv", ".tsv", ".xls", ".xlsx", ".parquet", ".h5"]

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
        if any(value.endswith(extension) for extension in valid_file_extensions):
            return value
        else:
            raise ValueError(f"File has an unsupported extension: {value}")

    # If none of the conditions match, raise an error
    raise ValueError("Input must be a non-path string, a valid file path, a list, or a pandas DataFrame.")


def main():
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

    # build parser arguments
    build_desc = "Build a mutation-containing reference sequence (MCRS) file."
    parser_build = parent_subparsers.add_parser(
        "build",
        parents=[parent],
        description=build_desc,
        help=build_desc,
        add_help=True,
        formatter_class=CustomHelpFormatter,
    )
    parser_build.add_argument(
        "sequences",
        type=str,
        nargs="+",
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
        type=str,
        required=False,
        help=extract_help_from_doc(build, "mut_id_column"),
    )
    parser_build.add_argument(
        "-gtf",
        "--gtf",
        default=None,
        type=str,
        required=False,
        help=extract_help_from_doc(build, "gtf"),
    )
    parser_build.add_argument(
        "-gtic",
        "--gtf_transcript_id_column",
        default=None,
        type=str,
        required=False,
        help=extract_help_from_doc(build, "gtf_transcript_id_column"),
    )
    parser_build.add_argument(
        "-w",
        "--w",
        default=30,
        type=int,
        required=False,
        help=extract_help_from_doc(build, "w"),
    )
    parser_build.add_argument(
        "-k",
        "--k",
        default=None,
        type=int,
        required=False,
        help=extract_help_from_doc(build, "k"),
    )
    parser_build.add_argument(
        "--insertion_size_limit",
        default=None,
        type=int,
        required=False,
        help=extract_help_from_doc(build, "insertion_size_limit"),
    )
    parser_build.add_argument(
        "-msl",
        "--min_seq_len",
        default=None,
        type=int,
        required=False,
        help=extract_help_from_doc(build, "min_seq_len"),
    )
    parser_build.add_argument(
        "-ma",
        "--max_ambiguous",
        default=None,
        type=int,
        required=False,
        help=extract_help_from_doc(build, "max_ambiguous"),
    )
    parser_build.add_argument(
        "-ofr",
        "--optimize_flanking_regions",
        default=False,
        action="store_true",
        required=False,
        help=extract_help_from_doc(build, "optimize_flanking_regions"),
    )
    parser_build.add_argument(
        "-rswk",
        "--remove_seqs_with_wt_kmers",
        default=False,
        action="store_true",
        required=False,
        help=extract_help_from_doc(build, "remove_seqs_with_wt_kmers"),
    )
    parser_build.add_argument(
        "-mi",
        "--merge_identical",
        default=False,
        action="store_true",
        required=False,
        help=extract_help_from_doc(build, "merge_identical"),
    )
    parser_build.add_argument(
        "-mirc",
        "--merge_identical_rc",
        default=False,
        action="store_true",
        required=False,
        help=extract_help_from_doc(build, "merge_identical_rc"),
    )
    parser_build.add_argument(
        "-koh",
        "--keep_original_headers",
        default=False,
        action="store_true",
        required=False,
        help=extract_help_from_doc(build, "keep_original_headers"),
    )
    parser_build.add_argument(
        "-udf",
        "--update_df",
        default=False,
        action="store_true",
        required=False,
        help=extract_help_from_doc(build, "update_df"),
    )
    parser_build.add_argument(
        "--translate",
        default=None,
        action="store_true",
        required=False,
        help="Translate the mutated sequences to amino acids. Only valid when used with `--update_df`.",
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
        "--translate_end",
        default=None,
        type=int_or_str,
        required=False,
        help=extract_help_from_doc(build, "translate_end"),
    )
    parser_build.add_argument(
        "-sfs",
        "--store_full_sequences",
        default=False,
        action="store_true",
        required=False,
        help=extract_help_from_doc(build, "store_full_sequences"),
    )
    parser_build.add_argument(
        "-ro",
        "--reference_out",
        default=None,
        type=str,
        required=False,
        help=extract_help_from_doc(build, "reference_out"),
    )
    parser_build.add_argument(
        "-o",
        "--out",
        default=None,
        type=str,
        required=False,
        help=extract_help_from_doc(build, "out"),
    )
    parser_build.add_argument(
        "-q",
        "--quiet",
        default=True,
        action="store_false",
        required=False,
        help="Do not print progress information.",
    )

    # NEW PARSER
    info_desc = "Describe the MCRS reference in a dataframe."
    parser_info = parent_subparsers.add_parser(
        "info",
        parents=[parent],
        description=info_desc,
        help=info_desc,
        add_help=True,
        formatter_class=CustomHelpFormatter,
    )

    parser_info.add_argument(
        "-m",
        "--mutations",
        type=strpath_or_strnonpath_or_df,
        required=True,
        help=extract_help_from_doc(info, "mutations"),
    )
    parser_info.add_argument(
        "-u",
        "--updated_df",
        type=strpath_or_df,
        required=True,
        help=extract_help_from_doc(info, "updated_df"),
    )
    parser_info.add_argument(
        "--id_to_header_csv",
        type=str,
        required=False,
        help=extract_help_from_doc(info, "id_to_header_csv"),
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
        "--seqid_cdna_column",
        type=str,
        required=False,
        default="seq_ID",
        help=extract_help_from_doc(info, "seqid_cdna_column"),
    )
    parser_info.add_argument(
        "--seqid_genome_column",
        type=str,
        required=False,
        default="chromosome",
        help=extract_help_from_doc(info, "seqid_genome_column"),
    )
    parser_info.add_argument(
        "--mutation_cdna_column",
        type=str,
        required=False,
        default="chromosome",
        help=extract_help_from_doc(info, "mutation_cdna_column"),
    )
    parser_info.add_argument(
        "--mutation_genome_column",
        type=str,
        required=False,
        default="chromosome",
        help=extract_help_from_doc(info, "mutation_genome_column"),
    )
    parser_info.add_argument("--gtf", type=str, required=False, help=extract_help_from_doc(info, "gtf"))
    parser_info.add_argument(
        "--mutation_metadata_df_out_path",
        type=str,
        required=False,
        default="out_dir_notebook/mutation_metadata_df.csv",
        help=extract_help_from_doc(info, "mutation_metadata_df_out_path"),
    )
    parser_info.add_argument(
        "--out_dir_notebook",
        type=str,
        required=False,
        default=".",
        help=extract_help_from_doc(info, "out_dir_notebook"),
    )
    parser_info.add_argument(
        "--reference_out_dir",
        type=str,
        required=False,
        default=".",
        help=extract_help_from_doc(info, "reference_out_dir"),
    )
    parser_info.add_argument(
        "--dlist_reference_source",
        type=str,
        required=False,
        default="ensembl_grch37_release93",
        help=extract_help_from_doc(info, "dlist_reference_source"),
    )
    parser_info.add_argument(
        "--ref_prefix",
        type=str,
        required=False,
        default="index",
        help=extract_help_from_doc(info, "ref_prefix"),
    )
    parser_info.add_argument(
        "-w",
        type=int,
        required=False,
        default=30,
        help=extract_help_from_doc(info, "w"),
    )
    parser_info.add_argument(
        "--remove_Ns",
        default=False,
        action="store_true",
        required=False,
        help=extract_help_from_doc(info, "remove_Ns"),
    )
    parser_info.add_argument(
        "--strandedness",
        default=False,
        action="store_true",
        required=False,
        help=extract_help_from_doc(info, "strandedness"),
    )
    parser_info.add_argument(
        "--bowtie_path",
        type=str,
        required=False,
        default=None,
        help=extract_help_from_doc(info, "bowtie_path"),
    )
    parser_info.add_argument(
        "--near_splice_junction_threshold",
        type=int,
        required=False,
        default=10,
        help=extract_help_from_doc(info, "near_splice_junction_threshold"),
    )
    parser_info.add_argument(
        "-t",
        "--threads",
        type=int,
        required=False,
        default=2,
        help=extract_help_from_doc(info, "threads"),
    )
    parser_info.add_argument(
        "--reference_cdna_fasta",
        type=str,
        required=False,
        default=None,
        help=extract_help_from_doc(info, "reference_cdna_fasta"),
    )
    parser_info.add_argument(
        "--reference_genome_fasta",
        type=str,
        required=False,
        default=None,
        help=extract_help_from_doc(info, "reference_genome_fasta"),
    )
    parser_info.add_argument(
        "--mutations_csv",
        type=str,
        required=False,
        default=None,
        help=extract_help_from_doc(info, "mutations_csv"),
    )
    parser_info.add_argument(
        "--save_exploded_df",
        default=False,
        action="store_true",
        required=False,
        help=extract_help_from_doc(info, "save_exploded_df"),
    )
    parser_info.add_argument(
        "-q",
        "--quiet",
        default=True,
        action="store_false",
        required=False,
        help="Do not print progress information.",
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
        "-m",
        "--mutation_metadata_df_path",
        default=None,
        type=str,
        required=True,
        help=extract_help_from_doc(filter, "mutation_metadata_df_path"),
    )
    parser_filter.add_argument(
        "-f",
        "--filters",
        nargs="+",  # Accept multiple sequential filters or a single JSON file
        type=str,
        required=True,
        help=extract_help_from_doc(filter, "filters"),
    )
    parser_filter.add_argument(
        "--output_mcrs_fasta",
        default=None,
        type=str,
        required=False,
        help=extract_help_from_doc(filter, "output_mcrs_fasta"),
    )
    parser_filter.add_argument(
        "--output_metadata_df",
        default=None,
        type=str,
        required=False,
        help=extract_help_from_doc(filter, "output_metadata_df"),
    )
    parser_filter.add_argument(
        "--dlist_fasta",
        default=None,
        type=str,
        required=False,
        help=extract_help_from_doc(filter, "dlist_fasta"),
    )
    parser_filter.add_argument(
        "--output_dlist_fasta",
        default=None,
        type=str,
        required=False,
        help=extract_help_from_doc(filter, "output_dlist_fasta"),
    )
    parser_filter.add_argument(
        "--create_t2g",
        default=None,
        type=str,
        required=False,
        help=extract_help_from_doc(filter, "create_t2g"),
    )
    parser_filter.add_argument(
        "--output_t2g",
        default=None,
        type=str,
        required=False,
        help=extract_help_from_doc(filter, "output_t2g"),
    )
    parser_filter.add_argument(
        "--id_to_header_csv",
        default=None,
        type=str,
        required=False,
        help=extract_help_from_doc(filter, "id_to_header_csv"),
    )
    parser_filter.add_argument(
        "--output_id_to_header_csv",
        default=None,
        type=str,
        required=False,
        help=extract_help_from_doc(filter, "output_id_to_header_csv"),
    )
    parser_filter.add_argument(
        "-q",
        "--quiet",
        default=False,
        action="store_false",
        required=False,
        help="Do not print progress information.",
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
        "--mutation_metadata_df",
        default=None,
        type=strpath_or_df,
        required=True,
        help=extract_help_from_doc(sim, "mutation_metadata_df"),
    )
    parser_sim.add_argument(
        "--fastq_output_path",
        default=None,
        type=str,
        required=False,
        help=extract_help_from_doc(sim, "fastq_output_path"),
    )
    parser_sim.add_argument(
        "--fastq_parent_path",
        default=None,
        type=str,
        required=False,
        help=extract_help_from_doc(sim, "fastq_parent_path"),
    )
    parser_sim.add_argument(
        "--read_df_parent",
        default=None,
        type=strpath_or_df,
        required=False,
        help=extract_help_from_doc(sim, "read_df_parent"),
    )
    parser_sim.add_argument(
        "--sample_type",
        default=None,
        type=str,
        required=False,
        help=extract_help_from_doc(sim, "sample_type"),
    )
    parser_sim.add_argument(
        "--number_of_mutations_to_sample",
        default=1500,
        type=int,
        required=False,
        help=extract_help_from_doc(sim, "number_of_mutations_to_sample"),
    )
    parser_sim.add_argument(
        "--strand",
        default=False,
        action="store_true",
        required=False,
        help=extract_help_from_doc(sim, "strand"),
    )
    parser_sim.add_argument(
        "--number_of_reads_per_sample",
        type=int,
        required=False,
        help=extract_help_from_doc(sim, "number_of_reads_per_sample"),
    )
    parser_sim.add_argument(
        "--number_of_reads_per_sample_m",
        type=int,
        required=False,
        help=extract_help_from_doc(sim, "number_of_reads_per_sample_m"),
    )
    parser_sim.add_argument(
        "--number_of_reads_per_sample_w",
        type=int,
        required=False,
        help=extract_help_from_doc(sim, "number_of_reads_per_sample_w"),
    )
    parser_sim.add_argument(
        "--read_length",
        default=150,
        type=int,
        required=False,
        help=extract_help_from_doc(sim, "read_length"),
    )
    parser_sim.add_argument(
        "--seed",
        default=42,
        type=int,
        required=False,
        help=extract_help_from_doc(sim, "seed"),
    )
    parser_sim.add_argument(
        "--add_noise",
        default=False,
        action="store_true",
        required=False,
        help=extract_help_from_doc(sim, "add_noise"),
    )
    parser_sim.add_argument(
        "--error_rate",
        default=0.0001,
        type=int,
        required=False,
        help=extract_help_from_doc(sim, "error_rate"),
    )
    parser_sim.add_argument(
        "--max_errors",
        default=float("inf"),
        type=is_int_or_float_or_inf,
        required=False,
        help=extract_help_from_doc(sim, "max_errors"),
    )
    parser_sim.add_argument(
        "--with_replacement",
        default=False,
        action="store_true",
        required=False,
        help=extract_help_from_doc(sim, "with_replacement"),
    )
    parser_sim.add_argument(
        "--sequences",
        default=None,
        type=str,
        required=False,
        help=extract_help_from_doc(sim, "sequences"),
    )
    parser_sim.add_argument(
        "--mutation_metadata_df_path",
        default=None,
        type=str,
        required=False,
        help=extract_help_from_doc(sim, "mutation_metadata_df_path"),
    )
    parser_sim.add_argument(
        "--reference_out_dir",
        default=None,
        type=str,
        required=False,
        help=extract_help_from_doc(sim, "reference_out_dir"),
    )
    parser_sim.add_argument(
        "--out_dir_vk_build",
        default=None,
        type=str,
        required=False,
        help=extract_help_from_doc(sim, "out_dir_vk_build"),
    )
    parser_sim.add_argument(
        "--seq_id_column",
        default=None,
        type=str,
        required=False,
        help=extract_help_from_doc(sim, "seq_id_column"),
    )
    parser_sim.add_argument(
        "--mut_column",
        default=None,
        type=str,
        required=False,
        help=extract_help_from_doc(sim, "mut_column"),
    )
    parser_sim.add_argument(
        "--gtf",
        default=None,
        type=str,
        required=False,
        help=extract_help_from_doc(sim, "gtf"),
    )
    parser_sim.add_argument(
        "--gtf_transcript_id_column",
        default=None,
        type=str,
        required=False,
        help=extract_help_from_doc(sim, "gtf_transcript_id_column"),
    )
    parser_sim.add_argument(
        "--sequences_cdna",
        default=None,
        type=str,
        required=False,
        help=extract_help_from_doc(sim, "sequences_cdna"),
    )
    parser_sim.add_argument(
        "--seq_id_column_cdna",
        default=None,
        type=str,
        required=False,
        help=extract_help_from_doc(sim, "seq_id_column_cdna"),
    )
    parser_sim.add_argument(
        "--mut_column_cdna",
        default=None,
        type=str,
        required=False,
        help=extract_help_from_doc(sim, "mut_column_cdna"),
    )
    parser_sim.add_argument(
        "--sequences_genome",
        default=None,
        type=str,
        required=False,
        help=extract_help_from_doc(sim, "sequences_genome"),
    )
    parser_sim.add_argument(
        "--seq_id_column_genome",
        default=None,
        type=str,
        required=False,
        help=extract_help_from_doc(sim, "seq_id_column_genome"),
    )
    parser_sim.add_argument(
        "--mut_column_genome",
        default=None,
        type=str,
        required=False,
        help=extract_help_from_doc(sim, "mut_column_genome"),
    )
    parser_sim.add_argument(
        "-f",
        "--filters",
        nargs="+",  # Accept multiple sequential filters or a single JSON file
        type=str,
        required=True,
        help=extract_help_from_doc(sim, "filters"),
    )
    parser_sim.add_argument(
        "-q",
        "--quiet",
        default=True,
        action="store_false",
        required=False,
        help="Do not print progress information.",
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
        "-q",
        "--quiet",
        default=True,
        action="store_false",
        required=False,
        help="Do not print progress information.",
    )

    ### Define return values
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
        "clean": parser_clean,
        "summarize": parser_summarize,
    }

    if len(sys.argv) == 2:
        if sys.argv[1] in command_to_parser:
            command_to_parser[sys.argv[1]].print_help(sys.stderr)
        else:
            parent_parser.print_help(sys.stderr)
        sys.exit(1)

    ## build return
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
            gtf=args.gtf,
            gtf_transcript_id_column=args.gtf_transcript_id_column,
            w=args.w,
            mut_column=args.mut_column,
            mut_id_column=args.mut_id_column,
            seq_id_column=args.seq_id_column,
            min_seq_len=args.min_seq_len,
            max_ambiguous=args.max_ambiguous,
            optimize_flanking_regions=args.optimize_flanking_regions,
            remove_seqs_with_wt_kmers=args.remove_seqs_with_wt_kmers,
            merge_identical=args.merge_identical,
            merge_identical_rc=args.merge_identical_rc,
            update_df=args.update_df,
            store_full_sequences=args.store_full_sequences,
            translate=args.translate,
            translate_start=args.translate_start,
            translate_end=args.translate_end,
            reference_out=args.reference_out,
            out=args.out,
            verbose=args.quiet,
            **kwargs,
        )

        # Print list of mutated sequences if any are returned (this should only happen when out=None)
        if build_results:
            for mut_seq in build_results:
                print(mut_seq)

    ## info return
    if args.command == "info":
        info_results = info(
            mutations=args.mutations,
            mcrs_fa=args.mcrs_fa,
            sequences_cdna=args.sequences_cdna,
            seq_id_column_cdna=args.seq_id_column_cdna,
            mut_column_cdna=args.mut_column_cdna,
            sequences_genome=args.sequences_genome,
            seq_id_column_genome=args.seq_id_column_genome,
            mut_column_genome=args.mut_column_genome,
            gtf=args.gtf,
            mutation_metadata_df_out_path=args.mutation_metadata_df_out_path,
            out_dir_notebook=args.out_dir_notebook,
            reference_out_dir=args.reference_out_dir,
            grch_mutations=args.grch_mutations,
            cosmic_release=args.cosmic_release,
            dlist_reference_source=args.dlist_reference_source,
            id_to_header_csv=args.id_to_header_csv,
            updated_df=args.updated_df,
            ref_prefix=args.ref_prefix,
            w=args.w,
            remove_Ns=args.remove_Ns,
            optimize_flanking_regions=args.optimize_flanking_regions,
            strandedness=args.strandedness,
            bowtie_path=args.bowtie_path,
            run_comprehensive_dlist=args.run_comprehensive_dlist,
            perform_additional_pseudoalignment=args.perform_additional_pseudoalignment,
            near_splice_junction_threshold=args.near_splice_junction_threshold,
            columns_to_include=args.columns_to_include,
            threads=args.threads,
            verbose=args.quiet,
            **kwargs,
        )

        # * optionally do something with info_results (e.g., save, or print to console)

    ## filter return
    if args.command == "filter":
        filter_rules = process_filters(args.filters)

        filter_results = filter(
            mcrs_fa=args.build_fasta,
            mutation_metadata_df_path=args.info_csv,
            output_fasta=args.output_fasta,
            kv_build_source=args.kv_build_source,
            verbose=args.quiet,
            filters=filter_rules,
            **kwargs,
        )

        # * optionally do something with filter_results (e.g., save, or print to console)

    ## sim return
    if args.command == "sim":
        filter_rules = process_filters(args.filters)

        simulated_df_dict = sim(
            mutation_metadata_df=args.mutation_metadata_df,
            fastq_output_path=args.fastq_output_path,
            fastq_parent_path=args.fastq_parent_path,
            read_df_parent=args.read_df_parent,
            sample_type=args.sample_type,
            n=args.n,
            strand=args.strand,
            number_of_reads_per_sample=args.number_of_reads_per_sample,
            read_length=args.read_length,
            seed=args.seed,
            add_noise=args.add_noise,
            with_replacement=args.with_replacement,
            filters=filter_rules,
            verbose=args.quiet,
            **kwargs,
        )

        # * optionally do something with sim_results (e.g., save, or print to console)

    ## clean return
    if args.command == "clean":
        clean_results = clean(verbose=args.quiet, **kwargs)

        # * optionally do something with clean_results (e.g., save, or print to console)

    ## summarize return
    if args.command == "summarize":
        summarize_results = summarize(verbose=args.quiet, **kwargs)

        # * optionally do something with summarize_results (e.g., save, or print to console)

    ## summarize return
    if args.command == "fastqpp":
        summarize_results = summarize(verbose=args.quiet, **kwargs)
