"""varseek filter and specific helper functions."""

import ast
import csv
import logging
import os
import re
import time
from pathlib import Path

import pandas as pd

from .utils import (
    check_file_path_is_string_with_valid_extension,
    create_identity_t2g,
    extract_documentation_file_blocks,
    fasta_summary_stats,
    filter_fasta,
    make_function_parameter_to_value_dict,
    make_mapping_dict,
    print_varseek_dry_run,
    report_time_elapsed,
    safe_literal_eval,
    save_params_to_config_file,
    save_run_info,
    set_varseek_logging_level_and_filehandler,
    set_up_logger,
    determine_write_mode,
    count_chunks,
    save_csv_chunk
)

logger = logging.getLogger(__name__)
logger = set_up_logger(logger, logging_level="INFO", save_logs=False, log_dir=None)


def apply_filters(df, filters, filtering_report_text_out=None):
    logger.info("Initial variant report")
    filtering_report_dict = make_filtering_report(df, filtering_report_text_out=filtering_report_text_out)
    initial_filtering_report_dict = filtering_report_dict.copy()

    for individual_filter in filters:
        column = individual_filter["column"]
        rule = individual_filter["rule"]
        value = individual_filter["value"]

        if column not in df.columns:
            # skip this iteration
            continue

        message = f"{column} {rule} {value}"

        logger.info(message)

        if rule == "greater_than":
            df = df.loc[(df[column].astype(float) > float(value)) | (df[column].isnull())]
        elif rule == "greater_or_equal":
            df = df.loc[(df[column].astype(float) >= float(value)) | (df[column].isnull())]
        elif rule == "less_than":
            df = df.loc[(df[column].astype(float) < float(value)) | (df[column].isnull())]
        elif rule == "less_or_equal":
            df = df.loc[(df[column].astype(float) <= float(value)) | (df[column].isnull())]
        elif rule == "between_inclusive":
            value_min, value_max = value.split(",")
            value_min, value_max = float(value_min), float(value_max)
            if value_min >= value_max:
                raise ValueError(f"Invalid range: {value}. Minimum value must be less than maximum value.")
            df = df.loc[((df[column] >= value_min) & (df[column] <= value_max) | (df[column].isnull()))]
        elif rule == "between_exclusive":
            value_min, value_max = value.split(",")
            value_min, value_max = float(value_min), float(value_max)
            if value_min >= value_max:
                raise ValueError(f"Invalid range: {value}. Minimum value must be less than maximum value.")
            df = df.loc[((df[column] > value_min) & (df[column] < value_max) | (df[column].isnull()))]
        elif rule == "top_percent":
            # Calculate the cutoff for the top percent
            percent_value = df[column].quantile((100 - float(value)) / 100)
            # Keep rows where the column value is NaN or greater than or equal to the percent value
            df = df.loc[(df[column].isnull()) | (df[column] >= percent_value)]
        elif rule == "bottom_percent":
            # Calculate the cutoff for the bottom percent
            percent_value = df[column].quantile(float(value) / 100)
            # Keep rows where the column value is NaN or less than or equal to the percent value
            df = df.loc[(df[column].isnull()) | (df[column] <= percent_value)]
        elif rule == "equal":
            df = df.loc[df[column].astype(str) == str(value)]
        elif rule == "not_equal":
            df = df.loc[df[column].astype(str) != str(value)]
        elif rule in {"is_in", "is_not_in"}:
            if value.endswith(".txt"):
                value = set(convert_txt_to_list(value))
            else:
                try:
                    value = ast.literal_eval(value)
                    if not isinstance(value, (set, list, tuple)):
                        raise ValueError("Value must be a set, list, tuple, or path to text file")
                except ValueError as exc:
                    raise ValueError("Value must be a set, list, tuple, or path to text file") from exc
            if rule == "is_in":
                df = df.loc[df[column].isin(set(value))]
            else:
                df = df.loc[~df[column].isin(set(value))]
        elif rule == "is_null":
            df = df.loc[df[column].isnull()]
        elif rule == "is_not_null":
            df = df.loc[df[column].notnull()]
        elif rule == "is_true":
            df = df.loc[df[column] == True]  # df.loc[df[column]] does not work when NaN values are present
        elif rule == "is_false":
            df = df.loc[df[column] == False]
        elif rule == "is_not_true":
            df = df.loc[(df[column] != True)]
        elif rule == "is_not_false":
            df = df.loc[(df[column] != False)]
        else:
            raise ValueError(f"Rule '{rule}' not recognized")

        filtering_report_dict = make_filtering_report(df, filtering_report_text_out=filtering_report_text_out, prior_filtering_report_dict=filtering_report_dict)

    number_of_variants_total_difference = initial_filtering_report_dict["number_of_variants_total"] - filtering_report_dict["number_of_variants_total"]
    number_of_vcrss_difference = initial_filtering_report_dict["number_of_vcrss"] - filtering_report_dict["number_of_vcrss"]
    number_of_unique_variants_difference = initial_filtering_report_dict["number_of_unique_variants"] - filtering_report_dict["number_of_unique_variants"]
    number_of_merged_variants_difference = initial_filtering_report_dict["number_of_merged_variants"] - filtering_report_dict["number_of_merged_variants"]

    message = f"Total variants filtered: {number_of_variants_total_difference}; total VCRSs filtered: {number_of_vcrss_difference}; unique variants filtered: {number_of_unique_variants_difference}; merged variants filtered: {number_of_merged_variants_difference}"
    logger.info(message)

    # Save the report string to the specified path
    if isinstance(filtering_report_text_out, str):
        with open(filtering_report_text_out, "a", encoding="utf-8") as file:
            file.write(message)

    return df


def prepare_filters_list(filters):
    filter_list = []

    if isinstance(filters, str) and filters.endswith(".txt"):
        filters = convert_txt_to_list(filters)
    elif isinstance(filters, str) and not filters.endswith(".txt"):
        filters = [filters]

    for f in filters:
        f_split_by_equal = f.split("=")
        col_rule = f_split_by_equal[0]

        if col_rule.count(":") != 1:  # was missing the ":" in between COLUMN and RULE
            raise ValueError(f"Filter format invalid: {f}. Missing colon. Expected 'COLUMN:RULE' or 'COLUMN:RULE=VALUE'")

        column, rule = col_rule.split(":")

        if rule not in all_possible_filter_rules:  # had a rule that was not one of the rules that allowed this
            raise ValueError(f"Filter format invalid: {f}. Invalid rule: {rule}.")

        if f.count("=") == 0:
            if rule not in filter_rules_that_expect_no_value:  # had 0 '=' and was not one of the rules that allowed this
                raise ValueError(f"Filter format invalid: {f}. Requires a VALUE for rule {rule}. Expected 'COLUMN:RULE=VALUE'")
            value = None
        elif f.count("=") == 1:
            if rule not in filter_rules_that_expect_single_numeric_value and rule not in filter_rules_that_expect_comma_separated_pair_of_numerics_value and rule not in filter_rules_that_expect_string_value and rule not in filter_rules_that_expect_text_file_or_list_value:  # had 1 '=' and was not one of the rules that allowed this
                raise ValueError(f"Filter format invalid: {f}. Requires no VALUE for rule {rule}. Expected 'COLUMN:RULE'")
            value = f_split_by_equal[1]
        else:  # had more than 1 '='
            raise ValueError(f"Filter format invalid: {f}. Too many '='s. Expected 'COLUMN:RULE' or 'COLUMN:RULE=VALUE'")

        if rule in filter_rules_that_expect_single_numeric_value:  # expects float-like
            try:
                value = float(value)
            except ValueError:
                raise ValueError(f"Filter format invalid: {f}. Expected single numeric value for rule {rule}. 'COLUMN:RULE=VALUE'")
        elif rule in filter_rules_that_expect_comma_separated_pair_of_numerics_value:  # expects pair of comma-separated floats
            try:
                value_min, value_max = value.split(",")
                value_min, value_max = float(value_min), float(value_max)
            except ValueError:
                raise ValueError(f"Filter format invalid: {f}. Expected a pair of comma-separated numeric values for rule {rule}. 'COLUMN:RULE=VALUE'")
        elif rule in filter_rules_that_expect_string_value:  # expects string
            pass
        elif rule in filter_rules_that_expect_text_file_or_list_value:  # expects text file or list
            if value.endswith(".txt"):
                pass
            elif (value[0] == "[" and value[-1] == "]") or (value[0] == "{" and value[-1] == "}") or (value[0] == "(" and value[-1] == ")"):
                pass
            elif isinstance(value, (list, set, tuple)):
                pass
            else:
                raise ValueError(f"Filter format invalid: {f}. Expected a text file path or list for rule {rule}. 'COLUMN:RULE=VALUE'")
            # test for list in a more thorough way (could replace the conditional above with the "[" etc checks)
            # try:
            #     value = ast.literal_eval(value)
            # except ValueError:
            #     raise ValueError(f"Filter format invalid: {f}. Expected a text file path or list for rule {rule}. 'COLUMN:RULE=VALUE'")
        elif rule in filter_rules_that_expect_no_value:  # expects no value
            pass  # lack of "=" checked earlier
        else:
            raise ValueError(f"Filter format invalid: {f}. Invalid rule: {rule}.")  # redundant with the above but keep anyways

        if rule in {"is_true", "is_not_true"}:
            value = True
        if rule in {"is_false", "is_not_false"}:
            value = False

        filter_list.append({"column": column, "rule": rule, "value": value})  # put filter_list into a list of dicts, where each dict is {"column": column, "rule": rule, "value": value}

    return filter_list


def convert_txt_to_list(txt_path):
    with open(txt_path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]

def make_filtering_report(variant_metadata_df, vcrs_header_column="vcrs_header", filtering_report_text_out=None, prior_filtering_report_dict=None):
    if "semicolon_count" not in variant_metadata_df.columns:  # already checked 'and vcrs_header_column in variant_metadata_df.columns' before
        variant_metadata_df["semicolon_count"] = variant_metadata_df[vcrs_header_column].str.count(";")

    # number of VCRSs
    number_of_vcrss = len(variant_metadata_df)

    # number of unique variants
    number_of_unique_variants = (variant_metadata_df["semicolon_count"] == 0).sum()

    # number of merged variants
    number_of_merged_variants = (variant_metadata_df.loc[variant_metadata_df["semicolon_count"] > 0, "semicolon_count"] + 1).sum()  # equivalent to doing (1) variant_metadata_df["semicolon_count"] += 1, (2) variant_metadata_df.loc[variant_metadata_df["semicolon_count"] == 1, "semicolon_count"] = np.nan, and (3) number_of_merged_variants = int(variant_metadata_df["semicolon_count"].sum())

    # number of total variants
    number_of_variants_total = number_of_unique_variants + number_of_merged_variants

    if prior_filtering_report_dict:
        number_of_variants_total_difference = prior_filtering_report_dict["number_of_variants_total"] - number_of_variants_total
        number_of_vcrss_difference = prior_filtering_report_dict["number_of_vcrss"] - number_of_vcrss
        number_of_unique_variants_difference = prior_filtering_report_dict["number_of_unique_variants"] - number_of_unique_variants
        number_of_merged_variants_difference = prior_filtering_report_dict["number_of_merged_variants"] - number_of_merged_variants
        filtering_report = f"Number of total variants: {number_of_variants_total} ({number_of_variants_total_difference} filtered); VCRSs: {number_of_vcrss} ({number_of_vcrss_difference} filtered); unique variants: {number_of_unique_variants} ({number_of_unique_variants_difference} filtered); merged variants: {number_of_merged_variants} ({number_of_merged_variants_difference} filtered)\n"
    else:
        filtering_report = f"Number of total variants: {number_of_variants_total}; VCRSs: {number_of_vcrss}; unique variants: {number_of_unique_variants}; merged variants: {number_of_merged_variants}\n"

    logger.info(filtering_report)

    # Save the report string to the specified path
    if isinstance(filtering_report_text_out, str):
        if os.path.dirname(filtering_report_text_out):
            os.makedirs(os.path.dirname(filtering_report_text_out), exist_ok=True)
        filtering_report_write_mode = "a" if os.path.exists(filtering_report_text_out) else "w"
        with open(filtering_report_text_out, filtering_report_write_mode, encoding="utf-8") as file:
            file.write(filtering_report)

    return {"number_of_vcrss": number_of_vcrss, "number_of_unique_variants": number_of_unique_variants, "number_of_merged_variants": number_of_merged_variants, "number_of_variants_total": number_of_variants_total}


def print_list_filter_rules():
    filter_md_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "docs", "filter.md")  # Get the filter.md file relative to varseek_filter.py
    column_blocks = extract_documentation_file_blocks(filter_md_path, start_pattern=r"^COLUMN:RULE", stop_pattern=r"^$")  # COLUMN:RULE to new line
    for block in column_blocks:
        print(block)


def validate_input_filter(params_dict):
    # Type-checking for paths
    if not isinstance(params_dict.get("input_dir"), (str, Path)):  # I will enforce that input_dir exists later, as otherwise it will throw an error when I call this through vk ref before vk build's out exists
        raise ValueError(f"Invalid value for input_dir: {params_dict.get('input_dir')}")
    if params_dict.get("out") is not None and not isinstance(params_dict.get("out"), (str, Path)):
        raise ValueError(f"Invalid output directory: {params_dict.get('out')}")

    # filters
    filters = params_dict["filters"]

    if not isinstance(filters, (str, list, tuple, set, Path)) and filters:  # also checks boolean - having filters empty (empty string, None, etc) is valid due to the check in vk ref - but if someone actually tries running vk filter directly with empty filters, then they will get an exception in section 0 before they even get this far
        raise ValueError(f"Invalid filters: {filters}")

    if filters and filters != "None":
        if isinstance(filters, (str, Path)):
            if os.path.isfile(filters):
                if not filters.endswith(".txt"):
                    raise ValueError(f"Invalid filters: {filters}")
            else:
                filters = [str(filters)]

        for individual_filter in filters:  # more thorough parsing provided in prepare_filters_list
            match = re.match(filter_regex, individual_filter)
            if not match:
                raise ValueError(f"Invalid filter: {individual_filter}")

    # file paths
    for param_name, file_type in {
        "variants_updated_vk_info_csv": "csv",
        "variants_updated_exploded_vk_info_csv": "csv",
        "id_to_header_csv": "csv",
        "dlist_fasta": "fasta",
        "variants_updated_filtered_csv_out": "csv",
        "variants_updated_exploded_filtered_csv_out": "csv",
        "id_to_header_filtered_csv_out": "csv",
        "dlist_filtered_fasta_out": "fasta",
        "vcrs_filtered_fasta_out": "fasta",
        "vcrs_t2g_filtered_out": "t2g",
        "wt_vcrs_filtered_fasta_out": "fasta",
        "wt_vcrs_t2g_filtered_out": "t2g",
    }.items():
        check_file_path_is_string_with_valid_extension(params_dict.get(param_name), param_name, file_type)

    # boolean
    for param_name in ["save_wt_vcrs_fasta_and_t2g", "save_variants_updated_filtered_csvs", "return_variants_updated_filtered_csv_df", "dry_run", "overwrite", "list_filter_rules"]:
        param_value = params_dict.get(param_name)
        if not isinstance(param_value, bool):
            raise ValueError(f"{param_name} must be a boolean. Got {param_value} of type {type(param_value)}.")


all_possible_filter_rules = {"greater_than", "greater_or_equal", "less_than", "less_or_equal", "between_inclusive", "between_exclusive", "top_percent", "bottom_percent", "equal", "not_equal", "is_in", "is_not_in", "is_true", "is_false", "is_not_true", "is_not_false", "is_null", "is_not_null"}
filter_rules_that_expect_single_numeric_value = {"greater_than", "greater_or_equal", "less_than", "less_or_equal", "top_percent", "bottom_percent"}
filter_rules_that_expect_comma_separated_pair_of_numerics_value = {"between_inclusive", "between_exclusive"}
filter_rules_that_expect_string_value = {"equal", "not_equal"}
filter_rules_that_expect_text_file_or_list_value = {"is_in", "is_not_in"}
filter_rules_that_expect_no_value = {"is_true", "is_false", "is_not_true", "is_not_false", "is_null", "is_not_null"}
all_possible_filter_rules_regex = "|".join(map(re.escape, all_possible_filter_rules))
filter_regex = rf"^(?P<column>\w+):(?P<rule>(?:{all_possible_filter_rules_regex}))(?:=(?P<value>.+))?$"

@report_time_elapsed
def filter(
    input_dir,
    filters,
    variants_updated_vk_info_csv=None,  # input variant metadata df
    variants_updated_exploded_vk_info_csv=None,  # input exploded variant metadata df
    id_to_header_csv=None,  # input id to header csv
    dlist_fasta=None,  # input dlist
    vcrs_id_column="vcrs_id",  # column name for vcrs id
    vcrs_sequence_column="vcrs_sequence",  # column name for vcrs sequence
    out=None,  # output directory
    variants_updated_filtered_csv_out=None,  # output metadata df
    variants_updated_exploded_filtered_csv_out=None,  # output exploded variant metadata df
    id_to_header_filtered_csv_out=None,  # output id to header csv
    dlist_filtered_fasta_out=None,  # output dlist fasta
    vcrs_filtered_fasta_out=None,  # output vcrs fasta
    vcrs_t2g_filtered_out=None,  # output t2g
    wt_vcrs_filtered_fasta_out=None,  # output wt vcrs fasta
    wt_vcrs_t2g_filtered_out=None,  # output t2g for wt vcrs fasta
    save_wt_vcrs_fasta_and_t2g=False,
    save_variants_updated_filtered_csvs=False,
    return_variants_updated_filtered_csv_df=False,
    chunksize=None,
    dry_run=False,
    list_filter_rules=False,
    overwrite=False,
    logging_level=None,
    save_logs=False,
    log_out_dir=None,
    **kwargs,
):
    """
    Filter variants based on the provided filters and save the filtered variants to a fasta file.

    # Required input arguments:
    - input_dir     (str) Path to the directory containing the input files. Corresponds to `out` in the varseek info function.
    - filters       (str or list[str]) Filter or list of filters to apply to the variant reference fasta. Each filter should be in the format COLUMN-RULE=VALUE or COLUMN-RULE (for boolean evaluation). For details, run vk filter --list_filter_rules, or see the documentation at https://github.com/pachterlab/varseek/blob/main/docs/filter.md

    # Optional input arguments:
    - variants_updated_vk_info_csv                (str) Path to the updated dataframe containing the VCRS headers and sequences. Corresponds to `variants_updated_csv_out` in the varseek build function. Only needed if the original file was changed or renamed. Default: None (will find it in `input_dir` if it exists).
    - variants_updated_exploded_vk_info_csv       (str) Path to the updated exploded dataframe containing the VCRS headers and sequences. Corresponds to `variants_updated_exploded_csv_out` in the varseek build function. Only needed if the original file was changed or renamed. Default: None (will find it in `input_dir` if it exists).
    - id_to_header_csv                             (str) Path to the csv file containing the mapping of IDs to headers generated from varseek build corresponding to vcrs_fasta. Corresponds to `id_to_header_csv_out` in the varseek build function. Only needed if the original file was changed or renamed. Default: None (will find it in `input_dir` if it exists).
    - dlist_fasta                                  (str) Path to the dlist fasta file. Default: None (will find it in `input_dir` if it exists).
    - vcrs_id_column                               (str) Column name for the VCRS ID in the variant metadata dataframe. Default: "vcrs_id".
    - vcrs_sequence_column                        (str) Column name for the VCRS sequence in the variant metadata dataframe. Default: "vcrs_sequence".

    # Optional output file paths: (only needed if changing/customizing file names or locations):
    - out                                          (str) Path to the directory where the output files will be saved. Default: `input_dir`.
    - variants_updated_filtered_csv_out            (str) Path to the filtered variant metadata dataframe. Default: None (will be saved in `out`).
    - variants_updated_exploded_filtered_csv_out   (str) Path to the filtered exploded variant metadata dataframe. Default: None (will be saved in `out`).
    - id_to_header_filtered_csv_out                (str) Path to the filtered id to header csv. Default: None (will be saved in `out`).
    - dlist_filtered_fasta_out                     (str) Path to the filtered dlist fasta file. Default: None (will be saved in `out`).
    - vcrs_filtered_fasta_out                      (str) Path to the filtered vcrs fasta file. Default: None (will be saved in `out`).
    - vcrs_t2g_filtered_out                        (str) Path to the filtered t2g file. Default: None (will be saved in `out`).
    - wt_vcrs_filtered_fasta_out                   (str) Path to the filtered wt vcrs fasta file. Default: None (will be saved in `out`).
    - wt_vcrs_t2g_filtered_out                     (str) Path to the filtered t2g file for wt vcrs fasta. Default: None (will be saved in `out`).

    # Returning and saving of optional output
    - save_wt_vcrs_fasta_and_t2g                   (bool) If True, save the filtered wt vcrs fasta and t2g files. Default: False.
    - save_variants_updated_filtered_csvs         (bool) If True, save the filtered variant metadata dataframe. Default: False.
    - return_variants_updated_filtered_csv_df     (bool) If True, return the filtered variant metadata dataframe. Default: False.

    # General arguments:
    - chunksize                                    (int) Number of variants to process at a time. If None, then all variants will be processed at once. Default: None.
    - dry_run                                      (bool) If True, print the parameters and exit without running the function. Default: False.
    - list_filter_rules                            (bool) If True, print the available filter rules and exit without running the function. Default: False.
    - overwrite                                    (bool) If True, overwrite the output files if they already exist. Default: False.
    - logging_level                                (str) Logging level. Can also be set with the environment variable VARSEEK_LOGGING_LEVEL. Default: INFO.
    - save_logs                                    (True/False) Whether to save logs to a file. Default: False.
    - log_out_dir                                  (str) Directory to save logs. Default: None (do not save logs).

    # Hidden arguments:
    - filter_all_dlists                            (bool) If True, filter all dlists. Default: False.
    - dlist_genome_fasta                           (str) Path to the genome dlist fasta file. Default: None.
    - dlist_cdna_fasta                             (str) Path to the cDNA dlist fasta file. Default: None.
    - dlist_genome_filtered_fasta_out              (str) Path to the filtered genome dlist fasta file. Default: None.
    - dlist_cdna_filtered_fasta_out                (str) Path to the filtered cDNA dlist fasta file. Default: None.
    - save_vcrs_filtered_fasta_and_t2g             (bool) If True, save the filtered vcrs fasta and t2g files. Default: True.
    - use_IDs                                      (bool) If True, use IDs instead of headers. Default: False.
    - make_internal_copies              (bool) Whether to make internal copies of the input dataframes. Default: True
    """
    # * 0. Informational arguments that exit early
    if list_filter_rules:
        print_list_filter_rules()
        return

    if not filters or filters == "None":
        raise ValueError("No filters provided. Please provide filters to apply.")

    # * 1. logger and set out folder (must to it up here or else logger and config will save in the wrong place)
    if out is None:
        out = input_dir if input_dir else "."

    if save_logs and not log_out_dir:
        log_out_dir = os.path.join(out, "logs")
    if not kwargs.get("running_within_chunk_iteration", False):
        set_varseek_logging_level_and_filehandler(logging_level=logging_level, save_logs=save_logs, log_dir=log_out_dir)

    if isinstance(filters, (list, tuple)) and len(filters) == 1:
        filters = filters[0]

    # * 1.5 Chunk iteration
    if chunksize and return_variants_updated_filtered_csv_df:
        raise ValueError("Cannot return variants_updated_filtered_csv_df when using chunksize. Please set return_variants_updated_filtered_csv_df to False.")
    if chunksize is not None:
        params_dict = make_function_parameter_to_value_dict(1)
        for key in ["variants_updated_vk_info_csv", "id_to_header_csv", "chunksize"]:
            params_dict.pop(key, None)
        
        variants_updated_vk_info_csv = os.path.join(input_dir, "variants_updated_vk_info.csv") if not variants_updated_vk_info_csv else variants_updated_vk_info_csv  # copy-paste from below (sorry)
        id_to_header_csv = os.path.join(input_dir, "id_to_header_mapping.csv") if (not id_to_header_csv and os.path.isfile(os.path.join(input_dir, "id_to_header_mapping.csv"))) else None  # copy-paste from below
        
        total_chunks = count_chunks(variants_updated_vk_info_csv, chunksize)
        for i in range(0, total_chunks):
            chunk_number = i + 1  # start at 1
            logger.info(f"Processing chunk {chunk_number}/{total_chunks}")
            variants_updated_vk_info_csv_chunk = save_csv_chunk(csv_path=variants_updated_vk_info_csv, chunk_number=chunk_number, chunksize=chunksize)
            id_to_header_csv_chunk = save_csv_chunk(csv_path=id_to_header_csv, chunk_number=chunk_number, chunksize=chunksize)
            filter(variants_updated_vk_info_csv=variants_updated_vk_info_csv_chunk, id_to_header_csv=id_to_header_csv_chunk, id_to_header_csv_full=id_to_header_csv, chunksize=None, chunk_number=chunk_number, total_chunks=total_chunks, running_within_chunk_iteration=True, **params_dict)  # running_within_chunk_iteration here for logger setup and report_time_elapsed decorator
            if chunk_number == total_chunks:
                for tmp_file in [variants_updated_vk_info_csv_chunk, id_to_header_csv_chunk]:
                    if isinstance(tmp_file, str) and os.path.exists(tmp_file):
                        os.remove(tmp_file)
                return
            
    chunk_number = kwargs.get("chunk_number", 1)
    first_chunk = (chunk_number == 1)

    # * 2. Type-checking
    params_dict = make_function_parameter_to_value_dict(1)
    validate_input_filter(params_dict)

    if not os.path.isdir(input_dir):  # only use os.path.isdir when I require that a directory already exists; checked outside validate_input_info to avoid raising issue when type-checking within vk ref
        raise ValueError(f"Input directory '{input_dir}' does not exist. Please provide a valid directory.")

    # * 3. Dry-run
    if dry_run:
        print_varseek_dry_run(params_dict, function_name="filter")
        return

    # * 4. Save params to config file and run info file
    config_file = os.path.join(out, "config", "vk_filter_config.json")
    save_params_to_config_file(params_dict, config_file)

    run_info_file = os.path.join(out, "config", "vk_filter_run_info.txt")
    save_run_info(run_info_file)

    # * 5. Set up default folder/file input paths, and make sure the necessary ones exist
    # have the option to filter other dlists as kwargs
    filter_all_dlists = kwargs.get("filter_all_dlists", False)
    dlist_genome_fasta = kwargs.get("dlist_genome_fasta", None)
    dlist_cdna_fasta = kwargs.get("dlist_cdna_fasta", None)
    dlist_genome_filtered_fasta_out = kwargs.get("dlist_genome_filtered_fasta_out", None)
    dlist_cdna_filtered_fasta_out = kwargs.get("dlist_cdna_filtered_fasta_out", None)
    save_vcrs_filtered_fasta_and_t2g = kwargs.get("save_vcrs_filtered_fasta_and_t2g", True)
    use_IDs = kwargs.get("use_IDs", False)
    make_internal_copies = kwargs.get("make_internal_copies", True)

    if filter_all_dlists:
        if not dlist_genome_fasta:
            dlist_genome_fasta = os.path.join(input_dir, "dlist_genome.fa")
        if not dlist_cdna_fasta:
            dlist_cdna_fasta = os.path.join(input_dir, "dlist_cdna.fa")
        if not dlist_genome_filtered_fasta_out:
            dlist_genome_filtered_fasta_out = os.path.join(out, "dlist_genome_filtered.fa")
        if not dlist_cdna_filtered_fasta_out:
            dlist_cdna_filtered_fasta_out = os.path.join(out, "dlist_cdna_filtered.fa")
        for output_file in [dlist_genome_filtered_fasta_out, dlist_cdna_filtered_fasta_out]:
            if output_file and os.path.dirname(output_file):
                os.makedirs(os.path.dirname(output_file), exist_ok=True)

    # define input file names if not provided
    if variants_updated_vk_info_csv is None:
        variants_updated_vk_info_csv = os.path.join(input_dir, "variants_updated_vk_info.csv")
    if variants_updated_exploded_vk_info_csv is None:
        variants_updated_exploded_vk_info_csv = os.path.join(input_dir, "variants_updated_exploded_vk_info.csv")
    if dlist_fasta is None:
        dlist_fasta = os.path.join(input_dir, "dlist.fa")
    if id_to_header_csv is None:
        id_to_header_csv = os.path.join(input_dir, "id_to_header_mapping.csv")

    # set input file names to None if they do not exist
    if not ((isinstance(variants_updated_vk_info_csv, str) and os.path.isfile(variants_updated_vk_info_csv)) or isinstance(variants_updated_vk_info_csv, pd.DataFrame)):
        raise FileNotFoundError(f"Variant metadata file not found at {variants_updated_vk_info_csv}.")
    if not ((isinstance(variants_updated_exploded_vk_info_csv, str) and os.path.isfile(variants_updated_exploded_vk_info_csv)) or isinstance(variants_updated_exploded_vk_info_csv, pd.DataFrame)):
        # logger.info(f"Exploded variant metadata file not found at {variants_updated_exploded_vk_info_csv}. Skipping filtering of exploded variant metadata.")
        variants_updated_exploded_vk_info_csv = None
    if not os.path.isfile(dlist_fasta):
        # logger.warning(f"d-list file not found at {dlist_fasta}. Skipping filtering of d-list.")
        dlist_fasta = None
    if not os.path.isfile(id_to_header_csv):
        # logger.warning(f"ID to header csv file not found at {id_to_header_csv}. Skipping filtering of ID to header csv.")
        id_to_header_csv = None

    # * 6. Set up default folder/file output paths, and make sure they don't exist unless overwrite=True
    # if someone specifies an output path, then it should be saved
    if wt_vcrs_filtered_fasta_out or wt_vcrs_t2g_filtered_out:
        save_wt_vcrs_fasta_and_t2g = True
    if variants_updated_filtered_csv_out or variants_updated_exploded_vk_info_csv:
        save_variants_updated_filtered_csvs = True

    # define output file names if not provided
    if not variants_updated_filtered_csv_out:  # variants_updated_vk_info_csv must exist or else an exception will be raised from earlier
        variants_updated_filtered_csv_out = os.path.join(out, "variants_updated_filtered.csv")
    if not variants_updated_exploded_filtered_csv_out:
        variants_updated_exploded_filtered_csv_out = os.path.join(out, "variants_updated_exploded_filtered.csv")
    if not id_to_header_filtered_csv_out:
        id_to_header_filtered_csv_out = os.path.join(out, "id_to_header_mapping_filtered.csv")
    if not dlist_filtered_fasta_out:
        dlist_filtered_fasta_out = os.path.join(out, "dlist_filtered.fa")
    if not vcrs_filtered_fasta_out:  # this file must be created
        vcrs_filtered_fasta_out = os.path.join(out, "vcrs_filtered.fa")
    if not vcrs_t2g_filtered_out:  # this file must be created
        vcrs_t2g_filtered_out = os.path.join(out, "vcrs_t2g_filtered.txt")
    if not wt_vcrs_filtered_fasta_out:
        wt_vcrs_filtered_fasta_out = os.path.join(out, "wt_vcrs_filtered.fa")
    if not wt_vcrs_t2g_filtered_out:
        wt_vcrs_t2g_filtered_out = os.path.join(out, "wt_vcrs_t2g_filtered.txt")
    filtering_report_text_out = os.path.join(out, "filtering_report.txt")

    # make sure directories of all output files exist
    output_files = [variants_updated_filtered_csv_out, variants_updated_exploded_filtered_csv_out, id_to_header_filtered_csv_out, dlist_filtered_fasta_out, vcrs_filtered_fasta_out, vcrs_t2g_filtered_out, wt_vcrs_filtered_fasta_out, wt_vcrs_t2g_filtered_out]
    for output_file in output_files:
        if os.path.isfile(output_file) and not overwrite and first_chunk:
            raise ValueError(f"Output file '{output_file}' already exists. Set 'overwrite=True' to overwrite it.")
        if os.path.dirname(output_file):
            os.makedirs(os.path.dirname(output_file), exist_ok=True)

    # * 7. Define kwargs defaults
    # defined at the beginning of (5) here, as they were needed in that section

    # * 8. Start the actual function
    # filters must either be a dict (as described in docs) or a path to a JSON file
    if isinstance(filters, (list, tuple, set, str)):
        filters = prepare_filters_list(filters)
    elif isinstance(filters, dict):
        pass  # filters is already a dict from argparse
    else:
        raise ValueError(f"Invalid filters: {filters}")

    if isinstance(variants_updated_vk_info_csv, str):
        variant_metadata_df = pd.read_csv(variants_updated_vk_info_csv)
    elif isinstance(variants_updated_vk_info_csv, pd.DataFrame):
        if make_internal_copies:
            variant_metadata_df = variants_updated_vk_info_csv.copy()
        else:
            variant_metadata_df = variants_updated_vk_info_csv
    else:
        raise ValueError(f"Invalid variant metadata dataframe/csv path: {variants_updated_vk_info_csv}")

    vcrs_header_column = "vcrs_header"

    if not variant_metadata_df[vcrs_id_column].iloc[0].startswith("vcrs"):  # use_IDs was False in vk build and header column is vcrs_id_column
        vcrs_header_column = vcrs_id_column
    else:
        if vcrs_header_column not in variant_metadata_df.columns:
            if id_to_header_csv and os.path.isfile(id_to_header_csv):
                id_to_header_dict = make_mapping_dict(id_to_header_csv, dict_key="id")

                if id_to_header_dict is not None:
                    variant_metadata_df[vcrs_header_column] = variant_metadata_df[vcrs_id_column].map(id_to_header_dict)
            else:
                raise ValueError(f"ID to header mapping file not found at {id_to_header_csv}, and vcrs_id provides an ID that must be replaced. Please provide a valid file.")

    if "semicolon_count" not in variant_metadata_df.columns and vcrs_header_column in variant_metadata_df.columns:  # adding for reporting purposes
        variant_metadata_df["semicolon_count"] = variant_metadata_df[vcrs_header_column].str.count(";")

    filtered_df = apply_filters(variant_metadata_df, filters, filtering_report_text_out=filtering_report_text_out)  #$$$ the real meat of the function

    if kwargs.get("called_from_vk_sim"):  # return early because I don't need anything else
        return filtered_df

    filtered_df = filtered_df.copy()  # here to avoid pandas warning about assigning to a slice rather than a copy

    if "semicolon_count" in filtered_df.columns:
        filtered_df = filtered_df.drop(columns=["semicolon_count"])

    if save_variants_updated_filtered_csvs:
        filtered_df.to_csv(variants_updated_filtered_csv_out, index=False, header=first_chunk, mode=determine_write_mode(variants_updated_filtered_csv_out, overwrite=overwrite, first_chunk=first_chunk))

    # make vcrs_filtered_fasta_out
    if use_IDs:
        output_fasta_header_column = vcrs_id_column
    else:
        output_fasta_header_column = vcrs_header_column
    filtered_df[output_fasta_header_column] = filtered_df[output_fasta_header_column].astype(str)

    if save_vcrs_filtered_fasta_and_t2g:
        filtered_df["fasta_format"] = ">" + filtered_df[output_fasta_header_column] + "\n" + filtered_df[vcrs_sequence_column] + "\n"

        mode=determine_write_mode(vcrs_filtered_fasta_out, overwrite=overwrite, first_chunk=first_chunk)
        with open(vcrs_filtered_fasta_out, mode, encoding="utf-8") as fasta_file:
            fasta_file.write("".join(filtered_df["fasta_format"].values))

        filtered_df.drop(columns=["fasta_format"], inplace=True)

        # make vcrs_t2g_filtered_out
        create_identity_t2g(vcrs_filtered_fasta_out, vcrs_t2g_filtered_out, mode=determine_write_mode(vcrs_t2g_filtered_out, overwrite=overwrite, first_chunk=first_chunk))

    # make wt_vcrs_filtered_fasta_out and wt_vcrs_t2g_filtered_out iff save_wt_vcrs_fasta_and_t2g is True
    if save_wt_vcrs_fasta_and_t2g:
        variants_with_exactly_1_wt_sequence_per_row = filtered_df[[output_fasta_header_column, "wt_sequence"]].copy()

        variants_with_exactly_1_wt_sequence_per_row["wt_sequence"] = variants_with_exactly_1_wt_sequence_per_row["wt_sequence"].apply(safe_literal_eval)

        if isinstance(variants_with_exactly_1_wt_sequence_per_row["wt_sequence"][0], list):  # remove the rows with multiple WT counterparts for 1 VCRS, and convert the list of strings to string
            # Step 1: Filter rows where the length of the set of the list in `wt_sequence` is 1
            variants_with_exactly_1_wt_sequence_per_row = variants_with_exactly_1_wt_sequence_per_row[variants_with_exactly_1_wt_sequence_per_row["wt_sequence"].apply(lambda x: len(set(x)) == 1)]

            # Step 2: Convert the list to a string
            variants_with_exactly_1_wt_sequence_per_row["wt_sequence"] = variants_with_exactly_1_wt_sequence_per_row["wt_sequence"].apply(lambda x: x[0])

        variants_with_exactly_1_wt_sequence_per_row["fasta_format_wt"] = ">" + variants_with_exactly_1_wt_sequence_per_row[output_fasta_header_column] + "\n" + variants_with_exactly_1_wt_sequence_per_row["wt_sequence"] + "\n"

        mode=determine_write_mode(wt_vcrs_filtered_fasta_out, overwrite=overwrite, first_chunk=first_chunk)
        with open(wt_vcrs_filtered_fasta_out, mode, encoding="utf-8") as fasta_file:
            fasta_file.write("".join(variants_with_exactly_1_wt_sequence_per_row["fasta_format_wt"].values))

        filtered_df.drop(columns=["fasta_format_wt"], inplace=True)

        create_identity_t2g(wt_vcrs_filtered_fasta_out, wt_vcrs_t2g_filtered_out, mode=determine_write_mode(wt_vcrs_t2g_filtered_out, overwrite=overwrite, first_chunk=first_chunk))

    if kwargs.get("running_within_chunk_iteration", False):
        if chunk_number != kwargs.get("total_chunks", 0):
            return  # go back to the iteration
        else:
            filtered_df_vcrs_ids = set(pd.read_csv(variants_updated_filtered_csv_out, usecols=[vcrs_id_column])[vcrs_id_column])
    else:
        filtered_df.reset_index(drop=True, inplace=True)
        filtered_df_vcrs_ids = set(filtered_df[vcrs_id_column])  # no need to use output_fasta_header_column here because output_fasta_header_column is only necessary when saving the fasta files (this is using the IDs just to check for membership, not to save to a file any differently)

    # make id_to_header_filtered_csv_out iff id_to_header_csv exists
    if kwargs.get("id_to_header_csv_full"):  # for chunking
        id_to_header_csv = kwargs.get("id_to_header_csv_full")
    if id_to_header_csv and os.path.isfile(id_to_header_csv):
        if not chunksize:
            id_to_header_df = pd.read_csv(id_to_header_csv)
            id_to_header_df = id_to_header_df[id_to_header_df['vcrs_id'].isin(filtered_df_vcrs_ids)]
            id_to_header_df.to_csv(id_to_header_filtered_csv_out, index=False)
            del id_to_header_df
        else:
            first_chunk_inner = True
            for chunk in pd.read_csv(id_to_header_csv, chunksize=chunksize):
                chunk = chunk[chunk['vcrs_id'].isin(filtered_df_vcrs_ids)]  # filter the chink
                if not chunk.empty:
                    chunk.to_csv(id_to_header_filtered_csv_out, mode="w" if first_chunk_inner else "a", index=False, header=first_chunk_inner)  # Append data to CSV, writing header only for the first chunk
                    first_chunk_inner = False
            

    # make variants_updated_exploded_filtered_csv_out iff variants_updated_exploded_vk_info_csv exists
    if save_variants_updated_filtered_csvs and variants_updated_exploded_vk_info_csv and os.path.isfile(variants_updated_exploded_vk_info_csv):
        if not chunksize:
            variant_metadata_df_exploded = pd.read_csv(variants_updated_exploded_vk_info_csv)
            filtered_variant_metadata_df_exploded = variant_metadata_df_exploded[variant_metadata_df_exploded[vcrs_id_column].isin(filtered_df_vcrs_ids)]  # Filter variant_metadata_df_exploded based on these unique values
            filtered_variant_metadata_df_exploded.to_csv(variants_updated_exploded_filtered_csv_out, index=False)
            del filtered_variant_metadata_df_exploded
        else:
            first_chunk_inner = True
            for chunk in pd.read_csv(variants_updated_exploded_vk_info_csv, chunksize=chunksize):
                chunk = chunk[chunk[vcrs_id_column].isin(filtered_df_vcrs_ids)]  # filter the chink
                if not chunk.empty:
                    chunk.to_csv(id_to_header_filtered_csv_out, mode="w" if first_chunk_inner else "a", index=False, header=first_chunk_inner)  # Append data to CSV, writing header only for the first chunk
                    first_chunk_inner = False

    # make dlist_filtered_fasta_out iff dlist_fasta exists
    if dlist_fasta and os.path.isfile(dlist_fasta):
        filter_fasta(dlist_fasta, dlist_filtered_fasta_out, filtered_df_vcrs_ids)
        # TODO: when use_IDs=False (which is the default), convert the IDs to headers (in a copied file is fine) - will take some parsing because the dlist headers also have the k-mer position of the original VCRS, so I probably want to remove these or otherwise deal with these to ensure seamless swapping

    if filter_all_dlists:
        if dlist_genome_fasta and os.path.isfile(dlist_genome_fasta):
            filter_fasta(dlist_genome_fasta, dlist_genome_filtered_fasta_out, filtered_df_vcrs_ids)  # TODO: same as above (when use_IDs=False...)
        if dlist_cdna_fasta and os.path.isfile(dlist_cdna_fasta):
            filter_fasta(dlist_cdna_fasta, dlist_cdna_filtered_fasta_out, filtered_df_vcrs_ids)  # TODO: same as above (when use_IDs=False...)

    if save_vcrs_filtered_fasta_and_t2g:
        logger.info(f"Output fasta file with filtered variants: {vcrs_filtered_fasta_out}")
        logger.info(f"t2g file containing mutated sequences created at {vcrs_t2g_filtered_out}.")
    if dlist_filtered_fasta_out and os.path.isfile(dlist_filtered_fasta_out):
        logger.info(f"Filtered dlist fasta created at {dlist_filtered_fasta_out}.")

    if return_variants_updated_filtered_csv_df:
        return filtered_df
