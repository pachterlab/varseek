import os
import csv
import time
import ast
import re
import numpy as np
import pandas as pd
from tqdm import tqdm
from pdb import set_trace as st
from .utils import (
    set_up_logger,
    filter_fasta,
    create_mutant_t2g,
    fasta_summary_stats,
    safe_literal_eval,
    save_params_to_config_file,
    make_function_parameter_to_value_dict,
    check_file_path_is_string_with_valid_extension,
    print_varseek_dry_run,
    report_time_elapsed,
    extract_documentation_file_blocks,
    save_run_info
)

logger = set_up_logger()

def apply_filters(df, filters, verbose=False, filtering_report_text_out=None):
    logger.info("Initial mutation report")
    filtering_report_dict = make_filtering_report(df, logger=logger, verbose=verbose, filtering_report_text_out=filtering_report_text_out)
    initial_filtering_report_dict = filtering_report_dict.copy()
    
    for filter in filters:
        column = filter["column"]
        rule = filter["rule"]
        value = filter["value"]

        if column not in df.columns:
            # skip this iteration
            continue

        message = f"{column} {rule} {value}"
        print(message) if not logger else logger.info(message)

        if rule == "greater_than":
            df = df.loc[(df[column].astype(float) > float(value)) | (df[column].isnull())]
        if rule == "greater_or_equal":
            df = df.loc[(df[column].astype(float) >= float(value)) | (df[column].isnull())]
        elif rule == "less_than":
            df = df.loc[(df[column].astype(float) < float(value)) | (df[column].isnull())]
        elif rule == "less_or_equal":
            df = df.loc[(df[column].astype(float) <= float(value)) | (df[column].isnull())]
        elif rule == "between_inclusive":
            value_min, value_max = value.split(",")
            value_min, value_max = float(value_min), float(value_max)
            df = df.loc[((df[column] >= value_min) & (df[column] <= value_max) | (df[column].isnull()))]
        elif rule == "between_exclusive":
            value_min, value_max = value.split(",")
            value_min, value_max = float(value_min), float(value_max)
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
                    if not isinstance(value, set) or not isinstance(value, list) or not isinstance(value, tuple):
                        raise ValueError(f"Value must be a set, list, tuple, or path to text file")
                except ValueError:
                    raise ValueError(f"Value must be a set, list, tuple, or path to text file")
            if rule == "is_in":
                df = df.loc[df[column].isin(set(value))]
            else:
                df = df.loc[~df[column].isin(set(value))]
        elif rule == "is_null":
            df = df.loc[df[column].isnull()]
        elif rule == "is_not_null":
            df = df.loc[df[column].notnull()]
        elif rule == "is_true":
            df = df.loc[df[column] == True]
        elif rule == "is_false":
            df = df.loc[df[column] == False]
        elif rule == "is_not_true":
            df = df.loc[(df[column] != True) | df[column].isnull()]
        elif rule == "is_not_false":
            df = df.loc[(df[column] != False) | df[column].isnull()]
        else:
            raise ValueError(f"Rule '{rule}' not recognized")
        
        filtering_report_dict = make_filtering_report(df, logger=logger, verbose=verbose, filtering_report_text_out=filtering_report_text_out, prior_filtering_report_dict=filtering_report_dict)

    if verbose:
        number_of_mutations_total_difference = initial_filtering_report_dict["number_of_mutations_total"] - filtering_report_dict["number_of_mutations_total"]
        number_of_vcrss_total_difference = initial_filtering_report_dict["number_of_vcrss_total"] - filtering_report_dict["number_of_vcrss_total"]
        number_of_unique_mutations_difference = initial_filtering_report_dict["number_of_unique_mutations"] - filtering_report_dict["number_of_unique_mutations"]
        number_of_merged_mutations_difference = initial_filtering_report_dict["number_of_merged_mutations"] - filtering_report_dict["number_of_merged_mutations"]

        message = f"Total mutations filtered: {number_of_mutations_total_difference}; total VCRSs filtered: {number_of_vcrss_total_difference}; unique mutations filtered: {number_of_unique_mutations_difference}; merged mutations filtered: {number_of_merged_mutations_difference}"
        print(message) if not logger else logger.info(message)

    return df



def prepare_filters_list(filters):
    filter_list = []

    if isinstance(filters, str) and filters.endswith(".txt"):
        filters = convert_txt_to_list(filters)

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
            if not value.endswith(".txt"):
                raise ValueError(f"Filter format invalid: {f}. Expected a text file path or list for rule {rule}. 'COLUMN:RULE=VALUE'")
            else:
                # test for list in a lightweight way
                if (value[0] == "[" and value[-1] == "]") or (value[0] == "{" and value[-1] == "}") or (value[0] == "(" and value[-1] == ")"):
                    raise ValueError(f"Filter format invalid: {f}. Expected a text file path or list for rule {rule}. 'COLUMN:RULE=VALUE'")
                # test for list in a more thorough way
                # try:
                #     value = ast.literal_eval(value)
                # except ValueError:
                #     raise ValueError(f"Filter format invalid: {f}. Expected a text file path or list for rule {rule}. 'COLUMN:RULE=VALUE'")
        elif rule in filter_rules_that_expect_no_value:  # expects no value
            pass
        else:
            raise ValueError(f"Filter format invalid: {f}. Invalid rule: {rule}.")  # redundant with the above but keep anyways
        
        if rule in {"is_true", "is_not_true"}:
            value = True
        if rule in {"is_false", "is_not_false"}:
            value = False

        filter_list.append({"column": column, "rule": rule, "value": value})  # put filter_list into a list of dicts, where each dict is {"column": column, "rule": rule, "value": value}
    
    return column


def convert_txt_to_list(txt_path):
    with open(txt_path, "r") as f:
        return [line.strip() for line in f if line.strip()]


def filter_id_to_header_csv(
    id_to_header_csv, id_to_header_csv_filtered, filtered_df_mcrs_ids
):
    with open(id_to_header_csv, mode="r") as infile, open(
        id_to_header_csv_filtered, mode="w", newline=""
    ) as outfile:
        reader = csv.reader(infile)
        writer = csv.writer(outfile)

        # Loop through each row in the input file
        for row in reader:
            key = row[0]  # Assuming the first column is the key
            # Write row to output file if key is in filtered_df_mcrs_ids
            if key in filtered_df_mcrs_ids:
                writer.writerow(row)

def make_filtering_report(mutation_metadata_df, logger = None, verbose = False, filtering_report_text_out = None, prior_filtering_report_dict = None):
    if "semicolon_count" not in mutation_metadata_df.columns:
        mutation_metadata_df["semicolon_count"] = mutation_metadata_df["mcrs_header"].str.count(";")
    
    # number of VCRSs
    number_of_vcrss = len(mutation_metadata_df)

    # number of unique mutations
    number_of_unique_mutations = (mutation_metadata_df["semicolon_count"] == 0).sum()

    # number of merged mutations
    number_of_merged_mutations = (mutation_metadata_df.loc[mutation_metadata_df["semicolon_count"] > 0, "semicolon_count"] + 1).sum()  # equivalent to doing (1) mutation_metadata_df["semicolon_count"] += 1, (2) mutation_metadata_df.loc[mutation_metadata_df["semicolon_count"] == 1, "semicolon_count"] = np.nan, and (3) number_of_merged_mutations = int(mutation_metadata_df["semicolon_count"].sum())
    
    # number of total mutations
    number_of_mutations_total = number_of_unique_mutations + number_of_merged_mutations

    if prior_filtering_report_dict:
        number_of_mutations_total_difference = prior_filtering_report_dict["number_of_mutations_total"] - number_of_mutations_total
        number_of_vcrss_difference = prior_filtering_report_dict["number_of_vcrss"] - number_of_vcrss
        number_of_unique_mutations_difference = prior_filtering_report_dict["number_of_unique_mutations"] - number_of_unique_mutations
        number_of_merged_mutations_difference = prior_filtering_report_dict["number_of_merged_mutations"] - number_of_merged_mutations
        filtering_report = f"Number of total mutations: {number_of_mutations_total} ({number_of_mutations_total_difference} filtered); VCRSs: {number_of_vcrss} ({number_of_vcrss_difference} filtered); unique mutations: {number_of_unique_mutations} ({number_of_unique_mutations_difference} filtered); merged mutations: {number_of_merged_mutations} ({number_of_merged_mutations_difference} filtered)\n"
    else:
        filtering_report = f"Number of total mutations: {number_of_mutations_total}; VCRSs: {number_of_vcrss}; unique mutations: {number_of_unique_mutations}; merged mutations: {number_of_merged_mutations}\n"

    if verbose:
        if logger:
            logger.info(filtering_report)
        else:
            print(filtering_report)

    # Save the report string to the specified path
    if isinstance(filtering_report_text_out, str):
        if os.path.dirname(filtering_report_text_out):
            os.makedirs(os.path.dirname(filtering_report_text_out), exist_ok=True)
        filtering_report_write_mode = "a" if os.path.exists(filtering_report_text_out) else "w"
        with open(filtering_report_text_out, filtering_report_write_mode) as file:
            file.write(filtering_report)
    
    return {"number_of_vcrss": number_of_vcrss, "number_of_unique_mutations": number_of_unique_mutations, "number_of_merged_mutations": number_of_merged_mutations, "number_of_mutations_total": number_of_mutations_total}

def print_list_filter_rules():
    filter_md_path = os.path.join(os.path.dirname(os.getcwd()), "docs", "filter.md")  # Get the filter.md file relative to varseek_filter.py
    column_blocks = extract_documentation_file_blocks(filter_md_path, start_pattern=r"^COLUMN:RULE", stop_pattern=r"^$")  # COLUMN:RULE to new line
    for block in column_blocks:
        print(block)

def validate_input_filter(params_dict):
    # directories
    input_dir = params_dict["input_dir"]
    out = params_dict["out"]
    # Type-checking for paths
    if not isinstance(input_dir, str) or not os.path.isdir(input_dir):
        raise ValueError(f"Invalid input directory: {input_dir}")
    if not isinstance(out, str) or not os.path.isdir(out):
        raise ValueError(f"Invalid input directory: {out}")
    
    # filters
    filters = params_dict["filters"]
    if isinstance(filters, str):
        if not os.path.isfile(filters) or not filters.endswith(".txt"):
            raise ValueError(f"Invalid filters: {filters}")
    
    if not (isinstance(filters, dict) or isinstance(filters, str) or isinstance(filters, list) or isinstance(filters, tuple) or isinstance(filters, set)):
        raise ValueError(f"Invalid filters: {filters}")
    
    # file paths
    for param_name, file_type in {
        "mutations_updated_vk_info_csv": "csv",
        "mutations_updated_exploded_vk_info_csv": "csv",
        "id_to_header_csv": "csv",
        "dlist_fasta": "fasta",
        "mutations_updated_filtered_csv_out": "csv",
        "mutations_updated_exploded_filtered_csv_out": "csv",
        "id_to_header_filtered_csv_out": "csv",
        "dlist_filtered_fasta_out": "fasta",
        "mcrs_filtered_fasta_out": "fasta",
        "mcrs_t2g_filtered_out": "t2g",
        "wt_mcrs_filtered_fasta_out": "fasta",
        "wt_mcrs_t2g_filtered_out": "t2g",
    }:
        check_file_path_is_string_with_valid_extension(params_dict.get(param_name), param_name, file_type)

    # boolean
    for param_name in ["save_wt_mcrs_fasta_and_t2g", "save_mutations_updated_filtered_csvs", "return_mutations_updated_filtered_csv_df", "dry_run", "overwrite", "verbose", "list_filter_rules"]:
        param_value = params_dict.get(param_name)
        if not isinstance(param_value, bool):
            raise ValueError(f"{param_name} must be a boolean. Got {param_value} of type {type(param_value)}.")

all_possible_filter_rules = {"greater_than", "greater_or_equal", "less_than", "less_or_equal", "between_inclusive", "between_exclusive", "top_percent", "bottom_percent", "equal", "not_equal", "is_in", "is_not_in", "is_true", "is_false", "is_not_true", "is_not_false", "is_null", "is_not_null"}
filter_rules_that_expect_single_numeric_value = {"greater_than", "greater_or_equal", "less_than", "less_or_equal", "top_percent", "bottom_percent"}
filter_rules_that_expect_comma_separated_pair_of_numerics_value = {"between_inclusive", "between_exclusive"}
filter_rules_that_expect_string_value = {"equal", "not_equal"}
filter_rules_that_expect_text_file_or_list_value = {"is_in", "is_not_in"}
filter_rules_that_expect_no_value = {"is_true", "is_false", "is_not_true", "is_not_false", "is_null", "is_not_null"}

def filter(
    input_dir,
    filters,
    mutations_updated_vk_info_csv=None,  # input mutation metadata df
    mutations_updated_exploded_vk_info_csv=None,  # input exploded mutation metadata df
    id_to_header_csv=None,  # input id to header csv
    dlist_fasta=None,  # input dlist
    out=None,  # output directory
    mutations_updated_filtered_csv_out=None,  # output metadata df
    mutations_updated_exploded_filtered_csv_out=None,  # output exploded mutation metadata df
    id_to_header_filtered_csv_out=None,  # output id to header csv
    dlist_filtered_fasta_out=None,  # output dlist fasta
    mcrs_filtered_fasta_out=None,  # output mcrs fasta
    mcrs_t2g_filtered_out=None,  # output t2g
    wt_mcrs_filtered_fasta_out=None,  # output wt mcrs fasta
    wt_mcrs_t2g_filtered_out=None,  # output t2g for wt mcrs fasta
    save_wt_mcrs_fasta_and_t2g=False,
    save_mutations_updated_filtered_csvs=False,
    return_mutations_updated_filtered_csv_df=False,
    dry_run=False,
    list_filter_rules=False,
    overwrite=False,
    verbose=True,
    **kwargs,
):
    """
    Filter mutations based on the provided filters and save the filtered mutations to a fasta file.

    # Required input arguments:
    - input_dir     (str) Path to the directory containing the input files. Corresponds to `out` in the varseek info function.
    - filters       (str or list[str]) String or list of filters to apply. If a string, it should be a path to a txt file containing the filters. If a list, it should be a list of strings in the format COLUMNNAME-RULE=VALUE. See documentation for details, or run vk.filter(list_filter_rules=True).

    # Optional input arguments:
    - mutations_updated_vk_info_csv                (str) Path to the updated dataframe containing the MCRS headers and sequences. Corresponds to `mutations_updated_csv_out` in the varseek build function. Only needed if the original file was changed or renamed. Default: None (will find it in `input_dir` if it exists).
    - mutations_updated_exploded_vk_info_csv       (str) Path to the updated exploded dataframe containing the MCRS headers and sequences. Corresponds to `mutations_updated_exploded_csv_out` in the varseek build function. Only needed if the original file was changed or renamed. Default: None (will find it in `input_dir` if it exists).
    - id_to_header_csv                             (str) Path to the csv file containing the mapping of IDs to headers generated from varseek build corresponding to mcrs_fasta. Corresponds to `id_to_header_csv_out` in the varseek build function. Only needed if the original file was changed or renamed. Default: None (will find it in `input_dir` if it exists).
    - dlist_fasta                                  (str) Path to the dlist fasta file. Default: None (will find it in `input_dir` if it exists).

    # Optional output file paths: (only needed if changing/customizing file names or locations):
    - mutations_updated_filtered_csv_out           (str) Path to the filtered mutation metadata dataframe. Default: None (will be saved in `out`).
    - mutations_updated_exploded_filtered_csv_out  (str) Path to the filtered exploded mutation metadata dataframe. Default: None (will be saved in `out`).
    - id_to_header_filtered_csv_out                (str) Path to the filtered id to header csv. Default: None (will be saved in `out`).
    - dlist_filtered_fasta_out                     (str) Path to the filtered dlist fasta file. Default: None (will be saved in `out`).
    - mcrs_filtered_fasta_out                      (str) Path to the filtered mcrs fasta file. Default: None (will be saved in `out`).
    - mcrs_t2g_filtered_out                        (str) Path to the filtered t2g file. Default: None (will be saved in `out`).
    - wt_mcrs_filtered_fasta_out                   (str) Path to the filtered wt mcrs fasta file. Default: None (will be saved in `out`).
    - wt_mcrs_t2g_filtered_out                     (str) Path to the filtered t2g file for wt mcrs fasta. Default: None (will be saved in `out`).

    # Returning and saving of optional output
    - save_wt_mcrs_fasta_and_t2g                   (bool) If True, save the filtered wt mcrs fasta and t2g files. Default: False.
    - save_mutations_updated_filtered_csvs         (bool) If True, save the filtered mutation metadata dataframe. Default: False.
    - return_mutations_updated_filtered_csv_df     (bool) If True, return the filtered mutation metadata dataframe. Default: False.

    # General arguments:
    - dry_run                                      (bool) If True, print the parameters and exit without running the function. Default: False.
    - list_filter_rules                            (bool) If True, print the available filter rules and exit without running the function. Default: False.
    - overwrite                                    (bool) If True, overwrite the output files if they already exist. Default: False.
    - verbose                                      (bool) If True, print progress messages. Default: True.

    # Hidden arguments:
    filter_all_dlists (bool) If True, filter all dlists. Default: False.
    dlist_genome_fasta (str) Path to the genome dlist fasta file. Default: None.
    dlist_cdna_fasta (str) Path to the cDNA dlist fasta file. Default: None.
    dlist_genome_filtered_fasta_out (str) Path to the filtered genome dlist fasta file. Default: None.
    dlist_cdna_filtered_fasta_out (str) Path to the filtered cDNA dlist fasta file. Default: None.
    save_mcrs_filtered_fasta_and_t2g (bool) If True, save the filtered mcrs fasta and t2g files. Default: True.
    """
    #* 0. Informational arguments that exit early
    if list_filter_rules:
        print_list_filter_rules()
    
    #* 1. Start timer
    start_time = time.perf_counter()

    #* 2. Type-checking
    params_dict = make_function_parameter_to_value_dict(1)
    validate_input_filter(params_dict)

    #* 3. Dry-run
    if dry_run:
        print_varseek_dry_run(params_dict, function_name="filter")
        return None
    if out is None:
        out = input_dir if input_dir else "."
    
    #* 4. Save params to config file and run info file
    config_file = os.path.join(out, "config", "vk_filter_config.json")
    save_params_to_config_file(params_dict, config_file)

    run_info_file = os.path.join(out, "config", "vk_filter_run_info.txt")
    save_run_info(run_info_file)

    #* 5. Set up default folder/file input paths, and make sure the necessary ones exist
    # have the option to filter other dlists as kwargs
    filter_all_dlists = kwargs.get("filter_all_dlists", False)
    dlist_genome_fasta = kwargs.get("dlist_genome_fasta", None)
    dlist_cdna_fasta = kwargs.get("dlist_cdna_fasta", None)
    dlist_genome_filtered_fasta_out = kwargs.get("dlist_genome_filtered_fasta_out", None)
    dlist_cdna_filtered_fasta_out = kwargs.get("dlist_cdna_filtered_fasta_out", None)
    save_mcrs_filtered_fasta_and_t2g = kwargs.get("save_mcrs_filtered_fasta_and_t2g", True)

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
    if not mutations_updated_vk_info_csv:
        mutations_updated_vk_info_csv = os.path.join(input_dir, "mutation_metadata_df_updated_vk_info.csv")
    if not mutations_updated_exploded_vk_info_csv:
        mutations_updated_exploded_vk_info_csv = os.path.join(input_dir, "mutation_metadata_df_updated_vk_info_exploded.csv")
    if not dlist_fasta:
        dlist_fasta = os.path.join(input_dir, "dlist.fa")
    if not id_to_header_csv:
        id_to_header_csv = os.path.join(input_dir, "id_to_header.csv")

    # set input file names to None if they do not exist
    if not os.path.isfile(mutations_updated_vk_info_csv) and not isinstance(mutations_updated_vk_info_csv, pd.DataFrame):
        raise FileNotFoundError(f"Mutation metadata file not found at {mutations_updated_vk_info_csv}.")
    if not os.path.isfile(mutations_updated_exploded_vk_info_csv):
        logger.warning(f"Exploded mutation metadata file not found at {mutations_updated_exploded_vk_info_csv}. Skipping filtering of exploded mutation metadata.")
        mutations_updated_exploded_vk_info_csv = None
    if not os.path.isfile(dlist_fasta):
        logger.warning(f"d-list file not found at {dlist_fasta}. Skipping filtering of d-list.")
        dlist_fasta = None
    if not os.path.isfile(id_to_header_csv):
        logger.warning(f"ID to header csv file not found at {id_to_header_csv}. Skipping filtering of ID to header csv.")
        id_to_header_csv = None

    #* 6. Set up default folder/file output paths, and make sure they don't exist unless overwrite=True    
    # define output file names if not provided
    if not mutations_updated_filtered_csv_out:  # mutations_updated_vk_info_csv must exist or else an exception will be raised from earlier
        mutations_updated_filtered_csv_out = os.path.join(out, "mutation_metadata_df_filtered.csv")
    if (mutations_updated_exploded_vk_info_csv and os.path.isfile(mutations_updated_exploded_vk_info_csv)) and not mutations_updated_exploded_filtered_csv_out:
        mutations_updated_exploded_filtered_csv_out = os.path.join(out, "mutation_metadata_df_updated_vk_info_exploded_filtered.csv")
    if (id_to_header_csv and os.path.isfile(id_to_header_csv)) and not id_to_header_filtered_csv_out:
        id_to_header_filtered_csv_out = os.path.join(out, "id_to_header_mapping_filtered.csv")    
    if (dlist_fasta and os.path.isfile(dlist_fasta)) and not dlist_filtered_fasta_out:
        dlist_filtered_fasta_out = os.path.join(out, "dlist_filtered.fa")
    if not mcrs_filtered_fasta_out:  # this file must be created
        mcrs_filtered_fasta_out = os.path.join(out, "mcrs_filtered.fa")
    if not mcrs_t2g_filtered_out:    # this file must be created
        mcrs_t2g_filtered_out = os.path.join(out, "mcrs_t2g_filtered.txt")
    if not wt_mcrs_filtered_fasta_out:
        wt_mcrs_filtered_fasta_out = os.path.join(out, "wt_mcrs_filtered.fa")
    if not wt_mcrs_t2g_filtered_out:
        wt_mcrs_t2g_filtered_out = os.path.join(out, "wt_mcrs_t2g_filtered.txt")

    # make sure directories of all output files exist
    output_files = [mutations_updated_filtered_csv_out, mutations_updated_exploded_filtered_csv_out, id_to_header_filtered_csv_out, dlist_filtered_fasta_out, mcrs_filtered_fasta_out, mcrs_t2g_filtered_out, wt_mcrs_filtered_fasta_out, wt_mcrs_t2g_filtered_out]
    for output_file in output_files:
        if os.path.isfile(output_file) and not overwrite:
            raise ValueError(f"Output file '{output_file}' already exists. Set 'overwrite=True' to overwrite it.")
        if os.path.dirname(output_file):
            os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
    #* 7. Define kwargs defaults
    # defined at the beginning of (5) here, as they were needed in that section

    #* 8. Start the actual function
    # filters must either be a dict (as described in docs) or a path to a JSON file
    if isinstance(filters, list) or isinstance(filters, tuple) or isinstance(filters, set) or (isinstance(filters, str) and filters.endswith(".txt")):
        filters = prepare_filters_list(filters)
    elif isinstance(filters, dict):
        pass  # filters is already a dict from argparse
    else:
        raise ValueError(f"Invalid filters: {filters}")

    if isinstance(mutations_updated_vk_info_csv, str):
        mutation_metadata_df = pd.read_csv(mutations_updated_vk_info_csv)
    else:
        mutation_metadata_df = mutations_updated_vk_info_csv

    if "semicolon_count" not in mutation_metadata_df.columns:  # adding for reporting purposes
        mutation_metadata_df["semicolon_count"] = mutation_metadata_df["mcrs_header"].str.count(";")

    filtering_report_text_out = os.path.join(out, "filtering_report.txt")
    filtered_df = apply_filters(mutation_metadata_df, filters, verbose=verbose, logger=logger, filtering_report_text_out=filtering_report_text_out)
    filtered_df = filtered_df.copy()  # here to avoid pandas warning about assigning to a slice rather than a copy

    if "semicolon_count" in mutation_metadata_df.columns:
        mutation_metadata_df = mutation_metadata_df.drop(columns=["semicolon_count"])

    if save_mutations_updated_filtered_csvs:
        filtered_df.to_csv(mutations_updated_filtered_csv_out, index=False)

    # make mcrs_filtered_fasta_out
    filtered_df["mcrs_id"] = filtered_df["mcrs_id"].astype(str)

    filtered_df["fasta_format"] = ">" + filtered_df["mcrs_id"] + "\n" + filtered_df["mcrs_sequence"] + "\n"

    if save_mcrs_filtered_fasta_and_t2g:
        with open(mcrs_filtered_fasta_out, "w") as fasta_file:
            fasta_file.write("".join(filtered_df["fasta_format"].values))

        filtered_df.drop(columns=["fasta_format"], inplace=True)

        # make mcrs_t2g_filtered_out
        create_mutant_t2g(mcrs_filtered_fasta_out, mcrs_t2g_filtered_out)

    # make wt_mcrs_filtered_fasta_out and wt_mcrs_t2g_filtered_out iff save_wt_mcrs_fasta_and_t2g is True
    if save_wt_mcrs_fasta_and_t2g:
        mutations_with_exactly_1_wt_sequence_per_row = filtered_df[["mcrs_id", "wt_sequence"]].copy()

        mutations_with_exactly_1_wt_sequence_per_row["wt_sequence"] = mutations_with_exactly_1_wt_sequence_per_row["wt_sequence"].apply(safe_literal_eval)

        if isinstance(mutations_with_exactly_1_wt_sequence_per_row["wt_sequence"][0], list):  # remove the rows with multiple WT counterparts for 1 MCRS, and convert the list of strings to string
            # Step 1: Filter rows where the length of the set of the list in `wt_sequence` is 1
            mutations_with_exactly_1_wt_sequence_per_row = mutations_with_exactly_1_wt_sequence_per_row[mutations_with_exactly_1_wt_sequence_per_row["wt_sequence"].apply(lambda x: len(set(x)) == 1)]

            # Step 2: Convert the list to a string
            mutations_with_exactly_1_wt_sequence_per_row["wt_sequence"] = mutations_with_exactly_1_wt_sequence_per_row["wt_sequence"].apply(lambda x: x[0])

        mutations_with_exactly_1_wt_sequence_per_row["fasta_format_wt"] = ">" + mutations_with_exactly_1_wt_sequence_per_row["mcrs_id"] + "\n" + mutations_with_exactly_1_wt_sequence_per_row["wt_sequence"] + "\n"

        with open(wt_mcrs_filtered_fasta_out, "w") as fasta_file:
            fasta_file.write("".join(mutations_with_exactly_1_wt_sequence_per_row["fasta_format_wt"].values))

        create_mutant_t2g(wt_mcrs_filtered_fasta_out, wt_mcrs_t2g_filtered_out)

        fasta_summary_stats(mcrs_filtered_fasta_out)

    filtered_df.reset_index(drop=True, inplace=True)

    filtered_df_mcrs_ids = set(filtered_df["mcrs_id"])

    # make mutations_updated_exploded_filtered_csv_out iff mutations_updated_exploded_vk_info_csv exists
    if save_mutations_updated_filtered_csvs and mutations_updated_exploded_vk_info_csv and os.path.isfile(mutations_updated_exploded_vk_info_csv):
        mutation_metadata_df_exploded = pd.read_csv(mutations_updated_exploded_vk_info_csv)

        # Filter mutation_metadata_df_exploded based on these unique values
        filtered_mutation_metadata_df_exploded = mutation_metadata_df_exploded[
            mutation_metadata_df_exploded['mcrs_id'].isin(filtered_df_mcrs_ids)
        ]

        filtered_mutation_metadata_df_exploded.to_csv(mutations_updated_exploded_filtered_csv_out, index=False)

        # Delete the DataFrame from memory
        del filtered_mutation_metadata_df_exploded

    # make dlist_filtered_fasta_out iff dlist_fasta exists
    if dlist_fasta and os.path.isfile(dlist_fasta):
        filter_fasta(dlist_fasta, dlist_filtered_fasta_out, filtered_df_mcrs_ids)
    
    if filter_all_dlists:
        if dlist_genome_fasta and os.path.isfile(dlist_genome_fasta):
            filter_fasta(dlist_genome_fasta, dlist_genome_filtered_fasta_out, filtered_df_mcrs_ids)
        if dlist_cdna_fasta and os.path.isfile(dlist_cdna_fasta):
            filter_fasta(dlist_cdna_fasta, dlist_cdna_filtered_fasta_out, filtered_df_mcrs_ids)

    # make id_to_header_filtered_csv_out iff id_to_header_csv exists
    if id_to_header_csv and os.path.isfile(id_to_header_csv):
        filter_id_to_header_csv(id_to_header_csv, id_to_header_filtered_csv_out, filtered_df_mcrs_ids)

    if verbose:
        logger.info(f"Output fasta file with filtered mutations: {mcrs_filtered_fasta_out}")
        logger.info(f"t2g file containing mutated sequences created at {mcrs_t2g_filtered_out}.")
        if dlist_filtered_fasta_out and os.path.isfile(dlist_filtered_fasta_out):
            logger.info(f"Filtered dlist fasta created at {dlist_filtered_fasta_out}.")

    # Report time
    report_time_elapsed(start_time, logger=logger, verbose=verbose)

    if return_mutations_updated_filtered_csv_df:
        return filtered_df

