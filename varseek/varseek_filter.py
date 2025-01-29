import os
import csv
import time
import numpy as np
import pandas as pd
from tqdm import tqdm
from pdb import set_trace as st
from .utils import (
    set_up_logger,
    apply_filters,
    filter_fasta,
    prepare_filters_json,
    prepare_filters_list,
    create_mutant_t2g,
    filter_id_to_header_csv,
    fasta_summary_stats,
    safe_literal_eval,
    save_params_to_config_file,
    make_function_parameter_to_value_dict,
    check_file_path_is_string_with_valid_extension,
    print_varseek_dry_run,
    report_time_elapsed
)

logger = set_up_logger()

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

def validate_input_filter(
    input_dir,
    filters,
    out,
    mutations_updated_vk_info_csv,
    mutations_updated_exploded_vk_info_csv,
    id_to_header_csv,
    dlist_fasta,
    mutations_updated_filtered_csv_out,
    mutations_updated_exploded_filtered_csv_out,
    id_to_header_filtered_csv_out,
    dlist_filtered_fasta_out,
    mcrs_filtered_fasta_out,
    mcrs_t2g_filtered_out,
    wt_mcrs_filtered_fasta_out,
    wt_mcrs_t2g_filtered_out,
    save_wt_mcrs_fasta_and_t2g,
    save_mutations_updated_filtered_csvs,
    return_mutations_updated_filtered_csv_df,
    verbose,
    **kwargs
):
    
    # Type-checking for paths
    if not isinstance(input_dir, str) or not os.path.isdir(input_dir):
        raise ValueError(f"Invalid input directory: {input_dir}")
    if not isinstance(out, str) or not os.path.isdir(out):
        raise ValueError(f"Invalid input directory: {out}")
    
    check_file_path_is_string_with_valid_extension(mutations_updated_vk_info_csv, "mutations_updated_vk_info_csv", "csv")
    check_file_path_is_string_with_valid_extension(mutations_updated_exploded_vk_info_csv, "mutations_updated_exploded_vk_info_csv", "csv")
    check_file_path_is_string_with_valid_extension(id_to_header_csv, "id_to_header_csv", "csv")
    check_file_path_is_string_with_valid_extension(dlist_fasta, "dlist_fasta", "fasta")
    check_file_path_is_string_with_valid_extension(mutations_updated_filtered_csv_out, "mutations_updated_filtered_csv_out", "csv")
    check_file_path_is_string_with_valid_extension(mutations_updated_exploded_filtered_csv_out, "mutations_updated_exploded_filtered_csv_out", "csv")
    check_file_path_is_string_with_valid_extension(id_to_header_filtered_csv_out, "id_to_header_filtered_csv_out", "csv")
    check_file_path_is_string_with_valid_extension(dlist_filtered_fasta_out, "dlist_filtered_fasta_out", "fasta")
    check_file_path_is_string_with_valid_extension(mcrs_filtered_fasta_out, "mcrs_filtered_fasta_out", "fasta")
    check_file_path_is_string_with_valid_extension(mcrs_t2g_filtered_out, "mcrs_t2g_filtered_out", "t2g")
    check_file_path_is_string_with_valid_extension(wt_mcrs_filtered_fasta_out, "wt_mcrs_filtered_fasta_out", "fasta")
    check_file_path_is_string_with_valid_extension(wt_mcrs_t2g_filtered_out, "wt_mcrs_t2g_filtered_out", "t2g")

    # Validate boolean parameters
    for param_name, param_value in {
        "save_wt_mcrs_fasta_and_t2g": save_wt_mcrs_fasta_and_t2g,
        "save_mutations_updated_filtered_csvs": save_mutations_updated_filtered_csvs,
        "return_mutations_updated_filtered_csv_df": return_mutations_updated_filtered_csv_df,
        "verbose": verbose,
    }.items():
        if not isinstance(param_value, bool):
            raise ValueError(f"{param_name} must be a boolean. Got {type(param_value)}.")



def filter(
    input_dir,
    filters,
    out=".",
    mutations_updated_vk_info_csv=None,  # input mutation metadata df
    mutations_updated_exploded_vk_info_csv=None,  # input exploded mutation metadata df
    id_to_header_csv=None,  # input id to header csv
    dlist_fasta=None,  # input dlist
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
    verbose=True,
    dry_run=False,
    overwrite=False,
    **kwargs,
):
    """
    Filter mutations based on the provided filters and save the filtered mutations to a fasta file.

    Required input arguments:
    - mutations_updated_vk_info_csv     (str) Path to the csv file containing the mutation metadata dataframe.

    Additional input arguments:
    - filters                       (dict) Dictionary containing the filters to apply to the mutation metadata dataframe.
    - mcrs_filtered_fasta_out                  (str) Path to the output fasta file containing the filtered mutations.
    - mutations_updated_filtered_csv_out            (str) Path to the output csv file containing the filtered mutation metadata dataframe.
    - dlist_fasta                   (str) Path to the dlist fasta file.
    - mcrs_t2g_filtered_out                    (str) Path to the output t2g file.
    - id_to_header_csv              (str) Path to the id to header csv file.
    - verbose                       (bool) Whether to print the logs or not.
    """
    #* 1. Start timer
    start_time = time.perf_counter()

    #* 2. Type-checking
    params_dict = make_function_parameter_to_value_dict(1)
    validate_input_filter(**params_dict)

    #* 3. Dry-run
    if dry_run:
        print_varseek_dry_run(params_dict, function_name="filter")
        return None
    
    #* 4. Save params to config file
    config_file = os.path.join(out, "config", "vk_filter_config.json")
    save_params_to_config_file(config_file)

    #* 5. Set up default folder/file input paths, and make sure the necessary ones exist
    os.makedirs(out, exist_ok=True)

    # have the option to filter other dlists as kwargs
    filter_all_dlists = kwargs.get("filter_all_dlists", False)
    dlist_genome_fasta = kwargs.get("dlist_genome_fasta", None)
    dlist_cdna_fasta = kwargs.get("dlist_cdna_fasta", None)
    dlist_genome_filtered_fasta_out = kwargs.get("dlist_genome_filtered_fasta_out", None)
    dlist_cdna_filtered_fasta_out = kwargs.get("dlist_cdna_filtered_fasta_out", None)

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
    if not os.path.isfile(mutations_updated_vk_info_csv):
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
        
    #* 7. Start the actual function
    # filters must either be a dict (as described in docs) or a path to a JSON file
    if type(filters) == str and filters.endswith(".json"):
        filters = prepare_filters_json(filters)
    elif type(filters) == list or (type(filters) == str and filters.endswith(".txt")):
        filters = prepare_filters_list(filters)
    else:
        pass  # filters is already a dict from argparse

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

    filtered_df.reset_index(drop=True, inplace=True)

    fasta_summary_stats(mcrs_filtered_fasta_out)

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
    report_time_elapsed(start_time)

    if return_mutations_updated_filtered_csv_df:
        return filtered_df

