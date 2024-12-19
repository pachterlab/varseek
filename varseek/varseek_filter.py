import os
import csv
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
    save_params_to_config_file
)

logger = set_up_logger()


def filter(
    mutation_metadata_df_path,  # input mutation metadata df
    filters,
    input_dir=None,
    out=".",
    mutation_metadata_df_exploded_path=None,  # input exploded mutation metadata df
    id_to_header_csv=None,  # input id to header csv
    dlist_fasta=None,  # input dlist
    output_metadata_df=None,  # output metadata df
    output_mutation_metadata_df_exploded=None,  # output exploded mutation metadata df
    output_id_to_header_csv=None,  # output id to header csv
    output_dlist_fasta="dlist_filtered.fa",  # output dlist fasta
    output_mcrs_fasta="mcrs_filtered.fa",  # output mcrs fasta
    output_t2g="t2g_filtered",  # output t2g
    output_wt_mcrs_fa=None,  # output wt mcrs fasta
    output_t2g_wt=None,  # output t2g for wt mcrs fasta
    make_wt_mcrs_files=False,
    return_df=True,
    verbose=True,
    **kwargs,
):
    """
    Filter mutations based on the provided filters and save the filtered mutations to a fasta file.

    Required input arguments:
    - mutation_metadata_df_path     (str) Path to the csv file containing the mutation metadata dataframe.

    Additional input arguments:
    - filters                       (dict) Dictionary containing the filters to apply to the mutation metadata dataframe.
    - output_mcrs_fasta                  (str) Path to the output fasta file containing the filtered mutations.
    - output_metadata_df            (str) Path to the output csv file containing the filtered mutation metadata dataframe.
    - dlist_fasta                   (str) Path to the dlist fasta file.
    - output_t2g                    (str) Path to the output t2g file.
    - id_to_header_csv              (str) Path to the id to header csv file.
    - verbose                       (bool) Whether to print the logs or not.
    """
    config_file = os.path.join(out, "config", "vk_filter_config.json")
    save_params_to_config_file(config_file)
    
    os.makedirs(out, exist_ok=True)
    
    # define input file names if not provided
    if not mutation_metadata_df_path:
        mutation_metadata_df_path = os.path.join(input_dir, "mutation_metadata_df_updated_vk_info.csv")
    if not mutation_metadata_df_exploded_path:
        mutation_metadata_df_exploded_path = os.path.join(input_dir, "mutation_metadata_df_updated_vk_info_exploded.csv")
    if not dlist_fasta:
        dlist_fasta = os.path.join(input_dir, "dlist.fa")
    if not id_to_header_csv:
        id_to_header_csv = os.path.join(input_dir, "id_to_header.csv")

    # set input file names to None if they do not exist
    if not os.path.exists(mutation_metadata_df_path):
        raise FileNotFoundError(f"Mutation metadata file not found at {mutation_metadata_df_path}.")
    if not os.path.exists(mutation_metadata_df_exploded_path):
        logger.warning(f"Exploded mutation metadata file not found at {mutation_metadata_df_exploded_path}. Skipping filtering of exploded mutation metadata.")
        mutation_metadata_df_exploded_path = None
    if not os.path.exists(dlist_fasta):
        logger.warning(f"Exploded mutation metadata file not found at {dlist_fasta}. Skipping filtering of exploded mutation metadata.")
        dlist_fasta = None
    if not os.path.exists(id_to_header_csv):
        logger.warning(f"Exploded mutation metadata file not found at {id_to_header_csv}. Skipping filtering of exploded mutation metadata.")
        id_to_header_csv = None

    # define output file names if not provided
    if not output_metadata_df:  # mutation_metadata_df_path must exist or else an exception will be raised from earlier
        output_metadata_df = os.path.join(out, "mutation_metadata_df_filtered.csv")
    if (mutation_metadata_df_exploded_path and os.path.exists(mutation_metadata_df_exploded_path)) and not output_mutation_metadata_df_exploded:
        output_mutation_metadata_df_exploded = os.path.join(out, "mutation_metadata_df_updated_vk_info_exploded_filtered.csv")
    if (id_to_header_csv and os.path.exists(id_to_header_csv)) and not output_id_to_header_csv:
        output_id_to_header_csv = os.path.join(out, "id_to_header_mapping_filtered.csv")
    if (dlist_fasta and os.path.exists(dlist_fasta)) and not output_dlist_fasta:
        output_dlist_fasta = os.path.join(out, "dlist_filtered.fa")
    if not output_mcrs_fasta:  # this file must be created
        output_mcrs_fasta = os.path.join(out, "mcrs_filtered.fa")
    if not output_t2g:    # this file must be created
        output_t2g = os.path.join(out, "mcrs_t2g_filtered.txt")
    if make_wt_mcrs_files:
        if not output_wt_mcrs_fa:
            output_wt_mcrs_fa = os.path.join(out, "wt_mcrs_filtered.fa")
        if not output_t2g_wt:
            output_t2g_wt = os.path.join(out, "wt_mcrs_t2g_filtered.txt")
    else:
        output_wt_mcrs_fa = None
        output_t2g_wt = None

    # make sure directories of all output files exist
    output_files = [output_metadata_df, output_mutation_metadata_df_exploded, output_id_to_header_csv, output_dlist_fasta, output_mcrs_fasta, output_t2g, output_wt_mcrs_fa, output_t2g_wt]
    for output_file in output_files:
        if output_file and os.path.dirname(output_file):
            os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
    # filters must either be a dict (as described in docs) or a path to a JSON file
    if type(filters) == str and filters.endswith(".json"):
        filters = prepare_filters_json(filters)
    elif type(filters) == list or (type(filters) == str and filters.endswith(".txt")):
        filters = prepare_filters_list(filters)
    else:
        pass  # filters is already a dict from argparse

    if type(mutation_metadata_df_path) == str:
        mutation_metadata_df = pd.read_csv(mutation_metadata_df_path)
    else:
        mutation_metadata_df = mutation_metadata_df_path

    if verbose:
        number_of_mutations_initial = len(mutation_metadata_df)
        # TODO: if I want number of mutations (in addition to number of MCRSs), then calculate the length of the df exploded by header
        logger.info(f"Initial number of mutations: {number_of_mutations_initial}")

    filtered_df = apply_filters(mutation_metadata_df, filters, verbose=verbose, logger=logger)
    filtered_df = filtered_df.copy()  # here to avoid pandas warning about assigning to a slice rather than a copy

    filtered_df.to_csv(output_metadata_df, index=False)

    # make output_mcrs_fasta
    filtered_df["mcrs_id"] = filtered_df["mcrs_id"].astype(str)

    filtered_df["fasta_format"] = ">" + filtered_df["mcrs_id"] + "\n" + filtered_df["mcrs_sequence"] + "\n"

    with open(output_mcrs_fasta, "w") as fasta_file:
        fasta_file.write("".join(filtered_df["fasta_format"].values))

    filtered_df.drop(columns=["fasta_format"], inplace=True)

    # make output_t2g
    create_mutant_t2g(output_mcrs_fasta, output_t2g)

    # make output_wt_mcrs_fa and output_wt_mcrs_fa iff make_wt_mcrs_files is True
    if make_wt_mcrs_files:
        mutations_with_exactly_1_wt_sequence_per_row = filtered_df[["mcrs_id", "wt_sequence"]].copy()

        mutations_with_exactly_1_wt_sequence_per_row["wt_sequence"] = mutations_with_exactly_1_wt_sequence_per_row["wt_sequence"].apply(safe_literal_eval)

        if isinstance(mutations_with_exactly_1_wt_sequence_per_row["wt_sequence"][0], list):  # remove the rows with multiple WT counterparts for 1 MCRS, and convert the list of strings to string
            # Step 1: Filter rows where the length of the set of the list in `wt_sequence` is 1
            mutations_with_exactly_1_wt_sequence_per_row = mutations_with_exactly_1_wt_sequence_per_row[mutations_with_exactly_1_wt_sequence_per_row["wt_sequence"].apply(lambda x: len(set(x)) == 1)]

            # Step 2: Convert the list to a string
            mutations_with_exactly_1_wt_sequence_per_row["wt_sequence"] = mutations_with_exactly_1_wt_sequence_per_row["wt_sequence"].apply(lambda x: x[0])

        mutations_with_exactly_1_wt_sequence_per_row["fasta_format_wt"] = ">" + mutations_with_exactly_1_wt_sequence_per_row["mcrs_id"] + "\n" + mutations_with_exactly_1_wt_sequence_per_row["wt_sequence"] + "\n"

        with open(output_wt_mcrs_fa, "w") as fasta_file:
            fasta_file.write("".join(mutations_with_exactly_1_wt_sequence_per_row["fasta_format_wt"].values))

        create_mutant_t2g(output_wt_mcrs_fa, output_t2g_wt)

    filtered_df.reset_index(drop=True, inplace=True)

    fasta_summary_stats(output_mcrs_fasta)

    filtered_df_mcrs_ids = set(filtered_df["mcrs_id"])

    # make output_mutation_metadata_df_exploded iff mutation_metadata_df_exploded_path exists
    if mutation_metadata_df_exploded_path and os.path.exists(mutation_metadata_df_exploded_path):
        mutation_metadata_df_exploded = pd.read_csv(mutation_metadata_df_exploded_path)

        # Filter mutation_metadata_df_exploded based on these unique values
        filtered_mutation_metadata_df_exploded = mutation_metadata_df_exploded[
            mutation_metadata_df_exploded['mcrs_id'].isin(filtered_df_mcrs_ids)
        ]

        filtered_mutation_metadata_df_exploded.to_csv(output_mutation_metadata_df_exploded, index=False)

        # Delete the DataFrame from memory
        del filtered_mutation_metadata_df_exploded

    # make output_id_to_header_csv iff id_to_header_csv exists
    if dlist_fasta and os.path.exists(dlist_fasta):
        filter_fasta(dlist_fasta, output_dlist_fasta, filtered_df_mcrs_ids)

    # make output_id_to_header_csv iff id_to_header_csv exists
    if id_to_header_csv and os.path.exists(id_to_header_csv):
        filter_id_to_header_csv(id_to_header_csv, output_id_to_header_csv, filtered_df_mcrs_ids)

    if verbose:
        logger.info(f"Output fasta file with filtered mutations: {output_mcrs_fasta}")
        logger.info(f"t2g file containing mutated sequences created at {output_t2g}.")
        if output_dlist_fasta and os.path.exists(output_dlist_fasta):
            logger.info(f"Filtered dlist fasta created at {output_dlist_fasta}.")

    if return_df:
        return filtered_df
    else:
        return None
