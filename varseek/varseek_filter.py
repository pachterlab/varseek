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
    return_mutations_updated_filtered_csv_df=True,
    verbose=True,
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
    config_file = os.path.join(out, "config", "vk_filter_config.json")
    save_params_to_config_file(config_file)
    
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
    if not os.path.exists(mutations_updated_vk_info_csv):
        raise FileNotFoundError(f"Mutation metadata file not found at {mutations_updated_vk_info_csv}.")
    if not os.path.exists(mutations_updated_exploded_vk_info_csv):
        logger.warning(f"Exploded mutation metadata file not found at {mutations_updated_exploded_vk_info_csv}. Skipping filtering of exploded mutation metadata.")
        mutations_updated_exploded_vk_info_csv = None
    if not os.path.exists(dlist_fasta):
        logger.warning(f"d-list file not found at {dlist_fasta}. Skipping filtering of d-list.")
        dlist_fasta = None
    if not os.path.exists(id_to_header_csv):
        logger.warning(f"ID to header csv file not found at {id_to_header_csv}. Skipping filtering of ID to header csv.")
        id_to_header_csv = None

    # define output file names if not provided
    if not mutations_updated_filtered_csv_out:  # mutations_updated_vk_info_csv must exist or else an exception will be raised from earlier
        mutations_updated_filtered_csv_out = os.path.join(out, "mutation_metadata_df_filtered.csv")
    if (mutations_updated_exploded_vk_info_csv and os.path.exists(mutations_updated_exploded_vk_info_csv)) and not mutations_updated_exploded_filtered_csv_out:
        mutations_updated_exploded_filtered_csv_out = os.path.join(out, "mutation_metadata_df_updated_vk_info_exploded_filtered.csv")
    if (id_to_header_csv and os.path.exists(id_to_header_csv)) and not id_to_header_filtered_csv_out:
        id_to_header_filtered_csv_out = os.path.join(out, "id_to_header_mapping_filtered.csv")    
    if (dlist_fasta and os.path.exists(dlist_fasta)) and not dlist_filtered_fasta_out:
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
        if output_file and os.path.dirname(output_file):
            os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
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

    if verbose:
        number_of_mutations_initial = len(mutation_metadata_df)
        # TODO: if I want number of mutations (in addition to number of MCRSs), then calculate the length of the df exploded by header
        logger.info(f"Initial number of mutations: {number_of_mutations_initial}")

    filtered_df = apply_filters(mutation_metadata_df, filters, verbose=verbose, logger=logger)
    filtered_df = filtered_df.copy()  # here to avoid pandas warning about assigning to a slice rather than a copy

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
    if mutations_updated_exploded_vk_info_csv and os.path.exists(mutations_updated_exploded_vk_info_csv):
        mutation_metadata_df_exploded = pd.read_csv(mutations_updated_exploded_vk_info_csv)

        # Filter mutation_metadata_df_exploded based on these unique values
        filtered_mutation_metadata_df_exploded = mutation_metadata_df_exploded[
            mutation_metadata_df_exploded['mcrs_id'].isin(filtered_df_mcrs_ids)
        ]

        filtered_mutation_metadata_df_exploded.to_csv(mutations_updated_exploded_filtered_csv_out, index=False)

        # Delete the DataFrame from memory
        del filtered_mutation_metadata_df_exploded

    # make dlist_filtered_fasta_out iff dlist_fasta exists
    if dlist_fasta and os.path.exists(dlist_fasta):
        filter_fasta(dlist_fasta, dlist_filtered_fasta_out, filtered_df_mcrs_ids)
    
    if filter_all_dlists:
        if dlist_genome_fasta and os.path.exists(dlist_genome_fasta):
            filter_fasta(dlist_genome_fasta, dlist_genome_filtered_fasta_out, filtered_df_mcrs_ids)
        if dlist_cdna_fasta and os.path.exists(dlist_cdna_fasta):
            filter_fasta(dlist_cdna_fasta, dlist_cdna_filtered_fasta_out, filtered_df_mcrs_ids)

    # make id_to_header_filtered_csv_out iff id_to_header_csv exists
    if id_to_header_csv and os.path.exists(id_to_header_csv):
        filter_id_to_header_csv(id_to_header_csv, id_to_header_filtered_csv_out, filtered_df_mcrs_ids)

    if verbose:
        logger.info(f"Output fasta file with filtered mutations: {mcrs_filtered_fasta_out}")
        logger.info(f"t2g file containing mutated sequences created at {mcrs_t2g_filtered_out}.")
        if dlist_filtered_fasta_out and os.path.exists(dlist_filtered_fasta_out):
            logger.info(f"Filtered dlist fasta created at {dlist_filtered_fasta_out}.")

    if return_mutations_updated_filtered_csv_df:
        return filtered_df
    else:
        return None
