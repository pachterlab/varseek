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
)

logger = set_up_logger()


def filter(
    mutation_metadata_df_path,
    filters,
    output_mcrs_fasta="output_mcrs_fasta.fa",
    output_metadata_df=None,
    dlist_fasta=None,
    output_dlist_fasta=None,
    output_wt_mcrs_fa=None,
    create_t2g=False,
    output_t2g=None,
    output_t2g_wt=None,
    id_to_header_csv=None,
    output_id_to_header_csv=None,
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
    - create_t2g                    (bool) Whether to create the t2g file or not.
    - output_t2g                    (str) Path to the output t2g file.
    - id_to_header_csv              (str) Path to the id to header csv file.
    - verbose                       (bool) Whether to print the logs or not.
    """
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

    if output_metadata_df is not None:
        filtered_df.to_csv(output_metadata_df, index=False)

    filtered_df["mcrs_id"] = filtered_df["mcrs_id"].astype(str)

    if type(output_mcrs_fasta) == str:
        filtered_df["fasta_format"] = ">" + filtered_df["mcrs_id"] + "\n" + filtered_df["mcrs_sequence"] + "\n"

        with open(output_mcrs_fasta, "w") as fasta_file:
            fasta_file.write("".join(filtered_df["fasta_format"].values))

        filtered_df.drop(columns=["fasta_format"], inplace=True)

    if type(output_wt_mcrs_fa) == str:
        mutations_with_exactly_1_wt_sequence_per_row = filtered_df[["mcrs_id", "wt_sequence"]].copy()

        if type(filtered_df["wt_sequence"][0] == list):  # remove the rows with multiple WT counterparts for 1 MCRS, and convert the list of strings to string
            # Step 1: Filter rows where the length of the set of the list in `wt_sequence` is 1
            mutations_with_exactly_1_wt_sequence_per_row = mutations_with_exactly_1_wt_sequence_per_row[mutations_with_exactly_1_wt_sequence_per_row["wt_sequence"].apply(lambda x: len(set(x)) == 1)]

            # Step 2: Convert the list to a string
            mutations_with_exactly_1_wt_sequence_per_row["wt_sequence"] = mutations_with_exactly_1_wt_sequence_per_row["wt_sequence"].apply(lambda x: x[0])

        mutations_with_exactly_1_wt_sequence_per_row["fasta_format_wt"] = ">" + mutations_with_exactly_1_wt_sequence_per_row["mcrs_id"] + "\n" + mutations_with_exactly_1_wt_sequence_per_row["wt_sequence"] + "\n"

        with open(output_wt_mcrs_fa, "w") as fasta_file:
            fasta_file.write("".join(mutations_with_exactly_1_wt_sequence_per_row["wt_fasta_format"].values))

    filtered_df.reset_index(drop=True, inplace=True)

    fasta_summary_stats(output_mcrs_fasta)

    filtered_df_mcrs_ids = set(filtered_df["mcrs_id"])

    # if mcrs_fa is not None:
    #     filter_fasta(mcrs_fa, output_mcrs_fasta, filtered_df_mcrs_ids)

    if dlist_fasta is not None:
        if output_dlist_fasta is None:
            output_dlist_fasta = dlist_fasta.replace(".fa", "_filtered.fa")
        filter_fasta(dlist_fasta, output_dlist_fasta, filtered_df_mcrs_ids)

    if id_to_header_csv is not None:
        if output_id_to_header_csv is None:
            output_id_to_header_csv = id_to_header_csv.replace(".csv", "_filtered.csv")
        filter_id_to_header_csv(id_to_header_csv, output_id_to_header_csv, filtered_df_mcrs_ids)

    if create_t2g:
        if output_t2g is None:
            output_t2g = output_mcrs_fasta.replace(".fa", "_t2g.txt")

        create_mutant_t2g(output_mcrs_fasta, output_t2g)

        if output_t2g_wt is None:
            output_t2g_wt = output_wt_mcrs_fa.replace(".fa", "_t2g.txt")

        create_mutant_t2g(output_wt_mcrs_fa, output_t2g_wt)

    if verbose:
        logger.info(f"Output fasta file with filtered mutations: {output_mcrs_fasta}")
        logger.info(f"t2g file containing mutated sequences created at {output_t2g}.")

    if return_df:
        return filtered_df
