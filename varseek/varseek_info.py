# CELL
import os
import subprocess
import re
import numpy as np
import pandas as pd
from tqdm import tqdm
import ast
from varseek.varseek_build import reverse_complement

from varseek.constants import (
    supported_databases_and_corresponding_reference_sequence_type,
    mutation_pattern,
)
from varseek.utils import (
    set_up_logger,
    read_fasta,
    swap_ids_for_headers_in_fasta,
    make_mapping_dict,
    compare_cdna_and_genome,
    longest_homopolymer,
    triplet_stats,
    get_mcrss_that_pseudoalign_but_arent_dlisted,
    create_df_of_mcrs_to_self_headers,
    get_df_overlap,
    plot_histogram_of_nearby_mutations_7_5,
    compute_distance_to_closest_splice_junction,
    add_mcrs_mutation_type,
    plot_kat_histogram,
    explode_df,
    collapse_df,
    fasta_summary_stats,
    calculate_total_gene_info,
    calculate_nearby_mutations,
    align_to_normal_genome_and_build_dlist,
    safe_literal_eval,
    save_params_to_config_file
)

tqdm.pandas()
logger = set_up_logger()
pd.set_option("display.max_columns", None)


def add_some_mutation_information_when_cdna_and_genome_combined(df, columns_to_change):
    for column in columns_to_change:
        # Create new columns
        df[f"{column}_cdna"] = None
        df[f"{column}_genome"] = None
        df.loc[df["source"] == "cdna", f"{column}_cdna"] = df.loc[df["source"] == "cdna", column]
        df.loc[df["source"] == "genome", f"{column}_genome"] = df.loc[df["source"] == "genome", column]

    # Create a helper DataFrame by grouping based on 'header_cdna'
    grouped = df.groupby("header_cdna")

    for id_val, group in grouped:
        for column in columns_to_change:
            # Find the cdna_info from the 'cdna' row for this group
            cdna_info_value = group.loc[group["source"] == "cdna", f"{column}_cdna"].values
            genome_info_value = group.loc[group["source"] == "genome", f"{column}_genome"].values

            # If there's a cdna_info, update the genome row with it
            if len(cdna_info_value) > 0 and len(genome_info_value) > 0:
                df.loc[
                    (df["header_cdna"] == id_val) & (df["source"] == "genome"),
                    f"{column}_cdna",
                ] = cdna_info_value[0]
                df.loc[
                    (df["header_cdna"] == id_val) & (df["source"] == "cdna"),
                    f"{column}_genome",
                ] = genome_info_value[0]

    return df


def add_mutation_information(mutation_metadata_df, mutation_column="mutation", mcrs_source="cdna"):
    mutation_metadata_df[["nucleotide_positions", "actual_mutation"]] = mutation_metadata_df[mutation_column].str.extract(mutation_pattern)

    split_positions = mutation_metadata_df["nucleotide_positions"].str.split("_", expand=True)
    mutation_metadata_df[f"start_mutation_position"] = split_positions[0]

    if split_positions.shape[1] > 1:
        mutation_metadata_df[f"end_mutation_position"] = split_positions[1].fillna(split_positions[0])
    else:
        mutation_metadata_df[f"end_mutation_position"] = mutation_metadata_df["start_mutation_position"]

    mutation_metadata_df[["start_mutation_position", "end_mutation_position"]] = mutation_metadata_df[["start_mutation_position", "end_mutation_position"]].astype("Int64")

    if mcrs_source is not None:
        mutation_metadata_df[f"nucleotide_positions_{mcrs_source}"] = mutation_metadata_df[f"nucleotide_positions"]
        mutation_metadata_df[f"actual_mutation_{mcrs_source}"] = mutation_metadata_df[f"actual_mutation"]
        mutation_metadata_df[f"start_mutation_position_{mcrs_source}"] = mutation_metadata_df[f"start_mutation_position"]
        mutation_metadata_df[f"end_mutation_position_{mcrs_source}"] = mutation_metadata_df[f"end_mutation_position"]

    return mutation_metadata_df


def info(
    input_dir,
    mcrs_fasta=None,
    mutations_updated_csv=None,
    id_to_header_csv=None,  # if none then assume no swapping occurred
    columns_to_include="all",
    mcrs_id_column="mcrs_id",
    mcrs_sequence_column="mutant_sequence",
    mcrs_source_column="mcrs_source",  # if input df has concatenated cdna and header MCRS's, then I want to know whether it came from cdna or genome
    seqid_cdna_column="seq_ID",  # if input df has concatenated cdna and header MCRS's, then I want a way of mapping from cdna to genome  # TODO: implement these 4 column name arguments
    seqid_genome_column="chromosome",  # if input df has concatenated cdna and header MCRS's, then I want a way of mapping from cdna to genome
    mutation_cdna_column="mutation",  # if input df has concatenated cdna and header MCRS's, then I want a way of mapping from cdna to genome
    mutation_genome_column="mutation_genome",  # if input df has concatenated cdna and header MCRS's, then I want a way of mapping from cdna to genome
    gtf=None,  # for distance to nearest splice junction
    out=".",
    reference_out_dir=None,
    mutations_updated_vk_info_csv_out = None,
    mutations_updated_exploded_vk_info_csv_out = None,
    dlist_genome_fasta_out = None,
    dlist_cdna_fasta_out = None,
    dlist_combined_fasta_out = None,
    dlist_reference_source="ensembl_grch37_release93",
    w=30,
    max_ambiguous_mcrs=None,
    max_ambiguous_reference=None,
    strandedness=False,
    near_splice_junction_threshold=10,
    threads=2,
    reference_cdna_fasta=None,
    reference_genome_fasta=None,
    mutations_csv=None,
    save_mutations_updated_exploded_vk_info_csv=False,
    verbose=True,
    **kwargs,
):
    """
    Takes in an MCRS fasta file generated from varseek build, and returns a data frame with additional columns containing information about the mutations in the following format:

    Required input arguments:
    - mcrs_fasta     (str) Path to the fasta file containing the MCRS sequences generated by varseek build.

    Additional input arguments:
    - columns_to_include                 (list or str) List of columns to include in the output dataframe. Default: "all".
    - mutations_updated_csv                         (str) Path to the updated dataframe containing the MCRS headers and sequences. Default: None.
    - id_to_header_csv                   (str) Path to the csv file containing the mapping of IDs to headers generated from varseek build corresponding to mcrs_fa. Default: None.
    - mcrs_id_column                     (str) Name of the column containing the MCRS IDs. Default: 'mcrs_id'.
    - mcrs_sequence_column               (str) Name of the column containing the MCRS sequences. Default: 'mutant_sequence'.
    - mcrs_source_column                 (str) Name of the column containing the source of the MCRS (cdna or genome). Default: 'mcrs_source'.
    - seqid_cdna_column                  (str) Name of the column containing the cDNA sequence IDs. Default: 'seq_ID'.
    - seqid_genome_column                (str) Name of the column containing the genome sequence IDs. Default: 'chromosome'.
    - mutation_cdna_column               (str) Name of the column containing the cDNA mutations. Default: 'mutation'.
    - mutation_genome_column             (str) Name of the column containing the genome mutations. Default: 'mutation_genome'.
    - gtf                                (str) Path to the GTF file containing the gene annotations for the genome sequences. Default: None.
    - out                   (str) Path to the directory where the output files will be saved. Default: '.'.
    - reference_out_dir                  (str) Path to the directory where the reference files will be saved. Default: '.'.
    - dlist_reference_source             (str) Source of the reference sequences for the d-list. Currently supported: ensembl_grchNUMBER_releaseNUMBER or t2t. Default: 'ensembl_grch37_release93'.
    - w                                  (int) Length of the flanking regions to be optimized. Default: 30.
    - max_ambiguous                      (int) Maximum number of 'N' characters allowed in the matching d-list entry. Default: None (no 'N' filter will be applied)
    - strandedness                       (bool) Whether to consider MCRSs as stranded when aligning to the human reference and comparing MCRS k-mers to each other. strandedness True corresponds to treating forward and reverse-complement as distinct; False corresponds to treating them as the same. Should match the varseek build merge_identical_rc argument. Default: False.
    - near_splice_junction_threshold     (int) Maximum distance from a splice junction to be considered "near" a splice junction. Default: 10.
    - threads                            (int) Number of threads to use for bowtie2 and bowtie2-build. Default: 2.
    - reference_cdna_fasta               (str) Path to the cDNA reference fasta file. Default: None.
    - reference_genome_fasta             (str) Path to the genome reference fasta file. Default: None.
    - mutations_csv                      (str) Path to the csv file containing the mutations. Default: None.
    - save_exploded_df                   (bool) Whether to save the exploded dataframe. Default: False.
    - verbose                            (bool) Whether to print verbose output. Default: False.

    Part of kwargs:
    - bowtie_path                        (str) Path to the directory containing the bowtie2 and bowtie2-build executables. Default: None.
    """

    # CELL
    config_file = os.path.join(out, "config", "vk_info_config.json")
    save_params_to_config_file(config_file)    
    
    columns_to_explode = ["header", "order"]
    columns_not_successfully_added = []

    if not mcrs_fasta:
        mcrs_fasta = os.path.join(input_dir, "mcrs.fa")
    if not os.path.exists(mcrs_fasta):
        raise FileNotFoundError(f"File not found: {mcrs_fasta}")
    
    if not mutations_updated_csv:
        mutations_updated_csv = os.path.join(input_dir, "mutation_metadata_df.csv")
    if not os.path.exists(mutations_updated_csv):
        logger.warning(f"File not found: {mutations_updated_csv}")
        mutations_updated_csv = None
    
    if not id_to_header_csv:
        id_to_header_csv = os.path.join(input_dir, "id_to_header_mapping.csv")
    if not os.path.exists(id_to_header_csv):
        logger.warning(f"File not found: {id_to_header_csv}")
        id_to_header_csv = None

    if not reference_out_dir:
        reference_out_dir = os.path.join(out, "reference")

    os.makedirs(out, exist_ok=True)
    os.makedirs(reference_out_dir, exist_ok=True)

    if not mutations_updated_vk_info_csv_out:
        mutations_updated_vk_info_csv_out = os.path.join(out, "mutation_metadata_df_updated_vk_info.csv")
    if not mutations_updated_exploded_vk_info_csv_out:
        mutations_updated_exploded_vk_info_csv_out = os.path.join(out, "mutation_metadata_df_updated_vk_info_exploded.csv")
    if not dlist_genome_fasta_out:  #! these 3 dlist paths are copied in vk ref
        dlist_genome_fasta_out = os.path.join(out, "dlist_genome.fa")
    if not dlist_cdna_fasta_out:
        dlist_cdna_fasta_out = os.path.join(out, "dlist_cdna.fa")
    if not dlist_combined_fasta_out:
        dlist_combined_fasta_out = os.path.join(out, "dlist.fa")

    # make sure directories of all output files exist
    output_files = [mutations_updated_vk_info_csv_out, mutations_updated_exploded_vk_info_csv_out, dlist_genome_fasta_out, dlist_cdna_fasta_out, dlist_combined_fasta_out]
    for output_file in output_files:
        if output_file and os.path.dirname(output_file):
            os.makedirs(os.path.dirname(output_file), exist_ok=True)


    bowtie_path = kwargs.get("bowtie_path", None)

    # --np (N penalty) caps number of Ns in read (MCRS), reference (human reference genome/transcriptome), or both
    # --n-ceil (max_ambiguous_mcrs) caps number of Ns in read (MCRS) only
    # I have my remove_Ns_fasta function which caps number of Ns in reference (human reference genome/transcriptome) only
    if max_ambiguous_mcrs is None:  # no N-penalty for MCRS during d-listing
        max_ambiguous_mcrs = 99999  #! be careful of changing this number - it must be an int for bowtie2
    if max_ambiguous_reference is None:  # no N-penalty for reference during d-listing
        max_ambiguous_reference = 99999  #! be careful of changing this number - it is related to the condition in 'align_to_normal_genome_and_build_dlist' - max_ambiguous_reference < 9999
    
    if max_ambiguous_mcrs == 0 and max_ambiguous_reference == 0:  # probably redundant with the filters above but still nice to have
        N_penalty = 1
    else:
        N_penalty = 0

    k = w + 1

    output_stat_folder = f"{out}/stats"
    output_plot_folder = f"{out}/plots"

    os.makedirs(output_stat_folder, exist_ok=True)
    os.makedirs(output_plot_folder, exist_ok=True)

    # CELL

    if id_to_header_csv is not None:
        id_to_header_dict = make_mapping_dict(id_to_header_csv, dict_key="id")
        # header_to_id_dict = {v: k for k, v in id_to_header_dict.items()}
        temp_header_fa = mcrs_fasta.replace(".fa", "_with_headers.fa")
        swap_ids_for_headers_in_fasta(mcrs_fasta, id_to_header_csv, out_fasta=temp_header_fa)
    else:
        id_to_header_dict = None
        # header_to_id_dict = None
        temp_header_fa = mcrs_fasta

    # CELL
    # # Calculate lengths of lists in each column to explode
    # lengths_df = mutation_metadata_df[columns_to_explode].applymap(lambda x: len(x) if isinstance(x, list) else 0)

    # # Identify rows where list lengths differ across columns to explode
    # inconsistent_rows = lengths_df[lengths_df.nunique(axis=1) > 1]

    # # Display these problematic rows
    # print("Rows with inconsistent list lengths across columns to explode:")
    # inconsistent_rows

    # CELL
    output_fastx_stat_file = f"{output_stat_folder}/fastx_stats.txt"
    fasta_summary_stats(mcrs_fasta, output_file=output_fastx_stat_file)

    # CELL
    # columns_to_change = ['nucleotide_positions', 'start_mutation_position', 'end_mutation_position', 'actual_mutation']

    if mutations_updated_csv is None:  # does not support concatenated cdna and genome
        columns_original = []
        data = list(read_fasta(mcrs_fasta))
        mutation_metadata_df = pd.DataFrame(data, columns=[mcrs_id_column, "mcrs_sequence"])

        if id_to_header_dict is not None:
            mutation_metadata_df["mcrs_header"] = mutation_metadata_df[mcrs_id_column].map(id_to_header_dict)
        else:
            mutation_metadata_df["mcrs_header"] = mutation_metadata_df[mcrs_id_column]

        contains_enst = mutation_metadata_df["mcrs_header"].iloc[0].find("ENST") != -1  # TODO: this just differentiates cdna from genome based on searching for ENST, but there may be other ways to discern cDNA from genome

        if contains_enst:
            mcrs_source = "cdna"
        else:
            mcrs_source = "genome"

        mutation_metadata_df[mcrs_source_column] = mcrs_source

        mutation_metadata_df["header_list"] = mutation_metadata_df["mcrs_header"].str.split(";")
        mutation_metadata_df["order_list"] = mutation_metadata_df["header_list"].apply(lambda x: list(range(len(x))))

        mcrs_header_has_merged_values = mutation_metadata_df["mcrs_header"].apply(lambda x: isinstance(x, str) and ";" in x).any()

        if mcrs_header_has_merged_values:
            mutation_metadata_df_exploded = explode_df(mutation_metadata_df, columns_to_explode)
        else:
            mutation_metadata_df_exploded = mutation_metadata_df

        mutation_metadata_df_exploded[["seq_ID", "mutation"]] = mutation_metadata_df_exploded["header"].str.split(":", expand=True)

        mutation_metadata_df_exploded["seq_ID"] = mutation_metadata_df_exploded["seq_ID"].astype(str)

        mutation_metadata_df_exploded = add_mutation_information(mutation_metadata_df_exploded, mcrs_source=mcrs_source)

        if mcrs_source == "genome":
            mutation_metadata_df_exploded.rename(
                columns={"seq_ID": "chromosome", "mutation": "mutation_genome"},
                inplace=True,
            )

        columns_to_explode_extend_values = [col for col in mutation_metadata_df_exploded.columns if col not in [mcrs_id_column, "mcrs_header", "mcrs_sequence"] and col not in columns_to_explode]

    else:
        mutation_metadata_df = pd.read_csv(mutations_updated_csv)
        mutation_metadata_df.rename(
            columns={"header": "mcrs_header", mcrs_sequence_column: "mcrs_sequence"},
            inplace=True,
        )

        columns_original = mutation_metadata_df.columns.tolist()

        for column in mutation_metadata_df.columns:
            if column not in columns_to_explode + [
                mcrs_id_column,
                "mcrs_header",
                "mcrs_sequence",
                "mutant_sequence_rc",
            ]:  # alternative: check if the first and last characters are '[' and ']', respectively
                mutation_metadata_df[column] = mutation_metadata_df[column].apply(lambda x: (safe_literal_eval(x) if isinstance(x, str) and x.startswith("[") and x.endswith("]") else x))

        columns_to_explode.extend(
            [
                col
                for col in mutation_metadata_df.columns
                if col
                not in [
                    mcrs_id_column,
                    "mcrs_header",
                    "mcrs_sequence",
                    "mutant_sequence_rc",
                ]
            ]
        )
        mutation_metadata_df["header_list"] = mutation_metadata_df["mcrs_header"].str.split(";")
        mutation_metadata_df["order_list"] = mutation_metadata_df["header_list"].apply(lambda x: list(range(len(x))))

        if mcrs_source_column in mutation_metadata_df.columns:
            mcrs_source = mutation_metadata_df[mcrs_source_column].unique()
            if len(mcrs_source) > 1:
                mcrs_source = "combined"
            else:
                mcrs_source = mcrs_source[0]
        else:
            contains_enst = mutation_metadata_df["mcrs_header"].iloc[0].find("ENST") != -1  # TODO: this just differentiates cdna from genome based on searching for ENST, but there may be other ways to discern cDNA from genome

            if contains_enst:
                mcrs_source = "cdna"
            else:
                mcrs_source = "genome"

            mutation_metadata_df[mcrs_source_column] = mcrs_source

        if mcrs_source == "combined":
            mutation_metadata_df = add_some_mutation_information_when_cdna_and_genome_combined(mutation_metadata_df, columns_to_explode)
        else:
            mutation_metadata_df[f"nucleotide_positions_{mcrs_source}"] = mutation_metadata_df[f"nucleotide_positions"]
            mutation_metadata_df[f"actual_mutation_{mcrs_source}"] = mutation_metadata_df[f"actual_mutation"]
            mutation_metadata_df[f"start_mutation_position_{mcrs_source}"] = mutation_metadata_df[f"start_mutation_position"]
            mutation_metadata_df[f"end_mutation_position_{mcrs_source}"] = mutation_metadata_df[f"end_mutation_position"]

        columns_to_explode.extend(
            [
                f"nucleotide_positions_{mcrs_source}",
                f"actual_mutation_{mcrs_source}",
                f"start_mutation_position_{mcrs_source}",
                f"end_mutation_position_{mcrs_source}",
            ]
        )

        mcrs_header_has_merged_values = mutation_metadata_df["mcrs_header"].apply(lambda x: isinstance(x, str) and ";" in x).any()

        if mcrs_header_has_merged_values:
            mutation_metadata_df_exploded = explode_df(mutation_metadata_df, columns_to_explode)
        else:
            mutation_metadata_df_exploded = mutation_metadata_df

        if "chromosome" in mutation_metadata_df_exploded.columns and "mutation_genome" in mutation_metadata_df_exploded.columns:
            mutation_metadata_df_exploded["header_genome"] = mutation_metadata_df_exploded["chromosome"].astype(str) + ":" + mutation_metadata_df_exploded["mutation_genome"].astype(str)

        if "seq_ID" in mutation_metadata_df_exploded.columns and "mutation" in mutation_metadata_df_exploded.columns:
            mutation_metadata_df_exploded["header_cdna"] = mutation_metadata_df_exploded["seq_ID"].astype(str) + ":" + mutation_metadata_df_exploded["mutation"].astype(str)

        if "seq_ID" in mutation_metadata_df_exploded.columns and "mutation_cds" in mutation_metadata_df_exploded.columns:
            mutation_metadata_df_exploded["header_cds"] = mutation_metadata_df_exploded["seq_ID"].astype(str) + ":" + mutation_metadata_df_exploded["mutation_cds"].astype(str)

        columns_to_explode_extend_values = [
            "header_genome",
            "header_cdna",
            "header_cds",
        ]

        if mutation_metadata_df_exploded["mcrs_source"].unique()[0] == "cdna" and mutation_genome_column in mutation_metadata_df_exploded:
            mutation_metadata_df_exploded = add_mutation_information(
                mutation_metadata_df_exploded,
                mutation_column=mutation_genome_column,
                mcrs_source="genome",
            )
            columns_to_explode_extend_values.extend(
                [
                    f"nucleotide_positions_genome",
                    f"actual_mutation_genome",
                    f"start_mutation_position_genome",
                    f"end_mutation_position_genome",
                ]
            )
            # TODO: this is a little hacky (I set these values in the function and then reset them now)
            mutation_metadata_df_exploded[f"nucleotide_positions"] = mutation_metadata_df_exploded[f"nucleotide_positions_{mcrs_source}"]
            mutation_metadata_df_exploded[f"actual_mutation"] = mutation_metadata_df_exploded[f"actual_mutation_{mcrs_source}"]
            mutation_metadata_df_exploded[f"start_mutation_position"] = mutation_metadata_df_exploded[f"start_mutation_position_{mcrs_source}"]
            mutation_metadata_df_exploded[f"end_mutation_position"] = mutation_metadata_df_exploded[f"end_mutation_position_{mcrs_source}"]

        if mutation_metadata_df_exploded["mcrs_source"].unique()[0] == "genome" and mutation_cdna_column in mutation_metadata_df_exploded:
            mutation_metadata_df_exploded = add_mutation_information(
                mutation_metadata_df_exploded,
                mutation_column=mutation_cdna_column,
                mcrs_source="cdna",
            )
            columns_to_explode_extend_values.extend(
                [
                    f"nucleotide_positions_cdna",
                    f"actual_mutation_cdna",
                    f"start_mutation_position_cdna",
                    f"end_mutation_position_cdna",
                ]
            )
            mutation_metadata_df_exploded[f"nucleotide_positions"] = mutation_metadata_df_exploded[f"nucleotide_positions_{mcrs_source}"]
            mutation_metadata_df_exploded[f"actual_mutation"] = mutation_metadata_df_exploded[f"actual_mutation_{mcrs_source}"]
            mutation_metadata_df_exploded[f"start_mutation_position"] = mutation_metadata_df_exploded[f"start_mutation_position_{mcrs_source}"]
            mutation_metadata_df_exploded[f"end_mutation_position"] = mutation_metadata_df_exploded[f"end_mutation_position_{mcrs_source}"]

    # CELL

    if mcrs_source == "genome" or mcrs_source == "combined":
        mutation_metadata_df_exploded = mutation_metadata_df_exploded.loc[~((mutation_metadata_df_exploded[mcrs_source_column] == "genome") & ((pd.isna(mutation_metadata_df_exploded["chromosome"])) | (mutation_metadata_df_exploded["mutation_genome"].str.contains("g.nan", na=True))))]

    columns_to_explode.extend(columns_to_explode_extend_values)

    # CELL
    if columns_to_include == "all" or "cdna_and_genome_same" in columns_to_include:
        try:
            logger.info("Comparing cDNA and genome")
            mutation_metadata_df_exploded, columns_to_explode = compare_cdna_and_genome(
                mutation_metadata_df_exploded,
                reference_cdna_fasta=reference_cdna_fasta,
                reference_genome_fasta=reference_genome_fasta,
                mutations_csv=mutations_csv,
                w=w,
                mcrs_source=mcrs_source,
                columns_to_explode=columns_to_explode,
            )
        except Exception as e:
            logger.error(f"Error comparing cDNA and genome: {e}")
            columns_not_successfully_added.append("cdna_and_genome_same")

    # CELL

    if columns_to_include == "all" or "distance_to_nearest_splice_junction" in columns_to_include:
        # Add metadata: distance to nearest splice junction
        try:
            logger.info("Computing distance to nearest splice junction")
            mutation_metadata_df_exploded, columns_to_explode = compute_distance_to_closest_splice_junction(
                mutation_metadata_df_exploded,
                gtf,
                columns_to_explode=columns_to_explode,
                near_splice_junction_threshold=near_splice_junction_threshold,
            )
        except Exception as e:
            logger.error(f"Error computing distance to nearest splice junction: {e}")
            columns_not_successfully_added.append("distance_to_nearest_splice_junction")

    # CELL
    if columns_to_include == "all" or "number_of_mutations_in_this_gene_total" in columns_to_include or "header_with_gene_name" in columns_to_include:
        total_genes_output_stat_file = f"{output_stat_folder}/total_genes_and_transcripts.txt"
        try:
            logger.info("Calculating total gene info")
            mutation_metadata_df_exploded, columns_to_explode = calculate_total_gene_info(
                mutation_metadata_df_exploded,
                mcrs_id_column=mcrs_id_column,
                output_stat_file=total_genes_output_stat_file,
                output_plot_folder=output_plot_folder,
                columns_to_include=columns_to_include,
                columns_to_explode=columns_to_explode,
            )
        except Exception as e:
            logger.error(f"Error calculating total gene info: {e}")
            columns_not_successfully_added.extend(["number_of_mutations_in_this_gene_total", "header_with_gene_name"])

    # CELL
    # Calculate mutations within (k-1) of each mutation
    # compare transcript location for spliced only with cDNA header;
    # filter out genome rows where cdna and genome are the same (because I don't want to count spliced and unspliced as 2 separate things when they are the same - but maybe I do?) and compare genome location for all (both spliced and unspliced) with regular header (will be the sole way to add information for unspliced rows, and will add unspliced info for cdna comparisons);
    # take union of sets

    if columns_to_include == "all" or ("nearby_mutations" in columns_to_include or "nearby_mutations_count" in columns_to_include or "has_a_nearby_mutation" in columns_to_include):
        try:
            logger.info("Calculating nearby mutations")
            mutation_metadata_df_exploded, columns_to_explode = calculate_nearby_mutations(
                mcrs_source_column=mcrs_source_column,
                k=k,
                output_plot_folder=output_plot_folder,
                mcrs_source=mcrs_source,
                mutation_metadata_df_exploded=mutation_metadata_df_exploded,
                columns_to_explode=columns_to_explode,
            )
        except Exception as e:
            logger.error(f"Error calculating nearby mutations: {e}")
            columns_not_successfully_added.extend(["nearby_mutations", "nearby_mutations_count", "has_a_nearby_mutation"])

    # CELL
    if mcrs_header_has_merged_values:
        logger.info("Collapsing dataframe")
        mutation_metadata_df, columns_to_explode = collapse_df(
            mutation_metadata_df_exploded,
            columns_to_explode,
            columns_to_explode_extend_values=columns_to_explode_extend_values,
        )
    else:
        mutation_metadata_df = mutation_metadata_df_exploded

    # CELL

    mutation_metadata_df[mcrs_id_column] = mutation_metadata_df[mcrs_id_column].astype(str)

    if columns_to_include == "all" or "mcrs_header_length" in columns_to_include:
        try:
            logger.info("Calculating MCRS header length")
            mutation_metadata_df["mcrs_header_length"] = mutation_metadata_df["mcrs_header"].str.len()
        except Exception as e:
            logger.error(f"Error calculating MCRS header length: {e}")
            columns_not_successfully_added.append("mcrs_header_length")
    if columns_to_include == "all" or "mcrs_sequence_length" in columns_to_include:
        try:
            logger.info("Calculating MCRS sequence length")
            mutation_metadata_df["mcrs_sequence_length"] = mutation_metadata_df["mcrs_sequence"].str.len()
        except Exception as e:
            logger.error(f"Error calculating MCRS sequence length: {e}")
            columns_not_successfully_added.append("mcrs_sequence_length")

    # CELL

    # TODO: calculate if MCRS was optimized - compare MCRS_length to length of unoptimized - exclude subs, and calculate max([2*w + length(added) - length(removed)], [2*w - 1])

    if bowtie_path is not None:
        bowtie2_build = f"{bowtie_path}/bowtie2-build"
        bowtie2 = f"{bowtie_path}/bowtie2"
    else:
        bowtie2_build = "bowtie2-build"
        bowtie2 = "bowtie2"

    if "ensembl" in dlist_reference_source:
        reference_out_dir_sequences_dlist = f"{reference_out_dir}/{dlist_reference_source}"
    elif dlist_reference_source == "t2t":
        reference_out_dir_sequences_dlist = f"{reference_out_dir}/T2T/GCF_009914755.1"
    os.makedirs(reference_out_dir_sequences_dlist, exist_ok=True)

    # TODO: have more columns_to_include options that allows me to do cdna alone, genome alone, or both combined - currently it is either cdna+genome or nothing
    if columns_to_include == "all" or ("dlist" in columns_to_include or "number_of_alignments_to_normal_human_reference" in columns_to_include or "dlist_substring" in columns_to_include or "number_of_substring_matches_to_normal_human_reference" in columns_to_include):
        try:
            logger.info("Aligning to normal genome and building dlist")
            (
                mutation_metadata_df,
                ref_dlist_fa_genome,
                ref_dlist_gtf,
                sequence_names_set_union_genome_and_cdna,
            ) = align_to_normal_genome_and_build_dlist(
                mutations=mcrs_fasta,
                mcrs_id_column=mcrs_id_column,
                out_dir_notebook=out,
                dlist_fasta_file_genome_full=dlist_genome_fasta_out,
                dlist_fasta_file_cdna_full=dlist_cdna_fasta_out,
                dlist_fasta_file=dlist_combined_fasta_out,
                dlist_reference_source=dlist_reference_source,
                ref_prefix="index",
                strandedness=strandedness,
                threads=threads,
                N_penalty=N_penalty,
                max_ambiguous_mcrs=max_ambiguous_mcrs,
                max_ambiguous_reference=max_ambiguous_reference,
                k=k,
                output_stat_folder=output_stat_folder,
                mutation_metadata_df=mutation_metadata_df,
                bowtie2_build=bowtie2_build,
                bowtie2=bowtie2,
                reference_out_dir_sequences_dlist=reference_out_dir_sequences_dlist,
                logger=logger,
            )
        except Exception as e:
            logger.error(f"Error aligning to normal genome and building dlist: {e}")
            columns_not_successfully_added.extend(
                [
                    "dlist",
                    "number_of_alignments_to_normal_human_reference",
                    "dlist_substring",
                    "number_of_substring_matches_to_normal_human_reference",
                ]
            )

    # CELL
    kat_output = f"{out}/kat_output/kat.hist"
    try:
        kat_hist_command = [
            "kat",
            "hist",
            "-m",
            str(k),
            "--threads",
            str(threads),
            "-o",
            kat_output,
            mcrs_fasta,
        ]
        if strandedness:
            # insert as the second element
            kat_hist_command.insert(2, "--stranded")
        logger.info("Running KAT")
        subprocess.run(kat_hist_command, check=True)
    except Exception as e:
        logger.error(f"Error running KAT: {e}")

    if os.path.exists(kat_output):
        plot_kat_histogram(kat_output)

    # CELL

    if columns_to_include == "all" or ("pseudoaligned_to_human_reference" in columns_to_include or "pseudoaligned_to_human_reference_despite_not_truly_aligning" in columns_to_include):
        if not sequence_names_set_union_genome_and_cdna:
            sequence_names_set_union_genome_and_cdna = set()
            column_name = "pseudoaligned_to_human_reference"
        else:
            column_name = "pseudoaligned_to_human_reference_despite_not_truly_aligning"

        ref_folder_kb = f"{reference_out_dir_sequences_dlist}/kb_index"

        try:
            logger.info("Getting MCRSs that pseudoalign but aren't dlisted")
            mutation_metadata_df = get_mcrss_that_pseudoalign_but_arent_dlisted(
                mutation_metadata_df=mutation_metadata_df,
                mcrs_id_column=mcrs_id_column,
                mcrs_fa=mcrs_fasta,
                sequence_names_set=sequence_names_set_union_genome_and_cdna,
                human_reference_genome_fa=ref_dlist_fa_genome,
                human_reference_gtf=ref_dlist_gtf,
                out_dir_notebook=out,
                ref_folder_kb=ref_folder_kb,
                dlist_reference_source=dlist_reference_source,
                header_column_name=mcrs_id_column,
                additional_kb_extract_filtering_workflow="nac",
                k=k,
                threads=threads,
                strandedness=strandedness,
                column_name=column_name,
            )
        except Exception as e:
            logger.error(f"Error getting MCRSs that pseudoalign but aren't dlisted: {e}")
            columns_not_successfully_added.append(column_name)

    # CELL

    if columns_to_include == "all" or ("number_of_kmers_with_overlap_to_other_mcrs_items_in_mcrs_reference" in columns_to_include or "number_of_mcrs_items_with_overlapping_kmers_in_mcrs_reference" in columns_to_include):
        try:
            logger.info("Calculating overlap between MCRS items")
            df_overlap_stat_file = f"{output_stat_folder}/df_overlap_stat.txt"
            df_overlap = get_df_overlap(
                mcrs_fasta,
                out_dir_notebook=out,
                k=k,
                strandedness=strandedness,
                mcrs_id_column=mcrs_id_column,
                output_text_file=df_overlap_stat_file,
                output_plot_folder=output_plot_folder,
            )

            mutation_metadata_df = mutation_metadata_df.merge(df_overlap, on=mcrs_id_column, how="left")
            mutation_metadata_df["kmer_overlap_in_mcrs_reference"] = mutation_metadata_df["number_of_kmers_with_overlap_to_other_mcrs_items_in_mcrs_reference"].astype(bool)
            mutation_metadata_df["kmer_overlap_in_mcrs_reference"] = mutation_metadata_df["number_of_kmers_with_overlap_to_other_mcrs_items_in_mcrs_reference"].notna() & mutation_metadata_df["number_of_kmers_with_overlap_to_other_mcrs_items_in_mcrs_reference"].astype(bool)
        except Exception as e:
            logger.error(f"Error calculating overlap between MCRS items: {e}")
            columns_not_successfully_added.extend(
                [
                    "number_of_kmers_with_overlap_to_other_mcrs_items_in_mcrs_reference",
                    "number_of_mcrs_items_with_overlapping_kmers_in_mcrs_reference",
                ]
            )

    # CELL

    # Applying the function to the DataFrame
    if columns_to_include == "all" or ("longest_homopolymer_length" in columns_to_include or "longest_homopolymer" in columns_to_include):
        try:
            logger.info("Calculating longest homopolymer")
            (
                mutation_metadata_df["longest_homopolymer_length"],
                mutation_metadata_df["longest_homopolymer"],
            ) = zip(*mutation_metadata_df["mcrs_sequence"].apply(lambda x: (longest_homopolymer(x) if pd.notna(x) else (np.nan, np.nan))))
        except Exception as e:
            logger.error(f"Error calculating longest homopolymer: {e}")
            columns_not_successfully_added.extend(["longest_homopolymer_length", "longest_homopolymer"])

    # CELL

    if columns_to_include == "all" or ("num_distinct_triplets" in columns_to_include or "num_total_triplets" in columns_to_include or "triplet_complexity" in columns_to_include):
        logger.info("Calculating triplet stats")
        try:
            (
                mutation_metadata_df["num_distinct_triplets"],
                mutation_metadata_df["num_total_triplets"],
                mutation_metadata_df["triplet_complexity"],
            ) = zip(*mutation_metadata_df["mcrs_sequence"].apply(lambda x: (triplet_stats(x) if pd.notna(x) else (np.nan, np.nan, np.nan))))

            output_file_longest_homopolymer = f"{output_plot_folder}/longest_homopolymer.png"
            plot_histogram_of_nearby_mutations_7_5(
                mutation_metadata_df,
                "longest_homopolymer_length",
                bins=20,
                output_file=output_file_longest_homopolymer,
            )

            output_file_triplet_complexity = f"{output_plot_folder}/triplet_complexity.png"
            plot_histogram_of_nearby_mutations_7_5(
                mutation_metadata_df,
                "triplet_complexity",
                bins=20,
                output_file=output_file_triplet_complexity,
            )

        except Exception as e:
            logger.error(f"Error calculating triplet stats: {e}")
            columns_not_successfully_added.extend(["num_distinct_triplets", "num_total_triplets", "triplet_complexity"])

    # CELL
    # add metadata: MCRS mutation type
    if columns_to_include == "all" or "mcrs_mutation_type" in columns_to_include:
        try:
            logger.info("Adding MCRS mutation type")
            mutation_metadata_df = add_mcrs_mutation_type(mutation_metadata_df, mut_column="mcrs_header")
        except Exception as e:
            logger.error(f"Error adding MCRS mutation type: {e}")
            columns_not_successfully_added.append("mcrs_mutation_type")

    # CELL

    # Add metadata: ';' in mcrs_header
    if columns_to_include == "all" or ("concatenated_headers_in_mcrs" in columns_to_include or "number_of_mutations_in_mcrs_header" in columns_to_include):
        try:
            logger.info("Adding concatenated header info")
            mutation_metadata_df["concatenated_headers_in_mcrs"] = mutation_metadata_df["mcrs_header"].str.contains(";")
            mutation_metadata_df["number_of_mutations_in_mcrs_header"] = mutation_metadata_df["mcrs_header"].str.count(";") + 1
        except Exception as e:
            logger.error(f"Error adding concatenated headers in MCRS: {e}")
            columns_not_successfully_added.extend(["concatenated_headers_in_mcrs", "number_of_mutations_in_mcrs_header"])

    # CELL

    # Add metadata: mcrs_sequence_rc
    if columns_to_include == "all" or "mcrs_sequence_rc" in columns_to_include:
        try:
            logger.info("Adding MCRS reverse complement")
            mutation_metadata_df["mcrs_sequence_rc"] = mutation_metadata_df["mcrs_sequence"].apply(reverse_complement)
        except Exception as e:
            logger.error(f"Error adding MCRS reverse complement: {e}")
            columns_not_successfully_added.append("mcrs_sequence_rc")

    # CELL

    # Add metadata: mcrs substring and superstring (forward and rc)
    if columns_to_include == "all" or ("entries_for_which_this_mcrs_is_substring" in columns_to_include or "entries_for_which_this_mcrs_is_superstring" in columns_to_include or "mcrs_is_substring" in columns_to_include or "mcrs_is_superstring" in columns_to_include):
        mcrs_to_mcrs_bowtie_folder = f"{out}/bowtie_mcrs_to_mcrs"
        mcrs_sam_file = f"{mcrs_to_mcrs_bowtie_folder}/mutant_reads_to_mcrs_index.sam"
        substring_output_stat_file = f"{output_stat_folder}/substring_output_stat.txt"

        try:
            logger.info("Creating MCRS to self headers")
            substring_to_superstring_df, superstring_to_substring_df = create_df_of_mcrs_to_self_headers(
                mcrs_sam_file=mcrs_sam_file,
                mcrs_fa=mcrs_fasta,
                bowtie_mcrs_reference_folder=mcrs_to_mcrs_bowtie_folder,
                bowtie_path=bowtie_path,
                threads=threads,
                strandedness=strandedness,
                mcrs_id_column=mcrs_id_column,
                output_stat_file=substring_output_stat_file,
            )

            mutation_metadata_df[mcrs_id_column] = mutation_metadata_df[mcrs_id_column].astype(str)
            mutation_metadata_df = mutation_metadata_df.merge(substring_to_superstring_df, on=mcrs_id_column, how="left")
            mutation_metadata_df = mutation_metadata_df.merge(superstring_to_substring_df, on=mcrs_id_column, how="left")

            mutation_metadata_df["mcrs_is_substring"] = mutation_metadata_df["mcrs_is_substring"].fillna(False).astype(bool)
            mutation_metadata_df["mcrs_is_superstring"] = mutation_metadata_df["mcrs_is_superstring"].fillna(False).astype(bool)

        except Exception as e:
            logger.error(f"Error creating MCRS to self headers: {e}")
            columns_not_successfully_added.extend(
                [
                    "entries_for_which_this_mcrs_is_substring",
                    "entries_for_which_this_mcrs_is_superstring",
                    "mcrs_is_substring",
                    "mcrs_is_superstring",
                ]
            )

    # CELL
    logger.info("sorting mutation metadata by mcrs id")
    mutation_metadata_df = mutation_metadata_df.sort_values(by="mcrs_id").reset_index(drop=True)

    logger.info("Saving mutation metadata")
    mutation_metadata_df.to_csv(mutations_updated_vk_info_csv_out, index=False)

    # CELL

    if save_mutations_updated_exploded_vk_info_csv:
        logger.info("Saving exploded mutation metadata")
        mutation_metadata_df_exploded = explode_df(mutation_metadata_df, columns_to_explode)
        mutation_metadata_df_exploded.to_csv(mutations_updated_exploded_vk_info_csv_out, index=False)

    if verbose:
        logger.info(f"Saved mutation metadata to {mutations_updated_vk_info_csv_out}")
        logger.info(f"Columns: {mutation_metadata_df.columns}")
        logger.info(f"Columns successfully added: {set(mutation_metadata_df.columns.tolist()) - set(columns_original)}")
        logger.info(f"Columns not successfully added: {columns_not_successfully_added}")
