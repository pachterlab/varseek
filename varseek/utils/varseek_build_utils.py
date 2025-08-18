import os
import re
import tempfile
from collections import OrderedDict, defaultdict

import subprocess
import numpy as np
import gget

import pandas as pd
import pyfastx
from tqdm import tqdm

from varseek.constants import codon_to_amino_acid, mutation_pattern, supported_databases_and_corresponding_reference_sequence_type, complement
from varseek.utils.logger_utils import set_up_logger

import logging

logger = logging.getLogger(__name__)
logger = set_up_logger(logger, logging_level="INFO", save_logs=False, log_dir=None)

tqdm.pandas()


def convert_chromosome_value_to_int_when_possible(val):
    try:
        # Try to convert the value to a float, then to an int, and finally to a string
        return str(int(float(val)))
    except ValueError:
        # If conversion fails, keep the value as it is
        return str(val)


# Function to ensure unique IDs
def generate_unique_ids(num_ids, start=1, total_rows=None):
    if total_rows is None:
        total_rows = num_ids
    num_digits = len(str(total_rows + start - 1))
    generated_ids = [f"vcrs_{i:0{num_digits}}" for i in range(start, start + num_ids)]
    return list(generated_ids)

def translate_sequence(sequence, start=0, end=None):
    if end is None:  # If end is not provided, set it to the length of the sequence
        end = len(sequence)
    
    amino_acid_sequence = ""
    for i in range(start, end, 3):
        codon = sequence[i : (i + 3)].upper()
        amino_acid = codon_to_amino_acid.get(codon, "X")  # Use 'X' for unknown codons
        amino_acid_sequence += amino_acid

    amino_acid_sequence = amino_acid_sequence.strip("X")  # Remove leading and trailing 'X's
    return amino_acid_sequence


def wt_fragment_and_mutant_fragment_share_kmer(mutated_fragment: str, wildtype_fragment: str, k: int) -> bool:
    if len(mutated_fragment) <= k:
        return bool(mutated_fragment in wildtype_fragment)

    # else:
    for mutant_position in range(len(mutated_fragment) - k):
        mutant_kmer = mutated_fragment[mutant_position : (mutant_position + k)]
        if mutant_kmer in wildtype_fragment:
            # wt_position = wildtype_fragment.find(mutant_kmer)
            return True
    return False


def return_pyfastx_index_object_with_header_versions_removed(fasta_path):
    logger.info(f"Removing version numbers in fasta headers for {fasta_path}")
    fa_read_only = pyfastx.Fastx(fasta_path)
    with tempfile.NamedTemporaryFile(mode="w", suffix=".fa", encoding="utf-8", delete=True) as temp_fasta:
        temp_fasta_path = temp_fasta.name
        temp_fasta_index_path = temp_fasta_path + ".fxi"
        for name, seq in fa_read_only:
            name_without_version = name.split(".")[0]
            temp_fasta.write(f">{name_without_version}\n{seq}\n")
        temp_fasta.flush()
        logger.info(f"Building pyfastx index for {fasta_path}")
        fa = pyfastx.Fasta(temp_fasta_path, build_index=True)
    return fa, temp_fasta_index_path


# Helper function to find starting position of CDS in cDNA
def find_cds_position(cdna_seq, cds_seq):
    pos = cdna_seq.find(cds_seq)
    return pos if pos != -1 else None


def count_leading_Ns(seq):
    return len(seq) - len(seq.lstrip("N"))


def convert_mutation_cds_locations_to_cdna(input_csv_path, cdna_fasta_path, cds_fasta_path, output_csv_path=None, verbose=True, strip_leading_Ns_cds=False):
    # Load the CSV
    if isinstance(input_csv_path, str):
        logger.info(f"Loading CSV from {input_csv_path}")
        df = pd.read_csv(input_csv_path)
    elif isinstance(input_csv_path, pd.DataFrame):
        df = input_csv_path.copy()
    else:
        raise ValueError("input_csv_path must be a string or a pandas DataFrame")

    logger.info("Copying df internally to avoid in-place modifications")
    df_original = df.copy()

    bad_mutations_dict = {}

    logger.info("Removing unknown mutations")
    # get rids of mutations that are uncertain, ambiguous, intronic, posttranslational
    uncertain_mutations = df["mutation"].str.contains(r"\?").sum()

    ambiguous_position_mutations = df["mutation"].str.contains(r"\(|\)").sum()

    intronic_mutations = df["mutation"].str.contains(r"\+|\-").sum()

    posttranslational_region_mutations = df["mutation"].str.contains(r"\*").sum()

    logger.info("Removing unsupported mutation types")
    logger.info(f"Uncertain mutations: {uncertain_mutations}")
    logger.info(f"Ambiguous position mutations: {ambiguous_position_mutations}")
    logger.info(f"Intronic mutations: {intronic_mutations}")
    logger.info(f"Posttranslational region mutations: {posttranslational_region_mutations}")

    bad_mutations_dict["uncertain_mutations"] = uncertain_mutations
    bad_mutations_dict["ambiguous_position_mutations"] = ambiguous_position_mutations
    bad_mutations_dict["intronic_mutations"] = intronic_mutations
    bad_mutations_dict["posttranslational_region_mutations"] = posttranslational_region_mutations

    # Filter out bad mutations
    combined_pattern = re.compile(r"(\?|\(|\)|\+|\-|\*)")  # gets rids of mutations that are uncertain, ambiguous, intronic, posttranslational
    mask = df["mutation"].str.contains(combined_pattern)
    df = df[~mask]

    logger.info("Sorting df")
    df = df.sort_values(by="seq_ID")  # to make iterrows more efficient

    logger.info("Determining mutation positions")
    df[["nucleotide_positions", "actual_variant"]] = df["mutation"].str.extract(mutation_pattern)

    split_positions = df["nucleotide_positions"].str.split("_", expand=True)

    df["start_variant_position"] = split_positions[0]
    if split_positions.shape[1] > 1:
        df["end_variant_position"] = split_positions[1].fillna(split_positions[0])
    else:
        df["end_variant_position"] = df["start_variant_position"]

    df.loc[df["end_variant_position"].isna(), "end_variant_position"] = df["start_variant_position"]

    df[["start_variant_position", "end_variant_position"]] = df[["start_variant_position", "end_variant_position"]].astype(int)

    # # Rename the mutation column
    # df.rename(columns={"mutation": "mutation_cds"}, inplace=True)

    temp_fasta_index_path_cdna, temp_fasta_index_path_cds = None, None  # in case the block fails

    # put in try-except-finally block to ensure that the temp index files are erased no matter what
    try:
        # Load the FASTA files
        fa_cdna, temp_fasta_index_path_cdna = return_pyfastx_index_object_with_header_versions_removed(cdna_fasta_path)
        fa_cds, temp_fasta_index_path_cds = return_pyfastx_index_object_with_header_versions_removed(cds_fasta_path)

        number_bad = 0
        seq_id_previous = None

        iterator = tqdm(df.iterrows(), total=len(df), desc="Processing rows") if verbose else df.iterrows()

        # Process each row
        for index, row in iterator:
            seq_id = row["seq_ID"]

            if seq_id != seq_id_previous:
                if seq_id in fa_cdna and seq_id in fa_cds:
                    cdna_seq = fa_cdna[seq_id].seq
                    cds_seq = fa_cds[seq_id].seq
                    number_of_leading_ns = count_leading_Ns(cds_seq)  # modified in April 2025 - used to be count_leading_Ns(cdna_seq)
                    cds_seq = cds_seq.strip("N")
                    cds_start_pos = find_cds_position(cdna_seq, cds_seq)
                    seq_id_found_in_cdna_and_cds = True
                else:
                    seq_id_found_in_cdna_and_cds = False

            if (not seq_id_found_in_cdna_and_cds) or (cds_start_pos is None):
                df.at[index, "mutation_cdna"] = None
                number_bad += 1
            else:
                if strip_leading_Ns_cds:
                    df.at[index, "start_variant_position"] += cds_start_pos
                    df.at[index, "end_variant_position"] += cds_start_pos
                else:
                    df.at[index, "start_variant_position"] += cds_start_pos - number_of_leading_ns
                    df.at[index, "end_variant_position"] += cds_start_pos - number_of_leading_ns

                start = df.at[index, "start_variant_position"]
                end = df.at[index, "end_variant_position"]
                actual_variant = row["actual_variant"]

                if start == end:
                    df.at[index, "mutation_cdna"] = f"c.{start}{actual_variant}"
                else:
                    df.at[index, "mutation_cdna"] = f"c.{start}_{end}{actual_variant}"

            seq_id_previous = seq_id

        logger.info(f"Number of bad mutations: {number_bad}")
        logger.info("Merging dfs")

        if (df_original.duplicated(subset=["seq_ID", "mutation"]).sum() == 0) and (df.duplicated(subset=["seq_ID", "mutation"]).sum() == 0):  # this condition should be True if downloading with default gget cosmic, but in case the user wants duplicate rows then I'll give both options
            df_merged = df_original.set_index(["seq_ID", "mutation"]).join(df.set_index(["seq_ID", "mutation"])[["mutation_cdna"]], how="left").reset_index()
        else:
            df_merged = df_original.merge(df[["seq_ID", "mutation", "mutation_cdna"]], on=["seq_ID", "mutation"], how="left")  # new as of Feb 2025

        # Write to new CSV
        # if not output_csv_path:
        #     output_csv_path = input_csv_path
        if output_csv_path:
            logger.info(f"Saving output to {output_csv_path}")
            df_merged.to_csv(output_csv_path, index=False)  # new as of Feb 2025 (replaced df.to_csv with df_merged.to_csv)

        return df_merged, bad_mutations_dict

    except Exception as e:
        raise RuntimeError(f"Error converting CDS to cDNA: {e}") from e
    finally:
        logger.info("Cleaning up temporary files...")
        for temp_path in [temp_fasta_index_path_cdna, temp_fasta_index_path_cds]:
            if temp_path and os.path.exists(temp_path):
                os.remove(temp_path)


def convert_mutation_cds_locations_to_cdna_old(input_csv_path, cdna_fasta_path, cds_fasta_path, output_csv_path=None, verbose=True):
    # Load the CSV
    if isinstance(input_csv_path, str):
        logger.info(f"Loading CSV from {input_csv_path}")
        df = pd.read_csv(input_csv_path)
    elif isinstance(input_csv_path, pd.DataFrame):
        df = input_csv_path.copy()
    else:
        raise ValueError("input_csv_path must be a string or a pandas DataFrame")

    logger.info("Copying df internally to avoid in-place modifications")
    df_original = df.copy()

    bad_mutations_dict = {}

    logger.info("Removing unknown mutations")
    # get rids of mutations that are uncertain, ambiguous, intronic, posttranslational
    uncertain_mutations = df["mutation"].str.contains(r"\?").sum()

    ambiguous_position_mutations = df["mutation"].str.contains(r"\(|\)").sum()

    intronic_mutations = df["mutation"].str.contains(r"\+|\-").sum()

    posttranslational_region_mutations = df["mutation"].str.contains(r"\*").sum()

    logger.info("Removing unsupported mutation types")
    logger.info(f"Uncertain mutations: {uncertain_mutations}")
    logger.info(f"Ambiguous position mutations: {ambiguous_position_mutations}")
    logger.info(f"Intronic mutations: {intronic_mutations}")
    logger.info(f"Posttranslational region mutations: {posttranslational_region_mutations}")

    bad_mutations_dict["uncertain_mutations"] = uncertain_mutations
    bad_mutations_dict["ambiguous_position_mutations"] = ambiguous_position_mutations
    bad_mutations_dict["intronic_mutations"] = intronic_mutations
    bad_mutations_dict["posttranslational_region_mutations"] = posttranslational_region_mutations

    # Filter out bad mutations
    combined_pattern = re.compile(r"(\?|\(|\)|\+|\-|\*)")  # gets rids of mutations that are uncertain, ambiguous, intronic, posttranslational
    mask = df["mutation"].str.contains(combined_pattern)
    df = df[~mask]

    logger.info("Sorting df")
    df = df.sort_values(by="seq_ID")  # to make iterrows more efficient

    logger.info("Determining mutation positions")
    df[["nucleotide_positions", "actual_variant"]] = df["mutation"].str.extract(mutation_pattern)

    split_positions = df["nucleotide_positions"].str.split("_", expand=True)

    df["start_variant_position"] = split_positions[0]
    if split_positions.shape[1] > 1:
        df["end_variant_position"] = split_positions[1].fillna(split_positions[0])
    else:
        df["end_variant_position"] = df["start_variant_position"]

    df.loc[df["end_variant_position"].isna(), "end_variant_position"] = df["start_variant_position"]

    df[["start_variant_position", "end_variant_position"]] = df[["start_variant_position", "end_variant_position"]].astype(int)

    # # Rename the mutation column
    # df.rename(columns={"mutation": "mutation_cds"}, inplace=True)

    temp_fasta_index_path_cdna, temp_fasta_index_path_cds = None, None  # in case the block fails

    # put in try-except-finally block to ensure that the temp index files are erased no matter what
    try:
        # Load the FASTA files
        fa_cdna, temp_fasta_index_path_cdna = return_pyfastx_index_object_with_header_versions_removed(cdna_fasta_path)
        fa_cds, temp_fasta_index_path_cds = return_pyfastx_index_object_with_header_versions_removed(cds_fasta_path)

        number_bad = 0
        seq_id_previous = None

        iterator = tqdm(df.iterrows(), total=len(df), desc="Processing rows") if verbose else df.iterrows()

        # Process each row
        for index, row in iterator:
            seq_id = row["seq_ID"]

            if seq_id != seq_id_previous:
                if seq_id in fa_cdna and seq_id in fa_cds:
                    cdna_seq = fa_cdna[seq_id].seq
                    cds_seq = fa_cds[seq_id].seq
                    cds_seq = cds_seq.strip("N")
                    cds_start_pos = find_cds_position(cdna_seq, cds_seq)
                    seq_id_found_in_cdna_and_cds = True
                else:
                    seq_id_found_in_cdna_and_cds = False

            if (not seq_id_found_in_cdna_and_cds) or (cds_start_pos is None):
                df.at[index, "mutation_cdna"] = None
                number_bad += 1
            else:
                df.at[index, "start_variant_position"] += cds_start_pos
                df.at[index, "end_variant_position"] += cds_start_pos

                start = df.at[index, "start_variant_position"]
                end = df.at[index, "end_variant_position"]
                actual_variant = row["actual_variant"]

                if start == end:
                    df.at[index, "mutation_cdna"] = f"c.{start}{actual_variant}"
                else:
                    df.at[index, "mutation_cdna"] = f"c.{start}_{end}{actual_variant}"

            seq_id_previous = seq_id

        logger.info(f"Number of bad mutations: {number_bad}")
        logger.info("Merging dfs")

        if (df_original.duplicated(subset=["seq_ID", "mutation"]).sum() == 0) and (df.duplicated(subset=["seq_ID", "mutation"]).sum() == 0):  # this condition should be True if downloading with default gget cosmic, but in case the user wants duplicate rows then I'll give both options
            df_merged = df_original.set_index(["seq_ID", "mutation"]).join(df.set_index(["seq_ID", "mutation"])[["mutation_cdna"]], how="left").reset_index()
        else:
            df_merged = df_original.merge(df[["seq_ID", "mutation", "mutation_cdna"]], on=["seq_ID", "mutation"], how="left")  # new as of Feb 2025

        # Write to new CSV
        # if not output_csv_path:
        #     output_csv_path = input_csv_path
        if output_csv_path:
            logger.info(f"Saving output to {output_csv_path}")
            df_merged.to_csv(output_csv_path, index=False)  # new as of Feb 2025 (replaced df.to_csv with df_merged.to_csv)

        return df_merged, bad_mutations_dict

    except Exception as e:
        raise RuntimeError(f"Error converting CDS to cDNA: {e}") from e
    finally:
        logger.info("Cleaning up temporary files...")
        for temp_path in [temp_fasta_index_path_cdna, temp_fasta_index_path_cds]:
            if temp_path and os.path.exists(temp_path):
                os.remove(temp_path)


def find_matching_sequences_through_fasta(file_path, ref_sequence):
    headers_of_matching_sequences = []
    for header, sequence in pyfastx.Fastx(file_path):
        if sequence == ref_sequence:
            headers_of_matching_sequences.append(header)

    return headers_of_matching_sequences


def create_header_to_sequence_ordered_dict_from_fasta_after_semicolon_splitting(
    input_fasta,
):
    mutant_reference = OrderedDict()
    for mutant_reference_header, mutant_reference_sequence in pyfastx.Fastx(input_fasta):
        mutant_reference_header_individual_list = mutant_reference_header.split(";")
        for mutant_reference_header_individual in mutant_reference_header_individual_list:
            mutant_reference[mutant_reference_header_individual] = mutant_reference_sequence
    return mutant_reference


# def merge_genome_into_transcriptome_fasta(
#     mutation_reference_file_fasta_transcriptome,
#     mutation_reference_file_fasta_genome,
#     mutation_reference_file_fasta_combined,
#     cosmic_reference_file_mutation_csv,
# ):

#     # TODO: make header fasta from id fasta with id:header dict

#     mutant_reference_transcriptome = create_header_to_sequence_ordered_dict_from_fasta_after_semicolon_splitting(mutation_reference_file_fasta_transcriptome)
#     mutant_reference_genome = create_header_to_sequence_ordered_dict_from_fasta_after_semicolon_splitting(mutation_reference_file_fasta_genome)

#     cosmic_df = pd.read_csv(
#         cosmic_reference_file_mutation_csv,
#         usecols=["seq_ID", "mutation_cdna", "chromosome", "mutation_genome"],  # TODO: remove column hard-coding
#     )
#     cosmic_df["chromosome"] = cosmic_df["chromosome"].apply(convert_chromosome_value_to_int_when_possible)

#     mutant_reference_genome_to_keep = OrderedDict()

#     for header_genome, sequence_genome in mutant_reference_genome.items():
#         seq_id_genome, mutation_id_genome = header_genome.split(":", 1)
#         row_corresponding_to_genome = cosmic_df[(cosmic_df["chromosome"] == seq_id_genome) & (cosmic_df["mutation_genome"] == mutation_id_genome)]
#         seq_id_transcriptome_corresponding_to_genome = row_corresponding_to_genome["seq_ID"].iloc[0]
#         mutation_id_transcriptome_corresponding_to_genome = row_corresponding_to_genome["mutation_cdna"].iloc[0]
#         header_transcriptome_corresponding_to_genome = f"{seq_id_transcriptome_corresponding_to_genome}:{mutation_id_transcriptome_corresponding_to_genome}"

#         if header_transcriptome_corresponding_to_genome in mutant_reference_transcriptome:
#             if mutant_reference_transcriptome[header_transcriptome_corresponding_to_genome] != sequence_genome:
#                 header_genome_transcriptome_style = f"unspliced{header_transcriptome_corresponding_to_genome}"  # TODO: change when I change unspliced notation
#                 mutant_reference_genome_to_keep[header_genome_transcriptome_style] = sequence_genome
#         else:
#             header_genome_transcriptome_style = f"unspliced{header_transcriptome_corresponding_to_genome}"  # TODO: change when I change unspliced notation
#             mutant_reference_genome_to_keep[header_genome_transcriptome_style] = sequence_genome

#     mutant_reference_combined = OrderedDict(mutant_reference_transcriptome)
#     mutant_reference_combined.update(mutant_reference_genome_to_keep)

#     mutant_reference_combined = join_keys_with_same_values(mutant_reference_combined)

#     # initialize combined fasta file with transcriptome fasta
#     with open(mutation_reference_file_fasta_combined, "w", encoding="utf-8") as fasta_file:
#         for (
#             header_transcriptome,
#             sequence_transcriptome,
#         ) in mutant_reference_combined.items():
#             # write the header followed by the sequence
#             fasta_file.write(f">{header_transcriptome}\n{sequence_transcriptome}\n")

#     # TODO: make id fasta from header fasta with id:header dict

#     print(f"Combined fasta file created at {mutation_reference_file_fasta_combined}")


def join_keys_with_same_values(original_dict):
    # Step 1: Group keys by their values
    grouped_dict = defaultdict(list)
    for key, value in original_dict.items():
        grouped_dict[value].append(key)

    # Step 2: Create the new OrderedDict with concatenated keys
    concatenated_dict = OrderedDict((";".join(keys), value) for value, keys in grouped_dict.items())

    return concatenated_dict


def download_cosmic_mutations(gtf, gtf_transcript_id_column, reference_out_dir, cosmic_version, cosmic_email, cosmic_password, columns_to_keep, grch, mutations_original, sequences, cds_file, cdna_file, var_id_column, verbose):
    reference_out_cosmic = f"{reference_out_dir}/cosmic"
    if int(cosmic_version) == 100:
        mutations = f"{reference_out_cosmic}/CancerMutationCensus_AllData_Tsv_v{cosmic_version}_GRCh{grch}_v2/CancerMutationCensus_AllData_v{cosmic_version}_GRCh{grch}_mutation_workflow.csv"
    else:
        mutations = f"{reference_out_cosmic}/CancerMutationCensus_AllData_Tsv_v{cosmic_version}_GRCh{grch}/CancerMutationCensus_AllData_v{cosmic_version}_GRCh{grch}_mutation_workflow.csv"
    mutations_path = mutations

    if not os.path.isfile(mutations):  # DO NOT specify column names in gget cosmic - I instead code them in later
        gget.cosmic(
                    None,
                    grch_version=grch,
                    cosmic_version=cosmic_version,
                    out=reference_out_cosmic,
                    cosmic_project="cancer",
                    download_cosmic=True,
                    gget_mutate=True,
                    keep_genome_info=True,
                    remove_duplicates=True,
                    email=cosmic_email,
                    password=cosmic_password,
                )

        mutations = pd.read_csv(mutations_path)

        if gtf:
            if os.path.isfile(gtf):
                mutations = merge_gtf_transcript_locations_into_cosmic_csv(mutations, gtf, gtf_transcript_id_column=gtf_transcript_id_column, output_mutations_path=mutations_path)
                columns_to_keep.extend(
                            [
                                "start_transcript_position",
                                "end_transcript_position",
                                "strand",
                            ]
                        )

                if "CancerMutationCensus" in mutations_original or mutations_original == "cosmic_cmc":
                    logger.info("COSMIC CMC genome strand information is not fully accurate. Improving with gtf information.")
                    mutations = improve_genome_strand_information(mutations, mutation_genome_column_name="mutation_genome", output_mutations_path=mutations_path)
            else:
                raise ValueError(f"gtf file '{gtf}' does not exist.")
    else:
        mutations = pd.read_csv(mutations_path)

    if sequences == "cdna" or sequences.endswith(supported_databases_and_corresponding_reference_sequence_type[mutations_original]["sequence_file_names"]["cdna"].replace("GRCH_NUMBER", grch)):  # covers whether sequences == "cdna" or sequences == "PATH/TO/Homo_sapiens.GRCh37.cdna.all.fa"
        if "mutation_cdna" not in mutations.columns:
            logger.info("Adding in cdna information into COSMIC. Note that the 'mutation_cdna' column will refer to cDNA mutation notation, and the 'mutation' column will refer to CDS mutation notation.")
            mutations, bad_cosmic_mutations_dict = convert_mutation_cds_locations_to_cdna(input_csv_path=mutations, output_csv_path=mutations_path, cds_fasta_path=cds_file, cdna_fasta_path=cdna_file, verbose=verbose, strip_leading_Ns_cds=True)

            # uncertain_mutations += bad_cosmic_mutations_dict["uncertain_mutations"]
            # ambiguous_position_mutations += bad_cosmic_mutations_dict["ambiguous_position_mutations"]
            # intronic_mutations += bad_cosmic_mutations_dict["intronic_mutations"]
            # posttranslational_region_mutations += bad_cosmic_mutations_dict["posttranslational_region_mutations"]

        seq_id_column = supported_databases_and_corresponding_reference_sequence_type[mutations_original]["column_names"]["seq_id_cdna_column"]  # "seq_ID"
        var_column = supported_databases_and_corresponding_reference_sequence_type[mutations_original]["column_names"]["var_cdna_column"]  # "mutation_cdna"

    elif sequences == "genome" or sequences.endswith(supported_databases_and_corresponding_reference_sequence_type[mutations_original]["sequence_file_names"]["genome"].replace("GRCH_NUMBER", grch)):  # covers whether sequences == "genome" or sequences == "PATH/TO/Homo_sapiens.GRCh37.dna.primary_assembly.fa"
        mutations_path_no_duplications = mutations_path.replace(".csv", "_no_duplications.csv")
        if not os.path.isfile(mutations_path_no_duplications):
            logger.info("COSMIC genome location is not accurate for duplications. Dropping duplications in a copy of the csv file.")
            mutations = drop_duplication_mutations(mutations, mutations_path_no_duplications, logger)  # COSMIC incorrectly records genome positions of duplications
        else:
            mutations = pd.read_csv(mutations_path_no_duplications)

        seq_id_column = supported_databases_and_corresponding_reference_sequence_type[mutations_original]["column_names"]["seq_id_genome_column"]  # "chromosome"
        var_column = supported_databases_and_corresponding_reference_sequence_type[mutations_original]["column_names"]["var_genome_column"]  # "mutation_genome"

    elif sequences == "cds" or sequences.endswith(supported_databases_and_corresponding_reference_sequence_type[mutations_original]["sequence_file_names"]["cds"].replace("GRCH_NUMBER", grch)):  # covers whether sequences == "cds" or sequences == "PATH/TO/Homo_sapiens.GRCh37.cds.all.fa"
        seq_id_column = supported_databases_and_corresponding_reference_sequence_type[mutations_original]["column_names"]["seq_id_cds_column"]  # "seq_ID"
        var_column = supported_databases_and_corresponding_reference_sequence_type[mutations_original]["column_names"]["var_cds_column"]  # "mutation"

    var_id_column = supported_databases_and_corresponding_reference_sequence_type[mutations_original]["column_names"]["var_id_column"] if var_id_column is not None else None  # use the id column if the user wanted to; otherwise keep as default
    return mutations,mutations_path,seq_id_column,var_column,var_id_column,columns_to_keep


def download_cosmic_sequences(sequences, seq_id_column, gtf, gtf_transcript_id_column, reference_out_dir, cosmic_version, mutations, grch, logger):
    if grch == "37":
        gget_ref_species = "human_grch37"
    elif grch == "38":
        gget_ref_species = "human"
    else:
        gget_ref_species = grch
    
    ensembl_version = supported_databases_and_corresponding_reference_sequence_type[mutations]["database_version_to_reference_release"][cosmic_version]
    reference_out_sequences = f"{reference_out_dir}/ensembl_grch{grch}_release{ensembl_version}"  # matches vk info

    sequences_download_command = supported_databases_and_corresponding_reference_sequence_type[mutations]["sequence_download_commands"][sequences]
    sequences_download_command = sequences_download_command.replace("OUT_DIR", reference_out_sequences)
    sequences_download_command = sequences_download_command.replace("ENSEMBL_VERSION", ensembl_version)
    sequences_download_command = sequences_download_command.replace("SPECIES", gget_ref_species)

    genome_file = supported_databases_and_corresponding_reference_sequence_type[mutations]["sequence_file_names"]["genome"]
    genome_file = genome_file.replace("GRCH_NUMBER", grch)
    genome_file = f"{reference_out_sequences}/{genome_file}"

    gtf_file = supported_databases_and_corresponding_reference_sequence_type[mutations]["sequence_file_names"]["gtf"]
    gtf_file = gtf_file.replace("GRCH_NUMBER", grch)
    gtf_file = f"{reference_out_sequences}/{gtf_file}"

    cds_file = supported_databases_and_corresponding_reference_sequence_type[mutations]["sequence_file_names"]["cds"]
    cds_file = cds_file.replace("GRCH_NUMBER", grch)
    cds_file = f"{reference_out_sequences}/{cds_file}"

    cdna_file = supported_databases_and_corresponding_reference_sequence_type[mutations]["sequence_file_names"]["cdna"]
    cdna_file = cdna_file.replace("GRCH_NUMBER", grch)
    cdna_file = f"{reference_out_sequences}/{cdna_file}"

    files_to_download_list = []
    if sequences == "genome" and not os.path.isfile(genome_file):
        files_to_download_list.append("dna")
    if gtf and not os.path.isfile(gtf):
        gtf = gtf_file
        gtf_transcript_id_column = seq_id_column
        if not os.path.isfile(gtf):  # now that I have overridden the user-provided gtf with my gtf
            files_to_download_list.append("gtf")
    if (sequences == "cdna" or sequences == "cds") and not os.path.isfile(cds_file):
        files_to_download_list.append("cds")
    if sequences == "cdna" and not os.path.isfile(cdna_file):
        files_to_download_list.append("cdna")

    files_to_download = ",".join(files_to_download_list)
    sequences_download_command = sequences_download_command.replace("FILES_TO_DOWNLOAD", files_to_download)

    sequences_download_command_list = sequences_download_command.split(" ")

    if files_to_download_list:  # means that at least 1 of the necessary files must be downloaded
        logger.warning("Downloading reference sequences with %s. Note that this requires curl >=7.73.0", " ".join(sequences_download_command_list))
        subprocess.run(sequences_download_command_list, check=True)
        if "dna" in files_to_download_list:
            subprocess.run(["gunzip", f"{genome_file}.gz"], check=True)
        if "gtf" in files_to_download_list:
            subprocess.run(["gunzip", f"{gtf_file}.gz"], check=True)
        if "cds" in files_to_download_list:
            subprocess.run(["gunzip", f"{cds_file}.gz"], check=True)
        if "cdna" in files_to_download_list:
            subprocess.run(["gunzip", f"{cdna_file}.gz"], check=True)

    if sequences == "genome":
        sequences = genome_file
    elif sequences == "cds":
        sequences = cds_file
    elif sequences == "cdna":
        sequences = cdna_file
    return sequences,gtf,gtf_transcript_id_column,genome_file,cds_file,cdna_file


def merge_gtf_transcript_locations_into_cosmic_csv(mutations, gtf_path, gtf_transcript_id_column, output_mutations_path=None):
    # mutations = mutations.copy()  # commented out to save time, but be careful not to make unwanted changes to the original df (in general, if I set df = myfunc(df), then I likely welcome any modifications to df within the function; if not, be especially careful)

    gtf_df = pd.read_csv(
        gtf_path,
        sep="\t",
        comment="#",
        header=None,
        names=[
            "seqname",
            "source",
            "feature",
            "start",
            "end",
            "score",
            "strand",
            "frame",
            "attribute",
        ],
    )

    if "strand" in mutations.columns:
        mutations.rename(columns={"strand": "strand_original"}, inplace=True)

    gtf_df = gtf_df[gtf_df["feature"] == "transcript"]

    gtf_df["transcript_id"] = gtf_df["attribute"].str.extract('transcript_id "([^"]+)"')

    if not len(gtf_df["transcript_id"]) == len(set(gtf_df["transcript_id"])):
        raise ValueError("Duplicate transcript_id values found!")

    # Filter out rows where transcript_id is NaN
    gtf_df = gtf_df.dropna(subset=["transcript_id"])

    gtf_df = gtf_df[["transcript_id", "start", "end", "strand"]].rename(
        columns={
            "transcript_id": gtf_transcript_id_column,
            "start": "start_transcript_position",
            "end": "end_transcript_position",
        }
    )

    merged_df = pd.merge(mutations, gtf_df, on=gtf_transcript_id_column, how="left")

    # Fill NaN values
    merged_df["start_transcript_position"] = merged_df["start_transcript_position"].fillna(0)
    merged_df["end_transcript_position"] = merged_df["end_transcript_position"].fillna(9999999)
    merged_df["strand"] = merged_df["strand"].fillna(".")

    if output_mutations_path is not None:
        merged_df.to_csv(output_mutations_path, index=False)

    return merged_df


def drop_duplication_mutations(input_mutations, output, mutation_column="mutation_genome"):
    if isinstance(input_mutations, str):
        df = pd.read_csv(input_mutations)
    elif isinstance(input_mutations, pd.DataFrame):
        df = input_mutations
    else:
        raise ValueError("input_mutations must be a string or a DataFrame.")

    # count number of duplications
    num_dup = df[mutation_column].str.contains("dup", na=False).sum()
    logger.info(f"Number of duplication mutations that have been dropped: {num_dup}")

    df_no_dup_mutations = df.loc[~(df[mutation_column].str.contains("dup"))]

    df_no_dup_mutations.to_csv(output, index=False)

    return df_no_dup_mutations


def improve_genome_strand_information(cosmic_reference_file_mutation_csv, mutation_genome_column_name="mutation_genome", output_mutations_path=None):
    if isinstance(cosmic_reference_file_mutation_csv, str):
        df = pd.read_csv(cosmic_reference_file_mutation_csv)
    elif isinstance(cosmic_reference_file_mutation_csv, pd.DataFrame):
        df = cosmic_reference_file_mutation_csv
    else:
        raise ValueError("cosmic_reference_file_mutation_csv must be a string or a DataFrame.")

    df["strand_modified"] = df["strand"].replace(".", "+")

    genome_nucleotide_position_pattern = r"g\.(\d+)(?:_(\d+))?[A-Za-z]*"

    extracted_numbers = df[mutation_genome_column_name].str.extract(genome_nucleotide_position_pattern)
    extracted_numbers[1] = extracted_numbers[1].fillna(extracted_numbers[0])
    df["GENOME_START"] = extracted_numbers[0]
    df["GENOME_STOP"] = extracted_numbers[1]

    def complement_substitution(actual_variant):
        return "".join(complement.get(nucleotide, "N") for nucleotide in actual_variant[:])

    def reverse_complement_insertion(actual_variant):
        return "".join(complement.get(nucleotide, "N") for nucleotide in actual_variant[::-1])

    df[["nucleotide_positions", "actual_variant"]] = df["mutation"].str.extract(mutation_pattern)

    minus_sub_mask = (df["strand_modified"] == "-") & (df["actual_variant"].str.contains(">"))
    ins_delins_mask = (df["strand_modified"] == "-") & (df["actual_variant"].str.contains("ins"))

    df["actual_variant_rc"] = df["actual_variant"]

    df.loc[minus_sub_mask, "actual_variant_rc"] = df.loc[minus_sub_mask, "actual_variant"].apply(complement_substitution)

    df.loc[ins_delins_mask, ["variant_type", "mut_nucleotides"]] = df.loc[ins_delins_mask, "actual_variant"].str.extract(r"(delins|ins)([A-Z]+)").values

    df.loc[ins_delins_mask, "mut_nucleotides_rc"] = df.loc[ins_delins_mask, "mut_nucleotides"].apply(reverse_complement_insertion)

    df.loc[ins_delins_mask, "actual_variant_rc"] = df.loc[ins_delins_mask, "variant_type"] + df.loc[ins_delins_mask, "mut_nucleotides_rc"]

    df["actual_variant_final"] = np.where(df["strand_modified"] == "+", df["actual_variant"], df["actual_variant_rc"])

    df[mutation_genome_column_name] = np.where(
        df["GENOME_START"] != df["GENOME_STOP"],
        "g." + df["GENOME_START"].astype(str) + "_" + df["GENOME_STOP"].astype(str) + df["actual_variant_final"],
        "g." + df["GENOME_START"].astype(str) + df["actual_variant_final"],
    )

    df.drop(
        columns=["GENOME_START", "GENOME_STOP", "nucleotide_positions", "actual_variant", "actual_variant_rc", "variant_type", "mut_nucleotides", "mut_nucleotides_rc", "actual_variant_final", "strand_modified"],
        inplace=True,
    )  # drop all columns exceptmutation_genome_column_name(and the original ones)

    if output_mutations_path:
        df.to_csv(output_mutations_path, index=False)

    return df


def get_last_vcrs_number(filename):
    import csv
    if not os.path.isfile(filename):
        return 1
    with open(filename, "r") as f:
        last_line = list(csv.reader(f))[-1][0]  # Read last row, first column

    # Extract number using regex
    match = re.search(r"vcrs_(\d+)", last_line)
    number = int(match.group(1)) if match else None
    return number



def merge_fasta_file_headers(input_fasta, use_IDs=False, id_to_header_csv_out=None):
    output_fasta = input_fasta.replace(".fa", "_merged.fa")
    
    # Create two temporary files for the intermediate steps.
    with tempfile.NamedTemporaryFile(delete=False, mode='w+', suffix='.tsv') as temp_tsv:
        temp_tsv_name = temp_tsv.name
    with tempfile.NamedTemporaryFile(delete=False, mode='w+', suffix='.tsv') as sorted_tsv:
        sorted_tsv_name = sorted_tsv.name
    with tempfile.NamedTemporaryFile(delete=False, suffix='.fasta') as merged_fasta:
        merged_fasta_name = merged_fasta.name

    try:
        # Step 1: Convert FASTA to a single-line, tab-separated file (sequence<tab>header)
        awk_cmd = f"""awk 'BEGIN {{FS="\\n"; RS=">"; ORS=""}} 
            NR > 1 {{
                split($0, lines, "\\n");
                header = lines[1];
                seq = "";
                for (i = 2; i <= length(lines); i++) {{
                    seq = seq lines[i];
                }}
                print seq "\\t" header "\\n";
            }}' {input_fasta} > {temp_tsv_name}"""
        subprocess.run(awk_cmd, shell=True, check=True, executable="/bin/bash")

        # Step 2: Sort the temporary file by sequence (first column)
        sort_cmd = f"sort -k1,1 -k2,2 {temp_tsv_name} > {sorted_tsv_name}"
        subprocess.run(sort_cmd, shell=True, check=True, executable="/bin/bash")

        
        step_3_output_fasta = merged_fasta_name if use_IDs else output_fasta
        # Step 3: Merge headers for identical sequences
        awk_merge_cmd = f"""awk -F'\\t' 'BEGIN {{ prevSeq = ""; mergedHeader = "" }}
            {{
                if ($1 == prevSeq) {{
                    mergedHeader = mergedHeader ";" $2;
                }} else {{
                    if (prevSeq != "") {{
                        print ">" mergedHeader "\\n" prevSeq;
                    }}
                    prevSeq = $1;
                    mergedHeader = $2;
                }}
            }}
            END {{
                if (prevSeq != "") {{
                    print ">" mergedHeader "\\n" prevSeq;
                }}
            }}' {sorted_tsv_name} > {step_3_output_fasta}"""
        subprocess.run(awk_merge_cmd, shell=True, check=True, executable="/bin/bash")

        if use_IDs:
            # Step 4: Count the number of merged FASTA records.
            count_cmd = f"grep -c '^>' {merged_fasta_name}"
            count_output = subprocess.check_output(count_cmd, shell=True, executable="/bin/bash").strip()
            total_records = int(count_output)
            num_digits = len(str(total_records))

            # Step 5: Replace headers with generated IDs and write a mapping CSV.
            # This AWK command processes the merged FASTA, replacing headers with new IDs,
            # and outputs a mapping (new ID, original merged header) to a CSV file.
            awk_id_cmd = f"""awk -v num_digits={num_digits} -v mapping_file="{id_to_header_csv_out}" 'BEGIN {{
                    count = 1;
                    print "vcrs_id,vcrs_header" > mapping_file;
                }}
                /^>/ {{
                    original = substr($0, 2);
                    new_id = sprintf("vcrs_%0*d", num_digits, count);
                    print ">" new_id;
                    print new_id "," original >> mapping_file;
                    count++;
                    next;
                }}
                {{
                    print;
                }}' {merged_fasta_name} > {output_fasta}"""
            subprocess.run(awk_id_cmd, shell=True, check=True, executable="/bin/bash")

        os.remove(input_fasta)
        os.rename(output_fasta, input_fasta)

    finally:
        # Delete the temporary files.
        os.remove(temp_tsv_name)
        os.remove(sorted_tsv_name)
        if os.path.exists(merged_fasta_name):
            os.remove(merged_fasta_name)