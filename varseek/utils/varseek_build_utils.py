import re
from collections import OrderedDict, defaultdict

import pandas as pd
import pyfastx
from Bio import SeqIO
from tqdm import tqdm

from varseek.constants import codon_to_amino_acid, mutation_pattern

tqdm.pandas()


# Function to ensure unique IDs
def generate_unique_ids(num_ids):
    num_digits = len(str(num_ids))
    generated_ids = [f"vcrs_{i+1:0{num_digits}}" for i in range(num_ids)]
    return list(generated_ids)


def convert_chromosome_value_to_int_when_possible(val):
    try:
        # Try to convert the value to a float, then to an int, and finally to a string
        return str(int(float(val)))
    except ValueError:
        # If conversion fails, keep the value as it is
        return val


def translate_sequence(sequence, start, end):
    amino_acid_sequence = ""
    for i in range(start, end, 3):
        codon = sequence[i:(i + 3)].upper()
        amino_acid = codon_to_amino_acid.get(codon, "X")  # Use 'X' for unknown or incomplete codons
        amino_acid_sequence += amino_acid

    return amino_acid_sequence


def wt_fragment_and_mutant_fragment_share_kmer(mutated_fragment: str, wildtype_fragment: str, k: int) -> bool:
    if len(mutated_fragment) <= k:
        return bool(mutated_fragment in wildtype_fragment)

    # else:
    for mutant_position in range(len(mutated_fragment) - k):
        mutant_kmer = mutated_fragment[mutant_position:(mutant_position + k)]
        if mutant_kmer in wildtype_fragment:
            # wt_position = wildtype_fragment.find(mutant_kmer)
            return True
    return False


def convert_mutation_cds_locations_to_cdna(input_csv_path, cdna_fasta_path, cds_fasta_path, output_csv_path):
    # Load the CSV
    if isinstance(input_csv_path, str):
        df = pd.read_csv(input_csv_path)
    elif isinstance(input_csv_path, pd.DataFrame):
        df = input_csv_path
    else:
        raise ValueError("input_csv_path must be a string or a pandas DataFrame")

    df_original = df.copy()

    # get rids of mutations that are uncertain, ambiguous, intronic, posttranslational
    uncertain_mutations = df["mutation"].str.contains(r"\?").sum()

    ambiguous_position_mutations = df["mutation"].str.contains(r"\(|\)").sum()

    intronic_mutations = df["mutation"].str.contains(r"\+|\-").sum()

    posttranslational_region_mutations = df["mutation"].str.contains(r"\*").sum()

    print("Removing unsupported mutation types")
    print(f"Uncertain mutations: {uncertain_mutations}")
    print(f"Ambiguous position mutations: {ambiguous_position_mutations}")
    print(f"Intronic mutations: {intronic_mutations}")
    print(f"Posttranslational region mutations: {posttranslational_region_mutations}")

    # Filter out bad mutations
    combined_pattern = re.compile(r"(\?|\(|\)|\+|\-|\*)")  # gets rids of mutations that are uncertain, ambiguous, intronic, posttranslational
    mask = df["mutation"].str.contains(combined_pattern)
    df = df[~mask]

    df[["nucleotide_positions", "actual_mutation"]] = df["mutation"].str.extract(mutation_pattern)

    split_positions = df["nucleotide_positions"].str.split("_", expand=True)

    df["start_mutation_position"] = split_positions[0]
    if split_positions.shape[1] > 1:
        df["end_mutation_position"] = split_positions[1].fillna(split_positions[0])
    else:
        df["end_mutation_position"] = df["start_mutation_position"]

    df.loc[df["end_mutation_position"].isna(), "end_mutation_position"] = df["start_mutation_position"]

    df[["start_mutation_position", "end_mutation_position"]] = df[["start_mutation_position", "end_mutation_position"]].astype(int)

    # Rename the mutation column
    df.rename(columns={"mutation": "mutation_cds"}, inplace=True)

    # Load the FASTA files
    cdna_seqs = SeqIO.to_dict(SeqIO.parse(cdna_fasta_path, "fasta"))
    cds_seqs = SeqIO.to_dict(SeqIO.parse(cds_fasta_path, "fasta"))

    def remove_transcript_version_number(seq_dict):
        new_seq_dict = {}
        for key in seq_dict:
            new_key = key.split(".")[0]
            new_seq_dict[new_key] = seq_dict[key]
        return new_seq_dict

    cdna_seqs = remove_transcript_version_number(cdna_seqs)
    cds_seqs = remove_transcript_version_number(cds_seqs)

    # Helper function to find starting position of CDS in cDNA
    def find_cds_position(cdna_seq, cds_seq):
        pos = cdna_seq.find(cds_seq)
        return pos if pos != -1 else None

    def count_leading_Ns(seq):
        return len(seq) - len(seq.lstrip("N"))

    number_bad = 0

    # Process each row
    for index, row in df.iterrows():
        seq_id = row["seq_ID"]
        if seq_id in cdna_seqs and seq_id in cds_seqs:
            cdna_seq = str(cdna_seqs[seq_id].seq)
            cds_seq = str(cds_seqs[seq_id].seq)
            number_of_leading_ns = count_leading_Ns(cdna_seq)
            cds_seq = cds_seq.strip("N")

            cds_start_pos = find_cds_position(cdna_seq, cds_seq)
            if cds_start_pos is not None:
                df.at[index, "start_mutation_position"] += cds_start_pos - number_of_leading_ns
                df.at[index, "end_mutation_position"] += cds_start_pos - number_of_leading_ns

                start = df.at[index, "start_mutation_position"]
                end = df.at[index, "end_mutation_position"]
                actual_mutation = row["actual_mutation"]

                if start == end:
                    df.at[index, "mutation"] = f"c.{start}{actual_mutation}"
                else:
                    df.at[index, "mutation"] = f"c.{start}_{end}{actual_mutation}"
            else:
                df.at[index, "mutation"] = None
                number_bad += 1
        else:
            df.at[index, "mutation"] = None
            number_bad += 1

    df.drop(
        columns=[
            "nucleotide_positions",
            "actual_mutation",
            "start_mutation_position",
            "end_mutation_position",
        ],
        inplace=True,
    )

    df.rename(columns={"mutation": "mutation_cdna", "mutation_cds": "mutation"}, inplace=True)  # new as of Feb 2025
    df_merged = df_original.merge(df[['mutation', 'mutation_cdna']], on='mutation', how='left')  # new as of Feb 2025

    # Write to new CSV
    df_merged.to_csv(output_csv_path, index=False)  # new as of Feb 2025 (replaced df.to_csv with df_merged.to_csv)

    print(f"Number of bad mutations: {number_bad}")
    print(f"Output written to {output_csv_path}")

    return df


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


def merge_genome_into_transcriptome_fasta(
    mutation_reference_file_fasta_transcriptome,
    mutation_reference_file_fasta_genome,
    mutation_reference_file_fasta_combined,
    cosmic_reference_file_mutation_csv,
):

    # TODO: make header fasta from id fasta with id:header dict

    mutant_reference_transcriptome = create_header_to_sequence_ordered_dict_from_fasta_after_semicolon_splitting(mutation_reference_file_fasta_transcriptome)
    mutant_reference_genome = create_header_to_sequence_ordered_dict_from_fasta_after_semicolon_splitting(mutation_reference_file_fasta_genome)

    cosmic_df = pd.read_csv(
        cosmic_reference_file_mutation_csv,
        usecols=["seq_ID", "mutation_cdna", "chromosome", "mutation_genome"],
    )
    cosmic_df["chromosome"] = cosmic_df["chromosome"].apply(convert_chromosome_value_to_int_when_possible)

    mutant_reference_genome_to_keep = OrderedDict()

    for header_genome, sequence_genome in mutant_reference_genome.items():
        seq_id_genome, mutation_id_genome = header_genome.split(":", 1)
        row_corresponding_to_genome = cosmic_df[(cosmic_df["chromosome"] == seq_id_genome) & (cosmic_df["mutation_genome"] == mutation_id_genome)]
        seq_id_transcriptome_corresponding_to_genome = row_corresponding_to_genome["seq_ID"].iloc[0]
        mutation_id_transcriptome_corresponding_to_genome = row_corresponding_to_genome["mutation_cdna"].iloc[0]
        header_transcriptome_corresponding_to_genome = f"{seq_id_transcriptome_corresponding_to_genome}:{mutation_id_transcriptome_corresponding_to_genome}"

        if header_transcriptome_corresponding_to_genome in mutant_reference_transcriptome:
            if mutant_reference_transcriptome[header_transcriptome_corresponding_to_genome] != sequence_genome:
                header_genome_transcriptome_style = f"unspliced{header_transcriptome_corresponding_to_genome}"  # TODO: change when I change unspliced notation
                mutant_reference_genome_to_keep[header_genome_transcriptome_style] = sequence_genome
        else:
            header_genome_transcriptome_style = f"unspliced{header_transcriptome_corresponding_to_genome}"  # TODO: change when I change unspliced notation
            mutant_reference_genome_to_keep[header_genome_transcriptome_style] = sequence_genome

    mutant_reference_combined = OrderedDict(mutant_reference_transcriptome)
    mutant_reference_combined.update(mutant_reference_genome_to_keep)

    mutant_reference_combined = join_keys_with_same_values(mutant_reference_combined)

    # initialize combined fasta file with transcriptome fasta
    with open(mutation_reference_file_fasta_combined, "w", encoding="utf-8") as fasta_file:
        for (
            header_transcriptome,
            sequence_transcriptome,
        ) in mutant_reference_combined.items():
            # write the header followed by the sequence
            fasta_file.write(f">{header_transcriptome}\n{sequence_transcriptome}\n")

    # TODO: make id fasta from header fasta with id:header dict

    print(f"Combined fasta file created at {mutation_reference_file_fasta_combined}")


def join_keys_with_same_values(original_dict):
    # Step 1: Group keys by their values
    grouped_dict = defaultdict(list)
    for key, value in original_dict.items():
        grouped_dict[value].append(key)

    # Step 2: Create the new OrderedDict with concatenated keys
    concatenated_dict = OrderedDict((";".join(keys), value) for value, keys in grouped_dict.items())

    return concatenated_dict
