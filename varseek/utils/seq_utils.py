import os
import ast
import json
import shutil
import tempfile
import random
import string
import math
import re
import subprocess
import requests
from tqdm import tqdm

tqdm.pandas()
import gzip
from collections import OrderedDict, defaultdict
from typing import Callable

import numpy as np
import pandas as pd
import anndata as ad
import scanpy as sc
import pysam

from scipy.sparse import csr_matrix
from bisect import bisect_left
from sklearn.metrics import euclidean_distances
from sklearn.neighbors import NearestNeighbors

from Bio import SeqIO
from Bio.Seq import Seq
import pyfastx
import gget
from varseek.constants import complement, codon_to_amino_acid, mutation_pattern, technology_barcode_and_umi_dict
from varseek.utils.visualization_utils import (
    print_column_summary_stats,
    plot_histogram_notebook_1,
    plot_basic_bar_plot_from_dict,
    plot_descending_bar_plot,
    plot_histogram_of_nearby_mutations_7_5,
    plot_knn_tissue_frequencies,
    plot_ascending_bar_plot_of_cluster_distances,
    plot_jaccard_bar_plot,
)

# from varseek.utils.io_utils import read_fasta, read_fastq, process_sam_file, write_temp_fa

# dlist_pattern_utils = re.compile(r"^(\d+)_(\d+)$")   # re.compile(r"^(unspliced)?(\d+)(;(unspliced)?\d+)*_(\d+)$")   # re.compile(r"^(unspliced)?(ENST\d+:(?:c\.|g\.)\d+(_\d+)?([a-zA-Z>]+))(;(unspliced)?ENST\d+:(?:c\.|g\.)\d+(_\d+)?([a-zA-Z>]+))*_\d+$")
# TODO: change when I change unspliced notaiton
dlist_pattern_utils = re.compile(
    r"^(\d+)_(\d+)$"  # First pattern: digits underscore digits
    r"|^(unspliced)?(\d+)(;(unspliced)?\d+)*_(\d+)$"  # Second pattern: optional unspliced, digits, underscore, digits
    r"|^(unspliced)?(ENST\d+:(?:c\.|g\.)\d+(_\d+)?([a-zA-Z>]+))(;(unspliced)?ENST\d+:(?:c\.|g\.)\d+(_\d+)?([a-zA-Z>]+))*_\d+$"  # Third pattern: complex ENST pattern
)


def read_fasta(file_path, semicolon_split=False):
    with open(file_path, "r") as file:
        header = None
        sequence_lines = []
        for line in file:
            line = line.strip()
            if line.startswith(">"):
                if header is not None:
                    # Yield the previous entry
                    sequence = "".join(sequence_lines)
                    if semicolon_split:
                        for sub_header in header.split(";"):
                            yield sub_header, sequence
                    else:
                        yield header, sequence
                # Start a new record
                header = line[1:]  # Remove '>' character
                sequence_lines = []
            else:
                sequence_lines.append(line)
        # Yield the last entry after the loop ends
        if header is not None:
            sequence = "".join(sequence_lines)
            if semicolon_split:
                for sub_header in header.split(";"):
                    yield sub_header, sequence
            else:
                yield header, sequence


def get_header_set_from_fasta(synthetic_read_fa):
    return {header for header, _ in read_fasta(synthetic_read_fa)}


def create_mutant_t2g(
    mutation_reference_file_fasta, out="./cancer_mutant_reference_t2g.txt"
):
    if not os.path.exists(out):
        with open(mutation_reference_file_fasta, "r") as fasta, open(out, "w") as t2g:
            for line in fasta:
                if line.startswith(">"):
                    header = line[1:].strip()
                    t2g.write(f"{header}\t{header}\n")
    else:
        print(f"{out} already exists")


def load_t2g_as_dict(file_path):
    t2g_dict = {}
    with open(file_path, "r") as file:
        for line in file:
            # Strip any whitespace and split by tab
            key, value = line.strip().split("\t")
            t2g_dict[key] = value
    return t2g_dict


def process_sam_file(sam_file):
    with open(sam_file, "r") as sam:
        for line in sam:
            if line.startswith("@"):
                continue

            fields = line.split("\t")
            yield fields


def fasta_to_fastq(
    fasta_file,
    fastq_file,
    quality_score="I",
    k=None,
    add_noise=False,
    average_quality_for_noisy_reads=30,
):
    """
    Convert a FASTA file to a FASTQ file with a default quality score

    :param fasta_file: Path to the input FASTA file.
    :param fastq_file: Path to the output FASTQ file.
    :param quality_score: Default quality score to use for each base. Default is "I" (high quality).
    """
    with open(fastq_file, "w") as fastq:
        for sequence_id, sequence in read_fasta(fasta_file):
            if k is None or k >= len(sequence):
                if add_noise:
                    quality_scores = generate_noisy_quality_scores(
                        sequence, average_quality_for_noisy_reads
                    )
                else:
                    quality_scores = quality_score * len(sequence)
                fastq.write(f"@{sequence_id}\n")
                fastq.write(f"{sequence}\n")
                fastq.write("+\n")
                fastq.write(f"{quality_scores}\n")
            else:
                for i in range(len(sequence) - k + 1):
                    kmer = sequence[i : i + k]
                    if add_noise:
                        quality_scores = generate_noisy_quality_scores(
                            kmer, average_quality_for_noisy_reads
                        )
                    else:
                        quality_scores = quality_score * k

                    fastq.write(f"@{sequence_id}_{i}\n")
                    fastq.write(f"{kmer}\n")
                    fastq.write("+\n")
                    fastq.write(f"{quality_scores}\n")


def read_fastq(fastq_file):
    if fastq_file.endswith(".gz"):
        file = gzip.open(fastq_file, "rt")  # 'rt' mode is for reading text
    else:
        file = open(fastq_file, "r")  # 'r' mode is for reading text

    with file:
        while True:
            header = file.readline().strip()
            sequence = file.readline().strip()
            plus_line = file.readline().strip()
            quality = file.readline().strip()

            if not header:
                break

            yield header, sequence, plus_line, quality


import gzip
import shutil


def concatenate_fastqs(*input_files, output_file=None, delete_original_files=False):
    """
    Concatenate a variable number of FASTQ files (gzipped or not) into a single output file.

    Parameters:
    - output_file (str): Path to the output file.
    - *input_files (str): Paths to the input FASTQ files to concatenate.
    """
    # Detect if the files are gzipped based on file extension of the first input
    if not input_files:
        raise ValueError("No input files provided.")

    if not output_file:
        parts_filename = input_files[0].split(".", 1)
        input_files_dir = os.path.dirname(input_files[0])
        output_file = f"{input_files_dir}/combined.{parts_filename[1]}"

    input_files_space_separated = " ".join(list(input_files))
    cat_command = f"cat {input_files_space_separated} > {output_file}"
    subprocess.run(cat_command, shell=True, check=True)

    if delete_original_files:
        for file in list(input_files):
            os.remove(file)

    # is_gzipped = input_files[0].endswith(".gz")
    # open_func = gzip.open if is_gzipped else open
    # with open_func(output_file, 'wt' if is_gzipped else 'w') as outfile:
    #     for file in input_files:
    #         with open_func(file, 'rt' if is_gzipped else 'r') as infile:
    #             shutil.copyfileobj(infile, outfile)

    return output_file


def write_temp_fa(mcrs_fa):
    mcrs_fa_original = mcrs_fa
    mcrs_fa = mcrs_fa.replace(".fa", "_temp.fa")
    with open(mcrs_fa, "w") as outfile:
        for header, sequence in read_fasta(mcrs_fa_original):
            new_header = header.replace(">", "x")
            outfile.write(f">{new_header}\n{sequence}\n")

    return mcrs_fa, mcrs_fa_original


def fast_reverse_complement(sequence):
    complement_map = str.maketrans("ACGTNacgtn.", "TGCANtgcan.")
    return sequence.translate(complement_map)[::-1]


def reverse_complement(sequence):
    return "".join(complement.get(nucleotide, "N") for nucleotide in sequence[::-1])


# Function to generate a random numeric ID of specified length
def generate_random_id(id_length):
    # return
    # # Ensure no leading 0s
    # min_val = 10**(id_length - 1)
    # max_val = 10**id_length - 1
    # return "seq_" + str(random.randint(min_val, max_val))
    random_number_string = "".join(random.choices(string.digits, k=id_length))
    return "seq_" + random_number_string


def calculate_sufficient_id_length(num_mutations):
    number_of_mutations_scaled = num_mutations * 10000000000
    id_length = 10 ** math.floor(
        math.log10(number_of_mutations_scaled)
    )  # have a sufficient number of digits for the id, such that there is a <0.0000001% chance of collisions
    id_length = math.log10(id_length)
    return id_length


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
        codon = sequence[i : i + 3].upper()
        amino_acid = codon_to_amino_acid.get(
            codon, "X"
        )  # Use 'X' for unknown or incomplete codons
        amino_acid_sequence += amino_acid

    return amino_acid_sequence


def wt_fragment_and_mutant_fragment_share_kmer(
    mutated_fragment: str, wildtype_fragment: str, k: int
) -> bool:
    if len(mutated_fragment) <= k:
        if mutated_fragment in wildtype_fragment:
            return True
        else:
            return False
    else:
        for mutant_position in range(len(mutated_fragment) - k):
            mutant_kmer = mutated_fragment[mutant_position : mutant_position + k]
            if mutant_kmer in wildtype_fragment:
                # wt_position = wildtype_fragment.find(mutant_kmer)
                return True
        return False


def convert_mutation_cds_locations_to_cdna(
    input_csv_path, cdna_fasta_path, cds_fasta_path, output_csv_path
):
    # Load the CSV
    df = pd.read_csv(input_csv_path)

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
    combined_pattern = re.compile(
        r"(\?|\(|\)|\+|\-|\*)"
    )  # gets rids of mutations that are uncertain, ambiguous, intronic, posttranslational
    mask = df["mutation"].str.contains(combined_pattern)
    df = df[~mask]

    df[["nucleotide_positions", "actual_mutation"]] = df["mutation"].str.extract(
        mutation_pattern
    )

    split_positions = df["nucleotide_positions"].str.split("_", expand=True)

    df["start_mutation_position"] = split_positions[0]
    if split_positions.shape[1] > 1:
        df["end_mutation_position"] = split_positions[1].fillna(split_positions[0])
    else:
        df["end_mutation_position"] = df["start_mutation_position"]

    df.loc[df["end_mutation_position"].isna(), "end_mutation_position"] = df[
        "start_mutation_position"
    ]

    df[["start_mutation_position", "end_mutation_position"]] = df[
        ["start_mutation_position", "end_mutation_position"]
    ].astype(int)

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
                df.at[index, "start_mutation_position"] += (
                    cds_start_pos - number_of_leading_ns
                )
                df.at[index, "end_mutation_position"] += (
                    cds_start_pos - number_of_leading_ns
                )

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

    # Write to new CSV
    df.to_csv(output_csv_path, index=False)

    print(f"Number of bad mutations: {number_bad}")
    print(f"Output written to {output_csv_path}")


def find_matching_sequences_through_fasta(file_path, ref_sequence):
    headers_of_matching_sequences = []
    for header, sequence in read_fasta(file_path):
        if sequence == ref_sequence:
            headers_of_matching_sequences.append(header)

    return headers_of_matching_sequences


def create_header_to_sequence_ordered_dict_from_fasta_after_semicolon_splitting(
    input_fasta,
):
    mutant_reference = OrderedDict()
    for mutant_reference_header, mutant_reference_sequence in read_fasta(input_fasta):
        mutant_reference_header_individual_list = mutant_reference_header.split(";")
        for (
            mutant_reference_header_individual
        ) in mutant_reference_header_individual_list:
            mutant_reference[mutant_reference_header_individual] = (
                mutant_reference_sequence
            )
    return mutant_reference


def create_header_to_sequence_ordered_dict_from_fasta_WITHOUT_semicolon_splitting(input_fasta, low_memory = False):
    if low_memory:
        mutant_reference = pyfastx.Fasta(input_fasta, build_index=True)
    else:
        mutant_reference = OrderedDict()
        for mutant_reference_header, mutant_reference_sequence in read_fasta(input_fasta):
            mutant_reference[mutant_reference_header] = mutant_reference_sequence
    return mutant_reference


def merge_genome_into_transcriptome_fasta(
    mutation_reference_file_fasta_transcriptome,
    mutation_reference_file_fasta_genome,
    mutation_reference_file_fasta_combined,
    cosmic_reference_file_mutation_csv,
):

    # TODO: make header fasta from id fasta with id:header dict

    mutant_reference_transcriptome = (
        create_header_to_sequence_ordered_dict_from_fasta_after_semicolon_splitting(
            mutation_reference_file_fasta_transcriptome
        )
    )
    mutant_reference_genome = (
        create_header_to_sequence_ordered_dict_from_fasta_after_semicolon_splitting(
            mutation_reference_file_fasta_genome
        )
    )

    cosmic_df = pd.read_csv(
        cosmic_reference_file_mutation_csv,
        usecols=["seq_ID", "mutation", "chromosome", "mutation_genome"],
    )
    cosmic_df["chromosome"] = cosmic_df["chromosome"].apply(
        convert_chromosome_value_to_int_when_possible
    )

    mutant_reference_genome_to_keep = OrderedDict()

    for header_genome, sequence_genome in mutant_reference_genome.items():
        seq_id_genome, mutation_id_genome = header_genome.split(":", 1)
        row_corresponding_to_genome = cosmic_df[
            (cosmic_df["chromosome"] == seq_id_genome)
            & (cosmic_df["mutation_genome"] == mutation_id_genome)
        ]
        seq_id_transcriptome_corresponding_to_genome = row_corresponding_to_genome[
            "seq_ID"
        ].iloc[0]
        mutation_id_transcriptome_corresponding_to_genome = row_corresponding_to_genome[
            "mutation"
        ].iloc[0]
        header_transcriptome_corresponding_to_genome = f"{seq_id_transcriptome_corresponding_to_genome}:{mutation_id_transcriptome_corresponding_to_genome}"

        if (
            header_transcriptome_corresponding_to_genome
            in mutant_reference_transcriptome
        ):
            if (
                mutant_reference_transcriptome[
                    header_transcriptome_corresponding_to_genome
                ]
                != sequence_genome
            ):
                header_genome_transcriptome_style = f"unspliced{header_transcriptome_corresponding_to_genome}"  # TODO: change when I change unspliced notaiton
                mutant_reference_genome_to_keep[header_genome_transcriptome_style] = (
                    sequence_genome
                )
        else:
            header_genome_transcriptome_style = f"unspliced{header_transcriptome_corresponding_to_genome}"  # TODO: change when I change unspliced notaiton
            mutant_reference_genome_to_keep[header_genome_transcriptome_style] = (
                sequence_genome
            )

    mutant_reference_combined = OrderedDict(mutant_reference_transcriptome)
    mutant_reference_combined.update(mutant_reference_genome_to_keep)

    mutant_reference_combined = join_keys_with_same_values(mutant_reference_combined)

    # initialize combined fasta file with transcriptome fasta
    with open(mutation_reference_file_fasta_combined, "w") as fasta_file:
        for (
            header_transcriptome,
            sequence_transcriptome,
        ) in mutant_reference_combined.items():
            # write the header followed by the sequence
            fasta_file.write(f">{header_transcriptome}\n{sequence_transcriptome}\n")

    # TODO: make id fasta from header fasta with id:header dict

    print(f"Combined fasta file created at {mutation_reference_file_fasta_combined}")


def remove_dlist_duplicates(input_file, output_file=None):
    if output_file is None:
        output_file = input_file + ".tmp"  # Write to a temporary file

    # TODO: make header fasta from id fasta with id:header dict

    sequence_to_headers_dict = {}
    with open(input_file, "r") as file:
        while True:
            header = file.readline().strip()
            sequence = file.readline().strip()

            if not header:
                break

            if sequence in sequence_to_headers_dict:
                header = header[1:]  # Remove '>' character
                if header not in sequence_to_headers_dict[sequence]:
                    sequence_to_headers_dict[sequence] += f"~{header}"
            else:
                sequence_to_headers_dict[sequence] = header

    with open(output_file, "w") as file:
        for sequence, header in sequence_to_headers_dict.items():
            file.write(f"{header}\n{sequence}\n")

    # TODO: make id fasta from header fasta with id:header dict

    if output_file == input_file + ".tmp":
        os.replace(output_file, input_file)


def capitalize_sequences(input_file, output_file=None):
    if output_file is None:
        output_file = input_file + ".tmp"  # Write to a temporary file
    with open(input_file, "r") as infile, open(output_file, "w") as outfile:
        for line in infile:
            if line.startswith(">"):
                outfile.write(line)
            else:
                outfile.write(line.upper())

    if output_file == input_file + ".tmp":
        os.replace(output_file, input_file)


def parse_sam_and_extract_sequences(
    sam_file,
    ref_genome_file,
    output_fasta_file,
    k=31,
    dfk_length=None,
    capitalize=True,
    remove_duplicates=False,
    check_for_bad_cigars=True,
):
    if dfk_length is None:
        dfk_length = k + 2

    def read_reference_genome(ref_genome_file):
        genome = {}
        with open(ref_genome_file, "r") as f:
            current_chrom = None
            current_seq = []
            for line in f:
                if line.startswith(">"):
                    if current_chrom:
                        genome[current_chrom] = "".join(current_seq)
                    current_chrom = line[1:].split()[0]
                    current_seq = []
                else:
                    current_seq.append(line.strip())
            if current_chrom:
                genome[current_chrom] = "".join(current_seq)

        return genome

    ref_genome = read_reference_genome(ref_genome_file)

    with open(sam_file, "r") as f, open(output_fasta_file, "w") as dlist_fasta:
        bad_cigar = 0
        for line in f:
            if line.startswith("@"):
                continue  # Skip header lines
            parts = line.strip().split("\t")
            if parts[2] == "*":
                continue  # Skip unmapped reads or those not matching 31 bases

            if check_for_bad_cigars and (parts[5] != f"{k}="):
                bad_cigar += 1
                continue

            chromosome = parts[2]
            start_position = int(parts[3]) - 1
            end_position = start_position + k
            start_dfk_position = start_position - dfk_length
            end_dfk_position = end_position + dfk_length

            start_dfk_position = max(0, start_dfk_position)
            # end_dfk_position = min(len(ref_genome[chromosome]), end_dfk_position)  # not needed because python will grab the entire string if the end index is greater than the length of the string

            if chromosome in ref_genome:
                sequence = ref_genome[chromosome][start_dfk_position:end_dfk_position]
                # there may be duplicate headers in the file if the same k-mer aligns to multiple parts of the genome/transcriptome - but this shouldn't matter
                header = parts[0]
                if header and sequence:
                    dlist_fasta.write(f">{header}\n{sequence}\n")

        if check_for_bad_cigars:
            print(f"Skipped {bad_cigar} reads with bad CIGAR strings")

    if capitalize:
        print("Capitalizing sequences")
        capitalize_sequences(output_fasta_file)

    if remove_duplicates:  #!!! not fully working yet
        print("Removing duplicate sequences")
        remove_dlist_duplicates(output_fasta_file)


def combine_transcriptome_fasta(
    input_fasta, output_file, max_chunk_size=10000000, k=31
):
    joined_sequences = []
    current_size = 0
    current_chunk = 0

    with open(output_file, "w") as outfile:
        # Open the FASTA file and loop through each entry
        outfile.write(f">Joined_transcriptome_chunk_{current_chunk}\n")
        for _, sequence in read_fasta(input_fasta):
            current_size += len(sequence)
            if current_size <= max_chunk_size:
                joined_sequences.append(sequence)
            else:
                joined_sequences_string = ("N" * k).join(joined_sequences)  # type: ignore
                outfile.write(f"{joined_sequences_string}\n")
                joined_sequences = [sequence]
                current_size = len(sequence)
                current_chunk += 1
                outfile.write(f">Joined_transcriptome_chunk_{current_chunk}\n")

        # Write the remaining sequences to the final chunk
        joined_sequences_string = ("N" * k).join(joined_sequences)
        outfile.write(f"{joined_sequences_string}\n")

    print(f"Saved joined transcriptome to {output_file}")


def get_set_of_headers_from_sam(
    sam_file, id_to_header_csv=None, check_for_bad_cigars=False, k="", return_set=True
):
    sequence_names = []

    for fields in process_sam_file(sam_file):
        cigarstring = fields[5]
        if check_for_bad_cigars and (cigarstring != f"{k}="):
            continue

        sequence_name = fields[0]

        # alignment_position = int(fields[3]) + 1

        sequence_names.append(sequence_name)

    # Remove everything from the last underscore to the end of the string for each sequence name
    cleaned_sequence_names = [name.rsplit("_", 1)[0] for name in sequence_names]

    if id_to_header_csv is not None:
        id_to_header_dict = make_mapping_dict(id_to_header_csv, dict_key="id")
        cleaned_sequence_names = [
            id_to_header_dict[seq_id] for seq_id in cleaned_sequence_names
        ]

    if return_set:
        cleaned_sequence_names = set(cleaned_sequence_names)

    return cleaned_sequence_names


def filter_fasta(
    input_fasta, output_fasta=None, sequence_names_set=set(), keep="not_in_list"
):
    if output_fasta is None:
        output_fasta = input_fasta + ".tmp"  # Write to a temporary file

    os.makedirs(os.path.dirname(output_fasta), exist_ok=True)

    if keep == "not_in_list":
        condition = lambda header: header not in sequence_names_set
    elif keep == "in_list":
        condition = lambda header: header in sequence_names_set
    else:
        raise ValueError("Invalid value for 'keep' parameter")

    if keep == "not_in_list" and not sequence_names_set:
        print("No sequences to filter out")
        shutil.copyfile(input_fasta, output_fasta)
    else:
        with open(input_fasta, "r") as infile, open(output_fasta, "w") as outfile:
            write_entry = False
            for line in infile:
                if line.startswith(">"):
                    # Extract the header without the '>'
                    header = line[1:].strip()
                    if condition(header):
                        write_entry = True
                        outfile.write(line)
                    else:
                        write_entry = False
                elif write_entry:
                    outfile.write(line)

    if output_fasta == input_fasta + ".tmp":
        os.replace(output_fasta, input_fasta)


def find_genes_with_aligned_reads_for_kb_extract(adata_path, number_genes=None):
    # Load the AnnData object
    adata = ad.read_h5ad(adata_path)

    problematic_genes = adata.var[np.array(adata.X.sum(axis=0) >= 1)[0]].index.values

    if number_genes:
        problematic_genes = problematic_genes[:number_genes]

    problematic_genes_string = " ".join(problematic_genes)

    return problematic_genes_string


def split_qualities_based_on_sequence(nucleotide_sequence, quality_score_sequence):
    # Step 1: Split the original sequence by the delimiter and get the fragments
    fragments = nucleotide_sequence.split("N")

    # Step 2: Calculate the lengths of the fragments
    lengths = [len(fragment) for fragment in fragments]

    # Step 3: Use these lengths to split the associated sequence
    split_quality_score_sequence = []
    start = 0
    for length in lengths:
        split_quality_score_sequence.append(
            quality_score_sequence[start : start + length]
        )
        start += length + 1

    return split_quality_score_sequence


def phred_to_error_rate(phred_score):
    return 10 ** (-phred_score / 10)


def trim_edges_and_adaptors_off_fastq_reads(
    filename,
    filename_r2=None,
    filename_filtered=None,
    filename_filtered_r2=None,
    cut_mean_quality=13,
    qualified_quality_phred=None,
    unqualified_percent_limit=None,
    n_base_limit=None,
    length_required=None,
    fastp="fastp",
):

    output_dir = os.path.dirname(filename)

    # Define default output filenames if not provided
    if not filename_filtered:
        parts_filename = filename.split(".", 1)
        filename_filtered = f"{parts_filename[0]}_filtered.{parts_filename[1]}"

    try:
        fastp_command = [
            fastp,
            "-i",
            filename,
            "-o",
            filename_filtered,
            "--cut_front",
            "--cut_tail",
            "--cut_window_size",
            "4",
            "--cut_mean_quality",
            str(int(cut_mean_quality)),
            "-h",
            f"{output_dir}/fastp_report.html",
            "-j",
            f"{output_dir}/fastp_report.json",
        ]

        # Add optional parameters
        if qualified_quality_phred and unqualified_percent_limit:
            fastp_command += [
                "--qualified_quality_phred",
                str(int(qualified_quality_phred)),
                "--unqualified_percent_limit",
                str(int(unqualified_percent_limit)),
            ]
        else:
            fastp_command += [
                "--unqualified_percent_limit",
                str(100),
            ]  # * default is 40
        if n_base_limit and n_base_limit <= 50:
            fastp_command += ["--n_base_limit", str(int(n_base_limit))]
        else:
            fastp_command += ["--n_base_limit", str(50)]  # * default is 5; max is 50
        if length_required:
            fastp_command += ["--length_required", str(int(length_required))]
        else:
            fastp_command += ["--disable_length_filtering"]  # * default is 15

        # Paired-end handling
        if filename_r2:
            if not filename_filtered_r2:
                parts_filename_r2 = filename_r2.split(".", 1)
                filename_filtered_r2 = (
                    f"{parts_filename_r2[0]}_filtered.{parts_filename_r2[1]}"
                )

            fastp_command[3:3] = [
                "-I",
                filename_r2,
                "-O",
                filename_filtered_r2,
                "--detect_adapter_for_pe",
            ]

        # Run the command
        subprocess.run(fastp_command, check=True)
    except Exception as e:
        try:
            print(f"Error: {e}")
            print("fastp did not work. Trying seqtk")
            _ = trim_edges_of_fastq_reads_seqtk(
                filename,
                seqtk="seqtk",
                filename_filtered=filename_filtered,
                minimum_phred=cut_mean_quality,
                number_beginning=0,
                number_end=0,
            )
            if filename_r2:
                _ = trim_edges_of_fastq_reads_seqtk(
                    filename_r2,
                    seqtk="seqtk",
                    filename_filtered=filename_filtered_r2,
                    minimum_phred=cut_mean_quality,
                    number_beginning=0,
                    number_end=0,
                )
        except Exception as e:
            print(f"Error: {e}")
            print("seqtk did not work. Skipping QC")
            return filename, filename_r2

    return filename_filtered, filename_filtered_r2


# def trim_edges_and_adaptors_of_fastq_reads_depracated(filename, tool = "trimgalore", filename_r2 = None, filename_filtered = None, filename_filtered_r2 = None, minimum_phred = 13, min_len = 31, adaptor_fa = "TruSeq3"):
#     if tool == "trimgalore":
#         output_dir = os.path.dirname(filename)
#         trimgalore_command = f"trim_galore --quality {minimum_phred} --length {min_len} --output_dir {output_dir} {filename}"
#         if filename and filename_r2:
#             parts = trimgalore_command.split()  # Split the command into parts
#             parts.insert(1, "--paired")  # Insert "--paired" at the second position
#             parts.append(filename_r2)  # Append the second filename
#             trimgalore_command = " ".join(parts)  # Rejoin the parts into a single string
#         subprocess.run(trimgalore_command, shell=True, check=True)

#     elif tool == "trimmomatic":
#         parts = filename.split(".")
#         if filename and filename_r2:
#             assay = "PE"
#         else:
#             assay = "SE"

#         if not os.path.exists(adaptor_fa):
#             adaptor_fa = f"{adaptor_fa}-{assay}-2.fa"
#             download_adaptor_fa_command = f"curl -o {adaptor_fa} https://raw.githubusercontent.com/usadellab/Trimmomatic/main/adapters/{adaptor_fa}"
#             subprocess.run(download_adaptor_fa_command, shell=True, check=True)

#         if not filename_filtered:
#             filename_filtered = f"{parts[0]}_filtered." + ".".join(parts[1:])
#         if filename and filename_r2:
#             parts_r2 = filename_r2.split(".")
#             if not filename_filtered_r2:
#                 filename_filtered_r2 = f"{parts_r2[0]}_filtered." + ".".join(parts_r2[1:])
#             trimmomatic_command = f"java -jar trimmomatic-0.39.jar PE -phred33 {filename} {filename_r2} {filename_filtered} output_forward_unpaired.{".".join(parts[1:])} {filename_filtered_r2} output_reverse_unpaired.{".".join(parts_r2[1:])} ILLUMINACLIP-{adaptor_fa}:2:30:10:2:True LEADING:{minimum_phred} TRAILING:{minimum_phred} MINLEN:{min_len}"
#         else:
#             trimmomatic_command = f"java -jar trimmomatic-0.39.jar SE -phred33 {filename} {filename_filtered} ILLUMINACLIP-{adaptor_fa}:2:30:10:2:True LEADING:{minimum_phred} TRAILING:{minimum_phred} SLIDINGWINDOW:4:{minimum_phred} MINLEN:{min_len}"
#         subprocess.run(trimmomatic_command, shell=True, check=True)
#         if filename and filename_r2:
#             os.remove(f"output_forward_unpaired.{".".join(parts[1:])}")
#             os.remove(f"output_reverse_unpaired.{".".join(parts_r2[1:])}")


def trim_edges_of_fastq_reads_seqtk(
    filename,
    seqtk="seqtk",
    filename_filtered=None,
    minimum_phred=13,
    number_beginning=0,
    number_end=0,
):
    if filename_filtered is None:
        parts = filename.split(".", 1)
        filename_filtered = f"{parts[0]}_filtered.{parts[1]}"

    minimum_base_probability = phred_to_error_rate(minimum_phred)

    if number_beginning == 0 and number_end == 0:
        command = [seqtk, "trimfq", "-q", str(minimum_base_probability), filename]
    else:
        command = [
            seqtk,
            "trimfq",
            "-q",
            str(minimum_base_probability),
            "-b",
            str(number_beginning),
            "-e",
            str(number_end),
            filename,
        ]
    with open(filename_filtered, "w") as output_file:
        subprocess.run(command, stdout=output_file, check=True)
    return filename_filtered


# def replace_low_quality_base_with_N_and_split_fastq_reads_by_N(input_fastq_file, output_fastq_file = None, minimum_sequence_length=31, seqtk = None, minimum_base_quality = 20):
#     parts = input_fastq_file.split(".")
#     output_replace_low_quality_with_N = f"{parts[0]}_with_Ns." + ".".join(parts[1:])
#     replace_low_quality_base_with_N(input_fastq_file, filename_filtered = output_replace_low_quality_with_N, seqtk = seqtk, minimum_base_quality = minimum_base_quality)
#     split_fastq_reads_by_N(input_fastq_file, output_fastq_file = output_fastq_file, minimum_sequence_length = minimum_sequence_length)


def replace_low_quality_base_with_N(
    filename, filename_filtered=None, seqtk="seqtk", minimum_base_quality=20
):
    parts = filename.split(".", 1)
    if filename_filtered is None:
        filename_filtered = f"{parts[0]}_with_more_Ns.{parts[1]}"
    command = [
        seqtk,
        "seq",
        "-q",
        str(minimum_base_quality),
        "-n",
        "N",
        "-x",
        filename,
    ]  # to drop a read containing N, use -N
    command = " ".join(command)
    if ".gz" in parts[1]:
        command += f" | gzip > {filename_filtered}"
        # with open(filename_filtered, 'wb') as output_file:
        #     process = subprocess.Popen(command, stdout=subprocess.PIPE)
        #     subprocess.run(["gzip"], stdin=process.stdout, stdout=output_file, check=True)
        #     process.stdout.close()
        #     process.wait()
    else:
        command += f" > {filename_filtered}"
        # with open(filename_filtered, 'w') as output_file:
        #     subprocess.run(command, stdout=output_file, check=True)
    subprocess.run(command, shell=True, check=True)
    return filename_filtered


# TODO: write this
def check_if_read_has_index_and_umi_smartseq3(sequence):
    pass

def split_fastq_reads_by_N(
    input_fastq_file,
    output_fastq_file=None,
    minimum_sequence_length=31,
    technology="bulk",
    contains_barcodes_or_umis = False  # set to False for bulk and for the paired file of any single-cell technology
):
    if "smartseq" in technology.lower():
        barcode_key = "spacer"
    else:
        barcode_key = "barcode"
    
    if technology != "bulk" and contains_barcodes_or_umis:
        if technology_barcode_and_umi_dict[technology][f"{barcode_key}_end"] is not None:
            barcode_length = technology_barcode_and_umi_dict[technology][f"{barcode_key}_end"] - technology_barcode_and_umi_dict[technology][f"{barcode_key}tart"]
        else:
            barcode_length = 0

        if technology_barcode_and_umi_dict[technology]["umi_start"] is not None:
            umi_length = technology_barcode_and_umi_dict[technology]["umi_end"] - technology_barcode_and_umi_dict[technology]["umi_start"]
        else:
            umi_length = 0
        
        prefix_len = barcode_length + umi_length

    prefix_len_original = prefix_len
    
    parts = input_fastq_file.split(".", 1)

    is_gzipped = ".gz" in parts[1]
    open_func = gzip.open if is_gzipped else open

    if output_fastq_file is None:
        output_fastq_file = f"{parts[0]}_split_by_Ns.{parts[1]}"

    regex = re.compile(r"[^Nn]+")

    with open_func(output_fastq_file, "wt") as out_file:
        for header, sequence, plus_line, quality in read_fastq(input_fastq_file):
            header = header[1:]  # Remove '@' character
            if technology != "bulk" and contains_barcodes_or_umis:
                if technology == "SMARTSEQ3":
                    sc_read_has_index_and_umi = check_if_read_has_index_and_umi_smartseq3(sequence)  # TODO: write this
                    if not sc_read_has_index_and_umi:
                        prefix_len = 0

                barcode_and_umi_sequence = sequence[:prefix_len]
                sequence_without_barcode_and_umi = sequence[prefix_len:]
                barcode_and_umi_quality = quality[:prefix_len]
                quality_without_barcode_and_umi = quality[prefix_len:]
                
                prefix_len = prefix_len_original
            else:
                sequence_without_barcode_and_umi = sequence
                quality_without_barcode_and_umi = quality

            # Use regex to find all runs of non-"N" characters and their positions
            matches = list(regex.finditer(sequence_without_barcode_and_umi))

            # Extract sequence parts and their positions
            split_sequence = [match.group() for match in matches]
            positions = [(match.start(), match.end()) for match in matches]

            # Use the positions to split the quality scores
            split_qualities = [
                quality_without_barcode_and_umi[start:end] for start, end in positions
            ]

            if technology != "bulk" and contains_barcodes_or_umis:
                split_sequence = [
                    barcode_and_umi_sequence + sequence for sequence in split_sequence
                ]
                split_qualities = [
                    barcode_and_umi_quality + quality for quality in split_qualities
                ]

            number_of_subsequences = len(split_sequence)
            for i in range(number_of_subsequences):
                if len(split_sequence[i]) < minimum_sequence_length:
                    continue
                if number_of_subsequences == 1:
                    new_header_prefix = ""
                else:
                    new_header_prefix = f"nsplit{i}__"

                new_header = f"@{new_header_prefix}{header}"

                if plus_line == "+":
                    new_plus_line = plus_line
                else:
                    new_plus_line = f"{new_header_prefix}{plus_line}"

                out_file.write(
                    f"{new_header}\n{split_sequence[i]}\n{new_plus_line}\n{split_qualities[i]}\n"
                )

    print(f"Split reads written to {output_fastq_file}")

    return output_fastq_file


def count_number_of_nonidentical_mutation_fragments(
    input_object, id_to_header_csv=None
):
    i = 0
    if type(input_object) == str:
        if id_to_header_csv is not None:
            temp_header_fa = input_object.replace(".fa", "_with_headers.fa")
            swap_ids_for_headers_in_fasta(
                input_object, id_to_header_csv, out_fasta=temp_header_fa
            )  # * added
            input_object = temp_header_fa
        for header, _ in read_fasta(input_object):
            if ";" not in header:
                i += 1
        if id_to_header_csv is not None:
            os.remove(temp_header_fa)
    elif (
        type(input_object) == list
        or type(input_object) == tuple
        or type(input_object) == set
    ):
        if type(input_object) == list or type(input_object) == tuple:
            input_object = set(input_object)
        for header in input_object:
            if ";" not in header:
                i += 1
    return i


def parse_fastq(file_path):
    problematic_mutations = []
    with gzip.open(file_path, "rt") as file:  # 'rt' mode is for reading text
        while True:
            header = file.readline().strip()
            sequence = file.readline().strip()
            plus_line = file.readline().strip()
            quality = file.readline().strip()

            if not header:
                break

            problematic_mutations.append(header[1:])  # Remove '@' character

    return problematic_mutations


def remove_fasta_headers(input_file, output_file, keep_only_flanks=True, k=31):
    with open(input_file, "r") as infile, open(output_file, "w") as outfile:
        for line in infile:
            if line.startswith(">"):
                outfile.write(">\n")
            else:
                outfile.write(line)

    if keep_only_flanks:
        output_file_temp = output_file + ".tmp"

        with open(output_file, "r") as infile, open(output_file_temp, "w") as outfile:
            line = infile.readline()
            if ">" in line:
                outfile.write(">\n")
            else:
                outfile.write(f"{line[:k]}\n")
                outfile.write(">\n")
                outfile.write(f"{line[len(line) - k:]}\n")

        os.replace(output_file_temp, output_file)


def capitalize_sequences(input_file, output_file=None):
    if output_file is None:
        output_file = input_file + ".tmp"  # Write to a temporary file
    with open(input_file, "r") as infile, open(output_file, "w") as outfile:
        for line in infile:
            if line.startswith(">"):
                outfile.write(line)
            else:
                outfile.write(line.upper())

    if output_file == input_file + ".tmp":
        os.replace(output_file, input_file)


def extract_problematic_mutation_list(problematic_genes_string, kb_extract_out_dir):
    problematic_mutations = []
    for gene in problematic_genes_string.split():
        fastq_file = f"{kb_extract_out_dir}/{gene}/1.fastq.gz"
        problematic_mutations_specific = parse_fastq(fastq_file)
        problematic_mutations.extend(problematic_mutations_specific)

    problematic_mutations = list(set(problematic_mutations))

    return problematic_mutations


def drop_duplication_mutations(input, output, mutation_column="mutation_genome"):
    df = pd.read_csv(input)

    # count number of duplications
    num_dup = df[mutation_column].str.contains("dup", na=False).sum()
    print(f"Number of duplication mutations that have been dropped: {num_dup}")

    df_no_dup_mutations = df.loc[~(df[mutation_column].str.contains("dup"))]

    df_no_dup_mutations.to_csv(output, index=False)


def join_keys_with_same_values(original_dict):
    # Step 1: Group keys by their values
    grouped_dict = defaultdict(list)
    for key, value in original_dict.items():
        grouped_dict[value].append(key)

    # Step 2: Create the new OrderedDict with concatenated keys
    concatenated_dict = OrderedDict(
        (";".join(keys), value) for value, keys in grouped_dict.items()
    )

    return concatenated_dict


def sequence_match(mcrs_sequence, dlist_sequence, strandedness=False):
    if strandedness:
        # Check only forward strand
        return mcrs_sequence in dlist_sequence
    else:
        # Check both forward and reverse complement
        return (mcrs_sequence in dlist_sequence) or (
            mcrs_sequence in reverse_complement(dlist_sequence)
        )


def remove_mutations_which_are_a_perfect_substring_of_wt_reference_genome(
    mutation_reference_file_fasta,
    dlist_fasta_file,
    output_fasta=None,
    output_dlist=None,
    strandedness=False,
):
    # TODO: make header fasta from id fasta with id:header dict

    mutant_reference = (
        create_header_to_sequence_ordered_dict_from_fasta_WITHOUT_semicolon_splitting(
            mutation_reference_file_fasta
        )
    )

    headers_NOT_to_put_in_dlist = set()

    if output_fasta is None:
        output_fasta = mutation_reference_file_fasta + ".tmp"

    if output_dlist is None:
        output_dlist = dlist_fasta_file + ".tmp"

    i = 0
    for dlist_header, dlist_sequence in read_fasta(dlist_fasta_file):
        dlist_header_shortened = dlist_header.rsplit("_", 1)[0]
        if dlist_header_shortened in mutant_reference:
            if sequence_match(
                mutant_reference[dlist_header_shortened],
                dlist_sequence,
                strandedness=strandedness,
            ):
                # if mutant_reference[dlist_header_shortened] in dlist_sequence or mutant_reference[dlist_header_shortened] in reverse_complement(dlist_sequence):
                del mutant_reference[dlist_header_shortened]
                headers_NOT_to_put_in_dlist.add(dlist_header_shortened)
                semicolon_separated_headers = dlist_header_shortened.split(";")
                i += len(semicolon_separated_headers)

    with open(output_dlist, "w") as output_dlist:
        for dlist_header, dlist_sequence in read_fasta(dlist_fasta_file):
            if not dlist_header.rsplit("_", 1)[0] in headers_NOT_to_put_in_dlist:
                output_dlist.write(f">{dlist_header}\n{dlist_sequence}\n")

    with open(output_fasta, "w") as file:
        for header, sequence in mutant_reference.items():
            file.write(f">{header}\n{sequence}\n")

    # TODO: make id fasta from header fasta with id:header dict

    if output_fasta == mutation_reference_file_fasta + ".tmp":
        os.replace(output_fasta, mutation_reference_file_fasta)

    if output_dlist == dlist_fasta_file + ".tmp":
        os.replace(output_dlist, dlist_fasta_file)

    print(
        f"Removed {i} mutations which are a perfect substring of the wildtype reference genome"
    )


def select_contiguous_substring(sequence, kmer, read_length=150):
    sequence_length = len(sequence)
    kmer_length = len(kmer)

    # Find the starting position of the kmer in the sequence
    kmer_start = sequence.find(kmer)
    if kmer_start == -1:
        raise ValueError("The k-mer is not found in the sequence")

    # Determine the possible start positions for the 20-character substring
    min_start_position = max(0, kmer_start - (read_length - kmer_length))
    max_start_position = min(sequence_length - read_length, kmer_start)

    # Randomly select a start position within the valid range
    start_position = random.randint(min_start_position, max_start_position)

    # Extract the 20-character substring
    selected_substring = sequence[start_position : start_position + read_length]

    return selected_substring


def remove_Ns_fasta(fasta_file):
    fasta_file_temp = fasta_file + ".tmp"
    i = 0
    with open(fasta_file, "r") as infile, open(fasta_file_temp, "w") as outfile:
        for header, sequence in read_fasta(fasta_file):
            if "N" not in sequence.upper():
                outfile.write(f">{header}\n{sequence}\n")
            else:
                i += 1

    os.replace(fasta_file_temp, fasta_file)
    print(f"Removed {i} sequences with Ns from {fasta_file}")


def count_line_number_in_file(file):
    return sum(1 for _ in open(file))


def count_number_of_spliced_and_unspliced_headers(file):
    # TODO: make header fasta with id:header dict
    spliced_only_lines = 0
    unspliced_only_lines = 0
    spliced_and_unspliced_lines = 0
    for headers_concatenated, _ in read_fasta(file):
        headers_list = headers_concatenated.split(";")
        spliced = False
        unspliced = False
        for header in headers_list:
            if "unspliced" in header:  # TODO: change when I change unspliced notaiton
                unspliced = True
            else:
                spliced = True
        if spliced and unspliced:
            spliced_and_unspliced_lines += 1
        elif spliced:
            spliced_only_lines += 1
        elif unspliced:
            unspliced_only_lines += 1
        else:
            raise ValueError("No spliced or unspliced header found")

    # TODO: make id fasta from header fasta with id:header dict

    return spliced_only_lines, unspliced_only_lines, spliced_and_unspliced_lines


def check_if_header_is_in_set(headers_concatenated, header_set_from_mutation_fasta):
    for header in header_set_from_mutation_fasta:
        if headers_concatenated in header:
            return header  # Return the first match where it's a substring
    return headers_concatenated


def check_dlist_header(dlist_header, pattern):
    return bool(pattern.search(dlist_header))


def assert_proper_header(dlist_header, pattern):
    assert pattern.fullmatch(
        dlist_header
    ), f"Header {dlist_header} does not match the expected format"


def get_long_headers(fasta_file, length_threshold=250):
    return {
        header for header, _ in read_fasta(fasta_file) if len(header) > length_threshold
    }


def compute_unique_mutation_header_set(
    file,
    id_to_header_csv=None,
    mutation_fasta_file=None,
    dlist_style=False,
    remove_unspliced=True,
):
    if id_to_header_csv is not None:
        temp_header_fa = file.replace(".fa", "_with_headers.fa")
        swap_ids_for_headers_in_fasta(
            file, id_to_header_csv, out_fasta=temp_header_fa
        )  # * added
        file = temp_header_fa

    unique_headers = set()

    total_lines = sum(1 for _ in open(file)) // 2
    for headers_concatenated, _ in tqdm(
        read_fasta(file), total=total_lines, desc="Processing headers"
    ):
        headers_list = headers_concatenated.split("~")
        for header in headers_list:
            if dlist_style:
                if check_dlist_header(header, dlist_pattern_utils):
                    header = header.rsplit("_", 1)[0]

            individual_header_list = header.split(";")
            for individual_header in individual_header_list:
                if remove_unspliced:
                    if (
                        "unspliced" in individual_header
                    ):  # TODO: change when I change unspliced notaiton
                        individual_header = individual_header.split("unspliced")[
                            1
                        ]  # requires my unspliced mutation to have the format "unspliced{seq_id}_{mutation_id}"
                unique_headers.add(individual_header)

    if id_to_header_csv is not None:
        os.remove(temp_header_fa)

    return unique_headers


# def count_number_of_total_mutations(headers_list, id_to_header_csv=None):
#     if id_to_header_csv is not None:
#         temp_header_fa = file.replace(".fa", "_with_headers.fa")
#         swap_ids_for_headers_in_fasta(
#             file, id_to_header_csv, out_fasta=temp_header_fa
#         )  # * added
#         file = temp_header_fa
#     i = 0
#     for header in headers_list:
#         individual_header = header.split(";")
#         i += len(individual_header)

#     # TODO: make id fasta from header fasta with id:header dict
#     return i


def apply_enst_format(unique_mutations_genome, cosmic_reference_file_mutation_csv):
    # TODO: make header fasta with id:header dict
    unique_mutations_genome_enst_format = set()
    cosmic_df = pd.read_csv(
        cosmic_reference_file_mutation_csv,
        usecols=["seq_ID", "mutation", "chromosome", "mutation_genome"],
    )
    for header_genome in unique_mutations_genome:
        seq_id_genome, mutation_id_genome = header_genome.split(":", 1)
        row_corresponding_to_genome = cosmic_df[
            (cosmic_df["chromosome"] == seq_id_genome)
            & (cosmic_df["mutation_genome"] == mutation_id_genome)
        ]
        seq_id_transcriptome_corresponding_to_genome = row_corresponding_to_genome[
            "seq_ID"
        ].iloc[0]
        mutation_id_transcriptome_corresponding_to_genome = row_corresponding_to_genome[
            "mutation"
        ].iloc[0]
        header_genome = f"{seq_id_transcriptome_corresponding_to_genome}:{mutation_id_transcriptome_corresponding_to_genome}"
        unique_mutations_genome_enst_format.add(header_genome)
    # TODO: make header fasta with id:header dict
    return unique_mutations_genome_enst_format


def generate_kmers(sequence, k, strandedness=True):
    """Generate k-mers of length k from a sequence."""
    if strandedness:
        return [sequence[i : i + k] for i in range(len(sequence) - k + 1)]
    else:
        list_f = [sequence[i : i + k] for i in range(len(sequence) - k + 1)]
        sequence_rc = reverse_complement(sequence)
        list_rc = [sequence_rc[i : i + k] for i in range(len(sequence_rc) - k + 1)]
        return list_f + list_rc


# def generate_kmer_overlap_df_temp(sequences, k, mcrs_id_column="mcrs_id"):
#     # Dictionary to store all k-mers and the sequences they come from
#     results = []
#     kmer_to_sequences = defaultdict(set)

#     # Loop through each sequence and generate k-mers
#     # for seq_id, sequence in sequences.items():
#     for seq_id, sequence in tqdm(sequences.items(), desc="Generating k-mers", unit="sequence"):
#         kmers = generate_kmers(sequence, k)
#         for kmer in kmers:
#             kmer_to_sequences[kmer].add(seq_id)

#     # Create a list to store results

#     # Loop through each sequence again to check k-mer overlaps
#     # for seq_id, sequence in sequences.items():
#     for seq_id, sequence in tqdm(sequences.items(), desc="Checking overlaps", unit="sequence"):
#         kmers = generate_kmers(sequence, k)
#         overlapping_kmers = 0
#         distinct_sequences = set()
#         overlapping_kmers_list = []

#         for kmer in kmers:
#             if len(kmer_to_sequences[kmer]) > 1:  # Check if k-mer overlaps with any other sequence
#                 overlapping_kmers += 1
#                 distinct_sequences.update(kmer_to_sequences[kmer])
#                 overlapping_kmers_list.append(kmer)

#         # Remove the current sequence from the distinct sequences count
#         distinct_sequences.discard(seq_id)

#         # Store the results for this sequence
#         results.append(
#             {
#                 mcrs_id_column: seq_id,
#                 "number_of_kmers_with_overlap_to_other_mcrs_items_in_mcrs_reference": overlapping_kmers,
#                 "overlapping_kmers": overlapping_kmers_list,
#                 "number_of_mcrs_items_with_overlapping_kmers_in_mcrs_reference": len(distinct_sequences),
#                 "mcrs_items_with_overlapping_kmers_in_mcrs_reference": distinct_sequences,
#             }
#         )

#     # Convert results to a DataFrame
#     df = pd.DataFrame(results)

#     return df


# def count_kmer_overlaps(fasta_file, k=31, strandedness=False, mcrs_id_column="mcrs_id"):
#     """Count k-mer overlaps between sequences in the FASTA file."""
#     # Parse the fasta file and store sequences
#     sequences_f = {record.id: str(record.seq) for record in SeqIO.parse(fasta_file, "fasta")}
#     df = generate_kmer_overlap_df_temp(sequences_f, k, mcrs_id_column=mcrs_id_column)

#     if not strandedness:
#         sequences_rc = {record_id: reverse_complement(seq) for record_id, seq in sequences_f.items()}  # TODO: takes a long time
#         df_rc = generate_kmer_overlap_df_temp(sequences_rc, k, mcrs_id_column=mcrs_id_column)

#         # Concatenate the DataFrames
#         df_concat = pd.concat([df, df_rc], ignore_index=True)

#         # Identify the columns
#         int_columns = [
#             "number_of_kmers_with_overlap_to_other_mcrs_items_in_mcrs_reference",
#             "number_of_mcrs_items_with_overlapping_kmers_in_mcrs_reference",
#         ]
#         set_columns = [
#             "overlapping_kmers",
#             "mcrs_items_with_overlapping_kmers_in_mcrs_reference",
#         ]

#         # Define the union function for sets
#         def union_sets(series):
#             sets = [s for s in series if isinstance(s, set)]
#             return set.union(*sets) if sets else set()

#         # Create the aggregation dictionary
#         agg_dict = {col: "sum" for col in int_columns}
#         agg_dict.update({col: union_sets for col in set_columns})

#         # Group by mcrs_id_column and aggregate
#         df = df_concat.groupby(mcrs_id_column).agg(agg_dict).reset_index()

#     return df

from hashlib import md5


def hash_kmer(kmer):
    """Return the MD5 hash of a k-mer as a hexadecimal string."""
    return md5(kmer.encode("utf-8")).hexdigest()


def count_kmer_overlaps_new(
    fasta_file, k=31, strandedness=False, mcrs_id_column="mcrs_id"
):
    """Count k-mer overlaps between sequences in the FASTA file."""
    # Parse the FASTA file and store sequences
    id_and_sequence_list_of_tuples = [
        (record.id, str(record.seq)) for record in SeqIO.parse(fasta_file, "fasta")
    ]

    # Create a combined k-mer overlap dictionary
    kmer_to_seqids = defaultdict(set)
    for seq_id, sequence in tqdm(
        id_and_sequence_list_of_tuples, desc="Generating k-mers", unit="sequence"
    ):
        for kmer in generate_kmers(sequence, k, strandedness=strandedness):
            # kmer_to_seqids[kmer].add(seq_id)
            # TODO: erase the line above and try storing hashes of k-mers instead
            kmer_hash = hash_kmer(kmer)  # Hash the k-mer
            kmer_to_seqids[kmer_hash].add(seq_id)

    # Process forward sequences only, checking overlaps with both forward and reverse complement k-mers
    results = []
    for seq_id, sequence in tqdm(
        id_and_sequence_list_of_tuples, desc="Checking overlaps", unit="sequence"
    ):
        kmers = generate_kmers(sequence, k)
        overlapping_kmers = 0
        distinct_sequences_set = set()
        overlapping_kmers_set = set()

        for kmer in kmers:
            kmer_hash = hash_kmer(kmer)
            if strandedness:
                if len(kmer_to_seqids[kmer_hash]) > 1:
                    overlapping_kmers += 1
                    overlapping_kmers_set.add(kmer)
                    distinct_sequences_set.update(kmer_to_seqids[kmer_hash])
            else:
                kmer_rc = reverse_complement(kmer)
                kmer_rc_hash = hash_kmer(kmer_rc)
                if (
                    len(kmer_to_seqids[kmer_rc_hash]) > 1
                    or len(kmer_to_seqids[kmer_hash]) > 1
                ):
                    overlapping_kmers += 1
                    overlapping_kmers_set.add(kmer)
                    distinct_sequences_set.update(kmer_to_seqids[kmer_hash])
                    distinct_sequences_set.update(kmer_to_seqids[kmer_rc_hash])

        # Remove the current sequence from the distinct sequences count
        distinct_sequences_set.discard(seq_id)

        # Store results
        results.append(
            {
                mcrs_id_column: seq_id,
                "number_of_kmers_with_overlap_to_other_mcrs_items_in_mcrs_reference": overlapping_kmers,
                "overlapping_kmers": overlapping_kmers_set,
                "number_of_mcrs_items_with_overlapping_kmers_in_mcrs_reference": len(
                    distinct_sequences_set
                ),
                "mcrs_items_with_overlapping_kmers_in_mcrs_reference": distinct_sequences_set,
            }
        )

    # Convert results to a DataFrame
    df = pd.DataFrame(results)

    return df


# Not used
def convert_nonsemicolon_headers_to_semicolon_joined_headers(
    nonsemicolon_read_headers_set, semicolon_reference_headers_set
):
    # Step 1: Initialize the mapping dictionary
    component_to_item = {}

    # Step 2: Build the mapping from components to items in set2
    for item in semicolon_reference_headers_set:
        components = item.split(";")
        for component in components:
            if (
                component in nonsemicolon_read_headers_set
                and component not in component_to_item
            ):
                component_to_item[component] = item

    # Step 3: Create set1_updated using the mapped items
    semicolon_read_headers_set = set(component_to_item.values())

    return semicolon_read_headers_set


def create_read_header_to_reference_header_mapping_df(
    gget_mutate_reference_headers_set, mutation_df_synthetic_read_headers_set
):
    read_to_reference_header_mapping = {}

    for read in tqdm(gget_mutate_reference_headers_set, desc="Processing reads"):
        if read in mutation_df_synthetic_read_headers_set:
            read_to_reference_header_mapping[read] = read
        else:
            for reference_item in gget_mutate_reference_headers_set:
                if read in reference_item:
                    read_to_reference_header_mapping[read] = reference_item
                    break

    df = pd.DataFrame(
        list(gget_mutate_reference_headers_set.items()),
        columns=["reference_header", "read_header"],
    )

    return df


def create_df_of_dlist_headers(dlist_path, k=None, header_column_name="mcrs_id"):
    dlist_headers_list_updated = get_set_of_headers_from_sam(
        dlist_path, k=k, return_set=False
    )

    df = pd.DataFrame(
        dlist_headers_list_updated, columns=[header_column_name]
    ).drop_duplicates()

    df[f"number_of_alignments_to_normal_human_reference"] = df[header_column_name].map(
        pd.Series(dlist_headers_list_updated).value_counts()
    )

    df["dlist"] = True

    return df


def load_splice_junctions_from_gtf(gtf_file):
    """
    Load splice junction positions from a GTF file.

    Parameters:
    - gtf_file: Path to the GTF file.

    Returns:
    - splice_junctions: Dictionary mapping chromosomes to sorted lists of splice junction positions.
    """
    # Columns in GTF files are typically:
    # seqname, source, feature, start, end, score, strand, frame, attribute
    col_names = [
        "seqname",
        "source",
        "feature",
        "start",
        "end",
        "score",
        "strand",
        "frame",
        "attribute",
    ]

    # Read only exon features as they contain splice junctions
    gtf_df = pd.read_csv(
        gtf_file,
        sep="\t",
        comment="#",
        names=col_names,
        dtype={"seqname": str, "start": int, "end": int},
    )

    # Filter for exons
    exons_df = gtf_df[gtf_df["feature"] == "exon"]

    # Initialize dictionary to hold splice junctions per chromosome
    splice_junctions = {}

    # Group exons by chromosome
    for chrom, group in exons_df.groupby("seqname"):
        # Extract start and end positions
        positions = set(group["start"]).union(set(group["end"]))
        # Sort positions for efficient searching
        splice_junctions[chrom] = sorted(positions)

    return splice_junctions


def find_closest_distance(position, positions_list):
    """
    Find the minimal distance between a position and a sorted list of positions.

    Parameters:
    - position: The position to compare.
    - positions_list: Sorted list of positions.

    Returns:
    - min_distance: The minimal distance.
    """
    index = bisect_left(positions_list, position)
    min_distance = None

    if index == 0:
        min_distance = abs(positions_list[0] - position)
    elif index == len(positions_list):
        min_distance = abs(positions_list[-1] - position)
    else:
        prev_pos = positions_list[index - 1]
        next_pos = positions_list[index]
        min_distance = min(abs(prev_pos - position), abs(next_pos - position))

    return min_distance


def compute_distance_to_closest_splice_junction(
    mutation_metadata_df_exploded,
    reference_genome_gtf,
    columns_to_explode=None,
    near_splice_junction_threshold=10,
):
    """
    Compute the distance to the closest splice junction for each mutation.

    Parameters:
    - mutation_df: DataFrame with mutation data.
    - splice_junctions: Dictionary of splice junctions per chromosome.

    Returns:
    - mutation_df: DataFrame with an added column for distances.
    """
    if columns_to_explode is None:
        columns_to_explode = ["header"]
    else:
        columns_to_explode = columns_to_explode.copy()

    # mutation_metadata_df_exploded = explode_df(mutation_metadata_df, columns_to_explode)

    splice_junctions = load_splice_junctions_from_gtf(reference_genome_gtf)

    distances = []

    for idx, row in tqdm(
        mutation_metadata_df_exploded.iterrows(),
        total=len(mutation_metadata_df_exploded),
    ):
        if (
            pd.isna(row["chromosome"])
            or pd.isna(row["start_mutation_position_genome"])
            or pd.isna(row["end_mutation_position_genome"])
        ):
            distances.append(np.nan)
            continue

        try:
            chrom = str(row["chromosome"])
            start_pos = int(row["start_mutation_position_genome"])
            end_pos = int(row["end_mutation_position_genome"])
        except ValueError:
            distances.append(np.nan)
            continue

        if chrom in splice_junctions:
            junctions = splice_junctions[chrom]

            # Find closest splice junction to start position
            dist_start = find_closest_distance(start_pos, junctions)
            if start_pos != end_pos:
                # Find closest splice junction to end position
                dist_end = find_closest_distance(end_pos, junctions)

                # Minimal distance
                min_distance = min(dist_start, dist_end)
            else:
                min_distance = dist_start
        else:
            # If chromosome not in splice junctions, set distance to NaN or a large number
            min_distance = np.nan

        distances.append(min_distance)

    mutation_metadata_df_exploded["distance_to_nearest_splice_junction"] = distances

    mutation_metadata_df_exploded[
        f"is_near_splice_junction_{near_splice_junction_threshold}"
    ] = mutation_metadata_df_exploded["distance_to_nearest_splice_junction"].apply(
        lambda x: x <= near_splice_junction_threshold if pd.notna(x) else np.nan
    )

    columns_to_explode.extend(
        [
            "distance_to_nearest_splice_junction",
            f"is_near_splice_junction_{near_splice_junction_threshold}",
        ]
    )

    # mutation_metadata_df, columns_to_explode = collapse_df(mutation_metadata_df_exploded, columns_to_explode, columns_to_explode_extend_values = ["distance_to_nearest_splice_junction", f"is_near_splice_junction_{near_splice_junction_threshold}"])

    return mutation_metadata_df_exploded, columns_to_explode


def merge_synthetic_read_info_into_mutations_metadata_df(
    mutation_metadata_df, sampled_reference_df, sample_type="all"
):
    if sample_type != "m":
        mutation_metadata_df = mutation_metadata_df.merge(
            sampled_reference_df[
                [
                    "header",
                    "included_in_synthetic_reads_wt",
                    "number_of_reads_wt",
                    "list_of_read_starting_indices_wt",
                    "any_noisy_reads_wt",
                    "noisy_read_indices_wt",
                ]
            ],
            on="header",
            how="left",
            suffixes=("", "_new"),
        )
        mutation_metadata_df["included_in_synthetic_reads_wt"] = (
            mutation_metadata_df["included_in_synthetic_reads_wt"]
            | mutation_metadata_df["included_in_synthetic_reads_wt_new"]
        )

        mutation_metadata_df["any_noisy_reads_wt"] = (
            mutation_metadata_df["any_noisy_reads_wt"]
            | mutation_metadata_df["any_noisy_reads_wt_new"]
        )

        mutation_metadata_df["number_of_reads_wt"] = np.where(
            mutation_metadata_df["number_of_reads_wt"] == 0,
            mutation_metadata_df["number_of_reads_wt_new"],
            mutation_metadata_df["number_of_reads_wt"],
        )

        mutation_metadata_df["list_of_read_starting_indices_wt"] = np.where(
            pd.isna(mutation_metadata_df["list_of_read_starting_indices_wt"]),
            mutation_metadata_df["list_of_read_starting_indices_wt_new"],
            mutation_metadata_df["list_of_read_starting_indices_wt"],
        )

        mutation_metadata_df["noisy_read_indices_wt"] = np.where(
            pd.isna(mutation_metadata_df["noisy_read_indices_wt"]),
            mutation_metadata_df["noisy_read_indices_wt_new"],
            mutation_metadata_df["noisy_read_indices_wt"],
        )

        mutation_metadata_df = mutation_metadata_df.drop(
            columns=[
                "included_in_synthetic_reads_wt_new",
                "number_of_reads_wt_new",
                "list_of_read_starting_indices_wt_new",
                "any_noisy_reads_wt_new",
                "noisy_read_indices_wt_new",
            ]
        )

    if sample_type != "w":
        mutation_metadata_df = mutation_metadata_df.merge(
            sampled_reference_df[
                [
                    "header",
                    "included_in_synthetic_reads_mutant",
                    "number_of_reads_mutant",
                    "list_of_read_starting_indices_mutant",
                    "any_noisy_reads_mutant",
                    "noisy_read_indices_mutant",
                ]
            ],
            on="header",
            how="left",
            suffixes=("", "_new"),
        )
        mutation_metadata_df["included_in_synthetic_reads_mutant"] = (
            mutation_metadata_df["included_in_synthetic_reads_mutant"]
            | mutation_metadata_df["included_in_synthetic_reads_mutant_new"]
        )

        mutation_metadata_df["any_noisy_reads_mutant"] = (
            mutation_metadata_df["any_noisy_reads_mutant"]
            | mutation_metadata_df["any_noisy_reads_mutant_new"]
        )

        mutation_metadata_df["number_of_reads_mutant"] = np.where(
            mutation_metadata_df["number_of_reads_mutant"] == 0,
            mutation_metadata_df["number_of_reads_mutant_new"],
            mutation_metadata_df["number_of_reads_mutant"],
        )

        mutation_metadata_df["list_of_read_starting_indices_mutant"] = np.where(
            pd.isna(mutation_metadata_df["list_of_read_starting_indices_mutant"]),
            mutation_metadata_df["list_of_read_starting_indices_mutant_new"],
            mutation_metadata_df["list_of_read_starting_indices_mutant"],
        )

        mutation_metadata_df["noisy_read_indices_mutant"] = np.where(
            pd.isna(mutation_metadata_df["noisy_read_indices_mutant"]),
            mutation_metadata_df["noisy_read_indices_mutant_new"],
            mutation_metadata_df["noisy_read_indices_mutant"],
        )

        mutation_metadata_df = mutation_metadata_df.drop(
            columns=[
                "included_in_synthetic_reads_mutant_new",
                "number_of_reads_mutant_new",
                "list_of_read_starting_indices_mutant_new",
                "any_noisy_reads_mutant_new",
                "noisy_read_indices_mutant_new",
            ]
        )

    mutation_metadata_df["included_in_synthetic_reads"] = (
        mutation_metadata_df["included_in_synthetic_reads_mutant"]
        | mutation_metadata_df["included_in_synthetic_reads_wt"]
    )
    mutation_metadata_df["any_noisy_reads"] = (
        mutation_metadata_df["any_noisy_reads_mutant"]
        | mutation_metadata_df["any_noisy_reads_wt"]
    )

    return mutation_metadata_df


# def generate_synthetic_reads(mutation_metadata_df, fasta_output_path, sampled_reference_df_parent = None, read_df_parent = None, sample_type = "all", n=1500, number_of_reads_per_sample = "all", read_length=150, seed=42, add_noise=False):
#     if 'included_in_synthetic_reads' in mutation_metadata_df.columns:
#         mutation_metadata_df = mutation_metadata_df.loc[(~mutation_metadata_df['included_in_synthetic_reads_mutant']) & (~mutation_metadata_df['included_in_synthetic_reads_wt'])]

#     if n == "all":
#         sampled_reference_df = mutation_metadata_df
#     else:
#         # Randomly select n rows
#         sampled_reference_df = mutation_metadata_df.sample(n=n, random_state=seed)

#     mutant_list_of_dicts = []
#     wt_list_of_dicts = []


#     if sample_type == "m":
#         sampled_reference_df['included_in_synthetic_reads_mutant'] = True
#         new_column_names = ['number_of_reads_mutant', 'list_of_read_starting_indices_mutant']
#     elif sample_type == "w":
#         sampled_reference_df['included_in_synthetic_reads_wt'] = True
#         new_column_names = ['number_of_reads_wt', 'list_of_read_starting_indices_wt']
#     else:
#         sampled_reference_df['included_in_synthetic_reads_mutant'] = True
#         sampled_reference_df['included_in_synthetic_reads_wt'] = True
#         new_column_names = ['number_of_reads_mutant', 'list_of_read_starting_indices_mutant', 'number_of_reads_wt', 'list_of_read_starting_indices_wt']

#     new_column_dict = {key: [] for key in new_column_names}

#     # Write to a FASTA file
#     total_fragments = 0
#     with open(fasta_output_path, "a") as fa_file:
#         for row in sampled_reference_df.itertuples(index=False):
#             header = row.header
#             mcrs_header = row.mcrs_header
#             mcrs_mutation_type = row.mcrs_mutation_type
#             mutant_sequence = row.mutant_sequence
#             mutant_sequence_length = row.mutant_read_parent_sequence_length_150
#             wt_sequence = row.wt_sequence
#             wt_sequence_length = row.wt_read_parent_sequence_length_150

#             valid_starting_index_max_mutant = mutant_sequence_length - read_length + 1
#             valid_starting_index_max_wt = wt_sequence_length - read_length + 1

#             if number_of_reads_per_sample == "all":
#                 read_start_indices_mutant = list(range(valid_starting_index_max_mutant))
#                 read_start_indices_wt = list(range(valid_starting_index_max_wt))

#                 number_of_reads_mutant = len(read_start_indices_mutant)
#                 number_of_reads_wt = len(read_start_indices_wt)
#             else:
#                 valid_starting_index_max = min(valid_starting_index_max_mutant, valid_starting_index_max_wt)
#                 if seed is not None:
#                     random.seed(seed)
#                 number_of_reads_per_sample = int(number_of_reads_per_sample)
#                 number_of_reads = min(valid_starting_index_max, number_of_reads_per_sample)
#                 read_start_indices_mutant = random.sample(range(valid_starting_index_max), min(valid_starting_index_max, number_of_reads_per_sample))
#                 read_start_indices_wt = read_start_indices_mutant

#                 number_of_reads_mutant = number_of_reads
#                 number_of_reads_wt = number_of_reads

#             # Loop through each 150mer of the sequence
#             if sample_type != "w":
#                 new_column_dict['number_of_reads_mutant'].append(number_of_reads_mutant)
#                 new_column_dict['list_of_read_starting_indices_mutant'].append(read_start_indices_mutant)
#                 for i in read_start_indices_mutant:
#                     sequence_chunk = mutant_sequence[i:i+read_length]
#                     if add_noise:
#                         sequence_chunk = introduce_sequencing_errors(sequence_chunk)
#                     read_header = f"{header}_{i}M"
#                     fa_file.write(f">{read_header}\n{sequence_chunk}\n")
#                     mutant_dict = {"read_header": read_header, "read_sequence": sequence_chunk, "read_index": i, "reference_header": header, "mcrs_header": mcrs_header, "mcrs_mutation_type": mcrs_mutation_type, "mutant_read": True, "wt_read": False, "region_included_in_mcrs_reference": True}
#                     mutant_list_of_dicts.append(mutant_dict)
#                     total_fragments += 1
#             if sample_type != "m":
#                 new_column_dict['number_of_reads_wt'].append(number_of_reads_wt)
#                 new_column_dict['list_of_read_starting_indices_wt'].append(read_start_indices_wt)
#                 for i in read_start_indices_wt:
#                     sequence_chunk = wt_sequence[i:i+read_length]
#                     if add_noise:
#                         sequence_chunk = introduce_sequencing_errors(sequence_chunk)
#                     read_header = f"{header}_{i}W"
#                     fa_file.write(f">{read_header}\n{sequence_chunk}\n")
#                     wt_dict = {"read_header": read_header, "read_sequence": sequence_chunk, "read_index": i, "reference_header": header, "mcrs_header": mcrs_header, "mcrs_mutation_type": mcrs_mutation_type, "mutant_read": False, "wt_read": True, "region_included_in_mcrs_reference": True}
#                     wt_list_of_dicts.append(wt_dict)
#                     total_fragments += 1

#     for key in new_column_dict:
#         sampled_reference_df[key] = new_column_dict[key]

#     if mutant_list_of_dicts and wt_list_of_dicts:
#         read_df_mut = pd.DataFrame(mutant_list_of_dicts)
#         read_df_wt = pd.DataFrame(wt_list_of_dicts)
#         read_df = pd.concat([read_df_mut, read_df_wt], ignore_index=True)
#     elif mutant_list_of_dicts:
#         read_df = pd.DataFrame(mutant_list_of_dicts)
#     elif wt_list_of_dicts:
#         read_df = pd.DataFrame(wt_list_of_dicts)

#     if sampled_reference_df_parent is not None:
#         sampled_reference_df = pd.concat([sampled_reference_df_parent, sampled_reference_df], ignore_index=True)

#     if read_df_parent is not None:
#         read_df = pd.concat([read_df_parent, read_df], ignore_index=True)

#     mutation_metadata_df = merge_synthetic_read_info_into_mutations_metadata_df(mutation_metadata_df, sampled_reference_df, sample_type = sample_type)

#     print(f"Wrote {total_fragments} mutations to {fasta_output_path}")

#     return read_df, mutation_metadata_df


# generate synthetic reads here
def generate_synthetic_reads(
    mutation_metadata_df,
    fasta_output_path,
    sampled_reference_df_parent=None,
    read_df_parent=None,
    sample_type="all",
    n=1500,
    strandedness=None,
    number_of_reads_per_sample="all",
    read_length=150,
    seed=42,
    add_noise=False,
):
    if seed is not None:
        random.seed(seed)

    if "included_in_synthetic_reads" not in mutation_metadata_df.columns:
        mutation_metadata_df["included_in_synthetic_reads"] = False

    if "included_in_synthetic_reads_wt" not in mutation_metadata_df.columns:
        mutation_metadata_df["included_in_synthetic_reads_wt"] = False

    if "included_in_synthetic_reads_mutant" not in mutation_metadata_df.columns:
        mutation_metadata_df["included_in_synthetic_reads_mutant"] = False

    if "list_of_read_starting_indices_wt" not in mutation_metadata_df.columns:
        mutation_metadata_df["list_of_read_starting_indices_wt"] = None

    if "list_of_read_starting_indices_mutant" not in mutation_metadata_df.columns:
        mutation_metadata_df["list_of_read_starting_indices_mutant"] = None

    if "number_of_reads_wt" not in mutation_metadata_df.columns:
        mutation_metadata_df["number_of_reads_wt"] = 0

    if "number_of_reads_mutant" not in mutation_metadata_df.columns:
        mutation_metadata_df["number_of_reads_mutant"] = 0

    if "included_in_synthetic_reads" in mutation_metadata_df.columns:
        mutation_metadata_df = mutation_metadata_df.loc[
            (~mutation_metadata_df["included_in_synthetic_reads_mutant"])
            & (~mutation_metadata_df["included_in_synthetic_reads_wt"])
        ]

    if n == "all":
        sampled_reference_df = mutation_metadata_df
    else:
        # Randomly select n rows
        sampled_reference_df = mutation_metadata_df.sample(n=n, random_state=seed)

    mutant_list_of_dicts = []
    wt_list_of_dicts = []

    if sample_type == "m":
        sampled_reference_df["included_in_synthetic_reads_mutant"] = True
        new_column_names = [
            "number_of_reads_mutant",
            "list_of_read_starting_indices_mutant",
        ]
    elif sample_type == "w":
        sampled_reference_df["included_in_synthetic_reads_wt"] = True
        new_column_names = ["number_of_reads_wt", "list_of_read_starting_indices_wt"]
    else:
        sampled_reference_df["included_in_synthetic_reads_mutant"] = True
        sampled_reference_df["included_in_synthetic_reads_wt"] = True
        new_column_names = [
            "number_of_reads_mutant",
            "list_of_read_starting_indices_mutant",
            "number_of_reads_wt",
            "list_of_read_starting_indices_wt",
        ]

    new_column_dict = {key: [] for key in new_column_names}

    # Write to a FASTA file
    total_fragments = 0
    with open(fasta_output_path, "a") as fa_file:
        for row in sampled_reference_df.itertuples(index=False):
            header = row.header
            mcrs_header = row.mcrs_header
            mcrs_mutation_type = row.mcrs_mutation_type
            mutant_sequence = row.mutant_sequence
            mutant_sequence_rc = row.mutant_sequence_rc
            mutant_sequence_length = row.mutant_read_parent_sequence_length_150
            wt_sequence = row.wt_sequence
            wt_sequence_rc = row.wt_sequence_rc
            wt_sequence_length = row.wt_read_parent_sequence_length_150

            valid_starting_index_max_mutant = mutant_sequence_length - read_length + 1
            valid_starting_index_max_wt = wt_sequence_length - read_length + 1

            if number_of_reads_per_sample == "all":
                read_start_indices_mutant = list(range(valid_starting_index_max_mutant))
                read_start_indices_wt = list(range(valid_starting_index_max_wt))

                number_of_reads_mutant = len(read_start_indices_mutant)
                number_of_reads_wt = len(read_start_indices_wt)

            else:
                valid_starting_index_max = min(
                    valid_starting_index_max_mutant, valid_starting_index_max_wt
                )
                number_of_reads_per_sample = int(number_of_reads_per_sample)
                number_of_reads = min(
                    valid_starting_index_max, number_of_reads_per_sample
                )
                read_start_indices_mutant = random.sample(
                    range(valid_starting_index_max),
                    min(valid_starting_index_max, number_of_reads_per_sample),
                )
                read_start_indices_wt = read_start_indices_mutant

                number_of_reads_mutant = number_of_reads
                number_of_reads_wt = number_of_reads

            if strandedness is None:
                mutant_sequence_list = [
                    random.choice([(mutant_sequence, "f"), (mutant_sequence_rc, "r")])
                ]
                wt_sequence_list = [
                    random.choice([(wt_sequence, "f"), (wt_sequence_rc, "r")])
                ]
            elif strandedness[0] == "f":
                mutant_sequence_list = [(mutant_sequence, "f")]
                wt_sequence_list = [(wt_sequence, "f")]
            elif strandedness[0] == "r":
                mutant_sequence_list = [(mutant_sequence_rc, "r")]
                wt_sequence_list = [(wt_sequence_rc, "r")]
            elif strandedness == "both":
                mutant_sequence_list = [
                    (mutant_sequence, "f"),
                    (mutant_sequence_rc, "r"),
                ]
                wt_sequence_list = [(wt_sequence, "f"), (wt_sequence_rc, "r")]

            # Loop through each 150mer of the sequence
            if sample_type != "w":
                if number_of_reads_per_sample == "all":
                    number_of_reads_mutant = (
                        number_of_reads_mutant * 2
                    )  # since now both strands are being sampled
                new_column_dict["number_of_reads_mutant"].append(number_of_reads_mutant)
                new_column_dict["list_of_read_starting_indices_mutant"].append(
                    read_start_indices_mutant
                )
                for selected_sequence, selected_strand in mutant_sequence_list:
                    for i in read_start_indices_mutant:
                        sequence_chunk = selected_sequence[i : i + read_length]
                        if add_noise:
                            sequence_chunk = introduce_sequencing_errors(sequence_chunk)
                        read_header = f"{header}_{i}{selected_strand}M"
                        fa_file.write(f">{read_header}\n{sequence_chunk}\n")
                        mutant_dict = {
                            "read_header": read_header,
                            "read_sequence": sequence_chunk,
                            "read_index": i,
                            "read_strand": selected_strand,
                            "reference_header": header,
                            "mcrs_header": mcrs_header,
                            "mcrs_mutation_type": mcrs_mutation_type,
                            "mutant_read": True,
                            "wt_read": False,
                            "region_included_in_mcrs_reference": True,
                        }
                        mutant_list_of_dicts.append(mutant_dict)
                        total_fragments += 1
            if sample_type != "m":
                if number_of_reads_per_sample == "all":
                    number_of_reads_wt = (
                        number_of_reads_wt * 2
                    )  # since now both strands are being sampled
                new_column_dict["number_of_reads_wt"].append(number_of_reads_wt)
                new_column_dict["list_of_read_starting_indices_wt"].append(
                    read_start_indices_wt
                )
                for selected_sequence, selected_strand in wt_sequence_list:
                    for i in read_start_indices_wt:
                        sequence_chunk = selected_sequence[i : i + read_length]
                        if add_noise:
                            sequence_chunk = introduce_sequencing_errors(sequence_chunk)
                        read_header = f"{header}_{i}{selected_strand}W"
                        fa_file.write(f">{read_header}\n{sequence_chunk}\n")
                        wt_dict = {
                            "read_header": read_header,
                            "read_sequence": sequence_chunk,
                            "read_index": i,
                            "read_strand": selected_strand,
                            "reference_header": header,
                            "mcrs_header": mcrs_header,
                            "mcrs_mutation_type": mcrs_mutation_type,
                            "mutant_read": False,
                            "wt_read": True,
                            "region_included_in_mcrs_reference": True,
                        }
                        wt_list_of_dicts.append(wt_dict)
                        total_fragments += 1

    for key in new_column_dict:
        sampled_reference_df[key] = new_column_dict[key]

    if mutant_list_of_dicts and wt_list_of_dicts:
        read_df_mut = pd.DataFrame(mutant_list_of_dicts)
        read_df_wt = pd.DataFrame(wt_list_of_dicts)
        read_df = pd.concat([read_df_mut, read_df_wt], ignore_index=True)
    elif mutant_list_of_dicts:
        read_df = pd.DataFrame(mutant_list_of_dicts)
    elif wt_list_of_dicts:
        read_df = pd.DataFrame(wt_list_of_dicts)

    if sampled_reference_df_parent is not None:
        sampled_reference_df = pd.concat(
            [sampled_reference_df_parent, sampled_reference_df], ignore_index=True
        )

    if read_df_parent is not None:
        read_df = pd.concat([read_df_parent, read_df], ignore_index=True)

    mutation_metadata_df = merge_synthetic_read_info_into_mutations_metadata_df(
        mutation_metadata_df, sampled_reference_df, sample_type=sample_type
    )

    print(f"Wrote {total_fragments} mutations to {fasta_output_path}")

    return read_df, mutation_metadata_df


def select_rows(df, column, value):
    return df.loc[df[column] == value]


def read_fasta_as_tuples(file_path):
    # Read the FASTA file and create a list of (header, sequence) tuples
    fasta_entries = [
        (record.id, str(record.seq)) for record in SeqIO.parse(file_path, "fasta")
    ]
    return fasta_entries


def is_in_ranges(num, ranges):
    if not ranges:
        return False
    for start, end in ranges:
        if start <= num <= end:
            return True
    return False


def append_row(read_df, header_value, sequence_value, start_position, strand):
    # Create a new row where 'header' and 'seq_ID' are populated, and others are NaN
    new_row = pd.Series(
        {
            "read_header": header_value,
            "read_sequence": sequence_value,
            "read_index": start_position,
            "strand": strand,
            "reference_header": None,
            "mcrs_header": None,
            "mcrs_mutation_type": None,
            "mutant_read": False,
            "wt_read": True,
            "region_included_in_mcrs_reference": False,
            # All other columns will be NaN automatically
        }
    )

    return read_df.append(new_row, ignore_index=True)


def build_random_genome_read_df(
    reference_fasta_file_path,
    mutation_metadata_df,
    read_df=None,
    n=10,
    read_length=150,
    input_type="transcriptome",
    strand=None,
):
    # Collect all headers and sequences from the FASTA file
    fasta_entries = list(read_fasta(reference_fasta_file_path))
    if read_df is None:
        column_names = [
            "read_header",
            "read_sequence",
            "reference_header",
            "mcrs_header",
            "mutant_read",
            "wt_read",
            "region_included_in_mcrs_reference",
        ]
        read_df = pd.DataFrame(columns=column_names)

    if input_type == "genome":
        chromosomes = [str(x) for x in list(range(23))]
        chromosomes.extend(["X", "Y", "MT"])
        fasta_entries = [
            entry for entry in fasta_entries if entry[0] not in chromosomes
        ]
        fasta_entry_column = "chromosome"
        mcrs_start_column = "start_position_for_which_read_contains_mutation_genome"
        mcrs_end_column = "end_mutation_position_genome"
    elif input_type == "transcriptome":
        fasta_entry_column = "seq_ID"
        mcrs_start_column = "start_position_for_which_read_contains_mutation_cdna"
        mcrs_end_column = "end_mutation_position_cdna"

    i = 0
    num_loops = 0
    while i < n:
        # Choose a random entry (header, sequence) from the FASTA file
        random_transcript, random_sequence = random.choice(fasta_entries)

        random_transcript = random_transcript.split()[0]
        if input_type == "transcriptome":
            random_transcript = random_transcript.split(".")[0]

        # Choose a random integer between 1 and the sequence length
        start_position = random.randint(1, len(random_sequence))

        filtered_mutation_metadata_df = mutation_metadata_df.loc[
            mutation_metadata_df[fasta_entry_column] == random_transcript
        ]

        ranges = list(
            zip(
                filtered_mutation_metadata_df[mcrs_start_column],
                filtered_mutation_metadata_df[mcrs_end_column],
            )
        )  # if a mutation spans from positions 950-955 and read length=150, then a random sequence between 801-955 will contain the mutation, and thus should be the range of exclusion here

        if not is_in_ranges(start_position, ranges):
            end_position = start_position + read_length
            if strand is None:
                selected_strand = random.choice(["f", "r"])
            else:
                selected_strand = strand

            if selected_strand == "r":
                # start_position, end_position = len(random_sequence) - end_position, len(random_sequence) - start_position  # I am keeping adding the "f/r" in header so I don't need this
                random_sequence = reverse_complement(random_sequence)

            header = f"{random_transcript}:{start_position}_{end_position}_random{selected_strand}W"
            read_df = append_row(
                read_df, header, random_sequence, start_position, selected_strand
            )
            i += 1

        num_loops += 1
        if num_loops > n * 100:
            print(f"Exiting after only {i} mutations added due to long while loop")
            break

    return read_df


def get_header_set_from_fastq(fastq_file, output_format="set"):
    if output_format == "set":
        headers = {header[1:].strip() for header, _, _, _ in read_fastq(fastq_file)}
    elif output_format == "list":
        headers = [header[1:].strip() for header, _, _, _ in read_fastq(fastq_file)]
    return headers


def compare_cdna_and_genome(
    mutation_metadata_df_exploded,
    gget_mutate_temp_folder=None,
    reference_cdna_fasta="cdna",
    reference_genome_fasta="genome",
    mutations_csv=None,
    w=30,
    mcrs_source="cdna",
    columns_to_explode=None,
):
    from varseek.varseek_build import build

    if columns_to_explode is None:
        columns_to_explode = ["header"]
    else:
        columns_to_explode = columns_to_explode.copy()

    with tempfile.TemporaryDirectory() as gget_mutate_temp_folder:

        reference_out_dir_temp = f"{gget_mutate_temp_folder}/reference_out"

        gget_mutate_cdna_out_fasta = (
            f"{gget_mutate_temp_folder}/gget_mutate_cdna_{w}.fa"
        )
        gget_mutate_cdna_out_df = f"{gget_mutate_temp_folder}/gget_mutate_cdna_{w}.csv"

        if not os.path.exists(gget_mutate_cdna_out_df):
            build(
                sequences=reference_cdna_fasta,
                mutations=mutations_csv,
                out=gget_mutate_cdna_out_fasta,
                reference_out=reference_out_dir_temp,
                w=w,
                remove_seqs_with_wt_kmers=False,
                optimize_flanking_regions=False,
                max_ambiguous=None,
                merge_identical=False,
                merge_identical_rc=False,
                cosmic_email=os.getenv("COSMIC_EMAIL"),
                cosmic_password=os.getenv("COSMIC_PASSWORD"),
                update_df=True,
                update_df_out=gget_mutate_cdna_out_df,
            )

        cdna_updated_df = pd.read_csv(
            gget_mutate_cdna_out_df,
            usecols=[
                "header",
                "mutant_sequence",
                "seq_ID",
                "mutation",
                "mutation_type",
            ],
        )

        gget_mutate_genome_out_fasta = (
            f"{gget_mutate_temp_folder}/gget_mutate_genome_{w}.fa"
        )
        gget_mutate_genome_out_df = (
            f"{gget_mutate_temp_folder}/gget_mutate_genome_{w}.csv"
        )

        if not os.path.exists(gget_mutate_genome_out_df):
            build(
                sequences=reference_genome_fasta,
                mutations=mutations_csv,
                out=gget_mutate_genome_out_fasta,
                reference_out=reference_out_dir_temp,
                w=w,
                remove_seqs_with_wt_kmers=False,
                optimize_flanking_regions=False,
                max_ambiguous=None,
                merge_identical=False,
                merge_identical_rc=False,
                cosmic_email=os.getenv("COSMIC_EMAIL"),
                cosmic_password=os.getenv("COSMIC_PASSWORD"),
                update_df=True,
                update_df_out=gget_mutate_genome_out_df,
                seq_id_column="chromosome",
                mut_column="mutation_genome",
            )

        genome_updated_df = pd.read_csv(
            gget_mutate_genome_out_df,
            usecols=[
                "header",
                "mutant_sequence",
                "chromosome",
                "mutation_genome",
                "mutation_type",
                "seq_ID",
                "mutation",
            ],
        )

    combined_updated_df = cdna_updated_df.merge(
        genome_updated_df,
        on=["seq_ID", "mutation"],
        how="outer",
        suffixes=("_cdna", "_genome"),
    )

    combined_updated_df["cdna_and_genome_same"] = (
        combined_updated_df["mutant_sequence_cdna"]
        == combined_updated_df["mutant_sequence_genome"]
    )  # combined_updated_df['mutant_sequence_plus_genome']
    # combined_updated_df["cdna_and_genome_same"] = combined_updated_df["cdna_and_genome_same"].astype(str)

    if "cosmic" in mutations_csv:
        # cosmic is not reliable at recording duplication mutations at the genome level
        combined_updated_df.loc[
            (combined_updated_df["mutation_type_cdna"] == "duplication")
            | (combined_updated_df["mutation_type_genome"] == "duplication"),
            "cdna_and_genome_same",
        ] = np.nan

    if mcrs_source == "combined":
        column_to_merge = "header_cdna"
    else:
        column_to_merge = "header"
        combined_updated_df.rename(
            columns={f"header_{mcrs_source}": "header"}, inplace=True
        )

    # mutation_metadata_df_exploded = explode_df(mutation_metadata_df, columns_to_explode)

    mutation_metadata_df_exploded = mutation_metadata_df_exploded.merge(
        combined_updated_df[[column_to_merge, "cdna_and_genome_same"]],
        on=column_to_merge,
        how="left",
    )

    columns_to_explode.append("cdna_and_genome_same")

    # mutation_metadata_df, columns_to_explode = collapse_df(mutation_metadata_df_exploded, columns_to_explode, columns_to_explode_extend_values = ["cdna_and_genome_same"])

    # # mutation_metadata_df["cdna_and_genome_same"] = mutation_metadata_df["cdna_and_genome_same"].fillna("unsure")  # because I'm filling values with unsure, I must specify == True if indexing true values
    # # mutation_metadata_df = mutation_metadata_df.loc[~((mutation_metadata_df["cdna_and_genome_same"] == "True") & (mutation_metadata_df["mcrs_source"] == "genome"))]  #* uncomment to filter out rows derived from genome where cDNA and genome are the same (I used to filter these out because they are redundant and I only wanted to keep rows where genome differed from cDNA)

    # # # delete temp folder and all contents
    # shutil.rmtree(gget_mutate_temp_folder)

    return mutation_metadata_df_exploded, columns_to_explode


def get_mutation_type_series(mutation_series):
    # Extract mutation type id using the regex pattern
    mutation_type_id = mutation_series.str.extract(mutation_pattern)[1]

    # Define conditions and choices for mutation types
    conditions = [
        mutation_type_id.str.contains(">", na=False),
        mutation_type_id.str.contains("delins", na=False),
        mutation_type_id.str.contains("del", na=False)
        & ~mutation_type_id.str.contains("delins", na=False),
        mutation_type_id.str.contains("ins", na=False)
        & ~mutation_type_id.str.contains("delins", na=False),
        mutation_type_id.str.contains("dup", na=False),
        mutation_type_id.str.contains("inv", na=False),
    ]

    choices = [
        "substitution",
        "delins",
        "deletion",
        "insertion",
        "duplication",
        "inversion",
    ]

    # Determine mutation type
    mutation_type_array = np.select(conditions, choices, default="unknown")

    return mutation_type_array


def add_mcrs_mutation_type(mutations_df, mut_column="mcrs_header"):
    mutations_df = mutations_df.copy()

    # Split the mut_column by ';'
    mutations_df["mutation_list"] = mutations_df[mut_column].str.split(";")

    # Explode the mutation_list to get one mutation per row
    mutations_exploded = mutations_df.explode("mutation_list")

    # Apply the vectorized get_mutation_type_series function
    mutations_exploded["mcrs_mutation_type"] = get_mutation_type_series(
        mutations_exploded["mutation_list"]
    )

    # Reset index to keep track of original rows
    mutations_exploded.reset_index(inplace=True)

    # Group back to the original DataFrame, joining mutation types with ';'
    grouped_mutation_types = mutations_exploded.groupby("index")[
        "mcrs_mutation_type"
    ].apply(";".join)

    # Assign the 'mutation_type' back to mutations_df
    mutations_df["mcrs_mutation_type"] = grouped_mutation_types

    # Split 'mutation_type' by ';' to analyze unique mutation types
    mutations_df["mutation_type_split"] = mutations_df["mcrs_mutation_type"].str.split(
        ";"
    )

    # Calculate the number of unique mutation types
    mutations_df["unique_mutation_count"] = (
        mutations_df["mutation_type_split"].map(set).str.len()
    )

    # Replace 'mutation_type' with the single unique mutation type if unique_mutation_count == 1
    mask_single = mutations_df["unique_mutation_count"] == 1
    mutations_df.loc[mask_single, "mcrs_mutation_type"] = mutations_df.loc[
        mask_single, "mutation_type_split"
    ].str[0]

    # Replace entries containing ';' with 'mixed'
    mutations_df.loc[
        mutations_df["mcrs_mutation_type"].str.contains(";"), "mcrs_mutation_type"
    ] = "mixed"

    # Drop helper columns
    mutations_df.drop(
        columns=["mutation_list", "mutation_type_split", "unique_mutation_count"],
        inplace=True,
    )

    mutations_df.loc[mutations_df[mut_column].isna(), "mcrs_mutation_type"] = np.nan

    return mutations_df


def align_all_kmers_from_a_given_fasta_entry_to_all_other_kmers_in_the_file(
    my_header, my_sequence, reference_fasta, k=31
):
    with open(reference_fasta, "r") as fasta_handle:
        for record in SeqIO.parse(fasta_handle, "fasta"):
            sequence = str(record.seq)  # Convert the sequence to a string
            for start_position in range(0, len(my_sequence) - k + 1):
                kmer = my_sequence[start_position : start_position + k]
                if kmer in sequence:
                    if record.id in my_header:
                        print(
                            f"k-mer at position {start_position} found in its respective sequence: {record.id}"
                        )
                    else:
                        print(
                            f"k-mer at position {start_position} found in a DIFFERENT sequence: {record.id}"
                        )


def introduce_sequencing_errors(
    sequence, error_rate=0.0001, error_distribution=(0.85, 0.1, 0.05), max_errors=float("inf")  # error_distribution is (sub, del, ins)
):  # Illumina error rate is around 0.01% (1 in 10,000)
    # Define the possible bases
    bases = ["A", "T", "C", "G"]
    new_sequence = []
    number_errors = 0

    error_distribution_sub = error_distribution[0]
    error_distribution_del = error_distribution[1]
    error_distribution_ins = error_distribution[2]

    for base in sequence:
        if number_errors < max_errors and random.random() < error_rate:
            if random.random() < error_distribution_sub:  # Substitution
                new_base = random.choice([b for b in bases if b != base])
                new_sequence.append(new_base)
            elif random.random() < error_distribution_ins:  # Insertion
                new_sequence.append(random.choice(bases))
            else:  # Deletion
                continue  # Skip this base (deletion)
            number_errors += 1
        else:
            new_sequence.append(base)  # No error, keep base

    return "".join(new_sequence)


def generate_noisy_quality_scores(sequence, avg_quality=30):
    # Assume a normal distribution for quality scores, with some fluctuation
    qualities = [max(0, min(40, int(random.gauss(avg_quality, 5)))) for _ in sequence]
    # Convert qualities to ASCII Phred scores (33 is the offset)
    return "".join([chr(q + 33) for q in qualities])


def add_polyA_tail(sequence, tail_length_range=(10, 50)):
    tail_length = random.randint(*tail_length_range)
    return sequence + "A" * tail_length


def add_noise_to_read(sequence, error_rate=0.0001, avg_quality=30, add_polyA=True):
    # Step 1: Introduce sequencing errors
    noisy_sequence = introduce_sequencing_errors(sequence, error_rate=error_rate)

    # Step 2: Generate quality scores
    quality_scores = generate_noisy_quality_scores(
        noisy_sequence, avg_quality=avg_quality
    )

    # Step 3: Optionally add a poly(A) tail
    if add_polyA:
        noisy_sequence = add_polyA_tail(noisy_sequence)

    return noisy_sequence, quality_scores


def count_nearby_mutations_efficient(
    df, k, fasta_entry_column, start_column, end_column, header_column=None
):
    # Ensure 'seq_ID' is in the DataFrame
    if "seq_ID" not in df.columns:
        raise ValueError("The DataFrame must contain a 'seq_ID' column.")

    # Initialize counts_unique array
    counts_unique = np.zeros(len(df), dtype=int)

    # Group by seq_ID
    grouped = df.groupby(fasta_entry_column)
    len_grouped = len(grouped)

    z = 0
    for seq_id, group in grouped:
        # Extract positions and indices within the group
        indices_original = group.index.values  # Original indices in df
        starts = group[start_column].values
        ends = group[end_column].values
        N = len(group)

        # Proceed only if group has more than one mutation
        if N > 1:
            # Create mapping from group indices (0 to N-1) to original indices
            mapping = {i: idx for i, idx in enumerate(indices_original)}

            # Prepare DataFrames with positions and group indices (0 to N-1)
            df_starts = pd.DataFrame({"position": starts, "index": np.arange(N)})
            df_ends = pd.DataFrame({"position": ends, "index": np.arange(N)})

            # Sort by positions
            df_starts_sorted = df_starts.sort_values("position").reset_index(drop=True)
            df_ends_sorted = df_ends.sort_values("position").reset_index(drop=True)

            # Initialize counts
            counts = np.zeros(N, dtype=int)

            # Condition 1: For each start, count ends within [start - (k - 1), start + (k - 1)]
            positions_ends = df_ends_sorted["position"].values
            indices_ends = df_ends_sorted["index"].values
            for i in range(N):
                start = starts[i]
                left = start - (k - 1)
                right = start + (k - 1)
                left_idx = np.searchsorted(positions_ends, left, side="left")
                right_idx = np.searchsorted(positions_ends, right, side="right")
                nearby_indices = indices_ends[left_idx:right_idx]
                # Exclude self
                nearby_indices = nearby_indices[nearby_indices != i]
                counts[i] += len(nearby_indices)

            # Condition 2: For each end, count starts within [end - (k - 1), end + (k - 1)]
            positions_starts = df_starts_sorted["position"].values
            indices_starts = df_starts_sorted["index"].values
            for i in range(N):
                end = ends[i]
                left = end - (k - 1)
                right = end + (k - 1)
                left_idx = np.searchsorted(positions_starts, left, side="left")
                right_idx = np.searchsorted(positions_starts, right, side="right")
                nearby_indices = indices_starts[left_idx:right_idx]
                # Exclude self
                nearby_indices = nearby_indices[nearby_indices != i]
                counts[i] += len(nearby_indices)

            # Merging Conditions
            counts_unique_group = np.zeros(N, dtype=int)
            for i in range(N):
                # Condition 1
                start = starts[i]
                left1 = start - (k - 1)
                right1 = start + (k - 1)
                left_idx1 = np.searchsorted(positions_ends, left1, side="left")
                right_idx1 = np.searchsorted(positions_ends, right1, side="right")
                nearby_indices1 = set(indices_ends[left_idx1:right_idx1])
                nearby_indices1.discard(i)

                # Condition 2
                end = ends[i]
                left2 = end - (k - 1)
                right2 = end + (k - 1)
                left_idx2 = np.searchsorted(positions_starts, left2, side="left")
                right_idx2 = np.searchsorted(positions_starts, right2, side="right")
                nearby_indices2 = set(indices_starts[left_idx2:right_idx2])
                nearby_indices2.discard(i)

                # Combine indices
                nearby_indices_total = nearby_indices1.union(nearby_indices2)
                counts_unique_group[i] = len(nearby_indices_total)

            # Assign counts to counts_unique using mapping
            for i in range(N):
                idx_original = mapping[i]
                counts_unique[idx_original] = counts_unique_group[i]
        else:
            # Only one mutation in group; count is zero
            idx_original = indices_original[0]
            counts_unique[idx_original] = 0

        z += 1
        if z % 100 == 0:
            print(f"Processed {z}/{len_grouped} groups.")

    df["nearby_mutations_count"] = counts_unique
    return df


def count_nearby_mutations_efficient_with_identifiers(
    df, k, fasta_entry_column, start_column, end_column, header_column
):
    # Ensure the required columns are in the DataFrame
    required_columns = [fasta_entry_column, start_column, end_column, header_column]
    for col in required_columns:
        if col not in df.columns:
            raise ValueError(f"The DataFrame must contain the column '{col}'.")

    # Initialize counts_unique array and nearby_headers_list
    counts_unique = np.zeros(len(df), dtype=int)
    nearby_headers_list = [
        set() for _ in range(len(df))
    ]  # List of sets to store nearby headers per mutation

    # Group by fasta_entry_column
    grouped = df.groupby(fasta_entry_column)
    total_groups = len(grouped)

    # Use tqdm to create a progress bar over groups
    with tqdm(total=total_groups, desc="Processing groups") as pbar:
        for seq_id, group in grouped:
            # Extract positions, headers, and indices within the group
            indices_original = group.index.values  # Original indices in df
            group[start_column] = pd.to_numeric(group[start_column], errors="coerce")
            group[end_column] = pd.to_numeric(group[end_column], errors="coerce")

            starts = group[start_column].values
            ends = group[end_column].values
            headers = group[header_column].values  # Extract headers within the group
            N = len(group)

            # Proceed only if group has more than one mutation
            if N > 1:
                # Create mapping from group indices (0 to N-1) to original indices
                mapping = {i: idx for i, idx in enumerate(indices_original)}

                # Prepare DataFrames with positions, group indices, and headers
                df_starts = pd.DataFrame(
                    {"position": starts, "index": np.arange(N), "header": headers}
                )
                df_ends = pd.DataFrame(
                    {"position": ends, "index": np.arange(N), "header": headers}
                )

                # Sort by positions
                df_starts_sorted = df_starts.sort_values("position").reset_index(
                    drop=True
                )
                df_ends_sorted = df_ends.sort_values("position").reset_index(drop=True)

                # Initialize counts and nearby headers for this group
                counts_unique_group = np.zeros(N, dtype=int)
                nearby_headers_group = [set() for _ in range(N)]

                # Positions and indices for efficient search
                positions_starts = df_starts_sorted["position"].values
                indices_starts = df_starts_sorted["index"].values
                headers_starts = df_starts_sorted["header"].values
                positions_ends = df_ends_sorted["position"].values
                indices_ends = df_ends_sorted["index"].values
                headers_ends = df_ends_sorted["header"].values

                # Loop over mutations within the group
                for i in range(N):
                    # Condition 1: Other ends within (k - 1) of current start
                    start = starts[i]
                    left1 = start - (k - 1)
                    right1 = start + (k - 1)
                    left_idx1 = np.searchsorted(positions_ends, left1, side="left")
                    right_idx1 = np.searchsorted(positions_ends, right1, side="right")
                    nearby_indices1 = indices_ends[left_idx1:right_idx1]
                    nearby_headers1 = headers_ends[left_idx1:right_idx1]
                    # Exclude self
                    mask1 = nearby_indices1 != i
                    nearby_indices1 = nearby_indices1[mask1]
                    nearby_headers1 = nearby_headers1[mask1]

                    # Condition 2: Other starts within (k - 1) of current end
                    end = ends[i]
                    left2 = end - (k - 1)
                    right2 = end + (k - 1)
                    left_idx2 = np.searchsorted(positions_starts, left2, side="left")
                    right_idx2 = np.searchsorted(positions_starts, right2, side="right")
                    nearby_indices2 = indices_starts[left_idx2:right_idx2]
                    nearby_headers2 = headers_starts[left_idx2:right_idx2]
                    # Exclude self
                    mask2 = nearby_indices2 != i
                    nearby_indices2 = nearby_indices2[mask2]
                    nearby_headers2 = nearby_headers2[mask2]

                    # Combine indices and headers
                    nearby_indices_total = set(nearby_indices1).union(nearby_indices2)
                    nearby_headers_total = set(nearby_headers1).union(nearby_headers2)

                    # Update counts and nearby headers
                    counts_unique_group[i] = len(nearby_indices_total)
                    nearby_headers_group[i].update(nearby_headers_total)

                # Assign counts and nearby headers to counts_unique and nearby_headers_list
                for i in range(N):
                    idx_original = mapping[i]
                    counts_unique[idx_original] = counts_unique_group[i]
                    nearby_headers_list[idx_original].update(nearby_headers_group[i])
            else:
                # Only one mutation in group; count is zero
                idx_original = indices_original[0]
                counts_unique[idx_original] = 0
                nearby_headers_list[idx_original] = set()

            pbar.update(1)  # Update the progress bar

    # Convert sets to lists for the 'nearby_headers' column
    nearby_headers_list = [list(headers_set) for headers_set in nearby_headers_list]

    # Add counts and nearby headers to DataFrame
    df["nearby_mutations_count"] = counts_unique
    df["nearby_mutations"] = nearby_headers_list
    return df


def create_df_of_mcrs_to_self_headers(
    mcrs_sam_file,
    mcrs_fa,
    bowtie_mcrs_reference_folder,
    bowtie_path,
    threads=2,
    strandedness=False,
    mcrs_id_column="mcrs_id",
    output_stat_file=None,
):

    bowtie2_build = f"{bowtie_path}/bowtie2-build"
    bowtie2 = f"{bowtie_path}/bowtie2"

    if not os.path.exists(mcrs_sam_file):
        if not os.path.exists(bowtie_mcrs_reference_folder) or not os.listdir(
            bowtie_mcrs_reference_folder
        ):
            print("Running bowtie2 build")
            os.makedirs(bowtie_mcrs_reference_folder, exist_ok=True)
            bowtie_reference_prefix = os.path.join(bowtie_mcrs_reference_folder, "mcrs")
            subprocess.run(
                [
                    bowtie2_build,  # Path to the bowtie2-build executable
                    "--threads",
                    str(threads),  # Number of threads
                    mcrs_fa,  # Input FASTA file
                    bowtie_reference_prefix,  # Output reference folder
                ],
                check=True,
            )

        print("Running bowtie2 alignment")

        bowtie2_alignment_command = [
            bowtie2,  # Path to the bowtie2 executable
            "-a",  # Report all alignments
            "-f",  # Input file is in FASTA format
            "-p",
            str(threads),  # Number of threads
            "--xeq",  # Match different quality scores
            "--score-min",
            "C,0,0",  # Minimum score threshold
            "--np",
            "0",  # No penalty for ambiguous matches
            "--n-ceil",
            "L,0,999",  # N-ceiling
            "-R",
            "1",  # Maximum Re-seed attempts
            "-N",
            "0",  # Maximum mismatches in seed alignment
            "-L",
            "31",  # Length of seed substrings
            "-i",
            "C,1,0",  # Interval between seed extensions
            "--no-1mm-upfront",  # No mismatches upfront
            "--no-unal",  # Do not write unaligned reads
            "--no-hd",  # Suppress header lines in SAM output
            "-x",
            bowtie_reference_prefix,  # Reference folder for alignment
            "-U",
            mcrs_fa,  # Input FASTA file
            "-S",
            mcrs_sam_file,  # Output SAM file
        ]

        if strandedness:
            bowtie2_alignment_command.insert(3, "--norc")

        result = subprocess.run(
            bowtie2_alignment_command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=True,
        )

        if output_stat_file is not None:
            if os.path.exists(output_stat_file):
                write_mode = "a"
            else:
                write_mode = "w"
            with open(output_stat_file, write_mode) as f:
                f.write(f"bowtie alignment for {bowtie_reference_prefix}")
                f.write("Standard Output:\n")
                f.write(result.stdout)
                f.write("\n\nStandard Error:\n")
                f.write(result.stderr)
                f.write("\n\n")

    substring_to_superstring_list_dict = defaultdict(list)
    superstring_to_substring_list_dict = defaultdict(list)

    print("Processing SAM file")
    for fields in process_sam_file(mcrs_sam_file):
        read_name = fields[0]
        ref_name = fields[2]

        if read_name == ref_name:
            continue

        substring_to_superstring_list_dict[read_name].append(ref_name)
        superstring_to_substring_list_dict[ref_name].append(read_name)

    # convert to DataFrame
    substring_to_superstring_df = pd.DataFrame(
        substring_to_superstring_list_dict.items(),
        columns=[mcrs_id_column, "entries_for_which_this_mcrs_is_substring"],
    )
    superstring_to_substring_df = pd.DataFrame(
        superstring_to_substring_list_dict.items(),
        columns=[mcrs_id_column, "entries_for_which_this_mcrs_is_superstring"],
    )

    substring_to_superstring_df["mcrs_is_substring"] = True
    superstring_to_substring_df["mcrs_is_superstring"] = True

    substring_to_superstring_df[mcrs_id_column] = substring_to_superstring_df[
        mcrs_id_column
    ].astype(str)
    superstring_to_substring_df[mcrs_id_column] = superstring_to_substring_df[
        mcrs_id_column
    ].astype(str)

    return substring_to_superstring_df, superstring_to_substring_df


def keep_end_of_integer_sequence(mylist, dlist_side="left"):
    sorted_list = sorted(mylist)
    if dlist_side == "left":  # keep only minima
        result = [sorted_list[0]]
        for i in range(1, len(sorted_list)):
            # Check if the current number is NOT consecutive from the previous one
            if sorted_list[i] != sorted_list[i - 1] + 1:
                result.append(sorted_list[i])
    elif dlist_side == "right":  # keep only maxima
        result = []
        for i in range(1, len(sorted_list)):
            if sorted_list[i] != sorted_list[i - 1] + 1:
                # If not consecutive, add the last number of the previous sequence
                result.append(sorted_list[i - 1])

        # Add the last number in the list (it's the largest in its sequence)
        result.append(sorted_list[-1])

    return result


# #* Pall d-list strategy, abandoned half-way through
# def create_dfk_list_for_dlist(sam_file, mutation_fasta_file = None, check_for_bad_cigars = True, k = 31):
#     header_to_position_dict = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))  # 3 levels deep

#     if mutation_fasta_file is not None:
#         header_set_from_mutation_fasta = get_long_headers(mutation_fasta_file)

#     for fields in process_sam_file(sam_file):
#         cigarstring = fields[5]
#         if check_for_bad_cigars and (cigarstring != f'{k}='):
#             continue

#         sequence_name = fields[0]

#         if mutation_fasta_file is not None:
#             if not check_dlist_header(sequence_name, dlist_pattern_utils):
#                 sequence_name = check_if_header_is_in_set(sequence_name, header_set_from_mutation_fasta)
#                 sequence_name = f'{sequence_name}_000'  # 000 means unknown position

#         mcrs_header = sequence_name.rsplit('_', 1)[0]
#         alignment_entry = fields[2]
#         alignment_start_position = int(fields[3]) - 1  # 0-based indexing
#         alignment_end_position = alignment_start_position + k - 1

#         # eg k-mer overlap from position 100-104 (index 0) and k=5 - left start position should be 96 (left_start-k+1) (such that 1st k-mer contains start position ie 96-100), and left end position should be 103 ie list should go :104 (left_end+k-1) (such that no k-mer contains full overlap ie 99-103)
#         # and right start position should be 101 (101-105), and right end position should be 108 (104-108)
#         dlist_left_start_position = max((alignment_start_position - (k - 1)), 0)  # ensure that each k-mer contains the start position
#         dlist_left_end_position = alignment_start_position + (k - 1)  # ensure that no k-mer contains complete overlap  #!!! doesn't check for end of sequence - unsure if issue

#         dlist_right_start_position = max((alignment_end_position - (k - 2)), 0)
#         dlist_right_end_position = alignment_end_position + (k - 1)  #!!! doesn't check for end of sequence - unsure if issue

#         header_to_position_dict[mcrs_header][alignment_entry]["dlist_left_start_position_list"] = header_to_position_dict[mcrs_header][alignment_entry]["dlist_left_start_position_list"].append(dlist_left_start_position)
#         header_to_position_dict[mcrs_header][alignment_entry]["dlist_left_end_position_list"] = header_to_position_dict[mcrs_header][alignment_entry]["dlist_left_end_position_list"].append(dlist_left_end_position)
#         header_to_position_dict[mcrs_header][alignment_entry]["dlist_right_start_position_list"] = header_to_position_dict[mcrs_header][alignment_entry]["dlist_right_start_position_list"].append(dlist_right_start_position)
#         header_to_position_dict[mcrs_header][alignment_entry]["dlist_right_end_position_list"] = header_to_position_dict[mcrs_header][alignment_entry]["dlist_right_end_position_list"].append(dlist_right_end_position)

#     list_names = ["dlist_left_start_position_list", "dlist_left_end_position_list", "dlist_right_start_position_list", "dlist_right_end_position_list"]
#     for list_name in list_names:
#         sorted_list = sorted(header_to_position_dict[mcrs_header][alignment_entry][list_name])  #!!! # make sure to iterate through each combination
#         result = [sorted_list[0]]
#         for i in range(1, len(sorted_list)):
#             # Check if the current number is NOT consecutive from the previous one
#             if sorted_list[i] != sorted_list[i-1] + 1:
#                 result.append(sorted_list[i])


def run_kb_count_dry_run(
    index, t2g, fastq, kb_count_out, newer_kallisto, k=31, threads=1
):
    if not os.path.exists(newer_kallisto):
        kallisto_install_from_source_commands = "git clone https://github.com/pachterlab/kallisto.git && cd kallisto && mkdir build && cd build && cmake .. -DMAX_KMER_SIZE=64 && make"
        subprocess.run(kallisto_install_from_source_commands, shell=True, check=True)

    kb_count_dry_run = f"kb count -t {threads} -i {index} -g {t2g} -x bulk -k {k} --dry-run --parity single -o {kb_count_out} {fastq}"  # should be the same as the kb count run before with the exception of removing --h5ad, swapping in the newer kallisto for the kallisto bus command, and adding --union and --dfk-onlist  # TODO: add support for more kb arguments
    if "--h5ad" in kb_count_dry_run:
        kb_count_dry_run = kb_count_dry_run.replace("--h5ad", "")  # not supported

    result = subprocess.run(
        kb_count_dry_run, shell=True, stdout=subprocess.PIPE, text=True
    )
    commands = result.stdout.strip().split("\n")

    for cmd in commands:
        # print(f"Running command: {cmd}")
        if "kallisto bus" in cmd:
            cmd_split = cmd.split()
            cmd_split[0] = newer_kallisto
            cmd_split.insert(2, "--union")
            cmd_split.insert(3, "--dfk-onlist")
            cmd = " ".join(cmd_split)
        result = subprocess.run(cmd, shell=True)
        if result.returncode != 0:
            print(f"Command failed: {cmd}")
            break


def create_umi_to_barcode_dict(
    bus_file, bustools="bustools", barcode_length=16, key_to_use="umi"
):
    umi_to_barcode_dict = {}

    # Define the command
    # !/home/jrich/miniconda3/envs/cartf/lib/python3.10/site-packages/kb_python/bins/linux/bustools/bustools text -p -a -f -d /home/jrich/Desktop/CART_prostate_sc/TEMP_dlist_tests/kb_count_out_delaney/output.bus
    command = [
        bustools,
        "text",
        "-p",
        "-a",
        "-f",
        "-d",
        bus_file,
    ]

    # Run the command and capture the output
    result = subprocess.run(command, stdout=subprocess.PIPE, text=True)

    # Loop through each line of the output (excluding the last line 'Read in X BUS records')
    for line in result.stdout.strip().split("\n"):
        # Split the line into columns (assuming it's tab or space-separated)
        columns = line.split("\t")  # If columns are space-separated, use .split()
        if key_to_use == "umi":
            umi = columns[2]
        elif key_to_use == "fastq_header_position":
            umi = columns[5]
        barcode = columns[0]  # remember there will be A's for padding to 32 characters
        barcode = barcode[(32 - barcode_length) :]  # * remove the padding
        umi_to_barcode_dict[umi] = barcode

    return umi_to_barcode_dict


def check_if_read_dlisted_by_one_of_its_respective_dlist_sequences(
    mcrs_header, mcrs_header_to_seq_dict, dlist_header_to_seq_dict, k
):
    # do a bowtie (or manual) alignment of breaking the mcrs seq into k-mers and aligning to the dlist seqs dervied from the same mcrs header
    dlist_header_to_seq_dict_filtered = {
        key: value
        for key, value in dlist_header_to_seq_dict.items()
        if mcrs_header == key.rsplit("_", 1)[0]
    }
    mcrs_sequence = mcrs_header_to_seq_dict[mcrs_header]
    for i in range(len(mcrs_sequence) - k + 1):
        kmer = mcrs_sequence[i : i + k]
        for dlist_sequence in dlist_header_to_seq_dict_filtered.values():
            if kmer in dlist_sequence:
                return True
    return False


def convert_to_list_in_df(value, reference_length=0):
    if isinstance(value, str):
        try:
            # Safely convert string representation of a list to an actual list
            return ast.literal_eval(value)
        except (ValueError, SyntaxError):
            # If conversion fails, return an empty list or handle accordingly
            if reference_length == 0:
                return []
            else:
                return [np.nan] * reference_length
    return value  # If already a list, return as is


def safe_literal_eval(val):
    if isinstance(val, str) and val.startswith("[") and val.endswith("]"):
        val = (
            val.replace("np.nan", "None").replace("nan", "None").replace("<NA>", "None")
        )
        try:
            # Attempt to parse the string as a literal
            parsed_val = ast.literal_eval(val)
            # If it's a list with NaN values, replace each entry with np.nan
            if isinstance(parsed_val, list):
                return [
                    np.nan if isinstance(i, float) and np.isnan(i) else i
                    for i in parsed_val
                ]
            return parsed_val
        except (ValueError, SyntaxError):
            # If not a valid literal, return the original value
            return val
    else:
        return val


def increment_adata_based_on_dlist_fns(
    adata,
    mcrs_fasta,
    dlist_fasta,
    kb_count_out,
    t2g,
    fastq,
    newer_kallisto,
    k=31,
    mm=False,
    assay="bulk",
    bustools="bustools",
):
    run_kb_count_dry_run(
        index="index.idx",
        t2g=t2g,
        fastq=fastq,
        kb_count_out=kb_count_out,
        newer_kallisto=newer_kallisto,
        k=k,
        threads=1,
    )

    if not os.path.exists(f"{kb_count_out}/bus_df.csv"):
        bus_df = make_bus_df(
            kb_count_out,
            fastq,
            t2g=t2g,
            mm=mm,
            union=False,
            assay=assay,
            bustools=bustools,
        )
    else:
        bus_df = pd.read_csv(f"{kb_count_out}/bus_df.csv")

    # with open(f"{kb_count_out}/transcripts.txt") as f:
    #     dlist_index = str(sum(1 for line in file))

    n_rows, n_cols = adata.X.shape
    increment_matrix = csr_matrix((n_rows, n_cols))

    mcrs_header_to_seq_dict = (
        create_header_to_sequence_ordered_dict_from_fasta_WITHOUT_semicolon_splitting(
            mcrs_fasta
        )
    )
    dlist_header_to_seq_dict = (
        create_header_to_sequence_ordered_dict_from_fasta_WITHOUT_semicolon_splitting(
            dlist_fasta
        )
    )
    var_names_to_idx_in_adata_dict = {
        name: idx for idx, name in enumerate(adata.var_names)
    }

    # Apply to the whole column at once
    bus_df["gene_names_final"] = bus_df["gene_names_final"].apply(safe_literal_eval)  # TODO: consider looking through gene_names_final_set rather than gene_names_final for possible speedup (but make sure safe_literal_eval supports this)

    # iterate through bus_df rows
    for _, row in bus_df.iterrows():
        if "dlist" in row["gene_names_final"] and (
            mm or len(row["gene_names_final"]) == 2
        ):  # don't replace with row['counted_in_count_matrix'] because this is the bus from when I ran union
            read_dlisted_by_one_of_its_respective_dlist_sequences = False
            for mcrs_header in row["gene_names_final"]:
                if mcrs_header != "dlist":
                    read_dlisted_by_one_of_its_respective_dlist_sequences = (
                        check_if_read_dlisted_by_one_of_its_respective_dlist_sequences(
                            mcrs_header=mcrs_header,
                            mcrs_header_to_seq_dict=mcrs_header_to_seq_dict,
                            dlist_header_to_seq_dict=dlist_header_to_seq_dict,
                            k=k,
                        )
                    )
                    if read_dlisted_by_one_of_its_respective_dlist_sequences:
                        break
            if not read_dlisted_by_one_of_its_respective_dlist_sequences:
                # barcode_idx = [i for i, name in enumerate(adata.obs_names) if barcode.endswith(name)][0]  # if I did not remove the padding
                barcode_idx = np.where(adata.obs_names == row["barcode"])[0][
                    0
                ]  # if I previously removed the padding
                mcrs_idxs = [
                    var_names_to_idx_in_adata_dict[header]
                    for header in row["gene_names_final"]
                    if header in var_names_to_idx_in_adata_dict
                ]

                increment_matrix[barcode_idx, mcrs_idxs] += row["count"]

    # print("Gene list:", list(adata.var.index))
    # print(
    #     "Increment matrix",
    #     (increment_matrix.toarray() if hasattr(increment_matrix, "toarray") else increment_matrix),
    # )
    # print(
    #     "Adata matrix original",
    #     adata.X.toarray() if hasattr(adata.X, "toarray") else adata.X,
    # )

    if not isinstance(adata.X, csr_matrix):
        adata.X = adata.X.tocsr()

    if not isinstance(increment_matrix, csr_matrix):
        increment_matrix = increment_matrix.tocsr()

    # Add the two sparse matrices
    adata.X = adata.X + increment_matrix

    adata.X = csr_matrix(adata.X)

    # print(
    #     "Adata matrix final",
    #     adata.X.toarray() if hasattr(adata.X, "toarray") else adata.X,
    # )

    return adata


def contains_kmer_in_mcrs(read_sequence, mcrs_sequence, k):
    return any(
        read_sequence[i : i + k] in mcrs_sequence
        for i in range(len(read_sequence) - k + 1)
    )


def check_for_read_kmer_in_mcrs(read_df, unique_mcrs_df, k, subset=None, strand=None):
    """
    Adds a column 'read_contains_kmer_in_mcrs' to read_df_subset indicating whether a k-mer
    from the read_sequence exists in the corresponding mcrs_sequence.

    Parameters:
    - read_df_subset: The subset of the read_df DataFrame (e.g., read_df.loc[read_df['FN']])
    - unique_mcrs_df: DataFrame containing 'mcrs_header' and 'mcrs_sequence' for lookups
    - k: The length of the k-mers to check for

    Returns:
    - The original DataFrame with the new 'read_contains_kmer_in_mcrs' column
    """

    def check_row_for_kmer(row):
        read_sequence = row["read_sequence"]
        if strand != "r":
            mcrs_sequence = mcrs_sequence_dict.get(row["mcrs_header"], "")
            contains_kmer_in_mcrs_f = contains_kmer_in_mcrs(
                read_sequence, mcrs_sequence, k
            )
            if strand == "f":
                return contains_kmer_in_mcrs_f
        if strand != "f":
            mcrs_sequence_rc = mcrs_sequence_dict_rc.get(row["mcrs_header"], "")
            contains_kmer_in_mcrs_r = contains_kmer_in_mcrs(
                Seq(read_sequence).reverse_complement(), mcrs_sequence_rc, k
            )
            if strand == "r":
                return contains_kmer_in_mcrs_r
        return contains_kmer_in_mcrs_f or contains_kmer_in_mcrs_r

    # Step 1: Create a dictionary to map 'mcrs_header' to 'mcrs_sequence' for fast lookups
    if strand != "r":
        mcrs_sequence_dict = unique_mcrs_df.set_index("mcrs_header")[
            "mcrs_sequence"
        ].to_dict()
    if strand != "f":
        mcrs_sequence_dict_rc = unique_mcrs_df.set_index("mcrs_header")[
            "mcrs_sequence_rc"
        ].to_dict()

    # Step 4: Initialize the column with NaN in the original read_df subset
    if "read_contains_kmer_in_mcrs" not in read_df.columns:
        read_df["read_contains_kmer_in_mcrs"] = np.nan

    # Step 5: Apply the function and update the 'read_contains_kmer_in_mcrs' column
    if subset is None:
        read_df["read_contains_kmer_in_mcrs"] = read_df.apply(
            lambda row: check_row_for_kmer(row), axis=1
        )
    else:
        read_df.loc[read_df[subset], "read_contains_kmer_in_mcrs"] = read_df.loc[
            read_df[subset]
        ].apply(lambda row: check_row_for_kmer(row, k), axis=1)

    return read_df


def get_valid_ensembl_gene_id(
    row, transcript_column: str = "seq_ID", gene_column: str = "gene_name"
):
    ensembl_gene_id = get_ensembl_gene_id(row[transcript_column])
    if ensembl_gene_id == "Unknown":
        return row[gene_column]
    return ensembl_gene_id


def get_ensembl_gene_id(transcript_id: str, verbose: bool = False):
    try:
        url = f"https://rest.ensembl.org/lookup/id/{transcript_id}?expand=1"
        response = requests.get(url, headers={"Content-Type": "application/json"})

        if not response.ok:
            response.raise_for_status()

        data = response.json()

        return data.get("Parent")
    except Exception as e:
        if verbose:
            print(f"Error for: {transcript_id}")
        return "Unknown"


def get_ensembl_gene_id_bulk(transcript_ids: list[str]) -> dict[str, str]:
    if not transcript_ids:
        return {}

    try:
        url = f"https://rest.ensembl.org/lookup/id/"
        response = requests.post(
            url,
            json={"ids": transcript_ids},
            headers={"Content-Type": "application/json"},
        )

        if not response.ok:
            response.raise_for_status()

        data = response.json()

        return {
            transcript_id: data[transcript_id].get("Parent")
            for transcript_id in transcript_ids
            if data[transcript_id]
        }
    except Exception as e:
        print(f"Failed to fetch gene IDs from Ensembl: {e}")
        raise e


def get_ensembl_gene_name_bulk(gene_ids: list[str]) -> dict[str, str]:
    if not gene_ids:
        return {}

    try:
        url = f"https://rest.ensembl.org/lookup/id/"
        response = requests.post(
            url, json={"ids": gene_ids}, headers={"Content-Type": "application/json"}
        )

        if not response.ok:
            response.raise_for_status()

        data = response.json()

        return {
            gene_id: data[gene_id].get("display_name")
            for gene_id in gene_ids
            if data[gene_id]
        }
    except Exception as e:
        print(f"Failed to fetch gene names from Ensembl: {e}")
        raise e


def get_valid_ensembl_gene_id_bulk(
    df: pd.DataFrame,
) -> Callable[[pd.Series, str, str], str]:
    map_: dict[str, str] | None = None

    def f(
        row: pd.Series,
        transcript_column: str = "seq_ID",
        gene_column: str = "gene_name",
    ):
        # logger.info(f"Row: {row}")
        nonlocal map_
        if map_ is None:
            all_transcript_ids = df[transcript_column].unique()
            map_ = get_ensembl_gene_id_bulk(list(all_transcript_ids))

        ensembl_gene_id = map_.get(row[transcript_column], "Unknown")
        if ensembl_gene_id == "Unknown":
            return row[gene_column]

        return ensembl_gene_id

    return f


# # Example usage
# transcript_id = "ENST00000562955"
# gene_id = get_ensembl_gene_id(transcript_id)
# gene_id


def run_bowtie_build_dlist(ref_fa, ref_folder, ref_prefix, bowtie2_build, threads=2):
    if not os.path.exists(ref_folder) or not os.listdir(ref_folder):
        print("Running bowtie2 build")
        os.makedirs(ref_folder, exist_ok=True)
        bowtie_reference_prefix = os.path.join(ref_folder, ref_prefix)
        subprocess.run(
            [
                bowtie2_build,  # Path to the bowtie2-build executable
                "--threads",
                str(threads),  # Number of threads
                ref_fa,  # Input FASTA file
                bowtie_reference_prefix,  # Output reference folder
            ],
            check=True,
        )

        print("Bowtie2 build complete")


def run_bowtie_alignment_dlist(
    output_sam_file,
    read_fa,
    ref_folder,
    ref_prefix,
    bowtie2,
    threads=2,
    k=31,
    strandedness=False,
    N_penalty=1,
    max_Ns_per_read_length=0,
    output_stat_file=None,
):
    if not os.path.exists(output_sam_file):
        print("Running bowtie2 alignment")

        os.makedirs(os.path.dirname(output_sam_file), exist_ok=True)

        bowtie_reference_prefix = os.path.join(ref_folder, ref_prefix)

        bowtie2_alignment_command = [
            bowtie2,  # Path to the bowtie2 executable
            "-a",  # Report all alignments
            "-f",  # Input file is in FASTA format
            "-p",
            str(threads),  # Number of threads
            "--xeq",  # Match different quality scores
            "--score-min",
            "C,0,0",  # Minimum score threshold
            "--np",
            str(N_penalty),  # No penalty for ambiguous matches
            "--n-ceil",
            f"L,0,{max_Ns_per_read_length}",  # N-ceiling
            "-F",
            f"{k},1",
            "-R",
            "1",  # Maximum Re-seed attempts
            "-N",
            "0",  # Maximum mismatches in seed alignment
            "-L",
            "31",  # Length of seed substrings
            "-i",
            "C,1,0",  # Interval between seed extensions
            "--no-1mm-upfront",  # No mismatches upfront
            "--no-unal",  # Do not write unaligned reads
            "--no-hd",  # Suppress header lines in SAM output
            "-x",
            bowtie_reference_prefix,  # Reference folder for alignment
            "-U",
            read_fa,  # Input FASTA file
            "-S",
            output_sam_file,  # Output SAM file
        ]

        if strandedness:
            bowtie2_alignment_command.insert(3, "--norc")

        result = subprocess.run(
            bowtie2_alignment_command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=True,
        )

        if output_stat_file is not None:
            if os.path.exists(output_stat_file):
                write_mode = "a"
            else:
                write_mode = "w"
            with open(output_stat_file, write_mode) as f:
                f.write(f"bowtie alignment for {bowtie_reference_prefix}")
                f.write("Standard Output:\n")
                f.write(result.stdout)
                f.write("\n\nStandard Error:\n")
                f.write(result.stderr)
                f.write("\n\n")

        print("Bowtie2 alignment complete")


import csv


def make_mapping_dict(id_to_header_csv, dict_key="id"):
    mapping_dict = {}
    with open(id_to_header_csv, newline="") as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            seq_id, header = row
            if dict_key == "id":
                mapping_dict[seq_id] = header
            elif dict_key == "header":
                mapping_dict[header] = seq_id
    return mapping_dict


def swap_ids_for_headers_in_fasta(in_fasta, id_to_header_csv, out_fasta=None):
    if out_fasta is None:
        out_fasta = in_fasta.replace(".fa", "_with_headers.fa")

    if id_to_header_csv.endswith(".csv"):
        id_to_header = make_mapping_dict(id_to_header_csv, dict_key="id")
    else:
        id_to_header = id_to_header_csv

    with open(out_fasta, "w") as output_file:
        for seq_id, sequence in read_fasta(in_fasta):
            output_file.write(f">{id_to_header[seq_id]}\n{sequence}\n")

    print("Swapping complete")


def swap_headers_for_ids_in_fasta(in_fasta, id_to_header_csv, out_fasta=None):
    if out_fasta is None:
        out_fasta = in_fasta.replace(".fa", "_with_ids.fa")

    if id_to_header_csv.endswith(".csv"):
        header_to_id = make_mapping_dict(id_to_header_csv, dict_key="header")
    else:
        header_to_id = id_to_header_csv

    with open(out_fasta, "w") as output_file:
        for header, sequence in read_fasta(in_fasta):
            output_file.write(f">{header_to_id[header]}\n{sequence}\n")

    print("Swapping complete")


def get_mcrs_headers_that_are_substring_dlist(
    mutation_reference_file_fasta,
    dlist_fasta_file,
    header_column_name="mcrs_id",
    strandedness=False,
):
    mcrs_headers_that_are_substring_dlist = []

    mutant_reference = (
        create_header_to_sequence_ordered_dict_from_fasta_WITHOUT_semicolon_splitting(
            mutation_reference_file_fasta
        )
    )  # TODO: replace with pyfastx

    for dlist_header, dlist_sequence in read_fasta(dlist_fasta_file):
        mcrs_header = dlist_header.rsplit("_", 1)[0]
        if sequence_match(
            mutant_reference[mcrs_header], dlist_sequence, strandedness=strandedness
        ):
            mcrs_headers_that_are_substring_dlist.append(mcrs_header)

    df = pd.DataFrame(
        mcrs_headers_that_are_substring_dlist, columns=[header_column_name]
    ).drop_duplicates()

    df[f"number_of_substring_matches_to_normal_human_reference"] = df[
        header_column_name
    ].map(pd.Series(mcrs_headers_that_are_substring_dlist).value_counts())

    df["dlist_substring"] = True

    return df


def concatenate_fasta(fasta_files, output_file):
    with open(output_file, "w") as outfile:
        for fasta_file in fasta_files:
            with open(fasta_file, "r") as infile:
                outfile.write(infile.read())


def longest_homopolymer(sequence):
    # Use regex to find all homopolymer stretches (e.g., A+, C+, G+, T+)
    homopolymers = re.findall(r"(A+|C+|G+|T+)", sequence)

    if homopolymers:
        # Find the length of the longest homopolymer
        max_length = len(max(homopolymers, key=len))

        # Collect all homopolymers that have the same length as the longest
        longest_homopolymers = [h for h in homopolymers if len(h) == max_length]

        # If there is only one longest homopolymer, return it as a string
        if len(longest_homopolymers) == 1:
            return max_length, longest_homopolymers[0]
        # If there are multiple longest homopolymers, return them as a list
        else:
            return max_length, sorted(list(set(longest_homopolymers)))
    else:
        return 0, None  # If no homopolymer is found


def triplet_stats(sequence):
    # Create a list of 3-mers (triplets) from the sequence
    triplets = [sequence[i : i + 3] for i in range(len(sequence) - 2)]

    # Number of distinct triplets
    distinct_triplets = set(triplets)

    # Number of total triplets
    total_triplets = len(triplets)

    # Triplet complexity: ratio of distinct triplets to total triplets
    triplet_complexity = (
        len(distinct_triplets) / total_triplets if total_triplets > 0 else 0
    )

    return len(distinct_triplets), total_triplets, triplet_complexity


def get_mcrss_that_pseudoalign_but_arent_dlisted(
    mutation_metadata_df,
    mcrs_id_column,
    mcrs_fa,
    sequence_names_set,
    human_reference_genome_fa,
    human_reference_gtf,
    out_dir_notebook=".",
    ref_folder_kb=None,
    dlist_reference_source="",
    header_column_name="mcrs_id",
    additional_kb_extract_filtering_workflow="nac",
    k=31,
    threads=2,
    strandedness=False,
    column_name="pseudoaligned_to_human_reference_despite_not_truly_aligning",
):
    if ref_folder_kb is None:
        ref_folder_kb = out_dir_notebook
    mcrs_fa_filtered_bowtie = mcrs_fa.replace(".fa", "_filtered_bowtie.fa")
    mcrs_fQ_filtered_bowtie = mcrs_fa_filtered_bowtie.replace(".fa", ".fq")
    kb_ref_wt = f"{ref_folder_kb}/{dlist_reference_source}_reference"
    os.makedirs(kb_ref_wt, exist_ok=True)
    kb_human_reference_index_file = f"{kb_ref_wt}/index.idx"
    kb_human_reference_t2g_file = f"{kb_ref_wt}/t2g.txt"
    kb_human_reference_f1_file = f"{kb_ref_wt}/f1.fasta"
    if additional_kb_extract_filtering_workflow == "standard":
        kb_ref_workflow_line = ["--workflow=standard"]
    elif additional_kb_extract_filtering_workflow == "nac":
        kb_human_reference_f2_file = f"{kb_ref_wt}/f2.fasta"
        kb_human_reference_c1_file = f"{kb_ref_wt}/c1.fasta"
        kb_human_reference_c2_file = f"{kb_ref_wt}/c2.fasta"
        kb_ref_workflow_line = [
            "-f2",
            kb_human_reference_f2_file,
            "-c1",
            kb_human_reference_c1_file,
            "-c2",
            kb_human_reference_c2_file,
            "--workflow=nac",
            "--make-unique",
        ]

    filter_fasta(mcrs_fa, mcrs_fa_filtered_bowtie, sequence_names_set)

    fasta_to_fastq(mcrs_fa_filtered_bowtie, mcrs_fQ_filtered_bowtie, k=None)

    kb_ref_command = (
        [
            "kb",
            "ref",
            "-i",
            kb_human_reference_index_file,
            "-g",
            kb_human_reference_t2g_file,
            "-f1",
            kb_human_reference_f1_file,
        ]
        + kb_ref_workflow_line
        + [
            "--d-list=None",
            "-k",
            str(k),
            "-t",
            str(threads),
            human_reference_genome_fa,
            human_reference_gtf,
        ]
    )

    if not os.path.exists(kb_human_reference_index_file):
        subprocess.run(kb_ref_command, check=True)
        # subprocess.run(" ".join(kb_ref_command), shell=True, check=True)

    kb_extract_out_dir_bowtie_filtered = (
        f"{out_dir_notebook}/kb_extract_bowtie_filtered"
    )

    kb_extract_command = [
        "kb",
        "extract",
        "--extract_all_fast",
        "--mm",
        "--verbose",
        "-t",
        str(threads),
        "-k",
        str(k),
        "-o",
        kb_extract_out_dir_bowtie_filtered,
        "-i",
        kb_human_reference_index_file,
        "-g",
        kb_human_reference_t2g_file,
        mcrs_fQ_filtered_bowtie,
    ]

    if strandedness:
        kb_extract_command = (
            kb_extract_command[:4] + ["--strand", "forward"] + kb_extract_command[4:]
        )

    try:
        subprocess.run(kb_extract_command, check=True)

        kb_extract_output_fastq_file = (
            f"{kb_extract_out_dir_bowtie_filtered}/all/1.fastq.gz"
        )

        problematic_mutations_total = parse_fastq(kb_extract_output_fastq_file)

        df = pd.DataFrame(
            problematic_mutations_total, columns=[header_column_name]
        ).drop_duplicates()

        df[column_name] = True

        df[mcrs_id_column] = df[mcrs_id_column].astype(str)

        mutation_metadata_df = pd.merge(
            mutation_metadata_df,
            df,
            on=mcrs_id_column,
            how="left",
        )

    except Exception as e:
        print("No reads pseudoaligned - setting entire column to False")
        mutation_metadata_df[column_name] = False

    return mutation_metadata_df


def get_df_overlap(
    mcrs_fa,
    out_dir_notebook=".",
    k=31,
    strandedness=False,
    mcrs_id_column="mcrs_id",
    output_text_file=None,
    output_plot_folder=None,
):
    df_overlap_save_path = f"{out_dir_notebook}/kmer_overlap_stats.csv"
    df_overlap = count_kmer_overlaps_new(
        mcrs_fa, k=k, strandedness=strandedness, mcrs_id_column=mcrs_id_column
    )
    df_overlap.to_csv(df_overlap_save_path, index=False)

    print_column_summary_stats(
        df_overlap,
        "number_of_kmers_with_overlap_to_other_mcrs_items_in_mcrs_reference",
        output_file=output_text_file,
    )
    print_column_summary_stats(
        df_overlap,
        "number_of_mcrs_items_with_overlapping_kmers_in_mcrs_reference",
        output_file=output_text_file,
    )

    kmer_plot_file = f"{output_plot_folder}/kmer_overlap_histogram.png"

    plot_histogram_notebook_1(
        df_overlap,
        column="number_of_kmers_with_overlap_to_other_mcrs_items_in_mcrs_reference",
        x_label="Number of K-mers with overlap",
        title="Histogram of K-mers with Overlap",
        output_plot_file=kmer_plot_file,
    )

    mcrs_plot_file = f"{output_plot_folder}/mcrs_overlap_histogram.png"

    plot_histogram_notebook_1(
        df_overlap,
        column="number_of_mcrs_items_with_overlapping_kmers_in_mcrs_reference",
        x_label="Number of MCRS items with Overlapping K-mers",
        title="Histogram of MCRS items with Overlapping K-mers",
        output_plot_file=mcrs_plot_file,
    )

    df_overlap[mcrs_id_column] = df_overlap[mcrs_id_column].astype(str)

    return df_overlap


from pdb import set_trace as st


def apply_filters(df, filters, verbose=False, logger=None):
    for filter in filters:
        column = filter["column"]
        rule = filter["rule"]
        value = filter["value"]

        if column not in df.columns:
            # skip this iteration
            continue

        len_df_start = len(df)
        # TODO: if I want number of mutations (in addition to number of MCRSs), then calculate the length of the df exploded by header

        if rule == "min":
            df = df.loc[df[column].astype(float) >= float(value)]
        elif rule == "max":
            df = df.loc[df[column].astype(float) <= float(value)]
        elif rule == "between":
            value_min, value_max = value.split(",")
            value_min, value_max = float(value_min), float(value_max)
            df = df.loc[(df[column] >= value_min) & (df[column] <= value_max)]
        elif rule == "equal":
            df = df.loc[df[column].astype(str) == str(value)]
        elif rule == "notequal":
            df = df.loc[df[column].astype(str) != str(value)]
        elif rule == "isin":
            # check if value is a path to a txt file
            if type(value) == str:
                if value.endswith(".txt"):
                    value = set(convert_txt_to_list(value))
                else:
                    value = ast.literal_eval(value)
            df = df.loc[df[column].isin(set(value))]
        elif rule == "isnotin":
            if type(value) == str:
                if value.endswith(".txt"):
                    value = set(convert_txt_to_list(value))
                else:
                    value = ast.literal_eval(value)
            df = df.loc[~df[column].isin(set(value))]
        elif rule == "isnull":
            df = df.loc[df[column].isnull()]
        elif rule == "isnotnull":
            df = df.loc[df[column].notnull()]
        elif rule == "istrue":
            df = df.loc[df[column] == True]
        elif rule == "isfalse":
            df = df.loc[df[column] == False]
        elif rule == "isnottrue":
            df = df.loc[(df[column] != True) | df[column].isnull()]
        elif rule == "isnotfalse":
            df = df.loc[(df[column] != False) | df[column].isnull()]

        len_df_end = len(df)
        number_filtered = len_df_start - len_df_end

        if verbose:
            logger.info(
                f"Filtered {number_filtered} mutations {column} with {rule} {value} - {len_df_end} mutations remaining"
            )

    return df


def parse_filters(args):
    filters = []
    for col_rule, value in args:
        column, rule = col_rule.split("-")
        if rule in ["min", "max"]:
            value = float(value) if "." in value else int(value)
        elif rule in ["istrue", "isfalse", "isnottrue", "isnotfalse"]:
            value = True if "true" in rule else False
        filters.append({"column": column, "rule": rule, "value": value})
    return filters


def prepare_filters_json(filters):
    with open(filters, "r") as f:
        return json.load(f)


def prepare_filters_list(filters):
    filter_list = []

    if type(filters) == str and filters.endswith(".txt"):
        filters = convert_txt_to_list(filters)

    for f in filters:
        try:
            # from pdb import set_trace as st; st()
            f_split = f.split("=")
            if len(f_split) == 1 and any(
                substring in f_split[0]
                for substring in [
                    "istrue",
                    "isfalse",
                    "isnottrue",
                    "isnotfalse",
                    "isnull",
                    "isnotnull",
                ]
            ):
                col_rule = f_split[0]
                value = None
            elif len(f_split) == 2:
                col_rule, value = f_split
            else:
                raise ValueError(
                    f"Filter format invalid: {f}. Expected 'Column-RULE=VALUE'"
                )
            filter_list.append((col_rule, value))
        except ValueError:
            raise ValueError(
                f"Filter format invalid: {f}. Expected 'Column-RULE=VALUE'"
            )
    filters = parse_filters(filter_list)
    return filters


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


def explode_df(mutation_metadata_df, columns_to_explode=None):
    if columns_to_explode is None:
        columns_to_explode = ["header", "order"]
    else:  # * remove with set
        columns_to_explode = columns_to_explode.copy()
        columns_to_explode = list(set(columns_to_explode))
    if "header_list" not in mutation_metadata_df.columns:
        mutation_metadata_df["header_list"] = mutation_metadata_df[
            "mcrs_header"
        ].str.split(";")
    if "order_list" not in mutation_metadata_df.columns:
        mutation_metadata_df["order_list"] = mutation_metadata_df["header_list"].apply(
            lambda x: list(range(len(x)))
        )

    mutation_metadata_df["header"] = mutation_metadata_df["header_list"]
    mutation_metadata_df["order"] = mutation_metadata_df["order_list"]

    # for column in columns_to_explode:
    #     mutation_metadata_df[column] = mutation_metadata_df.apply(
    #         lambda row: convert_to_list_in_df(row[column], len(row['header']) if isinstance(row['header'], list) else 1),
    #         axis=1
    #     )

    print("About to apply safe evals")
    for column in tqdm(columns_to_explode, desc="Checking columns"):
        mutation_metadata_df[column] = mutation_metadata_df.apply(
            lambda row: safe_literal_eval(row[column]), axis=1
        )

    mutation_metadata_df_exploded = mutation_metadata_df.explode(
        list(columns_to_explode)
    ).reset_index(drop=True)

    return mutation_metadata_df_exploded


def collapse_df(
    mutation_metadata_df_exploded,
    columns_to_explode=None,
    columns_to_explode_extend_values=None,
):
    if columns_to_explode is None:
        columns_to_explode = ["header", "order"]
    else:  # * remove with set
        columns_to_explode = columns_to_explode.copy()

    if columns_to_explode_extend_values:
        if type(columns_to_explode_extend_values) == list:
            columns_to_explode.extend(
                columns_to_explode_extend_values
            )  # * .update(items) for set
        elif type(columns_to_explode_extend_values) == str:
            columns_to_explode.append(
                columns_to_explode_extend_values
            )  # * .add(items) for set

    for column in list(columns_to_explode):
        mutation_metadata_df_exploded[column] = mutation_metadata_df_exploded[
            column
        ].apply(lambda x: tuple(x) if isinstance(x, list) else x)

    mutation_metadata_df = (
        mutation_metadata_df_exploded.sort_values("order")
        .groupby("mcrs_header", as_index=False)
        .agg(
            {
                **{col: list for col in list(columns_to_explode)},  # list these values
                **{
                    col: "first"
                    for col in mutation_metadata_df_exploded.columns
                    if col not in list(columns_to_explode) + ["mcrs_header"]
                },
            }  # Take the first value for other columns
        )
        .reset_index(drop=True)
    )

    return mutation_metadata_df, columns_to_explode


# TODO: save in txt file
def fasta_summary_stats(fa, output_file=None):
    try:
        if type(fa) == str:
            fa = pyfastx.Fasta(fa)

        try:
            gc_content = fa.gc_content
        except Exception as e:
            gc_content = None

        nucleotide_composition = fa.composition
        total_sequence_length = fa.size
        longest_sequence = fa.longest
        longest_sequence_length = len(longest_sequence)
        longest_sequence_name = longest_sequence.name
        shortest_sequence = fa.shortest
        shortest_sequence_length = len(shortest_sequence)
        shortest_sequence_name = shortest_sequence.name
        mean_sequence_length = fa.mean
        median_sequence_length = fa.median
        number_of_sequences_longer_than_mean = fa.count(math.ceil(mean_sequence_length))

        summary = [
            f"Total sequence length: {total_sequence_length}",
            f"GC content: {gc_content}",
            f"Nucleotide composition: {nucleotide_composition}",
            f"Longest sequence length: {longest_sequence_length}",
            f"Longest sequence name: {longest_sequence_name}",
            f"Shortest sequence length: {shortest_sequence_length}",
            f"Shortest sequence name: {shortest_sequence_name}",
            f"Mean sequence length: {mean_sequence_length}",
            f"Median sequence length: {median_sequence_length}",
            f"Number of sequences longer than mean: {number_of_sequences_longer_than_mean}",
        ]

        # Print to console and save to file
        if output_file is not None:
            with open(output_file, "w") as f:
                for line in summary:
                    f.write(line + "\n")  # Write to file with a newline

    except Exception as e:
        print(f"Error: {e}")


def compare_dicts(dict1, dict2):
    # Find keys that are only in one of the dictionaries
    keys_only_in_dict1 = dict1.keys() - dict2.keys()
    keys_only_in_dict2 = dict2.keys() - dict1.keys()

    # Find keys that are in both dictionaries with differing values
    differing_values = {
        k: (dict1[k], dict2[k])
        for k in dict1.keys() & dict2.keys()
        if dict1[k] != dict2[k]
    }

    # Report results
    if keys_only_in_dict1:
        print("Keys only in dict1:", keys_only_in_dict1)
    if keys_only_in_dict2:
        print("Keys only in dict2:", keys_only_in_dict2)
    if differing_values:
        print("Keys with differing values:", differing_values)
    if not keys_only_in_dict1 and not keys_only_in_dict2 and not differing_values:
        print("Dictionaries are identical.")


def calculate_total_gene_info(
    mutation_metadata_df_exploded,
    mcrs_id_column="mcrs_id",
    output_stat_file=None,
    output_plot_folder=None,
    columns_to_include="all",
    columns_to_explode=None,
):
    if columns_to_explode is None:
        columns_to_explode = ["header"]
    else:
        columns_to_explode = columns_to_explode.copy()

    number_of_mutations_total = len(
        mutation_metadata_df_exploded[mcrs_id_column].unique()
    )
    number_of_transcripts_total = len(mutation_metadata_df_exploded["seq_ID"].unique())
    number_of_genes_total = len(mutation_metadata_df_exploded["gene_name"].unique())

    metadata_counts_dict = {
        "Mutations_total": number_of_mutations_total,
        "Transcripts_total": number_of_transcripts_total,
        "Genes_total": number_of_genes_total,
    }

    if output_stat_file is not None:
        with open(output_stat_file, "w") as f:
            for key, value in metadata_counts_dict.items():
                f.write(f"{key}: {value}\n")

    output_plot_file_basic_bar_plot = f"{output_plot_folder}/basic_bar_plot.png"

    plot_basic_bar_plot_from_dict(
        metadata_counts_dict,
        "Counts",
        log_scale=True,
        output_file=output_plot_file_basic_bar_plot,
    )

    if columns_to_include == "all" or "header_with_gene_name" in columns_to_include:
        mutation_metadata_df_exploded["header_with_gene_name"] = (
            mutation_metadata_df_exploded["header"].str.split(":", n=1).str[0]
            + "("
            + mutation_metadata_df_exploded["gene_name"]
            + "):"
            + mutation_metadata_df_exploded["header"].str.split(":", n=1).str[1]
        )

    if (
        columns_to_include == "all"
        or "number_of_mutations_in_this_gene_total" in columns_to_include
    ):
        gene_counts = mutation_metadata_df_exploded["gene_name"].value_counts()
        mutation_metadata_df_exploded["number_of_mutations_in_this_gene_total"] = (
            mutation_metadata_df_exploded["gene_name"].map(gene_counts)
        )

        output_plot_file_descending_bar_plot = (
            f"{output_plot_folder}/descending_bar_plot.png"
        )

        plot_descending_bar_plot(
            gene_counts,
            x_label="Gene Name",
            y_label="Number of Occurrences",
            tick_interval=5000,
            output_file=output_plot_file_descending_bar_plot,
        )

    columns_to_explode.append("number_of_mutations_in_this_gene_total")

    return mutation_metadata_df_exploded, columns_to_explode


def calculate_nearby_mutations(
    mcrs_source_column,
    k,
    output_plot_folder,
    mcrs_source,
    mutation_metadata_df_exploded,
    columns_to_explode=None,
):
    if columns_to_explode is None:
        columns_to_explode = ["header", "order"]
    else:  # * remove with set
        columns_to_explode = columns_to_explode.copy()

    columns_to_explode_extend_values = [
        "nearby_mutations",
        "nearby_mutations_count",
        "has_a_nearby_mutation",
    ]

    if mcrs_source != "combined":
        mutation_metadata_df_exploded_copy = mutation_metadata_df_exploded.copy()
        mutation_metadata_df_exploded_copy = (
            count_nearby_mutations_efficient_with_identifiers(
                mutation_metadata_df_exploded_copy,
                k=k,
                fasta_entry_column="seq_ID",
                start_column="start_mutation_position",
                end_column="end_mutation_position",
                header_column="header",
            )
        )
        mutation_metadata_df_exploded = mutation_metadata_df_exploded.merge(
            mutation_metadata_df_exploded_copy[["header", "nearby_mutations"]],
            on="header",
            how="left",
        )
        mutation_metadata_df_exploded["nearby_mutations"] = (
            mutation_metadata_df_exploded["nearby_mutations"].apply(
                lambda x: [] if isinstance(x, float) and pd.isna(x) else x
            )
        )

    else:
        # find other mutations within (k-1) of each mutation for cDNA
        mutation_metadata_df_exploded_cdna = mutation_metadata_df_exploded.loc[
            mutation_metadata_df_exploded[mcrs_source_column] == "cdna"
        ].reset_index(drop=True)
        mutation_metadata_df_exploded_cdna = count_nearby_mutations_efficient_with_identifiers(
            mutation_metadata_df_exploded_cdna,
            k=k,
            fasta_entry_column="seq_ID",
            start_column="start_mutation_position_cdna",
            end_column="end_mutation_position_cdna",
            header_column="header",
        )  # * change to header_column='header_cdna' (along with below) if I don't want to distinguish between spliced and unspliced variants being close
        mutation_metadata_df_exploded = mutation_metadata_df_exploded.merge(
            mutation_metadata_df_exploded_cdna[["header", "nearby_mutations"]],
            on="header",
            how="left",
        )
        mutation_metadata_df_exploded.rename(
            columns={"nearby_mutations": "nearby_mutations_cdna"}, inplace=True
        )
        mutation_metadata_df_exploded["nearby_mutations_cdna"] = (
            mutation_metadata_df_exploded["nearby_mutations_cdna"].apply(
                lambda x: [] if isinstance(x, float) and pd.isna(x) else x
            )
        )
        # Step 1: Create two new columns for the length of each list, treating NaN as 0
        mutation_metadata_df_exploded["nearby_mutations_count_cdna"] = (
            mutation_metadata_df_exploded["nearby_mutations_cdna"].apply(
                lambda x: len(x) if isinstance(x, list) else 0
            )
        )
        mutation_metadata_df_exploded["has_a_nearby_mutation_cdna"] = (
            mutation_metadata_df_exploded["nearby_mutations_count_cdna"] > 0
        )
        columns_to_explode_extend_values.extend(
            [
                "nearby_mutations_cdna",
                "nearby_mutations_count_cdna",
                "has_a_nearby_mutation_cdna",
            ]
        )

        # find other mutations within (k-1) of each mutation for genome
        mutation_metadata_df_exploded_genome = (
            mutation_metadata_df_exploded.copy()
        )  # mutation_metadata_df.loc[(mutation_metadata_df[mcrs_source_column] == "cdna") | (mutation_metadata_df['cdna_and_genome_same'] != "True")].reset_index(drop=True)  #* uncomment this filtering if I only want to keep genome cases that differ from cdna
        mutation_metadata_df_exploded_genome = count_nearby_mutations_efficient_with_identifiers(
            mutation_metadata_df_exploded_genome,
            k=k,
            fasta_entry_column="chromosome",
            start_column="start_mutation_position_genome",
            end_column="end_mutation_position_genome",
            header_column="header",
        )  # * change to header_column='header_cdna' (along with above) if I don't want to distinguish between spliced and unspliced variants being close
        mutation_metadata_df_exploded = mutation_metadata_df_exploded.merge(
            mutation_metadata_df_exploded_genome[["header", "nearby_mutations"]],
            on="header",
            how="left",
        )
        mutation_metadata_df_exploded.rename(
            columns={"nearby_mutations": "nearby_mutations_genome"}, inplace=True
        )
        mutation_metadata_df_exploded["nearby_mutations_genome"] = (
            mutation_metadata_df_exploded["nearby_mutations_genome"].apply(
                lambda x: [] if isinstance(x, float) and pd.isna(x) else x
            )
        )
        # Step 1: Create two new columns for the length of each list, treating NaN as 0
        mutation_metadata_df_exploded["nearby_mutations_count_genome"] = (
            mutation_metadata_df_exploded["nearby_mutations_genome"].apply(
                lambda x: len(x) if isinstance(x, list) else 0
            )
        )
        mutation_metadata_df_exploded["has_a_nearby_mutation_genome"] = (
            mutation_metadata_df_exploded["nearby_mutations_count_genome"] > 0
        )
        columns_to_explode_extend_values.extend(
            [
                "nearby_mutations_genome",
                "nearby_mutations_count_genome",
                "has_a_nearby_mutation_genome",
            ]
        )

        mutation_metadata_df_exploded["nearby_mutations"] = (
            mutation_metadata_df_exploded.apply(
                lambda row: list(
                    set(
                        (row["nearby_mutations_cdna"])
                        + (row["nearby_mutations_genome"])
                    )
                ),
                axis=1,
            )
        )

    mutation_metadata_df_exploded["nearby_mutations_count"] = (
        mutation_metadata_df_exploded["nearby_mutations"].apply(
            lambda x: len(x) if isinstance(x, list) else 0
        )
    )
    mutation_metadata_df_exploded["has_a_nearby_mutation"] = (
        mutation_metadata_df_exploded["nearby_mutations_count"] > 0
    )
    print(
        f"Number of mutations with nearby mutations: {mutation_metadata_df_exploded['has_a_nearby_mutation'].sum()} {mutation_metadata_df_exploded['has_a_nearby_mutation'].sum() / len(mutation_metadata_df_exploded) * 100:.2f}%"
    )
    bins = min(int(mutation_metadata_df_exploded["nearby_mutations_count"].max()), 1000)
    nearby_mutations_output_plot_file = (
        f"{output_plot_folder}/nearby_mutations_histogram.png"
    )
    plot_histogram_of_nearby_mutations_7_5(
        mutation_metadata_df_exploded,
        column="nearby_mutations_count",
        bins=bins,
        output_file=nearby_mutations_output_plot_file,
    )

    columns_to_explode.extend(columns_to_explode_extend_values)

    return mutation_metadata_df_exploded, columns_to_explode


def align_to_normal_genome_and_build_dlist(
    mutations,
    mcrs_id_column,
    out_dir_notebook,
    dlist_reference_source,
    ref_prefix,
    remove_Ns,
    strandedness,
    threads,
    N_penalty,
    max_Ns_per_read_length,
    k,
    output_stat_folder,
    mutation_metadata_df,
    bowtie2_build,
    bowtie2,
    reference_out_dir_sequences_dlist,
    logger=None,
):
    bowtie_stat_file = f"{output_stat_folder}/bowtie_alignment.txt"

    if "ensembl" in dlist_reference_source:
        match = re.search(
            r"grch(\d+)_release(\d+)", dlist_reference_source, re.IGNORECASE
        )
        ensembl_grch = match.group(1)
        ensembl_release = match.group(2)
        if ensembl_grch == "37":
            ensembl_grch_gget = "human_grch37"
        else:
            ensembl_grch_gget = ensembl_grch
        if ensembl_grch == "37" and ensembl_release == "93":
            ensembl_release_gtf = "87"
        else:
            ensembl_release_gtf = ensembl_release

        # TODO: add each of these as arguments
        ref_dlist_fa_genome = f"{reference_out_dir_sequences_dlist}/Homo_sapiens.GRCh{ensembl_grch}.dna.primary_assembly.fa"
        ref_dlist_fa_cdna = f"{reference_out_dir_sequences_dlist}/Homo_sapiens.GRCh{ensembl_grch}.cdna.all.fa"
        ref_dlist_gtf = f"{reference_out_dir_sequences_dlist}/Homo_sapiens.GRCh{ensembl_grch}.{ensembl_release_gtf}.gtf"

        files_to_download_list = []
        file_dict = {
            "dna": ref_dlist_fa_genome,
            "cdna": ref_dlist_fa_cdna,
            "gtf": ref_dlist_gtf,
        }
        for file in file_dict:
            if not os.path.exists(file_dict[file]):
                files_to_download_list.append(file)
        if files_to_download_list:
            files_to_download = ",".join(files_to_download_list)
            gget_ref_command_dlist = [
                "gget",
                "ref",
                "-w",
                files_to_download,
                "-r",
                ensembl_release,
                "--out_dir",
                reference_out_dir_sequences_dlist,
                "-d",
                ensembl_grch_gget,
            ]
            subprocess.run(gget_ref_command_dlist, check=True)
            for file in files_to_download_list:
                subprocess.run(["gunzip", f"{file_dict[file]}.gz"])

    elif dlist_reference_source == "t2t":
        # TODO: check if these files already exist in reference_out_dir, and if so, skip downloading
        ref_dlist_fa_genome, ref_dlist_fa_cdna, ref_dlist_gtf = (
            download_t2t_reference_files(reference_out_dir_sequences_dlist)
        )

    ref_folder_genome_bowtie = (
        f"{reference_out_dir_sequences_dlist}/bowtie_index_genome"
    )
    ref_prefix_genome_full = f"{ref_folder_genome_bowtie}/{ref_prefix}"
    output_sam_file_genome = (
        f"{out_dir_notebook}/bowtie_mcrs_kmers_to_genome/alignment.sam"
    )

    if not os.path.exists(ref_folder_genome_bowtie) or not os.listdir(
        ref_folder_genome_bowtie
    ):
        run_bowtie_build_dlist(
            ref_fa=ref_dlist_fa_genome,
            ref_folder=ref_folder_genome_bowtie,
            ref_prefix=ref_prefix_genome_full,
            bowtie2_build=bowtie2_build,
            threads=threads,
        )

    if not os.path.exists(output_sam_file_genome):
        run_bowtie_alignment_dlist(
            output_sam_file=output_sam_file_genome,
            read_fa=mutations,
            ref_folder=ref_folder_genome_bowtie,
            ref_prefix=ref_prefix_genome_full,
            k=k,
            bowtie2=bowtie2,
            threads=threads,
            strandedness=strandedness,
            N_penalty=N_penalty,
            max_Ns_per_read_length=max_Ns_per_read_length,
            output_stat_file=bowtie_stat_file,
        )

    dlist_genome_df = create_df_of_dlist_headers(
        output_sam_file_genome, header_column_name=mcrs_id_column, k=k
    )

    dlist_fasta_file_genome_full = f"{out_dir_notebook}/dlist_genome.fa"
    if not os.path.exists(dlist_fasta_file_genome_full):
        parse_sam_and_extract_sequences(
            output_sam_file_genome,
            ref_dlist_fa_genome,
            dlist_fasta_file_genome_full,
            k=k,
            capitalize=True,
            remove_duplicates=False,
        )

    dlist_substring_genome_df = get_mcrs_headers_that_are_substring_dlist(
        mutation_reference_file_fasta=mutations,
        dlist_fasta_file=dlist_fasta_file_genome_full,
        strandedness=strandedness,
        header_column_name=mcrs_id_column,
    )

    dlist_genome_df[mcrs_id_column] = dlist_genome_df[mcrs_id_column].astype(str)
    dlist_substring_genome_df[mcrs_id_column] = dlist_substring_genome_df[
        mcrs_id_column
    ].astype(str)
    dlist_genome_df = pd.merge(
        dlist_genome_df, dlist_substring_genome_df, on=mcrs_id_column, how="left"
    )
    dlist_genome_df["dlist_substring"] = dlist_genome_df["dlist_substring"].fillna(
        False
    )

    if remove_Ns:
        remove_Ns_fasta(dlist_fasta_file_genome_full)

    ref_folder_cdna_bowtie = f"{reference_out_dir_sequences_dlist}/bowtie_index_cdna"
    ref_prefix_cdna_full = f"{ref_folder_cdna_bowtie}/{ref_prefix}"
    output_sam_file_cdna = f"{out_dir_notebook}/bowtie_mcrs_kmers_to_cdna/alignment.sam"

    if not os.path.exists(ref_folder_cdna_bowtie) or not os.listdir(
        ref_folder_cdna_bowtie
    ):
        run_bowtie_build_dlist(
            ref_fa=ref_dlist_fa_cdna,
            ref_folder=ref_folder_cdna_bowtie,
            ref_prefix=ref_prefix_cdna_full,
            bowtie2_build=bowtie2_build,
            threads=threads,
        )

    if not os.path.exists(output_sam_file_cdna):
        run_bowtie_alignment_dlist(
            output_sam_file=output_sam_file_cdna,
            read_fa=mutations,
            ref_folder=ref_folder_cdna_bowtie,
            ref_prefix=ref_prefix_cdna_full,
            k=k,
            bowtie2=bowtie2,
            threads=threads,
            strandedness=strandedness,
            N_penalty=N_penalty,
            max_Ns_per_read_length=max_Ns_per_read_length,
            output_stat_file=bowtie_stat_file,
        )

    dlist_cdna_df = create_df_of_dlist_headers(
        output_sam_file_cdna, header_column_name=mcrs_id_column, k=k
    )

    dlist_fasta_file_cdna_full = f"{out_dir_notebook}/dlist_cdna.fa"
    if not os.path.exists(dlist_fasta_file_cdna_full):
        parse_sam_and_extract_sequences(
            output_sam_file_cdna,
            ref_dlist_fa_cdna,
            dlist_fasta_file_cdna_full,
            k=k,
            capitalize=True,
            remove_duplicates=False,
        )

    dlist_substring_cdna_df = get_mcrs_headers_that_are_substring_dlist(
        mutation_reference_file_fasta=mutations,
        dlist_fasta_file=dlist_fasta_file_cdna_full,
        strandedness=strandedness,
        header_column_name=mcrs_id_column,
    )

    dlist_cdna_df[mcrs_id_column] = dlist_cdna_df[mcrs_id_column].astype(str)

    dlist_cdna_df[mcrs_id_column] = dlist_cdna_df[mcrs_id_column].astype(str)
    dlist_substring_cdna_df[mcrs_id_column] = dlist_substring_cdna_df[
        mcrs_id_column
    ].astype(str)
    dlist_cdna_df = pd.merge(
        dlist_cdna_df, dlist_substring_cdna_df, on=mcrs_id_column, how="left"
    )
    dlist_cdna_df["dlist_substring"] = dlist_cdna_df["dlist_substring"].fillna(False)

    if remove_Ns:
        remove_Ns_fasta(dlist_fasta_file_cdna_full)

    dlist_fasta_file = f"{out_dir_notebook}/dlist.fa"

    # concatenate d-lists into one file
    with open(dlist_fasta_file, "w") as outfile:
        # Write the contents of the first input file to the output file
        with open(dlist_fasta_file_genome_full, "r") as infile1:
            outfile.write(infile1.read())

            # Write the contents of the second input file to the output file
        with open(dlist_fasta_file_genome_full, "r") as infile2:
            outfile.write(infile2.read())

    dlist_combined_df = dlist_cdna_df.merge(
        dlist_genome_df,
        on=mcrs_id_column,
        how="outer",
        suffixes=("_cdna", "_genome"),
    )

    dlist_combined_df["dlist_cdna"] = (
        dlist_combined_df["dlist_cdna"].fillna(False).astype(bool)
    )
    dlist_combined_df["dlist_genome"] = (
        dlist_combined_df["dlist_genome"].fillna(False).astype(bool)
    )
    dlist_combined_df["dlist_substring_cdna"] = (
        dlist_combined_df["dlist_substring_cdna"].fillna(False).astype(bool)
    )
    dlist_combined_df["dlist_substring_genome"] = (
        dlist_combined_df["dlist_substring_genome"].fillna(False).astype(bool)
    )

    dlist_combined_df["dlist"] = "none"  # default to 'none'
    dlist_combined_df.loc[
        dlist_combined_df["dlist_cdna"] & dlist_combined_df["dlist_genome"], "dlist"
    ] = "cdna_and_genome"
    dlist_combined_df.loc[
        dlist_combined_df["dlist_cdna"] & ~dlist_combined_df["dlist_genome"], "dlist"
    ] = "cdna"
    dlist_combined_df.loc[
        ~dlist_combined_df["dlist_cdna"] & dlist_combined_df["dlist_genome"], "dlist"
    ] = "genome"

    dlist_combined_df.drop(columns=["dlist_cdna", "dlist_genome"], inplace=True)

    dlist_combined_df["dlist_substring"] = "none"  # default to 'none'
    dlist_combined_df.loc[
        dlist_combined_df["dlist_substring_cdna"]
        & dlist_combined_df["dlist_substring_genome"],
        "dlist_substring",
    ] = "cdna_and_genome"
    dlist_combined_df.loc[
        dlist_combined_df["dlist_substring_cdna"]
        & ~dlist_combined_df["dlist_substring_genome"],
        "dlist_substring",
    ] = "cdna"
    dlist_combined_df.loc[
        ~dlist_combined_df["dlist_substring_cdna"]
        & dlist_combined_df["dlist_substring_genome"],
        "dlist_substring",
    ] = "genome"

    dlist_combined_df.drop(
        columns=["dlist_substring_cdna", "dlist_substring_genome"], inplace=True
    )

    mutation_metadata_df = mutation_metadata_df.merge(
        dlist_combined_df[
            [
                mcrs_id_column,
                "dlist",
                "dlist_substring",
                "number_of_alignments_to_normal_human_reference_cdna",
                "number_of_alignments_to_normal_human_reference_genome",
                "number_of_substring_matches_to_normal_human_reference_cdna",
                "number_of_substring_matches_to_normal_human_reference_genome",
            ]
        ],
        on=mcrs_id_column,
        how="left",
    )
    mutation_metadata_df["number_of_alignments_to_normal_human_reference_cdna"] = (
        mutation_metadata_df["number_of_alignments_to_normal_human_reference_cdna"]
        .fillna(0)
        .astype(int)
    )
    mutation_metadata_df["number_of_alignments_to_normal_human_reference_genome"] = (
        mutation_metadata_df["number_of_alignments_to_normal_human_reference_genome"]
        .fillna(0)
        .astype(int)
    )
    mutation_metadata_df[
        "number_of_substring_matches_to_normal_human_reference_cdna"
    ] = (
        mutation_metadata_df[
            "number_of_substring_matches_to_normal_human_reference_cdna"
        ]
        .fillna(0)
        .astype(int)
    )
    mutation_metadata_df[
        "number_of_substring_matches_to_normal_human_reference_genome"
    ] = (
        mutation_metadata_df[
            "number_of_substring_matches_to_normal_human_reference_genome"
        ]
        .fillna(0)
        .astype(int)
    )

    mutation_metadata_df["number_of_alignments_to_normal_human_reference"] = (
        mutation_metadata_df["number_of_alignments_to_normal_human_reference_cdna"]
        + mutation_metadata_df["number_of_alignments_to_normal_human_reference_genome"]
    )
    mutation_metadata_df["number_of_substring_matches_to_normal_human_reference"] = (
        mutation_metadata_df[
            "number_of_substring_matches_to_normal_human_reference_cdna"
        ]
        + mutation_metadata_df[
            "number_of_substring_matches_to_normal_human_reference_genome"
        ]
    )
    mutation_metadata_df["dlist"] = mutation_metadata_df["dlist"].fillna("none")
    mutation_metadata_df["dlist_substring"] = mutation_metadata_df[
        "dlist_substring"
    ].fillna("none")

    # TODO: for those that dlist in the genome, add an additional check to see if they filter in coding regions (I already check for spliced with cDNA, but I don't distinguish unspliced coding vs noncoding)

    count_cdna_unique = (mutation_metadata_df["dlist"] == "cdna").sum()
    count_genome_unique = (mutation_metadata_df["dlist"] == "genome").sum()
    count_cdna_and_genome_intersection = (
        mutation_metadata_df["dlist"] == "cdna_and_genome"
    ).sum()
    count_cdna_total = (
        (mutation_metadata_df["dlist"] == "cdna")
        | (mutation_metadata_df["dlist"] == "cdna_and_genome")
    ).sum()
    count_genome_total = (
        (mutation_metadata_df["dlist"] == "genome")
        | (mutation_metadata_df["dlist"] == "cdna_and_genome")
    ).sum()
    count_cdna_or_genome_union = (
        (mutation_metadata_df["dlist"] == "cdna")
        | (mutation_metadata_df["dlist"] == "genome")
        | (mutation_metadata_df["dlist"] == "cdna_and_genome")
    ).sum()

    log_messages = [
        f"Unique to cDNA: {count_cdna_unique}",
        f"Unique to genome: {count_genome_unique}",
        f"Shared between cDNA and genome: {count_cdna_and_genome_intersection}",
        f"Total in cDNA: {count_cdna_total}",
        f"Total in genome: {count_genome_total}",
        f"Total in cDNA or genome: {count_cdna_or_genome_union}",
    ]

    # Log the messages
    for message in log_messages:
        if logger:
            logger.info(message)
        else:
            print(message)

    if os.path.exists(bowtie_stat_file):
        write_mode = "a"
    else:
        write_mode = "w"

        # Re-print and save the messages to a text file
    with open(bowtie_stat_file, write_mode) as f:  # Use 'a' to append to the file
        f.write("Bowtie alignment statistics summary\n")
        for message in log_messages:
            f.write(message + "\n")  # Write to file

    sequence_names_set_genome = get_set_of_headers_from_sam(output_sam_file_genome, k=k)
    sequence_names_set_cdna = get_set_of_headers_from_sam(output_sam_file_cdna, k=k)
    sequence_names_set_union_genome_and_cdna = (
        sequence_names_set_genome | sequence_names_set_cdna
    )
    return (
        mutation_metadata_df,
        ref_dlist_fa_genome,
        ref_dlist_gtf,
        sequence_names_set_union_genome_and_cdna,
    )


def download_t2t_reference_files(reference_out_dir_sequences_dlist):
    ref_dlist_fa_genome = (
        f"{reference_out_dir_sequences_dlist}/GCF_009914755.1_T2T-CHM13v2.0_genomic.fna"
    )
    ref_dlist_fa_cdna = f"{reference_out_dir_sequences_dlist}/rna.fna"
    ref_dlist_gtf = f"{reference_out_dir_sequences_dlist}/genomic.gtf"

    # Step 1: Download the ZIP file using wget
    download_url = "https://api.ncbi.nlm.nih.gov/datasets/v2alpha/genome/accession/GCF_009914755.1/download?include_annotation_type=GENOME_FASTA&include_annotation_type=RNA_FASTA&include_annotation_type=GENOME_GTF&hydrated=FULLY_HYDRATED"
    zip_file = f"{reference_out_dir_sequences_dlist}/t2t.zip"
    subprocess.run(["wget", "-O", zip_file, download_url], check=True)

    # Step 2: Unzip the downloaded file
    temp_dir = f"{reference_out_dir_sequences_dlist}/temp"
    subprocess.run(["unzip", zip_file, "-d", temp_dir], check=True)

    # Step 3: Move the files from the extracted directory to the target folder
    extracted_path = f"{temp_dir}/ncbi_dataset/data/GCF_009914755.1/"
    subprocess.run(
        f"mv {extracted_path}* {reference_out_dir_sequences_dlist}",
        shell=True,
        check=True,
    )

    # Step 4: Remove the temporary folder
    subprocess.run(["rm", "-rf", temp_dir], check=True)
    return ref_dlist_fa_genome, ref_dlist_fa_cdna, ref_dlist_gtf


# first tries pysam, then samtools, then finally custom
def create_fai(fasta_path, fai_path=None):
    try:
        import pysam

        pysam.faidx(fasta_path)
    except Exception as e:
        try:
            import subprocess

            samtools_faidx_command = f"samtools faidx {fasta_path}"
            subprocess.run(samtools_faidx_command, shell=True, check=True)
        except Exception as e:
            if fai_path is None:
                fai_path = fasta_path + ".fai"
            with open(fasta_path, "r") as fasta_file, open(fai_path, "w") as fai_file:
                offset = 0  # Track byte offset of each sequence
                seq_name = None
                seq_start = 0
                seq_length = 0
                line_length = 0
                line_length_with_newline = 0

                for line in fasta_file:
                    if line.startswith(">"):
                        # Write the previous sequence's info if it exists
                        if seq_name is not None:
                            fai_file.write(
                                f"{seq_name.split()[0]}\t{seq_length}\t{seq_start}\t{line_length}\t{line_length_with_newline}\n"
                            )

                        # Initialize new sequence information
                        seq_name = line[1:].strip()
                        seq_start = offset + len(line)  # Starting byte offset of the sequence
                        seq_length = 0  # Reset sequence length

                    else:
                        # Record line length info for the first line of the sequence
                        if seq_length == 0:
                            line_length = len(line.strip())
                            line_length_with_newline = len(line)

                        seq_length += len(line.strip())  # Update total sequence length

                    # Update byte offset for each line
                    offset += len(line)

                # Write the last sequence's info after the loop
                if seq_name is not None:
                    fai_file.write(
                        f"{seq_name.split()[0]}\t{seq_length}\t{seq_start}\t{line_length}\t{line_length_with_newline}\n"
                    )


def compute_intersection(set1, set2):
    return len(set1.intersection(set2))


def compute_jaccard_index(set1, set2):
    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))
    return intersection / union


# * TODO: this considers only the largest number of nearest neighbors, but not the ordinality, which is especially important because if I look for 15 nearest neighbors but a cancer type of interest only has 3 samples in my dataset then it will have a hard time winning out
def compute_knn(adata_unknown, adata_combined_ccle_rnaseq, out_dir=".", k=10):
    # Combine PCA embeddings from both datasets
    combined_pca_embeddings = np.vstack(
        [adata_combined_ccle_rnaseq.obsm["X_pca"], adata_unknown.obsm["X_pca"]]
    )

    # Fit the k-NN model on the reference dataset
    reference_embeddings = combined_pca_embeddings.obsm["X_pca"]
    knn_model = NearestNeighbors(n_neighbors=k, metric="euclidean")
    knn_model.fit(reference_embeddings)

    # Query the nearest neighbors for each point in `adata_unknown`
    unknown_embeddings = adata_unknown.obsm["X_pca"]
    distances, indices = knn_model.kneighbors(unknown_embeddings)

    tissue_counts = plot_knn_tissue_frequencies(
        indices,
        adata_combined_ccle_rnaseq,
        output_plot_file=f"{out_dir}/knn_tissue_frequencies.png",
    )

    plurality_tissue_counts = np.argmax(tissue_counts, axis=1)
    print("True cancer type: ", adata_unknown.uns["sample_name"])
    print("Predicted cancer type: ", plurality_tissue_counts)
    print(
        "Predicted cancer type and true cancer type match: ",
        adata_unknown.uns["sample_name"] == plurality_tissue_counts,
    )


def compute_cluster_centroid_distances(
    adata_unknown, adata_combined_ccle_rnaseq, out_dir="."
):
    # Ensure PCA has been performed on `adata_combined_ccle_rnaseq` and `adata_unknown`
    # Get the PCA coordinates for both datasets
    reference_pca = adata_combined_ccle_rnaseq.obsm["X_pca"]
    unknown_pca = adata_unknown.obsm["X_pca"]

    # Calculate the centroids of each Leiden cluster
    cluster_centroids = {}
    for cluster in adata_combined_ccle_rnaseq.obs["tissue"].unique():
        # Select cells in the current cluster and compute the centroid
        cluster_cells = reference_pca[
            adata_combined_ccle_rnaseq.obs["tissue"] == cluster
        ]
        centroid = cluster_cells.mean(axis=0)
        cluster_centroids[cluster] = centroid

    # Convert centroids to an array for easy distance calculation
    centroid_matrix = np.array(list(cluster_centroids.values()))

    # Calculate distances from the unknown point to each centroid
    distances = euclidean_distances(unknown_pca, centroid_matrix)

    sorted_distances = sorted(
        zip(cluster_centroids.keys(), distances[0]), key=lambda x: x[1]
    )

    cluster_centroid_distance_file = f"{out_dir}/cluster_centroid_distances.txt"
    # Write the sorted distances to a file
    with open(cluster_centroid_distance_file, "w") as f:
        for cluster, dist in sorted_distances:
            f.write(
                f"Distance from unknown sample to centroid of cluster {cluster}: {dist}\n"
            )

    cluster_centroid_distance_plot_file = f"{out_dir}/cluster_centroid_distances.png"
    plot_ascending_bar_plot_of_cluster_distances(
        sorted_distances, output_plot_file=cluster_centroid_distance_plot_file
    )

    # Output the closest cluster
    closest_cluster, closest_distance = sorted_distances[0]
    print("True cancer type: ", adata_unknown.uns["sample_name"])
    print("Predicted cancer type and distance: ", closest_cluster, closest_distance)
    print(
        "Predicted cancer type and true cancer type match: ",
        adata_unknown.uns["sample_name"] == closest_cluster,
    )


def compute_jaccard_indices(
    adata_unknown,
    grouped_tissue_dir,
    jaccard_type="mutation",
    out_dir=".",
    jaccard_or_intersection="jaccard",
):
    if jaccard_type == "mutation":
        column_name = "header_with_gene_name"
        text_file = "sorted_mutations.txt"
    elif jaccard_type == "mutated_genes":
        column_name = "gene_name"
        text_file = "sorted_mutated_genes.txt"
    else:
        raise ValueError("Invalid jaccard_type. Must be 'mutation' or 'mutated_genes'")
    # Initialize a dictionary to store mutations for each tissue
    tissue_mutations = {}

    adata_unknown_unique_mutations = set(adata_unknown.var[column_name])

    # Loop through each subfolder in `grouped_tissue_dir`
    for tissue in os.listdir(grouped_tissue_dir):
        specific_tissue_dir = os.path.join(grouped_tissue_dir, tissue)
        sorted_mutations_file = os.path.join(specific_tissue_dir, text_file)

        # Initialize an empty set for mutations in this tissue
        mutations = set()

        # Check if the file exists in the current tissue folder
        if os.path.isfile(sorted_mutations_file):
            with open(sorted_mutations_file, "r") as f:
                for line in f:
                    mutation_name = line.split()[
                        0
                    ]  # Grab the first column (mutation name)
                    mutations.add(mutation_name)

        if jaccard_or_intersection == "jaccard":
            jaccard_index_mutations = compute_jaccard_index(
                mutations, adata_unknown_unique_mutations
            )
        elif jaccard_or_intersection == "intersection":
            jaccard_index_mutations = compute_intersection(
                mutations, adata_unknown_unique_mutations
            )

        tissue_mutations[tissue] = jaccard_index_mutations

    tissues = list(adata_unknown_unique_mutations.keys())
    jaccard_values = list(adata_unknown_unique_mutations.values())

    sorted_data = sorted(zip(jaccard_values, tissues), reverse=True)
    tissues, jaccard_values = zip(*sorted_data)

    plot_jaccard_bar_plot(
        tissues,
        jaccard_values,
        output_plot_file=f"{out_dir}/jaccard_bar_plot_{jaccard_type}.png",
    )

    # Find the tissue with the maximum Jaccard index
    max_tissue = max(
        adata_unknown_unique_mutations, key=adata_unknown_unique_mutations.get
    )
    max_jaccard = adata_unknown_unique_mutations[max_tissue]

    print("True cancer type: ", adata_unknown.uns["sample_name"])
    print("Predicted cancer type and jaccard for mutations: ", max_tissue, max_jaccard)
    print(
        "Predicted cancer type and true cancer type match: ",
        adata_unknown.uns["sample_name"] == max_tissue,
    )


# TODO: write this
def compute_mx_metric():
    pass
    # return a list of marker mutations for each cancer type, and provide these as input along with the unknown sample's mutation matrix to *mx assign* (might only work for single-cell)
    # - identify cancer-specific marker mutations (or genes, and just apply to all of it's mutations in my database) from the literature ("old")
    # - return a list of marker mutations for each cancer type from my cell lines ("new")
    # - merge these with *ec merge*
    # - run *ec index* on the merged marker list
    # - run *mx extract* on the output of *ec index*
    # - run *mx clean* on the output of *mx extract*
    # - run *mx assign* on the output of *mx clean*


def predict_cancer_type(
    adata_path,
    adata_combined_ccle_rnaseq,
    pca_components,
    grouped_tissue_dir,
    sample_name,
    sample_to_cancer_type,
    metric="knn",
    k=10,
    use_binary_matrix=False,
):
    sample_dir = os.path.dirname(os.path.dirname(adata_path))
    adata_unknown = sc.read_h5ad(adata_path)
    adata_unknown.uns["sample_name"] = sample_name
    adata_unknown.uns["cancer_type"] = sample_to_cancer_type[sample_name]

    if use_binary_matrix:
        adata_unknown.X = (adata_unknown.X > 0).astype(int)

    # Identify common genes between the two datasets
    common_genes = adata_unknown.var_names.intersection(
        adata_combined_ccle_rnaseq.var_names
    )

    # Subset adata_unknown to include only these common genes
    adata_unknown = adata_unknown[:, common_genes].copy()

    # Step 2: Center the new data using the mean of the original data
    # Note: adata_original.X.mean(axis=0) assumes both datasets have the same set of genes/variables
    mean_center = np.mean(adata_combined_ccle_rnaseq.X, axis=0)
    adata_unknown_centered = adata_unknown.X - mean_center

    # Step 3: Project the new data into the PCA space of the original data
    # This will give you the PCA coordinates for adata_unknown in the same space as adata_original
    adata_unknown.obsm["X_pca"] = adata_unknown_centered.dot(pca_components)

    # Optional: Store the explained variance for reference if needed
    adata_unknown.uns["pca"] = adata_combined_ccle_rnaseq.uns[
        "pca"
    ]  # Copy explained variance, etc.

    if metric == "knn":
        compute_knn(
            adata_unknown=adata_unknown,
            adata_combined_ccle_rnaseq=adata_combined_ccle_rnaseq,
            out_dir=sample_dir,
            k=k,
        )
    elif metric == "euclidean":
        compute_cluster_centroid_distances(
            adata_unknown=adata_unknown,
            adata_combined_ccle_rnaseq=adata_combined_ccle_rnaseq,
            out_dir=sample_dir,
        )
    elif metric == "jaccard":
        compute_jaccard_indices(
            adata_unknown=adata_unknown,
            grouped_tissue_dir=grouped_tissue_dir,
            out_dir=sample_dir,
        )
    elif metric == "intersection":
        compute_jaccard_indices(
            adata_unknown=adata_unknown,
            grouped_tissue_dir=grouped_tissue_dir,
            out_dir=sample_dir,
        )
    elif metric == "mx":
        compute_mx_metric()


# to be clear, this removes double counting of the same MCRS on each paired end, which is true valid when fragment length < 2*read length
def decrement_adata_matrix_when_split_by_Ns_or_running_paired_end_in_single_end_mode(
    adata,
    fastq,
    kb_count_out,
    t2g,
    mm,
    bustools="bustools",
    split_Ns=False,
    paired_end_fastqs=False,
    paired_end_suffix_length=2,
    assay="bulk",
    keep_only_insertions=True,
):
    assert (
        split_Ns or paired_end_fastqs
    ), "At least one of split_Ns or paired_end_fastqs must be True"
    if assay != "bulk":
        raise ValueError("This function currently only works with bulk RNA-seq data")

    if not os.path.exists(f"{kb_count_out}/bus_df.csv"):
        bus_df = make_bus_df(
            kb_count_out,
            fastq,
            t2g=t2g,
            mm=mm,
            union=False,
            assay=assay,
            bustools=bustools,
        )
    else:
        bus_df = pd.read_csv(f"{kb_count_out}/bus_df.csv")

    if not "mcrs_mutation_type" in adata.var.columns:
        adata.var = add_mcrs_mutation_type(adata.var, mut_column="mcrs_header")

    if keep_only_insertions:  # valid when fragment length >= 2*read length
        # Can only count for insertions (lengthens the MCRS)
        mutation_types_with_a_chance_of_being_double_counted_after_N_split = {
            "insertion",
            "delins",
            "mixed",
        }

        # Filter and retrieve the set of 'mcrs_header' values
        potentially_double_counted_reference_items = set(
            adata.var["mcrs_id"][
                adata.var["mcrs_mutation_type"].isin(
                    mutation_types_with_a_chance_of_being_double_counted_after_N_split
                )
            ]
        )

        # filter bus_df to only keep rows where bus_df['gene_names_final'] contains a gene that is in potentially_double_counted_reference_items
        pattern = "|".join(potentially_double_counted_reference_items)
        bus_df = bus_df[bus_df["gene_names_final"].str.contains(pattern, regex=True)]

    n_rows, n_cols = adata.X.shape
    decrement_matrix = csr_matrix((n_rows, n_cols))
    bus_df["gene_names_final"] = bus_df["gene_names_final"].apply(safe_literal_eval)

    tested_suffixes = set()

    var_names_to_idx_in_adata_dict = {
        name: idx for idx, name in enumerate(adata.var_names)
    }

    for _, row in bus_df.iterrows():
        if row["counted_in_count_matrix"]:
            suffix = row["fastq_header"]
            if paired_end_fastqs:
                suffix = suffix[:-paired_end_suffix_length]  # assumes the form nsplit{i}__READHEADERpairedendportion
            if split_Ns and (
                suffix.startswith("nsplit") or suffix.startswith("@nsplit")
            ):
                suffix = suffix.split("__")[1]  # assumes the form nsplit{i}__READHEADER
            if (
                suffix not in tested_suffixes
            ):  # here to make sure I don't double-count the decrementing
                filtered_bus_df = bus_df[
                    bus_df["gene_names_final"].str.contains(suffix)
                ]
                # Calculate the count of matching rows with the same 'EC' and 'barcode'
                count = (
                    sum(
                        1
                        for _, item in filtered_bus_df.iterrows()
                        if item["EC"] == row["EC"] and item["barcode"] == row["barcode"]
                    )
                    - 1
                )  # Subtract 1 to avoid counting the current row itself

                if count > 0:
                    barcode_idx = np.where(adata.obs_names == row["barcode"])[0][
                        0
                    ]  # if I previously removed the padding
                    mcrs_idxs = [
                        var_names_to_idx_in_adata_dict[header]
                        for header in row["gene_names_final"]
                        if header in var_names_to_idx_in_adata_dict
                    ]
                    decrement_matrix[barcode_idx, mcrs_idxs] += count
                tested_suffixes.add(suffix)

    if not isinstance(adata.X, csr_matrix):
        adata.X = adata.X.tocsr()

    if not isinstance(decrement_matrix, csr_matrix):
        decrement_matrix = decrement_matrix.tocsr()

    # Add the two sparse matrices
    adata.X = adata.X - decrement_matrix

    adata.X = csr_matrix(adata.X)

    return adata


def remove_adata_columns(adata, values_of_interest, operation, var_column_name):
    if type(values_of_interest) == str and values_of_interest.endswith(".txt"):
        with open(values_of_interest, "r") as f:
            values_of_interest_set = {line.strip() for line in f}
    elif type(values_of_interest) == list:
        values_of_interest_set = set(values_of_interest)

    # Step 2: Filter adata.var based on whether 'mcrs_id' is in the set
    columns_to_remove = adata.var.index[
        adata.var[var_column_name].isin(values_of_interest_set)
    ]

    # Step 3: Remove the corresponding columns in adata.X and rows in adata.var
    if operation == "keep":
        adata = adata[:, adata.var_names.isin(columns_to_remove)]
    elif operation == "exclude":
        adata = adata[:, ~adata.var_names.isin(columns_to_remove)]

    return adata


def trim_edges_off_reads_fastq_list(
    rnaseq_fastq_files,
    parity,
    minimum_base_quality_trim_reads=0,
    qualified_quality_phred=0,
    unqualified_percent_limit=100,
    n_base_limit=None,
    length_required=None,
    fastp="fastp",
):
    rnaseq_fastq_files_quality_controlled = []
    if parity == "single":
        for i in range(len(rnaseq_fastq_files)):
            rnaseq_fastq_file, _ = trim_edges_and_adaptors_off_fastq_reads(
                filename=rnaseq_fastq_files[i],
                filename_r2=None,
                filename_filtered=None,
                filename_filtered_r2=None,
                cut_mean_quality=minimum_base_quality_trim_reads,
                qualified_quality_phred=qualified_quality_phred,
                unqualified_percent_limit=unqualified_percent_limit,
                n_base_limit=n_base_limit,
                length_required=length_required,
                fastp=fastp,
            )
            rnaseq_fastq_files_quality_controlled.append(rnaseq_fastq_file)
    elif parity == "paired":
        for i in range(0, len(rnaseq_fastq_files), 2):
            rnaseq_fastq_file, rnaseq_fastq_file_2 = (
                trim_edges_and_adaptors_off_fastq_reads(
                    filename=rnaseq_fastq_files[i],
                    filename_r2=rnaseq_fastq_files[i + 1],
                    filename_filtered=None,
                    filename_filtered_r2=None,
                    cut_mean_quality=minimum_base_quality_trim_reads,
                    qualified_quality_phred=qualified_quality_phred,
                    unqualified_percent_limit=unqualified_percent_limit,
                    n_base_limit=n_base_limit,
                    length_required=length_required,
                    fastp=fastp,
                )
            )
            rnaseq_fastq_files_quality_controlled.extend(
                [rnaseq_fastq_file, rnaseq_fastq_file_2]
            )

    return rnaseq_fastq_files_quality_controlled


def run_fastqc_and_multiqc(rnaseq_fastq_files_quality_controlled, fastqc_out_dir):
    rnaseq_fastq_files_quality_controlled_string = " ".join(
        rnaseq_fastq_files_quality_controlled
    )
    try:
        fastqc_command = (
            f"fastqc -o {fastqc_out_dir} {rnaseq_fastq_files_quality_controlled_string}"
        )
        subprocess.run(fastqc_command, shell=True, check=True)
    except subprocess.CalledProcessError as e:
        print("Error running fastqc")
        print(e)

    try:
        multiqc_command = f"multiqc --filename multiqc --outdir {fastqc_out_dir} {fastqc_out_dir}/*fastqc*"
        subprocess.run(multiqc_command, shell=True, check=True)
    except subprocess.CalledProcessError as e:
        print("Error running multiqc")
        print(e)


def replace_low_quality_bases_with_N_list(
    rnaseq_fastq_files, minimum_base_quality, seqtk
):
    rnaseq_fastq_files_replace_low_quality_bases_with_N = []
    for i in range(len(rnaseq_fastq_files)):
        rnaseq_fastq_file = rnaseq_fastq_files[i]
        rnaseq_fastq_file = replace_low_quality_base_with_N(
            rnaseq_fastq_file, seqtk=seqtk, minimum_base_quality=minimum_base_quality
        )
        rnaseq_fastq_files_replace_low_quality_bases_with_N.append(rnaseq_fastq_file)
        # # delete the file in rnaseq_fastq_files[i]
        # os.remove(rnaseq_fastq_files[i])
    return rnaseq_fastq_files_replace_low_quality_bases_with_N


# TODO: enable single vs paired end mode (single end works as-is; paired end requires 2 files as input, and for every line it splits in file 1, I will add a line of all Ns in file 2)
def split_reads_by_N_list(
    rnaseq_fastq_files_replace_low_quality_bases_with_N,
    minimum_sequence_length,
    delete_original_files=True,
):
    rnaseq_fastq_files_split_reads_by_N = []
    for i in range(len(rnaseq_fastq_files_replace_low_quality_bases_with_N)):
        rnaseq_fastq_file = rnaseq_fastq_files_replace_low_quality_bases_with_N[i]
        rnaseq_fastq_file = split_fastq_reads_by_N(
            rnaseq_fastq_file, minimum_sequence_length=minimum_sequence_length
        )  # TODO: would need a way of postprocessing to make sure I don't double-count fragmented reads - I would need to see where each fragmented read aligns - perhaps with kb extract or pseudobam
        # replace_low_quality_base_with_N_and_split_fastq_reads_by_N(input_fastq_file = rnaseq_fastq_file, output_fastq_file = None, minimum_sequence_length=k, seqtk = seqtk, minimum_base_quality = minimum_base_quality_replace_with_N)
        rnaseq_fastq_files_split_reads_by_N.append(rnaseq_fastq_file)
        # # delete the file in rnaseq_fastq_files_replace_low_quality_bases_with_N[i]
        if delete_original_files:
            os.remove(rnaseq_fastq_files_replace_low_quality_bases_with_N[i])
    return rnaseq_fastq_files_split_reads_by_N


def intersect_lists(series):
    return list(set.intersection(*map(set, series)))


def map_transcripts_to_genes(transcript_list, mapping_dict):
    return [mapping_dict.get(transcript, "Unknown") for transcript in transcript_list]


#* only works when kb count was run with --num (as this means that each row of the BUS file corresponds to exactly one read)
def make_bus_df(
    kallisto_out,
    fastq_file_list,  # make sure this is in the same order as passed into kb count - [sample1, sample2, etc] OR [sample1_pair1, sample1_pair2, sample2_pair1, sample2_pair2, etc]
    t2g_file,
    mm=False,
    union=False,
    assay="bulk",  # technology flag of kb
    parity="single",
    bustools="bustools",
):
    print("loading in transcripts")
    with open(f"{kallisto_out}/transcripts.txt") as f:
        transcripts = (
            f.read().splitlines()
        )  # get transcript at index 0 with transcript[0], and index of transcript named "name" with transcript.index("name")

    transcripts.append("dlist")  # add dlist to the end of the list

    if assay == "bulk" or "smartseq" in assay.lower():  # smartseq does not have barcodes
        print("loading in barcodes")
        with open(f"{kallisto_out}/matrix.sample.barcodes") as f:
            barcodes = (
                f.read().splitlines()
            )  # get transcript at index 0 with transcript[0], and index of transcript named "name" with transcript.index("name")
    else:
        try:
            barcode_start = technology_barcode_and_umi_dict[assay]["barcode_start"]
            barcode_end = technology_barcode_and_umi_dict[assay]["barcode_end"]
            umi_start = technology_barcode_and_umi_dict[assay]["umi_start"]
            umi_end = technology_barcode_and_umi_dict[assay]["umi_end"]
        except KeyError:
            print(f"Assay {assay} currently not supported. Supported are {list(technology_barcode_and_umi_dict.keys())}")

        pass  # TODO: write this (will involve technology parameter to get barcode from read)

    fastq_header_df = pd.DataFrame(columns=['read_index', 'fastq_header', 'barcode'])

    if parity == "paired":
        fastq_header_df['fastq_header_pair'] = None

    if type(fastq_file_list) == str:
        fastq_file_list = [fastq_file_list]
    
    skip_upcoming_fastq = False
    
    for i, fastq_file in enumerate(fastq_file_list):
        if skip_upcoming_fastq:
            skip_upcoming_fastq = False
            continue
        # important for temp files
        fastq_file = str(fastq_file)

        print("loading in fastq headers")
        if (
            fastq_file.endswith(".fastq")
            or fastq_file.endswith(".fq")
            or fastq_file.endswith(".fastq.gz")
            or fastq_file.endswith(".fq.gz")
        ):
            fastq_header_list = get_header_set_from_fastq(fastq_file, output_format="list")
        elif fastq_file.endswith(".txt"):
            with open(fastq_file) as f:
                fastq_header_list = f.read().splitlines()

        if assay == "bulk" or "smartseq" in assay.lower():
            barcode_list = barcodes[i]
        else:
            fq_dict = pyfastx.Fastq(fastq_file, build_index=True)
            barcode_list = [fq_dict[i].seq[barcode_start:barcode_end] for i in range(len(fq_dict))]

        new_rows = pd.DataFrame({
            'read_index': range(len(fastq_header_list)),  # Position/index values
            'fastq_header': fastq_header_list,                 # List values
            'barcode': barcode_list
        })

        if parity == "paired":
            fastq_file_pair = str(fastq_file_list[i + 1])
            if (
                fastq_file_pair.endswith(".fastq")
                or fastq_file_pair.endswith(".fq")
                or fastq_file_pair.endswith(".fastq.gz")
                or fastq_file_pair.endswith(".fq.gz")
            ):
                new_rows['fastq_header_pair'] = get_header_set_from_fastq(fastq_file_pair, output_format="list")
            elif fastq_file_pair.endswith(".txt"):
                with open(fastq_file_pair) as f:
                    new_rows['fastq_header_pair'] = f.read().splitlines()

            skip_upcoming_fastq = True  # because it will be the pair

        fastq_header_df = pd.concat([fastq_header_df, new_rows], ignore_index=True)

    # Get equivalence class that matches to 0-indexed line number of target ID
    print("loading in ec matrix")
    ec_df = pd.read_csv(
        f"{kallisto_out}/matrix.ec",
        sep="\t",
        header=None,
        names=["EC", "transcript_ids"],
    )
    ec_df["transcript_ids"] = ec_df["transcript_ids"].astype(str)
    ec_df["transcript_ids_list"] = ec_df["transcript_ids"].str.split(",")
    ec_df["transcript_ids_list"] = ec_df["transcript_ids_list"].apply(
        lambda x: list(map(int, x))
    )
    ec_df["transcript_ids_list"] = ec_df["transcript_ids"].apply(
        lambda x: list(map(int, x.split(",")))
    )
    ec_df["transcript_names"] = ec_df["transcript_ids_list"].apply(
        lambda ids: [transcripts[i] for i in ids]
    )

    # Get bus output (converted to txt)
    bus_file = f"{kallisto_out}/output.bus"
    bus_text_file = f"{kallisto_out}/output_sorted_bus.txt"
    if not os.path.exists(bus_text_file):
        print("running bustools text")
        bus_txt_file_existed_originally = False
        create_bus_txt_file_command = (
            f"{bustools} text -o {bus_text_file} -f {bus_file}"
        )
        subprocess.run(create_bus_txt_file_command, shell=True, check=True)
        # /home/jrich/miniconda3/envs/cartf/lib/python3.10/site-packages/kb_python/bins/linux/bustools/bustools text -p -a -f -d /home/jrich/Desktop/CART_prostate_sc/TEMP_dlist_tests/kb_count_out_delaney/output.bus
    else:
        bus_txt_file_existed_originally = True

    print("loading in bus df")
    bus_df = pd.read_csv(
        bus_text_file,
        sep="\t",
        header=None,
        names=["barcode", "UMI", "EC", "count", "read_index"],
    )

    if not bus_txt_file_existed_originally:
        os.remove(bus_text_file)

    bus_df = bus_df.merge(fastq_header_df, on=['read_index', 'barcode'], how="left")

    print("merging ec df into bus df")
    bus_df = bus_df.merge(ec_df, on="EC", how="left")

    if assay != "bulk":
        bus_df_collapsed_1 = bus_df.groupby(
            ["barcode", "UMI", "EC"], as_index=False
        ).agg(
            {
                "count": "sum",  # Sum counts
                "read_index": lambda x: list(x),  # Combine ints in a list
                "fastq_header": lambda x: list(x),  # Combine strings in a list
                "transcript_ids": "first",  # Take the first value for all other columns
                "transcript_ids_list": "first",  # Take the first value for all other columns
                "transcript_names": "first",  # Take the first value for all other columns
            }
        )

        bus_df_collapsed_2 = bus_df_collapsed_1.groupby(
            ["barcode", "UMI"], as_index=False
        ).agg(
            {
                "EC": lambda x: list(x),
                "count": "sum",  # Sum the 'count' column
                "read_index": lambda x: sum(x, []),  # Concatenate lists in 'read_index'
                "fastq_header": lambda x: sum(
                    x, []
                ),  # Concatenate lists in 'fastq_header'
                "transcript_ids": lambda x: ",".join(
                    x
                ),  # Join strings in 'transcript_ids_list' with commas  # may contain duplicates indices
                "transcript_ids_list": lambda x: sum(
                    x, []
                ),  # Concatenate lists for 'transcript_ids_list'
                "transcript_names": lambda x: sum(
                    x, []
                ),  # Concatenate lists for 'transcript_names'
            }
        )

        # Add new columns for the intersected lists
        bus_df_collapsed_2["transcript_names_final"] = (
            bus_df_collapsed_1.groupby(["barcode", "UMI"])["transcript_names"]
            .apply(intersect_lists)
            .values
        )
        bus_df_collapsed_2["transcript_ids_list_final"] = (
            bus_df_collapsed_1.groupby(["barcode", "UMI"])["transcript_ids_list"]
            .apply(intersect_lists)
            .values
        )

        bus_df = bus_df_collapsed_2

    else:  # assay == "bulk"
        # bus_df.rename(columns={"transcript_ids_list": "transcript_ids_list_final", "transcript_names": "transcript_names_final"}, inplace=True)
        bus_df["transcript_ids_list_final"] = bus_df["transcript_ids_list"]
        bus_df["transcript_names_final"] = bus_df["transcript_names"]

    print("loading in t2g df")
    t2g_df = pd.read_csv(
        t2g_file, sep="\t", header=None, names=["transcript_id", "gene_name"]
    )
    t2g_dict = dict(zip(t2g_df["transcript_id"], t2g_df["gene_name"]))

    print("Apply the mapping function to create gene name columns")
    # mapping transcript to gene names
    bus_df["gene_names"] = bus_df["transcript_names"].apply(
        lambda x: map_transcripts_to_genes(x, t2g_dict)
    )
    bus_df["gene_names_final"] = bus_df["transcript_names_final"].apply(
        lambda x: map_transcripts_to_genes(x, t2g_dict)
    )

    bus_df["gene_names_final_set"] = bus_df["gene_names_final"].apply(set)

    print("added counted in matrix column")
    if union or mm:
        # union or mm gets added to count matrix as long as dlist is not included in the EC
        bus_df["counted_in_count_matrix"] = bus_df["transcript_names_final"].apply(
            lambda x: "dlist" not in x
        )
    else:
        # only gets added to the count matrix if EC has exactly 1 gene
        bus_df["counted_in_count_matrix"] = bus_df["gene_names_final_set"].apply(
            lambda x: len(x) == 1
        )

    # adata_path = f"{kallisto_out}/counts_unfiltered/adata.h5ad"
    # adata = sc.read_h5ad(adata_path)
    # barcode_length = len(adata.obs.index[0])
    # bus_df['barcode_without_padding'] = bus_df['barcode'].str[(32 - barcode_length):]

    # so now I can iterate through this dataframe for the columns where counted_in_count_matrix is True - barcode will be the cell/sample (adata row), gene_names_final will be the list of gene name(s) (adata column), and count will be the number added to this entry of the matrix (always 1 for bulk)

    # save bus_df
    print("saving bus df")
    bus_df.to_csv(f"{kallisto_out}/bus_df.csv", index=False)
    return bus_df


# TODO: test
def match_paired_ends_after_single_end_run(
    bus_df_path, gene_name_type="mcrs_id", id_to_header_csv=None
):
    if os.path.exists(bus_df_path):
        bus_df = pd.read_csv(bus_df_path)
    else:
        raise FileNotFoundError(f"{bus_df_path} does not exist")

    paired_end_suffix_length = 2  # * only works for /1 and /2 notation
    bus_df["fastq_header_without_paired_end_suffix"] = bus_df["fastq_header"].str[
        :-paired_end_suffix_length
    ]

    # get the paired ends side-by-side
    df_1 = bus_df[
        bus_df["fastq_header"].str.endswith("/1")
    ].copy()  # * only works for /1 and /2 notation
    df_2 = bus_df[bus_df["fastq_header"].str.endswith("/2")].copy()

    # Remove the "/1" and "/2" suffix for merging on entry numbers
    df_1["entry_number"] = df_1["fastq_name"].str.extract(r"(\d+)/1").astype(int)
    df_2["entry_number"] = df_2["fastq_name"].str.extract(r"(\d+)/2").astype(int)

    # Merge based on entry numbers to create paired columns
    paired_df = pd.merge(df_1, df_2, on="entry_number", suffixes=("_1", "_2"))

    # Select and rename columns
    paired_df = paired_df.rename(
        columns={
            "fastq_name_1": "fastq_header",
            "gene_names_final_1": "gene_names_final",
            "fastq_name_2": "fastq_header_pair",
            "gene_names_final_2": "gene_names_final_pair",
        }
    )

    # Merge paired information back into the original bus_df
    bus_df = bus_df.merge(
        paired_df[
            [
                "fastq_header",
                "gene_names_final",
                "fastq_header_pair",
                "gene_names_final_pair",
            ]
        ],
        on=["fastq_header", "gene_names_final"],
        how="left",
    )

    bus_df["gene_names_final"] = bus_df["gene_names_final"].apply(safe_literal_eval)
    bus_df["gene_names_final_pair"] = bus_df["gene_names_final_pair"].apply(
        safe_literal_eval
    )

    if gene_name_type == "mcrs_id":
        id_to_header_dict = make_mapping_dict(id_to_header_csv, dict_key="id")

        bus_df["mcrs_header_list"] = bus_df["gene_names_final"].apply(
            lambda gene_list: [id_to_header_dict.get(gene, gene) for gene in gene_list]
        )

        bus_df["mcrs_header_list_pair"] = bus_df["gene_names_final_pair"].apply(
            lambda gene_list: [id_to_header_dict.get(gene, gene) for gene in gene_list]
        )

        bus_df["ensembl_transcript_list"] = [
            value.split(":")[0] for value in bus_df["mcrs_header_list"]
        ]
        bus_df["ensembl_transcript_list_pair"] = [
            value.split(":")[0] for value in bus_df["mcrs_header_list_pair"]
        ]

        # TODO: map ENST to ENSG
        bus_df["gene_list"] = ""
        bus_df["gene_list_pair"] = ""
    else:
        bus_df["gene_list"] = bus_df["gene_names_final"]
        bus_df["gene_list_pair"] = bus_df["gene_names_final_pair"]

    bus_df["paired_ends_map_to_different_genes"] = bus_df.apply(
        lambda row: (
            isinstance(row["gene_list"], list)
            and bool(row["gene_list"])
            and isinstance(row["gene_list_pair"], list)
            and bool(row["gene_list_pair"])
            and not set(row["gene_list"]).intersection(row["gene_list_pair"])
        ),
        axis=1,
    )

    return bus_df


# TODO: unsure if this works for sc
def adjust_mutation_adata_by_normal_gene_matrix(adata, kb_output_mutation, kb_output_standard, id_to_header_csv = None, mutation_metadata_csv = None, adata_output_path = None, t2g_mutation = None, t2g_standard = None, fastq_file_list = None, mm = False, union = False, assay = "bulk", parity = "single", bustools = "bustools"):
    if not adata:
        adata = f"{kb_output_mutation}/counts_unfiltered/adata.h5ad"
    if type(adata) == str:
        adata = sc.read_h5ad(adata)
    
    bus_df_mutation_path = f"{kb_output_mutation}/bus_df.csv"
    bus_df_standard_path = f"{kb_output_standard}/bus_df.csv"
    assert id_to_header_csv or mutation_metadata_csv, "Either id_to_header_csv or mutation_metadata_csv must be provided"
    
    if not os.path.exists(bus_df_mutation_path):
        bus_df_mutation = make_bus_df(
            kallisto_out = kb_output_mutation,
            fastq_file_list = fastq_file_list,  # make sure this is in the same order as passed into kb count - [sample1, sample2, etc] OR [sample1_pair1, sample1_pair2, sample2_pair1, sample2_pair2, etc]
            t2g_file = t2g_mutation,
            mm=mm,
            union=union,
            assay=assay,
            parity=parity,
            bustools=bustools,
        )
    else:
        bus_df_mutation = pd.read_csv(bus_df_mutation_path)

    bus_df_mutation["gene_names_final"] = bus_df_mutation["gene_names_final"].apply(safe_literal_eval)
    bus_df_mutation.rename(columns={"gene_names_final": "MCRS_ids_final", "count": "count_value"}, inplace=True)

    if not id_to_header_csv:
        id_to_header_dict = make_mapping_dict(id_to_header_csv, dict_key="id")
        bus_df_mutation['MCRS_headers_final'] = bus_df_mutation['MCRS_ids_final'].apply(
            lambda name_list: [id_to_header_dict.get(name, name) for name in name_list]
        )

        bus_df_mutation['transcripts_MCRS'] = bus_df_mutation['MCRS_headers_final'].apply(
            lambda string_list: tuple({s.split(':')[0] for s in string_list})
        )
    
    elif not mutation_metadata_csv:
        mutation_metadata = pd.read_csv(mutation_metadata_csv)

        id_to_transcript_dict = dict(zip(mutation_metadata['mcrs_id'], mutation_metadata['seq_ID']))
        bus_df_mutation['transcripts_MCRS'] = bus_df_mutation['MCRS_ids_final'].apply(
            lambda name_list: tuple({id_to_transcript_dict.get(name, name) for name in name_list})
        )

    if not os.path.exists(bus_df_standard_path):
        bus_df_standard = make_bus_df(
            kallisto_out = kb_output_standard,
            fastq_file_list = fastq_file_list,  # make sure this is in the same order as passed into kb count - [sample1, sample2, etc] OR [sample1_pair1, sample1_pair2, sample2_pair1, sample2_pair2, etc]
            t2g_file = t2g_standard,
            mm=mm,
            union=union,
            assay=assay,
            parity=parity,
            bustools=bustools,
        )
    else:
        bus_df_standard = pd.read_csv(bus_df_standard_path, usecols=['barcode', 'UMI', 'fastq_header', 'transcript_names_final'])

    bus_df_standard["transcript_names_final"] = bus_df_standard["transcript_names_final"].apply(safe_literal_eval)
    bus_df_standard['transcripts_standard'] = bus_df_standard['transcript_names_final'].apply(
        lambda name_list: tuple(
            [re.match(r'^(ENST\d+)', name).group(0) if re.match(r'^(ENST\d+)', name) else name for name in name_list]
        )
    )

    bus_df_mutation = bus_df_mutation.merge(bus_df_standard[['barcode', 'UMI', 'fastq_header', 'transcripts_standard']], on=['barcode', 'UMI', 'fastq_header'], how='left')

    bus_df_mutation['mcrs_matrix_received_a_count_from_a_read_that_aligned_to_a_different_gene'] = bus_df_mutation.apply(
        lambda row: (
            row['counted_in_count_matrix'] and
            any(transcript in row['transcripts_standard'] for transcript in row['transcripts_mcrs'])
        ),
        axis=1
    )

    n_rows, n_cols = adata.X.shape
    decrement_matrix = csr_matrix((n_rows, n_cols))

    var_names_to_idx_in_adata_dict = {
        name: idx for idx, name in enumerate(adata.var_names)
    }

    # iterate through the rows where the erroneous counting occurred
    for row in bus_df_mutation.loc[
        bus_df_mutation['mcrs_matrix_received_a_count_from_a_read_that_aligned_to_a_different_gene']
    ].itertuples():    
        barcode_idx = np.where(adata.obs_names == row.barcode)[0][
            0
        ]  # if I previously removed the padding
        mcrs_idxs = [
            var_names_to_idx_in_adata_dict[header]
            for header in row.MCRS_ids_final
            if header in var_names_to_idx_in_adata_dict
        ]

        decrement_matrix[barcode_idx, mcrs_idxs] += row.count_value

    if not isinstance(adata.X, csr_matrix):
        adata.X = adata.X.tocsr()

    if not isinstance(decrement_matrix, csr_matrix):
        decrement_matrix = decrement_matrix.tocsr()

    # Add the two sparse matrices
    adata.X = adata.X - decrement_matrix

    adata.X = csr_matrix(adata.X)

    # save adata
    if not adata_output_path:
        adata_output_path = f"{kb_output_mutation}/counts_unfiltered/adata_adjusted_by_gene_alignments.h5ad"
    
    adata.write(adata_output_path)

    return adata


def match_adata_orders(adata, adata_ref):
    # Ensure cells (obs) are in the same order
    adata = adata[adata_ref.obs_names]

    # Add missing genes to adata
    missing_genes = adata_ref.var_names.difference(adata.var_names)
    padding_matrix = csr_matrix((adata.n_obs, len(missing_genes)))  # Sparse zero matrix

    # Create a padded AnnData for missing genes
    adata_padded = ad.AnnData(
        X=padding_matrix,
        obs=adata.obs,
        var=pd.DataFrame(index=missing_genes)
    )

    # Concatenate the original and padded AnnData objects
    adata_padded = ad.concat([adata, adata_padded], axis=1)

    # Reorder genes to match adata_ref
    adata_padded = adata_padded[:, adata_ref.var_names]

    return adata_padded

def make_vaf_matrix(adata_mutant_mcrs_path, adata_wt_mcrs_path, adata_vaf_output = None, mutant_vcf = None):
    import scipy.sparse as sp
    adata_mutant_mcrs = sc.read_h5ad(adata_mutant_mcrs_path)
    adata_wt_mcrs = sc.read_h5ad(adata_wt_mcrs_path)

    adata_mutant_mcrs_path_out = adata_mutant_mcrs_path.replace('.h5ad', '_with_vaf.h5ad')
    adata_wt_mcrs_path_out = adata_wt_mcrs_path.replace('.h5ad', '_with_vaf.h5ad')

    adata_wt_mcrs_padded = match_adata_orders(adata = adata_wt_mcrs, adata_ref = adata_mutant_mcrs)

    # Perform element-wise division (handle sparse matrices)
    mutant_X = adata_mutant_mcrs.X
    wt_X = adata_wt_mcrs_padded.X

    if sp.issparse(mutant_X) and sp.issparse(wt_X):
        # Calculate the denominator: mutant_X + wt_X (element-wise addition for sparse matrices)
        denominator = mutant_X + wt_X
        
        # Avoid division by zero by setting zeros in the denominator to NaN
        denominator.data[denominator.data == 0] = np.nan
        
        # Calculate VAF: mutant_X / (mutant_X + wt_X)
        result_matrix = mutant_X.multiply(1 / denominator)
        
        # Handle NaNs and infinities resulting from division
        result_matrix.data[np.isnan(result_matrix.data)] = 0.0  # Set NaNs to 0
        result_matrix.data[np.isinf(result_matrix.data)] = 0.0  # Set infinities to 0
    else:
        # Calculate VAF for dense matrices
        denominator = mutant_X + wt_X
        result_matrix = np.nan_to_num(mutant_X / denominator, nan=0.0, posinf=0.0, neginf=0.0)

    # Create a new AnnData object with the result
    adata_result = ad.AnnData(
        X=result_matrix,
        obs=adata_mutant_mcrs.obs,
        var=adata_mutant_mcrs.var
    )

    if not adata_vaf_output:
        adata_vaf_output = "./adata_vaf.h5ad"

    # Save the result as an AnnData object
    adata_result.write(adata_vaf_output)

    # merge wt allele depth into mutant adata
    # Ensure indices of adata2.var and adata1.var are aligned
    merged_var = adata_mutant_mcrs.var.copy()  # Start with adata1.var

    # Add the "mcrs_count" from adata2 as "wt_count" into adata1.var
    merged_var["wt_count"] = adata_wt_mcrs.var["mcrs_count"].rename("wt_count")

    # Assign the updated var back to adata1
    adata_mutant_mcrs.var = merged_var

    # Ensure there are no division by zero errors
    mcrs_count = adata_mutant_mcrs.var["mcrs_count"]
    wt_count = adata_mutant_mcrs.var["wt_count"]

    # Calculate VAF
    adata_mutant_mcrs.var["vaf_across_samples"] = mcrs_count / (mcrs_count + wt_count)

    # wherever wt_count has a NaN, I want adata_mutant_mcrs.var["vaf_across_samples"] to have a NaN
    adata_mutant_mcrs.var.loc[wt_count.isna(), "vaf_across_samples"] = pd.NA

    adata_mutant_mcrs.write(adata_mutant_mcrs_path_out)
    adata_wt_mcrs.write(adata_wt_mcrs_path_out)

    return adata_vaf_output


# convert gatk output vcf to pandas df
def vcf_to_dataframe(vcf_file, additional_columns = True):
    """Convert a VCF file to a Pandas DataFrame."""    
    vcf = pysam.VariantFile(vcf_file)
    
    # List to store VCF rows
    vcf_data = []
    
    # Fetch each record in the VCF
    for record in vcf.fetch():
        # For each record, extract the desired fields
        vcf_row = {
            'CHROM': record.chrom,
            'POS': record.pos,
            'ID': record.id,
            'REF': record.ref,
            'ALT': ','.join(record.alts),  # ALT can be multiple
        }

        if additional_columns:
            vcf_row['QUAL'] = record.qual
            vcf_row['FILTER'] = ';'.join(record.filter.keys()) if record.filter else None,  # FILTER keys

            # Add INFO fields
            for key, value in record.info.items():
                vcf_row[f'INFO_{key}'] = value
            
            # Add per-sample data (FORMAT fields)
            for sample, sample_data in record.samples.items():
                for format_key, format_value in sample_data.items():
                    vcf_row[f'{sample}_{format_key}'] = format_value

        
        # Append the row to the list
        vcf_data.append(vcf_row)
    
    # Convert the list to a Pandas DataFrame
    df = pd.DataFrame(vcf_data)
    
    return df

def add_mutation_type(mutations, mut_column):
    mutations["mutation_type_id"] = mutations[mut_column].str.extract(mutation_pattern)[1]

    # Define conditions and choices for the mutation types
    conditions = [
        mutations["mutation_type_id"].str.contains(">", na=False),
        mutations["mutation_type_id"].str.contains("delins", na=False),
        mutations["mutation_type_id"].str.contains("del", na=False) & ~mutations["mutation_type_id"].str.contains("delins", na=False),
        mutations["mutation_type_id"].str.contains("ins", na=False) & ~mutations["mutation_type_id"].str.contains("delins", na=False),
        mutations["mutation_type_id"].str.contains("dup", na=False),
        mutations["mutation_type_id"].str.contains("inv", na=False),
    ]

    choices = [
        "substitution",
        "delins",
        "deletion",
        "insertion",
        "duplication",
        "inversion",
    ]

    # Assign the mutation types
    mutations["mutation_type"] = np.select(conditions, choices, default="unknown")

    # Drop the temporary mutation_type_id column
    mutations.drop(columns=["mutation_type_id"], inplace=True)

    return mutations

def add_vcf_info_to_cosmic_tsv(cosmic_tsv, reference_genome_fasta, cosmic_df_out = None, cosmic_cdna_info_csv = None, mutation_source = "cds"):
    # load in COSMIC tsv with columns CHROM, POS, ID, REF, ALT
    cosmic_df = pd.read_csv(cosmic_tsv, sep="\t", usecols=["Mutation genome position GRCh37", "GENOMIC_WT_ALLELE_SEQ", "GENOMIC_MUT_ALLELE_SEQ", "ACCESSION_NUMBER", "Mutation CDS", "MUTATION_URL"])

    if mutation_source == "cdna":
        cosmic_cdna_info_df = pd.read_csv(cosmic_cdna_info_csv, usecols=["mutation_id", "mutation"])
        cosmic_cdna_info_df = cosmic_cdna_info_df.rename(columns={"mutation": "Mutation cDNA"})

    cosmic_df = add_mutation_type(cosmic_df, "Mutation CDS")

    cosmic_df["ACCESSION_NUMBER"] = cosmic_df["ACCESSION_NUMBER"].str.split(".").str[0]

    cosmic_df[['CHROM', 'GENOME_POS']] = cosmic_df['Mutation genome position GRCh37'].str.split(':', expand=True)
    # cosmic_df['CHROM'] = cosmic_df['CHROM'].apply(convert_chromosome_value_to_int_when_possible)
    cosmic_df[['POS', 'GENOME_END_POS']] = cosmic_df['GENOME_POS'].str.split('-', expand=True)

    cosmic_df = cosmic_df.rename(
        columns={
            "GENOMIC_WT_ALLELE_SEQ": "REF",
            "GENOMIC_MUT_ALLELE_SEQ": "ALT",
            "MUTATION_URL": "mutation_id"
        }
    )

    if mutation_source == "cds":
        cosmic_df['ID'] = cosmic_df['ACCESSION_NUMBER'] + ":" + cosmic_df['Mutation CDS']
    elif mutation_source == "cdna":
        cosmic_df["mutation_id"] = cosmic_df["mutation_id"].str.extract(r"id=(\d+)")
        cosmic_df['mutation_id'] = cosmic_df['mutation_id'].astype(int, errors='raise')
        cosmic_df = cosmic_df.merge(cosmic_cdna_info_df[['mutation_id', 'Mutation cDNA']], on='mutation_id', how='left')
        cosmic_df['ID'] = cosmic_df['ACCESSION_NUMBER'] + ":" + cosmic_df['Mutation cDNA']
        cosmic_df.drop(columns=["Mutation cDNA"], inplace=True)

    cosmic_df = cosmic_df.dropna(subset=['CHROM', 'POS'])
    cosmic_df = cosmic_df.dropna(subset=['ID'])  # a result of intron mutations and COSMIC duplicates that get dropped before cDNA determination


    # reference_genome_fasta
    reference_genome = pysam.FastaFile(reference_genome_fasta)

    def get_nucleotide_from_reference(chromosome, position):
        # pysam is 0-based, so subtract 1 from the position
        return reference_genome.fetch(chromosome, int(position) - 1, int(position))

    def get_complement(nucleotide_sequence):
        return ''.join([complement[nuc] for nuc in nucleotide_sequence])

    # Insertion, get original nucleotide (not in COSMIC df)
    cosmic_df.loc[
        (cosmic_df['GENOME_END_POS'].astype(int) != 1) & (cosmic_df['mutation_type'] == 'insertion'), 'original_nucleotide'
    ] = cosmic_df.loc[
        (cosmic_df['GENOME_END_POS'].astype(int) != 1) & (cosmic_df['mutation_type'] == 'insertion'), ['CHROM', 'POS']
    ].progress_apply(
        lambda row: get_nucleotide_from_reference(row['CHROM'], int(row['POS'])),
        axis=1
    )

    # Deletion, get new nucleotide (not in COSMIC df)
    cosmic_df.loc[
        (cosmic_df['POS'].astype(int) != 1) & (cosmic_df['mutation_type'] == 'deletion'), 'original_nucleotide'
    ] = cosmic_df.loc[
        (cosmic_df['POS'].astype(int) != 1) & (cosmic_df['mutation_type'] == 'deletion'), ['CHROM', 'POS']
    ].progress_apply(
        lambda row: get_nucleotide_from_reference(row['CHROM'], int(row['POS']) - 1),
        axis=1
    )

    # Duplication
    cosmic_df.loc[cosmic_df['mutation_type'] == 'duplication', 'original_nucleotide'] = cosmic_df.loc[cosmic_df['ID'].str.contains('dup', na=False), 'ALT'].str[-1]

    # deal with start of 1, insertion
    cosmic_df.loc[
        (cosmic_df['GENOME_END_POS'].astype(int) == 1) & (cosmic_df['mutation_type'] == 'insertion'), 'original_nucleotide'
    ] = cosmic_df.loc[
        (cosmic_df['GENOME_END_POS'].astype(int) == 1) & (cosmic_df['mutation_type'] == 'insertion'), ['CHROM', 'POS']
    ].progress_apply(
        lambda row: get_nucleotide_from_reference(row['CHROM'], int(row['GENOME_END_POS'])),
        axis=1
    )

    # deal with start of 1, deletion
    cosmic_df.loc[
        (cosmic_df['POS'].astype(int) == 1) & (cosmic_df['mutation_type'] == 'deletion'), 'original_nucleotide'
    ] = cosmic_df.loc[
        (cosmic_df['POS'].astype(int) == 1) & (cosmic_df['mutation_type'] == 'deletion'), ['CHROM', 'POS']
    ].progress_apply(
        lambda row: get_nucleotide_from_reference(row['CHROM'], int(row['GENOME_END_POS']) + 1),
        axis=1
    )

    # # deal with (-) strand - commented out because the vcf should all be relative to the forward strand, not the cdna
    # cosmic_df.loc[cosmic_df['strand'] == '-', 'original_nucleotide'] = cosmic_df.loc[cosmic_df['strand'] == '-', 'original_nucleotide'].apply(get_complement)




    # ins and dup, starting position not 1
    cosmic_df.loc[(((cosmic_df['mutation_type'] == 'insertion') | (cosmic_df['mutation_type'] == 'duplication')) & (cosmic_df['POS'].astype(int) != 1)), 'ref_updated'] = cosmic_df.loc[(((cosmic_df['mutation_type'] == 'insertion') | (cosmic_df['mutation_type'] == 'duplication')) & (cosmic_df['POS'].astype(int) != 1)), 'original_nucleotide']
    cosmic_df.loc[(((cosmic_df['mutation_type'] == 'insertion') | (cosmic_df['mutation_type'] == 'duplication')) & (cosmic_df['POS'].astype(int) != 1)), 'alt_updated'] = cosmic_df.loc[(((cosmic_df['mutation_type'] == 'insertion') | (cosmic_df['mutation_type'] == 'duplication')) & (cosmic_df['POS'].astype(int) != 1)), 'original_nucleotide'] + cosmic_df.loc[(((cosmic_df['mutation_type'] == 'insertion') | (cosmic_df['mutation_type'] == 'duplication')) & (cosmic_df['POS'].astype(int) != 1)), 'ALT']

    # ins and dup, starting position 1
    cosmic_df.loc[(((cosmic_df['mutation_type'] == 'insertion') | (cosmic_df['mutation_type'] == 'duplication')) & (cosmic_df['POS'].astype(int) == 1)), 'ref_updated'] = cosmic_df.loc[(((cosmic_df['mutation_type'] == 'insertion') | (cosmic_df['mutation_type'] == 'duplication')) & (cosmic_df['POS'].astype(int) == 1)), 'original_nucleotide']
    cosmic_df.loc[(((cosmic_df['mutation_type'] == 'insertion') | (cosmic_df['mutation_type'] == 'duplication')) & (cosmic_df['POS'].astype(int) == 1)), 'alt_updated'] = cosmic_df.loc[(((cosmic_df['mutation_type'] == 'insertion') | (cosmic_df['mutation_type'] == 'duplication')) & (cosmic_df['POS'].astype(int) == 1)), 'ALT'] + cosmic_df.loc[(((cosmic_df['mutation_type'] == 'insertion') | (cosmic_df['mutation_type'] == 'duplication')) & (cosmic_df['POS'].astype(int) == 1)), 'original_nucleotide']


    # del, starting position not 1
    cosmic_df.loc[((cosmic_df['mutation_type'] == 'deletion') & (cosmic_df['POS'].astype(int) != 1)), 'ref_updated'] = cosmic_df.loc[((cosmic_df['mutation_type'] == 'deletion') & (cosmic_df['POS'].astype(int) != 1)), 'original_nucleotide'] + cosmic_df.loc[((cosmic_df['mutation_type'] == 'deletion') & (cosmic_df['POS'].astype(int) != 1)), 'REF']
    cosmic_df.loc[((cosmic_df['mutation_type'] == 'deletion') & (cosmic_df['POS'].astype(int) != 1)), 'alt_updated'] = cosmic_df.loc[((cosmic_df['mutation_type'] == 'deletion') & (cosmic_df['POS'].astype(int) != 1)), 'original_nucleotide']

    # del, starting position 1
    cosmic_df.loc[((cosmic_df['mutation_type'] == 'deletion') & (cosmic_df['POS'].astype(int) == 1)), 'ref_updated'] = cosmic_df.loc[((cosmic_df['mutation_type'] == 'deletion') & (cosmic_df['POS'].astype(int) == 1)), 'REF'] + cosmic_df.loc[((cosmic_df['mutation_type'] == 'deletion') & (cosmic_df['POS'].astype(int) == 1)), 'original_nucleotide']
    cosmic_df.loc[((cosmic_df['mutation_type'] == 'deletion') & (cosmic_df['POS'].astype(int) == 1)), 'alt_updated'] = cosmic_df.loc[((cosmic_df['mutation_type'] == 'deletion') & (cosmic_df['POS'].astype(int) == 1)), 'original_nucleotide']



    # Deletion, update position (should refer to 1 BEFORE the deletion)
    cosmic_df.loc[
        (cosmic_df['POS'].astype(int) != 1) & (cosmic_df['mutation_type'] == 'deletion'), 'POS'
    ] = cosmic_df.loc[
        (cosmic_df['POS'].astype(int) != 1) & (cosmic_df['mutation_type'] == 'deletion'), 'POS'
    ].progress_apply(
        lambda pos: int(pos) - 1
    )

    # deal with start of 1, deletion update position (should refer to 1 after the deletion)
    cosmic_df.loc[
        (cosmic_df['POS'].astype(int) == 1) & (cosmic_df['mutation_type'] == 'deletion'), 'POS'
    ] = cosmic_df.loc[
        (cosmic_df['POS'].astype(int) == 1) & (cosmic_df['mutation_type'] == 'deletion'), 'GENOME_END_POS'
    ].astype(int) + 1

    # Insertion, update position when pos=1 (should refer to 1)
    cosmic_df.loc[
        (cosmic_df['GENOME_END_POS'].astype(int) == 1) & (cosmic_df['mutation_type'] == 'insertion'), 'POS'
    ] = 1

    cosmic_df['ref_updated'] = cosmic_df['ref_updated'].fillna(cosmic_df['REF'])
    cosmic_df['alt_updated'] = cosmic_df['alt_updated'].fillna(cosmic_df['ALT'])
    cosmic_df.rename(columns={'ALT': 'alt_cosmic', 'alt_updated': 'ALT', 'REF': 'ref_cosmic', 'ref_updated': 'REF'}, inplace=True)
    cosmic_df.drop(columns=["Mutation genome position GRCh37", "GENOME_POS", "GENOME_END_POS", "ACCESSION_NUMBER", "Mutation CDS", "mutation_id", 'ref_cosmic', 'alt_cosmic', 'original_nucleotide', 'mutation_type'], inplace=True)  # 'strand'

    num_rows_with_na = cosmic_df.isna().any(axis=1).sum()
    assert num_rows_with_na == 0, f"Number of rows with NA values: {num_rows_with_na}"

    cosmic_df['POS'] = cosmic_df['POS'].astype(np.int64)

    if cosmic_df_out:
        cosmic_df.to_csv(cosmic_df_out, index=False)

    return cosmic_df


# TODO: make sure this works for rows with just ID and everything else blank (due to different mutations being concatenated)
def write_to_vcf(adata_var, output_file):
    """
    Write adata.var DataFrame to a VCF file.

    Parameters:
        adata_var (pd.DataFrame): DataFrame with VCF columns (CHROM, POS, REF, ALT, ID, DP, AF, NS).
        output_file (str): Path to the output VCF file.
    """
    # Open VCF file for writing
    with open(output_file, "w") as vcf_file:
        # Write VCF header
        vcf_file.write("##fileformat=VCFv4.2\n")
        vcf_file.write('##INFO=<ID=DP,Number=1,Type=Integer,Description="Total Depth">\n')
        vcf_file.write('##INFO=<ID=AF,Number=A,Type=Float,Description="Variant Allele Frequency">\n')
        vcf_file.write('##INFO=<ID=NS,Number=1,Type=Integer,Description="Number of Samples">\n')
        vcf_file.write("#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\n")
        
        # Write each row of the DataFrame
        for _, row in adata_var.iterrows():
            # Construct INFO field
            info_fields = [
                f"DP={row['DP']}" if pd.notna(row['DP']) else None,
                f"AF={row['AF']}" if pd.notna(row['AF']) else None,
                f"NS={row['NS']}" if pd.notna(row['NS']) else None,
            ]
            info = ";".join(filter(None, info_fields))
            
            # Write VCF row
            vcf_file.write(
                f"{row['CHROM']}\t{row['POS']}\t{row['ID']}\t{row['REF']}\t{row['ALT']}\t.\tPASS\t{info}\n"
            )

# TODO: make sure this works for rows with just ID and everything else blank (due to different mutations being concatenated)
def write_vcfs_for_rows(adata, adata_wt_mcrs, adata_vaf, output_dir):
    """
    Write a VCF file for each row (variant) in adata.var.

    Parameters:
        adata: AnnData object with mutant counts.
        adata_wt_mcrs: AnnData object with wild-type counts.
        adata_vaf: AnnData object with VAF values.
        output_dir: Directory to save VCF files.
    """
    for idx, row in adata.var.iterrows():
        # Extract VCF fields from adata.var
        chrom = row["CHROM"]
        pos = row["POS"]
        var_id = row["ID"]
        ref = row["REF"]
        alt = row["ALT"]
        mcrs_id = row["mcrs_id"]  # This is the index for the column in the matrices
        
        # Extract corresponding matrix values
        mutant_counts = adata[:, mcrs_id].X.flatten()  # Extract as 1D array
        wt_counts = adata_wt_mcrs[:, mcrs_id].X.flatten()  # Extract as 1D array
        vaf_values = adata_vaf[:, mcrs_id].X.flatten()  # Extract as 1D array
        
        # Create VCF file for the row
        output_file = f"{output_dir}/{var_id}.vcf"
        with open(output_file, "w") as vcf_file:
            # Write VCF header
            vcf_file.write("##fileformat=VCFv4.2\n")
            vcf_file.write('##INFO=<ID=RD,Number=1,Type=Integer,Description="Total Depth">\n')
            vcf_file.write('##INFO=<ID=AF,Number=A,Type=Float,Description="Allele Frequency">\n')
            vcf_file.write('##INFO=<ID=NS,Number=1,Type=Integer,Description="Number of Samples">\n')
            vcf_file.write("#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\n")
            
            # Iterate through samples (rows in the matrix)
            for sample_idx in range(len(mutant_counts)):
                # Calculate RD and AF
                rd = mutant_counts[sample_idx] + wt_counts[sample_idx]
                af = vaf_values[sample_idx]
                
                # INFO field
                info = f"RD={int(rd)};AF={af:.3f};NS=1"
                
                # Write VCF row
                vcf_file.write(f"{chrom}\t{pos}\t{var_id}\t{ref}\t{alt}\t.\tPASS\t{info}\n")