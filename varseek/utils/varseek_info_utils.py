import ast
import gzip
import hashlib
import math
import os
import random
import re
import shutil
import subprocess
import sys
from bisect import bisect_left
from collections import defaultdict

import numpy as np
import pandas as pd
import pyfastx
from tqdm import tqdm

tqdm.pandas()

from varseek.utils.logger_utils import splitext_custom, get_file_name_without_extensions_or_full_path, get_printlog
from varseek.utils.seq_utils import (
    create_header_to_sequence_ordered_dict_from_fasta_WITHOUT_semicolon_splitting,
    fasta_to_fastq,
    filter_fasta,
    make_mapping_dict,
    reverse_complement,
    safe_literal_eval,
    get_header_set_from_fastq
)
from varseek.utils.visualization_utils import (
    plot_basic_bar_plot_from_dict,
    plot_descending_bar_plot,
    plot_histogram_notebook_1,
    plot_histogram_of_nearby_mutations_7_5,
    print_column_summary_stats,
)

tqdm.pandas()


def remove_dlist_duplicates(input_file, output_file=None):
    if output_file is None:
        output_file = input_file + ".tmp"  # Write to a temporary file

    # TODO: replace with pyfastx (and don't forget to erase the `header = header[1:]` line when I do)
    # TODO: make header fasta from id fasta with id:header dict

    try:
        sequence_to_headers_dict = {}
        with open(input_file, "r", encoding="utf-8") as file:
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

        with open(output_file, "w", encoding="utf-8") as file:
            for sequence, header in sequence_to_headers_dict.items():
                file.write(f"{header}\n{sequence}\n")

        if output_file == input_file + ".tmp":
            os.replace(output_file, input_file)

    # TODO: make id fasta from header fasta with id:header dict

    except Exception as e:
        if os.path.exists(output_file):
            os.remove(output_file)
        raise e
        


def capitalize_sequences(input_file, output_file=None):
    if output_file is None:
        output_file = input_file + ".tmp"  # Write to a temporary file
    try:
        with open(input_file, "r", encoding="utf-8") as infile, open(output_file, "w", encoding="utf-8") as outfile:
            for line in infile:
                if line.startswith(">"):
                    outfile.write(line)
                else:
                    outfile.write(line.upper())
        if output_file == input_file + ".tmp":
            os.replace(output_file, input_file)
    except Exception as e:
        if os.path.exists(output_file):
            os.remove(output_file)
        raise e
            


def parse_sam_and_extract_sequences(
    sam_file,
    ref_genome_file,
    output_fasta_file,
    k=31,
    dfk_length=None,
    capitalize=True,
    remove_duplicates=False,
    check_for_bad_cigars=True,
    logger=None
):
    logger_info = get_printlog(logger=logger)

    if dfk_length is None:
        dfk_length = k + 2

    ref_genome = {header: sequence for header, sequence in pyfastx.Fastx(ref_genome_file)}

    with open(sam_file, "r", encoding="utf-8") as f, open(output_fasta_file, "w", encoding="utf-8") as dlist_fasta:
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
            logger_info(f"Skipped {bad_cigar} reads with bad CIGAR strings")

    if capitalize:
        logger_info("Capitalizing sequences")
        capitalize_sequences(output_fasta_file)

    if remove_duplicates:  #!!! not fully working yet
        logger_info("Removing duplicate sequences")
        remove_dlist_duplicates(output_fasta_file)

def process_sam_file(sam_file):
    with open(sam_file, "r", encoding="utf-8") as sam:
        for line in sam:
            if line.startswith("@"):
                continue

            fields = line.split("\t")
            yield fields


def get_set_of_headers_from_sam(sam_file, id_to_header_csv=None, check_for_bad_cigars=False, k="", return_set=True):
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
        cleaned_sequence_names = [id_to_header_dict[seq_id] for seq_id in cleaned_sequence_names]

    if return_set:
        cleaned_sequence_names = set(cleaned_sequence_names)

    return cleaned_sequence_names


def sequence_match(vcrs_sequence, dlist_sequence, strandedness=False):
    if strandedness:
        # Check only forward strand
        return vcrs_sequence in dlist_sequence
    else:
        # Check both forward and reverse complement
        return (vcrs_sequence in dlist_sequence) or (vcrs_sequence in reverse_complement(dlist_sequence))

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
    selected_substring = sequence[start_position : (start_position + read_length)]

    return selected_substring


def remove_Ns_fasta(fasta_file, max_ambiguous_reference=0, logger=None):
    logger_info = get_printlog(logger=logger)
    fasta_file_temp = fasta_file + ".tmp"
    try:
        i = 0
        if max_ambiguous_reference == 0:  # no Ns allowed
            condition = lambda sequence: "N" not in sequence.upper()
        else:  # at most max_ambiguous_reference Ns
            condition = lambda sequence: sequence.upper().count("N") <= max_ambiguous_reference
        with open(fasta_file_temp, "w", encoding="utf-8") as outfile:
            for header, sequence in pyfastx.Fastx(fasta_file):
                if condition(sequence):
                    outfile.write(f">{header}\n{sequence}\n")
                else:
                    i += 1
        
        os.replace(fasta_file_temp, fasta_file)
        logger_info(f"Removed {i} sequences with Ns from {fasta_file}")
    except Exception as e:
        if os.path.exists(fasta_file_temp):
            os.remove(fasta_file_temp)
        raise e
    

# there used to be a count_nearby_mutations_efficient function that I erased on 2/25/25
def count_nearby_mutations_efficient_with_identifiers(df, k, fasta_entry_column, start_column, end_column, header_column):
    # Ensure the required columns are in the DataFrame
    required_columns = [fasta_entry_column, start_column, end_column, header_column]
    for col in required_columns:
        if col not in df.columns:
            raise ValueError(f"The DataFrame must contain the column '{col}'.")

    # Initialize counts_unique array and nearby_headers_list
    counts_unique = np.zeros(len(df), dtype=int)
    nearby_headers_list = [set() for _ in range(len(df))]  # List of sets to store nearby headers per mutation

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
                mapping = dict(enumerate(indices_original))

                # Prepare DataFrames with positions, group indices, and headers
                df_starts = pd.DataFrame({"position": starts, "index": np.arange(N), "header": headers})
                df_ends = pd.DataFrame({"position": ends, "index": np.arange(N), "header": headers})

                # Sort by positions
                df_starts_sorted = df_starts.sort_values("position").reset_index(drop=True)
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
    df["nearby_variants_count"] = counts_unique
    df["nearby_variants"] = nearby_headers_list
    return df


def create_df_of_vcrs_to_self_headers(
    vcrs_sam_file,
    vcrs_fa,
    bowtie_vcrs_reference_folder,
    bowtie_path=None,
    threads=2,
    strandedness=False,
    vcrs_id_column="vcrs_id",
    output_stat_file=None,
    logger=None,
):
    logger_info = get_printlog(logger=logger)
    if not bowtie_path:
        bowtie2_build = "bowtie2-build"
        bowtie2 = "bowtie2"
    else:
        bowtie2_build = f"{bowtie_path}/bowtie2-build"
        bowtie2 = f"{bowtie_path}/bowtie2"

    if not os.path.exists(vcrs_sam_file):
        if not os.path.exists(bowtie_vcrs_reference_folder) or not os.listdir(bowtie_vcrs_reference_folder):
            logger_info("Running bowtie2 build")
            os.makedirs(bowtie_vcrs_reference_folder, exist_ok=True)
            bowtie_reference_prefix = os.path.join(bowtie_vcrs_reference_folder, "vcrs")
            subprocess.run(
                [
                    bowtie2_build,  # Path to the bowtie2-build executable
                    "--threads",
                    str(threads),  # Number of threads
                    vcrs_fa,  # Input FASTA file
                    bowtie_reference_prefix,  # Output reference folder
                ],
                check=True,
            )

        logger_info("Running bowtie2 alignment")

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
            vcrs_fa,  # Input FASTA file
            "-S",
            vcrs_sam_file,  # Output SAM file
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
            with open(output_stat_file, write_mode, encoding="utf-8") as f:
                f.write(f"bowtie alignment for {bowtie_reference_prefix}")
                f.write("Standard Output:\n")
                f.write(result.stdout)
                f.write("\n\nStandard Error:\n")
                f.write(result.stderr)
                f.write("\n\n")

    substring_to_superstring_list_dict = defaultdict(list)
    superstring_to_substring_list_dict = defaultdict(list)

    logger_info("Processing SAM file")
    for fields in process_sam_file(vcrs_sam_file):
        read_name = fields[0]
        ref_name = fields[2]

        if read_name == ref_name:
            continue

        substring_to_superstring_list_dict[read_name].append(ref_name)
        superstring_to_substring_list_dict[ref_name].append(read_name)

    # convert to DataFrame
    substring_to_superstring_df = pd.DataFrame(
        substring_to_superstring_list_dict.items(),
        columns=[vcrs_id_column, "VCRSs_for_which_this_VCRS_is_a_substring"],
    )
    superstring_to_substring_df = pd.DataFrame(
        superstring_to_substring_list_dict.items(),
        columns=[vcrs_id_column, "VCRSs_for_which_this_VCRS_is_a_superstring"],
    )

    substring_to_superstring_df["VCRS_is_a_substring_of_another_VCRS"] = True
    superstring_to_substring_df["VCRS_is_a_superstring_of_another_VCRS"] = True

    substring_to_superstring_df[vcrs_id_column] = substring_to_superstring_df[vcrs_id_column].astype(str)
    superstring_to_substring_df[vcrs_id_column] = superstring_to_substring_df[vcrs_id_column].astype(str)

    return substring_to_superstring_df, superstring_to_substring_df


def compare_cdna_and_genome(mutation_metadata_df_exploded, varseek_build_temp_folder="vk_build_tmp", reference_cdna_fasta="cdna", reference_genome_fasta="genome", mutations_csv=None, w=30, variant_source="cdna", columns_to_explode=None, seq_id_column_cdna="seq_ID", var_column_cdna="mutation_cdna", seq_id_column_genome="chromosome", var_column_genome="mutation_genome", delete_temp_dir=True):
    from varseek.varseek_build import build

    if columns_to_explode is None:
        columns_to_explode = ["header"]
    else:
        columns_to_explode = columns_to_explode.copy()

    reference_out_dir_temp = f"{varseek_build_temp_folder}/reference_out"
    varseek_build_cdna_out_df = f"{varseek_build_temp_folder}/varseek_build_cdna_{w}.csv"

    if not os.path.exists(varseek_build_cdna_out_df):
        build(
            sequences=reference_cdna_fasta,
            variants=mutations_csv,
            out=varseek_build_temp_folder,
            reference_out_dir=reference_out_dir_temp,
            w=w,
            remove_seqs_with_wt_kmers=False,
            optimize_flanking_regions=False,
            min_seq_len=None,
            max_ambiguous=None,
            required_insertion_overlap_length=None,
            merge_identical=False,
            cosmic_email=os.getenv("COSMIC_EMAIL"),
            cosmic_password=os.getenv("COSMIC_PASSWORD"),
            save_variants_updated_csv=True,
            variants_updated_csv_out=varseek_build_cdna_out_df,
            seq_id_column=seq_id_column_cdna,
            var_column=var_column_cdna,
        )

    cdna_updated_df = pd.read_csv(
        varseek_build_cdna_out_df,
        usecols=[
            "header",
            "vcrs_sequence",
            seq_id_column_cdna,
            var_column_cdna,
            "variant_type",
        ],
    )

    varseek_build_genome_out_df = f"{varseek_build_temp_folder}/varseek_build_genome_{w}.csv"

    if not os.path.exists(varseek_build_genome_out_df):
        build(
            sequences=reference_genome_fasta,
            variants=mutations_csv,
            out=varseek_build_temp_folder,
            reference_out_dir=reference_out_dir_temp,
            w=w,
            remove_seqs_with_wt_kmers=False,
            optimize_flanking_regions=False,
            min_seq_len=None,
            max_ambiguous=None,
            required_insertion_overlap_length=None,
            merge_identical=False,
            cosmic_email=os.getenv("COSMIC_EMAIL"),
            cosmic_password=os.getenv("COSMIC_PASSWORD"),
            save_variants_updated_csv=True,
            variants_updated_csv_out=varseek_build_genome_out_df,
            seq_id_column=seq_id_column_genome,
            var_column=var_column_genome,
        )

    genome_updated_df = pd.read_csv(
        varseek_build_genome_out_df,
        usecols=[
            "header",
            "vcrs_sequence",
            seq_id_column_genome,
            var_column_genome,
            "variant_type",
            seq_id_column_cdna,
            var_column_cdna,
        ],
    )

    combined_updated_df = cdna_updated_df.merge(
        genome_updated_df,
        on=[seq_id_column_cdna, var_column_cdna],
        how="outer",
        suffixes=("_cdna", "_genome"),
    )

    combined_updated_df["cdna_and_genome_same"] = combined_updated_df["vcrs_sequence_cdna"] == combined_updated_df["vcrs_sequence_genome"]
    # combined_updated_df["cdna_and_genome_same"] = combined_updated_df["cdna_and_genome_same"].astype(str)

    if "cosmic" in mutations_csv:
        # cosmic is not reliable at recording duplication mutations at the genome level
        combined_updated_df.loc[
            (combined_updated_df["variant_type_cdna"] == "duplication") | (combined_updated_df["variant_type_genome"] == "duplication"),
            "cdna_and_genome_same",
        ] = np.nan

    if variant_source == "combined":
        column_to_merge = "header_cdna"
    else:
        column_to_merge = "header"
        combined_updated_df.rename(columns={f"header_{variant_source}": "header"}, inplace=True)

    # mutation_metadata_df_exploded = explode_df(mutation_metadata_df, columns_to_explode)

    mutation_metadata_df_exploded = mutation_metadata_df_exploded.merge(
        combined_updated_df[[column_to_merge, "cdna_and_genome_same"]],
        on=column_to_merge,
        how="left",
    )

    columns_to_explode.append("cdna_and_genome_same")

    # mutation_metadata_df, columns_to_explode = collapse_df(mutation_metadata_df_exploded, columns_to_explode, columns_to_explode_extend_values = ["cdna_and_genome_same"])

    # # mutation_metadata_df["cdna_and_genome_same"] = mutation_metadata_df["cdna_and_genome_same"].fillna("unsure")  # because I'm filling values with unsure, I must specify == True if indexing true values
    # # mutation_metadata_df = mutation_metadata_df.loc[~((mutation_metadata_df["cdna_and_genome_same"] == "True") & (mutation_metadata_df["variant_source"] == "genome"))]  #* uncomment to filter out rows derived from genome where cDNA and genome are the same (I used to filter these out because they are redundant and I only wanted to keep rows where genome differed from cDNA)

    # delete temp folder and all contents
    if delete_temp_dir:
        shutil.rmtree(varseek_build_temp_folder)

    return mutation_metadata_df_exploded, columns_to_explode


def check_dlist_header(dlist_header, pattern):
    return bool(pattern.search(dlist_header))


def get_long_headers(fasta_file, length_threshold=250):
    return {header for header, _ in pyfastx.Fastx(fasta_file) if len(header) > length_threshold}


def hash_kmer_security_specified(kmer):
    """Return the MD5 hash of a k-mer as a hexadecimal string."""
    return hashlib.md5(kmer.encode("utf-8"), usedforsecurity=False).hexdigest()


def hash_kmer(kmer):
    """Return the MD5 hash of a k-mer as a hexadecimal string."""
    return hashlib.md5(kmer.encode("utf-8")).hexdigest()


# using hash has the upside of less memory when k > 32 (because hashes are fixed 32 length), but introduce the chance of collisions
def count_kmer_overlaps_new(fasta_file, k=31, strandedness=False, vcrs_id_column="vcrs_id", use_hash=False):
    """Count k-mer overlaps between sequences in the FASTA file."""
    # Parse the FASTA file and store sequences
    fasta_read_only = pyfastx.Fastx(fasta_file)  # new Feb 2025

    if sys.version_info >= (3, 9):
        hash_kmer_function = hash_kmer_security_specified
    else:
        hash_kmer_function = hash_kmer

    # Create a combined k-mer overlap dictionary
    kmer_to_seqids = defaultdict(set)
    for seq_id, sequence in tqdm(fasta_read_only, desc="Generating k-mers", unit="sequence"):
        for kmer in generate_kmers(sequence, k, strandedness=strandedness):
            if use_hash:
                kmer = hash_kmer_function(kmer)  # Hash the k-mer
            kmer_to_seqids[kmer].add(seq_id)

    # Process forward sequences only, checking overlaps with both forward and reverse complement k-mers
    fasta_read_only = pyfastx.Fastx(fasta_file)  # new Feb 2025 - repeated because the previous loop has already exhausted the generator
    results = []
    for seq_id, sequence in tqdm(fasta_read_only, desc="Checking overlaps", unit="sequence"):
        kmers = generate_kmers(sequence, k)
        overlapping_kmers = 0
        distinct_sequences_set = set()
        overlapping_kmers_set = set()

        for kmer in kmers:
            kmer_original = kmer  # only necessary because of the possibility of using hash
            if use_hash:
                kmer = hash_kmer(kmer)
            if strandedness:
                if len(kmer_to_seqids[kmer]) > 1:
                    overlapping_kmers += 1
                    overlapping_kmers_set.add(kmer_original)
                    distinct_sequences_set.update(kmer_to_seqids[kmer])
            else:
                kmer_rc = reverse_complement(kmer)
                if use_hash:
                    kmer_rc = hash_kmer(kmer_rc)
                if len(kmer_to_seqids[kmer_rc]) > 1 or len(kmer_to_seqids[kmer]) > 1:
                    overlapping_kmers += 1
                    overlapping_kmers_set.add(kmer_original)
                    distinct_sequences_set.update(kmer_to_seqids[kmer])
                    distinct_sequences_set.update(kmer_to_seqids[kmer_rc])

        # Remove the current sequence from the distinct sequences count
        distinct_sequences_set.discard(seq_id)

        # Store results
        results.append(
            {
                vcrs_id_column: seq_id,
                "number_of_kmers_with_overlap_to_other_VCRSs": overlapping_kmers,
                "number_of_other_VCRSs_with_overlapping_kmers": len(distinct_sequences_set),
                "VCRSs_with_overlapping_kmers": distinct_sequences_set,
                # "overlapping_kmers": overlapping_kmers_set,
            }
        )

    # Convert results to a DataFrame
    df = pd.DataFrame(results)

    return df


def generate_kmers(sequence, k, strandedness=True):
    """Generate k-mers of length k from a sequence."""
    if strandedness:
        return [sequence[i : (i + k)] for i in range(len(sequence) - k + 1)]
    else:
        list_f = [sequence[i : (i + k)] for i in range(len(sequence) - k + 1)]
        sequence_rc = reverse_complement(sequence)
        list_rc = [sequence_rc[i : (i + k)] for i in range(len(sequence_rc) - k + 1)]
        return list_f + list_rc


def get_vcrs_headers_that_are_substring_dlist(
    mutation_reference_file_fasta,
    dlist_fasta_file,
    header_column_name="vcrs_id",
    strandedness=False,
):
    vcrs_headers_that_are_substring_dlist = []

    mutant_reference = create_header_to_sequence_ordered_dict_from_fasta_WITHOUT_semicolon_splitting(mutation_reference_file_fasta)

    for dlist_header, dlist_sequence in pyfastx.Fastx(dlist_fasta_file):
        vcrs_header = dlist_header.rsplit("_", 1)[0]
        if sequence_match(mutant_reference[vcrs_header], dlist_sequence, strandedness=strandedness):
            vcrs_headers_that_are_substring_dlist.append(vcrs_header)

    df = pd.DataFrame(vcrs_headers_that_are_substring_dlist, columns=[header_column_name]).drop_duplicates()

    df["substring_alignment_to_reference_count"] = df[header_column_name].map(pd.Series(vcrs_headers_that_are_substring_dlist).value_counts())

    df["substring_alignment_to_reference"] = True

    return df


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
    triplets = [sequence[i : (i + 3)] for i in range(len(sequence) - 2)]

    # Number of distinct triplets
    distinct_triplets = set(triplets)

    # Number of total triplets
    total_triplets = len(triplets)

    # Triplet complexity: ratio of distinct triplets to total triplets
    triplet_complexity = len(distinct_triplets) / total_triplets if total_triplets > 0 else 0

    return len(distinct_triplets), total_triplets, triplet_complexity


def get_vcrss_that_pseudoalign_but_arent_dlisted(
    mutation_metadata_df,
    vcrs_id_column,
    vcrs_fa,
    sequence_names_set,
    human_reference_genome_fa,
    human_reference_gtf,
    out_dir_notebook=".",
    ref_folder_kb=None,
    header_column_name="vcrs_id",
    additional_kb_extract_filtering_workflow="nac",
    k=31,
    threads=2,
    strandedness=False,
    column_name="pseudoaligned_to_human_reference_despite_not_truly_aligning",
):
    if ref_folder_kb is None:
        ref_folder_kb = out_dir_notebook
    vcrs_fa_base, vcrs_fa_ext = splitext_custom(vcrs_fa)
    vcrs_fa_filtered_bowtie = f"{vcrs_fa_base}_filtered_bowtie{vcrs_fa_ext}"
    vcrs_fQ_filtered_bowtie = f"{vcrs_fa_base}_filtered_bowtie.fq"
    kb_ref_wt = f"{ref_folder_kb}/reference"
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
    else:
        raise ValueError("additional_kb_extract_filtering_workflow must be either 'standard' or 'nac'")

    filter_fasta(vcrs_fa, vcrs_fa_filtered_bowtie, sequence_names_set)

    fasta_to_fastq(vcrs_fa_filtered_bowtie, vcrs_fQ_filtered_bowtie, k=None)

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

    kb_extract_out_dir_bowtie_filtered = f"{out_dir_notebook}/kb_extract_bowtie_filtered"

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
        vcrs_fQ_filtered_bowtie,
    ]

    if strandedness:
        kb_extract_command = kb_extract_command[:4] + ["--strand", "forward"] + kb_extract_command[4:]

    try:
        subprocess.run(kb_extract_command, check=True)

        kb_extract_output_fastq_file = f"{kb_extract_out_dir_bowtie_filtered}/all/1.fastq.gz"

        problematic_mutations_total = get_header_set_from_fastq(kb_extract_output_fastq_file, "list")

        df = pd.DataFrame(problematic_mutations_total, columns=[header_column_name]).drop_duplicates()

        df[column_name] = True

        df[vcrs_id_column] = df[vcrs_id_column].astype(str)

        mutation_metadata_df = pd.merge(
            mutation_metadata_df,
            df,
            on=vcrs_id_column,
            how="left",
        )

    except Exception:
        print("No reads pseudoaligned - setting entire column to False")
        mutation_metadata_df[column_name] = False

    return mutation_metadata_df


def get_df_overlap(
    vcrs_fa,
    out_dir_notebook=".",
    k=31,
    strandedness=False,
    vcrs_id_column="vcrs_id",
    output_text_file=None,
    output_plot_folder=None,
):
    df_overlap_save_path = f"{out_dir_notebook}/kmer_overlap_stats.csv"
    df_overlap = count_kmer_overlaps_new(vcrs_fa, k=k, strandedness=strandedness, vcrs_id_column=vcrs_id_column)
    df_overlap.to_csv(df_overlap_save_path, index=False)

    print_column_summary_stats(
        df_overlap,
        "number_of_kmers_with_overlap_to_other_VCRSs",
        output_file=output_text_file,
    )
    print_column_summary_stats(
        df_overlap,
        "number_of_other_VCRSs_with_overlapping_kmers",
        output_file=output_text_file,
    )

    kmer_plot_file = f"{output_plot_folder}/kmer_overlap_histogram.png"

    plot_histogram_notebook_1(
        df_overlap,
        column="number_of_kmers_with_overlap_to_other_VCRSs",
        x_label="Number of K-mers with overlap",
        title="Histogram of K-mers with Overlap",
        output_plot_file=kmer_plot_file,
    )

    vcrs_plot_file = f"{output_plot_folder}/vcrs_overlap_histogram.png"

    plot_histogram_notebook_1(
        df_overlap,
        column="number_of_other_VCRSs_with_overlapping_kmers",
        x_label="Number of VCRS items with Overlapping K-mers",
        title="Histogram of VCRS items with Overlapping K-mers",
        output_plot_file=vcrs_plot_file,
    )

    df_overlap[vcrs_id_column] = df_overlap[vcrs_id_column].astype(str)

    return df_overlap


def convert_to_list_in_df(value, reference_length=0):
    if isinstance(value, str):
        try:
            # Safely convert string representation of a list to an actual list
            return ast.literal_eval(value)
        except (ValueError, SyntaxError):
            # If conversion fails, return an empty list or handle accordingly
            if reference_length == 0:
                return []
            # else:
            return [np.nan] * reference_length
    return value  # If already a list, return as is


def explode_df(mutation_metadata_df, columns_to_explode=None, verbose=False, logger=None):
    logger_info = get_printlog(logger=logger)
    if columns_to_explode is None:
        columns_to_explode = ["header", "order"]
    else:  # * remove with set
        columns_to_explode = columns_to_explode.copy()
        columns_to_explode = list(set(columns_to_explode))
    if "header_list" not in mutation_metadata_df.columns:
        mutation_metadata_df["header_list"] = mutation_metadata_df["vcrs_header"].str.split(";")
    if "order_list" not in mutation_metadata_df.columns:
        mutation_metadata_df["order_list"] = mutation_metadata_df["header_list"].apply(lambda x: list(range(len(x))))

    mutation_metadata_df["header"] = mutation_metadata_df["header_list"]
    mutation_metadata_df["order"] = mutation_metadata_df["order_list"]

    # for column in columns_to_explode:
    #     mutation_metadata_df[column] = mutation_metadata_df.apply(
    #         lambda row: convert_to_list_in_df(row[column], len(row['header']) if isinstance(row['header'], list) else 1),
    #         axis=1
    #     )

    logger_info("About to apply safe evals")
    if verbose:
        for column in tqdm(columns_to_explode, desc="Checking columns"):
            mutation_metadata_df[column] = mutation_metadata_df[column].progress_apply(safe_literal_eval)
    else:
        for column in columns_to_explode:
            mutation_metadata_df[column] = mutation_metadata_df[column].apply(safe_literal_eval)

    mutation_metadata_df_exploded = mutation_metadata_df.explode(list(columns_to_explode)).reset_index(drop=True)

    return mutation_metadata_df_exploded


def collapse_df(
    mutation_metadata_df_exploded,
    columns_to_explode=None
):
    if columns_to_explode is None:
        columns_to_explode = ["header", "order"]
    else:  # * remove with set
        columns_to_explode = columns_to_explode.copy()

    for column in list(columns_to_explode):
        mutation_metadata_df_exploded[column] = mutation_metadata_df_exploded[column].apply(lambda x: tuple(x) if isinstance(x, list) else x)

    mutation_metadata_df = (
        mutation_metadata_df_exploded.sort_values("order")
        .groupby("vcrs_header", as_index=False)
        .agg(
            {
                **{col: list for col in list(columns_to_explode)},  # list these values
                **{col: "first" for col in mutation_metadata_df_exploded.columns if col not in list(columns_to_explode) + ["vcrs_header"]},
            }  # Take the first value for other columns
        )
        .reset_index(drop=True)
    )

    return mutation_metadata_df, columns_to_explode


# TODO: save in txt file
def fasta_summary_stats(fa, output_file=None):
    try:
        if isinstance(fa, str):
            fa = pyfastx.Fasta(fa)

        try:
            gc_content = fa.gc_content
        except Exception:
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
            with open(output_file, "w", encoding="utf-8") as f:
                for line in summary:
                    f.write(line + "\n")  # Write to file with a newline

    except Exception as e:
        print(f"Error: {e}")


def create_df_of_dlist_headers(dlist_path, k=None, header_column_name="vcrs_id"):
    dlist_headers_list_updated = get_set_of_headers_from_sam(dlist_path, k=k, return_set=False)

    df = pd.DataFrame(dlist_headers_list_updated, columns=[header_column_name]).drop_duplicates()

    df["alignment_to_reference_count_total"] = df[header_column_name].map(pd.Series(dlist_headers_list_updated).value_counts())

    df["alignment_to_reference"] = True

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
    seq_id_genome_column="chromosome"
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
        if pd.isna(row[seq_id_genome_column]) or pd.isna(row["start_variant_position_genome"]) or pd.isna(row["end_variant_position_genome"]):
            distances.append(np.nan)
            continue

        try:
            chrom = str(row[seq_id_genome_column])
            start_pos = int(row["start_variant_position_genome"])
            end_pos = int(row["end_variant_position_genome"])
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

    mutation_metadata_df_exploded[f"is_near_splice_junction_{near_splice_junction_threshold}"] = mutation_metadata_df_exploded["distance_to_nearest_splice_junction"].apply(lambda x: x <= near_splice_junction_threshold if pd.notna(x) else np.nan)

    columns_to_explode.extend(
        [
            "distance_to_nearest_splice_junction",
            f"is_near_splice_junction_{near_splice_junction_threshold}",
        ]
    )

    # mutation_metadata_df, columns_to_explode = collapse_df(mutation_metadata_df_exploded, columns_to_explode, columns_to_explode_extend_values = ["distance_to_nearest_splice_junction", f"is_near_splice_junction_{near_splice_junction_threshold}"])

    return mutation_metadata_df_exploded, columns_to_explode


def run_bowtie_build_dlist(ref_fa, ref_folder, ref_prefix, bowtie2_build, threads=2, logger=None):
    logger_info = get_printlog(logger=logger)
    if not os.path.exists(ref_folder) or not os.listdir(ref_folder):
        logger_info("Running bowtie2 build")
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

        logger_info("Bowtie2 build complete")


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
    max_ambiguous_vcrs=0,
    output_stat_file=None,
    logger=None
):
    logger_info = get_printlog(logger=logger)
    if not os.path.exists(output_sam_file):
        logger_info("Running bowtie2 alignment")

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
            f"C,0,{max_ambiguous_vcrs}",  # N-ceiling
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
            with open(output_stat_file, write_mode, encoding="utf-8") as f:
                f.write(f"bowtie alignment for {bowtie_reference_prefix}")
                f.write("Standard Output:\n")
                f.write(result.stdout)
                f.write("\n\nStandard Error:\n")
                f.write(result.stderr)
                f.write("\n\n")

        logger_info("Bowtie2 alignment complete")


def calculate_total_gene_info(
    mutation_metadata_df_exploded,
    vcrs_id_column="vcrs_id",
    gene_name_column="gene_name",
    output_stat_file=None,
    output_plot_folder=None,
    columns_to_include="all",
    columns_to_explode=None,
):
    if columns_to_explode is None:
        columns_to_explode = ["header"]
    else:
        columns_to_explode = columns_to_explode.copy()

    number_of_mutations_total = len(mutation_metadata_df_exploded[vcrs_id_column].unique())
    number_of_transcripts_total = len(mutation_metadata_df_exploded["seq_ID_used_for_vcrs"].unique())
    number_of_genes_total = len(mutation_metadata_df_exploded[gene_name_column].unique())

    metadata_counts_dict = {
        "Mutations_total": number_of_mutations_total,
        "Transcripts_total": number_of_transcripts_total,
        "Genes_total": number_of_genes_total,
    }

    if output_stat_file is not None:
        with open(output_stat_file, "w", encoding="utf-8") as f:
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
        mutation_metadata_df_exploded["header_with_gene_name"] = mutation_metadata_df_exploded["header"].str.split(":", n=1).str[0] + "(" + mutation_metadata_df_exploded[gene_name_column] + "):" + mutation_metadata_df_exploded["header"].str.split(":", n=1).str[1]

    if columns_to_include == "all" or "number_of_variants_in_this_gene_total" in columns_to_include:
        gene_counts = mutation_metadata_df_exploded[gene_name_column].value_counts()
        mutation_metadata_df_exploded["number_of_variants_in_this_gene_total"] = mutation_metadata_df_exploded[gene_name_column].map(gene_counts)

        output_plot_file_descending_bar_plot = f"{output_plot_folder}/descending_bar_plot.png"

        plot_descending_bar_plot(
            gene_counts,
            x_label="Gene Name",
            y_label="Number of Occurrences",
            tick_interval=5000,
            output_file=output_plot_file_descending_bar_plot,
        )

    columns_to_explode.append("number_of_variants_in_this_gene_total")

    return mutation_metadata_df_exploded, columns_to_explode


def calculate_nearby_mutations(
    variant_source_column,
    k,
    output_plot_folder,
    variant_source,
    mutation_metadata_df_exploded,
    columns_to_explode=None,
    seq_id_cdna_column="seq_ID",
    seq_id_genome_column="chromosome",
    logger=None
):
    logger_info = get_printlog(logger=logger)
    if columns_to_explode is None:
        columns_to_explode = ["header", "order"]
    else:  # * remove with set
        columns_to_explode = columns_to_explode.copy()

    columns_to_explode_extend_values = [
        "nearby_variants",
        "nearby_variants_count",
        "has_a_nearby_variant",
    ]

    if variant_source != "combined":
        mutation_metadata_df_exploded_copy = mutation_metadata_df_exploded.copy()
        mutation_metadata_df_exploded_copy = count_nearby_mutations_efficient_with_identifiers(
            mutation_metadata_df_exploded_copy,
            k=k,
            fasta_entry_column="seq_ID_used_for_vcrs",
            start_column="start_variant_position",
            end_column="end_variant_position",
            header_column="header",
        )
        mutation_metadata_df_exploded = mutation_metadata_df_exploded.merge(
            mutation_metadata_df_exploded_copy[["header", "nearby_variants"]],
            on="header",
            how="left",
        )
        mutation_metadata_df_exploded["nearby_variants"] = mutation_metadata_df_exploded["nearby_variants"].apply(lambda x: [] if isinstance(x, float) and pd.isna(x) else x)

    else:
        # find other mutations within (k-1) of each mutation for cDNA
        start_column = "start_variant_position_cdna" if "start_variant_position_cdna" in mutation_metadata_df_exploded.columns else "start_variant_position"
        end_column = "end_variant_position_cdna" if "end_variant_position_cdna" in mutation_metadata_df_exploded.columns else "end_variant_position"

        mutation_metadata_df_exploded_cdna = mutation_metadata_df_exploded.loc[mutation_metadata_df_exploded[variant_source_column] == "cdna"].reset_index(drop=True)
        mutation_metadata_df_exploded_cdna = count_nearby_mutations_efficient_with_identifiers(
            mutation_metadata_df_exploded_cdna,
            k=k,
            fasta_entry_column=seq_id_cdna_column,
            start_column=start_column,
            end_column=end_column,
            header_column="header",
        )  # * change to header_column='header_cdna' (along with below) if I don't want to distinguish between spliced and unspliced variants being close
        mutation_metadata_df_exploded = mutation_metadata_df_exploded.merge(
            mutation_metadata_df_exploded_cdna[["header", "nearby_variants"]],
            on="header",
            how="left",
        )
        mutation_metadata_df_exploded.rename(columns={"nearby_variants": "nearby_mutations_cdna"}, inplace=True)
        mutation_metadata_df_exploded["nearby_mutations_cdna"] = mutation_metadata_df_exploded["nearby_mutations_cdna"].apply(lambda x: [] if isinstance(x, float) and pd.isna(x) else x)
        # Step 1: Create two new columns for the length of each list, treating NaN as 0
        mutation_metadata_df_exploded["nearby_mutations_count_cdna"] = mutation_metadata_df_exploded["nearby_mutations_cdna"].apply(lambda x: len(x) if isinstance(x, list) else 0)
        mutation_metadata_df_exploded["has_a_nearby_mutation_cdna"] = mutation_metadata_df_exploded["nearby_mutations_count_cdna"] > 0
        columns_to_explode_extend_values.extend(
            [
                "nearby_mutations_cdna",
                "nearby_mutations_count_cdna",
                "has_a_nearby_mutation_cdna",
            ]
        )

        # find other mutations within (k-1) of each mutation for genome
        mutation_metadata_df_exploded_genome = mutation_metadata_df_exploded.copy()  # mutation_metadata_df.loc[(mutation_metadata_df[variant_source_column] == "cdna") | (mutation_metadata_df['cdna_and_genome_same'] != "True")].reset_index(drop=True)  #* uncomment this filtering if I only want to keep genome cases that differ from cdna
        mutation_metadata_df_exploded_genome = count_nearby_mutations_efficient_with_identifiers(
            mutation_metadata_df_exploded_genome,
            k=k,
            fasta_entry_column=seq_id_genome_column,
            start_column="start_variant_position_genome",
            end_column="end_variant_position_genome",
            header_column="header",
        )  # * change to header_column='header_cdna' (along with above) if I don't want to distinguish between spliced and unspliced variants being close
        mutation_metadata_df_exploded = mutation_metadata_df_exploded.merge(
            mutation_metadata_df_exploded_genome[["header", "nearby_variants"]],
            on="header",
            how="left",
        )
        mutation_metadata_df_exploded.rename(columns={"nearby_variants": "nearby_mutations_genome"}, inplace=True)
        mutation_metadata_df_exploded["nearby_mutations_genome"] = mutation_metadata_df_exploded["nearby_mutations_genome"].apply(lambda x: [] if isinstance(x, float) and pd.isna(x) else x)
        # Step 1: Create two new columns for the length of each list, treating NaN as 0
        mutation_metadata_df_exploded["nearby_mutations_count_genome"] = mutation_metadata_df_exploded["nearby_mutations_genome"].apply(lambda x: len(x) if isinstance(x, list) else 0)
        mutation_metadata_df_exploded["has_a_nearby_mutation_genome"] = mutation_metadata_df_exploded["nearby_mutations_count_genome"] > 0
        columns_to_explode_extend_values.extend(
            [
                "nearby_mutations_genome",
                "nearby_mutations_count_genome",
                "has_a_nearby_mutation_genome",
            ]
        )

        mutation_metadata_df_exploded["nearby_variants"] = mutation_metadata_df_exploded.apply(
            lambda row: list(set((row["nearby_mutations_cdna"]) + (row["nearby_mutations_genome"]))),
            axis=1,
        )

    mutation_metadata_df_exploded["nearby_variants_count"] = mutation_metadata_df_exploded["nearby_variants"].apply(lambda x: len(x) if isinstance(x, list) else 0)
    mutation_metadata_df_exploded["has_a_nearby_variant"] = mutation_metadata_df_exploded["nearby_variants_count"] > 0
    logger_info(f"Number of mutations with nearby mutations: {mutation_metadata_df_exploded['has_a_nearby_variant'].sum()} {mutation_metadata_df_exploded['has_a_nearby_variant'].sum() / len(mutation_metadata_df_exploded) * 100:.2f}%")
    bins = min(int(mutation_metadata_df_exploded["nearby_variants_count"].max()), 1000)
    nearby_mutations_output_plot_file = f"{output_plot_folder}/nearby_mutations_histogram.png"
    plot_histogram_of_nearby_mutations_7_5(
        mutation_metadata_df_exploded,
        column="nearby_variants_count",
        bins=bins,
        output_file=nearby_mutations_output_plot_file,
    )

    columns_to_explode.extend(columns_to_explode_extend_values)

    return mutation_metadata_df_exploded, columns_to_explode


def align_to_normal_genome_and_build_dlist(
    mutations,
    vcrs_id_column,
    out_dir_notebook,
    reference_out,
    ref_prefix,
    strandedness,
    threads,
    N_penalty,
    max_ambiguous_vcrs,
    max_ambiguous_reference,
    k,
    output_stat_folder,
    mutation_metadata_df,
    bowtie2_build,
    bowtie2,
    dlist_reference_genome_fasta,
    dlist_reference_cdna_fasta,
    dlist_fasta_file_genome_full=None,
    dlist_fasta_file_cdna_full=None,
    dlist_fasta_file=None,
    logger=None,
):
    logger_info = get_printlog(logger=logger)
    bowtie_stat_file = f"{output_stat_folder}/bowtie_alignment.txt"

    ref_folder_genome_bowtie = f"{reference_out}/bowtie_index_genome"
    ref_prefix_genome_full = f"{ref_folder_genome_bowtie}/{ref_prefix}"
    output_sam_file_genome = f"{out_dir_notebook}/bowtie_vcrs_kmers_to_genome/alignment.sam"

    if not os.path.exists(ref_folder_genome_bowtie) or not os.listdir(ref_folder_genome_bowtie):
        run_bowtie_build_dlist(
            ref_fa=dlist_reference_genome_fasta,
            ref_folder=ref_folder_genome_bowtie,
            ref_prefix=ref_prefix_genome_full,
            bowtie2_build=bowtie2_build,
            threads=threads,
            logger=logger,
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
            max_ambiguous_vcrs=max_ambiguous_vcrs,
            output_stat_file=bowtie_stat_file,
            logger=logger
        )

    dlist_genome_df = create_df_of_dlist_headers(output_sam_file_genome, header_column_name=vcrs_id_column, k=k)
    dlist_genome_df.rename(columns={"alignment_to_reference_count_total": "alignment_to_reference_count_genome"}, inplace=True)

    if not dlist_fasta_file_genome_full:
        dlist_fasta_file_genome_full = f"{out_dir_notebook}/dlist_genome.fa"
    if not os.path.exists(dlist_fasta_file_genome_full):
        parse_sam_and_extract_sequences(
            output_sam_file_genome,
            dlist_reference_genome_fasta,
            dlist_fasta_file_genome_full,
            k=k,
            capitalize=True,
            remove_duplicates=False,
            logger=logger
        )

    dlist_substring_genome_df = get_vcrs_headers_that_are_substring_dlist(
        mutation_reference_file_fasta=mutations,
        dlist_fasta_file=dlist_fasta_file_genome_full,
        strandedness=strandedness,
        header_column_name=vcrs_id_column,
    )

    dlist_genome_df[vcrs_id_column] = dlist_genome_df[vcrs_id_column].astype(str)
    dlist_substring_genome_df[vcrs_id_column] = dlist_substring_genome_df[vcrs_id_column].astype(str)
    dlist_genome_df = pd.merge(dlist_genome_df, dlist_substring_genome_df, on=vcrs_id_column, how="left")
    dlist_genome_df["substring_alignment_to_reference"] = dlist_genome_df["substring_alignment_to_reference"].fillna(False)

    if max_ambiguous_reference < 9999:  #! be careful of changing this number - it is related to the condition in varseek info - max_ambiguous_reference = 99999
        remove_Ns_fasta(dlist_fasta_file_genome_full, max_ambiguous_reference=max_ambiguous_reference, logger=logger)

    ref_folder_cdna_bowtie = f"{reference_out}/bowtie_index_transcriptome"
    ref_prefix_cdna_full = f"{ref_folder_cdna_bowtie}/{ref_prefix}"
    output_sam_file_cdna = f"{out_dir_notebook}/bowtie_vcrs_kmers_to_transcriptome/alignment.sam"

    if not os.path.exists(ref_folder_cdna_bowtie) or not os.listdir(ref_folder_cdna_bowtie):
        run_bowtie_build_dlist(
            ref_fa=dlist_reference_cdna_fasta,
            ref_folder=ref_folder_cdna_bowtie,
            ref_prefix=ref_prefix_cdna_full,
            bowtie2_build=bowtie2_build,
            threads=threads,
            logger=logger
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
            max_ambiguous_vcrs=max_ambiguous_vcrs,
            output_stat_file=bowtie_stat_file,
            logger=logger
        )

    dlist_cdna_df = create_df_of_dlist_headers(output_sam_file_cdna, header_column_name=vcrs_id_column, k=k)
    dlist_cdna_df.rename(columns={"alignment_to_reference_count_total": "alignment_to_reference_count_cdna"}, inplace=True)

    if not dlist_fasta_file_cdna_full:
        dlist_fasta_file_cdna_full = f"{out_dir_notebook}/dlist_cdna.fa"
    if not os.path.exists(dlist_fasta_file_cdna_full):
        parse_sam_and_extract_sequences(
            output_sam_file_cdna,
            dlist_reference_cdna_fasta,
            dlist_fasta_file_cdna_full,
            k=k,
            capitalize=True,
            remove_duplicates=False,
            logger=logger
        )

    dlist_substring_cdna_df = get_vcrs_headers_that_are_substring_dlist(
        mutation_reference_file_fasta=mutations,
        dlist_fasta_file=dlist_fasta_file_cdna_full,
        strandedness=strandedness,
        header_column_name=vcrs_id_column,
    )

    dlist_cdna_df[vcrs_id_column] = dlist_cdna_df[vcrs_id_column].astype(str)

    dlist_cdna_df[vcrs_id_column] = dlist_cdna_df[vcrs_id_column].astype(str)
    dlist_substring_cdna_df[vcrs_id_column] = dlist_substring_cdna_df[vcrs_id_column].astype(str)
    dlist_cdna_df = pd.merge(dlist_cdna_df, dlist_substring_cdna_df, on=vcrs_id_column, how="left")
    dlist_cdna_df["substring_alignment_to_reference"] = dlist_cdna_df["substring_alignment_to_reference"].fillna(False)

    if max_ambiguous_reference < 9999:  #! be careful of changing this number - it is related to the condition in varseek info - max_ambiguous_reference = 99999
        remove_Ns_fasta(dlist_fasta_file_cdna_full, max_ambiguous_reference=max_ambiguous_reference, logger=logger)

    if not dlist_fasta_file:
        dlist_fasta_file = f"{out_dir_notebook}/dlist.fa"

    # concatenate d-lists into one file
    with open(dlist_fasta_file, "w", encoding="utf-8") as outfile:
        # Write the contents of the first input file to the output file
        with open(dlist_fasta_file_genome_full, "r", encoding="utf-8") as infile1:
            outfile.write(infile1.read())

            # Write the contents of the second input file to the output file
        with open(dlist_fasta_file_genome_full, "r", encoding="utf-8") as infile2:
            outfile.write(infile2.read())

    dlist_combined_df = dlist_cdna_df.merge(
        dlist_genome_df,
        on=vcrs_id_column,
        how="outer",
        suffixes=("_cdna", "_genome"),
    )

    mutation_metadata_df = mutation_metadata_df.merge(
        dlist_combined_df[
            [
                vcrs_id_column,
                "alignment_to_reference_cdna",
                "alignment_to_reference_genome",
                "substring_alignment_to_reference_cdna",
                "substring_alignment_to_reference_genome",
                "alignment_to_reference_count_cdna",
                "alignment_to_reference_count_genome",
                "substring_alignment_to_reference_count_cdna",
                "substring_alignment_to_reference_count_genome",
            ]
        ],
        on=vcrs_id_column,
        how="left",
    )

    mutation_metadata_df["alignment_to_reference_cdna"] = mutation_metadata_df["alignment_to_reference_cdna"].fillna(False).astype(bool)
    mutation_metadata_df["alignment_to_reference_genome"] = mutation_metadata_df["alignment_to_reference_genome"].fillna(False).astype(bool)
    mutation_metadata_df['alignment_to_reference'] = (mutation_metadata_df['alignment_to_reference_cdna'] | mutation_metadata_df['alignment_to_reference_genome']).astype(bool)

    mutation_metadata_df["substring_alignment_to_reference_cdna"] = mutation_metadata_df["substring_alignment_to_reference_cdna"].fillna(False).astype(bool)
    mutation_metadata_df["substring_alignment_to_reference_genome"] = mutation_metadata_df["substring_alignment_to_reference_genome"].fillna(False).astype(bool)
    mutation_metadata_df['substring_alignment_to_reference'] = (mutation_metadata_df['substring_alignment_to_reference_cdna'] | mutation_metadata_df['substring_alignment_to_reference_genome']).astype(bool)

    mutation_metadata_df["alignment_to_reference_count_cdna"] = mutation_metadata_df["alignment_to_reference_count_cdna"].fillna(0).astype(int)
    mutation_metadata_df["alignment_to_reference_count_genome"] = mutation_metadata_df["alignment_to_reference_count_genome"].fillna(0).astype(int)
    mutation_metadata_df["alignment_to_reference_count_total"] = mutation_metadata_df["alignment_to_reference_count_cdna"] + mutation_metadata_df["alignment_to_reference_count_genome"]
    
    mutation_metadata_df["substring_alignment_to_reference_count_cdna"] = mutation_metadata_df["substring_alignment_to_reference_count_cdna"].fillna(0).astype(int)
    mutation_metadata_df["substring_alignment_to_reference_count_genome"] = mutation_metadata_df["substring_alignment_to_reference_count_genome"].fillna(0).astype(int)
    mutation_metadata_df["substring_alignment_to_reference_count_total"] = mutation_metadata_df["substring_alignment_to_reference_count_cdna"] + mutation_metadata_df["substring_alignment_to_reference_count_genome"]

    # TODO: for those that dlist in the genome, add an additional check to see if they filter in coding regions (I already check for spliced with cDNA, but I don't distinguish unspliced coding vs noncoding) - add to column alignment_to_reference_coding_genome, substring_alignment_to_reference_coding_genome, alignment_to_reference_count_coding_genome, substring_alignment_to_reference_count_coding_genome

    count_cdna_unique = ((mutation_metadata_df["alignment_to_reference_cdna"]) & (~mutation_metadata_df["alignment_to_reference_genome"])).sum()
    count_genome_unique = ((mutation_metadata_df["alignment_to_reference_genome"]) & (~mutation_metadata_df["alignment_to_reference_cdna"])).sum()
    count_cdna_and_genome_intersection = ((mutation_metadata_df["alignment_to_reference_genome"]) & (mutation_metadata_df["alignment_to_reference_cdna"])).sum()
    count_cdna_total = mutation_metadata_df["alignment_to_reference_cdna"].sum()
    count_genome_total = mutation_metadata_df["alignment_to_reference_genome"].sum()
    count_cdna_or_genome_union = mutation_metadata_df["alignment_to_reference"].sum()

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
        logger_info(message)

    if os.path.exists(bowtie_stat_file):
        write_mode = "a"
    else:
        write_mode = "w"

    # Re-print and save the messages to a text file
    with open(bowtie_stat_file, write_mode, encoding="utf-8") as f:  # Use 'a' to append to the file
        f.write("Bowtie alignment statistics summary\n")
        for message in log_messages:
            f.write(message + "\n")  # Write to file

    sequence_names_set_genome = get_set_of_headers_from_sam(output_sam_file_genome, k=k)
    sequence_names_set_cdna = get_set_of_headers_from_sam(output_sam_file_cdna, k=k)
    sequence_names_set_union_genome_and_cdna = sequence_names_set_genome | sequence_names_set_cdna
    return (mutation_metadata_df, sequence_names_set_union_genome_and_cdna)


def identify_variant_source(mutation_metadata_df_exploded, variant_source_column = "variant_source"):
    # intentionally modifies mutation_metadata_df_exploded in-place
    choices = ["cdna", "genome"]

    conditions = [
        mutation_metadata_df_exploded["variant_used_for_vcrs"].str.startswith("c.", na=False),
        mutation_metadata_df_exploded["variant_used_for_vcrs"].str.startswith("g.", na=False),
    ]  # if it finds ":c.", make it "cdna"; if it finds ":g.", make it "genome"; if it finds both or neither, make it "unknown"

    mutation_metadata_df_exploded[variant_source_column] = np.select(
        conditions, choices, default="unknown"
    )