"""varseek sequencing utilities."""

import ast
import csv
import gzip
import os
import random
import re
import shutil
import subprocess
from collections import OrderedDict
from typing import Callable

import anndata as ad
import numpy as np
import pandas as pd
import pyfastx
import requests
from Bio.Seq import Seq
from tqdm import tqdm

from varseek.constants import (
    complement,
    complement_trans,
    fastq_extensions,
    mutation_pattern,
)
from varseek.utils.logger_utils import get_printlog, splitext_custom

tqdm.pandas()

# dlist_pattern_utils = re.compile(r"^(\d+)_(\d+)$")   # re.compile(r"^(unspliced)?(\d+)(;(unspliced)?\d+)*_(\d+)$")   # re.compile(r"^(unspliced)?(ENST\d+:(?:c\.|g\.)\d+(_\d+)?([a-zA-Z>]+))(;(unspliced)?ENST\d+:(?:c\.|g\.)\d+(_\d+)?([a-zA-Z>]+))*_\d+$")
# TODO: change when I change unspliced notation
dlist_pattern_utils = re.compile(
    r"^(\d+)_(\d+)$"  # First pattern: digits underscore digits
    r"|^(unspliced)?(\d+)(;(unspliced)?\d+)*_(\d+)$"  # Second pattern: optional unspliced, digits, underscore, digits
    r"|^(unspliced)?(ENST\d+:(?:c\.|g\.)\d+(_\d+)?([a-zA-Z>]+))(;(unspliced)?ENST\d+:(?:c\.|g\.)\d+(_\d+)?([a-zA-Z>]+))*_\d+$"  # Third pattern: complex ENST pattern
)


def read_fastq(fastq_file):
    is_gzipped = fastq_file.endswith(".gz")
    open_func = gzip.open if is_gzipped else open
    open_mode = "rt" if is_gzipped else "r"

    try:
        with open_func(fastq_file, open_mode) as file:
            while True:
                header = file.readline().strip()
                sequence = file.readline().strip()
                plus_line = file.readline().strip()
                quality = file.readline().strip()

                if not header:
                    break

                yield header, sequence, plus_line, quality
    except Exception as e:
        raise RuntimeError(f"Error reading FASTQ file '{fastq_file}': {e}") from e


def read_fasta(file_path, semicolon_split=False):
    is_gzipped = file_path.endswith(".gz")
    open_func = gzip.open if is_gzipped else open
    open_mode = "rt" if is_gzipped else "r"

    with open_func(file_path, open_mode) as file:
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
    return {header for header, _ in pyfastx.Fastx(synthetic_read_fa)}


def create_identity_t2g(mutation_reference_file_fasta, out="./cancer_mutant_reference_t2g.txt"):
    if not os.path.exists(out):
        with open(mutation_reference_file_fasta, "r", encoding="utf-8") as fasta, open(out, "w", encoding="utf-8") as t2g:
            for line in fasta:
                if line.startswith(">"):
                    header = line[1:].strip()
                    t2g.write(f"{header}\t{header}\n")
    else:
        print(f"{out} already exists")


def load_t2g_as_dict(file_path):
    t2g_dict = {}
    with open(file_path, "r", encoding="utf-8") as file:
        for line in file:
            # Strip any whitespace and split by tab
            key, value = line.strip().split("\t")
            t2g_dict[key] = value
    return t2g_dict


def generate_noisy_quality_scores(sequence, avg_quality=30, sd_quality=5, seed=None):
    if seed:
        random.seed(seed)

    # Assume a normal distribution for quality scores, with some fluctuation
    qualities = [max(0, min(40, int(random.gauss(avg_quality, sd_quality)))) for _ in sequence]
    # Convert qualities to ASCII Phred scores (33 is the offset)
    return "".join([chr(q + 33) for q in qualities])


def fasta_to_fastq(fasta_file, fastq_file, quality_score="I", k=None, add_noise=False, average_quality_for_noisy_reads=30, sd_quality_for_noisy_reads=5, seed=None, gzip_output=False):
    """
    Convert a FASTA file to a FASTQ file with a default quality score

    :param fasta_file: Path to the input FASTA file.
    :param fastq_file: Path to the output FASTQ file.
    :param quality_score: Default quality score to use for each base. Default is "I" (high quality).
    """
    if seed:
        random.seed(seed)
    open_func = gzip.open if gzip_output else open
    mode = "wt" if gzip_output else "w"
    with open_func(fastq_file, mode) as fastq:
        for sequence_id, sequence in pyfastx.Fastx(fasta_file):
            if k is None or k >= len(sequence):
                if add_noise:
                    quality_scores = generate_noisy_quality_scores(sequence, average_quality_for_noisy_reads, sd_quality_for_noisy_reads)  # don't pass seed in here since it is already set earlier
                else:
                    quality_scores = quality_score * len(sequence)
                fastq.write(f"@{sequence_id}\n")
                fastq.write(f"{sequence}\n")
                fastq.write("+\n")
                fastq.write(f"{quality_scores}\n")
            else:
                for i in range(len(sequence) - k + 1):
                    kmer = sequence[i : (i + k)]
                    if add_noise:
                        quality_scores = generate_noisy_quality_scores(kmer, average_quality_for_noisy_reads, sd_quality_for_noisy_reads)  # don't pass seed in here since it is already set earlier
                    else:
                        quality_scores = quality_score * k

                    fastq.write(f"@{sequence_id}_{i}\n")
                    fastq.write(f"{kmer}\n")
                    fastq.write("+\n")
                    fastq.write(f"{quality_scores}\n")


def reverse_complement(sequence):
    if pd.isna(sequence):  # Check if the sequence is NaN
        return np.nan
    return sequence.translate(complement_trans)[::-1]


def slow_reverse_complement(sequence):
    return "".join(complement.get(nucleotide, "N") for nucleotide in sequence[::-1])


def filter_fasta(input_fasta, output_fasta=None, sequence_names_set=None, keep="not_in_list"):
    if sequence_names_set is None:
        sequence_names_set = set()

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
        with open(input_fasta, "r", encoding="utf-8") as infile, open(output_fasta, "w", encoding="utf-8") as outfile:
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


def count_line_number_in_file(file):
    with open(file, "r", encoding="utf-8") as f:
        return sum(1 for _ in f)


def count_number_of_spliced_and_unspliced_headers(file):
    # TODO: make header fasta with id:header dict
    spliced_only_lines = 0
    unspliced_only_lines = 0
    spliced_and_unspliced_lines = 0
    for headers_concatenated, _ in pyfastx.Fastx(file):
        headers_list = headers_concatenated.split(";")
        spliced = False
        unspliced = False
        for header in headers_list:
            if "unspliced" in header:  # TODO: change when I change unspliced notation
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


def apply_enst_format(unique_mutations_genome, cosmic_reference_file_mutation_csv):
    # TODO: make header fasta with id:header dict
    unique_mutations_genome_enst_format = set()
    cosmic_df = pd.read_csv(
        cosmic_reference_file_mutation_csv,
        usecols=["seq_ID", "mutation_cdna", "chromosome", "mutation_genome"],
    )
    for header_genome in unique_mutations_genome:
        seq_id_genome, mutation_id_genome = header_genome.split(":", 1)
        row_corresponding_to_genome = cosmic_df[(cosmic_df["chromosome"] == seq_id_genome) & (cosmic_df["mutation_genome"] == mutation_id_genome)]
        seq_id_transcriptome_corresponding_to_genome = row_corresponding_to_genome["seq_ID"].iloc[0]
        mutation_id_transcriptome_corresponding_to_genome = row_corresponding_to_genome["mutation_cdna"].iloc[0]
        header_genome = f"{seq_id_transcriptome_corresponding_to_genome}:{mutation_id_transcriptome_corresponding_to_genome}"
        unique_mutations_genome_enst_format.add(header_genome)
    # TODO: make header fasta with id:header dict
    return unique_mutations_genome_enst_format


# Not used
def convert_nonsemicolon_headers_to_semicolon_joined_headers(nonsemicolon_read_headers_set, semicolon_reference_headers_set):
    # Step 1: Initialize the mapping dictionary
    component_to_item = {}

    # Step 2: Build the mapping from components to items in set2
    for item in semicolon_reference_headers_set:
        components = item.split(";")
        for component in components:
            if component in nonsemicolon_read_headers_set and component not in component_to_item:
                component_to_item[component] = item

    # Step 3: Create set1_updated using the mapped items
    semicolon_read_headers_set = set(component_to_item.values())

    return semicolon_read_headers_set


def create_read_header_to_reference_header_mapping_df(varseek_build_reference_headers_set, mutation_df_synthetic_read_headers_set):
    read_to_reference_header_mapping = {}

    for read in tqdm(varseek_build_reference_headers_set, desc="Processing reads"):
        if read in mutation_df_synthetic_read_headers_set:
            read_to_reference_header_mapping[read] = read
        else:
            for reference_item in varseek_build_reference_headers_set:
                if read in reference_item:
                    read_to_reference_header_mapping[read] = reference_item
                    break

    df = pd.DataFrame(
        list(varseek_build_reference_headers_set.items()),
        columns=["reference_header", "read_header"],
    )

    return df


def safe_literal_eval(val):
    if isinstance(val, str) and val.startswith("[") and val.endswith("]"):
        val = val.replace("np.nan", "None").replace("nan", "None").replace("<NA>", "None")
        try:
            # Attempt to parse the string as a literal
            parsed_val = ast.literal_eval(val)
            # If it's a list with NaN values, replace each entry with np.nan
            if isinstance(parsed_val, list):
                return [np.nan if isinstance(i, float) and np.isnan(i) else i for i in parsed_val]
            return parsed_val
        except (ValueError, SyntaxError):
            # If not a valid literal, return the original value
            return val
    else:
        return val


def get_header_set_from_fastq(fastq_file, output_format="set"):
    if output_format == "set":
        headers = {header[1:].strip() for header, _, _ in pyfastx.Fastx(fastq_file)}
    elif output_format == "list":
        headers = [header[1:].strip() for header, _, _ in pyfastx.Fastx(fastq_file)]
    else:
        raise ValueError(f"Invalid output_format: {output_format}")
    return headers


def create_header_to_sequence_ordered_dict_from_fasta_WITHOUT_semicolon_splitting(input_fasta, low_memory=False):
    if low_memory:
        mutant_reference = pyfastx.Fasta(input_fasta, build_index=True)
    else:
        mutant_reference = OrderedDict()
        for mutant_reference_header, mutant_reference_sequence in pyfastx.Fastx(input_fasta):
            mutant_reference[mutant_reference_header] = mutant_reference_sequence
    return mutant_reference


def contains_kmer_in_vcrs(read_sequence, vcrs_sequence, k):
    return any(read_sequence[i : (i + k)] in vcrs_sequence for i in range(len(read_sequence) - k + 1))


def check_for_read_kmer_in_vcrs(read_df, unique_vcrs_df, k, subset=None, strand=None):
    """
    Adds a column 'read_contains_kmer_in_vcrs' to read_df_subset indicating whether a k-mer
    from the read_sequence exists in the corresponding vcrs_sequence.

    Parameters:
    - read_df_subset: The subset of the read_df DataFrame (e.g., read_df.loc[read_df['FN']])
    - unique_vcrs_df: DataFrame containing 'vcrs_header' and 'vcrs_sequence' for lookups
    - k: The length of the k-mers to check for

    Returns:
    - The original DataFrame with the new 'read_contains_kmer_in_vcrs' column
    """

    # Step 1: Create a dictionary to map 'vcrs_header' to 'vcrs_sequence' for fast lookups
    vcrs_sequence_dict = unique_vcrs_df.set_index("vcrs_header")["vcrs_sequence"].to_dict() if strand != "r" else {}
    vcrs_sequence_dict_rc = unique_vcrs_df.set_index("vcrs_header")["vcrs_sequence_rc"].to_dict() if strand != "f" else {}

    def check_row_for_kmer(row, strand, k, vcrs_sequence_dict, vcrs_sequence_dict_rc):
        read_sequence = row["read_sequence"]

        contains_kmer_in_vcrs_f = False
        contains_kmer_in_vcrs_r = False

        if strand != "r":
            vcrs_sequence = vcrs_sequence_dict.get(row["vcrs_header"], "")
            contains_kmer_in_vcrs_f = contains_kmer_in_vcrs(read_sequence, vcrs_sequence, k)
            if strand == "f":
                return contains_kmer_in_vcrs_f

        if strand != "f":
            vcrs_sequence_rc = vcrs_sequence_dict_rc.get(row["vcrs_header"], "")
            contains_kmer_in_vcrs_r = contains_kmer_in_vcrs(Seq(read_sequence).reverse_complement(), vcrs_sequence_rc, k)
            if strand == "r":
                return contains_kmer_in_vcrs_r

        return contains_kmer_in_vcrs_f or contains_kmer_in_vcrs_r

    # Step 4: Initialize the column with NaN in the original read_df subset
    if "read_contains_kmer_in_vcrs" not in read_df.columns:
        read_df["read_contains_kmer_in_vcrs"] = np.nan

    # Step 5: Apply the function and update the 'read_contains_kmer_in_vcrs' column
    if subset is None:
        read_df["read_contains_kmer_in_vcrs"] = read_df.apply(lambda row: check_row_for_kmer(row, strand, k, vcrs_sequence_dict, vcrs_sequence_dict_rc), axis=1)
    else:
        read_df.loc[read_df[subset], "read_contains_kmer_in_vcrs"] = read_df.loc[read_df[subset]].apply(lambda row: check_row_for_kmer(row, strand, k, vcrs_sequence_dict, vcrs_sequence_dict_rc), axis=1)

    return read_df


def get_valid_ensembl_gene_id(row, transcript_column: str = "seq_ID", gene_column: str = "gene_name"):
    ensembl_gene_id = get_ensembl_gene_id(row[transcript_column])
    if ensembl_gene_id == "Unknown":
        return row[gene_column]
    return ensembl_gene_id


def get_ensembl_gene_id(transcript_id: str, verbose: bool = False):
    try:
        url = f"https://rest.ensembl.org/lookup/id/{transcript_id}?expand=1"
        response = requests.get(url, headers={"Content-Type": "application/json"}, timeout=10)

        if not response.ok:
            response.raise_for_status()

        data = response.json()

        return data.get("Parent")
    except Exception:
        if verbose:
            print(f"Error for: {transcript_id}")
        return "Unknown"


def get_ensembl_gene_id_bulk(transcript_ids: list[str]) -> dict[str, str]:
    if not transcript_ids:
        return {}

    try:
        url = "https://rest.ensembl.org/lookup/id/"
        response = requests.post(
            url,
            json={"ids": transcript_ids},
            headers={"Content-Type": "application/json"},
            timeout=10,
        )

        if not response.ok:
            response.raise_for_status()

        data = response.json()

        return {transcript_id: data[transcript_id].get("Parent") for transcript_id in transcript_ids if data[transcript_id]}
    except Exception as e:
        print(f"Failed to fetch gene IDs from Ensembl: {e}")
        raise e


def get_ensembl_gene_name_bulk(gene_ids: list[str]) -> dict[str, str]:
    if not gene_ids:
        return {}

    try:
        url = "https://rest.ensembl.org/lookup/id/"
        response = requests.post(url, json={"ids": gene_ids}, headers={"Content-Type": "application/json"}, timeout=10)

        if not response.ok:
            response.raise_for_status()

        data = response.json()

        return {gene_id: data[gene_id].get("display_name") for gene_id in gene_ids if data[gene_id]}
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


def make_mapping_dict(id_to_header_csv, dict_key="id"):
    mapping_dict = {}
    with open(id_to_header_csv, newline="", encoding="utf-8") as csvfile:
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
        base, ext = splitext_custom(in_fasta)
        out_fasta = f"{base}_with_headers{ext}"

    if id_to_header_csv.endswith(".csv"):
        id_to_header = make_mapping_dict(id_to_header_csv, dict_key="id")
    else:
        id_to_header = id_to_header_csv

    with open(out_fasta, "w", encoding="utf-8") as output_file:
        for seq_id, sequence in pyfastx.Fastx(in_fasta):
            output_file.write(f">{id_to_header[seq_id]}\n{sequence}\n")

    print("Swapping complete")


def swap_headers_for_ids_in_fasta(in_fasta, id_to_header_csv, out_fasta=None):
    if out_fasta is None:
        base, ext = splitext_custom(in_fasta)
        out_fasta = f"{base}_with_ids{ext}"

    if id_to_header_csv.endswith(".csv"):
        header_to_id = make_mapping_dict(id_to_header_csv, dict_key="header")
    else:
        header_to_id = id_to_header_csv

    with open(out_fasta, "w", encoding="utf-8") as output_file:
        for header, sequence in pyfastx.Fastx(in_fasta):
            output_file.write(f">{header_to_id[header]}\n{sequence}\n")

    print("Swapping complete")


def compare_dicts(dict1, dict2):
    # Find keys that are only in one of the dictionaries
    keys_only_in_dict1 = dict1.keys() - dict2.keys()
    keys_only_in_dict2 = dict2.keys() - dict1.keys()

    # Find keys that are in both dictionaries with differing values
    differing_values = {k: (dict1[k], dict2[k]) for k in dict1.keys() & dict2.keys() if dict1[k] != dict2[k]}

    # Report results
    if keys_only_in_dict1:
        print("Keys only in dict1:", keys_only_in_dict1)
    if keys_only_in_dict2:
        print("Keys only in dict2:", keys_only_in_dict2)
    if differing_values:
        print("Keys with differing values:", differing_values)
    if not keys_only_in_dict1 and not keys_only_in_dict2 and not differing_values:
        print("Dictionaries are identical.")


def download_t2t_reference_files(reference_out_dir_sequences_dlist):
    ref_dlist_fa_genome = f"{reference_out_dir_sequences_dlist}/GCF_009914755.1_T2T-CHM13v2.0_genomic.fna"
    ref_dlist_fa_cdna = f"{reference_out_dir_sequences_dlist}/rna.fna"
    ref_dlist_gtf = f"{reference_out_dir_sequences_dlist}/genomic.gtf"

    if os.path.exists(ref_dlist_fa_genome) and os.path.exists(ref_dlist_fa_cdna) and os.path.exists(ref_dlist_gtf):
        return ref_dlist_fa_genome, ref_dlist_fa_cdna, ref_dlist_gtf

    # Step 1: Download the ZIP file using wget
    download_url = "https://api.ncbi.nlm.nih.gov/datasets/v2alpha/genome/accession/GCF_009914755.1/download?include_annotation_type=GENOME_FASTA&include_annotation_type=RNA_FASTA&include_annotation_type=GENOME_GTF&hydrated=FULLY_HYDRATED"
    zip_file = f"{reference_out_dir_sequences_dlist}/t2t.zip"
    subprocess.run(["wget", "-O", zip_file, download_url], check=True)

    # Step 2: Unzip the downloaded file
    temp_dir = f"{reference_out_dir_sequences_dlist}/temp"
    subprocess.run(["unzip", zip_file, "-d", temp_dir], check=True)

    try:
        # Step 3: Move the files from the extracted directory to the target folder
        extracted_path = f"{temp_dir}/ncbi_dataset/data/GCF_009914755.1/"
        destination = reference_out_dir_sequences_dlist
        for filename in os.listdir(extracted_path):
            shutil.move(os.path.join(extracted_path, filename), destination)  # new Feb 2025 (used subprocess)
    except Exception as e:
        raise RuntimeError(f"Error moving files: {e}") from e
    finally:
        # Step 4: Remove the temporary folder
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir, ignore_errors=True)  # new Feb 2025 (used subprocess)

    return ref_dlist_fa_genome, ref_dlist_fa_cdna, ref_dlist_gtf


def download_ensembl_reference_files(reference_out_dir_sequences_dlist, grch="37", ensembl_release="93"):
    grch = str(grch)
    ensembl_release = str(ensembl_release)
    ensembl_grch_gget = "human_grch37" if grch == "37" else grch
    ensembl_release_gtf = "87" if (grch == "37" and ensembl_release == "93") else ensembl_release

    ref_dlist_fa_genome = f"{reference_out_dir_sequences_dlist}/Homo_sapiens.GRCh{grch}.dna.primary_assembly.fa"
    ref_dlist_fa_cdna = f"{reference_out_dir_sequences_dlist}/Homo_sapiens.GRCh{grch}.cdna.all.fa"
    ref_dlist_gtf = f"{reference_out_dir_sequences_dlist}/Homo_sapiens.GRCh{grch}.{ensembl_release_gtf}.gtf"

    files_to_download_list = []
    file_dict = {
        "dna": ref_dlist_fa_genome,
        "cdna": ref_dlist_fa_cdna,
        "gtf": ref_dlist_gtf,
    }

    for file_source, file_path in file_dict.items():
        if not os.path.exists(file_path):
            files_to_download_list.append(file_source)

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
            subprocess.run(["gunzip", f"{file_dict[file]}.gz"], check=True)

    return ref_dlist_fa_genome, ref_dlist_fa_cdna, ref_dlist_gtf


def get_mutation_type_series(mutation_series):
    # Extract mutation type id using the regex pattern
    mutation_type_id = mutation_series.str.extract(mutation_pattern)[1]

    # Define conditions and choices for mutation types
    conditions = [
        mutation_type_id.str.contains(">", na=False),
        mutation_type_id.str.contains("delins", na=False),
        mutation_type_id.str.contains("del", na=False) & ~mutation_type_id.str.contains("delins", na=False),
        mutation_type_id.str.contains("ins", na=False) & ~mutation_type_id.str.contains("delins", na=False),
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


def add_vcrs_mutation_type(mutations_df, var_column="vcrs_header"):
    mutations_df = mutations_df.copy()

    # Split the var_column by ';'
    mutations_df["mutation_list"] = mutations_df[var_column].str.split(";")

    # Explode the mutation_list to get one mutation per row
    mutations_exploded = mutations_df.explode("mutation_list")

    # Apply the vectorized get_mutation_type_series function
    mutations_exploded["vcrs_mutation_type"] = get_mutation_type_series(mutations_exploded["mutation_list"])

    # Reset index to keep track of original rows
    mutations_exploded.reset_index(inplace=True)

    # Group back to the original DataFrame, joining mutation types with ';'
    grouped_mutation_types = mutations_exploded.groupby("index")["vcrs_mutation_type"].apply(";".join)

    # Assign the 'mutation_type' back to mutations_df
    mutations_df["vcrs_mutation_type"] = grouped_mutation_types

    # Split 'mutation_type' by ';' to analyze unique mutation types
    mutations_df["mutation_type_split"] = mutations_df["vcrs_mutation_type"].str.split(";")

    # Calculate the number of unique mutation types
    mutations_df["unique_mutation_count"] = mutations_df["mutation_type_split"].map(set).str.len()

    # Replace 'mutation_type' with the single unique mutation type if unique_mutation_count == 1
    mask_single = mutations_df["unique_mutation_count"] == 1
    mutations_df.loc[mask_single, "vcrs_mutation_type"] = mutations_df.loc[mask_single, "mutation_type_split"].str[0]

    # Replace entries containing ';' with 'mixed'
    mutations_df.loc[mutations_df["vcrs_mutation_type"].str.contains(";"), "vcrs_mutation_type"] = "mixed"

    # Drop helper columns
    mutations_df.drop(
        columns=["mutation_list", "mutation_type_split", "unique_mutation_count"],
        inplace=True,
    )

    mutations_df.loc[mutations_df[var_column].isna(), "vcrs_mutation_type"] = np.nan

    return mutations_df


def add_mutation_type(mutations, var_column):
    mutations["mutation_type_id"] = mutations[var_column].str.extract(mutation_pattern)[1]

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


rnaseq_fastq_filename_pattern_bulk = re.compile(r"([^/]+)_(\d+)\.(fastq|fq)(\.gz)?$")  # eg SRR8615037_1.fastq.gz
rnaseq_fastq_filename_pattern_illumina = re.compile(r"^([\w.-]+)_L\d+_R[12]_\d{3}\.(fastq|fq)(\.gz)?$")  # SAMPLE_LANE_R[12]_001.fastq.gz where SAMPLE is letters, numbers, underscores; LANE is numbers with optional leading 0s; pair is either 1 or 2; and it has .fq or .fastq extension (or .fq.gz or .fastq.gz)


def bulk_sort_order_for_kb_count_fastqs(filepath):
    # Define order for read types
    read_type_order = {"1": 0, "2": 1}

    match = rnaseq_fastq_filename_pattern_bulk.search(filepath)
    if not match:
        raise ValueError(f"Invalid SRA-style FASTQ filename: {filepath}")

    sample_number, read_type = match.groups()

    return (sample_number, read_type_order.get(read_type, 999))


def illumina_sort_order_for_kb_count_fastqs(filepath):
    # Define order for file types
    file_type_order = {"R1": 0, "R2": 1, "I1": 2, "I2": 3}

    # Split the filepath into parts by '/'
    path_parts = filepath.split("/")

    # # Extract the parent folder (2nd to last part)
    # parent_folder = path_parts[-2]

    # Extract the filename (last part of the path)
    filename = path_parts[-1]

    # Split filename by '_' to extract file type and lane information
    parts = filename.split("_")

    # Extract lane number; assuming lane info is of the format 'L00X'
    lane = int(parts[-3][1:4])  # e.g., extracts '001' from 'L001'

    # Get the order value for the file type, e.g., 'R1'
    file_type = parts[-2].split(".")[0]  # e.g., extracts 'R1' from 'R1_001.fastq.gz'

    # Return a tuple to sort by:
    # 1. Numerically by lane
    # 2. Order of file type (R1, R2)
    return (lane, file_type_order.get(file_type, 999))


def sort_fastq_files_for_kb_count(fastq_files, technology=None, multiplexed=None, logger=None, check_only=False, verbose=True):
    printlog = get_printlog(verbose, logger)

    file_name_format = None

    for fastq_file in fastq_files:
        if not fastq_file.endswith(fastq_extensions):  # check for valid extension
            message = f"File {fastq_file} does not have a valid FASTQ extension of one of the following: {fastq_extensions}."
            raise ValueError(message)  # invalid regardless of order

        if bool(rnaseq_fastq_filename_pattern_bulk.match(fastq_file)):
            file_name_format = "bulk"
        elif bool(rnaseq_fastq_filename_pattern_illumina.match(fastq_file)):  # check for Illumina file naming convention
            file_name_format = "illumina"
        else:
            message = f"File {fastq_file} does not match the expected bulk file naming convention of SAMPLE_PAIR.EXT where SAMPLE is sample name, PAIR is 1/2, and EXT is a fastq extension - or the Illumina file naming convention of SAMPLE_LANE_R[12]_001.fastq.gz, where SAMPLE is letters, numbers, underscores; LANE is numbers with optional leading 0s; pair is either R1 or R2; and it has .fq or .fastq extension (or .fq.gz or .fastq.gz)."
            if check_only:
                printlog(message)
            else:
                message += "\nRaising exception and exiting because sort_fastqs=True, which requires standard bulk or Illumina file naming convention. Please check fastq file names or set sort_fastqs=False."
                raise ValueError(message)

    if technology is None:
        printlog("No technology specified, so defaulting to None when checking file order (i.e., will not drop index files from fastq file list)")
    if "smartseq" in technology.lower() and multiplexed is None:
        printlog("Multiplexed not specified with smartseq technology, so defaulting to None when checking file order (i.e., will not drop index files from fastq file list)")
        multiplexed = True

    if technology is None or technology == "10xv1" or ("smartseq" in technology.lower() and multiplexed):  # keep the index I1/I2 files (pass into kb count) for 10xv1 or multiplexed smart-seq
        filtered_files = fastq_files
    else:  # remove the index files
        printlog(f"Removing index files from fastq files list, as they are not utilized in kb count with technology {technology}")
        filtered_files = [f for f in fastq_files if not any(x in os.path.basename(f) for x in ["I1", "I2"])]

    if file_name_format == "illumina":
        sorted_files = sorted(filtered_files, key=illumina_sort_order_for_kb_count_fastqs)
    elif file_name_format == "bulk":
        sorted_files = sorted(filtered_files, key=bulk_sort_order_for_kb_count_fastqs)
    else:
        sorted_files = sorted(filtered_files, key=bulk_sort_order_for_kb_count_fastqs)  # default to bulk

    if check_only:
        if sorted_files == fastq_files:
            printlog("Fastq files are in the expected order")
        else:
            printlog("Fastq files are not in the expected order. Fastq files are expected to be sorted (in order) by (a) SAMPLE, (b) LANE, and (c) PARITY (R1/R2). Index files (I1/I2) are not included in the sort order except for technology=10xv1 and multiplexed smartseq. To enable automatic sorting, set sort_fastqs=True.")
        return fastq_files
    else:
        return sorted_files


def load_in_fastqs(fastqs):
    if len(fastqs) != 1:
        return fastqs
    fastqs = fastqs[0]
    if not os.path.exists(fastqs):
        raise ValueError(f"File/folder {fastqs} does not exist")
    if os.path.isdir(fastqs):
        files = []
        for file in os.listdir(fastqs):  # make fastqs list from fastq files in immediate child directory
            if (os.path.isfile(os.path.join(fastqs, file))) and (any(file.lower().endswith((ext, f"{ext}.zip", f"{ext}.gz")) for ext in fastq_extensions)):
                files.append(file)
        if len(files) == 0:
            raise ValueError(f"No fastq files found in {fastqs}")  # redundant with type-checking below, but prints a different error message (informs that the directory has no fastqs, rather than simply telling the user that no fastqs were provided)
    elif os.path.isfile(fastqs):
        if file.lower().endswith("txt"):  # make fastqs list from items in txt file
            with open(fastqs, "r", encoding="utf-8") as f:
                files = [line.strip() for line in f.readlines()]
            if len(files) == 0:
                raise ValueError(f"No fastq files found in {fastqs}")  # redundant with type-checking below, but prints a different error message (informs that the text file has no fastqs, rather than simply telling the user that no fastqs were provided)
        elif any(fastqs.lower().endswith((ext, f"{ext}.zip", f"{ext}.gz")) for ext in fastq_extensions):
            files = [fastqs]
        else:
            raise ValueError(f"File {fastqs} is not a fastq file, text file, or directory")
    return files
