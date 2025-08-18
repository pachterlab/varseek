import os
import random

import numpy as np
import pandas as pd
import pyfastx
from tqdm import tqdm
import logging

from varseek.utils.logger_utils import splitext_custom, set_up_logger
from varseek.utils.seq_utils import (
    add_mutation_information,
    fasta_to_fastq,
    reverse_complement,
)

tqdm.pandas()

logger = logging.getLogger(__name__)
logger = set_up_logger(logger, logging_level="INFO", save_logs=False, log_dir=None)


def merge_synthetic_read_info_into_variants_metadata_df(mutation_metadata_df, sampled_reference_df, sample_type="all", header_column="header"):
    columns_to_merge_mutant = ["included_in_synthetic_reads_mutant", "number_of_reads_mutant", "list_of_read_starting_indices_mutant", "any_noisy_reads_mutant", "noisy_read_indices_mutant"]
    columns_to_merge_wt = ["included_in_synthetic_reads_wt", "number_of_reads_wt", "list_of_read_starting_indices_wt", "any_noisy_reads_wt", "noisy_read_indices_wt"]

    columns_to_merge = [header_column]
    if sample_type == "m":
        columns_to_merge += columns_to_merge_mutant
    elif sample_type == "w":
        columns_to_merge += columns_to_merge_wt
    elif sample_type == "all":
        columns_to_merge += columns_to_merge_mutant + columns_to_merge_wt
    else:
        raise ValueError(f"Invalid sample_type: {sample_type}. Expected 'm', 'w', or 'all'.")
    
    mutation_metadata_df_new = mutation_metadata_df.merge(
        sampled_reference_df[columns_to_merge],
        on=header_column,
        how="left",
        suffixes=("", "_new"),
    )
    
    if sample_type != "m":
        mutation_metadata_df_new["included_in_synthetic_reads_wt"] = mutation_metadata_df_new["included_in_synthetic_reads_wt"] | mutation_metadata_df_new["included_in_synthetic_reads_wt_new"]

        mutation_metadata_df_new["any_noisy_reads_wt"] = mutation_metadata_df_new["any_noisy_reads_wt"] | mutation_metadata_df_new["any_noisy_reads_wt_new"]

        mutation_metadata_df_new["number_of_reads_wt"] = np.where(
            (mutation_metadata_df_new["number_of_reads_wt"] == 0) | (mutation_metadata_df_new["number_of_reads_wt"].isna()),
            mutation_metadata_df_new["number_of_reads_wt_new"],
            mutation_metadata_df_new["number_of_reads_wt"],
        )

        mutation_metadata_df_new["list_of_read_starting_indices_wt"] = np.where(
            pd.isna(mutation_metadata_df_new["list_of_read_starting_indices_wt"]),
            mutation_metadata_df_new["list_of_read_starting_indices_wt_new"],
            mutation_metadata_df_new["list_of_read_starting_indices_wt"],
        )

        mutation_metadata_df_new["noisy_read_indices_wt"] = np.where(
            pd.isna(mutation_metadata_df_new["noisy_read_indices_wt"]),
            mutation_metadata_df_new["noisy_read_indices_wt_new"],
            mutation_metadata_df_new["noisy_read_indices_wt"],
        )

        mutation_metadata_df_new = mutation_metadata_df_new.drop(
            columns=[
                "included_in_synthetic_reads_wt_new",
                "number_of_reads_wt_new",
                "list_of_read_starting_indices_wt_new",
                "any_noisy_reads_wt_new",
                "noisy_read_indices_wt_new",
            ]
        )

    if sample_type != "w":
        mutation_metadata_df_new["included_in_synthetic_reads_mutant"] = mutation_metadata_df_new["included_in_synthetic_reads_mutant"] | mutation_metadata_df_new["included_in_synthetic_reads_mutant_new"]

        mutation_metadata_df_new["any_noisy_reads_mutant"] = mutation_metadata_df_new["any_noisy_reads_mutant"] | mutation_metadata_df_new["any_noisy_reads_mutant_new"]

        mutation_metadata_df_new["number_of_reads_mutant"] = np.where(
            (mutation_metadata_df_new["number_of_reads_mutant"] == 0) | (mutation_metadata_df_new["number_of_reads_mutant"].isna()),
            mutation_metadata_df_new["number_of_reads_mutant_new"],
            mutation_metadata_df_new["number_of_reads_mutant"],
        )

        mutation_metadata_df_new["list_of_read_starting_indices_mutant"] = np.where(
            pd.isna(mutation_metadata_df_new["list_of_read_starting_indices_mutant"]),
            mutation_metadata_df_new["list_of_read_starting_indices_mutant_new"],
            mutation_metadata_df_new["list_of_read_starting_indices_mutant"],
        )

        mutation_metadata_df_new["noisy_read_indices_mutant"] = np.where(
            pd.isna(mutation_metadata_df_new["noisy_read_indices_mutant"]),
            mutation_metadata_df_new["noisy_read_indices_mutant_new"],
            mutation_metadata_df_new["noisy_read_indices_mutant"],
        )

        mutation_metadata_df_new = mutation_metadata_df_new.drop(
            columns=[
                "included_in_synthetic_reads_mutant_new",
                "number_of_reads_mutant_new",
                "list_of_read_starting_indices_mutant_new",
                "any_noisy_reads_mutant_new",
                "noisy_read_indices_mutant_new",
            ]
        )

    mutation_metadata_df_new["included_in_synthetic_reads"] = mutation_metadata_df_new["included_in_synthetic_reads_mutant"] | mutation_metadata_df_new["included_in_synthetic_reads_wt"]
    mutation_metadata_df_new["any_noisy_reads"] = mutation_metadata_df_new["any_noisy_reads_mutant"] | mutation_metadata_df_new["any_noisy_reads_wt"]

    return mutation_metadata_df_new


def is_in_ranges(num, ranges):
    if not ranges:
        return False
    for start, end in ranges:
        if start <= num <= end:
            return True
    return False


def append_row(read_df, id_value, header_value, sequence_value, start_position, strand, added_noise=False):
    # Create a new row where 'header' and 'seq_ID' are populated, and others are NaN
    new_row = pd.Series(
        {
            "read_id": id_value,
            "read_header": header_value,
            "read_sequence": sequence_value,
            "read_index": start_position,
            "read_strand": strand,
            "reference_header": None,
            "vcrs_id": None,
            "vcrs_header": None,
            "vcrs_variant_type": None,
            "mutant_read": False,
            "wt_read": True,
            "region_included_in_vcrs_reference": False,
            "noise_added": added_noise,
            # All other columns will be NaN automatically
        }
    )

    return pd.concat([read_df, pd.DataFrame([new_row])], ignore_index=True)  # concat returns a new df, and does NOT modify the original df in-place   # old (gives warning): return read_df.append(new_row, ignore_index=True)


def introduce_sequencing_errors(sequence, error_rate=0.0001, error_distribution=(0.85, 0.1, 0.05), max_errors=float("inf"), seed=None):  # Illumina error rate is around 0.01% (1 in 10,000); error_distribution is (sub, del, ins)
    # Define the possible bases
    bases = ["A", "T", "C", "G"]
    new_sequence = []
    number_errors = 0

    error_distribution_sub = error_distribution[0]
    error_distribution_del = error_distribution[1]
    error_distribution_ins = error_distribution[2]

    if seed:
        random.seed(seed)

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


def build_random_genome_read_df(
    reference_fasta_file_path,
    mutation_metadata_df=None,
    seq_id_column="seq_ID",
    var_column="mutation",
    input_type="transcriptome",
    read_df=None,
    read_df_out=None,
    fastq_output_path="random_reads.fq",
    fastq_parent_path=None,
    n=10,
    read_length=150,
    strand=None,
    add_noise_sequencing_error=False,
    add_noise_base_quality=False,
    error_rate=0.0001,
    error_distribution=(0.85, 0.1, 0.05),  # sub, del, ins
    max_errors=float("inf"),
    seed=42,
):
    if input_type == "cdna":
        input_type = "transcriptome"  # for backwards compatibility
    if input_type not in ["genome", "transcriptome"]:
        raise ValueError(f"Invalid input_type: {input_type}. Expected 'genome' or 'transcriptome'.")
    if mutation_metadata_df is not None:
        if f"start_variant_position_{input_type}" not in mutation_metadata_df.columns or f"end_variant_position_{input_type}" not in mutation_metadata_df.columns:
            add_mutation_information(mutation_metadata_df, mutation_column=var_column, variant_source=input_type)
        mutation_metadata_df[f"start_position_for_which_read_contains_mutation_{input_type}"] = mutation_metadata_df[f"start_variant_position_{input_type}"] - read_length + 1

    # Collect all headers and sequences from the FASTA file
    fastq_output_path_base, fastq_output_path_ext = splitext_custom(fastq_output_path)
    fasta_output_path_temp = fastq_output_path_base + "_temp.fa"

    fasta_entries = list(pyfastx.Fastx(reference_fasta_file_path))
    if read_df is None:
        column_names = ["read_id", "read_header", "read_sequence", "reference_header", "vcrs_header", "mutant_read", "wt_read", "region_included_in_vcrs_reference", "noise_added"]
        read_df = pd.DataFrame(columns=column_names)

    fasta_entry_column = seq_id_column
    vcrs_start_column = f"start_position_for_which_read_contains_mutation_{input_type}"
    vcrs_end_column = f"end_variant_position_{input_type}"

    if seed:
        random.seed(seed)

    i = 0
    num_loops = 0
    with open(fasta_output_path_temp, "a", encoding="utf-8") as fa_file:
        while i < n:
            # Choose a random entry (header, sequence) from the FASTA file
            random_transcript, random_sequence = random.choice(fasta_entries)

            len_random_sequence = len(random_sequence)

            if len_random_sequence < read_length:
                continue

            random_transcript = random_transcript.split()[0]  # grab ENST from long transcript name string
            if input_type == "transcriptome":
                random_transcript = random_transcript.split(".")[0]  # strip version number from ENST

            # Choose a random integer between 1 and the sequence_length-read_length as start position
            start_position = random.randint(0, len_random_sequence - read_length)  # positions are 0-index

            if mutation_metadata_df is not None:
                filtered_mutation_metadata_df = mutation_metadata_df.loc[mutation_metadata_df[fasta_entry_column] == random_transcript]

                ranges = list(
                    zip(
                        filtered_mutation_metadata_df[vcrs_start_column],
                        filtered_mutation_metadata_df[vcrs_end_column],
                    )
                )  # if a mutation spans from positions 950-955 and read length=150, then a random sequence between 801-955 will contain the mutation, and thus should be the range of exclusion here
            else:
                ranges = None

            if not is_in_ranges(start_position, ranges):
                end_position = start_position + read_length  # positions are still 0-index
                if strand is None:
                    selected_strand = random.choice(["f", "r"])
                else:
                    selected_strand = strand

                random_sequence = random_sequence[start_position:end_position]  # positions are 0-index
                start_position += 1  # positions are now 1-index
                end_position += 1

                if selected_strand == "r":
                    # start_position, end_position = len(random_sequence) - end_position, len(random_sequence) - start_position  # I am keeping adding the "f/r" in header so I don't need this
                    random_sequence = reverse_complement(random_sequence)  # I slice the sequence first and then take the rc

                noise_str = ""
                if add_noise_sequencing_error:
                    random_sequence_old = random_sequence
                    random_sequence = introduce_sequencing_errors(
                        random_sequence,
                        error_rate=error_rate,
                        error_distribution=error_distribution,
                        max_errors=max_errors,
                    )  # no need to pass seed here since it's already set
                    if random_sequence != random_sequence_old:
                        noise_str = "n"

                wt_id = f"wt_{input_type}_random{selected_strand}W{noise_str}_{i}"
                header = f"{random_transcript}:{start_position}_{end_position}_random{selected_strand}W{noise_str}_{i}"
                read_df = append_row(read_df, wt_id, header, random_sequence, start_position, selected_strand, added_noise=bool(noise_str))

                fa_file.write(f">{header}\n{random_sequence}\n")

                i += 1

            num_loops += 1
            if num_loops > n * 100:
                print(f"Exiting after only {i} mutations added due to long while loop")
                break

    fasta_to_fastq(fasta_output_path_temp, fastq_output_path, add_noise=add_noise_base_quality)  # no need to pass seed here since it's already set

    os.remove(fasta_output_path_temp)

    if fastq_parent_path:
        if not os.path.exists(fastq_parent_path) or os.path.getsize(fastq_parent_path) == 0:
            # write to a new file
            write_mode = "w"
        else:
            write_mode = "a"
        with open(fastq_output_path, "r", encoding="utf-8") as new_file:
            file_content_new = new_file.read()

        # Now write both contents to read_fa_path
        with open(fastq_parent_path, write_mode, encoding="utf-8") as parent_file:
            parent_file.write(file_content_new)

    if read_df_out is not None:
        read_df.to_csv(read_df_out, index=False)

    return read_df
