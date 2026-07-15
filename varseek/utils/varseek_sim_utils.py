import ast
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


# ---------------------------------------------------------------------------
# Self-contained variant dataframe filtering used by vk sim.
# Ported from the (now removed) vk filter module so that sim has no dependency
# on varseek.filter. Supports the same "COLUMN:RULE[=VALUE]" filter strings.
# ---------------------------------------------------------------------------
all_possible_filter_rules = {"greater_than", "greater_or_equal", "less_than", "less_or_equal", "between_inclusive", "between_exclusive", "top_percent", "bottom_percent", "equal", "not_equal", "is_in", "is_not_in", "is_true", "is_false", "is_not_true", "is_not_false", "is_null", "is_not_null"}
filter_rules_that_expect_single_numeric_value = {"greater_than", "greater_or_equal", "less_than", "less_or_equal", "top_percent", "bottom_percent"}
filter_rules_that_expect_comma_separated_pair_of_numerics_value = {"between_inclusive", "between_exclusive"}
filter_rules_that_expect_string_value = {"equal", "not_equal"}
filter_rules_that_expect_text_file_or_list_value = {"is_in", "is_not_in"}
filter_rules_that_expect_no_value = {"is_true", "is_false", "is_not_true", "is_not_false", "is_null", "is_not_null"}


def convert_txt_to_list(txt_path):
    with open(txt_path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def prepare_filters_list(filters):
    """Parse a list of ``COLUMN:RULE[=VALUE]`` strings into filter dicts."""
    filter_list = []

    if isinstance(filters, str) and filters.endswith(".txt"):
        filters = convert_txt_to_list(filters)
    elif isinstance(filters, str) and not filters.endswith(".txt"):
        filters = [filters]

    for f in filters:
        f_split_by_equal = f.split("=")
        col_rule = f_split_by_equal[0]

        if col_rule.count(":") != 1:
            raise ValueError(f"Filter format invalid: {f}. Missing colon. Expected 'COLUMN:RULE' or 'COLUMN:RULE=VALUE'")

        column, rule = col_rule.split(":")

        if rule not in all_possible_filter_rules:
            raise ValueError(f"Filter format invalid: {f}. Invalid rule: {rule}.")

        if f.count("=") == 0:
            if rule not in filter_rules_that_expect_no_value:
                raise ValueError(f"Filter format invalid: {f}. Requires a VALUE for rule {rule}. Expected 'COLUMN:RULE=VALUE'")
            value = None
        elif f.count("=") == 1:
            if rule not in filter_rules_that_expect_single_numeric_value and rule not in filter_rules_that_expect_comma_separated_pair_of_numerics_value and rule not in filter_rules_that_expect_string_value and rule not in filter_rules_that_expect_text_file_or_list_value:
                raise ValueError(f"Filter format invalid: {f}. Requires no VALUE for rule {rule}. Expected 'COLUMN:RULE'")
            value = f_split_by_equal[1]
        else:
            raise ValueError(f"Filter format invalid: {f}. Too many '='s. Expected 'COLUMN:RULE' or 'COLUMN:RULE=VALUE'")

        if rule in filter_rules_that_expect_single_numeric_value:
            try:
                value = float(value)
            except ValueError as exc:
                raise ValueError(f"Filter format invalid: {f}. Expected single numeric value for rule {rule}. 'COLUMN:RULE=VALUE'") from exc
        elif rule in filter_rules_that_expect_comma_separated_pair_of_numerics_value:
            try:
                value_min, value_max = value.split(",")
                value_min, value_max = float(value_min), float(value_max)
            except ValueError as exc:
                raise ValueError(f"Filter format invalid: {f}. Expected a pair of comma-separated numeric values for rule {rule}. 'COLUMN:RULE=VALUE'") from exc
        elif rule in filter_rules_that_expect_string_value:
            pass
        elif rule in filter_rules_that_expect_text_file_or_list_value:
            if value.endswith(".txt"):
                pass
            elif (value[0] == "[" and value[-1] == "]") or (value[0] == "{" and value[-1] == "}") or (value[0] == "(" and value[-1] == ")"):
                pass
            elif isinstance(value, (list, set, tuple)):
                pass
            else:
                raise ValueError(f"Filter format invalid: {f}. Expected a text file path or list for rule {rule}. 'COLUMN:RULE=VALUE'")
        elif rule in filter_rules_that_expect_no_value:
            pass
        else:
            raise ValueError(f"Filter format invalid: {f}. Invalid rule: {rule}.")

        if rule in {"is_true", "is_not_true"}:
            value = True
        if rule in {"is_false", "is_not_false"}:
            value = False

        filter_list.append({"column": column, "rule": rule, "value": value})

    return filter_list


def apply_filters_to_df(df, filters):
    """Apply ``COLUMN:RULE[=VALUE]`` filter strings to ``df`` and return the result.

    Rows whose filter column is null are retained for numeric/range rules (matching
    the original vk filter semantics). Filters referencing an absent column are skipped.
    """
    for individual_filter in prepare_filters_list(filters):
        column = individual_filter["column"]
        rule = individual_filter["rule"]
        value = individual_filter["value"]

        if column not in df.columns:
            continue

        logger.info(f"{column} {rule} {value}")

        if rule == "greater_than":
            df = df.loc[(df[column].astype(float) > float(value)) | (df[column].isnull())]
        elif rule == "greater_or_equal":
            df = df.loc[(df[column].astype(float) >= float(value)) | (df[column].isnull())]
        elif rule == "less_than":
            df = df.loc[(df[column].astype(float) < float(value)) | (df[column].isnull())]
        elif rule == "less_or_equal":
            df = df.loc[(df[column].astype(float) <= float(value)) | (df[column].isnull())]
        elif rule == "between_inclusive":
            value_min, value_max = value.split(",")
            value_min, value_max = float(value_min), float(value_max)
            if value_min >= value_max:
                raise ValueError(f"Invalid range: {value}. Minimum value must be less than maximum value.")
            df = df.loc[((df[column] >= value_min) & (df[column] <= value_max) | (df[column].isnull()))]
        elif rule == "between_exclusive":
            value_min, value_max = value.split(",")
            value_min, value_max = float(value_min), float(value_max)
            if value_min >= value_max:
                raise ValueError(f"Invalid range: {value}. Minimum value must be less than maximum value.")
            df = df.loc[((df[column] > value_min) & (df[column] < value_max) | (df[column].isnull()))]
        elif rule == "top_percent":
            percent_value = df[column].quantile((100 - float(value)) / 100)
            df = df.loc[(df[column].isnull()) | (df[column] >= percent_value)]
        elif rule == "bottom_percent":
            percent_value = df[column].quantile(float(value) / 100)
            df = df.loc[(df[column].isnull()) | (df[column] <= percent_value)]
        elif rule == "equal":
            df = df.loc[df[column].astype(str) == str(value)]
        elif rule == "not_equal":
            df = df.loc[df[column].astype(str) != str(value)]
        elif rule in {"is_in", "is_not_in"}:
            if isinstance(value, str) and value.endswith(".txt"):
                value = set(convert_txt_to_list(value))
            else:
                try:
                    value = ast.literal_eval(value)
                    if not isinstance(value, (set, list, tuple)):
                        raise ValueError("Value must be a set, list, tuple, or path to text file")
                except ValueError as exc:
                    raise ValueError("Value must be a set, list, tuple, or path to text file") from exc
            if rule == "is_in":
                df = df.loc[df[column].isin(set(value))]
            else:
                df = df.loc[~df[column].isin(set(value))]
        elif rule == "is_null":
            df = df.loc[df[column].isnull()]
        elif rule == "is_not_null":
            df = df.loc[df[column].notnull()]
        elif rule == "is_true":
            df = df.loc[df[column] == True]  # noqa: E712
        elif rule == "is_false":
            df = df.loc[df[column] == False]  # noqa: E712
        elif rule == "is_not_true":
            df = df.loc[(df[column] != True)]  # noqa: E712
        elif rule == "is_not_false":
            df = df.loc[(df[column] != False)]  # noqa: E712
        else:
            raise ValueError(f"Rule '{rule}' not recognized")

    return df


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


def assign_strands(read_start_indices_mutant, strand, seed=None):
    if strand in ("f", "r"):
        return [(idx, strand) for idx in read_start_indices_mutant]
    elif strand == "random":
        if seed:
            random.seed(seed)
        return [(idx, random.choice(["f", "r"])) for idx in read_start_indices_mutant]
    elif strand == "both":
        half = len(read_start_indices_mutant) // 2
        return [(idx, "f") for idx in read_start_indices_mutant[:half]] + [(idx, "r") for idx in read_start_indices_mutant[half:]]
    else:
        raise ValueError("strand must be 'f', 'r', 'random', or 'both'")
