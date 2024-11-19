import os
import subprocess
import re
import numpy as np
import pandas as pd
from tqdm import tqdm
import varseek
import random

from .utils import (
    set_up_logger,
    introduce_sequencing_errors,
    merge_synthetic_read_info_into_mutations_metadata_df,
    fasta_to_fastq,
)


tqdm.pandas()
logger = set_up_logger()


def sim(
    mutation_metadata_df,
    fastq_output_path=None,
    fastq_parent_path=None,
    read_df_parent=None,
    read_df_out=None,
    mutation_metadata_df_out=None,
    sample_type="all",
    number_of_mutations_to_sample=1500,
    strand=None,
    number_of_reads_per_sample=None,
    number_of_reads_per_sample_m=None,
    number_of_reads_per_sample_w=None,
    read_length=150,
    seed=42,
    k=None,
    w=None,
    add_noise=False,
    error_rate=0.0001,
    max_errors=float("inf"),
    with_replacement=False,
    sequences=None,
    mutation_metadata_df_path=None,
    reference_out_dir=".",
    out_dir_vk_build=".",
    seq_id_column="seq_ID",
    mut_column="mutation",
    gtf=None,
    gtf_transcript_id_column=None,  # "transcript_id",
    sequences_cdna=None,
    seq_id_column_cdna=None,  # "seq_ID",
    mut_column_cdna=None,  # "mutation",
    sequences_genome=None,
    seq_id_column_genome=None,  # "chromosome",
    mut_column_genome=None,  # "mutation_genome",
    filters=None,
    verbose=True,
    **kwargs,
):
    """
    Create synthetic RNA-seq dataset with mutation reads.

    Required input arguments:
    - mutation_metadata_df     (str or pd.DataFrame) Path to the csv file containing the mutation metadata dataframe.
    - fastq_output_path        (str) Path to the output fastq file containing the simulated reads.
    - fastq_parent_path       (str) Path to the parent fastq file containing the simulated reads.
    - read_df_parent           (str or pd.DataFrame) Path to the csv file containing the read dataframe.
    - sample_type              (str) Type of reads to simulate. Possible values: 'm', 'w', 'all'.
    - number_of_mutations_to_sample     (int) Number of mutations to sample.
    - strand                   (str) Strand to simulate reads from. Possible values: 'f', 'r', 'both', or None.
    - number_of_reads_per_sample (int) Number of reads to simulate per mutation. Not used if both number_of_reads_per_sample_m and number_of_reads_per_sample_w are provided.
    - number_of_reads_per_sample_m (int) Number of reads to simulate per mutation for mutant reads. Only used if sample_type is 'm' and number_of_reads_per_sample is None.
    - number_of_reads_per_sample_w (int) Number of reads to simulate per mutation for wild-type reads. Only used if sample_type is 'w' and number_of_reads_per_sample is None.
    - read_length              (int) Length of the reads to simulate.
    - seed                     (int) Seed for random number generation.
    - add_noise                (bool) Whether to add noise to the reads.
    - with_replacement         (bool) Whether to sample with replacement.
    - sequences                (str) Path to the fasta file containing the sequences.
    - reference_out_dir        (str) Path to the output directory for the reference files.
    - out_dir_vk_build         (str) Path to the output directory for the vk_build files.
    - seq_id_column            (str) Name of the column containing the sequence IDs.
    - mut_column               (str) Name of the column containing the mutations.
    - gtf                      (str) Path to the GTF file.
    - gtf_transcript_id_column (str) Name of the column containing the transcript IDs in the GTF file.
    - seq_id_column_cdna       (str) Name of the column containing the sequence IDs for cDNA sequences.
    - mut_column_cdna          (str) Name of the column containing the mutations for cDNA sequences.
    - seq_id_column_genome     (str) Name of the column containing the sequence IDs for genome sequences.
    - mut_column_genome        (str) Name of the column containing the mutations for genome sequences.
    - filters                  (dict) Dictionary containing the filters to apply to the mutation metadata dataframe.
    - **kwargs                 (dict) Additional keyword arguments to pass to varseek.build.
    """
    if number_of_reads_per_sample is None and number_of_reads_per_sample_m is None and number_of_reads_per_sample_w is None:
        number_of_reads_per_sample = "all"
        number_of_reads_per_sample_m = "all"
        number_of_reads_per_sample_w = "all"

    if number_of_reads_per_sample_m is not None and number_of_reads_per_sample_w is not None:
        number_of_reads_per_sample = None

    if number_of_reads_per_sample_m is None and number_of_reads_per_sample_w is None and sample_type == "all":
        number_of_reads_per_sample_m = number_of_reads_per_sample
        number_of_reads_per_sample_w = number_of_reads_per_sample

    if number_of_reads_per_sample_m is None and sample_type == "m":
        number_of_reads_per_sample_m = number_of_reads_per_sample
        number_of_reads_per_sample_w = 0

    if number_of_reads_per_sample_w is None and sample_type == "w":
        number_of_reads_per_sample_w = number_of_reads_per_sample
        number_of_reads_per_sample_m = 0

    if type(mutation_metadata_df) == str and os.path.exists(mutation_metadata_df):
        mutation_metadata_df = pd.read_csv(mutation_metadata_df)

    if (type(mutation_metadata_df) == str and not os.path.exists(mutation_metadata_df)) or "mutant_sequence_read_parent" not in mutation_metadata_df.columns or "wt_sequence_read_parent" not in mutation_metadata_df.columns:  # TODO: debug when a subset of columns is already in df
        print("cannot find mutant sequence read parent")
        update_df_out = f"{out_dir_vk_build}/sim_data_df.csv"

        if k and w:
            assert k > w, "k must be greater than w"
            read_w = read_length - (k - w)
        else:
            read_w = read_length - 1

        if sequences_cdna is not None and sequences_genome is not None:
            update_df_out_cdna = update_df_out.replace(".csv", "_cdna.csv")
            if not os.path.exists(update_df_out_cdna):
                varseek.build(
                    sequences=sequences_cdna,
                    mutations=mutation_metadata_df_path,
                    out=out_dir_vk_build,
                    reference_out=reference_out_dir,
                    w=read_w,
                    remove_seqs_with_wt_kmers=False,
                    optimize_flanking_regions=False,
                    min_seq_len=read_length,
                    cosmic_email=os.getenv("COSMIC_EMAIL"),
                    cosmic_password=os.getenv("COSMIC_PASSWORD"),
                    update_df=True,
                    update_df_out=update_df_out_cdna,
                    seq_id_column=seq_id_column_cdna,
                    mut_column=mut_column_cdna,
                )

            update_df_out_genome = update_df_out.replace(".csv", "_genome.csv")
            if not os.path.exists(update_df_out_genome):
                varseek.build(
                    sequences=sequences_genome,
                    mutations=mutation_metadata_df_path,
                    out=out_dir_vk_build,
                    reference_out=reference_out_dir,
                    w=read_w,
                    remove_seqs_with_wt_kmers=False,
                    optimize_flanking_regions=False,
                    min_seq_len=read_length,
                    cosmic_email=os.getenv("COSMIC_EMAIL"),
                    cosmic_password=os.getenv("COSMIC_PASSWORD"),
                    update_df=True,
                    update_df_out=update_df_out_genome,
                    seq_id_column=seq_id_column_genome,
                    mut_column=mut_column_genome,
                    gtf=gtf,
                    gtf_transcript_id_column=gtf_transcript_id_column,
                )

            # Load the CSV files
            df_cdna = pd.read_csv(update_df_out_cdna)
            df_genome = pd.read_csv(update_df_out_genome)

            # Concatenate vertically (appending rows)
            sim_data_df = pd.concat([df_cdna, df_genome], axis=0)

            # Save the result to a new CSV
            sim_data_df.to_csv(update_df_out, index=False)
        else:
            # TODO: add more support for column names based on whether cDNA or genome specified
            if not os.path.exists(update_df_out):
                print("running varseek build")
                varseek.build(
                    sequences=sequences,
                    mutations=mutation_metadata_df_path,
                    out=out_dir_vk_build,
                    reference_out=reference_out_dir,
                    w=read_w,
                    remove_seqs_with_wt_kmers=False,
                    optimize_flanking_regions=False,
                    min_seq_len=read_length,
                    cosmic_email=os.getenv("COSMIC_EMAIL"),
                    cosmic_password=os.getenv("COSMIC_PASSWORD"),
                    update_df=True,
                    update_df_out=update_df_out,
                    seq_id_column=seq_id_column,  # uncomment for genome support
                    mut_column=mut_column,
                    gtf=gtf,
                    gtf_transcript_id_column=gtf_transcript_id_column,
                )

            sim_data_df = pd.read_csv(update_df_out)

        sim_data_df.rename(
            columns={
                "mutant_sequence": "mutant_sequence_read_parent",
                "wt_sequence": "wt_sequence_read_parent",
            },
            inplace=True,
        )

        sim_data_df["mutant_sequence_read_parent_rc"] = sim_data_df["mutant_sequence_read_parent"].apply(varseek.varseek_build.reverse_complement)
        sim_data_df["mutant_sequence_read_parent_length"] = sim_data_df["mutant_sequence_read_parent"].str.len()

        sim_data_df["wt_sequence_read_parent_rc"] = sim_data_df["wt_sequence_read_parent"].apply(varseek.varseek_build.reverse_complement)
        sim_data_df["wt_sequence_read_parent_length"] = sim_data_df["wt_sequence_read_parent"].str.len()

        mutation_metadata_df = pd.merge(
            mutation_metadata_df,
            sim_data_df[
                [
                    "header",
                    "mutant_sequence_read_parent",
                    "mutant_sequence_read_parent_rc",
                    "mutant_sequence_read_parent_length",
                    "wt_sequence_read_parent",
                    "wt_sequence_read_parent_rc",
                    "wt_sequence_read_parent_length",
                ]
            ],
            on="header",
            how="left",
            suffixes=("", "_read_parent"),
        )

    filters.extend(["mutant_sequence_read_parent-isnotnull", "wt_sequence_read_parent-isnotnull"])
    filters = list(set(filters))

    if filters:
        filtered_df = varseek.filter(
            mutation_metadata_df,
            filters=filters,
            return_df=True,
            output_mcrs_fasta=None,
        )  # filter to include only rows not already in mutation and whatever condition I would like
    else:
        filtered_df = mutation_metadata_df

    if fastq_output_path is None:
        fastq_output_path = "./synthetic_reads.fq"

    fasta_output_path_temp = fastq_output_path.replace(".fq", "_temp.fa")

    if seed is not None:
        random.seed(seed)

    column_and_default_value_list_of_tuples = [
        ("included_in_synthetic_reads", False),
        ("included_in_synthetic_reads_wt", False),
        ("included_in_synthetic_reads_mutant", False),
        ("list_of_read_starting_indices_wt", None),
        ("list_of_read_starting_indices_mutant", None),
        ("number_of_reads_wt", 0),
        ("number_of_reads_mutant", 0),
        ("any_noisy_reads_wt", False),
        ("noisy_read_indices_wt", None),
        ("any_noisy_reads_mutant", False),
        ("noisy_read_indices_mutant", None),
    ]

    for column, default_value in column_and_default_value_list_of_tuples:
        if column not in mutation_metadata_df.columns:
            mutation_metadata_df[column] = default_value

    if number_of_mutations_to_sample == "all":
        sampled_reference_df = filtered_df
    else:
        # Randomly select number_of_mutations_to_sample rows
        number_of_mutations_to_sample = min(number_of_mutations_to_sample, len(filtered_df))
        sampled_reference_df = filtered_df.sample(n=number_of_mutations_to_sample, random_state=seed)

    if sampled_reference_df.empty:
        print("No mutations to sample")
        # return dict with empty values
        return {
            "read_df": pd.DataFrame(),
            "mutation_metadata_df": pd.DataFrame(),
        }

    mutant_list_of_dicts = []
    wt_list_of_dicts = []

    if sample_type == "m":
        sampled_reference_df["included_in_synthetic_reads_mutant"] = True
        new_column_names = [
            "number_of_reads_mutant",
            "list_of_read_starting_indices_mutant",
            "any_noisy_reads_mutant",
            "noisy_read_indices_mutant",
        ]
    elif sample_type == "w":
        sampled_reference_df["included_in_synthetic_reads_wt"] = True
        new_column_names = [
            "number_of_reads_wt",
            "list_of_read_starting_indices_wt",
            "any_noisy_reads_wt",
            "noisy_read_indices_wt",
        ]
    else:
        sampled_reference_df["included_in_synthetic_reads_mutant"] = True
        sampled_reference_df["included_in_synthetic_reads_wt"] = True
        new_column_names = [
            "number_of_reads_mutant",
            "list_of_read_starting_indices_mutant",
            "number_of_reads_wt",
            "list_of_read_starting_indices_wt",
            "any_noisy_reads_mutant",
            "noisy_read_indices_mutant",
            "any_noisy_reads_wt",
            "noisy_read_indices_wt",
        ]

    new_column_dict = {key: [] for key in new_column_names}
    noisy_read_indices_mutant = []
    noisy_read_indices_wt = []

    with_replacement_original = with_replacement

    # Write to a FASTA file
    total_fragments = 0
    skipped = 0
    with open(fasta_output_path_temp, "a") as fa_file:
        for row in sampled_reference_df.itertuples(index=False):
            # try:
            header = row.header
            mcrs_id = row.mcrs_id
            mcrs_header = row.mcrs_header
            mcrs_mutation_type = row.mcrs_mutation_type
            mutant_sequence = row.mutant_sequence_read_parent
            mutant_sequence_rc = row.mutant_sequence_read_parent_rc
            mutant_sequence_length = row.mutant_sequence_read_parent_length
            wt_sequence = row.wt_sequence_read_parent
            wt_sequence_rc = row.wt_sequence_read_parent_rc
            wt_sequence_length = row.wt_sequence_read_parent_length

            valid_starting_index_max_mutant = int(mutant_sequence_length - read_length + 1)
            valid_starting_index_max_wt = int(wt_sequence_length - read_length + 1)

            if number_of_reads_per_sample == "all":  # sample all reads from sample_type (wt and/or mutant)
                read_start_indices_mutant = list(range(valid_starting_index_max_mutant))
                read_start_indices_wt = list(range(valid_starting_index_max_wt))

                number_of_reads_mutant = len(read_start_indices_mutant)
                number_of_reads_wt = len(read_start_indices_wt)

            elif number_of_reads_per_sample is None:  # sample number_of_reads_per_sample_m from mutant (if sample_type != "w") and number_of_reads_per_sample_w from wt (if sample_type != "m")
                number_of_reads_per_sample_m = int(number_of_reads_per_sample_m)
                number_of_reads_per_sample_w = int(number_of_reads_per_sample_w)
                number_of_reads_mutant = min(valid_starting_index_max_mutant, number_of_reads_per_sample_m)
                number_of_reads_wt = min(valid_starting_index_max_wt, number_of_reads_per_sample_w)

                if number_of_reads_per_sample_m > valid_starting_index_max_mutant:
                    logger.info("Setting with_replacement = True for this round")
                    with_replacement = True

                if with_replacement:
                    read_start_indices_mutant = random.choices(
                        range(valid_starting_index_max_mutant),
                        k=min(
                            valid_starting_index_max_mutant,
                            number_of_reads_per_sample_m,
                        ),
                    )
                else:
                    read_start_indices_mutant = random.sample(
                        range(valid_starting_index_max_mutant),
                        min(
                            valid_starting_index_max_mutant,
                            number_of_reads_per_sample_m,
                        ),
                    )

                with_replacement = with_replacement_original

                if number_of_reads_per_sample_w > valid_starting_index_max_wt:
                    logger.info("Setting with_replacement = True for this round")
                    with_replacement = True

                if with_replacement:
                    read_start_indices_wt = random.choices(
                        range(valid_starting_index_max_wt),
                        k=min(valid_starting_index_max_wt, number_of_reads_per_sample_w),
                    )
                else:
                    read_start_indices_wt = random.sample(
                        range(valid_starting_index_max_wt),
                        min(valid_starting_index_max_wt, number_of_reads_per_sample_w),
                    )

                with_replacement = with_replacement_original

            else:  # sample number_of_reads_per_sample (int) from sample_type (wt and/or mutant), and in the same locations if sample_type == "all"
                valid_starting_index_max = min(valid_starting_index_max_mutant, valid_starting_index_max_wt)
                number_of_reads_per_sample = int(number_of_reads_per_sample)
                number_of_reads = min(valid_starting_index_max, number_of_reads_per_sample)

                if number_of_reads_per_sample > valid_starting_index_max:
                    logger.info("Setting with_replacement = True for this round")
                    with_replacement = True

                if with_replacement:
                    read_start_indices_mutant = random.choices(
                        range(valid_starting_index_max),
                        k=min(valid_starting_index_max, number_of_reads_per_sample),
                    )
                else:
                    read_start_indices_mutant = random.sample(
                        range(valid_starting_index_max),
                        min(valid_starting_index_max, number_of_reads_per_sample),
                    )

                with_replacement = with_replacement_original

                read_start_indices_wt = read_start_indices_mutant

                number_of_reads_mutant = number_of_reads
                number_of_reads_wt = number_of_reads

            if strand == False or strand is None:
                mutant_sequence_list = [random.choice([(mutant_sequence, "f"), (mutant_sequence_rc, "r")])]
                wt_sequence_list = [random.choice([(wt_sequence, "f"), (wt_sequence_rc, "r")])]
            elif strand[0] == "f":
                mutant_sequence_list = [(mutant_sequence, "f")]
                wt_sequence_list = [(wt_sequence, "f")]
            elif strand[0] == "r":
                mutant_sequence_list = [(mutant_sequence_rc, "r")]
                wt_sequence_list = [(wt_sequence_rc, "r")]
            elif strand == "both" or strand == True:
                mutant_sequence_list = [
                    (mutant_sequence, "f"),
                    (mutant_sequence_rc, "r"),
                ]
                wt_sequence_list = [(wt_sequence, "f"), (wt_sequence_rc, "r")]

            # Loop through each 150mer of the sequence
            if sample_type != "w":
                if number_of_reads_per_sample == "all" and (strand == "both" or strand == True):
                    number_of_reads_mutant = number_of_reads_mutant * 2  # since now both strands are being sampled

                new_column_dict["number_of_reads_mutant"].append(number_of_reads_mutant)
                new_column_dict["list_of_read_starting_indices_mutant"].append(read_start_indices_mutant)

                for selected_sequence, selected_strand in mutant_sequence_list:
                    for i in read_start_indices_mutant:
                        sequence_chunk = selected_sequence[i : i + read_length]
                        noise_str = ""
                        if add_noise:
                            sequence_chunk_old = sequence_chunk
                            sequence_chunk = introduce_sequencing_errors(
                                sequence_chunk,
                                error_rate=error_rate,
                                max_errors=max_errors,
                            )
                            if sequence_chunk != sequence_chunk_old:
                                noise_str = "n"
                                noisy_read_indices_mutant.append(i)

                        read_id = f"{mcrs_id}_{i}{selected_strand}M{noise_str}"
                        read_header = f"{header}_{i}{selected_strand}M{noise_str}"
                        fa_file.write(f">{read_id}\n{sequence_chunk}\n")
                        mutant_dict = {
                            "read_id": read_id,
                            "read_header": read_header,
                            "read_sequence": sequence_chunk,
                            "read_index": i,
                            "read_strand": selected_strand,
                            "reference_header": header,
                            "mcrs_id": mcrs_id,
                            "mcrs_header": mcrs_header,
                            "mcrs_mutation_type": mcrs_mutation_type,
                            "mutant_read": True,
                            "wt_read": False,
                            "region_included_in_mcrs_reference": True,
                            "noise_added": bool(noise_str),
                        }
                        mutant_list_of_dicts.append(mutant_dict)
                        total_fragments += 1

                new_column_dict["any_noisy_reads_mutant"].append(bool(noisy_read_indices_mutant))
                new_column_dict["noisy_read_indices_mutant"].append(noisy_read_indices_mutant)
                noisy_read_indices_mutant = []

            if sample_type != "m":
                if number_of_reads_per_sample == "all" and (strand == "both" or strand == True):
                    number_of_reads_wt = number_of_reads_wt * 2  # since now both strands are being sampled
                new_column_dict["number_of_reads_wt"].append(number_of_reads_wt)
                new_column_dict["list_of_read_starting_indices_wt"].append(read_start_indices_wt)
                for selected_sequence, selected_strand in wt_sequence_list:
                    for i in read_start_indices_wt:
                        sequence_chunk = selected_sequence[i : i + read_length]
                        noise_str = ""
                        if add_noise:
                            sequence_chunk_old = sequence_chunk
                            sequence_chunk = introduce_sequencing_errors(
                                sequence_chunk,
                                error_rate=error_rate,
                                max_errors=max_errors,
                            )
                            if sequence_chunk != sequence_chunk_old:
                                noise_str = "n"
                                noisy_read_indices_wt.append(i)

                        read_id = f"{mcrs_id}_{i}{selected_strand}W{noise_str}"
                        read_header = f"{header}_{i}{selected_strand}W{noise_str}"
                        fa_file.write(f">{read_id}\n{sequence_chunk}\n")
                        wt_dict = {
                            "read_id": read_id,
                            "read_header": read_header,
                            "read_sequence": sequence_chunk,
                            "read_index": i,
                            "read_strand": selected_strand,
                            "reference_header": header,
                            "mcrs_id": mcrs_id,
                            "mcrs_header": mcrs_header,
                            "mcrs_mutation_type": mcrs_mutation_type,
                            "mutant_read": False,
                            "wt_read": True,
                            "region_included_in_mcrs_reference": True,
                            "noise_added": bool(noise_str),
                        }
                        wt_list_of_dicts.append(wt_dict)
                        total_fragments += 1

                new_column_dict["noisy_read_indices_wt"].append(noisy_read_indices_wt)
                new_column_dict["any_noisy_reads_wt"].append(bool(noisy_read_indices_wt))
                noisy_read_indices_wt = []
            # except Exception as e:
            #     skipped += 1

    if skipped > 0:
        print(f"Skipped {skipped} mutations due to errors")

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

    fasta_to_fastq(fasta_output_path_temp, fastq_output_path, add_noise=add_noise)

    # Read the contents of the files first
    if fastq_parent_path and os.path.exists(fastq_parent_path) and os.path.getsize(fastq_parent_path) != 0:
        with open(fastq_output_path, "r") as new_file:
            file_content_new = new_file.read()

        # Now write both contents to read_fa_path
        with open(fastq_parent_path, "a") as parent_file:
            parent_file.write(file_content_new)
    else:
        fastq_parent_path = fastq_output_path

    if read_df_parent is not None:
        if type(read_df_parent) == str:
            read_df_parent = pd.read_csv(read_df_parent)
        read_df = pd.concat([read_df_parent, read_df], ignore_index=True)

    mutation_metadata_df = merge_synthetic_read_info_into_mutations_metadata_df(mutation_metadata_df, sampled_reference_df, sample_type=sample_type)

    mutation_metadata_df["tumor_purity"] = mutation_metadata_df["number_of_reads_mutant"] / mutation_metadata_df["number_of_reads_wt"]

    mutation_metadata_df["tumor_purity"] = np.where(
        np.isnan(mutation_metadata_df["tumor_purity"]),
        np.nan,  # Keep NaN as NaN
        mutation_metadata_df["tumor_purity"],  # Keep the result for valid divisions
    )

    os.remove(fasta_output_path_temp)

    print(f"Wrote {total_fragments} mutations to {fastq_output_path}")

    simulated_df_dict = {
        "read_df": read_df,
        "mutation_metadata_df": mutation_metadata_df,
    }

    if read_df_out is not None:
        read_df.to_csv(read_df_out, index=False)

    if mutation_metadata_df_out is not None:
        mutation_metadata_df.to_csv(mutation_metadata_df_out, index=False)

    return simulated_df_dict
