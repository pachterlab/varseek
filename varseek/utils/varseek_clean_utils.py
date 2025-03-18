import os
import re
import subprocess
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import pyfastx
import scipy.sparse as sp
from scipy.sparse import csr_matrix
import json
import logging
from tqdm import tqdm

from varseek.constants import (
    complement,
    fastq_extensions,
    technology_barcode_and_umi_dict,
)
from varseek.utils.seq_utils import (
    add_variant_type,
    add_vcrs_variant_type,
    create_header_to_sequence_ordered_dict_from_fasta_WITHOUT_semicolon_splitting,
    get_header_set_from_fastq,
    make_mapping_dict,
    safe_literal_eval,
    sort_fastq_files_for_kb_count,
    load_in_fastqs
)
from varseek.utils.logger_utils import set_up_logger

logger = logging.getLogger(__name__)
logger = set_up_logger(logger, logging_level="INFO", save_logs=False, log_dir=None)

tqdm.pandas()


def run_kb_count_dry_run(index, t2g, fastq, kb_count_out, newer_kallisto, k=31, threads=1):
    # if not os.path.exists(newer_kallisto):  # uncommented because the newest release of kb has the correct kallisto version
    #     kallisto_install_from_source_commands = "git clone https://github.com/pachterlab/kallisto.git && cd kallisto && git checkout 0397342 && mkdir build && cd build && cmake .. -DMAX_KMER_SIZE=64 && make"
    #     subprocess.run(kallisto_install_from_source_commands, shell=True, check=True)

    kb_count_dry_run = ["kb", "count", "-t", str(threads), "-i", index, "-g", t2g, "-x", "bulk", "-k", str(k), "--dry-run", "--parity", "single", "-o", kb_count_out, fastq]  # should be the same as the kb count run before with the exception of removing --h5ad, swapping in the newer kallisto for the kallisto bus command, and adding --union and --dfk-onlist  # TODO: add support for more kb arguments
    if "--h5ad" in kb_count_dry_run:
        kb_count_dry_run.remove("--h5ad")  # not supported

    result = subprocess.run(kb_count_dry_run, stdout=subprocess.PIPE, text=True, check=True)  # used to be shell (changed Feb 2025)
    commands = result.stdout.strip().split("\n")

    for cmd in commands:
        # print(f"Running command: {cmd}")
        cmd_split = cmd.split()
        if "kallisto bus" in cmd:
            cmd_split[0] = newer_kallisto
            cmd_split.insert(2, "--union")
            cmd_split.insert(3, "--dfk-onlist")
        result = subprocess.run(cmd_split, check=True)
        if result.returncode != 0:
            print(f"Command failed: {cmd}")
            break


def create_umi_to_barcode_dict(bus_file, bustools="bustools", barcode_length=16, key_to_use="umi"):
    umi_to_barcode_dict = {}

    # Define the command
    # bustools text -p -a -f -d output.bus
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
    result = subprocess.run(command, stdout=subprocess.PIPE, text=True, check=True)

    # Loop through each line of the output (excluding the last line 'Read in X BUS records')
    for line in result.stdout.strip().split("\n"):
        # Split the line into columns (assuming it's tab or space-separated)
        columns = line.split("\t")  # If columns are space-separated, use .split()
        if key_to_use == "umi":
            umi = columns[2]
        elif key_to_use == "fastq_header_position":
            umi = columns[5]
        else:
            raise ValueError("key_to_use must be either 'umi' or 'fastq_header_position'")
        barcode = columns[0]  # remember there will be A's for padding to 32 characters
        barcode = barcode[(32 - barcode_length) :]  # * remove the padding
        umi_to_barcode_dict[umi] = barcode

    return umi_to_barcode_dict


def check_if_read_dlisted_by_one_of_its_respective_dlist_sequences(vcrs_header, vcrs_header_to_seq_dict, dlist_header_to_seq_dict, k):
    # do a bowtie (or manual) alignment of breaking the vcrs seq into k-mers and aligning to the dlist seqs dervied from the same vcrs header
    dlist_header_to_seq_dict_filtered = {key: value for key, value in dlist_header_to_seq_dict.items() if vcrs_header == key.rsplit("_", 1)[0]}
    vcrs_sequence = vcrs_header_to_seq_dict[vcrs_header]
    for i in range(len(vcrs_sequence) - k + 1):
        kmer = vcrs_sequence[i : (i + k)]
        for dlist_sequence in dlist_header_to_seq_dict_filtered.values():
            if kmer in dlist_sequence:
                return True
    return False


def increment_adata_based_on_dlist_fns(adata, vcrs_fasta, dlist_fasta, kb_count_out, index, t2g, fastq, newer_kallisto, k=31, mm=False, technology="bulk", bustools="bustools", ignore_barcodes=False):
    run_kb_count_dry_run(
        index=index,
        t2g=t2g,
        fastq=fastq,
        kb_count_out=kb_count_out,
        newer_kallisto=newer_kallisto,
        k=k,
        threads=1,
    )

    if not os.path.exists(f"{kb_count_out}/bus_df.csv"):
        bus_df = make_bus_df(kb_count_out, fastq, t2g_file=t2g, mm=mm, union=False, technology=technology, bustools=bustools, ignore_barcodes=ignore_barcodes)
    else:
        bus_df = pd.read_csv(f"{kb_count_out}/bus_df.csv")

    # with open(f"{kb_count_out}/transcripts.txt", encoding="utf-8") as f:
    #     dlist_index = str(sum(1 for line in file))

    n_rows, n_cols = adata.X.shape
    increment_matrix = csr_matrix((n_rows, n_cols))

    vcrs_header_to_seq_dict = create_header_to_sequence_ordered_dict_from_fasta_WITHOUT_semicolon_splitting(vcrs_fasta)
    dlist_header_to_seq_dict = create_header_to_sequence_ordered_dict_from_fasta_WITHOUT_semicolon_splitting(dlist_fasta)
    var_names_to_idx_in_adata_dict = {name: idx for idx, name in enumerate(adata.var_names)}

    # Apply to the whole column at once
    bus_df["gene_names_final"] = bus_df["gene_names_final"].apply(safe_literal_eval)  # TODO: consider looking through gene_names_final_set rather than gene_names_final for possible speedup (but make sure safe_literal_eval supports this)

    # iterate through bus_df rows
    for _, row in bus_df.iterrows():
        if "dlist" in row["gene_names_final"] and (mm or len(row["gene_names_final"]) == 2):  # don't replace with row['counted_in_count_matrix'] because this is the bus from when I ran union
            read_dlisted_by_one_of_its_respective_dlist_sequences = False
            for vcrs_header in row["gene_names_final"]:
                if vcrs_header != "dlist":
                    read_dlisted_by_one_of_its_respective_dlist_sequences = check_if_read_dlisted_by_one_of_its_respective_dlist_sequences(
                        vcrs_header=vcrs_header,
                        vcrs_header_to_seq_dict=vcrs_header_to_seq_dict,
                        dlist_header_to_seq_dict=dlist_header_to_seq_dict,
                        k=k,
                    )
                    if read_dlisted_by_one_of_its_respective_dlist_sequences:
                        break
            if not read_dlisted_by_one_of_its_respective_dlist_sequences:
                # barcode_idx = [i for i, name in enumerate(adata.obs_names) if barcode.endswith(name)][0]  # if I did not remove the padding
                barcode_idx = np.where(adata.obs_names == row["barcode"])[0][0]  # if I previously removed the padding
                vcrs_idxs = [var_names_to_idx_in_adata_dict[header] for header in row["gene_names_final"] if header in var_names_to_idx_in_adata_dict]

                increment_matrix[barcode_idx, vcrs_idxs] += row["count"]

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


# to be clear, this removes double counting of the same VCRS on each paired end, which is valid when fragment length < 2*read length OR for long insertions that make VCRS very long (such that the VCRS spans across both ends even when considering the region between the ends)
def decrement_adata_matrix_when_split_by_Ns_or_running_paired_end_in_single_end_mode(adata, fastq, kb_count_out, t2g, mm, bustools="bustools", split_Ns=False, paired_end_fastqs=False, paired_end_suffix_length=2, technology="bulk", keep_only_insertions=True, ignore_barcodes=False):
    if not split_Ns and not paired_end_fastqs:
        raise ValueError("At least one of split_Ns or paired_end_fastqs must be True")
    if technology.lower() != "bulk":
        raise ValueError("This function currently only works with bulk RNA-seq data")

    if not os.path.exists(f"{kb_count_out}/bus_df.csv"):
        bus_df = make_bus_df(kb_count_out, fastq, t2g_file=t2g, mm=mm, union=False, technology=technology, bustools=bustools, ignore_barcodes=ignore_barcodes)
    else:
        bus_df = pd.read_csv(f"{kb_count_out}/bus_df.csv")

    if "vcrs_variant_type" not in adata.var.columns:
        adata.var = add_vcrs_variant_type(adata.var, var_column="vcrs_header")

    if keep_only_insertions:  # valid when fragment length >= 2*read length
        # Can only count for insertions (lengthens the VCRS)
        variant_types_with_a_chance_of_being_double_counted_after_N_split = {
            "insertion",
            "delins",
            "mixed",
        }

        # Filter and retrieve the set of 'vcrs_header' values
        potentially_double_counted_reference_items = set(adata.var["vcrs_id"][adata.var["vcrs_variant_type"].isin(variant_types_with_a_chance_of_being_double_counted_after_N_split)])

        # filter bus_df to only keep rows where bus_df['gene_names_final'] contains a gene that is in potentially_double_counted_reference_items
        pattern = "|".join(potentially_double_counted_reference_items)
        bus_df = bus_df[bus_df["gene_names_final"].str.contains(pattern, regex=True)]

    n_rows, n_cols = adata.X.shape
    decrement_matrix = csr_matrix((n_rows, n_cols))
    bus_df["gene_names_final"] = bus_df["gene_names_final"].apply(safe_literal_eval)

    tested_read_header_bases = set()

    var_names_to_idx_in_adata_dict = {name: idx for idx, name in enumerate(adata.var_names)}

    for _, row in bus_df.iterrows():
        if row["counted_in_count_matrix"]:
            read_header_base = row["fastq_header"]
            if split_Ns:  # assumes the form READHEADERpairedendportion:START-END
                read_header_base = read_header_base.rsplit(":", 1)[0]  # now will be of the form READHEADERpairedendportion
            if paired_end_fastqs:  # assumes the form READHEADERpairedendportion
                read_header_base = read_header_base[:-paired_end_suffix_length]  # now will be of the form READHEADER
            if read_header_base not in tested_read_header_bases:  # here to make sure I don't double-count the decrementing
                filtered_bus_df = bus_df[bus_df["gene_names_final"].str.contains(read_header_base)]
                # Calculate the count of matching rows with the same 'EC' and 'barcode'
                count = sum(1 for _, item in filtered_bus_df.iterrows() if item["EC"] == row["EC"] and item["barcode"] == row["barcode"]) - 1  # Subtract 1 to avoid counting the current row itself

                if count > 0:
                    barcode_idx = np.where(adata.obs_names == row["barcode"])[0][0]  # if I previously removed the padding
                    vcrs_idxs = [var_names_to_idx_in_adata_dict[header] for header in row["gene_names_final"] if header in var_names_to_idx_in_adata_dict]
                    decrement_matrix[barcode_idx, vcrs_idxs] += count
                tested_read_header_bases.add(read_header_base)

    if not isinstance(adata.X, csr_matrix):
        adata.X = adata.X.tocsr()

    if not isinstance(decrement_matrix, csr_matrix):
        decrement_matrix = decrement_matrix.tocsr()

    # Add the two sparse matrices
    adata.X = adata.X - decrement_matrix

    adata.X = csr_matrix(adata.X)

    return adata


def remove_adata_columns(adata, values_of_interest, operation, var_column_name):
    if isinstance(values_of_interest, str) and values_of_interest.endswith(".txt"):
        with open(values_of_interest, "r", encoding="utf-8") as f:
            values_of_interest_set = {line.strip() for line in f}
    elif isinstance(values_of_interest, (list, tuple, set)):
        values_of_interest_set = set(values_of_interest)
    else:
        raise ValueError("values_of_interest must be a list, tuple, set, or a file path ending with .txt")

    # Step 2: Filter adata.var based on whether 'vcrs_id' is in the set
    columns_to_remove = adata.var.index[adata.var[var_column_name].isin(values_of_interest_set)]

    # Step 3: Remove the corresponding columns in adata.X and rows in adata.var
    if operation == "keep":
        adata = adata[:, adata.var_names.isin(columns_to_remove)]
    elif operation == "exclude":
        adata = adata[:, ~adata.var_names.isin(columns_to_remove)]

    return adata


def intersect_lists(series):
    return list(set.intersection(*map(set, series)))


def map_transcripts_to_genes(transcript_list, mapping_dict):
    return [mapping_dict.get(transcript, "Unknown") for transcript in transcript_list]


def make_good_barcodes_and_file_index_tuples(barcodes, include_file_index=False):
    if isinstance(barcodes, (str, Path)):
        with open(barcodes, encoding="utf-8") as f:
            barcodes = f.read().splitlines()

    good_barcodes_list = []
    for i in range(len(barcodes)):
        good_barcode_index = i // 2
        good_barcode = barcodes[good_barcode_index]
        if include_file_index:
            file_index = i % 2
            good_barcodes_list.append((good_barcode, str(file_index)))
        else:
            good_barcodes_list.append(good_barcode)

    bad_to_good_barcode_dict = dict(zip(barcodes, good_barcodes_list))
    return bad_to_good_barcode_dict



def make_bus_df(kb_count_out, fastq_file_list, t2g_file=None, mm=False, union=False, technology="bulk", parity="single", bustools="bustools", check_only=False):  # make sure this is in the same order as passed into kb count - [sample1, sample2, etc] OR [sample1_pair1, sample1_pair2, sample2_pair1, sample2_pair2, etc]
    with open(f"{kb_count_out}/kb_info.json", 'r') as f:
        kb_info_data = json.load(f)
    if "--num" not in kb_info_data.get("call", ""):
        raise ValueError("This function only works when kb count was run with --num (as this means that each row of the BUS file corresponds to exactly one read)")
    if "--parity paired" in kb_info_data.get("call", ""):
        vcrs_parity = "paired"  # same as parity_kb_count in vk count/clean
    else:
        vcrs_parity = "single"
    dlist_none_pattern = r"dlist\s*(?:=|\s+)?(?:'None'|None|\"None\")"
    if "dlist" not in kb_info_data.get("call", "") or re.search(dlist_none_pattern, kb_info_data.get("call", "")):
        used_dlist = False
    else:
        used_dlist = True
        
    fastq_file_list = load_in_fastqs(fastq_file_list)
    fastq_file_list = sort_fastq_files_for_kb_count(fastq_file_list, technology=technology, check_only=check_only)
    
    print("loading in transcripts")
    with open(f"{kb_count_out}/transcripts.txt", encoding="utf-8") as f:
        transcripts = f.read().splitlines()  # get transcript at index 0 with transcript[0], and index of transcript named "name" with transcript.index("name")
    transcripts.append("dlist")  # add dlist to the end of the list
    
    if technology != "bulk":
        parity = "single"

    technology = technology.lower()
    if technology == "bulk":
        #* barcodes
        print("loading in barcodes")
        with open(f"{kb_count_out}/matrix.sample.barcodes", encoding="utf-8") as f:
            barcodes = f.read().splitlines()

        if len(fastq_file_list) // 2 == len(barcodes):
            vcrs_parity = "paired"  # just a sanity check
            barcodes = [x for x in barcodes for _ in range(2)]  # converts ["AAA", "AAC"] to ["AAA", "AAA", "AAC", "AAC"]
        
        #* fastq df
        fastq_header_df = pd.DataFrame(columns=["read_index", "fastq_header", "barcode"])
        for i, fastq_file in enumerate(fastq_file_list):
            fastq_file = str(fastq_file)  # important for temp files
            fastq_header_list = [header.strip() for header, _, _ in tqdm(pyfastx.Fastx(fastq_file), desc="Processing FASTQ headers")]
            barcode_list = barcodes[i]

            new_rows = pd.DataFrame({"read_index": range(len(fastq_header_list)), "fastq_header": fastq_header_list, "barcode": barcode_list})
            fastq_header_df = pd.concat([fastq_header_df, new_rows], ignore_index=True)
        
        #* ec df
        # Get equivalence class that matches to 0-indexed line number of target ID
        print("loading in ec matrix")
        ec_df = pd.read_csv(
            f"{kb_count_out}/matrix.ec",
            sep="\t",
            header=None,
            names=["EC", "transcript_ids"],
        )

        ec_df["transcript_ids"] = ec_df["transcript_ids"].astype(str)
        ec_df["transcript_ids_list"] = ec_df["transcript_ids"].apply(lambda x: tuple(map(int, x.split(","))))
        ec_df["transcript_names"] = ec_df["transcript_ids_list"].apply(lambda ids: tuple(transcripts[i] for i in ids))
        ec_df.drop(columns=["transcript_ids", "transcript_ids_list"], inplace=True)  # drop transcript_ids

        #* t2g
        if t2g_file is not None:
            print("loading in t2g df")
            t2g_df = pd.read_csv(t2g_file, sep="\t", header=None, names=["transcript_id", "gene_name"])
            t2g_dict = dict(zip(t2g_df["transcript_id"], t2g_df["gene_name"]))

        #* bus
        bus_file = f"{kb_count_out}/output.bus"
        bus_text_file = f"{kb_count_out}/output_sorted_bus.txt"
        if not os.path.exists(bus_text_file):
            print("running bustools text")
            create_bus_txt_file_command = [bustools, "text", "-o", bus_text_file, "-f", bus_file]  # bustools text -p -a -f -d output.bus
            subprocess.run(create_bus_txt_file_command, check=True)
        print("loading in bus df")
        bus_df = pd.read_csv(
            bus_text_file,
            sep="\t",
            header=None,
            names=["barcode", "UMI", "EC", "count", "read_index"],
        )
        bus_df.drop(columns=["count"], inplace=True)  # drop count (it's always 1)

        # TODO: if I have low memory mode, then break up bus_df and loop from here through end
        print("Merging fastq header df and ec_df into bus df")
        bus_df = bus_df.merge(fastq_header_df, on=["read_index", "barcode"], how="left")
        bus_df = bus_df.merge(ec_df, on="EC", how="left")

        if parity == "paired" and vcrs_parity == "single":
            bad_to_good_barcode_dict = make_good_barcodes_and_file_index_tuples(barcodes, include_file_index=True)
            
            bus_df[['corrected_barcode', 'file_index']] = bus_df['barcode'].map(bad_to_good_barcode_dict).apply(pd.Series)
            bus_df.drop(columns=["barcode"], inplace=True)
            bus_df.rename(columns={"corrected_barcode": "barcode"}, inplace=True)

            dup_mask = bus_df.duplicated(subset=['barcode', 'UMI', 'EC', 'read_index'], keep=False)  # Identify all rows that have duplicates (including first occurrences)
            bus_df.loc[dup_mask, 'file_index'] = 'both'  # Set 'file_index' to 'all' for all duplicated rows
            bus_df = bus_df.drop_duplicates(subset=['barcode', 'UMI', 'EC', 'read_index'], keep='first')  # Drop duplicate rows, keeping only the first occurrence
        else:
            bus_df["file_index"] = "0"
        bus_df["file_index"] = bus_df["file_index"].astype("category")

        if t2g_file is not None:
            print("Apply the mapping function to create gene name columns")
            bus_df["gene_names"] = bus_df["transcript_names"].progress_apply(lambda x: map_transcripts_to_genes(x, t2g_dict))
            print("Taking set of gene_names")
            bus_df["gene_names"] = bus_df["gene_names"].progress_apply(lambda x: sorted(tuple(set(x))))
        else:
            bus_df["gene_names"] = bus_df["transcript_names"]

        print("Determining what counts in count matrix")
        if union or mm:
            # union or mm gets added to count matrix as long as dlist is not included in the EC
            if used_dlist:
                bus_df["counted_in_count_matrix"] = bus_df["transcript_names_final"].progress_apply(lambda x: "dlist" not in x)
            else:
                bus_df["counted_in_count_matrix"] = True
        else:
            # only gets added to the count matrix if EC has exactly 1 gene
            bus_df["counted_in_count_matrix"] = bus_df["gene_names"].progress_apply(lambda x: len(x) == 1)
        
        print("Saving bus df as parquet")
        bus_df.to_parquet(f"{kb_count_out}/bus_df.parquet", index=False)
        return bus_df
        
    
    
    else:
        for i, fastq_file in enumerate(fastq_file_list):
            pass
        pass #!!! WRITE FOR SINGLE-CELL


def make_bus_df_original(kallisto_out, fastq_file_list, t2g_file, mm=False, union=False, technology="bulk", parity="single", bustools="bustools", ignore_barcodes=False):  # make sure this is in the same order as passed into kb count - [sample1, sample2, etc] OR [sample1_pair1, sample1_pair2, sample2_pair1, sample2_pair2, etc]  # technology flag of kb
    print("loading in transcripts")
    with open(f"{kallisto_out}/transcripts.txt", encoding="utf-8") as f:
        transcripts = f.read().splitlines()  # get transcript at index 0 with transcript[0], and index of transcript named "name" with transcript.index("name")

    transcripts.append("dlist")  # add dlist to the end of the list

    technology = technology.lower()

    if technology == "bulk" or "smartseq" in technology.lower():  # smartseq does not have barcodes
        print("loading in barcodes")
        with open(f"{kallisto_out}/matrix.sample.barcodes", encoding="utf-8") as f:
            barcodes = f.read().splitlines()  # get transcript at index 0 with transcript[0], and index of transcript named "name" with transcript.index("name")
    else:
        if technology == "bulk" and ignore_barcodes:
            raise ValueError("ignore_barcodes is only supported for bulk RNA-seq data")

        try:
            barcode_start = technology_barcode_and_umi_dict[technology]["barcode_start"]
            barcode_end = technology_barcode_and_umi_dict[technology]["barcode_end"]
            umi_start = technology_barcode_and_umi_dict[technology]["umi_start"]
            umi_end = technology_barcode_and_umi_dict[technology]["umi_end"]
        except KeyError:
            print(f"technology {technology} currently not supported. Supported are {list(technology_barcode_and_umi_dict.keys())}")

        pass  # TODO: write this (will involve technology parameter to get barcode from read)

    fastq_header_df = pd.DataFrame(columns=["read_index", "fastq_header", "barcode"])

    if parity == "paired":
        fastq_header_df["fastq_header_pair"] = None

    if isinstance(fastq_file_list, (str, Path)):
        fastq_file_list = [str(fastq_file_list)]

    skip_upcoming_fastq = False

    for i, fastq_file in enumerate(fastq_file_list):
        if skip_upcoming_fastq:
            skip_upcoming_fastq = False
            continue
        # important for temp files
        fastq_file = str(fastq_file)

        print("loading in fastq headers")
        if fastq_file.endswith(fastq_extensions):
            fastq_header_list = get_header_set_from_fastq(fastq_file, output_format="list")
        elif fastq_file.endswith(".txt"):
            with open(fastq_file, encoding="utf-8") as f:
                fastq_header_list = f.read().splitlines()
        else:
            raise ValueError(f"fastq file {fastq_file} does not have a supported extension")

        if technology == "bulk" or "smartseq" in technology.lower():
            if ignore_barcodes:
                barcode_list = barcodes[0]
            else:
                barcode_list = barcodes[i]
        else:
            fq_dict = pyfastx.Fastq(fastq_file, build_index=True)
            barcode_list = [fq_dict[i].seq[barcode_start:barcode_end] for i in range(len(fq_dict))]

        new_rows = pd.DataFrame({"read_index": range(len(fastq_header_list)), "fastq_header": fastq_header_list, "barcode": barcode_list})  # Position/index values  # List values

        if parity == "paired":
            fastq_file_pair = str(fastq_file_list[i + 1])
            if fastq_file_pair.endswith(fastq_extensions):
                new_rows["fastq_header_pair"] = get_header_set_from_fastq(fastq_file_pair, output_format="list")
            elif fastq_file_pair.endswith(".txt"):
                with open(fastq_file_pair, encoding="utf-8") as f:
                    new_rows["fastq_header_pair"] = f.read().splitlines()

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
    ec_df["transcript_ids_list"] = ec_df["transcript_ids_list"].apply(lambda x: list(map(int, x)))
    ec_df["transcript_ids_list"] = ec_df["transcript_ids"].apply(lambda x: list(map(int, x.split(","))))
    ec_df["transcript_names"] = ec_df["transcript_ids_list"].apply(lambda ids: [transcripts[i] for i in ids])

    print("loading in t2g df")
    t2g_df = pd.read_csv(t2g_file, sep="\t", header=None, names=["transcript_id", "gene_name"])
    t2g_dict = dict(zip(t2g_df["transcript_id"], t2g_df["gene_name"]))

    # Get bus output (converted to txt)
    bus_file = f"{kallisto_out}/output.bus"
    bus_text_file = f"{kallisto_out}/output_sorted_bus.txt"
    if not os.path.exists(bus_text_file):
        print("running bustools text")
        bus_txt_file_existed_originally = False
        create_bus_txt_file_command = [bustools, "text", "-o", bus_text_file, "-f", bus_file]
        subprocess.run(create_bus_txt_file_command, check=True)
        # bustools text -p -a -f -d output.bus
    else:
        bus_txt_file_existed_originally = True

    print("loading in bus df")
    bus_df = pd.read_csv(
        bus_text_file,
        sep="\t",
        header=None,
        names=["barcode", "UMI", "EC", "count", "read_index"],
    )

    if ignore_barcodes:
        bus_df["barcode"] = barcodes[0]  # set all barcodes to the first barcode in barcodes list

    if not bus_txt_file_existed_originally:
        os.remove(bus_text_file)

    # TODO: if I have low memory mode, then break up bus_df and loop from here through end
    bus_df = bus_df.merge(fastq_header_df, on=["read_index", "barcode"], how="left")

    print("merging ec df into bus df")
    bus_df = bus_df.merge(ec_df, on="EC", how="left")

    if technology != "bulk":
        bus_df_collapsed_1 = bus_df.groupby(["barcode", "UMI", "EC"], as_index=False).agg(
            {
                "count": "sum",  # Sum counts
                "read_index": lambda x: list(x),  # Combine ints in a list
                "fastq_header": lambda x: list(x),  # Combine strings in a list
                "transcript_ids": "first",  # Take the first value for all other columns
                "transcript_ids_list": "first",  # Take the first value for all other columns
                "transcript_names": "first",  # Take the first value for all other columns
            }
        )

        bus_df_collapsed_2 = bus_df_collapsed_1.groupby(["barcode", "UMI"], as_index=False).agg(
            {
                "EC": lambda x: list(x),
                "count": "sum",  # Sum the 'count' column
                "read_index": lambda x: sum(x, []),  # Concatenate lists in 'read_index'
                "fastq_header": lambda x: sum(x, []),  # Concatenate lists in 'fastq_header'
                "transcript_ids": lambda x: ",".join(x),  # Join strings in 'transcript_ids_list' with commas  # may contain duplicates indices
                "transcript_ids_list": lambda x: sum(x, []),  # Concatenate lists for 'transcript_ids_list'
                "transcript_names": lambda x: sum(x, []),  # Concatenate lists for 'transcript_names'
            }
        )

        # Add new columns for the intersected lists
        bus_df_collapsed_2["transcript_names_final"] = bus_df_collapsed_1.groupby(["barcode", "UMI"])["transcript_names"].apply(intersect_lists).values
        bus_df_collapsed_2["transcript_ids_list_final"] = bus_df_collapsed_1.groupby(["barcode", "UMI"])["transcript_ids_list"].apply(intersect_lists).values

        bus_df = bus_df_collapsed_2

    else:  # technology == "bulk"
        # bus_df.rename(columns={"transcript_ids_list": "transcript_ids_list_final", "transcript_names": "transcript_names_final"}, inplace=True)
        bus_df["transcript_ids_list_final"] = bus_df["transcript_ids_list"]
        bus_df["transcript_names_final"] = bus_df["transcript_names"]

    print("Apply the mapping function to create gene name columns")
    # mapping transcript to gene names
    bus_df["gene_names"] = bus_df["transcript_names"].apply(lambda x: map_transcripts_to_genes(x, t2g_dict))
    bus_df["gene_names_final"] = bus_df["transcript_names_final"].apply(lambda x: map_transcripts_to_genes(x, t2g_dict))

    bus_df["gene_names_final_set"] = bus_df["gene_names_final"].apply(set)

    print("added counted in matrix column")
    if union or mm:
        # union or mm gets added to count matrix as long as dlist is not included in the EC
        bus_df["counted_in_count_matrix"] = bus_df["transcript_names_final"].apply(lambda x: "dlist" not in x)
    else:
        # only gets added to the count matrix if EC has exactly 1 gene
        bus_df["counted_in_count_matrix"] = bus_df["gene_names_final_set"].apply(lambda x: len(x) == 1)

    # adata_path = f"{kallisto_out}/counts_unfiltered/adata.h5ad"
    # adata = ad.read_h5ad(adata_path)
    # barcode_length = len(adata.obs.index[0])
    # bus_df['barcode_without_padding'] = bus_df['barcode'].str[(32 - barcode_length):]

    # so now I can iterate through this dataframe for the columns where counted_in_count_matrix is True - barcode will be the cell/sample (adata row), gene_names_final will be the list of gene name(s) (adata column), and count will be the number added to this entry of the matrix (always 1 for bulk)

    # save bus_df
    print("saving bus df")
    bus_df.to_csv(f"{kallisto_out}/bus_df.csv", index=False)
    return bus_df


# TODO: test
def match_paired_ends_after_single_end_run(bus_df_path, gene_name_type="vcrs_id", id_to_header_csv=None):
    if os.path.exists(bus_df_path):
        bus_df = pd.read_csv(bus_df_path)
    else:
        raise FileNotFoundError(f"{bus_df_path} does not exist")

    paired_end_suffix_length = 2  # * only works for /1 and /2 notation
    bus_df["fastq_header_without_paired_end_suffix"] = bus_df["fastq_header"].str[:-paired_end_suffix_length]

    # get the paired ends side-by-side
    df_1 = bus_df[bus_df["fastq_header"].str.endswith("/1")].copy()  # * only works for /1 and /2 notation
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
    bus_df["gene_names_final_pair"] = bus_df["gene_names_final_pair"].apply(safe_literal_eval)

    if gene_name_type == "vcrs_id":
        id_to_header_dict = make_mapping_dict(id_to_header_csv, dict_key="id")

        bus_df["vcrs_header_list"] = bus_df["gene_names_final"].apply(lambda gene_list: [id_to_header_dict.get(gene, gene) for gene in gene_list])

        bus_df["vcrs_header_list_pair"] = bus_df["gene_names_final_pair"].apply(lambda gene_list: [id_to_header_dict.get(gene, gene) for gene in gene_list])

        bus_df["ensembl_transcript_list"] = [value.split(":")[0] for value in bus_df["vcrs_header_list"]]
        bus_df["ensembl_transcript_list_pair"] = [value.split(":")[0] for value in bus_df["vcrs_header_list_pair"]]

        # TODO: map ENST to ENSG
        bus_df["gene_list"] = ""
        bus_df["gene_list_pair"] = ""
    else:
        bus_df["gene_list"] = bus_df["gene_names_final"]
        bus_df["gene_list_pair"] = bus_df["gene_names_final_pair"]

    bus_df["paired_ends_map_to_different_genes"] = bus_df.apply(
        lambda row: (isinstance(row["gene_list"], list) and bool(row["gene_list"]) and isinstance(row["gene_list_pair"], list) and bool(row["gene_list_pair"]) and not set(row["gene_list"]).intersection(row["gene_list_pair"])),
        axis=1,
    )

    return bus_df


# TODO: unsure if this works for sc
def adjust_variant_adata_by_normal_gene_matrix(adata, kb_count_vcrs_dir, kb_count_reference_genome_dir, id_to_header_csv=None, adata_output_path=None, vcrs_t2g=None, t2g_standard=None, fastq_file_list=None, mm=False, union=False, technology="bulk", parity="single", bustools="bustools", ignore_barcodes=False, check_only=False):
    if not adata:
        adata = f"{kb_count_vcrs_dir}/counts_unfiltered/adata.h5ad"
    if isinstance(adata, str):
        adata = ad.read_h5ad(adata)

    fastq_file_list = load_in_fastqs(fastq_file_list)
    fastq_file_list = sort_fastq_files_for_kb_count(fastq_file_list, technology=technology, check_only=check_only)

    bus_df_mutation_path = f"{kb_count_vcrs_dir}/bus_df.csv"
    bus_df_standard_path = f"{kb_count_reference_genome_dir}/bus_df.csv"

    if not os.path.exists(bus_df_mutation_path):
        bus_df_mutation = make_bus_df(
            kb_count_out=kb_count_vcrs_dir,
            fastq_file_list=fastq_file_list,
            t2g_file=vcrs_t2g,
            mm=mm,
            union=union,
            technology=technology,
            parity=parity,
            bustools=bustools,
            check_only=check_only
        )
    else:
        bus_df_mutation = pd.read_csv(bus_df_mutation_path)

    bus_df_mutation["gene_names_final"] = bus_df_mutation["gene_names_final"].apply(safe_literal_eval)
    bus_df_mutation.rename(columns={"gene_names_final": "VCRS_headers_final", "count": "count_value"}, inplace=True)

    if id_to_header_csv:
        bus_df_mutation.rename(columns={"VCRS_headers_final": "VCRS_ids_final"}, inplace=True)
        id_to_header_dict = make_mapping_dict(id_to_header_csv, dict_key="id")
        bus_df_mutation["VCRS_headers_final"] = bus_df_mutation["VCRS_ids_final"].apply(lambda name_list: [id_to_header_dict.get(name, name) for name in name_list])

    bus_df_mutation["transcripts_VCRS"] = bus_df_mutation["VCRS_headers_final"].apply(lambda string_list: tuple({s.split(":")[0] for s in string_list}))

    if not os.path.exists(bus_df_standard_path):
        bus_df_standard = make_bus_df(
            kallisto_out=kb_count_reference_genome_dir,
            fastq_file_list=fastq_file_list,  # make sure this is in the same order as passed into kb count - [sample1, sample2, etc] OR [sample1_pair1, sample1_pair2, sample2_pair1, sample2_pair2, etc]
            t2g_file=t2g_standard,
            mm=mm,
            union=union,
            technology=technology,
            parity=parity,
            bustools=bustools,
        )
    else:
        bus_df_standard = pd.read_csv(bus_df_standard_path, usecols=["barcode", "UMI", "fastq_header", "transcript_names_final"])

    bus_df_standard["transcript_names_final"] = bus_df_standard["transcript_names_final"].apply(safe_literal_eval)
    bus_df_standard["transcripts_standard"] = bus_df_standard["transcript_names_final"].apply(lambda name_list: tuple(re.match(r"^(ENST\d+)", name).group(0) if re.match(r"^(ENST\d+)", name) else name for name in name_list))

    if ignore_barcodes:
        columns_for_merging = ["UMI", "fastq_header", "transcripts_standard"]
        columns_for_merging_without_transcripts_standard = ["UMI", "fastq_header"]
    else:
        columns_for_merging = ["barcode", "UMI", "fastq_header", "transcripts_standard"]
        columns_for_merging_without_transcripts_standard = ["barcode", "UMI", "fastq_header"]

    bus_df_mutation = bus_df_mutation.merge(bus_df_standard[columns_for_merging], on=columns_for_merging_without_transcripts_standard, how="left", suffixes=("", "_standard"))  # keep barcode designations of mutation bus df (which aligns with the adata object)

    # TODO: I think this might be the inverse logic in the "any" line
    bus_df_mutation["vcrs_matrix_received_a_count_from_a_read_that_aligned_to_a_different_gene"] = bus_df_mutation.apply(lambda row: (row["counted_in_count_matrix"] and any(transcript in row["transcripts_standard"] for transcript in row["transcripts_vcrs"])), axis=1)

    n_rows, n_cols = adata.X.shape
    decrement_matrix = csr_matrix((n_rows, n_cols))

    var_names_to_idx_in_adata_dict = {name: idx for idx, name in enumerate(adata.var_names)}

    # iterate through the rows where the erroneous counting occurred
    for row in bus_df_mutation.loc[bus_df_mutation["vcrs_matrix_received_a_count_from_a_read_that_aligned_to_a_different_gene"]].itertuples():
        barcode_idx = np.where(adata.obs_names == row.barcode)[0][0]  # if I previously removed the padding
        vcrs_idxs = [var_names_to_idx_in_adata_dict[header] for header in row.VCRS_ids_final if header in var_names_to_idx_in_adata_dict]

        decrement_matrix[barcode_idx, vcrs_idxs] += row.count_value

    if not isinstance(adata.X, csr_matrix):
        adata.X = adata.X.tocsr()

    if not isinstance(decrement_matrix, csr_matrix):
        decrement_matrix = decrement_matrix.tocsr()

    # Add the two sparse matrices
    adata.X = adata.X - decrement_matrix

    adata.X = csr_matrix(adata.X)

    # save adata
    if not adata_output_path:
        adata_output_path = f"{kb_count_vcrs_dir}/counts_unfiltered/adata_adjusted_by_gene_alignments.h5ad"

    adata.write(adata_output_path)

    return adata


def match_adata_orders(adata, adata_ref):
    # Ensure cells (obs) are in the same order
    adata = adata[adata_ref.obs_names]

    # Add missing genes to adata
    missing_genes = adata_ref.var_names.difference(adata.var_names)
    padding_matrix = csr_matrix((adata.n_obs, len(missing_genes)))  # Sparse zero matrix

    # Create a padded AnnData for missing genes
    adata_padded = ad.AnnData(X=padding_matrix, obs=adata.obs, var=pd.DataFrame(index=missing_genes))

    # Concatenate the original and padded AnnData objects
    adata_padded = ad.concat([adata, adata_padded], axis=1)

    # Reorder genes to match adata_ref
    adata_padded = adata_padded[:, adata_ref.var_names]

    return adata_padded


def make_vaf_matrix(adata_mutant_vcrs_path, adata_wt_vcrs_path, adata_vaf_output=None, mutant_vcf=None):
    adata_mutant_vcrs = ad.read_h5ad(adata_mutant_vcrs_path)
    adata_wt_vcrs = ad.read_h5ad(adata_wt_vcrs_path)

    adata_mutant_vcrs_path_out = adata_mutant_vcrs_path.replace(".h5ad", "_with_vaf.h5ad")
    adata_wt_vcrs_path_out = adata_wt_vcrs_path.replace(".h5ad", "_with_vaf.h5ad")

    adata_wt_vcrs_padded = match_adata_orders(adata=adata_wt_vcrs, adata_ref=adata_mutant_vcrs)

    # Perform element-wise division (handle sparse matrices)
    mutant_X = adata_mutant_vcrs.X
    wt_X = adata_wt_vcrs_padded.X

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
    adata_result = ad.AnnData(X=result_matrix, obs=adata_mutant_vcrs.obs, var=adata_mutant_vcrs.var)

    if not adata_vaf_output:
        adata_vaf_output = "./adata_vaf.h5ad"

    # Save the result as an AnnData object
    adata_result.write(adata_vaf_output)

    # merge wt allele depth into mutant adata
    # Ensure indices of adata2.var and adata1.var are aligned
    merged_var = adata_mutant_vcrs.var.copy()  # Start with adata1.var

    # Add the "vcrs_count" from adata2 as "wt_count" into adata1.var
    merged_var["wt_count"] = adata_wt_vcrs.var["vcrs_count"].rename("wt_count")

    # Assign the updated var back to adata1
    adata_mutant_vcrs.var = merged_var

    # Ensure there are no division by zero errors
    vcrs_count = adata_mutant_vcrs.var["vcrs_count"]
    wt_count = adata_mutant_vcrs.var["wt_count"]

    # Calculate VAF
    adata_mutant_vcrs.var["vaf_across_samples"] = vcrs_count / (vcrs_count + wt_count)

    # wherever wt_count has a NaN, I want adata_mutant_vcrs.var["vaf_across_samples"] to have a NaN
    adata_mutant_vcrs.var.loc[wt_count.isna(), "vaf_across_samples"] = pd.NA

    adata_mutant_vcrs.write(adata_mutant_vcrs_path_out)
    adata_wt_vcrs.write(adata_wt_vcrs_path_out)

    return adata_vaf_output


def add_vcf_info_to_cosmic_tsv(cosmic_tsv, reference_genome_fasta, cosmic_df_out=None, cosmic_cdna_info_csv=None, mutation_source="cds"):
    import pysam

    # load in COSMIC tsv with columns CHROM, POS, ID, REF, ALT
    cosmic_df = pd.read_csv(cosmic_tsv, sep="\t", usecols=["Mutation genome position GRCh37", "GENOMIC_WT_ALLELE_SEQ", "GENOMIC_MUT_ALLELE_SEQ", "ACCESSION_NUMBER", "Mutation CDS", "MUTATION_URL"])

    if mutation_source == "cdna":
        cosmic_cdna_info_df = pd.read_csv(cosmic_cdna_info_csv, usecols=["mutation_id", "mutation_cdna"])  # TODO: remove column hard-coding
        cosmic_cdna_info_df = cosmic_cdna_info_df.rename(columns={"mutation_cdna": "Mutation cDNA"})

    cosmic_df = add_variant_type(cosmic_df, "Mutation CDS")

    cosmic_df["ACCESSION_NUMBER"] = cosmic_df["ACCESSION_NUMBER"].str.split(".").str[0]

    cosmic_df[["CHROM", "GENOME_POS"]] = cosmic_df["Mutation genome position GRCh37"].str.split(":", expand=True)
    # cosmic_df['CHROM'] = cosmic_df['CHROM'].apply(convert_chromosome_value_to_int_when_possible)
    cosmic_df[["POS", "GENOME_END_POS"]] = cosmic_df["GENOME_POS"].str.split("-", expand=True)

    cosmic_df = cosmic_df.rename(columns={"GENOMIC_WT_ALLELE_SEQ": "REF", "GENOMIC_MUT_ALLELE_SEQ": "ALT", "MUTATION_URL": "mutation_id"})

    if mutation_source == "cds":
        cosmic_df["ID"] = cosmic_df["ACCESSION_NUMBER"] + ":" + cosmic_df["Mutation CDS"]
    elif mutation_source == "cdna":
        cosmic_df["mutation_id"] = cosmic_df["mutation_id"].str.extract(r"id=(\d+)")
        cosmic_df["mutation_id"] = cosmic_df["mutation_id"].astype(int, errors="raise")
        cosmic_df = cosmic_df.merge(cosmic_cdna_info_df[["mutation_id", "Mutation cDNA"]], on="mutation_id", how="left")
        cosmic_df["ID"] = cosmic_df["ACCESSION_NUMBER"] + ":" + cosmic_df["Mutation cDNA"]
        cosmic_df.drop(columns=["Mutation cDNA"], inplace=True)

    cosmic_df = cosmic_df.dropna(subset=["CHROM", "POS"])
    cosmic_df = cosmic_df.dropna(subset=["ID"])  # a result of intron mutations and COSMIC duplicates that get dropped before cDNA determination

    # reference_genome_fasta
    reference_genome = pysam.FastaFile(reference_genome_fasta)

    def get_nucleotide_from_reference(chromosome, position):
        # pysam is 0-based, so subtract 1 from the position
        return reference_genome.fetch(chromosome, int(position) - 1, int(position))

    def get_complement(nucleotide_sequence):
        return "".join([complement[nuc] for nuc in nucleotide_sequence])

    # Insertion, get original nucleotide (not in COSMIC df)
    cosmic_df.loc[(cosmic_df["GENOME_END_POS"].astype(int) != 1) & (cosmic_df["variant_type"] == "insertion"), "original_nucleotide"] = cosmic_df.loc[(cosmic_df["GENOME_END_POS"].astype(int) != 1) & (cosmic_df["variant_type"] == "insertion"), ["CHROM", "POS"]].progress_apply(lambda row: get_nucleotide_from_reference(row["CHROM"], int(row["POS"])), axis=1)

    # Deletion, get new nucleotide (not in COSMIC df)
    cosmic_df.loc[(cosmic_df["POS"].astype(int) != 1) & (cosmic_df["variant_type"] == "deletion"), "original_nucleotide"] = cosmic_df.loc[(cosmic_df["POS"].astype(int) != 1) & (cosmic_df["variant_type"] == "deletion"), ["CHROM", "POS"]].progress_apply(lambda row: get_nucleotide_from_reference(row["CHROM"], int(row["POS"]) - 1), axis=1)

    # Duplication
    cosmic_df.loc[cosmic_df["variant_type"] == "duplication", "original_nucleotide"] = cosmic_df.loc[cosmic_df["ID"].str.contains("dup", na=False), "ALT"].str[-1]

    # deal with start of 1, insertion
    cosmic_df.loc[(cosmic_df["GENOME_END_POS"].astype(int) == 1) & (cosmic_df["variant_type"] == "insertion"), "original_nucleotide"] = cosmic_df.loc[(cosmic_df["GENOME_END_POS"].astype(int) == 1) & (cosmic_df["variant_type"] == "insertion"), ["CHROM", "POS"]].progress_apply(lambda row: get_nucleotide_from_reference(row["CHROM"], int(row["GENOME_END_POS"])), axis=1)

    # deal with start of 1, deletion
    cosmic_df.loc[(cosmic_df["POS"].astype(int) == 1) & (cosmic_df["variant_type"] == "deletion"), "original_nucleotide"] = cosmic_df.loc[(cosmic_df["POS"].astype(int) == 1) & (cosmic_df["variant_type"] == "deletion"), ["CHROM", "POS"]].progress_apply(lambda row: get_nucleotide_from_reference(row["CHROM"], int(row["GENOME_END_POS"]) + 1), axis=1)

    # # deal with (-) strand - commented out because the vcf should all be relative to the forward strand, not the cdna
    # cosmic_df.loc[cosmic_df['strand'] == '-', 'original_nucleotide'] = cosmic_df.loc[cosmic_df['strand'] == '-', 'original_nucleotide'].apply(get_complement)

    # ins and dup, starting position not 1
    cosmic_df.loc[(((cosmic_df["variant_type"] == "insertion") | (cosmic_df["variant_type"] == "duplication")) & (cosmic_df["POS"].astype(int) != 1)), "ref_updated"] = cosmic_df.loc[(((cosmic_df["variant_type"] == "insertion") | (cosmic_df["variant_type"] == "duplication")) & (cosmic_df["POS"].astype(int) != 1)), "original_nucleotide"]
    cosmic_df.loc[(((cosmic_df["variant_type"] == "insertion") | (cosmic_df["variant_type"] == "duplication")) & (cosmic_df["POS"].astype(int) != 1)), "alt_updated"] = cosmic_df.loc[(((cosmic_df["variant_type"] == "insertion") | (cosmic_df["variant_type"] == "duplication")) & (cosmic_df["POS"].astype(int) != 1)), "original_nucleotide"] + cosmic_df.loc[(((cosmic_df["variant_type"] == "insertion") | (cosmic_df["variant_type"] == "duplication")) & (cosmic_df["POS"].astype(int) != 1)), "ALT"]

    # ins and dup, starting position 1
    cosmic_df.loc[(((cosmic_df["variant_type"] == "insertion") | (cosmic_df["variant_type"] == "duplication")) & (cosmic_df["POS"].astype(int) == 1)), "ref_updated"] = cosmic_df.loc[(((cosmic_df["variant_type"] == "insertion") | (cosmic_df["variant_type"] == "duplication")) & (cosmic_df["POS"].astype(int) == 1)), "original_nucleotide"]
    cosmic_df.loc[(((cosmic_df["variant_type"] == "insertion") | (cosmic_df["variant_type"] == "duplication")) & (cosmic_df["POS"].astype(int) == 1)), "alt_updated"] = cosmic_df.loc[(((cosmic_df["variant_type"] == "insertion") | (cosmic_df["variant_type"] == "duplication")) & (cosmic_df["POS"].astype(int) == 1)), "ALT"] + cosmic_df.loc[(((cosmic_df["variant_type"] == "insertion") | (cosmic_df["variant_type"] == "duplication")) & (cosmic_df["POS"].astype(int) == 1)), "original_nucleotide"]

    # del, starting position not 1
    cosmic_df.loc[((cosmic_df["variant_type"] == "deletion") & (cosmic_df["POS"].astype(int) != 1)), "ref_updated"] = cosmic_df.loc[((cosmic_df["variant_type"] == "deletion") & (cosmic_df["POS"].astype(int) != 1)), "original_nucleotide"] + cosmic_df.loc[((cosmic_df["variant_type"] == "deletion") & (cosmic_df["POS"].astype(int) != 1)), "REF"]
    cosmic_df.loc[((cosmic_df["variant_type"] == "deletion") & (cosmic_df["POS"].astype(int) != 1)), "alt_updated"] = cosmic_df.loc[((cosmic_df["variant_type"] == "deletion") & (cosmic_df["POS"].astype(int) != 1)), "original_nucleotide"]

    # del, starting position 1
    cosmic_df.loc[((cosmic_df["variant_type"] == "deletion") & (cosmic_df["POS"].astype(int) == 1)), "ref_updated"] = cosmic_df.loc[((cosmic_df["variant_type"] == "deletion") & (cosmic_df["POS"].astype(int) == 1)), "REF"] + cosmic_df.loc[((cosmic_df["variant_type"] == "deletion") & (cosmic_df["POS"].astype(int) == 1)), "original_nucleotide"]
    cosmic_df.loc[((cosmic_df["variant_type"] == "deletion") & (cosmic_df["POS"].astype(int) == 1)), "alt_updated"] = cosmic_df.loc[((cosmic_df["variant_type"] == "deletion") & (cosmic_df["POS"].astype(int) == 1)), "original_nucleotide"]

    # Deletion, update position (should refer to 1 BEFORE the deletion)
    cosmic_df.loc[(cosmic_df["POS"].astype(int) != 1) & (cosmic_df["variant_type"] == "deletion"), "POS"] = cosmic_df.loc[(cosmic_df["POS"].astype(int) != 1) & (cosmic_df["variant_type"] == "deletion"), "POS"].progress_apply(lambda pos: int(pos) - 1)

    # deal with start of 1, deletion update position (should refer to 1 after the deletion)
    cosmic_df.loc[(cosmic_df["POS"].astype(int) == 1) & (cosmic_df["variant_type"] == "deletion"), "POS"] = cosmic_df.loc[(cosmic_df["POS"].astype(int) == 1) & (cosmic_df["variant_type"] == "deletion"), "GENOME_END_POS"].astype(int) + 1

    # Insertion, update position when pos=1 (should refer to 1)
    cosmic_df.loc[(cosmic_df["GENOME_END_POS"].astype(int) == 1) & (cosmic_df["variant_type"] == "insertion"), "POS"] = 1

    cosmic_df["ref_updated"] = cosmic_df["ref_updated"].fillna(cosmic_df["REF"])
    cosmic_df["alt_updated"] = cosmic_df["alt_updated"].fillna(cosmic_df["ALT"])
    cosmic_df.rename(columns={"ALT": "alt_cosmic", "alt_updated": "ALT", "REF": "ref_cosmic", "ref_updated": "REF"}, inplace=True)
    cosmic_df.drop(columns=["Mutation genome position GRCh37", "GENOME_POS", "GENOME_END_POS", "ACCESSION_NUMBER", "Mutation CDS", "mutation_id", "ref_cosmic", "alt_cosmic", "original_nucleotide", "variant_type"], inplace=True)  # 'strand'

    num_rows_with_na = cosmic_df.isna().any(axis=1).sum()
    if num_rows_with_na > 0:
        raise ValueError(f"Number of rows with NA values: {num_rows_with_na}")

    cosmic_df["POS"] = cosmic_df["POS"].astype(np.int64)

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
    with open(output_file, "w", encoding="utf-8") as vcf_file:
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
                f"DP={row['DP']}" if pd.notna(row["DP"]) else None,
                f"AF={row['AF']}" if pd.notna(row["AF"]) else None,
                f"NS={row['NS']}" if pd.notna(row["NS"]) else None,
            ]
            info = ";".join(filter(None, info_fields))

            # Write VCF row
            vcf_file.write(f"{row['CHROM']}\t{row['POS']}\t{row['ID']}\t{row['REF']}\t{row['ALT']}\t.\tPASS\t{info}\n")


# TODO: make sure this works for rows with just ID and everything else blank (due to different mutations being concatenated)
def write_vcfs_for_rows(adata, adata_wt_vcrs, adata_vaf, output_dir):
    """
    Write a VCF file for each row (variant) in adata.var.

    Parameters:
        adata: AnnData object with mutant counts.
        adata_wt_vcrs: AnnData object with wild-type counts.
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
        vcrs_id = row["vcrs_id"]  # This is the index for the column in the matrices

        # Extract corresponding matrix values
        mutant_counts = adata[:, vcrs_id].X.flatten()  # Extract as 1D array
        wt_counts = adata_wt_vcrs[:, vcrs_id].X.flatten()  # Extract as 1D array
        vaf_values = adata_vaf[:, vcrs_id].X.flatten()  # Extract as 1D array

        # Create VCF file for the row
        output_file = f"{output_dir}/{var_id}.vcf"
        with open(output_file, "w", encoding="utf-8") as vcf_file:
            # Write VCF header
            vcf_file.write("##fileformat=VCFv4.2\n")
            vcf_file.write('##INFO=<ID=RD,Number=1,Type=Integer,Description="Total Depth">\n')
            vcf_file.write('##INFO=<ID=AF,Number=A,Type=Float,Description="Allele Frequency">\n')
            vcf_file.write('##INFO=<ID=NS,Number=1,Type=Integer,Description="Number of Samples">\n')
            vcf_file.write("#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\n")

            # Iterate through samples (rows in the matrix)
            for sample_idx, mutant_count in enumerate(mutant_counts):
                # Calculate RD and AF
                rd = mutant_count + wt_counts[sample_idx]
                af = vaf_values[sample_idx]

                # INFO field
                info = f"RD={int(rd)};AF={af:.3f};NS=1"

                # Write VCF row
                vcf_file.write(f"{chrom}\t{pos}\t{var_id}\t{ref}\t{alt}\t.\tPASS\t{info}\n")
