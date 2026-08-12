import os
import re
import shlex
import shutil
import subprocess
from pathlib import Path

import anndata
import anndata as ad
import numpy as np
import sys
import pandas as pd
import pyfastx
import pysam
from scipy.sparse import csr_matrix, issparse
from scipy.stats import mode
import json
import logging
import gzip
from tqdm import tqdm
import hashlib

from varseek.constants import (
    complement,
    fastq_extensions,
    technology_info,
    mutation_pattern,
    HGVS_pattern_general,
    technology_valid_values,
    non_single_cell_technologies
)
from varseek.utils.seq_utils import (
    add_variant_type,
    add_vcrs_variant_type,
    create_header_to_sequence_ordered_dict_from_fasta_WITHOUT_semicolon_splitting,
    get_header_set_from_fastq,
    make_mapping_dict,
    safe_literal_eval,
    sort_fastq_files_for_kb_count,
    load_in_fastqs,
    parquet_column_list_to_tuple,
    parquet_column_tuple_to_list,
    get_ensembl_gene_id_from_transcript_id_bulk,
    make_good_barcodes_and_file_index_tuples,
    load_adata_from_mtx,
    strip_gene_name_from_seq_id,
)
from varseek.utils.coordinate_conversion import (
    build_transcript_exon_model,
    convert_variants_transcript_to_genome,
)
from varseek.utils.logger_utils import set_up_logger, count_chunks, determine_write_mode, check_file_path_is_string_with_valid_extension, is_valid_int
from varseek.utils.visualization_utils import plot_cdna_locations

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

def hash_list(lst):
    return hashlib.md5("::".join(lst).encode()).hexdigest()

def intersect_lists(series):
    return list(set.intersection(*map(set, series)))


def map_transcripts_to_genes(transcript_list, mapping_dict):
    return [mapping_dict.get(transcript, transcript) for transcript in transcript_list]  # if it cannot find a transcript in the mapping_dict, it will return the transcript itself (this works if I give a non-transcript too, like a chromosome name)

def map_transcripts_to_genes_versionless(transcript_list, mapping_dict):
    return [mapping_dict.get(transcript, transcript).split(".")[0] for transcript in transcript_list]  # if it cannot find a transcript in the mapping_dict, it will return the transcript itself (this works if I give a non-transcript too, like a chromosome name)

def find_hamming_1_match(barcode, whitelist):
    bases = ["A", "C", "G", "T"]
    barcode_list = list(barcode)

    for i in range(len(barcode)):
        original_base = barcode_list[i]
        for base in bases:
            if base != original_base:
                barcode_list[i] = base
                mutated_barcode = "".join(barcode_list)
                if mutated_barcode in whitelist:
                    return mutated_barcode  # Return the first match
        barcode_list[i] = original_base  # Restore original base
    return None  # No match found

def make_bus_df(kb_count_out, fastq_file_list=None, technology=None, t2g_file=None, mm=False, parity="single", bustools="bustools", fastq_sorting_check_only=True, chunksize=None, bad_to_good_barcode_dict=None, correct_barcodes_of_hamming_distance_one=False, save_type="parquet", add_fastq_headers=True, save_transcript_names=True, add_counted_in_count_matrix=True, strip_version_number_from_genes=False):  # make sure this is in the same order as passed into kb count - [sample1, sample2, etc] OR [sample1_pair1, sample1_pair2, sample2_pair1, sample2_pair2, etc]
    with open(f"{kb_count_out}/kb_info.json", 'r') as f:
        kb_info_data = json.load(f)
    if "--num" not in kb_info_data.get("call", ""):
        raise ValueError("This function only works when kb count was run with --num (as this means that each row of the BUS file corresponds to exactly one read)")
    if "--parity paired" in kb_info_data.get("call", ""):
        vcrs_parity = "paired"  # same as parity_kb_count in vk count/clean
    else:
        vcrs_parity = "single"
    if technology is None:
        match = re.search(r"-x (\S+)", kb_info_data.get("call", ""))
        technology = match.group(1) if match else None
        if technology is None:
            raise ValueError("Please provide technology to continue")
        else:
            logger.info("Using technology from kb_info.json:", technology)

    dlist_none_pattern = r"dlist\s*(?:=|\s+)?(?:'None'|None|\"None\")"
    if "dlist" not in kb_info_data.get("call", "") or re.search(dlist_none_pattern, kb_info_data.get("call", "")):
        used_dlist = False
    else:
        used_dlist = True
        
    if fastq_file_list is not None:
        fastq_file_list = load_in_fastqs(fastq_file_list)
        fastq_file_list = sort_fastq_files_for_kb_count(fastq_file_list, technology=technology, check_only=fastq_sorting_check_only)
    
    #* transcripts
    logger.info("loading in transcripts")
    with open(f"{kb_count_out}/transcripts.txt", encoding="utf-8") as f:
        transcripts = f.read().splitlines()  # get transcript at index 0 with transcript[0], and index of transcript named "name" with transcript.index("name")
    transcripts.append("dlist")  # add dlist to the end of the list

    #* ec df
    # Get equivalence class that matches to 0-indexed line number of target ID
    logger.info("loading in ec matrix")
    ec_df = pd.read_csv(
        f"{kb_count_out}/matrix.ec",
        sep="\t",
        header=None,
        names=["EC", "transcript_ids"],
    )
    ec_df["transcript_ids"] = ec_df["transcript_ids"].astype(str)
    ec_df["transcript_ids_list"] = ec_df["transcript_ids"].apply(lambda x: tuple(map(int, x.split(","))))
    ec_df["transcript_names"] = ec_df["transcript_ids_list"].apply(lambda ids: list(transcripts[i] for i in ids))
    ec_df.drop(columns=["transcript_ids", "transcript_ids_list"], inplace=True)  # drop transcript_ids

    #* t2g
    if t2g_file is not None:
        logger.info("loading in t2g df")
        t2g_dict = make_t2g_dict(t2g_file)

    number_of_files = technology_info[technology.upper()]["num_files"]
    if isinstance(number_of_files, dict):
        number_of_files = number_of_files[vcrs_parity]
    barcode_file_index_tuple = technology_info[technology.upper()]["barcode_position"]
    if barcode_file_index_tuple is None:
        barcode_file_index = None
    else:
        if isinstance(barcode_file_index_tuple[0], tuple):
            barcode_file_index = list(set(group[0] for group in barcode_file_index_tuple))[0]
            if len(barcode_file_index) > 1:
                raise ValueError(f"Error: {technology} has multiple file indices for barcodes, please specify which one to use in kb count")
            barcode_start_positions_list = [group[1] for group in barcode_file_index_tuple]  # get the start positions for each group
            barcode_end_positions_list = [group[2] for group in barcode_file_index_tuple]  # get the end positions for each group
        else:
            barcode_file_index = barcode_file_index_tuple[0]
            barcode_start_positions_list = [barcode_file_index_tuple[1]]  # if only one group, then use the start position
            barcode_end_positions_list = [barcode_file_index_tuple[2]]
    transcript_file_index = technology_info[technology.upper()]["transcript_file_index"]

    technology = technology.lower()
    if technology not in {"bulk", "smartseq2"}:
        parity = "single"

    #* barcodes
    barcodes = None
    if barcode_file_index is None:  # including bulk
        logger.info("loading in barcodes")
        with open(f"{kb_count_out}/matrix.sample.barcodes", encoding="utf-8") as f:
            barcodes = f.read().splitlines()

    if barcodes and (len(fastq_file_list) // 2 == len(barcodes)):
        vcrs_parity = "paired"  # just a sanity check
        barcodes = [x for x in barcodes for _ in range(2)]  # converts ["AAA", "AAC"] to ["AAA", "AAA", "AAC", "AAC"]
    
    #* fastq df
    if add_fastq_headers:
        fastq_header_list, barcode_list, fastq_header_file_index = None, None, None
        fastq_header_df = pd.DataFrame(columns=["read_index", "fastq_header", "barcode", "file_index"])
        for i, fastq_file in enumerate(fastq_file_list):
            if i % number_of_files == transcript_file_index:
                fastq_file = str(fastq_file)  # important for temp files
                fastq_header_list = [header.strip() for header, _, _ in tqdm(pyfastx.Fastx(fastq_file), desc="Processing FASTQ headers")]
                if barcode_file_index is None:  # including bulk
                    barcode_list = barcodes[i]
                fastq_header_file_index = i
            elif barcode_file_index is not None and i % number_of_files == barcode_file_index:  # bulk handled right above
                barcode_list = []
                for _, seq, _ in tqdm(pyfastx.Fastx(fastq_file), desc=f"Retrieving FASTQ barcodes from file {(i // number_of_files) + 1} of {len(fastq_file_list) // number_of_files}"):
                    barcode = ""
                    for start_pos, end_pos in zip(barcode_start_positions_list, barcode_end_positions_list):
                        barcode += seq[start_pos:end_pos]  # extract the barcode from the sequence
                    barcode_list.append(barcode)
            else:
                continue

            if fastq_header_list is not None and barcode_list is not None:
                num_rows = len(fastq_header_list)
                new_rows = pd.DataFrame({"read_index": range(num_rows), "fastq_header": fastq_header_list, "barcode": barcode_list, "file_index": [fastq_header_file_index] * num_rows})
                fastq_header_df = pd.concat([fastq_header_df, new_rows], ignore_index=True)
                fastq_header_list, barcode_list, fastq_header_file_index = None, None, None

    #* bus
    bus_file = f"{kb_count_out}/output.bus"
    bus_text_file = f"{kb_count_out}/output_sorted_bus.txt"
    if not os.path.exists(bus_text_file):
        logger.info("running bustools text")
        create_bus_txt_file_command = [bustools, "text", "-o", bus_text_file, "-f", bus_file]  # bustools text -p -a -f -d output.bus
        subprocess.run(create_bus_txt_file_command, check=True)
    logger.info("loading in bus df")
    if chunksize:
        total_chunks = count_chunks(bus_text_file, chunksize)
    else:
        chunksize = sys.maxsize  # ensures 1 chunk
        total_chunks = 1
    
    bus_out_path = f"{kb_count_out}/bus_df.{save_type}"
    for i, bus_df in enumerate(pd.read_csv(bus_text_file, sep=r"\s+", header=None, names=["barcode", "UMI", "EC", "count", "read_index"], usecols=["barcode", "UMI", "EC", "read_index"], chunksize=chunksize)):
        if total_chunks > 1:
            logger.info(f"Processing chunk {i+1}/{total_chunks}")

        logger.info("Merging fastq header df and ec_df into bus df")
        bus_df = bus_df.merge(ec_df, on="EC", how="left")
        if add_fastq_headers:
            bus_df = bus_df.merge(fastq_header_df, on=["read_index", "barcode"], how="left")
            bus_df["file_index"] = bus_df["file_index"].astype("category")
            bus_df["read_index"] = bus_df["read_index"].astype("int64")

        if parity == "paired" and vcrs_parity == "single":
            if not bad_to_good_barcode_dict:
                bad_to_good_barcode_dict = make_good_barcodes_and_file_index_tuples(barcodes, include_file_index=False)  # TODO: for the non-bulk technologies with transcripts in >1 file (ie SMARTSEQ2, SMARTSEQ3, STORMSEQ), write a custom technology to be able to do something analogous to running in single-end mode and then later combining results
            
            bus_df['corrected_barcode'] = bus_df['barcode'].map(bad_to_good_barcode_dict).apply(pd.Series)
            bus_df.drop(columns=["barcode"], inplace=True)
            bus_df.rename(columns={"corrected_barcode": "barcode"}, inplace=True)

        if barcode_file_index is None:  # including bulk:
            bus_df["barcode"] = bus_df["barcode"].astype("category")
            bus_df["UMI"] = bus_df["UMI"].astype("category")
        else:
            if correct_barcodes_of_hamming_distance_one:
                #* Barcode Hamming distance 1 correction
                # Load whitelist into a set
                with open(f"{kb_count_out}/counts_unfiltered/cells_x_genes.barcodes.txt", encoding="utf-8") as f:  # use f"{kb_count_out}/10x_version3_whitelist.txt" for full list of valid barcodes
                    whitelist = set(line.strip() for line in f)
                
                bus_df["barcode_true"] = None
                
                # exact matches
                bus_df.loc[bus_df["barcode"].isin(whitelist), "barcode_true"] = bus_df["barcode"]

                # Find Hamming-1 matches for remaining None values
                # tqdm.pandas(desc="Correcting unmatched barcodes to Hamming-1 matches")
                bus_df.loc[pd.isna(bus_df["barcode_true"]), "barcode_true"] = bus_df.loc[
                    pd.isna(bus_df["barcode_true"])
                ].apply(lambda row: find_hamming_1_match(row["barcode"], whitelist), axis=1)

                bus_df = bus_df.dropna(subset=["barcode_true"])  # Drop rows where "barcode_true" is None
                bus_df = bus_df.drop(columns=["barcode"]).rename(columns={"barcode_true": "barcode"})       # Drop the "barcode" column, and rename "barcode_true" to "barcode"

        if t2g_file is not None:
            logger.info("Apply the mapping function to create gene name columns")
            if strip_version_number_from_genes and "." in bus_df['transcript_names'].iloc[0][0]:
                bus_df["gene_names"] = bus_df["transcript_names"].progress_apply(lambda x: map_transcripts_to_genes_versionless(x, t2g_dict))
            else:
                bus_df["gene_names"] = bus_df["transcript_names"].progress_apply(lambda x: map_transcripts_to_genes(x, t2g_dict))
            logger.info("Taking set of gene_names")
            bus_df["gene_names"] = bus_df["gene_names"].progress_apply(lambda x: list(sorted(set(x))))
        else:
            bus_df["gene_names"] = bus_df["transcript_names"]

        if not save_transcript_names:
            bus_df.drop(columns=["transcript_names", "EC"], inplace=True)


        if add_counted_in_count_matrix:
            logger.info("Determining what counts in count matrix")
            if mm:
                # mm gets added to count matrix as long as dlist is not included in the EC (same with union - union controls whether unioned reads make it to bus file, and mm controls whether multimapped/unioned reads are counted in adata)
                if used_dlist:
                    bus_df["counted_in_count_matrix"] = bus_df["gene_names"].progress_apply(lambda x: "dlist" not in x)
                else:
                    bus_df["counted_in_count_matrix"] = True
                bus_df["count_matrix_value"] = np.where(bus_df["counted_in_count_matrix"] & (bus_df["gene_names"].str.len() > 0), 1/bus_df["gene_names"].str.len(), 0)  # 0 for rows where bus_df["counted_in_count_matrix"] is False, and for rows where it's True, it's equal to the length of bus_df["gene_names"]
            else:
                # only gets added to the count matrix if EC has exactly 1 gene and no dlist entry
                if used_dlist:
                    bus_df["counted_in_count_matrix"] = bus_df["gene_names"].progress_apply(lambda x: len(x) == 1 and x != ("dlist",))
                else:
                    bus_df["counted_in_count_matrix"] = bus_df["gene_names"].str.len() == 1
                bus_df["count_matrix_value"] = np.where(bus_df["counted_in_count_matrix"], 1, 0)
        
        logger.info(f"Saving bus df to {bus_out_path}") if total_chunks == 1 else logger.info(f"Saving chunk {i+1}/{total_chunks} of bus df to {bus_out_path}")
        first_chunk = (i == 0)
        if save_type == "parquet":  # parquet benefits over csv: much smaller file size (~10% of csv size), and saves data types upon saving and loading
            # parquet_column_tuple_to_list(bus_df)  # parquet doesn't like tuples - it only likes lists - if wanting to load back in the data as a tuple later, then use parquet_column_list_to_tuple - commented out because I don't use tuples
            if total_chunks == 1:  # no appending needed, so either engine is fine
                bus_df.to_parquet(bus_out_path, index=False)
            else:  # need append function, which is only supported by fastparquet (ie not pyarrow)
                bus_df.to_parquet(bus_out_path, index=False, engine="fastparquet", append=not first_chunk)
        elif save_type == "csv":  # csv benefits over parquet: human-readable, can be read/iterated in chunks, and supports tuples (which take about 3/4 the RAM of strings/lists)
            bus_df.csv(bus_out_path, index=False, header=first_chunk, mode=determine_write_mode(bus_out_path, overwrite=True, first_chunk=first_chunk))
    
    logger.info("Finished processing bus df")
    
    if total_chunks > 1:
        logger.info("Returning the last chunk of bus_df")
    return bus_df
        
# @profile
def adjust_variant_adata_by_normal_gene_matrix(kb_count_vcrs_dir, kallisto_quant_reference_genome_dir, technology, t2g_standard, adata=None, fastq_file_list=None, adata_output_path=None, mm=False, parity="single", bustools="bustools", fastq_sorting_check_only=False, save_type="parquet", count_reads_that_dont_pseudoalign_to_reference_genome=True, drop_reads_where_the_pairs_mapped_to_different_genes=False, avoid_paired_double_counting=False, add_fastq_headers=False, seq_id_column="seq_ID", gene_id_column="gene_id", variant_source=None, gtf=None, skip_transcripts_without_genes=False, mistake_ratio=None):
    if not adata:
        adata = f"{kb_count_vcrs_dir}/counts_unfiltered/adata.h5ad"
    if isinstance(adata, str):
        adata = ad.read_h5ad(adata)
    adata = adata.copy()  # make a copy to avoid modifying the original adata

    # if not adata_output_path:
    #     adata_output_path = f"{kb_count_vcrs_dir}/counts_unfiltered/adata_adjusted_with_reference_genome_alignment.h5ad"
    # if adata_output_path and os.path.dirname(adata_output_path):
    #     os.makedirs(os.path.dirname(adata_output_path), exist_ok=True)

    if fastq_file_list is not None:
        fastq_file_list = load_in_fastqs(fastq_file_list)
        fastq_file_list = sort_fastq_files_for_kb_count(fastq_file_list, technology=technology, check_only=fastq_sorting_check_only)

    #* create a dataframe of the BUS file for VCRSs with useful added information
    bus_df_mutation_path = f"{kb_count_vcrs_dir}/bus_df.{save_type}"
    if not os.path.exists(bus_df_mutation_path):
        logger.info("Making VCRS BUS df")
        bus_df_mutation = make_bus_df(
            kb_count_out=kb_count_vcrs_dir,
            fastq_file_list=fastq_file_list,
            technology=technology,
            t2g_file=None,
            mm=mm,
            parity=parity,
            bustools=bustools,
            fastq_sorting_check_only=True,
            chunksize=None,
            correct_barcodes_of_hamming_distance_one=True,
            save_type=save_type,
            add_fastq_headers=add_fastq_headers,
            add_counted_in_count_matrix=False,
            save_transcript_names=False,
            strip_version_number_from_genes=False
        )
        bus_df_mutation = bus_df_mutation[["barcode", "UMI", "read_index", "gene_names"]].copy()
    else:
        logger.info("Loading in VCRS BUS df from existing file")
        if save_type == "parquet":
            bus_df_mutation = pd.read_parquet(bus_df_mutation_path, columns=["barcode", "UMI", "read_index", "gene_names"])
        elif save_type == "csv":
            bus_df_mutation = pd.read_csv(bus_df_mutation_path, usecols=["barcode", "UMI", "read_index", "gene_names"])
        else:
            raise ValueError(f"Unsupported save type: {save_type}")

    bus_df_mutation.rename(columns={"gene_names": "vcrs_names"}, inplace=True)

    #* merge adata.var into bus_df_mutation if helpful (i.e., if I had a var_id_column that I must map to HGVS format and/or I previously determined gene_id)
    columns_to_merge = {"vcrs_id"}
    first_vcrs_name = bus_df_mutation.loc[0, "vcrs_names"][0]
    if not re.fullmatch(HGVS_pattern_general, first_vcrs_name):  #* if the user used var_id_column in vk ref, then ensure that my vcrs_header column is in chained HGVS format
        columns_to_merge.add("vcrs_header")  # only add vcrs_header if the first vcrs_name is not in HGVS format
    if gene_id_column in adata.var.columns:
        columns_to_merge.add(gene_id_column)
    if len(columns_to_merge) <= 1:
        columns_to_merge = []
    if len(columns_to_merge) > 1 or gene_id_column not in bus_df_mutation.columns:
        logger.info("Merging adata.var into VCRS BUS df")
        bus_df_mutation = merge_bus_df_and_adata_var(bus_df=bus_df_mutation, adata_var=adata.var, vcrs_column_bus="vcrs_names", vcrs_column_adata="vcrs_id", gene_id_column=gene_id_column, variant_source=variant_source, seq_id_column=seq_id_column, reference_genome_t2g=t2g_standard, gtf=gtf, columns_to_merge=columns_to_merge)
        first_vcrs_name = bus_df_mutation.loc[0, "vcrs_names"][0]

    #* create a dataframe of the BUS file for normal reference genome
    bus_df_standard_path = f"{kallisto_quant_reference_genome_dir}/bus_df.{save_type}"
    if not os.path.exists(bus_df_standard_path):
        logger.info("Making standard reference genome BUS df")
        bus_df_standard = make_bus_df(
            kb_count_out=kallisto_quant_reference_genome_dir,
            fastq_file_list=fastq_file_list,
            technology=technology,
            t2g_file=t2g_standard,
            mm=False,  # doesn't matter if mm True or False - mm'ed reads will appear in the BUS file regardless; mm only affects count matrix, but I don't care about count matrix
            parity=parity,
            bustools=bustools,
            fastq_sorting_check_only=True,
            chunksize=None,
            correct_barcodes_of_hamming_distance_one=True,
            add_fastq_headers=add_fastq_headers,
            add_counted_in_count_matrix=False,
            save_transcript_names=False,
            strip_version_number_from_genes=False
        )
        bus_df_standard = bus_df_standard[["barcode", "read_index", "gene_names"]].copy()
    else:
        logger.info("Loading in standard reference genome BUS df from existing file")
        if save_type == "parquet":
            bus_df_standard = pd.read_parquet(bus_df_standard_path, columns=["barcode", "read_index", "gene_names"])
        elif save_type == "csv":
            bus_df_standard = pd.read_csv(bus_df_standard_path, usecols=["barcode", "read_index", "gene_names"])
        else:
            raise ValueError(f"Unsupported save type: {save_type}")
        
    bus_df_standard.rename(columns={"gene_names": "genes_standard"}, inplace=True)
    
    #* strip off the version number from gene ID
    logger.info("Stripping off the version number from gene ID")
    if "." in bus_df_mutation['genes_vcrs'].iloc[0][0][0]:
        bus_df_mutation["genes_vcrs"] = bus_df_mutation["genes_vcrs"].apply(lambda outer: [[s.partition(".")[0] for s in inner] for inner in outer])
    if "." in bus_df_standard['genes_standard'].iloc[0][0]:
        bus_df_standard["genes_standard"] = bus_df_standard["genes_standard"].apply(lambda lst: [s.split(".")[0] for s in lst])

    #* merge normal genome read alignments (gene_names from its bus df) into VCRS's bus df by barcode + read_index
    logger.info("Merging VCRS BUS df with standard reference genome BUS df")
    bus_df = bus_df_mutation.merge(bus_df_standard, on=["barcode", "read_index"], how="left")  # will have columns barcode, UMI, read_index, vcrs_names, genes_vcrs, genes_standard
    del bus_df_mutation, bus_df_standard
    mask = bus_df['genes_standard'].isna()
    bus_df['pseudoaligns_to_reference_genome'] = ~mask    # will have columns barcode, UMI, read_index, vcrs_names, genes_vcrs, genes_standard, pseudoaligns_to_reference_genome
    bus_df.loc[mask, 'genes_standard'] = pd.Series([[] for _ in range(mask.sum())], index=bus_df[mask].index)  # replace NaNs in genes_standard with empty lists

    #* remove barcode padding
    logger.info("Removing barcode padding")
    bus_df.rename(columns={"barcode": "barcode_with_padding"}, inplace=True)
    true_barcode_length = len(adata.obs.index[0])  # in contrast to the length of 32 in BUS file
    bus_df["barcode"] = bus_df["barcode_with_padding"].str[-true_barcode_length:]  # removes padding - keeps only the last true_barcode_length characters
    bus_df.drop(columns=["barcode_with_padding"], inplace=True)

    #* if parity == paired and vcrs_parity == single: (1) make a df copy; (2) groupby same barcode + read_index; (3) take union of mapped VCRSs, and the union of mapped genes; (4) assign these exclusively to the first read of the pair, and give the 2nd read of the pair nothing
    with open(f"{kb_count_vcrs_dir}/kb_info.json", 'r') as f:
        kb_info_data_vcrs = json.load(f)
    if "--parity paired" in kb_info_data_vcrs.get("call", ""):
        vcrs_parity = "paired"  # same as parity_kb_count in vk count/clean
    else:
        vcrs_parity = "single"

    with open(f"{kallisto_quant_reference_genome_dir}/kb_info.json", 'r') as f:
        kb_info_data_normal = json.load(f)
    if "--parity paired" in kb_info_data_normal.get("call", ""):
        normal_parity = "paired"
    else:
        normal_parity = "single"

    if technology.upper() not in {"BULK", "SMARTSEQ2"}:
        parity, vcrs_parity, normal_parity = None, None, None

    # override adata
    if parity == "paired" and vcrs_parity == "single" and not adata.uns.get("corrected_barcodes", None):  # eg convert ["AAA", "AAC", "AAG", "AAT"] to ["AAA", "AAC"]
        new_index = adata.obs.index[:len(adata.obs.index) // 2]
        new_obs = pd.DataFrame(index=new_index)
        adata = ad.AnnData(
            X=csr_matrix((len(new_obs), adata.shape[1])),
            obs=new_obs,
            var=adata.var.copy(),
            uns=adata.uns.copy()
        )

    if not (parity == "paired" and vcrs_parity == "single"):
        avoid_paired_double_counting = False

    if parity == "paired" and (normal_parity == "single" or (vcrs_parity == "single" and avoid_paired_double_counting)):
        logger.info("Adjusting adata to adjust for paired-end technology run in single-end mode")
        # (1) make a df copy
        bus_df_copy = bus_df[["barcode", "read_index", "genes_standard"]].copy()
        bus_df_copy["original_sets_populated"] = bus_df_copy["genes_standard"].apply(lambda x: len(x) > 0)

        # (2) groupby same barcode + read_index; (3) take union of mapped VCRSs, and the union of mapped genes
        agg_dict = {}
        if vcrs_parity == "single" and avoid_paired_double_counting:
            agg_dict["vcrs_names"] = lambda x: sorted(set.intersection(*map(set, x))),  # take the intersection of vcrs_names to determine which would be double-counted (should be few or none)
        if normal_parity == "single":
            agg_dict["genes_standard"] = lambda x: sorted(set.union(*map(set, x)))
            agg_dict["original_sets_populated"] = "all"  # Logical AND
        
        columns_to_keep = agg_dict.keys()
        if list(columns_to_keep) != bus_df_copy.columns.tolist():
            bus_df_copy = bus_df_copy[list(columns_to_keep)].copy()

        bus_df_copy = bus_df_copy.groupby(["barcode", "read_index"]).agg(agg_dict).reset_index()

        double_counting_dict = {}
        if vcrs_parity == "single" and avoid_paired_double_counting:
            bus_df_copy2 = bus_df_copy.loc[bus_df_copy["vcrs_names"].apply(lambda x: len(x) > 0)]  # keep only the ones that would be double-counted
            if not bus_df_copy2.empty:
                double_counting_dict = {
                    (row["barcode"], row["read_index"]): {name: False for name in row["vcrs_names"]}
                    for _, row in bus_df_copy2.iterrows()
                }  # this dict has the read identifier as the key (read index and barcode), and the value is a dict, where the key of the inner dict is the VCRS(s) that would be double-counted as a result of the parity paired and vcrs_parity single thing, and the value is whether or not this has been added to the count matrix by one of these pairs yet (default False, and will update to True as I go through count matrix) - eg {("AAACCTGAGTACGCCC", 42): {"VCR1": False, "VCR2": False}, ("AAACCTGAGGTGTTAG", 17): {"VCR3": False}, ...}
            else:
                avoid_paired_double_counting = False
            del bus_df_copy2
            bus_df_copy.drop(columns=["vcrs_names"], inplace=True)

        # (3.5) if drop_reads_where_the_pairs_mapped_to_different_genes=True, drop rows where both original sets were non-empty but the intersection is empty (i.e., each pair mapped to a different gene)
        if normal_parity == "single":
            if drop_reads_where_the_pairs_mapped_to_different_genes:  # this is the normal behavior for kb count with --parity paired - but because I have more information (I have pair1 genome, pair2 genome, pair1 VCRS, and pair2 VCRS, I opt to keep this default off)
                bus_df_copy = bus_df_copy[bus_df_copy["original_sets_populated"] == False]  # drop rows where both original sets were non-empty but the intersection is empty
            bus_df_copy.drop(columns=["original_sets_populated"], inplace=True)  # drop original_sets_populated column

            # (4) assign these exclusively to the first read of the pair, and give the 2nd read of the pair nothing
            bus_df = bus_df[["barcode", "UMI", "read_index", "vcrs_names", "genes_vcrs", "pseudoaligns_to_reference_genome"]].copy()
            bus_df = bus_df.merge(bus_df_copy, on=["barcode", "read_index"], how="inner")  # inner join instead of left because I dropped rows above where the original sets were non-empty but the intersection is empty
        del bus_df_copy

    #* groupby barcode + UMI, taking the list of read_index, the mode of vcrs_names, and the union of genes_standard
    if technology.upper() not in {"BULK", "SMARTSEQ2"}:  # all technologies that have UMIs
        logger.info("Grouping by barcode and UMI to aggregate read indices, VCRS names, and genes")

        # groupby barcode + UMI, taking the list of read_index, the mode of vcrs_names (and the correspond genes_vcrs), and the union of genes_standard
        bus_df["original_index"] = np.arange(len(bus_df))
        bus_df = bus_df.groupby(["barcode", "UMI"], sort=False).apply(barcode_and_umi_agg).reset_index().sort_values("original_index").drop(columns="original_index").reset_index(drop=True)
        # bus_df = bus_df.groupby(["barcode", "UMI"]).agg({
        #     "read_index": list,  
        #     "vcrs_names": lambda x: mode(x, keepdims=True)[0][0],  
        #     "genes_standard": lambda x: sorted(set.union(*map(set, x)))  
        # }).reset_index()
    
    #* if skip_transcripts_without_genes=True, then for any rows without a gene name for the transcript found in the t2g (i.e., the genes_vcrs column is an ENST, which would be guaranteed to be tossed if the read aligned to any gene in the reference genome), set the reference genome gene alignment to empty list so these don't get tossed (it's more likely an incomplete t2g than a bad alignment)
    if skip_transcripts_without_genes:
        mask = bus_df['genes_vcrs'].astype(str).str.contains("ENST", na=False)
        bus_df.loc[mask, 'genes_standard'] = pd.Series([[] for _ in range(mask.sum())], index=bus_df[mask].index)
        bus_df.loc[mask, "pseudoaligns_to_reference_genome"] = False

    #* take the set of normal reference genome alignment genes in a new column
    logger.info("Taking the set of normal reference genome alignment genes")
    bus_df["genes_standard_set"] = bus_df["genes_standard"].apply(lambda x: set(x))

    if mistake_ratio == 0:
        raise ValueError("Mistake ratio must be in the range (0, 1]")
    if mistake_ratio:
        # Make genes_vcrs_set (flattened union of inner lists)
        bus_df['genes_vcrs_set'] = bus_df['genes_vcrs'].apply(
            lambda lst: set(g for inner in lst for g in inner)
        )

        # Make genes_intersection column
        bus_df['genes_intersection'] = bus_df.apply(
            lambda row: row['genes_vcrs_set'] & row['genes_standard_set'],
            axis=1
        )

        # Filter rows where genes_standard_set is non-empty AND genes_intersection is empty
        filtered_df = bus_df[
            (bus_df['genes_standard_set'].apply(bool)) &
            (bus_df['genes_intersection'].apply(lambda s: len(s) == 0))
        ]

        # Count gene occurrences from genes_standard_set
        from collections import Counter
        gene_counter_original = Counter(g for s in bus_df['genes_vcrs_set'] for g in s)
        gene_counter = Counter(g for s in filtered_df['genes_vcrs_set'] for g in s)

        gene_ratio_dict = {
            gene: gene_counter[gene] / gene_counter_original[gene]
            for gene in gene_counter
            if gene_counter_original[gene] > 0
        }

        set_of_genes_with_highest_ratio_of_mistakes = {gene for gene, ratio in gene_ratio_dict.items() if ratio > mistake_ratio}
    else:
        set_of_genes_with_highest_ratio_of_mistakes = {}

    #* loop through bus df (as zipped iterators)
    row_indices = []
    col_indices = []
    data_values = []

    number_of_counts_added = 0
    number_of_counts_removed = 0
    number_of_reads_changed = 0
    mm_original = True if "--mm" in kb_info_data_vcrs.get("call", "") else False
    logging_level = logger.getEffectiveLevel()

    logger.info("Looping through bus_df to adjust adata")  #? consider just updating adata instead of making from scratch - would compare final_counts to original counts (1 / vcrs_names_list)
    for vcrs_names_list, genes_vcrs_outer_list, genes_standard_set, barcode, pseudoaligns_to_reference_genome, read_index in tqdm(zip(bus_df["vcrs_names"], bus_df["genes_vcrs"], bus_df["genes_standard_set"], bus_df["barcode"], bus_df["pseudoaligns_to_reference_genome"], bus_df["read_index"]), total=len(bus_df), desc="Adjusting adata by looping through bus_df"):
        #* look at all VCRSs to which the read mapped, and keep only VCRSs whose corresponding gene is in the gene_intersection column above
        vcrs_names_list_final = []
        for vcrs_name, gene_vcrs_inner_list in zip(vcrs_names_list, genes_vcrs_outer_list):
            gene_vcrs_inner_set = set(gene_vcrs_inner_list)
            if "dlist" in gene_vcrs_inner_set:
                continue
            
            if gene_vcrs_inner_set & genes_standard_set or (count_reads_that_dont_pseudoalign_to_reference_genome and not pseudoaligns_to_reference_genome) or (mistake_ratio is not None and len(gene_vcrs_inner_set & set_of_genes_with_highest_ratio_of_mistakes)):  # (1) non-empty intersection between gene corresponding to VCRS and reference genome gene OR (2) if the read does not pseudoalign to the reference genome, count it anyway if the flag is True OR (3) if the read belongs to a variant corresponding to a gene with a high ratio of mistakes (and I am checking for this condition)
                vcrs_names_list_final.append(vcrs_name)
        
        if logging_level <= 20:  # since I only want to log this if the logging level is higher than INFO
            number_of_counts_added, number_of_counts_removed, number_of_reads_changed = normal_genome_validation_tallying(vcrs_names_list_final=vcrs_names_list_final, vcrs_names_list=vcrs_names_list, number_of_counts_added=number_of_counts_added, number_of_counts_removed=number_of_counts_removed, number_of_reads_changed=number_of_reads_changed, mm=mm, mm_original=mm_original)
            
        length_vcrs_names_list_final = len(vcrs_names_list_final)
        if length_vcrs_names_list_final == 0 or (not mm and length_vcrs_names_list_final > 1):
            continue
        # counts_final = 1 if not mm else 1 / length_vcrs_names_list_final
        counts_final = 1  # I use 1 rather than 1 / len(vcrs_names_list_final) because it is perfectly valid for a single read to map to 2 variants            
        row_idx = adata.obs.index.get_loc(barcode)

        for vcrs_name_final in vcrs_names_list_final:
            if avoid_paired_double_counting:
                if double_counting_dict.get((barcode, read_index), {}).get(vcrs_name_final, None):  # a 3-state system - True means it has already been counted, False means it has not already been counted, and None means it is not in the dict at all
                    continue  # if it has already been counted, then skip this iteration
                elif double_counting_dict.get((barcode, read_index), {}).get(vcrs_name_final, None) is False:
                    double_counting_dict[(barcode, read_index)][vcrs_name_final] = True  # if it has not been counted yet, then set to True (so it won't be counted in the future) but continue through the iteration
            
            col_idx = adata.var.index.get_loc(vcrs_name_final)
            
            row_indices.append(row_idx)
            col_indices.append(col_idx)
            data_values.append(counts_final)

    if logging_level <= 20:  # since I only want to log this if the logging level is higher than INFO
        logger.info(f"Number of counts added: {number_of_counts_added}")
        logger.info(f"Number of counts removed: {number_of_counts_removed}")
        logger.info(f"Number of reads that with changed count behavior: {number_of_reads_changed} / {len(bus_df)} ({100 * number_of_reads_changed / len(bus_df):.2f}%)")

    if len(data_values) == 0:
        logger.warning("No valid updates found in the bus_df. Returning the original AnnData object.")
        return adata

    #* copy adata.var and adata.obs and adata.uns as-is, all in the same order; initialize a sparse matrix (to remake adata from scratch)
    logger.info("Constructing the new AnnData object with updated counts")
    # Construct a sparse matrix with the updates
    data_values = np.array(data_values, dtype=np.float64)
    update_matrix = csr_matrix((data_values, (row_indices, col_indices)), shape=adata.shape)

    adata_new = ad.AnnData(
        X=update_matrix,  # Sparse zero matrix
        obs=adata.obs.copy(),
        var=adata.var.copy(),
        uns=adata.uns.copy()
    )

    #* save adata
    if adata_output_path:
        logger.info(f"Saving adjusted AnnData object to {adata_output_path}")
        adata_new.write(adata_output_path)

    return adata_new


# ======================================================================================
# pseudobam-based normal-genome QC (replacement for adjust_variant_adata_by_normal_gene_matrix)
#
# Instead of running a full second `kb count` against the reference genome and comparing
# read->gene assignments, this aligns the reads directly with `kallisto quant --pseudobam`
# (or `--genomebam` when the VCRS reference is genomic DNA but the headers are transcript
# HGVSc) and compares each read's actual alignment coordinate to the HGVS coordinate of the
# VCRS it was assigned to. Reads whose true alignment is inconsistent with their assigned
# variant's locus are dropped from the VCRS BUS file, and the count matrix is regenerated
# with `bustools count`.
# ======================================================================================

_FASTA_EXTS = (".fa", ".fasta", ".fna", ".fa.gz", ".fasta.gz", ".fna.gz")


def needs_k64(kallisto_path, index_path):
    result = subprocess.run(
        [kallisto_path, "inspect", index_path],
        capture_output=True,
        text=True,
    )

    output = result.stdout + result.stderr

    if "Length k of k-mers cannot exceed or be equal to 32" in output:
        return True

    return False

def _resolve_kb_binary(binary, name):
    """Return `binary` if provided, else resolve the path bundled with kb-python via `kb info`."""
    if binary:
        return binary
    cmd = f"kb info | grep '{name}:' | awk '{{print $3}}' | sed 's/[()]//g'"
    try:
        resolved = subprocess.run(cmd, shell=True, executable="/bin/bash", stdout=subprocess.PIPE, text=True, check=True).stdout.strip()
    except subprocess.CalledProcessError:
        resolved = ""
    return resolved or name


def _infer_reference_sequences_type(sequences):
    """Best-effort guess of whether the VCRS reference was built from RNA (transcriptome) or DNA (genome).

    Returns 'rna', 'dna', or None (unknown). Feeds the pseudobam-QC alignment strategy (see
    _determine_pseudobam_mode). Callers may pass an explicit value to override this heuristic.
    """
    if not isinstance(sequences, (str, Path)):
        return None
    s = str(sequences).lower()
    if any(k in s for k in ("cdna", "cds", "transcript", "_rna", "rna.")):
        return "rna"
    if any(k in s for k in ("genome", "dna", "primary_assembly", "toplevel")):
        return "dna"
    return None


def _determine_pseudobam_mode(adata_var, variant_source=None, reference_sequences_type=None, sequences=None, reads_type=None):
    """Resolve the pseudobam-QC strategy from three independent axes.

    The three axes are:
      - reads_type ('rna'/'dna'): the sequencing molecule. RNA-seq reads span spliced exon
        junctions, so they cannot pseudoalign to a raw genome; DNA/genomic reads align contiguously.
        Taken from the `technology` (the virtual "dna" technology => 'dna', any RNA technology =>
        'rna'). When None, it defaults to `reference_sequences_type` (i.e. reads assumed to match the
        reference: dna-dna or rna-rna) so the --genomebam path is opt-in for the minority
        RNA-reads-against-DNA-reference case.
      - reference_sequences_type ('rna'/'dna'): whether the VCRS reference was built from a
        transcriptome (RNA) or a genome (DNA). Determines the coordinate space alignments land in.
      - header_type ('hgvsc'/'hgvsg'): the coordinate system the VCRS HGVS headers are written in.

    Returns a dict with:
      - header_type                 : 'hgvsc' | 'hgvsg'
      - reference_sequences_type    : 'rna' | 'dna' | None
      - reads_type                  : 'rna' | 'dna' | None
      - use_genomebam (bool)        : True iff RNA reads against a DNA/genome reference. In that case
                                      the reads are aligned to a cDNA index (extracted from the genome
                                      + GTF via `kb ref --workflow standard`) and projected back onto
                                      the genome with `kallisto quant --genomebam`. Otherwise the
                                      reference is indexed directly (`--workflow custom`) and aligned
                                      with `--pseudobam`.
      - convert_transcript_to_genome (bool) : True iff HGVSc headers are compared in genome space
                                      (i.e. a DNA/genome reference), so the transcript-coordinate loci
                                      must be lifted to genome coordinates to match the alignments.
                                      Independent of use_genomebam: DNA reads + DNA ref + HGVSc
                                      headers also align in genome space (via --pseudobam) and need
                                      the same lift, while RNA reads + DNA ref + HGVSg headers use
                                      --genomebam but need no lift (headers are already genomic).
    """
    # header type
    if variant_source in ("transcriptome", "cdna", "c", "hgvsc"):
        header_type = "hgvsc"
    elif variant_source in ("genome", "g", "hgvsg"):
        header_type = "hgvsg"
    else:
        sample = adata_var.index.to_series().astype(str).head(2000).str.split(";").explode()
        has_c = sample.str.contains(r":c\.", regex=True, na=False).any()
        has_g = sample.str.contains(r":g\.", regex=True, na=False).any()
        if has_c and has_g:
            raise ValueError("The VCRS headers contain a mix of HGVSc (c.) and HGVSg (g.) variants. Please split them and run the pseudobam QC separately per coordinate system, or pass variant_source explicitly.")
        header_type = "hgvsg" if has_g else "hgvsc"

    ref_type = reference_sequences_type or _infer_reference_sequences_type(sequences)
    if reads_type is None:
        reads_type = ref_type  # assume reads match the reference (dna-dna or rna-rna)

    # HGVSg headers are genomic coordinates, so the reference is a genome regardless of what the
    # `sequences` path looked like -> alignments (and the loci they are compared to) are genome-space.
    reference_is_genome_space = (ref_type == "dna") or (header_type == "hgvsg")

    use_genomebam = (reads_type == "rna") and reference_is_genome_space
    convert_transcript_to_genome = (header_type == "hgvsc") and reference_is_genome_space

    return {
        "header_type": header_type,
        "reference_sequences_type": ref_type,
        "reads_type": reads_type,
        "use_genomebam": use_genomebam,
        "convert_transcript_to_genome": convert_transcript_to_genome,
    }


def _build_reference_genome_index(reference_genome_index, use_genomebam, sequences=None, gtf=None, threads=2, kallisto="kallisto", bustools="bustools"):
    """Build the standard reference kallisto index for the pseudobam QC if it does not already exist.

    The index space is dictated by `use_genomebam`:
      - use_genomebam=True (RNA reads against a DNA/genome `sequences` reference): the RNA reads span
        spliced junctions and cannot pseudoalign to the raw genome, so they are aligned to a cDNA index
        and then projected back onto the genome with kallisto's --genomebam. The cDNA is extracted from
        the genome FASTA + GTF with `kb ref --workflow standard` (i.e. a cDNA index):
        `kb ref --workflow standard -i INDEX -g t2g.txt -f1 cdna.fa -t THREADS genome.fa genome.gtf`.
      - use_genomebam=False (RNA reads + cDNA reference, or DNA reads + genome reference): `sequences`
        is already in the space the reads align to, so it is indexed directly with `kb ref --workflow custom`.
    """
    from .varseek_build_utils import run_kb_ref  # local import to avoid a circular import at module load time

    if os.path.exists(str(reference_genome_index)):
        return reference_genome_index
    if not sequences or not os.path.exists(str(sequences)):
        raise ValueError(f"reference_genome_index '{reference_genome_index}' does not exist and cannot be built because `sequences` was not provided (or does not exist).")

    out_dir = os.path.dirname(str(reference_genome_index)) or "."
    os.makedirs(out_dir, exist_ok=True)

    if use_genomebam:
        if not gtf or not os.path.exists(str(gtf)):
            raise ValueError("Building the pseudobam reference index for the RNA-reads-against-a-DNA-(genome)-`sequences` case requires a `gtf`; the cDNA is extracted from the genome FASTA + GTF with `kb ref --workflow standard`.")
        t2g = os.path.splitext(str(reference_genome_index))[0] + "_t2g.txt"
        f1 = os.path.join(out_dir, "reference_genome_index_cdna.fasta")
        logger.info(f"reference_genome_index '{reference_genome_index}' not found; building a cDNA index from genome '{sequences}' + gtf '{gtf}' with `kb ref --workflow standard`.")
        run_kb_ref(index=str(reference_genome_index), workflow="standard", dna_fasta=str(sequences), t2g=str(t2g), f1=str(f1), gtf=str(gtf), threads=threads, kallisto=kallisto, bustools=bustools, tmp=os.path.join(out_dir, "kb_ref_tmp_pseudobam"))
    else:
        logger.info(f"reference_genome_index '{reference_genome_index}' not found; building it from '{sequences}' with `kb ref --workflow custom`.")
        run_kb_ref(index=str(reference_genome_index), workflow="custom", dna_fasta=str(sequences), threads=threads, kallisto=kallisto, bustools=bustools)
    return reference_genome_index


def _generate_chromosomes_file(out_path, sequences=None, gtf=None):
    """Write a tab-separated <chromosome>\t<length> file (needed by kallisto --genomebam).

    Lengths come from the genome FASTA (`sequences`) when available (exact), otherwise from the
    maximum coordinate per seqname in the GTF (an upper-bound approximation that is sufficient
    for projecting alignments).
    """
    sizes = {}
    if sequences is not None and isinstance(sequences, (str, Path)) and os.path.isfile(str(sequences)) and str(sequences).endswith(_FASTA_EXTS):
        fa = pyfastx.Fasta(str(sequences), build_index=True)
        for name in fa.keys():
            sizes[str(name).split()[0].split(".")[0]] = len(fa[name])
    elif gtf is not None and os.path.isfile(str(gtf)):
        gdf = pd.read_csv(str(gtf), sep="\t", comment="#", header=None, usecols=[0, 4], names=["seqname", "end"], dtype={"seqname": str})
        gdf["seqname"] = gdf["seqname"].str.split(".").str[0]
        sizes = gdf.groupby("seqname")["end"].max().to_dict()
    else:
        raise ValueError("A genome FASTA (`sequences`) or a `gtf` is required to generate the chromosomes file for --genomebam.")
    with open(out_path, "w", encoding="utf-8") as f:
        for name, length in sizes.items():
            f.write(f"{name}\t{int(length)}\n")
    return out_path


def _group_fastqs_for_quant(fastq_file_list, technology, parity, barcodes):
    """Yield (group_barcode_or_None, [fastq files for this quant run], is_single_end).

    For bulk / smart-seq (barcode == sample) each sample is quantified separately so alignments
    can be attributed to a barcode unambiguously and read-name collisions across samples cannot
    over-validate. For barcoded single-cell technologies the cDNA-read files are quantified in a
    single run and each read's barcode is taken from the BUS file downstream (group barcode None).
    """
    tech = technology.upper()
    info = technology_info[tech]
    num_files = info["num_files"]
    if isinstance(num_files, dict):
        num_files = num_files[parity]
    transcript_file_index = info["transcript_file_index"]
    barcode_position = info["barcode_position"]

    groups = []
    if barcode_position is None:  # bulk / smart-seq2 -> one sample per barcode
        n_samples = len(fastq_file_list) // num_files
        for s in range(n_samples):
            files = [str(fastq_file_list[s * num_files + j]) for j in range(num_files)]
            bc = barcodes[s] if barcodes is not None and s < len(barcodes) else None
            groups.append((bc, files, num_files == 1))
    else:  # barcoded single-cell -> quant the cDNA reads only, in one pooled run
        cdna_files = [str(fastq_file_list[i]) for i in range(len(fastq_file_list)) if i % num_files == transcript_file_index]
        groups.append((None, cdna_files, True))
    return groups


def _run_kallisto_quant_group(kallisto, reference_genome_index, out_dir, fastq_files, is_single_end, use_genomebam, threads=2, fragment_length=51, fragment_sd=5, gtf=None, chromosomes=None, overwrite=False):
    """Run `kallisto quant` for one fastq group and return the path to pseudoalignments.bam."""
    os.makedirs(out_dir, exist_ok=True)
    bam_path = os.path.join(out_dir, "pseudoalignments.bam")
    if os.path.exists(bam_path) and not overwrite:
        logger.info(f"Reusing existing pseudoalignments.bam at {bam_path}")
        return bam_path

    command = [kallisto, "quant", "-i", str(reference_genome_index), "-o", out_dir, "-t", str(threads)]
    if is_single_end:
        command += ["--single", "-l", str(fragment_length), "-s", str(fragment_sd)]
    if use_genomebam:
        if not gtf:
            raise ValueError("--genomebam requires a `gtf` (RNA reads against a DNA/genome reference).")
        command += ["--genomebam", "--gtf", str(gtf), "--chromosomes", str(chromosomes)]
    else:
        command += ["--pseudobam"]
    command += [str(f) for f in fastq_files]

    logger.info(f"Running kallisto quant: {' '.join(command)}")
    subprocess.run(command, check=True)
    if not os.path.exists(bam_path):
        raise FileNotFoundError(f"kallisto quant did not produce {bam_path}. Check that this kallisto build supports --pseudobam/--genomebam.")
    return bam_path


# --------------------------------------------------------------------------------------
# True (non-pseudo) alignment for the pseudobam QC: bowtie2 / STAR instead of kallisto quant.
# The choice is dictated by the same `use_genomebam` signal that picks --genomebam over
# --pseudobam, because it encodes the one case that needs a splice-aware aligner: RNA reads
# against a genome reference (reads span exon-exon junctions). Everything else (DNA reads
# against a genome, RNA reads against a transcriptome) aligns contiguously -> bowtie2.
# The commands mirror the ones in vk denovo.
# --------------------------------------------------------------------------------------

_STAR_INDEX_MARKER_FILES = ("SA", "SAindex", "Genome", "chrName.txt")
_BOWTIE2_INDEX_SUFFIXES = (".1.bt2", ".1.bt2l")
_BOWTIE2_INDEX_FILE_PATTERN = re.compile(r"\.(?:rev\.)?\d+\.bt2l?$")


def _bowtie2_index_prefix(reference_genome_index):
    """`bowtie2 -x` takes the index *prefix*; accept either the prefix or one of its .bt2 files."""
    return _BOWTIE2_INDEX_FILE_PATTERN.sub("", str(reference_genome_index))


def _detect_reference_genome_index_type(reference_genome_index):
    """Return 'STAR' | 'bowtie2' | 'kallisto' | None for whatever sits at `reference_genome_index`.

    None means nothing recognizable is on disk yet (and the path does not name a kind by
    extension), so the caller is free to build whichever index type it needs there.
    """
    if not reference_genome_index:
        return None
    path = str(reference_genome_index)
    if os.path.isdir(path):
        return "STAR" if set(os.listdir(path)) & set(_STAR_INDEX_MARKER_FILES) else None
    if _BOWTIE2_INDEX_FILE_PATTERN.search(path) or any(os.path.exists(_bowtie2_index_prefix(path) + suffix) for suffix in _BOWTIE2_INDEX_SUFFIXES):
        return "bowtie2"
    if path.endswith(".idx"):
        return "kallisto"
    return "kallisto" if os.path.isfile(path) else None


def _infer_read_length_from_fastq(fastq_file):
    """Read length of the first record of `fastq_file` (STAR's --sjdbOverhang wants read length - 1)."""
    opener = gzip.open if str(fastq_file).endswith(".gz") else open
    with opener(str(fastq_file), "rt") as f:
        f.readline()  # skip the header line
        return len(f.readline().strip())


def _build_true_aligner_index(reference_genome_index, true_aligner, sequences=None, gtf=None, threads=2, read_length=None, star="STAR", bowtie2_build="bowtie2-build"):
    """Build the STAR genome directory / bowtie2 index if it does not already exist.

    Returns the handle the aligner actually wants: the genome directory for STAR, the index
    prefix for bowtie2.
    """
    from varseek.utils.varseek_denovo_utils import run as run_shell  # local import to keep the module import graph acyclic

    if true_aligner == "STAR":
        genome_dir = str(reference_genome_index)
        if os.path.isdir(genome_dir) and set(os.listdir(genome_dir)) & set(_STAR_INDEX_MARKER_FILES):
            return genome_dir
        if not sequences or not os.path.exists(str(sequences)):
            raise ValueError(f"The STAR genome index '{genome_dir}' does not exist and cannot be built because `sequences` was not provided (or does not exist).")
        if not gtf or not os.path.exists(str(gtf)):
            raise ValueError("Building a STAR genome index requires a `gtf` (the splice junctions are what make STAR able to align RNA reads to a genome).")
        os.makedirs(genome_dir, exist_ok=True)
        sjdb_overhang = (read_length or 100) - 1
        # STAR's default --genomeSAindexNbases (14) only suits genome-scale references; STAR itself
        # recommends min(14, log2(genomeLength)/2 - 1) and errors out on small references otherwise.
        genome_length = max(os.path.getsize(str(sequences)), 2)
        sa_index_nbases = max(4, min(14, int(np.log2(genome_length) / 2 - 1)))
        logger.info(f"STAR genome index '{genome_dir}' not found; building it from '{sequences}' + gtf '{gtf}'.")
        run_shell(f"{star} --runThreadN {threads} --runMode genomeGenerate --genomeDir {shlex.quote(genome_dir)} --genomeFastaFiles {shlex.quote(str(sequences))} --sjdbGTFfile {shlex.quote(str(gtf))} --sjdbOverhang {sjdb_overhang} --genomeSAindexNbases {sa_index_nbases} --limitSjdbInsertNsj 1000000 --limitBAMsortRAM 0")
        return genome_dir

    prefix = _bowtie2_index_prefix(reference_genome_index)
    if any(os.path.exists(prefix + suffix) for suffix in _BOWTIE2_INDEX_SUFFIXES):
        return prefix
    if not sequences or not os.path.exists(str(sequences)):
        raise ValueError(f"The bowtie2 index '{prefix}' does not exist and cannot be built because `sequences` was not provided (or does not exist).")
    os.makedirs(os.path.dirname(prefix) or ".", exist_ok=True)
    logger.info(f"bowtie2 index '{prefix}' not found; building it from '{sequences}'.")
    run_shell(f"{bowtie2_build} {shlex.quote(str(sequences))} {shlex.quote(prefix)}")
    return prefix


def _run_true_aligner_group(true_aligner, reference_genome_index, out_dir, fastq_files, is_single_end, threads=2, read_length=None, overwrite=False, star="STAR", bowtie2="bowtie2", samtools="samtools", bowtie2_options="--xeq --very-sensitive"):
    """Align one fastq group with bowtie2/STAR and return the path to a coordinate-sorted BAM.

    Same contract as :func:`_run_kallisto_quant_group`: one BAM per group whose read names are the
    FASTQ headers, ready for :func:`_parse_pseudobam`.
    """
    from varseek.utils.varseek_denovo_utils import run as run_shell  # local import to keep the module import graph acyclic

    os.makedirs(out_dir, exist_ok=True)
    bam_path = os.path.join(out_dir, "aligned.sorted.bam")
    if os.path.exists(bam_path) and not overwrite:
        logger.info(f"Reusing existing alignment BAM at {bam_path}")
        return bam_path

    files = [str(f) for f in fastq_files]
    # a non-single-end group is a flat [R1, R2, R1, R2, ...] list (see _group_fastqs_for_quant)
    mates_1, mates_2 = (files, []) if is_single_end else (files[0::2], files[1::2])

    if true_aligner == "STAR":
        star_prefix = os.path.join(out_dir, "star_")
        sjdb_overhang = (read_length or 100) - 1
        reads_in = ",".join(mates_1) + (" " + ",".join(mates_2) if mates_2 else "")
        command = (
            f"{star} --runThreadN {threads} --genomeDir {shlex.quote(str(reference_genome_index))} "
            f"--readFilesIn {reads_in} --sjdbOverhang {sjdb_overhang} "
            f"--outFileNamePrefix {shlex.quote(star_prefix)} --outSAMtype BAM SortedByCoordinate "
            "--outSAMattributes NH HI AS nM MD --outSAMmapqUnique 60 --twopassMode Basic "
            "--limitSjdbInsertNsj 1000000 --limitBAMsortRAM 0"
        )
        if any(f.endswith(".gz") for f in files):
            command += " --readFilesCommand zcat"
        logger.info(f"Running STAR alignment: {command}")
        run_shell(command)
        os.replace(f"{star_prefix}Aligned.sortedByCoord.out.bam", bam_path)
    else:
        index_prefix = _bowtie2_index_prefix(reference_genome_index)
        reads_in = f"-1 {','.join(mates_1)} -2 {','.join(mates_2)}" if mates_2 else f"-U {','.join(mates_1)}"
        command = f"{bowtie2} {bowtie2_options} --threads {threads} -x {shlex.quote(index_prefix)} {reads_in} | {samtools} view -bS - | {samtools} sort -o {shlex.quote(bam_path)}"
        logger.info(f"Running bowtie2 alignment: {command}")
        # pipefail (hence bash) so a bowtie2 failure is not masked by samtools sort exiting 0 on empty input
        subprocess.run(f"set -o pipefail; {command}", shell=True, executable="/bin/bash", check=True)

    if not os.path.exists(bam_path):
        raise FileNotFoundError(f"{true_aligner} did not produce {bam_path}.")
    return bam_path


def _resolve_true_aligner(pseudobam_validation_aligner, use_genomebam):
    """Pick (and sanity-check) the true aligner for the requested mode.

    Splice-aware STAR is required exactly when `use_genomebam` is True (RNA reads against a genome
    reference, i.e. reads that span exon-exon junctions); bowtie2 is correct otherwise.
    """
    expected = "STAR" if use_genomebam else "bowtie2"
    requested = {"star": "STAR", "bowtie2": "bowtie2"}.get(str(pseudobam_validation_aligner).lower())
    if requested is not None and requested != expected:
        reason = (
            "splice-aware STAR is required to align RNA reads across exon-exon junctions on a genome reference"
            if expected == "STAR"
            else "bowtie2 is appropriate because no splicing occurs in this alignment"
        )
        raise ValueError(f"pseudobam_validation_aligner='{pseudobam_validation_aligner}' is incompatible with use_genomebam={use_genomebam}. Use pseudobam_validation_aligner='{expected}' ({reason}), or 'true' to select it automatically.")
    return expected


def _parse_pseudobam(bam_path):
    """Return {read_name: {'refs': set(ref), 'pos_by_ref': {ref: [1-based positions]}}}.

    Aggregates over all (primary + secondary) alignment records for each read so multimapping
    reads contribute the union of their alignment loci. Reference-name version suffixes are
    stripped so `ENST0001.3` and `ENST0001` compare equal.
    """
    read_to_aln = {}
    with pysam.AlignmentFile(bam_path, "rb") as bam:
        for read in bam.fetch(until_eof=True):
            if read.is_unmapped or read.reference_name is None:
                continue
            qname = read.query_name
            ref = read.reference_name.split(".")[0]
            pos = read.reference_start + 1  # pysam is 0-based; HGVS is 1-based
            entry = read_to_aln.get(qname)
            if entry is None:
                entry = {"refs": set(), "pos_by_ref": {}}
                read_to_aln[qname] = entry
            entry["refs"].add(ref)
            entry["pos_by_ref"].setdefault(ref, []).append(pos)
    return read_to_aln


def _vcrs_individual_headers_frame(adata_var):
    """Return a long DataFrame [vcrs_id, header] with one row per individual (semicolon-split) HGVS header."""
    df = pd.DataFrame(index=adata_var.index.copy())
    df["vcrs_id"] = adata_var.index.astype(str)
    first = str(adata_var.index[0]).split(";")[0]
    if re.search(r":[cg]\.", first):  # looks like an HGVS header (robust to version dots the strict pattern rejects)
        header_source = df["vcrs_id"]
    elif "vcrs_header" in adata_var.columns:
        header_source = adata_var["vcrs_header"].astype(str).values
    else:
        raise ValueError("Cannot recover HGVS headers: adata.var index is not HGVS-formatted and there is no 'vcrs_header' column.")
    out = pd.DataFrame({"vcrs_id": df["vcrs_id"].values, "header": pd.Series(header_source, index=df.index).str.split(";").values})
    out = out.explode("header", ignore_index=True)
    out["header"] = out["header"].astype(str)
    return out


def _build_vcrs_locus_map(adata_var, convert_transcript_to_genome, gtf=None, seq_id_column="seq_ID", var_column="mutation"):
    """Map each vcrs_id to the reference locus/loci where a genuine read should align.

    Returns {vcrs_id: {'refs': set(seq_id/chrom), 'intervals': {ref: [(lo, hi), ...]}}}, all in the
    same coordinate space as the pseudoalignments.bam. When `convert_transcript_to_genome` is True
    (HGVSc headers compared against a genome-space alignment), the transcript-coordinate loci are
    lifted to genome coordinates; otherwise the header coordinates are used as-is (transcript space
    for HGVSc against a cDNA reference, genome space for HGVSg).
    """
    frame = _vcrs_individual_headers_frame(adata_var)
    parts = frame["header"].str.split(":", n=1, expand=True)
    frame["seq_id"] = strip_gene_name_from_seq_id(parts[0]).str.split(".").str[0]  # drop any '(GENE)' annotation, then the version suffix
    frame["mutation"] = parts[1] if parts.shape[1] > 1 else None

    if convert_transcript_to_genome:
        model = build_transcript_exon_model(str(gtf))
        conv_in = pd.DataFrame({"vcrs_id": frame["vcrs_id"].values, seq_id_column: frame["seq_id"].values, var_column: frame["mutation"].values})
        converted, _ = convert_variants_transcript_to_genome(conv_in, seq_id_column, var_column, model)
        frame = pd.DataFrame({"vcrs_id": converted["vcrs_id"].values, "seq_id": converted[seq_id_column].astype(str).str.split(".").str[0].values, "mutation": converted[var_column].values})

    extracted = frame["mutation"].str.extract(mutation_pattern)
    pos_split = extracted[0].astype(str).str.split("_", expand=True)
    start = pd.to_numeric(pos_split[0].str.replace(r"[^0-9]", "", regex=True).replace("", np.nan), errors="coerce")
    if pos_split.shape[1] > 1:
        end = pd.to_numeric(pos_split[1].str.replace(r"[^0-9]", "", regex=True).replace("", np.nan), errors="coerce").fillna(start)
    else:
        end = start

    locus_map = {}
    for vcrs_id, seq_id, lo, hi in zip(frame["vcrs_id"], frame["seq_id"], start, end):
        if not isinstance(seq_id, str) or not seq_id:
            continue
        entry = locus_map.setdefault(vcrs_id, {"refs": set(), "intervals": {}})
        entry["refs"].add(seq_id)
        if pd.notna(lo):
            lo_i, hi_i = int(min(lo, hi)), int(max(lo, hi))
            entry["intervals"].setdefault(seq_id, []).append((lo_i, hi_i))
    return locus_map


def _read_alignment_is_consistent(aln, vcrs_locus, check_position, position_tolerance):
    """True if the read's actual alignment `aln` is consistent with `vcrs_locus` (same ref, and
    optionally within `position_tolerance` of a variant interval on that ref)."""
    shared_refs = aln["refs"] & vcrs_locus["refs"]
    if not shared_refs:
        return False
    if not check_position:
        return True
    for ref in shared_refs:
        intervals = vcrs_locus["intervals"].get(ref)
        if not intervals:
            return True  # no position annotation for this ref -> name match is all we can verify
        for pos in aln["pos_by_ref"].get(ref, ()):
            for lo, hi in intervals:
                if lo - position_tolerance <= pos <= hi + position_tolerance:
                    return True
    return False


def _last_byte(path):
    """Last byte of a file (b"" when empty) -- used to tell whether it ends in a newline."""
    with open(path, "rb") as f:
        f.seek(0, os.SEEK_END)
        if f.tell() == 0:
            return b""
        f.seek(-1, os.SEEK_END)
        return f.read(1)


def _recount_from_filtered_bus(kb_count_vcrs_dir, kept_records, new_ec_records, bustools, kallisto_threads, vcrs_t2g, new_counts_dir, mm):
    """Write the kept BUS records to a new BUS file and regenerate the count matrix via bustools.

    `kept_records` is a DataFrame with columns [barcode, UMI, EC, read_index] and, when the per-file
    barcodes of a paired-end bulk run were collapsed, `raw_barcode`; `new_ec_records` is a list of
    (ec_id, [transcript indices]) equivalence classes to append to matrix.ec (for reads whose EC was
    narrowed to a consistent subset). Returns the path to the produced cells_x_genes.mtx.

    `read_index` is written into the BUS flag column, the same place kallisto's `--num` puts it, so the
    filtered BUS stays traceable back to individual reads (`bustools text -f`). The flag is reset for
    the sorted BUS that feeds `bustools count`, so the counts are identical to what they would be with
    an all-zero flag column: sorting is barcode/UMI/EC/flag, and distinct flags would otherwise stop
    records that differ only by read from collapsing.

    `new_counts_dir` is written as a self-contained kb-count-style directory: the BUS file, the
    matrix.ec (original classes plus the newly created ones) and the transcripts.txt those EC indices
    refer to, so the new equivalence classes can be interpreted without the original run's files.
    """
    os.makedirs(new_counts_dir, exist_ok=True)
    filtered_txt = os.path.join(new_counts_dir, "output_filtered_bus.txt")
    filtered_bus = os.path.join(new_counts_dir, "output_filtered.bus")
    sorted_bus = os.path.join(new_counts_dir, "output_filtered.sorted.bus")

    out = kept_records[["barcode", "UMI", "EC"]].copy()
    out["count"] = 1
    out["read_index"] = kept_records["read_index"].to_numpy()
    out.to_csv(filtered_txt, sep="\t", header=False, index=False)

    # The per-file barcode is collapsed in the BUS itself (see the caller), so when it was collapsed the
    # fastq a read came from is only recoverable from this sidecar.
    if "raw_barcode" in kept_records.columns:
        provenance = os.path.join(new_counts_dir, "output_filtered_reads.tsv")
        kept_records[["raw_barcode", "barcode", "UMI", "EC", "read_index"]].to_csv(provenance, sep="\t", index=False)
        logger.info(f"Wrote per-read provenance (pre-collapse barcode + read index) to {provenance}")

    subprocess.run([bustools, "fromtext", "-o", filtered_bus, filtered_txt], check=True)
    subprocess.run([bustools, "sort", "-t", str(kallisto_threads), "--no-flags", "-o", sorted_bus, filtered_bus], check=True)

    # Write the new run's matrix.ec (original equivalence classes + the ones created when a read's EC was
    # narrowed to the subset of VCRS consistent with its true alignment) and the transcripts.txt whose
    # line numbers those EC entries index into. The transcript list itself is unchanged -- the new ECs are
    # new *combinations* of existing transcripts -- so it is copied verbatim; rewriting it would shift the
    # indices that both the original and the new EC entries refer to.
    ec = os.path.join(new_counts_dir, "matrix.ec")
    src_ec = os.path.join(kb_count_vcrs_dir, "matrix.ec")
    with open(src_ec, "rb") as src, open(ec, "wb") as dst:
        shutil.copyfileobj(src, dst)  # streamed: matrix.ec is routinely hundreds of MB
        if new_ec_records:
            if _last_byte(src_ec) not in (b"\n", b""):
                dst.write(b"\n")
            for ec_id, idxs in new_ec_records:
                dst.write(f"{ec_id}\t{','.join(str(i) for i in idxs)}\n".encode("utf-8"))
    if new_ec_records:
        logger.info(f"Wrote {len(new_ec_records)} new equivalence class(es) to {ec}")

    transcripts_txt = os.path.join(new_counts_dir, "transcripts.txt")
    shutil.copyfile(os.path.join(kb_count_vcrs_dir, "transcripts.txt"), transcripts_txt)

    count_prefix = os.path.join(new_counts_dir, "cells_x_genes")

    # Prefer reusing the exact bustools count invocation kb recorded, swapping input bus, matrix.ec and -o.
    count_cmd = _extract_bustools_count_command(kb_count_vcrs_dir, bustools, count_prefix, sorted_bus, vcrs_t2g, ec, transcripts_txt, mm)
    logger.info(f"Regenerating count matrix: {' '.join(count_cmd)}")
    subprocess.run(count_cmd, check=True)
    return f"{count_prefix}.mtx"


def _extract_bustools_count_command(kb_count_vcrs_dir, bustools, count_prefix, sorted_bus, vcrs_t2g, ec, transcripts_txt, mm):
    """Build the `bustools count` command, reusing kb_info's recorded flags when available."""
    kb_info_path = os.path.join(kb_count_vcrs_dir, "kb_info.json")
    recorded = None
    try:
        with open(kb_info_path, "r", encoding="utf-8") as f:
            kb_info = json.load(f)
        for c in kb_info.get("commands", []):
            tokens = c.split() if isinstance(c, str) else list(c)
            if any("bustools" in str(t) for t in tokens) and "count" in tokens:
                recorded = tokens
    except (FileNotFoundError, json.JSONDecodeError, KeyError):
        recorded = None

    if recorded:
        cmd = list(recorded)
        cmd[0] = bustools  # normalize the binary path
        # swap the -o prefix
        if "-o" in cmd:
            cmd[cmd.index("-o") + 1] = count_prefix
        # swap the -e equivalence-class matrix (augmented with the newly-created ECs) and the
        # -t transcripts.txt its indices point into (both now live in the new counts dir)
        if "-e" in cmd:
            cmd[cmd.index("-e") + 1] = ec
        if "-t" in cmd:
            cmd[cmd.index("-t") + 1] = transcripts_txt
        # the trailing positional argument is the input .bus file
        for i in range(len(cmd) - 1, 0, -1):
            if str(cmd[i]).endswith(".bus"):
                cmd[i] = sorted_bus
                break
        else:
            cmd.append(sorted_bus)
        return cmd

    # Fallback: construct from known paths (mirrors kb count's bustools count call).
    if not vcrs_t2g:
        raise ValueError("Could not find a recorded bustools count command in kb_info.json and no vcrs_t2g was provided; cannot regenerate the count matrix. Pass vcrs_t2g explicitly.")
    cmd = [bustools, "count", "--genecounts", "-o", count_prefix, "-g", str(vcrs_t2g), "-e", ec, "-t", transcripts_txt]
    if mm:
        cmd.insert(2, "--multimapping")
    cmd.append(sorted_bus)
    return cmd


def join_list_columns_for_h5ad(var, sep=";"):
    """Copy of an adata.var with its list-valued columns joined into `sep`-separated strings.

    A merged VCRS carries one entry per variant it represents, so vk clean holds columns like seq_ID
    and mutation as Python lists (`['1', '2', '4']`) for most of its run. h5py cannot write those --
    it raises "Can't implicitly convert non-string objects to strings" -- so any adata written before
    vk clean's final conversion has to do the same conversion first. Read back with e.g.
    `adata.var[col].apply(lambda x: x.split(";") if isinstance(x, str) else x)`.

    Membership is tested per value rather than on the first row, which can be NaN for a VCRS the
    variant merge found nothing for and would then skip a column that does hold lists further down.
    """
    out = var.copy()
    for col in out.columns:
        is_list = out[col].map(lambda x: isinstance(x, list))
        if is_list.any():
            out[col] = out[col].map(lambda x: sep.join(map(str, x)) if isinstance(x, list) else x)
    return out


def _load_and_align_recounted_adata(mtx_file, adata_template):
    """Load the regenerated mtx and reindex it onto the template's obs/var (same shape)."""
    adata_new = load_adata_from_mtx(mtx_file)

    # load_adata_from_mtx keeps the VCRS ids / barcodes as columns (gene_id / cell_barcode), not the index
    true_barcode_length = len(str(adata_template.obs.index[0]))
    obs_source = adata_new.obs["cell_barcode"] if "cell_barcode" in adata_new.obs.columns else adata_new.obs.index.to_series()
    var_source = adata_new.var["gene_id"] if "gene_id" in adata_new.var.columns else adata_new.var.index.to_series()
    new_obs_names = pd.Index(obs_source.astype(str)).str[-true_barcode_length:].to_numpy()
    new_var_names = var_source.astype(str).to_numpy()

    obs_map = {bc: i for i, bc in enumerate(adata_template.obs.index.astype(str))}
    var_map = {v: i for i, v in enumerate(adata_template.var.index.astype(str))}

    # vectorized reindex over the nonzeros: map each new row/col onto the template position (-1 if absent)
    row_target = np.array([obs_map.get(bc, -1) for bc in new_obs_names], dtype=np.int64)
    col_target = np.array([var_map.get(v, -1) for v in new_var_names], dtype=np.int64)

    coo = adata_new.X.tocoo()
    rows = row_target[coo.row]
    cols = col_target[coo.col]
    keep = (rows >= 0) & (cols >= 0)
    final = csr_matrix((coo.data[keep].astype(np.float64), (rows[keep], cols[keep])), shape=adata_template.shape)
    return ad.AnnData(X=final, obs=adata_template.obs.copy(), var=adata_template.var.copy(), uns=adata_template.uns.copy())


# --------------------------------------------------------------------------- #
# Read counts straight from the BUS file (bustools' multimapping division undone)
# --------------------------------------------------------------------------- #
# `bustools count --multimapping --cm` splits a multimapped read's count across the targets of
# its equivalence class *in integer arithmetic*, so the value a VCRS lands in the matrix is
#
#       sum over ECs containing it of  floor(reads_in_EC / targets_in_EC)
#
# A VCRS whose reads only ever fall in equivalence classes holding fewer reads than targets
# therefore receives a hard **0** -- not a small number, zero -- and is indistinguishable from a
# VCRS that no read ever matched. No threshold on that matrix can recover it, because there is
# nothing left to threshold. Counting reads *compatible with* each VCRS instead makes the call a
# threshold on evidence rather than on an artefact of integer division.


def compute_vcrs_read_counts(bus_dir, bus_file="output.bus", bustools="bustools"):
    """(barcodes, vcrs_names, matrix) of reads compatible with each VCRS, read out of a BUS file.

    `bus_dir` is a kb-count-style directory holding the BUS file plus the `matrix.ec` and
    `transcripts.txt` its equivalence-class indices refer to. Every record of the BUS
    contributes its full count to *each* target of its equivalence class, which is exactly what
    `bustools count` does minus the `--cm` division.

    The returned matrix is barcodes x transcripts, so summing it over rows reproduces a
    whole-run per-VCRS read count and a single-sample bulk run yields one row.
    """
    import collections

    transcripts_path = os.path.join(bus_dir, "transcripts.txt")
    ec_path = os.path.join(bus_dir, "matrix.ec")
    bus_path = os.path.join(bus_dir, bus_file)
    for required in (transcripts_path, ec_path, bus_path):
        if not os.path.exists(required):
            raise FileNotFoundError(f"Cannot count reads per VCRS: {required} does not exist. Re-run kb count with delete_intermediate_files=False so the BUS file and its equivalence classes are kept.")

    with open(transcripts_path, encoding="utf-8") as f:
        vcrs_names = f.read().splitlines()

    bus_txt = f"{bus_path}.txt"
    if not os.path.exists(bus_txt):
        subprocess.run([bustools, "text", "-o", bus_txt, bus_path], check=True)

    # (barcode, EC) -> reads. Reads are summed per equivalence class first so that each EC's
    # target list is only expanded once, however many records share it.
    reads_per_barcode_ec = collections.Counter()
    with open(bus_txt, encoding="utf-8") as fh:
        for line in fh:
            fields = line.split("\t")
            reads_per_barcode_ec[(fields[0], int(fields[2]))] += int(fields[3])

    # matrix.ec is routinely hundreds of MB for a catalog reference, and only the classes the
    # reads actually hit are needed, so it is streamed and filtered rather than held whole.
    ec_ids_needed = {ec_id for _barcode, ec_id in reads_per_barcode_ec}
    ec_targets = {}
    with open(ec_path, encoding="utf-8") as fh:
        for line in fh:
            ec_id, targets = line.rstrip("\n").split("\t")
            ec_id = int(ec_id)
            if ec_id in ec_ids_needed:
                ec_targets[ec_id] = np.fromstring(targets, dtype=np.int64, sep=",")
    missing_ecs = ec_ids_needed - set(ec_targets)
    if missing_ecs:
        raise ValueError(f"{len(missing_ecs)} equivalence class(es) in {bus_path} are absent from {ec_path} (e.g. {sorted(missing_ecs)[:5]}); the BUS file and matrix.ec do not belong to the same run.")

    barcodes = sorted({barcode for barcode, _ec_id in reads_per_barcode_ec})
    barcode_index = {barcode: i for i, barcode in enumerate(barcodes)}
    rows, cols, values = [], [], []
    for (barcode, ec_id), n_reads in reads_per_barcode_ec.items():
        targets = ec_targets[ec_id]
        rows.append(np.full(targets.size, barcode_index[barcode], dtype=np.int64))
        cols.append(targets)
        values.append(np.full(targets.size, n_reads, dtype=np.float64))

    shape = (len(barcodes), len(vcrs_names))
    if not values:
        return barcodes, vcrs_names, csr_matrix(shape, dtype=np.float64)
    matrix = csr_matrix((np.concatenate(values), (np.concatenate(rows), np.concatenate(cols))), shape=shape)  # duplicate (row, col) pairs are summed
    return barcodes, vcrs_names, matrix


def apply_read_count_matrix(adata, bus_dir, min_reads=None, bus_file="output.bus", bustools="bustools"):
    """Replace `adata.X` with reads-compatible-per-VCRS and zero everything below `min_reads`.

    The matrix `bustools count --cm` produced is kept in `adata.layers["bustools_counts"]` and the
    per-VCRS read totals in `adata.var["read_count"]`, so nothing is lost -- but `adata.X`, which is
    what every downstream filter and the VCF writer read, becomes the read counts.
    """
    barcodes, vcrs_names, read_matrix = compute_vcrs_read_counts(bus_dir, bus_file=bus_file, bustools=bustools)

    # Barcodes are matched on their trailing characters the same way the recounted matrix is (kb
    # pads bulk barcodes), and VCRSs by name rather than by position, so neither a reordering
    # between kb count and the AnnData nor a padded barcode can silently misalign the counts.
    barcode_length = len(str(adata.obs.index[0]))
    obs_map = {str(barcode): i for i, barcode in enumerate(adata.obs.index)}
    var_map = {str(vcrs): i for i, vcrs in enumerate(adata.var.index)}
    row_target = np.array([obs_map.get(str(barcode)[-barcode_length:], -1) for barcode in barcodes], dtype=np.int64)
    col_target = np.array([var_map.get(str(vcrs), -1) for vcrs in vcrs_names], dtype=np.int64)

    coo = read_matrix.tocoo()
    rows, cols = row_target[coo.row], col_target[coo.col]
    keep = (rows >= 0) & (cols >= 0)
    reads = csr_matrix((coo.data[keep], (rows[keep], cols[keep])), shape=adata.shape)

    n_unmatched_vcrs = int((col_target < 0).sum())
    if n_unmatched_vcrs:
        logger.warning(f"{n_unmatched_vcrs:,} / {len(vcrs_names):,} VCRSs in {os.path.join(bus_dir, 'transcripts.txt')} are not columns of the count matrix and were skipped when counting reads.")

    bustools_counts = np.asarray(adata.X.sum(axis=0)).ravel()
    read_counts = np.asarray(reads.sum(axis=0)).ravel()
    adata.layers["bustools_counts"] = adata.X.copy()
    adata.X = reads
    adata.var["read_count"] = read_counts

    logger.info(f"Read counts from {os.path.join(bus_dir, bus_file)}: {int((read_counts > 0).sum()):,} VCRSs have at least one compatible read, versus {int((bustools_counts > 0).sum()):,} detected by `bustools count --cm` ({int(((read_counts > 0) & (bustools_counts == 0)).sum()):,} of them zeroed by its multimapping division).")

    if min_reads is not None and min_reads > 1:
        n_before = int((read_counts > 0).sum())
        adata.X = adata.X.multiply(adata.X >= min_reads).tocsr() if issparse(adata.X) else adata.X * (adata.X >= min_reads)
        n_after = int((np.asarray(adata.X.sum(axis=0)).ravel() > 0).sum())
        logger.info(f"min_reads={min_reads}: {n_before - n_after:,} VCRSs fell below the threshold and were zeroed -> {n_after:,} VCRSs called.")
    return adata


# --------------------------------------------------------------------------- #
# Variant allele fraction from the pseudobam
# --------------------------------------------------------------------------- #
# The VCRS count matrix only ever counts reads *compatible with the variant* -- the ALT side.
# It has no notion of how many reads at the same locus carry the reference allele, so a VCRS
# that a handful of reads match by coincidence (a catalog entry that mimics the sample's real
# haplotype, a mismapped paralogue) is indistinguishable from a genuine heterozygote. The
# alignments gathered for the position check already answer that question, so the reference
# depth is read straight off them here.

_HGVS_VAF_PATTERNS = (
    # order matters: delins must be tried before del, and dup before the bare-position forms
    ("delins", re.compile(r"^(?P<seq>.+):g\.(?P<start>\d+)(?:_(?P<end>\d+))?delins(?P<alt>[ACGTN]+)$")),
    ("del", re.compile(r"^(?P<seq>.+):g\.(?P<start>\d+)(?:_(?P<end>\d+))?del(?P<ref>[ACGTN]*)$")),
    ("ins", re.compile(r"^(?P<seq>.+):g\.(?P<start>\d+)_(?P<end>\d+)ins(?P<alt>[ACGTN]+)$")),
    ("dup", re.compile(r"^(?P<seq>.+):g\.(?P<start>\d+)(?:_(?P<end>\d+))?dup(?P<alt>[ACGTN]*)$")),
    ("sub", re.compile(r"^(?P<seq>.+):g\.(?P<start>\d+)(?P<ref>[ACGTN]+)>(?P<alt>[ACGTN]+)$")),
)


def parse_hgvsg_for_vaf(header):
    """(seq_id, start, end, kind, alt) for one HGVS genomic string, or None if unparseable.

    `kind` is "snv" for a single-base substitution and "indel" for everything else, which is
    what selects between `min_snp_vaf` and `min_indel_vaf`. Multi-base substitutions (MNVs)
    count as indels: they are not scored base-by-base below.
    """
    for name, pattern in _HGVS_VAF_PATTERNS:
        match = pattern.match(header)
        if not match:
            continue
        parts = match.groupdict()
        seq_id = parts["seq"].split("(")[0].split(".")[0]
        start = int(parts["start"])
        end = int(parts["end"]) if parts.get("end") else start
        alt = parts.get("alt") or ""
        if name == "sub":
            ref = parts.get("ref") or ""
            kind = "snv" if len(ref) == 1 and len(alt) == 1 else "indel"
            return seq_id, start, start + max(len(ref), 1) - 1, kind, alt
        return seq_id, start, end, "indel", alt
    return None


def _resolve_bam_contig(bam, seq_id):
    """Match an HGVS seq_id to a BAM reference name, tolerating a 'chr' prefix on either side."""
    refs = set(bam.references)
    for candidate in (seq_id, f"chr{seq_id}", seq_id[3:] if seq_id.startswith("chr") else None):
        if candidate and candidate in refs:
            return candidate
    return None


def _ensure_indexed_bam(bam_path, samtools="samtools", threads=1):
    """A coordinate-sorted, indexed copy of `bam_path` (the original when it already is one).

    Random access by locus is required to read the reference depth; alignment output from
    kallisto/bowtie2 is name-ordered, so it is sorted once into a sibling file and reused.
    """
    try:
        with pysam.AlignmentFile(bam_path, "rb") as bam:
            if bam.has_index():
                return bam_path
    except (ValueError, OSError):
        pass

    sorted_path = str(bam_path).replace(".bam", "") + ".vaf_sorted.bam"
    if not os.path.exists(sorted_path + ".bai"):
        logger.info(f"VAF filter: sorting and indexing {bam_path} for random access")
        try:
            pysam.sort("-@", str(max(1, int(threads) - 1)), "-o", sorted_path, str(bam_path))
            pysam.index(sorted_path)
        except Exception as exc:  # pylint: disable=broad-except
            logger.warning(f"VAF filter: could not sort/index {bam_path} ({exc}); VAF cannot be computed from it.")
            return None
    return sorted_path


def compute_vaf_for_variants(bam_paths, headers, slack=10, max_depth=100000, samtools="samtools", threads=1):
    """{header: (depth, alt_fragments, vaf)} for each HGVS genomic string in `headers`.

    Counting is per *fragment* (read name), not per record, so a read pair covering the locus
    contributes once -- matching how the VCRS matrix treats a pair once
    `avoid_paired_double_counting` has been applied. Substitutions are scored by the base at
    the variant position; indels by an I/D operation within `slack` bp of the interval, which
    keeps left-aligned representations of the same event together.
    """
    parsed = {}
    for header in headers:
        info = parse_hgvsg_for_vaf(str(header))
        if info is not None:
            parsed[header] = info
    if not parsed:
        return {}

    totals = {header: [set(), set()] for header in parsed}  # header -> [covering, alt-carrying]
    for bam_path in bam_paths:
        indexed = _ensure_indexed_bam(bam_path, samtools=samtools, threads=threads)
        if indexed is None:
            continue
        with pysam.AlignmentFile(indexed, "rb") as bam:
            for header, (seq_id, start, end, kind, alt) in parsed.items():
                contig = _resolve_bam_contig(bam, seq_id)
                if contig is None:
                    continue
                covering, alt_frags = totals[header]
                try:
                    fetched = bam.fetch(contig, max(0, start - 1 - slack - 1000), end + slack + 1000)
                except (ValueError, OSError):
                    continue
                for read in fetched:
                    if read.is_unmapped or read.query_sequence is None or read.reference_end is None:
                        continue
                    # the read must actually span the variant's first base to say anything about it
                    if not read.reference_start < start <= read.reference_end:
                        continue
                    covering.add(read.query_name)
                    if kind == "snv":
                        for qpos, rpos in read.get_aligned_pairs(matches_only=True):
                            if rpos + 1 == start:
                                if read.query_sequence[qpos].upper() == alt[:1].upper():
                                    alt_frags.add(read.query_name)
                                break
                    else:
                        ref_pos = read.reference_start
                        for op, length in (read.cigartuples or ()):
                            if op in (0, 7, 8):        # M / = / X
                                ref_pos += length
                            elif op == 2:              # D
                                if start - slack <= ref_pos + 1 <= end + slack + length:
                                    alt_frags.add(read.query_name)
                                ref_pos += length
                            elif op == 1:              # I
                                if start - slack <= ref_pos <= end + slack:
                                    alt_frags.add(read.query_name)

    out = {}
    for header, (covering, alt_frags) in totals.items():
        depth = len(covering)
        n_alt = len(alt_frags)
        out[header] = (depth, n_alt, (n_alt / depth) if depth else float("nan"))
    return out


def apply_vaf_filter(adata, bam_paths, min_snp_vaf=None, min_indel_vaf=None, slack=10, samtools="samtools", threads=1):
    """Zero out VCRSs whose locus does not reach the required allele fraction in the alignments.

    Only VCRSs that still carry counts are evaluated -- a catalog reference has millions of
    columns and all but a few thousand are already empty, so the pileup work is proportional to
    the number of calls rather than to the size of the reference.

    A VCRS whose header merges several variants passes if *any* of them passes (each judged
    against the threshold for its own kind), so merging never makes a call stricter. A variant
    whose locus is absent from the BAM, or whose HGVS cannot be parsed, is left alone: the
    filter only ever removes calls it has positive evidence against.
    """
    if min_snp_vaf is None and min_indel_vaf is None:
        return adata

    counts = np.asarray(adata.X.sum(axis=0)).ravel()
    called = np.flatnonzero(counts > 0)
    if called.size == 0:
        return adata

    vcrs_ids = [str(v) for v in adata.var_names[called]]
    individual = sorted({part for vid in vcrs_ids for part in vid.split(";") if part})
    logger.info(f"VAF filter: computing allele fractions for {len(individual):,} variant(s) across {len(vcrs_ids):,} VCRS(s) with counts")
    vaf_by_variant = compute_vaf_for_variants(bam_paths, individual, slack=slack, samtools=samtools, threads=threads)

    thresholds = {"snv": min_snp_vaf, "indel": min_indel_vaf}
    failing, n_evaluated = [], 0
    vaf_column, depth_column = {}, {}
    for row, vid in zip(called, vcrs_ids):
        verdicts, best_vaf, best_depth = [], float("nan"), 0
        for part in vid.split(";"):
            if not part:
                continue
            info = parse_hgvsg_for_vaf(part)
            stats = vaf_by_variant.get(part)
            if info is None or stats is None:
                continue
            depth, _n_alt, vaf = stats
            if depth == 0 or np.isnan(vaf):
                continue
            threshold = thresholds.get(info[3])
            if threshold is None:      # this kind is not being filtered -> the VCRS survives
                verdicts.append(True)
            else:
                verdicts.append(vaf >= threshold)
            if np.isnan(best_vaf) or vaf > best_vaf:
                best_vaf, best_depth = vaf, depth
        if not verdicts:               # nothing measurable -> leave the call alone
            continue
        n_evaluated += 1
        vaf_column[vid], depth_column[vid] = best_vaf, best_depth
        if not any(verdicts):
            failing.append(row)

    if vaf_column:
        adata.var["pseudobam_vaf"] = [vaf_column.get(str(v), np.nan) for v in adata.var_names]
        adata.var["pseudobam_vaf_depth"] = [depth_column.get(str(v), np.nan) for v in adata.var_names]

    if n_evaluated == 0:
        logger.warning(
            "VAF filter requested but no call could be measured, so nothing was filtered. The filter needs "
            "HGVS *genomic* headers (`seq:g.…`) whose seq_id appears in the alignment BAM; transcript-space "
            "(`c.`) headers and loci absent from the BAM are skipped by design."
        )
    logger.info(
        f"VAF filter (min_snp_vaf={min_snp_vaf}, min_indel_vaf={min_indel_vaf}): "
        f"{len(failing):,} / {n_evaluated:,} evaluated VCRSs fell below the threshold and were zeroed "
        f"({len(vcrs_ids) - n_evaluated:,} could not be measured and were left alone)."
    )
    if failing:
        X = adata.X.tolil() if issparse(adata.X) else adata.X
        X[:, failing] = 0
        adata.X = X.tocsr() if issparse(adata.X) else X
    return adata


def adjust_variant_adata_by_pseudobam(
    kb_count_vcrs_dir,
    technology,
    reference_genome_index=None,
    kallisto_quant_reference_genome_dir=None,
    adata=None,
    fastq_file_list=None,
    adata_output_path=None,
    parity="single",
    mm=False,
    bustools="bustools",
    kallisto="kallisto",
    threads=2,
    variant_source=None,
    reference_sequences_type=None,
    reads_type=None,
    sequences=None,
    gtf=None,
    chromosomes=None,
    vcrs_t2g=None,
    check_alignment_position=False,
    alignment_position_tolerance=200,
    min_reads=None,
    min_snp_vaf=None,
    min_indel_vaf=None,
    count_reads_that_dont_pseudoalign_to_reference_genome=True,
    avoid_paired_double_counting=False,
    fragment_length=51,
    fragment_sd=5,
    seq_id_column="seq_ID",
    var_column="mutation",
    fastq_sorting_check_only=False,
    save_type="parquet",
    overwrite=False,
    pseudobam_validation_aligner="pseudo",  # "pseudo" (kallisto quant --pseudobam/--genomebam), "true" (bowtie2/STAR, picked from use_genomebam), or an explicit "STAR"/"bowtie2"
    reference_bam=None,  # skip alignment entirely and use this BAM (or one BAM per fastq group)
    read_length=None,  # only used for STAR's --sjdbOverhang; inferred from the first fastq when None
    star="STAR",
    bowtie2="bowtie2",
    samtools="samtools",
):
    """Faster/more-accurate replacement for adjust_variant_adata_by_normal_gene_matrix.

    Aligns the reads against the standard reference, compares each read's true alignment coordinate
    to the HGVS locus of its assigned VCRS, drops reads that aligned somewhere inconsistent with the
    variant, and regenerates the VCRS count matrix with `bustools count`.

    How the alignment is obtained is set by `pseudobam_validation_aligner`:
      - "pseudo" (default): `kallisto quant --pseudobam`, or `--genomebam` when RNA reads are aligned
        against a DNA/genome reference (see _determine_pseudobam_mode). Fast, and reuses the kallisto
        index. `reference_genome_index` is a kallisto index.
      - "true" / "STAR" / "bowtie2": a real aligner. Splice-aware STAR is used exactly when
        `use_genomebam` would have been True (RNA reads against a genome reference); bowtie2 otherwise.
        `reference_genome_index` is the STAR genome directory / bowtie2 index prefix, built from
        `sequences` (+ `gtf` for STAR) if it does not exist.

    `reference_bam` short-circuits all of that: pass a BAM that was already aligned against the same
    reference (e.g. the one vk denovo produced) and it is used as-is. Either a single BAM covering all
    reads, or one BAM per fastq group in the same order as the fastqs.

    `min_snp_vaf` / `min_indel_vaf` additionally require the variant's locus to reach an allele
    fraction in those same alignments before the VCRS is allowed to keep its counts. This is the
    one thing the VCRS matrix cannot see on its own: it counts only reads compatible with the
    variant, so a coincidental match and a real heterozygote look identical until the reference
    allele is counted too.
    """
    if kallisto_quant_reference_genome_dir is None:
        kallisto_quant_reference_genome_dir = os.path.join(kb_count_vcrs_dir, "pseudobam_qc")

    if not adata:
        adata = f"{kb_count_vcrs_dir}/counts_unfiltered/adata.h5ad"
    if isinstance(adata, str):
        adata = ad.read_h5ad(adata)
    adata = adata.copy()

    bustools = _resolve_kb_binary(bustools, "bustools")
    kallisto = _resolve_kb_binary(kallisto, "kallisto")

    pb = _determine_pseudobam_mode(adata.var, variant_source=variant_source, reference_sequences_type=reference_sequences_type, sequences=sequences, reads_type=reads_type)
    header_type = pb["header_type"]
    use_genomebam = pb["use_genomebam"]
    convert_transcript_to_genome = pb["convert_transcript_to_genome"]
    logger.info(f"pseudobam QC: header_type={header_type}, reference_sequences_type={pb['reference_sequences_type']}, reads_type={pb['reads_type']}, use_genomebam={use_genomebam}, convert_transcript_to_genome={convert_transcript_to_genome}")

    # ---- resolve how the alignments are obtained: a supplied BAM, kallisto quant, or bowtie2/STAR ----
    alignment_mode = "bam" if reference_bam else str(pseudobam_validation_aligner).lower()
    if alignment_mode in {"pseudo", "kallisto", "pseudobam", "genomebam"}:
        alignment_mode = "pseudo"
    elif alignment_mode in {"true", "real", "star", "bowtie2"}:
        true_aligner = _resolve_true_aligner(pseudobam_validation_aligner, use_genomebam)
        alignment_mode = "true"
    elif alignment_mode != "bam":
        raise ValueError(f"pseudobam_validation_aligner must be one of 'pseudo', 'true', 'STAR', 'bowtie2'. Got '{pseudobam_validation_aligner}'.")

    if alignment_mode != "bam":
        index_type = _detect_reference_genome_index_type(reference_genome_index)
        if alignment_mode == "pseudo":
            if index_type in {"STAR", "bowtie2"}:
                raise ValueError(f"pseudobam_validation_aligner='pseudo' aligns with kallisto, but reference_genome_index '{reference_genome_index}' is a {index_type} index. Pass a kallisto index, or set pseudobam_validation_aligner='true'.")
            if needs_k64(kallisto, reference_genome_index):
                kallisto += "_k64"
        else:
            if index_type == "kallisto":
                raise ValueError(f"pseudobam_validation_aligner='{pseudobam_validation_aligner}' aligns with {true_aligner}, but reference_genome_index '{reference_genome_index}' is a kallisto index. Pass a {true_aligner} index, or set pseudobam_validation_aligner='pseudo'.")
            if index_type is not None and index_type != true_aligner:
                raise ValueError(f"reference_genome_index '{reference_genome_index}' is a {index_type} index, but use_genomebam={use_genomebam} requires a {true_aligner} index (RNA reads against a genome reference need splice-aware STAR; everything else aligns contiguously with bowtie2).")
            if not reference_genome_index:
                reference_genome_index = os.path.join(kallisto_quant_reference_genome_dir, "star_index" if true_aligner == "STAR" else "bowtie2_index")

    if fastq_file_list is None:
        raise ValueError("fastq_file_list is required for the pseudobam QC (the reads must be matched to the alignments by FASTQ header).")
    fastq_file_list = load_in_fastqs(fastq_file_list)
    fastq_file_list = sort_fastq_files_for_kb_count(fastq_file_list, technology=technology, check_only=fastq_sorting_check_only)

    if vcrs_t2g is None:
        try:
            with open(f"{kb_count_vcrs_dir}/kb_info.json", "r", encoding="utf-8") as f:
                call = json.load(f).get("call", "")
            m = re.search(r"-g (\S+)", call)
            vcrs_t2g = m.group(1) if m else None
        except (FileNotFoundError, json.JSONDecodeError):
            vcrs_t2g = None

    # ---- parity of the VCRS kb count run (needed for consistent BUS/barcode handling) ----
    with open(f"{kb_count_vcrs_dir}/kb_info.json", "r", encoding="utf-8") as f:
        kb_info_data_vcrs = json.load(f)
    vcrs_parity = "paired" if "--parity paired" in kb_info_data_vcrs.get("call", "") else "single"
    if technology.upper() not in {"BULK", "SMARTSEQ2"}:
        vcrs_parity = "single"

    # ---- split the fastqs into the groups that get aligned (and attributed to a barcode) together ----
    barcodes = None
    if os.path.exists(f"{kb_count_vcrs_dir}/matrix.sample.barcodes"):
        with open(f"{kb_count_vcrs_dir}/matrix.sample.barcodes", encoding="utf-8") as f:
            barcodes = f.read().splitlines()
    groups = _group_fastqs_for_quant(fastq_file_list, technology, vcrs_parity, barcodes)

    align_by_group = {}
    vaf_bam_paths = []  # the alignments the VAF filter reads reference depth from, whatever produced them
    if alignment_mode == "bam":
        # ---- use BAM(s) the caller already produced against the same reference (e.g. from vk denovo) ----
        bam_list = [str(reference_bam)] if isinstance(reference_bam, (str, Path)) else [str(b) for b in reference_bam]
        for bam_path in bam_list:
            if not os.path.exists(bam_path):
                raise FileNotFoundError(f"reference_bam '{bam_path}' does not exist.")
        if len(bam_list) == 1:
            if len(groups) > 1:
                logger.warning(f"A single reference_bam was given for {len(groups)} fastq groups (samples); it is used for all of them. Reads whose names collide across samples cannot be told apart.")
            parsed = _parse_pseudobam(bam_list[0])
            align_by_group = {group_barcode: parsed for group_barcode, _, _ in groups}
        elif len(bam_list) == len(groups):
            align_by_group = {group_barcode: _parse_pseudobam(bam_path) for (group_barcode, _, _), bam_path in zip(groups, bam_list)}
        else:
            raise ValueError(f"reference_bam has {len(bam_list)} BAMs but the fastqs form {len(groups)} group(s). Pass either one BAM covering all reads or one BAM per group, in the same order as the fastqs.")
        vaf_bam_paths = list(bam_list)
    elif alignment_mode == "pseudo":
        # ---- generate chromosomes file for --genomebam if needed ----
        if use_genomebam and not chromosomes:
            chromosomes = os.path.join(kallisto_quant_reference_genome_dir, "chromosomes.txt")
            os.makedirs(kallisto_quant_reference_genome_dir, exist_ok=True)
            if not os.path.exists(chromosomes) or overwrite:
                _generate_chromosomes_file(chromosomes, sequences=sequences, gtf=gtf)

        # Build the standard reference index on the fly if it does not exist. When RNA reads are aligned against a
        # DNA/genome `sequences` reference (use_genomebam), the index is built as a cDNA index via
        # `kb ref --workflow standard`; otherwise `sequences` is indexed directly with `kb ref --workflow custom`.
        reference_genome_index = _build_reference_genome_index(reference_genome_index, use_genomebam, sequences=sequences, gtf=gtf, threads=threads, kallisto=kallisto, bustools=bustools)

        # ---- run kallisto quant per fastq group and parse the pseudoalignments ----
        for gi, (group_barcode, group_files, is_single_end) in enumerate(groups):
            group_out = os.path.join(kallisto_quant_reference_genome_dir, f"quant_group_{gi}")
            bam_path = _run_kallisto_quant_group(
                kallisto, reference_genome_index, group_out, group_files, is_single_end, use_genomebam,
                threads=threads, fragment_length=fragment_length, fragment_sd=fragment_sd,
                gtf=gtf, chromosomes=chromosomes, overwrite=overwrite,
            )
            align_by_group[group_barcode] = _parse_pseudobam(bam_path)
            vaf_bam_paths.append(bam_path)
    else:
        # ---- true alignment: STAR when the reads must be spliced onto a genome, bowtie2 otherwise ----
        from varseek.utils.varseek_denovo_utils import check_tool  # local import to keep the module import graph acyclic

        check_tool(star if true_aligner == "STAR" else bowtie2)
        if true_aligner == "bowtie2":
            check_tool(samtools)
        if read_length is None:
            read_length = _infer_read_length_from_fastq(fastq_file_list[0])
        logger.info(f"pseudobam QC: aligning with {true_aligner} (read_length={read_length}) against '{reference_genome_index}'")

        reference_genome_index = _build_true_aligner_index(
            reference_genome_index, true_aligner, sequences=sequences, gtf=gtf,
            threads=threads, read_length=read_length, star=star, bowtie2_build=f"{bowtie2}-build",
        )

        for gi, (group_barcode, group_files, is_single_end) in enumerate(groups):
            group_out = os.path.join(kallisto_quant_reference_genome_dir, f"align_group_{gi}")
            bam_path = _run_true_aligner_group(
                true_aligner, reference_genome_index, group_out, group_files, is_single_end,
                threads=threads, read_length=read_length, overwrite=overwrite,
                star=star, bowtie2=bowtie2, samtools=samtools,
            )
            align_by_group[group_barcode] = _parse_pseudobam(bam_path)
            vaf_bam_paths.append(bam_path)

    single_cell = groups[0][0] is None  # barcode comes from the BUS file rather than the group

    # For paired-end BULK/SMARTSEQ2 data run through kb count in single mode, each fastq of a pair
    # became its own pseudo-sample/barcode. The per-read alignment lookup below is keyed by that raw
    # per-file barcode, so we only collapse pairs back to their shared "good" barcode when writing the
    # kept records for the recount (so the recounted matrix lines up with the barcode-corrected adata
    # template; mirrors correct_adata_barcodes_for_running_paired_data_in_single_mode and the
    # gene_matrix path). Cost is one dict lookup per kept read.
    bad_to_good_barcode_dict = None
    if parity == "paired" and vcrs_parity == "single" and technology.upper() in {"BULK", "SMARTSEQ2"} and barcodes:
        bad_to_good_barcode_dict = make_good_barcodes_and_file_index_tuples(barcodes, include_file_index=False)

    # ---- build the per-read VCRS BUS df (one row per read) with fastq headers to match the BAM ----
    logger.info("Making VCRS BUS df (with fastq headers)")
    bus_df = make_bus_df(
        kb_count_out=kb_count_vcrs_dir,
        fastq_file_list=fastq_file_list,
        technology=technology,
        t2g_file=None,
        mm=mm,
        parity=vcrs_parity,
        bustools=bustools,
        fastq_sorting_check_only=True,
        chunksize=None,
        correct_barcodes_of_hamming_distance_one=False,
        save_type=save_type,
        add_fastq_headers=True,
        add_counted_in_count_matrix=False,
        save_transcript_names=True,
        strip_version_number_from_genes=False,
    )
    bus_df = bus_df[["barcode", "UMI", "EC", "read_index", "fastq_header", "gene_names"]].copy()
    bus_df.rename(columns={"gene_names": "vcrs_names"}, inplace=True)

    n_missing_header = int(bus_df["fastq_header"].isna().sum())
    if n_missing_header:
        logger.warning(
            f"{n_missing_header} / {len(bus_df)} reads could not be matched to a FASTQ header (e.g. barcode-corrected "
            "single-cell reads whose corrected barcode differs from the raw barcode). These reads cannot be checked "
            "against the pseudobam and are treated as non-aligning (kept iff count_reads_that_dont_pseudoalign_to_reference_genome=True)."
        )

    # ---- build the VCRS -> reference-locus map ----
    logger.info("Building VCRS locus map from HGVS headers")
    locus_map = _build_vcrs_locus_map(adata.var, convert_transcript_to_genome, gtf=gtf, seq_id_column=seq_id_column, var_column=var_column)

    # ---- compare each read's alignment to its assigned VCRS loci ----
    # A read is dropped only if NONE of its assigned VCRS are consistent with where the read truly
    # aligned. If only SOME of a multi-VCRS read's variants are consistent, the read is kept but its
    # equivalence class (EC) is rewritten to the consistent subset (so, e.g., a read mapping to a
    # gene's true variant but also spuriously k-mer-matching a decoy variant keeps the true one).
    logger.info("Comparing read alignments to assigned VCRS loci")
    with open(f"{kb_count_vcrs_dir}/transcripts.txt", encoding="utf-8") as f:
        transcripts_list = f.read().splitlines()
    name_to_idx = {name: i for i, name in enumerate(transcripts_list)}

    # existing equivalence classes: frozenset(transcript indices) -> EC id
    ec_index_map = {}
    max_ec = -1
    with open(f"{kb_count_vcrs_dir}/matrix.ec", encoding="utf-8") as f:
        for line in f:
            ec_id_str, ids_str = line.rstrip("\n").split("\t")
            ec_id = int(ec_id_str)
            ec_index_map[frozenset(int(x) for x in ids_str.split(","))] = ec_id
            max_ec = max(max_ec, ec_id)
    new_ec_records = []  # (ec_id, [sorted transcript indices]) to append to matrix.ec

    # Collect one entry per surviving read as [corrected_barcode, raw_barcode, umi, read_index,
    # frozenset(consistent transcript idx)].
    # The alignment lookup below is keyed by the *raw* per-file barcode, but we store the collapsed
    # ("good") barcode for the recount. The collapse is applied exactly once here -- note it is NOT
    # idempotent, since a good barcode (e.g. "AAC") can also be a raw key, so it must not be re-mapped
    # downstream. The raw barcode is carried alongside it because it is the only record of which fastq
    # a read came from once the collapse has happened. EC ids are assigned only after the optional
    # double-counting dedup below (which can further narrow a read's transcript set).
    per_read = []
    n_dropped = n_rewritten = n_no_align = 0
    for barcode, umi, read_index, header, vcrs_names in zip(bus_df["barcode"].astype(str), bus_df["UMI"].astype(str), bus_df["read_index"], bus_df["fastq_header"].astype(str), bus_df["vcrs_names"]):
        vcrs_list = list(vcrs_names) if isinstance(vcrs_names, (list, tuple)) else [vcrs_names]
        group_lookup = align_by_group[None] if single_cell else align_by_group.get(barcode, {})
        aln = group_lookup.get(header)

        if aln is None:  # read did not pseudoalign to the reference at all
            n_no_align += 1
            if not count_reads_that_dont_pseudoalign_to_reference_genome:
                n_dropped += 1
                continue
            consistent = [v for v in vcrs_list if v != "dlist"]  # cannot verify -> keep all
        else:
            consistent = []
            for vcrs_name in vcrs_list:
                if vcrs_name == "dlist":
                    continue
                vcrs_locus = locus_map.get(vcrs_name)
                if vcrs_locus is None or _read_alignment_is_consistent(aln, vcrs_locus, check_alignment_position, alignment_position_tolerance):
                    consistent.append(vcrs_name)  # locus None -> cannot verify -> keep

        if not consistent:
            n_dropped += 1
            continue

        consistent_idx = frozenset(name_to_idx[v] for v in consistent if v in name_to_idx)
        if not consistent_idx:
            n_dropped += 1
            continue
        if len(consistent) != len([v for v in vcrs_list if v != "dlist"]):
            n_rewritten += 1
        corrected_barcode = bad_to_good_barcode_dict.get(barcode, barcode) if bad_to_good_barcode_dict else barcode
        per_read.append([corrected_barcode, barcode, umi, read_index, consistent_idx])

    # ---- optional paired double-counting removal (paired-end BULK/SMARTSEQ2 run in single mode) ----
    # Both mates of a fragment now share (corrected_barcode, read_index). Count each consistent VCRS at
    # most once per fragment: the first mate to carry a given transcript keeps it; later mates have it
    # removed from their equivalence class (and are dropped entirely if nothing unique remains). Mirrors
    # the gene_matrix path's "assign to the first read of the pair, give the second read nothing".
    n_double_counted = n_dedup_narrowed = 0
    if avoid_paired_double_counting and bad_to_good_barcode_dict:
        seen_by_fragment = {}  # (corrected_barcode, read_index) -> set of transcript idx already counted for this fragment
        deduped = []
        for corrected_barcode, raw_barcode, umi, read_index, consistent_idx in per_read:
            already = seen_by_fragment.setdefault((corrected_barcode, read_index), set())
            new_idx = consistent_idx - already
            if not new_idx:  # every VCRS on this mate was already counted by its pair -> drop
                n_double_counted += 1
                continue
            if new_idx != consistent_idx:
                n_dedup_narrowed += 1
            already |= new_idx
            deduped.append([corrected_barcode, raw_barcode, umi, read_index, frozenset(new_idx)])
        per_read = deduped

    # ---- assign (possibly new) equivalence-class ids to each kept read ----
    kept_barcodes, kept_raw_barcodes, kept_umis, kept_read_indices, kept_ecs = [], [], [], [], []
    for corrected_barcode, raw_barcode, umi, read_index, consistent_idx in per_read:
        ec_id = ec_index_map.get(consistent_idx)
        if ec_id is None:  # need a new EC for this consistent subset
            max_ec += 1
            ec_id = max_ec
            ec_index_map[consistent_idx] = ec_id
            new_ec_records.append((ec_id, sorted(consistent_idx)))
        kept_barcodes.append(corrected_barcode)
        kept_raw_barcodes.append(raw_barcode)
        kept_umis.append(umi)
        kept_read_indices.append(read_index)
        kept_ecs.append(ec_id)

    logger.info(f"pseudobam QC: {n_dropped} / {len(bus_df)} reads dropped as inconsistent with their assigned VCRS; "
                f"{n_rewritten} reads had their equivalence class narrowed to the consistent variant(s); "
                f"{n_double_counted} paired mates dropped and {n_dedup_narrowed} narrowed to avoid double counting; "
                f"{n_no_align} reads did not pseudoalign to the reference.")

    if n_dropped == 0 and n_rewritten == 0 and n_double_counted == 0 and n_dedup_narrowed == 0:
        logger.info("No inconsistent reads found; returning the original AnnData object.")
        # the position check changed nothing, so the read counts come from the original BUS
        if min_reads is not None:
            adata = apply_read_count_matrix(adata, kb_count_vcrs_dir, min_reads=min_reads, bus_file="output.bus", bustools=bustools)
        # the position check changed nothing, but the allele-fraction check is independent of it
        return apply_vaf_filter(adata, vaf_bam_paths, min_snp_vaf=min_snp_vaf, min_indel_vaf=min_indel_vaf, samtools=samtools, threads=threads)

    # ---- regenerate the count matrix from the filtered/rewritten BUS ----
    new_counts_dir = os.path.join(kb_count_vcrs_dir, "counts_unfiltered_pseudobam")
    kept_records = pd.DataFrame({"barcode": kept_barcodes, "UMI": kept_umis, "EC": kept_ecs})

    # The read index goes into the BUS flag column so the filtered BUS stays traceable to individual
    # reads. A missing index would silently become read 0, so any read whose index did not survive is
    # flagged rather than guessed at.
    read_index_series = pd.to_numeric(pd.Series(kept_read_indices), errors="coerce")
    n_no_index = int(read_index_series.isna().sum())
    if n_no_index:
        logger.warning(f"{n_no_index} / {len(read_index_series)} kept reads have no usable read index; writing -1 for those in the BUS flag column.")
    kept_records["read_index"] = read_index_series.fillna(-1).astype("int64").to_numpy()
    if len(kept_records) > 1 and (kept_records["read_index"] == 0).all():
        logger.warning("Every kept read has read index 0, so the filtered BUS cannot be traced back to individual reads. This usually means kb count was run without --num.")
    if bad_to_good_barcode_dict:  # per-file barcodes were collapsed; keep the originals recoverable
        kept_records["raw_barcode"] = kept_raw_barcodes
    mtx_file = _recount_from_filtered_bus(kb_count_vcrs_dir, kept_records, new_ec_records, bustools, threads, vcrs_t2g, new_counts_dir, mm)
    adata_new = _load_and_align_recounted_adata(mtx_file, adata)

    # Read counts are taken from the *filtered* BUS, so the position check is not silently undone
    # by recounting the reads it dropped. Applied before the VAF filter so that the two precision
    # filters compose: the VAF filter only pileups VCRSs that still hold counts, and on the
    # bustools matrix that set is missing everything its multimapping division zeroed.
    if min_reads is not None:
        adata_new = apply_read_count_matrix(adata_new, new_counts_dir, min_reads=min_reads, bus_file="output_filtered.sorted.bus", bustools=bustools)

    # Allele-fraction filter, applied after the recount so only VCRSs that still hold counts are
    # pileup'd (a catalog reference has millions of columns and nearly all are already empty).
    adata_new = apply_vaf_filter(adata_new, vaf_bam_paths, min_snp_vaf=min_snp_vaf, min_indel_vaf=min_indel_vaf, samtools=samtools, threads=threads)

    adata_output_path = adata_output_path or os.path.join(new_counts_dir, "adata.h5ad")
    if adata_output_path:
        logger.info(f"Saving adjusted AnnData object to {adata_output_path}")
        # This is an intermediate write: vk clean only joins var's list columns into strings at the very
        # end of its run, so the join has to happen here too. It is applied to the written copy only --
        # the returned object keeps its lists, which the rest of vk clean still reads as lists.
        var_with_lists = adata_new.var
        adata_new.var = join_list_columns_for_h5ad(var_with_lists)
        try:
            adata_new.write(adata_output_path)
        finally:
            adata_new.var = var_with_lists

    return adata_new


def merge_bus_df_and_adata_var(bus_df, adata_var, vcrs_column_bus="vcrs_names", vcrs_column_adata="vcrs_id", gene_id_column="gene_id", variant_source=None, seq_id_column="seq_ID", var_column="mutation", reference_genome_t2g=None, gtf=None, columns_to_merge="all"):
    bus_df_columns_original = bus_df.columns.tolist()
    bus_df_columns_original.remove(vcrs_column_bus)

    # Step 1: Explode by vcrs_names - eg [VCRS1;VCRS2, VCRS3] (1 row) --> VCRS1;VCRS2, VCRS3 (2 rows)
    logger.info("Exploding bus_df by vcrs_names")
    exploded = bus_df.explode(vcrs_column_bus, ignore_index=True)

    # Step 2: Add original row index for regrouping
    exploded['original_index'] = pd.Series(
        bus_df.index.repeat(bus_df[vcrs_column_bus].str.len())
    ).reset_index(drop=True)

    # Step 3: Merge with adata_var
    if columns_to_merge is not None and len(columns_to_merge) > 0:
        logger.info("Merging exploded bus_df with adata.var")
        if columns_to_merge == "all":
            columns_to_merge = adata_var.columns.tolist()
        adata_var_subset = adata_var.loc[:, list(columns_to_merge)].copy()  # only keep the necessary columns to merge
        merged = exploded.merge(
            adata_var_subset,
            left_on=vcrs_column_bus,
            right_on=vcrs_column_adata,
            how='left',
        )
    else:
        merged = exploded.copy()

    if "vcrs_header" not in merged.columns:  # means that we did not use a var_id_column, and thus "vcrs_names" represents our (HGVS-like) headers
        merged["vcrs_header"] = merged[vcrs_column_bus]

    #* Step 3.5 Explode and collapse by semicolon - eg VCRS1;VCRS2, VCRS3 (2 rows) --> VCRS1, VCRS2, VCRS3 (3 rows)
    if gene_id_column not in merged.columns:  # I won't enter this condition if vk clean set me up properly
        logger.info("Exploding merged bus_df by vcrs_names to add gene_id_column")
        merged['vcrs_header_individual'] = merged['vcrs_header'].str.split(';')  # don't let the naming be confusing - it will only be original AFTER exploding; before, it will be a list
        merged_exploded = merged[["vcrs_header", "vcrs_header_individual"]].copy().explode('vcrs_header_individual', ignore_index=True).drop_duplicates()
        include_position_information = False if variant_source == "transcriptome" else True
        logger.info("Adding information from variant header to adata.var")
        add_information_from_variant_header_to_adata_var_exploded(merged_exploded, seq_id_column=seq_id_column, var_column=var_column, variant_source=variant_source, t2g_file=reference_genome_t2g, include_position_information=include_position_information, gtf=gtf, gene_id_column=gene_id_column)
        logger.info("Merging exploded bus_df with adata.var again to add gene_id_column")
        grouped_merged = (
            merged_exploded[["vcrs_header", gene_id_column]].groupby("vcrs_header", as_index=False)
            .agg(
                {
                    gene_id_column: list,
                })
            .reset_index(drop=True)
        )
        merged = merged.merge(grouped_merged[["vcrs_header", gene_id_column]], on='vcrs_header', how='left', suffixes=('', '_merged'))
        merged.drop(columns=["vcrs_header_individual"], inplace=True, errors='ignore')
        del merged_exploded, grouped_merged

    # Step 4: Group back by original row index
    logger.info("Grouping back by original row index to aggregate the data")
    # bus_df = (merged
    #         .groupby('original_index')
    #         .agg({
    #           **{col: "first" for col in merged.columns if col in bus_df_columns_original},
    #           **{col: list for col in merged.columns if col not in bus_df_columns_original}
    #         })
    #         .reset_index(drop=True)
    #         .drop(columns=['original_index', vcrs_column_adata], errors='ignore')
    # )
    bus_df_tmp = (merged[['original_index', 'vcrs_header', gene_id_column]]
        .groupby('original_index')
        .agg({
            "vcrs_header": list,
            gene_id_column: list
        })
        .reset_index(drop=True)
        .drop(columns=['original_index', vcrs_column_adata], errors='ignore')
    )
    
    # bus_df = bus_df.merge(bus_df_tmp[['vcrs_key', gene_id_column]], on='vcrs_key', how='left', suffixes=('', '_merged'))  # crashes RAM on laptop
    bus_df[["vcrs_names", gene_id_column]] = bus_df_tmp[["vcrs_header", gene_id_column]]
    del bus_df_tmp

    if gene_id_column in bus_df.columns:  # this must be true now
        bus_df.rename(columns={gene_id_column: "genes_vcrs"}, inplace=True)
    # if "vcrs_header" in bus_df.columns:
    #     bus_df.drop(columns=["vcrs_names"], inplace=True, errors='ignore')  # drop the vcrs_names column (it might be a non-HGVS ID that I want to get rid of)
    #     bus_df.rename(columns={"vcrs_header": "vcrs_names"}, inplace=True)  # make sure vcrs_names is in HGVS format

    return bus_df

    
def barcode_and_umi_agg(group):
    # Compute the union for vcrs_names, and grab the corresponding genes_vcrs - I take the union rather than the mode as I previously did because each UMI only guarantees that the reads are derived from the same gene, but this does not mean that they must be duplicates of each other (i.e., they can cover different parts of the gene)
    vcrs_names_union, vcrs_genes_union = [], []
    for vcrs_name, vcrs_gene in zip(group["vcrs_names"], group["genes_vcrs"]):
        if vcrs_name not in vcrs_names_union:
            vcrs_names_union.append(vcrs_name)
            vcrs_genes_union.append(vcrs_gene)
    
    read_index_agg = list(group["read_index"])  # Aggregate read_index into a list
    genes_standard_agg = sorted(set.union(*map(set, group["genes_standard"])))  # Aggregate genes_standard using union of sets
    pseudoaligns_to_reference_genome = len(genes_standard_agg) > 0
    
    return pd.Series({
        "read_index": read_index_agg,
        "vcrs_names": vcrs_names_union,
        "genes_standard": genes_standard_agg,
        "genes_vcrs": vcrs_genes_union,
        "pseudoaligns_to_reference_genome": pseudoaligns_to_reference_genome,
        "original_index": list(group["original_index"])
    })


def normal_genome_validation_tallying(vcrs_names_list_final, vcrs_names_list, number_of_counts_added=0, number_of_counts_removed=0, number_of_reads_changed=0, mm=False, mm_original=False):
    len_vcrs_names_list_final = len(vcrs_names_list_final)
    len_vcrs_names_list = len(vcrs_names_list)

    if mm and len_vcrs_names_list_final > 1 and (len_vcrs_names_list_final < len_vcrs_names_list):
        if not mm_original:
            number_of_counts_added += len_vcrs_names_list_final  # it was NOT counted originally (due to mapping to >1 VCRS and lack of mm_original) but is counted now (due to mm)
        else:
            number_of_counts_removed += len_vcrs_names_list - len_vcrs_names_list_final  # it was counted originally (due to mm_original) and is counted now (due to mm), but to a lesser extent
        number_of_reads_changed += 1
    
    elif len_vcrs_names_list_final == 1 and (len_vcrs_names_list_final < len_vcrs_names_list):
        if not mm_original:
            number_of_counts_added += 1  # it was NOT counted originally (due to mapping to >1 VCRS and lack of mm_original) but is counted now (with or without mm, due to length 1)
        else:
            number_of_counts_removed += len_vcrs_names_list - len_vcrs_names_list_final  # it was counted originally (due to mm_original) and is counted now (with or without mm, due to length 1), but to a lesser extent
        number_of_reads_changed += 1
    
    elif len_vcrs_names_list_final == 0:
        if not mm_original:
            if len_vcrs_names_list == 1:  # when mm_original False, only originally counted when length 1
                number_of_counts_removed += 1  # it was counted (due to length 1) originally but is NOT counted now (due to length 0)    # the read was only originally counted if it mapped to exactly 1 VCRS
                number_of_reads_changed += 1
        else:
            number_of_counts_removed += len_vcrs_names_list  # it was counted originally (due to mm_original) but is NOT counted now (due to length 0)    # all of these reads were originally counted but now are not
            number_of_reads_changed += 1
    
    elif len_vcrs_names_list_final > 1 and mm and not mm_original:
        number_of_counts_added += len_vcrs_names_list_final  # no difference in initial and final VCRSs, but simply a difference in multimapping behavior
    
    elif len_vcrs_names_list_final > 1 and mm_original and not mm:
        number_of_counts_removed += len_vcrs_names_list - len_vcrs_names_list_final  # no difference in initial and final VCRSs, but simply a difference in multimapping behavior
    
    return number_of_counts_added, number_of_counts_removed, number_of_reads_changed

































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

        info = technology_info.get(technology.upper())
        barcode_umi = info["barcode_umi"] if info is not None else None
        if barcode_umi is None:
            print(f"technology {technology} currently not supported. Supported are {[t for t, i in technology_info.items() if i['barcode_umi'] is not None]}")
        else:
            barcode_start = barcode_umi["barcode_start"]
            barcode_end = barcode_umi["barcode_end"]
            umi_start = barcode_umi["umi_start"]
            umi_end = barcode_umi["umi_end"]

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
def match_paired_ends_after_single_end_run(bus_df_path, gene_name_type="vcrs_id", id_to_header_dataframe=None):
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
        id_to_header_dict = make_mapping_dict(id_to_header_dataframe, dict_key="id")

        bus_df["vcrs_header_list"] = bus_df["gene_names_final"].apply(lambda gene_list: [id_to_header_dict.get(gene, gene) for gene in gene_list])

        bus_df["vcrs_header_list_pair"] = bus_df["gene_names_final_pair"].apply(lambda gene_list: [id_to_header_dict.get(gene, gene) for gene in gene_list])

        bus_df["ensembl_transcript_list"] = [strip_gene_name_from_seq_id(value.split(":")[0]) for value in bus_df["vcrs_header_list"]]
        bus_df["ensembl_transcript_list_pair"] = [strip_gene_name_from_seq_id(value.split(":")[0]) for value in bus_df["vcrs_header_list_pair"]]

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
def adjust_variant_adata_by_normal_gene_matrix_original(adata, kb_count_vcrs_dir, kallisto_quant_reference_genome_dir, id_to_header_dataframe=None, adata_output_path=None, vcrs_t2g=None, t2g_standard=None, fastq_file_list=None, mm=False, technology="bulk", parity="single", bustools="bustools", ignore_barcodes=False, check_only=False, chunksize=None):
    if not adata:
        adata = f"{kb_count_vcrs_dir}/counts_unfiltered/adata.h5ad"
    if isinstance(adata, str):
        adata = ad.read_h5ad(adata)

    fastq_file_list = load_in_fastqs(fastq_file_list)
    fastq_file_list = sort_fastq_files_for_kb_count(fastq_file_list, technology=technology, check_only=check_only)

    bus_df_mutation_path = f"{kb_count_vcrs_dir}/bus_df.csv"
    bus_df_standard_path = f"{kallisto_quant_reference_genome_dir}/bus_df.csv"

    if not os.path.exists(bus_df_mutation_path):
        bus_df_mutation = make_bus_df(
            kb_count_out=kb_count_vcrs_dir,
            fastq_file_list=fastq_file_list,
            t2g_file=vcrs_t2g,
            mm=mm,
            technology=technology,
            parity=parity,
            bustools=bustools,
            check_only=check_only,
            chunksize=chunksize
        )
    else:
        bus_df_mutation = pd.read_csv(bus_df_mutation_path)

    bus_df_mutation["gene_names_final"] = bus_df_mutation["gene_names_final"].apply(safe_literal_eval)
    bus_df_mutation.rename(columns={"gene_names_final": "VCRS_headers_final", "count": "count_value"}, inplace=True)

    if id_to_header_dataframe:
        bus_df_mutation.rename(columns={"VCRS_headers_final": "VCRS_ids_final"}, inplace=True)
        id_to_header_dict = make_mapping_dict(id_to_header_dataframe, dict_key="id")
        bus_df_mutation["VCRS_headers_final"] = bus_df_mutation["VCRS_ids_final"].apply(lambda name_list: [id_to_header_dict.get(name, name) for name in name_list])

    bus_df_mutation["transcripts_VCRS"] = bus_df_mutation["VCRS_headers_final"].apply(lambda string_list: tuple({strip_gene_name_from_seq_id(s.split(":")[0]) for s in string_list}))

    if not os.path.exists(bus_df_standard_path):
        bus_df_standard = make_bus_df(
            kallisto_out=kallisto_quant_reference_genome_dir,
            fastq_file_list=fastq_file_list,  # make sure this is in the same order as passed into kb count - [sample1, sample2, etc] OR [sample1_pair1, sample1_pair2, sample2_pair1, sample2_pair2, etc]
            t2g_file=t2g_standard,
            mm=mm,
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

    if issparse(mutant_X) and issparse(wt_X):
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


def add_vcf_info_to_cosmic_tsv(cosmic_tsv=None, reference_genome_fasta=None, cosmic_df_out=None, sequences="cds", cosmic_version=101, cosmic_email=None, cosmic_password=None):
    import gget
    from varseek.utils.varseek_build_utils import convert_mutation_cds_locations_to_cdna

    if cosmic_tsv is None:
        cosmic_tsv = f"CancerMutationCensus_AllData_Tsv_v{cosmic_version}_GRCh37/CancerMutationCensus_AllData_v{cosmic_version}_GRCh37.tsv"
    if reference_genome_fasta is None:
        reference_genome_fasta = "Homo_sapiens.GRCh37.dna.primary_assembly.fa"
    
    cosmic_cdna_info_df = None
    cosmic_cdna_info_csv = cosmic_tsv.replace(".tsv", "_mutation_workflow.csv")
    reference_genome_fasta_dir = os.path.dirname(reference_genome_fasta) if os.path.dirname(reference_genome_fasta) else "."

    if not os.path.exists(cosmic_tsv) or (not os.path.exists(cosmic_cdna_info_csv) and sequences == "cdna"):
        reference_out_cosmic = os.path.dirname(os.path.dirname(cosmic_tsv)) if os.path.dirname(os.path.dirname(cosmic_tsv)) else "."
        gget.cosmic(
            None,
            grch_version=37,
            cosmic_version=cosmic_version,
            out=reference_out_cosmic,
            mutation_class="cancer",
            download_cosmic=True,
            gget_mutate=True,
            keep_genome_info=True,
            remove_duplicates=True,
            email=cosmic_email,
            password=cosmic_password,
        )
        if sequences == "cdna":
            cds_file = os.path.join(reference_genome_fasta_dir, "Homo_sapiens.GRCh37.cds.all.fa")
            cdna_file = os.path.join(reference_genome_fasta_dir, "Homo_sapiens.GRCh37.cdna.all.fa")
            if not os.path.exists(cds_file):
                subprocess.run(["gget", "ref", "-w", "cds", "-r", "93", "--out_dir", reference_genome_fasta_dir, "-d", "human_grch37"], check=True)
                subprocess.run(["gunzip", f"{cds_file}.gz"], check=True)
            if not os.path.exists(cdna_file):
                subprocess.run(["gget", "ref", "-w", "cdna", "-r", "93", "--out_dir", reference_genome_fasta_dir, "-d", "human_grch37"], check=True)
                subprocess.run(["gunzip", f"{cdna_file}.gz"], check=True)
            cosmic_cdna_info_df = convert_mutation_cds_locations_to_cdna(input_csv_path=cosmic_cdna_info_csv, output_csv_path=cosmic_cdna_info_csv, cds_fasta_path=cds_file, cdna_fasta_path=cdna_file)
    if not os.path.exists(reference_genome_fasta):
        subprocess.run(["gget", "ref", "-w", "dna", "-r", "93", "--out_dir", reference_genome_fasta_dir, "-d", "human_grch37"], check=True)
        subprocess.run(["gunzip", f"{reference_genome_fasta}.gz"], check=True)

    # load in COSMIC tsv with columns CHROM, POS, ID, REF, ALT
    cosmic_df = pd.read_csv(cosmic_tsv, sep="\t", usecols=["Mutation genome position GRCh37", "GENOMIC_WT_ALLELE_SEQ", "GENOMIC_MUT_ALLELE_SEQ", "ACCESSION_NUMBER", "Mutation CDS", "MUTATION_URL"])

    if sequences == "cdna":
        if not isinstance(cosmic_cdna_info_df, pd.DataFrame):
            cosmic_cdna_info_df = pd.read_csv(cosmic_cdna_info_csv, usecols=["mutation_id", "mutation_cdna"])
        cosmic_cdna_info_df = cosmic_cdna_info_df.rename(columns={"mutation_cdna": "Mutation cDNA"})

    cosmic_df = add_variant_type(cosmic_df, "Mutation CDS")

    cosmic_df["ACCESSION_NUMBER"] = cosmic_df["ACCESSION_NUMBER"].str.split(".").str[0]

    cosmic_df[["CHROM", "GENOME_POS"]] = cosmic_df["Mutation genome position GRCh37"].str.split(":", expand=True)
    # cosmic_df['CHROM'] = cosmic_df['CHROM'].apply(convert_chromosome_value_to_int_when_possible)
    cosmic_df[["POS", "GENOME_END_POS"]] = cosmic_df["GENOME_POS"].str.split("-", expand=True)

    cosmic_df = cosmic_df.rename(columns={"GENOMIC_WT_ALLELE_SEQ": "REF", "GENOMIC_MUT_ALLELE_SEQ": "ALT", "MUTATION_URL": "mutation_id"})

    if sequences == "cds":
        cosmic_df["ID"] = cosmic_df["ACCESSION_NUMBER"] + ":" + cosmic_df["Mutation CDS"]
    elif sequences == "cdna":
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
    cosmic_df.loc[(cosmic_df["GENOME_END_POS"].astype(int) == 1) & (cosmic_df["variant_type"] == "insertion"), "original_nucleotide"] = cosmic_df.loc[(cosmic_df["GENOME_END_POS"].astype(int) == 1) & (cosmic_df["variant_type"] == "insertion"), ["CHROM", "GENOME_END_POS"]].progress_apply(lambda row: get_nucleotide_from_reference(row["CHROM"], int(row["GENOME_END_POS"])), axis=1)

    # deal with start of 1, deletion
    cosmic_df.loc[(cosmic_df["POS"].astype(int) == 1) & (cosmic_df["variant_type"] == "deletion"), "original_nucleotide"] = cosmic_df.loc[(cosmic_df["POS"].astype(int) == 1) & (cosmic_df["variant_type"] == "deletion"), ["CHROM", "GENOME_END_POS"]].progress_apply(lambda row: get_nucleotide_from_reference(row["CHROM"], int(row["GENOME_END_POS"]) + 1), axis=1)

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
def write_to_vcf(adata_var, output_file, save_vcf_samples=False, adata=None, buffer_size=10_000):
    """
    Write adata.var DataFrame to a VCF file.

    Parameters:
        adata_var (pd.DataFrame): DataFrame with VCF columns (CHROM, POS, REF, ALT, ID, AO, AF, NS).
        output_file (str): Path to the output VCF file.
    """
    if save_vcf_samples:
        filtered_VCRSs = adata_var['ID'].astype(str).tolist()
        adata_filtered = adata[:, adata.var['vcrs_id'].isin(set(filtered_VCRSs))].copy()  # Subset adata to keep only the variables in filtered_ids
        if adata_var['ID'].tolist() != adata_filtered.var['vcrs_id'].tolist():  # different orders
            correct_order = adata_filtered.var.set_index('vcrs_id').loc[adata_var['ID']].index  # Get the correct order of indices based on adata_var['ID']
            adata_filtered = adata_filtered[:, correct_order].copy()  # Reorder adata_filtered.var and adata_filtered.X
    
    # Open VCF file for writing
    with open(output_file, "w", encoding="utf-8") as vcf_file:
        # TODO: eventually add ref depth in addition to alt depth (I would add RO (ref depth) and DP (ref+alt depth) and AF (alt/[ref+alt]) to INFO, add DP to FORMAT/samples, and either add RO or AD to FORMAT/samples (AD is more standardized but would change the output of the varseek pipeline))
        # Write VCF header
        vcf_file.write("##fileformat=VCFv4.2\n")
        vcf_file.write("##source=varseek\n")
        vcf_file.write('##INFO=<ID=NS,Number=1,Type=Integer,Description="Number of Samples">\n')
        vcf_file.write('##INFO=<ID=AO,Number=1,Type=Integer,Description="ALT Depth">\n')
        # vcf_file.write('##INFO=<ID=RO,Number=1,Type=Integer,Description="REF Depth">\n')
        # vcf_file.write('##INFO=<ID=DP,Number=1,Type=Integer,Description="Total depth">\n')
        # vcf_file.write('##INFO=<ID=AF,Number=A,Type=Float,Description="Variant Allele Frequency">\n')
        headers = "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO"
        if save_vcf_samples:
            vcf_file.write('##FORMAT=<ID=AO,Number=1,Type=Integer,Description="ALT Depth per sample">\n')
            # vcf_file.write('##FORMAT=<ID=RO,Number=1,Type=Integer,Description="REF Depth per sample">\n')  #? use RO or AD but not both
            # vcf_file.write('##FORMAT=<ID=AD,Number=R,Type=Integer,Description="Allelic depths for the REF and ALT alleles per sample">\n')
            # vcf_file.write('##FORMAT=<ID=DP,Number=1,Type=Integer,Description="Total Depth per sample">\n')
            headers += "\tFORMAT\t" + "\t".join(adata_filtered.obs_names)
        vcf_file.write(f"{headers}\n")

        # Extract all column data as NumPy arrays (faster access)
        chroms, poss, ids, refs, alts, aos, nss, afs = (
            adata_var["CHROM"].values, adata_var["POS"].values, adata_var["ID"].values,
            adata_var["REF"].values, adata_var["ALT"].values,
            adata_var["AO"].values if "AO" in adata_var else np.full(len(adata_var), np.nan),  # Handle optional AO column,
            adata_var["NS"].values if "NS" in adata_var else np.full(len(adata_var), np.nan),  # Handle optional NS column,
            adata_var["AF"].values if "AF" in adata_var else np.full(len(adata_var), np.nan)  # Handle optional AF column
        )

        # Iterate over pre-extracted values
        buffer = []
        for idx, (chrom, pos, id_, ref, alt, ao, ns, af) in enumerate(zip(chroms, poss, ids, refs, alts, aos, nss, afs)):
            # Construct INFO field efficiently
            info_fields = [f"AO={int(ao)}" if pd.notna(ao) else None,
                        f"NS={ns}" if pd.notna(ns) else None]
            if pd.notna(af):
                info_fields.append(f"AF={af}")

            info = ";".join(filter(None, info_fields))

            vcf_line = f"{chrom}\t{pos}\t{id_}\t{ref}\t{alt}\t.\tPASS\t{info}"
            if save_vcf_samples:
                X_col = adata_filtered.X[:, idx]
                if hasattr(X_col, "toarray"):  # Check if it's sparse
                    X_col = X_col.toarray()
                vcf_line += "\tAO\t" + "\t".join(map(str, X_col.flatten().tolist()))

            buffer.append(f"{vcf_line}\n")

            # Write to file in chunks
            if len(buffer) >= buffer_size:
                vcf_file.writelines(buffer)
                buffer.clear()  # Reset buffer
        
        # Write any remaining lines
        if buffer:
            vcf_file.writelines(buffer)

def cleaned_adata_to_vcf(variant_data, vcf_data_df, output_vcf = "variants.vcf", save_vcf_samples=False, adata=None):
    # variant_data should be adata or adata.var/df
    # if variant_data is adata, then adata will be automatically populated; if it is df, then adata will be None unless explicitely provided
    if isinstance(variant_data, str) and os.path.isfile(variant_data) and variant_data.endswith(".h5ad"):
        adata = ad.read_h5ad(variant_data)
        adata_var = adata.var
    elif isinstance(variant_data, ad.AnnData):
        adata = variant_data
        adata_var = variant_data.var
    elif isinstance(variant_data, str) and os.path.isfile(variant_data) and variant_data.endswith(".csv"):
        adata_var = pd.read_csv(variant_data)
    elif isinstance(variant_data, pd.DataFrame):
        adata_var = variant_data

    # Ensure proper columns
    if isinstance(vcf_data_df, str) and os.path.isfile(vcf_data_df) and vcf_data_df.endswith(".csv"):
        vcf_data_df = pd.read_csv(vcf_data_df)
    elif isinstance(vcf_data_df, pd.DataFrame):
        pass
    else:
        raise ValueError("vcf_data_df must be a CSV file path or a pandas DataFrame")
    
    if any(col not in vcf_data_df.columns for col in ["ID", "CHROM", "POS", "REF", "ALT"]):
        raise ValueError("vcf_data_df must contain columns ID, CHROM, POS, REF, ALT")
    if any(col not in adata_var.columns for col in ["vcrs_id", "vcrs_count", "number_obs"]):
        raise ValueError("adata_var must contain columns vcrs_id, vcrs_count, number_obs")
    if save_vcf_samples and not isinstance(adata, ad.AnnData):
        raise ValueError("adata must be provided as an anndata object or path to an anndata object if save_vcf_samples is True")

    output_vcf = str(output_vcf)  # for Path

    # only keep the VCRSs that have a count > 0, and only keep relevant columns
    adata_var_temp = adata_var[["vcrs_id", "vcrs_count", "number_obs"]].loc[adata_var["vcrs_count"] > 0].copy()

    # make copy column that won't be exploded so that I know how to groupby later
    adata_var_temp["vcrs_id_copy"] = adata_var_temp["vcrs_id"]

    # rename to have VCF-like column names
    adata_var_temp.rename(columns={"vcrs_count": "AO", "number_obs": "NS", "vcrs_id": "ID"}, inplace=True)

    # explode across semicolons so that I can merge in vcf_data_df
    adata_var_temp = adata_var_temp.assign(
        ID=adata_var_temp["ID"].str.split(";")
    ).explode("ID").reset_index(drop=True)

    # merge in vcf_data_df (eg cosmic_df)
    adata_var_temp = adata_var_temp.merge(vcf_data_df, on="ID", how="left")

    # collapse across semicolons so that I get my VCRSs back
    adata_var_temp = (
        adata_var_temp
        .groupby("vcrs_id_copy", sort=False)  # Group by vcrs_id_copy while preserving order
        .agg({
            "ID": lambda x: ";".join(x),  # Reconstruct ID as a single string
            "CHROM": set,  # Collect CHROM values in the same order as rows
            "POS": set,    # Collect POS values
            "REF": set,    # Collect REF values
            "ALT": set,    # Collect ALT values
            "AO": "first",
            "NS": "first",
        })
        .reset_index()  # Reset index for cleaner result
        .drop(columns=["vcrs_id_copy"])
    )

    # only keep the VCRSs that have a single value for CHROM, POS, REF, ALT - there could be some merged headers that have identical VCF information (eg same genomic mutation but for different splice variants), so I can't just drop all merged headers
    for col in ["CHROM", "POS", "REF", "ALT"]:
        adata_var_temp = adata_var_temp[adata_var_temp[col].apply(lambda x: len(set(x)) == 1)].copy()
        adata_var_temp[col] = adata_var_temp[col].apply(lambda x: list(x)[0])

    # write to VCF
    buffer_size = 10_000 if not save_vcf_samples else 1_000  # ensure buffer is smaller when using samples
    write_to_vcf(adata_var_temp, output_vcf, save_vcf_samples=save_vcf_samples, adata=adata, buffer_size=buffer_size)



# # TODO: make sure this works for rows with just ID and everything else blank (due to different mutations being concatenated)
# def write_vcfs_for_rows(adata, adata_wt_vcrs, adata_vaf, output_dir):
#     """
#     Write a VCF file for each row (variant) in adata.var.

#     Parameters:
#         adata: AnnData object with mutant counts.
#         adata_wt_vcrs: AnnData object with wild-type counts.
#         adata_vaf: AnnData object with VAF values.
#         output_dir: Directory to save VCF files.
#     """
#     for idx, row in adata.var.iterrows():
#         # Extract VCF fields from adata.var
#         chrom = row["CHROM"]
#         pos = row["POS"]
#         var_id = row["ID"]
#         ref = row["REF"]
#         alt = row["ALT"]
#         vcrs_id = row["vcrs_id"]  # This is the index for the column in the matrices

#         # Extract corresponding matrix values
#         mutant_counts = adata[:, vcrs_id].X.flatten()  # Extract as 1D array
#         wt_counts = adata_wt_vcrs[:, vcrs_id].X.flatten()  # Extract as 1D array
#         vaf_values = adata_vaf[:, vcrs_id].X.flatten()  # Extract as 1D array

#         # Create VCF file for the row
#         output_file = f"{output_dir}/{var_id}.vcf"
#         with open(output_file, "w", encoding="utf-8") as vcf_file:
#             # Write VCF header
#             vcf_file.write("##fileformat=VCFv4.2\n")
#             vcf_file.write('##INFO=<ID=RD,Number=1,Type=Integer,Description="Total Depth">\n')
#             vcf_file.write('##INFO=<ID=AF,Number=A,Type=Float,Description="Allele Frequency">\n')
#             vcf_file.write('##INFO=<ID=NS,Number=1,Type=Integer,Description="Number of Samples">\n')
#             vcf_file.write("#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\n")

#             # Iterate through samples (rows in the matrix)
#             for sample_idx, mutant_count in enumerate(mutant_counts):
#                 # Calculate RD and AF
#                 rd = mutant_count + wt_counts[sample_idx]
#                 af = vaf_values[sample_idx]

#                 # INFO field
#                 info = f"RD={int(rd)};AF={af:.3f};NS=1"

#                 # Write VCF row
#                 vcf_file.write(f"{chrom}\t{pos}\t{var_id}\t{ref}\t{alt}\t.\tPASS\t{info}\n")



def assign_gene_id(seq_id, position, gene_df):
    try:
        pos = int(position)
    except:
        return "unknown"
    match = gene_df[
        (gene_df["chromosome"] == seq_id) & 
        (gene_df["start"] <= pos) & 
        (gene_df["end"] >= pos)
    ]
    return match["gene_id"].iloc[0] if not match.empty else "unknown"

def assign_gene_id_original(seq_id, variant, position, gene_df):
    if ":c." in variant:
        return seq_id  # transcriptome variant → seq_id is transcript
    try:
        pos = int(position)
    except:
        return "unknown"
    match = gene_df[
        (gene_df["chromosome"] == seq_id) & 
        (gene_df["start"] <= pos) & 
        (gene_df["end"] >= pos)
    ]
    return match["gene_id"].iloc[0] if not match.empty else "unknown"

def assign_gene_ids_row(row, gene_df):
    return [
        assign_gene_id_original(seq_id, variant, pos, gene_df)
        for seq_id, variant, pos in zip(row["chromosomes"], row["variants_tmp"], row["start_nucleotide_positions"])
    ]
    

# def assign_transcript_id(row, transcript_df):
#     seq_id = row["chromosome"]
#     position = row["variant_start_genome_position"]
#     source = row["variant_source"]

#     if source == "genome":
#         # Query transcript_df for transcript mapping
#         match = transcript_df[
#             (transcript_df["chromosome"] == seq_id) & 
#             (transcript_df["start"] <= int(position)) & 
#             (transcript_df["end"] >= int(position))
#         ]
#         transcript_id = match["transcript_ID"] if not match.empty else "unknown"
#     else:
#         transcript_id = seq_id  # If not genome, use seq_ID directly

#     return transcript_id

#     # use it with adata_var_exploded.apply(lambda row: assign_transcript_id(row, transcript_df), axis=1)

def map_genome_to_cds(chrom_pos, transcript_id, gtf_df):
    cds_df = gtf_df[
        (gtf_df["feature"] == "exon") &
        (gtf_df["transcript_ID"] == transcript_id)
    ].copy()

    if cds_df.empty:
        return None  # no CDS mapping found

    strand = cds_df.iloc[0]["strand"]
    if strand == "+":
        cds_df = cds_df.sort_values(by="start")
    else:
        cds_df = cds_df.sort_values(by="end", ascending=False)

    current_cds_base = 1
    for _, row in cds_df.iterrows():
        start, end = row["start"], row["end"]

        if start <= chrom_pos <= end:
            offset = chrom_pos - start if strand == "+" else end - chrom_pos
            return current_cds_base + offset
        else:
            current_cds_base += (end - start + 1)

    return None  # position not in CDS

def assign_transcript_and_cds(row, transcript_df, gtf_df):
    seq_id = row["chromosome"]
    position_start = int(row["start_variant_position_genome"])
    position_end = int(row["end_variant_position_genome"])
    source = row["variant_source"]

    if source == "genome":
        match = transcript_df[
            (transcript_df["chromosome"] == seq_id) & 
            (transcript_df["start"] <= position_start) & 
            (transcript_df["end"] >= position_end)
        ]

        if match.empty:
            return pd.Series(["unknown", None, None], index=["transcript_ID", "start_variant_position", "end_variant_position"])
        
        transcript_id = match.iloc[0]["transcript_ID"]
    else:
        transcript_id = seq_id

    cds_start = map_genome_to_cds(position_start, transcript_id, gtf_df)
    cds_end = map_genome_to_cds(position_end, transcript_id, gtf_df)

    return pd.Series([transcript_id, cds_start, cds_end], index=["transcript_ID", "start_variant_position", "end_variant_position"])


# def assign_transcript_ids(adata_var_exploded, transcript_df):
#     # Merge transcript_df into adata_var_exploded based on genome position
#     merged_df = adata_var_exploded.merge(
#         transcript_df,
#         on="chromosome",
#         how="left",
#         suffixes=("", "_transcript")  # Avoid column name conflicts
#     )

#     # Assign transcript_ID based on conditions
#     merged_df["transcript_ID"] = merged_df.apply(
#         lambda row: row["transcript_ID_transcript"] 
#         if (row["variant_source"] == "genome") and (row["start"] <= row["start_variant_position_genome"] <= row["end"])
#         else row["chromosome"],
#         axis=1
#     )

#     # Keep only necessary columns
#     adata_var_exploded = merged_df[adata_var_exploded.columns.tolist() + ["transcript_ID"]]

def get_variant_sources_normal_genome(vcr_list):
    result = []
    for v in vcr_list:
        if ":c." in v:
            result.append("transcriptome")
        elif ":g." in v:
            result.append("genome")
        else:
            result.append("unknown")
    results_unique = set(result)
    if results_unique == {"transcriptome"} or results_unique == {"genome"}:
        unique_value = results_unique.pop()  # get the single value from the set
    else:
        unique_value = "multiple"
    return result, unique_value


def remove_variants_from_adata_for_stranded_technologies(adata, strand_bias_end, read_length, header_column="vcrs_id", variant_position_annotations=None, variant_source=None, gtf=None, forgiveness=100, seq_id_column="seq_ID", var_column="mutation", plot_histogram=True, out="."):
    #* Type-checking
    if isinstance(adata, str):  # adata is anndata object or path to h5ad
        adata = ad.read_h5ad(adata)
    elif isinstance(adata, ad.AnnData):
        pass
    else:
        raise ValueError("adata must be an AnnData object or a path to an AnnData object")
    
    if strand_bias_end not in {"5p", "3p"}:
        raise ValueError("strand_bias_end must be either '5p' or '3p'")
    
    if isinstance(read_length, (str, float)):
        read_length = int(read_length)
    if not isinstance(read_length, int):
        raise ValueError("read_length must be an integer")
    
    if header_column not in adata.var.columns:
        raise ValueError(f"header_column {header_column} not found in adata.var columns")
    
    if variant_position_annotations in {"chromosome", "gene"}:
        variant_source = "genome"
    elif variant_position_annotations in {"cdna", "cds"}:
        variant_source = "transcriptome"
    elif variant_position_annotations is None:
        pass
    else:
        raise ValueError("variant_position_annotations must be either 'chromosome', 'gene', 'cdna', 'cds', or None.")
    if variant_source not in {None, "transcriptome", "genome"}:
        raise ValueError("variant_source must be either None, 'transcriptome', or 'genome'")

    
    #* Load in gtf df if needed
    if not (variant_position_annotations == "cdna" and strand_bias_end == "5p"):
        if gtf is None:
            raise ValueError("gtf must be provided if variant_position_annotations is not 'cdna' or strand_bias_end is '3p'")
        if isinstance(gtf, str):
            gtf_cols = ["chromosome", "source", "feature", "start", "end", "score", "strand", "frame", "attributes"]
            gtf_df = pd.read_csv(gtf, sep="\t", comment="#", names=gtf_cols)
        elif isinstance(gtf, pd.DataFrame):
            gtf_df = gtf.copy()
        else:
            raise ValueError("gtf must be a path to a GTF file or a pandas DataFrame")
        if "transcript_ID" not in gtf_df.columns:
            gtf_df["region_length"] = gtf_df["end"] - gtf_df["start"] + 1  # this corresponds to the unspliced transcript length, NOT the spliced transcript/CDS length (which is what I care about)
            gtf_df["transcript_ID"] = gtf_df["attributes"].str.extract(r'transcript_id "([^"]+)"')
            transcript_df = gtf_df[gtf_df["feature"] == "transcript"].copy()
            transcript_df = transcript_df.drop_duplicates(subset="transcript_ID", keep="first")
        else:
            transcript_df = gtf_df.copy()
        
        five_prime_lengths = []
        three_prime_lengths = []
        transcript_lengths = []
        for transcript_id in transcript_df["transcript_ID"]:
            specific_transcript_df = gtf_df.loc[gtf_df["transcript_ID"] == transcript_id].copy()
            five_sum = specific_transcript_df[(specific_transcript_df["feature"] == "five_prime_utr")]["region_length"].sum()
            three_sum = specific_transcript_df[(specific_transcript_df["feature"] == "three_prime_utr")]["region_length"].sum()
            exon_sum = specific_transcript_df[(specific_transcript_df["feature"] == "exon")]["region_length"].sum()
            transcript_length = five_sum + three_sum + exon_sum

            five_prime_lengths.append(five_sum)
            three_prime_lengths.append(three_sum)
            transcript_lengths.append(transcript_length)
        transcript_df["five_prime_utr_length"] = five_prime_lengths
        transcript_df["three_prime_utr_length"] = three_prime_lengths
        transcript_df["transcript_length"] = transcript_lengths

        transcript_df["utr_length_preceding_transcript"] = transcript_df["five_prime_utr_length"]
        # # incorrect code, since five_prime_utr always references the transcript and not the genome, and therefore five_prime_utr is always the one upstream
        # transcript_df["utr_length_preceding_transcript"] = transcript_df.apply(
        #     lambda row: row["three_prime_utr_length"] if row["strand"] == "-" else row["five_prime_utr_length"],
        #     axis=1
        # )

    #* Explode adata.var
    adata_var = adata.var.copy()
    if plot_histogram and "vcrs_count" not in adata_var.columns:
        adata_var["vcrs_count"] = adata.X.sum(axis=0).A1 if hasattr(adata.X, "A1") else np.asarray(adata.X.sum(axis=0)).flatten()
        adata_var["vcrs_count"] = adata_var["vcrs_count"].fillna(0).astype("Int32")
    adata_var_exploded = adata_var.assign(vcrs_header_individual=adata_var[header_column].str.split(";")).explode("vcrs_header_individual")

    #* Add column information - seq_id, variant, nucleotide positions, etc
    adata_var_exploded = add_information_from_variant_header_to_adata_var_exploded(adata_var_exploded, seq_id_column=seq_id_column, var_column=var_column, variant_source=variant_source, include_gene_information=False)
    unique_vcrs_sources = adata_var_exploded["variant_source"].unique()
    variant_source = "combined" if len(unique_vcrs_sources) > 1 else unique_vcrs_sources[0]

    #* Classify variant source
    if not variant_position_annotations:
        if variant_source == "genome":
            logger.warning("variant_position_annotations not specified for adjusting by strand bias function, so assuming chromosome position annotations.")
            variant_position_annotations = "chromosome"  # could be chromosome or gene, but HGVSG (as well as VCF) is chromosome
        elif variant_source == "transcriptome":
            logger.warning("variant_position_annotations not specified for adjusting by strand bias function, so assuming CDS position annotations.")
            variant_position_annotations = "cds"  # could be cds or cdna, but HGVSC is cds
    if variant_position_annotations != "cdna" and gtf is None:
        raise ValueError("gtf must be provided if variants annotated from a source other than cDNA (i.e., chromosome position, gene position, or CDS position) are present in adata.var")
    if variant_position_annotations is None:  # only if variant_source is combined
        raise ValueError("variant_position_annotations must be specified, or all annotations must come from the same source (i.e., no mix of g.'s and c.'s).")

    #* Assign transcript ID for genome variants
    if variant_source != "transcriptome":
        if variant_position_annotations == "chromosome":
            adata_var_exploded.rename(columns={"seq_ID": "chromosome", "start_variant_position": "start_variant_position_genome", "end_variant_position": "end_variant_position_genome"}, inplace=True)
            # assign_transcript_ids(adata_var_exploded, transcript_df)  # faster (works with merge instead of apply) but RAM-intensive (left-merges GTF into adata_var_exploded by chromosome alone)
            # TODO: note that this assigns a chromosome position to the first transcript ID that meets criteria (within the range of chromosome and mutation position), but perhaps I want to consider all transcript IDs in the future and use the most conservative of them
            forgiveness += 50  # a hot fix, to be removed when I address the TODO above
            logger.warning("Doing strand bias correction for chromosome position variants. This may not be accurate if the chromosome position overlaps multiple transcripts.")
            adata_var_exploded[["transcript_ID", "start_variant_position", "end_variant_position"]] = (
                adata_var_exploded.apply(lambda row: assign_transcript_and_cds(row, transcript_df, gtf_df), axis=1)
            )
        elif variant_position_annotations == "gene":
            # TODO: implement this - I need to convert gene positions to CDS positions.
            raise NotImplementedError("Removing variants based on gene positions is not implemented yet. Please provide chromosome, cDNA, or CDS annotations for strand bias removal.")
        variant_position_annotations = "cds"  # I have now converted chromosome information to cds information
        variant_source = "transcriptome"
    else:
        adata_var_exploded.rename(columns={"seq_ID": "transcript_ID"}, inplace=True)

    #* merge transcript_df into adata_var_exploded
    if not (variant_position_annotations == "cdna" and strand_bias_end == "5p"):
        adata_var_exploded = adata_var_exploded.merge(
            transcript_df[["transcript_ID", "transcript_length", "five_prime_utr_length", "three_prime_utr_length", "utr_length_preceding_transcript", "strand"]],
            on="transcript_ID",
            how="left"
        ).set_index(adata_var_exploded.index)
    
    #* change CDS --> cDNA by adding the preceding UTR lengths
    if variant_position_annotations == "cdna":
        pass  # RNA-seq captures cDNA (including the UTRs), so positions are correct relative to what the sequencer sees
    elif variant_position_annotations == "cds":  # add the UTR lengths to the start and end positions for genome variants - 5' UTR for + strand, 3' UTR for - strand
        adata_var_exploded["start_variant_position"] += adata_var_exploded["utr_length_preceding_transcript"]
        adata_var_exploded["end_variant_position"] += adata_var_exploded["utr_length_preceding_transcript"]

    #* Plot strand bias before filtering out
    if plot_histogram:
        adata_var_with_alignment = adata_var_exploded.loc[adata_var_exploded["vcrs_count"] > 0].copy() if "vcrs_count" in adata_var_exploded.columns else adata_var_exploded.copy()
        plot_cdna_locations(adata_var_with_alignment, cdna_sequence_length_column="transcript_length", seq_id_column="transcript_ID", start_variant_position_cdna_column="start_variant_position", end_variant_position_cdna_column="end_variant_position", sequence_side=strand_bias_end, log_x=True, log_y=True, read_length_cutoff=read_length, save_path = os.path.join(out, f"strand_bias_{strand_bias_end}_prefiltering.png"))
        del adata_var_with_alignment

    #* Filter based on strand bias
    if strand_bias_end == "5p":  #* 5': mutation start is less than or equal to read length
        adata_var_exploded = adata_var_exploded[adata_var_exploded["start_variant_position"] <= read_length + forgiveness]  # forgiveness allows for small discrepancies in read alignment (eg if the person had a natural deletion, or the reference genome is not 100% accurate)
    else:  #* 3': mutation end is greater than or equal to (transcript length - read length)        
        adata_var_exploded = adata_var_exploded[adata_var_exploded["end_variant_position"] >= (adata_var_exploded["transcript_length"] - read_length - forgiveness)]

    #* Collapse
    adata_var = adata_var_exploded.groupby(adata_var_exploded.index)["vcrs_header_individual"].apply(lambda x: ";".join(sorted(x))).reset_index()

    valid_indices = set(adata_var["index"])  # Get the valid column indices from df_collapsed
    cols_to_keep = [i for i in range(adata.n_vars) if str(i) in valid_indices]  # Identify columns to keep (i.e., only the valid indices)
    adata_var.drop(columns=["index"], inplace=True)  # Drop the index column

    # Subset adata
    adata = adata[:, cols_to_keep]
    adata.var.reset_index(drop=True, inplace=True)
    adata_var.rename(columns={"vcrs_header_individual": header_column}, inplace=True)
    adata.var[header_column] = adata_var[header_column].values  # will fix cases like where ENST0000001:c.50G>A;ENST0000006:c.1001G>A --> ENST0000001:c.50G>A

    return adata



def kb_extract_all_alternative(fastq_file_list, t2g_file, technology, index_file=None, kb_count_out_dir="kb_count_out", kb_extract_out_dir="kb_extract_out", gzip_output=True, mm=False, union=False, parity="single", strand=None, threads=2, overwrite=False, verbose=False, tmp=None, keep_tmp=False, aa=False, kallisto="kallisto", bustools="bustools"):
    if not overwrite and (os.path.exists(kb_extract_out_dir) and len(os.listdir(kb_extract_out_dir)) > 0):
        raise ValueError(f"Output directory '{kb_extract_out_dir}' already exists and is not empty. Set 'overwrite=True' to overwrite existing files or choose a different output directory.")
    if overwrite or not os.path.exists(kb_count_out_dir) or len(os.listdir(kb_count_out_dir)) == 0:
        kb_count_command = ["kb", "count", "-t", str(threads), "-i", index_file, "-g", t2g_file, "-x", technology, "--num", "-o", kb_count_out_dir]
        if mm:
            kb_count_command.append("--mm")
        if union:
            kb_count_command.append("--union")
        if technology.lower() in {"bulk", "smartseq2"}:
            kb_count_command += ["--parity", parity]
        if strand:
            kb_count_command += ["--strand", strand]
        if verbose:
            kb_count_command.append("--verbose")
        if tmp:
            kb_count_command += ["--tmp", tmp]
        if keep_tmp:
            kb_count_command += ["--keep-tmp"]
        if aa:
            kb_count_command.append("--aa")
        if overwrite:
            kb_count_command.append("--overwrite")
        if kallisto != "kallisto":
            kb_count_command += ["--kallisto", kallisto]
        if bustools != "bustools":
            kb_count_command += ["--bustools", bustools]
        kb_count_command += fastq_file_list
        subprocess.run(kb_count_command, check=True)
    
    fastq_file_list_pyfastx = []
    for fastq_file in fastq_file_list:
        fastq_file_list_pyfastx.append(pyfastx.Fastq(fastq_file, build_index=True))

    bus_df = make_bus_df(kb_count_out_dir, fastq_file_list, t2g_file, technology=technology, mm=mm, bustools=bustools, correct_barcodes_of_hamming_distance_one=False)
    bus_df = bus_df[bus_df["counted_in_count_matrix"]]  # to only keep reads that were counted in count matrix

    open_func = gzip.open if gzip_output else open
    mode = "wt" if gzip_output else "w"  # 'wt' for text mode in gzip

    unique_genes = sorted(set().union(*bus_df["gene_names"]))
    for gene_name in unique_genes:  # Get unique gene names
        print(f"Processing {gene_name}")
        temp_df = bus_df[bus_df["gene_names"].apply(lambda x: gene_name in x)]

        gene_dir = os.path.join(kb_extract_out_dir, gene_name)
        os.makedirs(gene_dir, exist_ok=True)
        
        aligned_reads_file = os.path.join(gene_dir, "1.fastq")
        with open_func(aligned_reads_file, mode) as f:
            for header, file_index, read_index in zip(temp_df["fastq_header"], temp_df["file_index"], temp_df["read_index"]):
                fastq_file = fastq_file_list_pyfastx[file_index]
                sequence = fastq_file[read_index].seq
                qualities = fastq_file[read_index].qual
                f.write(f"@{header}\n{sequence}\n+\n{qualities}\n")
            
            # # an old solution - imperfect because (1) it loops through each fastq each time, (2) it does lookup by header rather than index, and (3) it must convert all keys into a set to check for membership
            # for header in temp_df["fastq_header"].tolist():
            #     for fastq_file in fastq_file_list_pyfastx:
            #         if header in set(fastq_file.keys()):
            #             sequence = fastq_file[header].seq
            #             qualities = fastq_file[header].qual
            #             f.write(f"@{header}\n{sequence}\n+\n{qualities}\n")
                        


def remove_columns_from_usecols_that_are_not_in_df(df, col_list):
    if isinstance(df, pd.DataFrame):
        pass
    elif isinstance(df, str) and os.path.exists(df):  # just read in columns
        if df.endswith(".csv"):
            df = pd.read_csv(df, nrows=0)
        elif df.endswith(".tsv"):
            df = pd.read_csv(df, sep="\t", nrows=0)
        elif df.endswith(".parquet"):
            df = pd.read_parquet(df)
    
    if col_list is None:
        return None
    
    if isinstance(col_list, str):
        col_list = [col_list]
    
    if not isinstance(col_list, list):
        raise ValueError("The usecols parameter must be a string or a list of strings.")
    
    col_list_final = col_list.copy()
    for col in col_list:
        if col not in df.columns:
            logger.warning(f"Column '{col}' specified in usecols is not in the DataFrame. It will be removed from the usecols list.")
            col_list_final.remove(col)
    
    return col_list_final
        

def merge_variants_with_adata(variants, adata_var_exploded, seq_id_column, var_column, var_id_column=None, variants_usecols=None):
    # ensure I use the correct columns (default all)
    if var_id_column is None:
        var_id_column = "vcrs_header"
    if isinstance(variants_usecols, list) and var_id_column not in variants_usecols:
        variants_usecols.append(var_id_column)  # it's ok if it swiftly gets removed in the next func, becasue I don't use variants_usecols in this scope anymore

    variants_usecols = remove_columns_from_usecols_that_are_not_in_df(variants, variants_usecols)
    
    # load in variants, or filter by correct columns
    if isinstance(variants, pd.DataFrame):
        if isinstance(variants_usecols, list):
            variants = variants[variants_usecols].copy()
    elif isinstance(variants, str) and os.path.exists(variants):
        if variants.endswith(".csv"):
            variants = pd.read_csv(variants, usecols=variants_usecols)
        elif variants.endswith(".tsv"):
            variants = pd.read_csv(variants, sep="\t", usecols=variants_usecols)
        elif variants.endswith(".parquet"):
            variants = pd.read_parquet(variants, columns=variants_usecols)
        elif variants.endswith(".vcf"):
            raise ValueError("vcrs headers must be in HGVS format when variants is a VCF file.")
        else:
            raise ValueError("variants must be a DataFrame, a CSV/TSV, or a VCF.")
    else:
        raise ValueError("The variants parameter must be a DataFrame, a CSV/TSV file, or a VCF file. Please provide a valid input.")
    if 'vcrs_header' not in variants.columns:
        variants['vcrs_header'] = variants[seq_id_column].astype(str) + ":" + variants[var_column].astype(str)
    merged_var = adata_var_exploded.merge(
            variants,
            left_on='vcrs_id_individual',
            right_on=var_id_column,
            how='left',
            suffixes=('', '_variant'),
        )
    if "vcrs_header_variant" in merged_var.columns:  # this col was already in adata.var
        merged_var.drop(columns=['vcrs_header_variant'], inplace=True)
    return merged_var

def make_t2g_dict(t2g_file, strip_versions=False, column_indices=(0,1)):
    col1, col2 = column_indices
    if t2g_file is None or not os.path.isfile(t2g_file):
        raise ValueError(f"The specified t2g file '{t2g_file}' does not exist.")
    t2g_df = pd.read_csv(t2g_file, sep="\t", header=None)
    t2g_df.rename(columns={col1: "transcript_id", col2: "gene_name"}, inplace=True)
    t2g_df = t2g_df[["transcript_id", "gene_name"]].copy()  # keep only first 2 columns
    t2g_dict = dict(zip(t2g_df["transcript_id"], t2g_df["gene_name"]))
    t2g_dict["dlist"] = "dlist"

    if strip_versions:
        t2g_dict = {key.split(".")[0]: val.split(".")[0] for key, val in t2g_dict.items()}  # strip off the version number

    return t2g_dict


def make_t2g_dict_from_gtf(gtf):
    if isinstance(gtf, str):
        gtf_cols = ["chromosome", "source", "feature", "start", "end", "score", "strand", "frame", "attributes"]
        gtf_df = pd.read_csv(gtf, sep="\t", comment="#", names=gtf_cols)
    elif isinstance(gtf, pd.DataFrame):
        gtf_df = gtf.copy()
    else:
        raise ValueError("gtf must be a path to a GTF file or a pandas DataFrame")
    
    def extract_id(attr_str, key):
        match = re.search(fr'{key} "([^"]+)"', attr_str)
        return match.group(1) if match else None

    # Filter for rows that have 'transcript' as feature
    transcript_df = gtf_df[gtf_df['feature'] == 'transcript'].copy()

    # Extract transcript_id and gene_id
    transcript_df['transcript_id'] = transcript_df['attributes'].apply(lambda x: extract_id(x, 'transcript_id'))
    transcript_df['gene_id'] = transcript_df['attributes'].apply(lambda x: extract_id(x, 'gene_id'))

    # Build the dictionary
    t2g_dict = dict(zip(transcript_df['transcript_id'], transcript_df['gene_id']))
    t2g_dict["dlist"] = "dlist"
    
    return t2g_dict

def add_information_from_variant_header_to_adata_var_exploded(adata_var_exploded, vcrs_header_individual_column="vcrs_header_individual", seq_id_column="seq_ID", var_column="mutation", gene_id_column="gene_id", variant_source=None, include_position_information=True, include_gene_information=True, t2g_file=None, gtf=None):
    if seq_id_column not in adata_var_exploded.columns or var_column not in adata_var_exploded.columns:
        adata_var_exploded[[seq_id_column, var_column]] = adata_var_exploded[vcrs_header_individual_column].str.split(":", n=1, expand=True)
        adata_var_exploded[seq_id_column] = strip_gene_name_from_seq_id(adata_var_exploded[seq_id_column])  # drop any '(GENE)' annotation from the seq_ID

    adata_var_exploded[seq_id_column] = adata_var_exploded[seq_id_column].str.split(".").str[0]  # strip off the version number

    if include_position_information:
        if "nucleotide_positions" not in adata_var_exploded.columns and not "actual_variant" in adata_var_exploded.columns:
            adata_var_exploded[["nucleotide_positions", "actual_variant"]] = adata_var_exploded[var_column].str.extract(mutation_pattern)

        if "start_variant_position" not in adata_var_exploded.columns or "end_variant_position" not in adata_var_exploded.columns:
            split_positions = adata_var_exploded["nucleotide_positions"].str.split("_", expand=True)
            adata_var_exploded["start_variant_position"] = split_positions[0]
            if split_positions.shape[1] > 1:
                adata_var_exploded["end_variant_position"] = split_positions[1].fillna(split_positions[0])
            else:
                adata_var_exploded["end_variant_position"] = adata_var_exploded["start_variant_position"]
            adata_var_exploded.loc[adata_var_exploded["end_variant_position"].isna(), "end_variant_position"] = adata_var_exploded["start_variant_position"]
            adata_var_exploded[["start_variant_position", "end_variant_position"]] = adata_var_exploded[["start_variant_position", "end_variant_position"]].astype(int)
    
    if not variant_source:  # detect automatically per-variant
        identify_variant_source(adata_var_exploded, variant_column=var_column, variant_source_column="variant_source", choices = ("transcriptome", "genome"))
        unique_vcrs_sources = adata_var_exploded["variant_source"].unique()
        variant_source = "combined" if len(unique_vcrs_sources) > 1 else unique_vcrs_sources[0]
    else:
        adata_var_exploded["variant_source"] = variant_source
    
    if (variant_source != "genome" and (t2g_file is None and gtf is None)) or (variant_source != "transcriptome" and gtf is None):
        include_gene_information = False
    
    if include_gene_information and gene_id_column not in adata_var_exploded.columns:
        if variant_source != "genome":
            if t2g_file and os.path.isfile(t2g_file):  # use normal reference genome t2g to map transcript to gene
                t2g_dict = make_t2g_dict(t2g_file)
            elif gtf is not None:  # use gtf to make t2g to map transcript to gene
                t2g_dict = make_t2g_dict_from_gtf(gtf)
            else:
                raise ValueError("No t2g file or gtf file provided to map transcript to gene.")  # I don't use get_ensembl_gene_id_from_transcript_id_bulk because I would need to provide reference genome version (37 vs 38) and species - and at that point I might as well just download the gtf (and it is long)
                # logger.warning("Retrieving ensembl gene IDs from Ensembl API. This may take a while if there are many transcripts. Please provide t2g or gtf file to speed this up.")
                # t2g_dict = get_ensembl_gene_id_from_transcript_id_bulk(adata_var_exploded[seq_id_column].tolist())
            # adata_var_exploded[gene_id_column] = (adata_var_exploded[seq_id_column].map(t2g_dict).fillna(adata_var_exploded[seq_id_column]))
            
            t2g_dict = {key.split(".")[0]: val.split(".")[0] for key, val in t2g_dict.items()}  # strip off the version number

            transcriptome_mask = adata_var_exploded["variant_source"] == "transcriptome"
            adata_var_exploded.loc[transcriptome_mask, gene_id_column] = (
                adata_var_exploded.loc[transcriptome_mask, seq_id_column]
                .map(t2g_dict)
                .fillna(adata_var_exploded.loc[transcriptome_mask, seq_id_column])
            )
        if variant_source != "transcriptome":  # use gtf to determine gene for a given position on a given chromosome
            if isinstance(gtf, str):
                gtf_cols = ["chromosome", "source", "feature", "start", "end", "score", "strand", "frame", "attributes"]
                gtf_df = pd.read_csv(gtf, sep="\t", comment="#", names=gtf_cols)
            elif isinstance(gtf, pd.DataFrame):
                gtf_df = gtf.copy()
            else:
                raise ValueError("gtf must be a path to a GTF file or a pandas DataFrame")
            gtf_df["gene_ID"] = gtf_df["attributes"].str.extract(r'gene_id "([^"]+)"')
            gene_df = gtf_df[gtf_df["feature"] == "gene"].copy()
            gene_df = gene_df.drop_duplicates(subset="gene_ID", keep="first")
            
            genome_mask = adata_var_exploded["variant_source"] == "genome"
            adata_var_exploded.loc[genome_mask, gene_id_column] = adata_var_exploded.loc[genome_mask].apply(
                lambda row: assign_gene_id(
                    row[seq_id_column],
                    row['start_variant_position'],
                    gene_df
                ),
                axis=1
            )

        number_of_unmapped_genes = ((adata_var_exploded[gene_id_column] == adata_var_exploded[seq_id_column]).sum())
        if number_of_unmapped_genes > 0:
            logger.warning(f"{number_of_unmapped_genes} variants were not mapped to a gene. Please check the variant source and position annotations.")
    
    if gene_id_column in adata_var_exploded.columns:
        # adata_var_exploded['vcrs_header_with_gene_name'] = ("(" + adata_var_exploded[gene_id_column].astype(str) + ")" + adata_var_exploded[vcrs_header_individual_column].astype(str))
        header_parts = adata_var_exploded[vcrs_header_individual_column].str.split(":", n=1, expand=True)
        # strip any '(GENE)' annotation build may have added, so we don't double-annotate the seq_ID
        header_seq_id = strip_gene_name_from_seq_id(header_parts[0])
        header_mutation = header_parts[1] if header_parts.shape[1] > 1 else ""
        adata_var_exploded['vcrs_header_with_gene_name'] = (
            adata_var_exploded[gene_id_column].astype(str) + "(" +
            header_seq_id + "):" +
            header_mutation
        )

    return adata_var_exploded


def identify_variant_source(mutation_metadata_df_exploded, variant_column="variant_used_for_vcrs", variant_source_column="variant_source", choices=("cdna", "genome")):
    conditions = [
        mutation_metadata_df_exploded[variant_column].str.startswith("c.", na=False),
        mutation_metadata_df_exploded[variant_column].str.startswith("g.", na=False),
    ]  # if it finds ":c.", make it "cdna"; if it finds ":g.", make it "genome"; if it finds both or neither, make it "unknown"

    mutation_metadata_df_exploded[variant_source_column] = np.select(conditions, choices, default="unknown")



def prepare_set(vcrs_id_set_to_exclusively_keep):
    if vcrs_id_set_to_exclusively_keep is None:
        return None  # Keep None as None

    elif isinstance(vcrs_id_set_to_exclusively_keep, (list, tuple)):
        return set(vcrs_id_set_to_exclusively_keep)  # Convert list/tuple to set

    elif isinstance(vcrs_id_set_to_exclusively_keep, str) and os.path.isfile(vcrs_id_set_to_exclusively_keep) and vcrs_id_set_to_exclusively_keep.endswith(".txt"):
        # Load lines from text file, stripping whitespace
        with open(vcrs_id_set_to_exclusively_keep, "r", encoding="utf-8") as f:
            return set(line.strip() for line in f if line.strip())  # Ignore empty lines

    elif isinstance(vcrs_id_set_to_exclusively_keep, set):
        return vcrs_id_set_to_exclusively_keep  # Already a set, return as is

    else:
        raise ValueError("Invalid input: must be None, a list, tuple, set, or a path to a text file")


scanpy_conditions = ["filter_cells_by_min_counts", "filter_cells_by_min_genes", "filter_genes_by_min_cells", "filter_cells_by_max_mt_content", "doublet_detection"]


def _validate_clean_params(params_dict):
    # check if adata_vcrs is of type Anndata
    adata_vcrs = params_dict["adata_vcrs"]
    if not isinstance(adata_vcrs, (str, Path, anndata.AnnData)):  # I will enforce that adata_vcrs exists later, as otherwise it will throw an error when I call this through vk count before kb count can run
        raise ValueError("adata_vcrs must be an AnnData object or a file path to an h5ad object.")
    if isinstance(adata_vcrs, (str, Path)):
        check_file_path_is_string_with_valid_extension(adata_vcrs, "adata_vcrs", "h5ad")

    # technology
    technology = params_dict.get("technology", None)
    technology_valid_values_lower = {x.lower() for x in technology_valid_values}
    if technology is None or technology.lower() not in technology_valid_values_lower:
        raise ValueError(f"Technology must be one of {technology_valid_values_lower}")

    # min_counts may be fractional, so it is checked as a number rather than an int
    min_counts = params_dict.get("min_counts")
    try:
        if isinstance(min_counts, bool):
            raise TypeError
        min_counts_as_float = float(min_counts)
    except (TypeError, ValueError):
        min_counts_as_float = None
    if min_counts_as_float is None or min_counts_as_float <= 0:
        raise ValueError(f"min_counts must be a number > 0. Got {min_counts} of type {type(min_counts)}.")

    for param_name, min_value, optional_value in [
        ("read_length", 1, True),
        ("filter_cells_by_min_genes", 0, True),  # optional True means that it can be None
        ("filter_genes_by_min_cells", 0, True),
    ]:
        param_value = params_dict.get(param_name)
        if not is_valid_int(param_value, ">=", min_value, optional=optional_value):
            must_be_value = f"an integer >= {min_value}" if not optional_value else f"an integer >= {min_value} or None"
            raise ValueError(f"{param_name} must be {must_be_value}. Got {param_value} of type {type(param_value)}.")

    if params_dict.get("cpm_normalization") and (not params_dict.get("adata_reference_genome") and not params_dict.get("kallisto_quant_reference_genome_dir")):
        raise ValueError("adata_reference_genome or kallisto_quant_reference_genome_dir must be provided if cpm_normalization=True.")

    if technology not in non_single_cell_technologies:
        if params_dict.get("filter_cells_by_min_counts") is True:
            try:
                from kneed import KneeLocator
            except ImportError:
                raise ImportError("kneed is required for filter_cells_by_min_counts=True. See pyproject.toml project.optional-dependencies for version recommendation. Install it with:\n" "  pip install kneed")
        for condition in scanpy_conditions:
            if params_dict.get(condition):
                if not params_dict.get("adata_reference_genome") and not params_dict.get("kallisto_quant_reference_genome_dir"):
                    raise ValueError(f"adata_reference_genome or kallisto_quant_reference_genome_dir must be provided if {condition}=True.")
                try:
                    import scanpy as sc
                except ImportError:
                    raise ImportError("Scanpy is required for this function. See pyproject.toml project.optional-dependencies for version recommendation. Install it with:\n" "  pip install scanpy")

    # filter_cells_by_min_counts - gets special treatment because it can also be True for automatic calculation
    filter_cells_by_min_counts = params_dict.get("filter_cells_by_min_counts", None)
    if filter_cells_by_min_counts is not None and not is_valid_int(filter_cells_by_min_counts, ">=", 1):
        raise ValueError(f"filter_cells_by_min_counts must be an integer >= 1 (for manual threshold), True (for automatic threshold calculation with kneed's KneeLocator), or None (for no threshold). Got {filter_cells_by_min_counts} of type {type(filter_cells_by_min_counts)}.")

    # filter_cells_by_max_mt_content - special treatment because it is between rather than lower-bounded only
    if not is_valid_int(params_dict.get("filter_cells_by_max_mt_content"), "between", min_value_inclusive=0, max_value_inclusive=100, optional=True):
        raise ValueError(f"filter_cells_by_max_mt_content must be an integer between 0 and 100, or None. Got {params_dict.get('filter_cells_by_max_mt_content')}.")

    # boolean
    for param_name in ["use_binary_matrix", "drop_empty_columns", "pseudobam_validation", "doublet_detection", "remove_doublets", "cpm_normalization", "sum_rows", "drop_multi_variant_vcrs", "rename_vcrs_to_variant", "mm", "save_vcf", "dry_run", "overwrite", "account_for_strand_bias"]:
        if not isinstance(params_dict.get(param_name), bool):
            raise ValueError(f"{param_name} must be a boolean. Got {param_name} of type {type(params_dict.get(param_name))}.")
    if not isinstance(params_dict.get("multiplexed"), bool) and params_dict.get("multiplexed") is not None:
        raise ValueError(f"multiplexed must be a boolean or None. Got {params_dict.get('multiplexed')} of type {type(params_dict.get('multiplexed'))}.")

    # sets
    for param_name in ["vcrs_id_set_to_exclusively_keep", "vcrs_id_set_to_exclude", "gene_set_to_exclusively_keep", "gene_set_to_exclude"]:
        param_value = params_dict.get(param_name, None)
        if param_value is not None and not isinstance(param_value, (set, list, tuple) and not (isinstance(param_value, str) and param_value.endswith(".txt") and os.path.isfile(param_value))):  # checks if it is (1) None, (2) a set/list/tuple, or (3) a string path to a txt file that exists
            raise ValueError(f"{param_name} must be a set. Got {param_name} of type {type(param_value)}.")

    parity_valid_values = {"single", "paired"}
    if params_dict["parity"] not in parity_valid_values:
        raise ValueError(f"Parity must be one of {parity_valid_values}")
    
    strand_bias_end_valid_values = {"5p", "3p", None}
    if params_dict["strand_bias_end"] not in strand_bias_end_valid_values:
        raise ValueError(f"strand_bias_end must be one of {strand_bias_end_valid_values}")
    
    if params_dict.get("account_for_strand_bias") and not params_dict.get("strand_bias_end"):
        raise ValueError("strand_bias_end must be provided if account_for_strand_bias=True.")

    # $ type checking of the directory and text file performed earlier by load_in_fastqs
    fastqs = params_dict["fastqs"]  # tuple
    if fastqs:
        for fastq in fastqs:
            check_file_path_is_string_with_valid_extension(fastq, variable_name=fastq, file_type="fastq")  # ensure that all fastq files have valid extension
            if not os.path.isfile(fastq):  # ensure that all fastq files exist
                raise ValueError(f"File {fastq} does not exist")

    # file paths
    for param_name, file_type in {
        "config": ["json", "yaml"],
        "vcrs_index": "index",
        "vcrs_t2g": "t2g",
        "adata_reference_genome": "h5ad",
        "adata_vcrs_clean_out": "h5ad",
        "adata_reference_genome_clean_out": "h5ad",
        "vcf_out": "vcf",
    }.items():
        check_file_path_is_string_with_valid_extension(params_dict.get(param_name), param_name, file_type)

    # directories
    for param_name in ["vk_ref_dir", "kb_count_vcrs_dir", "kallisto_quant_reference_genome_dir"]:
        if not isinstance(params_dict.get(param_name), (str, Path)) and params_dict.get(param_name) is not None:
            raise ValueError(f"Directory {param_name} {params_dict.get(param_name)} is not a string or None")
        if params_dict.get(param_name) and not params_dict.get("dry_run") and (not os.path.isdir(params_dict.get(param_name)) or len(os.listdir(params_dict.get(param_name))) == 0):  # including the dry_run condition so that vk count dry run does not throw an error
            raise ValueError(f"Directory {params_dict.get(param_name)} does not exist")
    if not isinstance(params_dict.get("out"), (str, Path)):
        raise ValueError(f"Out directory {params_dict.get('out')} is not a string")

    pseudobam_validation_aligner = params_dict.get("pseudobam_validation_aligner", "pseudo")
    pseudobam_validation_aligner_valid_values = {"pseudo", "true", "star", "bowtie2"}
    if not isinstance(pseudobam_validation_aligner, str) or pseudobam_validation_aligner.lower() not in pseudobam_validation_aligner_valid_values:
        raise ValueError(f"pseudobam_validation_aligner must be one of 'pseudo', 'true', 'STAR', 'bowtie2'. Got {pseudobam_validation_aligner}.")

    reference_bam = params_dict.get("reference_bam")
    if reference_bam is not None:
        reference_bam_list = [reference_bam] if isinstance(reference_bam, (str, Path)) else reference_bam
        if not isinstance(reference_bam_list, (list, tuple)):
            raise ValueError(f"reference_bam must be a path to a BAM file, a list of such paths, or None. Got {type(reference_bam)}.")
        for bam in reference_bam_list:
            check_file_path_is_string_with_valid_extension(bam, "reference_bam", "bam")
            if not params_dict.get("dry_run") and not os.path.isfile(bam):
                raise ValueError(f"reference_bam file {bam} does not exist")

    if params_dict.get("pseudobam_validation"):
        kb_count_vcrs_dir = params_dict.get("kb_count_vcrs_dir")
        if kb_count_vcrs_dir and os.path.exists(kb_count_vcrs_dir):
            run_info_json = os.path.join(kb_count_vcrs_dir, "run_info.json")
            if os.path.exists(run_info_json):
                with open(run_info_json, "r") as f:
                    data = json.load(f)
                if "--num" not in data["call"]:
                    raise ValueError("--num must be included in the kb count run used to produce kb_count_vcrs_dir when pseudobam_validation=True. Please re-run kb count (vk count adds --num automatically), or point kb_count_vcrs_dir to such a run.")
