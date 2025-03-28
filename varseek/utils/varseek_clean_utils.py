import os
import re
import subprocess
from pathlib import Path

import anndata as ad
import numpy as np
import sys
import pandas as pd
import pyfastx
import pysam
from scipy.sparse import csr_matrix, issparse
import json
import logging
from tqdm import tqdm

from varseek.constants import (
    complement,
    fastq_extensions,
    technology_barcode_and_umi_dict,
    mutation_pattern,
    technology_to_file_index_with_transcripts_mapping
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
    parquet_column_tuple_to_list
)
from varseek.utils.logger_utils import set_up_logger, count_chunks, determine_write_mode
from varseek.utils.varseek_info_utils import identify_variant_source

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

def make_bus_df(kb_count_out, fastq_file_list, t2g_file=None, mm=False, technology="bulk", parity="single", bustools="bustools", check_only=False, chunksize=None, bad_to_good_barcode_dict=None, save_type="parquet"):  # make sure this is in the same order as passed into kb count - [sample1, sample2, etc] OR [sample1_pair1, sample1_pair2, sample2_pair1, sample2_pair2, etc]
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
    
    #* transcripts
    print("loading in transcripts")
    with open(f"{kb_count_out}/transcripts.txt", encoding="utf-8") as f:
        transcripts = f.read().splitlines()  # get transcript at index 0 with transcript[0], and index of transcript named "name" with transcript.index("name")
    transcripts.append("dlist")  # add dlist to the end of the list

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
        t2g_dict["dlist"] = "dlist"

    technology = technology.lower()
    if technology != "bulk":
        parity = "single"
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
            if vcrs_parity == "paired" and i % 2 == 1:  #!!! technology_to_file_index_with_transcripts_mapping
                continue  # if vcrs_parity is paired, then only include the read headers from file0
            fastq_file = str(fastq_file)  # important for temp files
            fastq_header_list = [header.strip() for header, _, _ in tqdm(pyfastx.Fastx(fastq_file), desc="Processing FASTQ headers")]
            barcode_list = barcodes[i]

            new_rows = pd.DataFrame({"read_index": range(len(fastq_header_list)), "fastq_header": fastq_header_list, "barcode": barcode_list})
            fastq_header_df = pd.concat([fastq_header_df, new_rows], ignore_index=True)

        #* bus
        bus_file = f"{kb_count_out}/output.bus"
        bus_text_file = f"{kb_count_out}/output_sorted_bus.txt"
        if not os.path.exists(bus_text_file):
            print("running bustools text")
            create_bus_txt_file_command = [bustools, "text", "-o", bus_text_file, "-f", bus_file]  # bustools text -p -a -f -d output.bus
            subprocess.run(create_bus_txt_file_command, check=True)
        print("loading in bus df")
        if chunksize:
            total_chunks = count_chunks(bus_text_file, chunksize)
        else:
            chunksize = sys.maxsize  # ensures 1 chunk
            total_chunks = 1
        
        bus_out_path = f"{kb_count_out}/bus_df.{save_type}"
        for i, bus_df in enumerate(pd.read_csv(bus_text_file, sep=r"\s+", header=None, names=["barcode", "UMI", "EC", "count", "read_index"], usecols=["barcode", "UMI", "EC", "read_index"], chunksize=chunksize)):
            if total_chunks > 1:
                print(f"Processing chunk {i+1}/{total_chunks}")

            print("Merging fastq header df and ec_df into bus df")
            bus_df = bus_df.merge(fastq_header_df, on=["read_index", "barcode"], how="left")
            bus_df = bus_df.merge(ec_df, on="EC", how="left")

            if parity == "paired" and vcrs_parity == "single":
                if not bad_to_good_barcode_dict:
                    bad_to_good_barcode_dict = make_good_barcodes_and_file_index_tuples(barcodes, include_file_index=True)
                
                bus_df[['corrected_barcode', 'file_index']] = bus_df['barcode'].map(bad_to_good_barcode_dict).apply(pd.Series)
                bus_df.drop(columns=["barcode"], inplace=True)
                bus_df.rename(columns={"corrected_barcode": "barcode"}, inplace=True)

                # # problematic because (1) it erases the fact that the 2nd read really did have an alignment and (2) if misses cases where read1_1 maps to VCRS1 (EC1), and read1_2 maps to VCRS1 and VCRS2 (EC2) - VCRS1 will be double-counted, but this wouldn't catch those cases
                # dup_mask = bus_df.duplicated(subset=['barcode', 'UMI', 'EC', 'read_index'], keep=False)  # Identify all rows that have duplicates (including first occurrences)
                # bus_df.loc[dup_mask, 'file_index'] = 'both'  # Set 'file_index' to 'both' for all duplicated rows
                # bus_df = bus_df.drop_duplicates(subset=['barcode', 'UMI', 'EC', 'read_index'], keep='first')  # Drop duplicate rows, keeping only the first occurrence
            else:
                bus_df["file_index"] = "0"
            bus_df["file_index"] = bus_df["file_index"].astype("category")
            if technology == "bulk":
                bus_df["barcode"] = bus_df["barcode"].astype("category")

            if t2g_file is not None:
                print("Apply the mapping function to create gene name columns")
                bus_df["gene_names"] = bus_df["transcript_names"].progress_apply(lambda x: map_transcripts_to_genes(x, t2g_dict))
                print("Taking set of gene_names")
                bus_df["gene_names"] = bus_df["gene_names"].progress_apply(lambda x: sorted(tuple(set(x))))
            else:
                bus_df["gene_names"] = bus_df["transcript_names"]

            print("Determining what counts in count matrix")
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
            
            print(f"Saving bus df to {bus_out_path}") if total_chunks == 1 else print(f"Saving chunk {i+1}/{total_chunks} of bus df to {bus_out_path}")
            first_chunk = (i == 0)
            if save_type == "parquet":  # parquet benefits over csv: much smaller file size (~10% of csv size), and saves data types upon saving and loading
                parquet_column_tuple_to_list(bus_df)  # parquet doesn't like tuples - it only likes lists - if wanting to load back in the data as a tuple later, then use parquet_column_list_to_tuple
                bus_df.to_parquet(bus_out_path, index=False, append=not first_chunk)
            elif save_type == "csv":  # csv benefits over parquet: human-readable, can be read/iterated in chunks, and supports tuples (which take about 3/4 the RAM of strings/lists)
                bus_df.csv(bus_out_path, index=False, header=first_chunk, mode=determine_write_mode(bus_out_path, overwrite=True, first_chunk=first_chunk))
        
        if total_chunks > 1:
            print("Returning the last chunk of bus_df")
        return bus_df
        
    
    
    else:  #!!! WRITE FOR SINGLE-CELL
        for i, fastq_file in enumerate(fastq_file_list):
            pass

        #* barcodes
        # Load whitelist into a set
        with open(f"{kb_count_out}/counts_unfiltered/cells_x_genes.barcodes.txt", encoding="utf-8") as f:  # use f"{kb_count_out}/10x_version3_whitelist.txt" for full list of valid barcodes
            whitelist = set(line.strip() for line in f)
        
        # correct for bad barcodes
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


#!!! add back here


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
        adata_filtered = adata[:, adata.var['vcrs_header'].isin(set(filtered_VCRSs))].copy()  # Subset adata to keep only the variables in filtered_ids
        if adata_var['ID'].tolist() != adata_filtered.var['vcrs_header'].tolist():  # different orders
            correct_order = adata_filtered.var.set_index('vcrs_header').loc[adata_var['ID']].index  # Get the correct order of indices based on adata_var['ID']
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
        chroms, poss, ids, refs, alts, dps, nss, afs = (
            adata_var["CHROM"].values, adata_var["POS"].values, adata_var["ID"].values,
            adata_var["REF"].values, adata_var["ALT"].values, adata_var["AO"].values,
            adata_var["NS"].values,
            adata_var["AF"].values if "AF" in adata_var else np.full(len(adata_var), np.nan)  # Handle optional AF column
        )

        # Iterate over pre-extracted values
        buffer = []
        for idx, (chrom, pos, id_, ref, alt, dp, ns, af) in enumerate(zip(chroms, poss, ids, refs, alts, dps, nss, afs)):
            # Construct INFO field efficiently
            info_fields = [f"AO={int(dp)}" if pd.notna(dp) else None,
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
    if any(col not in adata_var.columns for col in ["vcrs_header", "vcrs_count", "number_obs"]):
        raise ValueError("adata_var must contain columns vcrs_header, vcrs_count, number_obs")
    if save_vcf_samples and not isinstance(adata, ad.AnnData):
        raise ValueError("adata must be provided as an anndata object or path to an anndata object if save_vcf_samples is True")
    
    output_vcf = str(output_vcf)  # for Path
    
    # only keep the VCRSs that have a count > 0, and only keep relevant columns
    adata_var_temp = adata_var[["vcrs_header", "vcrs_count", "number_obs"]].loc[adata_var["vcrs_count"] > 0].copy()

    # make copy column that won't be exploded so that I know how to groupby later
    adata_var_temp["vcrs_header_copy"] = adata_var_temp["vcrs_header"]

    # rename to have VCF-like column names
    adata_var_temp.rename(columns={"vcrs_count": "AO", "number_obs": "NS", "vcrs_header": "ID"}, inplace=True)

    # explode across semicolons so that I can merge in vcf_data_df
    adata_var_temp = adata_var_temp.assign(
        ID=adata_var_temp["ID"].str.split(";")
    ).explode("ID").reset_index(drop=True)

    # merge in vcf_data_df (eg cosmic_df)
    adata_var_temp = adata_var_temp.merge(vcf_data_df, on="ID", how="left")

    # collapse across semicolons so that I get my VCRSs back
    adata_var_temp = (
        adata_var_temp
        .groupby("vcrs_header_copy", sort=False)  # Group by vcrs_header_copy while preserving order
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
        .drop(columns=["vcrs_header_copy"])
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



def assign_transcript_id(row, transcript_df):
    seq_id = row["chromosome"]
    position = row["variant_start_genome_position"]
    source = row["variant_source"]

    if source == "genome":
        # Query transcript_df for transcript mapping
        match = transcript_df[
            (transcript_df["chromosome"] == seq_id) & 
            (transcript_df["start"] <= int(position)) & 
            (transcript_df["end"] >= int(position))
        ]
        transcript_id = match["transcript_ID"] if not match.empty else "unknown"
    else:
        transcript_id = seq_id  # If not genome, use seq_ID directly

    return transcript_id

    # use it with adata_var_exploded.apply(lambda row: assign_transcript_id(row, transcript_df), axis=1)


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


def remove_variants_from_adata_for_stranded_technologies(adata, strand_bias_end, read_length, header_column="vcrs_header", variant_source=None, gtf=None):
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
    
    if variant_source not in {None, "transcriptome", "genome"}:
        raise ValueError("variant_source must be either None, 'transcriptome', or 'genome'")
    
    #* Load in gtf df if needed
    if variant_source == "genome" or strand_bias_end == "3p":
        if gtf is None:
            raise ValueError("gtf must be provided if variant_source is 'genome' or strand_bias_end is '3p'")
        if isinstance(gtf, str):
            gtf_cols = ["chromosome", "source", "feature", "start", "end", "score", "strand", "frame", "attributes"]
            gtf_df = pd.read_csv(gtf, sep="\t", comment="#", names=gtf_cols)
            transcript_df = gtf_df[gtf_df["feature"] == "transcript"].copy()
            transcript_df["transcript_ID"] = transcript_df["attributes"].str.extract(r'transcript_id "([^"]+)"')
        elif isinstance(gtf, pd.DataFrame):
            transcript_df = gtf
        else:
            raise ValueError("gtf must be a path to a GTF file or a pandas DataFrame")
        transcript_df = transcript_df.drop_duplicates(subset="transcript_ID", keep="first")

    #* Explode adata.var
    adata_var = adata.var.copy()
    adata_var_exploded = adata_var.assign(vcrs_header=adata_var["vcrs_header"].str.split(";")).explode("vcrs_header")

    #* Split variant column into seq_ID and variant
    adata_var_exploded[["seq_ID", "variant"]] = adata_var_exploded[header_column].str.split(":", expand=True)

    #* Split variant into nucleotide positions and actual variant
    adata_var_exploded[["nucleotide_positions", "actual_variant"]] = adata_var_exploded["variant"].str.extract(mutation_pattern)

    #* Classify variant source
    if not variant_source:  # detect automatically per-variant
        identify_variant_source(adata_var_exploded, variant_column="variant", variant_source_column="variant_source", choices = ("transcriptome", "genome"))
    
    unique_variant_sources = adata_var_exploded["variant_source"].unique()
    if len(unique_variant_sources) == 1:
        variant_source = unique_variant_sources[0]
    if variant_source != "transcriptome" and gtf is None:
        raise ValueError("gtf must be provided if genome-derived variants are present in adata.var")

    #* Find transcript start and end positions
    split_positions = adata_var_exploded["nucleotide_positions"].str.split("_", expand=True)
    adata_var_exploded["start_variant_position"] = split_positions[0]
    if split_positions.shape[1] > 1:
        adata_var_exploded["end_variant_position"] = split_positions[1].fillna(split_positions[0])
    else:
        adata_var_exploded["end_variant_position"] = adata_var_exploded["start_variant_position"]

    adata_var_exploded.loc[adata_var_exploded["end_variant_position"].isna(), "end_variant_position"] = adata_var_exploded["start_variant_position"]
    adata_var_exploded[["start_variant_position", "end_variant_position"]] = adata_var_exploded[["start_variant_position", "end_variant_position"]].astype(int)
    
    #* Assign transcript ID for genome variants
    if variant_source != "transcriptome":
        adata_var_exploded.rename(columns={"seq_ID": "chromosome", "start_variant_position": "start_variant_position_genome", "end_variant_position": "end_variant_position_genome"}, inplace=True)
        # assign_transcript_ids(adata_var_exploded, transcript_df)  # faster (works with merge instead of apply) but RAM-intensive (left-merges GTF into adata_var_exploded by chromosome alone)
        adata_var_exploded["transcript_ID"] = adata_var_exploded.apply(lambda row: assign_transcript_id(row, transcript_df), axis=1)
    else:
        adata_var_exploded.rename(columns={"seq_ID": "transcript_ID"}, inplace=True)

    #* Filter based on strand bias
    if strand_bias_end == "5p":  #* 5': mutation start is less than or equal to read length
        adata_var_exploded = adata_var_exploded[adata_var_exploded["start_variant_position"] <= read_length]
    else:  #* 3': mutation end is greater than or equal to (transcript length - read length)
        adata_var_exploded = adata_var_exploded.merge(
            transcript_df[["transcript_ID", "start", "end"]],
            on="transcript_ID",
            how="left"
        ).rename(columns={"start": "start_transcript_position_genome", "end": "end_transcript_position_genome"}).set_index(adata_var_exploded.index)

        adata_var_exploded["transcript_length"] = adata_var_exploded["end_transcript_position_genome"] - adata_var_exploded["start_transcript_position_genome"] + 1
        
        adata_var_exploded = adata_var_exploded[adata_var_exploded["end_variant_position"] >= (adata_var_exploded["transcript_length"] - read_length)]

    #* Collapse
    adata_var = adata_var_exploded.groupby(adata_var_exploded.index)["vcrs_header"].apply(lambda x: ";".join(sorted(x))).reset_index()

    valid_indices = set(adata_var["index"])  # Get the valid column indices from df_collapsed
    cols_to_keep = [i for i in range(adata.n_vars) if str(i) in valid_indices]  # Identify columns to keep (i.e., only the valid indices)
    adata_var.drop(columns=["index"], inplace=True)  # Drop the index column

    # Subset adata
    adata = adata[:, cols_to_keep]
    adata.var.reset_index(drop=True, inplace=True)
    adata.var["vcrs_header"] = adata_var["vcrs_header"].values  # will fix cases like where ENST0000001:c.50G>A;ENST0000006:c.1001G>A --> ENST0000001:c.50G>A

    return adata



def kb_extract_all_alternative(fastq_file_list, kb_count_out_dir, t2g_file, technology, kb_extract_out_dir="kb_extract_out", mm=False, bustools="bustools"):
    fastq_file_list_pyfastx = []
    for fastq_file in fastq_file_list:
        fastq_file_list_pyfastx.append(pyfastx.Fastq(fastq_file, build_index=True))

    bus_df = make_bus_df(kb_count_out_dir, fastq_file_list, t2g_file, technology=technology, mm=mm, bustools=bustools)
    bus_df = bus_df[bus_df["counted_in_count_matrix"]]  # to only keep reads that were counted in count matrix
    bus_df["gene_names_str"] = bus_df["gene_names"].apply(lambda x: x[0])  # cast to string

    for gene_name in bus_df["gene_names_str"].unique():  # Get unique gene names
        print(f"Processing {gene_name}")
        temp_df = bus_df[bus_df["gene_names_str"] == gene_name]  # Filter
        fastq_headers = temp_df["fastq_header"].tolist()  # Get values as a list

        gene_dir = os.path.join(kb_extract_out_dir, gene_name)
        os.makedirs(gene_dir, exist_ok=True)
        
        aligned_reads_file = os.path.join(gene_dir, "1.fastq")
        with open(aligned_reads_file, "w") as f:
            for header in fastq_headers:
                for fastq_file in fastq_file_list_pyfastx:
                    if header in fastq_file.index:
                        sequence = fastq_file[header].seq
                        qualities = fastq_file[header].qual
                        f.write(f"@{header}\n{sequence}\n+\n{qualities}\n")
                        break