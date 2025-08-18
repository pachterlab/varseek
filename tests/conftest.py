import os
import random
from scipy.sparse import issparse
import tempfile
import anndata as ad
import json
from pathlib import Path
from io import StringIO

import numpy as np
import pandas as pd
import sys
import pytest

import varseek as vk
from varseek.utils import (
    create_header_to_sequence_ordered_dict_from_fasta_WITHOUT_semicolon_splitting,
    create_identity_t2g,
    make_mapping_dict,
    reverse_complement,
    add_mutation_information,
    add_variant_type
)

def pytest_ignore_collect(path, config):  # skip test_bustools.py on Mac due to kb python issues
    if sys.platform == "darwin" and "test_bustools.py" in str(path):
        return True


def compare_two_dataframes_without_regard_for_order_of_rows_or_columns(df1_path, df2_path, columns_to_drop=None, head=False, only_check_intersection_of_columns=False):
    nrows = 5 if head else None
    df1 = df1_path if isinstance(df1_path, pd.DataFrame) else pd.read_csv(df1_path, nrows=nrows)
    df2 = df2_path if isinstance(df2_path, pd.DataFrame) else pd.read_csv(df2_path, nrows=nrows)
    if head:
        df1 = df1.head()
        df2 = df2.head()
        
    if columns_to_drop:
        if isinstance(columns_to_drop, str):
            columns_to_drop = [columns_to_drop]
        
        df1 = df1.drop(columns=columns_to_drop, errors='ignore')
        df2 = df2.drop(columns=columns_to_drop, errors='ignore')

    if only_check_intersection_of_columns:
        df1_cols = set(df1.columns)
        df2_cols = set(df2.columns)

        only_in_df1 = df1_cols - df2_cols
        only_in_df2 = df2_cols - df1_cols

        print("Columns only in df1:", only_in_df1)
        print("Columns only in df2:", only_in_df2)

        # Get intersection of columns
        common_cols = df1_cols & df2_cols

        if len(common_cols) == 0:
            raise ValueError("No common columns to compare.")

        # Create new DataFrames with only common columns
        df1 = df1[list(common_cols)]
        df2 = df2[list(common_cols)]
    
    # # Sort by all columns and reset index to ignore both row and column order  # commented out Feb 2025
    # df1 = df1.sort_values(by=list(df1.columns)).reset_index(drop=True)
    # df2 = df2.sort_values(by=list(df2.columns)).reset_index(drop=True)

    pd.testing.assert_frame_equal(df1.fillna('NULL_FOR_TESTING'), df2.fillna('NULL_FOR_TESTING'), check_like=True, check_exact=False)  # check_dtype=False

def compare_two_fastas_without_regard_for_order_of_entries(fasta1, fasta2):
    fasta1_dict = create_header_to_sequence_ordered_dict_from_fasta_WITHOUT_semicolon_splitting(fasta1)
    fasta2_dict = create_header_to_sequence_ordered_dict_from_fasta_WITHOUT_semicolon_splitting(fasta2)
    assert dict(fasta1_dict) == dict(fasta2_dict)

def compare_two_t2gs(t2g_path, t2g_path_ground_truth):
    t2g = pd.read_csv(t2g_path, sep="\t", header=None, names=["transcript_id", "gene_id"])
    t2g_ground_truth = pd.read_csv(t2g_path_ground_truth, sep="\t", header=None, names=["transcript_id", "gene_id"])
    t2g_dict = t2g.set_index("transcript_id")["gene_id"].to_dict()
    t2g_ground_truth_dict = t2g_ground_truth.set_index("transcript_id")["gene_id"].to_dict()
    assert t2g_dict == t2g_ground_truth_dict

def compare_two_id_to_header_mappings(id_to_header_csv, id_to_header_csv_ground_truth):
    id_to_header_dict = make_mapping_dict(id_to_header_csv, dict_key="id")
    id_to_header_dict_ground_truth = make_mapping_dict(id_to_header_csv_ground_truth, dict_key="id")
    assert id_to_header_dict == id_to_header_dict_ground_truth

def compare_two_fastqs(fastq1, fastq2):
    with open(fastq1, 'r') as f1, open(fastq2, 'r') as f2:
        lines1 = f1.readlines()
        lines2 = f2.readlines()

    assert len(lines1) == len(lines2), "Fastq files differ in number of lines."
    
    for line1, line2 in zip(lines1, lines2):
        assert line1.strip() == line2.strip(), f"Fastq files differ at line: {line1.strip()} vs {line2.strip()}"

def compare_two_anndata_objects(adata1, adata2, check_obsm=False, check_varm=False, check_layers=False):
    adata1 = ad.read_h5ad(adata1) if isinstance(adata1, str) else adata1
    adata2 = ad.read_h5ad(adata2) if isinstance(adata2, str) else adata2
    assert np.array_equal(adata1.X.toarray() if issparse(adata1.X) else adata1.X, adata2.X.toarray() if issparse(adata2.X) else adata2.X)
    assert adata1.obs.equals(adata2.obs)
    assert adata1.var.equals(adata2.var) 
    assert adata1.uns == adata2.uns 
    if check_obsm:
        assert (adata1.obsm.keys() == adata2.obsm.keys() and all(np.array_equal(adata1.obsm[k], adata2.obsm[k]) for k in adata1.obsm.keys()))
    if check_varm:
        assert (adata1.varm.keys() == adata2.varm.keys() and all(np.array_equal(adata1.varm[k], adata2.varm[k]) for k in adata1.varm.keys()))
    if check_layers:
        assert (adata1.layers.keys() == adata2.layers.keys() and all(np.array_equal(adata1.layers[key], adata2.layers[key]) for key in adata1.layers.keys()))


def compare_two_vcfs(vcf1, vcf2):
    import pysam

    # Open VCF files
    vcf1_file = pysam.VariantFile(vcf1)
    vcf2_file = pysam.VariantFile(vcf2)

    # Compare headers
    vcf1_headers = str(vcf1_file.header)
    vcf2_headers = str(vcf2_file.header)
    
    assert vcf1_headers == vcf2_headers, "Headers differ between VCF files."

    # Extract and sort variant records
    v1_records = sorted((rec.chrom, rec.pos, rec.ref, tuple(rec.alts), rec.qual, rec.filter.keys(), rec.info.items()) 
                        for rec in vcf1_file)
    
    v2_records = sorted((rec.chrom, rec.pos, rec.ref, tuple(rec.alts), rec.qual, rec.filter.keys(), rec.info.items()) 
                        for rec in vcf2_file)

    # Compare number of records
    assert len(v1_records) == len(v2_records), f"Number of records differ: {len(v1_records)} vs {len(v2_records)}"

    # Compare variants record by record
    for i, (rec1, rec2) in enumerate(zip(v1_records, v2_records)):
        assert rec1 == rec2, f"Difference at record {i+1}: {rec1} vs {rec2}"

    print("VCF files are identical.")


def compare_two_vk_summarize_txt_files(file1, file2):
    with open(file1, "r") as f1, open(file2, "r") as f2:
        lines1 = f1.readlines()
        lines2 = f2.readlines()

    assert lines1 == lines2, "Files differ."

def compare_two_jsons(file1, file2):
    # Load JSON files
    with open(file1) as f1, open(file2) as f2:
        json1 = json.load(f1)
        json2 = json.load(f2)

    # Compare as dictionaries
    print(json1 == json2)


import hashlib
def compute_checksum(file_path, algorithm='sha256'):
    """Compute file checksum using hashlib (pure Python)."""
    hash_func = hashlib.new(algorithm)
    with open(file_path, 'rb') as f:
        while chunk := f.read(8192):  # Read in chunks
            hash_func.update(chunk)
    return hash_func.hexdigest()

def compare_two_files_by_checksum(file1, file2):
    checksum1 = compute_checksum(file1)
    checksum2 = compute_checksum(file2)
    assert checksum1 == checksum2


@pytest.fixture
def vcrs_id_and_header_and_sequence_standard_lists():
    vcrs_id_list = [
        "vcrs_1204954474446204",
        "vcrs_4723197439168244",
        "vcrs_1693806423259989",
        "vcrs_8256702678403708",
        "vcrs_1784404960707341",
        "vcrs_7524932564184340",
        "vcrs_2241452516841814",
        "vcrs_9556672898923933",
        "vcrs_9627237534759445",
        "vcrs_3545345645923316",
        "vcrs_9762246785550270"
    ]
    vcrs_header_list = [
        "ENST00000312553:c.1758C>T", 
        "ENST00000312553:c.1503G>A;ENST00000527353:c.1503G>A", 
        "ENST00000312553:c.941C>A", 
        "ENST00000527353:c.9G>T", 
        "ENST000001441512:c.1411del", 
        "ENST000001441512:c.1411_1413del", 
        "ENST00000371237:c.660_661insATTGCAAA", 
        "ENST00000371237:c.306del;ENST00000371410:c.111_113delinsAG", 
        "ENST00000371226:c.351dup", 
        "ENST00000371226:c.5512_5515inv", 
        "ENST00000371225:c.907G>T;ENST00000371226:c.907G>T;ENST00000371227:c.907G>T;ENST00000371228:c.907G>T;ENST00000371229:c.907G>T"
    ]
    # w=54/k=55 (each sequence ~109bp)
    vcrs_sequence_list = [
        "AGGGGCTCCCCGAGTCACTTGAGTACCTGTACCTGCAGAACAACAAGATTAGTGTGGTGCCCGCCAATGCCTTCGACTCCACGCCCAACCTCAAGGGGATCTTTCTCAG",
        "GCTCGCTGGACCTGTCGGGCAACCGGCTGCACACGCTGCCACCTGGGCTGCCTCAAAATGTCCATGTGCTGAAGGTCAAGCGCAATGAGCTGGCTGCCTTGGCACGAGG",
        "AAGAACAACAAGCTGGAGAAGATCCCCCCGGGGGCCTTCAGCGAGCTGAGCAGCATGCGCGAGCTATACCTGCAGAACAACTACCTGACTGACGAGGGCCTGGACAACG",
        "CTCAACCTCAGCTACAACCGCATCACCAGCCCGCAGGTGCACCGCGACGCCTTCTGCAAGCTGCGCCTGCTGCGCTCGCTGGACCTGTCGGGCAACCGGCTGCACACGC",
        "CACCGGTGCACCTGTACAACAACGCGCTGGAGCGCGTGCCCAGTGGCCTGCCTAGCCGCGTGCGCACCCTCATGATCCTGCACAACCAGATCACAGGCATTGGCCGCG",
        "TTAACAAGCTGGCTGTGGCCGTGGTGGACAGTGCCTTCCGGAGGCTGAAGCTCCTGCAGGTCTTGGACATTGAAGGCAACTTAGAGTTTGGTGACATTTCCAAGGA",
        "GAAGCACCTGCAGGTCTTGGACATTGAAGGCAACTTAGAGTTTGGTGACATTTCTAATTGCAAAAGGACCGTGGCCGCTTGGGGAAGGAAAAGGAGGAGGAGGAAGAGAGGAGGAG",
        "CAGCGAGCTGAGCAGCCTGCGCGAGCTATACCTGCAGAACAACTACCTGACTAGGATGAGGGCCTGGACAACGAGACCTTCTGGAAGCTCTCGCCTGAGTACCTGGAT",
        "AATGAGCTGGCTGCCTTGGCACGAGGGGCGCTGGTGGGCATGGCTCAGCTGCATGAGCTGTACCTCACCAGCAACCGACTGCGCAGCCGAGCCCTGGGCCCCC",
        "GAGGAGGAGCCGGTGCTGGTACTGAGCCCTGAGGAGCCCGGGCCTGGCCCAGCCTCGGTCAGCTGCCCCCGAGACTGTGCCTGTTCCCAGGAGGGCGTCGTGGACTGTG",
        "CGCGAAGACTTTGCCACCACCTACTTCCTGGAGGAGCTCAACCTCAGCTACAACAGCATCACCAGCCCGCAGGTGCACCGCGACGCCTTCCGCAAGCTGCGCCTGCTGC"
    ]

    vcrs_genome_header_list = [
        "1:g.53546494C>T", 
        "1:g.53544534G>A;1:g.53544534G>A", 
        "1:g.53543408C>A", 
        "4:g.53544461G>T", 
        "X:g.1411del", 
        "X:g.73643517_73643519del", 
        "7:g.660141_660142insATTGCAAA", 
        "7:g.306511del;12:g.1132221_1132223delinsAG", 
        "21:g.35100dup",
        "21:g.5515278_5515281inv", 
        "9:g.907144G>T;9:g.907144G>T;9:g.907144G>T;9:g.907144G>T;9:g.907144G>T"
    ]

    assert len(vcrs_id_list) == len(vcrs_header_list) and len(vcrs_id_list) == len(vcrs_sequence_list) and len(vcrs_id_list) == len(vcrs_genome_header_list), "Not all list lengths are equal"


    return {
        'vcrs_id_list': vcrs_id_list, 
        'vcrs_header_list': vcrs_header_list, 
        'vcrs_sequence_list': vcrs_sequence_list, 
        'vcrs_genome_header_list': vcrs_genome_header_list
    }


def extract_seq_ids_and_mutations(vcrs_header_list):
    # Initialize lists to hold IDs and mutations separately
    ids_list = []
    mutations_list = []
    
    # Loop through each header in the list
    for header in vcrs_header_list:
        # Extract IDs and mutations separately for each entry
        ids = [item.split(':')[0] for item in header.split(';')]
        mutations = [item.split(':')[1] for item in header.split(';')]
        
        ids_list.append(ids)
        mutations_list.append(mutations)
    
    return ids_list, mutations_list



@pytest.fixture
def toy_mutation_metadata_df_path(vcrs_id_and_header_and_sequence_standard_lists, tmp_path):
    # Create a temporary CSV file
    temp_csv_file = tmp_path / "toy_mutation_metadata.csv"
    
    # Data to write to CSV
    vcrs_id_list = vcrs_id_and_header_and_sequence_standard_lists['vcrs_id_list']
    vcrs_header_list = vcrs_id_and_header_and_sequence_standard_lists['vcrs_header_list']
    vcrs_sequence_list = vcrs_id_and_header_and_sequence_standard_lists['vcrs_sequence_list']
    vcrs_genome_header_list = vcrs_id_and_header_and_sequence_standard_lists['vcrs_genome_header_list']
    seq_id_list, mutation_list = extract_seq_ids_and_mutations(vcrs_header_list)
    chromosome_list, mutation_genome_list = extract_seq_ids_and_mutations(vcrs_genome_header_list)
    
    data = {
        'vcrs_id': vcrs_id_list,
        'vcrs_header': vcrs_header_list,
        'vcrs_sequence': vcrs_sequence_list,
        'seq_ID': seq_id_list,
        'mutation': mutation_list,
        'chromosome': chromosome_list,
        'mutation_genome': mutation_genome_list
    }

    df = pd.DataFrame(data)

    df = add_mutation_information(df, mutation_column = "mutation", variant_source = None)
    df = add_variant_type(df, var_column = "mutation")
    df["vcrs_sequence_rc"] = df["vcrs_sequence"].apply(reverse_complement)

    df.to_csv(str(temp_csv_file), index=False)
    
    return str(temp_csv_file)

@pytest.fixture
def toy_mutation_metadata_df_with_read_parents_path(toy_mutation_metadata_df_path, tmp_path):
    temp_csv_file = tmp_path / "toy_mutation_metadata_with_read_parents.csv"
    
    nucleotides = ['A', 'T', 'C', 'G']

    df = pd.read_csv(toy_mutation_metadata_df_path)
    
    random.seed(42)
    padding_left = ''.join(random.choices(nucleotides, k=95))  # for w=54 above and read_length=150
    padding_right = ''.join(random.choices(nucleotides, k=95))
    df['mutant_sequence_read_parent'] = padding_left + df['vcrs_sequence'] + padding_right
    df['wt_sequence_read_parent'] = df['mutant_sequence_read_parent']   # TODO: this means that testing only supports mutant sequences for now

    df['header'] = df['vcrs_id']
    df['vcrs_variant_type'] = 'unknown'
    df["mutant_sequence_read_parent_rc"] = df["mutant_sequence_read_parent"].apply(reverse_complement)
    df["mutant_sequence_read_parent_length"] = df["mutant_sequence_read_parent"].apply(len)
    df["wt_sequence_read_parent_rc"] = df["wt_sequence_read_parent"].apply(reverse_complement)
    df["wt_sequence_read_parent_length"] = df["wt_sequence_read_parent"].apply(len)

    df.to_csv(str(temp_csv_file), index=False)

    return str(temp_csv_file)


@pytest.fixture
def toy_vcrs_fa_path(vcrs_id_and_header_and_sequence_standard_lists, tmp_path):
    # Create a temporary FASTA file
    temp_fasta_file = tmp_path / "toy_vcrs.fasta"

    vcrs_id_list = vcrs_id_and_header_and_sequence_standard_lists['vcrs_id_list']
    vcrs_sequence_list = vcrs_id_and_header_and_sequence_standard_lists['vcrs_sequence_list']

    with open(str(temp_fasta_file), 'w', encoding="utf-8") as fasta_file:
        for i in range(len(vcrs_id_list)):
            fasta_file.write(f">{vcrs_id_list[i]}\n")
            fasta_file.write(f"{vcrs_sequence_list[i]}\n")
        
    return str(temp_fasta_file)


@pytest.fixture
def dlist_file_small_path(vcrs_id_and_header_and_sequence_standard_lists, tmp_path):

    temp_dlist_fasta_file = tmp_path / "toy_dlist.fasta"
    temp_dlist_fasta_file = str(temp_dlist_fasta_file)
    
    vcrs_id_list = vcrs_id_and_header_and_sequence_standard_lists['vcrs_id_list']
    vcrs_sequence_list = vcrs_id_and_header_and_sequence_standard_lists['vcrs_sequence_list']

    # w=54/k=55, dfk_length = k + 2 = 57   -   dlist_left_flank + (vcrs_dlist_end - vcrs_dlist_start) + dlist_right_flank = 169
    dlist_information = [
        {"vcrs_id": vcrs_id_list[2], "vcrs_sequence": vcrs_sequence_list[2], "dlist_left_flank": "A"*50, "dlist_right_flank": "T"*60, "vcrs_dlist_start": 0, "vcrs_dlist_end": 59},
        {"vcrs_id": vcrs_id_list[5], "vcrs_sequence": vcrs_sequence_list[5], "dlist_left_flank": "A"*0, "dlist_right_flank": "T"*0, "vcrs_dlist_start": 0, "vcrs_dlist_end": "end"},  # substring
        {"vcrs_id": vcrs_id_list[8], "vcrs_sequence": vcrs_sequence_list[8], "dlist_left_flank": "A"*0, "dlist_right_flank": "T"*0, "vcrs_dlist_start": 44, "vcrs_dlist_end": "end"},
        {"vcrs_id": vcrs_id_list[9], "vcrs_sequence": vcrs_sequence_list[9], "dlist_left_flank": "A"*57, "dlist_right_flank": "T"*57, "vcrs_dlist_start": 12, "vcrs_dlist_end": 67}
    ]

    k = 55
    dfk_length = k + 2
    intended_dlist_entry_length = k + 2*dfk_length
    with open(temp_dlist_fasta_file, 'a', encoding="utf-8") as fasta_file:
        for dlist_entry in dlist_information:
            if dlist_entry["vcrs_dlist_end"] == "end":
                dlist_entry["vcrs_dlist_end"] = len(dlist_entry['vcrs_sequence'][dlist_entry["vcrs_dlist_start"]:])
                dlist_entry_is_end = True
            else:
                dlist_entry_is_end = False
                
            vcrs_sequence_fragment_in_dlist = dlist_entry['vcrs_sequence'][dlist_entry["vcrs_dlist_start"]:dlist_entry["vcrs_dlist_end"]]
            
            if dlist_entry_is_end:
                dfk_combined_length = intended_dlist_entry_length - len(vcrs_sequence_fragment_in_dlist)
                dfk_left_length = dfk_combined_length // 2
                dfk_right_length = dfk_combined_length - dfk_left_length
                dlist_entry["dlist_left_flank"] = "A"*dfk_left_length
                dlist_entry["dlist_right_flank"] = "T"*dfk_right_length
            
            dlist_sequence = f"{dlist_entry['dlist_left_flank']}{vcrs_sequence_fragment_in_dlist}{dlist_entry['dlist_right_flank']}"
            
            assert len(dlist_sequence) == intended_dlist_entry_length, f"The length of the dlist sequence is not the intended {intended_dlist_entry_length}, but is instead {len(dlist_sequence)}"

            fasta_file.write(f">{dlist_entry['vcrs_id']}_{dlist_entry['vcrs_dlist_start']}\n")
            fasta_file.write(f"{dlist_sequence}\n")

    return temp_dlist_fasta_file


@pytest.fixture
def toy_id_to_header_mapping_csv_path(vcrs_id_and_header_and_sequence_standard_lists, tmp_path):
    vcrs_id_list = vcrs_id_and_header_and_sequence_standard_lists['vcrs_id_list']
    vcrs_header_list = vcrs_id_and_header_and_sequence_standard_lists['vcrs_header_list']

    temp_id_to_header_mapping_csv = tmp_path / "toy_id_to_header_mapping.csv"

    # write csv
    with open(str(temp_id_to_header_mapping_csv), 'w', encoding="utf-8") as id_to_header_mapping_out:
        id_to_header_mapping_out.write("vcrs_id,vcrs_header\n")  # add headers
        
        for i in range(len(vcrs_id_list)):
            id_to_header_mapping_out.write(f"{vcrs_id_list[i]},{vcrs_header_list[i]}\n")

    return str(temp_id_to_header_mapping_csv)

@pytest.fixture
def toy_t2g_path(toy_vcrs_fa_path, tmp_path):
    temp_t2g = tmp_path / "toy_t2g.txt"

    create_identity_t2g(toy_vcrs_fa_path, out=str(temp_t2g))

    return str(temp_t2g)


@pytest.fixture
def toy_mutation_metadata_df_exploded():
    # Sample input DataFrame
    data = {
        "mutation": ["c.101A>G", "c.1211_1212insAAG", "c.256del", "c.256_260del", "c.401_403delinsG", "c.501delinsGA", "c.301_303dup", "c.301dup", "c.599_602inv", "c.101A>G", "c.108A>G"],
        "nucleotide_positions": ["101", "1211_1212", "256", "256_260", "401_403", "501", "301_303", "301", "599_602", "101", "108"],
        "actual_variant": ["A>G", "insAAG", "del", "del", "delinsG", "delinsGA", "dup", "dup", "inv", "A>G", "A>G"],
        "start_variant_position": [101, 1211, 256, 256, 401, 501, 301, 301, 599, 101, 108],
        "end_variant_position": [101, 1212, 256, 260, 403, 501, 303, 301, 602, 101, 108],
        "nucleotide_positions_cdna": ["101", "1211_1212", "256", "256_260", "401_403", "501", "301_303", "301", "599_602", "101", "108"],
        "actual_variant_cdna": ["A>G", "insAAG", "del", "del", "delinsG", "delinsGA", "dup", "dup", "inv", "A>G", "A>G"],
        "start_variant_position_cdna": [101, 1211, 256, 256, 401, 501, 301, 301, 599, 101, 108],
        "end_variant_position_cdna": [101, 1212, 256, 260, 403, 501, 303, 301, 602, 101, 108],
        "seq_ID": ["ENST1", "ENST2", "ENST3", "ENST4", "ENST5", "ENST6", "ENST7", "ENST8", "ENST9", "ENST1", "ENST11"],
        "vcrs_id": ["vcrs_1", "vcrs_2", "vcrs_3", "vcrs_4", "vcrs_5", "vcrs_6", "vcrs_7", "vcrs_8", "vcrs_9", "vcrs_1", "vcrs_1"],
        "vcrs_header": ["ENST1:c.101A>G;ENST1:c.101A>G;ENST11:c.108A>G", "ENST2:c.1211_1212insAAG", "ENST3:c.256del", "ENST4:c.256_260del", "ENST5:c.401_403delinsG", "ENST6:c.501delinsGA", "ENST7:c.301_303dup", "ENST8:c.301dup", "ENST9:c.599_602inv", "ENST1:c.101A>G;ENST1:c.101A>G;ENST11:c.108A>G", "ENST1:c.101A>G;ENST1:c.101A>G;ENST11:c.108A>G"],
        "vcrs_sequence": ["ATGCAAGGCTA", "TGGTGCATA", "GCTAGCGGCATA", "GACATCATCAG", "TTGACGGTACA", "GTCCCATACCGA", "GGCGTTGCAGCA", "GCCCAATGACAG", "ACATAGACAGGA", "ATGCAAGGCTA", "ATGCAAGGCTA"]
    }
    mutation_metadata_df = pd.DataFrame(data)

    mutation_metadata_df[["start_variant_position", "end_variant_position"]] = mutation_metadata_df[["start_variant_position", "end_variant_position"]].astype("Int64")
    mutation_metadata_df[["start_variant_position_cdna", "end_variant_position_cdna"]] = mutation_metadata_df[["start_variant_position_cdna", "end_variant_position_cdna"]].astype("Int64")

    mutation_metadata_df["header_list"] = mutation_metadata_df["vcrs_header"].str.split(";")
    mutation_metadata_df['order_list'] = mutation_metadata_df['header_list'].apply(lambda x: list(range(len(x))))

    mutation_metadata_df["header"] = mutation_metadata_df["header_list"]
    mutation_metadata_df["order"] = mutation_metadata_df["order_list"]

    return mutation_metadata_df


@pytest.fixture
def toy_mutation_metadata_df_collapsed():
    # Sample input DataFrame
    data = {
        "mutation": [["c.101A>G", "c.101A>G", "c.108A>G"], ["c.1211_1212insAAG"], ["c.256del"], ["c.256_260del"], ["c.401_403delinsG"], ["c.501delinsGA"], ["c.301_303dup"], ["c.301dup"], ["c.599_602inv"]],
        "nucleotide_positions": [["101", "101", "108"], ["1211_1212"], ["256"], ["256_260"], ["401_403"], ["501"], ["301_303"], ["301"], ["599_602"]],
        "actual_variant": [["A>G", "A>G", "A>G"], ["insAAG"], ["del"], ["del"], ["delinsG"], ["delinsGA"], ["dup"], ["dup"], ["inv"]],
        "start_variant_position": [[101, 101, 108], [1211], [256], [256], [401], [501], [301], [301], [599]],
        "end_variant_position": [[101, 101, 108], [1212], [256], [260], [403], [501], [303], [301], [602]],
        "nucleotide_positions_cdna": [["101", "101", "108"], ["1211_1212"], ["256"], ["256_260"], ["401_403"], ["501"], ["301_303"], ["301"], ["599_602"]],
        "actual_variant_cdna": [["A>G", "A>G", "A>G"], ["insAAG"], ["del"], ["del"], ["delinsG"], ["delinsGA"], ["dup"], ["dup"], ["inv"]],
        "start_variant_position_cdna": [[101, 101, 108], [1211], [256], [256], [401], [501], [301], [301], [599]],
        "end_variant_position_cdna": [[101, 101, 108], [1212], [256], [260], [403], [501], [303], [301], [602]],
        "seq_ID": [["ENST1", "ENST1", "ENST11"], ["ENST2"], ["ENST3"], ["ENST4"], ["ENST5"], ["ENST6"], ["ENST7"], ["ENST8"], ["ENST9"]],
        "vcrs_id": ["vcrs_1", "vcrs_2", "vcrs_3", "vcrs_4", "vcrs_5", "vcrs_6", "vcrs_7", "vcrs_8", "vcrs_9"],
        "vcrs_header": ["ENST1:c.101A>G;ENST1:c.101A>G;ENST11:c.108A>G", "ENST2:c.1211_1212insAAG", "ENST3:c.256del", "ENST4:c.256_260del", "ENST5:c.401_403delinsG", "ENST6:c.501delinsGA", "ENST7:c.301_303dup", "ENST8:c.301dup", "ENST9:c.599_602inv"],
        "vcrs_sequence": ["ATGCAAGGCTA", "TGGTGCATA", "GCTAGCGGCATA", "GACATCATCAG", "TTGACGGTACA", "GTCCCATACCGA", "GGCGTTGCAGCA", "GCCCAATGACAG", "ACATAGACAGGA"]
    }

    mutation_metadata_df = pd.DataFrame(data)

    return mutation_metadata_df


@pytest.fixture
def toy_sequences_fasta_for_vk_ref(tmp_path):
    fasta_path = tmp_path / "toy_reference.fasta"
    fasta_content = """>seq1
GCATGCTAGCGCGCGCCCTCTCTTAGAGCATCGAGCTACGAGCGAGTCCAGATGCCTGATGTACGCGCGAGCGAGAGAGGAGAGAAAAGACTCGCA
>seq2
GCGCGCGCGCATCGATCGCACTGCGCAGAAAGAGAGCGGGCCCCGCTACGAGCATCGACGAGCGACTGCGGGGGCCGCAACACGCGCGCGCAGAGC
"""
    fasta_path.write_text(fasta_content)
    return fasta_path

@pytest.fixture
def toy_variants_csv_for_vk_ref(tmp_path):
    csv_path = tmp_path / "toy_variants.csv"
    
    data = {
        "seq_ID": ["seq1", "seq1", "seq2"],
        "mutation": ["c.21T>A", "c.30_31insGG", "c.48A>C"],
    }

    df = pd.DataFrame(data)
    df.to_csv(csv_path, index=False)
    return csv_path


@pytest.fixture
def vcf_file_and_corresponding_sequences(tmp_path):
    vcf_content = """\
##fileformat=VCFv4.0
##fileDate=20090805
##source=myImputationProgramV3.1
##reference=1000GenomesPilot-NCBI36
##phasing=partial
##INFO=<ID=NS,Number=1,Type=Integer,Description="Number of Samples With Data">
##INFO=<ID=AN,Number=1,Type=Integer,Description="Total number of alleles in called genotypes">
##INFO=<ID=AC,Number=.,Type=Integer,Description="Allele count in genotypes, for each ALT allele, in the same order as listed">
##INFO=<ID=DP,Number=1,Type=Integer,Description="Total Depth">
##INFO=<ID=AF,Number=.,Type=Float,Description="Allele Frequency">
##INFO=<ID=AA,Number=1,Type=String,Description="Ancestral Allele">
##INFO=<ID=DB,Number=0,Type=Flag,Description="dbSNP membership, build 129">
##INFO=<ID=H2,Number=0,Type=Flag,Description="HapMap2 membership">
##FILTER=<ID=q10,Description="Quality below 10">
##FILTER=<ID=s50,Description="Less than 50% of samples have data">
##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">
##FORMAT=<ID=GQ,Number=1,Type=Integer,Description="Genotype Quality">
##FORMAT=<ID=DP,Number=1,Type=Integer,Description="Read Depth">
##FORMAT=<ID=HQ,Number=2,Type=Integer,Description="Haplotype Quality">
##ALT=<ID=DEL:ME:ALU,Description="Deletion of ALU element">
##ALT=<ID=CNV,Description="Copy number variable region">
#CHROM	POS	ID	REF	ALT	QUAL	FILTER	INFO	FORMAT	NA00001	NA00002	NA00003
19	11	single_sub_mid;g.11A>C	A	C	1	PASS	.	GT:HQ	0|0:10,10	0|0:10,10	0/1:3,3
19	22	exploded_sub_mid;g.22G>A,g.22G>C,g.22G>T	G	A,C,T	1	PASS	NS=3;DP=14;AF=0.5;DB;H2	GT:GQ:DP:HQ	0|0:48:1:51,51	1|0:48:8:51,51	1/1:43:5:.,.
19	1	sub_begin;g.1T>A	T	A	1	PASS	NS=3;DP=11;AF=0.017	GT:GQ:DP:HQ	0|0:49:3:58,50	0|1:3:5:65,3	0/0:41:3:.,.
19	33	empty_alt;None	T	.	None	FAIL	NS=3;DP=13;AA=T	GT:GQ:DP:HQ	0|0:54:.:56,60	0|0:48:4:51,51	0/0:61:2:.,.
19	34	exploded_ins_mid;g.34_35insA,g.34_35insAC	G	GA,GAC	0	PASS	NS=3;DP=9;AA=G;AN=6;AC=3,1	GT:GQ:DP	0/1:.:4	0/2:17:2	1/1:40:3
19	1	ins_begin;None	T	AGT	None	FAIL	NS=3;DP=9;AA=G;AN=6;AC=3,1	GT:GQ:DP	0/1:.:4	0/2:17:2	1/1:40:3
19	40	single_del_mid;g.41del	AG	A	0	PASS	NS=3;DP=9;AA=G;AN=6;AC=3,1	GT:GQ:DP	0/1:.:4	0/2:17:2	1/1:40:3
19	40	multi_del_mid;g.41_44del	AGCAT	A	0	PASS	NS=3;DP=9;AA=G;AN=6;AC=3,1	GT:GQ:DP	0/1:.:4	0/2:17:2	1/1:40:3
19	1	single_del_begin;g.1del	TC	C	0	PASS	NS=3;DP=9;AA=G;AN=6;AC=3,1	GT:GQ:DP	0/1:.:4	0/2:17:2	1/1:40:3
19	1	multi_del_begin;g.1_4del	TCATC	C	0	PASS	NS=3;DP=9;AA=G;AN=6;AC=3,1	GT:GQ:DP	0/1:.:4	0/2:17:2	1/1:40:3
19	60	delins;g.60_62delinsAA	TCG	AA	0	PASS	NS=3;DP=9;AA=G;AN=6;AC=3,1	GT:GQ:DP	0/1:.:4	0/2:17:2	1/1:40:3
19	40	duplication;g.40dup	A	AA	0	PASS	NS=3;DP=9;AA=G;AN=6;AC=3,1	GT:GQ:DP	0/1:.:4	0/2:17:2	1/1:40:3
19	40	inversion;g.40_43inv	TCGT	ACGA	0	PASS	NS=3;DP=9;AA=G;AN=6;AC=3,1	GT:GQ:DP	0/1:.:4	0/2:17:2	1/1:40:3
19	50	duplication;g.47_50dup(initiallyg.50_51insGCCC)	C	CGCCC	0	PASS	NS=3;DP=9;AA=G;AN=6;AC=3,1	GT:GQ:DP	0/1:.:4	0/2:17:2	1/1:40:3
"""
    
    vcf_path = tmp_path / "test.vcf"
    vcf_path.write_text(vcf_content)

    vcf_sequence_dict = {"19": "TCATCGAACTAGCAGCTCGACGACGCACATCGTGGATCCAGCATCAGCCCCCTCTCGAGTCGCATCGCATCG"}
                              # 123456789_123456789_123456789_123456789_123456789_123456789_123456789_123
                              #          10        20        30        40        50        60        70

    fasta_content = ""
    for key, value in vcf_sequence_dict.items():
        fasta_content += f">{key}\n{value}\n"

    fasta_path = tmp_path / "test.fasta"
    fasta_path.write_text(fasta_content)

    vcf_output_ground_truth_df_data = """vcrs_header,seq_ID,mutation,variant_type,wt_sequence,vcrs_sequence,nucleotide_positions,actual_variant,start_variant_position,end_variant_position,vcrs_id
19:g.11A>C,19,g.11A>C,substitution,CGAACTAGCAGCT,CGAACTCGCAGCT,11,A>C,11,11,vcrs_01
19:g.22G>A,19,g.22G>A,substitution,CTCGACGACGCAC,CTCGACAACGCAC,22,G>A,22,22,vcrs_02
19:g.22G>C,19,g.22G>C,substitution,CTCGACGACGCAC,CTCGACCACGCAC,22,G>C,22,22,vcrs_03
19:g.22G>T,19,g.22G>T,substitution,CTCGACGACGCAC,CTCGACTACGCAC,22,G>T,22,22,vcrs_04
19:g.1T>A,19,g.1T>A,substitution,TCATCGA,ACATCGA,1,T>A,1,1,vcrs_05
19:g.34_35insA,19,g.34_35insA,insertion,ATCGTGGATCCA,ATCGTGAGATCCA,34_35,insA,34,35,vcrs_06
19:g.34_35insAC,19,g.34_35insAC,insertion,ATCGTGGATCCA,ATCGTGACGATCCA,34_35,insAC,34,35,vcrs_07
19:g.41del,19,g.41del,deletion,GATCCAGCATCAG,GATCCACATCAG,41,del,41,41,vcrs_08
19:g.41_44del,19,g.41_44del,deletion,GATCCAGCATCAGCCC,GATCCACAGCCC,41_44,del,41,44,vcrs_09
19:g.1del,19,g.1del,deletion,TCATCGA,CATCGA,1,del,1,1,vcrs_10
19:g.1_4del,19,g.1_4del,deletion,TCATCGAACT,CGAACT,1_4,del,1,4,vcrs_11
19:g.60_62delinsAA,19,g.60_62delinsAA,delins,CTCGAGTCGCATCGC,CTCGAGAACATCGC,60_62,delinsAA,60,62,vcrs_12
19:g.40dup,19,g.40dup,duplication,GATCCAGCATCA,GATCCAAGCATCA,40,dup,40,40,vcrs_13
19:g.40_43inv,19,g.40_43inv,inversion,GGATCCAGCATCAGCC,GGATCCTGCTTCAGCC,40_43,inv,40,43,vcrs_14
19:g.47_50dup,19,g.47_50dup,duplication,CAGCCCCCTCTC,CAGCCCGCCCCCTCTC,47_50,dup,47,50,vcrs_15
"""

    vcf_output_ground_truth_df = pd.read_csv(StringIO(vcf_output_ground_truth_df_data))

    return vcf_path, fasta_path, vcf_output_ground_truth_df