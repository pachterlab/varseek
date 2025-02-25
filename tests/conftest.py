import os
import random
import tempfile

import numpy as np
import pandas as pd
import sys
import pytest

import varseek as vk
from varseek.utils import (
    create_header_to_sequence_ordered_dict_from_fasta_WITHOUT_semicolon_splitting,
    create_identity_t2g,
    make_mapping_dict,
    reverse_complement
)

def pytest_ignore_collect(path, config):  # skip test_bustools.py on Mac due to kb python issues
    if sys.platform == "darwin" and "test_bustools.py" in str(path):
        return True


def compare_two_dataframes_without_regard_for_order_of_rows_or_columns(df1_path, df2_path, columns_to_drop=None, head=False):
    df1 = pd.read_csv(df1_path)
    df2 = pd.read_csv(df2_path)

    if columns_to_drop:
        if type(columns_to_drop) == str:
            columns_to_drop = [columns_to_drop]
        
        df1 = df1.drop(columns=columns_to_drop, errors='ignore')
        df2 = df2.drop(columns=columns_to_drop, errors='ignore')

    # Sort by all columns and reset index to ignore both row and column order
    df1_sorted = df1.sort_values(by=list(df1.columns)).reset_index(drop=True)
    df2_sorted = df2.sort_values(by=list(df2.columns)).reset_index(drop=True)

    if head:
        df1 = df1.head()
        df2 = df2.head()

    pd.testing.assert_frame_equal(df1_sorted, df2_sorted, check_like=True)

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

    df = vk.varseek_info.add_mutation_information(df, mutation_column = "mutation", variant_source = None)
    df = vk.utils.add_mutation_type(df, var_column = "mutation")
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
    df['vcrs_mutation_type'] = 'unknown'
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