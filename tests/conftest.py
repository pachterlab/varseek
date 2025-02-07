import os
import random
import tempfile

import numpy as np
import pandas as pd
import pytest

import varseek as vk
from varseek.utils import (
    create_header_to_sequence_ordered_dict_from_fasta_WITHOUT_semicolon_splitting,
    create_mutant_t2g,
    make_mapping_dict,
)


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
def mcrs_id_and_header_and_sequence_standard_lists():
    mcrs_id_list = [
        "seq1204954474446204",
        "seq4723197439168244",
        "seq1693806423259989",
        "seq8256702678403708",
        "seq1784404960707341",
        "seq7524932564184340",
        "seq2241452516841814",
        "seq9556672898923933",
        "seq9627237534759445",
        "seq3545345645923316",
        "seq9762246785550270"
    ]
    mcrs_header_list = [
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
    mcrs_sequence_list = [
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

    mcrs_genome_header_list = [
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

    assert len(mcrs_id_list) == len(mcrs_header_list) and len(mcrs_id_list) == len(mcrs_sequence_list) and len(mcrs_id_list) == len(mcrs_genome_header_list), "Not all list lengths are equal"


    return {
        'mcrs_id_list': mcrs_id_list, 
        'mcrs_header_list': mcrs_header_list, 
        'mcrs_sequence_list': mcrs_sequence_list, 
        'mcrs_genome_header_list': mcrs_genome_header_list
    }


def extract_seq_ids_and_mutations(mcrs_header_list):
    # Initialize lists to hold IDs and mutations separately
    ids_list = []
    mutations_list = []
    
    # Loop through each header in the list
    for header in mcrs_header_list:
        # Extract IDs and mutations separately for each entry
        ids = [item.split(':')[0] for item in header.split(';')]
        mutations = [item.split(':')[1] for item in header.split(';')]
        
        ids_list.append(ids)
        mutations_list.append(mutations)
    
    return ids_list, mutations_list



@pytest.fixture
def toy_mutation_metadata_df_path(mcrs_id_and_header_and_sequence_standard_lists):
    # Create a temporary CSV file
    temp_csv_file = tempfile.NamedTemporaryFile(delete=False, suffix='.csv')
    
    # Data to write to CSV
    mcrs_id_list = mcrs_id_and_header_and_sequence_standard_lists['mcrs_id_list']
    mcrs_header_list = mcrs_id_and_header_and_sequence_standard_lists['mcrs_header_list']
    mcrs_sequence_list = mcrs_id_and_header_and_sequence_standard_lists['mcrs_sequence_list']
    mcrs_genome_header_list = mcrs_id_and_header_and_sequence_standard_lists['mcrs_genome_header_list']
    seq_id_list, mutation_list = extract_seq_ids_and_mutations(mcrs_header_list)
    chromosome_list, mutation_genome_list = extract_seq_ids_and_mutations(mcrs_genome_header_list)
    
    data = {
        'mcrs_id': mcrs_id_list,
        'mcrs_header': mcrs_header_list,
        'mcrs_sequence': mcrs_sequence_list,
        'seq_ID': seq_id_list,
        'mutation': mutation_list,
        'chromosome': chromosome_list,
        'mutation_genome': mutation_genome_list
    }

    df = pd.DataFrame(data)

    df = vk.varseek_info.add_mutation_information(df, mutation_column = "mutation", mcrs_source = None)
    df = vk.varseek_build.add_mutation_type(df, mut_column = "mutation")
    df["mutant_sequence_rc"] = df["mcrs_sequence"].apply(vk.varseek_build.reverse_complement)

    df.to_csv(temp_csv_file.name, index=False)
    
    yield temp_csv_file.name
    
    # Cleanup
    os.remove(temp_csv_file.name)

@pytest.fixture
def toy_mutation_metadata_df_with_read_parents_path(toy_mutation_metadata_df_path):
    temp_csv_file = tempfile.NamedTemporaryFile(delete=False, suffix='.csv')
    nucleotides = ['A', 'T', 'C', 'G']

    df = pd.read_csv(toy_mutation_metadata_df_path)
    
    random.seed(42)
    padding_left = ''.join(random.choices(nucleotides, k=95))  # for w=54 above and read_length=150
    padding_right = ''.join(random.choices(nucleotides, k=95))
    df['mutant_sequence_read_parent'] = padding_left + df['mcrs_sequence'] + padding_right
    df['wt_sequence_read_parent'] = df['mutant_sequence_read_parent']   # TODO: this means that testing only supports mutant sequences for now

    df['header'] = df['mcrs_id']
    df['mcrs_mutation_type'] = 'unknown'
    df["mutant_sequence_read_parent_rc"] = df["mutant_sequence_read_parent"].apply(vk.varseek_build.reverse_complement)
    df["mutant_sequence_read_parent_length"] = df["mutant_sequence_read_parent"].apply(len)
    df["wt_sequence_read_parent_rc"] = df["wt_sequence_read_parent"].apply(vk.varseek_build.reverse_complement)
    df["wt_sequence_read_parent_length"] = df["wt_sequence_read_parent"].apply(len)

    df.to_csv(temp_csv_file.name, index=False)

    yield temp_csv_file.name

    os.remove(temp_csv_file.name)


@pytest.fixture
def toy_mcrs_fa_path(mcrs_id_and_header_and_sequence_standard_lists):
    # Create a temporary FASTA file
    temp_fasta_file = tempfile.NamedTemporaryFile(delete=False, suffix='.fasta')

    mcrs_id_list = mcrs_id_and_header_and_sequence_standard_lists['mcrs_id_list']
    mcrs_sequence_list = mcrs_id_and_header_and_sequence_standard_lists['mcrs_sequence_list']

    with open(temp_fasta_file.name, 'w') as fasta_file:
        for i in range(len(mcrs_id_list)):
            fasta_file.write(f">{mcrs_id_list[i]}\n")
            fasta_file.write(f"{mcrs_sequence_list[i]}\n")
        
    yield temp_fasta_file.name
    
    # Cleanup
    os.remove(temp_fasta_file.name)


@pytest.fixture
def dlist_file_small_path(mcrs_id_and_header_and_sequence_standard_lists):

    temp_dlist_fasta_file = tempfile.NamedTemporaryFile(delete=False, suffix='.fasta')
    
    mcrs_id_list = mcrs_id_and_header_and_sequence_standard_lists['mcrs_id_list']
    mcrs_sequence_list = mcrs_id_and_header_and_sequence_standard_lists['mcrs_sequence_list']

    # w=54/k=55, dfk_length = k + 2 = 57   -   dlist_left_flank + (mcrs_dlist_end - mcrs_dlist_start) + dlist_right_flank = 169
    dlist_information = [
        {"mcrs_id": mcrs_id_list[2], "mcrs_sequence": mcrs_sequence_list[2], "dlist_left_flank": "A"*50, "dlist_right_flank": "T"*60, "mcrs_dlist_start": 0, "mcrs_dlist_end": 59},
        {"mcrs_id": mcrs_id_list[5], "mcrs_sequence": mcrs_sequence_list[5], "dlist_left_flank": "A"*0, "dlist_right_flank": "T"*0, "mcrs_dlist_start": 0, "mcrs_dlist_end": "end"},  # substring
        {"mcrs_id": mcrs_id_list[8], "mcrs_sequence": mcrs_sequence_list[8], "dlist_left_flank": "A"*0, "dlist_right_flank": "T"*0, "mcrs_dlist_start": 44, "mcrs_dlist_end": "end"},
        {"mcrs_id": mcrs_id_list[9], "mcrs_sequence": mcrs_sequence_list[9], "dlist_left_flank": "A"*57, "dlist_right_flank": "T"*57, "mcrs_dlist_start": 12, "mcrs_dlist_end": 67}
    ]

    k = 55
    dfk_length = k + 2
    intended_dlist_entry_length = k + 2*dfk_length
    with open(temp_dlist_fasta_file.name, 'a') as fasta_file:
        for dlist_entry in dlist_information:
            if dlist_entry["mcrs_dlist_end"] == "end":
                dlist_entry["mcrs_dlist_end"] = len(dlist_entry['mcrs_sequence'][dlist_entry["mcrs_dlist_start"]:])
                dlist_entry_is_end = True
            else:
                dlist_entry_is_end = False
                
            mcrs_sequence_fragment_in_dlist = dlist_entry['mcrs_sequence'][dlist_entry["mcrs_dlist_start"]:dlist_entry["mcrs_dlist_end"]]
            
            if dlist_entry_is_end:
                dfk_combined_length = intended_dlist_entry_length - len(mcrs_sequence_fragment_in_dlist)
                dfk_left_length = dfk_combined_length // 2
                dfk_right_length = dfk_combined_length - dfk_left_length
                dlist_entry["dlist_left_flank"] = "A"*dfk_left_length
                dlist_entry["dlist_right_flank"] = "T"*dfk_right_length
            
            dlist_sequence = f"{dlist_entry['dlist_left_flank']}{mcrs_sequence_fragment_in_dlist}{dlist_entry['dlist_right_flank']}"
            
            assert len(dlist_sequence) == intended_dlist_entry_length, f"The length of the dlist sequence is not the intended {intended_dlist_entry_length}, but is instead {len(dlist_sequence)}"

            fasta_file.write(f">{dlist_entry['mcrs_id']}_{dlist_entry['mcrs_dlist_start']}\n")
            fasta_file.write(f"{dlist_sequence}\n")

    yield temp_dlist_fasta_file.name

    # Cleanup
    os.remove(temp_dlist_fasta_file.name)


@pytest.fixture
def toy_id_to_header_mapping_csv_path(mcrs_id_and_header_and_sequence_standard_lists):
    mcrs_id_list = mcrs_id_and_header_and_sequence_standard_lists['mcrs_id_list']
    mcrs_header_list = mcrs_id_and_header_and_sequence_standard_lists['mcrs_header_list']

    temp_id_to_header_mapping_csv = tempfile.NamedTemporaryFile(delete=False, suffix='.csv')

    # write csv
    with open(temp_id_to_header_mapping_csv.name, 'w') as id_to_header_mapping_out:
        for i in range(len(mcrs_id_list)):
            id_to_header_mapping_out.write(f"{mcrs_id_list[i]},{mcrs_header_list[i]}\n")

    yield temp_id_to_header_mapping_csv.name

    # Cleanup
    os.remove(temp_id_to_header_mapping_csv.name)

@pytest.fixture
def toy_t2g_path(toy_mcrs_fa_path):
    temp_t2g = tempfile.NamedTemporaryFile(delete=False, suffix='.txt')

    create_mutant_t2g(toy_mcrs_fa_path, out=temp_t2g.name)

    yield temp_t2g.name

    # Cleanup
    os.remove(temp_t2g.name)


@pytest.fixture
def toy_mutation_metadata_df_exploded():
    # Sample input DataFrame
    data = {
        "mutation": ["c.101A>G", "c.1211_1212insAAG", "c.256del", "c.256_260del", "c.401_403delinsG", "c.501delinsGA", "c.301_303dup", "c.301dup", "c.599_602inv", "c.101A>G", "c.108A>G"],
        "nucleotide_positions": ["101", "1211_1212", "256", "256_260", "401_403", "501", "301_303", "301", "599_602", "101", "108"],
        "actual_mutation": ["A>G", "insAAG", "del", "del", "delinsG", "delinsGA", "dup", "dup", "inv", "A>G", "A>G"],
        "start_mutation_position": [101, 1211, 256, 256, 401, 501, 301, 301, 599, 101, 108],
        "end_mutation_position": [101, 1212, 256, 260, 403, 501, 303, 301, 602, 101, 108],
        "nucleotide_positions_cdna": ["101", "1211_1212", "256", "256_260", "401_403", "501", "301_303", "301", "599_602", "101", "108"],
        "actual_mutation_cdna": ["A>G", "insAAG", "del", "del", "delinsG", "delinsGA", "dup", "dup", "inv", "A>G", "A>G"],
        "start_mutation_position_cdna": [101, 1211, 256, 256, 401, 501, 301, 301, 599, 101, 108],
        "end_mutation_position_cdna": [101, 1212, 256, 260, 403, 501, 303, 301, 602, 101, 108],
        "seq_ID": ["ENST1", "ENST2", "ENST3", "ENST4", "ENST5", "ENST6", "ENST7", "ENST8", "ENST9", "ENST1", "ENST11"],
        "mcrs_id": ["seq_1", "seq_2", "seq_3", "seq_4", "seq_5", "seq_6", "seq_7", "seq_8", "seq_9", "seq_1", "seq_1"],
        "mcrs_header": ["ENST1:c.101A>G;ENST1:c.101A>G;ENST11:c.108A>G", "ENST2:c.1211_1212insAAG", "ENST3:c.256del", "ENST4:c.256_260del", "ENST5:c.401_403delinsG", "ENST6:c.501delinsGA", "ENST7:c.301_303dup", "ENST8:c.301dup", "ENST9:c.599_602inv", "ENST1:c.101A>G;ENST1:c.101A>G;ENST11:c.108A>G", "ENST1:c.101A>G;ENST1:c.101A>G;ENST11:c.108A>G"],
        "mcrs_sequence": ["ATGCAAGGCTA", "TGGTGCATA", "GCTAGCGGCATA", "GACATCATCAG", "TTGACGGTACA", "GTCCCATACCGA", "GGCGTTGCAGCA", "GCCCAATGACAG", "ACATAGACAGGA", "ATGCAAGGCTA", "ATGCAAGGCTA"]
    }
    mutation_metadata_df = pd.DataFrame(data)

    mutation_metadata_df[["start_mutation_position", "end_mutation_position"]] = mutation_metadata_df[["start_mutation_position", "end_mutation_position"]].astype("Int64")
    mutation_metadata_df[["start_mutation_position_cdna", "end_mutation_position_cdna"]] = mutation_metadata_df[["start_mutation_position_cdna", "end_mutation_position_cdna"]].astype("Int64")

    mutation_metadata_df["header_list"] = mutation_metadata_df["mcrs_header"].str.split(";")
    mutation_metadata_df['order_list'] = mutation_metadata_df['header_list'].apply(lambda x: list(range(len(x))))

    mutation_metadata_df["header"] = mutation_metadata_df["header_list"]
    mutation_metadata_df["order"] = mutation_metadata_df["order_list"]

    yield mutation_metadata_df


@pytest.fixture
def toy_mutation_metadata_df_collapsed():
    # Sample input DataFrame
    data = {
        "mutation": [["c.101A>G", "c.101A>G", "c.108A>G"], ["c.1211_1212insAAG"], ["c.256del"], ["c.256_260del"], ["c.401_403delinsG"], ["c.501delinsGA"], ["c.301_303dup"], ["c.301dup"], ["c.599_602inv"]],
        "nucleotide_positions": [["101", "101", "108"], ["1211_1212"], ["256"], ["256_260"], ["401_403"], ["501"], ["301_303"], ["301"], ["599_602"]],
        "actual_mutation": [["A>G", "A>G", "A>G"], ["insAAG"], ["del"], ["del"], ["delinsG"], ["delinsGA"], ["dup"], ["dup"], ["inv"]],
        "start_mutation_position": [[101, 101, 108], [1211], [256], [256], [401], [501], [301], [301], [599]],
        "end_mutation_position": [[101, 101, 108], [1212], [256], [260], [403], [501], [303], [301], [602]],
        "nucleotide_positions_cdna": [["101", "101", "108"], ["1211_1212"], ["256"], ["256_260"], ["401_403"], ["501"], ["301_303"], ["301"], ["599_602"]],
        "actual_mutation_cdna": [["A>G", "A>G", "A>G"], ["insAAG"], ["del"], ["del"], ["delinsG"], ["delinsGA"], ["dup"], ["dup"], ["inv"]],
        "start_mutation_position_cdna": [[101, 101, 108], [1211], [256], [256], [401], [501], [301], [301], [599]],
        "end_mutation_position_cdna": [[101, 101, 108], [1212], [256], [260], [403], [501], [303], [301], [602]],
        "seq_ID": [["ENST1", "ENST1", "ENST11"], ["ENST2"], ["ENST3"], ["ENST4"], ["ENST5"], ["ENST6"], ["ENST7"], ["ENST8"], ["ENST9"]],
        "mcrs_id": ["seq_1", "seq_2", "seq_3", "seq_4", "seq_5", "seq_6", "seq_7", "seq_8", "seq_9"],
        "mcrs_header": ["ENST1:c.101A>G;ENST1:c.101A>G;ENST11:c.108A>G", "ENST2:c.1211_1212insAAG", "ENST3:c.256del", "ENST4:c.256_260del", "ENST5:c.401_403delinsG", "ENST6:c.501delinsGA", "ENST7:c.301_303dup", "ENST8:c.301dup", "ENST9:c.599_602inv"],
        "mcrs_sequence": ["ATGCAAGGCTA", "TGGTGCATA", "GCTAGCGGCATA", "GACATCATCAG", "TTGACGGTACA", "GTCCCATACCGA", "GGCGTTGCAGCA", "GCCCAATGACAG", "ACATAGACAGGA"]
    }

    mutation_metadata_df = pd.DataFrame(data)

    yield mutation_metadata_df
