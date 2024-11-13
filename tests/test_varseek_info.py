import pytest
import tempfile
import varseek as vk
import pandas as pd
import numpy as np
import os
from pdb import set_trace as st
from varseek.utils import compare_dicts, collapse_df, explode_df, compare_cdna_and_genome, compute_distance_to_closest_splice_junction, calculate_nearby_mutations, compare_cdna_and_genome, align_to_normal_genome_and_build_dlist, get_mcrss_that_pseudoalign_but_arent_dlisted, get_df_overlap, longest_homopolymer, triplet_stats, add_mcrs_mutation_type, create_df_of_mcrs_to_self_headers, count_kmer_overlaps
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from Bio import SeqIO
import tempfile

from varseek.varseek_info import add_mutation_information



@pytest.fixture
def temporary_output_files():
    with tempfile.NamedTemporaryFile(suffix=".csv") as output_metadata_df, \
         tempfile.NamedTemporaryFile(suffix=".fasta") as output_mcrs_fasta, \
         tempfile.NamedTemporaryFile(suffix=".fasta") as output_dlist_fasta, \
         tempfile.NamedTemporaryFile(suffix=".csv") as output_id_to_header_csv, \
         tempfile.NamedTemporaryFile(suffix=".txt") as output_t2g:
        
        # Dictionary of temporary paths
        temp_files = {
            "output_metadata_df": output_metadata_df.name,
            "output_mcrs_fasta": output_mcrs_fasta.name,
            "output_dlist_fasta": output_dlist_fasta.name,
            "output_id_to_header_csv": output_id_to_header_csv.name,
            "output_t2g": output_t2g.name
        }
        
        yield temp_files  # Provide paths to test

        # Temporary files are automatically cleaned up after `yield`


def test_add_mutation_information(toy_mutation_metadata_df_exploded):
    expected_df = toy_mutation_metadata_df_exploded

    column_to_drop_list = ["header", "header_list", "order", "order_list"]
    for column in column_to_drop_list:
        if column in toy_mutation_metadata_df_exploded.columns:
            toy_mutation_metadata_df_exploded.drop(column, axis=1, inplace=True)

    mutation_metadata_df = toy_mutation_metadata_df_exploded[["mutation", "seq_ID", "mcrs_id", "mcrs_header", "mcrs_sequence"]].copy()
    output_df = add_mutation_information(mutation_metadata_df, mutation_column="mutation", mcrs_source="cdna")
    output_df = output_df[expected_df.columns]

    # Assert that the output matches the expected DataFrame
    pd.testing.assert_frame_equal(output_df, expected_df)

def test_collapse(toy_mutation_metadata_df_exploded, toy_mutation_metadata_df_collapsed):
    expected_df = toy_mutation_metadata_df_collapsed

    mutation_metadata_df_exploded = toy_mutation_metadata_df_exploded
    columns_to_explode = [col for col in mutation_metadata_df_exploded.columns if col not in ['mcrs_id', 'mcrs_header', 'mcrs_sequence']]
    output_df, _ = collapse_df(mutation_metadata_df_exploded, columns_to_explode = columns_to_explode, columns_to_explode_extend_values = None)
    output_df = output_df[expected_df.columns]

    # Assert that the output matches the expected DataFrame
    pd.testing.assert_frame_equal(output_df, expected_df)


def test_explode(toy_mutation_metadata_df_exploded, toy_mutation_metadata_df_collapsed):
    expected_df = toy_mutation_metadata_df_exploded

    mutation_metadata_df_collapsed = toy_mutation_metadata_df_collapsed
    columns_to_explode = [col for col in mutation_metadata_df_collapsed.columns if col not in ['mcrs_id', 'mcrs_header', 'mcrs_sequence']]
    output_df = explode_df(mutation_metadata_df_collapsed, columns_to_explode = columns_to_explode)
    output_df = output_df[expected_df.columns]

    output_df = output_df.sort_values(by=['mcrs_id']).reset_index(drop=True)
    expected_df = expected_df.sort_values(by=['mcrs_id']).reset_index(drop=True)

    output_df[["start_mutation_position", "end_mutation_position", "start_mutation_position_cdna", "end_mutation_position_cdna"]] = output_df[["start_mutation_position", "end_mutation_position", "start_mutation_position_cdna", "end_mutation_position_cdna"]].astype("Int64")
    expected_df[["start_mutation_position", "end_mutation_position", "start_mutation_position_cdna", "end_mutation_position_cdna"]] = expected_df[["start_mutation_position", "end_mutation_position", "start_mutation_position_cdna", "end_mutation_position_cdna"]].astype("Int64")

    # Assert that the output matches the expected DataFrame
    pd.testing.assert_frame_equal(output_df, expected_df)


def mock_load_splice_junctions_from_gtf(gtf_path):
    # Simulate a GTF file with splice junctions
    # Format: {"chromosome": [junction1, junction2, ...]}
    return {
        "1": [100, 200, 300],
        "2": [150, 250, 350]
    }

@pytest.fixture
def mock_helpers(monkeypatch):
    monkeypatch.setattr("varseek.utils.seq_utils.load_splice_junctions_from_gtf", mock_load_splice_junctions_from_gtf)

def test_compute_distance_to_closest_splice_junction(mock_helpers):
    # Create a toy DataFrame with mutations
    data = {
        "chromosome": ["1", "1", "2", "2", "2", "2", "2", "3"],  # Chromosome 3 is to check missing junctions
        "start_mutation_position_genome": [95, 210, 240, 248, 250, 550, 700, 400],
        "end_mutation_position_genome": [105, 210, 260, 254, 250, 550, 700, 400]
    }
    mutation_metadata_df_exploded = pd.DataFrame(data)

    # Expected results
    expected_distances = [5, 10, 10, 2, 0, 200, 350, np.nan]
    expected_is_near = [True, True, True, True, True, False, False, np.nan]  # Using threshold 10

    # Run the function with the toy DataFrame and mock GTF path
    output_df, columns_to_explode = compute_distance_to_closest_splice_junction(
        mutation_metadata_df_exploded,
        reference_genome_gtf="mock_path",
        columns_to_explode=None,
        near_splice_junction_threshold=10
    )

    # Check that the computed distances are as expected
    pd.testing.assert_series_equal(
        output_df["distance_to_nearest_splice_junction"],
        pd.Series(expected_distances, name="distance_to_nearest_splice_junction"),
        check_dtype=False
    )

    # Check the is_near_splice_junction column
    pd.testing.assert_series_equal(
        output_df["is_near_splice_junction_10"],
        pd.Series(expected_is_near, name="is_near_splice_junction_10"),
        check_dtype=False
    )

    # Check that columns to explode includes the correct columns
    assert "distance_to_nearest_splice_junction" in columns_to_explode
    assert "is_near_splice_junction_10" in columns_to_explode

def mock_plot_histogram_of_nearby_mutations_7_5(mutation_metadata_df_exploded, column, bins, output_file):
    pass

# Use a pytest fixture to patch the function with the mock version
@pytest.fixture
def mock_helpers_visualization(monkeypatch):
    #* notice I use seq_utils and not visualization_utils
    monkeypatch.setattr("varseek.utils.seq_utils.plot_histogram_of_nearby_mutations_7_5", mock_plot_histogram_of_nearby_mutations_7_5)

def test_calculate_nearby_mutations(mock_helpers_visualization):
    # Create a toy DataFrame
    data = {
        "seq_ID": ["ENST0001", "ENST0001", "ENST0001", "ENST0001"],
        "start_mutation_position": [100, 101, 112, 123],
        "end_mutation_position": [105, 101, 112, 123],
        "header": ["ENST0001:c.101_105del", "ENST0001:c.101C>A", "ENST0001:c.112G>C", "ENST0001:c.123A>G"]
    }
    mutation_metadata_df_exploded = pd.DataFrame(data)
    
    # Run the function with the mock data
    output_df, columns_to_explode = calculate_nearby_mutations(
        mcrs_source_column="seq_ID",
        k=10,
        output_plot_folder="mock_folder",
        mcrs_source="not_combined",
        mutation_metadata_df_exploded=mutation_metadata_df_exploded,
        columns_to_explode=None
    )

    # Check that the expected columns are in the output
    assert "nearby_mutations" in output_df.columns
    assert "nearby_mutations_count" in output_df.columns
    assert "has_a_nearby_mutation" in output_df.columns

    # Verify contents of 'nearby_mutations_count' and 'has_a_nearby_mutation'
    expected_nearby_mutations = [['ENST0001:c.101C>A', "ENST0001:c.112G>C"], ["ENST0001:c.101_105del"], ['ENST0001:c.101_105del'], []]
    expected_nearby_mutations = [sorted(sublist) for sublist in expected_nearby_mutations]
    expected_nearby_mutations_count = [2, 1, 1, 0]
    expected_has_a_nearby_mutation = [True, True, True, False]

    # sort output_df["nearby_mutations"]
    output_df["nearby_mutations"] = output_df["nearby_mutations"].apply(lambda x: sorted(x))
    pd.testing.assert_series_equal(
        output_df["nearby_mutations"],
        pd.Series(expected_nearby_mutations, name="nearby_mutations"),
        check_dtype=False
    )
    
    pd.testing.assert_series_equal(
        output_df["nearby_mutations_count"],
        pd.Series(expected_nearby_mutations_count, name="nearby_mutations_count"),
        check_dtype=False
    )

    pd.testing.assert_series_equal(
        output_df["has_a_nearby_mutation"],
        pd.Series(expected_has_a_nearby_mutation, name="has_a_nearby_mutation"),
        check_dtype=False
    )



def test_longest_homopolymer_in_series():
    sequences = pd.Series([
        "AAAAA",         # Single long homopolymer of A
        "ACGTACGT",      # No homopolymer longer than 1
        "GGGGGTT",       # Longest is GGGGG
        np.nan,          # NaN sequence, should return NaN for both outputs
        "CCCAAAAATTT",   # Multiple homopolymers, longest is AAAAA
        "TTTCCCGGG"      # Multiple equal-length homopolymers
    ], name="mcrs_sequence")

    # Apply the function and unpack results into two new columns
    sequences_df = pd.DataFrame({"mcrs_sequence": sequences})
    sequences_df["longest_homopolymer_length"], sequences_df["longest_homopolymer"] = zip(
        *sequences_df["mcrs_sequence"].apply(lambda x: longest_homopolymer(x) if pd.notna(x) else (np.nan, np.nan))
    )

    # Expected results
    expected_lengths = [5, 1, 5, np.nan, 5, 3]
    expected_homopolymers = ["AAAAA", "A", "GGGGG", np.nan, "AAAAA", ["CCC", "GGG", "TTT"]]

    # Test each column separately
    pd.testing.assert_series_equal(
        sequences_df["longest_homopolymer_length"],
        pd.Series(expected_lengths, name="longest_homopolymer_length"),
        check_dtype=False
    )

    # # For homopolymer values, handle cases with lists
    # for i, expected in enumerate(expected_homopolymers):
    #     if isinstance(expected, list):
    #         # Sort lists for consistent comparison
    #         assert sorted(sequences_df["longest_homopolymer"].iloc[i]) == sorted(expected)
    #     else:
    #         # Direct comparison for single homopolymer strings or NaN
    #         assert sequences_df["longest_homopolymer"].iloc[i] == expected


def test_triplet_stats_in_series():
    # Create a Series with test sequences
    sequences = pd.Series([
        "AAATTTCCCGGG",  # 10 distinct triplets, 10 total triplets, complexity = 1.0
        "AAAAAA",        # 1 distinct triplet (AAA), 4 total triplets, complexity = 0.25
        "ATCGATCG",      # 4 distinct triplets, 6 total triplets, complexity = 0.67
        np.nan,          # NaN sequence, should return NaN for all outputs
        "ATATAT",        # 2 distinct triplets (ATA, TAT), 4 total triplets, complexity = 0.5
    ], name="mcrs_sequence")

    # Apply the function and unpack results into three new columns
    sequences_df = pd.DataFrame({"mcrs_sequence": sequences})
    sequences_df["num_distinct_triplets"], sequences_df["num_total_triplets"], sequences_df["triplet_complexity"] = zip(
        *sequences_df["mcrs_sequence"].apply(lambda x: triplet_stats(x) if pd.notna(x) else (np.nan, np.nan, np.nan))
    )

    # Expected results
    expected_num_distinct_triplets = [10, 1, 4, np.nan, 2]
    expected_num_total_triplets = [10, 4, 6, np.nan, 4]
    expected_triplet_complexity = [1.0, 0.25, 0.67, np.nan, 0.5]

    # Test each column separately
    pd.testing.assert_series_equal(
        sequences_df["num_distinct_triplets"],
        pd.Series(expected_num_distinct_triplets, name="num_distinct_triplets"),
        check_dtype=False
    )

    pd.testing.assert_series_equal(
        sequences_df["num_total_triplets"],
        pd.Series(expected_num_total_triplets, name="num_total_triplets"),
        check_dtype=False
    )

    pd.testing.assert_series_equal(
        sequences_df["triplet_complexity"],
        pd.Series(expected_triplet_complexity, name="triplet_complexity"),
        check_dtype=False,
        atol=0.01  # Allows a small tolerance for floating point comparison
    )

def test_count_kmer_overlaps(mock_helpers):
    # Create a temporary FASTA file for testing
    sequences_list = ["ATCGATCGATCG", "GCTAGCTAGCTA", "GATCTTTGCTA", "TTTTTTTTTT"]
    with tempfile.NamedTemporaryFile("w+", suffix=".fasta") as fasta_file:
        for i in range(len(sequences_list)):
            # write normally
            fasta_file.write(f">seq_{i}\n{sequences_list[i]}\n")
        fasta_file.seek(0)

        # Run the function with the temporary FASTA file (strandedness=False)
        df = count_kmer_overlaps(fasta_file.name, k=4, strandedness=True, mcrs_id_column="mcrs_id")

        # Expected results
        expected_data = {
            "mcrs_id": ["seq_0", "seq_1", "seq_2", "seq_3"],
            "number_of_kmers_with_overlap_to_other_mcrs_items_in_mcrs_reference": [2, 3, 2, 0],
            "number_of_mcrs_items_with_overlapping_kmers_in_mcrs_reference": [1, 1, 2, 0],
            "overlapping_kmers": [["GATC", "GATC"], ["GCTA", "GCTA", "GCTA"], ["GATC", "GCTA"], []],
            "mcrs_items_with_overlapping_kmers_in_mcrs_reference": [{"seq_2"}, {"seq_2"}, {"seq_0", "seq_1"}, set()],
        }
        expected_df = pd.DataFrame(expected_data)

        # set columns in same order
        df = df[expected_df.columns]

        # Assertions: Check that the columns and values match the expected DataFrame
        pd.testing.assert_frame_equal(df, expected_df)


def test_add_mcrs_mutation_type(mock_helpers):
    # Create a toy DataFrame
    data = {
        "mcrs_header": [
            "ENST2:c.1211_1212insAAG",                             # Single insertion mutation
            "ENST1:c.101A>G;ENST1:c.101A>G;ENST11:c.108A>G",       # Mixed mutations with substitution duplicates
            "ENST1:c.101A>G;ENST2:c.1211_1212insAAG",              # Mixed insertion and substitution
            "ENST1:c.101_102delinsAG",                              # Mixed deletion and insertion
            "ENST1:c.101dup",                                       # Duplication mutation
            "ENST1:c.101A>G",                                       # Single substitution mutation
            np.nan                                                 # NaN mutation entry
        ]
    }
    mutation_metadata_df = pd.DataFrame(data)

    # Run the function
    result_df = add_mcrs_mutation_type(mutation_metadata_df, mut_column="mcrs_header")

    # Expected results
    expected_mutation_type = ["insertion", "substitution", "mixed", "delins", "duplication", "substitution", np.nan]
    expected_columns = ["mcrs_header", "mcrs_mutation_type"]

    # Assertions
    # Check that the output has the expected columns
    assert all(col in result_df.columns for col in expected_columns)

    # Check the values in the mcrs_mutation_type column
    pd.testing.assert_series_equal(
        result_df["mcrs_mutation_type"],
        pd.Series(expected_mutation_type, name="mcrs_mutation_type"),
        check_dtype=False
    )

    # Check that NaN values remain NaN
    assert result_df.loc[result_df["mcrs_header"].isna(), "mcrs_mutation_type"].isna().all()



    

# compare_cdna_and_genome - requires having a toy reference genome and toy reference transcriptome, and it generally just relies on vk build working for both cdna and genome
# align_to_normal_genome_and_build_dlist
# get_mcrss_that_pseudoalign_but_arent_dlisted
# create_df_of_mcrs_to_self_headers - pretty simple logic - bowtie align MCRS's to themselves, go through the SAM file, and consider the reads the substrings and the reference items the superstrings