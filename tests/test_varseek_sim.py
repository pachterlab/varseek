import os
import tempfile
from pdb import set_trace as st

import numpy as np
import pandas as pd
import pytest

import varseek as vk
from varseek.utils import (
    compare_dicts,
    create_header_to_sequence_ordered_dict_from_fasta_WITHOUT_semicolon_splitting,
    make_mapping_dict,
)


@pytest.fixture
def temporary_output_files():
    with tempfile.NamedTemporaryFile(suffix=".csv") as mutation_metadata_df_out, \
        tempfile.NamedTemporaryFile(suffix=".csv") as read_df_out, \
        tempfile.NamedTemporaryFile(suffix=".fq") as fastq_output_path:
        
        # Dictionary of temporary paths
        temp_files = {
            "mutation_metadata_df_out": mutation_metadata_df_out.name,
            "read_df_out": read_df_out.name,
            "fastq_output_path": fastq_output_path.name
        }
        
        yield temp_files  # Provide paths to test

def test_basic_sim(toy_mutation_metadata_df_with_read_parents_path, temporary_output_files):
    filters = []

    strand = "f"
    number_of_mutations_to_sample = 5
    seed = 42
    read_length = 150
    add_noise = False
    error_rate = 0.01
    max_errors = 0

    mutation_metadata_df_out, read_df_out, fastq_output_path = temporary_output_files["mutation_metadata_df_out"], temporary_output_files["read_df_out"], temporary_output_files["fastq_output_path"]

    simulated_df_dict_from_test = vk.sim(
        mutation_metadata_df = toy_mutation_metadata_df_with_read_parents_path,
        fastq_output_path = fastq_output_path,
        sample_type="m",
        number_of_mutations_to_sample=number_of_mutations_to_sample,
        strand=strand,
        number_of_reads_per_sample="all",  # not used when number_of_reads_per_sample_m and number_of_reads_per_sample_w are provided
        read_length=read_length,
        seed=seed,
        add_noise=add_noise,
        error_rate=error_rate,
        max_errors=max_errors,
        with_replacement=False,
        sequences=None,
        mutation_metadata_df_path=None,
        seq_id_column="seq_ID",
        mut_column="mutation",
        reference_out_dir=None,
        out_dir_vk_build=None,
        filters=filters,
        read_df_out=read_df_out,
        mutation_metadata_df_out=mutation_metadata_df_out,
    )

    read_df_from_test, mutation_metadata_df_from_test = simulated_df_dict_from_test["read_df"], simulated_df_dict_from_test["mutation_metadata_df"]
    
    output_metadata_df_expected = pd.read_csv(toy_mutation_metadata_df_with_read_parents_path)

    for _, row in read_df_from_test.iterrows():
        read_index = row['read_index']
        mcrs_id = row['mcrs_id']
        read_sequence = row['read_sequence']
        read_length = len(read_sequence)

        # Find the row in mutation_metadata_df_from_test that matches mcrs_id
        matching_row = output_metadata_df_expected.loc[output_metadata_df_expected['mcrs_id'] == mcrs_id]
        
        if matching_row.empty:
            raise ValueError(f"No match found for mcrs_id: {mcrs_id}")

        # Extract the mcrs_sequence
        mcrs_sequence = matching_row.iloc[0]['mutant_sequence_read_parent']
        
        # Extract the slice from mcrs_sequence
        mcrs_sequence_slice = mcrs_sequence[read_index : read_index + read_length]

        # Assert that the extracted slice is equal to read_sequence
        assert mcrs_sequence_slice == read_sequence, (
            f"Mismatch for mcrs_id {mcrs_id} at index {read_index}:\n"
            f"Expected: {read_sequence}\nFound: {mcrs_sequence_slice}"
        )