import os
import tempfile
from pdb import set_trace as st

import pandas as pd
import pytest
from datetime import datetime
from pathlib import Path

import varseek as vk

store_out_in_permanent_paths = False
tests_dir = Path(__file__).resolve().parent
pytest_permanent_out_dir_base = tests_dir / "pytest_output" / Path(__file__).stem
current_datetime = datetime.now().strftime("date_%Y_%m_%d_time_%H%M_%S")

@pytest.fixture
def out_dir(tmp_path, request):
    """Fixture that returns the appropriate output directory for each test."""
    if store_out_in_permanent_paths:
        current_test_function_name = request.node.name
        out = Path(f"{pytest_permanent_out_dir_base}/{current_datetime}/{current_test_function_name}")
    else:
        out = tmp_path / "out_vk_sim"

    out.mkdir(parents=True, exist_ok=True)  # Ensure the directory exists
    return out


@pytest.fixture
def temporary_output_files():
    with tempfile.NamedTemporaryFile(suffix=".csv") as variants_updated_csv_out, \
        tempfile.NamedTemporaryFile(suffix=".csv") as reads_csv_out, \
        tempfile.NamedTemporaryFile(suffix=".fq") as reads_fastq_out:
        
        # Dictionary of temporary paths
        temp_files = {
            "variants_updated_csv_out": variants_updated_csv_out.name,
            "reads_csv_out": reads_csv_out.name,
            "reads_fastq_out": reads_fastq_out.name
        }
        
        yield temp_files  # Provide paths to test

def test_basic_sim(toy_mutation_metadata_df_with_read_parents_path, temporary_output_files, out_dir):
    filters = []

    strand = "f"
    number_of_variants_to_sample = 5
    seed = 42
    read_length = 150
    add_noise = False
    error_rate = 0.01
    max_errors = 0

    variants_updated_csv_out, reads_csv_out, reads_fastq_out = temporary_output_files["variants_updated_csv_out"], temporary_output_files["reads_csv_out"], temporary_output_files["reads_fastq_out"]

    simulated_df_dict_from_test = vk.sim(
        variants = toy_mutation_metadata_df_with_read_parents_path,
        number_of_variants_to_sample=number_of_variants_to_sample,
        number_of_reads_per_variant_alt="all",
        strand=strand,
        read_length=read_length,
        filters=filters,
        add_noise=add_noise,
        error_rate=error_rate,
        max_errors=max_errors,
        with_replacement=False,
        sequences=None,
        seq_id_column="seq_ID",
        var_column="mutation",
        reference_out_dir=None,
        vk_build_out_dir=None,
        out=out_dir,
        reads_fastq_out = reads_fastq_out,
        reads_csv_out=reads_csv_out,
        variants_updated_csv_out=variants_updated_csv_out,
        seed=seed,
    )

    read_df_from_test, mutation_metadata_df_from_test = simulated_df_dict_from_test["read_df"], simulated_df_dict_from_test["variants"]
    
    output_metadata_df_expected = pd.read_csv(toy_mutation_metadata_df_with_read_parents_path)

    for _, row in read_df_from_test.iterrows():
        read_index = row['read_index']
        vcrs_id = row['vcrs_id']
        read_sequence = row['read_sequence']
        read_length = len(read_sequence)

        # Find the row in mutation_metadata_df_from_test that matches vcrs_id
        matching_row = output_metadata_df_expected.loc[output_metadata_df_expected['vcrs_id'] == vcrs_id]
        
        if matching_row.empty:
            raise ValueError(f"No match found for vcrs_id: {vcrs_id}")

        # Extract the vcrs_sequence
        vcrs_sequence = matching_row.iloc[0]['mutant_sequence_read_parent']
        
        # Extract the slice from vcrs_sequence
        vcrs_sequence_slice = vcrs_sequence[read_index : read_index + read_length]

        # Assert that the extracted slice is equal to read_sequence
        assert vcrs_sequence_slice == read_sequence, (
            f"Mismatch for vcrs_id {vcrs_id} at index {read_index}:\n"
            f"Expected: {read_sequence}\nFound: {vcrs_sequence_slice}"
        )