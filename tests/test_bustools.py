import os
import shutil
import subprocess
import tempfile

import numpy as np
import pandas as pd
import pytest
import anndata as ad

import varseek as vk
from varseek.utils import create_identity_t2g, make_bus_df


@pytest.fixture
def temp_fastq_file(tmp_path):
    fastq_content = (
        "@seq1\n"
        "CCCAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGA\n"
        "+\n"
        "IIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIII\n"
        "@seq2\n"
        "CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC\n"
        "+\n"
        "IIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIII\n"
        "@seq3\n"
        "CCCAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGA\n"
        "+\n"
        "IIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIII\n"
        "@seq4\n"
        "TTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTT\n"
        "+\n"
        "IIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIII\n"
        "@seq5\n"
        "AAGGAAGGAAAAGGAAGGAAAAGGAAGGAAAAGGAAGGAAAAGGAAGGAAAAGGAAGGAAAAGGAAGGAAAAGGAAGGAAAAGGAAGGAAAAGGAAGGAAAAGGAAGGAAAAGGAAGGAAAAGGAAGGAAAAGGAAGGAAAAGGAAGGAA\n"
        "+\n"
        "IIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIII\n"
    )
    fastq_path = tmp_path / "test_sequences.fastq"
    fastq_path.write_text(fastq_content)
    return fastq_path

@pytest.fixture
def temp_fasta_file(tmp_path):
    fasta_content = (
        ">vcrs1\n"
        "AGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGTAT\n"
        ">vcrs2\n"
        "CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCAAGGAAGGAAAAGGAAGGAAAAGGAAGGAAAAGGAAGGAAAAGGAAGGAAAAGGAAGGAAAAGGAAGG\n"
        ">vcrs3\n"
        "ATGCTGACTGACTGCTGACTGCTAGCTGACGTCATCAGTACGTACGATGCTGACTGACTGCTGACTGCTAGCTGACGTCATCAGTACGTACG\n"
        ">vcrs4\n"
        "AAGGAAGGAAAAGGAAGGAAAAGGAAGGAAAAGGAAGGAAAAGGAAGGAAAAGGAAGGAAAAGGAAGGAAA\n"
        ">vcrs5\n"
        "AAGGAAGGAAAAGGAAGGAAAAGGAAGGAAAAGGAAGGAAAAGGAAGGAA\n"
        ">vcrs6\n"
        "AGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGTTT\n"
    )
    fasta_path = tmp_path / "test_sequences.fasta"
    fasta_path.write_text(fasta_content)
    return fasta_path

@pytest.fixture
def temp_index_file(tmp_path):
    # Create a temporary index file path within the tmp_path directory
    return tmp_path / "mutation_index.idx"

@pytest.fixture
def temp_t2g_file(tmp_path):
    # Create a temporary index file path within the tmp_path directory
    return tmp_path / "mutation_t2g.txt"

@pytest.fixture
def temp_kb_count_out_folder(tmp_path):
    # Create a subdirectory within tmp_path if needed
    temp_dir = tmp_path / "temp_data"
    temp_dir.mkdir()  # Create the directory
    return temp_dir

from pdb import set_trace as st


def test_bustools_df_bulk(temp_fastq_file, temp_fasta_file, temp_index_file, temp_t2g_file, temp_kb_count_out_folder, tmp_path_factory):
    k = "31"
    kb_ref_command = ["kb", "ref", "--workflow", "custom", "-t", "2", "-i", str(temp_index_file), "--d-list", "None", "-k", k, str(temp_fasta_file)]
    subprocess.run(kb_ref_command, check=True)
    create_identity_t2g(temp_fasta_file, temp_t2g_file)
    
    kb_count_command = ["kb", "count", "-t", "2", "-k", str(k), "-i", str(temp_index_file), "-g", str(temp_t2g_file), "-x", "bulk", "--strand", "forward", "--num", "--h5ad", "--parity", "single", "-o", str(temp_kb_count_out_folder), str(temp_fastq_file)]
    subprocess.run(kb_count_command, check=True)

    bustools = None
    if not bustools:
        bustools_binary_path_command = "kb info | grep 'bustools:' | awk '{print $3}' | sed 's/[()]//g'"
        bustools = subprocess.run(bustools_binary_path_command, shell=True, executable="/bin/bash", stdout=subprocess.PIPE, text=True, check=True).stdout.strip()

    bus_df = make_bus_df(kallisto_out = temp_kb_count_out_folder, fastq_file_list = temp_fastq_file, t2g_file = temp_t2g_file, mm = False, union = False, technology = "bulk", bustools = bustools)
    read_to_ref_dict = dict(zip(bus_df['fastq_header'], bus_df['gene_names_final']))

    assert read_to_ref_dict == {'seq1': ['vcrs1', 'vcrs6'], 'seq2': ['vcrs2'], 'seq3': ['vcrs1', 'vcrs6'], 'seq5': ['vcrs2', 'vcrs4', 'vcrs5']}

    adata_path = f"{temp_kb_count_out_folder}/counts_unfiltered/adata.h5ad"
    adata = ad.read_h5ad(adata_path)

    assert np.array_equal(adata.X.toarray(), np.array([[0., 1., 0., 0., 0., 0.]]))

    # # manually remove basetemp
    # try:
    #     shutil.rmtree(tmp_path_factory.getbasetemp())
    # except Exception as e:
    #     pass

# @pytest.fixture(scope="session", autouse=True)
# def notify_basetemp_location(tmp_path_factory):
#     # Run all tests first
#     yield

#     # Print the message at the end of the test session
#     basetemp = tmp_path_factory.getbasetemp()
#     print(f"\nRemove temp dir with `rm -rf {basetemp}`")