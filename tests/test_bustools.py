import os
import shutil
import subprocess
import tempfile

import numpy as np
import pandas as pd
import pytest
import anndata as ad
import sys

import varseek as vk
from varseek.utils import create_identity_t2g, make_bus_df

#$ TOGGLE THIS SECTION TO HAVE THIS FILE RECOGNIZED BY PYTEST (commented out means it will be recognized, uncommented means it will be hidden)
# # If "tests/test_bustools.py" is not explicitly in the command line arguments, skip this module.
# if not any("tests/test_bustools.py" in arg for arg in sys.argv):
#     pytest.skip("Skipping test_bustools.py due issues with kallisto compiling in some environments (e.g., GitHub actions, some MacOS systems); run this file by explicity including the file i.e., 'pytest tests/test_bustools.py'", allow_module_level=True)


@pytest.fixture
def temp_fastq_file(tmp_path):
    fastq_content = (
        "@seq1_1\n"
        "CCCAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGA\n"
        "+\n"
        "IIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIII\n"
        "@seq2_1\n"
        "CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC\n"
        "+\n"
        "IIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIII\n"
        "@seq3_1\n"
        "CCCAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGA\n"
        "+\n"
        "IIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIII\n"
        "@seq4_1\n"
        "TTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTT\n"
        "+\n"
        "IIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIII\n"
        "@seq5_1\n"
        "AAGGAAGGAAAAGGAAGGAAAAGGAAGGAAAAGGAAGGAAAAGGAAGGAAAAGGAAGGAAAAGGAAGGAAAAGGAAGGAAAAGGAAGGAAAAGGAAGGAAAAGGAAGGAAAAGGAAGGAAAAGGAAGGAAAAGGAAGGAAAAGGAAGGAA\n"
        "+\n"
        "IIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIII\n"
    )
    fastq_path = tmp_path / "test_sequences.fastq"
    fastq_path.write_text(fastq_content)
    return fastq_path


@pytest.fixture
def temp_fastq_file_pair(tmp_path):
    fastq_content = (
        "@seq1_2\n"
        "GAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGTATTCTCGACTACGACACGACACGACACGACACGACACGACACGACACGACACGACACGAC\n"
        "+\n"
        "IIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIII\n"
        "@seq2_2\n"
        "CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC\n"
        "+\n"
        "IIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIII\n"
        "@seq3_2\n"
        "AAGGAAGGAAAAGGAAGGAAAAGGAAGGAAAAGGAAGGAAAAGGAAGGAAAAGGAAGGAAAAGGAAGGAAAAGGAAGGAAAAGGAAGGAAAAGGAAGGAAAAGGAAGGAAAAGGAAGGAAAAGGAAGGAAAAGGAAGGAAAAGGAAGGAA\n"
        "+\n"
        "IIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIII\n"
        "@seq4_2\n"
        "TTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTT\n"
        "+\n"
        "IIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIII\n"
        "@seq5_2\n"
        "AAGGAAGGAAAAGGAAGGAAAAGGAAGGAAAAGGAAGGAAAAGGAAGGAAAAGGAAGGAAAAGGAAGGAAAAGGAAGGAAAAGGAAGGAAAAGGAAGGAAAAGGAAGGAAAAGGAAGGAAAAGGAAGGAAAAGGAAGGAAAAGGAAGGAA\n"
        "+\n"
        "IIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIII\n"
    )
    fastq_path = tmp_path / "test_sequences_paired.fastq"
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



def test_bustools_df_bulk_parity_single(temp_fastq_file, temp_fasta_file, temp_index_file, temp_t2g_file, temp_kb_count_out_folder):
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

    bus_df = make_bus_df(kb_count_out = temp_kb_count_out_folder, fastq_file_list = temp_fastq_file, t2g_file = temp_t2g_file, mm = False, technology = "bulk", bustools = bustools, fastq_sorting_check_only=True)
    read_to_ref_dict = dict(zip(bus_df['fastq_header'], bus_df['gene_names']))

    assert read_to_ref_dict == {'seq1_1': ['vcrs1', 'vcrs6'], 'seq2_1': ['vcrs2'], 'seq3_1': ['vcrs1', 'vcrs6'], 'seq5_1': ['vcrs2', 'vcrs4', 'vcrs5']}

    adata_path = f"{temp_kb_count_out_folder}/counts_unfiltered/adata.h5ad"
    adata = ad.read_h5ad(adata_path)

    assert np.array_equal(adata.X.toarray(), np.array([[0., 1., 0., 0., 0., 0.]]))

def test_bustools_df_bulk_parity_paired(temp_fastq_file, temp_fastq_file_pair, temp_fasta_file, temp_index_file, temp_t2g_file, temp_kb_count_out_folder):
    k = "31"
    kb_ref_command = ["kb", "ref", "--workflow", "custom", "-t", "2", "-i", str(temp_index_file), "--d-list", "None", "-k", k, str(temp_fasta_file)]
    subprocess.run(kb_ref_command, check=True)
    create_identity_t2g(temp_fasta_file, temp_t2g_file)
    
    kb_count_command = ["kb", "count", "-t", "2", "-k", str(k), "-i", str(temp_index_file), "-g", str(temp_t2g_file), "-x", "BULK", "--num", "--h5ad", "--parity", "paired", "-o", str(temp_kb_count_out_folder), str(temp_fastq_file), str(temp_fastq_file_pair)]  # note: "--strand forward" will cause an issue
    subprocess.run(kb_count_command, check=True)

    bustools = None
    if not bustools:
        bustools_binary_path_command = "kb info | grep 'bustools:' | awk '{print $3}' | sed 's/[()]//g'"
        bustools = subprocess.run(bustools_binary_path_command, shell=True, executable="/bin/bash", stdout=subprocess.PIPE, text=True, check=True).stdout.strip()

    bus_df = make_bus_df(kb_count_out = temp_kb_count_out_folder, fastq_file_list = [temp_fastq_file, temp_fastq_file_pair], t2g_file = temp_t2g_file, mm = False, technology = "bulk", bustools = bustools, fastq_sorting_check_only=True)
    read_to_ref_dict = dict(zip(bus_df['fastq_header'], bus_df['gene_names']))

    assert read_to_ref_dict == {'seq1_1': ['vcrs1'], 'seq2_1': ['vcrs2'], 'seq5_1': ['vcrs2', 'vcrs4', 'vcrs5']}

    adata_path = f"{temp_kb_count_out_folder}/counts_unfiltered/adata.h5ad"
    adata = ad.read_h5ad(adata_path)

    assert np.array_equal(adata.X.toarray(), np.array([[1., 1., 0., 0., 0., 0.]]))







@pytest.fixture
def temp_fastq_R1_complex(tmp_path):
    fastq_content = (
        "@read0_mapsto_vcrs1_R1\n"
        "AAACCCAAGAAACACTTTTTTTTTTTTT\n"
        "+\n"
        "IIIIIIIIIIIIIIIIIIIIIIIIIIII\n"
        "@read1_mapsto_vcrs1_same_barcode_and_umi_R1\n"
        "AAACCCAAGAAACACTTTTTTTTTTTTT\n"
        "+\n"
        "IIIIIIIIIIIIIIIIIIIIIIIIIIII\n"
        "@read2_mapsto_vcrs1_different_barcode_R1\n"
        "TATCAGGAGCTAAGTGTTTTTTTTTTTT\n"
        "+\n"
        "IIIIIIIIIIIIIIIIIIIIIIIIIIII\n"
        "@read3_mapsto_vcrs2_same_barcode_R1\n"
        "TATCAGGAGCTAAGTGTTTTTTTTTTTT\n"
        "+\n"
        "IIIIIIIIIIIIIIIIIIIIIIIIIIII\n"
        "@read4_mapsto_vcrs2_different_umi_R1\n"
        "TATCAGGAGCTAAGTGTTTTTTTTTTTA\n"
        "+\n"
        "IIIIIIIIIIIIIIIIIIIIIIIIIIII\n"
        "@read5_mapsto_vcrs1_and_vcrs2_union_R1\n"
        "TATCAGGAGCTAAGTGTTTTTTTTTTTG\n"
        "+\n"
        "IIIIIIIIIIIIIIIIIIIIIIIIIIII\n"
        "@read6_mapsto_vcrs3_and_vcrs4_and_vcrs5_multimap_R1\n"
        "TATCAGGAGCTAAGTGTTTTTTTTTTCC\n"
        "+\n"
        "IIIIIIIIIIIIIIIIIIIIIIIIIIII\n"
        "@read7_mapsto_vcrs3_and_vcrs4_and_vcrs5_multimap_different_umi_R1\n"
        "TATCAGGAGCTAAGTGTTTTTTTTTAAA\n"
        "+\n"
        "IIIIIIIIIIIIIIIIIIIIIIIIIIII\n"
        "@read8_mapsto_vcrs1_barcode1_but_hamming_distance1_R1\n"
        "GAACCCAAGAAACACTTTTTTTTTAACA\n"
        "+\n"
        "IIIIIIIIIIIIIIIIIIIIIIIIIIII\n"
        "@read9_mapsto_vcrs1_and_vcrs2_barcode1_same_umi_as_read10_R1\n"
        "AAACCCAAGAAACACTTTTTTTAAAAAA\n"
        "+\n"
        "IIIIIIIIIIIIIIIIIIIIIIIIIIII\n"
        "@read10_mapsto_vcrs1_and_vcrs6_barcode1_same_umi_as_read9_R1\n"
        "AAACCCAAGAAACACTTTTTTTAAAAAA\n"
        "+\n"
        "IIIIIIIIIIIIIIIIIIIIIIIIIIII\n"
    )
    fastq_path = tmp_path / "test_sequences_R1.fastq"
    fastq_path.write_text(fastq_content)
    return fastq_path


@pytest.fixture
def temp_fastq_R2_complex(tmp_path):
    fastq_content = (
        "@read0_mapsto_vcrs1_R2\n"
        "CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC\n"
        "+\n"
        "IIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIII\n"
        "@read1_mapsto_vcrs1_same_barcode_and_umi_R2\n"
        "CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC\n"
        "+\n"
        "IIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIII\n"
        "@read2_mapsto_vcrs1_different_barcode_R2\n"
        "CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC\n"
        "+\n"
        "IIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIII\n"
        "@read3_mapsto_vcrs2_same_barcode_R2\n"
        "AGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGA\n"
        "+\n"
        "IIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIII\n"
        "@read4_mapsto_vcrs2_different_umi_R2\n"
        "AGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGA\n"
        "+\n"
        "IIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIII\n"
        "@read5_mapsto_vcrs1_and_vcrs2_union_R2\n"
        "CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGA\n"
        "+\n"
        "IIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIII\n"
        "@read6_mapsto_vcrs3_and_vcrs4_and_vcrs5_multimap_R2\n"
        "CCACCACCACCACCACCACCACCACCACCACCACCACCACCACCACCACCACCACCACCACCACCACCACCACCACCACCACCACCACCACCACCACCACCACCACCACCACCACCACCACCACCACCACCACCACCACCACCACCACCA\n"
        "+\n"
        "IIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIII\n"
        "@read7_mapsto_vcrs3_and_vcrs4_and_vcrs5_multimap_different_umi_R2\n"
        "CCACCACCACCACCACCACCACCACCACCACCACCACCACCACCACCACCACCACCACCACCACCACCACCACCACCACCACCACCACCACCACCACCACCACCACCACCACCACCACCACCACCACCACCACCACCACCACCACCACCA\n"
        "+\n"
        "IIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIII\n"
        "@read8_mapsto_vcrs1_barcode1_but_hamming_distance2_R2\n"
        "CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC\n"
        "+\n"
        "IIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIII\n"
        "@read9_mapsto_vcrs1_and_vcrs2_barcode1_same_umi_as_read10_R2\n"
        "CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGA\n"
        "+\n"
        "IIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIII\n"
        "@read10_mapsto_vcrs1_and_vcrs6_barcode1_same_umi_as_read9_R2\n"
        "CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCGATGATGATGATGATGATGATGATGATGATGATGATGATGATGATGATGATGATGATGATGATGATGATGATGATGATGATGATGA\n"
        "+\n"
        "IIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIII\n"
    )
    fastq_path = tmp_path / "test_sequences_paired.fastq"
    fastq_path.write_text(fastq_content)
    return fastq_path

@pytest.fixture
def temp_fasta_file_complex(tmp_path):
    fasta_content = (
        ">vcrs1\n"
        "CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC\n"
        ">vcrs2\n"
        "AGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGA\n"
        ">vcrs3\n"
        "CCACCACCACCACCACCACCACCACCACCACCACCACCACCACCACCACCACCACCACCACCACCACCACCACCACCACCACCACCACCACCACCACCACCAG\n"
        ">vcrs4\n"
        "CCACCACCACCACCACCACCACCACCACCACCACCACCACCACCACCACCACCACCACCACCACCACCACCACCACCACCACCACCACCACCACCACCACCAT\n"
        ">vcrs5\n"
        "CCACCACCACCACCACCACCACCACCACCACCACCACCACCACCACCACCACCACCACCACCACCACCACCACCACCACCACCACCACCACCACCACCACCAA\n"
        ">vcrs6\n"
        "GATGATGATGATGATGATGATGATGATGATGATGATGATGATGATGATGATGATGATGATGATGATGATGATGATGATGATGATGATGATGATGATGATGATG\n"
    )
    fasta_path = tmp_path / "test_sequences.fasta"
    fasta_path.write_text(fasta_content)
    return fasta_path



def test_bustools_df_10xv3(temp_fastq_R1_complex, temp_fastq_R2_complex, temp_fasta_file_complex, temp_index_file, temp_t2g_file, temp_kb_count_out_folder):
    mm = False
    union = False

    k = "31"
    kb_ref_command = ["kb", "ref", "--workflow", "custom", "-t", "2", "-i", str(temp_index_file), "--d-list", "None", "-k", k, str(temp_fasta_file_complex)]
    subprocess.run(kb_ref_command, check=True)
    create_identity_t2g(temp_fasta_file_complex, temp_t2g_file)
    
    kb_count_command = ["kb", "count", "-t", "2", "-k", str(k), "-i", str(temp_index_file), "-g", str(temp_t2g_file), "-x", "10XV3", "--num", "--h5ad", "-o", str(temp_kb_count_out_folder), str(temp_fastq_R1_complex), str(temp_fastq_R2_complex)]
    if mm:
        kb_count_command.insert(2, "--mm")
    if union:
        kb_count_command.insert(2, "--union")
    subprocess.run(kb_count_command, check=True)

    bustools = None
    if not bustools:
        bustools_binary_path_command = "kb info | grep 'bustools:' | awk '{print $3}' | sed 's/[()]//g'"
        bustools = subprocess.run(bustools_binary_path_command, shell=True, executable="/bin/bash", stdout=subprocess.PIPE, text=True, check=True).stdout.strip()

    bus_df = make_bus_df(kb_count_out = temp_kb_count_out_folder, fastq_file_list = [temp_fastq_R1_complex, temp_fastq_R2_complex], t2g_file = temp_t2g_file, mm = mm, technology = "10XV3", bustools = bustools, fastq_sorting_check_only=True)
    read_to_ref_dict = dict(zip(bus_df['fastq_header'], bus_df['gene_names']))

    adata_path = f"{temp_kb_count_out_folder}/counts_unfiltered/adata.h5ad"
    adata = ad.read_h5ad(adata_path)

    # Convert adata.X to a DataFrame (if not already)
    matrix_df = pd.DataFrame(adata.X.toarray() if hasattr(adata.X, "toarray") else adata.X, 
                    index=adata.obs.index, 
                    columns=adata.var.index)

    read_to_ref_dict_gt = {
        'read0_mapsto_vcrs1_R2': ['vcrs1'],  # count_matrix_data_gt["AAACCCAAGAAACACT"]["vcrs1"] = 1
        'read1_mapsto_vcrs1_same_barcode_and_umi_R2': ['vcrs1'],  # because it has duplicate UMI as read0, it doesn't count for count matrix
        'read2_mapsto_vcrs1_different_barcode_R2': ['vcrs1'],  # count_matrix_data_gt["TATCAGGAGCTAAGTG"]["vcrs1"] = 1
        'read3_mapsto_vcrs2_same_barcode_R2': ['vcrs2'],  # count_matrix_data_gt["TATCAGGAGCTAAGTG"]["vcrs2"] = 1
        'read4_mapsto_vcrs2_different_umi_R2': ['vcrs2'],  # count_matrix_data_gt["TATCAGGAGCTAAGTG"]["vcrs2"] = 2
        'read5_mapsto_vcrs1_and_vcrs2_union_R2': [],  # doesn't count for count matrix OR show up in bus file unless --union is used
        'read6_mapsto_vcrs3_and_vcrs4_and_vcrs5_multimap_R2': ['vcrs3', 'vcrs4', 'vcrs5'],  # doesn't count for count matrix unless --mm is used (but shows up in bus file regardless)
        'read7_mapsto_vcrs3_and_vcrs4_and_vcrs5_multimap_different_umi_R2': ['vcrs3', 'vcrs4', 'vcrs5'],  # doesn't count for count matrix unless --mm is used (but shows up in bus file regardless)
        'read8_mapsto_vcrs1_barcode1_but_hamming_distance2_R2': ['vcrs1'],  # count_matrix_data_gt["AAACCCAAGAAACACT"]["vcrs1"] = 2
        'read9_mapsto_vcrs1_and_vcrs2_barcode1_same_umi_as_read10_R2': [],  # doesn't count for VCRS1 without union (won't show up in BUS file without union); VCRS2 doesn't count regardless because it has the same UMI as read10, but read10 doesn't map to VCRS2 
        'read10_mapsto_vcrs1_and_vcrs6_barcode1_same_umi_as_read9_R2': [],  # doesn't count due to same barcode and UMI as read9
    }
    if union:
        read_to_ref_dict_gt['read5_mapsto_vcrs1_and_vcrs2_union_R2'] = ['vcrs1', 'vcrs2']
        read_to_ref_dict_gt['read9_mapsto_vcrs1_and_vcrs2_barcode1_same_umi_as_read10_R2'] = ['vcrs1', 'vcrs2']
        read_to_ref_dict_gt['read10_mapsto_vcrs1_and_vcrs6_barcode1_same_umi_as_read9_R2'] = ['vcrs1', 'vcrs6']

    read_to_ref_dict_gt = {k: v for k, v in read_to_ref_dict_gt.items() if v != []}  # remove empty keys


    count_matrix_data_gt = {
        "AAACCCAAGAAACACT": {"vcrs1": 2, "vcrs2": 0, "vcrs3": 0, "vcrs4": 0, "vcrs5": 0, "vcrs6": 0},
        "TATCAGGAGCTAAGTG": {"vcrs1": 1, "vcrs2": 2, "vcrs3": 0, "vcrs4": 0, "vcrs5": 0, "vcrs6": 0},
    }
    if union and mm:  # notably, won't show up in count matrix unless mm is also used
        count_matrix_data_gt["AAACCCAAGAAACACT"]["vcrs1"] += (1/1)  # each unioned read adds (1/n), where n is the number of VCRSs to which the read maps - this comes from read9/10
        count_matrix_data_gt["TATCAGGAGCTAAGTG"]["vcrs1"] += (1/2)  # each unioned read adds (1/n), where n is the number of VCRSs to which the read maps - this comes from read5
        count_matrix_data_gt["TATCAGGAGCTAAGTG"]["vcrs2"] += (1/2)  # each unioned read adds (1/n), where n is the number of VCRSs to which the read maps - this comes from read5
    if mm:
        count_matrix_data_gt["TATCAGGAGCTAAGTG"]["vcrs3"] += (1/3) * 2  # each multimapped read adds (1/n), where n is the number of VCRSs to which the read maps, and reads 6 and 7 are both multimappers hence the *2
        count_matrix_data_gt["TATCAGGAGCTAAGTG"]["vcrs4"] += (1/3) * 2  # each multimapped read adds (1/n), where n is the number of VCRSs to which the read maps, and reads 6 and 7 are both multimappers hence the *2
        count_matrix_data_gt["TATCAGGAGCTAAGTG"]["vcrs5"] += (1/3) * 2  # each multimapped read adds (1/n), where n is the number of VCRSs to which the read maps, and reads 6 and 7 are both multimappers hence the *2
    # count_matrix_data_gt_with_multimap = {
    #     "AAACCCAAGAAACACT": {"vcrs1": 3, "vcrs2": 0, "vcrs3": 0, "vcrs4": 0, "vcrs5": 0, "vcrs6": 0},
    #     "TATCAGGAGCTAAGTG": {"vcrs1": 1.5, "vcrs2": 2.5, "vcrs3": 0.67, "vcrs4": 0.67, "vcrs5": 0.67, "vcrs6": 0},
    # }

    matrix_df_gt = pd.DataFrame(count_matrix_data_gt)
    matrix_df_gt = matrix_df_gt.T
    matrix_df_gt = matrix_df_gt.astype("float64")
    
    # ensure no rounding errors (eg 0.6667 vs 0.66666666...)
    matrix_df = matrix_df.round(3)
    matrix_df_gt = matrix_df_gt.round(3)

    assert read_to_ref_dict == read_to_ref_dict_gt  # from BUS file
    assert matrix_df.equals(matrix_df_gt)  # from adata

def test_bustools_df_10xv3_MM(temp_fastq_R1_complex, temp_fastq_R2_complex, temp_fasta_file_complex, temp_index_file, temp_t2g_file, temp_kb_count_out_folder):
    mm = True
    union = False
    
    k = "31"
    kb_ref_command = ["kb", "ref", "--workflow", "custom", "-t", "2", "-i", str(temp_index_file), "--d-list", "None", "-k", k, str(temp_fasta_file_complex)]
    subprocess.run(kb_ref_command, check=True)
    create_identity_t2g(temp_fasta_file_complex, temp_t2g_file)
    
    kb_count_command = ["kb", "count", "-t", "2", "-k", str(k), "-i", str(temp_index_file), "-g", str(temp_t2g_file), "-x", "10XV3", "--num", "--h5ad", "-o", str(temp_kb_count_out_folder), str(temp_fastq_R1_complex), str(temp_fastq_R2_complex)]
    if mm:
        kb_count_command.insert(2, "--mm")
    if union:
        kb_count_command.insert(2, "--union")
    subprocess.run(kb_count_command, check=True)

    bustools = None
    if not bustools:
        bustools_binary_path_command = "kb info | grep 'bustools:' | awk '{print $3}' | sed 's/[()]//g'"
        bustools = subprocess.run(bustools_binary_path_command, shell=True, executable="/bin/bash", stdout=subprocess.PIPE, text=True, check=True).stdout.strip()

    bus_df = make_bus_df(kb_count_out = temp_kb_count_out_folder, fastq_file_list = [temp_fastq_R1_complex, temp_fastq_R2_complex], t2g_file = temp_t2g_file, mm = mm, technology = "10XV3", bustools = bustools, fastq_sorting_check_only=True)
    read_to_ref_dict = dict(zip(bus_df['fastq_header'], bus_df['gene_names']))

    adata_path = f"{temp_kb_count_out_folder}/counts_unfiltered/adata.h5ad"
    adata = ad.read_h5ad(adata_path)

    # Convert adata.X to a DataFrame (if not already)
    matrix_df = pd.DataFrame(adata.X.toarray() if hasattr(adata.X, "toarray") else adata.X, 
                    index=adata.obs.index, 
                    columns=adata.var.index)

    read_to_ref_dict_gt = {
        'read0_mapsto_vcrs1_R2': ['vcrs1'],  # count_matrix_data_gt["AAACCCAAGAAACACT"]["vcrs1"] = 1
        'read1_mapsto_vcrs1_same_barcode_and_umi_R2': ['vcrs1'],  # because it has duplicate UMI as read0, it doesn't count for count matrix
        'read2_mapsto_vcrs1_different_barcode_R2': ['vcrs1'],  # count_matrix_data_gt["TATCAGGAGCTAAGTG"]["vcrs1"] = 1
        'read3_mapsto_vcrs2_same_barcode_R2': ['vcrs2'],  # count_matrix_data_gt["TATCAGGAGCTAAGTG"]["vcrs2"] = 1
        'read4_mapsto_vcrs2_different_umi_R2': ['vcrs2'],  # count_matrix_data_gt["TATCAGGAGCTAAGTG"]["vcrs2"] = 2
        'read5_mapsto_vcrs1_and_vcrs2_union_R2': [],  # doesn't count for count matrix OR show up in bus file unless --union is used
        'read6_mapsto_vcrs3_and_vcrs4_and_vcrs5_multimap_R2': ['vcrs3', 'vcrs4', 'vcrs5'],  # doesn't count for count matrix unless --mm is used (but shows up in bus file regardless)
        'read7_mapsto_vcrs3_and_vcrs4_and_vcrs5_multimap_different_umi_R2': ['vcrs3', 'vcrs4', 'vcrs5'],  # doesn't count for count matrix unless --mm is used (but shows up in bus file regardless)
        'read8_mapsto_vcrs1_barcode1_but_hamming_distance2_R2': ['vcrs1'],  # count_matrix_data_gt["AAACCCAAGAAACACT"]["vcrs1"] = 2
        'read9_mapsto_vcrs1_and_vcrs2_barcode1_same_umi_as_read10_R2': [],  # doesn't count for VCRS1 without union (won't show up in BUS file without union); VCRS2 doesn't count regardless because it has the same UMI as read10, but read10 doesn't map to VCRS2 
        'read10_mapsto_vcrs1_and_vcrs6_barcode1_same_umi_as_read9_R2': [],  # doesn't count due to same barcode and UMI as read9
    }
    if union:
        read_to_ref_dict_gt['read5_mapsto_vcrs1_and_vcrs2_union_R2'] = ['vcrs1', 'vcrs2']
        read_to_ref_dict_gt['read9_mapsto_vcrs1_and_vcrs2_barcode1_same_umi_as_read10_R2'] = ['vcrs1', 'vcrs2']
        read_to_ref_dict_gt['read10_mapsto_vcrs1_and_vcrs6_barcode1_same_umi_as_read9_R2'] = ['vcrs1', 'vcrs6']

    read_to_ref_dict_gt = {k: v for k, v in read_to_ref_dict_gt.items() if v != []}  # remove empty keys

    count_matrix_data_gt = {
        "AAACCCAAGAAACACT": {"vcrs1": 2, "vcrs2": 0, "vcrs3": 0, "vcrs4": 0, "vcrs5": 0, "vcrs6": 0},
        "TATCAGGAGCTAAGTG": {"vcrs1": 1, "vcrs2": 2, "vcrs3": 0, "vcrs4": 0, "vcrs5": 0, "vcrs6": 0},
    }
    if union and mm:  # notably, won't show up in count matrix unless mm is also used
        count_matrix_data_gt["AAACCCAAGAAACACT"]["vcrs1"] += (1/1)  # each unioned read adds (1/n), where n is the number of VCRSs to which the read maps - this comes from read9/10
        count_matrix_data_gt["TATCAGGAGCTAAGTG"]["vcrs1"] += (1/2)  # each unioned read adds (1/n), where n is the number of VCRSs to which the read maps - this comes from read5
        count_matrix_data_gt["TATCAGGAGCTAAGTG"]["vcrs2"] += (1/2)  # each unioned read adds (1/n), where n is the number of VCRSs to which the read maps - this comes from read5
    if mm:
        count_matrix_data_gt["TATCAGGAGCTAAGTG"]["vcrs3"] += (1/3) * 2  # each multimapped read adds (1/n), where n is the number of VCRSs to which the read maps, and reads 6 and 7 are both multimappers hence the *2
        count_matrix_data_gt["TATCAGGAGCTAAGTG"]["vcrs4"] += (1/3) * 2  # each multimapped read adds (1/n), where n is the number of VCRSs to which the read maps, and reads 6 and 7 are both multimappers hence the *2
        count_matrix_data_gt["TATCAGGAGCTAAGTG"]["vcrs5"] += (1/3) * 2  # each multimapped read adds (1/n), where n is the number of VCRSs to which the read maps, and reads 6 and 7 are both multimappers hence the *2
    # count_matrix_data_gt_with_multimap = {
    #     "AAACCCAAGAAACACT": {"vcrs1": 3, "vcrs2": 0, "vcrs3": 0, "vcrs4": 0, "vcrs5": 0, "vcrs6": 0},
    #     "TATCAGGAGCTAAGTG": {"vcrs1": 1.5, "vcrs2": 2.5, "vcrs3": 0.67, "vcrs4": 0.67, "vcrs5": 0.67, "vcrs6": 0},
    # }

    matrix_df_gt = pd.DataFrame(count_matrix_data_gt)
    matrix_df_gt = matrix_df_gt.T
    matrix_df_gt = matrix_df_gt.astype("float64")

    # ensure no rounding errors (eg 0.6667 vs 0.66666666...)
    matrix_df = matrix_df.round(3)
    matrix_df_gt = matrix_df_gt.round(3)

    assert read_to_ref_dict == read_to_ref_dict_gt  # from BUS file
    assert matrix_df.equals(matrix_df_gt)  # from adata

def test_bustools_df_10xv3_MM_union(temp_fastq_R1_complex, temp_fastq_R2_complex, temp_fasta_file_complex, temp_index_file, temp_t2g_file, temp_kb_count_out_folder):
    mm = True
    union = True
    
    k = "31"
    kb_ref_command = ["kb", "ref", "--workflow", "custom", "-t", "2", "-i", str(temp_index_file), "--d-list", "None", "-k", k, str(temp_fasta_file_complex)]
    subprocess.run(kb_ref_command, check=True)
    create_identity_t2g(temp_fasta_file_complex, temp_t2g_file)
    
    kb_count_command = ["kb", "count", "-t", "2", "-k", str(k), "-i", str(temp_index_file), "-g", str(temp_t2g_file), "-x", "10XV3", "--num", "--h5ad", "-o", str(temp_kb_count_out_folder), str(temp_fastq_R1_complex), str(temp_fastq_R2_complex)]
    if mm:
        kb_count_command.insert(2, "--mm")
    if union:
        kb_count_command.insert(2, "--union")
    subprocess.run(kb_count_command, check=True)

    bustools = None
    if not bustools:
        bustools_binary_path_command = "kb info | grep 'bustools:' | awk '{print $3}' | sed 's/[()]//g'"
        bustools = subprocess.run(bustools_binary_path_command, shell=True, executable="/bin/bash", stdout=subprocess.PIPE, text=True, check=True).stdout.strip()

    bus_df = make_bus_df(kb_count_out = temp_kb_count_out_folder, fastq_file_list = [temp_fastq_R1_complex, temp_fastq_R2_complex], t2g_file = temp_t2g_file, mm = mm, technology = "10XV3", bustools = bustools, fastq_sorting_check_only=True)
    read_to_ref_dict = dict(zip(bus_df['fastq_header'], bus_df['gene_names']))

    adata_path = f"{temp_kb_count_out_folder}/counts_unfiltered/adata.h5ad"
    adata = ad.read_h5ad(adata_path)

    # Convert adata.X to a DataFrame (if not already)
    matrix_df = pd.DataFrame(adata.X.toarray() if hasattr(adata.X, "toarray") else adata.X, 
                    index=adata.obs.index, 
                    columns=adata.var.index)

    read_to_ref_dict_gt = {
        'read0_mapsto_vcrs1_R2': ['vcrs1'],  # count_matrix_data_gt["AAACCCAAGAAACACT"]["vcrs1"] = 1
        'read1_mapsto_vcrs1_same_barcode_and_umi_R2': ['vcrs1'],  # because it has duplicate UMI as read0, it doesn't count for count matrix
        'read2_mapsto_vcrs1_different_barcode_R2': ['vcrs1'],  # count_matrix_data_gt["TATCAGGAGCTAAGTG"]["vcrs1"] = 1
        'read3_mapsto_vcrs2_same_barcode_R2': ['vcrs2'],  # count_matrix_data_gt["TATCAGGAGCTAAGTG"]["vcrs2"] = 1
        'read4_mapsto_vcrs2_different_umi_R2': ['vcrs2'],  # count_matrix_data_gt["TATCAGGAGCTAAGTG"]["vcrs2"] = 2
        'read5_mapsto_vcrs1_and_vcrs2_union_R2': [],  # doesn't count for count matrix OR show up in bus file unless --union is used
        'read6_mapsto_vcrs3_and_vcrs4_and_vcrs5_multimap_R2': ['vcrs3', 'vcrs4', 'vcrs5'],  # doesn't count for count matrix unless --mm is used (but shows up in bus file regardless)
        'read7_mapsto_vcrs3_and_vcrs4_and_vcrs5_multimap_different_umi_R2': ['vcrs3', 'vcrs4', 'vcrs5'],  # doesn't count for count matrix unless --mm is used (but shows up in bus file regardless)
        'read8_mapsto_vcrs1_barcode1_but_hamming_distance2_R2': ['vcrs1'],  # count_matrix_data_gt["AAACCCAAGAAACACT"]["vcrs1"] = 2
        'read9_mapsto_vcrs1_and_vcrs2_barcode1_same_umi_as_read10_R2': [],  # doesn't count for VCRS1 without union (won't show up in BUS file without union); VCRS2 doesn't count regardless because it has the same UMI as read10, but read10 doesn't map to VCRS2 
        'read10_mapsto_vcrs1_and_vcrs6_barcode1_same_umi_as_read9_R2': [],  # doesn't count due to same barcode and UMI as read9
    }
    if union:
        read_to_ref_dict_gt['read5_mapsto_vcrs1_and_vcrs2_union_R2'] = ['vcrs1', 'vcrs2']
        read_to_ref_dict_gt['read9_mapsto_vcrs1_and_vcrs2_barcode1_same_umi_as_read10_R2'] = ['vcrs1', 'vcrs2']
        read_to_ref_dict_gt['read10_mapsto_vcrs1_and_vcrs6_barcode1_same_umi_as_read9_R2'] = ['vcrs1', 'vcrs6']

    read_to_ref_dict_gt = {k: v for k, v in read_to_ref_dict_gt.items() if v != []}  # remove empty keys

    count_matrix_data_gt = {
        "AAACCCAAGAAACACT": {"vcrs1": 2, "vcrs2": 0, "vcrs3": 0, "vcrs4": 0, "vcrs5": 0, "vcrs6": 0},
        "TATCAGGAGCTAAGTG": {"vcrs1": 1, "vcrs2": 2, "vcrs3": 0, "vcrs4": 0, "vcrs5": 0, "vcrs6": 0},
    }
    if union and mm:  # notably, won't show up in count matrix unless mm is also used
        count_matrix_data_gt["AAACCCAAGAAACACT"]["vcrs1"] += (1/1)  # each unioned read adds (1/n), where n is the number of VCRSs to which the read maps - this comes from read9/10
        count_matrix_data_gt["TATCAGGAGCTAAGTG"]["vcrs1"] += (1/2)  # each unioned read adds (1/n), where n is the number of VCRSs to which the read maps - this comes from read5
        count_matrix_data_gt["TATCAGGAGCTAAGTG"]["vcrs2"] += (1/2)  # each unioned read adds (1/n), where n is the number of VCRSs to which the read maps - this comes from read5
    if mm:
        count_matrix_data_gt["TATCAGGAGCTAAGTG"]["vcrs3"] += (1/3) * 2  # each multimapped read adds (1/n), where n is the number of VCRSs to which the read maps, and reads 6 and 7 are both multimappers hence the *2
        count_matrix_data_gt["TATCAGGAGCTAAGTG"]["vcrs4"] += (1/3) * 2  # each multimapped read adds (1/n), where n is the number of VCRSs to which the read maps, and reads 6 and 7 are both multimappers hence the *2
        count_matrix_data_gt["TATCAGGAGCTAAGTG"]["vcrs5"] += (1/3) * 2  # each multimapped read adds (1/n), where n is the number of VCRSs to which the read maps, and reads 6 and 7 are both multimappers hence the *2
    # count_matrix_data_gt_with_multimap = {
    #     "AAACCCAAGAAACACT": {"vcrs1": 3, "vcrs2": 0, "vcrs3": 0, "vcrs4": 0, "vcrs5": 0, "vcrs6": 0},
    #     "TATCAGGAGCTAAGTG": {"vcrs1": 1.5, "vcrs2": 2.5, "vcrs3": 0.67, "vcrs4": 0.67, "vcrs5": 0.67, "vcrs6": 0},
    # }

    matrix_df_gt = pd.DataFrame(count_matrix_data_gt)
    matrix_df_gt = matrix_df_gt.T
    matrix_df_gt = matrix_df_gt.astype("float64")

    # ensure no rounding errors (eg 0.6667 vs 0.66666666...)
    matrix_df = matrix_df.round(3)
    matrix_df_gt = matrix_df_gt.round(3)

    assert read_to_ref_dict == read_to_ref_dict_gt  # from BUS file
    assert matrix_df.equals(matrix_df_gt)  # from adata