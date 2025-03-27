import os
import tempfile
import anndata as ad
from pdb import set_trace as st

import numpy as np
import pandas as pd
import pytest
from datetime import datetime
from pathlib import Path

from varseek.utils import (
    cleaned_adata_to_vcf,
    vcf_to_dataframe,
    remove_variants_from_adata_for_stranded_technologies
)

from .conftest import (
    compare_two_anndata_objects,
)

store_out_in_permanent_paths = True
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
        out = tmp_path / "out_vk_clean"

    out.mkdir(parents=True, exist_ok=True)  # Ensure the directory exists
    return out

def test_cleaned_adata_to_vcf(out_dir):
    # Define the data
    adata_var_toy = pd.DataFrame({
        "vcrs_header": ["header1", "header2;header3", "header4;header5;header6", "header7", "header8"],
        "vcrs_count": [1, 4, 2, 99, 0],
        "number_obs": [1, 2, 1, 3, 0],
    })

    cosmic_data_toy = pd.DataFrame({
        "ID": ["header1", "header2", "header3", "header4", "header5", "header6", "header7", "header8"],
        "CHROM": ["chr1", "chr2", "chr2", "chr4", "chr4", "chr6", "chr7", "chr8"],
        "POS": ["1", "2", "2", "4", "4", "6", "7", "8"],
        "REF": ["A", "A", "A", "A", "A", "A", "A", "A"],
        "ALT": ["C", "G", "G", "C", "C", "C", "C", "C"]
    })

    output_vcf = out_dir / "output.vcf"

    cleaned_adata_to_vcf(adata_var_toy, vcf_data_df=cosmic_data_toy, output_vcf = output_vcf, save_vcf_samples=False)

    test_vcf_df = vcf_to_dataframe(output_vcf, additional_columns=False)
    test_vcf_df["CHROM"] = test_vcf_df["CHROM"].astype(str)
    ground_truth_vcf_df = pd.DataFrame({
        "CHROM": ["chr1", "chr2", "chr7"],
        "POS": [1, 2, 7],
        "ID": ["header1", "header2;header3", "header7"],
        "REF": ["A", "A", "A"],
        "ALT": ["C", "G", "C"],
        # "QUAL": [None, None, None],
        # "FILTER": ["PASS", "PASS", "PASS"],
        # "INFO_AO": [1, 4, 99],
        # "INFO_NS": [1, 2, 3],
    })
    assert test_vcf_df.equals(ground_truth_vcf_df), "The VCF file does not match the expected output."


def test_cleaned_adata_to_vcf_with_samples(out_dir):
    # Define the data
    adata_var_toy = pd.DataFrame({
        "vcrs_header": ["header1", "header2;header3", "header4;header5;header6", "header7", "header8"],
        "vcrs_count": [1, 4, 2, 99, 0],
        "number_obs": [1, 2, 1, 3, 0],
    })

    # Define the specified X matrix (4 samples × 5 genes)
    X = np.array([
        [0, 1, 0, 1, 0],
        [0, 0, 0, 0, 0],
        [1, 3, 0, 90, 0],
        [0, 0, 2, 8, 0]
    ])

    # Define sample metadata
    obs = pd.DataFrame(index=[f"sample_{i+1}" for i in range(4)])  # Sample names

    # Create AnnData object
    adata = ad.AnnData(X=X, var=adata_var_toy, obs=obs)

    cosmic_data_toy = pd.DataFrame({
        "ID": ["header1", "header2", "header3", "header4", "header5", "header6", "header7", "header8"],
        "CHROM": ["chr1", "chr2", "chr2", "chr4", "chr4", "chr6", "chr7", "chr8"],
        "POS": ["1", "2", "2", "4", "4", "6", "7", "8"],
        "REF": ["A", "A", "A", "A", "A", "A", "A", "A"],
        "ALT": ["C", "G", "G", "C", "C", "C", "C", "C"]
    })

    output_vcf = out_dir / "output.vcf"

    cleaned_adata_to_vcf(adata, vcf_data_df=cosmic_data_toy, output_vcf = output_vcf, save_vcf_samples=True)

    test_vcf_df = vcf_to_dataframe(output_vcf, additional_columns=True)
    test_vcf_df["CHROM"] = test_vcf_df["CHROM"].astype(str)
    ground_truth_vcf_df = pd.DataFrame({
        "CHROM": ["chr1", "chr2", "chr7"],
        "POS": [1, 2, 7],
        "ID": ["header1", "header2;header3", "header7"],
        "REF": ["A", "A", "A"],
        "ALT": ["C", "G", "C"],
        "QUAL": [None, None, None],
        "FILTER": [("PASS",), ("PASS",), ("PASS",)],
        "INFO_AO": [1, 4, 99],
        "INFO_NS": [1, 2, 3],
        "sample_1_AO": [0, 1, 1],
        "sample_2_AO": [0, 0, 0],
        "sample_3_AO": [1, 3, 90],
        "sample_4_AO": [0, 0, 8],
    })
    assert test_vcf_df.equals(ground_truth_vcf_df), "The VCF file does not match the expected output."


@pytest.fixture
def adata_for_strand_bias_testing():
    """Fixture to create a test AnnData object."""
    # Define the data
    adata_var_toy = pd.DataFrame({
        "vcrs_header": ["ENST0000001:c.10A>G", "ENST0000001:c.11A>G;ENST0000001:c.12A>G", "ENST0000001:c.11_12insAAT", "ENST0000001:c.12del;ENST0000002:c.14del", "ENST0000001:c.101del;ENST0000003:c.100A>T", "ENST0000001:c.50G>A;ENST0000006:c.1001G>A", "ENST0000001:c.89_93del"],
        "vcrs_count": [1, 4, 2, 99, 0, 1, 1],
        "number_obs": [1, 2, 1, 3, 0, 1, 1],
    })

    # Define the specified X matrix (4 samples × 5 genes)
    X = np.array([
        [0, 1, 0, 1, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0],
        [1, 3, 0, 90, 0, 1, 0],
        [0, 0, 2, 8, 0, 0, 1]
    ])

    # Define sample metadata
    obs = pd.DataFrame(index=[f"sample_{i+1}" for i in range(X.shape[0])])  # Sample names

    # Create AnnData object
    return ad.AnnData(X=X, var=adata_var_toy, obs=obs)

@pytest.fixture
def transcript_id_df_for_strand_bias_testing():
    transcript_id_df = pd.DataFrame({
        "transcript_ID": ["ENST0000001", "ENST0000002", "ENST0000003", "ENST0000004", "ENST0000005", "ENST0000006", "ENST0000007"],
        "start": [1, 200, 1000, 7001, 9000, 12000, 50000],
        "end": [120, 240, 1140, 8000, 10000, 12100, 50500],
    })

    return transcript_id_df

def test_strand_bias_filtering_5p(adata_for_strand_bias_testing):
    strand_bias_end = "5p"
    read_length = "90"
    
    adata = remove_variants_from_adata_for_stranded_technologies(adata=adata_for_strand_bias_testing, strand_bias_end=strand_bias_end, read_length=read_length, header_column="vcrs_header", variant_source=None, gtf=None)

    adata_var_gt = pd.DataFrame({
        "vcrs_header": ["ENST0000001:c.10A>G", "ENST0000001:c.11A>G;ENST0000001:c.12A>G", "ENST0000001:c.11_12insAAT", "ENST0000001:c.12del;ENST0000002:c.14del", "ENST0000001:c.50G>A", "ENST0000001:c.89_93del"],
        "vcrs_count": [1, 4, 2, 99, 1, 1],
        "number_obs": [1, 2, 1, 3, 1, 1],
    })

    adata_x_gt = np.array([
        [0, 1, 0, 1, 0, 0],
        [0, 0, 0, 0, 0, 0],
        [1, 3, 0, 90, 1, 0],
        [0, 0, 2, 8, 0, 1]
    ])

    adata_obs_gt = adata_for_strand_bias_testing.obs

    adata_gt = ad.AnnData(X=adata_x_gt, var=adata_var_gt, obs=adata_obs_gt)

    compare_two_anndata_objects(adata, adata_gt)

def test_strand_bias_filtering_3p(adata_for_strand_bias_testing, transcript_id_df_for_strand_bias_testing):
    strand_bias_end = "3p"
    read_length = "90"
    
    adata = remove_variants_from_adata_for_stranded_technologies(adata=adata_for_strand_bias_testing, strand_bias_end=strand_bias_end, read_length=read_length, header_column="vcrs_header", variant_source=None, gtf=transcript_id_df_for_strand_bias_testing)

    #!!! ENST0000003:c.100A>T
    #!!! ENST0000001:c.50G>A;ENST0000006:c.1001G>A got split into separate
    adata_var_gt = pd.DataFrame({
        "vcrs_header": ["ENST0000002:c.14del", "ENST0000001:c.101del;ENST0000003:c.100A>T", "ENST0000001:c.50G>A;ENST0000006:c.1001G>A", "ENST0000001:c.89_93del"],
        "vcrs_count": [99, 0, 1, 1],
        "number_obs": [3, 0, 1, 1],
    })

    adata_x_gt = np.array([
        [1, 0, 0, 0],
        [0, 0, 0, 0],
        [90, 0, 1, 0],
        [8, 0, 0, 1]
    ])

    adata_obs_gt = adata_for_strand_bias_testing.obs

    adata_gt = ad.AnnData(X=adata_x_gt, var=adata_var_gt, obs=adata_obs_gt)

    compare_two_anndata_objects(adata, adata_gt)
















from varseek.utils import create_identity_t2g, make_bus_df
import subprocess


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


@pytest.fixture
def temp_fastq_R1(tmp_path):
    fastq_content = (
        "@seq1\n"
        "XXXX\n"
        "+\n"
        "IIII\n"
        "@seq2\n"
        "XXXX\n"
        "+\n"
        "IIII\n"
        "@seq3\n"
        "XXXX\n"
        "+\n"
        "IIII\n"
        "@seq4\n"
        "XXXX\n"
        "+\n"
        "IIII\n"
        "@seq5\n"
        "XXXX\n"
        "+\n"
        "IIII\n"
    )
    fastq_path = tmp_path / "test_sequences_R1.fastq"
    fastq_path.write_text(fastq_content)
    return fastq_path

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

    bus_df = make_bus_df(kb_count_out = temp_kb_count_out_folder, fastq_file_list = temp_fastq_file, t2g_file = temp_t2g_file, mm = False, technology = "bulk", bustools = bustools, check_only=True)
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

    bus_df = make_bus_df(kb_count_out = temp_kb_count_out_folder, fastq_file_list = [temp_fastq_file, temp_fastq_file_pair], t2g_file = temp_t2g_file, mm = False, technology = "bulk", bustools = bustools, check_only=True)
    read_to_ref_dict = dict(zip(bus_df['fastq_header'], bus_df['gene_names']))

    assert read_to_ref_dict == {'seq1_1': ['vcrs1'], 'seq2_1': ['vcrs2'], 'seq5_1': ['vcrs2', 'vcrs4', 'vcrs5']}

    adata_path = f"{temp_kb_count_out_folder}/counts_unfiltered/adata.h5ad"
    adata = ad.read_h5ad(adata_path)

    assert np.array_equal(adata.X.toarray(), np.array([[1., 1., 0., 0., 0., 0.]]))

def test_bustools_df_10xv3(temp_fastq_file, temp_fastq_R1, temp_fasta_file, temp_index_file, temp_t2g_file, temp_kb_count_out_folder):
    k = "31"
    kb_ref_command = ["kb", "ref", "--workflow", "custom", "-t", "2", "-i", str(temp_index_file), "--d-list", "None", "-k", k, str(temp_fasta_file)]
    subprocess.run(kb_ref_command, check=True)
    create_identity_t2g(temp_fasta_file, temp_t2g_file)
    
    kb_count_command = ["kb", "count", "-t", "2", "-k", str(k), "-i", str(temp_index_file), "-g", str(temp_t2g_file), "-x", "10XV3", "--num", "--h5ad", "-o", str(temp_kb_count_out_folder), str(temp_fastq_R1), str(temp_fastq_file)]
    subprocess.run(kb_count_command, check=True)

    bustools = None
    if not bustools:
        bustools_binary_path_command = "kb info | grep 'bustools:' | awk '{print $3}' | sed 's/[()]//g'"
        bustools = subprocess.run(bustools_binary_path_command, shell=True, executable="/bin/bash", stdout=subprocess.PIPE, text=True, check=True).stdout.strip()

    bus_df = make_bus_df(kb_count_out = temp_kb_count_out_folder, fastq_file_list = [temp_fastq_R1, temp_fastq_file], t2g_file = temp_t2g_file, mm = False, technology = "bulk", bustools = bustools, check_only=True)
    read_to_ref_dict = dict(zip(bus_df['fastq_header'], bus_df['gene_names']))

    assert read_to_ref_dict == {'seq1_1': ['vcrs1', 'vcrs6'], 'seq2_1': ['vcrs2'], 'seq3_1': ['vcrs1', 'vcrs6'], 'seq5_1': ['vcrs2', 'vcrs4', 'vcrs5']}

    adata_path = f"{temp_kb_count_out_folder}/counts_unfiltered/adata.h5ad"
    adata = ad.read_h5ad(adata_path)

    assert np.array_equal(adata.X.toarray(), np.array([[1., 1., 0., 0., 0., 0.]]))