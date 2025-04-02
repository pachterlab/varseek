import os
import tempfile
import anndata as ad
from pdb import set_trace as st

import numpy as np
import pandas as pd
import pytest
from datetime import datetime
import subprocess
from pathlib import Path

from varseek.utils import (
    adjust_variant_adata_by_normal_gene_matrix
)

from .conftest import (
    compare_two_anndata_objects,
    create_identity_t2g
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
        out = tmp_path / "out_vk_clean_normal_genome"

    out.mkdir(parents=True, exist_ok=True)  # Ensure the directory exists
    return out

@pytest.fixture
def bustools():
    bustools_binary_path_command = "kb info | grep 'bustools:' | awk '{print $3}' | sed 's/[()]//g'"
    bustools = subprocess.run(bustools_binary_path_command, shell=True, executable="/bin/bash", stdout=subprocess.PIPE, text=True, check=True).stdout.strip()
    return bustools

@pytest.fixture
def NG_vcrs_reference_fasta(tmp_path):
    fasta_content = (
        ">ENST1:mut\n"
        "AGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGTAC\n"
        ">ENST2:mut\n"
        "CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCAAGGAAGGAAAAGGAAGGAAAAGGAAGGAAAAGGAAGGAAAAGGAAGGAAAAGGAAGGAAAAGGAAGGCCCCCAAAAACCCCCAAAAACCCCCAAAAACCCCCAAAAA\n"
        ">ENST3:mut\n"
        "ATGCTGACTGACTGCTGACTGCTAGCTGACGTCATCAGTACGTACGATGCTGACTGACTGCTGACTGCTAGCTGACGTCATCAGTACGTACGCCCCACCCCACCCCACCCCACCCCACCCCACCCCACCCCACCCCACCCCACCCCACCCCA\n"
        ">ENST4:mut\n"
        "AAGGAAGGAAAAGGAAGGAAAAGGAAGGAAAAGGAAGGAAAAGGAAGGAAAAGGAAGGAAAAGGAAGGAAA\n"
        ">ENST5:mut\n"
        "TTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTA\n"
        ">ENST6:mut\n"
        "TTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTG\n"
        ">ENST7:mut\n"
        "AGCCATCGACTAGCTACGACTGCATGCTGACTGCATACGCATGACGTCAGCGAAAAAATACTCAGTT\n"
        ">ENST8:mut\n"
        "AGCCATCGACTAGCTACGACTGCATGCTGACTGCATACGCATGACGTCAGCGAAAAAATACTCAGTG\n"
        ">ENST9:mut\n"
        "CCGGGCCCGGGAAATTTAAATTTCCGGGCCCGGGAAATTTAAATTTCCGGGCCCGGGAAATTTAAATTTCCGGGCCCGGGAAATTTAAATTT\n"
    )
    tmp_fasta_path = tmp_path / "vcrs.fasta"
    tmp_fasta_path.write_text(fasta_content)
    return tmp_fasta_path

@pytest.fixture
def NG_vcrs_reference_t2g(tmp_path, NG_vcrs_reference_fasta):
    tmp_t2g_path = tmp_path / "vcrs_t2g.txt"
    create_identity_t2g(NG_vcrs_reference_fasta, tmp_t2g_path)
    return tmp_t2g_path

@pytest.fixture
def NG_vcrs_reference_index(tmp_path, NG_vcrs_reference_fasta):
    tmp_index_path = tmp_path / "mutation_index.idx"
    kb_ref_command_vcrs = ["kb", "ref", "--workflow", "custom", "-t", "2", "-i", str(tmp_index_path), "--d-list", "None", "-k", "31", str(NG_vcrs_reference_fasta)]
    subprocess.run(kb_ref_command_vcrs, check=True)
    return tmp_index_path

@pytest.fixture
def NG_normal_reference_fasta(tmp_path):
    fasta_content = (
        ">ENST1\n"
        "AGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGTATTATTATTATTATTATTATTATTATTATTATTATTATTATTATTATTATTATTATTATTATTA\n"
        ">ENST2\n"
        "AGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGCCCCCCCCCCAAAAACCCCCAAAAACCCCCAAAAACCCCCAAAAACCCCCAAAAACCCCCAAAAA\n"
        ">ENST3\n"
        "ATGCTGACTGACTGCTGACTGCTAGCTGACGTCATCAGTACGTACGATGCTGACTGACTGCTGACTGCTAGCTGACGTCATCAGTACGTACG\n"
        ">ENST4\n"
        "CCCCACCCCACCCCACCCCACCCCACCCCACCCCACCCCACCCCACCCCACCCCACCCCA\n"
        ">ENST5\n"
        "TTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTA\n"
        ">ENST6\n"
        "TTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTG\n"
        ">ENST7\n"
        "AGCCATCGACTAGCTACGACTGCATGCTGACTGCATACGCATGACGTCAGCGAAAAAATACTCAGT\n"
        ">ENST8\n"
        "CCGACGTCAGTACTGCTGATAGCTGAGTCGTCATGACGTCAGTACGTGGGGCGCGCCCCTCTATTATCGTCATGCAGT\n"
        ">ENST9\n"
        "CGCTAGTAGTAGTAGTAGCGCGCACCGCACCGCACCGCACCGCACCGCACCCGATCGATCGATCGATCGATCGATCGA\n"
    )
    tmp_fasta_path = tmp_path / "normal.fasta"
    tmp_fasta_path.write_text(fasta_content)
    return tmp_fasta_path

@pytest.fixture
def NG_normal_reference_t2g(tmp_path):
    t2g_content = (
        "ENST1\tgene1\n"
        "ENST2\tgene2\n"
        "ENST3\tgene3\n"
        "ENST4\tgene4\n"
        "ENST5\tgene5\n"
        "ENST6\tgene6\n"
        "ENST7\tgene7\n"
        "ENST8\tgene8\n"
        "ENST9\tgene9\n"
    )
    tmp_t2g_path = tmp_path / "normal_t2g.txt"
    tmp_t2g_path.write_text(t2g_content)
    return tmp_t2g_path

@pytest.fixture
def NG_normal_reference_index(tmp_path, NG_normal_reference_t2g, NG_normal_reference_fasta):
    tmp_index_path = tmp_path / "normal_index.idx"
    kb_ref_command_normal = ["kb", "ref", "-t", "2", "--workflow", "custom", "-i", str(tmp_index_path), "-g", str(NG_normal_reference_t2g), "--d-list", "None", "-k", "31", str(NG_normal_reference_fasta)]
    subprocess.run(kb_ref_command_normal, check=True)
    return tmp_index_path

@pytest.fixture
def NG_fastq_bulk_1(tmp_path):
    fastq_content = (
        "@read0_mapsto_ENST1:mut_and_NORMAL_ENST1\n"
        "AGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGTATGCCGTACGGCTATGCAGTACGTCTGCAGTACTGCATGACTGACTGCA\n"
        "+\n"
        "IIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIII\n"
        "@read1_mapsto_VCRS_ENST3:mut_and_NORMAL_ENST4\n"
        "CCCCACCCCACCCCACCCCACCCCACCCCACCCCACCCCACCCCACCCCACCCCACCCCACCCCACCCCACCCCACCCCACCCCACCCCACCCCACCCCACCCCACCCCACCCCACCCCACCCCACCCCACCCCACCCCACCCCACCCCA\n"
        "+\n"
        "IIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIII\n"
        "@read2_mapsto_VCRS_ENST1:mut_and_NORMAL_ENST1_and_ENST2\n"
        "AGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAAAAAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAAAAAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAAAAAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAAAAAGAGAGAGAGAGAG\n"
        "+\n"
        "IIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIII\n"
        "@read3_MM_mapsto_VCRS_ENST5:mut_and_ENST6:mut_and_NORMAL_ENST5_and_ENST6\n"
        "TTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTT\n"
        "+\n"
        "IIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIII\n"
        "@read4_MM_mapsto_VCRS_ENST7:mut_and_ENST8:mut_and_NORMAL_ENST7\n"
        "AGCCATCGACTAGCTACGACTGCATGCTGACTGCATACGCATGACGTCAGCGAAAAAATACTCAGCGCGGCGCGAGCCATCGACTAGCTACGACTGCATGCTGACTGCATACGCATGACGTCAGCGAAAAAATACTCAGCGTAGCATCGG\n"
        "+\n"
        "IIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIII\n"
        "@read5_mapsto_VCRS_ENST9:mut_and_NORMAL_none\n"
        "CCGGGCCCGGGAAATTTAAATTTCCGGGCCCGGGAAATTTAAATTTCCGGGCCCGGGACCGGGCCCGGGAAATTTAAATTTCCGGGCCCGGGAAATTTAAATTTCCGGGCCCGGGACCGGGCCCGGGAAATTTAAATTTCCGGGCCCGGG\n"
        "+\n"
        "IIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIII\n"
    )

    fastq_path = tmp_path / "NG_fastq_bulk_1.fastq"
    fastq_path.write_text(fastq_content)
    return fastq_path



@pytest.fixture
def NG_fastq_bulk_2(tmp_path):
    fastq_content = (
        "@read0_pair_mapsto_VCRS_none_and_NORMAL_ENST1\n"
        "AGAGAGAGAGAGAGAGAGAGAGAGAGAGTATAGAGAGAGAGAGAGAGAGAGAGAGAGAGTATAGAGAGAGAGAGAGAGAGAGAGAGAGAGTATAGAGAGAGAGAGAGAGAGAGAGAGAGAGTATAGAGAGAGAGAGAGAGAGAGAGAGAG\n"
        "+\n"
        "IIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIII\n"
        "@read1_pair_mapsto_VCRS_ENST4:mut_and_NORMAL_none\n"
        "AAGGAAGGAAAAGGAAGGAAAAGGAAGGAAAAGGAAGGAAAAGGAAGGAAAAGGAAGGAAAAGGAAGGAAAAAGGAAGGAAAAGGAAGGAAAAGGAAGGAAAAGGAAGGAAAAGGAAGGAAAAGGAAGGAAAAGGAAGGAAAAAGGAAGG\n"
        "+\n"
        "IIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIII\n"
        "@read2_pair_mapsto_VCRS_ENST2:mut_and_ENST4:mut_and_NORMAL_ENST2\n"
        "CCCCCAAAAACCCCCAAAAACCCCCAAAAACCCCCAAAAACCCCCAAAAACCCCCAAAAACCCCCAAAAACCCCCAAAAACCCCCAAAAACCCCCAAAAACCCCCAAAAACCCCCAAAAACCCCCAAAAACCCCCAAAAACCCCCAAAAA\n"
        "+\n"
        "IIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIII\n"
        "@read3_pair_mapsto_VCRS_none_and_NORMAL_none\n"
        "ACGTGCACGCCAGCACACGAACGTGCACGCCAGCACACGAACGTGCACGCCAGCACACGAACGTGCACGCCAGCACACGAACGTGCACGCCAGCACACGAACGTGCACGCCAGCACACGAACGTGCACGCCAGCACACGAACGTGCACGC\n"
        "+\n"
        "IIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIII\n"
        "@read4_pair_mapsto_VCRS_none_and_NORMAL_none\n"
        "ACGTGCACGCCAGCACACGAACGTGCACGCCAGCACACGAACGTGCACGCCAGCACACGAACGTGCACGCCAGCACACGAACGTGCACGCCAGCACACGAACGTGCACGCCAGCACACGAACGTGCACGCCAGCACACGAACGTGCACGC\n"
        "+\n"
        "IIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIII\n"
        "@read5_pair_mapsto_VCRS_none_and_NORMAL_none\n"
        "ACGTGCACGCCAGCACACGAACGTGCACGCCAGCACACGAACGTGCACGCCAGCACACGAACGTGCACGCCAGCACACGAACGTGCACGCCAGCACACGAACGTGCACGCCAGCACACGAACGTGCACGCCAGCACACGAACGTGCACGC\n"
        "+\n"
        "IIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIII\n"
    )
    fastq_path = tmp_path / "NG_fastq_bulk_2.fastq"
    fastq_path.write_text(fastq_content)
    return fastq_path

@pytest.fixture
def NG_fastq_10xv3_R1(tmp_path):
    fastq_content = ""
    fastq_path = tmp_path / "NG_fastq_10xv3_R1.fastq"
    fastq_path.write_text(fastq_content)
    return fastq_path

@pytest.fixture
def NG_fastq_10xv3_R2(tmp_path):
    fastq_content = ""
    fastq_path = tmp_path / "NG_fastq_10xv3_R2.fastq"
    fastq_path.write_text(fastq_content)
    return fastq_path



def test_adjust_variant_adata_by_normal_gene_matrix_bulk_single(out_dir, NG_vcrs_reference_fasta, NG_vcrs_reference_t2g, NG_vcrs_reference_index, NG_normal_reference_fasta, NG_normal_reference_t2g, NG_normal_reference_index, NG_fastq_bulk_1, NG_fastq_bulk_2, bustools):
    technology = "BULK"
    mm = False
    union = False
    parity = "single"
    vcrs_parity = "single"
    count_reads_that_dont_pseudoalign_to_reference_genome = True

    fastqs = [str(NG_fastq_bulk_1)]

    kb_count_out_vcrs = out_dir / "kb_count_vcrs"
    kb_count_out_normal = out_dir / "kb_count_normal"
    
    kb_count_command_vcrs = ["kb", "count", "-t", "2", "-k", "31", "-i", str(NG_vcrs_reference_index), "-g", str(NG_vcrs_reference_t2g), "-x", technology, "--num", "--h5ad", "-o", str(kb_count_out_vcrs)] + fastqs
    if mm:
        kb_count_command_vcrs.insert(2, "--mm")
    if union:
        kb_count_command_vcrs.insert(2, "--union")
    if technology in {"BULK", "SMARTSEQ2"}:
        kb_count_command_vcrs.insert(2, vcrs_parity)
        kb_count_command_vcrs.insert(2, "--parity")
    subprocess.run(kb_count_command_vcrs, check=True)

    kb_count_command_normal = ["kb", "count", "-t", "2", "-k", "31", "-i", str(NG_normal_reference_index), "-g", str(NG_normal_reference_t2g), "-x", technology, "--num", "--h5ad", "-o", str(kb_count_out_normal)] + fastqs
    if technology in {"BULK", "SMARTSEQ2"}:
        kb_count_command_normal.insert(2, parity)
        kb_count_command_normal.insert(2, "--parity")
    subprocess.run(kb_count_command_normal, check=True)

    adata = adjust_variant_adata_by_normal_gene_matrix(kb_count_vcrs_dir=kb_count_out_vcrs, kb_count_reference_genome_dir=kb_count_out_normal, fastq_file_list=fastqs, technology=technology, t2g_standard=NG_normal_reference_t2g, adata_output_path=None, mm=mm, parity=parity, bustools=bustools, fastq_sorting_check_only=True, save_type="parquet", count_reads_that_dont_pseudoalign_to_reference_genome=count_reads_that_dont_pseudoalign_to_reference_genome)

    # Convert adata.X to a DataFrame (if not already)
    matrix_df = pd.DataFrame(adata.X.toarray() if hasattr(adata.X, "toarray") else adata.X, 
                    index=adata.obs.index, 
                    columns=adata.var.index)

    count_matrix_data_gt = {
        "AAAAAAAAAAAAAAAA": {"ENST1:mut": 2, "ENST2:mut": 0, "ENST3:mut": 0, "ENST4:mut": 0, "ENST5:mut": 0, "ENST6:mut": 0, "ENST7:mut": 1, "ENST8:mut": 0, "ENST9:mut": 0},
    }
    if mm:
        count_matrix_data_gt["AAAAAAAAAAAAAAAA"]["ENST5:mut"] += 1
        count_matrix_data_gt["AAAAAAAAAAAAAAAA"]["ENST6:mut"] += 1
    if count_reads_that_dont_pseudoalign_to_reference_genome:
        count_matrix_data_gt["AAAAAAAAAAAAAAAA"]["ENST9:mut"] += 1

    matrix_df_gt = pd.DataFrame(count_matrix_data_gt)
    matrix_df_gt = matrix_df_gt.T
    matrix_df_gt = matrix_df_gt.astype("float64")

    # ensure no rounding errors (eg 0.6667 vs 0.66666666...)
    matrix_df = matrix_df.round(3)
    matrix_df_gt = matrix_df_gt.round(3)

    assert matrix_df.equals(matrix_df_gt)  # from adata

    bus_df_vcrs = pd.read_parquet(kb_count_out_vcrs / "bus_df.parquet", columns=["fastq_header", "gene_names"])
    bus_df_vcrs.rename(columns={"gene_names": "vcrs_names"}, inplace=True)
    read_to_ref_vcrs_dict = dict(zip(bus_df_vcrs['fastq_header'], bus_df_vcrs['vcrs_names']))

    bus_df_normal = pd.read_parquet(kb_count_out_normal / "bus_df.parquet", columns=["fastq_header", "gene_names"])
    read_to_ref_normal_dict = dict(zip(bus_df_normal['fastq_header'], bus_df_normal['gene_names']))

    read_to_ref_vcrs_dict_gt = {
        "read0_mapsto_ENST1:mut_and_NORMAL_ENST1": ["ENST1:mut"],
        "read1_mapsto_VCRS_ENST3:mut_and_NORMAL_ENST4": ["ENST3:mut"],
        "read2_mapsto_VCRS_ENST1:mut_and_NORMAL_ENST1_and_ENST2": ["ENST1:mut"],
        "read3_MM_mapsto_VCRS_ENST5:mut_and_ENST6:mut_and_NORMAL_ENST5_and_ENST6": ["ENST5:mut", "ENST6:mut"],
        "read4_MM_mapsto_VCRS_ENST7:mut_and_ENST8:mut_and_NORMAL_ENST7": ["ENST7:mut", "ENST8:mut"],
        "read5_mapsto_VCRS_ENST9:mut_and_NORMAL_none": ["ENST9:mut"],
    }
    read_to_ref_normal_dict_gt = {
        "read0_mapsto_ENST1:mut_and_NORMAL_ENST1": ["gene1"],
        "read1_mapsto_VCRS_ENST3:mut_and_NORMAL_ENST4": ["gene4"],
        "read2_mapsto_VCRS_ENST1:mut_and_NORMAL_ENST1_and_ENST2": ["gene1", "gene2"],
        "read3_MM_mapsto_VCRS_ENST5:mut_and_ENST6:mut_and_NORMAL_ENST5_and_ENST6": ["gene5", "gene6"],
        "read4_MM_mapsto_VCRS_ENST7:mut_and_ENST8:mut_and_NORMAL_ENST7": ["gene7"],
        "read5_mapsto_VCRS_ENST9:mut_and_NORMAL_none": [],
    }

    read_to_ref_vcrs_dict_gt = {k: v for k, v in read_to_ref_vcrs_dict_gt.items() if v != []}  # remove empty keys
    read_to_ref_normal_dict_gt = {k: v for k, v in read_to_ref_normal_dict_gt.items() if v != []}  # remove empty keys

    assert read_to_ref_vcrs_dict == read_to_ref_vcrs_dict_gt  # from BUS file
    assert read_to_ref_normal_dict == read_to_ref_normal_dict_gt  # from BUS file

def test_adjust_variant_adata_by_normal_gene_matrix_bulk_paired_but_run_as_single(out_dir, NG_vcrs_reference_fasta, NG_vcrs_reference_t2g, NG_vcrs_reference_index, NG_normal_reference_fasta, NG_normal_reference_t2g, NG_normal_reference_index, NG_fastq_bulk_1, NG_fastq_bulk_2, bustools):
    technology = "BULK"
    mm = False
    union = False
    parity = "paired"
    vcrs_parity = "single"
    count_reads_that_dont_pseudoalign_to_reference_genome = True

    fastqs = [str(NG_fastq_bulk_1), str(NG_fastq_bulk_2)]

    kb_count_out_vcrs = out_dir / "kb_count_vcrs"
    kb_count_out_normal = out_dir / "kb_count_normal"
    
    kb_count_command_vcrs = ["kb", "count", "-t", "2", "-k", "31", "-i", str(NG_vcrs_reference_index), "-g", str(NG_vcrs_reference_t2g), "-x", technology, "--num", "--h5ad", "-o", str(kb_count_out_vcrs)] + fastqs
    if mm:
        kb_count_command_vcrs.insert(2, "--mm")
    if union:
        kb_count_command_vcrs.insert(2, "--union")
    if technology in {"BULK", "SMARTSEQ2"}:
        kb_count_command_vcrs.insert(2, vcrs_parity)
        kb_count_command_vcrs.insert(2, "--parity")
    subprocess.run(kb_count_command_vcrs, check=True)

    kb_count_command_normal = ["kb", "count", "-t", "2", "-k", "31", "-i", str(NG_normal_reference_index), "-g", str(NG_normal_reference_t2g), "-x", technology, "--num", "--h5ad", "-o", str(kb_count_out_normal)] + fastqs
    if technology in {"BULK", "SMARTSEQ2"}:
        kb_count_command_normal.insert(2, parity)
        kb_count_command_normal.insert(2, "--parity")
    subprocess.run(kb_count_command_normal, check=True)

    adata = adjust_variant_adata_by_normal_gene_matrix(kb_count_vcrs_dir=kb_count_out_vcrs, kb_count_reference_genome_dir=kb_count_out_normal, fastq_file_list=fastqs, technology=technology, t2g_standard=NG_normal_reference_t2g, adata_output_path=None, mm=mm, parity=parity, bustools=bustools, fastq_sorting_check_only=True, save_type="parquet", count_reads_that_dont_pseudoalign_to_reference_genome=count_reads_that_dont_pseudoalign_to_reference_genome)

    # Convert adata.X to a DataFrame (if not already)
    matrix_df = pd.DataFrame(adata.X.toarray() if hasattr(adata.X, "toarray") else adata.X, 
                    index=adata.obs.index, 
                    columns=adata.var.index)

    count_matrix_data_gt = {
        "AAAAAAAAAAAAAAAA": {"ENST1:mut": 2, "ENST2:mut": 0, "ENST3:mut": 0, "ENST4:mut": 0, "ENST5:mut": 0, "ENST6:mut": 0, "ENST7:mut": 1, "ENST8:mut": 0, "ENST9:mut": 0},
    }
    if mm:
        count_matrix_data_gt["AAAAAAAAAAAAAAAA"]["ENST5:mut"] += 1
        count_matrix_data_gt["AAAAAAAAAAAAAAAA"]["ENST6:mut"] += 1
    if count_reads_that_dont_pseudoalign_to_reference_genome:
        count_matrix_data_gt["AAAAAAAAAAAAAAAA"]["ENST9:mut"] += 1
    if parity == "paired" and vcrs_parity == "single":
        count_matrix_data_gt["AAAAAAAAAAAAAAAA"]["ENST1:mut"] -= 1
        count_matrix_data_gt["AAAAAAAAAAAAAAAA"]["ENST2:mut"] += 1
        count_matrix_data_gt["AAAAAAAAAAAAAAAA"]["ENST4:mut"] += 1

    matrix_df_gt = pd.DataFrame(count_matrix_data_gt)
    matrix_df_gt = matrix_df_gt.T
    matrix_df_gt = matrix_df_gt.astype("float64")

    # ensure no rounding errors (eg 0.6667 vs 0.66666666...)
    matrix_df = matrix_df.round(3)
    matrix_df_gt = matrix_df_gt.round(3)

    assert matrix_df.equals(matrix_df_gt)  # from adata

    bus_df_vcrs = pd.read_parquet(kb_count_out_vcrs / "bus_df.parquet", columns=["fastq_header", "gene_names"])
    bus_df_vcrs.rename(columns={"gene_names": "vcrs_names"}, inplace=True)
    read_to_ref_vcrs_dict = dict(zip(bus_df_vcrs['fastq_header'], bus_df_vcrs['vcrs_names']))

    bus_df_normal = pd.read_parquet(kb_count_out_normal / "bus_df.parquet", columns=["fastq_header", "gene_names"])
    read_to_ref_normal_dict = dict(zip(bus_df_normal['fastq_header'], bus_df_normal['gene_names']))

    read_to_ref_vcrs_dict_gt = {
        "read0_mapsto_ENST1:mut_and_NORMAL_ENST1": ["ENST1:mut"],
        "read1_mapsto_VCRS_ENST3:mut_and_NORMAL_ENST4": ["ENST3:mut"],
        "read2_mapsto_VCRS_ENST1:mut_and_NORMAL_ENST1_and_ENST2": ["ENST1:mut"],
        "read3_MM_mapsto_VCRS_ENST5:mut_and_ENST6:mut_and_NORMAL_ENST5_and_ENST6": ["ENST5:mut", "ENST6:mut"],
        "read4_MM_mapsto_VCRS_ENST7:mut_and_ENST8:mut_and_NORMAL_ENST7": ["ENST7:mut", "ENST8:mut"],
        "read5_mapsto_VCRS_ENST9:mut_and_NORMAL_none": ["ENST9:mut"],
        "read0_pair_mapsto_VCRS_none_and_NORMAL_ENST1": [],
        "read1_pair_mapsto_VCRS_ENST4:mut_and_NORMAL_none": ["ENST4:mut"],
        "read2_pair_mapsto_VCRS_ENST2:mut_and_ENST4:mut_and_NORMAL_ENST2": ["ENST2:mut", "ENST4:mut"],
        "read3_pair_mapsto_VCRS_none_and_NORMAL_none": [],
        "read4_pair_mapsto_VCRS_none_and_NORMAL_none": [],
        "read5_pair_mapsto_VCRS_none_and_NORMAL_none": [],
    }
    read_to_ref_normal_dict_gt = {
        "read0_mapsto_ENST1:mut_and_NORMAL_ENST1": ["gene1"],
        "read1_mapsto_VCRS_ENST3:mut_and_NORMAL_ENST4": ["gene4"],
        "read2_mapsto_VCRS_ENST1:mut_and_NORMAL_ENST1_and_ENST2": ["gene2"], # ["gene1", "gene2"] if run as single, but the pair only aligns to gene2, so this only gets gene2
        "read3_MM_mapsto_VCRS_ENST5:mut_and_ENST6:mut_and_NORMAL_ENST5_and_ENST6": ["gene5", "gene6"],
        "read4_MM_mapsto_VCRS_ENST7:mut_and_ENST8:mut_and_NORMAL_ENST7": ["gene7"],
        "read5_mapsto_VCRS_ENST9:mut_and_NORMAL_none": [],
        # "read0_pair_mapsto_VCRS_none_and_NORMAL_ENST1": ["gene1"],
        # "read1_pair_mapsto_VCRS_ENST4:mut_and_NORMAL_none": [],
        # "read2_pair_mapsto_VCRS_ENST2:mut_and_ENST4:mut_and_NORMAL_ENST2": ["gene2"],
        # "read3_pair_mapsto_VCRS_none_and_NORMAL_none": [],
        # "read4_pair_mapsto_VCRS_none_and_NORMAL_none": [],
        # "read5_pair_mapsto_VCRS_none_and_NORMAL_none": [],
    }

    read_to_ref_vcrs_dict_gt = {k: v for k, v in read_to_ref_vcrs_dict_gt.items() if v != []}  # remove empty keys
    read_to_ref_normal_dict_gt = {k: v for k, v in read_to_ref_normal_dict_gt.items() if v != []}  # remove empty keys

    assert read_to_ref_vcrs_dict == read_to_ref_vcrs_dict_gt  # from BUS file
    assert read_to_ref_normal_dict == read_to_ref_normal_dict_gt  # from BUS file

def test_adjust_variant_adata_by_normal_gene_matrix_bulk_mm(out_dir, NG_vcrs_reference_fasta, NG_vcrs_reference_t2g, NG_vcrs_reference_index, NG_normal_reference_fasta, NG_normal_reference_t2g, NG_normal_reference_index, NG_fastq_bulk_1, NG_fastq_bulk_2, bustools):
    technology = "BULK"
    mm = True
    union = False
    parity = "single"
    vcrs_parity = "single"
    count_reads_that_dont_pseudoalign_to_reference_genome = True

    fastqs = [str(NG_fastq_bulk_1), str(NG_fastq_bulk_2)]

    kb_count_out_vcrs = out_dir / "kb_count_vcrs"
    kb_count_out_normal = out_dir / "kb_count_normal"
    
    kb_count_command_vcrs = ["kb", "count", "-t", "2", "-k", "31", "-i", str(NG_vcrs_reference_index), "-g", str(NG_vcrs_reference_t2g), "-x", technology, "--num", "--h5ad", "-o", str(kb_count_out_vcrs)] + fastqs
    if mm:
        kb_count_command_vcrs.insert(2, "--mm")
    if union:
        kb_count_command_vcrs.insert(2, "--union")
    if technology in {"BULK", "SMARTSEQ2"}:
        kb_count_command_vcrs.insert(2, vcrs_parity)
        kb_count_command_vcrs.insert(2, "--parity")
    subprocess.run(kb_count_command_vcrs, check=True)

    kb_count_command_normal = ["kb", "count", "-t", "2", "-k", "31", "-i", str(NG_normal_reference_index), "-g", str(NG_normal_reference_t2g), "-x", technology, "--num", "--h5ad", "-o", str(kb_count_out_normal)] + fastqs
    if technology in {"BULK", "SMARTSEQ2"}:
        kb_count_command_normal.insert(2, parity)
        kb_count_command_normal.insert(2, "--parity")
    subprocess.run(kb_count_command_normal, check=True)

    adata = adjust_variant_adata_by_normal_gene_matrix(adata, kb_count_vcrs_dir=kb_count_out_vcrs, kb_count_reference_genome_dir=kb_count_out_normal, fastq_file_list=fastqs, technology=technology, t2g_standard=NG_normal_reference_t2g, adata_output_path=None, mm=mm, parity=parity, bustools=bustools, fastq_sorting_check_only=True, save_type="parquet", count_reads_that_dont_pseudoalign_to_reference_genome=count_reads_that_dont_pseudoalign_to_reference_genome)

# from this point on, I will always use union
def test_adjust_variant_adata_by_normal_gene_matrix_bulk_mm_union(out_dir, NG_vcrs_reference_fasta, NG_vcrs_reference_t2g, NG_vcrs_reference_index, NG_normal_reference_fasta, NG_normal_reference_t2g, NG_normal_reference_index, NG_fastq_bulk_1, NG_fastq_bulk_2, bustools):
    technology = "BULK"
    mm = True
    union = True
    parity = "single"
    vcrs_parity = "single"
    count_reads_that_dont_pseudoalign_to_reference_genome = True

    fastqs = [str(NG_fastq_bulk_1), str(NG_fastq_bulk_2)]

    kb_count_out_vcrs = out_dir / "kb_count_vcrs"
    kb_count_out_normal = out_dir / "kb_count_normal"
    
    kb_count_command_vcrs = ["kb", "count", "-t", "2", "-k", "31", "-i", str(NG_vcrs_reference_index), "-g", str(NG_vcrs_reference_t2g), "-x", technology, "--num", "--h5ad", "-o", str(kb_count_out_vcrs)] + fastqs
    if mm:
        kb_count_command_vcrs.insert(2, "--mm")
    if union:
        kb_count_command_vcrs.insert(2, "--union")
    if technology in {"BULK", "SMARTSEQ2"}:
        kb_count_command_vcrs.insert(2, vcrs_parity)
        kb_count_command_vcrs.insert(2, "--parity")
    subprocess.run(kb_count_command_vcrs, check=True)

    kb_count_command_normal = ["kb", "count", "-t", "2", "-k", "31", "-i", str(NG_normal_reference_index), "-g", str(NG_normal_reference_t2g), "-x", technology, "--num", "--h5ad", "-o", str(kb_count_out_normal)] + fastqs
    if technology in {"BULK", "SMARTSEQ2"}:
        kb_count_command_normal.insert(2, parity)
        kb_count_command_normal.insert(2, "--parity")
    subprocess.run(kb_count_command_normal, check=True)

    adata = adjust_variant_adata_by_normal_gene_matrix(adata, kb_count_vcrs_dir=kb_count_out_vcrs, kb_count_reference_genome_dir=kb_count_out_normal, fastq_file_list=fastqs, technology=technology, t2g_standard=NG_normal_reference_t2g, adata_output_path=None, mm=mm, parity=parity, bustools=bustools, fastq_sorting_check_only=True, save_type="parquet", count_reads_that_dont_pseudoalign_to_reference_genome=count_reads_that_dont_pseudoalign_to_reference_genome)

def test_adjust_variant_adata_by_normal_gene_matrix_bulk_toss_empty(out_dir, NG_vcrs_reference_fasta, NG_vcrs_reference_t2g, NG_vcrs_reference_index, NG_normal_reference_fasta, NG_normal_reference_t2g, NG_normal_reference_index, NG_fastq_bulk_1, NG_fastq_bulk_2, bustools):
    technology = "BULK"
    mm = False
    union = True
    parity = "single"
    vcrs_parity = "single"
    count_reads_that_dont_pseudoalign_to_reference_genome = False

    fastqs = [str(NG_fastq_bulk_1), str(NG_fastq_bulk_2)]

    kb_count_out_vcrs = out_dir / "kb_count_vcrs"
    kb_count_out_normal = out_dir / "kb_count_normal"
    
    kb_count_command_vcrs = ["kb", "count", "-t", "2", "-k", "31", "-i", str(NG_vcrs_reference_index), "-g", str(NG_vcrs_reference_t2g), "-x", technology, "--num", "--h5ad", "-o", str(kb_count_out_vcrs)] + fastqs
    if mm:
        kb_count_command_vcrs.insert(2, "--mm")
    if union:
        kb_count_command_vcrs.insert(2, "--union")
    if technology in {"BULK", "SMARTSEQ2"}:
        kb_count_command_vcrs.insert(2, vcrs_parity)
        kb_count_command_vcrs.insert(2, "--parity")
    subprocess.run(kb_count_command_vcrs, check=True)

    kb_count_command_normal = ["kb", "count", "-t", "2", "-k", "31", "-i", str(NG_normal_reference_index), "-g", str(NG_normal_reference_t2g), "-x", technology, "--num", "--h5ad", "-o", str(kb_count_out_normal)] + fastqs
    if technology in {"BULK", "SMARTSEQ2"}:
        kb_count_command_normal.insert(2, parity)
        kb_count_command_normal.insert(2, "--parity")
    subprocess.run(kb_count_command_normal, check=True)

    adata = adjust_variant_adata_by_normal_gene_matrix(adata, kb_count_vcrs_dir=kb_count_out_vcrs, kb_count_reference_genome_dir=kb_count_out_normal, fastq_file_list=fastqs, technology=technology, t2g_standard=NG_normal_reference_t2g, adata_output_path=None, mm=mm, parity=parity, bustools=bustools, fastq_sorting_check_only=True, save_type="parquet", count_reads_that_dont_pseudoalign_to_reference_genome=count_reads_that_dont_pseudoalign_to_reference_genome)

def test_adjust_variant_adata_by_normal_gene_matrix_10x(out_dir, NG_vcrs_reference_fasta, NG_vcrs_reference_t2g, NG_vcrs_reference_index, NG_normal_reference_fasta, NG_normal_reference_t2g, NG_normal_reference_index, NG_fastq_10xv3_R1, NG_fastq_10xv3_R2, bustools):
    technology = "10XV3"
    mm = False
    union = True
    parity = None
    vcrs_parity = None
    count_reads_that_dont_pseudoalign_to_reference_genome = True

    fastqs = [str(NG_fastq_10xv3_R1), str(NG_fastq_10xv3_R2)]

    kb_count_out_vcrs = out_dir / "kb_count_vcrs"
    kb_count_out_normal = out_dir / "kb_count_normal"
    
    kb_count_command_vcrs = ["kb", "count", "-t", "2", "-k", "31", "-i", str(NG_vcrs_reference_index), "-g", str(NG_vcrs_reference_t2g), "-x", technology, "--num", "--h5ad", "-o", str(kb_count_out_vcrs)] + fastqs
    if mm:
        kb_count_command_vcrs.insert(2, "--mm")
    if union:
        kb_count_command_vcrs.insert(2, "--union")
    if technology in {"BULK", "SMARTSEQ2"}:
        kb_count_command_vcrs.insert(2, vcrs_parity)
        kb_count_command_vcrs.insert(2, "--parity")
    subprocess.run(kb_count_command_vcrs, check=True)

    kb_count_command_normal = ["kb", "count", "-t", "2", "-k", "31", "-i", str(NG_normal_reference_index), "-g", str(NG_normal_reference_t2g), "-x", technology, "--num", "--h5ad", "-o", str(kb_count_out_normal)] + fastqs
    if technology in {"BULK", "SMARTSEQ2"}:
        kb_count_command_normal.insert(2, parity)
        kb_count_command_normal.insert(2, "--parity")
    subprocess.run(kb_count_command_normal, check=True)

    adata = adjust_variant_adata_by_normal_gene_matrix(adata, kb_count_vcrs_dir=kb_count_out_vcrs, kb_count_reference_genome_dir=kb_count_out_normal, fastq_file_list=fastqs, technology=technology, t2g_standard=NG_normal_reference_t2g, adata_output_path=None, mm=mm, parity=parity, bustools=bustools, fastq_sorting_check_only=True, save_type="parquet", count_reads_that_dont_pseudoalign_to_reference_genome=count_reads_that_dont_pseudoalign_to_reference_genome)

def test_adjust_variant_adata_by_normal_gene_matrix_10x_mm(out_dir, NG_vcrs_reference_fasta, NG_vcrs_reference_t2g, NG_vcrs_reference_index, NG_normal_reference_fasta, NG_normal_reference_t2g, NG_normal_reference_index, NG_fastq_10xv3_R1, NG_fastq_10xv3_R2, bustools):
    technology = "10XV3"
    mm = True
    union = True
    parity = None
    vcrs_parity = None
    count_reads_that_dont_pseudoalign_to_reference_genome = True

    fastqs = [str(NG_fastq_10xv3_R1), str(NG_fastq_10xv3_R2)]

    kb_count_out_vcrs = out_dir / "kb_count_vcrs"
    kb_count_out_normal = out_dir / "kb_count_normal"
    
    kb_count_command_vcrs = ["kb", "count", "-t", "2", "-k", "31", "-i", str(NG_vcrs_reference_index), "-g", str(NG_vcrs_reference_t2g), "-x", technology, "--num", "--h5ad", "-o", str(kb_count_out_vcrs)] + fastqs
    if mm:
        kb_count_command_vcrs.insert(2, "--mm")
    if union:
        kb_count_command_vcrs.insert(2, "--union")
    if technology in {"BULK", "SMARTSEQ2"}:
        kb_count_command_vcrs.insert(2, vcrs_parity)
        kb_count_command_vcrs.insert(2, "--parity")
    subprocess.run(kb_count_command_vcrs, check=True)

    kb_count_command_normal = ["kb", "count", "-t", "2", "-k", "31", "-i", str(NG_normal_reference_index), "-g", str(NG_normal_reference_t2g), "-x", technology, "--num", "--h5ad", "-o", str(kb_count_out_normal)] + fastqs
    if technology in {"BULK", "SMARTSEQ2"}:
        kb_count_command_normal.insert(2, parity)
        kb_count_command_normal.insert(2, "--parity")
    subprocess.run(kb_count_command_normal, check=True)

    adata = adjust_variant_adata_by_normal_gene_matrix(adata, kb_count_vcrs_dir=kb_count_out_vcrs, kb_count_reference_genome_dir=kb_count_out_normal, fastq_file_list=fastqs, technology=technology, t2g_standard=NG_normal_reference_t2g, adata_output_path=None, mm=mm, parity=parity, bustools=bustools, fastq_sorting_check_only=True, save_type="parquet", count_reads_that_dont_pseudoalign_to_reference_genome=count_reads_that_dont_pseudoalign_to_reference_genome)


