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
        ">ENST1:c.1mut\n"
        "AGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGTAC\n"
        ">ENST2:c.1mut\n"
        "CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC\n"    #!!! AAGGAAGGAAAAGGAAGGAAAAGGAAGGAAAAGGAAGGAAAAGGAAGGAAAAGGAAGGAAAAGGAAGGCCCCCAAAAACCCCCAAAAACCCCCAAAAACCCCCAAAAA used to be appended
        ">ENST3:c.1mut\n"
        "ATGCTGACTGACTGCTGACTGCTAGCTGACGTCATCAGTACGTACGATGCTGACTGACTGCTGACTGCTAGCTGACGTCATCAGTACGTACGCCCCACCCCACCCCACCCCACCCCACCCCACCCCACCCCACCCCACCCCACCCCACCCCA\n"
        ">ENST4:c.1mut\n"
        "AAGGAAGGAAAAGGAAGGAAAAGGAAGGAAAAGGAAGGAAAAGGAAGGAAAAGGAAGGAAAAGGAAGGAAA\n"
        ">ENST5:c.1mut\n"
        "TTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTA\n"
        ">ENST6:c.1mut\n"
        "TTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTG\n"
        ">ENST7:c.1mut\n"
        "AGCCATCGACTAGCTACGACTGCATGCTGACTGCATACGCATGACGTCAGCGAAAAAATACTCAGTT\n"
        ">ENST8:c.1mut\n"
        "AGCCATCGACTAGCTACGACTGCATGCTGACTGCATACGCATGACGTCAGCGAAAAAATACTCAGTG\n"
        ">ENST9:c.1mut\n"
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
        "CCCCCAAAAACCCCCAAAAACCCCCAAAAACCCCCAAAAACCCCCAAAAACCCCCAAACCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCAAGGAAGGAAAAGGAAGGAAAAGGAAGGAAAAGGAAGGAAAA\n"
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

    adata = adjust_variant_adata_by_normal_gene_matrix(kb_count_vcrs_dir=kb_count_out_vcrs, kb_count_reference_genome_dir=kb_count_out_normal, fastq_file_list=fastqs, technology=technology, t2g_standard=NG_normal_reference_t2g, adata_output_path=None, mm=mm, parity=parity, bustools=bustools, fastq_sorting_check_only=True, save_type="parquet", count_reads_that_dont_pseudoalign_to_reference_genome=count_reads_that_dont_pseudoalign_to_reference_genome, variant_source="transcriptome", add_fastq_headers=True)

    # Convert adata.X to a DataFrame (if not already)
    matrix_df = pd.DataFrame(adata.X.toarray() if hasattr(adata.X, "toarray") else adata.X, 
                    index=adata.obs.index, 
                    columns=adata.var.index)

    count_matrix_data_gt = {
        "AAAAAAAAAAAAAAAA": {"ENST1:c.1mut": 2, "ENST2:c.1mut": 0, "ENST3:c.1mut": 0, "ENST4:c.1mut": 0, "ENST5:c.1mut": 0, "ENST6:c.1mut": 0, "ENST7:c.1mut": 1, "ENST8:c.1mut": 0, "ENST9:c.1mut": 0},
    }
    if mm:
        count_matrix_data_gt["AAAAAAAAAAAAAAAA"]["ENST5:c.1mut"] += 1
        count_matrix_data_gt["AAAAAAAAAAAAAAAA"]["ENST6:c.1mut"] += 1
    if count_reads_that_dont_pseudoalign_to_reference_genome:
        count_matrix_data_gt["AAAAAAAAAAAAAAAA"]["ENST9:c.1mut"] += 1

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
        "read0_mapsto_ENST1:mut_and_NORMAL_ENST1": ["ENST1:c.1mut"],
        "read1_mapsto_VCRS_ENST3:mut_and_NORMAL_ENST4": ["ENST3:c.1mut"],
        "read2_mapsto_VCRS_ENST1:mut_and_NORMAL_ENST1_and_ENST2": ["ENST1:c.1mut"],
        "read3_MM_mapsto_VCRS_ENST5:mut_and_ENST6:mut_and_NORMAL_ENST5_and_ENST6": ["ENST5:c.1mut", "ENST6:c.1mut"],
        "read4_MM_mapsto_VCRS_ENST7:mut_and_ENST8:mut_and_NORMAL_ENST7": ["ENST7:c.1mut", "ENST8:c.1mut"],
        "read5_mapsto_VCRS_ENST9:mut_and_NORMAL_none": ["ENST9:c.1mut"],
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

# from this point on, I will always use union
def test_adjust_variant_adata_by_normal_gene_matrix_bulk_paired_but_run_as_single(out_dir, NG_vcrs_reference_fasta, NG_vcrs_reference_t2g, NG_vcrs_reference_index, NG_normal_reference_fasta, NG_normal_reference_t2g, NG_normal_reference_index, NG_fastq_bulk_1, NG_fastq_bulk_2, bustools):
    technology = "BULK"  # includes non-multiplexed SMARTSEQ2
    mm = False
    union = True
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

    adata = adjust_variant_adata_by_normal_gene_matrix(kb_count_vcrs_dir=kb_count_out_vcrs, kb_count_reference_genome_dir=kb_count_out_normal, fastq_file_list=fastqs, technology=technology, t2g_standard=NG_normal_reference_t2g, adata_output_path=None, mm=mm, parity=parity, bustools=bustools, fastq_sorting_check_only=True, save_type="parquet", count_reads_that_dont_pseudoalign_to_reference_genome=count_reads_that_dont_pseudoalign_to_reference_genome, variant_source="transcriptome", add_fastq_headers=True)

    # Convert adata.X to a DataFrame (if not already)
    matrix_df = pd.DataFrame(adata.X.toarray() if hasattr(adata.X, "toarray") else adata.X, 
                    index=adata.obs.index, 
                    columns=adata.var.index)

    count_matrix_data_gt = {
        "AAAAAAAAAAAAAAAA": {"ENST1:c.1mut": 2, "ENST2:c.1mut": 0, "ENST3:c.1mut": 0, "ENST4:c.1mut": 0, "ENST5:c.1mut": 0, "ENST6:c.1mut": 0, "ENST7:c.1mut": 1, "ENST8:c.1mut": 0, "ENST9:c.1mut": 0},
    }
    if mm:
        count_matrix_data_gt["AAAAAAAAAAAAAAAA"]["ENST5:c.1mut"] += 1
        count_matrix_data_gt["AAAAAAAAAAAAAAAA"]["ENST6:c.1mut"] += 1
    if count_reads_that_dont_pseudoalign_to_reference_genome:
        count_matrix_data_gt["AAAAAAAAAAAAAAAA"]["ENST9:c.1mut"] += 1
    if parity == "paired" and vcrs_parity == "single":
        count_matrix_data_gt["AAAAAAAAAAAAAAAA"]["ENST1:c.1mut"] -= 1
        count_matrix_data_gt["AAAAAAAAAAAAAAAA"]["ENST2:c.1mut"] += 1
        count_matrix_data_gt["AAAAAAAAAAAAAAAA"]["ENST4:c.1mut"] += 1

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
        "read0_mapsto_ENST1:mut_and_NORMAL_ENST1": ["ENST1:c.1mut"],
        "read1_mapsto_VCRS_ENST3:mut_and_NORMAL_ENST4": ["ENST3:c.1mut"],
        "read2_mapsto_VCRS_ENST1:mut_and_NORMAL_ENST1_and_ENST2": ["ENST1:c.1mut"],
        "read3_MM_mapsto_VCRS_ENST5:mut_and_ENST6:mut_and_NORMAL_ENST5_and_ENST6": ["ENST5:c.1mut", "ENST6:c.1mut"],
        "read4_MM_mapsto_VCRS_ENST7:mut_and_ENST8:mut_and_NORMAL_ENST7": ["ENST7:c.1mut", "ENST8:c.1mut"],
        "read5_mapsto_VCRS_ENST9:mut_and_NORMAL_none": ["ENST9:c.1mut"],
        "read0_pair_mapsto_VCRS_none_and_NORMAL_ENST1": [],
        "read1_pair_mapsto_VCRS_ENST4:mut_and_NORMAL_none": ["ENST4:c.1mut"],
        "read2_pair_mapsto_VCRS_ENST2:mut_and_ENST4:mut_and_NORMAL_ENST2": ["ENST2:c.1mut", "ENST4:c.1mut"],
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
    union = True
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

    adata = adjust_variant_adata_by_normal_gene_matrix(kb_count_vcrs_dir=kb_count_out_vcrs, kb_count_reference_genome_dir=kb_count_out_normal, fastq_file_list=fastqs, technology=technology, t2g_standard=NG_normal_reference_t2g, adata_output_path=None, mm=mm, parity=parity, bustools=bustools, fastq_sorting_check_only=True, save_type="parquet", count_reads_that_dont_pseudoalign_to_reference_genome=count_reads_that_dont_pseudoalign_to_reference_genome, variant_source="transcriptome", add_fastq_headers=True)

    # Convert adata.X to a DataFrame (if not already)
    matrix_df = pd.DataFrame(adata.X.toarray() if hasattr(adata.X, "toarray") else adata.X, 
                    index=adata.obs.index, 
                    columns=adata.var.index)

    count_matrix_data_gt = {
        "AAAAAAAAAAAAAAAA": {"ENST1:c.1mut": 2, "ENST2:c.1mut": 0, "ENST3:c.1mut": 0, "ENST4:c.1mut": 0, "ENST5:c.1mut": 0, "ENST6:c.1mut": 0, "ENST7:c.1mut": 1, "ENST8:c.1mut": 0, "ENST9:c.1mut": 0},
    }
    if mm:
        count_matrix_data_gt["AAAAAAAAAAAAAAAA"]["ENST5:c.1mut"] += 1
        count_matrix_data_gt["AAAAAAAAAAAAAAAA"]["ENST6:c.1mut"] += 1
    if count_reads_that_dont_pseudoalign_to_reference_genome:
        count_matrix_data_gt["AAAAAAAAAAAAAAAA"]["ENST9:c.1mut"] += 1
    if parity == "paired" and vcrs_parity == "single":
        count_matrix_data_gt["AAAAAAAAAAAAAAAA"]["ENST1:c.1mut"] -= 1
        count_matrix_data_gt["AAAAAAAAAAAAAAAA"]["ENST2:c.1mut"] += 1
        count_matrix_data_gt["AAAAAAAAAAAAAAAA"]["ENST4:c.1mut"] += 1

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
        "read0_mapsto_ENST1:mut_and_NORMAL_ENST1": ["ENST1:c.1mut"],
        "read1_mapsto_VCRS_ENST3:mut_and_NORMAL_ENST4": ["ENST3:c.1mut"],
        "read2_mapsto_VCRS_ENST1:mut_and_NORMAL_ENST1_and_ENST2": ["ENST1:c.1mut"],
        "read3_MM_mapsto_VCRS_ENST5:mut_and_ENST6:mut_and_NORMAL_ENST5_and_ENST6": ["ENST5:c.1mut", "ENST6:c.1mut"],
        "read4_MM_mapsto_VCRS_ENST7:mut_and_ENST8:mut_and_NORMAL_ENST7": ["ENST7:c.1mut", "ENST8:c.1mut"],
        "read5_mapsto_VCRS_ENST9:mut_and_NORMAL_none": ["ENST9:c.1mut"],
        "read0_pair_mapsto_VCRS_none_and_NORMAL_ENST1": [],
        "read1_pair_mapsto_VCRS_ENST4:mut_and_NORMAL_none": ["ENST4:c.1mut"],
        "read2_pair_mapsto_VCRS_ENST2:mut_and_ENST4:mut_and_NORMAL_ENST2": ["ENST2:c.1mut", "ENST4:c.1mut"],
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

def test_adjust_variant_adata_by_normal_gene_matrix_bulk_toss_empty(out_dir, NG_vcrs_reference_fasta, NG_vcrs_reference_t2g, NG_vcrs_reference_index, NG_normal_reference_fasta, NG_normal_reference_t2g, NG_normal_reference_index, NG_fastq_bulk_1, NG_fastq_bulk_2, bustools):
    technology = "BULK"
    mm = False
    union = True
    parity = "paired"
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

    adata = adjust_variant_adata_by_normal_gene_matrix(kb_count_vcrs_dir=kb_count_out_vcrs, kb_count_reference_genome_dir=kb_count_out_normal, fastq_file_list=fastqs, technology=technology, t2g_standard=NG_normal_reference_t2g, adata_output_path=None, mm=mm, parity=parity, bustools=bustools, fastq_sorting_check_only=True, save_type="parquet", count_reads_that_dont_pseudoalign_to_reference_genome=count_reads_that_dont_pseudoalign_to_reference_genome, variant_source="transcriptome", add_fastq_headers=True)

    # Convert adata.X to a DataFrame (if not already)
    matrix_df = pd.DataFrame(adata.X.toarray() if hasattr(adata.X, "toarray") else adata.X, 
                    index=adata.obs.index, 
                    columns=adata.var.index)

    count_matrix_data_gt = {
        "AAAAAAAAAAAAAAAA": {"ENST1:c.1mut": 2, "ENST2:c.1mut": 0, "ENST3:c.1mut": 0, "ENST4:c.1mut": 0, "ENST5:c.1mut": 0, "ENST6:c.1mut": 0, "ENST7:c.1mut": 1, "ENST8:c.1mut": 0, "ENST9:c.1mut": 0},
    }
    if mm:
        count_matrix_data_gt["AAAAAAAAAAAAAAAA"]["ENST5:c.1mut"] += 1
        count_matrix_data_gt["AAAAAAAAAAAAAAAA"]["ENST6:c.1mut"] += 1
    if count_reads_that_dont_pseudoalign_to_reference_genome:
        count_matrix_data_gt["AAAAAAAAAAAAAAAA"]["ENST9:c.1mut"] += 1
    if parity == "paired" and vcrs_parity == "single":
        count_matrix_data_gt["AAAAAAAAAAAAAAAA"]["ENST1:c.1mut"] -= 1
        count_matrix_data_gt["AAAAAAAAAAAAAAAA"]["ENST2:c.1mut"] += 1
        count_matrix_data_gt["AAAAAAAAAAAAAAAA"]["ENST4:c.1mut"] += 1

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
        "read0_mapsto_ENST1:mut_and_NORMAL_ENST1": ["ENST1:c.1mut"],
        "read1_mapsto_VCRS_ENST3:mut_and_NORMAL_ENST4": ["ENST3:c.1mut"],
        "read2_mapsto_VCRS_ENST1:mut_and_NORMAL_ENST1_and_ENST2": ["ENST1:c.1mut"],
        "read3_MM_mapsto_VCRS_ENST5:mut_and_ENST6:mut_and_NORMAL_ENST5_and_ENST6": ["ENST5:c.1mut", "ENST6:c.1mut"],
        "read4_MM_mapsto_VCRS_ENST7:mut_and_ENST8:mut_and_NORMAL_ENST7": ["ENST7:c.1mut", "ENST8:c.1mut"],
        "read5_mapsto_VCRS_ENST9:mut_and_NORMAL_none": ["ENST9:c.1mut"],
        "read0_pair_mapsto_VCRS_none_and_NORMAL_ENST1": [],
        "read1_pair_mapsto_VCRS_ENST4:mut_and_NORMAL_none": ["ENST4:c.1mut"],
        "read2_pair_mapsto_VCRS_ENST2:mut_and_ENST4:mut_and_NORMAL_ENST2": ["ENST2:c.1mut", "ENST4:c.1mut"],
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



























@pytest.fixture
def NG_fastq_10xv3_R1(tmp_path):
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
def NG_fastq_10xv3_R2(tmp_path):
    fastq_content = (
        "@read0_mapsto_ENST1:c.1mut_R2\n"
        "CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC\n"
        "+\n"
        "IIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIII\n"
        "@read1_mapsto_ENST1:c.1mut_same_barcode_and_umi_R2\n"
        "CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC\n"
        "+\n"
        "IIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIII\n"
        "@read2_mapsto_ENST1:c.1mut_different_barcode_R2\n"
        "CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC\n"
        "+\n"
        "IIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIII\n"
        "@read3_mapsto_ENST2:c.1mut_same_barcode_R2\n"
        "AGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGA\n"
        "+\n"
        "IIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIII\n"
        "@read4_mapsto_ENST2:c.1mut_different_umi_R2\n"
        "AGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGA\n"
        "+\n"
        "IIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIII\n"
        "@read5_mapsto_ENST1:c.1mut_and_ENST2:c.1mut_union_R2\n"
        "CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGA\n"
        "+\n"
        "IIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIII\n"
        "@read6_mapsto_ENST3:c.1mut_and_ENST4:c.1mut_and_ENST5:c.1mut_multimap_R2\n"
        "CCACCACCACCACCACCACCACCACCACCACCACCACCACCACCACCACCACCACCACCACCACCACCACCACCACCACCACCACCACCACCACCACCACCACCACCACCACCACCACCACCACCACCACCACCACCACCACCACCACCA\n"
        "+\n"
        "IIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIII\n"
        "@read7_mapsto_ENST3:c.1mut_and_ENST4:c.1mut_and_ENST5:c.1mut_multimap_different_umi_R2\n"
        "CCACCACCACCACCACCACCACCACCACCACCACCACCACCACCACCACCACCACCACCACCACCACCACCACCACCACCACCACCACCACCACCACCACCACCACCACCACCACCACCACCACCACCACCACCACCACCACCACCACCA\n"
        "+\n"
        "IIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIII\n"
        "@read8_mapsto_ENST1:c.1mut_barcode1_but_hamming_distance2_R2\n"
        "CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC\n"
        "+\n"
        "IIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIII\n"
        "@read9_mapsto_ENST1:c.1mut_and_ENST2:c.1mut_barcode1_same_umi_as_read10_R2\n"
        "CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGA\n"
        "+\n"
        "IIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIII\n"
        "@read10_mapsto_ENST1:c.1mut_and_ENST6:c.1mut_barcode1_same_umi_as_read9_R2\n"
        "CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCGATGATGATGATGATGATGATGATGATGATGATGATGATGATGATGATGATGATGATGATGATGATGATGATGATGATGATGATGA\n"
        "+\n"
        "IIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIII\n"
    )
    fastq_path = tmp_path / "test_sequences_R2.fastq"
    fastq_path.write_text(fastq_content)
    return fastq_path


@pytest.fixture
def NG_vcrs_reference_fasta_sc(tmp_path):
    fasta_content = (
        ">ENST1:c.1mut\n"
        "CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC\n"
        ">ENST2:c.1mut\n"
        "AGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGA\n"
        ">ENST3:c.1mut\n"
        "CCACCACCACCACCACCACCACCACCACCACCACCACCACCACCACCACCACCACCACCACCACCACCACCACCACCACCACCACCACCACCACCACCACCAG\n"
        ">ENST4:c.1mut\n"
        "CCACCACCACCACCACCACCACCACCACCACCACCACCACCACCACCACCACCACCACCACCACCACCACCACCACCACCACCACCACCACCACCACCACCAT\n"
        ">ENST5:c.1mut\n"
        "CCACCACCACCACCACCACCACCACCACCACCACCACCACCACCACCACCACCACCACCACCACCACCACCACCACCACCACCACCACCACCACCACCACCAA\n"
        ">ENST6:c.1mut\n"
        "GATGATGATGATGATGATGATGATGATGATGATGATGATGATGATGATGATGATGATGATGATGATGATGATGATGATGATGATGATGATGATGATGATGATG\n"
    )
    tmp_fasta_path = tmp_path / "vcrs_sc.fasta"
    tmp_fasta_path.write_text(fasta_content)
    return tmp_fasta_path

@pytest.fixture
def NG_vcrs_reference_t2g_sc(tmp_path, NG_vcrs_reference_fasta_sc):
    tmp_t2g_path = tmp_path / "vcrs_t2g_sc.txt"
    create_identity_t2g(NG_vcrs_reference_fasta_sc, tmp_t2g_path)
    return tmp_t2g_path

@pytest.fixture
def NG_vcrs_reference_index_sc(tmp_path, NG_vcrs_reference_fasta_sc):
    tmp_index_path = tmp_path / "mutation_index_sc.idx"
    kb_ref_command_vcrs = ["kb", "ref", "--workflow", "custom", "-t", "2", "-i", str(tmp_index_path), "--d-list", "None", "-k", "31", str(NG_vcrs_reference_fasta_sc)]
    subprocess.run(kb_ref_command_vcrs, check=True)
    return tmp_index_path

@pytest.fixture
def NG_normal_reference_fasta_sc(tmp_path):
    fasta_content = (
        ">ENST1\n"
        "CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC\n"
        ">ENST2\n"
        "AGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGAGA\n"
        ">ENST3\n"
        "CCACCACCACCACCACCACCACCACCACCACCACCACCACCACCACCACCACCACCACCACCACCACCACCACCACCACCACCACCACCACCACCACCACCAG\n"
        ">ENST4\n"
        "CCACCACCACCACCACCACCACCACCACCACCACCACCACCACCACCACCACCACCACCACCACCACCACCACCACCACCACCACCACCACCACCACCACCAT\n"
        ">ENST5\n"
        "CCACCACCACCACCACCACCACCACCACCACCACCACCACCACCACCACCACCACCACCACCACCACCACCACCACCACCACCACCACCACCACCACCACCAA\n"
        ">ENST6\n"
        "GATGATGATGATGATGATGATGATGATGATGATGATGATGATGATGATGATGATGATGATGATGATGATGATGATGATGATGATGATGATGATGATGATGATG\n"
    )
    tmp_fasta_path = tmp_path / "normal_sc.fasta"
    tmp_fasta_path.write_text(fasta_content)
    return tmp_fasta_path

@pytest.fixture
def NG_normal_reference_t2g_sc(tmp_path):
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
    tmp_t2g_path = tmp_path / "normal_t2g_sc.txt"
    tmp_t2g_path.write_text(t2g_content)
    return tmp_t2g_path

@pytest.fixture
def NG_normal_reference_index_sc(tmp_path, NG_normal_reference_t2g_sc, NG_normal_reference_fasta_sc):
    tmp_index_path = tmp_path / "normal_index_sc.idx"
    kb_ref_command_normal = ["kb", "ref", "-t", "2", "--workflow", "custom", "-i", str(tmp_index_path), "-g", str(NG_normal_reference_t2g_sc), "--d-list", "None", "-k", "31", str(NG_normal_reference_fasta_sc)]
    subprocess.run(kb_ref_command_normal, check=True)
    return tmp_index_path



def test_adjust_variant_adata_by_normal_gene_matrix_10x(out_dir, NG_vcrs_reference_fasta_sc, NG_vcrs_reference_t2g_sc, NG_vcrs_reference_index_sc, NG_normal_reference_fasta_sc, NG_normal_reference_t2g_sc, NG_normal_reference_index_sc, NG_fastq_10xv3_R1, NG_fastq_10xv3_R2, bustools):
    technology = "10XV3"
    mm = True
    union = True
    parity = "paired"
    vcrs_parity = "single"
    count_reads_that_dont_pseudoalign_to_reference_genome = True

    fastqs = [str(NG_fastq_10xv3_R1), str(NG_fastq_10xv3_R2)]

    kb_count_out_vcrs = out_dir / "kb_count_vcrs"
    kb_count_out_normal = out_dir / "kb_count_normal"
    
    kb_count_command_vcrs = ["kb", "count", "-t", "2", "-k", "31", "-i", str(NG_vcrs_reference_index_sc), "-g", str(NG_vcrs_reference_t2g_sc), "-x", technology, "--num", "--h5ad", "-o", str(kb_count_out_vcrs)] + fastqs
    if mm:
        kb_count_command_vcrs.insert(2, "--mm")
    if union:
        kb_count_command_vcrs.insert(2, "--union")
    if technology in {"BULK", "SMARTSEQ2"}:
        kb_count_command_vcrs.insert(2, vcrs_parity)
        kb_count_command_vcrs.insert(2, "--parity")
    subprocess.run(kb_count_command_vcrs, check=True)

    kb_count_command_normal = ["kb", "count", "-t", "2", "-k", "31", "-i", str(NG_normal_reference_index_sc), "-g", str(NG_normal_reference_t2g_sc), "-x", technology, "--num", "--h5ad", "-o", str(kb_count_out_normal)] + fastqs
    if technology in {"BULK", "SMARTSEQ2"}:
        kb_count_command_normal.insert(2, parity)
        kb_count_command_normal.insert(2, "--parity")
    subprocess.run(kb_count_command_normal, check=True)

    adata = adjust_variant_adata_by_normal_gene_matrix(kb_count_vcrs_dir=kb_count_out_vcrs, kb_count_reference_genome_dir=kb_count_out_normal, fastq_file_list=fastqs, technology=technology, t2g_standard=NG_normal_reference_t2g_sc, adata_output_path=None, mm=mm, parity=parity, bustools=bustools, fastq_sorting_check_only=True, save_type="parquet", count_reads_that_dont_pseudoalign_to_reference_genome=count_reads_that_dont_pseudoalign_to_reference_genome, variant_source="transcriptome", add_fastq_headers=True)

    # Convert adata.X to a DataFrame (if not already)
    matrix_df = pd.DataFrame(adata.X.toarray() if hasattr(adata.X, "toarray") else adata.X, 
                    index=adata.obs.index, 
                    columns=adata.var.index)
    
    read_to_ref_dict_vcrs_gt = {
        'read0_mapsto_ENST1:c.1mut_R2': ['ENST1:c.1mut'],  # count_matrix_data_gt["AAACCCAAGAAACACT"]["ENST1:c.1mut"] = 1
        'read1_mapsto_ENST1:c.1mut_same_barcode_and_umi_R2': ['ENST1:c.1mut'],  # because it has duplicate UMI as read0, it doesn't count for count matrix
        'read2_mapsto_ENST1:c.1mut_different_barcode_R2': ['ENST1:c.1mut'],  # count_matrix_data_gt["TATCAGGAGCTAAGTG"]["ENST1:c.1mut"] = 1
        'read3_mapsto_ENST2:c.1mut_same_barcode_R2': ['ENST2:c.1mut'],  # count_matrix_data_gt["TATCAGGAGCTAAGTG"]["ENST2:c.1mut"] = 1
        'read4_mapsto_ENST2:c.1mut_different_umi_R2': ['ENST2:c.1mut'],  # count_matrix_data_gt["TATCAGGAGCTAAGTG"]["ENST2:c.1mut"] = 2
        'read5_mapsto_ENST1:c.1mut_and_ENST2:c.1mut_union_R2': [],  # doesn't count for count matrix OR show up in bus file unless --union is used
        'read6_mapsto_ENST3:c.1mut_and_ENST4:c.1mut_and_ENST5:c.1mut_multimap_R2': ['ENST3:c.1mut', 'ENST4:c.1mut', 'ENST5:c.1mut'],  # doesn't count for count matrix unless --mm is used (but shows up in bus file regardless)
        'read7_mapsto_ENST3:c.1mut_and_ENST4:c.1mut_and_ENST5:c.1mut_multimap_different_umi_R2': ['ENST3:c.1mut', 'ENST4:c.1mut', 'ENST5:c.1mut'],  # doesn't count for count matrix unless --mm is used (but shows up in bus file regardless)
        'read8_mapsto_ENST1:c.1mut_barcode1_but_hamming_distance2_R2': ['ENST1:c.1mut'],  # count_matrix_data_gt["AAACCCAAGAAACACT"]["ENST1:c.1mut"] = 2
        'read9_mapsto_ENST1:c.1mut_and_ENST2:c.1mut_barcode1_same_umi_as_read10_R2': [],  # doesn't count for ENST1:c.1mut without union (won't show up in BUS file without union); ENST2:c.1mut doesn't count regardless because it has the same UMI as read10, but read10 doesn't map to ENST2:c.1mut 
        'read10_mapsto_ENST1:c.1mut_and_ENST6:c.1mut_barcode1_same_umi_as_read9_R2': [],  # doesn't count due to same barcode and UMI as read9
    }
    if union:
        read_to_ref_dict_vcrs_gt['read5_mapsto_ENST1:c.1mut_and_ENST2:c.1mut_union_R2'] = ['ENST1:c.1mut', 'ENST2:c.1mut']
        read_to_ref_dict_vcrs_gt['read9_mapsto_ENST1:c.1mut_and_ENST2:c.1mut_barcode1_same_umi_as_read10_R2'] = ['ENST1:c.1mut', 'ENST2:c.1mut']
        read_to_ref_dict_vcrs_gt['read10_mapsto_ENST1:c.1mut_and_ENST6:c.1mut_barcode1_same_umi_as_read9_R2'] = ['ENST1:c.1mut', 'ENST6:c.1mut']

    bus_df_vcrs = pd.read_parquet(kb_count_out_vcrs / "bus_df.parquet", columns=["fastq_header", "gene_names"])
    bus_df_vcrs.rename(columns={"gene_names": "vcrs_names"}, inplace=True)
    read_to_ref_dict_vcrs = dict(zip(bus_df_vcrs['fastq_header'], bus_df_vcrs['vcrs_names']))
    
    read_to_ref_dict_vcrs_gt = {k: v for k, v in read_to_ref_dict_vcrs_gt.items() if v != []}  # remove empty keys

    count_matrix_data_gt = {
        "AAACCCAAGAAACACT": {"ENST1:c.1mut": 2, "ENST2:c.1mut": 0, "ENST3:c.1mut": 0, "ENST4:c.1mut": 0, "ENST5:c.1mut": 0, "ENST6:c.1mut": 0},
        "TATCAGGAGCTAAGTG": {"ENST1:c.1mut": 1, "ENST2:c.1mut": 1, "ENST3:c.1mut": 0, "ENST4:c.1mut": 0, "ENST5:c.1mut": 0, "ENST6:c.1mut": 0},
    }
    if union and mm:  # notably, won't show up in count matrix unless mm is also used
        count_matrix_data_gt["AAACCCAAGAAACACT"]["ENST1:c.1mut"] += (1/1)  # each unioned read adds (1/n), where n is the number of VCRSs to which the read maps - this comes from read9/10
        count_matrix_data_gt["AAACCCAAGAAACACT"]["ENST2:c.1mut"] += (1/1)  # each unioned read adds (1/n), where n is the number of VCRSs to which the read maps - this comes from read9/10
        count_matrix_data_gt["TATCAGGAGCTAAGTG"]["ENST1:c.1mut"] += (1/1)  # each unioned read adds (1/n), where n is the number of VCRSs to which the read maps - this comes from read5
        count_matrix_data_gt["TATCAGGAGCTAAGTG"]["ENST2:c.1mut"] += (1/1)  # each unioned read adds (1/n), where n is the number of VCRSs to which the read maps - this comes from read5
    if mm:
        count_matrix_data_gt["TATCAGGAGCTAAGTG"]["ENST3:c.1mut"] += (1/1) * 2  # each multimapped read adds (1/n), where n is the number of VCRSs to which the read maps, and reads 6 and 7 are both multimappers hence the *2
        count_matrix_data_gt["TATCAGGAGCTAAGTG"]["ENST4:c.1mut"] += (1/1) * 2  # each multimapped read adds (1/n), where n is the number of VCRSs to which the read maps, and reads 6 and 7 are both multimappers hence the *2
        count_matrix_data_gt["TATCAGGAGCTAAGTG"]["ENST5:c.1mut"] += (1/1) * 2  # each multimapped read adds (1/n), where n is the number of VCRSs to which the read maps, and reads 6 and 7 are both multimappers hence the *2
    # count_matrix_data_gt_with_multimap = {
    #     "AAACCCAAGAAACACT": {"ENST1:c.1mut": 3, "ENST2:c.1mut": 0, "ENST3:c.1mut": 0, "ENST4:c.1mut": 0, "ENST5:c.1mut": 0, "ENST6:c.1mut": 0},
    #     "TATCAGGAGCTAAGTG": {"ENST1:c.1mut": 1.5, "ENST2:c.1mut": 2.5, "ENST3:c.1mut": 0.67, "ENST4:c.1mut": 0.67, "ENST5:c.1mut": 0.67, "ENST6:c.1mut": 0},
    # }

    matrix_df_gt = pd.DataFrame(count_matrix_data_gt)
    matrix_df_gt = matrix_df_gt.T
    matrix_df_gt = matrix_df_gt.astype("float64")
    
    # ensure no rounding errors (eg 0.6667 vs 0.66666666...)
    matrix_df = matrix_df.round(3)
    matrix_df_gt = matrix_df_gt.round(3)

    assert read_to_ref_dict_vcrs == read_to_ref_dict_vcrs_gt  # from BUS file
    assert matrix_df.equals(matrix_df_gt)  # from adata