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
    remove_variants_from_adata_for_stranded_technologies,
    adjust_variant_adata_by_normal_gene_matrix
)

from .conftest import (
    compare_two_anndata_objects,
)

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
        "feature": ["exon", "exon", "exon", "exon", "exon", "exon", "exon"],
        "region_length": [120, 40, 140, 1000, 1000, 100, 500],
        "strand": ["+", "-", "+", "+", "-", "+", "-"]
    })

    return transcript_id_df

def test_strand_bias_filtering_5p(adata_for_strand_bias_testing):
    strand_bias_end = "5p"
    read_length = "90"
    
    adata = remove_variants_from_adata_for_stranded_technologies(adata=adata_for_strand_bias_testing, strand_bias_end=strand_bias_end, read_length=read_length, header_column="vcrs_header", variant_source="transcriptome", variant_position_annotations="cdna", gtf=None, plot_histogram=False)

    adata_var_gt = pd.DataFrame({
        "vcrs_header": ["ENST0000001:c.10A>G", "ENST0000001:c.11A>G;ENST0000001:c.12A>G", "ENST0000001:c.11_12insAAT", "ENST0000001:c.12del;ENST0000002:c.14del", "ENST0000001:c.101del;ENST0000003:c.100A>T", "ENST0000001:c.50G>A", "ENST0000001:c.89_93del"],
        "vcrs_count": [1, 4, 2, 99, 0, 1, 1],
        "number_obs": [1, 2, 1, 3, 0, 1, 1],
    })

    adata_x_gt = np.array([
        [0, 1, 0, 1, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0],
        [1, 3, 0, 90, 0, 1, 0],
        [0, 0, 2, 8, 0, 0, 1]
    ])

    adata_obs_gt = adata_for_strand_bias_testing.obs

    adata_gt = ad.AnnData(X=adata_x_gt, var=adata_var_gt, obs=adata_obs_gt)

    compare_two_anndata_objects(adata, adata_gt)

def test_strand_bias_filtering_3p(adata_for_strand_bias_testing, transcript_id_df_for_strand_bias_testing):
    strand_bias_end = "3p"
    read_length = "90"
    
    adata = remove_variants_from_adata_for_stranded_technologies(adata=adata_for_strand_bias_testing, strand_bias_end=strand_bias_end, read_length=read_length, header_column="vcrs_header", variant_source="transcriptome", variant_position_annotations="cdna", gtf=transcript_id_df_for_strand_bias_testing, plot_histogram=False)

    #!!! ENST0000003:c.100A>T
    #!!! ENST0000001:c.50G>A;ENST0000006:c.1001G>A got split into separate
    adata_var_gt = pd.DataFrame({
        "vcrs_header": ["ENST0000001:c.10A>G", "ENST0000001:c.11A>G;ENST0000001:c.12A>G", "ENST0000001:c.11_12insAAT", "ENST0000001:c.12del;ENST0000002:c.14del", "ENST0000001:c.101del;ENST0000003:c.100A>T", "ENST0000001:c.50G>A;ENST0000006:c.1001G>A", "ENST0000001:c.89_93del"],
        "vcrs_count": [1, 4, 2, 99, 0, 1, 1],
        "number_obs": [1, 2, 1, 3, 0, 1, 1],
    })

    adata_x_gt = np.array([
        [0, 1, 0, 1, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0],
        [1, 3, 0, 90, 0, 1, 0],
        [0, 0, 2, 8, 0, 0, 1]
    ])

    adata_obs_gt = adata_for_strand_bias_testing.obs

    adata_gt = ad.AnnData(X=adata_x_gt, var=adata_var_gt, obs=adata_obs_gt)

    compare_two_anndata_objects(adata, adata_gt)
