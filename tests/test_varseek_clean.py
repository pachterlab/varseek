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
    vcf_to_dataframe
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
