"""Tests for vk summarize.

These exercise the report/TSV/plot outputs and, importantly, the degenerate inputs that
`summarize` has to survive: a bare AnnData that never went through vk clean, missing or
all-NaN gene columns, non-numeric count columns, and headers that are not in HGVS format.
"""

import os
from datetime import datetime
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import pytest
from scipy import sparse as sp

import varseek as vk
from varseek.utils.varseek_summarize_utils import (
    _bucket_distribution,
    _explode_gene_stats,
    _fmt_float,
    _fmt_int,
    _top_table,
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
        out = tmp_path / "out_vk_summarize"

    out.mkdir(parents=True, exist_ok=True)
    return out


# HGVS-format headers, matching what vk clean puts in adata.var["vcrs_id"] (and on the var index).
# The third entry is a merged VCRS (semicolon-separated), which summarize must not choke on.
TOY_HEADERS = [
    "ENST00000000001:c.100A>T",
    "ENST00000000002:c.200G>C",
    "ENST00000000003:c.300del;ENST00000000004:c.400del",
    "ENST00000000005:c.500A>G",
]
TOY_GENES = ["BRCA1", "TP53", "EGFR;KRAS", "TP53"]

# 3 observations x 4 variants. Variant 1 is undetected everywhere; obs 2 has no counts at all.
TOY_COUNTS = np.array(
    [
        [5, 0, 2, 0],
        [3, 0, 0, 1],
        [0, 0, 0, 0],
    ],
    dtype=float,
)


def make_adata(counts=None, headers=None, genes=None, sparse=False, add_gene_column=True, index_only=False):
    """Build a small AnnData shaped like a post-vk-clean variant count matrix."""
    counts = TOY_COUNTS if counts is None else counts
    headers = TOY_HEADERS if headers is None else headers
    genes = TOY_GENES if genes is None else genes

    X = sp.csr_matrix(counts) if sparse else counts
    var = pd.DataFrame(index=pd.Index(headers, name="variant"))
    if not index_only:
        # vk clean sets vcrs_id from the var index and exempts it from the vcrs_* -> variant_* rename
        var["vcrs_id"] = headers
    if add_gene_column:
        var["gene_name"] = genes[: len(headers)]
    obs = pd.DataFrame(index=[f"cell{i}" for i in range(counts.shape[0])])
    return ad.AnnData(X=X, obs=obs, var=var)


def read_stats(out):
    with open(os.path.join(out, "varseek_summarize_stats.txt"), encoding="utf-8") as f:
        return f.read()


# ---------------------------------------------------------------------------
# Core behavior
# ---------------------------------------------------------------------------


def test_summarize_writes_expected_outputs(out_dir):
    adata = make_adata()
    result = vk.summarize(adata, out=str(out_dir))

    assert isinstance(result, ad.AnnData)
    assert os.path.isfile(os.path.join(out_dir, "varseek_summarize_stats.txt"))
    for tsv in ("variants_by_count.tsv", "variants_by_prevalence.tsv", "variants_per_cell.tsv"):
        assert os.path.isfile(os.path.join(out_dir, "specific_stats", tsv)), f"missing {tsv}"
    assert os.path.isfile(os.path.join(out_dir, "specific_stats", "genes_by_variant_count.tsv"))
    assert os.listdir(os.path.join(out_dir, "plots")), "no plots were produced"


def test_summarize_headline_numbers_are_correct(out_dir):
    adata = make_adata()
    vk.summarize(adata, out=str(out_dir))
    stats = read_stats(out_dir)

    # 4 variants in the panel, 3 of them with count > 0 (variant index 1 is never detected)
    assert "Variants screened (panel):   4" in stats
    assert "Variants detected (count>0): 3 (75.0% of panel)" in stats
    # 3 cells, 2 of which carry at least one variant
    assert "Cells profiled:            3" in stats
    assert "Cells with >=1 variant:     2 (66.7%)" in stats
    assert f"Total variant counts:        {int(TOY_COUNTS.sum())}" in stats


def test_summarize_per_variant_stats_match_matrix(out_dir):
    adata = make_adata()
    vk.summarize(adata, out=str(out_dir))

    by_count = pd.read_csv(os.path.join(out_dir, "specific_stats", "variants_by_count.tsv"), sep="\t")
    counted = dict(zip(by_count["variant"], by_count["total_counts"]))
    assert counted[TOY_HEADERS[0]] == TOY_COUNTS[:, 0].sum()
    assert counted[TOY_HEADERS[2]] == TOY_COUNTS[:, 2].sum()
    assert TOY_HEADERS[1] not in counted  # undetected variants are excluded

    detected = dict(zip(by_count["variant"], by_count["n_cells_detected"]))
    assert detected[TOY_HEADERS[0]] == int((TOY_COUNTS[:, 0] > 0).sum())

    per_cell = pd.read_csv(os.path.join(out_dir, "specific_stats", "variants_per_cell.tsv"), sep="\t")
    assert list(per_cell["total_variant_counts"]) == list(TOY_COUNTS.sum(axis=1))
    assert list(per_cell["n_variants_detected"]) == list((TOY_COUNTS > 0).sum(axis=1))


def test_summarize_gene_stats_split_merged_genes(out_dir):
    adata = make_adata()
    vk.summarize(adata, out=str(out_dir))

    genes = pd.read_csv(os.path.join(out_dir, "specific_stats", "genes_by_variant_count.tsv"), sep="\t", index_col=0)
    # variant 2 maps to "EGFR;KRAS" - each gene gets credited separately
    assert set(genes.index) == {"BRCA1", "EGFR", "KRAS", "TP53"}
    assert genes.loc["EGFR", "n_variants_detected"] == 1
    assert genes.loc["KRAS", "n_variants_detected"] == 1
    # TP53 has two variants but only one of them is detected
    assert genes.loc["TP53", "n_variants_detected"] == 1


def test_summarize_sparse_matches_dense(out_dir, tmp_path):
    dense_out = out_dir
    sparse_out = tmp_path / "sparse_out"
    vk.summarize(make_adata(sparse=False), out=str(dense_out))
    vk.summarize(make_adata(sparse=True), out=str(sparse_out))

    def report_body(out):
        # the trailing "Detailed tables:"/"Plots:" lines name the output dir, which differs by design
        return [line for line in read_stats(out).splitlines() if not line.startswith(("Detailed tables:", "Plots:"))]

    assert report_body(dense_out) == report_body(sparse_out)


def test_summarize_bulk_uses_sample_wording(out_dir):
    adata = make_adata()
    vk.summarize(adata, technology="bulk", out=str(out_dir))
    stats = read_stats(out_dir)

    assert "Samples profiled:" in stats
    assert "Variants detected per sample" in stats
    assert "Per-sample breakdown" in stats  # only emitted for bulk with <= 100 observations
    assert "Cells profiled:" not in stats
    assert os.path.isfile(os.path.join(out_dir, "specific_stats", "variants_per_sample.tsv"))


def test_summarize_top_values_limits_tables(out_dir):
    adata = make_adata()
    vk.summarize(adata, top_values=1, out=str(out_dir))
    stats = read_stats(out_dir)

    assert "Top 1 variants by total counts" in stats
    # the highest-count variant is the only one listed; the runner-up is not
    assert TOY_HEADERS[0] in stats
    assert TOY_HEADERS[3] not in stats


def test_summarize_reads_h5ad_from_disk(out_dir, tmp_path):
    h5ad_path = tmp_path / "adata.h5ad"
    make_adata().write(h5ad_path)

    vk.summarize(str(h5ad_path), out=str(out_dir))
    assert "Variants screened (panel):   4" in read_stats(out_dir)


def test_summarize_accepts_path_object(out_dir, tmp_path):
    """Path inputs used to pass validation and then fail in the loader."""
    h5ad_path = tmp_path / "adata.h5ad"
    make_adata().write(h5ad_path)

    vk.summarize(h5ad_path, out=str(out_dir))
    assert os.path.isfile(os.path.join(out_dir, "varseek_summarize_stats.txt"))


def test_summarize_does_not_leave_scratch_columns_on_input(out_dir):
    adata = make_adata()
    vk.summarize(adata, out=str(out_dir))
    assert "_display_name" not in adata.var.columns


def test_summarize_dry_run_writes_nothing(out_dir):
    adata = make_adata()
    assert vk.summarize(adata, out=str(out_dir), dry_run=True) is None
    assert not os.path.isfile(os.path.join(out_dir, "varseek_summarize_stats.txt"))
    assert not os.path.isdir(os.path.join(out_dir, "plots"))


def test_summarize_refuses_to_overwrite_by_default(out_dir):
    vk.summarize(make_adata(), out=str(out_dir))
    with pytest.raises(FileExistsError):
        vk.summarize(make_adata(), out=str(out_dir))
    vk.summarize(make_adata(), out=str(out_dir), overwrite=True)  # succeeds with overwrite


# ---------------------------------------------------------------------------
# Robustness: degenerate / non-vk-clean inputs
# ---------------------------------------------------------------------------


def test_summarize_without_gene_column(out_dir):
    adata = make_adata(add_gene_column=False)
    vk.summarize(adata, out=str(out_dir))

    stats = read_stats(out_dir)
    assert "Gene-level stats skipped" in stats
    assert not os.path.isfile(os.path.join(out_dir, "specific_stats", "genes_by_variant_count.tsv"))


def test_summarize_with_all_nan_gene_column(out_dir):
    adata = make_adata()
    adata.var["gene_name"] = np.nan
    vk.summarize(adata, out=str(out_dir))

    assert "Genes with detected variants: 0" in read_stats(out_dir)


def test_summarize_with_missing_named_gene_column_warns_and_continues(out_dir, caplog):
    adata = make_adata(add_gene_column=False)
    vk.summarize(adata, gene_name_column="not_a_column", out=str(out_dir))

    assert "Gene-level stats skipped" in read_stats(out_dir)
    assert any("not found in adata.var" in record.message for record in caplog.records)


def test_summarize_with_index_only_var(out_dir):
    """A hand-built AnnData with no vcrs_id column still summarizes off the index."""
    adata = make_adata(index_only=True, add_gene_column=False)
    vk.summarize(adata, out=str(out_dir))

    stats = read_stats(out_dir)
    assert "Variants screened (panel):   4" in stats
    assert TOY_HEADERS[0] in stats


def test_summarize_with_holes_in_vcrs_id_falls_back_to_index(out_dir):
    """A vcrs_id column with NaNs must not produce "nan" labels or kill the plots."""
    adata = make_adata(add_gene_column=False)
    adata.var["vcrs_id"] = [TOY_HEADERS[0], np.nan, None, TOY_HEADERS[3]]
    vk.summarize(adata, out=str(out_dir))

    stats = read_stats(out_dir)
    assert TOY_HEADERS[0] in stats
    assert TOY_HEADERS[2] in stats  # recovered from the var index
    assert "nan" not in stats
    plots = os.listdir(os.path.join(out_dir, "plots"))
    assert any(p.startswith("substitutions_") for p in plots), "specialized plots should still be produced"


def test_summarize_with_non_hgvs_headers_skips_specialized_plots(out_dir, caplog):
    adata = make_adata(headers=["v1", "v2", "v3", "v4"])
    vk.summarize(adata, out=str(out_dir))

    # the core report is still produced
    assert "Variants screened (panel):   4" in read_stats(out_dir)
    assert any("not in HGVS format" in record.message for record in caplog.records)
    plots = os.listdir(os.path.join(out_dir, "plots"))
    assert not any(p.startswith("substitutions_") for p in plots)


def test_summarize_with_precomputed_variant_count_column(out_dir):
    """vk clean's rename_vcrs_to_variant leaves variant_count/variant_detected behind."""
    adata = make_adata()
    adata.var["variant_count"] = TOY_COUNTS.sum(axis=0)
    adata.var["variant_detected"] = TOY_COUNTS.sum(axis=0) > 0
    vk.summarize(adata, out=str(out_dir))

    assert "Variants detected (count>0): 3 (75.0% of panel)" in read_stats(out_dir)


def test_summarize_coerces_non_numeric_count_column(out_dir):
    """A count column carried in as strings (or holding NaN) must not abort the report."""
    adata = make_adata()
    adata.var["variant_count"] = ["5", "0", np.nan, "1"]
    vk.summarize(adata, out=str(out_dir))

    stats = read_stats(out_dir)
    assert "Variants screened (panel):   4" in stats
    # "5" and "1" parse to counts; the NaN is treated as 0
    assert "Variants detected (count>0): 2" in stats


def test_summarize_all_zero_matrix(out_dir):
    adata = make_adata(counts=np.zeros((3, 4)))
    vk.summarize(adata, out=str(out_dir))

    stats = read_stats(out_dir)
    assert "Variants detected (count>0): 0 (0.0% of panel)" in stats
    assert "Cells with >=1 variant:     0 (0.0%)" in stats


def test_summarize_single_cell_single_variant(out_dir):
    adata = make_adata(counts=np.array([[7.0]]), headers=[TOY_HEADERS[0]], genes=["BRCA1"])
    vk.summarize(adata, out=str(out_dir))

    stats = read_stats(out_dir)
    assert "Variants screened (panel):   1" in stats
    assert "Total variant counts:        7" in stats


def test_summarize_rejects_empty_adata(out_dir):
    empty = ad.AnnData(X=np.zeros((0, 0)))
    with pytest.raises(ValueError, match="empty"):
        vk.summarize(empty, out=str(out_dir))


def test_summarize_rejects_bad_extension(out_dir):
    with pytest.raises(ValueError, match="Invalid file extension"):
        vk.summarize("adata.csv", out=str(out_dir))


def test_summarize_rejects_wrong_type(out_dir):
    with pytest.raises(TypeError):
        vk.summarize(42, out=str(out_dir))


def test_summarize_rejects_missing_file(out_dir):
    with pytest.raises(ValueError, match="does not exist"):
        vk.summarize("/nonexistent/path/adata.h5ad", out=str(out_dir))


def test_summarize_no_longer_accepts_strand_bias_arguments(out_dir):
    """plot_strand_bias and its companions were removed from the summarize interface."""
    import inspect

    params = inspect.signature(vk.summarize).parameters
    for removed in ("plot_strand_bias", "strand_bias_end", "cdna_fasta", "seq_id_cdna_column", "start_variant_position_cdna_column", "end_variant_position_cdna_column", "read_length"):
        assert removed not in params, f"{removed} should have been removed from summarize"


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------


def test_fmt_int_and_float_handle_bad_values():
    assert _fmt_int(1234) == "1,234"
    assert _fmt_int(1234.6) == "1,235"
    assert _fmt_int(np.nan) == "n/a"
    assert _fmt_int(np.inf) == "n/a"
    assert _fmt_int("not a number") == "n/a"
    assert _fmt_int(None) == "n/a"

    assert _fmt_float(1234.56) == "1,234.6"
    assert _fmt_float(np.nan) == "n/a"
    assert _fmt_float(None) == "n/a"


def test_bucket_distribution():
    buckets = [("0", 0, 0), ("1", 1, 1), ("2+", 2, float("inf"))]
    rows = _bucket_distribution(np.array([0, 0, 1, 5, 9]), buckets)
    assert [(label, n) for label, n, _ in rows] == [("0", 2), ("1", 1), ("2+", 2)]
    assert rows[0][2] == pytest.approx(0.4)


def test_bucket_distribution_on_empty_input():
    rows = _bucket_distribution(np.array([]), [("0", 0, 0)])
    assert rows == [("0", 0, 0.0)]


def test_explode_gene_stats_handles_missing_and_merged_genes():
    var_df = pd.DataFrame(
        {
            "gene_name": ["A;B", "A", np.nan, "nan", "C ; A"],
            "vcrs_count": [10.0, 5.0, 3.0, 2.0, 1.0],
        }
    )
    genes = _explode_gene_stats(var_df, "gene_name")

    assert set(genes.index) == {"A", "B", "C"}
    assert genes.loc["A", "n_variants_detected"] == 3  # "A;B", "A", and "C ; A"
    assert genes.loc["A", "total_counts"] == pytest.approx(16.0)
    assert genes.loc["B", "n_variants_detected"] == 1
    # sorted by n_variants_detected descending
    assert genes.index[0] == "A"


def test_explode_gene_stats_with_no_usable_genes():
    var_df = pd.DataFrame({"gene_name": [np.nan, None], "vcrs_count": [1.0, 2.0]})
    genes = _explode_gene_stats(var_df, "gene_name")
    assert genes.empty
    assert list(genes.columns) == ["n_variants_detected", "total_counts"]


def test_top_table_renders_and_handles_empty():
    table = _top_table([("BRCA1", "10", "2")], ["gene", "counts", "cells"])
    lines = table.split("\n")
    assert lines[0].split() == ["gene", "counts", "cells"]
    assert lines[1].split() == ["BRCA1", "10", "2"]
    assert _top_table([], ["gene"]) == "  (none)"
