"""Read counts per VCRS from the BUS file (vk clean's `min_reads`).

The point of `min_reads` is that `bustools count --multimapping --cm` gives a VCRS
`sum over ECs of floor(reads_in_EC / targets_in_EC)`, so a VCRS whose reads only ever land in
equivalence classes holding fewer reads than targets receives a hard 0 that no `min_counts` can
recover. These tests pin that behaviour on a hand-built BUS directory.
"""

import anndata as ad
import numpy as np
import pandas as pd
import pytest
from scipy.sparse import csr_matrix

from varseek.utils import apply_read_count_matrix, compute_vcrs_read_counts

VCRS_NAMES = ["1:g.100A>T", "1:g.200C>G", "1:g.300G>A"]


@pytest.fixture
def bus_dir(tmp_path):
    """A kb-count-style directory whose middle record is a 2-read, 3-target equivalence class.

    `bustools count --cm` would give each of that class's three targets floor(2/3) = 0.
    """
    d = tmp_path / "kb_count_out_vcrs"
    d.mkdir()
    (d / "transcripts.txt").write_text("\n".join(VCRS_NAMES) + "\n")
    (d / "matrix.ec").write_text("0\t0\n1\t1\n2\t0,1,2\n")
    # barcode, UMI, EC, count -- the columns `bustools text` writes
    (d / "output.bus.txt").write_text(
        "AAAA\tTTTT\t0\t5\n"
        "AAAA\tTTTT\t2\t2\n"
        "CCCC\tTTTT\t1\t3\n"
    )
    (d / "output.bus").write_bytes(b"")  # only the .txt is read once it exists
    return d


@pytest.fixture
def adata_bustools():
    """The matrix `bustools count --cm` would have produced, with the columns deliberately
    ordered differently from transcripts.txt so name-based alignment is actually exercised."""
    var_names = [VCRS_NAMES[2], VCRS_NAMES[0], VCRS_NAMES[1]]
    X = csr_matrix(np.array([[0.0, 5.0, 0.0], [0.0, 0.0, 3.0]]))  # C, A, B for two barcodes
    return ad.AnnData(X=X, obs=pd.DataFrame(index=["AAAA", "CCCC"]), var=pd.DataFrame(index=var_names))


def test_compute_vcrs_read_counts_does_not_divide_across_the_equivalence_class(bus_dir):
    barcodes, names, matrix = compute_vcrs_read_counts(str(bus_dir))

    assert names == VCRS_NAMES
    assert barcodes == ["AAAA", "CCCC"]
    # every target of an EC gets that EC's full read count, and a VCRS in several ECs sums them
    np.testing.assert_array_equal(matrix.toarray(), np.array([[7.0, 2.0, 2.0], [0.0, 3.0, 0.0]]))


def test_apply_read_count_matrix_replaces_X_and_keeps_the_bustools_matrix(bus_dir, adata_bustools):
    adata = apply_read_count_matrix(adata_bustools, str(bus_dir))

    # aligned by name, not by position: adata's columns are C, A, B
    np.testing.assert_array_equal(np.asarray(adata.X.sum(axis=0)).ravel(), [2.0, 7.0, 5.0])
    np.testing.assert_array_equal(adata.var["read_count"].to_numpy(), [2.0, 7.0, 5.0])
    np.testing.assert_array_equal(np.asarray(adata.layers["bustools_counts"].sum(axis=0)).ravel(), [0.0, 5.0, 3.0])


def test_min_reads_recovers_a_vcrs_that_bustools_zeroed(bus_dir, adata_bustools):
    """The third VCRS has two compatible reads but a bustools count of 0, so it is callable at
    min_reads=2 and unreachable at any min_counts."""
    adata = apply_read_count_matrix(adata_bustools, str(bus_dir), min_reads=2)

    called = {str(v) for v, c in zip(adata.var_names, np.asarray(adata.X.sum(axis=0)).ravel()) if c > 0}
    assert called == set(VCRS_NAMES)
    assert np.asarray(adata.layers["bustools_counts"].sum(axis=0)).ravel()[0] == 0


def test_min_reads_thresholds_per_entry(bus_dir, adata_bustools):
    adata = apply_read_count_matrix(adata_bustools, str(bus_dir), min_reads=3)

    # barcode AAAA holds A=7 (kept), B=2 and C=2 (both below 3); barcode CCCC holds B=3 (kept)
    np.testing.assert_array_equal(adata.X.toarray(), np.array([[0.0, 7.0, 0.0], [0.0, 0.0, 3.0]]))


def test_missing_bus_file_is_reported(tmp_path):
    with pytest.raises(FileNotFoundError, match="Cannot count reads per VCRS"):
        compute_vcrs_read_counts(str(tmp_path))


def test_bus_file_and_ec_from_different_runs_are_reported(bus_dir):
    (bus_dir / "matrix.ec").write_text("0\t0\n1\t1\n")  # EC 2 removed
    with pytest.raises(ValueError, match="do not belong to the same run"):
        compute_vcrs_read_counts(str(bus_dir))
