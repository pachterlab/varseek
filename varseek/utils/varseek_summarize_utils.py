"""Helper functions for varseek summarize."""

import numpy as np
import pandas as pd
from scipy import sparse as sp


def _as_1d(array_like):
    """Flatten a numpy array or the result of a (possibly sparse) matrix reduction to 1-D."""
    return np.asarray(array_like).flatten()


def _counts_per_obs(X):
    """Total variant counts per observation (cell/sample)."""
    return _as_1d(X.sum(axis=1))


def _variants_detected_per_obs(X):
    """Number of distinct variants with count > 0 per observation (cell/sample)."""
    if sp.issparse(X):
        return _as_1d((X != 0).sum(axis=1))
    return _as_1d((np.asarray(X) != 0).sum(axis=1))


def _detected_per_var(X):
    """Number of observations (cells/samples) in which each variant is detected."""
    if sp.issparse(X):
        return _as_1d((X != 0).sum(axis=0))
    return _as_1d((np.asarray(X) != 0).sum(axis=0))


def _fmt_int(value):
    """Format an integer with thousands separators."""
    return f"{int(round(float(value))):,}"


def _fmt_float(value, decimals=1):
    return f"{float(value):,.{decimals}f}"


def _distribution_line(values):
    """One-line mean/median/min/max/std summary of a numeric array."""
    values = np.asarray(values, dtype=float)
    if values.size == 0:
        return "mean n/a  median n/a  min n/a  max n/a"
    return (
        f"mean {_fmt_float(values.mean())}  "
        f"median {_fmt_float(np.median(values))}  "
        f"min {_fmt_int(values.min())}  "
        f"max {_fmt_int(values.max())}  "
        f"std {_fmt_float(values.std())}"
    )


def _bucket_distribution(values, buckets):
    """Return (label, count, fraction) rows binning ``values`` into ``buckets``.

    ``buckets`` is a list of (label, low, high) with inclusive bounds; use
    ``float('inf')`` for an open upper bound.
    """
    values = np.asarray(values)
    total = max(values.size, 1)
    rows = []
    for label, low, high in buckets:
        n = int(((values >= low) & (values <= high)).sum())
        rows.append((label, n, n / total))
    return rows


def _explode_gene_stats(var_df, gene_column, count_column="vcrs_count"):
    """Aggregate per-gene stats over detected variants.

    Gene columns store one-or-more gene identifiers as a semicolon-separated
    string (a single VCRS can map to multiple genes). Each distinct gene a
    variant maps to gets credited with that variant. Returns a DataFrame indexed
    by gene with ``n_variants_detected`` and ``total_counts`` columns.
    """
    n_variants = {}
    total_counts = {}
    for gene_value, count in zip(var_df[gene_column], var_df[count_column]):
        if gene_value is None or (isinstance(gene_value, float) and np.isnan(gene_value)):
            continue
        genes = {g for g in str(gene_value).split(";") if g and g.lower() != "nan"}
        for gene in genes:
            n_variants[gene] = n_variants.get(gene, 0) + 1
            total_counts[gene] = total_counts.get(gene, 0.0) + float(count)
    if not n_variants:
        return pd.DataFrame(columns=["n_variants_detected", "total_counts"])
    genes_df = pd.DataFrame(
        {
            "n_variants_detected": pd.Series(n_variants),
            "total_counts": pd.Series(total_counts),
        }
    )
    genes_df.index.name = "gene"
    return genes_df.sort_values(["n_variants_detected", "total_counts"], ascending=False)


def _top_table(rows, headers, indent="  "):
    """Render a small right-padded text table from a list of tuples."""
    if not rows:
        return f"{indent}(none)"
    str_rows = [tuple(str(cell) for cell in row) for row in rows]
    widths = [max(len(headers[i]), *(len(r[i]) for r in str_rows)) for i in range(len(headers))]
    lines = [indent + "  ".join(h.ljust(widths[i]) for i, h in enumerate(headers))]
    for r in str_rows:
        lines.append(indent + "  ".join(cell.ljust(widths[i]) for i, cell in enumerate(r)))
    return "\n".join(lines)
