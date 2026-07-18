#!/usr/bin/env python3
"""
Summarise the k-mer equivalence-class tables as a distribution of "additional maps".

For an anchor, additional maps = ec_size - 1, i.e. how many OTHER positions share its
window. positions.parquet only stores anchors whose EC has 2+ members, so the entire
0-additional-map population (singletons) is absent from it and has to be recovered as

    singletons = total_anchors - member_rows

where total_anchors is counted straight off the FASTA the tables were built from,
using the same anchor rule as build_kmer_map (skip_truncated: a valid run of length L
yields max(0, L - k + 1) anchors, independent of --anchor).
"""

import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

# Same lookup build_kmer_map uses: only uppercase A/C/G/T are valid; anything else
# (N, soft-masked lowercase, ...) breaks the run.
_VALID = np.zeros(256, dtype=bool)
for _b in b"ACGT":
    _VALID[_b] = True


def count_anchors(fasta, ks):
    """Total anchors per k, counting valid-base runs exactly as build_kmer_map does."""
    totals = {k: 0 for k in ks}

    def flush(seq):
        if not seq:
            return
        valid = _VALID[np.frombuffer(seq.encode("latin-1"), dtype=np.uint8)]
        if not valid.any():
            return
        edges = np.diff(valid.view(np.int8))
        starts = np.flatnonzero(edges == 1) + 1
        ends = np.flatnonzero(edges == -1) + 1
        if valid[0]:
            starts = np.concatenate(([0], starts))
        if valid[-1]:
            ends = np.concatenate((ends, [valid.size]))
        lens = (ends - starts).astype(np.int64)
        for k in ks:
            totals[k] += int(np.maximum(lens - k + 1, 0).sum())

    parts = []
    with open(fasta) as f:
        for line in f:
            if line.startswith(">"):
                flush("".join(parts))
                parts = []
            else:
                parts.append(line.strip())
    flush("".join(parts))
    return totals


def ec_size_histogram(positions_parquet):
    """bincount over the ec_size column, read one row group at a time."""
    pf = pq.ParquetFile(positions_parquet)
    hist = np.zeros(2, dtype=np.int64)
    for batch in pf.iter_batches(batch_size=1 << 22, columns=["ec_size"]):
        sizes = batch.column("ec_size").to_numpy(zero_copy_only=False).astype(np.int64)
        counts = np.bincount(sizes)
        if counts.size > hist.size:
            hist = np.pad(hist, (0, counts.size - hist.size))
        hist[: counts.size] += counts
    return hist


def main():
    fasta = "/home/jrich/Desktop/varseek-examples/trash/data/reference/t2t/rna.fna"
    dirs = {55: "kmer_locations_k55_cdna", 75: "kmer_locations_k75_cdna", 95: "kmer_locations_k95_cdna"}

    print("Counting anchors in FASTA...", flush=True)
    totals = count_anchors(fasta, list(dirs))
    for k, n in totals.items():
        print(f"  k={k}: {n:,} anchors", flush=True)

    out = {}
    for k, d in dirs.items():
        print(f"Histogramming {d}...", flush=True)
        hist = ec_size_histogram(Path(d) / "positions.parquet")
        members = int(hist.sum())
        singletons = totals[k] - members
        # hist[s] = anchors in an EC of size s; index 1 is the recovered singletons.
        hist = hist.copy()
        hist[1] = singletons
        out[k] = {
            "total_anchors": totals[k],
            "member_rows": members,
            "singletons": singletons,
            "hist": hist.tolist(),
        }
        print(f"  k={k}: {members:,} members, {singletons:,} singletons, max ec_size={hist.size - 1}", flush=True)

    Path("kmer_map_stats.json").write_text(json.dumps(out))
    print("Wrote kmer_map_stats.json")


if __name__ == "__main__":
    main()
