#!/usr/bin/env python3
"""
Simple, readable k-mer equivalence-class builder.

This is the straightforward dict-of-lists version of build_kmer_map.py: every
anchor's context window is a plain string key, and positions that share a window
are grouped by appending to that key's list. It trades the packed-int64 / lexsort
machinery for clarity, so it is ideal for small/moderate inputs (a transcriptome,
a single chromosome) but NOT for whole-genome scale, where ~3e9 distinct string
keys blow past ~1 TB of RAM. Use build_kmer_map.py there.

Output is a single DataFrame with one row per anchor:

    seq_id | position | ec | ec_len [| gene]

where `ec` is the ";"-joined equivalence class (every interchangeable location,
including the row's own), `ec_len` is its member count (semicolons + 1), and
`gene` (only when a GTF is given) is the gene of that row's own location.
"""

from collections import defaultdict
from pathlib import Path
import argparse
import logging
import resource
import sys
import time

import pandas as pd
from tqdm import tqdm

max_ram_gb = 500  # 300 GB
MAX_RAM = max_ram_gb * 1024**3

soft, hard = resource.getrlimit(resource.RLIMIT_AS)
resource.setrlimit(resource.RLIMIT_AS, (MAX_RAM, MAX_RAM))

# Reuse the genuinely simple, shared helpers rather than re-implementing them.
from build_kmer_map import fasta_reader, anchor_extents, _count_seqs

logger = logging.getLogger(__name__)

_VALID = set("ACGT")


def _valid_runs(seq):
    """
    Yield [start, end) index ranges of maximal runs of A/C/G/T bases.

    Lowercase (soft-masked) and N bases break a run, so a window never spans one,
    matching build_kmer_map.py's behaviour.
    """
    start = None
    for i, base in enumerate(seq):
        if base in _VALID:
            if start is None:
                start = i
        elif start is not None:
            yield start, i
            start = None
    if start is not None:
        yield start, len(seq)


def _genes_for(refs, prefixes, positions, gtf):
    """
    Resolve the gene name for each (ref, prefix, pos), using varseek's own GTF
    logic so these line up with the headers vk build writes. Returns "" where no
    gene is found.

    Genes are resolved over the UNIQUE dedup keys, not once per anchor: a
    transcript (c.) gene depends only on its seq_id, so a whole transcriptome
    collapses to its few thousand distinct transcripts; genome (g.) genes depend
    on position and so are only deduplicated on exact repeats.
    """
    from varseek.utils.seq_utils import gene_name_lookups_from_gtf
    from varseek.utils.varseek_build_utils import compute_gene_name_series_for_headers

    lookups = gene_name_lookups_from_gtf(gtf)
    dedup_keys = [(r, "" if p == "c" else f"{p}.{pos}") for r, p, pos in zip(refs, prefixes, positions)]
    uniq_keys = list(dict.fromkeys(dedup_keys))

    # "c.1" is a throwaway position for transcript rows (the ENST->gene path ignores
    # it); genome rows carry their real "g.<pos>" mutation.
    frame = pd.DataFrame({
        "seq_id": [key[0] for key in uniq_keys],
        "mutation": [key[1] or "c.1" for key in uniq_keys],
    })
    gene_of_key = dict(zip(uniq_keys, compute_gene_name_series_for_headers(frame, "seq_id", "mutation", gtf, lookups=lookups).tolist()))
    return [gene_of_key[key] or "" for key in dedup_keys]


def build_kmer_df(inputs, left_extent, right_extent, skip_truncated=False, gtf=None, seq_id_column="seq_id", position_column="position"):
    """
    Group every anchor's context window with a plain dict, then flatten to a df.

    For each maximal run of valid bases, every anchor's window is extracted as a
    substring and appended to that window's member list. Two anchors are
    interchangeable iff they share the same (alignment, window bases) key, so the
    key carries `left_len` too — with --skip-truncated every window is full so the
    alignment is constant, but truncated edge windows of different lengths must not
    collide.

    Parameters
    ----------
    inputs : list[(fasta_path, prefix)]
        prefix is "c" (transcript) or "g" (genomic); it becomes the HGVS-style tag.
    left_extent, right_extent : int
        Max bases of context on each side of the anchor.
    skip_truncated : bool
        When True only anchors with a full window on both sides are emitted (edge
        anchors are skipped), reproducing the original single-k-mer builder.
    """
    k = left_extent + right_extent + 1

    # Parallel per-anchor arrays, plus window-key -> [anchor indices] groups.
    refs, poss, prefixes = [], [], []
    groups = defaultdict(list)

    for fasta, prefix in inputs:
        logger.info("Processing %s...", Path(fasta).name)
        total = _count_seqs(fasta)

        for ref, seq in tqdm(fasta_reader(fasta), desc="Building k-mer map", unit="sequence", total=total):
            for rs, re_ in _valid_runs(seq):
                run = seq[rs:re_]
                L = re_ - rs

                if skip_truncated:
                    if L < k:
                        continue  # no anchor in this run has a full window
                    anchor_range = range(left_extent, L - right_extent)
                else:
                    anchor_range = range(L)

                for j in anchor_range:
                    lo = max(0, j - left_extent)
                    hi = min(L, j + right_extent + 1)
                    # (alignment, window bases) fully identifies interchangeable anchors.
                    key = (j - lo, run[lo:hi])
                    groups[key].append(len(refs))
                    refs.append(ref)
                    poss.append(rs + j + 1)  # 1-based anchor position
                    prefixes.append(prefix)

    n = len(refs)
    logger.info("Collected %d anchors in %d equivalence classes.", n, len(groups))

    genes = _genes_for(refs, prefixes, poss, gtf) if gtf else [""] * n
    labels = [f"{r}({g}):{p}.{pos}" if g else f"{r}:{p}.{pos}" for r, g, p, pos in zip(refs, genes, prefixes, poss)]

    # One header per group; broadcast it (and its member count) back onto every member.
    ec = [None] * n
    ec_len = [0] * n
    for idxs in tqdm(groups.values(), desc="Joining EC headers", unit="EC"):
        header = ";".join(sorted(labels[i] for i in idxs))
        size = header.count(";") + 1
        for i in idxs:
            ec[i] = header
            ec_len[i] = size

    data = {seq_id_column: refs, position_column: poss, "ec": ec, "ec_len": ec_len}
    if gtf:
        data["gene"] = genes
    return pd.DataFrame(data)


def _write(df, out):
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.suffix.lower() in (".parquet", ".pq"):
        df.to_parquet(out, index=False)
    else:
        df.to_csv(out, index=False)
    logger.info("Wrote %d rows to %s", len(df), out)


def build_kmer_index_simple(cdna=None, dna=None, k=31, anchor="middle", gtf=None, out="kmer_ec.parquet", skip_truncated=False, seq_id_column="seq_id", position_column="position"):
    start = time.perf_counter()
    if not cdna and not dna:
        raise SystemExit("provide at least one FASTA via --cdna and/or --dna")

    try:
        left_extent, right_extent = anchor_extents(anchor, k)
    except ValueError as e:
        raise SystemExit(str(e))

    inputs = [(fasta, "c") for fasta in (cdna or [])] + [(fasta, "g") for fasta in (dna or [])]
    df = build_kmer_df(inputs, left_extent, right_extent, skip_truncated=skip_truncated, gtf=gtf, seq_id_column=seq_id_column, position_column=position_column)
    _write(df, out)
    logger.info("Finished building simple k-mer table in %.1f seconds.", time.perf_counter() - start)
    return df


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")

    parser = argparse.ArgumentParser(description="Simple dict-based k-mer equivalence-class table builder (small/moderate inputs).")
    parser.add_argument("--cdna", nargs="*", default=[], metavar="FASTA", help="Transcript/cDNA FASTA(s); labelled 'c.'.")
    parser.add_argument("--dna", nargs="*", default=[], metavar="FASTA", help="Genomic FASTA(s); labelled 'g.'.")
    parser.add_argument("-k", type=int, default=31, help="Window size (default: 31). Must be odd when --anchor is middle.")
    parser.add_argument("--anchor", choices=["l", "left", "r", "right", "m", "middle"], default="middle", help="Which base the position labels: l/left (window start), r/right (window end), or m/middle (center, requires odd k; default).")
    parser.add_argument("--skip-truncated", action="store_true", help="Only emit anchors with a full window; skip edge/truncated k-mers.")
    parser.add_argument("--gtf", default=None, help="Optional GTF; adds a 'gene' column and bakes gene names into headers ('<ref>(<GENE>):<prefix>.<pos>').")
    parser.add_argument("-o", "--out", default="kmer_ec.parquet", help="Output table (.parquet or .csv).")
    parser.add_argument("--seq_id_column", default="seq_id", help="Name for the seq-id column (default: seq_id).")
    parser.add_argument("--position_column", default="position", help="Name for the position column (default: position).")

    build_kmer_index_simple(**vars(parser.parse_args()))


if __name__ == "__main__":
    try:
        main()
    except MemoryError:
        print("❌ Memory limit exceeded — exiting")  # might just print 'Segmentation fault (core dumped)' rather than this
        sys.exit(1)

# Build the table:
# python build_kmer_map_simple.py \
#     -o kmer_ec.parquet \
#     -k 31 \
#     --skip-truncated \
#     --gtf  /home/jrich/Desktop/varseek-examples/trash/data/reference/Homo_sapiens.GRCh38.gtf \
#     --cdna /home/jrich/Desktop/varseek-examples/trash/data/reference/Homo_sapiens.GRCh38.cdna.all.fa
