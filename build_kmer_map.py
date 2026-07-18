#!/usr/bin/env python3

from pathlib import Path
import resource
from tqdm import tqdm
import time
import argparse
import logging

import numpy as np
import pandas as pd

max_ram_gb = 500  # 300 GB
MAX_RAM = max_ram_gb * 1024**3

soft, hard = resource.getrlimit(resource.RLIMIT_AS)
resource.setrlimit(resource.RLIMIT_AS, (MAX_RAM, MAX_RAM))

logger = logging.getLogger(__name__)


DNA = {
    "A": 0,
    "C": 1,
    "G": 2,
    "T": 3,
}

# Byte -> 2-bit code lookup for vectorised packing. Only uppercase A/C/G/T map to
# 0..3; every other byte (N, lowercase soft-masked bases, anything else) maps to -1
# and breaks the window, exactly like the DNA.get(base, -1) path used to.
_CODE_LUT = np.full(256, -1, dtype=np.int8)
for _base, _code in DNA.items():
    _CODE_LUT[ord(_base)] = _code


def _count_seqs(path):
    """Count FASTA records cheaply in one binary pass (each header is a '>' byte)."""
    n = 0
    size = Path(path).stat().st_size
    with open(path, "rb") as f, tqdm(total=size, desc=f"Counting sequences in {Path(path).name}", unit="B", unit_scale=True, leave=False) as bar:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            n += chunk.count(b">")
            bar.update(len(chunk))
    return n


def fasta_reader(filename):
    """
    Simple FASTA parser.

    Yields
    ------
    header : str
    sequence : str
    """
    header = None
    seq = []

    with open(filename) as f:
        for line in f:
            line = line.strip()

            if not line:
                continue

            if line.startswith(">"):
                if header is not None:
                    yield header, "".join(seq)

                header = line[1:].split()[0]
                seq = []

            else:
                # Keep the original case: lowercase (soft-masked) bases are
                # intentionally NOT upper-cased, so they map to -1 in
                # build_kmer_map and break the window, just like an N or any
                # other non-ACGT character.
                seq.append(line)

        if header is not None:
            yield header, "".join(seq)


# Bases packed per int64 limb. 31 bases = 62 bits keeps each limb non-negative, so
# a window wider than 31 bases (k > 31) is carried across several int64 limbs and
# grouped by comparing the limbs together. k is otherwise unbounded.
_BASES_PER_LIMB = 31


def build_kmer_map(inputs, left_extent, right_extent, skip_truncated=False):
    """
    Pack every anchor's context window into flat, sort-ready arrays.

    Every valid (A/C/G/T) base is an anchor. Its window spans up to `left_extent`
    bases to the left and `right_extent` to the right, truncated at the sequence
    ends and at any invalid base (so a window never crosses an N or soft-masked
    base). Two anchors are interchangeable iff they share the same (left_len,
    right_len, packed_window) key; grouping equal keys is deferred to
    build_ec_tables, which sorts these arrays instead of hashing per base.

    The heavy lifting is vectorised: each maximal run of valid bases is converted
    to int8 codes, its rolling packed windows are built with k NumPy shift-or
    passes (O(k) per run, not per base), and every anchor's key/position is read
    off with array indexing. Positions are carried as integers (ref_id, 1-based
    anchor pos) rather than formatted strings, so the billions of singleton
    windows that build_ec_tables discards never pay for string construction.

    The packed window is right-aligned (the anchor's rightmost base is the low
    2 bits) and split across ceil(k / 31) int64 limbs, so k is unbounded (e.g.
    k = 91 uses 3 limbs); truncation just masks off the high (leftmost) bits.

    Parameters
    ----------
    inputs : list[(str, str)]
        (fasta_path, prefix) pairs, where prefix is "c" (transcript) or "g"
        (genomic); it becomes the HGVS-style tag in each location label.
    left_extent, right_extent : int
        Max bases of context on each side of the anchor.
    skip_truncated : bool
        When True, only anchors with a full window on both sides are emitted
        (edge anchors near a run/sequence boundary are skipped), reproducing the
        original single-k-mer builder, which recorded a position only once k
        valid bases were available. Such positions become singletons at query
        time. When False (default) every valid base is an anchor.

    Returns
    -------
    arrays : tuple of np.ndarray (left_len i8, right_len i8, packed i64[N, n_limbs],
        ref_id i32, position i32), one entry per anchor across all inputs.
        position is the 1-based coordinate of the anchor base; ref_id indexes
        into ref_names.
    ref_names : list[str]
        ref_id -> ref name.
    seq_prefix : dict[str, str]
        Maps each ref name to the prefix ("c"/"g") it was labelled with, so a
        singleton position's HGVS-like label can be reconstructed at query time.
    """

    # A window never exceeds k = left_extent + right_extent + 1 bases; it needs
    # n_limbs int64s to hold its 2*k bits.
    k = left_extent + right_extent + 1
    n_limbs = (k + _BASES_PER_LIMB - 1) // _BASES_PER_LIMB

    ref_names = []
    seq_prefix = {}
    lefts, rights, packeds, ref_id_chunks, pos_chunks = [], [], [], [], []

    for fasta, prefix in inputs:
        logger.info("Processing %s...", Path(fasta).name)
        total = _count_seqs(fasta)

        for ref, seq in tqdm(fasta_reader(fasta), desc="Building k-mer map", unit="sequence", total=total):

            ref_id = len(ref_names)
            ref_names.append(ref)
            seq_prefix[ref] = prefix

            codes = _CODE_LUT[np.frombuffer(seq.encode("latin-1"), dtype=np.uint8)]
            valid = codes >= 0
            if not valid.any():
                continue

            # Maximal runs of valid (A/C/G/T) bases; a run boundary is exactly where
            # the window used to stop, so windows never span an N or masked base.
            edges = np.diff(valid.view(np.int8))
            run_starts = np.flatnonzero(edges == 1) + 1
            run_ends = np.flatnonzero(edges == -1) + 1
            if valid[0]:
                run_starts = np.concatenate(([0], run_starts))
            if valid[-1]:
                run_ends = np.concatenate((run_ends, [valid.size]))

            for rs, re_ in zip(run_starts.tolist(), run_ends.tolist()):
                L = re_ - rs
                if skip_truncated and L < k:
                    continue  # no anchor in this run has a full window
                c = codes[rs:re_].astype(np.int64)

                # rw_limbs[m][t] = the packed value (right-aligned, base at run index t
                # in the low 2 bits) of the run's bases at offsets [31m, 31m+31) behind
                # t, i.e. limb m of the window ending at t. Each limb is built with up
                # to 31 shift-or passes; bits per offset are disjoint within the limb.
                rw_limbs = []
                for m in range(n_limbs):
                    limb = np.zeros(L, dtype=np.int64)
                    d0 = _BASES_PER_LIMB * m
                    for d in range(d0, min(k, d0 + _BASES_PER_LIMB)):
                        if d >= L:
                            break
                        limb[d:] |= c[: L - d] << (2 * (d - d0))
                    rw_limbs.append(limb)

                # Anchor set: full-window-only when skipping, else every base. The
                # window ends at hi and has `span` bases; its limbs are rw_limbs[*][hi]
                # masked down to `span` bases (drops the leftmost/high overflow).
                if skip_truncated:
                    i = np.arange(left_extent, L - right_extent)
                    left_len = np.full(i.size, left_extent)
                    right_len = np.full(i.size, right_extent)
                else:
                    i = np.arange(L)
                    left_len = np.minimum(i, left_extent)
                    right_len = np.minimum(L - 1 - i, right_extent)
                span = left_len + right_len + 1
                hi = i + right_len

                packed = np.empty((i.size, n_limbs), dtype=np.int64)
                for m in range(n_limbs):
                    keep = np.clip(span - _BASES_PER_LIMB * m, 0, _BASES_PER_LIMB)  # bases of limb m inside window
                    mask = (np.int64(1) << (2 * keep).astype(np.int64)) - 1         # keep==31 -> full-limb mask
                    packed[:, m] = rw_limbs[m][hi] & mask

                lefts.append(left_len.astype(np.int8))
                rights.append(right_len.astype(np.int8))
                packeds.append(packed)
                ref_id_chunks.append(np.full(i.size, ref_id, dtype=np.int32))
                pos_chunks.append((rs + i + 1).astype(np.int32))

    if packeds:
        arrays = (
            np.concatenate(lefts),
            np.concatenate(rights),
            np.concatenate(packeds, axis=0),
            np.concatenate(ref_id_chunks),
            np.concatenate(pos_chunks),
        )
    else:
        arrays = (
            np.empty(0, np.int8), np.empty(0, np.int8), np.empty((0, n_limbs), np.int64),
            np.empty(0, np.int32), np.empty(0, np.int32),
        )

    return arrays, ref_names, seq_prefix


def anchor_extents(anchor, k):
    """
    Translate an anchor choice + k into (left_extent, right_extent).

    anchor : "l"/"left", "r"/"right", or "m"/"middle" (middle requires odd k).
    """
    a = anchor[0]

    if a == "m":
        if k % 2 == 0:
            raise ValueError(f"k must be odd when anchor is middle (got {k})")
        return k // 2, k // 2

    if a == "l":
        return 0, k - 1

    if a == "r":
        return k - 1, 0

    raise ValueError(f"invalid anchor: {anchor!r}")


def parse_label(label):
    """Split a "<ref>:<prefix>.<pos>" location label into (ref, prefix, int pos)."""
    ref, rest = label.rsplit(":", 1)  # ref never contains ":"; rest is "<prefix>.<pos>"
    prefix, pos = rest.split(".", 1)
    return ref, prefix, int(pos)


def _annotate_labels(refs, prefixes, positions, gtf):
    """
    Bake gene names into location labels, matching varseek's header convention:
    "<ref>(<GENE>):<prefix>.<pos>" when a gene is found, plain "<ref>:<prefix>.<pos>"
    otherwise. With no GTF, every label is left plain.

    Genes are resolved with varseek's own GTF logic (ENST -> gene for c. variants,
    chromosome+position -> gene for g. variants), so these headers line up exactly
    with the ones vk build writes.
    """
    plain = [f"{r}:{p}.{pos}" for r, p, pos in zip(refs, prefixes, positions)]

    if gtf is None:
        return plain

    from varseek.utils.seq_utils import gene_name_lookups_from_gtf
    from varseek.utils.varseek_build_utils import compute_gene_name_series_for_headers

    # Parse the (potentially multi-GB) GTF exactly once, then resolve gene names over the
    # UNIQUE dedup keys rather than once per member. A transcript (c.) variant's gene depends
    # only on its seq_id, so its key drops the position — a transcriptome's millions of anchors
    # collapse to its few thousand distinct transcripts. Genome (g.) variants keep the position
    # (gene depends on it), so they are not deduplicated beyond exact repeats.
    lookups = gene_name_lookups_from_gtf(gtf)

    dedup_keys = [(r, "" if p == "c" else f"{p}.{pos}") for r, p, pos in zip(refs, prefixes, positions)]
    uniq_keys = list(dict.fromkeys(dedup_keys))

    # "c.1" is a throwaway position for transcript rows: the ENST->gene path ignores it, so any
    # valid c. mutation yields the same gene. Genome rows carry their real "g.<pos>" mutation.
    frame = pd.DataFrame({
        "seq_id": [k[0] for k in uniq_keys],
        "mutation": [k[1] or "c.1" for k in uniq_keys],
    })
    gene_of_key = dict(zip(uniq_keys, compute_gene_name_series_for_headers(frame, "seq_id", "mutation", gtf, lookups=lookups).tolist()))

    return [f"{r}({g}):{p}.{pos}" if (g := gene_of_key[k]) else plabel for r, p, pos, k, plabel in zip(refs, prefixes, positions, dedup_keys, plain)]


def build_ec_tables(kmer_arrays, ref_names, seq_prefix, gtf=None, seq_id_column="seq_id", position_column="position"):
    """
    Turn the packed anchor arrays into three compact, lookup-ready tables.

    Anchors are grouped by sorting on their (left_len, right_len, packed) key so
    equal keys become adjacent runs — a single O(N log N) sort in place of a
    per-base hash table. Only keys shared by 2+ anchors form an equivalence class
    (EC); a position absent from any EC is its own singleton class and is
    reconstructed at query time, so singletons are never materialised here. The
    EC's members are stored once (normalised by an integer ec_id) rather than
    repeating the joined class on every row, so storage stays O(total members).

    seq_id_column / position_column name the seq-id and position columns in the
    output tables; pass the same names to substitute() so the lookup merge lines up.

    Returns
    -------
    positions_df : DataFrame [<seq_id_column>, <position_column>, ec_id, ec_size]
        One row per ambiguous position; the (seq_id, position) lookup key. ec_size
        is the number of members in the row's EC (>= 2), so callers can filter or
        weight without a second lookup into ecs_df.
    ecs_df : DataFrame [ec_id, header]
        One row per EC; header is the ";"-joined, gene-annotated member labels
        (sorted for a stable canonical form) — the "full equivalence class".
    seq_meta_df : DataFrame [<seq_id_column>, prefix]
        The prefix each ref was labelled with, to reconstruct singleton headers.
    """
    logger.info("Assembling equivalence-class tables...")

    left, right, packed, ref_ids, positions = kmer_arrays
    seq_meta_df = pd.DataFrame({seq_id_column: list(seq_prefix), "prefix": list(seq_prefix.values())})

    N = packed.shape[0]
    if N == 0:
        positions_df = pd.DataFrame({seq_id_column: [], position_column: [], "ec_id": [], "ec_size": []})
        positions_df["ec_size"] = positions_df["ec_size"].astype("int32")
        ecs_df = pd.DataFrame({"ec_id": [], "header": []})
        logger.info("Built 0 equivalence classes covering 0 positions.")
        return positions_df, ecs_df, seq_meta_df

    # Sort so identical (left_len, right_len, packed) keys are adjacent. packed is
    # (N, n_limbs); each limb column is its own lexsort key. The last lexsort key is
    # primary, so this groups by the full (left, right, all limbs) tuple.
    t0 = time.perf_counter()
    order = np.lexsort((*packed.T, right, left))
    left, right, packed = left[order], right[order], packed[order]
    ref_ids, positions = ref_ids[order], positions[order]
    logger.info("Sorted %d anchors in %.1fs", N, time.perf_counter() - t0)

    # Group boundaries: a new group starts wherever any key field (any limb
    # included) differs from the previous row.
    t0 = time.perf_counter()
    same = (left[1:] == left[:-1]) & (right[1:] == right[:-1]) & (packed[1:] == packed[:-1]).all(axis=1)
    starts = np.concatenate(([0], np.flatnonzero(~same) + 1))
    sizes = np.diff(np.concatenate((starts, [N])))

    # Only groups of 2+ members are ECs; number them contiguously in group order.
    is_ec = sizes >= 2
    n_ec = int(is_ec.sum())
    ec_id_of_group = np.full(sizes.size, -1, dtype=np.int64)
    ec_id_of_group[is_ec] = np.arange(n_ec)

    # Broadcast each group's ec_id / size back onto its members, then keep only
    # members that belong to an EC. Sorted order keeps each EC's members contiguous.
    elem_ec = np.repeat(ec_id_of_group, sizes)
    elem_size = np.repeat(sizes, sizes)
    member = elem_ec >= 0

    m_ec = elem_ec[member]
    m_size = elem_size[member].astype(np.int32)
    logger.info("Grouped %d anchors into %d ECs (%d member rows) in %.1fs", N, n_ec, int(member.sum()), time.perf_counter() - t0)

    t0 = time.perf_counter()
    flat_refs = [ref_names[r] for r in ref_ids[member].tolist()]
    flat_pos = positions[member].tolist()
    flat_prefix = [seq_prefix[r] for r in flat_refs]

    labels = _annotate_labels(flat_refs, flat_prefix, flat_pos, gtf)
    logger.info("Built %d member labels%s in %.1fs", len(labels), " (with GTF gene names)" if gtf else "", time.perf_counter() - t0)

    # One header per EC: sort+join the labels of its (contiguous) members.
    t0 = time.perf_counter()
    ec_sizes = sizes[is_ec]
    headers = [None] * n_ec
    s = 0
    for j, sz in enumerate(tqdm(ec_sizes.tolist(), desc="Joining EC headers", unit="EC")):
        headers[j] = ";".join(sorted(labels[s:s + sz]))
        s += sz
    logger.info("Joined %d EC headers in %.1fs", n_ec, time.perf_counter() - t0)

    positions_df = pd.DataFrame({seq_id_column: flat_refs, position_column: flat_pos, "ec_id": m_ec, "ec_size": m_size})
    # int32 keeps the column small while still covering huge repeat classes (int16
    # tops out at 32767, which a repetitive k-mer's EC can exceed).
    positions_df["ec_size"] = positions_df["ec_size"].astype("int32")
    ecs_df = pd.DataFrame({"ec_id": np.arange(n_ec), "header": headers})

    logger.info("Built %d equivalence classes covering %d positions.", len(ecs_df), len(positions_df))
    return positions_df, ecs_df, seq_meta_df


def write_ec_tables(positions_df, ecs_df, seq_meta_df, outdir):
    """Write the three tables to <outdir> as parquet (typed, compressed, fast to load)."""
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    positions_df.to_parquet(outdir / "positions.parquet", index=False)
    ecs_df.to_parquet(outdir / "ecs.parquet", index=False)
    seq_meta_df.to_parquet(outdir / "seq_meta.parquet", index=False)
    logger.info("Wrote tables to %s/", outdir)


def substitute(query_df, table_dir, fill_singletons=True, seq_id_column="seq_id", position_column="position"):
    """
    Substitute HGVS-like locations with their equivalence-class header.

    Parameters
    ----------
    query_df : DataFrame with a seq-id column and a position column (1-based anchor
        position), named by seq_id_column / position_column.
    table_dir : directory holding positions.parquet / ecs.parquet / seq_meta.parquet.
    fill_singletons : if True, positions absent from any multi-member EC get their own
        plain "<seq_id>:<prefix>.<pos>" label (their class is just themselves); if False
        they are left as NaN.
    seq_id_column / position_column : the seq-id and position column names, matching
        those the tables were built with.

    Returns
    -------
    A copy of query_df with added 'header' (the full equivalence class) and
    'ec_size' (member count) columns.
    """
    table_dir = Path(table_dir)
    positions = pd.read_parquet(table_dir / "positions.parquet")
    ecs = pd.read_parquet(table_dir / "ecs.parquet")

    out = query_df.copy()
    out[seq_id_column] = out[seq_id_column].astype(str)
    out[position_column] = out[position_column].astype(int)

    out = out.merge(positions, on=[seq_id_column, position_column], how="left")
    out = out.merge(ecs, on="ec_id", how="left")
    out = out.drop(columns="ec_id")

    if fill_singletons:
        prefix_map = pd.read_parquet(table_dir / "seq_meta.parquet")
        prefix_map = dict(zip(prefix_map[seq_id_column], prefix_map["prefix"]))
        missing = out["header"].isna()
        known = missing & out[seq_id_column].isin(prefix_map)
        out.loc[missing, "header"] = [
            f"{sid}:{prefix_map[sid]}.{pos}" if sid in prefix_map else pd.NA
            for sid, pos in zip(out.loc[missing, seq_id_column], out.loc[missing, position_column])
        ]
        out.loc[known, "ec_size"] = 1  # a position with no shared window is its own class

    out["ec_size"] = out["ec_size"].astype("Int32")  # nullable: unresolved rows stay <NA>
    return out


def _read_query(path):
    """Load a query table (seq_id, position) from parquet or CSV by extension."""
    if isinstance(path, pd.DataFrame):
        return path
    path = Path(path)
    if path.suffix.lower() in (".parquet", ".pq"):
        return pd.read_parquet(path)
    return pd.read_csv(path)


def _write_query(df, path):
    path = Path(path)
    if path.suffix.lower() in (".parquet", ".pq"):
        df.to_parquet(path, index=False)
    else:
        df.to_csv(path, index=False)



def build_kmer_index(cdna=None, dna=None, k=31, anchor="middle", gtf=None, out="kmer_ec_tables", seq_id_column="seq_id", position_column="position", skip_truncated=False):
    start_time = time.perf_counter()
    if not cdna and not dna:
        raise SystemExit("provide at least one FASTA via --cdna and/or --dna")

    try:
        left_extent, right_extent = anchor_extents(anchor, k)
    except ValueError as e:
        raise SystemExit(str(e))

    inputs = [(fasta, "c") for fasta in cdna] + [(fasta, "g") for fasta in dna]

    kmer_arrays, ref_names, seq_prefix = build_kmer_map(inputs, left_extent, right_extent, skip_truncated=skip_truncated)
    positions_df, ecs_df, seq_meta_df = build_ec_tables(kmer_arrays, ref_names, seq_prefix, gtf=gtf, seq_id_column=seq_id_column, position_column=position_column)
    write_ec_tables(positions_df, ecs_df, seq_meta_df, out)
    logger.info("Finished building k-mer equivalence-class tables in %.1f seconds.", time.perf_counter() - start_time)


def map_to_kmer_index(table, query, no_singletons=False, out=None, seq_id_column="seq_id", position_column="position"):
    start_time = time.perf_counter()
    query = _read_query(query)
    for col in (seq_id_column, position_column):
        if col not in query.columns:
            raise SystemExit(f"query is missing required column '{col}'")

    result = substitute(query, table, fill_singletons=not no_singletons, seq_id_column=seq_id_column, position_column=position_column)
    if out:
        _write_query(result, out)
        logger.info("Wrote %d substituted rows to %s", len(result), out)
    logger.info("Finished mapping %d query rows to k-mer equivalence-class headers in %.1f seconds.", len(result), time.perf_counter() - start_time)
    return result


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")

    parser = argparse.ArgumentParser(description="Build k-mer equivalence-class tables and substitute HGVS-like locations with their EC header.")
    sub = parser.add_subparsers(dest="command", required=True)

    build = build_kmer_index
    b = sub.add_parser("build", help="Build the equivalence-class tables from FASTA(s).")
    b.add_argument("--cdna", nargs="*", default=[], metavar="FASTA", help="Transcript/cDNA FASTA(s); labelled 'c.'.")
    b.add_argument("--dna", nargs="*", default=[], metavar="FASTA", help="Genomic FASTA(s); labelled 'g.'.")
    b.add_argument("-k", type=int, default=31, help="Window size (default: 31). Must be odd when --anchor is middle. Any k is allowed (windows wider than 31 bases use multiple int64 limbs).")
    b.add_argument("--anchor", choices=["l", "left", "r", "right", "m", "middle"], default="middle", help="Which base the position labels: l/left (window start), r/right (window end), or m/middle (center, requires odd k; default).")
    b.add_argument("--skip-truncated", action="store_true", help="Only emit anchors with a full window; skip edge/truncated k-mers (as the original single-k-mer builder did). Skipped positions become singletons at query time.")
    b.add_argument("--gtf", default=None, help="Optional GTF; bakes gene names into headers ('<ref>(<GENE>):<prefix>.<pos>').")
    b.add_argument("-o", "--out", default="kmer_ec_tables", help="Output directory for the parquet tables.")
    b.add_argument("--seq_id_column", default="seq_id", help="Name for the seq-id column in the output tables (default: seq_id).")
    b.add_argument("--position_column", default="position", help="Name for the position column in the output tables (default: position).")
    b.set_defaults(func=build)

    map = map_to_kmer_index
    m = sub.add_parser("map", help="Substitute a query of (seq_id, position) with EC headers.")
    m.add_argument("--table", required=True, help="Directory of parquet tables produced by 'build'.")
    m.add_argument("--query", required=True, help="Query CSV/parquet with columns seq_id, position.")
    m.add_argument("--no-singletons", action="store_true", help="Leave positions with no shared EC as NaN instead of their own label.")
    m.add_argument("-o", "--out", help="Output CSV/parquet with an added 'header' column.")
    m.add_argument("--seq_id_column", default="seq_id", help="Name of the seq-id column in the query (default: seq_id).")
    m.add_argument("--position_column", default="position", help="Name of the position column in the query (default: position).")
    m.set_defaults(func=map)

    kwargs = vars(parser.parse_args())
    func = kwargs.pop("func")
    kwargs.pop("command", None)  # dispatch-only keys aren't handler parameters
    func(**kwargs)


if __name__ == "__main__":
    main()

# Build the tables:
# python build_kmer_map.py build \
#     -o kmer_ec_tables \
#     -k 31 \
#     --gtf  /home/jrich/Desktop/varseek-examples/trash/data/reference/Homo_sapiens.GRCh38.gtf \
#     --cdna /home/jrich/Desktop/varseek-examples/trash/data/reference/Homo_sapiens.GRCh38.cdna.all.fa \
#     --dna  /home/jrich/Desktop/varseek-examples/trash/data/reference/Homo_sapiens.GRCh38.dna.primary_assembly.fa
#
# Substitute a query with EC headers:
# python build_kmer_map.py map --table kmer_ec_tables --query queries.csv -o queries_with_ec.csv