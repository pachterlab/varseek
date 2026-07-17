#!/usr/bin/env python3

from collections import defaultdict
from pathlib import Path
from tqdm import tqdm
import time
import argparse

import pandas as pd


DNA = {
    "A": 0,
    "C": 1,
    "G": 2,
    "T": 3,
}


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


def build_kmer_map(inputs, left_extent, right_extent):
    """
    Group reference positions into k-mer equivalence classes.

    Every valid (A/C/G/T) base is treated as an anchor. The window spans up to
    `left_extent` bases to the left of the anchor and `right_extent` to the
    right, truncated at the sequence ends and at any invalid base (so a window
    never crosses an N or soft-masked base). The anchor itself is the base that
    the position label refers to; where it sits in the window is controlled by
    the caller (left / right / middle) via the two extents.

    Parameters
    ----------
    inputs : list[(str, str)]
        (fasta_path, prefix) pairs, where prefix is "c" (transcript) or "g"
        (genomic); it becomes the HGVS-style tag in each location label.
    left_extent, right_extent : int
        Max bases of context on each side of the anchor.

    Returns
    -------
    table : dict[(int, int, int), list[str]]
        Maps (left_len, right_len, packed_window) to the location labels
        ("<ref>:<prefix>.<anchor_position>") that share that window, where
        anchor_position is the 1-based coordinate of the anchor base. left_len /
        right_len record how many bases of context the window actually has on
        each side, so a truncated edge window cannot collide with a longer one
        that happens to share a suffix/prefix.
    seq_prefix : dict[str, str]
        Maps each ref name to the prefix ("c"/"g") it was labelled with, so a
        singleton position's HGVS-like label can be reconstructed at query time.
    """

    table = defaultdict(list)
    seq_prefix = {}

    # A window never exceeds k = left_extent + right_extent + 1 bases. Keeping a
    # rolling packed value masked to the last k bases (win_mask) lets every anchor
    # window be read off as a masked suffix, so packing is O(1) per base instead of
    # re-scanning the whole window. span_masks[s] isolates the low s bases (2 bits
    # each); precomputed so the per-anchor mask is never rebuilt as a big int.
    k = left_extent + right_extent + 1
    win_mask = (1 << (2 * k)) - 1
    span_masks = [(1 << (2 * s)) - 1 for s in range(k + 1)]

    for fasta, prefix in inputs:
        print(f"Processing {Path(fasta).name}...")
        fasta_len = sum(1 for _ in fasta_reader(fasta))

        for ref, seq in tqdm(fasta_reader(fasta), desc="Building k-mer map", unit="sequence", total=fasta_len):

            seq_prefix[ref] = prefix
            n = len(seq)
            codes = [DNA.get(base, -1) for base in seq]  # -1 for any non-ACGT base

            # Walk maximal runs of valid (A/C/G/T) bases; a run boundary is exactly
            # where the explicit expansion used to stop, so windows never span an N.
            j = 0
            while j < n:
                if codes[j] < 0:
                    j += 1
                    continue

                start = j
                while j < n and codes[j] >= 0:
                    j += 1
                run = j - start  # number of valid bases in [start, j)

                # Roll a masked packed window across the run. rw[i] holds the packed
                # value of the last min(k, i + 1) bases ending at run index i (MSB
                # first), each step a single shift-or-mask on a bounded-width int.
                rw = [0] * run
                w = 0
                for i in range(run):
                    w = ((w << 2) | codes[start + i]) & win_mask
                    rw[i] = w

                # One key per anchor. Context is truncated at the run ends: at most
                # left_extent bases to the left, right_extent to the right, capped by
                # how much of the run is actually there. The anchor window ends at
                # hi and has `span` bases, so it's the low `span` bases of rw[hi].
                for i in range(run):
                    left_len = i if i < left_extent else left_extent
                    right_avail = run - 1 - i
                    right_len = right_avail if right_avail < right_extent else right_extent
                    hi = i + right_len
                    span = left_len + right_len + 1
                    packed = rw[hi] & span_masks[span]
                    key = (left_len, right_len, packed)  # (left_len, right_len, window)
                    table[key].append(f"{ref}:{prefix}.{start + i + 1}")

    return table, seq_prefix


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

    # Reuse varseek's mapping over the flat member list in one vectorised call.
    from varseek.utils.varseek_build_utils import compute_gene_name_series_for_headers

    members = pd.DataFrame({"seq_id": refs, "mutation": [f"{p}.{pos}" for p, pos in zip(prefixes, positions)]})
    genes = compute_gene_name_series_for_headers(members, "seq_id", "mutation", gtf).tolist()

    return [f"{r}({g}):{p}.{pos}" if g else plabel for r, p, pos, g, plabel in zip(refs, prefixes, positions, genes, plain)]


def build_ec_tables(table, seq_prefix, gtf=None, seq_id_column="seq_id", position_column="position"):
    """
    Turn the raw k-mer map into three compact, lookup-ready tables.

    Only windows shared by 2+ locations form an equivalence class (EC); a position
    absent from the map is its own singleton class. The EC's members are stored once
    (normalised by an integer ec_id) rather than repeating the joined class on every
    row, so storage stays O(total members) instead of O(sum of class_size^2).

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
    print("Assembling equivalence-class tables...")

    classes = [m for m in table.values() if len(m) >= 2]

    # Flatten every ambiguous member, tagged with its ec_id (contiguous by class)
    # and its EC's member count.
    flat_refs, flat_prefix, flat_pos, flat_ec, flat_size = [], [], [], [], []
    for ec_id, members in enumerate(tqdm(classes, desc="Flattening classes", unit="EC")):
        size = len(members)
        for label in members:
            ref, prefix, pos = parse_label(label)
            flat_refs.append(ref)
            flat_prefix.append(prefix)
            flat_pos.append(pos)
            flat_ec.append(ec_id)
            flat_size.append(size)

    labels = _annotate_labels(flat_refs, flat_prefix, flat_pos, gtf)

    # Group the (now annotated) labels back into one header per EC.
    headers = [None] * len(classes)
    start = 0
    for ec_id, members in enumerate(classes):
        end = start + len(members)
        headers[ec_id] = ";".join(sorted(labels[start:end]))
        start = end

    positions_df = pd.DataFrame({seq_id_column: flat_refs, position_column: flat_pos, "ec_id": flat_ec, "ec_size": flat_size})
    # int32 keeps the column small while still covering huge repeat classes (int16
    # tops out at 32767, which a repetitive k-mer's EC can exceed).
    positions_df["ec_size"] = positions_df["ec_size"].astype("int32")
    ecs_df = pd.DataFrame({"ec_id": range(len(classes)), "header": headers})
    seq_meta_df = pd.DataFrame({seq_id_column: list(seq_prefix), "prefix": list(seq_prefix.values())})

    print(f"Built {len(ecs_df)} equivalence classes covering {len(positions_df)} positions.")
    return positions_df, ecs_df, seq_meta_df


def write_ec_tables(positions_df, ecs_df, seq_meta_df, outdir):
    """Write the three tables to <outdir> as parquet (typed, compressed, fast to load)."""
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    positions_df.to_parquet(outdir / "positions.parquet", index=False)
    ecs_df.to_parquet(outdir / "ecs.parquet", index=False)
    seq_meta_df.to_parquet(outdir / "seq_meta.parquet", index=False)
    print(f"Wrote tables to {outdir}/")


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



def build_kmer_index(cdna=None, dna=None, k=31, anchor="middle", gtf=None, out="kmer_ec_tables", seq_id_column="seq_id", position_column="position"):
    start_time = time.perf_counter()
    if not cdna and not dna:
        raise SystemExit("provide at least one FASTA via --cdna and/or --dna")

    try:
        left_extent, right_extent = anchor_extents(anchor, k)
    except ValueError as e:
        raise SystemExit(str(e))

    inputs = [(fasta, "c") for fasta in cdna] + [(fasta, "g") for fasta in dna]

    table, seq_prefix = build_kmer_map(inputs, left_extent, right_extent)
    positions_df, ecs_df, seq_meta_df = build_ec_tables(table, seq_prefix, gtf=gtf, seq_id_column=seq_id_column, position_column=position_column)
    write_ec_tables(positions_df, ecs_df, seq_meta_df, out)
    print(f"Finished building k-mer equivalence-class tables in {time.perf_counter() - start_time:.1f} seconds.")


def map_to_kmer_index(table, query, no_singletons=False, out=None, seq_id_column="seq_id", position_column="position"):
    start_time = time.perf_counter()
    query = _read_query(query)
    for col in (seq_id_column, position_column):
        if col not in query.columns:
            raise SystemExit(f"query is missing required column '{col}'")

    result = substitute(query, table, fill_singletons=not no_singletons, seq_id_column=seq_id_column, position_column=position_column)
    if out:
        _write_query(result, out)
        print(f"Wrote {len(result)} substituted rows to {out}")
    print(f"Finished mapping {len(result)} query rows to k-mer equivalence-class headers in {time.perf_counter() - start_time:.1f} seconds.")
    return result


def main():
    parser = argparse.ArgumentParser(description="Build k-mer equivalence-class tables and substitute HGVS-like locations with their EC header.")
    sub = parser.add_subparsers(dest="command", required=True)

    build = build_kmer_index
    b = sub.add_parser("build", help="Build the equivalence-class tables from FASTA(s).")
    b.add_argument("--cdna", nargs="*", default=[], metavar="FASTA", help="Transcript/cDNA FASTA(s); labelled 'c.'.")
    b.add_argument("--dna", nargs="*", default=[], metavar="FASTA", help="Genomic FASTA(s); labelled 'g.'.")
    b.add_argument("-k", type=int, default=31, help="Window size (default: 31). Must be odd when --anchor is middle.")
    b.add_argument("--anchor", choices=["l", "left", "r", "right", "m", "middle"], default="middle", help="Which base the position labels: l/left (window start), r/right (window end), or m/middle (center, requires odd k; default).")
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