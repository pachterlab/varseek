import gzip

import pandas as pd

from varseek.utils.seq_utils import read_fasta, reverse_complement


def make_intergenic_fasta(
    fasta,
    gtf,
    out,
    feature="gene",
    min_length=1,  # use k
    flank=0
):
    """
    Write a FASTA containing only the intergenic regions of a genome.

    Intergenic regions are the stretches of each chromosome that are NOT covered
    by any feature (default: "gene") in the GTF. Overlapping features are merged
    before the complement is taken, so each intergenic region is reported once.

    Args:
        fasta       (str) Path to the genome DNA FASTA (optionally gzipped).
        gtf         (str) Path to the GTF annotation (optionally gzipped).
        out         (str) Path to the output FASTA (gzipped if it ends with ".gz").
        feature     (str) GTF feature type that defines "genic" regions. Default: "gene".
        min_length  (int) Minimum length of an intergenic region to keep. Default: 1.
        flank       (int) Number of bases to trim from each side of every genic interval
                          before taking the complement (i.e. exclude promoter/UTR-adjacent
                          bases from the intergenic output). Default: 0.

    Returns:
        out (str) Path to the written FASTA.
    """
    # --- read the genic intervals from the GTF ---------------------------------
    gtf_df = pd.read_csv(
        gtf,
        sep="\t",
        comment="#",
        header=None,
        names=["seqname", "source", "feature", "start", "end", "score", "strand", "frame", "attribute"],
        dtype={"seqname": str},
    )
    gtf_df = gtf_df[gtf_df["feature"] == feature]

    # GTF is 1-based inclusive; convert to 0-based half-open [start, end).
    # Merging/complementing is strand-agnostic for intergenic regions.
    genic_by_chrom = {}
    for seqname, group in gtf_df.groupby("seqname"):
        intervals = []
        for start, end in zip(group["start"], group["end"]):
            lo = (start - 1) - flank
            hi = end + flank
            if hi > lo:
                intervals.append((lo, hi))
        genic_by_chrom[seqname] = _merge_intervals(intervals)

    # --- walk the FASTA and emit the complement of the genic intervals ----------
    is_gzipped = out.endswith(".gz")
    open_func = gzip.open if is_gzipped else open
    open_mode = "wt" if is_gzipped else "w"

    n_regions = 0
    with open_func(out, open_mode) as out_fh:
        for header, sequence in read_fasta(fasta):
            chrom = header.split()[0]  # strip description after the first whitespace
            chrom_len = len(sequence)
            genic = genic_by_chrom.get(chrom, [])

            for lo, hi in _complement_intervals(genic, chrom_len):
                lo = max(lo, 0)
                hi = min(hi, chrom_len)
                if hi - lo < min_length:
                    continue
                subseq = sequence[lo:hi]
                # 1-based inclusive coordinates in the header for readability
                out_fh.write(f">{chrom}:{lo + 1}-{hi} intergenic\n")
                for i in range(0, len(subseq), 60):
                    out_fh.write(subseq[i:i + 60] + "\n")
                n_regions += 1

    print(f"Wrote {n_regions} intergenic regions to {out}", flush=True)
    return out


def make_transcriptome_fasta(fasta, gtf, out, feature="exon", id_key="transcript_id"):
    """
    Write a spliced-transcript (cDNA) FASTA from a genome FASTA + GTF.

    For each transcript, its `feature` intervals (default: "exon") are collected,
    sorted, and the genomic sequence for each is concatenated in transcript order
    (i.e. introns are dropped / spliced out). Minus-strand transcripts are
    reverse-complemented so the output is in mRNA 5'->3' orientation. This mirrors
    kb-python's "f1" cDNA FASTA. Pass feature="CDS" to keep only coding sequence.

    Args:
        fasta    (str) Path to the genome DNA FASTA (optionally gzipped).
        gtf      (str) Path to the GTF annotation (optionally gzipped).
        out      (str) Path to the output FASTA (gzipped if it ends with ".gz").
        feature  (str) GTF feature to concatenate per transcript. Default: "exon"
                       (use "CDS" for coding sequence only).
        id_key   (str) GTF attribute used to group features. Default: "transcript_id".

    Returns:
        out (str) Path to the written FASTA.
    """
    # --- read the exon (or CDS) intervals from the GTF, grouped by chromosome ----
    gtf_df = pd.read_csv(
        gtf,
        sep="\t",
        comment="#",
        header=None,
        names=["seqname", "source", "feature", "start", "end", "score", "strand", "frame", "attribute"],
        dtype={"seqname": str},
    )
    gtf_df = gtf_df[gtf_df["feature"] == feature].copy()
    gtf_df[id_key] = gtf_df["attribute"].str.extract(f'{id_key} "([^"]+)"')
    gtf_df = gtf_df.dropna(subset=[id_key])

    # chrom -> {transcript_id: {"strand": s, "exons": [(start0, end), ...]}}
    exons_by_chrom = {}
    for row in gtf_df.itertuples(index=False):
        chrom_txs = exons_by_chrom.setdefault(row.seqname, {})
        tx = chrom_txs.setdefault(getattr(row, id_key), {"strand": row.strand, "exons": []})
        tx["exons"].append((row.start - 1, row.end))  # GTF 1-based inclusive -> 0-based half-open

    # --- walk the FASTA and stitch each transcript's exons together -------------
    is_gzipped = out.endswith(".gz")
    open_func = gzip.open if is_gzipped else open
    open_mode = "wt" if is_gzipped else "w"

    n_transcripts = 0
    with open_func(out, open_mode) as out_fh:
        for header, sequence in read_fasta(fasta):
            chrom = header.split()[0]  # strip description after the first whitespace
            chrom_len = len(sequence)
            chrom_txs = exons_by_chrom.get(chrom, {})

            # emit transcripts in genomic order (by leftmost exon) for determinism
            for tx_id in sorted(chrom_txs, key=lambda t: min(e[0] for e in chrom_txs[t]["exons"])):
                tx = chrom_txs[tx_id]
                exons = sorted(tx["exons"])  # ascending genomic order
                spliced = "".join(sequence[max(lo, 0):min(hi, chrom_len)] for lo, hi in exons)
                if tx["strand"] == "-":
                    spliced = reverse_complement(spliced)
                if not spliced:
                    continue
                out_fh.write(f">{tx_id}\n")
                for i in range(0, len(spliced), 60):
                    out_fh.write(spliced[i:i + 60] + "\n")
                n_transcripts += 1

    print(f"Wrote {n_transcripts} transcripts to {out}", flush=True)
    return out


def _merge_intervals(intervals):
    """Merge a list of [start, end) intervals; returns sorted, non-overlapping list."""
    if not intervals:
        return []
    intervals = sorted(intervals)
    merged = [list(intervals[0])]
    for lo, hi in intervals[1:]:
        if lo <= merged[-1][1]:  # overlapping or adjacent
            merged[-1][1] = max(merged[-1][1], hi)
        else:
            merged.append([lo, hi])
    return [tuple(iv) for iv in merged]


def _complement_intervals(intervals, length):
    """Given merged [start, end) intervals within [0, length), yield the gaps between them."""
    cursor = 0
    for lo, hi in intervals:
        if lo > cursor:
            yield (cursor, lo)
        cursor = max(cursor, hi)
    if cursor < length:
        yield (cursor, length)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Write a FASTA of intergenic regions from a genome FASTA + GTF.")
    parser.add_argument("fasta", help="Genome DNA FASTA (optionally gzipped)")
    parser.add_argument("gtf", help="GTF annotation (optionally gzipped)")
    parser.add_argument("out", help="Output FASTA path (gzipped if it ends with .gz)")
    parser.add_argument("--feature", default="gene", help="GTF feature defining genic regions (default: gene)")
    parser.add_argument("--min-length", type=int, default=1, help="Minimum intergenic region length to keep")
    parser.add_argument("--flank", type=int, default=0, help="Bases to trim from each side of every genic interval")
    args = parser.parse_args()

    make_intergenic_fasta(args.fasta, args.gtf, args.out, feature=args.feature, min_length=args.min_length, flank=args.flank)
