"""Local genome <-> transcript variant coordinate conversion for varseek build.

`vk build` applies HGVS variant positions directly against the supplied ``sequences``,
so the variant annotation and the reference must share a coordinate system. This module
projects variants between genome coordinates (``g.``) and transcript/cDNA coordinates
(``c.``) using only a GTF (for exon structure) -- no external database (unlike the
``hgvs`` package's UTA-backed ``g_to_t``/``t_to_g``).

Following varseek's internal convention, ``c.`` positions here are **cDNA/transcript-
relative** (1-based along the spliced transcript), matching the cDNA fasta that build
mutates -- consistent with :func:`convert_mutation_cds_locations_to_cdna`.

The two public entry points, :func:`convert_variants_genome_to_transcript` and
:func:`convert_variants_transcript_to_genome`, take a variants DataFrame and return a new
DataFrame with the ``seq_id_column``/``var_column`` rewritten (and, for genome->transcript,
one output row per overlapping transcript), plus a report dict counting dropped variants.
"""

import logging
import re

import pandas as pd

from varseek.constants import mutation_pattern
from varseek.utils.logger_utils import set_up_logger
from varseek.utils.seq_utils import reverse_complement

logger = logging.getLogger(__name__)
logger = set_up_logger(logger, logging_level="INFO", save_logs=False, log_dir=None)

# variant strings containing any of these markers are intronic/uncertain/ambiguous and cannot be
# projected between coordinate systems (same set build itself filters out downstream).
_UNSUPPORTED_MARKER_PATTERN = re.compile(r"[\?\(\)\+\-\*]")


def _strip_version(value):
    """Drop an Ensembl-style version suffix (``ENST0001.3`` -> ``ENST0001``)."""
    return str(value).split(".")[0]


def build_transcript_exon_model(gtf_path):
    """Parse a GTF into an in-memory exon model supporting both conversion directions.

    Returns a dict with two views onto the same exons:

    - ``transcripts``: ``{transcript_id: {"chrom", "strand", "exons": [exon, ...]}}`` where
      ``exons`` is ordered in transcript (5'->3') order. Used for transcript->genome.
    - ``chrom_exons``: ``{chrom: [exon, ...]}`` (all exons on a chromosome). Used to find every
      transcript overlapping a genomic position for genome->transcript.

    Each ``exon`` is a dict with genomic bounds ``g_start``/``g_end`` (1-based inclusive, GTF
    convention, always ``g_start <= g_end``), ``strand``, the exon's cDNA range
    ``cdna_lo``/``cdna_hi`` (1-based, along the spliced transcript), and ``transcript_id``.
    Transcript and chromosome ids have Ensembl version suffixes stripped to match how build
    keys its sequence dict.
    """
    gtf_df = pd.read_csv(
        gtf_path,
        sep="\t",
        comment="#",
        header=None,
        names=["seqname", "source", "feature", "start", "end", "score", "strand", "frame", "attribute"],
        dtype={"seqname": str, "feature": str, "start": int, "end": int, "strand": str, "attribute": str},
    )

    gtf_df = gtf_df[gtf_df["feature"] == "exon"].copy()
    if gtf_df.empty:
        raise ValueError(f"No exon features found in GTF '{gtf_path}'. A GTF with exon records is required for coordinate conversion.")

    gtf_df["transcript_id"] = gtf_df["attribute"].str.extract(r'transcript_id "([^"]+)"')[0]
    gtf_df = gtf_df.dropna(subset=["transcript_id"])
    gtf_df["transcript_id"] = gtf_df["transcript_id"].map(_strip_version)
    gtf_df["seqname"] = gtf_df["seqname"].map(_strip_version)

    transcripts = {}
    chrom_exons = {}

    # group by transcript so we can lay out cDNA coordinates exon-by-exon in transcript order
    for transcript_id, tx_rows in gtf_df.groupby("transcript_id", sort=False):
        chrom = tx_rows["seqname"].iloc[0]
        strand = tx_rows["strand"].iloc[0]
        # transcript order: 5'->3' along the transcript. On + strand that is ascending genomic
        # coordinate; on - strand it is descending.
        ascending = strand != "-"
        tx_rows = tx_rows.sort_values("start", ascending=ascending)

        exons = []
        cdna_cursor = 1  # 1-based cDNA coordinate of the next exon's 5' base
        for g_start, g_end in zip(tx_rows["start"], tx_rows["end"]):
            length = g_end - g_start + 1
            exon = {
                "g_start": int(g_start),
                "g_end": int(g_end),
                "strand": strand,
                "cdna_lo": cdna_cursor,
                "cdna_hi": cdna_cursor + length - 1,
                "transcript_id": transcript_id,
                "chrom": chrom,
            }
            exons.append(exon)
            chrom_exons.setdefault(chrom, []).append(exon)
            cdna_cursor += length

        transcripts[transcript_id] = {"chrom": chrom, "strand": strand, "exons": exons}

    return {"transcripts": transcripts, "chrom_exons": chrom_exons}


def _genomic_to_cdna(exon, g_pos):
    """cDNA position of ``g_pos`` within ``exon``, or ``None`` if outside the exon."""
    if not (exon["g_start"] <= g_pos <= exon["g_end"]):
        return None
    if exon["strand"] == "-":
        return exon["cdna_lo"] + (exon["g_end"] - g_pos)
    return exon["cdna_lo"] + (g_pos - exon["g_start"])


def _cdna_to_genomic(exon, c_pos):
    """Genomic position of cDNA ``c_pos`` within ``exon``, or ``None`` if outside the exon."""
    if not (exon["cdna_lo"] <= c_pos <= exon["cdna_hi"]):
        return None
    if exon["strand"] == "-":
        return exon["g_end"] - (c_pos - exon["cdna_lo"])
    return exon["g_start"] + (c_pos - exon["cdna_lo"])


def _parse_variant(var):
    """Split an HGVS ``c.``/``g.`` variant string into (start, end, variant_type, actual).

    Returns ``None`` if the variant is unsupported for conversion (intronic/uncertain/ambiguous
    markers, or not matching the expected substitution/indel grammar). Positions are 1-based ints.
    """
    if not isinstance(var, str) or _UNSUPPORTED_MARKER_PATTERN.search(var):
        return None
    match = re.search(mutation_pattern, var)
    if not match:
        return None
    positions, actual = match.group(1), match.group(2)

    if ">" in actual:
        variant_type = "substitution"
    elif actual.startswith("delins"):
        variant_type = "delins"
    elif actual.startswith("ins"):
        variant_type = "insertion"
    elif actual == "del":
        variant_type = "deletion"
    elif actual == "dup":
        variant_type = "duplication"
    elif actual == "inv":
        variant_type = "inversion"
    else:
        return None

    pieces = positions.split("_")
    try:
        start = int(pieces[0])
        end = int(pieces[1]) if len(pieces) > 1 else start
    except (ValueError, IndexError):
        return None
    return start, end, variant_type, actual


def _reverse_complement_actual(actual, variant_type):
    """Reverse-complement the reference/inserted bases of a variant for a minus-strand transcript.

    ``del``/``dup``/``inv`` carry no explicit bases and are returned unchanged.
    """
    if variant_type == "substitution":
        ref, alt = actual.split(">")
        return f"{reverse_complement(ref)}>{reverse_complement(alt)}"
    if variant_type in ("insertion", "delins"):
        prefix = "delins" if variant_type == "delins" else "ins"
        seq = actual[len(prefix):]
        return f"{prefix}{reverse_complement(seq)}"
    return actual


def _format_variant(prefix, new_start, new_end, actual):
    if new_start == new_end:
        return f"{prefix}.{new_start}{actual}"
    return f"{prefix}.{new_start}_{new_end}{actual}"


def _build_converted_row(row, seq_id_column, var_column, new_seq_id, new_prefix, exon, start, end, variant_type, actual):
    """Project one variant through ``exon`` and return an updated copy of ``row`` (or ``None``).

    ``None`` means the variant's two endpoints do not both fall inside this single exon (it spans
    an exon-exon junction / an intron), which cannot be represented as one contiguous variant in
    the target coordinate system.
    """
    to_target = _genomic_to_cdna if new_prefix == "c" else _cdna_to_genomic
    mapped_start = to_target(exon, start)
    mapped_end = to_target(exon, end)
    if mapped_start is None or mapped_end is None:
        return None  # junction-spanning / outside this exon

    new_start, new_end = sorted((mapped_start, mapped_end))
    new_actual = _reverse_complement_actual(actual, variant_type) if exon["strand"] == "-" else actual

    new_row = row.copy()
    new_row[seq_id_column] = new_seq_id
    new_row[var_column] = _format_variant(new_prefix, new_start, new_end, new_actual)
    return new_row


def _empty_report():
    return {
        "unsupported_variant": 0,      # intronic/uncertain/ambiguous or unparseable
        "seq_id_not_in_gtf": 0,        # transcript (c->g) or chromosome (g->c) absent from GTF
        "no_overlapping_exon": 0,      # variant falls in an intron / intergenic region
        "junction_spanning": 0,        # endpoints in different exons -> not contiguous in target
        "input_rows": 0,
        "output_rows": 0,
    }


def convert_variants_transcript_to_genome(df, seq_id_column, var_column, model):
    """Project transcript/cDNA (``c.``) variants to genome (``g.``) coordinates.

    ``seq_id_column`` is expected to hold transcript ids; on output it holds chromosome names and
    ``var_column`` holds ``g.`` variants. Returns ``(converted_df, report)``. Each input variant maps
    to at most one genomic variant (its transcript's exon structure); variants whose transcript is
    absent from the GTF, that are unsupported, or that span an exon-exon junction are dropped and
    counted in ``report``.
    """
    report = _empty_report()
    report["input_rows"] = len(df)
    transcripts = model["transcripts"]
    out_rows = []

    for _, row in df.iterrows():
        transcript_id = _strip_version(row[seq_id_column])
        parsed = _parse_variant(row[var_column])
        if parsed is None:
            report["unsupported_variant"] += 1
            continue
        start, end, variant_type, actual = parsed

        entry = transcripts.get(transcript_id)
        if entry is None:
            report["seq_id_not_in_gtf"] += 1
            continue

        # locate the exon containing the start of the cDNA variant
        target_exon = next((exon for exon in entry["exons"] if exon["cdna_lo"] <= start <= exon["cdna_hi"]), None)
        if target_exon is None:
            report["no_overlapping_exon"] += 1
            continue

        new_row = _build_converted_row(row, seq_id_column, var_column, entry["chrom"], "g", target_exon, start, end, variant_type, actual)
        if new_row is None:
            report["junction_spanning"] += 1
            continue
        out_rows.append(new_row)

    converted = pd.DataFrame(out_rows, columns=df.columns) if out_rows else df.iloc[0:0].copy()
    report["output_rows"] = len(converted)
    _log_report("transcript->genome", report)
    return converted, report


def convert_variants_genome_to_transcript(df, seq_id_column, var_column, model):
    """Project genome (``g.``) variants to transcript/cDNA (``c.``) coordinates.

    ``seq_id_column`` is expected to hold chromosome names; on output it holds transcript ids and
    ``var_column`` holds ``c.`` variants. A single genomic position can fall within exons of several
    transcripts/isoforms, so this **expands rows**: one output row per transcript whose exon fully
    contains the variant. Variants in introns/intergenic regions, spanning a junction, on an unknown
    chromosome, or otherwise unsupported are dropped and counted in ``report``. Returns
    ``(converted_df, report)``.
    """
    report = _empty_report()
    report["input_rows"] = len(df)
    chrom_exons = model["chrom_exons"]
    out_rows = []

    for _, row in df.iterrows():
        chrom = _strip_version(row[seq_id_column])
        parsed = _parse_variant(row[var_column])
        if parsed is None:
            report["unsupported_variant"] += 1
            continue
        start, end, variant_type, actual = parsed
        var_lo, var_hi = min(start, end), max(start, end)

        exons = chrom_exons.get(chrom)
        if not exons:
            report["seq_id_not_in_gtf"] += 1
            continue

        # every exon that FULLY contains the variant span (both endpoints in the same exon)
        containing_exons = [exon for exon in exons if exon["g_start"] <= var_lo and var_hi <= exon["g_end"]]
        if not containing_exons:
            # distinguish "touches an exon but crosses a junction" from "purely intronic/intergenic"
            touches_exon = any(exon["g_start"] <= var_hi and var_lo <= exon["g_end"] for exon in exons)
            report["junction_spanning" if touches_exon else "no_overlapping_exon"] += 1
            continue

        for exon in containing_exons:
            new_row = _build_converted_row(row, seq_id_column, var_column, exon["transcript_id"], "c", exon, start, end, variant_type, actual)
            if new_row is not None:
                out_rows.append(new_row)

    converted = pd.DataFrame(out_rows, columns=df.columns) if out_rows else df.iloc[0:0].copy()
    report["output_rows"] = len(converted)
    _log_report("genome->transcript", report)
    if report["input_rows"]:
        logger.info("genome->transcript expansion: %d input variant(s) -> %d transcript-level row(s)", report["input_rows"], report["output_rows"])
    return converted, report


def _log_report(direction, report):
    dropped = report["unsupported_variant"] + report["seq_id_not_in_gtf"] + report["no_overlapping_exon"] + report["junction_spanning"]
    if dropped:
        logger.warning(
            "Coordinate conversion (%s) dropped %d variant(s): %d unsupported, %d seq_id-not-in-GTF, %d intronic/intergenic, %d junction-spanning.",
            direction, dropped, report["unsupported_variant"], report["seq_id_not_in_gtf"], report["no_overlapping_exon"], report["junction_spanning"],
        )
