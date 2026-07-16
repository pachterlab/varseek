"""Unit tests for local genome <-> transcript variant coordinate conversion.

Covers both strands, every supported variant type, row expansion for genome->transcript, and the
drop paths (intronic/intergenic, junction-spanning, unknown seq_id, unsupported variant).
"""

import pandas as pd
import pytest

from varseek.utils import (
    build_transcript_exon_model,
    convert_variants_genome_to_transcript,
    convert_variants_transcript_to_genome,
)

# A two-transcript, two-exon-each model on chromosome "1":
#   ENSTP (+ strand): exon g100-109 -> cDNA 1-10,  exon g200-209 -> cDNA 11-20
#   ENSTM (- strand): exon g400-409 -> cDNA 1-10,  exon g300-309 -> cDNA 11-20
# ENSTP2 (+ strand) overlaps ENSTP's first exon exactly, to exercise g->c expansion.
GTF_LINES = [
    ("1", "exon", 100, 109, "+", "ENSTP"),
    ("1", "exon", 200, 209, "+", "ENSTP"),
    ("1", "exon", 300, 309, "-", "ENSTM"),
    ("1", "exon", 400, 409, "-", "ENSTM"),
    ("1", "exon", 100, 109, "+", "ENSTP2"),
]


@pytest.fixture
def model(tmp_path):
    gtf = tmp_path / "mini.gtf"
    with open(gtf, "w") as fh:
        for seqname, feature, start, end, strand, tid in GTF_LINES:
            attr = f'transcript_id "{tid}"; gene_id "G_{tid}";'
            fh.write(f"{seqname}\ttest\t{feature}\t{start}\t{end}\t.\t{strand}\t.\t{attr}\n")
    return build_transcript_exon_model(str(gtf))


def _g2c(model, seq_id, mutation):
    df = pd.DataFrame({"seq_ID": [seq_id], "mutation": [mutation]})
    return convert_variants_genome_to_transcript(df, "seq_ID", "mutation", model)


def _c2g(model, seq_id, mutation):
    df = pd.DataFrame({"seq_ID": [seq_id], "mutation": [mutation]})
    return convert_variants_transcript_to_genome(df, "seq_ID", "mutation", model)


# ------------------------------- exon model -------------------------------

def test_model_layout(model):
    plus = model["transcripts"]["ENSTP"]
    assert plus["strand"] == "+" and plus["chrom"] == "1"
    assert [(e["g_start"], e["g_end"], e["cdna_lo"], e["cdna_hi"]) for e in plus["exons"]] == [(100, 109, 1, 10), (200, 209, 11, 20)]
    minus = model["transcripts"]["ENSTM"]
    # transcript-5' exon on the minus strand is the higher-genomic-coordinate one
    assert [(e["g_start"], e["g_end"], e["cdna_lo"], e["cdna_hi"]) for e in minus["exons"]] == [(400, 409, 1, 10), (300, 309, 11, 20)]


# --------------------------- genome -> transcript ---------------------------

def test_g2c_plus_substitution(model):
    out, rep = _g2c(model, "1", "g.105A>T")
    # g.105 -> cDNA 6 on ENSTP (and ENSTP2, which shares that exon) -> two rows
    assert set(zip(out["seq_ID"], out["mutation"])) == {("ENSTP", "c.6A>T"), ("ENSTP2", "c.6A>T")}
    assert rep["output_rows"] == 2


def test_g2c_minus_substitution_reverse_complements(model):
    out, _ = _g2c(model, "1", "g.405A>T")
    # minus strand: position flips (g.405 -> cDNA 5) and bases are complemented (A>T -> T>A)
    assert list(zip(out["seq_ID"], out["mutation"])) == [("ENSTM", "c.5T>A")]


def test_g2c_plus_deletion(model):
    out, _ = _g2c(model, "1", "g.205_207del")
    assert ("ENSTP", "c.16_18del") in set(zip(out["seq_ID"], out["mutation"]))


def test_g2c_plus_insertion(model):
    out, _ = _g2c(model, "1", "g.100_101insGG")
    assert ("ENSTP", "c.1_2insGG") in set(zip(out["seq_ID"], out["mutation"]))


def test_g2c_minus_insertion_reverse_complements(model):
    out, _ = _g2c(model, "1", "g.404_405insGG")
    # between g404|g405 on the minus strand -> between cDNA 5|6, inserted seq reverse-complemented
    assert list(zip(out["seq_ID"], out["mutation"])) == [("ENSTM", "c.5_6insCC")]


def test_g2c_intronic_dropped(model):
    out, rep = _g2c(model, "1", "g.150A>T")  # between the two exons of ENSTP
    assert out.empty
    assert rep["no_overlapping_exon"] == 1


def test_g2c_junction_spanning_dropped(model):
    out, rep = _g2c(model, "1", "g.108_201del")  # starts in exon1, ends in exon2 of ENSTP
    assert out.empty
    assert rep["junction_spanning"] == 1


def test_g2c_unknown_chromosome_dropped(model):
    out, rep = _g2c(model, "99", "g.105A>T")
    assert out.empty
    assert rep["seq_id_not_in_gtf"] == 1


def test_g2c_unsupported_variant_dropped(model):
    out, rep = _g2c(model, "1", "g.105-2A>T")  # intronic offset marker
    assert out.empty
    assert rep["unsupported_variant"] == 1


# --------------------------- transcript -> genome ---------------------------

def test_c2g_plus_roundtrip(model):
    out, _ = _c2g(model, "ENSTP", "c.6A>T")
    assert list(zip(out["seq_ID"], out["mutation"])) == [("1", "g.105A>T")]


def test_c2g_plus_exon2(model):
    out, _ = _c2g(model, "ENSTP", "c.16A>T")
    assert list(zip(out["seq_ID"], out["mutation"])) == [("1", "g.205A>T")]


def test_c2g_minus_roundtrip(model):
    out, _ = _c2g(model, "ENSTM", "c.5T>A")
    # inverse of the g->c minus substitution test
    assert list(zip(out["seq_ID"], out["mutation"])) == [("1", "g.405A>T")]


def test_c2g_minus_deletion(model):
    out, _ = _c2g(model, "ENSTM", "c.3_5del")
    # cDNA 3..5 on the minus strand maps to g.405..407
    assert list(zip(out["seq_ID"], out["mutation"])) == [("1", "g.405_407del")]


def test_c2g_junction_spanning_dropped(model):
    out, rep = _c2g(model, "ENSTP", "c.9_12del")  # crosses the exon1/exon2 cDNA boundary (10|11)
    assert out.empty
    assert rep["junction_spanning"] == 1


def test_c2g_unknown_transcript_dropped(model):
    out, rep = _c2g(model, "ENST_MISSING", "c.6A>T")
    assert out.empty
    assert rep["seq_id_not_in_gtf"] == 1


def test_c2g_version_suffix_stripped(model):
    out, _ = _c2g(model, "ENSTP.7", "c.6A>T")  # versioned transcript id still resolves
    assert list(zip(out["seq_ID"], out["mutation"])) == [("1", "g.105A>T")]


def test_extra_columns_preserved(model):
    df = pd.DataFrame({"seq_ID": ["ENSTP"], "mutation": ["c.6A>T"], "gene_name": ["MYGENE"]})
    out, _ = convert_variants_transcript_to_genome(df, "seq_ID", "mutation", model)
    assert out.iloc[0]["gene_name"] == "MYGENE"
