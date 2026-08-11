"""Tests for how vk denovo resolves the bam2vcf quality gates.

bam2vcf applies no statistical model, so min_mapq/min_baseq/min_vaf are the only thing
keeping it from emitting every sequencing error in the BAM. denovo defaults the two quality
gates to the benchmarked values instead of 0, while leaving them as no-ops for the other
callers (which express the same filters through mpileup's -q/-Q and --include). min_vaf
stays at 0 and only warns, because a fraction floor drops real low-fraction variants. The
signature keeps None as "unset" so an explicit value can still be told apart from a default.
"""

import inspect
import logging
import random

import pysam
import pytest

import varseek.varseek_denovo as vd
from varseek.varseek_denovo import (
    BAM2VCF_DEFAULT_MIN_BASEQ,
    BAM2VCF_DEFAULT_MIN_MAPQ,
    BAM2VCF_DEFAULT_MIN_VAF,
    BAM2VCF_LOW_MIN_VAF,
    BAM2VCF_SUGGESTED_MIN_VAF_DNA,
    DenovoParams,
    denovo,
)

READLEN = 100
VARIANT_POS1 = 151  # 1-based position of the planted substitution


@pytest.fixture(scope="module")
def tiny_bam(tmp_path_factory):
    """A BAM carrying one substitution supported by 6 reads, two of them low quality.

    One supporting read is mapped at MAPQ 5 and another carries base quality 5 at the
    variant, so the default gates must drop exactly those two.
    """
    out = tmp_path_factory.mktemp("denovo_defaults")
    random.seed(3)
    contig = "".join(random.choice("ACGT") for _ in range(1000))

    fasta = out / "ref.fa"
    with open(fasta, "w") as handle:
        handle.write(">chr1\n")
        for i in range(0, len(contig), 60):
            handle.write(contig[i : i + 60] + "\n")
    pysam.faidx(str(fasta))

    alt = {"A": "C", "C": "A", "G": "T", "T": "G"}[contig[VARIANT_POS1 - 1]]
    reads = [(i * 5, contig[i * 5 : i * 5 + READLEN], 60, 40) for i in range(20)]
    for i in range(6):
        pos0 = 60 + i
        seq = list(contig[pos0 : pos0 + READLEN])
        seq[VARIANT_POS1 - 1 - pos0] = alt
        reads.append((pos0, "".join(seq), 5 if i == 0 else 60, 5 if i == 1 else 40))

    bam = out / "in.bam"
    header = {"HD": {"VN": "1.6", "SO": "coordinate"}, "SQ": [{"SN": "chr1", "LN": len(contig)}]}
    with pysam.AlignmentFile(str(bam), "wb", header=header) as handle:
        for n, (pos0, seq, mapq, baseq) in enumerate(sorted(reads)):
            record = pysam.AlignedSegment()
            record.query_name, record.query_sequence, record.flag = f"r{n}", seq, 0
            record.reference_id, record.reference_start, record.mapping_quality = 0, pos0, mapq
            record.cigarstring = f"{len(seq)}M"
            record.query_qualities = pysam.qualitystring_to_array(chr(33 + 40) * len(seq))
            if baseq != 40:
                qualities = record.query_qualities
                qualities[VARIANT_POS1 - 1 - pos0] = baseq
                record.query_qualities = qualities
            handle.write(record)
    pysam.index(str(bam))
    return {"bam": str(bam), "fasta": str(fasta)}


def _call_denovo(monkeypatch, tiny_bam, tmp_path, technology="dna", **kwargs):
    """Run denovo on the fixture BAM, returning the kwargs bam_to_vcf actually received."""
    seen = {}
    real = vd.bam_to_vcf

    def spy(**kw):
        seen.update(kw)
        return real(**kw)

    monkeypatch.setattr(vd, "bam_to_vcf", spy)
    output = str(tmp_path / "out.vcf")
    denovo(inputs=[tiny_bam["bam"]], sequences=tiny_bam["fasta"], output=output,
           technology=technology, overwrite=True, **kwargs)
    records = [line for line in open(output) if not line.startswith("#")]
    return seen, records


def test_signature_leaves_quality_gates_unset():
    """None means "unset", which is what lets an explicit 0 differ from a default."""
    parameters = inspect.signature(denovo).parameters
    for name in ("min_vaf", "min_mapq", "min_baseq"):
        assert parameters[name].default is None


def test_defaults_reach_bam2vcf(monkeypatch, tiny_bam, tmp_path):
    seen, records = _call_denovo(monkeypatch, tiny_bam, tmp_path)
    assert seen["min_mapq"] == BAM2VCF_DEFAULT_MIN_MAPQ
    assert seen["min_baseq"] == BAM2VCF_DEFAULT_MIN_BASEQ
    assert BAM2VCF_DEFAULT_MIN_MAPQ > 0 and BAM2VCF_DEFAULT_MIN_BASEQ > 0
    # the low-mapq and low-baseq reads are excluded from AO
    assert len(records) == 1
    assert "AO=4;" in records[0]


def test_min_vaf_defaults_to_no_floor(monkeypatch, tiny_bam, tmp_path):
    """Sensitivity first: the fraction floor is the one gate that drops real variants."""
    assert BAM2VCF_DEFAULT_MIN_VAF == 0.0
    seen, _ = _call_denovo(monkeypatch, tiny_bam, tmp_path)
    assert seen["min_vaf"] == 0.0


def test_explicit_zeros_restore_the_unfiltered_behaviour(monkeypatch, tiny_bam, tmp_path):
    seen, records = _call_denovo(monkeypatch, tiny_bam, tmp_path, min_mapq=0, min_baseq=0, min_vaf=0.0)
    assert (seen["min_mapq"], seen["min_baseq"], seen["min_vaf"]) == (0, 0, 0.0)
    assert "AO=6;" in records[0]


def test_explicit_values_are_passed_through(monkeypatch, tiny_bam, tmp_path):
    seen, _ = _call_denovo(monkeypatch, tiny_bam, tmp_path, min_mapq=30, min_baseq=10, min_vaf=0.05)
    assert (seen["min_mapq"], seen["min_baseq"], seen["min_vaf"]) == (30, 10, 0.05)


def _vaf_warnings(caplog):
    return [r.message for r in caplog.records if r.levelno == logging.WARNING and "min_vaf" in r.message]


def test_low_min_vaf_warns_with_dna_advice(monkeypatch, tiny_bam, tmp_path, caplog):
    with caplog.at_level(logging.WARNING, logger=vd.logger.name):
        _call_denovo(monkeypatch, tiny_bam, tmp_path, technology="dna")
    warnings = _vaf_warnings(caplog)
    assert len(warnings) == 1
    assert f"min_vaf={BAM2VCF_SUGGESTED_MIN_VAF_DNA}" in warnings[0]
    assert "germline DNA" in warnings[0]


def test_low_min_vaf_warning_points_rna_at_min_counts(monkeypatch, tiny_bam, tmp_path, caplog):
    """A fraction floor is the wrong lever for RNA, so the advice differs."""
    with caplog.at_level(logging.WARNING, logger=vd.logger.name):
        _call_denovo(monkeypatch, tiny_bam, tmp_path, technology="10xv3")
    warnings = _vaf_warnings(caplog)
    assert len(warnings) == 1
    assert "allele-specific expression" in warnings[0]
    assert "min_counts" in warnings[0]
    assert f"min_vaf={BAM2VCF_SUGGESTED_MIN_VAF_DNA}" not in warnings[0]


def test_no_warning_once_a_floor_is_set(monkeypatch, tiny_bam, tmp_path, caplog):
    with caplog.at_level(logging.WARNING, logger=vd.logger.name):
        _call_denovo(monkeypatch, tiny_bam, tmp_path, min_vaf=BAM2VCF_LOW_MIN_VAF)
    assert _vaf_warnings(caplog) == []


@pytest.mark.parametrize("caller", ["bcftools", "cigar"])
def test_other_callers_unaffected_by_the_defaults(caller, tmp_path):
    """The defaults must not turn into a bam2vcf-only error for everyone else."""
    output = str(tmp_path / ("out.csv" if caller == "cigar" else "out.vcf"))
    DenovoParams(variant_caller=caller, output=output)


@pytest.mark.parametrize("kwargs", [{"min_mapq": 20}, {"min_baseq": 20}, {"min_vaf": 0.2}])
def test_explicit_values_still_rejected_for_other_callers(kwargs, tmp_path):
    with pytest.raises(ValueError, match="require variant_caller='bam2vcf'"):
        DenovoParams(variant_caller="bcftools", output=str(tmp_path / "out.vcf"), **kwargs)


@pytest.mark.parametrize("kwargs", [{"min_mapq": 0}, {"min_baseq": 0}, {"min_vaf": 0.0}])
def test_explicit_no_ops_allowed_for_other_callers(kwargs, tmp_path):
    DenovoParams(variant_caller="bcftools", output=str(tmp_path / "out.vcf"), **kwargs)


@pytest.mark.parametrize("kwargs, pattern", [
    ({"min_vaf": 1.5}, "between 0 and 1"),
    ({"min_vaf": 0.9, "max_vaf": 0.5}, "exceeds"),
    ({"min_mapq": -1}, "non-negative"),
])
def test_range_checks_survive(kwargs, pattern, tmp_path):
    with pytest.raises(ValueError, match=pattern):
        DenovoParams(output=str(tmp_path / "out.vcf"), **kwargs)
