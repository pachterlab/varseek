"""Tests for bam_to_vcf and the compiled varseek/cpp/bam2vcf.cpp helper it drives.

The compiled engine and the pysam fallback are held to the same standard: every test that
can run against both does, and test_engines_agree asserts they are byte-identical.
"""

import os
import random

import pysam
import pytest

from varseek.utils import bam_to_vcf
from varseek.utils.native import NativeBuildError, program_path

READLEN = 100


def _native_available():
    try:
        program_path("bam2vcf")
        return True
    except NativeBuildError:
        return False


NATIVE = _native_available()
ENGINES = ["python"] + (["native"] if NATIVE else [])
requires_native = pytest.mark.skipif(not NATIVE, reason="bam2vcf could not be compiled (no htslib or no C++ compiler)")


@pytest.fixture(scope="module")
def synthetic_bam(tmp_path_factory):
    """A small BAM with a known truth set, plus its reference FASTA.

    The reference deliberately contains an 8bp homopolymer at chr1:301-308 and a CAG repeat,
    so that indel left-alignment has something to bite on.
    """
    out = tmp_path_factory.mktemp("bam_to_vcf")
    random.seed(7)

    def rnd(n):
        return "".join(random.choice("ACGT") for _ in range(n))

    chr1 = rnd(300) + "AAAAAAAA" + rnd(200) + "CAGCAGCAGCAG" + rnd(480)
    chr2 = rnd(600)
    ref = {"chr1": chr1, "chr2": chr2}

    fasta = out / "ref.fa"
    with open(fasta, "w") as handle:
        for name, seq in ref.items():
            handle.write(f">{name}\n")
            for i in range(0, len(seq), 60):
                handle.write(seq[i : i + 60] + "\n")
    pysam.faidx(str(fasta))

    flip = {"A": "C", "C": "A", "G": "T", "T": "G"}
    reads = []   # (contig, 0-based start, sequence, cigar)
    truth = []   # (contig, 1-based VCF pos, REF, ALT, expected AO)

    def perfect(contig, pos0, length=READLEN):
        return ref[contig][pos0 : pos0 + length]

    # background depth; also exercises the NM==0 depth-only fast path
    for i in range(20):
        reads.append(("chr1", i * 40, perfect("chr1", i * 40), f"{READLEN}M"))

    # substitutions
    for pos1, n_reads, first_start in ((101, 5, 60), (150, 2, 100)):
        ref_base = chr1[pos1 - 1]
        alt_base = flip[ref_base]
        truth.append(("chr1", pos1, ref_base, alt_base, n_reads))
        for i in range(n_reads):
            pos0 = first_start + i
            seq = list(perfect("chr1", pos0))
            seq[pos1 - 1 - pos0] = alt_base
            reads.append(("chr1", pos0, "".join(seq), f"{READLEN}M"))

    # 2bp deletion of chr1:401-402 -> anchored at 400
    truth.append(("chr1", 400, chr1[399:402], chr1[399], 3))
    for i in range(3):
        pos0 = 350 + i
        dstart = 400 - pos0
        window = ref["chr1"][pos0 : pos0 + READLEN + 2]
        reads.append(("chr1", pos0, window[:dstart] + window[dstart + 2 : dstart + 2 + (READLEN - dstart)],
                      f"{dstart}M2D{READLEN - dstart}M"))

    # 3bp insertion between chr1:451 and 452
    truth.append(("chr1", 451, chr1[450], chr1[450] + "GGG", 4))
    for i in range(4):
        pos0 = 400 + i
        ioff = 451 - pos0
        window = ref["chr1"][pos0 : pos0 + READLEN - 3]
        reads.append(("chr1", pos0, window[:ioff] + "GGG" + window[ioff:], f"{ioff}M3I{READLEN - 3 - ioff}M"))

    # 1bp deletion of the LAST base of the chr1:301-308 homopolymer. Every A is equivalent,
    # so left-alignment must move this to an anchor at 300 while the as-aligned form sits at 307.
    homopolymer_truth = ("chr1", 300, chr1[299] + "A", chr1[299], 3)
    truth.append(homopolymer_truth)
    for i in range(3):
        pos0 = 250 + i
        dstart = 307 - pos0
        window = ref["chr1"][pos0 : pos0 + READLEN + 1]
        reads.append(("chr1", pos0, window[:dstart] + window[dstart + 1 : dstart + 1 + (READLEN - dstart)],
                      f"{dstart}M1D{READLEN - dstart}M"))

    # a spliced read whose SNV sits after the junction. MD says nothing about the intron, so
    # walking MD alone would shift this variant by the intron length.
    snv_pos = 720
    truth.append(("chr1", snv_pos, chr1[snv_pos - 1], flip[chr1[snv_pos - 1]], 3))
    for i in range(3):
        pos0 = 550 + i
        left, gap, right = 50, 100, 50
        exon2_start = pos0 + left + gap
        exon2 = list(ref["chr1"][exon2_start : exon2_start + right])
        exon2[(snv_pos - 1) - exon2_start] = flip[chr1[snv_pos - 1]]
        reads.append(("chr1", pos0, ref["chr1"][pos0 : pos0 + left] + "".join(exon2),
                      f"{left}M{gap}N{right}M"))

    # a second contig, to exercise the contig-change flush
    ref2_base = chr2[49]
    truth.append(("chr2", 50, ref2_base, flip[ref2_base], 4))
    for i in range(6):
        reads.append(("chr2", i * 30, perfect("chr2", i * 30), f"{READLEN}M"))
    for i in range(4):
        pos0 = 10 + i
        seq = list(perfect("chr2", pos0))
        seq[49 - pos0] = flip[ref2_base]
        reads.append(("chr2", pos0, "".join(seq), f"{READLEN}M"))

    header = {"HD": {"VN": "1.6", "SO": "coordinate"},
              "SQ": [{"SN": name, "LN": len(seq)} for name, seq in ref.items()]}
    unsorted_bam = out / "unsorted.bam"
    tids = {name: i for i, name in enumerate(ref)}
    with pysam.AlignmentFile(str(unsorted_bam), "wb", header=header) as bam:
        for i, (contig, pos0, seq, cigar) in enumerate(reads):
            rec = pysam.AlignedSegment()
            rec.query_name = f"r{i}"
            rec.query_sequence = seq
            rec.flag = 0
            rec.reference_id = tids[contig]
            rec.reference_start = pos0
            rec.mapping_quality = 60
            rec.cigarstring = cigar
            rec.query_qualities = pysam.qualitystring_to_array("I" * len(seq))
            bam.write(rec)

    no_md_bam = out / "sorted_no_md.bam"
    pysam.sort("-o", str(no_md_bam), str(unsorted_bam))
    md_bam = out / "sorted.bam"
    with open(md_bam, "wb") as handle:  # calmd fills in MD and NM
        handle.write(pysam.calmd("-b", str(no_md_bam), str(fasta), catch_stdout=True))

    return {
        "bam": str(md_bam),
        "bam_no_md": str(no_md_bam),
        "fasta": str(fasta),
        "ref": ref,
        "truth": truth,
        "homopolymer_truth": homopolymer_truth,
        "n_reads": len(reads),
        "dir": out,
    }


def read_records(path):
    """(chrom, pos, ref, alt, info dict) for every record, in file order."""
    records = []
    with pysam.VariantFile(path) as vcf:
        for rec in vcf:
            records.append((rec.chrom, rec.pos, rec.ref, rec.alts[0], dict(rec.info)))
    return records


# ---------------------------------------------------------------------------
# core behaviour
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("engine", ENGINES)
def test_recovers_planted_variants(synthetic_bam, tmp_path, engine):
    out = tmp_path / f"{engine}.vcf"
    stats = bam_to_vcf(synthetic_bam["bam"], output=str(out), reference=synthetic_bam["fasta"],
                       engine=engine, overwrite=True)
    assert stats["engine"] == engine
    assert stats["reads_used"] == synthetic_bam["n_reads"]

    got = {(chrom, pos, ref, alt): info for chrom, pos, ref, alt, info in read_records(str(out))}
    assert len(got) == len(synthetic_bam["truth"])
    for chrom, pos, ref, alt, expected_ao in synthetic_bam["truth"]:
        key = (chrom, pos, ref, alt)
        assert key in got, f"missing {key}; got {sorted(got)}"
        assert got[key]["AO"][0] == expected_ao


@requires_native
def test_engines_agree(synthetic_bam, tmp_path):
    """The fallback is only a safety net if it produces the same answer."""
    for reference in (None, synthetic_bam["fasta"]):
        outputs = []
        for engine in ("native", "python"):
            out = tmp_path / f"agree_{engine}_{'fa' if reference else 'md'}.vcf"
            bam_to_vcf(synthetic_bam["bam"], output=str(out), reference=reference,
                       engine=engine, overwrite=True)
            with open(out) as handle:  # the ##source line names the engine, so skip the header
                outputs.append([line for line in handle if not line.startswith("##")])
        assert outputs[0] == outputs[1]


@pytest.mark.parametrize("engine", ENGINES)
def test_ref_alleles_match_the_reference(synthetic_bam, tmp_path, engine):
    """Every REF allele must be exactly what the FASTA says, or downstream tools reject it."""
    out = tmp_path / f"refcheck_{engine}.vcf"
    bam_to_vcf(synthetic_bam["bam"], output=str(out), reference=synthetic_bam["fasta"],
               engine=engine, overwrite=True)
    ref = synthetic_bam["ref"]
    for chrom, pos, ref_allele, _alt, _info in read_records(str(out)):
        assert ref[chrom][pos - 1 : pos - 1 + len(ref_allele)] == ref_allele


def _independent_coverage(bam_path):
    """Per-position coverage counted directly off the CIGARs: {(chrom, pos1): depth}.

    Deliberately naive and independent of the ring buffer under test. Positions under a
    spliced gap (N) are not covered; positions under a deletion (D) are, since the read
    does span them.
    """
    coverage = {}
    with pysam.AlignmentFile(bam_path, "rb") as bam:
        for read in bam:
            if read.is_unmapped:
                continue
            chrom = bam.get_reference_name(read.reference_id)
            rpos = read.reference_start + 1
            for op, length in read.cigartuples:
                if op in (0, 2, 7, 8):  # M / D / = / X all span the reference
                    for pos in range(rpos, rpos + length):
                        coverage[(chrom, pos)] = coverage.get((chrom, pos), 0) + 1
                    rpos += length
                elif op == 3:  # N
                    rpos += length
    return coverage


@pytest.mark.parametrize("engine", ENGINES)
def test_depth_matches_independent_count(synthetic_bam, tmp_path, engine):
    """DP must agree with an independent count of reads spanning the position."""
    out = tmp_path / f"depth_{engine}.vcf"
    bam_to_vcf(synthetic_bam["bam"], output=str(out), reference=synthetic_bam["fasta"],
               engine=engine, overwrite=True)
    coverage = _independent_coverage(synthetic_bam["bam"])
    records = read_records(str(out))
    assert records
    for chrom, pos, _ref, _alt, info in records:
        assert info["DP"] == coverage.get((chrom, pos), 0), f"{chrom}:{pos} DP mismatch"


@pytest.mark.parametrize("engine", ENGINES)
def test_vaf_is_ao_over_dp(synthetic_bam, tmp_path, engine):
    out = tmp_path / f"vaf_{engine}.vcf"
    bam_to_vcf(synthetic_bam["bam"], output=str(out), reference=synthetic_bam["fasta"],
               engine=engine, overwrite=True)
    for _chrom, _pos, _ref, _alt, info in read_records(str(out)):
        assert info["AO"][0] / info["DP"] == pytest.approx(info["VAF"][0], rel=1e-5)


# ---------------------------------------------------------------------------
# normalization
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("engine", ENGINES)
def test_left_alignment_requires_reference(synthetic_bam, tmp_path, engine):
    """With a FASTA the homopolymer deletion is left-aligned; without one it stays as aligned."""
    chrom, pos, ref_allele, alt_allele, _ao = synthetic_bam["homopolymer_truth"]

    with_ref = tmp_path / f"norm_{engine}.vcf"
    bam_to_vcf(synthetic_bam["bam"], output=str(with_ref), reference=synthetic_bam["fasta"],
               engine=engine, overwrite=True)
    assert (chrom, pos, ref_allele, alt_allele) in {r[:4] for r in read_records(str(with_ref))}

    without_ref = tmp_path / f"nonorm_{engine}.vcf"
    bam_to_vcf(synthetic_bam["bam"], output=str(without_ref), engine=engine, overwrite=True)
    deletions = [r for r in read_records(str(without_ref)) if len(r[2]) > len(r[3])]
    assert (chrom, pos) not in {(r[0], r[1]) for r in deletions}
    assert 307 in {r[1] for r in deletions}  # reported where the aligner placed it


@pytest.mark.parametrize("engine", ENGINES)
def test_no_normalize_flag(synthetic_bam, tmp_path, engine):
    out = tmp_path / f"nonormflag_{engine}.vcf"
    bam_to_vcf(synthetic_bam["bam"], output=str(out), reference=synthetic_bam["fasta"],
               normalize=False, engine=engine, overwrite=True)
    deletion_positions = {r[1] for r in read_records(str(out)) if len(r[2]) > len(r[3])}
    assert 307 in deletion_positions and 300 not in deletion_positions


# ---------------------------------------------------------------------------
# filters and options
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("engine", ENGINES)
def test_min_count(synthetic_bam, tmp_path, engine):
    counts = {}
    for min_count in (1, 3, 4, 6):
        out = tmp_path / f"minc_{engine}_{min_count}.vcf"
        stats = bam_to_vcf(synthetic_bam["bam"], output=str(out), reference=synthetic_bam["fasta"],
                           min_count=min_count, engine=engine, overwrite=True)
        counts[min_count] = stats["records_emitted"]
        for _chrom, _pos, _ref, _alt, info in read_records(str(out)):
            assert info["AO"][0] >= min_count
    assert counts[1] > counts[3] >= counts[4] > counts[6]
    assert counts[6] == 0


@pytest.mark.parametrize("engine", ENGINES)
def test_min_vaf(synthetic_bam, tmp_path, engine):
    """min_vaf keeps only alleles at or above the fraction, independently of AO."""
    unfiltered = tmp_path / f"vaf_all_{engine}.vcf"
    bam_to_vcf(synthetic_bam["bam"], output=str(unfiltered), reference=synthetic_bam["fasta"],
               engine=engine, overwrite=True)
    observed = {(r[0], r[1]): r[4]["VAF"][0] for r in read_records(str(unfiltered))}
    assert min(observed.values()) < 0.3 < max(observed.values()), "fixture needs a spread of VAFs to test against"

    for threshold in (0.25, 0.5, 0.62):
        out = tmp_path / f"vaf_{engine}_{threshold}.vcf"
        stats = bam_to_vcf(synthetic_bam["bam"], output=str(out), reference=synthetic_bam["fasta"],
                           min_vaf=threshold, engine=engine, overwrite=True)
        kept = {(r[0], r[1]) for r in read_records(str(out))}
        expected = {key for key, vaf in observed.items() if vaf >= threshold - 1e-6}
        assert kept == expected
        assert stats["dropped_vaf"] == len(observed) - len(expected)


@pytest.mark.parametrize("engine", ENGINES)
def test_max_vaf(synthetic_bam, tmp_path, engine):
    """max_vaf drops the high-fraction (germline-looking) end."""
    unfiltered = tmp_path / f"maxvaf_all_{engine}.vcf"
    bam_to_vcf(synthetic_bam["bam"], output=str(unfiltered), reference=synthetic_bam["fasta"],
               engine=engine, overwrite=True)
    observed = {(r[0], r[1]): r[4]["VAF"][0] for r in read_records(str(unfiltered))}

    out = tmp_path / f"maxvaf_{engine}.vcf"
    bam_to_vcf(synthetic_bam["bam"], output=str(out), reference=synthetic_bam["fasta"],
               max_vaf=0.6, engine=engine, overwrite=True)
    kept = {(r[0], r[1]) for r in read_records(str(out))}
    assert kept == {key for key, vaf in observed.items() if vaf <= 0.6 + 1e-6}
    assert kept and len(kept) < len(observed)


@pytest.mark.parametrize("engine", ENGINES)
def test_vaf_window_combines_with_min_count(synthetic_bam, tmp_path, engine):
    out = tmp_path / f"vafwindow_{engine}.vcf"
    bam_to_vcf(synthetic_bam["bam"], output=str(out), reference=synthetic_bam["fasta"],
               min_count=3, min_vaf=0.3, max_vaf=0.9, engine=engine, overwrite=True)
    records = read_records(str(out))
    assert records
    for _chrom, _pos, _ref, _alt, info in records:
        assert info["AO"][0] >= 3
        assert 0.3 - 1e-6 <= info["VAF"][0] <= 0.9 + 1e-6


@pytest.mark.parametrize("engine", ENGINES)
def test_vaf_bounds_are_validated(synthetic_bam, tmp_path, engine):
    for min_vaf, max_vaf, pattern in ((-0.1, 1.0, "between 0 and 1"),
                                      (0.0, 1.5, "between 0 and 1"),
                                      (0.8, 0.2, "exceeds")):
        with pytest.raises(ValueError, match=pattern):
            bam_to_vcf(synthetic_bam["bam"], output=str(tmp_path / "bad.vcf"),
                       reference=synthetic_bam["fasta"], min_vaf=min_vaf, max_vaf=max_vaf,
                       engine=engine, overwrite=True)


@pytest.mark.parametrize("engine", ENGINES)
def test_skip_indels(synthetic_bam, tmp_path, engine):
    out = tmp_path / f"noindel_{engine}.vcf"
    bam_to_vcf(synthetic_bam["bam"], output=str(out), reference=synthetic_bam["fasta"],
               skip_indels=True, engine=engine, overwrite=True)
    for _chrom, _pos, ref_allele, alt_allele, _info in read_records(str(out)):
        assert len(ref_allele) == 1 and len(alt_allele) == 1


@pytest.mark.parametrize("engine", ENGINES)
def test_regions_bed(synthetic_bam, tmp_path, engine):
    bed = tmp_path / "regions.bed"
    bed.write_text("chr1\t400\t460\nchr2\t0\t100\n")
    out = tmp_path / f"regions_{engine}.vcf"
    bam_to_vcf(synthetic_bam["bam"], output=str(out), reference=synthetic_bam["fasta"],
               regions=str(bed), engine=engine, overwrite=True)
    records = read_records(str(out))
    assert {(r[0], r[1]) for r in records} == {("chr1", 400), ("chr1", 451), ("chr2", 50)}


@pytest.mark.parametrize("engine", ENGINES)
def test_min_mapq_excludes_reads(synthetic_bam, tmp_path, engine):
    out = tmp_path / f"mapq_{engine}.vcf"
    stats = bam_to_vcf(synthetic_bam["bam"], output=str(out), reference=synthetic_bam["fasta"],
                       min_mapq=61, engine=engine, overwrite=True)
    assert stats["reads_used"] == 0 and stats["records_emitted"] == 0


@pytest.mark.parametrize("engine", ENGINES)
def test_bgzip_output_is_indexable(synthetic_bam, tmp_path, engine):
    out = tmp_path / f"indexed_{engine}.vcf.gz"
    bam_to_vcf(synthetic_bam["bam"], output=str(out), reference=synthetic_bam["fasta"],
               index=True, engine=engine, overwrite=True)
    assert os.path.exists(str(out) + ".tbi")
    with pysam.VariantFile(str(out)) as vcf:  # random access proves the file is sorted and indexed
        fetched = [(rec.pos, rec.ref, rec.alts[0]) for rec in vcf.fetch("chr1", 440, 460)]
    assert fetched == [(451, "A", "AGGG")]


@pytest.mark.parametrize("engine", ENGINES)
def test_output_is_coordinate_sorted(synthetic_bam, tmp_path, engine):
    out = tmp_path / f"sorted_{engine}.vcf"
    bam_to_vcf(synthetic_bam["bam"], output=str(out), reference=synthetic_bam["fasta"],
               engine=engine, overwrite=True)
    seen = {}
    order = []
    for chrom, pos, _ref, _alt, _info in read_records(str(out)):
        if chrom not in seen:
            seen[chrom] = -1
            order.append(chrom)
        assert pos >= seen[chrom], f"{chrom}:{pos} out of order"
        seen[chrom] = pos
    assert order == ["chr1", "chr2"]  # BAM header order


# ---------------------------------------------------------------------------
# error handling
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("engine", ENGINES)
def test_missing_md_without_reference_raises(synthetic_bam, tmp_path, engine):
    """Without MD or a FASTA there is nothing to read mismatches out of; fail loudly."""
    out = tmp_path / f"nomd_{engine}.vcf"
    with pytest.raises((ValueError, RuntimeError), match="(?i)MD tag"):
        bam_to_vcf(synthetic_bam["bam_no_md"], output=str(out), engine=engine, overwrite=True)


@pytest.mark.parametrize("engine", ENGINES)
def test_missing_md_with_reference_is_fine(synthetic_bam, tmp_path, engine):
    """A FASTA removes the MD requirement entirely."""
    out = tmp_path / f"nomd_ok_{engine}.vcf"
    bam_to_vcf(synthetic_bam["bam_no_md"], output=str(out), reference=synthetic_bam["fasta"],
               engine=engine, overwrite=True)
    got = {r[:4] for r in read_records(str(out))}
    for chrom, pos, ref_allele, alt_allele, _ao in synthetic_bam["truth"]:
        assert (chrom, pos, ref_allele, alt_allele) in got


@pytest.mark.parametrize("engine", ENGINES)
def test_unsorted_bam_is_refused(synthetic_bam, tmp_path, engine):
    unsorted = tmp_path / "unsorted_header.bam"
    header = pysam.AlignmentFile(synthetic_bam["bam"], "rb").header.to_dict()
    header["HD"]["SO"] = "unsorted"
    with pysam.AlignmentFile(synthetic_bam["bam"], "rb") as src, \
            pysam.AlignmentFile(str(unsorted), "wb", header=header) as dst:
        for read in src:
            dst.write(read)
    out = tmp_path / f"unsorted_{engine}.vcf"
    with pytest.raises((ValueError, RuntimeError), match="(?i)coordinate"):
        bam_to_vcf(str(unsorted), output=str(out), reference=synthetic_bam["fasta"],
                   engine=engine, overwrite=True)
    # assume_sorted bypasses the check; this input really is sorted, so results are unchanged
    bam_to_vcf(str(unsorted), output=str(out), reference=synthetic_bam["fasta"],
               engine=engine, assume_sorted=True, overwrite=True)
    assert len(read_records(str(out))) == len(synthetic_bam["truth"])


def test_existing_output_requires_overwrite(synthetic_bam, tmp_path):
    out = tmp_path / "exists.vcf"
    out.write_text("")
    with pytest.raises(ValueError, match="already exists"):
        bam_to_vcf(synthetic_bam["bam"], output=str(out), reference=synthetic_bam["fasta"])


def test_bad_engine_rejected(synthetic_bam, tmp_path):
    with pytest.raises(ValueError, match="engine must be"):
        bam_to_vcf(synthetic_bam["bam"], output=str(tmp_path / "x.vcf"), engine="rust")


def test_missing_bam_rejected(tmp_path):
    with pytest.raises(ValueError, match="does not exist"):
        bam_to_vcf(str(tmp_path / "nope.bam"), output=str(tmp_path / "x.vcf"))


# ---------------------------------------------------------------------------
# mnv_max_gap: merging nearby changes within a read
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def mnv_bam(tmp_path_factory):
    """Reads that each carry a known cluster of changes.

    chr1 (1000 bp):
    - 3 reads with a 2-base substitution at 101-102
    - 2 reads with an SNV at 301 directly followed by a 2-base deletion at 302-303
    - 2 reads with SNVs at 501 and 505 (three reference bases between them)
    - 2 spliced reads (50M200N50M) whose SNVs sit on either side of the junction at 750 and 951
    """
    out = tmp_path_factory.mktemp("bam_to_vcf_mnv")
    random.seed(11)
    chrom = "".join(random.choice("ACGT") for _ in range(1000))
    fasta = out / "ref.fa"
    with open(fasta, "w") as handle:
        handle.write(">chr1\n")
        for i in range(0, len(chrom), 60):
            handle.write(chrom[i : i + 60] + "\n")
    pysam.faidx(str(fasta))
    flip = {"A": "C", "C": "A", "G": "T", "T": "G"}

    def mutate(pos0, positions1):
        seq = list(chrom[pos0 : pos0 + READLEN])
        for p1 in positions1:
            seq[p1 - 1 - pos0] = flip[seq[p1 - 1 - pos0]]
        return "".join(seq)

    reads = []
    for i in range(3):
        reads.append((60 + i, mutate(60 + i, [101, 102]), f"{READLEN}M"))
    for i in range(2):
        pos0 = 260 + i
        seq = list(chrom[pos0 : pos0 + READLEN + 2])
        seq[300 - pos0] = flip[seq[300 - pos0]]
        del seq[301 - pos0 : 303 - pos0]
        left = 301 - pos0
        reads.append((pos0, "".join(seq)[:READLEN], f"{left}M2D{READLEN - left}M"))
    for i in range(2):
        reads.append((460 + i, mutate(460 + i, [501, 505]), f"{READLEN}M"))
    for i in range(2):
        seq = list(chrom[700:750] + chrom[950:1000])
        seq[49] = flip[seq[49]]  # chr1:750
        seq[50] = flip[seq[50]]  # chr1:951
        reads.append((700, "".join(seq), "50M200N50M"))

    header = {"HD": {"VN": "1.6", "SO": "coordinate"}, "SQ": [{"SN": "chr1", "LN": len(chrom)}]}
    unsorted_bam = out / "unsorted.bam"
    with pysam.AlignmentFile(str(unsorted_bam), "wb", header=header) as bam:
        for i, (pos0, seq, cigar) in enumerate(reads):
            rec = pysam.AlignedSegment()
            rec.query_name = f"m{i}"
            rec.query_sequence = seq
            rec.flag = 0
            rec.reference_id = 0
            rec.reference_start = pos0
            rec.mapping_quality = 60
            rec.cigarstring = cigar
            rec.query_qualities = pysam.qualitystring_to_array("I" * len(seq))
            bam.write(rec)
    sorted_bam = out / "sorted.bam"
    pysam.sort("-o", str(sorted_bam), str(unsorted_bam))
    return {"bam": str(sorted_bam), "fasta": str(fasta), "ref": chrom, "flip": flip}


def _alleles(path):
    return {(pos, ref, alt): (info["AO"][0], info["TYPE"][0]) for _, pos, ref, alt, info in read_records(path)}


@pytest.mark.parametrize("engine", ENGINES)
def test_mnv_merging_off_by_default(mnv_bam, tmp_path, engine):
    out = tmp_path / f"{engine}.vcf"
    bam_to_vcf(mnv_bam["bam"], output=str(out), reference=mnv_bam["fasta"], engine=engine, overwrite=True)
    types = {t for _, t in _alleles(str(out)).values()}
    assert types <= {"snv", "ins", "del"}
    assert sum(1 for _, t in _alleles(str(out)).values() if t == "snv") == 7


@pytest.mark.parametrize("engine", ENGINES)
def test_mnv_merging_adjacent(mnv_bam, tmp_path, engine):
    ref, flip = mnv_bam["ref"], mnv_bam["flip"]
    out = tmp_path / f"{engine}.vcf"
    bam_to_vcf(mnv_bam["bam"], output=str(out), reference=mnv_bam["fasta"], engine=engine, overwrite=True, mnv_max_gap=0)
    got = _alleles(str(out))
    mnv_ref = ref[100:102]
    assert got[(101, mnv_ref, flip[mnv_ref[0]] + flip[mnv_ref[1]])] == (3, "mnv")
    # SNV at 301 then deletion of 302-303: one anchored complex allele
    assert got[(300, ref[299:303], ref[299] + flip[ref[300]])] == (2, "complex")
    # three reference bases apart: not merged at gap 0
    assert got[(501, ref[500], flip[ref[500]])] == (2, "snv")
    assert got[(505, ref[504], flip[ref[504]])] == (2, "snv")
    # a spliced gap is never merged across
    assert got[(750, ref[749], flip[ref[749]])] == (2, "snv")
    assert got[(951, ref[950], flip[ref[950]])] == (2, "snv")
    assert len(got) == 6


@pytest.mark.parametrize("engine", ENGINES)
def test_mnv_merging_with_gap(mnv_bam, tmp_path, engine):
    ref, flip = mnv_bam["ref"], mnv_bam["flip"]
    out = tmp_path / f"{engine}.vcf"
    bam_to_vcf(mnv_bam["bam"], output=str(out), reference=mnv_bam["fasta"], engine=engine, overwrite=True, mnv_max_gap=3)
    got = _alleles(str(out))
    alt = flip[ref[500]] + ref[501:504] + flip[ref[504]]
    assert got[(501, ref[500:505], alt)] == (2, "mnv")
    assert (750, ref[749], flip[ref[749]]) in got


def test_mnv_max_gap_rejects_negative(mnv_bam, tmp_path):
    with pytest.raises(ValueError, match="mnv_max_gap"):
        bam_to_vcf(mnv_bam["bam"], output=str(tmp_path / "x.vcf"), reference=mnv_bam["fasta"], mnv_max_gap=-1)


@requires_native
def test_engines_agree_with_mnv_merging(mnv_bam, tmp_path):
    for gap in (0, 3):
        outputs = []
        for engine in ("native", "python"):
            out = tmp_path / f"agree_{engine}_{gap}.vcf"
            bam_to_vcf(mnv_bam["bam"], output=str(out), reference=mnv_bam["fasta"], engine=engine, overwrite=True, mnv_max_gap=gap)
            with open(out) as handle:
                outputs.append([line for line in handle if not line.startswith("##")])
        assert outputs[0] == outputs[1]
