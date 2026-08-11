import shutil
import argparse
import subprocess
import sys
import os
import re
import logging
import shlex
import tempfile

from collections import Counter
import gzip
import pysam
from tqdm import tqdm
import pandas as pd
import matplotlib.pyplot as plt
import re
from typing import Literal, Optional, Union

from pydantic import BaseModel, ConfigDict, model_validator

logger = logging.getLogger(__name__)

from .utils import (
    make_function_parameter_to_value_dict,
    validate_call,
    vk_config,
    Parity,
    PositiveInt,
    NonNegativeInt,
)

from .utils.varseek_denovo_utils import (
    infer_hgvs_prefix,
    infer_reference_type_from_fasta,
    vcf_allele_to_hgvs,
    vcf_to_tsv,
    configure_logger,
    run,
    check_tool,
    parse_cigars,
    bam_to_vcf,
)


# bam2vcf reads variants straight off the alignment records and applies no statistical
# model, so these gates are all that stand between the caller and every sequencing error
# and mismapped read in the BAM. Left at zero it is a firehose: on 30x NA12878 chr20 it
# emits ~520k candidates versus ~127k at min_mapq=20, with SNP recall still above 97%.
# The quality gates are the values benchmarked in the NA12878 notebook (alongside k=41,
# w=40, min_counts=3) and cost essentially no recall, so they are safe to default on.
#
# min_vaf is deliberately NOT among them. varseek is a sensitive candidate generator, and a
# fraction floor is the one filter that removes *real* variants: somatic/subclonal alleles
# and allele-specific expression legitimately sit far below the germline 0.5. It stays at 0
# and denovo warns instead, so the choice is the caller's rather than a silent default.
BAM2VCF_DEFAULT_MIN_MAPQ = 20
BAM2VCF_DEFAULT_MIN_BASEQ = 20
BAM2VCF_DEFAULT_MIN_VAF = 0.0

# min_vaf below this is treated as "no meaningful fraction floor" and triggers the advisory
# warning. 0.05 is well under any germline expectation, so it does not fire on somatic work
# that has deliberately set a low-but-real floor.
BAM2VCF_LOW_MIN_VAF = 0.05

# What to suggest when the fraction floor is off. Germline DNA variants cluster at ~0.5/1.0,
# so a floor there is nearly free; RNA fractions are skewed by allele-specific expression
# and uneven coverage, so the advice is to lean on min_counts instead of a fraction.
BAM2VCF_SUGGESTED_MIN_VAF_DNA = 0.2


class DenovoParams(BaseModel):
    """Cross-field / filesystem validation for :func:`denovo`.

    Per-parameter type checks live on the ``denovo`` signature (validated by
    ``@validate_call``). This model captures the checks a single-value annotation
    cannot express: file-extension + existence of ``sequences``/``gtf``/``regions``,
    the ``output``/``output_tsv`` extension and overwrite-collision rules, and the
    ``tsv_reference_type`` prefix validity. Checks that *mutate* or *derive* state the
    body depends on (``min_counts`` reset, ``inputs`` normalization + input-type
    detection, and the aligner-vs-biology check) intentionally remain in ``denovo``.
    Instantiate it with the full params dict (extra keys are ignored)::

        DenovoParams(**make_function_parameter_to_value_dict(1))
    """

    model_config = ConfigDict(extra="ignore", arbitrary_types_allowed=True)

    sequences: object = None
    gtf: object = None
    regions: object = None
    output: object = "out.vcf.gz"
    output_tsv: object = None
    tsv_reference_type: object = "auto"
    overwrite: object = False
    variant_caller: object = "bam2vcf"
    min_vaf: object = None
    max_vaf: object = 1.0
    min_mapq: object = None
    min_baseq: object = None

    @model_validator(mode="after")
    def _validate(self):
        # min_vaf/min_mapq/min_baseq are None when the caller left them alone; denovo then
        # resolves them to the bam2vcf defaults, or to no-ops for the other callers. Only a
        # value the caller actually passed can conflict with a non-bam2vcf caller, so every
        # cross-caller check below tests the raw (pre-resolution) value.
        min_vaf = 0.0 if self.min_vaf is None else float(self.min_vaf)
        min_mapq = 0 if self.min_mapq is None else int(self.min_mapq)
        min_baseq = 0 if self.min_baseq is None else int(self.min_baseq)

        # VAF bounds are computed from INFO/AO over INFO/DP, which only bam2vcf emits.
        # Ignoring them for the other callers would misreport what was actually filtered.
        if not 0.0 <= min_vaf <= 1.0 or not 0.0 <= self.max_vaf <= 1.0:
            raise ValueError(f"--min-vaf and --max-vaf must lie between 0 and 1, got min_vaf={self.min_vaf}, max_vaf={self.max_vaf}")
        if min_vaf > self.max_vaf:
            raise ValueError(f"--min-vaf ({min_vaf}) exceeds --max-vaf ({self.max_vaf}), so no variant could pass")
        if ((self.min_vaf is not None and min_vaf > 0.0) or self.max_vaf < 1.0) and self.variant_caller != "bam2vcf":
            raise ValueError(
                f"--min-vaf/--max-vaf require variant_caller='bam2vcf' (got '{self.variant_caller}'); "
                "the bcftools caller can express a similar filter through --include, and the cigar caller reports no depth."
            )

        # bam2vcf reads alignment records directly, so read/base quality is the only handle it
        # has on misalignment and sequencing error -- bcftools instead expresses these through
        # mpileup's own -q/-Q. Silently dropping them would misreport what was filtered.
        if min_mapq < 0 or min_baseq < 0:
            raise ValueError(f"--min-mapq and --min-baseq must be non-negative, got min_mapq={self.min_mapq}, min_baseq={self.min_baseq}")
        if ((self.min_mapq is not None and min_mapq > 0) or (self.min_baseq is not None and min_baseq > 0)) and self.variant_caller != "bam2vcf":
            raise ValueError(
                f"--min-mapq/--min-baseq require variant_caller='bam2vcf' (got '{self.variant_caller}'); "
                "the bcftools caller sets mpileup's -q/-Q itself, and the cigar caller ignores qualities."
            )

        # The callers emit different formats, so the required extension depends on the caller:
        # only 'cigar' writes a CSV of HGVS strings; bam2vcf and bcftools both write VCF.
        if self.variant_caller == "cigar":
            if not self.output.endswith(".csv"):
                raise ValueError("when using the 'cigar' variant caller, --output must end with .csv")
        elif not self.output.endswith(".vcf") and not self.output.endswith(".vcf.gz"):
            raise ValueError("--output must end with .vcf or .vcf.gz")

        if self.sequences:
            valid_fasta_extensions = [".fa", ".fasta", ".fa.gz", ".fasta.gz", ".fna", ".fna.gz"]
            if not any(self.sequences.endswith(ext) for ext in valid_fasta_extensions):
                raise ValueError(f"-s/--sequences must be a FASTA file ending with {', '.join(valid_fasta_extensions)}")
            if not os.path.isfile(self.sequences):
                fasta_dir = os.path.dirname(self.sequences) or "."
                recommended_command = f"gget ref -r 111 -d -od {fasta_dir} -w dna human && gunzip {self.sequences}.gz"
                raise ValueError(f"FASTA reference '{self.sequences}' not found. Recommended command to download: {recommended_command}")

        if self.gtf:
            if not self.gtf.endswith(".gtf"):
                raise ValueError("--gtf must be a GTF file ending with .gtf")
            if not os.path.isfile(self.gtf):
                gtf_dir = os.path.dirname(self.gtf) or "."
                recommended_command = f"gget ref -r 111 -d -od {gtf_dir} -w gtf human && gunzip {self.gtf}.gz"
                raise ValueError(f"GTF file '{self.gtf}' not found. Recommended command to download: {recommended_command}")

        if self.regions:
            if not self.regions.endswith(".bed"):
                raise ValueError("--regions must be a BED file ending with .bed")
            if not os.path.isfile(self.regions):
                raise ValueError(f"regions BED file '{self.regions}' not found.")

        if os.path.exists(self.output) and not self.overwrite:
            raise ValueError(f"output file '{self.output}' already exists. Use --overwrite to overwrite.")

        if self.output_tsv:
            if not self.output_tsv.endswith(".tsv"):
                raise ValueError("--output-tsv must end with .tsv")
            if os.path.exists(self.output_tsv) and not self.overwrite:
                raise ValueError(f"TSV output file '{self.output_tsv}' already exists. Use --overwrite to overwrite.")
            infer_hgvs_prefix("chr1", reference_type=self.tsv_reference_type)

        return self


@validate_call(config=vk_config)
def denovo(
    inputs: Union[str, list[str]],
    sequences: str,
    parity: Parity = "single",
    gtf: Optional[str] = None,
    star_genome_index_dir: str = "genome_index",
    bowtie2_genome_index_prefix: str = "bowtie2_index",
    star_alignment_prefix: str = "star_",
    bowtie2_alignment_dir: str = "bowtie2_alignments",
    regions: Optional[str] = None,
    out_bam_dir: Optional[str] = None,
    output: str = "out.vcf.gz",
    output_tsv: Optional[str] = None,
    tsv_reference_type: str = "auto",
    read_length: Optional[PositiveInt] = None,
    min_counts: int = 3,
    min_vaf: Optional[float] = None,
    max_vaf: float = 1.0,
    min_mapq: Optional[NonNegativeInt] = None,
    min_baseq: Optional[NonNegativeInt] = None,
    aligner: Literal["STAR", "bowtie2"] = "bowtie2",
    technology: Optional[str] = None,  # kb-style technology string (as in vk ref/count); "DNA" => genomic DNA reads, anything else => RNA reads. Used to derive the read type for aligner validation.
    reference_type: Optional[str] = "auto",  # not a strict Literal: the body lowercases before validating
    variant_caller: Literal["bam2vcf", "bcftools", "cigar"] = "bam2vcf",
    bam2vcf_engine: Literal["auto", "native", "python"] = "auto",
    bowtie2_seed_length: Optional[PositiveInt] = None,
    bowtie2_score_min: Optional[str] = None,
    include: Optional[str] = None,
    skip_indels: bool = False,
    disable_baq: bool = True,
    max_depth: PositiveInt = 1000,
    split_bam_by_n: bool = False,
    disable_bcftools_norm: bool = False,
    rm_dup: Literal["exact", "snps", "indels", "both", "all", "none"] = "exact",
    bcftools_call_prior: Optional[str] = None,
    merge_bam_files: bool = False,
    strip_version_numbers: bool = False,
    tmp_dir: Optional[str] = None,
    threads: PositiveInt = 1,
    overwrite: bool = False,
    verbose: NonNegativeInt = 0,
    quiet: bool = False,
):
    """
    Call de novo variants from FASTQ or BAM inputs.

    # Required input arguments:
    - inputs                         (str or list[str]) Input BAMs or FASTQs. Pass one or more files as separate values. For paired FASTQs, pass alternating R1/R2 files.
    - sequences                      (str) Path to a reference genome FASTA file — the same file passed as `sequences` to vk build. Must end with one of: .fa, .fasta, .fna (optionally .gz).

    # Optional input arguments:
    - parity                         (str) Whether FASTQ inputs are single-end or paired-end. Use "single" to treat each FASTQ separately. Use "paired" to pair alternating inputs as R1/R2 and require an even number of FASTQs. Default: "single"
    - gtf                            (str) Genome annotation GTF file. Required for STAR genome generation.
    - star_genome_index_dir          (str) STAR genome index directory. Default: "genome_index"
    - bowtie2_genome_index_prefix    (str) Prefix for Bowtie2 genome index files. Default: "bowtie2_index"
    - star_alignment_prefix          (str) Prefix for STAR output BAM. Default: "star_"
    - bowtie2_alignment_dir          (str) Directory for Bowtie2 output BAMs. Default: "bowtie2_alignments"
    - regions                        (str) BED file of regions to restrict variant calling to. Default: None
    - out_bam_dir                    (str) Output directory for BAM files for Bowtie2 alignments when merge_bam_files=False. Default: None
    - output                         (str) Output VCF file for the bam2vcf and bcftools variant callers, or CSV file for cigar. Default: "out.vcf.gz"
    - output_tsv                     (str) Optional TSV output converted from the bcftools VCF with columns seq_id and variant. Default: None
    - tsv_reference_type             (str) Variant coordinate prefix for output_tsv. One of auto, dna, genome, cdna, transcriptome. Default: "auto"
    - read_length                    (int) Read length. Default: None
    - min_counts                     (int) Minimum count threshold for filtering, i.e. the minimum number of reads supporting a variant (INFO/AO for bam2vcf, INFO/AD[1] for bcftools). Default: 3
    - min_vaf                      (float) Minimum variant allele fraction (INFO/VAF, i.e. AO/DP), inclusive. Filters out low-fraction candidates that pass min_counts only because the site is deeply covered. Left at 0 so that somatic/subclonal and allele-specific variants, which legitimately sit far below the germline fraction, are not dropped by default; denovo warns when no floor is in effect. On germline DNA, 0.2 is a cheap specificity win. Only supported by the bam2vcf variant caller. Default: 0.0
    - max_vaf                      (float) Maximum variant allele fraction (INFO/VAF, i.e. AO/DP), inclusive. Lower it (e.g. 0.9) to drop likely-germline homozygous sites when hunting somatic variants. Only supported by the bam2vcf variant caller. Default: 1.0
    - min_mapq                       (int) Minimum read mapping quality; reads below it contribute to neither AO nor DP. Because bam2vcf trusts the aligner's placement, unfiltered multi-mapping reads in repeats turn every mismapped base into a candidate — on 30x NA12878 chr20, min_mapq=20 cut the candidate set from 520k to 127k while keeping SNP recall above 97%. Only supported by the bam2vcf variant caller. Set it to 0 to disable the filter. Default: 20 (bam2vcf), unset for the other callers
    - min_baseq                      (int) Minimum base quality at the variant position; low-quality bases contribute to neither AO nor DP. bam2vcf has no statistical model, so this is its only defence against sequencing error. Only supported by the bam2vcf variant caller. Set it to 0 to disable the filter. Default: 20 (bam2vcf), unset for the other callers
    - aligner                      (str) Aligner to use. One of STAR or bowtie2. Splice-aware STAR is required only when RNA reads are aligned to a genome (reads span exon-exon junctions); bowtie2 is correct otherwise (DNA reads to a genome, or RNA reads to a transcriptome). A mismatch against `technology`/`reference_type` raises. Default: "bowtie2"
    - technology                     (str) Sequencing technology, using the same vocabulary as vk ref/count (run `kb --list`; varseek additionally accepts "dna"). It fills in the read type used to validate `aligner`: "dna" (or "DNA") declares genomic DNA reads (-> bowtie2), any other technology declares RNA reads (-> STAR on a genome). Required to validate the aligner when the reference is a genome. Default: None
    - reference_type                 (str) Whether `sequences` is a genome or transcriptome, used to validate `aligner`. One of auto, genome, dna, transcriptome, cdna. "auto" infers it from the FASTA sequence IDs. Default: "auto"
    - variant_caller                 (str) Variant caller to use. One of bam2vcf, bcftools or cigar. "bam2vcf" reads every variant recorded in the alignments themselves (CIGAR indels plus mismatches from the MD tag or the reference) and writes a sites-only VCF with AO/DP/VAF; it is roughly 10x faster than the 'cigar' caller and is a sensitive candidate generator rather than a genotyper, so it applies no statistical model. "bcftools" runs mpileup/call/norm, which genotypes and filters but is far slower. "cigar" is the older Python walk and writes a CSV of HGVS strings instead of a VCF. Default: "bam2vcf"
    - bam2vcf_engine                 (str) Implementation used by the bam2vcf caller. "auto" uses the compiled C++ helper and falls back to Python if it cannot be built, "native" requires the compiled helper, "python" forces the pysam implementation. Default: "auto"
    - bowtie2_seed_length            (int) Seed length for Bowtie2 aligner. Default: None
    - bowtie2_score_min              (str) Bowtie2 score-min setting. Default: None
    - include                        (str) bcftools filter expression. Default: None
    - skip_indels                    (bool) Skip indels. Default: False
    - disable_baq                    (bool) Disable BAQ (Base Alignment Quality) computation in mpileup, passing -B. BAQ downweights base qualities near indels and in misalignment-prone regions, which raises specificity but suppresses marginal calls and costs a large amount of runtime (a per-read HMM realignment). Disabled by default because varseek's typical use is sensitive candidate discovery; set False to recover bcftools' stock specificity. Default: True
    - max_depth                      (int) Maximum reads per input file at any single position in mpileup, passed as -d. Only binds at positions deeper than this. Capping subsamples reads, so allele *fraction* is roughly preserved and only absolute counts shrink -- a variant is lost only if its capped ALT count falls under min_counts (roughly AF < min_counts/max_depth at capped sites). Raise it when hunting low-frequency somatic/subclonal variants; lower it for speed on deep RNA-seq pileups. Default: 1000
    - split_bam_by_n                 (bool) Split BAM by N in CIGAR using GATK SplitNCigarReads. Default: False
    - disable_bcftools_norm          (bool) Disable bcftools norm. Default: False
    - rm_dup                         (str) Duplicate-removal mode passed to `bcftools norm -d`, or "none" to omit -d entirely. Because -m -any runs first, every ALT of a multi-allelic site becomes its own record at the same POS -- and "all" collapses records by position alone, so only the first ALT survives. That silently discards the other alleles at exactly the sites that carry them (het indels, homopolymer/STR tracts), which is the opposite of what a sensitive candidate generator wants. "exact" dedups on CHROM+POS+REF+ALT and keeps every distinct allele. Default: "exact"
    - bcftools_call_prior            (str) Prior for bcftools call. Default: None
    - merge_bam_files                (bool) Merge multiple BAM files into one for variant calling with Bowtie2. Default: False
    - strip_version_numbers          (bool) Strip version numbers from sequence names in output. Default: False
    - tmp_dir                        (str) Temporary directory for intermediate files. Default: None
    - threads                        (int) Number of threads to use. Default: 1
    - overwrite                      (bool) Overwrite output files if they exist. Default: False
    - verbose                        (int) Increase output verbosity. 0 is WARNING, 1 is INFO, 2 or greater is DEBUG. Default: 0
    - quiet                          (bool) Suppress all output. Default: False
    """
    #* Configure logger
    configure_logger(verbose, quiet)

    #* Check tools (only the chosen caller's dependencies: bam2vcf and cigar need neither
    #  bcftools nor a BAM index, so requiring them up front would block those callers)
    if variant_caller == "bcftools":
        check_tool("bcftools")

    #* Validate arguments (per-parameter types on the signature via @validate_call;
    #  cross-field / filesystem checks in DenovoParams). aligner/variant_caller/parity
    #  are enforced by their signature types, so they need no re-check here.
    DenovoParams(**make_function_parameter_to_value_dict(1))

    #* Resolve the bam2vcf gates (must follow DenovoParams, which needs the raw values to
    #  tell "unset" from an explicit request). bam2vcf gets the defaults; the other callers
    #  get no-ops, since bcftools expresses these through mpileup's own -q/-Q and --include
    #  while cigar ignores qualities entirely, and DenovoParams has already refused an
    #  explicit value there.
    if variant_caller == "bam2vcf":
        defaulted = [f"{name}={value}" for name, value, given in (
            ("min_mapq", BAM2VCF_DEFAULT_MIN_MAPQ, min_mapq),
            ("min_baseq", BAM2VCF_DEFAULT_MIN_BASEQ, min_baseq),
        ) if given is None]
        if min_mapq is None:
            min_mapq = BAM2VCF_DEFAULT_MIN_MAPQ
        if min_baseq is None:
            min_baseq = BAM2VCF_DEFAULT_MIN_BASEQ
        if min_vaf is None:
            min_vaf = BAM2VCF_DEFAULT_MIN_VAF
        if defaulted:
            logger.info("bam2vcf quality defaults applied (%s); pass 0 to disable a filter", ", ".join(defaulted))
    else:
        min_mapq = 0 if min_mapq is None else min_mapq
        min_baseq = 0 if min_baseq is None else min_baseq
        min_vaf = 0.0 if min_vaf is None else min_vaf

    if min_counts < 2:
        min_counts = 0
        message = "Filtering by a minimum count threshold is highly recommended."
        if variant_caller == "bcftools":
            # bam2vcf and cigar read variants straight off the alignments, so singleton
            # indels do survive there; only mpileup drops them.
            message += " Additionally, indels observed once will not be output regardless of settings (bcftools mpileup behavior)."
        logger.warning(message)

    #* Advise on the fraction floor rather than imposing one. Without it bam2vcf reports every
    #  allele that clears min_counts, so a deeply covered site contributes a candidate for each
    #  recurrent error; min_counts alone cannot tell 3 reads out of 5 from 3 out of 3000. The
    #  advice is technology-dependent because the cost of a floor is: germline DNA variants sit
    #  at ~0.5/1.0 and lose almost nothing, whereas RNA fractions are pulled around by
    #  allele-specific expression, so there min_counts is the safer lever.
    if variant_caller == "bam2vcf" and min_vaf < BAM2VCF_LOW_MIN_VAF:
        reads_type = "dna" if (technology or "").upper() == "DNA" else ("rna" if technology else None)
        advice = (
            f"For germline DNA, min_vaf={BAM2VCF_SUGGESTED_MIN_VAF_DNA} removes most error-driven candidates at little recall cost."
            if reads_type == "dna"
            else "For RNA, a fraction floor is riskier because allele-specific expression puts real variants at low VAF, so prefer raising min_counts."
            if reads_type == "rna"
            else f"For germline DNA, min_vaf={BAM2VCF_SUGGESTED_MIN_VAF_DNA} is a reasonable floor; for RNA or somatic/subclonal work, prefer raising min_counts instead."
        )
        logger.warning(
            "min_vaf=%g applies no variant allele fraction floor, so bam2vcf will report every allele passing min_counts=%s, "
            "including recurrent sequencing errors at deeply covered sites. %s Keep min_vaf low if you are deliberately "
            "hunting low-fraction variants.",
            min_vaf, min_counts, advice,
        )

    #* Validate inputs
    if isinstance(inputs, str):
        inputs = [inputs]
    elif isinstance(inputs, tuple):
        inputs = list(inputs)
    elif not isinstance(inputs, list):
        raise ValueError("inputs must be a string or a list/tuple of strings")
    if not inputs:
        raise ValueError("at least one input file is required")
    if any("," in input_file for input_file in inputs):
        raise ValueError("inputs must be provided as separate values; comma-separated input lists are not supported")
    
    valid_fastq_extensions = [".fq", ".fastq", ".fq.gz", ".fastq.gz"]
    valid_bam_extensions = [".bam"]
    input_type = None
    for file in inputs:
        if any(file.endswith(ext) for ext in valid_fastq_extensions):
            if input_type is None:
                input_type = "fastq"
            elif input_type != "fastq":
                raise ValueError("all inputs must be of the same type (either FASTQ or BAM)")
        elif any(file.endswith(ext) for ext in valid_bam_extensions):
            if input_type is None:
                input_type = "bam"
            elif input_type != "bam":
                raise ValueError("all inputs must be of the same type (either FASTQ or BAM)")
        else:
            raise ValueError(f"input file '{file}' must be a FASTQ or BAM file")
        if not os.path.isfile(file):
            raise ValueError(f"input file '{file}' not found")

    if input_type == "bam":
        if parity != "single":
            raise ValueError("parity='paired' is only supported for FASTQ inputs")
        single_fastq_files = []
        paired_fastq_files = []
    elif parity == "single":
        single_fastq_files = inputs
        paired_fastq_files = []
    else:
        if len(inputs) % 2 != 0:
            raise ValueError("paired FASTQ inputs require an even number of files in alternating R1/R2 order")
        single_fastq_files = []
        paired_fastq_files = list(zip(inputs[0::2], inputs[1::2]))

    #* Validate the aligner against the biology (only matters when we actually align FASTQs).
    # Splice-aware STAR is required only when RNA reads are aligned to a genome (spliced reads span
    # exon-exon junctions); bowtie2 is correct otherwise (DNA->genome, or RNA->transcriptome).
    if input_type == "fastq":
        # Derive the read type from `technology` the same way vk count does: the virtual "DNA"
        # technology declares genomic DNA reads, any other (RNA) technology declares RNA reads.
        technology_normalized = (technology or "").upper() or None
        if technology_normalized == "DNA":
            reads_type_normalized = "dna"
        elif technology_normalized is not None:
            reads_type_normalized = "rna"
        else:
            reads_type_normalized = None

        ref_type = (reference_type or "auto").lower()
        if ref_type in {"dna", "genome", "genomic", "g"}:
            reference_is_genome = True
        elif ref_type in {"cdna", "transcriptome", "transcript", "c"}:
            reference_is_genome = False
        elif ref_type == "auto":
            reference_is_genome = infer_reference_type_from_fasta(sequences) == "genome"
        else:
            raise ValueError("reference_type must be one of auto, genome, dna, transcriptome, cdna")

        if reference_is_genome:
            if reads_type_normalized is None:
                raise ValueError(
                    "Cannot verify the aligner for a genome reference without knowing the read type. "
                    "Set technology to an RNA technology (spliced RNA reads -> STAR) or technology='dna' (DNA reads -> bowtie2)."
                )
            expected_aligner = "STAR" if reads_type_normalized == "rna" else "bowtie2"
        else:
            expected_aligner = "bowtie2"  # transcriptome reference: no splicing in alignment

        if aligner != expected_aligner:
            ref_desc = "genome" if reference_is_genome else "transcriptome"
            reads_desc = f"{reads_type_normalized} reads" if reads_type_normalized else "reads"
            reason = (
                "splice-aware STAR is required to align RNA reads across exon-exon junctions on a genome"
                if expected_aligner == "STAR"
                else "bowtie2 is appropriate because no splicing occurs in this alignment"
            )
            raise ValueError(
                f"aligner='{aligner}' is incompatible with a {ref_desc} reference and {reads_desc}. "
                f"Use aligner='{expected_aligner}' ({reason})."
            )

    #* Define derivative variables
    output_type = "-Oz" if output.endswith(".gz") else "-Ov"
    os.makedirs(os.path.dirname(output) or ".", exist_ok=True)


    do_filtering = (min_counts > 1) or (include is not None)
    if do_filtering:
        filter_expression = f"bcftools filter --threads {threads}"
        if include:
            filter_expression += f" -i '{include}'"
        if min_counts > 1:
            filter_expression += f" -i 'INFO/AD[1] >= {min_counts}'"

    #* Align reads if BAM doesn't exist
    bam_for_bcftools = None
    if input_type == "fastq":
        check_tool("samtools")
        if aligner == "STAR":
            check_tool("STAR")

            if read_length is None:
                first_fastq = single_fastq_files[0] if parity == "single" else paired_fastq_files[0][0]
                with gzip.open(first_fastq, "rt") if first_fastq.endswith(".gz") else open(first_fastq, "rt") as f:
                    f.readline()  # skip header
                    read_length = len(f.readline().strip())  # get read length

            read_length_minus_one = read_length - 1
            bam_for_bcftools = f"{star_alignment_prefix}Aligned.sortedByCoord.out.bam"
            
            if not os.path.exists(bam_for_bcftools):
                #* Build STAR genome if needed
                if not os.path.isdir(star_genome_index_dir) or not os.listdir(star_genome_index_dir):
                    logger.info(f"Building STAR genome index at {star_genome_index_dir}...")
                    star_build_command = f"STAR --runThreadN {threads} --runMode genomeGenerate --genomeDir {star_genome_index_dir} --genomeFastaFiles {sequences} --sjdbGTFfile {gtf} --sjdbOverhang {read_length_minus_one} --limitSjdbInsertNsj 1000000 --limitBAMsortRAM 0"
                    run(star_build_command)
                
                logger.info("Running STAR alignment...")
                if parity == "paired":
                    inputs_star = " ".join([",".join(fastq1 for fastq1, _ in paired_fastq_files), ",".join(fastq2 for _, fastq2 in paired_fastq_files)])
                else:
                    inputs_star = ",".join(single_fastq_files)
                # TODO: add merge_bam_files logic for STAR (right now it always merges into one BAM)
                cmd = f"""STAR --runThreadN {threads} \
                    --genomeDir {star_genome_index_dir} \
                    --readFilesIn {inputs_star} \
                    --sjdbOverhang {read_length_minus_one} \
                    --outFileNamePrefix {star_alignment_prefix} \
                    --outSAMtype BAM SortedByCoordinate \
                    --outSAMattributes NH HI AS nM MD \
                    --outSAMmapqUnique 60 \
                    --twopassMode Basic \
                    --limitSjdbInsertNsj 1000000 \
                    --limitBAMsortRAM 0"""
                if inputs[0].endswith(".gz"):
                    cmd += " --readFilesCommand zcat"
                run(cmd)

                #* Split spliced reads if requested
                if split_bam_by_n:
                    check_tool("gatk")
                    split_bam = f"{star_alignment_prefix}split_exons.sorted.bam"
                    if not os.path.exists(split_bam):
                        # TODO: rewrite without GATK dependency
                        split_cmd = f"gatk SplitNCigarReads -R {sequences} -I {bam_for_bcftools} -O {split_bam} --create-output-bam-index"
                        if tmp_dir:
                            split_cmd += f" --tmp-dir {tmp_dir}"
                        run(split_cmd)
                    bam_for_bcftools = split_bam

        elif aligner == "bowtie2":
            check_tool("bowtie2")
            
            bowtie2_options = ""
            if bowtie2_seed_length is not None:
                bowtie2_options += f" -L {bowtie2_seed_length}"
            if bowtie2_score_min is not None:
                bowtie2_options += f" --score-min {bowtie2_score_min}"
            
            bowtie2_genome_index_file = f"{bowtie2_genome_index_prefix}.1.bt2"
            bowtie_build_command = f"bowtie2-build {sequences} {bowtie2_genome_index_prefix}"

            if merge_bam_files:
                bam_for_bcftools = os.path.join(bowtie2_alignment_dir, "aligned.sorted.bam")
                if not os.path.exists(bam_for_bcftools):
                    if not os.path.exists(bowtie2_genome_index_file):
                        run(bowtie_build_command)
                    os.makedirs(bowtie2_alignment_dir, exist_ok=True)
                    if parity == "paired":
                        fastq1_arg = ",".join(fastq1 for fastq1, _ in paired_fastq_files)
                        fastq2_arg = ",".join(fastq2 for _, fastq2 in paired_fastq_files)
                        bowtie2_align_command = f"bowtie2 --xeq --very-sensitive {bowtie2_options} --threads {threads} -x {bowtie2_genome_index_prefix} -1 {fastq1_arg} -2 {fastq2_arg} | samtools view -bS - | samtools sort -o {bam_for_bcftools}"
                    else:
                        fastq_arg = ",".join(single_fastq_files)
                        bowtie2_align_command = f"bowtie2 --xeq --very-sensitive {bowtie2_options} --threads {threads} -x {bowtie2_genome_index_prefix} -U {fastq_arg} | samtools view -bS - | samtools sort -o {bam_for_bcftools}"
                    logger.info("Running Bowtie2 alignment...")
                    run(bowtie2_align_command)
            else:
                bam_for_bcftools = []
                first_fastq = inputs[0]
                if not out_bam_dir:
                    out_bam_dir = os.path.dirname(first_fastq)
                os.makedirs(out_bam_dir, exist_ok=True)
                sequences_base = os.path.basename(sequences)
                sequences_base = re.sub(r"\.(fa|fasta|fna)(\.gz)?$", "", sequences_base)
                sequences_base = sequences_base.replace(".", "_")
                if parity == "paired":
                    for fastq1, fastq2 in paired_fastq_files:
                        fq_base = re.sub(r"\..*", "", os.path.basename(fastq1))
                        bam_out = os.path.join(out_bam_dir, f"{fq_base}_aligned_to_{sequences_base}.bam")
                        if not os.path.exists(bam_out):
                            if not os.path.exists(bowtie2_genome_index_file):
                                run(bowtie_build_command)
                            bowtie2_align_command = f"bowtie2 --xeq --very-sensitive {bowtie2_options} --threads {threads} -x {bowtie2_genome_index_prefix} -1 {fastq1} -2 {fastq2} | samtools view -bS - | samtools sort -o {bam_out}"
                            run(bowtie2_align_command)
                        bam_for_bcftools.append(bam_out)
                else:
                    for fastq in single_fastq_files:
                        fq_base = re.sub(r"\..*", "", os.path.basename(fastq))
                        bam_out = os.path.join(out_bam_dir, f"{fq_base}_aligned_to_{sequences_base}.bam")
                        if not os.path.exists(bam_out):
                            if not os.path.exists(bowtie2_genome_index_file):
                                run(bowtie_build_command)
                            bowtie2_align_command = f"bowtie2 --xeq --very-sensitive {bowtie2_options} --threads {threads} -x {bowtie2_genome_index_prefix} -U {fastq} | samtools view -bS - | samtools sort -o {bam_out}"
                            run(bowtie2_align_command)
                        bam_for_bcftools.append(bam_out)
                
                bam_for_bcftools = " ".join(bam_for_bcftools)
        else:
            raise ValueError(f"aligner '{aligner}' not supported")
    elif input_type == "bam":
        bam_for_bcftools = " ".join(inputs)

    #* Index BAM (mpileup needs it; bam2vcf and cigar stream the BAM instead)
    assert isinstance(bam_for_bcftools, str)
    bam_files = shlex.split(bam_for_bcftools)
    for bam in bam_files:
        if not os.path.exists(bam):
            raise ValueError(f"BAM file '{bam}' not found")
        if variant_caller != "bcftools":
            continue
        bai = bam + ".bai"
        if not os.path.exists(bai):
            check_tool("samtools")
            run(f"samtools index -@ {threads} {bam}")
    
    #* bam2vcf: read every variant recorded in the alignment records themselves
    if variant_caller == "bam2vcf":
        if len(bam_files) > 1:
            raise NotImplementedError(
                f"the 'bam2vcf' variant caller reads a single BAM, but {len(bam_files)} were given. "
                "Pass merge_bam_files=True to combine them, or run one input at a time."
            )
        # These shape bcftools' statistical model, which bam2vcf has no counterpart for.
        # Silently ignoring a filter expression would misreport what was called, so refuse.
        if include:
            raise ValueError("--include is a bcftools filter expression and is only supported with variant_caller='bcftools'")
        if bcftools_call_prior:
            raise ValueError("--bcftools-call-prior is only supported with variant_caller='bcftools'")
        for name, value, default in (("max-depth", max_depth, 1000), ("rm-dup", rm_dup, "exact"), ("disable-baq", disable_baq, True)):
            if value != default:
                logger.info("--%s applies only to variant_caller='bcftools' and is ignored by bam2vcf", name)

        stats = bam_to_vcf(
            bam=bam_files[0],
            output=output,
            reference=sequences,
            regions=regions or None,
            min_count=min_counts,
            min_vaf=min_vaf,
            max_vaf=max_vaf,
            min_mapq=min_mapq,
            min_baseq=min_baseq,
            threads=threads,
            normalize=(not disable_bcftools_norm),
            skip_indels=skip_indels,
            index=output.endswith(".vcf.gz"),
            strip_version_numbers=strip_version_numbers,
            overwrite=True,  # DenovoParams already applied the overwrite policy
            engine=bam2vcf_engine,
            progress=(verbose >= 1),
            logger=logger,
        )
        logger.info(
            "bam2vcf (%s engine) wrote %s records at %s sites from %s reads",
            stats.get("engine"), stats.get("records_emitted"), stats.get("sites_emitted"), stats.get("reads_used"),
        )

        if output_tsv:
            vcf_to_tsv(output, output_tsv, reference_type=tsv_reference_type, overwrite=overwrite, logger=logger)

    #* bcftools mpileup
    elif variant_caller == "bcftools":
        if not output.endswith(".vcf") and not output.endswith(".vcf.gz"):
            raise ValueError("when using 'bcftools' variant caller, --output must end with .vcf or .vcf.gz")
        
        bcftools_cmd = f"bcftools mpileup --threads {threads} -A -f {sequences} -a INFO/AD -Q 0 -d {max_depth} -Ou"
        if regions:
            bcftools_cmd += f" -R {regions}"
        if disable_baq:
            bcftools_cmd += " -B"
        if skip_indels:
            bcftools_cmd += " -I"
        bcftools_cmd += f" {bam_for_bcftools}"

        #* bcftools filter
        if do_filtering:
            bcftools_cmd += f" | {filter_expression} -Ou"
        
        #* bcftools call
        bcftools_cmd += f" | bcftools call -m -A -v --threads {threads}"
        if bcftools_call_prior:
            bcftools_cmd += f" --prior {bcftools_call_prior}"

        #* optional: bcftools norm and additional filter (must repeat after normalization)
        if not disable_bcftools_norm:
            rm_dup_arg = "" if rm_dup == "none" else f" -d {rm_dup}"
            bcftools_cmd += f" -Ou | bcftools norm -f {sequences} -c s{rm_dup_arg} -m -any --threads {threads}"
            if do_filtering:
                bcftools_cmd += f" -Ou | {filter_expression}"

        bcftools_cmd += f" {output_type} -o {output}"

        run(bcftools_cmd)

        #* optional: strip version numbers
        if strip_version_numbers:
            tmp_fh = tempfile.NamedTemporaryFile(delete=False, suffix=".vcf.gz" if output_type == "-Oz" else ".vcf")
            tmp_file = tmp_fh.name
            tmp_fh.close()
            logger.info(f"Stripping version numbers in {output}...")
            if output_type == "-Oz":
                # compressed .vcf.gz
                cmd = f"""
                zcat {output} |
                awk '
                    BEGIN {{ OFS="\\t" }}
                    /^##contig=/ {{ sub(/\\.[0-9]+/, "", $0); print; next }}
                    /^#/ {{ print; next }}
                    {{ sub(/\\.[0-9]+$/, "", $1); print }}
                ' |
                bgzip -c > {tmp_file}
                """.strip()


            else:
                # uncompressed .vcf
                cmd = f"""
                awk '
                    BEGIN {{ OFS="\\t" }}
                    /^##contig=/ {{ sub(/\\.[0-9]+/, "", $0); print; next }}
                    /^#/ {{ print; next }}
                    {{ sub(/\\.[0-9]+$/, "", $1); print }}
                ' {output} > {tmp_file}
                """.strip()
            
            run(cmd)
            shutil.move(tmp_file, output)

        #* index compressed VCF
        if output.endswith(".vcf.gz"):
            run(f"bcftools index -f --threads {threads} {output}")

        if output_tsv:
            vcf_to_tsv(output, output_tsv, reference_type=tsv_reference_type, overwrite=overwrite, logger=logger)
    
    elif variant_caller == "cigar":
        if output_tsv:
            raise ValueError("--output-tsv is only supported with the 'bam2vcf' and 'bcftools' variant callers, which write VCF")
        if not output.endswith(".csv"):
            raise ValueError("when using 'cigar' variant caller, --output must end with .csv")
        if len(bam_files) > 1:
            raise NotImplementedError(
                f"the 'cigar' variant caller reads a single BAM, but {len(bam_files)} were given. "
                "Pass merge_bam_files=True to combine them, or run one input at a time."
            )
        parse_cigars(bam_path=bam_for_bcftools, total=None, out_plot=None, do_baq=(not disable_baq), regions=regions, min_threshold=min_counts, strip_version_numbers=strip_version_numbers, out_dataframe=output, logger=logger)
    else:
        raise ValueError(f"variant caller '{variant_caller}' not supported")

    logger.info(f"Program complete. VCF written to {output}")


read2vcf = denovo

def main():
    parser = argparse.ArgumentParser(description="STAR alignment + bcftools variant calling pipeline")
    parser.add_argument("inputs", nargs="+", help="Input BAMs or FASTQs. For paired FASTQs, pass alternating R1/R2 files.")
    parser.add_argument("-s", "--sequences", required=True, help="Path to a reference genome FASTA file — the same file passed as `sequences` to vk build")
    parser.add_argument("-p", "--parity", default="single", choices=["single", "paired"], help="FASTQ parity. Use paired to pair alternating R1/R2 inputs.")
    parser.add_argument("-g", "--gtf", default="", help="genome annotation GTF file")
    parser.add_argument("-xs", "--star-genome-index-dir", default="genome_index", help="STAR or Bowtie2 genome index directory")
    parser.add_argument("-xb", "--bowtie2-genome-index-prefix", default="bowtie2_index", help="prefix for Bowtie2 genome index files")
    parser.add_argument("--star-alignment-prefix", default="star_", help="prefix for STAR output BAM")
    parser.add_argument("--bowtie2-alignment-dir", default="bowtie2_alignments", help="directory for Bowtie2 output BAMs")
    parser.add_argument("-R", "--regions", default="", help="BED file of regions to restrict variant calling to")
    parser.add_argument("-obd", "--out-bam-dir", default="", help="Output directory for BAM files (for Bowtie2 aligner only when not merging BAMs)")
    parser.add_argument("-o", "--output", default="out.vcf.gz", help="Output VCF file (for the bam2vcf and bcftools variant callers) or CSV file (for the cigar variant caller)")
    parser.add_argument("--output-tsv", default="", help="Optional TSV output converted from the bcftools VCF with columns: seq_id, variant")
    parser.add_argument("--tsv-reference-type", default="auto", choices=["auto", "dna", "genome", "cdna", "transcriptome"], help="Variant coordinate prefix for --output-tsv. auto uses c. for transcript-like sequence IDs and g. otherwise.")
    parser.add_argument("-r", "--read-length", type=int, default=None, help="Read length (default: inferred from the first read)")
    parser.add_argument("-m", "--min-counts", type=int, default=3, help="Minimum count threshold for filtering (minimum reads supporting a variant)")
    parser.add_argument("--min-vaf", type=float, default=None, help=f"Minimum variant allele fraction AO/DP, inclusive; left off by default to keep somatic/subclonal variants, and {BAM2VCF_SUGGESTED_MIN_VAF_DNA} is a cheap specificity win on germline DNA (bam2vcf variant caller only, default {BAM2VCF_DEFAULT_MIN_VAF})")
    parser.add_argument("--max-vaf", type=float, default=1.0, help="Maximum variant allele fraction AO/DP, inclusive; lower it to drop likely-germline homozygous sites (bam2vcf variant caller only)")
    parser.add_argument("--min-mapq", type=int, default=None, help=f"Minimum read mapping quality; reads below it contribute to neither AO nor DP, and 0 disables the filter (bam2vcf variant caller only, default {BAM2VCF_DEFAULT_MIN_MAPQ})")
    parser.add_argument("--min-baseq", type=int, default=None, help=f"Minimum base quality at the variant position; low-quality bases contribute to neither AO nor DP, and 0 disables the filter (bam2vcf variant caller only, default {BAM2VCF_DEFAULT_MIN_BASEQ})")
    parser.add_argument("-a", "--aligner", default="bowtie2", choices=["STAR", "bowtie2"], help="Aligner to use: STAR (splice-aware; RNA reads to a genome) or bowtie2 (otherwise)")
    parser.add_argument("--technology", "--technology", default=None, help="Sequencing technology (same vocabulary as vk ref/count; 'dna' declares genomic DNA reads, any other technology declares RNA reads). Fills in the read type used to validate the aligner; required to validate the choice when the reference is a genome.")
    parser.add_argument("--reference-type", "--reference_type", default="auto", choices=["auto", "genome", "dna", "transcriptome", "cdna"], help="Whether --sequences is a genome or transcriptome, used to validate the aligner. auto infers it from the FASTA sequence IDs.")
    parser.add_argument("--variant-caller", default="bam2vcf", choices=["bam2vcf", "bcftools", "cigar"], help="Variant caller to use: bam2vcf (default; reads variants straight out of the alignment records, VCF with AO/DP/VAF), bcftools (mpileup/call/norm) or cigar (older Python walk, CSV of HGVS strings)")
    parser.add_argument("--bam2vcf-engine", default="auto", choices=["auto", "native", "python"], help="Implementation used by the bam2vcf caller: auto (compiled C++ with a Python fallback), native (require the compiled helper) or python")
    parser.add_argument("--bowtie2-seed-length", type=int, default=None, help="Seed length for Bowtie2 aligner")
    parser.add_argument("--bowtie2-score-min", default=None, help="Score minimum for Bowtie2 aligner")
    parser.add_argument("-i", "--include", default="", help="bcftools filter expression")
    parser.add_argument("-I", "--skip-indels", action="store_true", help="Skip indels")
    parser.add_argument("--enable-baq", dest="disable_baq", action="store_false", help="Enable BAQ computation in mpileup (disabled by default: BAQ raises specificity near indels but suppresses marginal calls and is a large runtime cost)")
    parser.add_argument("--max-depth", type=int, default=1000, help="Maximum reads per input file at any single position in mpileup (-d). Raise for low-frequency somatic/subclonal variants, lower for speed on deep pileups.")
    parser.add_argument("--split-bam-by-n", action="store_true", help="Split BAM by N in CIGAR (spliced reads)")
    parser.add_argument("--merge-bam-files", action="store_true", help="Merge multiple BAM files into one for variant calling (Bowtie2 only)")
    parser.add_argument("--strip-version-numbers", action="store_true", help="Strip version numbers from chromosome names in output VCF")
    parser.add_argument("--disable-bcftools-norm", action="store_true", help="Disable running bcftools norm")
    parser.add_argument("--rm-dup", default="exact", choices=["exact", "snps", "indels", "both", "all", "none"], help="Duplicate-removal mode for `bcftools norm -d` ('none' omits -d). 'all' collapses by position, so only the first ALT of a multi-allelic site survives -- use 'exact' to keep every distinct allele.")
    parser.add_argument("--bcftools-call-prior", default="", help="Prior for bcftools call")
    parser.add_argument("--tmp-dir", default="/tmp", help="Temporary directory for intermediate files") 
    parser.add_argument("-t", "--threads", type=int, default=1, help="Number of threads to use")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite output file if it exists")
    parser.add_argument("-v", "--verbose", action="count", default=0, help="Increase output verbosity (default logging.WARNING, -v logging.INFO, -vv for logging.DEBUG)") 
    parser.add_argument("-q", "--quiet", action="store_true", help="Suppress all output (overrides any verbose flag)") 
    args = parser.parse_args()

    denovo(
        inputs=args.inputs,
        sequences=args.sequences,
        parity=args.parity,
        gtf=args.gtf,
        star_genome_index_dir=args.star_genome_index_dir,
        bowtie2_genome_index_prefix=args.bowtie2_genome_index_prefix,
        star_alignment_prefix=args.star_alignment_prefix,
        bowtie2_alignment_dir=args.bowtie2_alignment_dir,
        regions=args.regions,
        out_bam_dir=args.out_bam_dir,
        output=args.output,
        output_tsv=args.output_tsv,
        tsv_reference_type=args.tsv_reference_type,
        read_length=args.read_length,
        min_counts=args.min_counts,
        aligner=args.aligner,
        technology=args.technology,
        reference_type=args.reference_type,
        variant_caller=args.variant_caller,
        bam2vcf_engine=args.bam2vcf_engine,
        min_vaf=args.min_vaf,
        max_vaf=args.max_vaf,
        min_mapq=args.min_mapq,
        min_baseq=args.min_baseq,
        bowtie2_seed_length=args.bowtie2_seed_length,
        bowtie2_score_min=args.bowtie2_score_min,
        include=args.include,
        skip_indels=args.skip_indels,
        disable_baq=args.disable_baq,
        max_depth=args.max_depth,
        split_bam_by_n=args.split_bam_by_n,
        merge_bam_files=args.merge_bam_files,
        strip_version_numbers=args.strip_version_numbers,
        disable_bcftools_norm=args.disable_bcftools_norm,
        rm_dup=args.rm_dup,
        bcftools_call_prior=args.bcftools_call_prior,
        tmp_dir=args.tmp_dir,
        threads=args.threads,
        overwrite=args.overwrite,
        verbose=args.verbose,
        quiet=args.quiet,
    )

if __name__ == "__main__":
    main()
