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

logger = logging.getLogger(__name__)

from .utils.varseek_denovo_utils import (
    infer_hgvs_prefix,
    infer_reference_type_from_fasta,
    vcf_allele_to_hgvs,
    vcf_to_tsv,
    configure_logger,
    run,
    check_tool,
    parse_cigars,
)


def denovo(
    inputs,
    sequences,
    parity="single",
    gtf=None,
    star_genome_index_dir="genome_index",
    bowtie2_genome_index_prefix="bowtie2_index",
    star_alignment_prefix="star_",
    bowtie2_alignment_dir="bowtie2_alignments",
    regions=None,
    out_bam_dir=None,
    output="out.vcf.gz",
    output_tsv=None,
    tsv_reference_type="auto",
    read_length=None,
    min_counts=3,
    aligner="bowtie2",
    reads_type=None,
    reference_type="auto",
    variant_caller="bcftools",
    bowtie2_seed_length=None,
    bowtie2_score_min=None,
    include=None,
    skip_indels=False,
    disable_baq=False,
    split_bam_by_n=False,
    disable_bcftools_norm=False,
    bcftools_call_prior=None,
    merge_bam_files=False,
    strip_version_numbers=False,
    tmp_dir=None,
    threads=1,
    overwrite=False,
    verbose=0,
    quiet=False,
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
    - output                         (str) Output VCF file for bcftools or CSV file for cigar. Default: "out.vcf.gz"
    - output_tsv                     (str) Optional TSV output converted from the bcftools VCF with columns seq_id and variant. Default: None
    - tsv_reference_type             (str) Variant coordinate prefix for output_tsv. One of auto, dna, genome, cdna, transcriptome. Default: "auto"
    - read_length                    (int) Read length. Default: None
    - min_counts                     (int) Minimum count threshold for filtering. Default: 3
    - aligner                        (str) Aligner to use. One of STAR or bowtie2. Splice-aware STAR is required only when RNA reads are aligned to a genome (reads span exon-exon junctions); bowtie2 is correct otherwise (DNA reads to a genome, or RNA reads to a transcriptome). A mismatch against `reads_type`/`reference_type` raises. Default: "bowtie2"
    - reads_type                     (str) Whether the reads are RNA (spliced) or DNA. One of "rna", "dna", or None. Only used to validate `aligner`; required to validate the choice when the reference is a genome. Default: None
    - reference_type                 (str) Whether `sequences` is a genome or transcriptome, used to validate `aligner`. One of auto, genome, dna, transcriptome, cdna. "auto" infers it from the FASTA sequence IDs. Default: "auto"
    - variant_caller                 (str) Variant caller to use. One of bcftools or cigar. Default: "bcftools"
    - bowtie2_seed_length            (int) Seed length for Bowtie2 aligner. Default: None
    - bowtie2_score_min              (str) Bowtie2 score-min setting. Default: None
    - include                        (str) bcftools filter expression. Default: None
    - skip_indels                    (bool) Skip indels. Default: False
    - disable_baq                    (bool) Disable BAQ computation in mpileup. Default: False
    - split_bam_by_n                 (bool) Split BAM by N in CIGAR using GATK SplitNCigarReads. Default: False
    - disable_bcftools_norm          (bool) Disable bcftools norm. Default: False
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

    #* Check tools
    for tool in ["bcftools"]:
        check_tool(tool)

    #* Validate flagged arguments
    if not output.endswith(".vcf") and not output.endswith(".vcf.gz"):
        raise ValueError("--output must end with .vcf or .vcf.gz")
    if sequences:
        valid_fasta_extensions = [".fa", ".fasta", ".fa.gz", ".fasta.gz", ".fna", ".fna.gz"]
        if not any(sequences.endswith(ext) for ext in valid_fasta_extensions):
            raise ValueError(f"-s/--sequences must be a FASTA file ending with {', '.join(valid_fasta_extensions)}")
        if not os.path.isfile(sequences):
            fasta_dir = os.path.dirname(sequences) or "."
            recommended_command = f"gget ref -r 111 -d -od {fasta_dir} -w dna human && gunzip {sequences}.gz"
            raise ValueError(f"FASTA reference '{sequences}' not found. Recommended command to download: {recommended_command}")
    if gtf:
        if not gtf.endswith(".gtf"):
            raise ValueError("--gtf must be a GTF file ending with .gtf")
        if not os.path.isfile(gtf):
            gtf_dir = os.path.dirname(gtf) or "."
            recommended_command = f"gget ref -r 111 -d -od {gtf_dir} -w gtf human && gunzip {gtf}.gz"
            raise ValueError(f"GTF file '{gtf}' not found. Recommended command to download: {recommended_command}")
    if regions:
        if not regions.endswith(".bed"):
            raise ValueError("--regions must be a BED file ending with .bed")
        if not os.path.isfile(regions):
            # recommended_command = f"awk '$3 == \"gene\" {{print $1, $4-1, $5, $10}}' OFS='\\t' {gtf} | sort -k1,1V -k2,2n -o {regions}"
            raise ValueError(f"regions BED file '{regions}' not found.")
    if min_counts < 2:
        min_counts = 0
        logger.warning("Filtering by a minimum count threshold is highly recommended. Additionally, indels observed once will not be output regardless of settings (bcftools mpileup behavior).")
    if not aligner in ["STAR", "bowtie2"]:
        raise ValueError("--aligner must be either 'STAR' or 'bowtie2'")
    if not variant_caller in ["bcftools", "cigar"]:
        raise ValueError("--variant-caller must be either 'bcftools' or 'cigar'")
    if parity not in ["single", "paired"]:
        raise ValueError("parity must be either 'single' or 'paired'")
    if os.path.exists(output) and not overwrite:
        raise ValueError(f"output file '{output}' already exists. Use --overwrite to overwrite.")
    if output_tsv:
        if not output_tsv.endswith(".tsv"):
            raise ValueError("--output-tsv must end with .tsv")
        if os.path.exists(output_tsv) and not overwrite:
            raise ValueError(f"TSV output file '{output_tsv}' already exists. Use --overwrite to overwrite.")
        infer_hgvs_prefix("chr1", reference_type=tsv_reference_type)
    
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
        reads_type_normalized = (reads_type or "").lower() or None
        if reads_type_normalized in {"dna", "genomic", "g"}:
            reads_type_normalized = "dna"
        elif reads_type_normalized in {"rna", "cdna", "transcriptomic", "c"}:
            reads_type_normalized = "rna"
        elif reads_type_normalized is not None:
            raise ValueError("reads_type must be one of 'rna', 'dna', or None")

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
                    "Set reads_type='rna' (spliced RNA reads -> STAR) or reads_type='dna' (DNA reads -> bowtie2)."
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

    #* Index BAM
    assert isinstance(bam_for_bcftools, str)
    bam_files = shlex.split(bam_for_bcftools)
    for bam in bam_files:
        if not os.path.exists(bam):
            raise ValueError(f"BAM file '{bam}' not found")
        bai = bam + ".bai"
        if not os.path.exists(bai):
            run(f"samtools index -@ {threads} {bam}")
    
    #* bcftools mpileup
    if variant_caller == "bcftools":
        if not output.endswith(".vcf") and not output.endswith(".vcf.gz"):
            raise ValueError("when using 'bcftools' variant caller, --output must end with .vcf or .vcf.gz")
        
        bcftools_cmd = f"bcftools mpileup --threads {threads} -A -f {sequences} -a INFO/AD -Q 0 -d 10000 -Ou"
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
            bcftools_cmd += f" -Ou | bcftools norm -f {sequences} -c s -d all -m -any --threads {threads}"
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
            raise ValueError("--output-tsv is only supported with the 'bcftools' variant caller")
        if not output.endswith(".csv"):
            raise ValueError("when using 'cigar' variant caller, --output must end with .csv")
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
    parser.add_argument("-o", "--output", default="out.vcf.gz", help="Output VCF file (For bcftools variant caller) or CSV file (for cigar variant caller)")
    parser.add_argument("--output-tsv", default="", help="Optional TSV output converted from the bcftools VCF with columns: seq_id, variant")
    parser.add_argument("--tsv-reference-type", default="auto", choices=["auto", "dna", "genome", "cdna", "transcriptome"], help="Variant coordinate prefix for --output-tsv. auto uses c. for transcript-like sequence IDs and g. otherwise.")
    parser.add_argument("-r", "--read-length", type=int, default=None, help="Read length (default: inferred from the first read)")
    parser.add_argument("-m", "--min-counts", type=int, default=3, help="Minimum count threshold for filtering")
    parser.add_argument("-a", "--aligner", default="bowtie2", choices=["STAR", "bowtie2"], help="Aligner to use: STAR (splice-aware; RNA reads to a genome) or bowtie2 (otherwise)")
    parser.add_argument("--reads-type", "--reads_type", default=None, choices=["rna", "dna"], help="Read type, used to validate the aligner. Required to validate the choice when the reference is a genome.")
    parser.add_argument("--reference-type", "--reference_type", default="auto", choices=["auto", "genome", "dna", "transcriptome", "cdna"], help="Whether --sequences is a genome or transcriptome, used to validate the aligner. auto infers it from the FASTA sequence IDs.")
    parser.add_argument("--variant-caller", default="bcftools", choices=["bcftools", "cigar"], help="Variant caller to use: bcftools or cigar")
    parser.add_argument("--bowtie2-seed-length", type=int, default=None, help="Seed length for Bowtie2 aligner")
    parser.add_argument("--bowtie2-score-min", default=None, help="Score minimum for Bowtie2 aligner")
    parser.add_argument("-i", "--include", default="", help="bcftools filter expression")
    parser.add_argument("-I", "--skip-indels", action="store_true", help="Skip indels")
    parser.add_argument("--disable-baq", action="store_true", help="Disable BAQ computation in mpileup")
    parser.add_argument("--split-bam-by-n", action="store_true", help="Split BAM by N in CIGAR (spliced reads)")
    parser.add_argument("--merge-bam-files", action="store_true", help="Merge multiple BAM files into one for variant calling (Bowtie2 only)")
    parser.add_argument("--strip-version-numbers", action="store_true", help="Strip version numbers from chromosome names in output VCF")
    parser.add_argument("--disable-bcftools-norm", action="store_true", help="Disable running bcftools norm")
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
        reads_type=args.reads_type,
        reference_type=args.reference_type,
        variant_caller=args.variant_caller,
        bowtie2_seed_length=args.bowtie2_seed_length,
        bowtie2_score_min=args.bowtie2_score_min,
        include=args.include,
        skip_indels=args.skip_indels,
        disable_baq=args.disable_baq,
        split_bam_by_n=args.split_bam_by_n,
        merge_bam_files=args.merge_bam_files,
        strip_version_numbers=args.strip_version_numbers,
        disable_bcftools_norm=args.disable_bcftools_norm,
        bcftools_call_prior=args.bcftools_call_prior,
        tmp_dir=args.tmp_dir,
        threads=args.threads,
        overwrite=args.overwrite,
        verbose=args.verbose,
        quiet=args.quiet,
    )

if __name__ == "__main__":
    main()
