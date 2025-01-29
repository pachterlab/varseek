import os
import pysam
import argparse
from varseek.utils import run_command_with_error_logging

parser = argparse.ArgumentParser(description="Run GATK Mutect2 on a set of reads and report the time and memory usage")

# Paths
parser.add_argument("--synthetic_read_fastq", help="Path to synthetic read FASTQ")
parser.add_argument("--reference_genome_fasta", help="Path to reference genome fasta")
parser.add_argument("--reference_genome_gtf", help="Path to reference genome GTF")
parser.add_argument("--star_genome_dir", default="", help="Path to star_genome_dir")
parser.add_argument("--tmp", default="tmp", help="Path to temp folder")

# Parameters
parser.add_argument("--threads", default=2, help="Number of threads")
parser.add_argument("--read_length", default=150, help="Read length")

# Executables
parser.add_argument("--STAR", default="STAR", help="Path to STAR executable")
parser.add_argument("--STRELKA_INSTALL_PATH", default="STRELKA_INSTALL_PATH", help="Path to STRELKA_INSTALL_PATH parent")

args = parser.parse_args()

star_genome_dir = args.star_genome_dir if args.star_genome_dir else os.path.join(args.tmp, "star_genome")
strelka2_output_dir = os.path.join(args.tmp, "strelka2_simulated_data_dir")
reference_genome_fasta = args.reference_genome_fasta
reference_genome_gtf = args.reference_genome_gtf
threads = args.threads
read_length_minus_one = args.read_length - 1
synthetic_read_fastq = args.synthetic_read_fastq

STAR = args.STAR
STRELKA_INSTALL_PATH = args.STRELKA_INSTALL_PATH

alignment_folder = f"{strelka2_output_dir}/alignment"
os.makedirs(strelka2_output_dir, exist_ok=True)
os.makedirs(star_genome_dir, exist_ok=True)
os.makedirs(alignment_folder, exist_ok=True)

out_file_name_prefix = f"{alignment_folder}/sample_"
aligned_and_unmapped_bam = f"{out_file_name_prefix}Aligned.sortedByCoord.out.bam"

#* Genome alignment with STAR
star_build_command = [
    STAR,
    "--runThreadN", str(threads),
    "--runMode", "genomeGenerate",
    "--genomeDir", star_genome_dir,
    "--genomeFastaFiles", reference_genome_fasta,
    "--sjdbGTFfile", reference_genome_gtf,
    "--sjdbOverhang", str(read_length_minus_one),
]

star_align_command = [
    STAR,
    "--runThreadN", str(threads),
    "--genomeDir", star_genome_dir,
    "--readFilesIn", synthetic_read_fastq,
    "--sjdbOverhang", str(read_length_minus_one),
    "--outFileNamePrefix", out_file_name_prefix,
    "--outSAMtype", "BAM", "SortedByCoordinate",
    "--outSAMunmapped", "Within",
    "--outSAMmapqUnique", "60",
    "--twopassMode", "Basic"
]

#* Strelka2 variant calling
strelka2_configure_command = [
    f"{STRELKA_INSTALL_PATH}/bin/configureStrelkaGermlineWorkflow.py",
    "--bam", aligned_and_unmapped_bam,
    "--referenceFasta", reference_genome_fasta,
    "--rna",
    "--runDir", strelka2_output_dir
]

strelka2_run_command = [
    f"{strelka2_output_dir}/runWorkflow.py",
    "-m", "local",
    "-j", str(threads)
]


# # commented out, as these should already be done prior to running this script
# if not os.listdir(star_genome_dir):
#     run_command_with_error_logging(star_build_command)

# if not os.path.exists(f"{reference_genome_fasta}.fai"):
#     _ = pysam.faidx(reference_genome_fasta)

if not os.path.exists(aligned_and_unmapped_bam):
    run_command_with_error_logging(star_align_command)

run_command_with_error_logging(strelka2_configure_command)

run_command_with_error_logging(strelka2_run_command)