import argparse
import os

import pysam

from varseek.utils import run_command_with_error_logging

parser = argparse.ArgumentParser(description="Run GATK Mutect2 on a set of reads and report the time and memory usage")

# Paths
parser.add_argument("--synthetic_read_fastq", help="Path to synthetic read FASTQ")
parser.add_argument("--reference_genome_fasta", help="Path to reference genome fasta")
parser.add_argument("--reference_genome_gtf", help="Path to reference genome GTF")
parser.add_argument("--star_genome_dir", default="", help="Path to star_genome_dir")
parser.add_argument("--aligned_and_unmapped_bam", default="", help="Path to aligned_and_unmapped_bam. If not provided, will be created")
parser.add_argument("--tmp", default="tmp", help="Path to temp folder")

# Parameters
parser.add_argument("--threads", default=2, help="Number of threads")
parser.add_argument("--read_length", default=150, help="Read length")

# Executables
parser.add_argument("--STAR", default="STAR", help="Path to STAR executable")
parser.add_argument("--VARSCAN_INSTALL_PATH", default="VARSCAN_INSTALL_PATH", help="Path to VARSCAN_INSTALL_PATH parent")

args = parser.parse_args()

star_genome_dir = args.star_genome_dir if args.star_genome_dir else os.path.join(args.tmp, "star_genome")
varscan_output_dir = os.path.join(args.tmp, "varscan_simulated_data_dir")
reference_genome_fasta = args.reference_genome_fasta
reference_genome_gtf = args.reference_genome_gtf
threads = args.threads
read_length_minus_one = args.read_length - 1
synthetic_read_fastq = args.synthetic_read_fastq
aligned_and_unmapped_bam = args.aligned_and_unmapped_bam

STAR = args.STAR
VARSCAN_INSTALL_PATH = args.VARSCAN_INSTALL_PATH


os.makedirs(varscan_output_dir, exist_ok=True)
os.makedirs(star_genome_dir, exist_ok=True)


alignment_folder = f"{varscan_output_dir}/alignment"
out_file_name_prefix = f"{alignment_folder}/sample_"
data_pileup_file = f"{varscan_output_dir}/simulated_data.pileup"

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

#* Samtools mpileup
samtools_mpileup_command = f"samtools mpileup -B -f {reference_genome_fasta} {aligned_and_unmapped_bam} > {data_pileup_file}"

#* Varscan variant calling
varscan_command = f"java -jar {VARSCAN_INSTALL_PATH} mpileup2cns {data_pileup_file} --output-vcf 1 --variants 1 --min-coverage 1 --min-reads2 1 --min-var-freq 0.0001 --strand-filter 0 --p-value 0.9999"


# # commented out, as these should already be done prior to running this script
# if not os.listdir(star_genome_dir):
#     run_command_with_error_logging(star_build_command)

# if not os.path.exists(f"{reference_genome_fasta}.fai"):
#     _ = pysam.faidx(reference_genome_fasta)

if not os.path.exists(aligned_and_unmapped_bam):
    os.makedirs(alignment_folder, exist_ok=True)
    aligned_and_unmapped_bam = f"{out_file_name_prefix}Aligned.sortedByCoord.out.bam"
    run_command_with_error_logging(star_align_command)

bam_index_file = f"{aligned_and_unmapped_bam}.bai"
if not os.path.exists(bam_index_file):
    _ = pysam.index(aligned_and_unmapped_bam)

run_command_with_error_logging(samtools_mpileup_command)

run_command_with_error_logging(varscan_command)