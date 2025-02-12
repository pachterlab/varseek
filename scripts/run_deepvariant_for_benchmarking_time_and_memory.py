import argparse
import os

import pysam

from varseek.utils import run_command_with_error_logging

parser = argparse.ArgumentParser(description="Run Deepvariant on a set of reads and report the time and memory usage")

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

args = parser.parse_args()

star_genome_dir = args.star_genome_dir if args.star_genome_dir else os.path.join(args.tmp, "star_genome")
deepvariant_output_dir = os.path.join(args.tmp, "deepvariant_simulated_data_dir")
reference_genome_fasta = args.reference_genome_fasta
reference_genome_gtf = args.reference_genome_gtf
threads = args.threads
read_length_minus_one = args.read_length - 1
synthetic_read_fastq = args.synthetic_read_fastq
aligned_and_unmapped_bam = args.aligned_and_unmapped_bam

STAR = args.STAR

os.makedirs(deepvariant_output_dir, exist_ok=True)
os.makedirs(star_genome_dir, exist_ok=True)

alignment_folder = f"{deepvariant_output_dir}/alignment"
out_file_name_prefix = f"{alignment_folder}/sample_"

deepvariant_vcf = os.path.join(deepvariant_output_dir, "results/variants/genome.vcf.gz")

intermediate_results = os.path.join(deepvariant_output_dir, "intermediate_results_dir")
os.makedirs(intermediate_results, exist_ok=True)

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

#* deepvariant variant calling
BIN_VERSION="1.4.0"

deepvariant_command = [
    "sudo", "docker", "run",
    "-v", f"{os.getcwd()}:{os.getcwd()}",
    "-w", os.getcwd(),
    f"google/deepvariant:{BIN_VERSION}",
    "run_deepvariant",
    "--model_type=WES",
    "--customized_model=model/model.ckpt",
    f"--ref={reference_genome_fasta}",
    f"--reads={aligned_and_unmapped_bam}",
    f"--output_vcf={deepvariant_vcf}",
    f"--num_shards={os.cpu_count()}",
    "--make_examples_extra_args=split_skip_reads=true,channel_list='BASE_CHANNELS'",
    "--intermediate_results_dir", intermediate_results
]

# # commented out, as these should already be done prior to running this script
# if not os.listdir(star_genome_dir):
#     run_command_with_error_logging(star_build_command)

# if not os.path.exists(f"{reference_genome_fasta}.fai"):
#     _ = pysam.faidx(reference_genome_fasta)

if not os.path.exists(aligned_and_unmapped_bam):
    aligned_and_unmapped_bam = f"{out_file_name_prefix}Aligned.sortedByCoord.out.bam"
    os.makedirs(alignment_folder, exist_ok=True)
    run_command_with_error_logging(star_align_command)

bam_index_file = f"{aligned_and_unmapped_bam}.bai"
if not os.path.exists(bam_index_file):
    _ = pysam.index(aligned_and_unmapped_bam)

run_command_with_error_logging(deepvariant_command)