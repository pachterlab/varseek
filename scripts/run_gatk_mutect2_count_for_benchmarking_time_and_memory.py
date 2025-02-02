import os
import argparse
import pysam
import subprocess
from varseek.utils import run_command_with_error_logging


parser = argparse.ArgumentParser(description="Run GATK Mutect2 on a set of reads and report the time and memory usage")

# Paths
parser.add_argument("--synthetic_read_fastq", help="Path to synthetic read FASTQ")
parser.add_argument("--reference_genome_fasta", help="Path to reference genome fasta")
parser.add_argument("--reference_genome_gtf", help="Path to reference genome GTF")
parser.add_argument("--genomes1000_vcf", help="Path to 1000 genomes vcf file")
parser.add_argument("--star_genome_dir", default="", help="Path to star_genome_dir")
parser.add_argument("--aligned_and_unmapped_bam", default="", help="Path to aligned_and_unmapped_bam. If not provided, will be created")
parser.add_argument("--tmp", default="tmp", help="Path to temp folder")

# Parameters
parser.add_argument("--threads", default=2, help="Number of threads")
parser.add_argument("--read_length", default=150, help="Read length")

# Executables
parser.add_argument("--STAR", default="STAR", help="Path to STAR executable")
parser.add_argument("--java", default="java", help="Path to java executable")
parser.add_argument("--picard_jar", default="picard.jar", help="Path to picard.jar executable")
parser.add_argument("--gatk", default="gatk", help="Path to gatk executable")

args = parser.parse_args()

star_genome_dir = args.star_genome_dir if args.star_genome_dir else os.path.join(args.tmp, "star_genome")
gatk_parent = os.path.join(args.tmp, "gatk")
reference_genome_fasta = args.reference_genome_fasta
reference_genome_gtf = args.reference_genome_gtf
genomes1000_vcf = args.genomes1000_vcf
threads = args.threads
read_length_minus_one = args.read_length - 1
synthetic_read_fastq = args.synthetic_read_fastq
aligned_and_unmapped_bam = args.aligned_and_unmapped_bam

STAR = args.STAR
java = args.java
picard_jar = args.picard_jar
gatk = args.gatk

for name, path in {"STAR": STAR, "java": java, "picard_jar": picard_jar, "gatk": gatk}.items():
    if not os.path.exists(name):
        raise FileNotFoundError(f"{name} not found.")
    elif not os.access(name, os.X_OK):
        raise PermissionError(f"{name} is not executable.")

java_home = os.path.dirname(os.path.dirname(java))

os.environ['JAVA_HOME'] = java_home
os.environ['PATH'] = f"{os.environ['JAVA_HOME']}/bin:" + os.environ['PATH']

os.makedirs(star_genome_dir, exist_ok=True)

alignment_folder = f"{gatk_parent}/alignment"
os.makedirs(alignment_folder, exist_ok=True)

gatk_supporting_files = f"{gatk_parent}/supporting_files"
os.makedirs(gatk_supporting_files, exist_ok=True)

plot_output_folder = f"{gatk_parent}/plots"
os.makedirs(plot_output_folder, exist_ok=True)

out_file_name_prefix = f"{alignment_folder}/sample_"

vcf_folder = f"{gatk_parent}/vcfs"
haplotypecaller_folder = f"{vcf_folder}/haplotypecaller"
mutect2_folder = f"{vcf_folder}/mutect2"

os.makedirs(vcf_folder, exist_ok=True)
os.makedirs(haplotypecaller_folder, exist_ok=True)
os.makedirs(mutect2_folder, exist_ok=True)

aligned_only_bam = f"{alignment_folder}/aligned_only.bam"
unmapped_bam = f"{alignment_folder}/unmapped.bam"
merged_bam = f"{alignment_folder}/merged.bam"

marked_duplicates_bam = f"{alignment_folder}/marked_duplicates.bam"
marked_dup_metrics_txt = f"{alignment_folder}/marked_dup_metrics.txt"

split_n_cigar_reads_bam = f"{alignment_folder}/split_n_cigar_reads.bam"
recal_data_table = f"{alignment_folder}/recal_data.table"
recalibrated_bam = f"{alignment_folder}/recalibrated.bam"
covariates_plot = f"{alignment_folder}/AnalyzeCovariates.pdf"
haplotypecaller_unfiltered_vcf = f"{haplotypecaller_folder}/haplotypecaller_output_unfiltered.g.vcf.gz"

haplotypecaller_filtered_vcf = f"{haplotypecaller_folder}/haplotypecaller_output_filtered.vcf.gz"
haplotypecaller_filtered_applied_vcf = f"{haplotypecaller_folder}/haplotypecaller_output_filtered_applied.vcf.gz"

panel_of_normals_vcf = f"{gatk_supporting_files}/1000g_pon.hg38.vcf.gz"
panel_of_normals_vcf_filtered = f"{gatk_supporting_files}/1000g_pon.hg38_filtered.vcf.gz"
mutect2_unfiltered_vcf = f"{mutect2_folder}/mutect2_output_unfiltered.g.vcf.gz"
mutect2_filtered_vcf = f"{mutect2_folder}/mutect2_output_filtered.vcf.gz"
mutect2_filtered_applied_vcf = f"{mutect2_folder}/mutect2_output_filtered_applied.vcf.gz"

reference_genome_dict = reference_genome_fasta.replace(".fa", ".dict")

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

fastq_to_sam_command = [
    java, "-jar", picard_jar, "FastqToSam",
    "-FASTQ", synthetic_read_fastq,
    "-OUTPUT", unmapped_bam,
    "-READ_GROUP_NAME", "rg1",
    "-SAMPLE_NAME", "sample1",
    "-LIBRARY_NAME", "lib1",
    "-PLATFORM_UNIT", "unit1",
    "-PLATFORM", "ILLUMINA",
    "-SEQUENCING_CENTER", "center1"
]

create_sequence_dict_command = [
    java, "-jar", picard_jar, "CreateSequenceDictionary",
    "-R", reference_genome_fasta,
    "-O", reference_genome_dict
]

merge_bam_alignment_command = [
    java, "-jar", picard_jar, "MergeBamAlignment",
    "--ALIGNED_BAM", aligned_and_unmapped_bam,
    "--UNMAPPED_BAM", unmapped_bam,
    "--OUTPUT", merged_bam,
    "--REFERENCE_SEQUENCE", reference_genome_fasta,
    "--SORT_ORDER", "coordinate",
    "--INCLUDE_SECONDARY_ALIGNMENTS", "false",
    "--VALIDATION_STRINGENCY", "SILENT"
]

mark_duplicates_command = [
    java, "-jar", picard_jar, "MarkDuplicates",
    "--INPUT", merged_bam,
    "--OUTPUT", marked_duplicates_bam,
    "--METRICS_FILE", marked_dup_metrics_txt,
    "--CREATE_INDEX", "true",
    "--VALIDATION_STRINGENCY", "SILENT"
]

split_n_cigar_reads_command = [
    gatk, "SplitNCigarReads",
    "-R", reference_genome_fasta,
    "-I", marked_duplicates_bam,
    "-O", split_n_cigar_reads_bam
]

index_feature_file_command = [
    gatk, "IndexFeatureFile",
    "-I", genomes1000_vcf
]

base_recalibrator_command = [
    gatk, "BaseRecalibrator",
    "-I", split_n_cigar_reads_bam,
    "-R", reference_genome_fasta,
    "--use-original-qualities",
    "--known-sites", genomes1000_vcf,
    "-O", recal_data_table
]

apply_bqsr_command = [
    gatk, "ApplyBQSR",
    "--add-output-sam-program-record",
    "-R", reference_genome_fasta,
    "-I", split_n_cigar_reads_bam,
    "--use-original-qualities",
    "--bqsr-recal-file", recal_data_table,
    "-O", recalibrated_bam
]

analyze_covariates_command = [
    gatk, "AnalyzeCovariates",
    "-bqsr", recal_data_table,
    "-plots", covariates_plot
]

mutect2_command = [
    gatk, "Mutect2",
    "-R", reference_genome_fasta,
    "-I", recalibrated_bam,
    "-O", mutect2_unfiltered_vcf,
    "--min-base-quality-score", "10"
]

filter_mutect_calls_command = [
    gatk, "FilterMutectCalls",
    "-R", reference_genome_fasta,
    "-V", mutect2_unfiltered_vcf,
    "-O", mutect2_filtered_vcf
]

select_variants_command = [
    gatk, "SelectVariants",
    "-V", mutect2_filtered_vcf,
    "--exclude-filtered", "true",
    "-O", mutect2_filtered_applied_vcf
]

# # commented out, as these should already be done prior to running this script
# reference_genome_fasta_url = "https://ftp.ensembl.org/pub/grch37/release-93/fasta/homo_sapiens/dna/Homo_sapiens.GRCh37.dna.primary_assembly.fa.gz"
# reference_genome_gtf_url = "https://ftp.ensembl.org/pub/grch37/release-93/gtf/homo_sapiens/Homo_sapiens.GRCh37.87.gtf.gz"
# genomes1000_vcf_url = "https://ftp.ensembl.org/pub/grch37/release-93/variation/vcf/homo_sapiens/1000GENOMES-phase_3.vcf.gz"

# download_reference_genome_fasta_command = ["wget", "-O", f"{reference_genome_fasta}.gz", reference_genome_fasta_url]
# unzip_reference_genome_fasta_command = ["gunzip", f"{reference_genome_fasta}.gz"]

# download_reference_genome_gtf_command = ["wget", "-O", f"{reference_genome_gtf}.gz", reference_genome_gtf_url]
# unzip_reference_genome_gtf_command = ["gunzip", f"{reference_genome_gtf}.gz"]

# download_1000_genomes_command = ["wget", "-O", f"{genomes1000_vcf}.gz", genomes1000_vcf_url]
# unzip_1000_genomes_command = ["gunzip", f"{genomes1000_vcf}.gz"]

# if not os.path.exists(reference_genome_fasta):
#     run_command_with_error_logging(download_reference_genome_fasta_command)
#     run_command_with_error_logging(unzip_reference_genome_fasta_command)

# if not os.path.exists(reference_genome_gtf):
#     run_command_with_error_logging(download_reference_genome_gtf_command)
#     run_command_with_error_logging(unzip_reference_genome_gtf_command)

# if not os.path.exists(genomes1000_vcf):
#     run_command_with_error_logging(download_1000_genomes_command)
#     run_command_with_error_logging(unzip_1000_genomes_command)

# if not os.listdir(star_genome_dir):
#     run_command_with_error_logging(star_build_command)

# if not os.path.exists(f"{reference_genome_fasta}.fai"):
#     _ = pysam.faidx(reference_genome_fasta)

if not os.path.exists(aligned_and_unmapped_bam):
    aligned_and_unmapped_bam = f"{out_file_name_prefix}Aligned.sortedByCoord.out.bam"
    run_command_with_error_logging(star_align_command)

run_command_with_error_logging(fastq_to_sam_command)

run_command_with_error_logging(create_sequence_dict_command)

run_command_with_error_logging(merge_bam_alignment_command)

run_command_with_error_logging(mark_duplicates_command)

run_command_with_error_logging(split_n_cigar_reads_command)

if not os.path.exists(f"{genomes1000_vcf}.idx"):
    run_command_with_error_logging(index_feature_file_command)

run_command_with_error_logging(base_recalibrator_command)

run_command_with_error_logging(apply_bqsr_command)

run_command_with_error_logging(analyze_covariates_command)

run_command_with_error_logging(mutect2_command)

run_command_with_error_logging(filter_mutect_calls_command)

run_command_with_error_logging(select_variants_command)