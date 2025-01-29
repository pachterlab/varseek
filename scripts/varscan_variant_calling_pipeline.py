import os
import pysam
import pandas as pd
import subprocess
from varseek.utils import vcf_to_dataframe, calculate_metrics, draw_confusion_matrix, create_venn_diagram, calculate_sensitivity_specificity, add_vcf_info_to_cosmic_tsv, merge_gatk_and_cosmic, safe_literal_eval

run_benchmarking = False  #!!! change to True when finished debugging the variant calling pipeline

### ARGUMENTS ###
synthetic_read_fastq = "/home/jrich/data/varseek_data_fresh/manuscript_worthy/vk_sim_2024dec17_complex_testing/synthetic_reads.fq"  #!!! update path
unique_mcrs_df_path = "/home/jrich/data/varseek_data_fresh/manuscript_worthy/vk_sim_2024dec17_complex_testing/unique_mcrs_df.csv"  #!!! update path
varscan_output_dir = "/home/jrich/data/varseek_data_fresh/manuscript_worthy/vk_sim_2024dec17_complex_testing/varscan_simulated_data_dir"

threads = 4
read_length = 150
mutation_source = "cdna"  # "cdna", "cds"

cosmic_tsv = "/home/jrich/data/varseek_data/reference/cosmic/CancerMutationCensus_AllData_Tsv_v100_GRCh37/CancerMutationCensus_AllData_v100_GRCh37.tsv"
cosmic_cdna_info_csv = "/home/jrich/data/varseek_data/reference/cosmic/CancerMutationCensus_AllData_Tsv_v100_GRCh37/CancerMutationCensus_AllData_v100_GRCh37_mutation_workflow_with_cdna.csv"

# if these paths don't exist then they will be created
reference_genome_fasta = "/home/jrich/data/varseek_data/reference/ensembl_grch37_release93/Homo_sapiens.GRCh37.dna.primary_assembly.fa"
reference_genome_gtf = "/home/jrich/data/varseek_data/reference/ensembl_grch37_release93/Homo_sapiens.GRCh37.87.gtf"
star_genome_dir = "/home/jrich/data/varseek_data/reference/ensembl_grch37_release93/star_reference"
star_alignment_dir = "/home/jrich/data/varseek_data_fresh/manuscript_worthy/vk_sim_2024dec17_complex_testing/star_alignment"

opt_dir = '/home/jrich/Desktop/variant_calling'
STAR = os.path.join(opt_dir, "STAR-2.7.11b/bin/Linux_x86_64/STAR")
VARSCAN_INSTALL_PATH = os.path.join(opt_dir, "VarScan.v2.3.9.jar")
### ARGUMENTS ###

varscan_benchmarking_output = os.path.join(varscan_output_dir, "benchmarking")

os.makedirs(star_genome_dir, exist_ok=True)
os.makedirs(star_alignment_dir, exist_ok=True)
os.makedirs(opt_dir, exist_ok=True)
os.makedirs(varscan_benchmarking_output, exist_ok=True)

out_file_name_prefix = f"{star_alignment_dir}/sample_"
aligned_and_unmapped_bam = f"{out_file_name_prefix}Aligned.sortedByCoord.out.bam"
read_length_minus_one = read_length - 1
star_tarball = os.path.join(opt_dir, "2.7.11b.tar.gz")
data_pileup_file = f"{varscan_output_dir}/simulated_data.pileup"
varscan_vcf = ""  #!!! update

#* Download software and reference files
if not os.path.exists(reference_genome_fasta):
    reference_genome_fasta_url = "https://ftp.ensembl.org/pub/grch37/release-93/fasta/homo_sapiens/dna/Homo_sapiens.GRCh37.dna.primary_assembly.fa.gz"
    download_reference_genome_fasta_command = ["wget", "-O", f"{reference_genome_fasta}.gz", reference_genome_fasta_url]
    unzip_reference_genome_fasta_command = ["gunzip", "-O", f"{reference_genome_fasta}.gz"]

    subprocess.run(download_reference_genome_fasta_command, check=True)
    subprocess.run(unzip_reference_genome_fasta_command, check=True)

if not os.path.exists(reference_genome_gtf):
    reference_genome_gtf_url = "https://ftp.ensembl.org/pub/grch37/release-93/gtf/homo_sapiens/Homo_sapiens.GRCh37.87.gtf.gz"
    download_reference_genome_gtf_command = ["wget", "-O", f"{reference_genome_gtf}.gz", reference_genome_gtf_url]
    unzip_reference_genome_gtf_command = ["gunzip", "-O", f"{reference_genome_gtf}.gz"]

    subprocess.run(download_reference_genome_gtf_command, check=True)
    subprocess.run(unzip_reference_genome_gtf_command, check=True)

if not os.path.exists(STAR):
    subprocess.run(["wget", "-O", star_tarball, "https://github.com/alexdobin/STAR/archive/2.7.11b.tar.gz"], check=True)
    subprocess.run(["tar", "-xzf", star_tarball, "-C", opt_dir], check=True)

if not os.path.exists(VARSCAN_INSTALL_PATH):
    subprocess.run(["wget", "-O", VARSCAN_INSTALL_PATH, "https://sourceforge.net/projects/varscan/files/VarScan.v2.3.9.jar/download"], check=True)

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

if not os.listdir(star_genome_dir):
    subprocess.run(star_build_command, check=True)

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

if not os.path.exists(aligned_and_unmapped_bam):
    subprocess.run(star_align_command, check=True)


#* Index reference genome
if not os.path.exists(f"{reference_genome_fasta}.fai"):
    _ = pysam.faidx(reference_genome_fasta)

#* BAM indexing (maybe not needed?) and sorting (maybe already done by STAR?)
samtools_index_command = ["samtools", "index", aligned_and_unmapped_bam]
subprocess.run(samtools_index_command, check=True)  # can possibly skip samtools sort below due to STAR mode --outSAMtype BAM SortedByCoordinate, but might need this step instead
# !samtools sort --threads $threads -o sorted.bam $aligned_and_unmapped_bam  # stated to be needed by VarScan documentation

#* Samtools mpileup
samtools_mpileup_command = f"samtools mpileup -B -f {reference_genome_fasta} {aligned_and_unmapped_bam} > {data_pileup_file}"
subprocess.run(samtools_mpileup_command, shell=True, check=True)

#* Varscan variant calling
varscan_command = f"java -jar {VARSCAN_INSTALL_PATH} mpileup2snp {data_pileup_file}"
subprocess.run(varscan_command, shell=True, check=True)

if not run_benchmarking:
    print("Skipping benchmarking")
    exit()

#* Merging into COSMIC#* Merging into COSMIC
# Convert VCF to DataFrame
df_varscan = vcf_to_dataframe(varscan_vcf, additional_columns = True)
df_varscan = df_varscan[['CHROM', 'POS', 'ID', 'REF', 'ALT', 'INFO_DP']].rename(columns={'INFO_DP': 'DP_varscan'})

cosmic_df = add_vcf_info_to_cosmic_tsv(cosmic_tsv=cosmic_tsv, reference_genome_fasta=reference_genome_fasta, cosmic_df_out = None, cosmic_cdna_info_csv = cosmic_cdna_info_csv, mutation_source = "cdna")

# load in unique_mcrs_df
unique_mcrs_df = pd.read_csv(unique_mcrs_df_path)
unique_mcrs_df.rename(columns={'received_an_aligned_read': 'mutation_detected_varseek', 'number_of_reads_aligned_to_this_item': 'DP_varseek', 'mutation_expression_prediction_error': 'mutation_expression_prediction_error_varseek', 'TP_crude': 'TP_varseek', 'FP_crude': 'FP_varseek', 'TN_crude': 'TN_varseek', 'FN_crude': 'FN_varseek', 'TP': 'TP_varseek_read_specific', 'FP': 'FP_varseek_read_specific', 'TN': 'TN_varseek_read_specific', 'FN': 'FN_varseek_read_specific'}, inplace=True)
unique_mcrs_df["header_list"] = unique_mcrs_df["header_list"].apply(safe_literal_eval)

varscan_cosmic_merged_df = merge_gatk_and_cosmic(df_varscan, cosmic_df, exact_position=False)  # change exact_position to True to merge based on exact position as before
id_set_varscan = set(varscan_cosmic_merged_df['ID'])

# Merge DP values into unique_mcrs_df
# Step 1: Remove rows with NaN values in 'ID' column
varscan_cosmic_merged_df_for_merging = varscan_cosmic_merged_df[['ID', 'DP_varscan']].dropna(subset=['ID']).rename(columns={'ID': 'mcrs_header'})

# Step 2: Drop duplicates from 'ID' column
varscan_cosmic_merged_df_for_merging = varscan_cosmic_merged_df_for_merging.drop_duplicates(subset=['mcrs_header'])

# Step 3: Left merge with unique_mcrs_df
unique_mcrs_df = pd.merge(
    unique_mcrs_df,               # Left DataFrame
    varscan_cosmic_merged_df_for_merging,         # Right DataFrame
    on='mcrs_header',
    how='left'
)

number_of_mutations_varscan = len(df_varscan.drop_duplicates(subset=['CHROM', 'POS', 'REF', 'ALT']))
number_of_cosmic_mutations_varscan = len(varscan_cosmic_merged_df.drop_duplicates(subset=['CHROM', 'POS', 'REF', 'ALT']))

# unique_mcrs_df['header_list'] each contains a list of strings. I would like to make a new column unique_mcrs_df['mutation_detected_varscan'] where each row is True if any value from the list unique_mcrs_df['mcrs_header'] is in the set id_set_mut  # keep in mind that my IDs are the mutation headers (ENST...), NOT mcrs headers or mcrs ids
unique_mcrs_df['mutation_detected_varscan'] = unique_mcrs_df['header_list'].apply(
    lambda header_list: any(header in id_set_varscan for header in header_list)
)

# calculate expression error
unique_mcrs_df['mutation_expression_prediction_error'] = unique_mcrs_df['DP_varscan'] - unique_mcrs_df['number_of_reads_mutant']  # positive means overpredicted, negative means underpredicted

unique_mcrs_df['TP'] = (unique_mcrs_df['included_in_synthetic_reads_mutant'] & unique_mcrs_df['mutation_detected_varscan'])
unique_mcrs_df['FP'] = (~unique_mcrs_df['included_in_synthetic_reads_mutant'] & unique_mcrs_df['mutation_detected_varscan'])
unique_mcrs_df['FN'] = (unique_mcrs_df['included_in_synthetic_reads_mutant'] & ~unique_mcrs_df['mutation_detected_varscan'])
unique_mcrs_df['TN'] = (~unique_mcrs_df['included_in_synthetic_reads_mutant'] & ~unique_mcrs_df['mutation_detected_varscan'])

varscan_stat_path = f"{varscan_benchmarking_output}/reference_metrics_varscan.txt"
metric_dictionary_reference = calculate_metrics(unique_mcrs_df, header_name = "mcrs_header", check_assertions = False, out = varscan_stat_path)
draw_confusion_matrix(metric_dictionary_reference)

true_set = set(unique_mcrs_df.loc[unique_mcrs_df['included_in_synthetic_reads_mutant'], 'mcrs_header'])
positive_set = set(unique_mcrs_df.loc[unique_mcrs_df['mutation_detected_varscan'], 'mcrs_header'])
create_venn_diagram(true_set, positive_set, TN = metric_dictionary_reference['TN'], out_path = f"{varscan_benchmarking_output}/venn_diagram_reference_cosmic_only_varscan.png")

noncosmic_mutation_id_set = {f'varscan_fp_{i}' for i in range(1, number_of_mutations_varscan - number_of_cosmic_mutations_varscan + 1)}

positive_set_including_noncosmic_mutations = positive_set.union(noncosmic_mutation_id_set)
false_positive_set = set(unique_mcrs_df.loc[unique_mcrs_df['FP'], 'mcrs_header'])
false_positive_set_including_noncosmic_mutations = false_positive_set.union(noncosmic_mutation_id_set)

FP_including_noncosmic = len(false_positive_set_including_noncosmic_mutations)
accuracy, sensitivity, specificity = calculate_sensitivity_specificity(metric_dictionary_reference['TP'], metric_dictionary_reference['TN'], FP_including_noncosmic, metric_dictionary_reference['FN'])

with open(varscan_stat_path, "a") as file:
    file.write(f"FP including non-cosmic: {FP_including_noncosmic}\n")
    file.write(f"accuracy including non-cosmic: {accuracy}\n")
    file.write(f"specificity including non-cosmic: {specificity}\n")



create_venn_diagram(true_set, positive_set_including_noncosmic_mutations, TN = metric_dictionary_reference['TN'], out_path = f"{varscan_benchmarking_output}/venn_diagram_reference_including_noncosmics_varscan.png")

unique_mcrs_df.rename(columns={'TP': 'TP_varscan', 'FP': 'FP_varscan', 'TN': 'TN_varscan', 'FN': 'FN_varscan', 'mutation_expression_prediction_error': 'mutation_expression_prediction_error_varscan'}, inplace=True)

unique_mcrs_df_out = unique_mcrs_df_path.replace(".csv", "_with_gatk.csv")
unique_mcrs_df.to_csv(unique_mcrs_df_out, index=False)