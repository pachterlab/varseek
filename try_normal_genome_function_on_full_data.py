import time
import os
import subprocess
from varseek.utils import adjust_variant_adata_by_normal_gene_matrix

# fastqs = ["/Users/joeyrich/Desktop/local/varseek/data/vk_count_out/fastqs_quality_controlled/pbmc_1k_v3_S1_L001_R1_001.fastq.gz", "/Users/joeyrich/Desktop/local/varseek/data/vk_count_out/fastqs_quality_controlled/pbmc_1k_v3_S1_L001_R2_001.fastq.gz", "/Users/joeyrich/Desktop/local/varseek/data/vk_count_out/fastqs_quality_controlled/pbmc_1k_v3_S1_L002_R1_001.fastq.gz", "/Users/joeyrich/Desktop/local/varseek/data/vk_count_out/fastqs_quality_controlled/pbmc_1k_v3_S1_L002_R2_001.fastq.gz"]
fastqs = ["/Users/joeyrich/Desktop/local/varseek/data/ccle_data_base/RNASeq_MELHO_SKIN/SRR8615233_1.fastq.gz"]  # SRR8615233_1_first_10mil
technology = "BULK"  # 10XV3

vcrs_index = "/Users/joeyrich/Desktop/local/varseek/data/vk_ref_out/vcrs_index.idx"
vcrs_t2g = "/Users/joeyrich/Desktop/local/varseek/data/vk_ref_out/vcrs_t2g_filtered.txt"
kb_count_out_vcrs = "/Users/joeyrich/Desktop/local/varseek/data/ccle_data_base/RNASeq_MELHO_SKIN/kb_count_out_vcrs"  # kb_count_out_vcrs_first_10mil
vcrs_parity = "single"

normal_index = "/Users/joeyrich/Desktop/local/varseek/data/reference/ensembl_grch37_release93/kb_ref_out_standard_workflow_k31/index.idx"
normal_t2g = "/Users/joeyrich/Desktop/local/varseek/data/reference/ensembl_grch37_release93/kb_ref_out_standard_workflow_k31/t2g.txt"
kb_count_out_normal = "/Users/joeyrich/Desktop/local/varseek/data/ccle_data_base/RNASeq_MELHO_SKIN/kb_count_out_normal"  # kb_count_out_normal_first_10mil
parity = "single"

#* VCRS
kb_count_vcrs_command = f"kb count -t 2 -k 51 --mm --union -i {vcrs_index} -g {vcrs_t2g} -x {technology} --num --h5ad -o {kb_count_out_vcrs} "
if technology in {"BULK", "SMARTSEQ2"}:
    kb_count_vcrs_command += f"--parity {vcrs_parity} "
kb_count_vcrs_command += " ".join(fastqs)

if not os.path.exists(kb_count_out_vcrs):
    print("VCRS output file does not exist, running kb count...")
    subprocess.run(kb_count_vcrs_command, shell=True, executable="/bin/bash", check=True)

#* Normal
kb_count_normal_genome_command = f"kb count -t 2 -k 31 -i {normal_index} -g {normal_t2g} -x {technology} --num --h5ad -o {kb_count_out_normal} "
if technology in {"BULK", "SMARTSEQ2"}:
    kb_count_normal_genome_command += f"--parity {parity} "
kb_count_normal_genome_command += " ".join(fastqs)

if not os.path.exists(kb_count_out_normal):
    print("Normal genome output file does not exist, running kb count...")
    subprocess.run(kb_count_normal_genome_command, shell=True, executable="/bin/bash", check=True)

#* My function
print("Running adjust_variant_adata_by_normal_gene_matrix")
start = time.time()
adata_adjusted_for_normal_genome = adjust_variant_adata_by_normal_gene_matrix(kb_count_vcrs_dir=kb_count_out_vcrs, kb_count_reference_genome_dir=kb_count_out_normal, fastq_file_list=fastqs, technology=technology, t2g_standard=normal_t2g, adata_output_path=None, mm=False, parity=parity, bustools="/Users/joeyrich/miniconda3/envs/varseek/lib/python3.10/site-packages/kb_python/bins/darwin/m1/bustools/bustools", fastq_sorting_check_only=True, save_type="parquet", count_reads_that_dont_pseudoalign_to_reference_genome=True, variant_source="transcriptome")
adata_adjusted_for_normal_genome.write_h5ad(kb_count_out_vcrs + "/adata_adjusted_for_normal_genome.h5ad")
end = time.time()
minutes, seconds = divmod(end - start, 60)
print(f"Time taken: {minutes} minutes and {seconds} seconds")