import time
import subprocess
from varseek.utils import adjust_variant_adata_by_normal_gene_matrix

fastqs = ["/Users/joeyrich/Desktop/local/varseek/data/vk_count_out/fastqs_quality_controlled/pbmc_1k_v3_S1_L001_R1_001.fastq.gz", "/Users/joeyrich/Desktop/local/varseek/data/vk_count_out/fastqs_quality_controlled/pbmc_1k_v3_S1_L001_R2_001.fastq.gz", "/Users/joeyrich/Desktop/local/varseek/data/vk_count_out/fastqs_quality_controlled/pbmc_1k_v3_S1_L002_R1_001.fastq.gz", "/Users/joeyrich/Desktop/local/varseek/data/vk_count_out/fastqs_quality_controlled/pbmc_1k_v3_S1_L002_R2_001.fastq.gz"]
normal_t2g = "/Users/joeyrich/Desktop/local/varseek/data/reference/ensembl_grch37_release93/kb_ref_out_standard_workflow_k31/t2g.txt"
kb_count_normal_genome_command = f"kb count -t 2 -k 31 -i /Users/joeyrich/Desktop/local/varseek/data/reference/ensembl_grch37_release93/kb_ref_out_standard_workflow_k31/index.idx -g {normal_t2g} -x 10XV3 --num --h5ad -o kb_count_out /Users/joeyrich/Desktop/local/varseek/data/vk_count_out_normal {fastqs}"

print("Running kb count on normal genome")
subprocess.run(kb_count_normal_genome_command, shell=True, executable="/bin/bash", check=True)

print("Running adjust_variant_adata_by_normal_gene_matrix")
# record time
start = time.time()
adjust_variant_adata_by_normal_gene_matrix(adata="/Users/joeyrich/Desktop/local/varseek/data/vk_count_out/adata_cleaned.h5ad", kb_count_vcrs_dir="/Users/joeyrich/Desktop/local/varseek/data/vk_count_out_vcrs", kb_count_reference_genome_dir="/Users/joeyrich/Desktop/local/varseek/data/vk_count_out_normal", fastq_file_list=fastqs, technology="10XV3", t2g_standard=normal_t2g, adata_output_path=None, mm=False, parity=None, bustools="/Users/joeyrich/miniconda3/envs/varseek/lib/python3.10/site-packages/kb_python/bins/darwin/m1/bustools/bustools", check_only=True, save_type="parquet", count_reads_that_dont_pseudoalign_to_reference_genome=True)
end = time.time()
minutes, seconds = divmod(end - start, 60)
print(f"Time taken: {minutes} minutes and {seconds} seconds")