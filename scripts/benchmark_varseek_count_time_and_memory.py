import os
import random
import shutil
import gget
import subprocess
import time
import logging

import numpy as np
import pandas as pd
import pysam

import varseek as vk
from varseek.utils import (
    is_program_installed,
    report_time_and_memory_of_script,
    run_command_with_error_logging,
    convert_mutation_cds_locations_to_cdna
)

logger = logging.getLogger(__name__)
logger.setLevel("INFO")
formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s", "%H:%M:%S")
console_handler = logging.StreamHandler()
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)


script_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(os.path.dirname(script_dir), "data")
reference_out_dir = os.path.join(data_dir, "reference")

### ARGUMENTS ###
number_of_reads_list = [1, 4, 16, 64, 256, 1024]  # number of reads, in millions  # for debugging: [0.001, 0.002]
dry_run = False  # only applies to the variant calling steps, not to the preparation (ie data downloads, etc)

read_length = 150
k = 51
w = 47
strand = None  # None for strand-agnostic (randomly-selected), "f" for forward, "r" for reverse, "both" for both - make sure this matches the reference genome (vk build command) - strand = True -> "f" or "r" here; strand = False -> None or "both" here - note that the strand is randomly selected per *transcript*, such that all drawn reads will come from the same strand no matter what
add_noise_sequencing_error = True
add_noise_base_quality = False
error_rate = 0.0001  # only if add_noise_sequencing_error=True
error_distribution = (0.85, 0.1, 0.05)  # sub, del, ins  # only if add_noise_sequencing_error=True
max_errors = float("inf")  # only if add_noise_sequencing_error=True
seq_id_column = "seq_ID"
var_column = "mutation_cdna"
threads = 16
random_seed = 42
qc_against_gene_matrix = False

# varseek ref parameters
vk_ref_out = os.path.join(data_dir, "vk_ref_out")
vk_ref_index_path = os.path.join(vk_ref_out, "vcrs_index.idx")  # for vk count
vk_ref_t2g_path = os.path.join(vk_ref_out, "vcrs_t2g_filtered.txt")  # for vk count
dlist_reference_source = "t2t"

# normal reference genome
reference_genome_index_path = os.path.join(reference_out_dir, "ensembl_grch37_release93", "index.idx")  # can either already exist or will be created; only used if qc_against_gene_matrix=True
reference_genome_t2g_path = os.path.join(reference_out_dir, "ensembl_grch37_release93", "t2g.txt")  # can either already exist or will be created; only used if qc_against_gene_matrix=True

cosmic_mutations_path = os.path.join(reference_out_dir, "cosmic", "CancerMutationCensus_AllData_Tsv_v101_GRCh37", "CancerMutationCensus_AllData_v101_GRCh37_mutation_workflow.csv")  # for vk sim
reference_cdna_path = os.path.join(reference_out_dir, "ensembl_grch37_release93", "Homo_sapiens.GRCh37.cdna.all.fa")  # for vk sim
reference_cds_path = os.path.join(reference_out_dir, "ensembl_grch37_release93", "Homo_sapiens.GRCh37.cds.all.fa")  # for vk sim
reference_genome_fasta = os.path.join(reference_out_dir, "ensembl_grch37_release93", "Homo_sapiens.GRCh37.dna.primary_assembly.fa")  # can either already exist or will be downloaded
reference_genome_gtf = os.path.join(reference_out_dir, "ensembl_grch37_release93", "Homo_sapiens.GRCh37.87.gtf")  # can either already exist or will be downloaded

seqtk = "seqtk"

output_dir = os.path.join(data_dir, "time_and_memory_benchmarking_out_dir")  #* change for each run
tmp_dir = "/data/benchmarking_tmp"  #!! replace with "tmp"
overwrite = False
### ARGUMENTS ###

# # set random seeds
# random.seed(random_seed)
# np.random.seed(random_seed)

if overwrite:
    for out_directory in [output_dir, tmp_dir]:
        if os.path.exists(out_directory):
            shutil.rmtree(out_directory)

# os.makedirs(output_dir)  # purposely not using exist_ok=True to ensure that the directory is non-existent  #* comment out for debugging to keep output_dir between runs
# os.makedirs(tmp_dir)  # purposely not using exist_ok=True to ensure that the directory is non-existent  #* comment out for debugging to keep tmp_dir between runs

if strand is None or strand == "both":
    kb_count_strand = "unstranded"
elif strand == "f":
    kb_count_strand = "forward"
elif strand == "r":
    kb_count_strand = "reverse"

vk_count_script_path = os.path.join(script_dir, "run_varseek_count_for_benchmarking.py")

# create synthetic reads
if k and w:
    if k <= w:
        raise ValueError("k must be greater than w")
    read_w = read_length - (k - w)  # note that this does not affect read length, just read *parent* length
else:
    read_w = read_length - 1

#* download COSMIC and sequences for vk sim if not already downloaded
# download cosmic and cdna
if not os.path.exists(reference_cdna_path):
    logger.info("Downloading cDNA")
    reference_cdna_dir = os.path.dirname(reference_cdna_path) if os.path.dirname(reference_cdna_path) else "."
    gget_ref_command = ["gget", "ref", "-w", "cdna", "-r", "93", "--out_dir", reference_cdna_dir, "-d", "human_grch37"]
    subprocess.run(gget_ref_command, check=True)
    subprocess.run(["gunzip", f"{reference_cdna_path}.gz"], check=True)
if not os.path.exists(reference_cds_path):
    logger.info("Downloading CDS")
    reference_cds_dir = os.path.dirname(reference_cds_path) if os.path.dirname(reference_cds_path) else "."
    gget_ref_command = ["gget", "ref", "-w", "cds", "-r", "93", "--out_dir", reference_cds_dir, "-d", "human_grch37"]
    subprocess.run(gget_ref_command, check=True)
    subprocess.run(["gunzip", f"{reference_cds_path}.gz"], check=True)

if not os.path.exists(cosmic_mutations_path):
    logger.info("Downloading COSMIC")
    reference_out_dir_cosmic = os.path.dirname(os.path.dirname(cosmic_mutations_path))
    gget.cosmic(
        None,
        grch_version=37,
        cosmic_version=101,
        out=reference_out_dir_cosmic,
        cosmic_project="cancer",
        download_cosmic=True,
        gget_mutate=True
    )

with open(cosmic_mutations_path) as f:
    number_of_cosmic_mutations = sum(1 for _ in f) - 1  # adjust for headers

cosmic_mutations = pd.read_csv(cosmic_mutations_path, nrows=2)
cosmic_mutations_path_original = cosmic_mutations_path.replace(".csv", "_original.csv")
if not os.path.exists(cosmic_mutations_path_original):
    shutil.copy(cosmic_mutations_path, cosmic_mutations_path_original)

if "mutation_cdna" not in cosmic_mutations.columns:
    logger.info("Converting CDS to cDNA in COSMIC")
    _, _ = convert_mutation_cds_locations_to_cdna(input_csv_path=cosmic_mutations_path, output_csv_path=cosmic_mutations_path, cds_fasta_path=reference_cds_path, cdna_fasta_path=reference_cdna_path, verbose=True)

#* Make synthetic reads corresponding to the largest value in number_of_reads_list - if desired, I can replace this with real data
number_of_reads_max = int(max(number_of_reads_list) * 10**6)  # convert to millions

number_of_reads_per_variant_alt=100
number_of_reads_per_variant_ref=150
number_of_reads_per_variant_total = number_of_reads_per_variant_alt + number_of_reads_per_variant_ref

if number_of_reads_max > (number_of_cosmic_mutations * number_of_reads_per_variant_total):
    raise ValueError("Max reads is too large. Either increase number_of_reads_per_variant_alt and/or number_of_reads_per_variant_ref, or choose a larger variant database.")

#* Download varseek index
if not os.path.exists(vk_ref_index_path) or not os.path.exists(vk_ref_t2g_path):
    vk.ref(variants="cosmic_cmc", sequences="cdna", w=w, k=k, out=vk_ref_out, dlist_reference_source=dlist_reference_source, download=True, index_out=vk_ref_index_path, t2g_out=vk_ref_t2g_path)
    # alternatively, to build from scratch: subprocess.run([os.path.join(script_dir, "run_vk_ref.py")], check=True)

#* install seqtk if not installed
# if not is_program_installed(seqtk):
#     raise ValueError("seqtk is required to run this script. Please install seqtk and ensure that it is in your PATH.")
#     # subprocess.run("git clone https://github.com/lh3/seqtk.git", shell=True, check=True)
#     # subprocess.run("cd seqtk && make", shell=True, check=True)
#     # seqtk = os.path.join(script_dir, "seqtk/seqtk")

#* Build normal genome reference (for vk clean in vk count) when qc_against_gene_matrix=True
if qc_against_gene_matrix and (not os.path.exists(reference_genome_index_path) or not os.path.exists(reference_genome_t2g_path)):  # download reference if does not exist
    if not os.path.exists(reference_genome_fasta) or not os.path.exists(reference_genome_gtf):
        reference_genome_out_dir = os.path.dirname(reference_genome_fasta) if os.path.dirname(reference_genome_fasta) else "."
        subprocess.run(["gget", "ref", "-w", "dna,gtf", "-r", "93", "--out_dir", reference_genome_out_dir, "-d", "human_grch37"], check=True)  # using grch37, ensembl 93 to agree with COSMIC
        subprocess.run(["gunzip", f"{reference_genome_fasta}.gz"], check=True)
        subprocess.run(["gunzip", f"{reference_genome_gtf}.gz"], check=True)
    reference_genome_f1 = os.path.join(reference_out_dir, "ensembl_grch37_release93", "f1.fasta")
    subprocess.run(["kb", "ref", "-t", str(threads), "-i", reference_genome_index_path, "-g", reference_genome_t2g_path, "-f1", reference_genome_f1, reference_genome_fasta, reference_genome_gtf], check=True)

output_file = os.path.join(output_dir, "time_and_memory_benchmarking_report.txt")
# file_mode = "a" if os.path.isfile(output_file) else "w"
# with open(output_file, file_mode, encoding="utf-8") as f:
#     f.write(f"Threads Argument: {threads}\n\n")

vk_sim_out_dir = os.path.join(tmp_dir, "vk_sim_out")

#* Run variant calling tools
for number_of_reads in number_of_reads_list:
    number_of_reads = int(number_of_reads * 10**6)  # convert to millions
    fastq_output_path = os.path.join(tmp_dir, f"reads_{number_of_reads}_fastq.fastq")
    if not os.path.isfile(fastq_output_path):
        # seqtk_sample_command = f"{seqtk} sample -s {random_seed} {fastq_output_path_max_reads} {number_of_reads} > {fastq_output_path}"
        # logger.info(f"Running seqtk sample for {number_of_reads} reads")
        # subprocess.run(seqtk_sample_command, shell=True, check=True)

        number_of_variants_to_sample = number_of_reads // number_of_reads_per_variant_total

        logger.info(f"Building synthetic reads for {number_of_reads} reads")
        _ = vk.sim(
            variants=cosmic_mutations_path,
            reads_fastq_out=fastq_output_path,
            number_of_variants_to_sample=number_of_variants_to_sample,
            strand=strand,
            number_of_reads_per_variant_alt=number_of_reads_per_variant_alt,
            number_of_reads_per_variant_ref=number_of_reads_per_variant_ref,
            read_length=read_length,
            seed=random_seed,
            add_noise_sequencing_error=add_noise_sequencing_error,
            add_noise_base_quality=add_noise_base_quality,
            error_rate=error_rate,
            error_distribution=error_distribution,
            max_errors=max_errors,
            with_replacement=True,
            gzip_reads_fastq_out=False,
            sequences=reference_cdna_path,
            seq_id_column=seq_id_column,
            var_column=var_column,
            variant_type_column=None,
            reference_out_dir=reference_out_dir,
            out=vk_sim_out_dir,
            k=k,
            w=w,
            make_dataframes=False
        )

    kb_count_reference_genome_out_dir = os.path.join(tmp_dir, f"kb_count_reference_genome_out_dir_{number_of_reads}")
    # # commented out because I now include this within vk count if I do it at all
    # if not os.path.exists(kb_count_reference_genome_out_dir):
    #     # kb count, reference genome
    #     kb_count_standard_index_command = [
    #         "kb",
    #         "count",
    #         "-t",
    #         str(threads),
    #         "-i",
    #         reference_genome_index_path,
    #         "-g",
    #         reference_genome_t2g_path,
    #         "-x",
    #         "bulk",
    #         "--h5ad",
    #         "--parity",
    #         "single",
    #         "--strand",
    #         kb_count_strand,
    #         "-o",
    #         kb_count_reference_genome_out_dir,
    #         fastq_output_path
    #     ]

    #     logger.info(f"kb count, reference genome, {number_of_reads} reads")
    #     script_title = f"kb count reference genome pseudoalignment {number_of_reads} reads {threads} threads"
    #     _ = report_time_and_memory_of_script(subprocess_script_path, output_file = kb_reference_output_file, argparse_flags = f'"{str(kb_count_standard_index_command)}"', script_title = script_title)
    #     # subprocess.run(kb_count_standard_index_command, check=True)
            
    #* Variant calling: varseek
    logger.info(f"varseek, {number_of_reads} reads")
    script_title = f"varseek {number_of_reads} reads {threads} threads"
    vk_count_out_tmp = os.path.join(tmp_dir, f"vk_count_{number_of_reads}_reads")
    argparse_flags = f"--index {vk_ref_index_path} --t2g {vk_ref_t2g_path} --technology bulk --threads {threads} -k {k} --out {vk_count_out_tmp} --kb_count_reference_genome_out_dir {kb_count_reference_genome_out_dir} --reference_genome_index {reference_genome_index_path} --reference_genome_t2g {reference_genome_t2g_path} --disable_clean --disable_summarize --fastqs {fastq_output_path}"
    print(f"python3 {vk_count_script_path} {argparse_flags}")
    if not dry_run:
        _ = report_time_and_memory_of_script(vk_count_script_path, output_file = output_file, argparse_flags = argparse_flags, script_title = script_title)

# delete tmp directory
# os.system(f"rm -rf {tmp_dir}")  #!!! uncomment later to delete tmp directory
