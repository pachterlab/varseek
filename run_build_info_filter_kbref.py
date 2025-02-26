import subprocess
import varseek as vk

dry_run = True  #! change to False
reference_out_dir = "/mnt/gpussd2/jrich/varseek/data/reference"
out = "/mnt/gpussd2/jrich/varseek/data/vk_build_info_filter_kbref"
w=47
k=51
filters=(
    "alignment_to_reference:is_not_true",
    "pseudoaligned_to_reference_despite_not_truly_aligning:is_not_true",  # filter out variants that pseudoaligned to human genome despite not truly aligning
    "num_distinct_triplets:greater_than=2",  # filters out VCRSs with <= 2 unique triplets
)

# subprocess.run(f"cp -r /mnt/gpussd2/jrich/varseek/data/vk_build_out {out}", shell=True, check=True)
# subprocess.run(f"rm -rf /mnt/gpussd2/jrich/varseek/data/vk_build_out_basic/variants_updated.csv", shell=True, check=True)

vk.build(
    variants="cosmic_cmc",
    sequences="cdna",
    w=w,
    k=k,
    out=out,
    reference_out_dir=reference_out_dir,
    gtf=True,  # just so that gtf information gets merged into cosmic df
    save_logs=True,
    verbose=True,
    save_variants_updated_csv=True,
    cosmic_email="jmrich@caltech.edu",
    cosmic_password="bopdit-xybRog-bycqu1",
    dry_run=dry_run,
)

vk.info(
    input_dir=out,
    k=k,
    out=out,
    reference_out_dir=reference_out_dir,
    save_logs=True,
    verbose=True,
    gtf="/mnt/gpussd2/jrich/varseek/data/reference/ensembl_grch37_release93/Homo_sapiens.GRCh37.87.gtf",
    save_variants_updated_exploded_vk_info_csv=True,
    dlist_reference_source="grch37",
    dlist_reference_ensembl_release=93,
    # columns_to_include="all",
    kallisto="/home/jrich/enter/envs/varseek/lib/python3.10/site-packages/kb_python/bins/linux/kallisto/kallisto_k64",
    bustools="/home/jrich/enter/envs/varseek/lib/python3.10/site-packages/kb_python/bins/linux/bustools/bustools",
    bowtie2_path="/mnt/gpussd2/jrich/opt/bowtie2-2.5.4-linux-x86_64",
    gene_name_column="gene_name",
    var_cdna_column="mutation_cdna",
    seq_id_cdna_column="seq_ID",
    var_genome_column="mutation_genome",
    seq_id_genome_column="chromosome",
    reference_cdna_fasta="/mnt/gpussd2/jrich/varseek/data/reference/ensembl_grch37_release93/Homo_sapiens.GRCh37.cdna.all.fa",
    reference_genome_fasta="/mnt/gpussd2/jrich/varseek/data/reference/ensembl_grch37_release93/Homo_sapiens.GRCh37.dna.primary_assembly.fa",
    variants="/mnt/gpussd2/jrich/varseek/data/reference/cosmic/CancerMutationCensus_AllData_Tsv_v101_GRCh37/CancerMutationCensus_AllData_v101_GRCh37_mutation_workflow.csv",
    w=w,
    dry_run=dry_run,
)

vk.filter(
    input_dir=out,
    filters=filters,
    reference_out_dir=reference_out_dir,
    save_logs=True,
    dry_run=dry_run,
)

kb_ref_command = [
    "kb",
    "ref",
    "--workflow",
    "custom",
    "-t",
    "2",
    "-i",
    f"{out}/index.idx",
    "--d-list",
    "None",
    "-k",
    str(k),
    "--overwrite",  # set overwrite here regardless of the overwrite argument because I would only even enter this block if kb count was only partially run (as seen by the lack of existing of file_signifying_successful_kb_ref_completion), in which case I should overwrite anyways
]

kb_ref_command.append(f"{out}/vcrs.fa")

if dry_run:
    print(" ".join(kb_ref_command))
else:
    subprocess.run(kb_ref_command, check=True)
