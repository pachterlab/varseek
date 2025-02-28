import os
import subprocess
import varseek as vk

dry_run=True  #! change to False
reference_out_dir = "/mnt/gpussd2/jrich/varseek/data/reference"
out = "/mnt/gpussd2/jrich/varseek/data/vk_ref"
w=47
k=51
filters=(
    "alignment_to_reference:is_not_true",
    "pseudoaligned_to_reference_despite_not_truly_aligning:is_not_true",  # filter out variants that pseudoaligned to human genome despite not truly aligning
    "num_distinct_triplets:greater_than=2",  # filters out VCRSs with <= 2 unique triplets
)

# subprocess.run(f"cp -r /mnt/gpussd2/jrich/varseek/data/vk_build_out {out}", shell=True, check=True)
# subprocess.run(f"rm -rf /mnt/gpussd2/jrich/varseek/data/vk_build_out_basic/variants_updated.csv", shell=True, check=True)

vk.ref(
    variants="cosmic_cmc",
    sequences="cdna",
    w=w,
    k=k,
    filters=filters,
    out=out,
    reference_out_dir=reference_out_dir,
    gtf=True,  # just so that gtf information gets merged into cosmic df
    save_logs=True,
    verbose=True,
    dry_run=dry_run,
    save_variants_updated_csv=True,
    save_variants_updated_exploded_vk_info_csv=True,
    dlist_reference_source="grch37",
    dlist_reference_ensembl_release=93,
    # columns_to_include="all",
    kallisto="/home/jrich/enter/envs/varseek/lib/python3.10/site-packages/kb_python/bins/linux/kallisto/kallisto_k64",
    bustools="/home/jrich/enter/envs/varseek/lib/python3.10/site-packages/kb_python/bins/linux/bustools/bustools",
    bowtie2_path="/mnt/gpussd2/jrich/opt/bowtie2-2.5.4-linux-x86_64"
)