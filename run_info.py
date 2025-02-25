import subprocess
input_dir = "/Users/joeyrich/Desktop/local/varseek/data/vk_info_test13"
subprocess.run(f"cp -r /Users/joeyrich/Desktop/local/varseek/data/vk_build_final {input_dir}", shell=True, check=True)
subprocess.run("head -n 20000 /Users/joeyrich/Desktop/local/varseek/data/vk_info_test13/vcrs.fa > /Users/joeyrich/Desktop/local/varseek/data/vk_info_test13/vcrs_head.fa", shell=True, check=True)

import varseek as vk
vk.info(
    input_dir=input_dir,
    reference_out_dir="/Users/joeyrich/Desktop/local/varseek/data/reference",
    w=54,
    k=59,
    dlist_reference_source="grch37",
    dlist_reference_ensembl_release=93,
    save_logs=True,
    verbose=True,
    vcrs_fasta='/Users/joeyrich/Desktop/local/varseek/data/vk_info_test13/vcrs_head.fa'
    # columns_to_include=[],
    # gene_name_column="gene_name",
)