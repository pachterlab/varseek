import subprocess
input_dir = "/Users/joeyrich/Desktop/local/varseek/data/vk_info_test3"
subprocess.run(f"cp -r /Users/joeyrich/Desktop/local/varseek/data/vk_build_final {input_dir}", shell=True, check=True)

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
    # gene_name_column="gene_name",
)