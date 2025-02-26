import subprocess
input_dir = "/Users/joeyrich/Desktop/local/varseek/data/vk_build_final"
filters=(
    "alignment_to_reference:is_not_true",
    "pseudoaligned_to_reference_despite_not_truly_aligning:is_not_true",  # filter out variants that pseudoaligned to human genome despite not truly aligning
    "num_distinct_triplets:greater_than=2",  # filters out VCRSs with <= 2 unique triplets
)

import varseek as vk
vk.filter(
    input_dir=input_dir,
    filters=filters,
    reference_out_dir="/Users/joeyrich/Desktop/local/varseek/data/reference",
    save_logs=True,
)