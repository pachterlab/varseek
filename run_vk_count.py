import os
import varseek as vk

data_dir = "/Users/joeyrich/Desktop/local/varseek/data"
reference_out_dir = os.path.join(data_dir, "reference")

fastqs = ["/Users/joeyrich/Desktop/local/varseek/data/ccle_data_base/RNASeq_MELHO_SKIN/SRR8615233_1_first_10mil.fastq"]  # SRR8615233_1_first_10mil
technology = "BULK"  # 10XV3
mm = True
union = True
qc_against_gene_matrix = False  #!!! change to True
parity = "single"
k = 51
out = "/Users/joeyrich/Desktop/local/varseek/data/ccle_data_base/vk_count_out_test_April8th"
threads = 4

variants = None if not qc_against_gene_matrix else os.path.join(reference_out_dir, "cosmic", "CancerMutationCensus_AllData_Tsv_v101_GRCh37", "CancerMutationCensus_AllData_v101_GRCh37_mutation_workflow.csv")
seq_id_column = "seq_ID"
var_column = "mutation_cdna"
gene_id_column = "gene_name"
variants_usecols = None  # all columns

vcrs_index = "/Users/joeyrich/Desktop/local/varseek/data/vk_ref_out/vcrs_index.idx"
vcrs_t2g = "/Users/joeyrich/Desktop/local/varseek/data/vk_ref_out/vcrs_t2g_filtered.txt"

vk.count(
    fastqs=fastqs,
    index=vcrs_index,
    t2g=vcrs_t2g,
    technology=technology,
    mm=mm,
    union=union,
    qc_against_gene_matrix=qc_against_gene_matrix,
    parity=parity,
    k=k,
    out=out,
    threads=threads,
    variants=variants,
    seq_id_column=seq_id_column,
    var_column=var_column,
    gene_id_column=gene_id_column,
    variants_usecols=variants_usecols,
    overwrite=False
)