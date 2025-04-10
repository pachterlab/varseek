import varseek as vk

fastqs = ["/Users/joeyrich/Desktop/local/varseek/data/ccle_data_base/RNASeq_MELHO_SKIN/SRR8615233_1_first_10mil.fastq"]  # SRR8615233_1_first_10mil
technology = "BULK"  # 10XV3
mm = True
union = True
qc_against_gene_matrix = True
parity = "single"
k = 51
out = "/Users/joeyrich/Desktop/local/varseek/data/ccle_data_base/vk_count_out_test_April8th"
threads = 4

vcrs_index = "/Users/joeyrich/Desktop/local/varseek/data/vk_ref_out/vcrs_index.idx"
vcrs_t2g = "/Users/joeyrich/Desktop/local/varseek/data/vk_ref_out/vcrs_t2g_filtered.txt"

vk.count(
    fastqs=fastqs,
    index=vcrs_index,
    t2g=vcrs_t2g,
    technology=technology,
    mm=mm,
    union=union,
    parity=parity,
    k=k,
    out=out,
    threads=threads,
    overwrite=True
)