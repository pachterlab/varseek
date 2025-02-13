🧹
Input table
| Parameter                                                           | File type                               | Required?           | Default Path                                                                  | Description             |
|----------------------------------------------------------------|--------------------------------------|------------------------|-------------------------------------------------------------------------|--------------------------|
| adata_vcrs                                                           | .h5ad                                   | True                    | N/A                                                                                | ...                           |
| adata_reference_genome                                    | .h5ad                                   | False                  | N/A                                                                                | ...                           |
| fastqs                                                                   | .fastq or List[.fastq] or .txt   | False                   | N/A                                                                                | ...                           |
| vk_ref_dir                                                             | directory                              | False                   | N/A                                                                               | ...                           |
| vcrs_index                                                           | .idx                                       | False                   | <vk_ref_dir>/mcrs_index.idx                                       | ...                           |
| vcrs_t2g                                                               | .txt                                        | False                   | <vk_ref_dir>/mcrs_t2g_filtered.txt                              | ...                           |
| mcrs_fasta                                                           | .fa                                         | False                   | <vk_ref_dir>/mcrs_filtered.fa                                      | ...                           |
| dlist_fasta                                                            | .fa                                        | False                   | <vk_ref_dir>/dlist_filtered.fa                                         | ...                           |
| kb_count_vcrs_dir                                               | directory                              | False                   | N/A                                                                               | ...                           |
| kb_count_reference_genome_dir                        | directory                              | False                   | N/A                                                                               | ...                           |


Output table
| Parameter                                                           | File type         | Flag                                                                           | Default Path                                                            | Description           |
|----------------------------------------------------------------|--------------------|---------------------------------------------------------------------|-------------------------------------------------------------------|-------------------------|
| out                                                                       | directory         | N/A                                                                            | "."                                                                           | ...                          |
| adata_vcrs_clean_out                  	                      | .h5ad             | N/A                                                                             | <out>/adata_cleaned.h5ad									   | ...                          |
| adata_reference_genome_clean_out                 | .h5ad             | adata_reference_genome or kb_count_reference_genome_dir must be provided             | <out>/adata_reference_genome_cleaned.h5ad	   | ...                          |
| vcf_out                  	                                              | .vcf                 | save_vcf                                                                    | <out>/vcrs.vcf												            | ...                          |

