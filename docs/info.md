📝
Input table
| Parameter                                                           | File type         | Required?           | Corresponding parameter from vk build  | Description             |
|----------------------------------------------------------------|--------------------|------------------------|------------------------------------------------------|---------------------------|
| input_dir                                                              | directory         | True                    | out                                                           | ...                             |
| vcrs_fasta                                                           | .fa                  | False                  | mcrs_fasta_out                                       | ...                             |
| id_to_header_csv                                                | .csv                | False                  | id_to_header_csv_out                            | ...                             |
| variants_updated_csv                                      | .csv                | False                  | mutations_updated_csv_out                   | ...                             |
| gtf                                                                        | .csv                | False                  | gtf                                                            | ...                             |
| dlist_reference_genome_fasta                            | .csv                | False                  | N/A                                                          | ...                             |
| dlist_reference_cdna_fasta                                 | .csv                | False                  | N/A                                                          | ...                             |
| dlist_reference_gtf                                               | .csv                | False                  | N/A                                                          | ...                             |







Output table
| Parameter                                                           | File type         | Flag                                                                           | Default Path                                                                                                     | Description           |
|----------------------------------------------------------------|--------------------|---------------------------------------------------------------------|------------------------------------------------------------------------------------------------------|-------------------------|
| out                                                                       | directory         | N/A                                                                            | <input_dir>                                                                                                       | ...                          |
| reference_out_dir                                                | directory         | N/A                                                                            | <out>                                                                                                                | ...                          |
| variants_updated_vk_info_csv_out                  | .csv                | N/A                                                                            | <out>/variants_updated_vk_info.csv                                        | ...                          |
| variants_updated_exploded_vk_info_csv_out  | .csv                | save_variants_updated_exploded_vk_info_csv   | <out>/variants_updated_exploded_vk_info.csv                        | ...                          |
| dlist_genome_fasta_out                                       | .fa                  | N/A                                                                           | <out>/dlist_genome.fa                                                                                     | ...                          |
| dlist_cdna_fasta_out                                            | .fa                  | N/A                                                                            | <out>/dlist_cdna.fa                                                                                          | ...                          |
| dlist_combined_fasta_out                                     | .fa                  | N/A                                                                            | <out>/dlist.fa                                                                                                   | ...                          |
