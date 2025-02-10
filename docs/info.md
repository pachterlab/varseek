📝
Input table
| Parameter                                                           | File type         | Required?           | Corresponding parameter from vk build  | Description             |
|----------------------------------------------------------------|--------------------|------------------------|------------------------------------------------------|---------------------------|
| input_dir                                                              | directory         | True                    | out                                                           | ...                             |
| mcrs_fasta                                                           | .fa                  | False                  | mcrs_fasta_out                                       | ...                             |
| id_to_header_csv                                                | .csv                | False                  | id_to_header_csv_out                            | ...                             |
| mutations_updated_csv                                      | .csv                | False                  | mutations_updated_csv_out                   | ...                             |
| gtf                                                                        | .csv                | False                  | gtf                                                            | ...                             |
| dlist_reference_genome_fasta                            | .csv                | False                  | N/A                                                          | ...                             |
| dlist_reference_cdna_fasta                                 | .csv                | False                  | N/A                                                          | ...                             |
| dlist_reference_gtf                                               | .csv                | False                  | N/A                                                          | ...                             |







Output table
| Parameter                                                           | File type         | Flag                                                                           | Default Path                                                                                                     | Description           |
|----------------------------------------------------------------|--------------------|---------------------------------------------------------------------|------------------------------------------------------------------------------------------------------|-------------------------|
| out                                                                       | directory         | N/A                                                                            | <input_dir>                                                                                                       | ...                          |
| reference_out_dir                                                | directory         | N/A                                                                            | <out>                                                                                                                | ...                          |
| mutations_updated_vk_info_csv_out                  | .csv                | N/A                                                                            | <out>/mutation_metadata_df_updated_vk_info.csv                                        | ...                          |
| mutations_updated_exploded_vk_info_csv_out  | .csv                | save_mutations_updated_exploded_vk_info_csv   | <out>/mutation_metadata_df_updated_vk_info_exploded.csv                        | ...                          |
| dlist_genome_fasta_out                                       | .fa                  | N/A                                                                           | <out>/dlist_genome.fa                                                                                     | ...                          |
| dlist_cdna_fasta_out                                            | .fa                  | N/A                                                                            | <out>/dlist_cdna.fa                                                                                          | ...                          |
| dlist_combined_fasta_out                                     | .fa                  | N/A                                                                            | <out>/dlist.fa                                                                                                   | ...                          |
