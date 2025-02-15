Input table
| Parameter                                                           | File type         | Required?           | Description             |
|----------------------------------------------------------------|--------------------|------------------------|---------------------------|
| variants                                                               | .csv or string   | True                    | ...                             |
| sequences                                                           | .fa/.fasta         | True                    | ...                             |


Output table
| Parameter                                                           | File type         | Flag                                                                           | Default Path                                                                                                     | Description           |
|----------------------------------------------------------------|--------------------|---------------------------------------------------------------------|------------------------------------------------------------------------------------------------------|-------------------------|
| out                                                                       | directory         | N/A                                                                            | "."                                                                                                                     | ...                          |
| index_out                  											  | .idx                | N/A                                                                            | <out>/vcrs_index.idx												                                         | ...                          |
| t2g_out         													      | .txt                | save_variants_updated_exploded_filtered_csv    | <out>/vcrs_t2g.txt                       																		 | ...                          |

