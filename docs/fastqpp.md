Input table
| Parameter                                                           | File type                               | Required?           | Description             |
|----------------------------------------------------------------|--------------------------------------|------------------------|---------------------------|
| fastqs                                                                   | .fastq or List[.fastq] or .txt   | True                    | ...                             |


Output table
| Parameter                                                           | File type         | Flag                                                                           | Default Path                                                                                                     | Description           |
|----------------------------------------------------------------|--------------------|---------------------------------------------------------------------|------------------------------------------------------------------------------------------------------|-------------------------|
| out                                                                       | directory         | N/A                                                                            | "."                                                                                                                     | ...                          |
| quality_control_fastqs_out_suffix                  	  | .fastq             | quality_control_fastqs                                                | <out>/<filename>_qc.<ext>												                             | ...                          |
| replace_low_quality_bases_with_N_out_suffix   | .fastq             | replace_low_quality_bases_with_N                          | <out>/<filename>_addedNs.<ext>												                     | ...                          |
| split_by_N_out_suffix                  	                      | .fastq             | split_reads_by_Ns                                                     | <out>/<filename>_splitNs.<ext>												                         | ...                          |
| concatenate_paired_fastqs_out_suffix                | .fastq             | concatenate_paired_fastqs                                       | <out>/<filename>_concatenated.<ext>								                             | ...                          |
