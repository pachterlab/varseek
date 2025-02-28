Input table
| Parameter                                                           | File type                               | Required?           | Description             |
|----------------------------------------------------------------|--------------------------------------|------------------------|---------------------------|
| fastqs                                                                   | .fastq or List[.fastq] or .txt   | True                    | ...                             |


Output table
| Parameter                                                           | File type         | Flag                                                                           | Default Path                                                                                                     | Description           |
|----------------------------------------------------------------|--------------------|---------------------------------------------------------------------|------------------------------------------------------------------------------------------------------|-------------------------|
| out                                                                       | directory         | N/A                                                                            | "."                                                                                                                     | ...                          |
| split_by_Ns_and_low_quality_bases_out_suffix   | .fastq             | split_reads_by_Ns_and_low_quality_bases                          | <out>/<filename>_addedNs.<ext>												                     | ...                          |
| concatenate_paired_fastqs_out_suffix                | .fastq             | concatenate_paired_fastqs                                       | <out>/<filename>_concatenated.<ext>								                             | ...                          |
