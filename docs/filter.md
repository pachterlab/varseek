Input table
| Parameter                                                           | File type         | Required?           | Corresponding parameter from vk build/info        | Description             |
|----------------------------------------------------------------|--------------------|------------------------|-----------------------------------------------------------------|---------------------------|
| input_dir                                                              | directory         | True                    | N/A                                                                        | ...                            |
| mutations_updated_vk_info_csv                         | .csv                | False                  | mutations_updated_vk_info_csv_out                   | ...                             |
| mutations_updated_exploded_vk_info_csv        | .csv                | False                  | mutations_updated_exploded_vk_info_csv_out   | ...                             |
| id_to_header_csv                                                | .csv                | False                  | id_to_header_csv_out                                          | ...                             |
| dlist_fasta                                                            | .fa                  | False                  | dlist_combined_fasta_out (or any other dlist)       | ...                             |


Output table
| Parameter                                                           | File type         | Flag                                                                           | Default Path                                                                                                     | Description           |
|----------------------------------------------------------------|--------------------|---------------------------------------------------------------------|------------------------------------------------------------------------------------------------------|-------------------------|
| out                                                                       | directory         | N/A                                                                            | .                                                                                                                        | ...                          |
| mutations_updated_filtered_csv_out                  | .csv                | N/A                                                                            | <out>/mutation_metadata_df_updated_filtered.csv                                         | ...                          |
| mutations_updated_exploded_filtered_csv_out  | .csv                | save_mutations_updated_exploded_filtered_csv    | <out>/mutation_metadata_df_updated_filtered_exploded.csv                        | ...                          |
| id_to_header_filtered_csv_out                            | .csv                | N/A                                                                            | <out>/id_to_header_mapping_filtered.csv                                                      | ...                          |
| dlist_filtered_fasta_out                                         | .fa                  | N/A                                                                            | <out>/dlist_filtered.fa                                                                                       | ...                          |
| mcrs_filtered_fasta_out                                       | .fa                  | N/A                                                                            | <out>/mcrs_filtered.fa                                                                                      | ...                          |
| mcrs_filtered_t2g_out                                          | .txt                 | N/A                                                                            | <out>/mcrs_t2g_filtered.txt                                                                              | ...                          |
| wt_mcrs_filtered_fasta_out                                  | .txt                 | N/A                                                                            | <out>/wt_mcrs_filtered.fa                                                                                | ...                          |
| wt_mcrs_filtered_t2g_out                                     | .txt                 | N/A                                                                            | <out>/wt_mcrs_t2g_filtered.txt                                                                       | ...                          |


Takes in:
- info csv
- mcrs fasta
- filters in the following format

COLUMN-RULE=VALUE
- COLUMN: column name in the info csv
- RULE: rule to apply to the column from the following list:
    - min: minimum value
    - max: maximum value
    - between: between two values (inclusive)
    - toppercent: top x% of values (eg 1 to keep top 1%)
    - bottompercent: bottom x% of values (eg 1 to keep bottom 1%)
    - equal: equal to value
    - notequal: not equal to value
    - isin: equals an element in value (set, or a txt file where each value is separated by a new line)
    - isnotin: does not equal an element in value (set, or a txt file where each value is separated by a new line)
    - istrue: is True
    - isfalse: is False
    - isnottrue: is not True (i.e., False or NaN)
    - isnotfalse: is not False (i.e., True or NaN)
    - isnull: is null
    - isnotnull: is not null
- VALUE: value to compare to
    - min, max: single numeric value (e.g., 1)
    - between: two numeric values separated by a comma (e.g., "1,2")
    - contains, notcontains: one or more numeric values separated by a comma (e.g., "1,2,3") for command line or python, or a list or set passed in through an fstring (python only)
    - equal, notequal: single value (e.g., "yes")
    - istrue, isfalse, isnottrue, isnotfalse, isnull, isnotnull: no value needed

on command line, simply list the filters as the last argument; in python, pass them as a list of strings

OR the filters can be passed in as a txt file - example:
COLUMN1-RULE1=VALUE1
COLUMN2-RULE2=VALUE2
COLUMN3-RULE3=VALUE3

While the order of filters does not affect the output filtered fasta file, it will affect the stats printed to the console when in verbose mode. The stats will be printed in the order of the filters.