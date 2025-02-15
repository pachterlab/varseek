Input table
| Parameter                                                           | File type         | Required?           | Corresponding parameter from vk build/info        | Description             |
|----------------------------------------------------------------|--------------------|------------------------|-----------------------------------------------------------------|---------------------------|
| input_dir                                                              | directory         | True                    | N/A                                                                        | ...                            |
| variants_updated_vk_info_csv                         | .csv                | False                  | variants_updated_vk_info_csv_out                   | ...                             |
| variants_updated_exploded_vk_info_csv        | .csv                | False                  | variants_updated_exploded_vk_info_csv_out   | ...                             |
| id_to_header_csv                                                | .csv                | False                  | id_to_header_csv_out                                          | ...                             |
| dlist_fasta                                                            | .fa                  | False                  | dlist_combined_fasta_out (or any other dlist)       | ...                             |


Output table
| Parameter                                                           | File type         | Flag                                                                           | Default Path                                                                                                     | Description           |
|----------------------------------------------------------------|--------------------|---------------------------------------------------------------------|------------------------------------------------------------------------------------------------------|-------------------------|
| out                                                                       | directory         | N/A                                                                            | <input_dir>                                                                                                       | ...                          |
| variants_updated_filtered_csv_out                  | .csv                | N/A                                                                            | <out>/mutation_metadata_df_updated_filtered.csv                                         | ...                          |
| variants_updated_exploded_filtered_csv_out  | .csv                | save_variants_updated_exploded_filtered_csv    | <out>/mutation_metadata_df_updated_filtered_exploded.csv                        | ...                          |
| id_to_header_filtered_csv_out                            | .csv                | N/A                                                                            | <out>/id_to_header_mapping_filtered.csv                                                      | ...                          |
| dlist_filtered_fasta_out                                         | .fa                  | N/A                                                                            | <out>/dlist_filtered.fa                                                                                       | ...                          |
| vcrs_filtered_fasta_out                                       | .fa                  | N/A                                                                            | <out>/vcrs_filtered.fa                                                                                      | ...                          |
| vcrs_filtered_t2g_out                                          | .txt                 | N/A                                                                            | <out>/vcrs_t2g_filtered.txt                                                                              | ...                          |
| wt_vcrs_filtered_fasta_out                                  | .txt                 | N/A                                                                            | <out>/wt_vcrs_filtered.fa                                                                                | ...                          |
| wt_vcrs_filtered_t2g_out                                     | .txt                 | N/A                                                                            | <out>/wt_vcrs_t2g_filtered.txt                                                                       | ...                          |


Takes in:
- info csv
- vcrs fasta
- filters in the following format

Only keeps values that pass the filters (e.g., COLUMN0:min=5 will only keep rows where COLUMN0 is greater than or equal to 5)

COLUMN:RULE or COLUMN:RULE=VALUE
- COLUMN: column name in the info csv
- RULE: rule to apply to the column from the following list:
    - greater_than:
        - greater than VALUE i.e., minimum (exclusive) - does not filter null values
        - VALUE: numeric
        - example: COLUMN1:greater_than=0
    - greater_or_equal: 
        - greater than or equal to VALUE i.e., minimum (inclusive) - does not filter null values
        - VALUE: numeric
        - example: COLUMN2:greater_or_equal=0
    - less_than:
        - less than or equal to VALUE i.e., maximum (inclusive) - does not filter null values
        - VALUE: numeric
        - COLUMN3:less_than=100
    - less_or_equal:
        - less than or equal to VALUE i.e., maximum (inclusive) - does not filter null values
        - VALUE: numeric
        - example: COLUMN4:less_or_equal=100
    - between_inclusive:
        - between two VALUEs (inclusive) - does not filter null values
        - VALUE: two numeric values separated by a comma
        - example: COLUMN5:between=0,100
    - between_exclusive:
        - between two VALUEs (exclusive) - does not filter null values
        - VALUE: two numeric values separated by a comma
        - example: COLUMN6:between=0,100
    - top_percent:
        - top VALUE% of numbers (inclusive)
        - VALUE: numeric
        - example: COLUMN7:top_percent=10
    - bottom_percent:
        - bottom VALUE% of numbers (inclusive)
        - VALUE: numeric
        - example: COLUMN8:bottom_percent=0.1
    - equal:
        - equal to VALUE
        - VALUE: string
        - example: COLUMN9:equal=cdna
    - not_equal:
        - not equal to VALUE
        - VALUE: string
        - example: COLUMN10:not_equal=cdna
    - is_in:
        - equals an element in VALUE i.e., in a set
        - VALUE: either (1) list or (2) path to text file, where each value is separated by a new line
        - example: COLUMN11:is_in=[1,2,3], or COLUMN11:is_in=myset.txt
    - is_not_in:
        - does not equal an element in value i.e., not in a set
        - VALUE: either (1) list or (2) path to text file, where each value is separated by a new line
        - example: COLUMN12:is_not_in=[1,2,3], or COLUMN12:is_not_in=myset.txt
    - is_true:
        - is True
        - VALUE: no value needed
        - example: COLUMN13:is_true
    - is_false:
        - is False
        - VALUE: no value needed
        - example: COLUMN14:is_false
    - is_not_true:
        - is not True (i.e., False or NaN)
        - VALUE: no value needed
        - example: COLUMN15:is_not_true
    - is_not_false:
        - is not False (i.e., True or NaN)
        - VALUE: no value needed
        - example: COLUMN16:is_not_false
    - is_null:
        - is null
        - VALUE: no value needed
        - example: COLUMN17:is_null
    - is_not_null:
        - is not null
        - VALUE: no value needed
        - example: COLUMN18:is_not_null

on command line, simply list the filters as the last argument; in python, pass them as a list of strings

OR the filters can be passed in as a txt file - example:
COLUMN1:RULE1=VALUE1
COLUMN2:RULE2=VALUE2
COLUMN3:RULE3=VALUE3

While the order of filters does not affect the output filtered fasta file, it will affect the stats printed to the console when in verbose mode. The stats will be printed in the order of the filters.