> Python arguments are equivalent to long-option arguments (`--arg`), unless otherwise specified. Flags are True/False arguments in Python. The manual for any varseek tool can be called from the command-line using the `-h` `--help` flag.  
# varseek build 🔨
Takes in nucleotide sequences and mutations (in [standard mutation annotation](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC1867422/) and returns mutated versions of the input sequences according to the provided mutations.  
Return format: Saves mutated sequences in FASTA format.

Input table
| Parameter                                                           | File type         | Required?           | Description             |
|----------------------------------------------------------------|--------------------|------------------------|---------------------------|
| variants                                                               | .csv or string   | True                    | ...                             |
| sequences                                                           | .fa/.fasta         | True                    | ...                             |

Output table
| Parameter                                                           | File type         | Flag                                                        | Default Path                                                        | Description           |
|----------------------------------------------------------------|--------------------|-----------------------------------------------------|----------------------------------------------------------------|-------------------------|
| out                                                                       | directory         | N/A                                                        | <input_dir>                                                           | ...                          |
| reference_out_dir                                                | directory         | N/A                                                        | <out>                                                                   | ...                          |
| mcrs_fasta_out                                                    | .fa                  | N/A                                                        | <out>/mcrs.fa                                                      | ...                          |
| id_to_header_csv_out                                         | .csv                | N/A                                                        | <out>/id_to_header_mapping.csv                      | ...                          |
| mutations_updated_csv_out                               | .csv                 | save_mutations_updated_csv=True     | <out>/mutation_metadata_df.csv                       | ...                          |
| mcrs_t2g_out                                                      | .txt                  | N/A                                                        | <out>/mcrs_t2g.txt                                              | ...                          |
| wt_mcrs_fasta_out                                              | .txt                  | save_wt_mcrs_fasta_and_t2g=True     | <out>/wt_mcrs.fa                                                | ...                          |
| wt_mcrs_t2g_out                                                 | .txt                  | save_wt_mcrs_fasta_and_t2g=True     | <out>/wt_mcrs_t2g.txt                                        | ...                          |
| removed_variants_text_out                                 | .txt                  | save_removed_variants_text=True      | <out>/removed_variants.txt                                | ...                          |
| filtering_report_text_out                                       | .txt                  | save_filtering_report_text=True           | <out>/filtering_report.txt                                      | ...                          |

**Required arguments**  
`-s` `--sequences`
Path to the fasta file containing the sequences to be mutated, e.g., 'seqs.fa'.
Sequence identifiers following the '>' character must correspond to the identifiers
in the seq_ID column of 'mutations'.

Example:
>seq1 (or ENSG00000106443)
ACTGCGATAGACT
>seq2
AGATCGCTAG

Alternatively: Input sequence(s) as a string or a list of strings,
e.g. 'AGCTAGCT' or ['ACTGCTAGCT', 'AGCTAGCT'].

NOTE: Only the letters until the first space or dot will be used as sequence identifiers
- Version numbers of Ensembl IDs will be ignored.
NOTE: When 'sequences' input is a genome, also see 'gtf' argument below.

Alternatively, if 'mutations' is a string specifying a supported database, 
sequences can be a string indicating the source upon which to apply the mutations.
See below for supported databases and sequences options.
To see the supported combinations of mutations and sequences, either
1) run `vk build --help` from the command line, or
2) run varseek.varseek_build.print_valid_values_for_mutations_and_sequences_in_varseek_build() in python

`-m` `--mutations`  
Path to csv or tsv file (str) (e.g., 'mutations.csv'), or DataFrame (DataFrame object),
containing information about the mutations in the following format:

| mutation         | mut_ID | seq_ID |
| c.2C>T           | mut1   | seq1   | -> Apply mutation 1 to sequence 1
| c.9_13inv        | mut2   | seq2   | -> Apply mutation 2 to sequence 2
| c.9_13inv        | mut2   | seq3   | -> Apply mutation 2 to sequence 3
| c.9_13delinsAAT  | mut3   | seq3   | -> Apply mutation 3 to sequence 3
| ...              | ...    | ...    |

'mutation' = Column containing the mutations to be performed written in standard mutation annotation (see below)
'seq_ID' = Column containing the identifiers of the sequences to be mutated (must correspond to the string following
the > character in the 'sequences' fasta file; do NOT include spaces or dots)
'mut_ID' = Column containing an identifier for each mutation (optional).

Alternatively: Input mutation(s) as a string or list, e.g., 'c.2C>T' or ['c.2C>T', 'c.1A>C'].
If a list is provided, the number of mutations must equal the number of input sequences.

For more information on the standard mutation annotation, see https://www.ncbi.nlm.nih.gov/pmc/articles/PMC1867422/.

Alternatively, 'mutations' can be a string specifying a supported database, which will automatically download
both the mutation database and corresponding reference sequence (if the 'sequences' is not a path).
To see the supported combinations of mutations and sequences, either
1) run `vk build --help` from the command line, or
2) run varseek.varseek_build.print_valid_values_for_mutations_and_sequences_in_varseek_build() in python


**Optional input-related arguments**  
`-mc` `--mut_column`  
Name of the column containing the mutations to be performed in `mutations`. Default: 'mutation'.  

`-sic` `--seq_id_column`  
Name of the column containing the IDs of the sequences to be mutated in `mutations`. Default: 'seq_ID'.

`-mic` `--mut_id_column`  
Name of the column containing the IDs of each mutation in `mutations`. Optional. Default: use <seq_ID>_<mutation> for each row.

`-gtf` `--gtf`  
Path to a .gtf file. When providing a genome fasta file as input for 'sequences', you can provide a .gtf file here and the input sequences will be defined according to the transcript boundaries, e.g. 'path/to/genome_annotation.gtf'. Default: None

`-gtic` `--gtf_transcript_id_column`  
Column name in the input `mutations` file containing the transcript ID. In this case, column `seq_id_column` should contain the chromosome number.  
Required when `gtf` is provided. Default: None  
  
**Optional mutant sequence generation/filtering arguments**  
`-w` `--w`  
Length of sequence windows flanking the mutation. Default: 30.  
If w > total length of the sequence, the entire sequence will be kept.  

`-k` `--k`  
Length of the k-mers to be considered when removed remove_seqs_with_wt_kmers.
If using kallisto in a later workflow, then this should correspond to kallisto k).
Must be greater than the value passed in for w. Default: w+1.

`--insertion_size_limit`
Maximum number of nucleotides allowed in an insertion-type mutation. Mutations with insertions larger than this will be dropped.
Default: None (no insertion size limit will be applied)

`-msl` `--min_seq_len`  
Minimum length of the mutant output sequence. Mutant sequences smaller than this will be dropped. Default: None (No length filter will be applied)

`-ma` `--max_ambiguous`                
Maximum number of 'N' (or 'n') characters allowed in the output sequence. Default: None (no 'N' filter will be applied)

`-riol` `--required_insertion_overlap_length`
Sets the Minimum number of nucleotides that must overlap between the inserted sequence and the flanking regions after flank optimization. Only effective when optimize_flanking_regions is also True. Default: None (No checking). If "all", then require the entire insertion and the following nucleotide

**Optional mutant sequence generation/filtering flags**  
`-ofr` `--optimize_flanking_regions`  
Whether to remove nucleotides from either end of the mutant sequence to ensure (when possible) that the mutant sequence does not contain any w-mers (where a w-mer is a subsequence of length w) also found in the wildtype/input sequence. Default: False

`-rswk` `--remove_seqs_with_wt_kmers`  
Removes output sequences where at least one (w+1)-mer (where a w-mer is a subsequence of length w) is also present in the wildtype/input sequence in the same region. If `--optimize_flanking_regions`, only sequences for which a wildtype w-mer is still present after optimization will be removed. Default: False

`-mi` `--merge_identical`          
Whether to merge identical mutant sequences in the output (identical sequences will be merged by concatenating the sequence headers for all identical sequences with semicolons). Default: False

`--strandedness`          
Whether to consider the forward and reverse-complement mutant sequences as distinct if merging identical sequences. Only effective when merge_identical is also True. Default: False (ie do not consider forward and reverse-complement sequences to be equivalent)

`-koh` `--keep_original_headers`
Whether to keep the original sequence headers in the output fasta file, or to replace them with unique IDs of the form 'vcrs_<int>. If False, then an additional file at the path <id_to_header_csv_out> will be formed that maps sequence IDs from the fasta file to the <mut_id_column>. Default: False.

**Optional arguments to generate additional output**   
`-smuc` `--save_mutations_updated_csv`   
Whether to update the input `mutations` DataFrame to include additional columns with the mutation type, wildtype nucleotide sequence, and mutant nucleotide sequence (only valid if `mutations` is a csv or tsv file). Default: False

`--save_wt_mcrs_fasta_and_t2g`
Whether to create a fasta file containing the wildtype sequence counterparts of the mutation-containing reference sequences (MCRSs) and the corresponding t2g. Default: False.

**Optional flags to modify additional output**  
`-sfs` `--store_full_sequences`         
Includes the complete wildtype and mutant sequences in the updated `mutations` DataFrame (not just the sub-sequence with k-length flanks). Only valid when used with `--update_df`.   

`-tr` `--translate`                  
Adds additional columns to the updated `mutations` DataFrame containing the wildtype and mutant amino acid sequences. Only valid when used with `--store_full_sequences`.   

`-ts` `--translate_start`              
(int or str) The position in the input nucleotide sequence to start translating, e.g. 5. If a string is provided, it should correspond to a column name in `mutations` containing the open reading frame start positions for each sequence/mutation. Only valid when used with `--translate`.  
Default: translates from the beginning of each sequence  

`-te` `--translate_end`                
(int or str) The position in the input nucleotide sequence to end translating, e.g. 35. If a string is provided, it should correspond to a column name in `mutations` containing the open reading frame end positions for each sequence/mutation. Only valid when used with `--translate`.  
Default: translates until the end of each sequence  
                                  
**Optional general arguments**
`-ro` `--reference_out`
Path to reference files to be downloaded if 'mutations' is a supported database and 'sequences' is not provided. Default: 'out' directory.  

`-o` `--out`   
Path to output folder containing created files (if fasta_out and/or update_df_out not supplied) Default: '.'.
Default: None -> returns a list of the mutated sequences to standard out.    
The identifiers (following the '>') of the mutated sequences in the output FASTA will be '>[seq_ID]_[mut_ID]'. 

**Optional general flags**  
`-q` `--quiet`   
Command-line only. Prevents progress information from being displayed.  
Python: Use `verbose=False` to prevent progress information from being displayed.  

### Examples
```bash
varseek build ATCGCTAAGCT -m 'c.4G>T'
```
```python
# Python
varseek.build("ATCGCTAAGCT", "c.4G>T")
```
&rarr; Returns ATCTCTAAGCT.  

<br/><br/>

**List of sequences with a mutation for each sequence provided in a list:**  
```bash
varseek build ATCGCTAAGCT TAGCTA -m 'c.4G>T' 'c.1_3inv' -o mut_fasta.fa
```
```python
# Python
varseek.build(["ATCGCTAAGCT", "TAGCTA"], ["c.4G>T", "c.1_3inv"], out="mut_fasta.fa")
```
&rarr; Saves 'mut_fasta.fa' file containing: 
```
>seq1_mut1  
ATCTCTAAGCT  
>seq2_mut2  
GATCTA
```

<br/><br/>

**One mutation applied to several sequences with adjusted `k`:**  
```bash
varseek build ATCGCTAAGCT TAGCTA -m 'c.1_3inv' -k 3
```
```python
# Python
varseek.build(["ATCGCTAAGCT", "TAGCTA"], "c.1_3inv", k=3)
```
&rarr; Returns ['CTAGCT', 'GATCTA'].  


<br/><br/>

**Add mutations to an entire genome with extended output**  
Main input:   
- mutation information as a `mutations` CSV (by having `seq_id_column` contain chromosome information, and `mut_column` contain mutation information with respect to genome coordinates)  
- the genome as the `sequences` file  

Since we are passing the path to a gtf file to the `gtf` argument, transcript boundaries will be respected (the genome will be split into transcripts). `gtf_transcript_id_column` specifies the name of the column in `mutations` containing the transcript IDs corresponding to the transcript IDs in the `gtf` file.  

The `optimize_flanking_regions` argument maximizes the length of the resulting mutation-containing sequences while maintaining specificity (no wildtype k-mer will be retained).

`update_df` activates the creation of a new CSV file with updated information about each input and output sequence. This new CSV file will be saved as `update_df_out`. Since `store_full_sequences` is activated, this new CSV file will not only contain the output sequences (restricted in size by flanking regiong of size `k`), but also the complete input and output sequences. This allows us to observe the mutation in the context of the entire sequence. Lastly, we are also adding the translated versions of the complete sequences by adding the with the `translate` flag, so we can observe how the resulting amino acid sequence is changed. The `translate_start` and `translate_end` arguments specify the names of the columns in `mutations` that contain the start and end positions of the open reading frame (start and end positions for translating the nucleotide sequence to an amino acid sequence), respectively.  

```bash
varseek build \
  -m mutations_input.csv \
  -o mut_fasta.fa \
  -w 4 \
  -sic Chromosome \
  -mic Mutation \
  -gtf genome_annotation.gtf \
  -gtic Ensembl_Transcript_ID \
  -ofr \
  -update_df \
  -udf_o mutations_updated.csv \
  -sfs \
  -tr \
  -ts Translate_Start \
  -te Translate_End \
  genome_reference.fa
```
```python
# Python
varseek.build(
  sequences="genome_reference.fa",
  mutations="mutations_input.csv",
  out="mut_fasta.fa",
  w=4,
  seq_id_column="Chromosome",
  mut_column="Mutation",
  gtf="genome_annotation.gtf",
  gtf_transcript_id_column="Ensembl_Transcript_ID",
  optimize_flanking_regions=True,
  update_df=True,
  update_df_out="mutations_updated.csv",
  store_full_sequences=True,
  translate=True,
  translate_start="Translate_Start",
  translate_end="Translate_End"
)
```
&rarr; Takes in a genome fasta ('genome_reference.fa') and gtf file ('genome_annotation.gtf') (these can be downloaded using [`gget ref`](ref.md)) as well as a 'mutations_input.csv' file containing: 
```
| Chromosome | Mutation          | Ensembl_Transcript_ID  | Translate_Start | Translate_End |
|------------|-------------------|------------------------|-----------------|---------------|
| 1          | g.224411A>C       | ENST00000193812        | 0               | 100           |
| 8          | g.25111del        | ENST00000174411        | 0               | 294           |
| X          | g.1011_1012insAA  | ENST00000421914        | 9               | 1211          |
``` 
&rarr; Saves 'mut_fasta.fa' file containing: 
```
>1:g.224411A>C  
TGCTCTGCT  
>8:g.25111del  
GAGTCGAT
>X:g.1011_1012insAA
TTAGAACTT
``` 
&rarr; Saves 'mutations_updated.csv' file containing: 
```

| Chromosome | Mutation          | Ensembl_Transcript_ID  | mutation_type | wt_sequence | mutant_sequence | wt_sequence_full  | mutant_sequence_full | wt_sequence_aa_full | mutant_sequence_aa_full |
|------------|-------------------|------------------------|---------------|-------------|-----------------|-------------------|----------------------|---------------------|-------------------------|
| 1          | g.224411A>C       | ENSMUST00000193812     | Substitution  | TGCTATGCT   | TGCTCTGCT       | ...TGCTATGCT...   | ...TGCTCTGCT...      | ...CYA...           | ...CSA...               |
| 8          | g.25111del        | ENST00000174411        | Deletion      | GAGTCCGAT   | GAGTCGAT        | ...GAGTCCGAT...   | ...GAGTCGAT...       | ...ESD...           | ...ES...                |
| X          | g.1011_1012insAA  | ENST00000421914        | Insertion     | TTAGCTT     | TTAGAACTT       | ...TTAGCTT...     | ...TTAGAACTT...      | ...A...             | ...EL...                |

```
