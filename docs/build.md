# varseek build 🔨
Takes in nucleotide sequences and variants (in [standard mutation annotation](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC1867422/) and returns mutated versions of the input sequences according to the provided variants.  
Return format: Saves mutated sequences in FASTA format.

### Examples
```bash
varseek build -s ATCGCTAAGCT -v 'c.4G>T'
```
```python
# Python
varseek.build(sequences="ATCGCTAAGCT", variants="c.4G>T")
```
&rarr; Returns ATCTCTAAGCT.  

<br/><br/>

**List of sequences with a mutation for each sequence provided in a list:**  
```bash
varseek build -s ATCGCTAAGCT TAGCTA -v 'c.4G>T' 'c.1_3inv' -o mut_fasta.fa
```
```python
# Python
varseek.build(sequences=["ATCGCTAAGCT", "TAGCTA"], variants=["c.4G>T", "c.1_3inv"], out="mut_fasta.fa")
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
varseek build -s ATCGCTAAGCT TAGCTA -v 'c.1_3inv' -k 3
```
```python
# Python
varseek.build(sequences=["ATCGCTAAGCT", "TAGCTA"], variants="c.1_3inv", k=3)
```
&rarr; Returns ['CTAGCT', 'GATCTA'].  


<br/><br/>

**Add variants to an entire genome with extended output**  
Main input:   
- mutation information as a `variants` CSV (by having `seq_id_column` contain chromosome information, and `var_column` contain mutation information with respect to genome coordinates)  
- the genome as the `sequences` file  

Since we are passing the path to a gtf file to the `gtf` argument, transcript boundaries will be respected (the genome will be split into transcripts). `gtf_transcript_id_column` specifies the name of the column in `variants` containing the transcript IDs corresponding to the transcript IDs in the `gtf` file.  

The `optimize_flanking_regions` argument maximizes the length of the resulting mutation-containing sequences while maintaining specificity (no wildtype k-mer will be retained).

`update_df` activates the creation of a new CSV file with updated information about each input and output sequence. This new CSV file will be saved as `update_df_out`. Since `store_full_sequences` is activated, this new CSV file will not only contain the output sequences (restricted in size by flanking regiong of size `k`), but also the complete input and output sequences. This allows us to observe the mutation in the context of the entire sequence. Lastly, we are also adding the translated versions of the complete sequences by adding the with the `translate` flag, so we can observe how the resulting amino acid sequence is changed. The `translate_start` and `translate_end` arguments specify the names of the columns in `variants` that contain the start and end positions of the open reading frame (start and end positions for translating the nucleotide sequence to an amino acid sequence), respectively.  

```bash
varseek build \
  -v variants_input.csv \
  -o mut_fasta.fa \
  -w 4 \
  -sic Chromosome \
  -vic Mutation \
  -gtf genome_annotation.gtf \
  -gtic Ensembl_Transcript_ID \
  -ofr \
  -update_df \
  -udf_o variants_updated.csv \
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
  variants="variants_input.csv",
  out="mut_fasta.fa",
  w=4,
  seq_id_column="chromosome",
  var_column="mutation",
  gtf="genome_annotation.gtf",
  gtf_transcript_id_column="Ensembl_Transcript_ID",
  optimize_flanking_regions=True,
  save_variants_updated_csv=True,
  variants_updated_csv_out="variants_updated.csv",
  store_full_sequences=True,
  translate=True,
  translate_start="Translate_Start",
  translate_end="Translate_End"
)
```
&rarr; Takes in a genome fasta ('genome_reference.fa') and gtf file ('genome_annotation.gtf') (these can be downloaded using `gget ref`) as well as a 'variants_input.csv' file containing: 
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
&rarr; Saves 'variants_updated.csv' file containing: 
```

| Chromosome | Mutation          | Ensembl_Transcript_ID  | variant_type  | wt_sequence | vcrs_sequence | wt_sequence_full  | vcrs_sequence_full| wt_sequence_aa_full | vcrs_sequence_aa_full |
|------------|-------------------|------------------------|---------------|-------------|-----------------|-------------------|----------------------|---------------------|-------------------------|
| 1          | g.224411A>C       | ENSMUST00000193812     | Substitution  | TGCTATGCT   | TGCTCTGCT       | ...TGCTATGCT...   | ...TGCTCTGCT...      | ...CYA...           | ...CSA...               |
| 8          | g.25111del        | ENST00000174411        | Deletion      | GAGTCCGAT   | GAGTCGAT        | ...GAGTCCGAT...   | ...GAGTCGAT...       | ...ESD...           | ...ES...                |
| X          | g.1011_1012insAA  | ENST00000421914        | Insertion     | TTAGCTT     | TTAGAACTT       | ...TTAGCTT...     | ...TTAGAACTT...      | ...A...             | ...EL...                |

```
