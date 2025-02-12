# varseek
[![pypi version](https://img.shields.io/pypi/v/varseek)](https://pypi.org/project/varseek)
[![image](https://anaconda.org/bioconda/varseek/badges/version.svg)](https://anaconda.org/bioconda/varseek)
# [![Downloads](https://static.pepy.tech/personalized-badge/varseek?period=total&units=international_system&left_color=grey&right_color=brightgreen&left_text=downloads)](https://pepy.tec$
# [![Conda](https://img.shields.io/conda/dn/bioconda/varseek?logo=Anaconda)](https://anaconda.org/bioconda/varseek)
[![license](https://img.shields.io/pypi/l/varseek)](LICENSE)
![status](https://github.com/pachterlab/varseek/actions/workflows/ci.yml/badge.svg)
# ![Code Coverage](https://img.shields.io/badge/Coverage-83%25-green.svg)

![alt text](https://github.com/pachterlab/varseek/blob/main/figures/logo.png?raw=true)

`varseek` is a free, open-source command-line tool and Python package that provides mutation screening of RNA-seq and DNA-seq data using k-mer-based alignment against a reference of known mutations. The name comes from "seeking variants" or, alternatively, "seeing k-variants" (where a "k-variant" is defined as a k-mer containing a variant).
  
![alt text](https://github.com/pachterlab/varseek/blob/main/figures/varseek_overview.png?raw=true)

The functions of `varseek` are described in the table below.

| Description                                                    | Bash                   | Python                    |
|----------------------------------------------------------------|------------------------|---------------------------|
| Build a mutation-containing reference sequence (MCRS) file     | `varseek build ...`       | `varseek.build(...)`         |
| Describe the MCRS reference in a dataframe                     | `varseek info ...`    | `varseek.info(...)`      |
| Filter the MCRS file based on the CSV generated from varseek info | `varseek filter ...`    | `varseek.filter(...)`      |
| Run standard processing on the mutation count matrix           | `varseek clean ...`        | `varseek.clean(...)`          |
| Analyze the mutation count matrix results                      | `varseek summarize ...`         | `varseek.summarize(...)`           |
| Create synthetic RNA-seq dataset with mutation reads           | `varseek sim ...`        | `varseek.sim(...)`          |

After aligning and generating a mutation count matrix with `varseek`, you can explore the data using our pre-built notebooks. The notebooks are described in the table below.

| Description                                   | Notebook                                                                 |
|-----------------------------------------------|--------------------------------------------------------------------------------------|
| Preprocessing the mutation count matrix       | [3_matrix_preprocessing.ipynb](./3_matrix_preprocessing.ipynb)                       |
| Sequence visualization of mutations           | [4_1_mutation_analysis_sequence_visualization.ipynb](./4_1_mutation_analysis_sequence_visualization.ipynb) |
| Heatmap visualization of mutation patterns    | [4_2_mutation_analysis_heatmaps.ipynb](./4_2_mutation_analysis_heatmaps.ipynb)       |
| Protein-level mutation analysis               | [4_3_mutation_analysis_protein_mutation.ipynb](./4_3_mutation_analysis_protein_mutation.ipynb) |
| Heatmap analysis of gene expression           | [5_1_gene_analysis_heatmaps.ipynb](./5_1_gene_analysis_heatmaps.ipynb)               |
| Drug-target analysis for genes                | [5_2_gene_analysis_drugs.ipynb](./5_2_gene_analysis_drugs.ipynb)                     |
| Pathway analysis using Enrichr                | [6_1_pathway_analysis_enrichr.ipynb](./6_1_pathway_analysis_enrichr.ipynb)           |
| Gene Ontology enrichment analysis (GOEA)      | [6_2_pathway_analysis_goea.ipynb](./6_2_pathway_analysis_goea.ipynb)                 |

You can find more examples of how to use varseek in the GitHub repository for our preprint [GitHub - pachterlab/RLSRWP_2025](https://github.com/pachterlab/RLSRWP_2025.git).

    
If you use `varseek` in a publication, please cite the following study:    
```
PAPER CITATION
```
Read the article here: PAPER DOI  

# Installation
```bash
pip install varseek
```
Or with conda:
```bash
conda install -c bioconda varseek
```

For use in Python:
```python
# Python
import varseek as vk
```

# 🪄 Quick start guide
Command line:
```bash
# Build a mutation-containing reference sequence (MCRS) reference file
$ vk build ...

# Describe the MCRS reference in a dataframe
$ vk info ...

# Filter the MCRS reference based on the CSV generated from vk info
$ vk filter ...

# Run standard processing on the mutation count matrix
$ vk clean ...

# Analyze the results of the mutation count matrix
$ vk summarize ...

# Create a synthetic RNA-seq dataset consisting of mutation-containing reads from the MCRS reference
$ vk sim ...

```
Python (Python / Jupyter Lab / Google Colab):
```python  
import varseek as vk

# Build a mutation-containing reference sequence (MCRS) reference file
vk.build(...)

# Describe the MCRS reference in a dataframe
vk.info(...)

# Filter the MCRS reference based on the CSV generated from vk info
vk.filter(...)

# Run standard processing on the mutation count matrix
vk.clean(...)

# Analyze the results of the mutation count matrix
vk.summarize(...)

# Create a synthetic RNA-seq dataset consisting of mutation-containing reads from the MCRS reference
vk.sim(...)
```


# Quick start guide
1. Acquire a reference - follow one of the below options:
a. Download pre-built reference – standard workflow
View all downloadable references: `vk ref --list_downloadable_references`
`vk ref --download --mutations MUTATIONS --sequences SEQUENCES`

b. Make custom reference – screen for user-defined variants
`vk ref --mutations MUTATIONS --sequences SEQUENCES ...`

c. Customize reference building process – customize the VCRS filtering process (e.g., add additional information by which to filter, add custom filtering logic, tune filtering parameters based on the results of intermediate steps, etc.)
`vk build --mutations MUTATIONS --sequences SEQUENCES ...`
(optional) `vk info --input_dir INPUT_DIR ...`
(optional) `vk filter --input_dir INPUT_DIR ...`
`kb ref --workflow custom --index INDEX ...`


2. Screen for variants - follow one of the below options:
a. Standard workflow
(optional) fastq quality control
`vk count --index INDEX --t2g T2G ... --fastqs FASTQ1 FASTQ2...`

b. Customize variant screening process - additional fastq preprocessing, custom count matrix processing
(optional) fastq quality control
(optional) `vk fastqpp ... --fastqs FASTQ1 FASTQ2...`
`kb count --index INDEX --t2g T2G ... --fastqs FASTQ1 FASTQ2...`
(optional) `vk clean --adata ADATA ...`
(optional) `vk summarize --adata ADATA ...`


3. Analyze results
a. View results of vk summarize (txt, vcf, Anndata - in OUT from vk count)
b. Jupyter - see varseek/notebooks for examples to get started, and [GitHub - pachterlab/RLSRWP_2025](https://github.com/pachterlab/RLSRWP_2025.git) for figures from our first preprint




FAQs:
- Q: I want to add a custom filter to my VCRS index. How can I do this?
- A: First run vk build with the desired parameters to generate the vcrs.fa file. Optionally run this file through vk info and vk filter if any filtering performed by these steps is desired. Then, write any necessary logic to filter undesired entries out of the VCRS reference file. Generate a new file with vk.utils.create_mutant_t2g. Then pass the filtered vcrs fasta file into kb ref with --workflow custom.
