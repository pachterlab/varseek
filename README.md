# varseek
[![pypi version](https://img.shields.io/pypi/v/varseek)](https://pypi.org/project/varseek)
![Downloads](https://static.pepy.tech/personalized-badge/varseek?period=total&units=international_system&left_color=grey&right_color=brightgreen&left_text=downloads)
[![license](https://img.shields.io/pypi/l/varseek)](LICENSE)
![status](https://github.com/pachterlab/varseek/actions/workflows/ci.yml/badge.svg)
![Code Coverage](https://img.shields.io/badge/Coverage-83%25-green.svg)

![alt text](https://github.com/pachterlab/varseek/blob/main/figures/logo.png?raw=true)

<!--[![image](https://anaconda.org/bioconda/varseek/badges/version.svg)](https://anaconda.org/bioconda/varseek)-->
<!--[![Conda](https://img.shields.io/conda/dn/bioconda/varseek?logo=Anaconda)](https://anaconda.org/bioconda/varseek)-->

`varseek` is a free, open-source command-line tool and Python package that provides variant screening of RNA-seq and DNA-seq data using k-mer-based alignment against a reference of known variants. The name comes from "seeking variants" or, alternatively, "seeing k-variants" (where a "k-variant" is defined as a k-mer containing a variant).
  
![alt text](https://github.com/pachterlab/varseek/blob/main/figures/varseek_overview_simple.png?raw=true)

The two commands used in a standard workflow are `varseek ref` and `varseek count`. `varseek ref` generates a variant-containing reference sequence (VCRS) index that serves as the basis for variant calling. `varseek count` pseudoaligns RNA-seq or DNA-seq reads against the VCRS index and generates a variant count matrix. The variant count matrix can be used for downstream analysis. Each step wraps around other steps within the varseek package and the kb-python package, as described below.

![alt text](https://github.com/pachterlab/varseek/blob/main/figures/varseek_overview.png?raw=true)

The functions of `varseek` are described in the table below.

| Description                                                       | Bash              | Python (with `import varseek as vk`) |
|-------------------------------------------------------------------|-------------------|--------------------------------------|
| Build a variant-containing reference sequence (VCRS) fasta file   | `vk build ...`    | `vk.build(...)`                      |
| Describe the VCRS reference in a dataframe for filtering          | `vk info ...`     | `vk.info(...)`                       |
| Filter the VCRS file based on the CSV generated from varseek info | `vk filter ...`   | `vk.filter(...)`                     |
| Preprocess the FASTQ files before pseudoalignment                 | `vk fastqpp ...`  | `vk.fastqpp(...)`                    |
| Process the variant count matrix                                  | `vk clean ...`    | `vk.clean(...)`                      |
| Analyze the variant count matrix results                          | `vk summarize ...`| `vk.summarize(...)`                  |
| Wrap vk build, vk info, vk filter, and kb ref                     | `vk ref ...`      | `vk.ref(...)`                        |
| Wrap vk fastqpp, kb count, vk clean, and vk summarize             | `vk count ...`    | `vk.count(...)`                      |
| Create synthetic RNA-seq dataset with variant-containing reads    | `vk sim ...`      | `vk.sim(...)`                        |

After aligning and generating a variant count matrix with `varseek`, you can explore the data using our pre-built notebooks. The notebooks are described in the table below.

| Description                                   | Notebook                                                                 |
|-----------------------------------------------|--------------------------------------------------------------------------------------|
| Preprocessing the variant count matrix        | [3_matrix_preprocessing.ipynb](./3_matrix_preprocessing.ipynb)                       |
| Sequence visualization of variants            | [4_1_variant_analysis_sequence_visualization.ipynb](./4_1_variant_analysis_sequence_visualization.ipynb) |
| Heatmap visualization of variant patterns     | [4_2_variant_analysis_heatmaps.ipynb](./4_2_variant_analysis_heatmaps.ipynb)       |
| Protein-level variant analysis                | [4_3_variant_analysis_protein_variant.ipynb](./4_3_variant_analysis_protein_variant.ipynb) |
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

# 🪄 Quick start guide
## 1. Acquire a Reference

Follow one of the below options:

### a. Download a Pre-built Reference
- (optional) View all downloadable references: `vk ref --list_downloadable_references`
- `vk ref --download --variants VARIANTS --sequences SEQUENCES`

### b. Make custom reference – screen for user-defined variants
- `vk ref --variants VARIANTS --sequences SEQUENCES ...`

### c. Customize reference building process – customize the VCRS filtering process (e.g., add additional information by which to filter, add custom filtering logic, tune filtering parameters based on the results of intermediate steps, etc.)
- `vk build --variants VARIANTS --sequences SEQUENCES ...`
- (optional) `vk info --input_dir INPUT_DIR ...`
- (optional) `vk filter --input_dir INPUT_DIR ...`
- `kb ref --workflow custom --index INDEX ...`


## 2. Screen for variants

Follow one of the below options:

### a. Standard workflow
- (optional) fastq quality control
- `vk count --index INDEX --t2g T2G ... --fastqs FASTQ1 FASTQ2...`

### b. Customize variant screening process - additional fastq preprocessing, custom count matrix processing
- (optional) fastq quality control
- (optional) `vk fastqpp ... --fastqs FASTQ1 FASTQ2...`
- `kb count --index INDEX --t2g T2G ... --fastqs FASTQ1 FASTQ2...`
- (optional) `kb count --index REFERENCE_INDEX --t2g REFERENCE_T2G ... --fastqs FASTQ1 FASTQ2...`
- (optional) `vk clean --adata ADATA ...`
- (optional) `vk summarize --adata ADATA ...`


**Examples for getting started:** [GitHub - pachterlab/varseek](https://github.com/pachterlab/varseek-examples.git)
**Manuscript**: ...
**Repository for manuscript figures**: [GitHub - pachterlab/RLSRP_2025](https://github.com/pachterlab/RLSRP_2025.git)
