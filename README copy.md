# varseek
[![pypi version](https://img.shields.io/pypi/v/varseek)](https://pypi.org/project/varseek)
[![license](https://img.shields.io/pypi/l/varseek)](LICENSE)
![status](https://github.com/pachterlab/varseek/actions/workflows/ci.yml/badge.svg)

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
import varseek
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






