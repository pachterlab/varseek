import os
import scanpy as sc
import numpy as np
import pandas as pd
import anndata

from varseek.utils import (
    plot_items_descending_order,
    plot_scree,
    plot_loading_contributions,
    find_resolution_for_target_clusters,
    plot_contingency_table,
    plot_knn_tissue_frequencies,
    plot_ascending_bar_plot_of_cluster_distances,
    plot_jaccard_bar_plot,
    plot_knee_plot,
    increment_adata_based_on_dlist_fns,
    decrement_adata_matrix_when_split_by_Ns_or_running_paired_end_in_single_end_mode,
    remove_adata_columns,
)


def summarize(
    adata_path,
    assay="bulk",
    output_dir=".",
    overwrite=False,
    top_values=10,
    verbose=False,
    **kwargs,
):
    os.makedirs(output_dir, exist_ok=True)
    stats_file = os.path.join(output_dir, "varseek_summarize_stats.txt")
    specific_stats_folder = os.path.join(output_dir, "specific_stats")
    plots_folder = os.path.join(output_dir, "plots")

    os.makedirs(specific_stats_folder, exist_ok=True)
    os.makedirs(plots_folder, exist_ok=True)

    if os.path.exists(stats_file):
        if not overwrite:
            raise FileExistsError(f"Stats file {stats_file} already exists. Please delete it or specify a different output directory.")
        else:
            os.remove(stats_file)
    if isinstance(adata_path, anndata.AnnData):
        adata = adata_path
    elif isinstance(adata_path, str):
        adata = sc.read_h5ad(adata_path)
    else:
        raise ValueError("adata_path must be a string (file path) or an AnnData object.")

    if "mcrs_id" not in adata.var.columns:
        adata.var["mcrs_id"] = adata.var_names

    adata.var_names = adata.var["mcrs_header"]

    if "mcrs_count" not in adata.var.columns:
        adata.var["mcrs_count"] = adata.X.sum(axis=0).A1 if hasattr(adata.X, "A1") else adata.X.sum(axis=0).flatten()

    # 1. Number of Mutations with Count > 0 in any Sample/Cell, and for bulk in particular, for each sample; then list the mutations
    with open(stats_file, "w") as f:
        mutations_with_any_count = (adata.X > 0).sum(axis=0)
        mutations_count_any_row = (mutations_with_any_count > 0).sum()
        line = f"Total mutations with count > 0 for any sample/cell: {mutations_count_any_row}"
        f.write(line)
        if assay == "bulk":
            for sample in adata.obs_names:
                count_nonzero_mutations = (adata[sample, :].X > 0).sum()
                line = f"Sample {sample} has {count_nonzero_mutations} mutations with count > 0."
                f.write(line)

    mutations_with_nonzero_counts = adata.var_names[mutations_with_any_count > 0]
    with open(f"{specific_stats_folder}/mutations_with_any_count.txt", "w") as f:
        for mutation in mutations_with_nonzero_counts:
            f.write(f"{mutation}\n")

    # 2. Mutations Present Across the Most Samples
    adata.var["number_of_samples_in_which_the_mutation_is_detected"] = (adata.X > 0).sum(axis=0).A1 if hasattr(adata.X, "A1") else (adata.X > 0).sum(axis=0)
    # Sort by number of samples and break ties with mcrs_count
    most_common_mutations = adata.var.sort_values(
        by=["number_of_samples_in_which_the_mutation_is_detected", "mcrs_count"],
        ascending=False,
    )
    mutation_names = most_common_mutations.index.tolist()
    mutation_names_top_n = mutation_names[:top_values]
    with open(stats_file, "a") as f:
        f.write(f"Mutations present across the most samples: {', '.join(mutation_names_top_n)}")
    with open(f"{specific_stats_folder}/mutations_present_across_the_most_samples.txt", "w") as f:
        f.write("Mutation\tNumber_of_Samples\tTotal_Counts\n")

        # Write each mutation's details
        for mutation in most_common_mutations.index:
            number_of_samples = adata.var.loc[mutation, "number_of_samples_in_which_the_mutation_is_detected"]
            total_counts = adata.var.loc[mutation, "mcrs_count"]
            f.write(f"{mutation}\t{number_of_samples}\t{total_counts}\n")

    # 3. Top 10 Mutations with Highest mcrs_count Across All Samples
    top_mutations_mcrs_count = adata.var.sort_values(by="mcrs_count", ascending=False)
    mutation_names = top_mutations_mcrs_count.index.tolist()
    mutation_names_top_n = mutation_names[:top_values]
    with open(stats_file, "a") as f:
        f.write(f"Mutations with highest counts across all samples: {', '.join(mutation_names_top_n)}")
    with open(f"{specific_stats_folder}/mutations_highest_mcrs_count.txt", "w") as f:
        f.write("Mutation\tNumber_of_Samples\tTotal_Counts\n")

        # Write each mutation's details
        for mutation in top_mutations_mcrs_count.index:
            number_of_samples = adata.var.loc[mutation, "number_of_samples_in_which_the_mutation_is_detected"]
            total_counts = adata.var.loc[mutation, "mcrs_count"]
            f.write(f"{mutation}\t{number_of_samples}\t{total_counts}\n")

    # --------------------------------------------------------------------------------------------------------
    # 4. Number of Genes with Count > 0 in any Sample/Cell, and for bulk in particular, for each sample; then list the genes
    with open(stats_file, "a") as f:
        # Sum mcrs_count for each gene
        gene_counts = adata.var.groupby("gene_name")["mcrs_count"].sum()

        # Count genes with non-zero mcrs_count across all samples
        genes_count_any_row = (gene_counts > 0).sum()
        line = f"Total genes with count > 0 in any sample/cell: {genes_count_any_row}\n"
        f.write(line)
        print(line.strip())  # For verification in console

        # For bulk assays, calculate counts for each sample
        if assay == "bulk":
            for sample in adata.obs_names:
                # Calculate mcrs_count per gene for the specific sample
                gene_counts_per_sample = adata[sample, :].to_df().gt(0).groupby(adata.var["gene_name"], axis=1).sum().gt(0).sum()
                line = f"Sample {sample} has {gene_counts_per_sample.sum()} genes with count > 0.\n"
                f.write(line)
                print(line.strip())  # For verification in console

    # List of genes with non-zero mcrs_count across all samples
    genes_with_nonzero_counts = gene_counts[gene_counts > 0].index
    with open(f"{specific_stats_folder}/genes_with_any_count.txt", "w") as f:
        for gene in genes_with_nonzero_counts:
            f.write(f"{gene}\n")

    # 5. Genes Present Across the Most Samples
    # Calculate the number of samples where each gene has at least one mutation with count > 0
    adata.var["detected"] = (adata.X > 0).astype(int)  # Binary matrix indicating presence per sample
    gene_sample_count = adata.var.groupby("gene_name")["detected"].sum()  # Sum across mutations per gene

    # Add the total mcrs_count per gene (summing across all mutations for that gene)
    gene_mcrs_count = adata.var.groupby("gene_name")["mcrs_count"].sum()

    # Combine both into a DataFrame for sorting
    gene_presence_df = pd.DataFrame(
        {
            "number_of_samples_in_which_the_gene_is_detected": gene_sample_count.max(axis=1),
            "total_mcrs_count": gene_mcrs_count,
        }
    )

    # Sort by number of samples and break ties with total mcrs_count
    most_common_genes = gene_presence_df.sort_values(
        by=["number_of_samples_in_which_the_gene_is_detected", "total_mcrs_count"],
        ascending=False,
    )

    # Write results to file
    with open(stats_file, "a") as f:
        gene_names_top_n = most_common_genes.index[:top_values].tolist()
        f.write(f"Genes present across the most samples: {', '.join(gene_names_top_n)}\n")

    with open(f"{specific_stats_folder}/genes_present_across_the_most_samples.txt", "w") as f:
        f.write("Gene\tNumber_of_Samples\tTotal_mcrs_count\n")
        for gene, row in most_common_genes.iterrows():
            f.write(f"{gene}\t{row['number_of_samples_in_which_the_gene_is_detected']}\t{row['total_mcrs_count']}\n")

    # 6. Top 10 Genes with Highest mcrs_count Across All Samples
    # Sort genes by total mcrs_count
    top_genes_mcrs_count = gene_presence_df.sort_values(by="total_mcrs_count", ascending=False)

    # Write top genes with highest mcrs_count to stats file and detailed file
    with open(stats_file, "a") as f:
        gene_names_top_n = top_genes_mcrs_count.index[:top_values].tolist()
        f.write(f"Genes with highest mcrs_count across all samples: {', '.join(gene_names_top_n)}\n")

    with open(f"{specific_stats_folder}/genes_highest_mcrs_count.txt", "w") as f:
        f.write("Gene\tNumber_of_Samples\tTotal_mcrs_count\n")
        for gene, row in top_genes_mcrs_count.iterrows():
            f.write(f"{gene}\t{row['number_of_samples_in_which_the_gene_is_detected']}\t{row['total_mcrs_count']}\n")

    # TODO: things to add
    # differentially expressed mutations/mutated genes
    # VAF - learn how transipedia calculated VAF from RNA data and incorporate this here
    # have a list of genes of interest as optional input, and if provided then output a csv with which of these mutations were found and a list of additional interesting info for each gene (including the number of cells in which this mutation was found in bulk - VAF (variant allele frequency))
    # bulk: log1p, pca - sc.pp.log1p(adata), sc.tl.pca(adata)
    # plot line plots/heatmaps from notebook 3
