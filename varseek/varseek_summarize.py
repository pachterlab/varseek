"""varseek summarize and specific helper functions."""
import os
import time

import anndata
import pandas as pd
import scanpy as sc

from varseek.utils import (
    check_file_path_is_string_with_valid_extension,
    is_valid_int,
    make_function_parameter_to_value_dict,
    print_varseek_dry_run,
    report_time_elapsed,
    save_params_to_config_file,
    save_run_info,
    set_up_logger,
)

from .constants import technology_valid_values

logger = set_up_logger()


def validate_input_summarize(params_dict):
    adata = params_dict["adata"]
    if not isinstance(adata, (str, anndata.AnnData)):
        raise TypeError("adata must be a string (file path) or an AnnData object.")
    if isinstance(adata, str):
        check_file_path_is_string_with_valid_extension(adata, adata, "h5ad")
        if not os.path.isfile(adata):  # ensure that all fastq files exist
            raise ValueError(f"File {adata} does not exist")

    if not is_valid_int(params_dict["top_values"], ">=", 1):
        raise ValueError(f"top_values must be an positive integer. Got {params_dict.get('top_values')}.")

    technology = params_dict.get("technology", None)
    technology_valid_values_lower = {x.lower() for x in technology_valid_values}
    if technology is not None:
        if technology.lower() not in technology_valid_values_lower:
            raise ValueError(f"Technology must be None or one of {technology_valid_values_lower}")

    if not isinstance(params_dict["out"], str):
        raise ValueError("out must be a string.")

    for param_name in ["dry_run", "overwrite", "verbose"]:
        if not isinstance(params_dict.get(param_name), bool):
            raise ValueError(f"{param_name} must be a boolean. Got {param_name} of type {type(params_dict.get(param_name))}.")


def summarize(
    adata,
    top_values=10,
    technology=None,
    out=".",
    dry_run=False,
    overwrite=False,
    verbose=True,
    **kwargs,
):
    """
    Summarize the results of the varseek analysis.

    # Required input arguments:
    - adata                             (str or Anndata) Anndata object or path to h5ad file.

    # Optional input arguments:
    - top_values                        (int) Number of top values to report. Default: 10
    - technology                        (str) Technology used to generate the data. To see list of spported technologies, run `kb --list`. For the purposes of this function, the only distinction that matters is bulk vs. non-bulk. Default: None
    - out                               (str) Output directory. Default: "."
    - dry_run                           (bool) If True, print the commands that would be run without actually running them. Default: False
    - overwrite                         (bool) Whether to overwrite existing files. Default: False
    - verbose                           (bool) Whether to print progress messages. Default: True

    # Hidden arguments (part of kwargs):
    - stats_file                        (str) Path to the stats file. Default: `out`/varseek_summarize_stats.txt
    - specific_stats_folder             (str) Path to the specific stats folder. Default: `out`/specific_stats
    - plots_folder                      (str) Path to the plots folder. Default: `out`/plots
    """

    # * 1. Start timer
    start_time = time.perf_counter()

    # * 2. Type-checking
    params_dict = make_function_parameter_to_value_dict(1)
    validate_input_summarize(params_dict)

    # * 3. Dry-run
    if dry_run:
        print_varseek_dry_run(params_dict, function_name="summarize")
        return

    # * 4. Save params to config file and run info file
    config_file = os.path.join(out, "config", "vk_summarize_config.json")
    save_params_to_config_file(params_dict, config_file)

    run_info_file = os.path.join(out, "config", "vk_summarize_run_info.txt")
    save_run_info(run_info_file)

    # * 5. Set up default folder/file input paths, and make sure the necessary ones exist
    # all input files for vk summarize are required in the varseek workflow, so this is skipped

    # * 6. Set up default folder/file output paths, and make sure they don't exist unless overwrite=True
    stats_file = os.path.join(out, "varseek_summarize_stats.txt") if not kwargs.get("stats_file") else kwargs["stats_file"]
    specific_stats_folder = os.path.join(out, "specific_stats") if not kwargs.get("specific_stats_folder") else kwargs["specific_stats_folder"]
    plots_folder = os.path.join(out, "plots") if not kwargs.get("plots_folder") else kwargs["plots_folder"]

    if not overwrite:
        for output_path in [stats_file, specific_stats_folder, plots_folder]:
            if os.path.exists(output_path):
                raise FileExistsError(f"Path {output_path} already exists. Please delete it or specify a different output directory.")

    os.makedirs(out, exist_ok=True)
    os.makedirs(specific_stats_folder, exist_ok=True)
    os.makedirs(plots_folder, exist_ok=True)

    # * 7. Define kwargs defaults
    # no kwargs

    # * 8. Start the actual function
    if isinstance(adata, anndata.AnnData):
        pass
    elif isinstance(adata, str):
        adata = sc.read_h5ad(adata)
    else:
        raise ValueError("adata must be a string (file path) or an AnnData object.")

    if "mcrs_id" not in adata.var.columns:
        adata.var["mcrs_id"] = adata.var_names

    adata.var_names = adata.var["mcrs_header"]

    if "mcrs_count" not in adata.var.columns:
        adata.var["mcrs_count"] = adata.X.sum(axis=0).A1 if hasattr(adata.X, "A1") else adata.X.sum(axis=0).flatten()

    # 1. Number of Mutations with Count > 0 in any Sample/Cell, and for bulk in particular, for each sample; then list the mutations
    with open(stats_file, "w", encoding="utf-8") as f:
        mutations_with_any_count = (adata.X > 0).sum(axis=0)
        mutations_count_any_row = (mutations_with_any_count > 0).sum()
        line = f"Total mutations with count > 0 for any sample/cell: {mutations_count_any_row}"
        f.write(line)
        if technology.lower() == "bulk":
            for sample in adata.obs_names:
                count_nonzero_mutations = (adata[sample, :].X > 0).sum()
                line = f"Sample {sample} has {count_nonzero_mutations} mutations with count > 0."
                f.write(line)

    mutations_with_nonzero_counts = adata.var_names[mutations_with_any_count > 0]
    with open(f"{specific_stats_folder}/mutations_with_any_count.txt", "w", encoding="utf-8") as f:
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
    with open(stats_file, "a", encoding="utf-8") as f:
        f.write(f"Mutations present across the most samples: {', '.join(mutation_names_top_n)}")
    with open(f"{specific_stats_folder}/mutations_present_across_the_most_samples.txt", "w", encoding="utf-8") as f:
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
    with open(stats_file, "a", encoding="utf-8") as f:
        f.write(f"Mutations with highest counts across all samples: {', '.join(mutation_names_top_n)}")
    with open(f"{specific_stats_folder}/mutations_highest_mcrs_count.txt", "w", encoding="utf-8") as f:
        f.write("Mutation\tNumber_of_Samples\tTotal_Counts\n")

        # Write each mutation's details
        for mutation in top_mutations_mcrs_count.index:
            number_of_samples = adata.var.loc[mutation, "number_of_samples_in_which_the_mutation_is_detected"]
            total_counts = adata.var.loc[mutation, "mcrs_count"]
            f.write(f"{mutation}\t{number_of_samples}\t{total_counts}\n")

    # --------------------------------------------------------------------------------------------------------
    # 4. Number of Genes with Count > 0 in any Sample/Cell, and for bulk in particular, for each sample; then list the genes
    with open(stats_file, "a", encoding="utf-8") as f:
        # Sum mcrs_count for each gene
        gene_counts = adata.var.groupby("gene_name")["mcrs_count"].sum()

        # Count genes with non-zero mcrs_count across all samples
        genes_count_any_row = (gene_counts > 0).sum()
        line = f"Total genes with count > 0 in any sample/cell: {genes_count_any_row}\n"
        f.write(line)

        # For bulk technologys, calculate counts for each sample
        if technology.lower() == "bulk":
            for sample in adata.obs_names:
                # Calculate mcrs_count per gene for the specific sample
                gene_counts_per_sample = adata[sample, :].to_df().gt(0).groupby(adata.var["gene_name"], axis=1).sum().gt(0).sum()
                line = f"Sample {sample} has {gene_counts_per_sample.sum()} genes with count > 0.\n"
                f.write(line)

    # List of genes with non-zero mcrs_count across all samples
    genes_with_nonzero_counts = gene_counts[gene_counts > 0].index
    with open(f"{specific_stats_folder}/genes_with_any_count.txt", "w", encoding="utf-8") as f:
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
    with open(stats_file, "a", encoding="utf-8") as f:
        gene_names_top_n = most_common_genes.index[:top_values].tolist()
        f.write(f"Genes present across the most samples: {', '.join(gene_names_top_n)}\n")

    with open(f"{specific_stats_folder}/genes_present_across_the_most_samples.txt", "w", encoding="utf-8") as f:
        f.write("Gene\tNumber_of_Samples\tTotal_mcrs_count\n")
        for gene, row in most_common_genes.iterrows():
            f.write(f"{gene}\t{row['number_of_samples_in_which_the_gene_is_detected']}\t{row['total_mcrs_count']}\n")

    # 6. Top 10 Genes with Highest mcrs_count Across All Samples
    # Sort genes by total mcrs_count
    top_genes_mcrs_count = gene_presence_df.sort_values(by="total_mcrs_count", ascending=False)

    # Write top genes with highest mcrs_count to stats file and detailed file
    with open(stats_file, "a", encoding="utf-8") as f:
        gene_names_top_n = top_genes_mcrs_count.index[:top_values].tolist()
        f.write(f"Genes with highest mcrs_count across all samples: {', '.join(gene_names_top_n)}\n")

    with open(f"{specific_stats_folder}/genes_highest_mcrs_count.txt", "w", encoding="utf-8") as f:
        f.write("Gene\tNumber_of_Samples\tTotal_mcrs_count\n")
        for gene, row in top_genes_mcrs_count.iterrows():
            f.write(f"{gene}\t{row['number_of_samples_in_which_the_gene_is_detected']}\t{row['total_mcrs_count']}\n")

    report_time_elapsed(start_time, logger=logger, verbose=verbose, function_name="summarize")

    # TODO: things to add
    # differentially expressed mutations/mutated genes
    # VAF - learn how transipedia calculated VAF from RNA data and incorporate this here
    # have a list of genes of interest as optional input, and if provided then output a csv with which of these mutations were found and a list of additional interesting info for each gene (including the number of cells in which this mutation was found in bulk - VAF (variant allele frequency))
    # bulk: log1p, pca - sc.pp.log1p(adata), sc.tl.pca(adata)
    # plot line plots/heatmaps from notebook 3
