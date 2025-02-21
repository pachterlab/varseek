"""varseek summarize and specific helper functions."""

import os
import time

import anndata
import pandas as pd
from pathlib import Path
import anndata as ad
import logging

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

logger = logging.getLogger(__name__)


def validate_input_summarize(params_dict):
    adata = params_dict["adata"]
    if not isinstance(adata, (str, Path, anndata.AnnData)):
        raise TypeError("adata must be a string (file path) or an AnnData object.")
    if isinstance(adata, (str, Path)):
        check_file_path_is_string_with_valid_extension(adata, "adata", "h5ad")    # I will enforce that adata exists later, as otherwise it will throw an error when I call this through vk count before kb count/vk clean can run

    if not is_valid_int(params_dict["top_values"], ">=", 1):
        raise ValueError(f"top_values must be an positive integer. Got {params_dict.get('top_values')}.")

    technology = params_dict.get("technology", None)
    technology_valid_values_lower = {x.lower() for x in technology_valid_values}
    if technology is not None:
        if technology.lower() not in technology_valid_values_lower:
            raise ValueError(f"Technology must be None or one of {technology_valid_values_lower}")

    if not isinstance(params_dict["out"], (str, Path)):
        raise ValueError("out must be a string or Path object.")

    if not isinstance(params_dict["vcrs_id_column"], str):
        raise ValueError("vcrs_id_column must be a string.")

    for param_name in ["dry_run", "overwrite"]:
        if not isinstance(params_dict.get(param_name), bool):
            raise ValueError(f"{param_name} must be a boolean. Got {param_name} of type {type(params_dict.get(param_name))}.")


def summarize(
    adata,
    top_values=10,
    technology=None,
    vcrs_header_column="vcrs_header",
    vcrs_id_column="vcrs_id",
    out=".",
    dry_run=False,
    overwrite=False,
    logging_level=None,
    save_logs=False,
    log_out_dir=None,
    **kwargs,
):
    """
    Summarize the results of the varseek analysis.

    # Required input arguments:
    - adata                             (str or Anndata) Anndata object or path to h5ad file.

    # Optional input arguments:
    - top_values                        (int) Number of top values to report. Default: 10
    - technology                        (str) Technology used to generate the data. To see list of spported technologies, run `kb --list`. For the purposes of this function, the only distinction that matters is bulk vs. non-bulk. Default: None
    - vcrs_id_column                    (str) Column name in adata.var that contains the vcrs_id. Default: "vcrs_id"
    - out                               (str) Output directory. Default: "."
    - dry_run                           (bool) If True, print the commands that would be run without actually running them. Default: False
    - overwrite                         (bool) Whether to overwrite existing files. Default: False
    - logging_level                     (str) Logging level. Can also be set with the environment variable VARSEEK_LOGGING_LEVEL. Default: INFO.
    - save_logs                         (True/False) Whether to save logs to a file. Default: False.
    - log_out_dir                       (str) Directory to save logs. Default: None (do not save logs).

    # Hidden arguments (part of kwargs):
    - stats_file                        (str) Path to the stats file. Default: `out`/varseek_summarize_stats.txt
    - specific_stats_folder             (str) Path to the specific stats folder. Default: `out`/specific_stats
    - plots_folder                      (str) Path to the plots folder. Default: `out`/plots
    """

    # * 1. Start timer
    start_time = time.perf_counter()

    # * 1.5. logger
    global logger
    if kwargs.get("logger") and isinstance(kwargs.get("logger"), logging.Logger):
        logger = kwargs.get("logger")
    else:
        if save_logs and not log_out_dir:
            log_out_dir = os.path.join(out, "logs")
        logger = set_up_logger(logger, logging_level=logging_level, save_logs=save_logs, log_dir=log_out_dir)

    # * 2. Type-checking
    params_dict = make_function_parameter_to_value_dict(1)
    validate_input_summarize(params_dict)

    if isinstance(adata, (str, Path)) and not os.path.isfile(adata) and not dry_run:  # only use os.path.isfile when I require that a directory already exists; checked outside validate_input_summarize to avoid raising issue when type-checking within vk count
        raise ValueError(f"adata file path {adata} does not exist.")

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

    # * 7.5 make sure ints are ints
    top_values = int(top_values)

    # * 8. Start the actual function
    if isinstance(adata, anndata.AnnData):
        pass
    elif isinstance(adata, str):
        adata = ad.read_h5ad(adata)
    else:
        raise ValueError("adata must be a string (file path) or an AnnData object.")

    if vcrs_id_column not in adata.var.columns:
        adata.var[vcrs_id_column] = adata.var_names

    adata.var_names = adata.var[vcrs_header_column]

    if "vcrs_count" not in adata.var.columns:
        adata.var["vcrs_count"] = adata.X.sum(axis=0).A1 if hasattr(adata.X, "A1") else adata.X.sum(axis=0).flatten()

    # 1. Number of Variants with Count > 0 in any Sample/Cell, and for bulk in particular, for each sample; then list the variants
    with open(stats_file, "w", encoding="utf-8") as f:
        variants_with_any_count = (adata.X > 0).sum(axis=0)
        variants_count_any_row = (variants_with_any_count > 0).sum()
        line = f"Total variants with count > 0 for any sample/cell: {variants_count_any_row}"
        f.write(line)
        if technology.lower() == "bulk":
            for sample in adata.obs_names:
                count_nonzero_variants = (adata[sample, :].X > 0).sum()
                line = f"Sample {sample} has {count_nonzero_variants} variants with count > 0."
                f.write(line)

    variants_with_nonzero_counts = adata.var_names[variants_with_any_count > 0]
    with open(f"{specific_stats_folder}/variants_with_any_count.txt", "w", encoding="utf-8") as f:
        for variant in variants_with_nonzero_counts:
            f.write(f"{variant}\n")

    # 2. Variants Present Across the Most Samples
    adata.var["number_of_samples_in_which_the_variant_is_detected"] = (adata.X > 0).sum(axis=0).A1 if hasattr(adata.X, "A1") else (adata.X > 0).sum(axis=0)
    # Sort by number of samples and break ties with vcrs_count
    most_common_variants = adata.var.sort_values(
        by=["number_of_samples_in_which_the_variant_is_detected", "vcrs_count"],
        ascending=False,
    )
    variant_names = most_common_variants.index.tolist()
    variant_names_top_n = variant_names[:top_values]
    with open(stats_file, "a", encoding="utf-8") as f:
        f.write(f"Variants present across the most samples: {', '.join(variant_names_top_n)}")
    with open(f"{specific_stats_folder}/variants_present_across_the_most_samples.txt", "w", encoding="utf-8") as f:
        f.write("Variant\tNumber_of_Samples\tTotal_Counts\n")

        # Write each variant's details
        for variant in most_common_variants.index:
            number_of_samples = adata.var.loc[variant, "number_of_samples_in_which_the_variant_is_detected"]
            total_counts = adata.var.loc[variant, "vcrs_count"]
            f.write(f"{variant}\t{number_of_samples}\t{total_counts}\n")

    # 3. Top 10 Variants with Highest vcrs_count Across All Samples
    top_variants_vcrs_count = adata.var.sort_values(by="vcrs_count", ascending=False)
    variant_names = top_variants_vcrs_count.index.tolist()
    variant_names_top_n = variant_names[:top_values]
    with open(stats_file, "a", encoding="utf-8") as f:
        f.write(f"Variants with highest counts across all samples: {', '.join(variant_names_top_n)}")
    with open(f"{specific_stats_folder}/variants_highest_vcrs_count.txt", "w", encoding="utf-8") as f:
        f.write("Variant\tNumber_of_Samples\tTotal_Counts\n")

        # Write each variant's details
        for variant in top_variants_vcrs_count.index:
            number_of_samples = adata.var.loc[variant, "number_of_samples_in_which_the_variant_is_detected"]
            total_counts = adata.var.loc[variant, "vcrs_count"]
            f.write(f"{variant}\t{number_of_samples}\t{total_counts}\n")

    # --------------------------------------------------------------------------------------------------------
    # 4. Number of Genes with Count > 0 in any Sample/Cell, and for bulk in particular, for each sample; then list the genes
    with open(stats_file, "a", encoding="utf-8") as f:
        # Sum vcrs_count for each gene
        gene_counts = adata.var.groupby("gene_name")["vcrs_count"].sum()

        # Count genes with non-zero vcrs_count across all samples
        genes_count_any_row = (gene_counts > 0).sum()
        line = f"Total genes with count > 0 in any sample/cell: {genes_count_any_row}\n"
        f.write(line)

        # For bulk technologys, calculate counts for each sample
        if technology.lower() == "bulk":
            for sample in adata.obs_names:
                # Calculate vcrs_count per gene for the specific sample
                gene_counts_per_sample = adata[sample, :].to_df().gt(0).groupby(adata.var["gene_name"], axis=1).sum().gt(0).sum()
                line = f"Sample {sample} has {gene_counts_per_sample.sum()} genes with count > 0.\n"
                f.write(line)

    # List of genes with non-zero vcrs_count across all samples
    genes_with_nonzero_counts = gene_counts[gene_counts > 0].index
    with open(f"{specific_stats_folder}/genes_with_any_count.txt", "w", encoding="utf-8") as f:
        for gene in genes_with_nonzero_counts:
            f.write(f"{gene}\n")

    # 5. Genes Present Across the Most Samples
    # Calculate the number of samples where each gene has at least one variant with count > 0
    adata.var["detected"] = (adata.X > 0).astype(int)  # Binary matrix indicating presence per sample
    gene_sample_count = adata.var.groupby("gene_name")["detected"].sum()  # Sum across variants per gene

    # Add the total vcrs_count per gene (summing across all variants for that gene)
    gene_vcrs_count = adata.var.groupby("gene_name")["vcrs_count"].sum()

    # Combine both into a DataFrame for sorting
    gene_presence_df = pd.DataFrame(
        {
            "number_of_samples_in_which_the_gene_is_detected": gene_sample_count.max(axis=1),
            "total_vcrs_count": gene_vcrs_count,
        }
    )

    # Sort by number of samples and break ties with total vcrs_count
    most_common_genes = gene_presence_df.sort_values(
        by=["number_of_samples_in_which_the_gene_is_detected", "total_vcrs_count"],
        ascending=False,
    )

    # Write results to file
    with open(stats_file, "a", encoding="utf-8") as f:
        gene_names_top_n = most_common_genes.index[:top_values].tolist()
        f.write(f"Genes present across the most samples: {', '.join(gene_names_top_n)}\n")

    with open(f"{specific_stats_folder}/genes_present_across_the_most_samples.txt", "w", encoding="utf-8") as f:
        f.write("Gene\tNumber_of_Samples\tTotal_vcrs_count\n")
        for gene, row in most_common_genes.iterrows():
            f.write(f"{gene}\t{row['number_of_samples_in_which_the_gene_is_detected']}\t{row['total_vcrs_count']}\n")

    # 6. Top 10 Genes with Highest vcrs_count Across All Samples
    # Sort genes by total vcrs_count
    top_genes_vcrs_count = gene_presence_df.sort_values(by="total_vcrs_count", ascending=False)

    # Write top genes with highest vcrs_count to stats file and detailed file
    with open(stats_file, "a", encoding="utf-8") as f:
        gene_names_top_n = top_genes_vcrs_count.index[:top_values].tolist()
        f.write(f"Genes with highest vcrs_count across all samples: {', '.join(gene_names_top_n)}\n")

    with open(f"{specific_stats_folder}/genes_highest_vcrs_count.txt", "w", encoding="utf-8") as f:
        f.write("Gene\tNumber_of_Samples\tTotal_vcrs_count\n")
        for gene, row in top_genes_vcrs_count.iterrows():
            f.write(f"{gene}\t{row['number_of_samples_in_which_the_gene_is_detected']}\t{row['total_vcrs_count']}\n")

    report_time_elapsed(start_time, logger=logger, function_name="summarize")

    # TODO: things to add
    # differentially expressed variants/mutated genes
    # VAF - learn how transipedia calculated VAF from RNA data and incorporate this here
    # have a list of genes of interest as optional input, and if provided then output a csv with which of these variants were found and a list of additional interesting info for each gene (including the number of cells in which this variant was found in bulk - VAF (variant allele frequency))
    # bulk: log1p, pca - sc.pp.log1p(adata), sc.tl.pca(adata)
    # plot line plots/heatmaps from notebook 3
