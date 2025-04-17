"""varseek summarize and specific helper functions."""

import logging
import os
import time
import re
import numpy as np
from pathlib import Path

import anndata
import anndata as ad
import pandas as pd

from varseek.utils import (
    check_file_path_is_string_with_valid_extension,
    is_valid_int,
    make_function_parameter_to_value_dict,
    get_varseek_dry_run,
    report_time_elapsed,
    save_params_to_config_file,
    save_run_info,
    set_up_logger,
    set_varseek_logging_level_and_filehandler,
    plot_items_descending_order,
    plot_histogram_with_zero_value,
    plot_cdna_locations,
    add_information_from_variant_header_to_adata_var_exploded,
    plot_substitution_heatmap,
    plot_variant_types,
    load_df_types_adata
)

from .constants import technology_valid_values, HGVS_pattern_general

logger = logging.getLogger(__name__)
logger = set_up_logger(logger, logging_level="INFO", save_logs=False, log_dir=None)


def validate_input_summarize(params_dict):
    adata = params_dict["adata"]
    if not isinstance(adata, (str, Path, anndata.AnnData)):
        raise TypeError("adata must be a string (file path) or an AnnData object.")
    if isinstance(adata, (str, Path)):
        check_file_path_is_string_with_valid_extension(adata, "adata", "h5ad")  # I will enforce that adata exists later, as otherwise it will throw an error when I call this through vk count before kb count/vk clean can run

    if not is_valid_int(params_dict["top_values"], ">=", 1):
        raise ValueError(f"top_values must be an positive integer. Got {params_dict.get('top_values')}.")

    technology = params_dict.get("technology", None)
    technology_valid_values_lower = {x.lower() for x in technology_valid_values}
    if technology is not None:
        if technology.lower() not in technology_valid_values_lower:
            raise ValueError(f"Technology must be None or one of {technology_valid_values_lower}")

    if not isinstance(params_dict["gene_name_column"], (str, type(None))):
        raise ValueError("gene_name_column must be a string or None. Got {params_dict.get('gene_name_column')}.")

    if not isinstance(params_dict["out"], (str, Path)):
        raise ValueError("out must be a string or Path object.")

    for param_name in ["dry_run", "overwrite"]:
        if not isinstance(params_dict.get(param_name), bool):
            raise ValueError(f"{param_name} must be a boolean. Got {param_name} of type {type(params_dict.get(param_name))}.")

@report_time_elapsed
def summarize(
    adata,
    top_values=10,
    technology=None,
    gene_name_column=None,
    out=".",
    dry_run=False,
    overwrite=False,
    logging_level=None,
    save_logs=False,
    log_out_dir=None,
    plot_strand_bias=False,
    strand_bias_end="auto",
    cdna_fasta=None,
    seq_id_cdna_column="seq_ID",
    start_variant_position_cdna_column="start_variant_position",
    end_variant_position_cdna_column="end_variant_position",
    read_length=None,
    **kwargs,
):
    """
    Summarize the results of the varseek analysis.

    # Required input arguments:
    - adata                             (str or Anndata) Anndata object or path to h5ad file.

    # Optional input arguments:
    - top_values                        (int) Number of top values to report. Default: 10
    - technology                        (str) Technology used to generate the data. To see list of spported technologies, run `kb --list`. For the purposes of this function, the only distinction that matters is bulk vs. non-bulk. Default: None
    - gene_name_column                  (str) Column name in adata.var that contains the gene names. Default: None (skip any uses of this).
    - plot_strand_bias                   (bool) Whether to plot strand bias. Default: False
    - out                               (str) Output directory. Default: "."
    - dry_run                           (bool) If True, print the commands that would be run without actually running them. Default: False
    - overwrite                         (bool) Whether to overwrite existing files. Default: False
    - logging_level                     (str) Logging level. Can also be set with the environment variable VARSEEK_LOGGING_LEVEL. Default: INFO.
    - save_logs                         (True/False) Whether to save logs to a file. Default: False.
    - log_out_dir                       (str) Directory to save logs. Default: None (do not save logs).

    # Only if plot_strand_bias=True
    - strand_bias_end                  (str) End of the strand bias plot. Options: 'auto', 'both', '5p', '3p'. Default: auto (auto-detected by technology)
    - cdna_fasta                       (str) Path to the cDNA FASTA file. Only if strand_bias_end != '5p'. Default: None
    - seq_id_cdna_column               (str) Column name in adata.var that contains the sequence ID for cDNA. Default: "seq_ID"
    - start_variant_position_cdna_column   (str) Column name in adata.var that contains the start variant position for cDNA. Default: "start_variant_position"
    - end_variant_position_cdna_column     (str) Column name in adata.var that contains the end variant position for cDNA. Default: "end_variant_position"
    - read_length                      (int) Read length for the FASTQ data. Only used to mark a vertical line on the plot. Default: None


    # Hidden arguments (part of kwargs):
    - stats_file                        (str) Path to the stats file. Default: `out`/varseek_summarize_stats.txt
    - specific_stats_folder             (str) Path to the specific stats folder. Default: `out`/specific_stats
    - plots_folder                      (str) Path to the plots folder. Default: `out`/plots
    """
    # * 1. logger
    if save_logs and not log_out_dir:
        log_out_dir = os.path.join(out, "logs")
    set_varseek_logging_level_and_filehandler(logging_level=logging_level, save_logs=save_logs, log_dir=log_out_dir)

    # * 2. Type-checking
    params_dict = make_function_parameter_to_value_dict(1)
    validate_input_summarize(params_dict)

    if isinstance(adata, (str, Path)) and not os.path.isfile(adata) and not dry_run:  # only use os.path.isfile when I require that a directory already exists; checked outside validate_input_summarize to avoid raising issue when type-checking within vk count
        raise ValueError(f"adata file path {adata} does not exist.")

    # * 3. Dry-run
    if dry_run:
        print(get_varseek_dry_run(params_dict, function_name="summarize"))
        return

    # * 4. Save params to config file and run info file
    config_file = os.path.join(out, "config", "vk_summarize_config.json")
    save_params_to_config_file(params_dict, config_file)

    run_info_file = os.path.join(out, "config", "vk_summarize_run_info.txt")
    save_run_info(run_info_file, params_dict=params_dict, function_name="summarize")

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
        # adata_dtypes_file_base = adata.replace(".h5ad", "_dtypes")  # matches vk clean
        adata = ad.read_h5ad(adata)
        # adata = load_df_types_adata(adata, adata_dtypes_file_base)
    else:
        raise ValueError("adata must be a string (file path) or an AnnData object.")
    
    logger.info("Calculating summary statistics for variants and genes. See %s for results", stats_file)

    # 1. Number of Variants with Count > 0 in any Sample/Cell, and for bulk in particular, for each sample; then list the variants
    logger.info("1. Number of Variants with Count > 0 in any Sample/Cell, and for bulk in particular, for each sample; then list the variants")
    if "vcrs_count" not in adata.var.columns:
        adata.var["vcrs_count"] = adata.X.sum(axis=0).A1 if hasattr(adata.X, "A1") else adata.X.sum(axis=0).flatten()
    if "vcrs_detected" not in adata.var.columns:
        adata.var["vcrs_detected"] = adata.var["vcrs_count"] > 0

    # Sort by vcrs_count
    vcrs_count_descending = adata.var.sort_values(
        by=["vcrs_count"],
        ascending=False,
    )

    vcrs_count_descending_greater_than_zero = vcrs_count_descending.loc[vcrs_count_descending["vcrs_count"] > 0]  # get all values greater than zero
    top_values_for_vcrs_count_descending = min(top_values, len(vcrs_count_descending_greater_than_zero))  # in case there are fewer than top_values variants with count > 0
    vcrs_count_descending_top_n = vcrs_count_descending_greater_than_zero.index.tolist()[:top_values_for_vcrs_count_descending]  # get top values

    with open(stats_file, "w", encoding="utf-8") as f:
        f.write(f"Total variants with count > 0 for any sample/cell: {len(vcrs_count_descending_greater_than_zero)}\n")
        if technology.lower() == "bulk" and len(adata.obs_names) < 100:  # make sure this is not too long (eg smart-seq, or a ton of bulk samples)
            for sample in adata.obs_names:
                count_nonzero_variants = (adata[sample, :].X > 0).sum()
                f.write(f"Sample {sample} has {count_nonzero_variants} variants with count > 0.\n")
        f.write(f"Variants with highest cumulative counts: {', '.join(vcrs_count_descending_top_n)}\n")

    with open(f"{specific_stats_folder}/variants_with_any_count.txt", "w", encoding="utf-8") as f:
        f.write("Variant\tTotal_Counts\n")
        for variant in vcrs_count_descending_greater_than_zero.index:
            total_counts = adata.var.loc[variant, "vcrs_count"]
            f.write(f"{variant}\t{total_counts}\n")

    skip_plots = False
    if "vcrs_header" not in adata.var.columns:
        adata.var["vcrs_header"] = adata.var.index
        first_vcrs_header = adata.var["vcrs_header"].iloc[0].split(';')[0]
        if not re.fullmatch(HGVS_pattern_general, first_vcrs_header):
            logger.warning("Please run vk clean, or add a column vcrs_header to adata.var with the variant headers in HGVS format to make all plots.")
            skip_plots = True
    
    x_column = "vcrs_header_with_gene_name" if "vcrs_header_with_gene_name" in adata.var.columns else "vcrs_id"
    plot_items_descending_order(adata.var, x_column = x_column, y_column = 'vcrs_count', item_range = (0,top_values), show_names=True, xlabel = "Variant", title = f"Top {top_values} Variants by Counts across All Samples", figsize = (15, 7), show=False, save_path=os.path.join(plots_folder, f"top_{top_values}_variants_descending_plot.png"))
    plot_items_descending_order(adata.var, x_column = x_column, y_column = 'vcrs_count', show_names=False, xlabel = "Variant Index", title = "Top Variants by Counts across All Samples", figsize = (15, 7), show=False, save_path=os.path.join(plots_folder, "variants_descending_plot.png"))
    plot_histogram_with_zero_value(adata.var, col = "vcrs_count", save_path = os.path.join(plots_folder, "variants_histogram.png"))

    if not skip_plots:
        adata_var_with_alignment = adata.var.loc[adata.var["vcrs_count"] > 0].copy() if "vcrs_count" in adata.var.columns else adata.var.copy()
        if plot_strand_bias:
            if seq_id_cdna_column not in adata_var_with_alignment.columns or start_variant_position_cdna_column not in adata_var_with_alignment.columns or end_variant_position_cdna_column not in adata_var_with_alignment.columns:
                logger.info("Adding information from variant header to a copy of adata.var. Note: this assumes vcrs_header transcripts and positions are accurate for cDNA")
                adata_var_with_alignment = add_information_from_variant_header_to_adata_var_exploded(adata_var_with_alignment, vcrs_header_individual_column="vcrs_header", seq_id_column=seq_id_cdna_column, var_column="variant", variant_source="placeholder", include_position_information=True, include_gene_information=False)
                adata_var_with_alignment.rename(columns={"start_variant_position": start_variant_position_cdna_column, "end_variant_position": end_variant_position_cdna_column}, inplace=True)
            
            plot_cdna_locations(adata_var_with_alignment, cdna_fasta=cdna_fasta, seq_id_column=seq_id_cdna_column, start_variant_position_cdna_column=start_variant_position_cdna_column, end_variant_position_cdna_column=end_variant_position_cdna_column, sequence_side=strand_bias_end, log_x=True, log_y=True, read_length_cutoff=read_length, save_path = os.path.join(plots_folder, f"strand_bias_{strand_bias_end}.png"))

        plot_substitution_heatmap(adata_var_with_alignment, variant_header_column="vcrs_header", count_column="vcrs_count", output_file=os.path.join(plots_folder, "substitutions_with_vcrs_count.png"), show=False, plot_type="bar")
        plot_substitution_heatmap(adata_var_with_alignment, variant_header_column="vcrs_header", count_column="vcrs_detected", output_file=os.path.join(plots_folder, "substitutions_with_vcrs_detected.png"), show=False, plot_type="bar")
        plot_variant_types(adata_var_with_alignment, variant_header_column="vcrs_header", variant_type_column = "variant_type", count_column="vcrs_count", output_file=os.path.join(plots_folder, "variant_type_with_vcrs_detected.png"), show=False)
        plot_variant_types(adata_var_with_alignment, variant_header_column="vcrs_header", variant_type_column = "variant_type", count_column="vcrs_detected", output_file=os.path.join(plots_folder, "variant_type_with_vcrs_detected.png"), show=False)

    # 2. Variants Present Across the Most Samples
    logger.info("2. Variants Present Across the Most Samples")
    if "number_of_samples_in_which_the_variant_is_detected" not in adata.var.columns:
        adata.var["number_of_samples_in_which_the_variant_is_detected"] = (adata.X > 0).sum(axis=0).A1 if hasattr(adata.X, "A1") else (adata.X > 0).sum(axis=0).A1

    # Sort by number of samples and break ties with vcrs_count
    number_of_samples_descending = adata.var.sort_values(
        by=["number_of_samples_in_which_the_variant_is_detected", "vcrs_count"],
        ascending=False,
    )

    number_of_samples_descending_greater_than_zero = number_of_samples_descending.loc[number_of_samples_descending["number_of_samples_in_which_the_variant_is_detected"] > 0]  # get all values greater than zero
    top_values_for_number_of_samples_descending = min(top_values, len(number_of_samples_descending_greater_than_zero))  # in case there are fewer than top_values variants with count > 0
    number_of_samples_descending_top_n = number_of_samples_descending.index.tolist()[:top_values_for_number_of_samples_descending]  # get top values

    with open(stats_file, "a", encoding="utf-8") as f:
        f.write(f"Variants present across the most samples: {', '.join(number_of_samples_descending_top_n)}\n")

    with open(f"{specific_stats_folder}/variants_present_across_the_most_samples.txt", "w", encoding="utf-8") as f:
        f.write("Variant\tNumber_of_Samples\tTotal_Counts\n")
        for variant in number_of_samples_descending_greater_than_zero.index:
            number_of_samples = adata.var.loc[variant, "number_of_samples_in_which_the_variant_is_detected"]
            total_counts = adata.var.loc[variant, "vcrs_count"]
            f.write(f"{variant}\t{number_of_samples}\t{total_counts}\n")

    # --------------------------------------------------------------------------------------------------------
    if gene_name_column:
        gene_counts = adata.var.groupby(gene_name_column)["vcrs_count"].sum()

        # 3. Number of Genes with Count > 0 in any Sample/Cell, and for bulk in particular, for each sample; then list the genes
        logger.info("3. Number of Genes with Count > 0 in any Sample/Cell, and for bulk in particular, for each sample; then list the genes")
        # Sort by vcrs_count
        vcrs_count_descending = gene_counts.var.sort_values(
            by=["vcrs_count"],
            ascending=False,
        )

        vcrs_count_descending_greater_than_zero = vcrs_count_descending.loc[vcrs_count_descending["vcrs_count"] > 0]  # get all values greater than zero
        top_values_for_vcrs_count_descending = min(top_values, len(vcrs_count_descending_greater_than_zero))  # in case there are fewer than top_values variants with count > 0
        vcrs_count_descending_top_n = vcrs_count_descending_greater_than_zero.index.tolist()[:top_values_for_vcrs_count_descending]  # get top values

        with open(stats_file, "a", encoding="utf-8") as f:
            f.write(f"Total genes with count > 0 for any sample/cell: {len(vcrs_count_descending_greater_than_zero)}\n")
            if technology.lower() == "bulk" and len(gene_counts.obs_names) < 100:  # make sure this is not too long (eg smart-seq, or a ton of bulk samples)
                for sample in gene_counts.obs_names:  #!!! make sure this is right
                    count_nonzero_variants = (gene_counts[sample, :].X > 0).sum()
                    f.write(f"Sample {sample} has {count_nonzero_variants} genes with count > 0.\n")
            f.write(f"Genes with highest cumulative counts: {', '.join(vcrs_count_descending_top_n)}\n")

        with open(f"{specific_stats_folder}/genes_with_any_count.txt", "w", encoding="utf-8") as f:
            f.write("Gene\tTotal_Counts\n")
            for gene in vcrs_count_descending_greater_than_zero.index:
                total_counts = adata.var.loc[gene, "vcrs_count"]
                f.write(f"{gene}\t{total_counts}\n")

        # 4. Genes Present Across the Most Samples
        logger.info("4. Genes Present Across the Most Samples")

        number_of_samples_descending = gene_counts.var.sort_values(
            by=["number_of_samples_in_which_the_variant_is_detected", "vcrs_count"],
            ascending=False,
        )

        number_of_samples_descending_greater_than_zero = number_of_samples_descending.loc[number_of_samples_descending["number_of_samples_in_which_the_variant_is_detected"] > 0]  # get all values greater than zero
        top_values_for_number_of_samples_descending = min(top_values, len(number_of_samples_descending_greater_than_zero))  # in case there are fewer than top_values variants with count > 0
        number_of_samples_descending_top_n = number_of_samples_descending.index.tolist()[:top_values_for_number_of_samples_descending]  # get top values

        with open(stats_file, "a", encoding="utf-8") as f:
            f.write(f"Genes present across the most samples: {', '.join(number_of_samples_descending_top_n)}\n")

        with open(f"{specific_stats_folder}/genes_present_across_the_most_samples.txt", "w", encoding="utf-8") as f:
            f.write("Variant\tNumber_of_Samples\tTotal_Counts\n")
            for variant in number_of_samples_descending_greater_than_zero.index:
                number_of_samples = gene_counts.var.loc[variant, "number_of_samples_in_which_the_variant_is_detected"]
                total_counts = gene_counts.var.loc[variant, "vcrs_count"]
                f.write(f"{variant}\t{number_of_samples}\t{total_counts}\n")

    # TODO: things to add
    # differentially expressed variants/mutated genes
    # VAF - learn how transipedia calculated VAF from RNA data and incorporate this here
    # have a list of genes of interest as optional input, and if provided then output a csv with which of these variants were found and a list of additional interesting info for each gene (including the number of cells in which this variant was found in bulk - VAF (variant allele frequency))
    # bulk: log1p, pca - sc.pp.log1p(adata), sc.tl.pca(adata)
    # plot line plots/heatmaps from notebook 3
