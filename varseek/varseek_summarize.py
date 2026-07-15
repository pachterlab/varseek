"""varseek summarize and specific helper functions."""

import logging
import os
import time
import re
import numpy as np
from pathlib import Path
from typing import Optional, Union

import anndata
import anndata as ad
import pandas as pd
from scipy import sparse as sp
from pydantic import BaseModel, ConfigDict, model_validator

from varseek.utils import (
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
    load_df_types_adata,
    load_adata_from_mtx,
    validate_call,
    vk_config,
    PositiveInt,
    Technology,
)

from .utils.varseek_summarize_utils import (
    _as_1d,
    _counts_per_obs,
    _variants_detected_per_obs,
    _detected_per_var,
    _fmt_int,
    _fmt_float,
    _distribution_line,
    _bucket_distribution,
    _explode_gene_stats,
    _top_table,
)

from .constants import HGVS_pattern_general

logger = logging.getLogger(__name__)
logger = set_up_logger(logger, logging_level="INFO", save_logs=False, log_dir=None)

# Advanced `summarize` parameters: still accepted on the command line (and in the Python signature,
# where @validate_call validates them), but hidden from `vk summarize --help` to keep it uncluttered.
# Consumed by main.py, which flips each matching argparse action's help to SUPPRESS. Matched by dest.
vk_summarize_hidden_from_help = {
    "stats_file",
    "specific_stats_folder",
    "plots_folder",
}


class SummarizeParams(BaseModel):
    """Validation for :func:`summarize` not covered by the typed signature.

    Per-parameter types (top_values, technology, gene_name_column, out, booleans)
    are enforced on the signature by ``@validate_call``; this model captures the
    polymorphic ``adata`` argument (str/Path/AnnData plus a ``.h5ad`` extension
    check when it is a path). Existence of ``adata`` is intentionally NOT checked
    here (it is enforced separately, after validation).
    """

    model_config = ConfigDict(extra="ignore", arbitrary_types_allowed=True)

    adata: object = None

    @model_validator(mode="after")
    def _validate(self):
        adata = self.adata
        if not isinstance(adata, (str, Path, anndata.AnnData)):
            raise TypeError("adata must be a string (file path) or an AnnData object.")
        if isinstance(adata, (str, Path)):
            adata = str(adata)
            if not adata.lower().endswith((".h5ad", ".h5")):  # I will enforce that adata exists later, as otherwise it will throw an error when I call this through vk count before kb count/vk clean can run
                raise ValueError("Invalid file extension for adata. Must be one of {'.h5ad', '.h5'}")
        return self


@report_time_elapsed
@validate_call(config=vk_config)
def summarize(
    adata: object,
    top_values: PositiveInt = 10,
    technology: Optional[Technology] = None,
    gene_name_column: Optional[str] = None,
    out: Union[str, Path] = ".",
    dry_run: bool = False,
    overwrite: bool = False,
    logging_level: Optional[Union[str, int]] = None,
    save_logs: bool = False,
    log_out_dir: Optional[Union[str, Path]] = None,
    plot_strand_bias: bool = False,
    strand_bias_end: str = "auto",
    cdna_fasta: Optional[str] = None,
    seq_id_cdna_column: str = "seq_ID",
    start_variant_position_cdna_column: str = "start_variant_position",
    end_variant_position_cdna_column: str = "end_variant_position",
    read_length: Optional[int] = None,
    # --- Advanced parameters: part of the Python signature (validated by @validate_call) but hidden from `vk summarize --help`. See the "Advanced parameters" docstring section. ---
    stats_file: Optional[Union[str, Path]] = None,
    specific_stats_folder: Optional[Union[str, Path]] = None,
    plots_folder: Optional[Union[str, Path]] = None,
):
    """
    Summarize the results of a varseek mutation screen.

    Produces a concise, human-readable overview of the screen (printed to the
    console and saved to a stats file), a set of per-item TSV tables, and plots.
    Headline metrics include the number of variants screened vs. detected, the
    distribution of variants and counts per cell/sample, the most frequently
    detected variants, and the genes carrying the most variants.

    # Required input arguments:
    - adata                             (str or Anndata) Anndata object or path to h5ad file (cells/samples x variants).

    # Optional input arguments:
    - top_values                        (int) Number of top values to report in tables/plots. Default: 10
    - technology                        (str) Technology used to generate the data. To see list of supported technologies, run `kb --list`. For the purposes of this function, the only distinction that matters is bulk vs. non-bulk (which changes the "cell" vs "sample" wording and enables a per-sample breakdown). Default: None
    - gene_name_column                  (str) Column name in adata.var that contains the gene identifiers. Default: None (auto-detect "gene_name" then "gene_id"; skip gene stats if neither is present).
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


    # # Advanced parameters (real `summarize` arguments, but hidden from the `vk summarize --help` CLI):
    - stats_file                        (str) Path to the stats file. Default: `out`/varseek_summarize_stats.txt
    - specific_stats_folder             (str) Path to the specific stats folder. Default: `out`/specific_stats
    - plots_folder                      (str) Path to the plots folder. Default: `out`/plots
    """
    # * 1. logger
    if save_logs and not log_out_dir:
        log_out_dir = os.path.join(out, "logs")
    set_varseek_logging_level_and_filehandler(logging_level=logging_level, save_logs=save_logs, log_dir=log_out_dir)

    # * 2. Type-checking (per-parameter types enforced by @validate_call; polymorphic adata by SummarizeParams)
    params_dict = make_function_parameter_to_value_dict(1)
    SummarizeParams(**params_dict)

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
    stats_file = stats_file or os.path.join(out, "varseek_summarize_stats.txt")
    specific_stats_folder = specific_stats_folder or os.path.join(out, "specific_stats")
    plots_folder = plots_folder or os.path.join(out, "plots")

    if not overwrite:
        for output_path in [stats_file, specific_stats_folder, plots_folder]:
            if os.path.exists(output_path):
                raise FileExistsError(f"Path {output_path} already exists. Please delete it or specify a different output directory.")

    os.makedirs(out, exist_ok=True)
    os.makedirs(specific_stats_folder, exist_ok=True)
    os.makedirs(plots_folder, exist_ok=True)


    # * 7.5 make sure ints are ints
    top_values = int(top_values)

    # * 8. Load adata
    if isinstance(adata, anndata.AnnData):
        pass
    elif isinstance(adata, str) and adata.endswith(".h5ad"):
        adata = ad.read_h5ad(adata)
    elif isinstance(adata, str) and adata.endswith(".mtx"):
        adata = load_adata_from_mtx(adata)
    else:
        raise ValueError("adata must be a string (file path) or an AnnData object.")

    # * 9. Normalize/derive the var-level columns we rely on (works even if vk clean wasn't run)
    if "vcrs_count" not in adata.var.columns:
        adata.var["vcrs_count"] = _as_1d(adata.X.sum(axis=0))
    adata.var["vcrs_count"] = adata.var["vcrs_count"].astype(float)
    if "vcrs_detected" not in adata.var.columns:
        adata.var["vcrs_detected"] = adata.var["vcrs_count"] > 0
    else:
        adata.var["vcrs_detected"] = adata.var["vcrs_detected"].astype(bool)
    # number of observations in which each variant is detected (recompute to be safe)
    adata.var["number_obs"] = _detected_per_var(adata.X).astype(int)

    # display name for each variant (prefer gene-annotated header, then HGVS header, then id/index)
    if "vcrs_header_with_gene_name" in adata.var.columns:
        adata.var["_display_name"] = adata.var["vcrs_header_with_gene_name"].astype(str)
    elif "vcrs_header" in adata.var.columns:
        adata.var["_display_name"] = adata.var["vcrs_header"].astype(str)
    else:
        adata.var["_display_name"] = adata.var.index.astype(str)

    # auto-detect a gene column if not explicitly provided
    gene_column = gene_name_column
    if gene_column is None:
        for candidate in ("gene_name", "gene_id"):
            if candidate in adata.var.columns:
                gene_column = candidate
                break
    elif gene_column not in adata.var.columns:
        logger.warning("gene_name_column '%s' not found in adata.var; skipping gene-level stats.", gene_column)
        gene_column = None

    is_bulk = technology is not None and str(technology).lower() == "bulk"
    obs_label = "sample" if is_bulk else "cell"
    obs_label_plural = "samples" if is_bulk else "cells"

    logger.info("Calculating summary statistics for the mutation screen. See %s for the full report.", stats_file)

    # * 10. Compute headline statistics
    counts_per_obs = _counts_per_obs(adata.X)
    variants_per_obs = _variants_detected_per_obs(adata.X)

    n_variants_panel = int(adata.n_vars)
    n_variants_detected = int((adata.var["vcrs_count"] > 0).sum())
    n_obs = int(adata.n_obs)
    total_counts = float(counts_per_obs.sum())
    pct_variants_detected = (100.0 * n_variants_detected / n_variants_panel) if n_variants_panel else 0.0
    n_obs_with_variant = int((variants_per_obs > 0).sum())
    pct_obs_with_variant = (100.0 * n_obs_with_variant / n_obs) if n_obs else 0.0

    # store per-obs metrics on adata.obs for downstream plotting/inspection
    adata.obs["n_variants_detected"] = variants_per_obs
    adata.obs["total_variant_counts"] = counts_per_obs

    # * 11. Build the report
    report_lines = []
    report_lines.append("=" * 66)
    report_lines.append("varseek mutation screening summary")
    report_lines.append("=" * 66)
    report_lines.append("")
    report_lines.append("Overview")
    report_lines.append(f"  Variants screened (panel):   {_fmt_int(n_variants_panel)}")
    report_lines.append(f"  Variants detected (count>0): {_fmt_int(n_variants_detected)} ({pct_variants_detected:.1f}% of panel)")
    report_lines.append(f"  {obs_label_plural.capitalize()} profiled:            {_fmt_int(n_obs)}")
    report_lines.append(f"  {obs_label_plural.capitalize()} with >=1 variant:     {_fmt_int(n_obs_with_variant)} ({pct_obs_with_variant:.1f}%)")
    report_lines.append(f"  Total variant counts:        {_fmt_int(total_counts)}")

    # Per-cell/sample distributions
    report_lines.append("")
    report_lines.append(f"Variants detected per {obs_label}")
    report_lines.append(f"  {_distribution_line(variants_per_obs)}")
    variant_buckets = [("0", 0, 0), ("1", 1, 1), ("2-5", 2, 5), ("6-10", 6, 10), ("11-20", 11, 20), ("21+", 21, float("inf"))]
    dist_rows = [(label, _fmt_int(n), f"{frac*100:.1f}%") for label, n, frac in _bucket_distribution(variants_per_obs, variant_buckets)]
    report_lines.append(_top_table(dist_rows, ["variants", obs_label_plural, "fraction"]))

    report_lines.append("")
    report_lines.append(f"Variant counts per {obs_label}")
    report_lines.append(f"  {_distribution_line(counts_per_obs)}")

    # Top variants by cumulative counts
    var_sorted_by_count = adata.var.sort_values(["vcrs_count", "number_obs"], ascending=False)
    var_detected = var_sorted_by_count.loc[var_sorted_by_count["vcrs_count"] > 0]
    top_by_count = var_detected.head(top_values)
    report_lines.append("")
    report_lines.append(f"Top {min(top_values, len(var_detected))} variants by total counts")
    report_lines.append(
        _top_table(
            [(row["_display_name"], _fmt_int(row["vcrs_count"]), _fmt_int(row["number_obs"])) for _, row in top_by_count.iterrows()],
            ["variant", "counts", obs_label_plural],
        )
    )

    # Top variants by prevalence (number of cells/samples detected)
    var_sorted_by_prev = adata.var.sort_values(["number_obs", "vcrs_count"], ascending=False)
    var_prev_detected = var_sorted_by_prev.loc[var_sorted_by_prev["number_obs"] > 0]
    top_by_prev = var_prev_detected.head(top_values)
    report_lines.append("")
    report_lines.append(f"Top {min(top_values, len(var_prev_detected))} variants by prevalence (# {obs_label_plural} detected)")
    report_lines.append(
        _top_table(
            [(row["_display_name"], _fmt_int(row["number_obs"]), _fmt_int(row["vcrs_count"])) for _, row in top_by_prev.iterrows()],
            ["variant", obs_label_plural, "counts"],
        )
    )

    # Gene-level stats
    genes_df = pd.DataFrame(columns=["n_variants_detected", "total_counts"])
    if gene_column is not None:
        genes_df = _explode_gene_stats(var_detected, gene_column, count_column="vcrs_count")
        report_lines.append("")
        report_lines.append(f"Genes with detected variants: {_fmt_int(len(genes_df))}")
        report_lines.append(f"Top {min(top_values, len(genes_df))} genes by number of detected variants")
        top_genes = genes_df.head(top_values)
        report_lines.append(
            _top_table(
                [(gene, _fmt_int(row["n_variants_detected"]), _fmt_int(row["total_counts"])) for gene, row in top_genes.iterrows()],
                ["gene", "variants", "counts"],
            )
        )
    else:
        report_lines.append("")
        report_lines.append("Gene-level stats skipped (no gene column found in adata.var; pass gene_name_column).")

    # Variant-type distribution
    if "variant_type" in adata.var.columns:
        type_counts = var_detected["variant_type"].astype(str).value_counts()
        report_lines.append("")
        report_lines.append("Detected variants by type")
        report_lines.append(
            _top_table(
                [(vtype, _fmt_int(n), f"{100.0*n/max(len(var_detected),1):.1f}%") for vtype, n in type_counts.items()],
                ["type", "variants", "fraction"],
            )
        )

    # Per-sample breakdown for bulk with a manageable number of samples
    if is_bulk and n_obs <= 100:
        report_lines.append("")
        report_lines.append("Per-sample breakdown")
        report_lines.append(
            _top_table(
                [(str(sample), _fmt_int(v), _fmt_int(c)) for sample, v, c in zip(adata.obs_names, variants_per_obs, counts_per_obs)],
                ["sample", "variants", "counts"],
            )
        )

    report_lines.append("")
    report_lines.append(f"Detailed tables: {specific_stats_folder}")
    report_lines.append(f"Plots:           {plots_folder}")
    report_lines.append("=" * 66)

    report = "\n".join(report_lines)
    print(report)
    with open(stats_file, "w", encoding="utf-8") as f:
        f.write(report + "\n")

    # * 12. Write detailed per-item TSV tables
    var_detected[["_display_name", "vcrs_count", "number_obs"]].rename(
        columns={"_display_name": "variant", "vcrs_count": "total_counts", "number_obs": f"n_{obs_label_plural}_detected"}
    ).to_csv(os.path.join(specific_stats_folder, "variants_by_count.tsv"), sep="\t", index=True, index_label="vcrs_id")

    var_prev_detected[["_display_name", "number_obs", "vcrs_count"]].rename(
        columns={"_display_name": "variant", "number_obs": f"n_{obs_label_plural}_detected", "vcrs_count": "total_counts"}
    ).to_csv(os.path.join(specific_stats_folder, "variants_by_prevalence.tsv"), sep="\t", index=True, index_label="vcrs_id")

    adata.obs[["n_variants_detected", "total_variant_counts"]].to_csv(
        os.path.join(specific_stats_folder, f"variants_per_{obs_label}.tsv"), sep="\t", index=True, index_label=obs_label
    )

    if gene_column is not None and not genes_df.empty:
        genes_df.to_csv(os.path.join(specific_stats_folder, "genes_by_variant_count.tsv"), sep="\t", index=True)

    # * 13. Core screening plots
    # variants detected per cell/sample
    plot_histogram_with_zero_value(adata.obs, col="n_variants_detected", save_path=os.path.join(plots_folder, f"variants_per_{obs_label}_histogram.png"))
    # counts per cell/sample
    plot_histogram_with_zero_value(adata.obs, col="total_variant_counts", save_path=os.path.join(plots_folder, f"counts_per_{obs_label}_histogram.png"))
    # top variants by counts (with and without names) - only meaningful if something was detected
    if n_variants_detected > 0:
        plot_items_descending_order(adata.var, x_column="_display_name", y_column="vcrs_count", item_range=(0, top_values), show_names=True, xlabel="Variant", title=f"Top {top_values} Variants by Counts across All {obs_label_plural.capitalize()}", figsize=(15, 7), show=False, save_path=os.path.join(plots_folder, f"top_{top_values}_variants_descending_plot.png"))
        plot_items_descending_order(adata.var, x_column="_display_name", y_column="vcrs_count", show_names=False, xlabel="Variant Index", title=f"Variants by Counts across All {obs_label_plural.capitalize()}", figsize=(15, 7), show=False, save_path=os.path.join(plots_folder, "variants_descending_plot.png"))
    plot_histogram_with_zero_value(adata.var, col="vcrs_count", save_path=os.path.join(plots_folder, "variants_histogram.png"))
    # top genes by number of detected variants
    if gene_column is not None and not genes_df.empty:
        genes_plot_df = genes_df.reset_index()
        plot_items_descending_order(genes_plot_df, x_column="gene", y_column="n_variants_detected", item_range=(0, top_values), show_names=True, xlabel="Gene", title=f"Top {top_values} Genes by Number of Detected Variants", figsize=(15, 7), show=False, save_path=os.path.join(plots_folder, f"top_{top_values}_genes_by_variant_count.png"))

    # * 14. Specialized plots (require HGVS-format vcrs_header, e.g. after vk clean)
    skip_plots = False
    if "vcrs_header" not in adata.var.columns:
        skip_plots = True
    else:
        first_vcrs_header = str(adata.var["vcrs_header"].iloc[0]).split(";")[0]
        if not re.fullmatch(HGVS_pattern_general, first_vcrs_header):
            logger.warning("Please run vk clean, or add a column vcrs_header to adata.var with the variant headers in HGVS format to make the specialized plots.")
            skip_plots = True

    if not skip_plots:
        adata_var_with_alignment = adata.var.loc[adata.var["vcrs_count"] > 0].copy()
        if plot_strand_bias:
            if seq_id_cdna_column not in adata_var_with_alignment.columns or start_variant_position_cdna_column not in adata_var_with_alignment.columns or end_variant_position_cdna_column not in adata_var_with_alignment.columns:
                logger.info("Adding information from variant header to a copy of adata.var. Note: this assumes vcrs_header transcripts and positions are accurate for cDNA")
                adata_var_with_alignment = add_information_from_variant_header_to_adata_var_exploded(adata_var_with_alignment, vcrs_header_individual_column="vcrs_header", seq_id_column=seq_id_cdna_column, var_column="variant", variant_source="placeholder", include_position_information=True, include_gene_information=False)
                adata_var_with_alignment.rename(columns={"start_variant_position": start_variant_position_cdna_column, "end_variant_position": end_variant_position_cdna_column}, inplace=True)

            plot_cdna_locations(adata_var_with_alignment, cdna_fasta=cdna_fasta, seq_id_column=seq_id_cdna_column, start_variant_position_cdna_column=start_variant_position_cdna_column, end_variant_position_cdna_column=end_variant_position_cdna_column, sequence_side=strand_bias_end, log_x=True, log_y=True, read_length_cutoff=read_length, save_path=os.path.join(plots_folder, f"strand_bias_{strand_bias_end}.png"))

        plot_substitution_heatmap(adata_var_with_alignment, variant_header_column="vcrs_header", count_column="vcrs_count", output_file=os.path.join(plots_folder, "substitutions_with_vcrs_count.png"), show=False, plot_type="bar")
        plot_substitution_heatmap(adata_var_with_alignment, variant_header_column="vcrs_header", count_column="vcrs_detected", output_file=os.path.join(plots_folder, "substitutions_with_vcrs_detected.png"), show=False, plot_type="bar")
        plot_variant_types(adata_var_with_alignment, variant_header_column="vcrs_header", variant_type_column="variant_type", count_column="vcrs_count", output_file=os.path.join(plots_folder, "variant_type_with_vcrs_count.png"), show=False)
        plot_variant_types(adata_var_with_alignment, variant_header_column="vcrs_header", variant_type_column="variant_type", count_column="vcrs_detected", output_file=os.path.join(plots_folder, "variant_type_with_vcrs_detected.png"), show=False)

    return adata
