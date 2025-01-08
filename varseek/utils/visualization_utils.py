import os
from collections import Counter
import shutil
import scanpy as sc
from rich.table import Table
from rich.console import Console
from collections import OrderedDict, defaultdict
import json

console = Console()

import numpy as np
import pandas as pd
import anndata as ad

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator, MultipleLocator
from matplotlib.patches import Rectangle
from matplotlib.ticker import LogLocator, FuncFormatter
from matplotlib_venn import venn2
import seaborn as sns

from scipy import stats
from scipy.sparse import csr_matrix
from scipy.stats import ttest_rel, t
from statsmodels.stats.contingency_tables import mcnemar


from varseek.constants import complement, codon_to_amino_acid, mutation_pattern

# Set global settings
plt.rcParams.update({
    'savefig.dpi': 450,             # Set resolution to 450 dpi
    'font.family': 'DejaVu Sans',   # Set font to Arial  # TODO: replace with Arial for Nature
    'pdf.fonttype': 42,             # Embed fonts as TrueType (keeps text editable)
    'ps.fonttype': 42,              # Same for PostScript files
    'savefig.format': 'pdf',        # Default save format as PNG
    'savefig.bbox': 'tight',        # Adjust bounding box to fit tightly
    'figure.facecolor': 'white',    # Set figure background to white (common for RGB)
    'savefig.transparent': False,   # Disable transparency
})

color_map_10 = plt.get_cmap("tab10").colors  # Default color map with 10 colors

color_map_20_original = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
    "#aec7e8", "#ffbb78", "#98df8a", "#ff9896", "#c5b0d5", "#c49c94", "#f7b6d2", "#c7c7c7", "#dbdb8d", "#9edae5"
]  # plotly category 20

color_map_20 = [
    "#f08925", "#1f77b4", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
    "#aec7e8", "#ffbb78", "#98df8a", "#ff9896", "#c5b0d5", "#c49c94", "#f7b6d2", "#c7c7c7", "#dbdb8d", "#9edae5"
]  # modified to swap 1 and 2 (orange first), and replaced the orange with varseek orange 

save_pdf_global = True if os.getenv('VARSEEK_SAVE_PDF') == "TRUE" else False
dpi = 450


def calculate_sensitivity_specificity(TP, TN, FP, FN):
    # Accuracy = (TP + TN) / (TP + TN + FP + FN)
    accuracy = (TP + TN) / (TP + TN + FP + FN)

    # Sensitivity (Recall) = TP / (TP + FN)
    sensitivity = TP / (TP + FN) if (TP + FN) > 0 else 1.0

    # Specificity = TN / (TN + FP)
    specificity = TN / (TN + FP) if (TN + FP) > 0 else 1.0

    return accuracy, sensitivity, specificity


def print_column_summary_stats(df_overlap, column, output_file=None):
    # Calculate statistics for the number_of_kmers_with_overlap_to_other_genes column
    kmers_mean = df_overlap[column].mean()
    kmers_median = df_overlap[column].median()
    kmers_mode = stats.mode(df_overlap[column])[0]
    kmers_max = df_overlap[column].max()
    kmers_variance = df_overlap[column].var()

    stats_summary = (
        f"Statistics for '{column}':\n"
        f"Mean: {kmers_mean}\n"
        f"Median: {kmers_median}\n"
        f"Mode: {kmers_mode}\n"
        f"Max Value: {kmers_max}\n"
        f"Variance: {kmers_variance}\n"
    )

    # Save the statistics to a text file
    if output_file is not None:
        if os.path.exists(output_file):
            writing_mode = "a"  # Append to the file if it already exists
        else:
            writing_mode = "w"
        with open(output_file, writing_mode) as f:
            f.write(stats_summary)

    # Print out the results to the console as well
    print(stats_summary)


def plot_histogram_notebook_1(
    df_overlap, column, x_label="x-axis", title="Histogram", output_plot_file=None
):
    # Define the bin range for the histograms
    bins = range(0, df_overlap[column].max() + 2)  # Bins for k-mers

    # Replace 0 values with a small value to be represented on log scale
    data = df_overlap[column].replace(
        0, 1e-10
    )  # Replace 0 with a very small value (1e-10)

    # Plot histogram
    plt.figure(figsize=(10, 6))
    plt.hist(
        data, bins=bins, alpha=0.7, color="blue", edgecolor="black", log=True
    )  # log=True for log scale
    plt.xlabel(x_label)
    plt.ylabel("Frequency (log10 scale)")
    plt.title(title)

    # Set y-axis to log10 with ticks at every power of 10
    plt.yscale("log")
    plt.gca().yaxis.set_major_locator(
        plt.LogLocator(base=10.0)
    )  # Ensure ticks are at powers of 10
    plt.gca().yaxis.set_minor_locator(
        plt.LogLocator(base=10.0, subs=np.arange(1.0, 10.0) * 0.1, numticks=10)
    )
    plt.gca().yaxis.set_minor_formatter(plt.NullFormatter())  # Hide minor tick labels

    # Set x-ticks dynamically
    max_value = df_overlap[column].max()
    step = max(1, int(np.ceil(max_value / 10)))  # Adjust for dynamic tick spacing
    plt.gca().xaxis.set_major_locator(plt.MultipleLocator(step))

    plt.grid(
        True, which="major", axis="y", ls="-"
    )  # Grid lines for both major and minor ticks

    if output_plot_file:
        plt.savefig(output_plot_file, format="png", dpi=dpi, bbox_inches="tight")
        if save_pdf_global:
            plt.savefig(output_plot_file.replace(".png", ".pdf"), format="pdf", dpi=dpi)

    plt.show()
    plt.close()


def plot_histogram_of_nearby_mutations_7_5(
    mutation_metadata_df, column, bins, output_file=None
):
    plt.figure(figsize=(10, 6))
    plt.hist(
        mutation_metadata_df[column], bins=bins, color="skyblue", edgecolor="black"
    )

    # Set titles and labels
    plt.title(f"Histogram of {column}", fontsize=16)
    plt.xlabel(column, fontsize=12)
    plt.ylabel("Frequency", fontsize=12)
    # plt.xscale('log')
    plt.yscale("log")

    # Display the plot
    plt.tight_layout()

    if output_file:
        plt.savefig(output_file, format="png", dpi=dpi)
        if save_pdf_global:
            plt.savefig(output_file.replace(".png", ".pdf"), format="pdf", dpi=dpi)

    plt.show()
    plt.close()


def retrieve_value_from_metric_file(key_of_interest, metric_file):
    metrics = {}
    with open(metric_file, "r") as file:
        for line in file:
            key, value = line.strip().split(": ")
            metrics[key] = value

    value_of_interest = metrics.get(key_of_interest)
    return value_of_interest


def calculate_metrics(
    df, header_name=None, check_assertions=False, crude=False, out=None, suffix=""
):
    if crude:
        suffix = "_crude"

    TP_column = f"TP{suffix}"
    FP_column = f"FP{suffix}"
    FN_column = f"FN{suffix}"
    TN_column = f"TN{suffix}"

    TP = df[TP_column].sum()
    FP = df[FP_column].sum()
    FN = df[FN_column].sum()
    TN = df[TN_column].sum()

    if header_name is not None:
        FPs = list(df.loc[df[FP_column], header_name])
        FNs = list(df.loc[df[FN_column], header_name])
    else:
        FPs = []
        FNs = []

    print(f"TP: {TP}, FP: {FP}, FN: {FN}, TN: {TN}")
    # if FP != 0:
    #     print(f"FPs: {FPs}")

    # if FN != 0:
    #     print(f"FNs: {FNs}")

    accuracy, sensitivity, specificity = calculate_sensitivity_specificity(
        TP, TN, FP, FN
    )

    print(f"Accuracy: {accuracy}, Sensitivity: {sensitivity}, Specificity: {specificity}")
    
    if f'mutation_expression_prediction_error{suffix}' in df.columns:
        mean_expression_error = df[f'mutation_expression_prediction_error{suffix}'].mean()
        median_expression_error = df[f'mutation_expression_prediction_error{suffix}'].median()
        mean_magnitude_expression_error = df[f'mutation_expression_prediction_error{suffix}'].abs().mean()
        median_magnitude_expression_error = df[f'mutation_expression_prediction_error{suffix}'].abs().median()
        print(f"Mean Expression Error: {mean_expression_error}, Median Expression Error: {median_expression_error}, Mean Magnitude Expression Error: {mean_magnitude_expression_error}, Median Magnitude Expression Error: {median_magnitude_expression_error}")
    else:
        mean_expression_error = "N/A"
        median_expression_error = "N/A"
        mean_magnitude_expression_error = "N/A"
        median_magnitude_expression_error = "N/A"

    if check_assertions:
        assert int(accuracy) == 1, "Accuracy is not 1"
        assert int(sensitivity) == 1, "Sensitivity is not 1"
        assert int(specificity) == 1, "Specificity is not 1"
        if f'mutation_expression_prediction_error{suffix}' in df.columns:
            assert int(mean_magnitude_expression_error) == 0, "Mean magnitude expression error is not 0"

    metric_dictionary = {
        "accuracy": accuracy,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "mean_expression_error": mean_expression_error,
        "median_expression_error": median_expression_error,
        "mean_magnitude_expression_error": mean_magnitude_expression_error,
        "median_magnitude_expression_error": median_magnitude_expression_error,
        "TP": TP,
        "FP": FP,
        "FN": FN,
        "TN": TN,
        "FPs": FPs,
        "FNs": FNs,
    }

    if out is not None:
        keys_to_save = [
            "accuracy",
            "sensitivity",
            "specificity",
            "TP",
            "FP",
            "FN",
            "TN",
            "mean_expression_error",
            "median_expression_error",
            "mean_magnitude_expression_error",
            "median_magnitude_expression_error",
        ]
        with open(out, "w") as file:
            for key in keys_to_save:
                file.write(f"{key}: {metric_dictionary[key]}\n")

    return metric_dictionary


def compute_grouped_metric(grouped_df, y_metric, crude=False):
    if crude:
        TP_column = "TP_crude"
        FP_column = "FP_crude"
        FN_column = "FN_crude"
        TN_column = "TN_crude"
    else:
        TP_column = "TP"
        FP_column = "FP"
        FN_column = "FN"
        TN_column = "TN"

    if y_metric == "accuracy":
        grouped_df[y_metric] = (grouped_df[TP_column] + grouped_df[TN_column]) / (
            grouped_df[TP_column]
            + grouped_df[TN_column]
            + grouped_df[FP_column]
            + grouped_df[FN_column]
        )  # * TODO: replace with len(grouped_df)
    elif y_metric == "sensitivity":
        grouped_df[y_metric] = grouped_df[TP_column] / (
            grouped_df[TP_column] + grouped_df[FN_column]
        )
        grouped_df.loc[
            (grouped_df[TP_column] + grouped_df[FN_column]) == 0, y_metric
        ] = 1.0
    elif y_metric == "specificity":
        grouped_df[y_metric] = grouped_df[TN_column] / (
            grouped_df[TN_column] + grouped_df[FP_column]
        )
        grouped_df.loc[
            (grouped_df[TN_column] + grouped_df[FP_column]) == 0, y_metric
        ] = 1.0
    elif y_metric == "expression_error":
        grouped_df[y_metric] = (
            grouped_df["mutation_expression_prediction_error"] / grouped_df["count"]
        )
    else:
        raise ValueError(f"Invalid y_metric: {y_metric}")

    return grouped_df


def convert_number_bin_into_labels(bins):
    bin_labels = []
    for i in range(1, len(bins)):
        if i == 1:  # First bin after the initial boundary
            bin_labels.append(f"≤{int(bins[i])}")
        elif bins[i - 1] + 1 == bins[i]:  # Consecutive numbers
            bin_labels.append(f"{int(bins[i])}")
        elif i == len(bins) - 1:  # Check if it's the last element in the bin list
            bin_labels.append(f"{int(bins[i-1])+1}+")
        else:  # Range bins
            bin_labels.append(f"{int(bins[i-1])+1}-{int(bins[i])}")

    return bin_labels


def create_stratified_metric_bar_plot(
    df,
    x_stratification,
    y_metric,
    overall_metric=None,
    log_x_axis=False,
    bins=None,
    x_axis_name=None,
    y_axis_name=None,
    title=None,
    display_numbers=False,
    out_path=None,
    crude=False,
):
    if bins is not None:
        labels = convert_number_bin_into_labels(bins)
        df["binned_" + x_stratification] = pd.cut(
            df[x_stratification], bins=bins, labels=labels, right=True
        )

        group_col = "binned_" + x_stratification
    else:
        group_col = x_stratification

    if y_metric != "expression_error":
        grouped_df = df.groupby(group_col).sum()
        grouped_df = compute_grouped_metric(grouped_df, y_metric, crude=crude)
        grouped_df = grouped_df.reset_index()
        bottom_value = 0
    else:
        # grouped_df = df.groupby(group_col)['mutation_expression_prediction_error'].var().reset_index()
        grouped_df = (
            df.groupby(group_col)["mutation_expression_prediction_error"]
            .apply(lambda x: x.abs().mean())
            .reset_index()
        )
        grouped_df.rename(
            columns={"mutation_expression_prediction_error": y_metric}, inplace=True
        )
        grouped_df[y_metric] += 0.05
        bottom_value = -0.05

    # # add counts to each row
    # group_size = df.groupby(group_col).size().reset_index(name='count')
    # group_size = grouped_df.merge(group_size, on=group_col)

    # Create a bar chart where the x axis is number_of_reads_mutant, and y axis is accuracy
    if group_col == "number_of_reads_mutant":
        had_zero = (grouped_df[group_col].astype(int) == 0).any()
        grouped_df = grouped_df[grouped_df[group_col].astype(int) != 0].reset_index(
            drop=True
        )
    else:
        had_zero = None

    plt.bar(
        grouped_df[group_col],
        grouped_df[y_metric],
        bottom=bottom_value,
        color="black",
        alpha=0.7,
    )

    if display_numbers:
        for i, value in enumerate(grouped_df[y_metric]):
            plt.text(
                i,
                bottom_value + 0.85,
                f"{value:.5f}",
                ha="center",
                va="bottom",
                fontsize=10,
                color="red",
            )

    # Add a horizontal line for the total average accuracy
    if overall_metric is not None:
        plt.axhline(
            y=overall_metric,
            color="gray",
            linestyle="--",
            label=f"Average {y_metric} ({overall_metric:.2f})",
        )

    if y_metric == "accuracy" or y_metric == "sensitivity" or y_metric == "specificity":
        plt.ylim(0, 1)
        plt.yticks(np.arange(0, 1.1, 0.1))  # Major ticks every 0.1
        plt.minorticks_on()
        plt.gca().yaxis.set_minor_locator(
            plt.MultipleLocator(0.05)
        )  # Minor ticks every 0.05

    if log_x_axis:
        plt.xscale("log")

    if bins is None and type(grouped_df[x_stratification][0]) != str:
        plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True, prune="both"))
        if had_zero:
            min_x = 0
        else:
            min_x = int(grouped_df[x_stratification].min())
        max_x = int(grouped_df[x_stratification].max())
        min_x_rounded = (min_x // 5) * 5

        range_x = max_x - min_x_rounded

        # ensure ticks are spaced by 1 OR a multiple of 5
        if range_x <= 10:
            step_size = 1  # Use a step of 1 if the range is small
            minor_step_size = None  # No minor ticks for step size of 1
        else:
            step_size = 5 * (
                (range_x // 10) // 5 + 1
            )  # Ensure step size is a multiple of 5
            minor_step_size = step_size / 5  # 4 minor ticks between each major tick

        # step_size = (max_x - min_x) // 10  # Adjust 10 to a lower number to reduce tick frequency
        x_ticks_int = np.arange(min_x_rounded, max_x + 1, step_size)
        # x_ticks_int = np.arange(min_x, max_x + 1, 1)  # Adjust the step size (e.g., 1, 2, 5) as needed
        plt.xticks(x_ticks_int)

        plt.gca().xaxis.set_major_formatter(plt.ScalarFormatter())
        plt.gca().xaxis.set_major_locator(MultipleLocator(step_size))

        if minor_step_size:
            plt.gca().xaxis.set_minor_locator(MultipleLocator(minor_step_size))
        else:
            plt.gca().xaxis.set_minor_locator(
                plt.NullLocator()
            )  # No minor ticks if step size is 1

    # Add labels and title
    if x_axis_name is None:
        x_axis_name = x_stratification
    if y_axis_name is None:
        y_axis_name = y_metric
    if title is None:
        title = f"{y_metric} vs. {x_stratification}"

    plt.xlabel(x_axis_name)
    plt.ylabel(y_axis_name)
    plt.title(title)
    plt.legend()

    if out_path is not None:
        if out_path == True:
            out_path = f"{y_metric}_vs_{x_stratification}.png"
        plt.savefig(out_path, bbox_inches="tight", dpi=dpi)
        if save_pdf_global:
            plt.savefig(out_path.replace(".png", ".pdf"), format="pdf", dpi=dpi)

    plt.show()
    plt.close()


def create_venn_diagram(true_set, positive_set, TN=None, mm=None, out_path=None):
    venn = venn2(
        [true_set, positive_set],
        set_labels=("Present in reads", "Detected by alignment"),
    )

    if mm is not None:
        mm_line = f"\n({mm} mm)"
    else:
        mm_line = ""

    # Modify the colors of the circles
    if venn.get_label_by_id("10") is not None:  # Make sure the intersection exists
        venn.get_patch_by_id("10").set_color("yellow")  # Left circle (Set 1)
        venn.get_label_by_id("10").set_fontsize(10)  # Label for Set 1 only
        venn.get_label_by_id("10").set_text(
            f'FN: {venn.get_label_by_id("10").get_text()}{mm_line}'
        )

    if venn.get_label_by_id("01") is not None:  # Make sure the intersection exists
        venn.get_patch_by_id("01").set_color("blue")  # Right circle (Set 2)
        venn.get_label_by_id("01").set_fontsize(10)  # Label for Set 2 only
        venn.get_label_by_id("01").set_text(
            f'FP: {venn.get_label_by_id("01").get_text()}'
        )

    if venn.get_patch_by_id("11") is not None:
        venn.get_patch_by_id("11").set_color("green")  # Intersection of Set 1 and Set 2
        venn.get_label_by_id("11").set_fontsize(10)  # Label for the intersection
        venn.get_label_by_id("11").set_text(
            f'TP: {venn.get_label_by_id("11").get_text()}'
        )

    for label in venn.set_labels:
        if label is not None:
            label.set_fontsize(10)

    for patch in venn.patches:
        if patch is not None:
            patch.set_edgecolor("black")
            patch.set_linewidth(1)
            patch.set_alpha(0.6)

    # Add a black rectangle around the Venn diagram
    plt.gca().add_patch(
        Rectangle(
            (0, 0),
            1,
            1,
            fill=False,
            edgecolor="black",
            lw=2,
            transform=plt.gca().transAxes,
        )
    )

    # Print the number 5 in the top left corner

    if TN is not None:
        plt.text(0.01, 0.96, f"TN: {TN}", fontsize=10, transform=plt.gca().transAxes)

    if out_path is not None:
        plt.savefig(out_path, bbox_inches="tight")

    # Show the plot
    plt.show()
    plt.close()


def plot_histogram(
    df,
    column_name,
    bins=None,
    log_scale=False,
    x_axis_label=None,
    y_axis_label=None,
    title=None,
    out_path=None,
):
    """
    Plot a histogram of the specified column in the DataFrame with custom bins and log x-axis.

    Parameters:
    df (pd.DataFrame): DataFrame containing the data.
    column_name (str): Column name to plot as a histogram.
    x_axis_label (str): Label for the x-axis. Default is the column name.
    y_axis_label (str): Label for the y-axis. Default is 'Frequency'.
    title (str): Title of the plot. Default is 'Histogram of {column_name}'.
    out_path (str): Path to save the figure. If None, it shows the plot instead of saving it.
    """

    # Define custom bins as per the user's request
    if not bins:
        if log_scale:
            bins = [0, 1, 2, 3, 4, 5, 6, 10, 20, 50, np.inf]
        else:
            bins = [-np.inf, -50, -20, -10, -5, -2, -1, 0, 1, 2, 5, 10, 20, 50, np.inf]

    if log_scale:
        df[column_name] = df[column_name].abs()
        bins = list([x for x in bins if x >= 0])

    # Plot the histogram with log x-axis
    plt.figure(figsize=(4, 3))
    plt.hist(df[column_name], bins=bins, color="blue", alpha=0.7, log=False)

    # Set log scale for the x-axis
    if log_scale:
        plt.xscale("log")

    # Add labels and title
    if x_axis_label is None:
        x_axis_label = column_name
    if y_axis_label is None:
        y_axis_label = "Frequency"
    if title is None:
        title = f"Histogram of {column_name}"

    plt.xlabel(x_axis_label)
    plt.ylabel(y_axis_label)
    plt.title(title)

    # Ensure only integer ticks are shown on the x-axis
    plt.gca().xaxis.set_major_formatter(plt.ScalarFormatter())
    plt.gca().xaxis.set_minor_formatter(plt.NullFormatter())

    # Save or show the plot
    if out_path is not None:
        plt.savefig(out_path, bbox_inches="tight", dpi=dpi)
        if save_pdf_global:
            plt.savefig(out_path.replace(".png", ".pdf"), format="pdf", dpi=dpi)
    else:
        plt.show()
        plt.close()


def synthetic_data_summary_plot(df, column, sort_ascending=True, out_path=None):
    # Step 1: Calculate the counts of each unique value
    value_counts = df[column].value_counts()

    # Step 2: Convert counts to percentages
    percentages = (value_counts / len(df)) * 100

    if sort_ascending:
        try:
            percentages = percentages.sort_index(ascending=True)
        except Exception as e:
            pass

    # Step 3: Plot the percentages as a bar plot
    plt.figure(figsize=(4, 3))
    ax = percentages.plot(kind="bar", color="gray", alpha=0.8)
    # plt.title('Percentage of Rows for Each Region')
    plt.xlabel(column)
    plt.ylabel("Percentage of Rows (%)")

    if len(percentages) == 1:
        # Only one unique value, so set only one x-tick
        plt.xticks([0], [percentages.index[0]])
    else:
        try:
            # If x-ticks are numbers, enforce integer-only ticks
            ax.xaxis.set_major_locator(
                MaxNLocator(integer=True)
            )  # Ensure x-ticks are integers
        except ValueError:
            pass

    plt.xticks(rotation=45, ha="right")  # Rotate x labels if they are long
    plt.tight_layout()

    if out_path is not None:
        plt.savefig(out_path)
        if save_pdf_global:
            plt.savefig(out_path.replace(".png", ".pdf"), format="pdf", dpi=dpi)

    # Show the plot
    plt.show()
    plt.close()


def plot_basic_bar_plot_from_dict(my_dict, y_axis, log_scale=False, output_file=None):
    plt.figure(figsize=(8, 6))
    plt.bar(list(my_dict.keys()), list(my_dict.values()), color="black", alpha=0.8)
    plt.ylabel(y_axis)

    # log y scale
    if log_scale:
        plt.yscale("log")

    plt.tight_layout()

    if output_file:
        plt.savefig(output_file, format="png", dpi=dpi)
        if save_pdf_global:
            plt.savefig(output_file.replace(".png", ".pdf"), format="pdf", dpi=dpi)

    plt.show()
    plt.close()


def plot_descending_bar_plot(
    gene_counts, x_label, y_label, tick_interval=None, output_file=None
):
    # Plot a histogram of gene names in descending order
    plt.figure(figsize=(10, 6))
    gene_counts.plot(kind="bar", color="skyblue")

    plt.xlabel(x_label)
    plt.ylabel(y_label)

    tick_interval = 5000  # Set this to 100 or higher depending on the number of genes

    if tick_interval is None:
        plt.xticks(rotation=90)
    else:
        plt.xticks(
            ticks=range(0, len(gene_counts), tick_interval),
            labels=range(1, len(gene_counts) + 1, tick_interval),
            rotation=90,
        )

    # Show the plot
    plt.tight_layout()

    if output_file:
        plt.savefig(output_file, format="png", dpi=dpi)
        if save_pdf_global:
            plt.savefig(output_file.replace(".png", ".pdf"), format="pdf", dpi=dpi)

    plt.show()
    plt.close()

def draw_confusion_matrix(metric_dictionary_reads, title = "Confusion Matrix", title_color = "black", suffix = "", additional_fp_key = "", output_file = None, show = True):
    confusion_matrix = {
        "TP": str(metric_dictionary_reads[f"TP{suffix}"]),  # True Positive
        "TN": str(metric_dictionary_reads[f"TN{suffix}"]),  # True Negative
        "FP": str(metric_dictionary_reads[f"FP{suffix}"]),  # False Positive
        "FN": str(metric_dictionary_reads[f"FN{suffix}"]),  # False Negative
    }

    if additional_fp_key in metric_dictionary_reads:
        additional_fp_text = " ".join(additional_fp_key.split()[1:])  # so if the key is FP including non-cosmic, then the text will be non-cosmic
        confusion_matrix['FP'] += f"\n{additional_fp_text}: {metric_dictionary_reads[additional_fp_key]}"

    # Convert confusion matrix into a 2x2 format
    data = [
        [confusion_matrix["TP"], confusion_matrix["FN"]],  # Actual Positive
        [confusion_matrix["FP"], confusion_matrix["TN"]],  # Actual Negative
    ]

    # Row and column labels
    rows = ["Actual Positive", "Actual Negative"]
    columns = ["Predicted Positive", "Predicted Negative"]

    # Create a pandas DataFrame for easier handling
    df = pd.DataFrame(data, index=rows, columns=columns)

    # Plot the table
    fig, ax = plt.subplots(figsize=(6, 3))  # Adjust size as needed
    ax.axis("off")  # Turn off the axis

    # Create the table
    table = ax.table(
        cellText=df.values,
        rowLabels=df.index,
        colLabels=df.columns,
        loc="center",
        cellLoc="center",
    )

    table.scale(1, 2)  # Adjust scaling of the table (optional)
    ax.text(0.5, 0.75, title, transform=ax.transAxes, ha="center", fontsize=14, color=title_color)
    table.set_fontsize(10)

    # Save the table as a PDF
    plt.tight_layout()
    if output_file:
        plt.savefig(output_file, bbox_inches="tight", pad_inches=0.5)
    if show:
        plt.show()
        plt.close()

def draw_confusion_matrix_rich(metric_dictionary_reads, title = "Confusion Matrix", suffix = "", additional_fp_key = ""):
    # Sample dictionary with confusion matrix values
    confusion_matrix = {
        "TP": metric_dictionary_reads[f"TP{suffix}"],  # True Positive
        "TN": metric_dictionary_reads[f"TN{suffix}"],  # True Negative
        "FP": metric_dictionary_reads[f"FP{suffix}"],  # False Positive
        "FN": metric_dictionary_reads[f"FN{suffix}"],  # False Negative
    }

    # Create a Rich Table to display the confusion matrix
    table = Table(title=title)

    # Add columns for the table
    table.add_column("", justify="center")
    table.add_column("Predicted Positive", justify="center")
    table.add_column("Predicted Negative", justify="center")

    fp_line = str(confusion_matrix["FP"])
    if additional_fp_key in metric_dictionary_reads:
        additional_fp_text = " ".join(additional_fp_key.split()[1:])  # so if the key is FP including non-cosmic, then the text will be non-cosmic
        fp_line += " (" + additional_fp_text + ": " + str(metric_dictionary_reads[additional_fp_key]) + ")"

    # Add rows for the confusion matrix
    table.add_row(
        "Actual Positive", str(confusion_matrix["TP"]), str(confusion_matrix["FN"])
    )
    table.add_row(
        "Actual Negative", fp_line, str(confusion_matrix["TN"])
    )

    # Display the table
    console.print(table)


def find_specific_value_from_metric_text_file(file_path, line):
    # file must be \n-separated and have the format "line: value"
    
    value = None

    # Read the file and extract the value
    with open(file_path, 'r') as file:
        for looping_line in file:
            if line in looping_line:
                value = int(looping_line.split(":")[1].strip())
                return value


def plot_kat_histogram(kat_hist, out_path=None):
    if out_path is None:
        base_name = os.path.basename(kat_hist).replace(".", "_")
        out_path = f"{base_name}_custom.png"

    # Read the data, skip the header lines
    data = pd.read_csv(kat_hist, sep=" ", comment="#", header=None)

    # Assign column names for easier reference
    data.columns = ["Frequency", "Distinct_kmers"]

    data = data[data["Distinct_kmers"] > 0]

    # Plot the histogram
    plt.figure(figsize=(10, 6))
    plt.bar(data["Frequency"], data["Distinct_kmers"], width=0.8, color="skyblue")

    plt.yscale("log")

    # Ensure x-axis values are integers
    plt.xticks(np.arange(data["Frequency"].min(), data["Frequency"].max() + 1, step=1))

    if len(data) == 1:
        plt.xlim([data["Frequency"].min() - 1, data["Frequency"].max() + 1])

    # Add labels and title
    plt.xlabel("55-mer Frequency")
    plt.ylabel("# of Distinct 55-mers")
    plt.title("55-mer Spectra for random_sequences.fasta")

    # Save the plot
    plt.savefig(out_path, format="png", dpi=dpi)
    if save_pdf_global:
        plt.savefig(out_path.replace(".png", ".pdf"), format="pdf", dpi=dpi)

    # Display the plot
    plt.show()
    plt.close()


def plot_items_descending_order(
    df,
    x_column,
    y_column,
    item_range=(0, 10),
    xlabel="x-axis",
    title="Title",
    save_path=None,
    figsize=(15, 7),
    show=False
):
    # Plot the line plot
    plt.figure(figsize=figsize)

    first_item = item_range[0]
    last_item = item_range[1]

    assert len(df) > first_item, f"First item index {first_item} is out of bounds"
    last_item = min(last_item, len(df))

    if first_item + last_item > 100:
        x_axis_type = range(first_item + 1, last_item + 1)

    else:
        x_axis_type = list(df[x_column])[first_item:last_item]

    plt.plot(x_axis_type, df.iloc[first_item:last_item][y_column], marker="o")
    plt.xticks(rotation=90)
    plt.xlabel(xlabel)
    plt.ylabel("Transcript Count")
    plt.title(title)
    plt.yscale("log")
    plt.grid(True)
    plt.tight_layout()

    # Save the plot
    if save_path:
        plt.savefig(save_path, dpi=dpi)
        if save_pdf_global:
            plt.savefig(save_path.replace(".png", ".pdf"), format="pdf", dpi=dpi)

    # Show the plot
    if show:
        plt.show()
    
    plt.close()


def plot_scree(adata, output_plot_file=None):
    variance_explained = adata.uns["pca"]["variance_ratio"]
    num_components = len(variance_explained)

    # Plot the scree plot
    plt.figure(figsize=(10, 5))
    plt.plot(
        np.arange(1, len(variance_explained) + 1),
        variance_explained,
        marker="o",
        linestyle="-",
    )
    plt.xticks(ticks=np.arange(1, num_components + 1))
    plt.xlabel("Principal Component")
    plt.ylabel("Variance Explained")
    plt.title("Scree Plot")
    if output_plot_file:
        os.makedirs(os.path.dirname(output_plot_file), exist_ok=True)
        plt.savefig(output_plot_file, format="png", dpi=dpi)
        if save_pdf_global:
            plt.savefig(output_plot_file.replace(".png", ".pdf"), format="pdf", dpi=dpi)

    plt.show()
    plt.close()


def plot_loading_contributions(
    adata,
    PC_index=0,
    top_genes_stats=100,
    top_genes_plot=10,
    output_stats_file=None,
    output_plot_file=None,
    show=False
):
    # Get PCA loadings for the selected component
    loadings = adata.varm["PCs"][:, PC_index]

    # Find indices of top genes by absolute loading values
    top_gene_indices_stats = np.argsort(np.abs(loadings))[::-1][:top_genes_stats]
    top_gene_names_stats = adata.var_names[top_gene_indices_stats]
    top_gene_loadings_stats = loadings[top_gene_indices_stats]

    if output_stats_file:
        os.makedirs(os.path.dirname(output_stats_file), exist_ok=True)
        with open(output_stats_file, "w") as f:
            for gene, loading in zip(top_gene_names_stats, top_gene_loadings_stats):
                f.write(f"{gene} {loading}\n")

    top_gene_indices_plot = np.argsort(np.abs(loadings))[::-1][:top_genes_plot]
    top_gene_names_plot = adata.var_names[top_gene_indices_plot]
    top_gene_loadings_plot = loadings[top_gene_indices_plot]

    # Plot as a horizontal bar chart
    plt.figure(figsize=(8, 6))
    plt.barh(top_gene_names_plot, top_gene_loadings_plot, color="skyblue")
    plt.xlabel("Contribution to PC1")
    plt.ylabel("Gene")
    plt.title("Top Gene Contributions to PC1")
    plt.gca().invert_yaxis()  # Invert Y-axis for descending order
    if output_plot_file:
        os.makedirs(os.path.dirname(output_plot_file), exist_ok=True)
        plt.savefig(output_plot_file, format="png", dpi=dpi)
        if save_pdf_global:
            plt.savefig(output_plot_file.replace(".png", ".pdf"), format="pdf", dpi=dpi)
    if show:
        plt.show()
    plt.close()


def find_resolution_for_target_clusters(
    adata, target_clusters, tolerance=3, max_iters=10
):
    # Initial bounds for resolution
    lower, upper = 0.1, 10
    assert max_iters > 0, "max_iters must be a positive integer"
    for i in range(max_iters):
        # Take the midpoint as the next test resolution
        adata_copy = adata.copy()
        resolution = (lower + upper) / 2
        sc.tl.leiden(adata_copy, resolution=resolution)
        num_clusters = adata_copy.obs["leiden"].nunique()

        print(
            f"Iteration {i + 1}: Resolution = {resolution}, Clusters = {num_clusters}"
        )

        # Check if the number of clusters is within the tolerance of the target
        if abs(num_clusters - target_clusters) <= tolerance:
            return adata_copy, resolution, num_clusters

        # Update bounds based on whether we have too many or too few clusters
        if num_clusters < target_clusters:
            lower = resolution
        else:
            upper = resolution

    return (
        adata_copy,
        resolution,
        num_clusters,
    )  # Return last tested resolution if exact match not found


def plot_contingency_table(
    adata, column1="tissue", column2="leiden", output_plot_file=None
):
    # Create a contingency table (counts of cells in each combination of tissue and leiden cluster)
    contingency_table = pd.crosstab(adata.obs[column1], adata.obs[column2])

    # Plot the heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(contingency_table, annot=True, cmap="Blues", fmt="d")
    plt.xlabel("Leiden Cluster")
    plt.ylabel("Tissue Type")
    plt.title(f"Heatmap of Agreement Between {column1} and {column2}")
    if output_plot_file:
        plt.savefig(output_plot_file)
    plt.show()
    plt.close()


def plot_knn_tissue_frequencies(
    indices, adata_combined_ccle_rnaseq, output_plot_file=None
):
    # Split the 'obs_names' to extract the tissue component
    neighbor_tissues = [
        adata_combined_ccle_rnaseq.obs_names[idx].split("_")[
            -1
        ]  # Extract tissue from "experiment_tissue"
        for neighbors in indices
        for idx in neighbors
    ]

    # Count occurrences of each tissue in the nearest neighbors
    tissue_counts = Counter(neighbor_tissues)

    # Plot the tissue frequencies
    plt.figure(figsize=(10, 6))
    plt.bar(tissue_counts.keys(), tissue_counts.values(), color="skyblue")
    plt.xlabel("Tissue Type")
    plt.ylabel("Frequency")
    plt.title("Frequency of Each Tissue in Nearest Neighbors")
    plt.xticks(rotation=45)
    if output_plot_file:
        plt.savefig(output_plot_file)
        if save_pdf_global:
            plt.savefig(output_plot_file.replace(".png", ".pdf"), format="pdf", dpi=dpi)
    plt.show()
    plt.close()

    return tissue_counts


def plot_ascending_bar_plot_of_cluster_distances(
    sorted_distances, output_plot_file=None
):
    # Separate clusters and distances for plotting
    clusters_sorted, distances_sorted = zip(*sorted_distances)

    plt.figure(figsize=(10, 6))
    plt.bar(clusters_sorted, distances_sorted, color="skyblue")
    plt.xlabel("Cluster")
    plt.ylabel("Distance to Unknown Sample")
    plt.title("Distance from Unknown Sample to Each Cluster Centroid (Ascending Order)")
    plt.xticks(rotation=45)
    if output_plot_file:
        plt.savefig(output_plot_file, format="png", dpi=dpi)
        if save_pdf_global:
            plt.savefig(output_plot_file.replace(".png", ".pdf"), format="pdf", dpi=dpi)
            
    plt.show()
    plt.close()


def plot_jaccard_bar_plot(tissues, jaccard_values, output_plot_file=None):
    sorted_data = sorted(zip(jaccard_values, tissues), reverse=True)
    sorted_jaccard_values, sorted_tissues = zip(*sorted_data)

    plt.figure(figsize=(10, 6))
    plt.bar(sorted_tissues, sorted_jaccard_values, color="skyblue")
    plt.xlabel("Tissue")
    plt.ylabel("Jaccard Index")
    plt.title("Jaccard Index for Each Tissue")
    plt.xticks(rotation=45)
    if output_plot_file:
        plt.savefig(output_plot_file, format="png", dpi=dpi)
        if save_pdf_global:
            plt.savefig(output_plot_file.replace(".png", ".pdf"), format="pdf", dpi=dpi)

    plt.show()
    plt.close()


def plot_knee_plot(
    umi_counts_sorted,
    knee_locator,
    min_counts_assessed_by_knee_plot=None,
    output_file=None,
):
    plt.plot(range(len(umi_counts_sorted)), umi_counts_sorted, marker=".")
    plt.axvline(
        knee_locator.knee,
        color="red",
        linestyle="--",
        label=f"Cutoff at UMI = {min_counts_assessed_by_knee_plot}",
    )
    plt.xlabel("Cell Rank")
    plt.ylabel("Total UMI Counts")
    plt.title("Knee Plot with Cutoff")
    plt.legend()
    if output_file:
        plt.savefig(output_file)
    plt.show()
    plt.close()


def plot_overall_metrics(metric_dict_collection, primary_metrics = ("accuracy", "sensitivity", "specificity"), display_numbers = False, unique_mcrs_df = None, show_p_values = False, bonferroni = True, output_file = None, show = True, output_file_p_values = None, filter_real_negatives = False):
    if not isinstance(primary_metrics, (str, list, tuple)):
        raise ValueError("Primary metrics must be a string, list, or tuple.")
    
    if unique_mcrs_df is not None:
        unique_mcrs_df = unique_mcrs_df.copy()

    if not filter_real_negatives and "expression_error" in primary_metrics:
        print("Warning: filtering real negatives is recommended when using expression error as a primary metric, but this setting is not currently enabled. Recommended: filter_real_negatives = True.")
    
    if filter_real_negatives:
        unique_mcrs_df = unique_mcrs_df[unique_mcrs_df['included_in_synthetic_reads_mutant'] == True]
    
    if isinstance(primary_metrics, str):
        primary_metrics = [primary_metrics]
    elif isinstance(primary_metrics, tuple):
        primary_metrics = list(primary_metrics)
    
    # Extract keys and values
    groups = list(metric_dict_collection.keys())  # Outer keys: varseek, mutect2, haplotypecaller
    colors = color_map_20[:len(groups)]

    # Prepare data
    x_primary = np.arange(len(primary_metrics))  # Positions for the metrics on the x-axis
    bar_width = 0.25  # Width of each primary_metrics
    offsets = np.arange(len(groups)) * bar_width - (len(groups) - 1) * bar_width / 2  # Centered offsets

    # Create the plot
    fig, ax1 = plt.subplots(figsize=(8, 6))
    y_values_primary_total = []  # y_values_primary_total holds all metrics across all tools - I could make this a nested dict if desired
    for i, group in enumerate(groups):
        y_values_primary = [metric_dict_collection[group][metric] for metric in primary_metrics]  # y_values_primary just holds all metrics for a specific tool
        bars_primary = ax1.bar(x_primary + offsets[i], y_values_primary, bar_width, label=group, color=colors[i])

        # Add value annotations for the primary metrics
        if display_numbers:
            for bar, value in zip(bars_primary, y_values_primary):
                ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                        f"{value:.3f}", ha="center", va="bottom", fontsize=10)
                
        y_values_primary_total.extend(y_values_primary)

    # Customize the plot
    # ax1.set_xlabel("Metrics", fontsize=12)
    # ax1.set_title("Comparison of Metrics Across Tools", fontsize=14)
    if "accuracy" in primary_metrics or "sensitivity" in primary_metrics or "specificity" in primary_metrics:
        ax1.set_ylim(0, 1.05)
    ax1.set_xticks(x_primary)
    ax1.set_xticklabels(primary_metrics, fontsize=12)

    metric_to_tool_to_p_value_dict_of_dicts = {}
    margins_of_error = {}
    if show_p_values or output_file_p_values:
        for metric in primary_metrics:
            margins_of_error[metric] = {}
            if metric in {"accuracy", "sensitivity", "specificity"}:
                tool_to_p_value_dict_aggregate = calculate_mcnemar(unique_mcrs_df, tools = groups, metric = metric)  # don't pass output file here because I want output file to include all p-values; and don't do bonferroni here for a similar reason
            elif metric in {"mean_magnitude_expression_error"}:
                tool_to_p_value_dict_aggregate = calculate_paired_t_test(unique_mcrs_df, column_root = "mutation_expression_prediction_error", tools = groups, take_absolute_value = True)
                for group in groups:  # calculate 95% confidence intervals
                    mean_value, margin_of_error = compute_95_confidence_interval_margin_of_error(unique_mcrs_df[f'mutation_expression_prediction_error_{group}'], take_absolute_value = True)
                    margins_of_error[metric][group] = (mean_value, margin_of_error)
            elif metric in {"mean_expression_error"}:
                tool_to_p_value_dict_aggregate = calculate_paired_t_test(unique_mcrs_df, column_root = "mutation_expression_prediction_error", tools = groups, take_absolute_value = False)
                for group in groups:  # calculate 95% confidence intervals
                    mean_value, margin_of_error = compute_95_confidence_interval_margin_of_error(unique_mcrs_df[f'mutation_expression_prediction_error_{group}'], take_absolute_value = False)
                    margins_of_error[metric][group] = (mean_value, margin_of_error)
            else:
                raise ValueError(f"Invalid metric for p-value calculation: {metric}. Valid options are 'accuracy', 'sensitivity', 'specificity', 'mean_magnitude_expression_error', 'mean_expression_error'")
            
            metric_to_tool_to_p_value_dict_of_dicts[metric] = tool_to_p_value_dict_aggregate

        if bonferroni:
            n_tests = count_leaves(metric_to_tool_to_p_value_dict_of_dicts)  # counts the number of leaves in the nested dictionary - makes it generalizable  # (len(groups)-1) * len(primary_metrics)
            for metric in primary_metrics:
                for group in groups:
                    if metric in metric_to_tool_to_p_value_dict_of_dicts and group in metric_to_tool_to_p_value_dict_of_dicts[metric]:
                        metric_to_tool_to_p_value_dict_of_dicts[metric][group] = min(metric_to_tool_to_p_value_dict_of_dicts[metric][group] * n_tests, 1.0)
        
        # Save to a file
        if output_file_p_values:
            with open(output_file_p_values, "w") as f:
                json.dump(metric_to_tool_to_p_value_dict_of_dicts, f, indent=4)

        # # toy p-values for accuracy, sensitivity, and specificity
        # metric_to_tool_to_p_value_dict_of_dicts = {"accuracy": {
        #     "gatk_mutect2": 0.0001,
        #     "gatk_haplotypecaller": 0.04
        # },
        # "sensitivity": {
        #     "gatk_mutect2": 0.007,
        #     "gatk_haplotypecaller": 0.99
        # },
        # "specificity": {
        #     "gatk_mutect2": 0.02,
        #     "gatk_haplotypecaller": 0.99
        # }}

        # # toy p-values for mean_magnitude_expression_error
        # metric_to_tool_to_p_value_dict_of_dicts = {"mean_magnitude_expression_error": {
        #     "gatk_mutect2": 0.0001,
        #     "gatk_haplotypecaller": 0.04
        # }}
        

        if show_p_values:
            for i, metric in enumerate(primary_metrics):
                if metric in metric_to_tool_to_p_value_dict_of_dicts:
                    number_of_p_values_in_this_cluster = 0
                    for j, group in enumerate(groups):
                        # 95% confidence intervals
                        if metric in margins_of_error and group in margins_of_error[metric]:
                            # Calculate error values
                            mean_value, margin_of_error = margins_of_error[metric][group]
                            if margin_of_error != 0:
                                yerr = [margin_of_error, margin_of_error]
                                x_value = x_primary[i] + offsets[j]

                                # Plot the point with the confidence interval
                                ax1.errorbar(
                                    x_value, mean_value, 
                                    yerr=np.array([yerr]).T,  # Transpose to match dimensions
                                    fmt='',  # Marker for the point
                                    capsize=5,  # Adds caps to the error bars
                                    label="Mean with 95% CI",
                                    color="black"
                                )

                        # p-values
                        if group in metric_to_tool_to_p_value_dict_of_dicts[metric]:
                            p_value = metric_to_tool_to_p_value_dict_of_dicts[metric][group]
                            if p_value >= 0.05:  #* increase these values to show more p-values for debugging
                                continue
                            elif p_value < 0.05 and p_value >= 0.01:
                                symbol = "*"
                            elif p_value < 0.01 and p_value >= 0.001:
                                symbol = "**"
                            else:
                                symbol = "***"
                            
                            start_x = x_primary[i] + offsets[0]  # assuming varseek is first element
                            end_x = x_primary[i] + offsets[j]

                            if metric in {"accuracy", "sensitivity", "specificity"}:
                                y_start = max(y_values_primary_total) + (number_of_p_values_in_this_cluster * 0.05) + 0.1
                            else:
                                y_start = (max(y_values_primary_total) + (number_of_p_values_in_this_cluster * 1.7)) * 1.08  # 1.7 (left constant) adjusts based on other bars; 1.08 (right constant) adjusts to make sure it doesn't hit the top bar
                            
                            y_end = y_start

                            ax1.plot([start_x, start_x, end_x, end_x], [y_start, y_end, y_end, y_start], lw=1.5, c="k")  # plot the bar
                            ax1.text((start_x + end_x) * .5, y_end, symbol, ha='center', va='bottom', color="k")  # plot the asterisk(s)

                            number_of_p_values_in_this_cluster += 1
        
    # ax1.legend(title="Tools", loc="upper left", bbox_to_anchor=(1.05, 1))
    ax1.grid(axis="y", linestyle="--", alpha=0.7)

    # Show the plot
    plt.tight_layout()
    if output_file:
        plt.savefig(output_file)
    if show:
        plt.show()
        plt.close()


def calculate_grouped_metric(grouped_df, y_metric, tool):
    TP_column = f"TP_{tool}"
    FP_column = f"FP_{tool}"
    FN_column = f"FN_{tool}"
    TN_column = f"TN_{tool}"
    mutation_expression_prediction_error_column = f"mutation_expression_prediction_error_{tool}"
    y_metric_output_column = f"{y_metric}_{tool}"

    if y_metric == "accuracy":
        grouped_df[y_metric_output_column] = (grouped_df[TP_column] + grouped_df[TN_column]) / (grouped_df[TP_column] + grouped_df[TN_column] + grouped_df[FP_column] + grouped_df[FN_column])
    elif y_metric == "sensitivity":
        grouped_df[y_metric_output_column] = grouped_df[TP_column] / (grouped_df[TP_column] + grouped_df[FN_column])
        grouped_df.loc[(grouped_df[TP_column] + grouped_df[FN_column]) == 0, y_metric] = 1.0
    elif y_metric == "specificity":
        grouped_df[y_metric_output_column] = grouped_df[TN_column] / (grouped_df[TN_column] + grouped_df[FP_column])
        grouped_df.loc[(grouped_df[TN_column] + grouped_df[FP_column]) == 0, y_metric] = 1.0
    elif y_metric == "mean_magnitude_expression_error" or y_metric == "mean_expression_error":
        grouped_df[y_metric_output_column] = (grouped_df[mutation_expression_prediction_error_column] / grouped_df["number_of_elements_in_the_group"])
    else:
        raise ValueError(f"Invalid y_metric: {y_metric}. Valid options are 'accuracy', 'sensitivity', 'specificity', and 'mutation_expression_prediction_error'")

    return grouped_df

def create_stratified_metric_line_plot(unique_mcrs_df, x_stratification, y_metric, tools, bins = None, keep_strict_bins = False, show_p_values = False, show_confidence_intervals = False, bonferroni = True, output_file = None, show = True, output_file_p_values = None, filter_real_negatives = False):
    assert x_stratification in unique_mcrs_df.columns, f"Invalid x_stratification: {x_stratification}"

    # removes unnecessary columns for the function
    columns_to_keep_for_function = list(set([x_stratification, 'included_in_synthetic_reads_mutant', 'number_of_reads_mutant']))
    for tool in tools:
        columns_to_keep_for_function.extend([f"TP_{tool}", f"TN_{tool}", f"FP_{tool}", f"FN_{tool}", f"mutation_expression_prediction_error_{tool}", f"mutation_detected_{tool}", f"DP_{tool}"])
    for column in columns_to_keep_for_function:
        if column not in unique_mcrs_df.columns:
            columns_to_keep_for_function.remove(column)

    unique_mcrs_df = unique_mcrs_df.loc[:, columns_to_keep_for_function].copy()  # make a copy to avoid modifying the original DataFrame
    # unique_mcrs_df = unique_mcrs_df.copy()  # make a copy to avoid modifying the original DataFrame

    if "expression_error" in y_metric and not filter_real_negatives and x_stratification not in {"number_of_reads_mutant"}:
        print("Warning: filtering real negatives is recommended when using expression error as a primary metric and stratifying by something other than number_of_reads_mutant, but this setting is not currently enabled. Recommended: filter_real_negatives = True.")

    if y_metric == "sensitivity" or filter_real_negatives:
        unique_mcrs_df = unique_mcrs_df[(unique_mcrs_df['included_in_synthetic_reads_mutant'] == True) & (unique_mcrs_df['number_of_reads_mutant'] > 0)]
    elif y_metric == "specificity":
        unique_mcrs_df = unique_mcrs_df[(unique_mcrs_df['included_in_synthetic_reads_mutant'] == False) & (unique_mcrs_df['number_of_reads_mutant'] == 0)]

    if keep_strict_bins:
        unique_mcrs_df = unique_mcrs_df[unique_mcrs_df[x_stratification].astype(int).isin(bins)]

    # Prepare for plotting
    plt.figure(figsize=(10, 6))

    x_values_raw = sorted(list(unique_mcrs_df[x_stratification].unique()))

    x_stratification_original = x_stratification  # to label x-axis, as x_stratification may be changed to "bin"

    if bins and not keep_strict_bins:  # remember bins are left-inclusive and right-exclusive
        if bins[-2] > x_values_raw[-1]:
            raise ValueError(f"Invalid bins: {bins}. The 2nd to last bin value {bins[-2]} is greater than the maximum value in the data {x_values_raw[-1]}")
        # list comprehension to assign labels list
        labels = [f"({bins[i]}, {bins[i+1]}]" for i in range(len(bins)-1)]  # eg bins [0, 0.25, 0.5, 0.75, 1] --> labels ["[0, 0.25)", "[0.25, 0.5)", "[0.5, 0.75)", "[0.75, 1)"]
        
        # replace "inf" with true start and end values
        if '-inf' in labels[0]:
            labels[0] = labels[0].replace('-inf', str(x_values_raw[0]))
        if 'inf' in labels[-1]:
            labels[-1] = labels[-1].replace('inf', str(x_values_raw[-1]))

        number_of_rows_before_filtering = len(unique_mcrs_df)
        # remove rows lower than lower bound or higher than upper bound
        unique_mcrs_df = unique_mcrs_df[(unique_mcrs_df[x_stratification] >= bins[0]) & (unique_mcrs_df[x_stratification] < bins[-1])]
        number_of_rows_after_filtering = len(unique_mcrs_df)
        
        if number_of_rows_before_filtering != number_of_rows_after_filtering:
            print(f"Removed {number_of_rows_before_filtering - number_of_rows_after_filtering} rows due to binning.")

        # Assign bins to a new column
        unique_mcrs_df["bin"] = pd.cut(unique_mcrs_df[x_stratification], bins=bins, labels=labels, right=True, include_lowest=False)

        x_values = labels
        x_stratification = "bin"
    else:
        x_values = x_values_raw

    if not keep_strict_bins:
        x_indices = range(len(x_values))
    else:
        x_indices = x_values
    
    if y_metric == "mutation_expression_prediction_error":
        for tool in tools:
            assert f"mutation_expression_prediction_error_{tool}" in unique_mcrs_df.columns, f"mutation_expression_prediction_error_{tool} not in unique_mcrs_df.columns"

    # created grouped_df
    if y_metric == "mean_magnitude_expression_error":  # calculate sum of magnitudes for this one column
        aggregation_functions = {}
        for tool in tools:
            # Group by tumor purity and calculate sensitivity
            aggregation_functions[f"mutation_expression_prediction_error_{tool}"] = lambda x: x.abs().sum()  # Sum of absolute values

        # Use the default sum for all other columns
        grouped_df = unique_mcrs_df.groupby(x_stratification).agg(
            {col: aggregation_functions.get(col, "sum") for col in unique_mcrs_df.columns if col != x_stratification}  # sum is the default aggregation function
        )
    else:  # including if y_metric == mean_expression_error:  # calculate sum for all columns
        grouped_df = unique_mcrs_df.groupby(x_stratification).sum(numeric_only = True)

    grouped_df["number_of_elements_in_the_group"] = unique_mcrs_df.groupby(x_stratification).size()
    
    # redundant code for calculating y-max (because I need this for setting p-value asterisk height)
    if y_metric in {"accuracy", "sensitivity", "specificity"}:
        custom_y_limit = 1.05
    else:
        custom_y_limit = 0
        for i, tool in enumerate(tools):
            grouped_df = calculate_grouped_metric(grouped_df, y_metric, tool)
            y_metric_tool_specific = f'{y_metric}_{tool}'
            custom_y_limit = max(custom_y_limit, grouped_df[y_metric_tool_specific].max())

    number_of_valid_p_values = 0
    nested_dict = lambda: defaultdict(nested_dict)
    metric_to_tool_to_p_value_dict_of_dicts = nested_dict()
    
    for i, tool in enumerate(tools):
        grouped_df = calculate_grouped_metric(grouped_df, y_metric, tool)
        y_metric_tool_specific = f'{y_metric}_{tool}'  # matches column created by calculate_grouped_metric - try not changing this name if possible

        # Plot sensitivity as a function of tumor purity
        plt.plot(x_indices, grouped_df[y_metric_tool_specific], label=tool, marker="o", color = color_map_20[i])  # use grouped_df.index to get the x-axis values and plot numerically (vs converting to categorical)

        if (show_p_values or output_file_p_values) and tool != "varseek":  # because varseek is the reference tool
            p_value_list = []
            confidence_intervals_list = []
            for x_value in x_values:
                filtered_unique_mcrs_df_for_p_value = unique_mcrs_df.loc[unique_mcrs_df[x_stratification] == x_value]
                if len(filtered_unique_mcrs_df_for_p_value) > 1:
                    number_of_valid_p_values += 1
                    if y_metric in {"accuracy", "sensitivity", "specificity"}:  # Mcnemar
                        p_value = calculate_individual_mcnemar(filtered_unique_mcrs_df_for_p_value, 'mutation_detected_varseek', f'mutation_detected_{tool}')
                        margin_of_error = 0

                    elif y_metric in {"mean_magnitude_expression_error"}:  # paired t-test
                        p_value = calculate_individual_paired_t_test(filtered_unique_mcrs_df_for_p_value, column1 = "mutation_expression_prediction_error_varseek", column2 = f"mutation_expression_prediction_error_{tool}", take_absolute_value = True)
                        _, margin_of_error = compute_95_confidence_interval_margin_of_error(filtered_unique_mcrs_df_for_p_value[f'mutation_expression_prediction_error_{tool}'], take_absolute_value = True)
                        

                    elif y_metric in {"mean_expression_error"}:
                        p_value = calculate_individual_paired_t_test(filtered_unique_mcrs_df_for_p_value, column1 = "mutation_expression_prediction_error_varseek", column2 = f"mutation_expression_prediction_error_{tool}", take_absolute_value = False)
                        _, margin_of_error = compute_95_confidence_interval_margin_of_error(filtered_unique_mcrs_df_for_p_value[f'mutation_expression_prediction_error_{tool}'], take_absolute_value = False)
                    
                    else:
                        raise ValueError(f"Invalid metric for p-value calculation: {y_metric}. Valid options are 'accuracy', 'sensitivity', 'specificity', 'mean_magnitude_expression_error', 'mean_expression_error'")
                    
                    p_value_list.append(p_value)
                    confidence_intervals_list.append(margin_of_error)
                else:
                    p_value_list.append(1.0)

            # bonferroni
            if bonferroni:
                n_tests = number_of_valid_p_values * (len(tools) - 1)  # because varseek is the reference tool
                p_value_list = [min((p_value * n_tests), 1.0) for p_value in p_value_list]
        
            # Plot '*' above points where p-value < 0.05
            if show_p_values:
                for x, y, p_value, margin_of_error in zip(x_indices, list(grouped_df[y_metric_tool_specific]), p_value_list, confidence_intervals_list):
                    if show_confidence_intervals and margin_of_error != 0:
                        # confidence interval errors
                        yerr = [margin_of_error, margin_of_error]

                        # Plot the point with the confidence interval
                        plt.errorbar(
                            x, y, 
                            yerr=np.array([yerr]).T,  # Transpose to match dimensions
                            fmt='',  # Marker for the point
                            capsize=5,  # Adds caps to the error bars
                            label="Mean with 95% CI",
                            color=color_map_20[i]
                        )

                    # p-values
                    metric_to_tool_to_p_value_dict_of_dicts[str(x)][tool] = p_value
                    if p_value >= 0.05:  #* increase these values to show more p-values for debugging
                        continue
                    elif p_value < 0.05 and p_value >= 0.01:
                        symbol = "*"
                    elif p_value < 0.01 and p_value >= 0.001:
                        symbol = "**"
                    else:
                        symbol = "***"
                    plt.text(x, y + (custom_y_limit*0.01), symbol, color=color_map_20[i], fontsize=12, ha='center')  # Slightly above the point
                    

    # # Set x-axis to log2 scale
    # if log:  # log can be False (default, not log) or True (defaults to 2) or int (log base)
    #     if log == True:
    #         log = 2
    #     plt.xscale("log", base=log)

    if output_file_p_values:
        with open(output_file_p_values, "w") as f:
            json.dump(metric_to_tool_to_p_value_dict_of_dicts, f, indent=4)

    # Customize plot
    if y_metric in {"accuracy", "sensitivity", "specificity"}:  #"accuracy" in primary_metrics or "sensitivity" in primary_metrics or "specificity" in primary_metrics:
        plt.ylim(0, custom_y_limit)

    x_values_new = []
    for x_value in x_values:  # add the number of elements in each stratification below the x-axis label
        number_of_elements = grouped_df.loc[grouped_df.index == x_value]['number_of_elements_in_the_group'].iloc[0]
        x_values_new.append(f"{x_value}\n(n={number_of_elements})")
    x_values = x_values_new

    plt.xticks(ticks=x_indices, labels=x_values)
    plt.xlabel(x_stratification_original, fontsize=12)
    plt.ylabel(y_metric, fontsize=12)
    # plt.legend(title="Tools")
    plt.grid(axis="y", linestyle="--", alpha=0.7)

    # Show the plot
    plt.tight_layout()
    if output_file:
        plt.savefig(output_file)
    if show:
        plt.show()
        plt.close()

def create_benchmarking_legend(tools, outfile = None, show = True):
    # Define colors
    colors = color_map_20[:len(tools)]

    # Create a figure for the legend
    fig, ax = plt.subplots(figsize=(0.1, 0.1))  # Adjust size as needed to change whitespace margins

    # Create proxy artists (invisible items for the legend)
    proxies = [plt.Line2D([0], [0], color=color, lw=4) for color in colors]

    # Add the legend to the figure
    ax.legend(proxies, tools, title="Legend", loc="center")
    ax.axis("off")  # Turn off the axes

    # Show the legend-only figure
    plt.tight_layout()
    if outfile:
        plt.savefig(outfile, bbox_inches="tight")
    if show:
        plt.show()
        plt.close()

def write_p_values_to_file(tool_to_p_value_dict, out_file):
    if out_file.endswith(".txt"):
        with open(out_file, "w") as f:
            for tool, p_value in tool_to_p_value_dict.items():
                f.write(f"{tool}: {p_value}\n")
    elif out_file.endswith(".json"):
        # Save to a file
        with open(out_file, "w") as f:
            json.dump(tool_to_p_value_dict, f, indent=4)
    else:
        raise ValueError(f"Invalid file extension: {out_file}. Accepted extensions are '.txt' and '.json'.")
        
def calculate_individual_mcnemar(df, column1, column2):
    # Counting the true/false values
    contingency_table = pd.crosstab(df[column1], df[column2])

    # McNemar's test
    exact = False if (contingency_table.values >= 25).all() else True
    result = mcnemar(contingency_table, exact=exact)
    return result.pvalue

# see the function above for simple mncnemar, as the following function was written specifically for figure 2c
def calculate_mcnemar(unique_mcrs_df, tools, metric = "accuracy", out_file = None, bonferroni = False, do_sensitivity_specificity_filtering = True):
    # unique_mcrs_df = unique_mcrs_df.copy()  # make a copy to avoid modifying the original DataFrame
    
    # Filtering the DataFrame
    if do_sensitivity_specificity_filtering:
        if metric == "sensitivity":
            filtered_df = unique_mcrs_df[unique_mcrs_df['included_in_synthetic_reads_mutant'] == True]
        elif metric == "specificity":
            filtered_df = unique_mcrs_df[unique_mcrs_df['included_in_synthetic_reads_mutant'] == False]
        elif metric == "accuracy":
            filtered_df = unique_mcrs_df
        else:
            raise ValueError(f"Invalid metric: {metric}. Accepted values are 'sensitivity', 'specificity', and 'accuracy'.")
    else:
        filtered_df = unique_mcrs_df
    

    tool_to_p_value_dict = {}
    for tool in tools:
        if tool == "varseek":
            continue  # Skip varseek as it is the reference tool

        tool_to_p_value_dict[tool] = calculate_individual_mcnemar(filtered_df, 'mutation_detected_varseek', f'mutation_detected_{tool}')

    # Bonferroni correction
    if bonferroni:
        n_tests = count_leaves(tool_to_p_value_dict)
        for tool in tools:
            tool_to_p_value_dict[tool] = min(tool_to_p_value_dict[tool] * n_tests, 1.0)

    if out_file:
        write_p_values_to_file(tool_to_p_value_dict, out_file)

    return tool_to_p_value_dict


def calculate_individual_paired_t_test(df, column1, column2, tails = 2, larger_column_expected = None, take_absolute_value = False):
    df = df.copy()  # make a copy to avoid modifying the original DataFrame
    
    if take_absolute_value:
        df[column1] = df[column1].abs()
        df[column2] = df[column2].abs()

    t_stat, p_value = ttest_rel(df[column1], df[column2])  # if one-tailed, then the sign of t_stat indicates if 1st arg > 2nd arg

    if tails == 1:
        if not larger_column_expected:
            raise ValueError("larger_column_expected must be provided when tails == 1")
        if larger_column_expected != column1 and larger_column_expected != column2:
            raise ValueError("larger_column_expected must be one of the two columns passed in")
        
        if t_stat < 0 and larger_column_expected == column1:  # corresponds to varseek being passed in first above
            p_value /= 2
        elif t_stat > 0 and larger_column_expected != column2:
            p_value /= 2
        else:
            p_value = 1.0

    return p_value

# make sure I've already computed the difference between predicted expression and true expression for each tool (i.e., DP_TOOL - number_of_reads_mutant)
# two-tailed - to make one-tailed, divide by 2 and make sure that 
def calculate_paired_t_test(unique_mcrs_df, column_root, tools, take_absolute_value = False, out_file = None, bonferroni = False, tails = 2, larger_column_expected = "varseek"):
    unique_mcrs_df = unique_mcrs_df.copy()  # make a copy to avoid modifying the original DataFrame

    tool_to_p_value_dict = {}
    for tool in tools:
        if tool == "varseek":
            continue  # Skip varseek as it is the reference tool

        if larger_column_expected == "varseek":
            larger_column_expected = f'{column_root}_varseek'
        elif larger_column_expected in tools:
            larger_column_expected = f'{column_root}_{tool}'

        p_value = calculate_individual_paired_t_test(unique_mcrs_df, column1 = f'{column_root}_varseek', column2 = f'{column_root}_{tool}', take_absolute_value = take_absolute_value, tails = tails, larger_column_expected = larger_column_expected)

        tool_to_p_value_dict[tool] = p_value

    # Bonferroni correction
    if bonferroni:
        n_tests = count_leaves(tool_to_p_value_dict)
        for tool in tools:
            tool_to_p_value_dict[tool] = min(tool_to_p_value_dict[tool] * n_tests, 1.0)

    if out_file:
        write_p_values_to_file(tool_to_p_value_dict, out_file)

    return tool_to_p_value_dict

def print_json(json_file):
    with open(json_file, "r") as f:
        data = json.load(f)
    print(json.dumps(data, indent=4))

def count_leaves(d):
    """
    Recursively counts the number of leaves in a nested dictionary.
    
    Args:
        d (dict): The dictionary to traverse.
    
    Returns:
        int: The count of leaves.
    """
    if not isinstance(d, dict):  # Base case: not a dictionary
        return 1
    return sum(count_leaves(value) for value in d.values())

def compute_95_confidence_interval_margin_of_error(values, take_absolute_value = False):
    # values = unique_mcrs_df[column]
    if take_absolute_value:
        values = values.abs()
    
    # Step 1: Compute mean
    mean = np.mean(values)

    # Step 2: Compute Standard Error of the Mean (SEM)
    sem = np.std(values, ddof=1) / np.sqrt(len(values))

    # Step 3: Degrees of freedom
    df = len(values) - 1

    # Step 4: Critical t-value for 95% confidence
    t_critical = t.ppf(0.975, df)  # Two-tailed 95% confidence

    # Step 5: Margin of error
    margin_of_error = t_critical * sem

    # # Step 6: Compute confidence interval
    # ci_lower = mean - margin_of_error
    # ci_upper = mean + margin_of_error

    return mean, margin_of_error  # , ci_lower, ci_upper



def create_stratified_metric_bar_plot_updated(unique_mcrs_df, x_stratification, y_metric, tools, display_numbers = False, show_p_values = False, show_confidence_intervals = True, bonferroni = True, output_file = None, show = True, output_file_p_values = None, filter_real_negatives = False):
    assert x_stratification in unique_mcrs_df.columns, f"Invalid x_stratification: {x_stratification}"

    # removes unnecessary columns for the function
    columns_to_keep_for_function = list(set([x_stratification, 'included_in_synthetic_reads_mutant', 'number_of_reads_mutant']))
    for tool in tools:
        columns_to_keep_for_function.extend([f"TP_{tool}", f"TN_{tool}", f"FP_{tool}", f"FN_{tool}", f"mutation_expression_prediction_error_{tool}", f"mutation_detected_{tool}", f"DP_{tool}"])
    for column in columns_to_keep_for_function:
        if column not in unique_mcrs_df.columns:
            columns_to_keep_for_function.remove(column)
    
    unique_mcrs_df = unique_mcrs_df.loc[:, columns_to_keep_for_function].copy()  # make a copy to avoid modifying the original DataFrame
    # unique_mcrs_df = unique_mcrs_df.copy()  # make a copy to avoid modifying the original DataFrame

    if x_stratification == "mcrs_mutation_type":
        # remove any values in "mcrs_mutation_type" equal to "mixed"
        unique_mcrs_df = unique_mcrs_df[unique_mcrs_df['mcrs_mutation_type'] != "mixed"]

    if "expression_error" in y_metric and not filter_real_negatives and x_stratification not in {"number_of_reads_mutant"}:
        print("Warning: filtering real negatives is recommended when using expression error as a primary metric and stratifying by something other than number_of_reads_mutant, but this setting is not currently enabled. Recommended: filter_real_negatives = True.")

    if y_metric == "sensitivity" or filter_real_negatives:
        unique_mcrs_df = unique_mcrs_df[unique_mcrs_df['included_in_synthetic_reads_mutant'] == True]
    elif y_metric == "specificity":
        unique_mcrs_df = unique_mcrs_df[unique_mcrs_df['included_in_synthetic_reads_mutant'] == False]

    # Prepare for plotting
    plt.figure(figsize=(10, 6))

    if y_metric == "mutation_expression_prediction_error":
        for tool in tools:
            assert f"mutation_expression_prediction_error_{tool}" in unique_mcrs_df.columns, f"mutation_expression_prediction_error_{tool} not in unique_mcrs_df.columns"

    # created grouped_df
    if y_metric == "mean_magnitude_expression_error":  # calculate sum of magnitudes for this one column
        aggregation_functions = {}
        for tool in tools:
            # Group by tumor purity and calculate sensitivity
            aggregation_functions[f"mutation_expression_prediction_error_{tool}"] = lambda x: x.abs().sum()  # Sum of absolute values

        # Use the default sum for all other columns
        grouped_df = unique_mcrs_df.groupby(x_stratification).agg(
            {col: aggregation_functions.get(col, "sum") for col in unique_mcrs_df.columns if col != x_stratification}  # sum is the default aggregation function
        )
    else:  # including if y_metric == mean_expression_error:  # calculate sum for all columns
        grouped_df = unique_mcrs_df.groupby(x_stratification).sum(numeric_only = True)

    grouped_df["number_of_elements_in_the_group"] = unique_mcrs_df.groupby(x_stratification).size()
    
    # redundant code for calculating y-max (because I need this for setting p-value asterisk height)
    if y_metric in {"accuracy", "sensitivity", "specificity"}:
        custom_y_limit = 1.05
    else:
        custom_y_limit = 0
        for i, tool in enumerate(tools):
            grouped_df = calculate_grouped_metric(grouped_df, y_metric, tool)
            y_metric_tool_specific = f'{y_metric}_{tool}'
            custom_y_limit = max(custom_y_limit, grouped_df[y_metric_tool_specific].max())

    number_of_valid_p_values = 0
    nested_dict = lambda: defaultdict(nested_dict)
    stratification_to_tool_to_p_value_dict_of_dicts = nested_dict()
    stratification_to_tool_to_error_dict_of_dicts = nested_dict()

    # Prepare data
    bar_names = unique_mcrs_df[x_stratification].unique()
    x_primary = np.arange(len(bar_names))  # Positions for the metrics on the x-axis
    bar_width = 0.25  # Width of each primary_metrics
    offsets = np.arange(len(tools)) * bar_width - (len(tools) - 1) * bar_width / 2  # Centered offsets

    # Create the plot
    y_values_primary_total = []  # y_values_primary_total holds all metrics across all tools - I could make this a nested dict if desired
    for i, tool in enumerate(tools):
        grouped_df = calculate_grouped_metric(grouped_df, y_metric, tool)
        y_metric_tool_specific = f'{y_metric}_{tool}'  # matches column created by calculate_grouped_metric - try not changing this name if possible
        y_values_primary = [grouped_df.loc[grouped_df.index == bar_name, y_metric_tool_specific][0] for bar_name in bar_names]  # y_values_primary just holds all metrics for a specific tool
        bars_primary = plt.bar(x_primary + offsets[i], y_values_primary, bar_width, label=tool, color=color_map_20[i])

        # Add value annotations for the primary metrics
        if display_numbers:
            for bar, value in zip(bars_primary, y_values_primary):
                plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                        f"{value:.3f}", ha="center", va="bottom", fontsize=10)
                
        y_values_primary_total.extend(y_values_primary)
                
        if (show_p_values or output_file_p_values) and tool != "varseek":  # because varseek is the reference tool
            p_value_list = []
            confidence_intervals_list = []
            for x_value in bar_names:
                filtered_unique_mcrs_df_for_p_value = unique_mcrs_df.loc[unique_mcrs_df[x_stratification] == x_value]
                if len(filtered_unique_mcrs_df_for_p_value) > 1:
                    number_of_valid_p_values += 1
                    if y_metric in {"accuracy", "sensitivity", "specificity"}:  # Mcnemar
                        p_value = calculate_individual_mcnemar(filtered_unique_mcrs_df_for_p_value, 'mutation_detected_varseek', f'mutation_detected_{tool}')
                        mean_value, margin_of_error = 0, 0

                    elif y_metric in {"mean_magnitude_expression_error"}:  # paired t-test
                        p_value = calculate_individual_paired_t_test(filtered_unique_mcrs_df_for_p_value, column1 = "mutation_expression_prediction_error_varseek", column2 = f"mutation_expression_prediction_error_{tool}", take_absolute_value = True)
                        mean_value, margin_of_error = compute_95_confidence_interval_margin_of_error(filtered_unique_mcrs_df_for_p_value[f'mutation_expression_prediction_error_{tool}'], take_absolute_value = True)

                    elif y_metric in {"mean_expression_error"}:
                        p_value = calculate_individual_paired_t_test(filtered_unique_mcrs_df_for_p_value, column1 = "mutation_expression_prediction_error_varseek", column2 = f"mutation_expression_prediction_error_{tool}", take_absolute_value = False)
                        mean_value, margin_of_error = compute_95_confidence_interval_margin_of_error(filtered_unique_mcrs_df_for_p_value[f'mutation_expression_prediction_error_{tool}'], take_absolute_value = False)
                    
                    else:
                        raise ValueError(f"Invalid metric for p-value calculation: {y_metric}. Valid options are 'accuracy', 'sensitivity', 'specificity', 'mean_magnitude_expression_error', 'mean_expression_error'")
                else:
                    p_value = 1.0
                    mean_value, margin_of_error = 0, 0

                p_value_list.append(p_value)
                confidence_intervals_list.append(margin_of_error)
                stratification_to_tool_to_p_value_dict_of_dicts[x_value][tool] = p_value
                stratification_to_tool_to_error_dict_of_dicts[x_value][tool] = (mean_value, margin_of_error)

            # bonferroni
            if bonferroni:
                n_tests = number_of_valid_p_values * (len(tools) - 1)  # because varseek is the reference tool
                p_value_list = [min((p_value * n_tests), 1.0) for p_value in p_value_list]
                for x_value in bar_names:
                    stratification_to_tool_to_p_value_dict_of_dicts[x_value][tool] = min((stratification_to_tool_to_p_value_dict_of_dicts[x_value][tool] * n_tests), 1.0)
        
    # Plot '*' above points where p-value < 0.05
    if show_p_values:
        for i, bar_name in enumerate(bar_names):
            if bar_name in stratification_to_tool_to_p_value_dict_of_dicts:
                number_of_p_values_in_this_cluster = 0
                for j, tool in enumerate(tools):
                    # 95% confidence intervals
                    if bar_name in stratification_to_tool_to_error_dict_of_dicts and tool in stratification_to_tool_to_error_dict_of_dicts[bar_name]:
                        # Calculate error values
                        mean_value, margin_of_error = stratification_to_tool_to_error_dict_of_dicts[bar_name][tool]
                        if margin_of_error != 0:
                            yerr = [margin_of_error, margin_of_error]
                            x_value = x_primary[i] + offsets[j]

                            # Plot the point with the confidence interval
                            plt.errorbar(
                                x_value, mean_value, 
                                yerr=np.array([yerr]).T,  # Transpose to match dimensions
                                fmt='',  # Marker for the point
                                capsize=5,  # Adds caps to the error bars
                                label="Mean with 95% CI",
                                color="black"
                            )

                    # p-values
                    if tool in stratification_to_tool_to_p_value_dict_of_dicts[bar_name]:
                        p_value = stratification_to_tool_to_p_value_dict_of_dicts[bar_name][tool]

                        if p_value >= 0.05:  #* increase these values to show more p-values for debugging
                            continue
                        elif p_value < 0.05 and p_value >= 0.01:
                            symbol = "*"
                        elif p_value < 0.01 and p_value >= 0.001:
                            symbol = "**"
                        else:
                            symbol = "***"
                        
                        start_x = x_primary[i] + offsets[0]  # assuming varseek is first element
                        end_x = x_primary[i] + offsets[j]

                        if y_metric in {"accuracy", "sensitivity", "specificity"}:
                            y_start = max(y_values_primary_total) + (number_of_p_values_in_this_cluster * 0.05) + 0.05
                        else:
                            y_start = (max(y_values_primary_total) + (number_of_p_values_in_this_cluster * 1.7)) * 1.08  # 1.7 (left constant) adjusts based on other bars; 1.08 (right constant) adjusts to make sure it doesn't hit the top bar
                        
                        y_end = y_start

                        plt.plot([start_x, start_x, end_x, end_x], [y_start, y_end, y_end, y_start], lw=1.5, c="k")  # plot the bar
                        plt.text((start_x + end_x) * .5, y_end, symbol, ha='center', va='bottom', color="k")  # plot the asterisk(s)

                        number_of_p_values_in_this_cluster += 1

    if output_file_p_values:
        with open(output_file_p_values, "w") as f:
            json.dump(stratification_to_tool_to_p_value_dict_of_dicts, f, indent=4)
    
    if y_metric in {"accuracy", "sensitivity", "specificity"}:  #"accuracy" in primary_metrics or "sensitivity" in primary_metrics or "specificity" in primary_metrics:
        plt.ylim(0, custom_y_limit)

    x_values_new = []
    for x_value in bar_names:  # add the number of elements in each stratification below the x-axis label
        number_of_elements = grouped_df.loc[grouped_df.index == x_value]['number_of_elements_in_the_group'].iloc[0]
        x_values_new.append(f"{x_value}\n(n={number_of_elements})")
    bar_names = x_values_new

    plt.xticks(ticks=x_primary, labels=bar_names, fontsize=12)
    plt.ylabel(y_metric, fontsize=12)

    # Show the plot
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.tight_layout()
    if output_file:
        plt.savefig(output_file)
    if show:
        plt.show()
        plt.close()



def plot_frequency_histogram(unique_mcrs_df, column_base, tools, fraction = False, output_file = None, show = True):
    """
    Plots a histogram of the mutation expression prediction errors for each tool.
    """
    errors_dict = {}

    plt.figure(figsize=(10, 6))
    for index, tool in enumerate(tools):
        errors_dict[tool] = unique_mcrs_df.loc[unique_mcrs_df[f'FP_{tool}'], f'{column_base}_{tool}']
        if fraction:
            total_count = len(errors_dict[tool])  # Total number of errors for this tool
            weights = [1 / total_count] * total_count  # Fractional weights for each error
            y_axis_label = "Fraction of FPs"
        else:
            weights = None  # No weights for absolute counts
            y_axis_label = "Number of FPs"
        plt.hist(errors_dict[tool], bins=30, alpha=0.6, label=tool, color=color_map_20[index], weights=weights)

    # Add labels, legend, and title
    plt.xscale('log', base=2)

    # Customize ticks to show all powers of 2
    log_locator = LogLocator(base=2.0, subs=[], numticks=30)  # `subs=[]` means only major ticks are shown
    log_formatter = FuncFormatter(lambda x, _: f'{int(x)}' if x >= 1 else '')

    ax = plt.gca()
    ax.xaxis.set_major_locator(log_locator)
    ax.xaxis.set_major_formatter(log_formatter)


    plt.xlabel('Counts Detected')
    plt.ylabel(y_axis_label)
    plt.title('Histogram of Counts Detected for FPs')
    plt.legend(loc='upper right')

    # Show the plot
    plt.tight_layout()

    if output_file:
        plt.savefig(output_file)
    if show:
        plt.show()
        plt.close()