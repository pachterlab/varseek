import os
from collections import Counter
import shutil
import scanpy as sc
from rich.table import Table
from rich.console import Console

console = Console()

import numpy as np
import pandas as pd
import anndata as ad

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator, MultipleLocator
from matplotlib.patches import Rectangle
from matplotlib_venn import venn2
import seaborn as sns

from scipy import stats
from scipy.sparse import csr_matrix

from varseek.constants import complement, codon_to_amino_acid, mutation_pattern

# Set global settings
plt.rcParams.update({
    'savefig.dpi': 450,             # Set resolution to 450 dpi
    'font.family': 'Arial',         # Set font to Arial
    'pdf.fonttype': 42,             # Embed fonts as TrueType (keeps text editable)
    'ps.fonttype': 42,              # Same for PostScript files
    'savefig.format': 'pdf',        # Default save format as PNG
    'savefig.bbox': 'tight',        # Adjust bounding box to fit tightly
    'figure.facecolor': 'white',    # Set figure background to white (common for RGB)
    'savefig.transparent': False,   # Disable transparency
})

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
    df, header_name=None, check_assertions=False, crude=False, out=None
):
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
    print(
        f"Accuracy: {accuracy}, Sensitivity: {sensitivity}, Specificity: {specificity}"
    )

    if check_assertions:
        assert int(accuracy) == 1, "Accuracy is not 1"
        assert int(sensitivity) == 1, "Sensitivity is not 1"
        assert int(specificity) == 1, "Specificity is not 1"

    metric_dictionary = {
        "accuracy": accuracy,
        "sensitivity": sensitivity,
        "specificity": specificity,
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


def draw_confusion_matrix(metric_dictionary_reads):
    # Sample dictionary with confusion matrix values
    confusion_matrix = {
        "TP": metric_dictionary_reads["TP"],  # True Positive
        "TN": metric_dictionary_reads["TN"],  # True Negative
        "FP": metric_dictionary_reads["FP"],  # False Positive
        "FN": metric_dictionary_reads["FN"],  # False Negative
    }

    # Create a Rich Table to display the confusion matrix
    table = Table(title="Confusion Matrix")

    # Add columns for the table
    table.add_column("", justify="center")
    table.add_column("Predicted Positive", justify="center")
    table.add_column("Predicted Negative", justify="center")

    # Add rows for the confusion matrix
    table.add_row(
        "Actual Positive", str(confusion_matrix["TP"]), str(confusion_matrix["FN"])
    )
    table.add_row(
        "Actual Negative", str(confusion_matrix["FP"]), str(confusion_matrix["TN"])
    )

    # Display the table
    console.print(table)


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
