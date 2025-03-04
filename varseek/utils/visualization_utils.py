"""varseek visualization utilities."""

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Rectangle
from matplotlib.ticker import MaxNLocator, MultipleLocator
from scipy import stats

# Set global settings
plt.rcParams.update(
    {
        "savefig.dpi": 450,  # Set resolution to 450 dpi
        "font.family": "DejaVu Sans",  # Set font to Arial  # TODO: replace with Arial for Nature
        "pdf.fonttype": 42,  # Embed fonts as TrueType (keeps text editable)
        "ps.fonttype": 42,  # Same for PostScript files
        "savefig.format": "pdf",  # Default save format as PNG
        "savefig.bbox": "tight",  # Adjust bounding box to fit tightly
        "figure.facecolor": "white",  # Set figure background to white (common for RGB)
        "savefig.transparent": False,  # Disable transparency
    }
)

color_map_10 = plt.get_cmap("tab10").colors  # Default color map with 10 colors

color_map_20_original = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf", "#aec7e8", "#ffbb78", "#98df8a", "#ff9896", "#c5b0d5", "#c49c94", "#f7b6d2", "#c7c7c7", "#dbdb8d", "#9edae5"]  # plotly category 20

color_map_20 = ["#f08925", "#1f77b4", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf", "#aec7e8", "#ffbb78", "#98df8a", "#ff9896", "#c5b0d5", "#c49c94", "#f7b6d2", "#c7c7c7", "#dbdb8d", "#9edae5"]  # modified to swap 1 and 2 (orange first), and replaced the orange with varseek orange

SAVE_PDF_GLOBAL = os.getenv("VARSEEK_SAVE_PDF") == "TRUE"
DPI = 450


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

    stats_summary = f"Statistics for '{column}':\n" f"Mean: {kmers_mean}\n" f"Median: {kmers_median}\n" f"Mode: {kmers_mode}\n" f"Max Value: {kmers_max}\n" f"Variance: {kmers_variance}\n"

    # Save the statistics to a text file
    if output_file is not None:
        if os.path.exists(output_file):
            writing_mode = "a"  # Append to the file if it already exists
        else:
            writing_mode = "w"
        with open(output_file, writing_mode, encoding="utf-8") as f:
            f.write(stats_summary)

    # Print out the results to the console as well
    print(stats_summary)


def plot_histogram_notebook_1(df_overlap, column, x_label="x-axis", title="Histogram", output_plot_file=None, show=False):
    # Define the bin range for the histograms
    bins = range(0, df_overlap[column].max() + 2)  # Bins for k-mers

    # Replace 0 values with a small value to be represented on log scale
    data = df_overlap[column].replace(0, 1e-10)  # Replace 0 with a very small value (1e-10)

    # Plot histogram
    plt.figure(figsize=(10, 6))
    plt.hist(data, bins=bins, alpha=0.7, color="blue", edgecolor="black", log=True)  # log=True for log scale
    plt.xlabel(x_label)
    plt.ylabel("Frequency (log10 scale)")
    plt.title(title)

    # Set y-axis to log10 with ticks at every power of 10
    plt.yscale("log")
    plt.gca().yaxis.set_major_locator(plt.LogLocator(base=10.0))  # Ensure ticks are at powers of 10
    plt.gca().yaxis.set_minor_locator(plt.LogLocator(base=10.0, subs=np.arange(1.0, 10.0) * 0.1, numticks=10))
    plt.gca().yaxis.set_minor_formatter(plt.NullFormatter())  # Hide minor tick labels

    # Set x-ticks dynamically
    max_value = df_overlap[column].max()
    step = max(1, int(np.ceil(max_value / 10)))  # Adjust for dynamic tick spacing
    plt.gca().xaxis.set_major_locator(plt.MultipleLocator(step))

    plt.grid(True, which="major", axis="y", ls="-")  # Grid lines for both major and minor ticks

    if output_plot_file:
        plt.savefig(output_plot_file, format="png", dpi=DPI, bbox_inches="tight")
        if SAVE_PDF_GLOBAL:
            plt.savefig(output_plot_file.replace(".png", ".pdf"), format="pdf", dpi=DPI)

    if show:
        plt.show()
    plt.close()


def plot_histogram_of_nearby_mutations_7_5(mutation_metadata_df, column, bins, output_file=None, show=False):
    plt.figure(figsize=(10, 6))
    plt.hist(mutation_metadata_df[column], bins=bins, color="skyblue", edgecolor="black")

    # Set titles and labels
    plt.title(f"Histogram of {column}", fontsize=16)
    plt.xlabel(column, fontsize=12)
    plt.ylabel("Frequency", fontsize=12)
    # plt.xscale('log')
    plt.yscale("log")

    # Display the plot
    plt.tight_layout()

    if output_file:
        plt.savefig(output_file, format="png", dpi=DPI)
        if SAVE_PDF_GLOBAL:
            plt.savefig(output_file.replace(".png", ".pdf"), format="pdf", dpi=DPI)

    if show:
        plt.show()
    plt.close()


def retrieve_value_from_metric_file(key_of_interest, metric_file):
    metrics = {}
    with open(metric_file, "r", encoding="utf-8") as file:
        for line in file:
            key, value = line.strip().split(": ")
            metrics[key] = value

    value_of_interest = metrics.get(key_of_interest)
    return value_of_interest


def calculate_metrics(df, header_name=None, check_assertions=False, crude=False, out=None, suffix=""):
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

    accuracy, sensitivity, specificity = calculate_sensitivity_specificity(TP, TN, FP, FN)

    print(f"Accuracy: {accuracy}, Sensitivity: {sensitivity}, Specificity: {specificity}")

    if f"mutation_expression_prediction_error{suffix}" in df.columns:
        mean_expression_error = df[f"mutation_expression_prediction_error{suffix}"].mean()
        median_expression_error = df[f"mutation_expression_prediction_error{suffix}"].median()
        mean_magnitude_expression_error = df[f"mutation_expression_prediction_error{suffix}"].abs().mean()
        median_magnitude_expression_error = df[f"mutation_expression_prediction_error{suffix}"].abs().median()
        print(f"Mean Expression Error: {mean_expression_error}, Median Expression Error: {median_expression_error}, Mean Magnitude Expression Error: {mean_magnitude_expression_error}, Median Magnitude Expression Error: {median_magnitude_expression_error}")
    else:
        mean_expression_error = "N/A"
        median_expression_error = "N/A"
        mean_magnitude_expression_error = "N/A"
        median_magnitude_expression_error = "N/A"

    if check_assertions:
        if int(accuracy) != 1:
            raise AssertionError(f"Accuracy is not 1: {accuracy}")
        if int(sensitivity) != 1:
            raise AssertionError(f"Sensitivity is not 1: {sensitivity}")
        if int(specificity) != 1:
            raise AssertionError(f"Specificity is not 1: {specificity}")
        if f"mutation_expression_prediction_error{suffix}" in df.columns:
            if int(mean_magnitude_expression_error) != 0:
                raise AssertionError(f"Mean magnitude expression error is not 0: {mean_magnitude_expression_error}")

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
        with open(out, "w", encoding="utf-8") as file:
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
        grouped_df[y_metric] = (grouped_df[TP_column] + grouped_df[TN_column]) / (grouped_df[TP_column] + grouped_df[TN_column] + grouped_df[FP_column] + grouped_df[FN_column])  # len(grouped_df)
    elif y_metric == "sensitivity":
        grouped_df[y_metric] = grouped_df[TP_column] / (grouped_df[TP_column] + grouped_df[FN_column])
        grouped_df.loc[(grouped_df[TP_column] + grouped_df[FN_column]) == 0, y_metric] = 1.0
    elif y_metric == "specificity":
        grouped_df[y_metric] = grouped_df[TN_column] / (grouped_df[TN_column] + grouped_df[FP_column])
        grouped_df.loc[(grouped_df[TN_column] + grouped_df[FP_column]) == 0, y_metric] = 1.0
    elif y_metric == "expression_error":
        grouped_df[y_metric] = grouped_df["mutation_expression_prediction_error"] / grouped_df["count"]
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


def create_stratified_metric_bar_plot(df, x_stratification, y_metric, overall_metric=None, log_x_axis=False, bins=None, x_axis_name=None, y_axis_name=None, title=None, display_numbers=False, out_path=None, crude=False, show=True):
    if bins is not None:
        labels = convert_number_bin_into_labels(bins)
        df["binned_" + x_stratification] = pd.cut(df[x_stratification], bins=bins, labels=labels, right=True)

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
        grouped_df = df.groupby(group_col)["mutation_expression_prediction_error"].apply(lambda x: x.abs().mean()).reset_index()
        grouped_df.rename(columns={"mutation_expression_prediction_error": y_metric}, inplace=True)
        grouped_df[y_metric] += 0.05
        bottom_value = -0.05

    # # add counts to each row
    # group_size = df.groupby(group_col).size().reset_index(name='count')
    # group_size = grouped_df.merge(group_size, on=group_col)

    # Create a bar chart where the x axis is number_of_reads_mutant, and y axis is accuracy
    if group_col == "number_of_reads_mutant":
        had_zero = (grouped_df[group_col].astype(int) == 0).any()
        grouped_df = grouped_df[grouped_df[group_col].astype(int) != 0].reset_index(drop=True)
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

    if y_metric in {"accuracy", "sensitivity", "specificity"}:
        plt.ylim(0, 1)
        plt.yticks(np.arange(0, 1.1, 0.1))  # Major ticks every 0.1
        plt.minorticks_on()
        plt.gca().yaxis.set_minor_locator(plt.MultipleLocator(0.05))  # Minor ticks every 0.05

    if log_x_axis:
        plt.xscale("log")

    if bins is None and not isinstance(grouped_df[x_stratification][0], str):
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
            step_size = 5 * ((range_x // 10) // 5 + 1)  # Ensure step size is a multiple of 5
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
            plt.gca().xaxis.set_minor_locator(plt.NullLocator())  # No minor ticks if step size is 1

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
        if out_path is True:
            out_path = f"{y_metric}_vs_{x_stratification}.png"
        plt.savefig(out_path, bbox_inches="tight", dpi=DPI)
        if SAVE_PDF_GLOBAL:
            plt.savefig(out_path.replace(".png", ".pdf"), format="pdf", dpi=DPI)

    if show:
        plt.show()
    plt.close()


def create_venn_diagram(true_set, positive_set, TN=None, mm=None, out_path=None, show=True):
    from matplotlib_venn import venn2

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
        venn.get_label_by_id("10").set_text(f'FN: {venn.get_label_by_id("10").get_text()}{mm_line}')

    if venn.get_label_by_id("01") is not None:  # Make sure the intersection exists
        venn.get_patch_by_id("01").set_color("blue")  # Right circle (Set 2)
        venn.get_label_by_id("01").set_fontsize(10)  # Label for Set 2 only
        venn.get_label_by_id("01").set_text(f'FP: {venn.get_label_by_id("01").get_text()}')

    if venn.get_patch_by_id("11") is not None:
        venn.get_patch_by_id("11").set_color("green")  # Intersection of Set 1 and Set 2
        venn.get_label_by_id("11").set_fontsize(10)  # Label for the intersection
        venn.get_label_by_id("11").set_text(f'TP: {venn.get_label_by_id("11").get_text()}')

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
    if show:
        plt.show()
    plt.close()


def plot_histogram(df, column_name, bins=None, log_scale=False, x_axis_label=None, y_axis_label=None, title=None, out_path=None, show=False):
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
        bins = list(x for x in bins if x >= 0)

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
        plt.savefig(out_path, bbox_inches="tight", dpi=DPI)
        if SAVE_PDF_GLOBAL:
            plt.savefig(out_path.replace(".png", ".pdf"), format="pdf", dpi=DPI)

    if show:
        plt.show()
    plt.close()


def synthetic_data_summary_plot(df, column, sort_ascending=True, out_path=None, show=False):
    # Step 1: Calculate the counts of each unique value
    value_counts = df[column].value_counts()

    # Step 2: Convert counts to percentages
    percentages = (value_counts / len(df)) * 100

    if sort_ascending:
        try:
            percentages = percentages.sort_index(ascending=True)
        except Exception:
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
            ax.xaxis.set_major_locator(MaxNLocator(integer=True))  # Ensure x-ticks are integers
        except ValueError:
            pass

    plt.xticks(rotation=45, ha="right")  # Rotate x labels if they are long
    plt.tight_layout()

    if out_path is not None:
        plt.savefig(out_path)
        if SAVE_PDF_GLOBAL:
            plt.savefig(out_path.replace(".png", ".pdf"), format="pdf", dpi=DPI)

    # Show the plot
    if show:
        plt.show()
    plt.close()


def plot_basic_bar_plot_from_dict(my_dict, y_axis, log_scale=False, output_file=None, show=False):
    plt.figure(figsize=(8, 6))
    plt.bar(list(my_dict.keys()), list(my_dict.values()), color="black", alpha=0.8)
    plt.ylabel(y_axis)

    # log y scale
    if log_scale:
        plt.yscale("log")

    plt.tight_layout()

    if output_file:
        plt.savefig(output_file, format="png", dpi=DPI)
        if SAVE_PDF_GLOBAL:
            plt.savefig(output_file.replace(".png", ".pdf"), format="pdf", dpi=DPI)

    if show:
        plt.show()
    plt.close()


def plot_descending_bar_plot(gene_counts, x_label, y_label, tick_interval=None, output_file=None, show=False):
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
        plt.savefig(output_file, format="png", dpi=DPI)
        if SAVE_PDF_GLOBAL:
            plt.savefig(output_file.replace(".png", ".pdf"), format="pdf", dpi=DPI)

    if show:
        plt.show()
    plt.close()


def draw_confusion_matrix(metric_dictionary_reads, title="Confusion Matrix", title_color="black", suffix="", additional_fp_key="", output_file=None, show=True):
    confusion_matrix = {
        "TP": str(metric_dictionary_reads[f"TP{suffix}"]),  # True Positive
        "TN": str(metric_dictionary_reads[f"TN{suffix}"]),  # True Negative
        "FP": str(metric_dictionary_reads[f"FP{suffix}"]),  # False Positive
        "FN": str(metric_dictionary_reads[f"FN{suffix}"]),  # False Negative
    }

    if additional_fp_key in metric_dictionary_reads:
        additional_fp_text = " ".join(additional_fp_key.split()[1:])  # so if the key is FP including non-cosmic, then the text will be non-cosmic
        confusion_matrix["FP"] += f"\n{additional_fp_text}: {metric_dictionary_reads[additional_fp_key]}"

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


def draw_confusion_matrix_rich(metric_dictionary_reads, title="Confusion Matrix", suffix="", additional_fp_key=""):
    from rich.console import Console
    from rich.table import Table

    console = Console()

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
    table.add_row("Actual Positive", str(confusion_matrix["TP"]), str(confusion_matrix["FN"]))
    table.add_row("Actual Negative", fp_line, str(confusion_matrix["TN"]))

    # Display the table
    console.print(table)


def find_specific_value_from_metric_text_file(file_path, line):
    # file must be \n-separated and have the format "line: value"

    value = None

    # Read the file and extract the value
    with open(file_path, "r", encoding="utf-8") as file:
        for looping_line in file:
            if line in looping_line:
                value = int(looping_line.split(":")[1].strip())
                return value

    return value


def plot_kat_histogram(kat_hist, out_path=None, show=False):
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
    plt.xlabel("k-mer Frequency")
    plt.ylabel("# of Distinct k-mers")
    plt.title("k-mer Spectra for random_sequences.fasta")

    # Save the plot
    plt.savefig(out_path, format="png", dpi=DPI)
    if SAVE_PDF_GLOBAL:
        plt.savefig(out_path.replace(".png", ".pdf"), format="pdf", dpi=DPI)

    # Display the plot
    if show:
        plt.show()
    plt.close()


def plot_items_descending_order(df, x_column, y_column, item_range=(0, 10), xlabel="x-axis", title="Title", save_path=None, figsize=(15, 7), show=False):
    # Plot the line plot
    plt.figure(figsize=figsize)

    first_item = item_range[0]
    last_item = item_range[1]

    if len(df) <= first_item:
        raise ValueError(f"First item index {first_item} is out of bounds")

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
        plt.savefig(save_path, dpi=DPI)
        if SAVE_PDF_GLOBAL:
            plt.savefig(save_path.replace(".png", ".pdf"), format="pdf", dpi=DPI)

    # Show the plot
    if show:
        plt.show()
    plt.close()


def plot_knee_plot(umi_counts_sorted, knee_locator, min_counts_assessed_by_knee_plot=None, output_file=None, show=False):
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
    if show:
        plt.show()
    plt.close()
