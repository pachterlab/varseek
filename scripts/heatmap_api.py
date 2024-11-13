import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from scripts.cbioportal_api import ints_between


def draw_heatmap(
        data: pd.DataFrame,
        title: str,
        x_label: str,
        y_label: str,
        scale_label: str | None = None,
        colormap: str = 'Reds',
        save_path: str = None,
        dpi: int = 300,
        show: bool = True,
        size: tuple[int, int] = (18, 4)
):
    """
    Draw a heatmap of the given data.

    :param data:        The data to be plotted.
    :param title:       The title of the plot.
    :param x_label:     The label of the x-axis.
    :param y_label:     The label of the y-axis.

    :param scale_label: The label of the scale. If None, no scale will be shown. (Default: None)
    :param colormap:    The colormap to be used. (Default: 'Reds')
    :param save_path:   The path where the plot will be saved. If None, the plot will not be saved. (Default: None)
    :param dpi:         The resolution of the saved plot. (Default: 300)
    :param show:        If True, the plot will be shown. (Default: True)
    :param size:        The size of the plot. (Default: (18, 4))

    :return: None
    """

    plt.figure(figsize=size)

    min_value = int(data.min().min())

    if data.isna().sum().sum() != 0:
        min_value -= 1
        nas_present = True
    else:
        nas_present = False

    max_value = max(int(data.max().max()), 1)

    levels = list(range(min_value, max_value + 1))
    data = data.fillna(min_value)

    colors_list = plt.get_cmap(colormap, len(levels))(range(len(levels)))
    if nas_present:
        colors_list = np.vstack([[0.5, 0.5, 0.5, 0.3], colors_list])  # grey color for -1
    cmap = ListedColormap(colors_list)

    # define the norm, with vmin set to min_value and vmax set to max_value
    norm = BoundaryNorm(
        boundaries=np.arange(min_value - 0.5, max_value + 1.5, 1),
        ncolors=cmap.N,
        clip=False
    )

    plt.imshow(data, cmap=cmap, norm=norm, aspect="auto")

    if scale_label:
        levels = ints_between(min_value, max_value, 25, 7)

        cbar = plt.colorbar(label=scale_label, ticks=levels)

        labels: list[str | int] = levels.copy()
        if nas_present:
            labels[0] = "NaN"

        cbar.ax.set_yticklabels(labels)

    plt.grid(
        which='both',
        axis='both',
        color='black',
        linestyle='-',
        linewidth=0.5
    )

    x_labels = data.columns

    plt.xticks(np.arange(len(data.columns)), x_labels, rotation=90)
    plt.yticks(np.arange(len(data.index)), data.index)

    plt.gca().set_xticks(np.arange(-0.5, len(data.columns), 1), minor=True)
    plt.gca().set_yticks(np.arange(-0.5, len(data.index), 1), minor=True)

    plt.grid(which="major", color="white", linestyle="-", linewidth=0.5, alpha=0)
    plt.gca().tick_params(which="minor", size=0)

    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)

    if show:
        plt.show()

    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=dpi)
