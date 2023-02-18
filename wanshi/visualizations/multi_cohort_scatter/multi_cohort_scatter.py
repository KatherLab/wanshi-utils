# !/usr/bin/env python3

__author__ = "Marco Gustav"
__copyright__ = "Copyright 2023, Kather Lab"
__license__ = "MIT"
__version__ = "0.1.0"
__maintainer__ = ["Marco Gustav", "Jeff"]
__email__ = "marco.gustav@tu-dresden.de"

import argparse
import os
from pathlib import Path
from typing import List

import matplotlib.patches as ptch
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn import metrics

# pd float shown with 2 decimal places
pd.set_option("display.float_format", lambda x: "%.5f" % x)


def plot_bubble(
    data_path_internal: str,
    data_paths_external: List[str],
    title: str,
    outpath: str,
    internal_cohort_name: str,
    color_scheme: list,
    format: str = "norm",
    zoom_x_lower_thresh: float = 0.75,
    zoom_x_upper_thresh: float = 0.95,
    zoom_y_lower_thresh: float = 0.75,
    zoom_y_upper_thresh: float = 0.95,
) -> None:
    """
    Plots a bubble plot of the median AUC values of multiple target predictions for multiple cohorts.

    Parameters:
        data_path_internal (str): The path to the internal cross-validation results.
        data_paths_external (list): The paths to the external validation results. path to the folder with subfolders that are called as 'format'
        title (str): The title of the plot.
        internal_cohort_name (str): The name of the internal cohort.
        format: List of strings with the format of the data (raw, norm). Can potentially be used as hue.
        color_scheme List(str): The color scheme of the plot.
        outpath (Path): Output path for saving .svg.
        zoom_x_lower_thresh (float): Left threshold for zoom box.
        zoom_x_upper_thresh (float): Right threshold for zoom box.
        zoom_y_lower_thresh (float): Bottom threshold for zoom box.
        zoom_y_upper_thresh (float): Top threshold for zoom box.

    Returns:
        None
    """

    # first element in list is always internal crossval
    data_paths = [data_path_internal] + data_paths_external

    # TODO: add colormap input
    # color scheme
    if color_scheme is None:
        color_scheme = [
            "#66CCEE",
            "#EE6677",
            "#4477AA",
            "#228833",
            "#CCBB44",
            "#AA3377",
        ]

    # create cohorts list for adding cohorts from path endings
    cohorts = []
    # create dataframe to fill with the content of the result files
    results = pd.DataFrame(columns=["target", "cohort", "format", "fold", "auc"])

    # fill dataframe with data
    # for each cohort in paths
    for path_name in data_paths:
        # be aware: path is extended by form argument
        data_path_form = f"{path_name}/{format}/"
        # fill list with cohort names taken from last string in path name
        cohort = str(path_name).split("/")[-1]
        cohorts.append(cohort)
        # read patient preds files and extract names and values of/for the predicted targets
        for folder, sub_folders, files in os.walk(data_path_form):
            target = folder.split(os.path.sep)[-2]
            for file in files:
                if not file.endswith("patient-preds.csv"):
                    continue
                preds = pd.read_csv(os.path.join(folder, file))
                # replace entries that are non binary with binary labels
                label_0 = preds.columns[3].split("_")[-1]
                label_1 = preds.columns[4].split("_")[-1]
                preds[f"{target}"] = preds[f"{target}"].replace(
                    [label_0, label_1], [0, 1]
                )
                # calculate auroc
                try:
                    fpr, tpr, thresholds = metrics.roc_curve(
                        preds[f"{target}"], preds[f"{target}_{label_1}"]
                    )
                    auc = metrics.auc(fpr, tpr)
                    # append new row to the DataFrame
                    new_row = pd.DataFrame(
                        {
                            "target": f"{target}",
                            "format": f"{folder.split(os.path.sep)[-3]}",
                            "fold": f"{folder.split(os.path.sep)[-1]}",
                            "auc": f"{auc}",
                            "cohort": f"{cohort}",
                        },
                        index=[0],
                    )
                    # concatenate existing df with new data from new_row
                    results = pd.concat([results, new_row], ignore_index=True)
                except ValueError:
                    continue
                except TypeError:
                    continue

    # manipulate dataframe for plotting
    results["auc"] = results["auc"].astype(float)
    # calculate median from all folds from crossvalidation
    results["median"] = results.groupby(["target", "cohort"])["auc"].transform("median")
    # create new df with median values
    median = results[["target", "cohort", "median"]].copy()
    median.drop_duplicates(inplace=True)
    median.reset_index(drop=True, inplace=True)
    median_pivot = median.pivot(index=["target"], columns="cohort", values="median")

    # create plot
    # input
    data = median_pivot.copy()
    # figure layout
    fig_width = 40
    fig_height = 15
    fig, axes = plt.subplots(1, 2, figsize=(fig_width, fig_height))
    fontsize = 20
    plt.suptitle(title, ha="center", va="top", fontsize=fontsize * 1.5)
    plt.subplots_adjust(top=0.95)
    # settings
    # sns.set(font_scale=2)
    fig.canvas.draw()

    # empty list for the annotations in the zoom
    annotations = []

    # create plots and annotations
    for cohort_num in range(1, 1 + len(cohorts[1:])):
        ax1 = sns.scatterplot(
            ax=axes[0],
            data=data,
            y=data[cohorts[0]],
            x=data[cohorts[cohort_num]],
            s=150,
            color=color_scheme[cohort_num - 1],
            alpha=0.7,
        )
        ax2 = sns.scatterplot(
            ax=axes[1],
            data=data,
            y=data[cohorts[0]],
            x=data[cohorts[cohort_num]],
            s=300,
            color=color_scheme[cohort_num - 1],
            alpha=0.7,
        )

        for i in range(data.shape[0]):
            # only annotate dots if they are in zoom range
            if (
                data[cohorts[cohort_num]][i] > zoom_x_lower_thresh
                and data[cohorts[0]][i] > zoom_y_lower_thresh
            ):
                x_coord = data[cohorts[cohort_num]][i]
                y_coord = data[cohorts[0]][i]
                x_text_coord = x_coord + 0.01
                y_text_coord = y_coord + 0.01
                annotations.append(
                    plt.annotate(
                        data[cohorts[cohort_num]].index[i],
                        xy=(x_coord, y_coord),
                        xytext=(x_text_coord, y_text_coord),
                        arrowprops=dict(
                            facecolor="tab:gray",
                            edgecolor="tab:gray",
                            linewidth=0.5,
                            alpha=0.5,
                        ),
                        rotation=0,
                        fontsize=fontsize,
                    )
                )
            else:
                pass

    # modify layout
    ax1.set_title("a")
    ax2.set_title("b")

    for ax in axes:
        ax.grid(visible=True, axis="both")
        ax.set_aspect("equal", "box")
        ax.set_xlabel("Median AUROC for external validation cohorts")
        ax.set_ylabel(
            f"Median AUROC for internal validation cohort: {internal_cohort_name}"
        )

        for item in (
            [ax.title, ax.xaxis.label, ax.yaxis.label]
            + ax.get_xticklabels()
            + ax.get_yticklabels()
        ):
            item.set_fontsize(fontsize)

    # create zoom patch for highlighting zoom region in ax1
    zoom = ptch.Rectangle(
        (zoom_x_lower_thresh, zoom_y_lower_thresh),
        zoom_x_upper_thresh - zoom_x_lower_thresh,
        zoom_y_upper_thresh - zoom_x_lower_thresh,
        fill=False,
        edgecolor="tab:gray",
        lw=2,
    )
    # ax1
    ax1.set(ylim=(0, 1), xlim=(0, 1), yticks=np.arange(0, 1.1, 0.1))
    ax1.axhline(
        y=0.5, color="tab:gray", label=False, linewidth=5, linestyle="--", alpha=0.5
    )
    ax1.axvline(
        x=0.5, color="tab:gray", label=False, linewidth=5, linestyle="--", alpha=0.5
    )
    # add zoom box
    ax1.add_patch(zoom)
    # Add text to the bottom right corner of the box
    ax1.annotate(
        f"{ax2.get_title()}",
        xy=(zoom_x_upper_thresh - 0.03, zoom_y_lower_thresh + 0.01),
        fontsize=fontsize,
    )

    # ax2
    ax2.set(
        ylim=(zoom_y_lower_thresh, zoom_y_upper_thresh),
        xlim=(zoom_x_lower_thresh, zoom_x_upper_thresh),
        yticks=np.arange(zoom_y_lower_thresh, 0.96, 0.05),
        xticks=np.arange(zoom_y_lower_thresh, 0.96, 0.05),
    )

    # create dict with numbers starting at 1 mapped to labels shown in zomm region
    name_to_num = {}

    annotation_names = set(ann.get_text() for ann in annotations)

    for i, name in enumerate(annotation_names):
        name_to_num[name] = i + 1

    # replace name by num
    for ann in annotations:
        name = ann.get_text()
        num = name_to_num[name]
        ann.set_text(num)

    # create custom legends
    # cohort names (color coded in figure)
    leg1 = ax1.legend(
        labels=cohorts[1:],
        bbox_to_anchor=(1.05, 1.0),
        bbox_transform=ax2.transAxes,
        loc="upper left",
        borderaxespad=0.0,
        fancybox=True,
        fontsize=fontsize,
    )

    # get height of top legend to align second legend below it
    leg_1_height = leg1.get_window_extent().height
    # Get the DPI of the figure
    dpi = fig.dpi
    # Convert the legend height from pixels to inches
    legend_height_inches = leg_1_height / dpi

    # legend 2 elements
    legend_elements = [
        plt.scatter([0], [0], marker="o", color="w", label=f"{num}: {name}")
        for name, num in name_to_num.items()
    ]

    # create second custom legend with numbers and labels
    leg2 = ax2.legend(
        handles=legend_elements,
        bbox_to_anchor=(1.05, zoom_y_upper_thresh - 0.1 * legend_height_inches),
        bbox_transform=ax2.transAxes,
        loc="upper left",
        borderaxespad=0.0,
        handlelength=0,
        handletextpad=0,
        fancybox=True,
        fontsize=fontsize,
    )

    for item in leg2.legendHandles:
        item.set_visible(False)

    # final aesthetics
    fig.subplots_adjust(right=0.75, left=0.08, bottom=-0.1, top=1.1)

    plt.savefig(f"{args.outpath}/{internal_cohort_name}.svg")
    plt.show()


def add_multi_cohort_scatter_args(
    parser: argparse.ArgumentParser,
) -> argparse.ArgumentParser:
    parser.add_argument(
        "--data-path-internal",
        required=True,
        type=Path,
        help="Folder path for results of internal crossvalidation. Contains folders with results for norm and/or raw that contain one folder for each target.",
    )
    parser.add_argument(
        "--internal-cohort-name",
        required=True,
        type=str,
        help="Name of cohort that was internal crossval performed on.",
    )
    parser.add_argument(
        "--data-paths-external",
        required=True,
        action="append",
        type=Path,
        help="List of folder paths for results of external crossvalidation. End of path should contain cohort name. Contains folders with results for norm and/or raw that contain one folder for each target.",
    )
    parser.add_argument(
        "--format",
        required=True,
        type=str,
        help="Format of tiles that were used. 'norm' or 'raw' for example. Should be the same as the name in the subfolder in your cohort folder.",
    )
    parser.add_argument(
        "--title",
        required=True,
        type=str,
        help="Title of figure.",
    )
    parser.add_argument(
        "--outpath", required=True, type=Path, help="Path to save the `.svg` to."
    )
    parser.add_argument(
        "--color-scheme",
        required=False,
        type=list,
        help="List with color codes in HEX.",
    )

    return parser


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create multi cohort scatterplot.")
    add_multi_cohort_scatter_args(parser)
    args = parser.parse_args()

    plot_bubble(
        data_path_internal=args.data_path_internal,
        data_paths_external=args.data_paths_external,
        title=args.title,
        internal_cohort_name=args.internal_cohort_name,
        format=args.format,
        color_scheme=args.color_scheme,
        outpath=args.outpath,
    )
