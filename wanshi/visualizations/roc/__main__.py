#!/usr/bin/env python3
from argparse import ArgumentParser
from pathlib import Path

import pandas as pd
from matplotlib import pyplot as plt

from . import (
    plot_decorated_rocs_for_subtypes,
    plot_multiple_decorated_roc_curves,
    plot_single_decorated_roc_curve,
    split_preds_into_groups,
)


def add_roc_curve_args(parser: ArgumentParser) -> ArgumentParser:
    parser.add_argument(
        "pred_csvs",
        metavar="PREDS_CSV",
        nargs="+",
        type=Path,
        help="Predictions to create ROC curves for.",
    )
    parser.add_argument(
        "--target-label",
        metavar="LABEL",
        required=True,
        type=str,
        help="The target label to calculate the ROC for.",
    )
    parser.add_argument(
        "--true-label",
        metavar="LABEL",
        required=True,
        type=str,
        help="The target label to calculate the ROC for.",
    )
    parser.add_argument(
        "-o",
        "--outpath",
        metavar="PATH",
        required=True,
        type=Path,
        help=(
            "Path to save the ROC to.  "
            "Has to have an image extension (e.g. `.svg`, `.png`, etc.)"
        ),
    )
    parser.add_argument(
        "--subgroup-label",
        metavar="LABEL",
        required=False,
        type=str,
        help="Column name in Clini where to get the subgroups from.",
    )
    parser.add_argument(
        "--subgroup",
        metavar="SUBGROUP",
        dest="subgroups",
        required=False,
        type=str,
        action="append",
        help=(
            "A subgroup to include in the ouput.  "
            "If none are given, a ROC curve for each of the subgroups will be created."
        ),
    )
    parser.add_argument(
        "--clini-table",
        metavar="PATH",
        required=False,
        type=Path,
        help="Path to get subgroup information from clini table from.",
    )
    parser.add_argument(
        "--n-bootstrap-samples",
        metavar="N",
        type=int,
        required=False,
        help="Number of bootstrapping samples to take for confidence interval generation.",
    )
    parser.add_argument(
        "--threshold-cmap",
        metavar="COLORMAP",
        type=plt.get_cmap,
        required=False,
        help="Draw Curve with threshold color.",
    )

    return parser


def read_table(path: Path) -> pd.DataFrame:
    """Loads a dataframe from a file."""
    match path.suffix:
        case ".xlsx":
            return pd.read_excel(path)
        case ".csv":
            return pd.read_csv(path)
        case suffix:
            raise ValueError("unknown filetype!", suffix)


if __name__ == "__main__":
    parser = ArgumentParser(description="Create a ROC Curve.")
    add_roc_curve_args(parser)
    parser.add_argument(
        "--figure-width",
        metavar="INCHES",
        type=float,
        required=False,
        help="Width of the figure in inches.",
        default=3.8,
    )
    args = parser.parse_args()

    # read all the patient preds
    # and transform their true / preds columns into np arrays
    preds_dfs = [pd.read_csv(p, dtype=str) for p in args.pred_csvs]
    y_trues = [df[args.target_label] == args.true_label for df in preds_dfs]
    y_preds = [
        pd.to_numeric(df[f"{args.target_label}_{args.true_label}"]) for df in preds_dfs
    ]

    roc_curve_figure_aspect_ratio = 1.08
    fig, ax = plt.subplots(
        figsize=(args.figure_width, args.figure_width * roc_curve_figure_aspect_ratio),
        dpi=300,
    )

    if len(preds_dfs) == 1:
        if args.subgroup_label:
            assert (
                len(preds_dfs) == 1
            ), "currently subgroup analysis is only supported for a singular set of predictions"
            if not args.clini_table:
                parser.error("missing argument: --clini-table")

            groups = split_preds_into_groups(
                preds_df=preds_dfs[0],
                clini_df=read_table(args.clini_table),
                target_label=args.target_label,
                true_label=args.true_label,
                subgroup_label=args.subgroup_label,
            )

            plot_decorated_rocs_for_subtypes(
                ax,
                groups,
                target_label=args.target_label,
                true_label=args.true_label,
                subgroup_label=args.subgroup_label,
                subgroups=args.subgroups,
                n_bootstrap_samples=args.n_bootstrap_samples,
            )
        else:
            plot_single_decorated_roc_curve(
                ax,
                y_trues[0],
                y_preds[0],
                title=f"{args.target_label} = {args.true_label}",
                n_bootstrap_samples=args.n_bootstrap_samples,
                threshold_cmap=args.threshold_cmap,
            )

    else:
        plot_multiple_decorated_roc_curves(
            ax,
            y_trues,
            y_preds,
            title=f"{args.target_label} = {args.true_label}",
            n_bootstrap_samples=args.n_bootstrap_samples,
        )

    fig.tight_layout()
    fig.savefig(args.outpath)
    plt.close(fig)
