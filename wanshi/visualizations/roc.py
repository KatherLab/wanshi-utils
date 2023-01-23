#!/usr/bin/env python3
import argparse
from collections import namedtuple
import logging
from pathlib import Path
from typing import Sequence, Optional, Tuple, Mapping, List

from matplotlib import pyplot as plt
import numpy as np
import numpy.typing as npt
import pandas as pd
import scipy.stats as st
from sklearn.metrics import roc_curve, roc_auc_score
from tqdm import trange

all = [
    "plot_roc_curve",
    "plot_roc_curves",
    "plot_rocs_for_subtypes",
    "plot_roc_curves_",
]


def add_roc_curve_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument(
        "pred_csvs",
        metavar="PREDS_CSV",
        nargs="+",
        type=Path,
        help="Predictions to create ROC curves for.",
    )
    parser.add_argument(
        "--target-label",
        required=True,
        type=str,
        help="The target label to calculate the ROC for.",
    )
    parser.add_argument(
        "--true-label",
        required=True,
        type=str,
        help="The target label to calculate the ROC for.",
    )
    parser.add_argument(
        "-o", "--outpath", required=True, type=Path, help="Path to save the `.svg` to."
    )
    parser.add_argument(
        "--subgroup-label",
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
        required=False,
        type=Path,
        help="Path to get subgroup information from Clini table from.",
    )
    parser.add_argument("--n-bootstrap-samples", type=int)

    return parser


def plot_single_decorated_roc_curve(
    ax: plt.Axes,
    y_true: npt.NDArray[np.bool_],
    y_pred: npt.NDArray[np.float_],
    *,
    title: Optional[str] = None,
    n_bootstrap_samples: Optional[int] = None,
) -> plt.Axes:
    """Plots a single ROC curve.

    Args:
        ax:  Axis to plot to.
        y_true:  A sequence of ground truths.
        y_pred:  A sequence of predictions.
        title:  Title of the plot.
    """
    plot_bootstrapped_roc_curve(
        ax, y_true, y_pred, label="AUC = {ci}", n_bootstrap_samples=n_bootstrap_samples
    )
    style_auc(ax)
    if title:
        ax.set_title(title)


def auc_str(auc: float, conf_range: Optional[float]) -> str:
    if conf_range:
        return f"AUC = ${auc:0.2f} \pm {conf_range:0.2f}$"
    else:
        return f"AUC = ${auc:0.2f}$"


def style_auc(ax: plt.Axes) -> None:
    ax.plot([0, 1], [0, 1], "r--")
    ax.set_aspect("equal")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.legend(loc="lower right")


TPA = namedtuple("TPA", ["true", "pred", "auc"])


def plot_multiple_decorated_roc_curves(
    ax: plt.Axes,
    y_trues: Sequence[npt.NDArray[np.bool_]],
    y_scores: Sequence[npt.NDArray[np.float_]],
    *,
    title: Optional[str] = None,
    n_bootstrap_samples: Optional[int] = None,
):
    """Plots a family of ROC curves.

    Args:
        ax:  Axis to plot to.
        y_trues:  Sequence of ground truth lists.
        y_scores:  Sequence of prediction lists.
        title:  Title of the plot.
    """
    # sort trues, preds, AUCs by AUC
    tpas = [TPA(t, p, roc_auc_score(t, p)) for t, p in zip(y_trues, y_scores)]
    tpas = sorted(tpas, key=lambda x: x.auc, reverse=True)

    # plot rocs
    for t, p, auc in tpas:
        auc, _ = plot_bootstrapped_roc_curve(
            ax, t, p, label="AUC = {ci}", n_bootstrap_samples=n_bootstrap_samples
        )

    # style plot
    style_auc(ax)

    # calculate confidence intervals and print title
    aucs = [x.auc for x in tpas]
    l, h = st.t.interval(0.95, len(aucs) - 1, loc=np.mean(aucs), scale=st.sem(aucs))
    conf_range = (h - l) / 2

    if title:
        ax.set_title(f"{title}\n({auc_str(auc, conf_range)})")
    else:
        ax.set_title(auc_str(auc, conf_range))


def split_preds_into_groups(
    preds_df: pd.DataFrame,
    *,
    clini_df: pd.DataFrame,
    target_label: str,
    true_label: str,
    subgroup_label: str,
) -> Mapping[str, Tuple[npt.NDArray[np.bool_], npt.NDArray[np.float_]]]:
    """Splits predictions into a mapping `subgroup_name -> (y_true, y_pred)."""
    groups = {}
    for subgroup, subgroup_patients in clini_df.PATIENT.groupby(
        clini_df[subgroup_label]
    ):
        subgroup_preds = preds_df[preds_df.PATIENT.isin(subgroup_patients)]
        y_true = subgroup_preds[target_label] == true_label
        y_pred = pd.to_numeric(subgroup_preds[f"{target_label}_{true_label}"])
        groups[subgroup] = (y_true.values, y_pred.values)

    return groups


def plot_decorated_rocs_for_subtypes(
    ax: plt.Axes,
    groups: Mapping[str, Tuple[npt.NDArray[np.bool_], npt.NDArray[np.float_]]],
    *,
    target_label: str,
    subgroup_label: str,
    subgroups: Optional[Sequence[str]] = None,
    n_bootstrap_samples: Optional[int] = None,
) -> None:
    """Plots a ROC for multiple groups."""
    tpas: List[Tuple[str, TPA]] = []
    for subgroup, (y_true, y_pred) in groups.items():
        if subgroups and subgroup not in subgroups:
            continue

        if len(np.unique(y_true)) <= 1:
            logging.warn(
                f"subgroup {subgroup} does only have samples of one class... skipping"
            )
            continue

        tpas.append((subgroup, TPA(y_true, y_pred, roc_auc_score(y_true, y_pred))))

    # sort trues, preds, AUCs by AUC
    tpas = sorted(tpas, key=lambda x: x[1].auc, reverse=True)

    # plot rocs
    for subgroup, (t, s, _) in tpas:
        plot_bootstrapped_roc_curve(
            ax=ax,
            y_true=t,
            y_score=s,
            label=f"{target_label} for {subgroup} (AUC = {{ci}})",
            n_bootstrap_samples=n_bootstrap_samples,
        )

    # style plot
    style_auc(ax)
    ax.legend(loc="lower right")
    ax.set_title(f"{target_label} Subgrouped by {subgroup_label}")


def plot_bootstrapped_roc_curve(
    ax: plt.Axes,
    y_true: npt.NDArray[np.bool_],
    y_score: npt.NDArray[np.float_],
    label: Optional[str],
    n_bootstrap_samples: Optional[int] = None,
):
    """Plots a roc curve with bootstrap interval.

    Args:
        ax:  The axes to plot onto.
        y_true:  The ground truths.
        y_score:  The predictions corresponding to the ground truths.
        label:  A label to attach to the curve.
            The string `{ci}` will be replaced with the AUC
            and the range of the confidence interval.
    """
    assert len(y_true) == len(y_score), "length of truths and scores does not match."
    conf_range = None
    if n_bootstrap_samples:
        # draw some confidence intervals based on bootstrapping
        # sample repeatedly (with replacement) from our data points,
        # interpolate along the resulting ROC curves
        # and then sample the bottom 0.025 / top 0.975 quantile point
        # for each sampled fpr-position
        rng = np.random.default_rng()
        interp_rocs = []
        interp_fpr = np.linspace(0, 1, num=1000)
        bootstrap_aucs = []
        for _ in trange(
            n_bootstrap_samples, desc="Bootstrapping ROC curves", leave=False
        ):
            sample_idxs = rng.choice(len(y_true), len(y_true))
            sample_y_true = y_true[sample_idxs]
            sample_y_score = y_score[sample_idxs]
            if len(np.unique(sample_y_true)) != 2:
                continue
            fpr, tpr, _ = roc_curve(sample_y_true, sample_y_score)
            interp_rocs.append(np.interp(interp_fpr, fpr, tpr))
            bootstrap_aucs.append(roc_auc_score(sample_y_true, sample_y_score))

        lower = np.quantile(interp_rocs, 0.025, axis=0)
        upper = np.quantile(interp_rocs, 0.975, axis=0)
        ax.fill_between(interp_fpr, lower, upper, alpha=0.5)
        conf_range = (
            np.quantile(bootstrap_aucs, 0.975) - np.quantile(bootstrap_aucs, 0.025)
        ) / 2

    fpr, tpr, _ = roc_curve(y_true, y_score)
    auc = roc_auc_score(y_true, y_score)
    ci_str = f"${auc:0.2f} \pm {conf_range:0.2f}$" if conf_range else f"${auc:0.2f}$"
    ax.plot(fpr, tpr, label=label.format(ci=ci_str) if label else "")
    return auc, conf_range


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
    parser = argparse.ArgumentParser(description="Create a ROC Curve.")
    add_roc_curve_args(parser)
    parser.add_argument(
        "--figure-width",
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
