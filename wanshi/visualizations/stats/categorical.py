#!/usr/bin/env python3
"""Calculate statistics for deployments on categorical targets."""

from pathlib import Path
from argparse import ArgumentParser

import pandas as pd
import scipy.stats as st
from sklearn import metrics
from tqdm import trange


__author__ = "Marko van Treeck"
__copyright__ = "Copyright 2022, Kather Lab"
__license__ = "MIT"
__version__ = "0.2.0"
__maintainer__ = "Marko van Treeck"
__email__ = "markovantreeck@gmail.com"


__all__ = ["categorical", "aggregate_categorical_stats", "bootstrapped_categorical"]


score_labels = [
    "roc_auc_score",
    "average_precision_score",
    "precision_score",
    "recall_score",
    "p_value",
    "count",
]


def categorical(preds_df: pd.DataFrame, target_label: str) -> pd.DataFrame:
    """Calculates some stats for categorical prediction tables.

    This will calculate the number of items, the AUROC, AUPRC and p value
    for a prediction file.
    """
    categories = preds_df[target_label].unique()
    y_true = preds_df[target_label]
    y_pred = (
        preds_df[[f"{target_label}_{cat}" for cat in categories]].applymap(float).values
    )

    stats_df = pd.DataFrame(index=categories)

    # class counts
    stats_df["count"] = pd.value_counts(y_true)

    # roc_auc
    stats_df["roc_auc_score"] = [
        metrics.roc_auc_score(y_true == cat, y_pred[:, i])
        if (y_true == cat).nunique() == 2
        else None  # in case the AUROC is not defined
        for i, cat in enumerate(categories)
    ]

    # average_precision
    stats_df["average_precision_score"] = [
        metrics.average_precision_score(y_true == cat, y_pred[:, i])
        for i, cat in enumerate(categories)
    ]

    # precision
    stats_df["precision_score"] = [
        metrics.precision_score(y_true == cat, y_pred[:, i] > 0.5)
        for i, cat in enumerate(categories)
    ]

    # recall
    stats_df["recall_score"] = [
        metrics.recall_score(y_true == cat, y_pred[:, i] > 0.5)
        for i, cat in enumerate(categories)
    ]

    # p values
    p_values = []
    for i, cat in enumerate(categories):
        pos_scores = y_pred[:, i][y_true == cat]
        neg_scores = y_pred[:, i][y_true != cat]
        p_values.append(st.ttest_ind(pos_scores, neg_scores).pvalue)
    stats_df["p_value"] = p_values

    assert set(score_labels) & set(stats_df.columns) == set(score_labels)

    return stats_df


def aggregate_categorical_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Calculates confidence intervals by grouping along the first multiindex level."""
    stats = {}
    for cat, data in df.groupby("level_1"):
        scores_df = data[score_labels]
        means, sems = scores_df.mean(), scores_df.sem()
        l, h = st.t.interval(
            confidence=0.95, df=len(scores_df) - 1, loc=means, scale=sems
        )
        cat_stats_df = (
            pd.DataFrame.from_dict({"mean": means, "95% conf": (h - l) / 2})
            .transpose()
            .unstack()
        )
        cat_stats_df[("count", "sum")] = data["count"].sum()
        stats[cat] = cat_stats_df

    return pd.DataFrame.from_dict(stats, orient="index")


def bootstrapped_categorical(
    preds_df: pd.DataFrame, *, target_label: str, n_samples: int = 10000
) -> pd.DataFrame:
    """Calculates categorical stats confidence intervals by bootstrapping."""
    # sample repeatedly and calculate statistics for each iteration
    sample_stats = []
    for _ in trange(n_samples, desc="Bootstrapping stats", leave=False):
        sample_df = preds_df.sample(frac=1, replace=True)
        sample_stats.append(categorical(sample_df, target_label))

    # calculate mean stats & 95% confidence intervals
    grouped = (
        pd.concat(sample_stats).reset_index(names=["category"]).groupby("category")
    )
    means = grouped.mean()
    lower = grouped.quantile(".025")
    upper = grouped.quantile(".975")
    confs = (upper - lower) / 2

    # We aggregate the stats in such a roundabout way to have the items in the
    # same order as the `categorical` function gave them to us, with the "mean"
    # and "95% conf" labels as a second level column header
    stats = {}
    for category in means.index:
        cat_stats_df = pd.DataFrame.from_dict(
            {"mean": means.loc[category], "95% conf": confs.loc[category]},
            orient="index",
        ).unstack()
        stats[category] = cat_stats_df

    return pd.DataFrame.from_dict(stats).transpose()


def add_stats_categorical_args(parser: ArgumentParser) -> ArgumentParser:
    parser.add_argument(
        "preds_csvs",
        metavar="PREDS_CSV",
        nargs="+",
        type=Path,
        help="CSV file containing predictions.",
    )
    parser.add_argument(
        "-o",
        "--outpath",
        required=True,
        type=Path,
        help="Directory to write statistics to.",
    )
    parser.add_argument(
        "--target-label",
        required=True,
        type=str,
        help="Target to generate statistics for.",
    )
    parser.add_argument(
        "--n-bootstrap-samples",
        type=int,
        help=(
            "Number of samples used during bootstrapping.  "
            "If not specified, confidence intervals will be calculated over the given PREDS_CSVs."
        ),
    )

    return parser


if __name__ == "__main__":
    parser = ArgumentParser(
        description="Calculate statistics for categorical deployments."
    )
    parser = add_stats_categorical_args(parser)
    args = parser.parse_args()

    args.outpath.mkdir(parents=True, exist_ok=True)

    if args.n_bootstrap_samples:
        # for bootstrapping we sample from all the available predictions
        preds_df = pd.concat([pd.read_csv(p, dtype=str) for p in args.preds_csvs])
        bootstrapped_stats = bootstrapped_categorical(
            preds_df, target_label=args.target_label, n_samples=args.bootstrap_n_samples
        )
        bootstrapped_stats.to_csv(
            args.outpath / f"{args.target_label}-categorical-stats-bootstrapped.csv"
        )
    else:
        # calculate 95% confidence intervals by comparing the different prediction CSVs' stats
        preds_dfs = {
            Path(p).parent.name: categorical(
                pd.read_csv(p, dtype=str), args.target_label
            )
            for p in args.preds_csvs
        }
        preds_df = pd.concat(preds_dfs).sort_index()
        preds_df.to_csv(
            args.outpath / f"{args.target_label}-categorical-stats-individual.csv"
        )
        stats_df = aggregate_categorical_stats(preds_df.reset_index())
        stats_df.to_csv(
            args.outpath / f"{args.target_label}-categorical-stats-aggregated.csv"
        )
