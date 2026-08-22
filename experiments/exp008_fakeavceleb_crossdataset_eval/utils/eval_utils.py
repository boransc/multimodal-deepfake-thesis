# source calibration / external metric helpers used by exp008

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def load_source_score(
    fusion_results,
    split,
):
    """
    load one AV-Deepfake1M++ fusion-score csv produced by exp007.

    these source-domain train/validation/test scores are what exp008 uses for
    threshold calibration and learned fusion fitting. FakeAVCeleb labels are not
    used for those steps.
    """
    path = (
        Path(fusion_results)
        / f"{split}_fusion_scores.csv"
    )

    if not path.exists():
        raise FileNotFoundError(
            path
        )

    return pd.read_csv(
        path
    )


def safe_auc(
    labels,
    probabilities,
):
    """
    calculate roc auc when both classes are present.

    some condition-specific subsets contain only one class. auc is undefined
    there, so np.nan is returned instead of raising an exception.
    """
    labels = np.asarray(
        labels,
        dtype=int,
    )

    probabilities = np.asarray(
        probabilities,
        dtype=float,
    )

    if (
        len(
            np.unique(
                labels
            )
        )
        < 2
    ):
        return np.nan

    return roc_auc_score(
        labels,
        probabilities,
    )


def calculate_metrics(
    labels,
    probabilities,
    threshold,
):
    """
    calculate the binary metrics used throughout the external evaluation.

    the hard metrics use the supplied threshold while auc is calculated directly
    from the continuous score. thresholds passed here are selected from the
    AV-Deepfake1M++ validation scores, not from FakeAVCeleb.
    """
    labels = np.asarray(
        labels,
        dtype=int,
    )

    probabilities = np.asarray(
        probabilities,
        dtype=float,
    )

    predictions = (
        probabilities
        >= threshold
    ).astype(int)

    return {
        "threshold": float(
            threshold
        ),
        "accuracy": accuracy_score(
            labels,
            predictions,
        ),
        "balanced_accuracy": balanced_accuracy_score(
            labels,
            predictions,
        ),
        "f1": f1_score(
            labels,
            predictions,
            zero_division=0,
        ),
        "precision": precision_score(
            labels,
            predictions,
            zero_division=0,
        ),
        "recall": recall_score(
            labels,
            predictions,
            zero_division=0,
        ),
        "auc": safe_auc(
            labels,
            probabilities,
        ),
    }


def tune_threshold(
    dataframe,
    score_column,
    target_column,
):
    """
    select a decision threshold using source validation balanced accuracy.

    thresholds from 0.01 to 0.99 are evaluated. balanced accuracy is the primary
    ordering criterion and f1 breaks ties.

    returns:
        selected threshold
        full threshold sweep dataframe
    """
    rows = []

    for threshold in np.linspace(
        0.01,
        0.99,
        197,
    ):
        rows.append({
            "threshold": threshold,
            **calculate_metrics(
                dataframe[
                    target_column
                ],
                dataframe[
                    score_column
                ],
                threshold,
            ),
        })

    table = pd.DataFrame(
        rows
    )

    best = (
        table.sort_values(
            [
                "balanced_accuracy",
                "f1",
            ],
            ascending=False,
        )
        .iloc[0]
    )

    return (
        float(
            best["threshold"]
        ),
        table,
    )


def cluster_bootstrap_metrics(
    dataframe,
    target_column,
    score_column,
    threshold,
    group_column="source",
    repetitions=1000,
    seed=42,
):
    """
    estimate source-cluster bootstrap confidence intervals.

    source identities, rather than individual videos, are resampled with
    replacement. this keeps the multiple controlled-view videos belonging to a
    source together within each bootstrap draw.

    only balanced accuracy and auc intervals are returned because those are the
    main controlled-view uncertainty summaries used by the experiment.
    """
    if repetitions <= 0:
        return None

    groups = sorted(
        dataframe[
            group_column
        ]
        .astype(str)
        .unique()
    )

    rng = np.random.default_rng(
        seed
    )

    bootstrap_rows = []

    for _ in range(
        repetitions
    ):
        sampled_groups = rng.choice(
            groups,
            size=len(groups),
            replace=True,
        )

        sampled_parts = []

        for (
            bootstrap_group_index,
            group_value,
        ) in enumerate(
            sampled_groups
        ):
            part = (
                dataframe[
                    dataframe[
                        group_column
                    ]
                    .astype(str)
                    == group_value
                ]
                .copy()
            )

            part[
                "_bootstrap_group"
            ] = (
                bootstrap_group_index
            )

            sampled_parts.append(
                part
            )

        sampled = pd.concat(
            sampled_parts,
            ignore_index=True,
        )

        metric_values = (
            calculate_metrics(
                sampled[
                    target_column
                ],
                sampled[
                    score_column
                ],
                threshold,
            )
        )

        bootstrap_rows.append({
            "balanced_accuracy": (
                metric_values[
                    "balanced_accuracy"
                ]
            ),
            "auc": (
                metric_values[
                    "auc"
                ]
            ),
        })

    table = pd.DataFrame(
        bootstrap_rows
    )

    return {
        "balanced_accuracy_ci_low": (
            table[
                "balanced_accuracy"
            ]
            .quantile(0.025)
        ),
        "balanced_accuracy_ci_high": (
            table[
                "balanced_accuracy"
            ]
            .quantile(0.975)
        ),
        "auc_ci_low": (
            table[
                "auc"
            ]
            .quantile(0.025)
        ),
        "auc_ci_high": (
            table[
                "auc"
            ]
            .quantile(0.975)
        ),
    }
