# helper code used by exp007
# kept in one file because this experiment is mostly score pooling / fusion rather than another feature pipeline

import gc
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from tqdm.auto import tqdm


def find_run_dir(candidates, experiments_dir):
    """
    find the first existing source experiment folder.

    exp007 depends on the completed visual and audio temporal runs rather than
    decoding the raw dataset again. the candidate names cover the folder names
    used during development.

    returns:
        matched experiment name
        full experiment path
    """
    experiments_dir = Path(experiments_dir)

    for name in candidates:
        path = experiments_dir / name

        if path.exists():
            return name, path

    raise FileNotFoundError(
        "Could not find any of: "
        + ", ".join(candidates)
    )


def find_feature_dir(run_dir, candidates, splits):
    """
    find a complete cached feature directory inside a source experiment.

    a candidate is only accepted if train/val/test all contain a shard manifest.
    this prevents exp007 from silently using a partial feature cache.
    """
    root = Path(run_dir) / "features"
    checked = []

    for name in candidates:
        path = root / name
        checked.append(str(path))

        if (
            path.exists()
            and all(
                (path / split / "shard_manifest.csv").exists()
                for split in splits
            )
        ):
            return path

    raise FileNotFoundError(
        "No complete feature dir found. Checked:"
        + "".join(checked)
    )


def add_modality_labels(df, condition_col):
    """
    add visual, audio and any-fake labels from the four dataset conditions.

    binary_label is the fusion target in exp007:
        0 = both modalities real
        1 = visual fake, audio fake, or both

    visual_fake/audio_fake are kept as separate columns so the condition-level
    analysis can still show which modality was actually manipulated.
    """
    df = df.copy()

    df["visual_fake"] = df[condition_col].isin([
        "fake_video_real_audio",
        "fake_video_fake_audio",
    ]).astype(int)

    df["audio_fake"] = df[condition_col].isin([
        "real_video_fake_audio",
        "fake_video_fake_audio",
    ]).astype(int)

    df["binary_label"] = (
        (df["visual_fake"] == 1)
        | (df["audio_fake"] == 1)
    ).astype(int)

    return df


def load_source_manifests(
    run_dir,
    splits,
    condition_col,
):
    """
    load train/val/test manifests from one source experiment and add modality labels.

    the original row order is preserved because cached shard indices refer back
    to these split rows.
    """
    out = {}

    for split in splits:
        path = (
            Path(run_dir)
            / "manifests"
            / f"{split}.csv"
        )

        if not path.exists():
            raise FileNotFoundError(path)

        out[split] = add_modality_labels(
            pd.read_csv(path).reset_index(drop=True),
            condition_col=condition_col,
        )

    return out


class TemporalLinearHead(nn.Module):
    """
    same lightweight temporal head architecture used by exp004 and exp005.

    exp007 does not train this class again. it only rebuilds the architecture so
    the saved visual/audio temporal checkpoints can be loaded and used to produce
    probabilities from the cached feature shards.

    input:
        [B, T, feature_dim]

    output:
        [B, T] logits
    """

    def __init__(self, input_dim, dropout=0.0):
        super().__init__()

        self.net = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Dropout(dropout),
            nn.Linear(input_dim, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


def load_checkpoint_model(
    checkpoint_path,
    fallback_input_dim,
):
    """
    rebuild a temporal head and load a saved source-model checkpoint.

    input_dim is read from the checkpoint when available, otherwise the known
    modality-specific fallback is used (1024 visual, 402 audio).

    returns:
        loaded model in eval mode
        raw checkpoint dictionary
        device string used for inference
    """
    device = (
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
    )

    input_dim = int(
        checkpoint.get(
            "input_dim",
            fallback_input_dim,
        )
    )

    model = TemporalLinearHead(
        input_dim=input_dim,
        dropout=0.0,
    ).to(device)

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.eval()

    return (
        model,
        checkpoint,
        device,
    )


def get_shard_paths(
    feature_dir,
    split,
):
    """
    read one cached feature shard manifest and return all shard paths.

    every listed file is checked before prediction starts so a missing shard
    fails early rather than halfway through score generation.
    """
    manifest_path = (
        Path(feature_dir)
        / split
        / "shard_manifest.csv"
    )

    manifest = pd.read_csv(
        manifest_path
    )

    paths = [
        Path(path)
        for path in manifest["path"].tolist()
    ]

    missing = [
        path
        for path in paths
        if not path.exists()
    ]

    if missing:
        raise FileNotFoundError(
            f"Missing shard files, first few: {missing[:5]}"
        )

    return paths, manifest


def temporal_pooling_scores(
    probs,
    topk_fractions,
):
    """
    reduce one clip's temporal probabilities into several clip-level scores.

    exp007 compares:
        max
        mean
        top-5% mean
        top-10% mean
        top-20% mean

    for 64 temporal positions, top-10% uses ceil(64 * 0.10) = 7 positions.
    pooling is done separately for visual and audio before the modalities are
    fused.
    """
    probs = np.asarray(
        probs,
        dtype=np.float32,
    )

    scores = {
        "max": float(
            np.max(probs)
        ),
        "mean": float(
            np.mean(probs)
        ),
    }

    sorted_probs = np.sort(
        probs
    )[::-1]

    T = len(
        sorted_probs
    )

    for frac in topk_fractions:
        k = max(
            1,
            int(
                np.ceil(
                    T * frac
                )
            ),
        )

        scores[
            f"top{int(frac * 100)}_mean"
        ] = float(
            np.mean(
                sorted_probs[:k]
            )
        )

    return scores


@torch.no_grad()
def predict_modality_split(
    model,
    feature_dir,
    source_manifest_df,
    split,
    modality_prefix,
    device,
    path_col,
    relative_path_col,
    condition_col,
    default_threshold,
    topk_fractions,
):
    """
    run one saved temporal model over one split and create clip-level scores.

    cached feature shards are loaded, passed through the source temporal head and
    converted to sigmoid probabilities. every video's temporal probabilities are
    then pooled with max/mean/top-k rules.

    the dataframe row index stored in each shard is used to recover the matching
    path, condition and modality labels from the original source manifest.

    the returned dataframe contains one row per clip plus the pooled modality
    scores used later for fusion.
    """
    shard_paths, shard_manifest = get_shard_paths(
        feature_dir,
        split,
    )

    rows = []

    for shard_path in tqdm(
        shard_paths,
        desc=f"predict {modality_prefix} {split}",
        leave=True,
        mininterval=5,
    ):
        payload = torch.load(
            shard_path,
            map_location="cpu",
        )

        X = (
            payload["features"]
            .float()
            .to(device)
        )

        indices = (
            payload["indices"]
            .long()
            .cpu()
            .numpy()
        )

        probs = (
            torch.sigmoid(
                model(X)
            )
            .detach()
            .cpu()
            .numpy()
        )

        for i, split_row_index in enumerate(indices):
            meta = source_manifest_df.iloc[
                int(split_row_index)
            ]

            pooled = temporal_pooling_scores(
                probs[i],
                topk_fractions=topk_fractions,
            )

            row = {
                "split": split,
                "split_row_index": int(
                    split_row_index
                ),
                "path": meta.get(
                    path_col,
                    None,
                ),
                "relative_path": meta.get(
                    relative_path_col,
                    None,
                ),
                "condition": meta.get(
                    condition_col,
                    None,
                ),
                "binary_label": int(
                    meta.get("binary_label")
                ),
                "visual_fake": int(
                    meta.get("visual_fake")
                ),
                "audio_fake": int(
                    meta.get("audio_fake")
                ),
                f"{modality_prefix}_positive_rate_default": float(
                    (
                        probs[i]
                        >= default_threshold
                    ).mean()
                ),
            }

            for key, value in pooled.items():
                row[
                    f"{modality_prefix}_{key}"
                ] = value

            rows.append(row)

        del (
            payload,
            X,
            probs,
        )

        gc.collect()

        if device == "cuda":
            torch.cuda.empty_cache()

    return pd.DataFrame(
        rows
    )


def stable_merge_key(df):
    """
    add the path key used to match visual and audio predictions for the same clip.

    relative_path is preferred because it is less tied to the jupyter filesystem.
    if it is missing for a row, the absolute path is used as the fallback.
    """
    df = df.copy()

    if (
        "relative_path" in df.columns
        and df["relative_path"].notna().any()
    ):
        df["merge_key"] = (
            df["relative_path"]
            .fillna(df["path"])
            .astype(str)
        )

    else:
        df["merge_key"] = (
            df["path"]
            .astype(str)
        )

    return df


def merge_split_predictions(
    split,
    visual_pred,
    audio_pred,
):
    """
    inner-join visual and audio clip scores on the stable path key.

    only clips available in both modality pipelines are kept. validate='one_to_one'
    also checks that the merge key is unique on both sides.

    the result is the common subset used for all exp007 fusion comparisons.
    """
    visual_pred = stable_merge_key(
        visual_pred
    )

    audio_pred = stable_merge_key(
        audio_pred
    )

    visual_cols = [
        "merge_key",
        "split",
        "path",
        "relative_path",
        "condition",
        "binary_label",
        "visual_fake",
        "audio_fake",
    ]

    visual_cols += [
        c
        for c in visual_pred.columns
        if (
            c.startswith("visual_")
            and c != "visual_fake"
        )
    ]

    audio_cols = [
        "merge_key"
    ] + [
        c
        for c in audio_pred.columns
        if (
            c.startswith("audio_")
            and c != "audio_fake"
        )
    ]

    merged = (
        visual_pred[
            visual_cols
        ]
        .merge(
            audio_pred[
                audio_cols
            ],
            on="merge_key",
            how="inner",
            validate="one_to_one",
        )
    )

    merged["split"] = split

    return merged


def add_fusion_scores(df):
    """
    add the simple rule-based late-fusion scores.

    for matching visual/audio pooling types:
        probabilistic OR = 1 - (1-v)(1-a)
        mean             = (v+a)/2
        max              = max(v,a)

    no fitting happens here. these are deterministic combinations of the two
    modality probabilities.
    """
    df = df.copy()

    pairs = [
        (
            "max",
            "visual_max",
            "audio_max",
        ),
        (
            "top5",
            "visual_top5_mean",
            "audio_top5_mean",
        ),
        (
            "top10",
            "visual_top10_mean",
            "audio_top10_mean",
        ),
        (
            "top20",
            "visual_top20_mean",
            "audio_top20_mean",
        ),
        (
            "mean",
            "visual_mean",
            "audio_mean",
        ),
    ]

    for label, visual_col, audio_col in pairs:
        if (
            visual_col in df.columns
            and audio_col in df.columns
        ):
            df[f"or_{label}"] = (
                1
                - (
                    1 - df[visual_col]
                )
                * (
                    1 - df[audio_col]
                )
            )

            df[f"mean_{label}"] = (
                df[visual_col]
                + df[audio_col]
            ) / 2

            df[f"max_{label}"] = (
                df[
                    [
                        visual_col,
                        audio_col,
                    ]
                ]
                .max(axis=1)
            )

    return df


def safe_auc(
    y_true,
    y_score,
):
    """
    calculate roc auc when both ground-truth classes are present.

    condition-only subsets can contain one class, where auc is undefined.
    """
    if (
        len(
            np.unique(y_true)
        )
        < 2
    ):
        return np.nan

    return roc_auc_score(
        y_true,
        y_score,
    )


def compute_binary_metrics(
    y_true,
    y_score,
    threshold=0.5,
):
    """
    calculate the binary metrics used for each unimodal/fusion score.

    threshold only affects the hard-prediction metrics. auc is calculated from
    the continuous score and therefore does not depend on the threshold.
    """
    y_true = (
        np.asarray(y_true)
        .astype(int)
    )

    y_score = (
        np.asarray(y_score)
        .astype(float)
    )

    y_pred = (
        y_score
        >= threshold
    ).astype(int)

    return {
        "threshold": float(
            threshold
        ),
        "accuracy": accuracy_score(
            y_true,
            y_pred,
        ),
        "balanced_accuracy": balanced_accuracy_score(
            y_true,
            y_pred,
        ),
        "f1": f1_score(
            y_true,
            y_pred,
            zero_division=0,
        ),
        "precision": precision_score(
            y_true,
            y_pred,
            zero_division=0,
        ),
        "recall": recall_score(
            y_true,
            y_pred,
            zero_division=0,
        ),
        "auc": safe_auc(
            y_true,
            y_score,
        ),
    }


def tune_threshold_on_val(
    val_df,
    score_col,
    target_col="binary_label",
):
    """
    select a classification threshold using validation balanced accuracy.

    thresholds from 0.01 to 0.99 are tested. balanced accuracy is the primary
    ranking metric and f1 breaks ties. the returned threshold is later applied
    unchanged to the test scores.

    this tunes the decision threshold only; it does not retrain a source model or
    alter the fusion score itself.
    """
    y_true = (
        val_df[target_col]
        .values
        .astype(int)
    )

    scores = (
        val_df[score_col]
        .values
        .astype(float)
    )

    rows = []

    for threshold in np.linspace(
        0.01,
        0.99,
        197,
    ):
        rows.append({
            "score_col": score_col,
            **compute_binary_metrics(
                y_true,
                scores,
                threshold=threshold,
            ),
        })

    sweep = pd.DataFrame(
        rows
    )

    best = (
        sweep
        .sort_values(
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
        best,
        sweep,
    )


def condition_metrics_for_score(
    df,
    score_col,
    threshold,
    method_name,
):
    """
    summarise one fusion/unimodal score separately for each manipulation condition.

    each condition itself has one binary ground-truth label, so the useful number
    here is the condition success rate / predicted fake rate rather than treating
    each condition as a normal two-class evaluation set.

    the table also keeps average visual/audio top10 and max scores so modality
    behaviour can be inspected alongside the final fused prediction.
    """
    rows = []

    for condition, group in df.groupby(
        "condition"
    ):
        y = (
            group["binary_label"]
            .values
            .astype(int)
        )

        scores = (
            group[score_col]
            .values
            .astype(float)
        )

        pred = (
            scores
            >= threshold
        ).astype(int)

        if condition == "real":
            success_rate = float(
                (
                    pred == 0
                ).mean()
            )

            interpretation = (
                "real recall / specificity"
            )

        else:
            success_rate = float(
                (
                    pred == 1
                ).mean()
            )

            interpretation = (
                "fake recall"
            )

        rows.append({
            "method": method_name,
            "score_col": score_col,
            "threshold": threshold,
            "condition": condition,
            "n": len(group),
            "true_binary_label": int(
                group["binary_label"]
                .iloc[0]
            ),
            "visual_fake": int(
                group["visual_fake"]
                .iloc[0]
            ),
            "audio_fake": int(
                group["audio_fake"]
                .iloc[0]
            ),
            "success_rate": success_rate,
            "interpretation": interpretation,
            "predicted_fake_rate": float(
                pred.mean()
            ),
            "mean_score": float(
                scores.mean()
            ),
            "mean_visual_top10": (
                float(
                    group["visual_top10_mean"]
                    .mean()
                )
                if "visual_top10_mean" in group
                else np.nan
            ),
            "mean_audio_top10": (
                float(
                    group["audio_top10_mean"]
                    .mean()
                )
                if "audio_top10_mean" in group
                else np.nan
            ),
            "mean_visual_max": (
                float(
                    group["visual_max"]
                    .mean()
                )
                if "visual_max" in group
                else np.nan
            ),
            "mean_audio_max": (
                float(
                    group["audio_max"]
                    .mean()
                )
                if "audio_max" in group
                else np.nan
            ),
        })

    return pd.DataFrame(
        rows
    )
