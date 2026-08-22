# temporal head / evaluation helpers used by exp006

import random
from pathlib import Path

import matplotlib.pyplot as plt
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
from tqdm import tqdm


def load_shard(path):
    """load one saved temporal feature shard onto cpu."""
    return torch.load(
        path,
        map_location="cpu",
    )


def get_shard_paths(
    temporal_feature_dir,
    split_name,
):
    """
    read one split's shard manifest and return the saved shard paths.

    all listed shard files are checked before training so a partially missing
    cache fails early.
    """
    manifest_path = (
        Path(temporal_feature_dir)
        / split_name
        / "shard_manifest.csv"
    )

    if not manifest_path.exists():
        raise FileNotFoundError(
            f"Missing shard manifest: {manifest_path}"
        )

    manifest = pd.read_csv(
        manifest_path
    )

    paths = [
        Path(p)
        for p in manifest[
            "path"
        ].tolist()
    ]

    missing = [
        p for p in paths
        if not p.exists()
    ]

    if missing:
        raise FileNotFoundError(
            f"Missing shard files. First few: {missing[:5]}"
        )

    return paths


def compute_frame_label_counts(
    shard_paths,
):
    """
    count positive/negative sampled-frame labels across the training shards.

    manipulated sampled frames are relatively rare, so these counts are used to
    calculate BCE pos_weight before the temporal head is trained.
    """
    positive = 0.0
    total = 0.0

    for path in tqdm(
        shard_paths,
        desc="count frame labels",
        leave=False,
    ):
        payload = load_shard(
            path
        )

        labels = (
            payload["frame_labels"]
            .float()
        )

        positive += float(
            labels.sum().item()
        )

        total += float(
            labels.numel()
        )

        del (
            payload,
            labels,
        )

    negative = (
        total
        - positive
    )

    return {
        "positive_frames": positive,
        "negative_frames": negative,
        "total_frames": total,
        "positive_rate": (
            positive / total
            if total > 0
            else 0.0
        ),
        "negative_rate": (
            negative / total
            if total > 0
            else 0.0
        ),
    }


class TemporalLinearHead(nn.Module):
    """
    simple per-frame classifier trained on frozen convnext features.

    input:
        [B, T, 1024]

    layers:
        LayerNorm(1024)
        Dropout
        Linear(1024 -> 1)

    the same head is applied independently to each sampled position. there is no
    attention, recurrence or temporal convolution, so this is a lightweight
    localisation baseline rather than an explicit temporal-context model.
    """

    def __init__(
        self,
        input_dim=1024,
        dropout=0.0,
    ):
        super().__init__()

        self.net = nn.Sequential(
            nn.LayerNorm(
                input_dim
            ),
            nn.Dropout(
                dropout
            ),
            nn.Linear(
                input_dim,
                1,
            ),
        )

    def forward(self, x):
        # x: [batch, sampled frames, feature dim]
        return self.net(
            x
        ).squeeze(-1)


def iter_temporal_minibatches(
    shard_paths,
    batch_size,
    shuffle_files=False,
    shuffle_within_shard=False,
):
    """
    yield smaller minibatches from the cached temporal feature shards.

    shard files can be shuffled between training epochs and video rows can be
    shuffled inside each shard. the 64 frame positions inside a video keep their
    original order.

    yields:
        features
        frame labels
        dataframe row indices
        video-level visual labels
    """
    paths = list(
        shard_paths
    )

    if shuffle_files:
        random.shuffle(
            paths
        )

    for path in paths:
        payload = load_shard(
            path
        )

        X = (
            payload["features"]
            .float()
        )

        y = (
            payload["frame_labels"]
            .float()
        )

        indices = (
            payload["indices"]
            .long()
        )

        video_labels = (
            payload["video_labels"]
            .long()
        )

        n = X.shape[0]
        order = torch.arange(
            n
        )

        if shuffle_within_shard:
            order = order[
                torch.randperm(n)
            ]

        for start in range(
            0,
            n,
            batch_size,
        ):
            idx = order[
                start:start + batch_size
            ]

            yield (
                X[idx],
                y[idx],
                indices[idx],
                video_labels[idx],
            )

        del (
            payload,
            X,
            y,
            indices,
            video_labels,
        )


def safe_auc(
    y_true,
    y_prob,
):
    """
    calculate roc auc when both classes are present.

    condition-specific subsets can contain only one ground-truth class, where
    auc is undefined. np.nan is returned instead of raising an exception.
    """
    if (
        len(np.unique(y_true))
        < 2
    ):
        return np.nan

    return roc_auc_score(
        y_true,
        y_prob,
    )


@torch.no_grad()
def evaluate_temporal_model(
    model,
    shard_paths,
    criterion,
    device,
    batch_size,
    threshold,
):
    """
    evaluate the temporal head across all sampled frames in the supplied shards.

    sigmoid converts each frame logit into a fake probability and threshold turns
    that into the hard prediction used by the classification metrics.

    row indices and video labels are repeated across the 64 temporal positions.
    the notebook later uses those repeated indices to rebuild clip-level scores,
    condition summaries and sampled-grid localisation results.

    returns:
        metric dictionary
        flattened frame labels
        flattened frame probabilities
        repeated row indices
        repeated video labels
    """
    model.eval()

    total_loss = 0.0
    total_frames = 0

    all_labels = []
    all_probs = []
    all_indices = []
    all_video_labels = []

    for (
        X,
        y,
        indices,
        video_labels,
    ) in iter_temporal_minibatches(
        shard_paths,
        batch_size=batch_size,
        shuffle_files=False,
        shuffle_within_shard=False,
    ):
        X = X.to(
            device
        )

        y = y.to(
            device
        )

        logits = model(
            X
        )

        loss = criterion(
            logits,
            y,
        )

        probs = torch.sigmoid(
            logits
        )

        total_loss += (
            float(loss.item())
            * int(y.numel())
        )

        total_frames += int(
            y.numel()
        )

        all_labels.append(
            y.detach()
            .cpu()
            .reshape(-1)
        )

        all_probs.append(
            probs.detach()
            .cpu()
            .reshape(-1)
        )

        B, T = y.shape

        repeated_indices = (
            indices
            .view(-1, 1)
            .repeat(1, T)
            .reshape(-1)
        )

        repeated_video_labels = (
            video_labels
            .view(-1, 1)
            .repeat(1, T)
            .reshape(-1)
        )

        all_indices.append(
            repeated_indices.cpu()
        )

        all_video_labels.append(
            repeated_video_labels.cpu()
        )

        del (
            X,
            y,
            logits,
            loss,
            probs,
        )

    y_true = (
        torch.cat(
            all_labels
        )
        .numpy()
        .astype(int)
    )

    y_prob = (
        torch.cat(
            all_probs
        )
        .numpy()
    )

    y_pred = (
        y_prob
        >= threshold
    ).astype(int)

    metrics = {
        "loss": (
            total_loss
            / max(
                1,
                total_frames,
            )
        ),
        "frame_accuracy": accuracy_score(
            y_true,
            y_pred,
        ),
        "frame_balanced_accuracy": balanced_accuracy_score(
            y_true,
            y_pred,
        ),
        "frame_f1": f1_score(
            y_true,
            y_pred,
            zero_division=0,
        ),
        "frame_precision": precision_score(
            y_true,
            y_pred,
            zero_division=0,
        ),
        "frame_recall": recall_score(
            y_true,
            y_pred,
            zero_division=0,
        ),
        "frame_auc": safe_auc(
            y_true,
            y_prob,
        ),
    }

    return (
        metrics,
        y_true,
        y_prob,
        torch.cat(
            all_indices
        ).numpy(),
        torch.cat(
            all_video_labels
        ).numpy(),
    )


def plot_temporal_examples(
    frame_df,
    clip_df,
    condition,
    plots_dir,
    n_examples=2,
):
    """
    plot a few high-scoring temporal predictions for one condition.

    clips are ordered by their maximum predicted frame probability. each selected
    example overlays the 64 predicted probabilities with the true sampled-grid
    visual labels.

    these plots are for inspection only; they do not change the model or select
    the classification threshold.
    """
    plots_dir = Path(
        plots_dir
    )

    condition_clips = (
        clip_df[
            clip_df["condition"]
            == condition
        ]
        .copy()
    )

    if len(condition_clips) == 0:
        print(
            "No clips for condition:",
            condition,
        )
        return

    selected = (
        condition_clips
        .sort_values(
            "clip_prob_max",
            ascending=False,
        )
        .head(n_examples)[
            "row_index"
        ]
        .tolist()
    )

    for row_index in selected:
        g = (
            frame_df[
                frame_df["row_index"]
                == row_index
            ]
            .sort_values(
                "timestep"
            )
        )

        plt.figure(
            figsize=(12, 4)
        )

        plt.plot(
            g["timestep"],
            g["frame_prob"],
            label="predicted fake probability",
        )

        plt.plot(
            g["timestep"],
            g["frame_true"],
            label="true visual fake label",
            linestyle="--",
        )

        plt.xlabel(
            "Sampled frame index"
        )

        plt.ylabel(
            "Value"
        )

        plt.title(
            f"{condition} | row_index={row_index}"
        )

        plt.ylim(
            -0.05,
            1.05,
        )

        plt.legend()

        plt.grid(
            True,
            alpha=0.3,
        )

        plt.tight_layout()

        out_path = (
            plots_dir
            / f"example_{condition}_row_{row_index}.png"
        )

        plt.savefig(
            out_path,
            dpi=200,
        )

        plt.show()
