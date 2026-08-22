# temporal head / evaluation helpers used by exp004

import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
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

from .feature_utils import load_shard


class TemporalLinearHead(nn.Module):
    """
    simple per-frame classifier trained on the frozen convnext features.

    input is [B, T, 1024]. layernorm and one linear projection are applied to
    each sampled frame independently, producing one logit per temporal position.

    there is no attention, recurrence or temporal convolution here. the model
    does not explicitly use neighbouring-frame context, which is why i treat it
    as a lightweight temporal localisation baseline rather than a full temporal
    model.

    output:
        [B, T] frame logits
    """

    def __init__(
        self,
        input_dim=1024,
        dropout=0.0,
    ):
        super().__init__()

        self.net = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Dropout(dropout),
            nn.Linear(input_dim, 1),
        )

    def forward(self, x):
        # x: [batch, sampled positions, feature dim]
        return self.net(x).squeeze(-1)


def iter_temporal_minibatches(
    shard_paths,
    batch_size,
    shuffle_files=False,
    shuffle_within_shard=False,
):
    """
    yield smaller training/evaluation batches from the saved feature shards.

    each shard already holds several complete videos. files can be shuffled
    between epochs and rows can also be shuffled inside a shard for training.
    the actual frame sequence inside each video is not shuffled.

    yields:
        X            [B, T, D]
        y            [B, T]
        row indices  [B]
        video labels [B]
    """
    paths = list(shard_paths)

    if shuffle_files:
        random.shuffle(paths)

    for path in paths:
        payload = load_shard(path)

        X = payload["features"].float()
        y = payload["frame_labels"].float()
        indices = payload["indices"].long()
        video_labels = payload["video_labels"].long()

        n = X.shape[0]
        order = torch.arange(n)

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


def count_videos_in_shards(shard_paths):
    """count how many complete videos are stored across a list of shards."""
    total = 0

    for path in shard_paths:
        payload = load_shard(path)
        total += int(
            payload["features"].shape[0]
        )
        del payload

    return total


def safe_auc(y_true, y_prob):
    """
    calculate roc auc when both classes are present.

    some condition-specific subsets can contain only one class. roc auc is not
    defined in that case, so np.nan is returned instead of raising an error.
    """
    if len(np.unique(y_true)) < 2:
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

    the saved [B,T,D] feature tensors are loaded in smaller minibatches. sigmoid
    converts each frame logit to a fake probability and the supplied threshold
    converts that probability to a hard frame prediction.

    besides the metric summary, the function returns the flattened frame labels
    and probabilities plus repeated video row indices/labels. those repeated
    indices are what allow the notebook to rebuild clip-level predictions and
    sampled-grid localisation results afterwards.

    metrics:
        loss
        frame accuracy
        frame balanced accuracy
        frame f1
        frame precision
        frame recall
        frame roc auc
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
        X = X.to(device)
        y = y.to(device)

        logits = model(X)
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
            indices.view(-1, 1)
            .repeat(1, T)
            .reshape(-1)
        )

        repeated_video_labels = (
            video_labels.view(-1, 1)
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
        torch.cat(all_labels)
        .numpy()
        .astype(int)
    )

    y_prob = (
        torch.cat(all_probs)
        .numpy()
    )

    y_pred = (
        y_prob >= threshold
    ).astype(int)

    metrics = {
        "loss": (
            total_loss
            / max(1, total_frames)
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
        torch.cat(all_indices).numpy(),
        torch.cat(all_video_labels).numpy(),
    )


def plot_temporal_examples(
    frame_df,
    clip_df,
    condition,
    plots_dir,
    n_examples=2,
):
    """
    plot a few high-scoring temporal predictions for one dataset condition.

    clips are sorted by their maximum predicted frame probability and the top
    examples are shown. each plot overlays:
        predicted fake probability at each sampled position
        true sampled-grid visual fake label

    the plots are mainly for inspecting where the frame classifier activates,
    not for selecting thresholds or changing the trained model.
    """
    plots_dir = Path(plots_dir)

    condition_clips = clip_df[
        clip_df["condition"] == condition
    ].copy()

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
        .head(n_examples)["row_index"]
        .tolist()
    )

    for row_index in selected:
        g = (
            frame_df[
                frame_df["row_index"] == row_index
            ]
            .sort_values("timestep")
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
        plt.ylabel("Value")
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
