# temporal model / evaluation helpers used by exp005

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
from tqdm.auto import tqdm


def load_shard(path):
    """load one saved audio feature shard onto cpu."""
    return torch.load(
        path,
        map_location="cpu",
    )


def get_shard_paths(
    audio_feature_dir,
    split_name,
):
    """
    read one split's shard manifest and return the stored shard paths.

    every listed file is checked before training starts so a partially missing
    feature cache fails early.
    """
    manifest_path = (
        Path(audio_feature_dir)
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
        for p in manifest["path"].tolist()
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


def compute_window_label_counts(shard_paths):
    """
    summarise the temporal label imbalance and audio decode coverage.

    the positive/negative window counts are used later for BCE pos_weight.
    this also checks how many audio-fake videos actually contain at least one
    positive sampled window, which is useful because sparse/short segments can
    be missed by the 64-centre sampling grid.
    """
    positive = 0.0
    total = 0.0

    decode_ok = 0
    videos = 0

    audio_fake_videos = 0
    audio_fake_with_positive_windows = 0

    for path in tqdm(
        shard_paths,
        desc="count window labels",
        leave=False,
    ):
        payload = load_shard(path)

        labels = (
            payload["window_labels"]
            .float()
        )

        video_labels = (
            payload["video_labels"]
            .long()
        )

        positive += float(
            labels.sum().item()
        )

        total += float(
            labels.numel()
        )

        decode_ok += int(
            payload["decode_ok"]
            .sum()
            .item()
        )

        videos += int(
            payload["decode_ok"]
            .numel()
        )

        audio_fake_videos += int(
            video_labels.sum().item()
        )

        per_video_positive = (
            labels.sum(dim=1) > 0
        ).long()

        audio_fake_with_positive_windows += int(
            (
                (video_labels == 1)
                & (per_video_positive == 1)
            )
            .sum()
            .item()
        )

        del (
            payload,
            labels,
            video_labels,
            per_video_positive,
        )

    negative = (
        total
        - positive
    )

    return {
        "positive_windows": positive,
        "negative_windows": negative,
        "total_windows": total,
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
        "decode_ok_videos": decode_ok,
        "total_videos": videos,
        "decode_ok_rate": (
            decode_ok / videos
            if videos > 0
            else 0.0
        ),
        "audio_fake_videos": audio_fake_videos,
        "audio_fake_with_positive_sampled_windows": (
            audio_fake_with_positive_windows
        ),
        "audio_fake_window_coverage_rate": (
            audio_fake_with_positive_windows
            / audio_fake_videos
            if audio_fake_videos > 0
            else 0.0
        ),
    }


class AudioTemporalLinearHead(nn.Module):
    """
    lightweight classifier used on each 402-d audio window feature.

    input:
        [B, T, feature_dim]

    layers:
        LayerNorm(feature_dim)
        Dropout
        Linear(feature_dim -> 1)

    the head is applied independently to each of the 64 windows. there is no
    explicit sequence model between neighbouring windows.
    """

    def __init__(
        self,
        input_dim,
        dropout=0.0,
    ):
        super().__init__()

        self.net = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Dropout(dropout),
            nn.Linear(input_dim, 1),
        )

    def forward(self, x):
        # x: [batch, windows, feature dim]
        return self.net(x).squeeze(-1)


def iter_audio_temporal_minibatches(
    shard_paths,
    batch_size,
    shuffle_files=False,
    shuffle_within_shard=False,
):
    """
    yield smaller minibatches from the cached audio feature shards.

    shard files can be shuffled between epochs and video rows can be shuffled
    inside each shard. the 64 temporal windows inside a video keep their order.

    yields:
        features
        window labels
        row indices
        video-level audio labels
        decode-ok flags
    """
    paths = list(
        shard_paths
    )

    if shuffle_files:
        random.shuffle(paths)

    for path in paths:
        payload = load_shard(path)

        X = (
            payload["features"]
            .float()
        )

        y = (
            payload["window_labels"]
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

        decode_ok = (
            payload["decode_ok"]
            .long()
        )

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
                decode_ok[idx],
            )

        del (
            payload,
            X,
            y,
            indices,
            video_labels,
            decode_ok,
        )


def safe_auc(y_true, y_prob):
    """
    calculate roc auc when the subset contains both classes.

    condition-level subsets can contain only one ground-truth class, in which
    case roc auc is undefined and np.nan is returned.
    """
    if len(np.unique(y_true)) < 2:
        return np.nan

    return roc_auc_score(
        y_true,
        y_prob,
    )


@torch.no_grad()
def evaluate_audio_temporal_model(
    model,
    shard_paths,
    criterion,
    device,
    batch_size,
    threshold,
):
    """
    evaluate the audio temporal head over every sampled window in the shards.

    sigmoid converts each window logit to an audio-fake probability. the supplied
    threshold creates the hard window predictions used by the classification
    metrics.

    row indices, video labels and decode flags are repeated across the 64 windows
    so the notebook can later rebuild clip-level summaries and condition-level
    analyses from the flattened window predictions.

    returns:
        metric dictionary
        flattened true window labels
        flattened fake probabilities
        repeated row indices
        repeated video labels
        repeated decode flags
    """
    model.eval()

    total_loss = 0.0
    total_windows = 0

    all_labels = []
    all_probs = []
    all_indices = []
    all_video_labels = []
    all_decode_ok = []

    for (
        X,
        y,
        indices,
        video_labels,
        decode_ok,
    ) in iter_audio_temporal_minibatches(
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

        total_windows += int(
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

        repeated_decode_ok = (
            decode_ok.view(-1, 1)
            .repeat(1, T)
            .reshape(-1)
        )

        all_indices.append(
            repeated_indices.cpu()
        )

        all_video_labels.append(
            repeated_video_labels.cpu()
        )

        all_decode_ok.append(
            repeated_decode_ok.cpu()
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
            / max(1, total_windows)
        ),
        "window_accuracy": accuracy_score(
            y_true,
            y_pred,
        ),
        "window_balanced_accuracy": balanced_accuracy_score(
            y_true,
            y_pred,
        ),
        "window_f1": f1_score(
            y_true,
            y_pred,
            zero_division=0,
        ),
        "window_precision": precision_score(
            y_true,
            y_pred,
            zero_division=0,
        ),
        "window_recall": recall_score(
            y_true,
            y_pred,
            zero_division=0,
        ),
        "window_auc": safe_auc(
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
        torch.cat(all_decode_ok).numpy(),
    )


def plot_audio_temporal_examples(
    window_df,
    clip_df,
    condition,
    plots_dir,
    n_examples=2,
):
    """
    plot a few high-scoring temporal audio predictions for one condition.

    clips are sorted by maximum predicted audio-fake probability. for each chosen
    clip the plot overlays the predicted probability at each sampled window and
    the true sampled-window audio label.

    these plots are mainly for inspecting localisation/shortcut behaviour, not
    for fitting the model or selecting the threshold.
    """
    plots_dir = Path(plots_dir)

    condition_clips = (
        clip_df[
            clip_df["condition"] == condition
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
        .head(n_examples)["row_index"]
        .tolist()
    )

    for row_index in selected:
        g = (
            window_df[
                window_df["row_index"] == row_index
            ]
            .sort_values("window_idx")
        )

        plt.figure(
            figsize=(12, 4)
        )

        plt.plot(
            g["window_idx"],
            g["window_prob"],
            label="predicted audio fake probability",
        )

        plt.plot(
            g["window_idx"],
            g["window_true"],
            label="true audio fake label",
            linestyle="--",
        )

        plt.xlabel(
            "Sampled audio window index"
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
