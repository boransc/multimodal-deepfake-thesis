# frozen visual feature + shard helpers for exp004

import gc
from datetime import datetime
from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.models import convnext_base, ConvNeXt_Base_Weights
from tqdm import tqdm
from IPython.display import display

from .data_utils import TemporalVideoFeatureDataset


class ConvNeXtFrameFeatureExtractor(nn.Module):
    """
    frozen convnext-base encoder used to get one 1024-d feature per frame.

    the final imagenet classification projection is replaced with identity.
    unlike the earlier clip-level feature pipeline, the T frame representations
    are kept separately rather than being averaged inside this module.

    input:
        [B, T, 3, H, W]

    output:
        [B, T, 1024]

    the notebook puts this model in eval mode for feature extraction. the
    temporal linear head is trained later from the saved feature shards.
    """

    def __init__(self):
        super().__init__()

        weights = ConvNeXt_Base_Weights.IMAGENET1K_V1
        self.backbone = convnext_base(weights=weights)

        # use the representation before convnext's final imagenet projection
        self.backbone.classifier[2] = nn.Identity()

    def forward(self, x):
        # combine batch + time so convnext sees a normal image batch
        B, T, C, H, W = x.shape

        x = x.reshape(B * T, C, H, W)
        frame_features = self.backbone(x)       # [B*T, 1024]
        frame_features = frame_features.reshape(B, T, -1)

        return frame_features                   # [B, T, 1024]


def make_feature_loader(
    csv_path,
    num_frames,
    image_size,
    feature_batch_size,
    num_workers,
    pin_memory,
    prefetch_factor,
    path_col,
    missing_visual_segment_policy,
    segment_col,
    fallback_segment_cols=None,
    shuffle=False,
):
    """
    build the dataloader used while extracting temporal convnext features.

    the TemporalVideoFeatureDataset handles both sampled video frames and the
    sampled-grid frame labels. num_workers/prefetch settings are kept explicit
    because video decoding was one of the heavier parts of this experiment.

    returns both the dataset and dataloader so the notebook can print their
    sizes before feature extraction.
    """
    dataset = TemporalVideoFeatureDataset(
        csv_path=csv_path,
        path_col=path_col,
        num_frames=num_frames,
        image_size=image_size,
        missing_visual_segment_policy=missing_visual_segment_policy,
        segment_col=segment_col,
        fallback_segment_cols=fallback_segment_cols,
    )

    loader_kwargs = {
        "dataset": dataset,
        "batch_size": feature_batch_size,
        "shuffle": shuffle,
        "num_workers": num_workers,
        "pin_memory": pin_memory,
        "persistent_workers": False,
    }

    if num_workers > 0:
        loader_kwargs["prefetch_factor"] = prefetch_factor
        loader_kwargs["timeout"] = 180

    return dataset, DataLoader(**loader_kwargs)


@torch.inference_mode()
def extract_temporal_feature_shards(
    model,
    loader,
    split_name,
    output_root,
    device,
    model_name,
    num_frames,
    image_size,
):
    """
    extract frozen frame features and save them to smaller shard files.

    each input batch contains the sampled frames plus their frame-level labels
    and timestamps. convnext produces [B, T, 1024] features, which are moved to
    cpu and stored as float16 to reduce disk usage.

    each saved shard contains:
        features      [B, T, 1024]
        frame_labels  [B, T]
        frame_times   [B, T]
        video_labels  [B]
        indices       [B]

    a shard_manifest.csv records the file paths, shapes and positive-frame
    counts. if a complete manifest already exists and every listed shard exists,
    extraction is skipped and the old manifest is reused.
    """
    model.eval()

    split_dir = Path(output_root) / split_name
    split_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    manifest_path = split_dir / "shard_manifest.csv"

    if manifest_path.exists():
        existing = pd.read_csv(manifest_path)

        if (
            len(existing) > 0
            and existing["path"]
            .apply(lambda p: Path(p).exists())
            .all()
        ):
            print(
                f"{split_name}: complete shard manifest already exists, "
                "skipping extraction."
            )
            display(existing.tail())
            return existing

    rows = []
    start_time = __import__("time").time()

    progress = tqdm(
        loader,
        desc=f"extract {split_name}",
        leave=True,
        mininterval=5,
    )

    for batch_idx, (
        frames,
        frame_labels,
        frame_times,
        video_labels,
        indices,
    ) in enumerate(progress):

        frames = frames.to(
            device,
            non_blocking=True,
        )

        with torch.amp.autocast(
            "cuda",
            enabled=(device == "cuda"),
        ):
            features = model(frames)

        # the cache is float16 because these shards can take several gb
        features_cpu = (
            features.detach()
            .cpu()
            .to(torch.float16)
        )

        frame_labels_cpu = (
            frame_labels.cpu()
            .to(torch.float16)
        )

        frame_times_cpu = (
            frame_times.cpu()
            .to(torch.float16)
        )

        video_labels_cpu = video_labels.cpu()
        indices_cpu = indices.cpu()

        shard_path = (
            split_dir
            / f"shard_{batch_idx:05d}.pt"
        )

        payload = {
            "features": features_cpu,
            "frame_labels": frame_labels_cpu,
            "frame_times": frame_times_cpu,
            "video_labels": video_labels_cpu,
            "indices": indices_cpu,
            "split": split_name,
            "model_name": model_name,
            "num_frames": num_frames,
            "image_size": image_size,
            "feature_dim": int(
                features_cpu.shape[-1]
            ),
            "created_at": datetime.now().isoformat(
                timespec="seconds"
            ),
        }

        torch.save(
            payload,
            shard_path,
        )

        rows.append({
            "split": split_name,
            "shard_idx": batch_idx,
            "path": str(shard_path),
            "n_videos": int(features_cpu.shape[0]),
            "num_frames": int(features_cpu.shape[1]),
            "feature_dim": int(features_cpu.shape[2]),
            "positive_frames": float(
                frame_labels_cpu.sum().item()
            ),
            "total_frames": int(
                frame_labels_cpu.numel()
            ),
            "size_mb": (
                shard_path.stat().st_size
                / 1024**2
            ),
        })

        if batch_idx % 10 == 0:
            done = sum(
                row["n_videos"]
                for row in rows
            )

            progress.set_postfix({
                "videos": done,
                "pos": int(
                    sum(
                        row["positive_frames"]
                        for row in rows
                    )
                ),
                "gpu": (
                    f"{torch.cuda.memory_reserved() / 1024**3:.1f}G"
                    if device == "cuda"
                    else "cpu"
                ),
            })

            print(
                f"{split_name} | shard {batch_idx}/{len(loader)} | "
                f"videos={done}",
                flush=True,
            )

        del (
            frames,
            features,
            features_cpu,
            frame_labels_cpu,
            frame_times_cpu,
            video_labels_cpu,
            indices_cpu,
            payload,
        )

        gc.collect()

        if device == "cuda":
            torch.cuda.empty_cache()

    manifest_df = pd.DataFrame(rows)
    elapsed = __import__("time").time() - start_time

    manifest_df["elapsed_total_sec"] = elapsed
    manifest_df.to_csv(
        manifest_path,
        index=False,
    )

    print(
        f"{split_name}: saved "
        f"{manifest_df['n_videos'].sum()} videos "
        f"across {len(manifest_df)} shards"
    )
    print(
        f"{split_name}: elapsed "
        f"{elapsed / 60:.2f} min"
    )
    print("Manifest:", manifest_path)

    return manifest_df


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
    read a split's shard manifest and return the saved shard paths.

    this also checks that every listed file still exists before training starts,
    so a partially missing feature cache fails early rather than halfway through
    an epoch.
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

    manifest = pd.read_csv(manifest_path)

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
            f"Missing shard files. First few: {missing[:5]}"
        )

    return paths


def compute_frame_label_counts(shard_paths):
    """
    count positive/negative frame labels across a set of feature shards.

    exp004 has a low proportion of manipulated sampled frames. these counts are
    later used to calculate BCE pos_weight so the positive class is not ignored
    during temporal-head training.
    """
    positive = 0.0
    total = 0.0

    for path in tqdm(
        shard_paths,
        desc="count frame labels",
        leave=False,
    ):
        payload = load_shard(path)
        labels = payload["frame_labels"].float()

        positive += float(
            labels.sum().item()
        )
        total += float(
            labels.numel()
        )

        del payload, labels

    negative = total - positive

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
