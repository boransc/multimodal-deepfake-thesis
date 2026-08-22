# video dataset + frozen convnext feature helpers for exp006

import gc
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.models import (
    convnext_base,
    ConvNeXt_Base_Weights,
)
from tqdm import tqdm
from IPython.display import display

from .data_utils import (
    get_visual_segments,
    frame_labels_from_segments,
)


class TemporalVideoFeatureDataset(Dataset):
    """
    sample each exp006 video at 64 positions and build matching visual labels.

    the csv already contains the split rows plus visual_fake and segment metadata.
    frames are sampled uniformly across the complete reported frame range and
    preprocessed with the imagenet normalisation expected by convnext-base.

    label behaviour:
        visual_fake == 0
            -> all 64 sampled visual labels are 0

        visual_fake == 1 with visual segments
            -> sampled positions inside the manipulation intervals are 1

        visual_fake == 1 without segments
            -> behaviour follows missing_visual_segment_policy. the final exp006
               config uses "drop", so those rows should already be gone.

    returns:
        frames       [T, 3, H, W]
        frame_labels [T]
        frame_times  [T]
        video label
        dataframe row index
    """

    def __init__(
        self,
        csv_path,
        path_col="path",
        num_frames=64,
        image_size=224,
        missing_visual_segment_policy="drop",
        segment_col="visual_fake_segments",
    ):
        self.df = (
            pd.read_csv(csv_path)
            .reset_index(drop=True)
        )

        self.path_col = path_col
        self.num_frames = num_frames
        self.missing_visual_segment_policy = (
            missing_visual_segment_policy
        )

        self.segment_col = segment_col

        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(
                (image_size, image_size)
            ),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[
                    0.485,
                    0.456,
                    0.406,
                ],
                std=[
                    0.229,
                    0.224,
                    0.225,
                ],
            ),
        ])

    def __len__(self):
        return len(self.df)

    def _labels_for_row(
        self,
        row,
        target_indices,
        target_times,
        duration_sec,
        total_frames,
    ):
        """
        make the sampled visual labels for one video.

        audio-only fakes are forced to visual-real here even if their metadata
        contains a strange visual segment value. that keeps the modality target
        tied to the condition label.
        """
        visual_fake = int(
            row.get(
                "visual_fake",
                0,
            )
        )

        if visual_fake == 0:
            return np.zeros(
                self.num_frames,
                dtype=np.float32,
            )

        segments = get_visual_segments(
            row,
            segment_col=self.segment_col,
        )

        if len(segments) == 0:
            if (
                self.missing_visual_segment_policy
                == "all_fake"
            ):
                return np.ones(
                    self.num_frames,
                    dtype=np.float32,
                )

            return np.zeros(
                self.num_frames,
                dtype=np.float32,
            )

        return frame_labels_from_segments(
            segments=segments,
            target_indices=target_indices,
            target_times=target_times,
            duration_sec=duration_sec,
            total_frames=total_frames,
        )

    def _sample_frames(
        self,
        video_path,
        row,
    ):
        """
        decode one video and collect the 64 uniformly spaced frame positions.

        fps is used to map frame positions into seconds for time-based segment
        annotations. the same 25 fps fallback is kept for clips with invalid fps.

        if decoding produces fewer frames than requested, the last valid frame is
        repeated so the returned tensor still has a fixed temporal length.
        """
        cap = cv2.VideoCapture(
            str(video_path)
        )

        if not cap.isOpened():
            raise RuntimeError(
                f"Could not open video: {video_path}"
            )

        total_frames = int(
            cap.get(
                cv2.CAP_PROP_FRAME_COUNT
            )
        )

        fps = float(
            cap.get(
                cv2.CAP_PROP_FPS
            )
        )

        if total_frames <= 0:
            cap.release()
            raise RuntimeError(
                f"No frames found: {video_path}"
            )

        if (
            fps <= 0
            or not np.isfinite(fps)
        ):
            fps = 25.0

        duration_sec = (
            total_frames
            / fps
        )

        target_indices = np.linspace(
            0,
            total_frames - 1,
            self.num_frames,
        ).astype(int)

        target_times = (
            target_indices
            / fps
        )

        target_set = set(
            target_indices.tolist()
        )

        frame_labels = self._labels_for_row(
            row=row,
            target_indices=target_indices,
            target_times=target_times,
            duration_sec=duration_sec,
            total_frames=total_frames,
        )

        frames = []
        current_idx = 0

        while True:
            ret, frame = cap.read()

            if not ret:
                break

            if current_idx in target_set:
                frame = cv2.cvtColor(
                    frame,
                    cv2.COLOR_BGR2RGB,
                )

                frame = self.transform(
                    frame
                )

                frames.append(frame)

                if (
                    len(frames)
                    == self.num_frames
                ):
                    break

            current_idx += 1

        cap.release()

        if len(frames) == 0:
            raise RuntimeError(
                f"No frames sampled: {video_path}"
            )

        while (
            len(frames)
            < self.num_frames
        ):
            frames.append(
                frames[-1]
            )

        return (
            torch.stack(frames),
            torch.tensor(
                frame_labels,
                dtype=torch.float32,
            ),
            torch.tensor(
                target_times,
                dtype=torch.float32,
            ),
        )

    def __getitem__(
        self,
        idx,
    ):
        row = self.df.iloc[idx]

        (
            frames,
            frame_labels,
            frame_times,
        ) = self._sample_frames(
            row[self.path_col],
            row,
        )

        visual_video_label = int(
            row.get(
                "visual_fake",
                int(
                    frame_labels
                    .max()
                    .item()
                    > 0
                ),
            )
        )

        return (
            frames,
            frame_labels,
            frame_times,
            torch.tensor(
                visual_video_label,
                dtype=torch.long,
            ),
            torch.tensor(
                idx,
                dtype=torch.long,
            ),
        )


class ConvNeXtFrameFeatureExtractor(nn.Module):
    """
    frozen convnext-base encoder that keeps one 1024-d vector per sampled frame.

    the final imagenet classification projection is replaced with identity.
    batch and temporal dimensions are flattened while convnext processes the
    images, then restored afterwards.

    input:
        [B, T, 3, H, W]

    output:
        [B, T, 1024]

    unlike the clip-level experiments, the T frame vectors are not averaged
    here because exp006 needs per-frame predictions for localisation.
    """

    def __init__(self):
        super().__init__()

        weights = (
            ConvNeXt_Base_Weights
            .IMAGENET1K_V1
        )

        self.backbone = convnext_base(
            weights=weights
        )

        self.backbone.classifier[2] = (
            nn.Identity()
        )

    def forward(self, x):
        # combine batch + time so convnext sees a normal image batch
        B, T, C, H, W = x.shape

        x = x.reshape(
            B * T,
            C,
            H,
            W,
        )

        frame_features = (
            self.backbone(x)
        )

        frame_features = (
            frame_features
            .reshape(
                B,
                T,
                -1,
            )
        )

        return frame_features


def make_feature_loader(
    csv_path,
    num_frames,
    image_size,
    feature_batch_size,
    num_workers,
    pin_memory,
    prefetch_factor,
    path_col="path",
    missing_visual_segment_policy="drop",
    segment_col="visual_fake_segments",
    shuffle=False,
):
    """
    build the video dataloader used for convnext feature extraction.

    the dataset creates both the sampled frames and sampled-grid visual labels.
    worker/prefetch settings are passed in from the notebook so the final exp006
    extraction settings stay visible there.
    """
    dataset = TemporalVideoFeatureDataset(
        csv_path=csv_path,
        path_col=path_col,
        num_frames=num_frames,
        image_size=image_size,
        missing_visual_segment_policy=missing_visual_segment_policy,
        segment_col=segment_col,
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
        loader_kwargs[
            "prefetch_factor"
        ] = prefetch_factor

        loader_kwargs[
            "timeout"
        ] = 180

    return (
        dataset,
        DataLoader(
            **loader_kwargs
        ),
    )


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
    extract frozen frame features and save them in smaller shard files.

    each input video produces [T,1024] convnext features plus frame labels and
    sampled timestamps. the cached features/labels/times are stored as float16
    to reduce disk usage.

    every shard stores:
        features      [B, T, 1024]
        frame_labels  [B, T]
        frame_times   [B, T]
        video_labels  [B]
        indices       [B]

    if a complete shard manifest already exists and all listed files still exist,
    extraction is skipped and that cache is reused.
    """
    model.eval()

    split_dir = (
        Path(output_root)
        / split_name
    )

    split_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    manifest_path = (
        split_dir
        / "shard_manifest.csv"
    )

    if manifest_path.exists():
        existing = pd.read_csv(
            manifest_path
        )

        if (
            len(existing) > 0
            and existing["path"]
            .apply(
                lambda p: Path(p).exists()
            )
            .all()
        ):
            print(
                f"{split_name}: complete shard manifest already exists, "
                "skipping extraction."
            )

            display(
                existing.tail()
            )

            return existing

    rows = []
    start_time = time.time()

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
            features = model(
                frames
            )

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

        video_labels_cpu = (
            video_labels.cpu()
        )

        indices_cpu = (
            indices.cpu()
        )

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
            "created_at": (
                datetime.now()
                .isoformat(
                    timespec="seconds"
                )
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
            "n_videos": int(
                features_cpu.shape[0]
            ),
            "num_frames": int(
                features_cpu.shape[1]
            ),
            "feature_dim": int(
                features_cpu.shape[2]
            ),
            "positive_frames": float(
                frame_labels_cpu
                .sum()
                .item()
            ),
            "total_frames": int(
                frame_labels_cpu
                .numel()
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

    manifest_df = pd.DataFrame(
        rows
    )

    elapsed = (
        time.time()
        - start_time
    )

    manifest_df[
        "elapsed_total_sec"
    ] = elapsed

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

    print(
        "Manifest:",
        manifest_path,
    )

    return manifest_df
