# data / label helpers used by exp004
# mainly the split labels, segment parsing and temporal video dataset

import ast
import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from torchvision import transforms


def find_existing_split_source(project_root, candidates):
    """
    find the earlier experiment folder containing the train/val/test split.

    exp004 reuses an existing split instead of generating another one. the
    candidate experiment names are checked in order and the first folder with
    train.csv, val.csv and test.csv is returned.

    returns:
        source experiment name
        source experiment directory
        train csv path
        validation csv path
        test csv path
    """
    experiments_dir = Path(project_root) / "experiments"

    for name in candidates:
        run_dir = experiments_dir / name
        manifest_dir = run_dir / "manifests"

        train_csv = manifest_dir / "train.csv"
        val_csv = manifest_dir / "val.csv"
        test_csv = manifest_dir / "test.csv"

        if train_csv.exists() and val_csv.exists() and test_csv.exists():
            return name, run_dir, train_csv, val_csv, test_csv

    raise FileNotFoundError(
        "No split source found. Checked candidates:\n"
        + "\n".join(str(Path(project_root) / "experiments" / c) for c in candidates)
    )


def add_modality_labels(df, condition_col="condition"):
    """
    add the visual, audio and overall fake labels from the condition name.

    the four AV-Deepfake conditions encode which modality was manipulated.
    visual_fake and audio_fake keep those two targets separate, while
    binary_label is 1 whenever either modality is fake.

    this is useful in exp004 because the temporal target is specifically the
    visual manipulation, even though the source manifest also has any-fake
    binary labels.
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
        (df["visual_fake"] == 1) | (df["audio_fake"] == 1)
    ).astype(int)

    return df


def is_missing_value(x):
    """small helper for recognising empty segment fields in the manifest."""
    if x is None:
        return True

    if isinstance(x, float) and np.isnan(x):
        return True

    if isinstance(x, str) and x.strip().lower() in ["", "nan", "none", "null"]:
        return True

    return False


def parse_segments(value):
    """
    turn a segment annotation stored in the manifest into a python list.

    the manifest value can already be a list/dict, or it can be a string
    representation of one. json is tried first and ast.literal_eval is used as
    a fallback for python-style list/dict strings.

    values that cannot be parsed are treated as having no usable segments.
    """
    if is_missing_value(value):
        return []

    if isinstance(value, list):
        return value

    if isinstance(value, tuple):
        return list(value)

    if isinstance(value, dict):
        for key in ["segments", "fake_segments", "visual_fake_segments", "intervals"]:
            if key in value:
                return parse_segments(value[key])

        return [value]

    if isinstance(value, str):
        s = value.strip()

        if s in ["[]", "{}", "()"]:
            return []

        try:
            return parse_segments(json.loads(s))
        except Exception:
            pass

        try:
            return parse_segments(ast.literal_eval(s))
        except Exception:
            pass

        return []

    return []


def get_segment_bounds(segment):
    """
    read the start/end values from one segment annotation.

    known frame-index keys are labelled as frames and known time keys are
    labelled as seconds. plain two-item lists/tuples are accepted but their unit
    stays unknown until infer_units() is called.

    returns:
        (start, end, unit), or None when the structure is not recognised
    """
    if segment is None:
        return None

    if isinstance(segment, dict):
        frame_keys = [
            ("start_frame", "end_frame"),
            ("frame_start", "frame_end"),
            ("start_idx", "end_idx"),
        ]

        time_keys = [
            ("start", "end"),
            ("start_time", "end_time"),
            ("time_start", "time_end"),
            ("begin", "finish"),
        ]

        for a, b in frame_keys:
            if a in segment and b in segment:
                return float(segment[a]), float(segment[b]), "frames"

        for a, b in time_keys:
            if a in segment and b in segment:
                return float(segment[a]), float(segment[b]), "seconds"

        return None

    if isinstance(segment, (list, tuple)) and len(segment) >= 2:
        return float(segment[0]), float(segment[1]), "unknown"

    return None


def get_row_segments(row, segment_col, fallback_segment_cols=None):
    """
    get the visual manipulation segments for one manifest row.

    exp004 prefers visual_fake_segments. fallback columns are only checked if
    the preferred column has no usable segments. in the final experiment the
    fallback list is empty, so generic fake_segments are not used for the
    visual target.
    """
    fallback_segment_cols = fallback_segment_cols or []
    segments = []

    if segment_col in row:
        segments = parse_segments(row.get(segment_col))

    if len(segments) == 0:
        for col in fallback_segment_cols:
            if col in row:
                segments = parse_segments(row.get(col))

                if len(segments) > 0:
                    break

    return segments


def infer_units(bounds_list, duration_sec, total_frames):
    """
    decide whether unlabelled segment bounds are more likely seconds or frames.

    explicit units always win. for unknown units, a segment ending well beyond
    the video duration but still within the total frame count is treated as
    frame indices. otherwise seconds are used.
    """
    known = [
        unit
        for _, _, unit in bounds_list
        if unit in ["seconds", "frames"]
    ]

    if len(known) > 0:
        return known[0]

    if len(bounds_list) == 0:
        return "seconds"

    max_end = max(end for _, end, _ in bounds_list)

    if duration_sec is not None and total_frames is not None:
        if max_end > duration_sec + 5 and max_end <= total_frames + 5:
            return "frames"

    return "seconds"


def frame_labels_from_segments(
    segments,
    target_indices,
    target_times,
    duration_sec=None,
    total_frames=None,
):
    """
    convert manipulation intervals into labels on the sampled 64-frame grid.

    the function first normalises every usable interval into start/end bounds.
    each sampled position is then marked 1 when its frame index or timestamp
    lies inside at least one manipulation interval.

    this means the labels describe the sampled grid used by the model, not every
    original video frame and not a continuous-time boundary representation.

    returns:
        float32 array with one 0/1 label for each sampled position
    """
    labels = np.zeros(len(target_indices), dtype=np.float32)

    bounds_list = []

    for seg in segments:
        bounds = get_segment_bounds(seg)

        if bounds is None:
            continue

        start, end, unit = bounds

        if end < start:
            start, end = end, start

        bounds_list.append((start, end, unit))

    if len(bounds_list) == 0:
        return labels

    inferred_unit = infer_units(
        bounds_list,
        duration_sec,
        total_frames,
    )

    for start, end, unit in bounds_list:
        unit = unit if unit in ["seconds", "frames"] else inferred_unit

        if unit == "frames":
            inside = (target_indices >= start) & (target_indices <= end)
        else:
            inside = (target_times >= start) & (target_times <= end)

        labels[inside] = 1.0

    return labels


def segment_availability_summary(
    df,
    split_name,
    condition_col,
    segment_col,
    fallback_segment_cols=None,
):
    """
    summarise how many rows in each condition have usable segment annotations.

    this was an early safety check before building temporal labels. it is
    especially useful for visually-fake rows because missing visual segments
    would otherwise make the temporal target unreliable.
    """
    rows = []

    for condition, group in df.groupby(condition_col):
        parsed_counts = group.apply(
            lambda row: len(
                get_row_segments(
                    row,
                    segment_col,
                    fallback_segment_cols,
                )
            ),
            axis=1,
        )

        visual_fake_value = (
            int(group["visual_fake"].iloc[0])
            if len(group)
            else None
        )

        rows.append({
            "split": split_name,
            "condition": condition,
            "n_rows": len(group),
            "visual_fake": visual_fake_value,
            "rows_with_segments": int((parsed_counts > 0).sum()),
            "rows_without_segments": int((parsed_counts == 0).sum()),
            "mean_segments_per_row": (
                float(parsed_counts.mean())
                if len(parsed_counts)
                else 0.0
            ),
        })

    return pd.DataFrame(rows)


def filter_missing_visual_segments(
    df,
    split_name,
    missing_visual_segment_policy,
    segment_col,
    fallback_segment_cols=None,
):
    """
    apply exp004's policy for visually-fake rows with no usable visual segments.

    a visual temporal model needs frame-level targets, so a visually-fake row
    without a segment annotation is ambiguous. the final setting uses "drop",
    which removes those rows before feature extraction/training.

    other supported values are kept because they were part of the notebook:
        all_fake  -> keep the row and later label every sampled frame fake
        keep_zero -> keep the row with zero-valued frame labels

    returns the filtered dataframe with a has_visual_segment_annotation column.
    """
    df = df.copy().reset_index(drop=True)

    fallback_segment_cols = fallback_segment_cols or []

    has_segments = df.apply(
        lambda row: len(
            get_row_segments(
                row,
                segment_col,
                fallback_segment_cols,
            )
        ) > 0,
        axis=1,
    )

    df["has_visual_segment_annotation"] = has_segments.astype(int)

    visual_fake_missing = (
        (df["visual_fake"] == 1)
        & (~has_segments)
    )

    print(f"{split_name}: total rows = {len(df)}")
    print(
        f"{split_name}: visual_fake rows = "
        f"{int((df['visual_fake'] == 1).sum())}"
    )
    print(
        f"{split_name}: visual_fake rows missing segments = "
        f"{int(visual_fake_missing.sum())}"
    )

    if missing_visual_segment_policy == "drop":
        before = len(df)
        df = df[~visual_fake_missing].copy().reset_index(drop=True)
        print(f"{split_name}: dropped {before - len(df)} rows")

    elif missing_visual_segment_policy in ["all_fake", "keep_zero"]:
        print(
            f"{split_name}: kept missing visual segments with policy "
            f"{missing_visual_segment_policy}"
        )

    else:
        raise ValueError(
            f"Unknown policy: {missing_visual_segment_policy}"
        )

    return df


def group_overlap_summary(train_df, val_df, test_df):
    """
    count source clip-group overlap between the three saved partitions.

    the expected result for the group-disjoint split is zero overlap for every
    train/validation/test pair. this checks clip_group only; it does not prove
    that identities are disjoint.
    """
    if "clip_group" not in train_df.columns:
        return {
            "train_val_group_overlap": None,
            "train_test_group_overlap": None,
            "val_test_group_overlap": None,
        }

    train_groups = set(train_df["clip_group"])
    val_groups = set(val_df["clip_group"])
    test_groups = set(test_df["clip_group"])

    return {
        "train_val_group_overlap": len(train_groups & val_groups),
        "train_test_group_overlap": len(train_groups & test_groups),
        "val_test_group_overlap": len(val_groups & test_groups),
    }


class TemporalVideoFeatureDataset(Dataset):
    """
    video dataset used to build exp004's frame-level feature shards.

    each video is sampled at num_frames uniformly spaced positions. the frames
    are imagenet-normalised for convnext and a visual fake label is produced for
    every sampled position from the row's visual manipulation segments.

    label behaviour:
        visual_fake == 0
            -> every sampled frame is labelled visually real

        visual_fake == 1 with segments
            -> frame_labels_from_segments() marks sampled positions inside the
               annotated visual manipulation interval(s)

        visual_fake == 1 without segments
            -> behaviour depends on missing_visual_segment_policy. with the
               final "drop" setting these rows should already have been removed.

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
        path_col,
        num_frames,
        image_size,
        missing_visual_segment_policy,
        segment_col,
        fallback_segment_cols=None,
    ):
        self.df = pd.read_csv(csv_path).reset_index(drop=True)
        self.path_col = path_col
        self.num_frames = num_frames
        self.missing_visual_segment_policy = missing_visual_segment_policy
        self.segment_col = segment_col
        self.fallback_segment_cols = fallback_segment_cols or []

        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
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
        """build the sampled visual labels for one video row."""
        visual_fake = int(row.get("visual_fake", 0))

        # audio-only and real clips should stay visually real at every position
        if visual_fake == 0:
            return np.zeros(
                self.num_frames,
                dtype=np.float32,
            )

        segments = get_row_segments(
            row,
            self.segment_col,
            self.fallback_segment_cols,
        )

        if len(segments) == 0:
            if self.missing_visual_segment_policy == "all_fake":
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

    def _sample_frames(self, video_path, row):
        """
        sample the 64 visual positions and return frames, labels and timestamps.

        opencv reads the clip sequentially. linspace is used to choose positions
        across the complete reported frame range. fps converts those positions
        to seconds for annotations stored as time values.

        if fps is missing/invalid the notebook's existing 25 fps fallback is
        used. if decoding gives too few frames, the final valid frame is repeated
        so the input tensor still has a fixed length.
        """
        cap = cv2.VideoCapture(str(video_path))

        if not cap.isOpened():
            raise RuntimeError(
                f"Could not open video: {video_path}"
            )

        total_frames = int(
            cap.get(cv2.CAP_PROP_FRAME_COUNT)
        )
        fps = float(
            cap.get(cv2.CAP_PROP_FPS)
        )

        if total_frames <= 0:
            cap.release()
            raise RuntimeError(
                f"No frames found: {video_path}"
            )

        if fps <= 0 or not np.isfinite(fps):
            fps = 25.0

        duration_sec = total_frames / fps

        target_indices = np.linspace(
            0,
            total_frames - 1,
            self.num_frames,
        ).astype(int)

        target_times = target_indices / fps
        target_set = set(target_indices.tolist())

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

                frame = self.transform(frame)
                frames.append(frame)

                if len(frames) == self.num_frames:
                    break

            current_idx += 1

        cap.release()

        if len(frames) == 0:
            raise RuntimeError(
                f"No frames sampled: {video_path}"
            )

        while len(frames) < self.num_frames:
            frames.append(frames[-1])

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

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        frames, frame_labels, frame_times = self._sample_frames(
            row[self.path_col],
            row,
        )

        visual_video_label = int(
            row.get(
                "visual_fake",
                int(frame_labels.max().item() > 0),
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
