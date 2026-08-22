# manifest / split / temporal label helpers used by exp006

import ast
import json
from pathlib import Path

import numpy as np
import pandas as pd


def find_manifest(candidates):
    """
    find the first usable full-validation manifest from the known locations.

    exp006 starts from the full usable validation manifest rather than reusing
    the balanced exp002/exp004 split. this helper only finds the source csv; it
    does not create or alter any rows.

    returns:
        Path to the first manifest that exists
    """
    for path in candidates:
        path = Path(path)

        if path.exists():
            return path

    raise FileNotFoundError(
        "No usable manifest found. Checked:\n"
        + "\n".join(str(p) for p in candidates)
    )


def load_metadata_json_if_available(candidates):
    """
    load the first metadata json that exists.

    the manifest normally already contains the fields needed for exp006. this
    is just a fallback for things such as visual_fake_segments if the source csv
    is missing them.

    returns:
        metadata dataframe (or None)
        path that was used (or None)
    """
    for path in candidates:
        path = Path(path)

        if path.exists():
            print("Loading metadata:", path)

            with open(path, "r") as f:
                data = json.load(f)

            meta_df = pd.DataFrame(data)
            print("Metadata shape:", meta_df.shape)

            return meta_df, path

    print("No metadata JSON found. Continuing with manifest columns.")
    return None, None


def enrich_with_metadata_if_needed(
    df,
    meta_df,
    relative_path_col="relative_path",
):
    """
    fill missing metadata columns from the optional metadata json.

    existing manifest values are kept. metadata is only used to add or fill the
    known fields when they are missing. the merge is based on relative_path/file
    and does not change the train/val/test split because the split is created
    later in exp006.

    if metadata is unavailable or there is no usable merge key, the dataframe is
    returned unchanged.
    """
    df = df.copy()

    if meta_df is None:
        return df

    needed_cols = [
        "file",
        "original",
        "split",
        "modify_type",
        "audio_model",
        "fake_segments",
        "audio_fake_segments",
        "visual_fake_segments",
        "video_frames",
        "audio_frames",
        "video_model",
    ]

    missing_cols = [
        c for c in needed_cols
        if c not in df.columns
    ]

    if len(missing_cols) == 0:
        print("Manifest already has key metadata columns.")
        return df

    print("Trying to enrich missing metadata columns:", missing_cols)

    if "file" not in meta_df.columns:
        print("Metadata has no file column, cannot enrich.")
        return df

    merge_left = (
        relative_path_col
        if relative_path_col in df.columns
        else "file"
    )

    if merge_left not in df.columns:
        print("No relative_path/file column in manifest, cannot enrich.")
        return df

    add_cols = [
        c for c in needed_cols
        if c in meta_df.columns
    ]

    meta = (
        meta_df[add_cols]
        .drop_duplicates("file")
    )

    merged = df.merge(
        meta,
        left_on=merge_left,
        right_on="file",
        how="left",
        suffixes=("", "_meta"),
    )

    for c in needed_cols:
        meta_c = c + "_meta"

        if c not in merged.columns and meta_c in merged.columns:
            merged[c] = merged[meta_c]

        elif c in merged.columns and meta_c in merged.columns:
            merged[c] = merged[c].combine_first(merged[meta_c])

    drop_cols = [
        c for c in merged.columns
        if c.endswith("_meta")
    ]

    merged = merged.drop(
        columns=drop_cols,
        errors="ignore",
    )

    return merged


def derive_clip_group(
    row,
    relative_path_col="relative_path",
    path_col="path",
):
    """
    derive the source clip-group key used for the group-disjoint split.

    relative_path is preferred because it is less tied to one machine. if that
    is unavailable, file/path is used instead. the parent directory becomes the
    clip_group value.

    this is clip-group separation, not identity-level separation.
    """
    if (
        relative_path_col in row
        and pd.notna(row[relative_path_col])
    ):
        return str(
            Path(str(row[relative_path_col])).parent
        ).replace("\\", "/")

    if "file" in row and pd.notna(row["file"]):
        return str(
            Path(str(row["file"])).parent
        ).replace("\\", "/")

    return str(
        Path(str(row[path_col])).parent
    ).replace("\\", "/")


def add_modality_labels(
    df,
    condition_col="condition",
    relative_path_col="relative_path",
    path_col="path",
):
    """
    add visual_fake, audio_fake, binary_label and clip_group.

    modality labels come directly from the four dataset conditions:
        real                         -> visual 0, audio 0
        fake_video_real_audio        -> visual 1, audio 0
        real_video_fake_audio        -> visual 0, audio 1
        fake_video_fake_audio        -> visual 1, audio 1

    binary_label is 1 when either modality is fake. exp006 uses visual_fake for
    temporal supervision but keeps the other labels for condition analysis.
    """
    df = df.copy()

    if condition_col not in df.columns:
        raise ValueError(
            f"Missing condition column: {condition_col}"
        )

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

    if "clip_group" not in df.columns:
        df["clip_group"] = df.apply(
            lambda row: derive_clip_group(
                row,
                relative_path_col=relative_path_col,
                path_col=path_col,
            ),
            axis=1,
        )

    return df


def is_missing_value(x):
    """check the empty forms that can appear in a segment field."""
    if x is None:
        return True

    if isinstance(x, float) and np.isnan(x):
        return True

    if isinstance(x, str) and x.strip().lower() in [
        "",
        "nan",
        "none",
        "null",
    ]:
        return True

    return False


def parse_segments(value):
    """
    turn a stored visual segment annotation into a normal python list.

    the field can already be a list/dict or can be a string version of one.
    json is tried first and ast.literal_eval is kept as a fallback for
    python-style strings.

    values that cannot be parsed are treated as having no usable segments.
    """
    if is_missing_value(value):
        return []

    if isinstance(value, list):
        return value

    if isinstance(value, tuple):
        return list(value)

    if isinstance(value, dict):
        for key in [
            "segments",
            "fake_segments",
            "visual_fake_segments",
            "intervals",
        ]:
            if key in value:
                return parse_segments(value[key])

        return [value]

    if isinstance(value, str):
        s = value.strip()

        if s in ["[]", "{}", "()"]:
            return []

        try:
            return parse_segments(
                json.loads(s)
            )
        except Exception:
            pass

        try:
            return parse_segments(
                ast.literal_eval(s)
            )
        except Exception:
            pass

        return []

    return []


def get_segment_bounds(segment):
    """
    read start/end values and an obvious unit from one visual segment.

    known frame-index keys are returned as frames and time keys are returned as
    seconds. a plain two-item list/tuple is accepted with unit='unknown' so the
    later inference logic can decide how to interpret it.
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
                return (
                    float(segment[a]),
                    float(segment[b]),
                    "frames",
                )

        for a, b in time_keys:
            if a in segment and b in segment:
                return (
                    float(segment[a]),
                    float(segment[b]),
                    "seconds",
                )

        return None

    if (
        isinstance(segment, (list, tuple))
        and len(segment) >= 2
    ):
        return (
            float(segment[0]),
            float(segment[1]),
            "unknown",
        )

    return None


def get_visual_segments(
    row,
    segment_col="visual_fake_segments",
):
    """
    get the visual manipulation segments for one row.

    exp006 deliberately uses visual_fake_segments and does not fall back to the
    generic fake_segments column.
    """
    if segment_col not in row:
        return []

    return parse_segments(
        row.get(segment_col)
    )


def infer_units(
    bounds_list,
    duration_sec,
    total_frames,
):
    """
    infer seconds vs frame indices when segment bounds have no explicit unit.

    explicit units always win. otherwise, values that extend well beyond the
    video duration but still fit the reported frame count are treated as frame
    indices; the remaining unknown values are treated as seconds.
    """
    known = [
        u for _, _, u in bounds_list
        if u in ["seconds", "frames"]
    ]

    if len(known) > 0:
        return known[0]

    if len(bounds_list) == 0:
        return "seconds"

    max_end = max(
        end for _, end, _ in bounds_list
    )

    if (
        duration_sec is not None
        and total_frames is not None
    ):
        if (
            max_end > duration_sec + 5
            and max_end <= total_frames + 5
        ):
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
    convert visual manipulation intervals onto the sampled frame grid.

    every one of the 64 sampled positions is labelled 1 when its frame index or
    timestamp falls inside at least one visual fake interval.

    this produces labels on the sampled grid only. it is not a dense label for
    every original frame and it is not a continuous-time boundary target.
    """
    labels = np.zeros(
        len(target_indices),
        dtype=np.float32,
    )

    bounds_list = []

    for seg in segments:
        bounds = get_segment_bounds(seg)

        if bounds is None:
            continue

        start, end, unit = bounds

        if end < start:
            start, end = end, start

        bounds_list.append(
            (start, end, unit)
        )

    if len(bounds_list) == 0:
        return labels

    inferred_unit = infer_units(
        bounds_list,
        duration_sec,
        total_frames,
    )

    for start, end, unit in bounds_list:
        unit = (
            unit
            if unit in ["seconds", "frames"]
            else inferred_unit
        )

        if unit == "frames":
            inside = (
                (target_indices >= start)
                & (target_indices <= end)
            )
        else:
            inside = (
                (target_times >= start)
                & (target_times <= end)
            )

        labels[inside] = 1.0

    return labels


def segment_count(
    row,
    segment_col="visual_fake_segments",
):
    """count the usable visual segments attached to one row."""
    return len(
        get_visual_segments(
            row,
            segment_col=segment_col,
        )
    )


def group_disjoint_random_split(
    df,
    group_col="clip_group",
    train_frac=0.70,
    val_frac=0.15,
    test_frac=0.15,
    seed=42,
):
    """
    create the natural-distribution group-disjoint split used by exp006.

    unique clip groups are sorted, shuffled with the configured seed and then
    divided approximately 70/15/15 by number of groups. every row belonging to a
    group follows that group into the same partition.

    there is no class balancing or condition balancing here. that is an important
    difference from exp004 and is why exp006 is not a controlled single-variable
    ablation of the balanced experiment.
    """
    assert (
        abs(
            train_frac
            + val_frac
            + test_frac
            - 1.0
        )
        < 1e-6
    )

    rng = np.random.default_rng(
        seed
    )

    groups = np.array(
        sorted(
            df[group_col]
            .dropna()
            .unique()
        )
    )

    rng.shuffle(groups)

    n_groups = len(groups)

    n_train = int(
        round(
            n_groups
            * train_frac
        )
    )

    n_val = int(
        round(
            n_groups
            * val_frac
        )
    )

    train_groups = set(
        groups[:n_train]
    )

    val_groups = set(
        groups[
            n_train:n_train + n_val
        ]
    )

    test_groups = set(
        groups[
            n_train + n_val:
        ]
    )

    train_df = (
        df[
            df[group_col]
            .isin(train_groups)
        ]
        .copy()
        .reset_index(drop=True)
    )

    val_df = (
        df[
            df[group_col]
            .isin(val_groups)
        ]
        .copy()
        .reset_index(drop=True)
    )

    test_df = (
        df[
            df[group_col]
            .isin(test_groups)
        ]
        .copy()
        .reset_index(drop=True)
    )

    return (
        train_df,
        val_df,
        test_df,
    )


def group_overlap_summary(
    train_df,
    val_df,
    test_df,
):
    """
    count clip-group overlap between train, validation and test.

    zero overlap is expected for every pair. this checks the derived clip_group
    only; it does not establish identity-disjoint evaluation.
    """
    train_groups = set(
        train_df["clip_group"]
    )

    val_groups = set(
        val_df["clip_group"]
    )

    test_groups = set(
        test_df["clip_group"]
    )

    return {
        "train_groups": len(train_groups),
        "val_groups": len(val_groups),
        "test_groups": len(test_groups),
        "train_val_group_overlap": len(
            train_groups & val_groups
        ),
        "train_test_group_overlap": len(
            train_groups & test_groups
        ),
        "val_test_group_overlap": len(
            val_groups & test_groups
        ),
    }
