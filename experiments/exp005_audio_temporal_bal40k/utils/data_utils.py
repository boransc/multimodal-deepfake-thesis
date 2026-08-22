# manifest / label helpers used by exp005

import ast
import json
from pathlib import Path

import numpy as np
import pandas as pd


def find_existing_split_source(project_root, candidates):
    """
    find the earlier experiment folder containing the split reused by exp005.

    this does not build another split. it checks the candidate experiment folders
    for train.csv, val.csv and test.csv and returns the first complete set.

    returns:
        source experiment name
        source experiment folder
        train csv
        validation csv
        test csv
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


def add_modality_labels(df, condition_col):
    """
    derive visual_fake, audio_fake and binary_label from the dataset condition.

    exp005 is specifically an audio task, so audio_fake is the important target:
        real                         -> 0
        fake_video_real_audio        -> 0
        real_video_fake_audio        -> 1
        fake_video_fake_audio        -> 1

    binary_label stays useful for general dataset checks but it is not the
    temporal supervision used by the audio model.
    """
    df = df.copy()

    if condition_col not in df.columns:
        raise ValueError(f"Missing condition column: {condition_col}")

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


def load_metadata_json_if_available(candidates):
    """
    load the first metadata json that exists.

    the copied split csvs normally already contain the temporal metadata. this
    helper was kept as a fallback in case audio_fake_segments or other metadata
    columns were missing from those csvs.
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

    print("No metadata JSON found. Continuing with split CSV columns.")
    return None, None


def enrich_with_metadata_if_needed(df, meta_df, relative_path_col):
    """
    fill missing temporal metadata columns from val_metadata.json when possible.

    the function only adds/fills the known metadata fields and keeps the existing
    split rows. it does not change the train/val/test membership.

    if there is no metadata dataframe, file key, or usable merge key, the input
    dataframe is returned unchanged.
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
        print("Split already has key metadata columns.")
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


def is_missing_value(x):
    """check the different empty forms that can appear in segment columns."""
    if x is None:
        return True

    if isinstance(x, float) and np.isnan(x):
        return True

    if isinstance(x, str) and x.strip().lower() in ["", "nan", "none", "null"]:
        return True

    return False


def parse_segments(value):
    """
    turn a stored segment annotation into a normal python list.

    segment columns can already contain lists/dicts or can contain string versions
    of them. json is tried first and ast.literal_eval is used as a fallback for
    python-style strings.

    anything that cannot be parsed is treated as having no usable segments.
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
            "audio_fake_segments",
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
    read start/end values and the obvious unit from one segment object.

    supported explicit forms include sample indices, frame indices and seconds.
    a plain [start, end] pair is returned with unit='unknown' so the later unit
    inference logic can decide how to interpret it.
    """
    if segment is None:
        return None

    if isinstance(segment, dict):
        sample_keys = [
            ("start_sample", "end_sample"),
            ("sample_start", "sample_end"),
        ]

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

        for a, b in sample_keys:
            if a in segment and b in segment:
                return float(segment[a]), float(segment[b]), "samples"

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


def get_audio_segments(row, audio_segment_col, fallback_segment_cols):
    """
    get the audio manipulation intervals for one manifest row.

    audio_fake_segments is preferred. the fallback list is normally empty for
    exp005 because generic fake_segments could refer to a non-audio manipulation.
    """
    segments = []

    if audio_segment_col in row:
        segments = parse_segments(
            row.get(audio_segment_col)
        )

    if len(segments) == 0:
        for col in fallback_segment_cols:
            if col in row:
                segments = parse_segments(
                    row.get(col)
                )

                if len(segments) > 0:
                    break

    return segments


def infer_audio_segment_unit(
    bounds_list,
    duration_sec,
    audio_sample_rate,
    audio_frames=None,
):
    """
    infer the unit for segment bounds that were stored without an explicit unit.

    order used by the original notebook:
    1. if the values fit within the clip duration, treat them as seconds
    2. if they fit the metadata audio_frames count, treat them as frame indices
    3. if they look like sample indices at the configured sample rate, use samples
    4. otherwise fall back to seconds
    """
    known = [
        u for _, _, u in bounds_list
        if u in ["seconds", "frames", "samples"]
    ]

    if len(known) > 0:
        return known[0]

    if len(bounds_list) == 0:
        return "seconds"

    max_end = max(
        end for _, end, _ in bounds_list
    )

    if max_end <= duration_sec + 1.0:
        return "seconds"

    if audio_frames is not None and pd.notna(audio_frames):
        try:
            audio_frames = float(audio_frames)

            if (
                audio_frames > 0
                and max_end <= audio_frames + 5
            ):
                return "frames"

        except Exception:
            pass

    if (
        max_end
        <= duration_sec * audio_sample_rate
        + audio_sample_rate
    ):
        return "samples"

    return "seconds"


def convert_bounds_to_seconds(
    start,
    end,
    unit,
    duration_sec,
    audio_sample_rate,
    audio_frames=None,
):
    """
    convert one segment interval into seconds.

    frame-index annotations are scaled across the full clip using audio_frames
    when that metadata exists. sample-index annotations are divided by the
    configured sample rate.
    """
    if unit == "seconds":
        return start, end

    if unit == "samples":
        return (
            start / audio_sample_rate,
            end / audio_sample_rate,
        )

    if unit == "frames":
        if audio_frames is not None and pd.notna(audio_frames):
            audio_frames = float(audio_frames)

            if audio_frames > 0:
                return (
                    (start / audio_frames) * duration_sec,
                    (end / audio_frames) * duration_sec,
                )

        # same fallback behaviour as the notebook if audio_frames is unavailable
        return start, end

    return start, end


def audio_window_labels_from_segments(
    segments,
    window_centres,
    duration_sec,
    audio_sample_rate,
    audio_frames=None,
):
    """
    convert annotated audio intervals into labels for the sampled temporal windows.

    every window centre is labelled 1 when it falls inside at least one fake
    audio segment. the output is therefore a label vector on the 64 sampled
    window centres, not a dense sample-by-sample audio mask.
    """
    labels = np.zeros(
        len(window_centres),
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

    inferred_unit = infer_audio_segment_unit(
        bounds_list,
        duration_sec,
        audio_sample_rate,
        audio_frames=audio_frames,
    )

    for start, end, unit in bounds_list:
        unit = (
            unit
            if unit in ["seconds", "frames", "samples"]
            else inferred_unit
        )

        start_sec, end_sec = convert_bounds_to_seconds(
            start,
            end,
            unit,
            duration_sec=duration_sec,
            audio_sample_rate=audio_sample_rate,
            audio_frames=audio_frames,
        )

        inside = (
            (window_centres >= start_sec)
            & (window_centres <= end_sec)
        )

        labels[inside] = 1.0

    return labels


def audio_segment_count(row, audio_segment_col, fallback_segment_cols):
    """count the usable audio manipulation intervals on one manifest row."""
    return len(
        get_audio_segments(
            row,
            audio_segment_col,
            fallback_segment_cols,
        )
    )


def audit_audio_segments(
    df,
    split_name,
    condition_col,
    audio_segment_col,
    fallback_segment_cols,
):
    """
    audit the relationship between modality labels and audio segment annotations.

    the useful cases to spot are:
    - audio_fake rows with no temporal segment labels
    - audio-real rows that still contain audio segment metadata

    audio-real conflicts are kept as audio-real negatives later; this audit just
    makes those metadata inconsistencies visible.
    """
    df = df.copy()

    df["audio_segment_count"] = df.apply(
        lambda row: audio_segment_count(
            row,
            audio_segment_col,
            fallback_segment_cols,
        ),
        axis=1,
    )

    df["has_audio_segment_annotation"] = (
        df["audio_segment_count"] > 0
    ).astype(int)

    df["audio_fake_missing_segments"] = (
        (df["audio_fake"] == 1)
        & (df["audio_segment_count"] == 0)
    ).astype(int)

    df["audio_real_with_segments"] = (
        (df["audio_fake"] == 0)
        & (df["audio_segment_count"] > 0)
    ).astype(int)

    summary = (
        df.groupby(condition_col)
        .agg(
            n_rows=("audio_fake", "size"),
            audio_fake=("audio_fake", "first"),
            rows_with_audio_segments=("has_audio_segment_annotation", "sum"),
            audio_fake_missing_segments=("audio_fake_missing_segments", "sum"),
            audio_real_with_segments=("audio_real_with_segments", "sum"),
            mean_audio_segments=("audio_segment_count", "mean"),
        )
        .reset_index()
    )

    summary.insert(
        0,
        "split",
        split_name,
    )

    return df, summary


def filter_audio_temporal_rows(
    df,
    split_name,
    missing_audio_segment_policy,
    condition_col,
):
    """
    apply the final policy for audio-fake rows missing temporal segment labels.

    with the final exp005 setting ('drop'), only rows that are audio-fake and
    missing audio temporal labels are removed. audio-real rows are kept even if
    their metadata contains odd segment entries because their modality target is
    still real audio.
    """
    before = len(df)

    if missing_audio_segment_policy == "drop":
        df = df[
            df["audio_fake_missing_segments"] == 0
        ].copy().reset_index(drop=True)

    elif missing_audio_segment_policy in ["all_fake", "keep_zero"]:
        df = df.copy().reset_index(drop=True)

    else:
        raise ValueError(
            f"Unknown MISSING_AUDIO_SEGMENT_POLICY: {missing_audio_segment_policy}"
        )

    print(
        f"{split_name}: before={before}, "
        f"after={len(df)}, dropped={before - len(df)}"
    )

    print(
        df[condition_col]
        .value_counts(dropna=False)
    )

    print("audio_fake distribution:")
    print(
        df["audio_fake"]
        .value_counts(dropna=False)
    )

    return df


def group_overlap_summary(train_df, val_df, test_df):
    """
    count clip_group overlap between train, validation and test.

    zero overlap is expected because exp005 reuses the earlier group-disjoint
    split. this checks source clip groups only, not identity-level separation.
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
        "train_groups": len(train_groups),
        "val_groups": len(val_groups),
        "test_groups": len(test_groups),
        "train_val_group_overlap": len(train_groups & val_groups),
        "train_test_group_overlap": len(train_groups & test_groups),
        "val_test_group_overlap": len(val_groups & test_groups),
    }
