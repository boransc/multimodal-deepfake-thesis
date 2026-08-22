# dataset/view helpers used by exp008

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd


def find_run(candidates, experiments_dir):
    """
    find the first existing source experiment folder.

    exp008 reuses already-trained exp004, exp005 and exp007 outputs. the
    candidate names cover the folder names used during the project so the
    notebook can locate the completed source runs without hard-coding only one
    development name.

    returns:
        matched run name
        matched run directory
    """
    experiments_dir = Path(experiments_dir)

    for name in candidates:
        path = experiments_dir / name

        if path.exists():
            return name, path

    raise FileNotFoundError(
        "No source run found:\n"
        + "\n".join(
            str(experiments_dir / name)
            for name in candidates
        )
    )


def result_dir(run):
    """
    find the results folder used by one completed experiment.

    some project runs used 'results' and some older folders used 'res', so both
    names are checked. nothing is created here because exp008 must use the
    existing source-domain fusion outputs.
    """
    run = Path(run)

    for name in ["results", "res"]:
        path = run / name

        if path.exists():
            return path

    raise FileNotFoundError(
        f"No results/res folder in {run}"
    )


def stable_key(value, seed=42):
    """
    create a deterministic hash used when selecting one FakeAVCeleb example.

    the same input value + seed always gives the same key, which means the
    source-matched views do not depend on dataframe row order.
    """
    return hashlib.sha1(
        f"{seed}|{value}".encode("utf-8")
    ).hexdigest()


def build_source_matched_four(
    full_manifest,
    seed=42,
    max_sources=None,
):
    """
    build the controlled four-condition FakeAVCeleb view.

    first, one physical video is selected deterministically for each
    source/condition pair using the relative path hash. only source identities
    that are represented in all four conditions are then kept.

    this gives one row per source in:
        real
        fake video / real audio
        real video / fake audio
        fake video / fake audio

    max_sources is only used for the small smoke run. when it is None all shared
    source identities are retained.

    returns:
        matched dataframe
        sorted list of shared source identities
    """
    one_per_source_condition = (
        full_manifest.assign(
            selection_key=full_manifest["relative_path"].map(
                lambda value: stable_key(
                    value,
                    seed=seed,
                )
            )
        )
        .sort_values(
            [
                "condition",
                "source",
                "selection_key",
            ]
        )
        .drop_duplicates(
            [
                "condition",
                "source",
            ],
            keep="first",
        )
    )

    condition_sources = [
        set(group["source"])
        for _, group in one_per_source_condition.groupby(
            "condition"
        )
    ]

    shared_sources = sorted(
        set.intersection(
            *condition_sources
        )
    )

    if max_sources is not None:
        shared_sources = sorted(
            shared_sources,
            key=lambda value: stable_key(
                value,
                seed=seed,
            ),
        )[:max_sources]

    matched = (
        one_per_source_condition[
            one_per_source_condition[
                "source"
            ].isin(shared_sources)
        ]
        .copy()
    )

    expected_rows = (
        len(shared_sources)
        * 4
    )

    if len(matched) != expected_rows:
        raise RuntimeError(
            f"Source-matched view has {len(matched)} rows; "
            f"expected {expected_rows}."
        )

    counts = (
        matched.groupby(
            "condition"
        )["source"]
        .nunique()
    )

    if counts.nunique() != 1:
        raise RuntimeError(
            "Source-matched condition counts differ: "
            f"{counts.to_dict()}"
        )

    return (
        matched.drop(
            columns=["selection_key"]
        ),
        shared_sources,
    )


def build_source_matched_binary(
    source_matched_four,
    seed=42,
):
    """
    build the controlled one-real / one-fake view used for binary evaluation.

    every source contributes its one real row plus exactly one fake row. the fake
    condition is assigned deterministically from the source id so the selection
    is reproducible and approximately spread across the three fake conditions.

    the result should therefore contain two rows per source with exactly balanced
    real/fake binary labels.
    """
    fake_conditions = [
        "fake_video_real_audio",
        "real_video_fake_audio",
        "fake_video_fake_audio",
    ]

    rows = []

    for source, group in source_matched_four.groupby(
        "source",
        sort=True,
    ):
        real_row = group[
            group["condition"] == "real"
        ]

        if len(real_row) != 1:
            raise RuntimeError(
                f"Expected one real row for source {source}, "
                f"found {len(real_row)}"
            )

        rows.append(
            real_row.iloc[0]
        )

        fake_index = (
            int(
                stable_key(
                    source,
                    seed=seed,
                ),
                16,
            )
            % len(fake_conditions)
        )

        fake_condition = (
            fake_conditions[
                fake_index
            ]
        )

        fake_row = group[
            group["condition"]
            == fake_condition
        ]

        if len(fake_row) != 1:
            raise RuntimeError(
                f"Expected one {fake_condition} row for source {source}, "
                f"found {len(fake_row)}"
            )

        rows.append(
            fake_row.iloc[0]
        )

    return (
        pd.DataFrame(rows)
        .reset_index(drop=True)
    )


def parse_bool_series(
    series,
    column_name,
):
    """
    safely convert one audited manifest column into real booleans.

    the metadata audit may have written bools directly or may have serialised
    common true/false strings. unknown values are rejected rather than silently
    being interpreted as truthy strings.
    """
    if pd.api.types.is_bool_dtype(
        series
    ):
        return series.astype(bool)

    normalised = (
        series.astype(str)
        .str.strip()
        .str.lower()
    )

    mapping = {
        "true": True,
        "1": True,
        "yes": True,
        "false": False,
        "0": False,
        "no": False,
    }

    unknown = sorted(
        set(normalised)
        - set(mapping)
    )

    if unknown:
        raise ValueError(
            f"Could not parse boolean column {column_name}; "
            f"unknown values: {unknown}"
        )

    return (
        normalised.map(mapping)
        .astype(bool)
    )


def manifest_fingerprint(
    manifest,
    pipeline_version,
):
    """
    create the short fingerprint used to separate resumable score shards.

    the fingerprint depends on the pipeline version plus each selected
    relative_path/condition pair. this prevents a cached score shard from a
    different evaluation manifest being reused accidentally.
    """
    fingerprint_input = (
        pipeline_version
        + "\n"
        + "\n".join(
            manifest["relative_path"].astype(str)
            + "|"
            + manifest["condition"].astype(str)
        )
    )

    return hashlib.md5(
        fingerprint_input.encode("utf-8")
    ).hexdigest()[:12]
