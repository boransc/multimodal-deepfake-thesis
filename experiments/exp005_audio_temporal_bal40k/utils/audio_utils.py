# audio decoding / feature helpers used by exp005

import gc
import json
import subprocess
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm.auto import tqdm
from IPython.display import display

from .data_utils import (
    get_audio_segments,
    audio_window_labels_from_segments,
)


def probe_audio_stream(video_path, ffprobe_bin):
    """
    read basic information for the first audio stream using ffprobe.

    returns the first stream dictionary plus an error string. if probing fails or
    the file has no audio stream, the stream result is None.
    """
    video_path = Path(video_path)

    cmd = [
        ffprobe_bin,
        "-v", "error",
        "-select_streams", "a:0",
        "-show_entries", "stream=index,codec_name,sample_rate,channels,duration",
        "-of", "json",
        str(video_path),
    ]

    proc = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )

    if proc.returncode != 0:
        return (
            None,
            proc.stderr[-1000:]
            if proc.stderr
            else "ffprobe failed",
        )

    try:
        data = json.loads(proc.stdout)
        streams = data.get("streams", [])

        if not streams:
            return None, "no audio streams found"

        return streams[0], ""

    except Exception as e:
        return None, f"ffprobe json parse failed: {e}"


def raw_s16le_to_float32(raw_bytes, sample_rate, channels=1):
    """
    convert little-endian signed 16-bit pcm bytes into float32 mono audio.

    multi-channel audio is averaged to mono. values are scaled to roughly
    [-1, 1]. an empty byte stream returns one second of zeros plus decode_ok=False.
    """
    audio_i16 = np.frombuffer(
        raw_bytes,
        dtype=np.int16,
    )

    if len(audio_i16) == 0:
        return (
            np.zeros(sample_rate, dtype=np.float32),
            False,
            "empty raw pcm",
        )

    channels = (
        int(channels)
        if channels is not None
        else 1
    )

    channels = max(channels, 1)

    if channels > 1:
        usable = (
            len(audio_i16) // channels
        ) * channels

        audio_i16 = (
            audio_i16[:usable]
            .reshape(-1, channels)
            .mean(axis=1)
        )

    audio = (
        audio_i16.astype(np.float32)
        / 32768.0
    )

    audio = np.nan_to_num(
        audio,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )

    return audio.astype(np.float32), True, ""


def run_ffmpeg_to_raw_s16le(cmd, sample_rate, channels=1):
    """
    run an ffmpeg command that writes raw pcm to stdout and convert the bytes.

    this helper remained in the original notebook as part of the corrected
    decoding work even though the final decode_audio_ffmpeg path performs the
    stream-copy command directly.
    """
    proc = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )

    stderr_text = (
        proc.stderr.decode(
            "utf-8",
            errors="ignore",
        )[-1500:]
        if proc.stderr
        else ""
    )

    if (
        proc.returncode != 0
        or len(proc.stdout) == 0
    ):
        return (
            np.zeros(sample_rate, dtype=np.float32),
            False,
            stderr_text,
        )

    return raw_s16le_to_float32(
        proc.stdout,
        sample_rate=sample_rate,
        channels=channels,
    )


def decode_audio_ffmpeg(
    video_path,
    ffmpeg_bin,
    ffprobe_bin,
    sample_rate,
):
    """
    extract the MP4 audio stream and interpret the copied bytes as pcm_s16le.

    this is the corrected path added after the first exp005 attempt produced
    zero-valued audio. ffprobe is used to locate the first audio stream and its
    channel count, then ffmpeg stream-copies that stream to stdout as raw data.

    the raw int16 samples are converted to mono float32. a failed, empty or
    all-zero decode returns a zero array together with decode_ok=False so the
    preflight/shard code can stop instead of training on bad features.
    """
    video_path = Path(video_path)

    stream, probe_err = probe_audio_stream(
        video_path,
        ffprobe_bin,
    )

    if stream is None:
        return (
            np.zeros(sample_rate, dtype=np.float32),
            False,
            f"probe failed: {probe_err}",
        )

    stream_index = stream.get(
        "index",
        1,
    )

    channels = int(
        stream.get("channels", 1) or 1
    )

    cmd = [
        ffmpeg_bin,
        "-nostdin",
        "-hide_banner",
        "-loglevel", "error",
        "-copy_unknown",
        "-i", str(video_path),
        "-map", f"0:{stream_index}",
        "-c", "copy",
        "-f", "data",
        "pipe:1",
    ]

    proc = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )

    err = (
        proc.stderr.decode(
            "utf-8",
            errors="ignore",
        )[-2000:]
        if proc.stderr
        else ""
    )

    if (
        proc.returncode != 0
        or len(proc.stdout) == 0
    ):
        return (
            np.zeros(sample_rate, dtype=np.float32),
            False,
            err,
        )

    raw = np.frombuffer(
        proc.stdout,
        dtype=np.int16,
    )

    if len(raw) == 0:
        return (
            np.zeros(sample_rate, dtype=np.float32),
            False,
            "empty raw stream",
        )

    if channels > 1:
        usable = (
            len(raw) // channels
        ) * channels

        raw = (
            raw[:usable]
            .reshape(-1, channels)
            .mean(axis=1)
        )

    audio = (
        raw.astype(np.float32)
        / 32768.0
    )

    audio = np.nan_to_num(
        audio,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )

    if np.max(np.abs(audio)) == 0:
        return (
            np.zeros(sample_rate, dtype=np.float32),
            False,
            "raw stream extracted but all zeros",
        )

    return audio.astype(np.float32), True, ""


def run_audio_decode_preflight(
    df,
    split_name,
    condition_col,
    path_col,
    sample_rate,
    ffmpeg_bin,
    ffprobe_bin,
    seed,
    n_per_condition=2,
):
    """
    test decoding on a few examples from every manipulation condition.

    this is the guard added after the failed first audio run. the sample includes
    real, video-only, audio-only and both-fake conditions so decode behaviour is
    checked before the full feature cache is built.

    the returned dataframe stores stream metadata, decode success, duration and
    simple signal magnitude checks for each sampled file.
    """
    rows = []

    sample_df = (
        df.groupby(
            condition_col,
            group_keys=False,
        )
        .apply(
            lambda g: g.sample(
                min(len(g), n_per_condition),
                random_state=seed,
            )
        )
        .reset_index(drop=True)
    )

    for i, row in sample_df.iterrows():
        stream, probe_err = probe_audio_stream(
            row[path_col],
            ffprobe_bin,
        )

        audio, ok, err = decode_audio_ffmpeg(
            row[path_col],
            ffmpeg_bin=ffmpeg_bin,
            ffprobe_bin=ffprobe_bin,
            sample_rate=sample_rate,
        )

        duration = (
            len(audio) / sample_rate
            if ok
            else 0.0
        )

        rows.append({
            "split": split_name,
            "row_index": int(i),
            "condition": row[condition_col],
            "path": row[path_col],
            "probe_codec": stream.get("codec_name") if stream else None,
            "probe_sample_rate": stream.get("sample_rate") if stream else None,
            "probe_channels": stream.get("channels") if stream else None,
            "probe_stream_index": stream.get("index") if stream else None,
            "probe_error": probe_err,
            "decode_ok": int(ok),
            "num_samples": int(len(audio)),
            "duration_sec": float(duration),
            "abs_mean": float(np.mean(np.abs(audio))) if len(audio) else 0.0,
            "abs_max": float(np.max(np.abs(audio))) if len(audio) else 0.0,
            "error": err,
        })

    return pd.DataFrame(rows)


def pad_or_crop_window(audio, center_sample, window_samples):
    """
    take a fixed-size waveform window centred on one sampled time.

    samples outside the available clip are zero-padded. this keeps every temporal
    window the same length, including the windows close to the start/end of a clip.
    """
    half = window_samples // 2
    start = int(center_sample - half)
    end = start + window_samples

    out = np.zeros(
        window_samples,
        dtype=np.float32,
    )

    src_start = max(start, 0)
    src_end = min(end, len(audio))

    if src_end > src_start:
        dst_start = src_start - start
        dst_end = (
            dst_start
            + (src_end - src_start)
        )

        out[dst_start:dst_end] = (
            audio[src_start:src_end]
        )

    return out


def logstft_stats(
    window_audio,
    n_fft,
    hop_length,
    win_length,
):
    """
    create the 402-d log-STFT summary used for one audio window.

    torch.stft produces n_fft/2 + 1 frequency bins (201 when n_fft=400). after
    taking log(1 + magnitude), mean and standard deviation are calculated across
    time for every frequency bin.

    with the final settings:
        201 frequency means
        201 frequency standard deviations
        -> 402 features per temporal window
    """
    wav = torch.tensor(
        window_audio,
        dtype=torch.float32,
    )

    if wav.numel() < win_length:
        pad = (
            win_length
            - wav.numel()
        )

        wav = torch.nn.functional.pad(
            wav,
            (0, pad),
        )

    spec = torch.stft(
        wav,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        window=torch.hann_window(win_length),
        return_complex=True,
        center=True,
    )

    mag = torch.log1p(
        torch.abs(spec)
    )

    mean = mag.mean(dim=1)
    std = mag.std(dim=1)

    return torch.cat(
        [mean, std],
        dim=0,
    )


def extract_audio_temporal_features(
    audio,
    sample_rate,
    num_audio_windows,
    audio_window_seconds,
    n_fft,
    hop_length,
    win_length,
):
    """
    convert one decoded waveform into the 64 temporal feature vectors.

    window centres are spaced uniformly across the clip. every centre gets a
    fixed 0.96-second waveform window (with the final config), then logstft_stats
    converts that waveform into the 402-d summary vector.

    returns:
        features       [num_audio_windows, feature_dim]
        window centres [num_audio_windows] in seconds
        decoded clip duration in seconds
    """
    if len(audio) == 0:
        audio = np.zeros(
            sample_rate,
            dtype=np.float32,
        )

    duration_sec = max(
        len(audio) / sample_rate,
        1.0 / sample_rate,
    )

    window_samples = int(
        audio_window_seconds
        * sample_rate
    )

    window_samples = max(
        window_samples,
        win_length,
    )

    if num_audio_windows == 1:
        centres_sec = np.array(
            [duration_sec / 2],
            dtype=np.float32,
        )

    else:
        centres_sec = np.linspace(
            0,
            duration_sec,
            num_audio_windows,
            endpoint=False,
            dtype=np.float32,
        )

        centres_sec = (
            centres_sec
            + (duration_sec / num_audio_windows) / 2
        )

    features = []

    for c_sec in centres_sec:
        center_sample = int(
            c_sec * sample_rate
        )

        window_audio = pad_or_crop_window(
            audio,
            center_sample,
            window_samples,
        )

        feat = logstft_stats(
            window_audio,
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
        )

        features.append(feat)

    features = torch.stack(
        features,
        dim=0,
    )

    return (
        features,
        torch.tensor(
            centres_sec,
            dtype=torch.float32,
        ),
        duration_sec,
    )


class AudioTemporalFeatureDataset(Dataset):
    """
    dataset that creates audio features and temporal labels for one video.

    each item:
    1. decodes the MP4 audio
    2. creates 64 log-STFT feature vectors
    3. creates 64 modality-specific audio fake labels from audio_fake_segments

    audio-real conditions are always labelled 0 across all windows, even if their
    metadata contains unexpected segment values. audio-fake rows use the segment
    intervals after unit conversion.

    returns:
        features       [64, 402]
        window_labels  [64]
        window_centres [64]
        audio clip label
        dataframe row index
        decode_ok flag
        decoded duration
    """

    def __init__(
        self,
        csv_path,
        path_col,
        audio_segment_col,
        fallback_segment_cols,
        missing_audio_segment_policy,
        audio_sample_rate,
        num_audio_windows,
        audio_window_seconds,
        n_fft,
        hop_length,
        win_length,
        ffmpeg_bin,
        ffprobe_bin,
    ):
        self.df = (
            pd.read_csv(csv_path)
            .reset_index(drop=True)
        )

        self.path_col = path_col
        self.audio_segment_col = audio_segment_col
        self.fallback_segment_cols = fallback_segment_cols
        self.missing_audio_segment_policy = missing_audio_segment_policy

        self.audio_sample_rate = audio_sample_rate
        self.num_audio_windows = num_audio_windows
        self.audio_window_seconds = audio_window_seconds

        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length

        self.ffmpeg_bin = ffmpeg_bin
        self.ffprobe_bin = ffprobe_bin

    def __len__(self):
        return len(self.df)

    def _labels_for_row(
        self,
        row,
        window_centres,
        duration_sec,
    ):
        """create the 64 sampled-window labels for one row."""
        audio_fake = int(
            row.get("audio_fake", 0)
        )

        # real and video-only-fake conditions must stay audio-real
        if audio_fake == 0:
            return np.zeros(
                self.num_audio_windows,
                dtype=np.float32,
            )

        segments = get_audio_segments(
            row,
            self.audio_segment_col,
            self.fallback_segment_cols,
        )

        if len(segments) == 0:
            if self.missing_audio_segment_policy == "all_fake":
                return np.ones(
                    self.num_audio_windows,
                    dtype=np.float32,
                )

            return np.zeros(
                self.num_audio_windows,
                dtype=np.float32,
            )

        audio_frames = row.get(
            "audio_frames",
            None,
        )

        return audio_window_labels_from_segments(
            segments=segments,
            window_centres=window_centres,
            duration_sec=duration_sec,
            audio_sample_rate=self.audio_sample_rate,
            audio_frames=audio_frames,
        )

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        audio, decode_ok, decode_error = decode_audio_ffmpeg(
            row[self.path_col],
            ffmpeg_bin=self.ffmpeg_bin,
            ffprobe_bin=self.ffprobe_bin,
            sample_rate=self.audio_sample_rate,
        )

        features, window_centres, duration_sec = extract_audio_temporal_features(
            audio,
            sample_rate=self.audio_sample_rate,
            num_audio_windows=self.num_audio_windows,
            audio_window_seconds=self.audio_window_seconds,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
        )

        window_labels = self._labels_for_row(
            row=row,
            window_centres=window_centres.numpy(),
            duration_sec=duration_sec,
        )

        audio_video_label = int(
            row.get(
                "audio_fake",
                int(window_labels.max() > 0),
            )
        )

        return (
            features.float(),
            torch.tensor(
                window_labels,
                dtype=torch.float32,
            ),
            window_centres.float(),
            torch.tensor(
                audio_video_label,
                dtype=torch.long,
            ),
            torch.tensor(
                idx,
                dtype=torch.long,
            ),
            torch.tensor(
                int(decode_ok),
                dtype=torch.long,
            ),
            torch.tensor(
                float(duration_sec),
                dtype=torch.float32,
            ),
        )


def make_audio_feature_loader(
    csv_path,
    feature_batch_size,
    num_workers,
    pin_memory,
    prefetch_factor,
    path_col,
    audio_segment_col,
    fallback_segment_cols,
    missing_audio_segment_policy,
    audio_sample_rate,
    num_audio_windows,
    audio_window_seconds,
    n_fft,
    hop_length,
    win_length,
    ffmpeg_bin,
    ffprobe_bin,
    shuffle=False,
):
    """
    build the dataloader used for audio feature extraction.

    the loader keeps the original notebook's worker/prefetch behaviour. the
    dataset does the actual ffmpeg decoding, log-STFT feature calculation and
    temporal label construction.
    """
    dataset = AudioTemporalFeatureDataset(
        csv_path=csv_path,
        path_col=path_col,
        audio_segment_col=audio_segment_col,
        fallback_segment_cols=fallback_segment_cols,
        missing_audio_segment_policy=missing_audio_segment_policy,
        audio_sample_rate=audio_sample_rate,
        num_audio_windows=num_audio_windows,
        audio_window_seconds=audio_window_seconds,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        ffmpeg_bin=ffmpeg_bin,
        ffprobe_bin=ffprobe_bin,
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

    return dataset, DataLoader(
        **loader_kwargs
    )


def extract_audio_feature_shards(
    loader,
    split_name,
    output_root,
    num_audio_windows,
    audio_sample_rate,
    audio_window_seconds,
):
    """
    cache the audio temporal features/labels in smaller shard files.

    this includes the guard added after the failed first run: if the first batch
    has zero successful audio decodes, extraction stops immediately instead of
    writing an all-zero feature cache.

    features/labels/times are saved as float16 to reduce disk usage. each shard
    also keeps video labels, row indices, decode flags and durations.

    shard_manifest.csv records the paths, shapes, positive-window counts and
    decode success rate for the split.
    """
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
    start_time = time.time()

    cumulative_decode_ok = 0
    cumulative_videos = 0

    progress = tqdm(
        loader,
        desc=f"extract audio {split_name}",
        leave=True,
        mininterval=5,
    )

    for batch_idx, (
        features,
        window_labels,
        window_times,
        video_labels,
        indices,
        decode_ok,
        durations,
    ) in enumerate(progress):

        batch_decode_ok = int(
            decode_ok.sum().item()
        )

        cumulative_decode_ok += (
            batch_decode_ok
        )

        cumulative_videos += int(
            decode_ok.numel()
        )

        # stop before reproducing the old all-zero decode failure
        if (
            batch_idx == 0
            and batch_decode_ok == 0
        ):
            raise RuntimeError(
                f"{split_name}: first feature batch decoded "
                f"0/{int(decode_ok.numel())} videos successfully. "
                "Stopping to avoid caching all-zero audio features. "
                "Check audio_decode_preflight.csv and ffmpeg setup."
            )

        features_cpu = (
            features.cpu()
            .to(torch.float16)
        )

        window_labels_cpu = (
            window_labels.cpu()
            .to(torch.float16)
        )

        window_times_cpu = (
            window_times.cpu()
            .to(torch.float16)
        )

        shard_path = (
            split_dir
            / f"shard_{batch_idx:05d}.pt"
        )

        payload = {
            "features": features_cpu,
            "window_labels": window_labels_cpu,
            "window_times": window_times_cpu,
            "video_labels": video_labels.cpu(),
            "indices": indices.cpu(),
            "decode_ok": decode_ok.cpu(),
            "durations": durations.cpu(),
            "split": split_name,
            "feature_type": "logstft_mean_std_pcm_s16le_decode",
            "num_audio_windows": num_audio_windows,
            "sample_rate": audio_sample_rate,
            "audio_window_seconds": audio_window_seconds,
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
            "num_windows": int(features_cpu.shape[1]),
            "feature_dim": int(features_cpu.shape[2]),
            "positive_windows": float(
                window_labels_cpu.sum().item()
            ),
            "total_windows": int(
                window_labels_cpu.numel()
            ),
            "audio_fake_videos": int(
                video_labels.sum().item()
            ),
            "decode_ok_count": batch_decode_ok,
            "decode_ok_rate_batch": (
                batch_decode_ok
                / max(1, int(decode_ok.numel()))
            ),
            "duration_mean_sec": float(
                durations.float()
                .mean()
                .item()
            ),
            "size_mb": (
                shard_path.stat().st_size
                / 1024**2
            ),
        })

        if batch_idx % 10 == 0:
            done = sum(
                r["n_videos"]
                for r in rows
            )

            progress.set_postfix({
                "videos": done,
                "pos": int(
                    sum(
                        r["positive_windows"]
                        for r in rows
                    )
                ),
                "decode_ok_rate": (
                    f"{cumulative_decode_ok / max(1, cumulative_videos):.3f}"
                ),
            })

        del (
            features_cpu,
            window_labels_cpu,
            window_times_cpu,
            payload,
        )

        gc.collect()

    manifest_df = pd.DataFrame(rows)
    elapsed = (
        time.time()
        - start_time
    )

    manifest_df[
        "elapsed_total_sec"
    ] = elapsed

    overall_decode_ok_rate = (
        manifest_df["decode_ok_count"].sum()
        / max(
            1,
            manifest_df["n_videos"].sum(),
        )
    )

    manifest_df[
        "decode_ok_rate_overall"
    ] = overall_decode_ok_rate

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
        f"{split_name}: "
        f"decode_ok_rate={overall_decode_ok_rate:.4f}"
    )

    print(
        f"{split_name}: elapsed "
        f"{elapsed / 60:.2f} min"
    )

    print(
        "Manifest:",
        manifest_path,
    )

    if overall_decode_ok_rate < 0.95:
        print(
            "WARNING: decode success below 95%. "
            "Inspect shard manifest and audio_decode_preflight.csv."
        )

    return manifest_df
