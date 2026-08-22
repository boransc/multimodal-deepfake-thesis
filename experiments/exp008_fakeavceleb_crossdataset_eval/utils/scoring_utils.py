# external visual/audio scoring helpers used by exp008

import gc
import json
import shutil
import subprocess
from pathlib import Path

import cv2
cv2.setNumThreads(0)

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
from tqdm.auto import tqdm


class TemporalLinearHead(nn.Module):
    """
    same lightweight temporal head architecture used by exp004/exp005.

    exp008 does not retrain this class. it rebuilds the architecture so the
    already-trained visual and audio temporal checkpoints can be loaded.

    input:
        [B, T, feature_dim]

    output:
        [B, T] logits
    """

    def __init__(
        self,
        input_dim,
    ):
        super().__init__()

        self.net = nn.Sequential(
            nn.LayerNorm(
                input_dim
            ),
            nn.Dropout(0.0),
            nn.Linear(
                input_dim,
                1,
            ),
        )

    def forward(
        self,
        x,
    ):
        return self.net(
            x
        ).squeeze(-1)


class ConvNeXtFrameFeatureExtractor(nn.Module):
    """
    frozen-style convnext-base frame encoder used for external visual scoring.

    the final imagenet classifier projection is replaced with identity so each
    sampled frame gives a 1024-d representation. the 64 frame representations
    stay separate because the saved visual temporal head scores every position.

    input:
        [B, T, 3, H, W]

    output:
        [B, T, 1024]
    """

    def __init__(self):
        super().__init__()

        self.backbone = convnext_base(
            weights=(
                ConvNeXt_Base_Weights
                .IMAGENET1K_V1
            )
        )

        self.backbone.classifier[2] = (
            nn.Identity()
        )

    def forward(
        self,
        x,
    ):
        (
            batch,
            time,
            channels,
            height,
            width,
        ) = x.shape

        flat = x.reshape(
            batch * time,
            channels,
            height,
            width,
        )

        features = (
            self.backbone(
                flat
            )
        )

        return features.reshape(
            batch,
            time,
            -1,
        )


def pool_scores(
    probabilities,
):
    """
    turn one 64-position probability sequence into clip-level scores.

    the same pooling family used in exp007 is kept here:
        max
        mean
        top 5% mean
        top 10% mean
        top 20% mean

    with 64 positions, top10 uses ceil(6.4) = 7 scores.
    """
    probabilities = np.asarray(
        probabilities,
        dtype=np.float32,
    )

    if (
        probabilities.ndim != 1
        or len(probabilities) == 0
    ):
        raise ValueError(
            "Expected a non-empty one-dimensional probability array."
        )

    output = {
        "max": float(
            probabilities.max()
        ),
        "mean": float(
            probabilities.mean()
        ),
    }

    sorted_scores = np.sort(
        probabilities
    )[::-1]

    for fraction in [
        0.05,
        0.10,
        0.20,
    ]:
        k = max(
            1,
            int(
                np.ceil(
                    len(sorted_scores)
                    * fraction
                )
            ),
        )

        output[
            f"top{int(fraction * 100)}_mean"
        ] = float(
            sorted_scores[:k]
            .mean()
        )

    return output


def shard_is_complete(
    path,
    expected_indices,
):
    """
    check whether a resumable csv score shard contains the expected videos.

    only the external_index column needs to be read. a shard is reused only when
    its set of indices exactly matches the current chunk.
    """
    path = Path(path)

    if not path.exists():
        return False

    try:
        shard = pd.read_csv(
            path,
            usecols=["external_index"],
        )
    except Exception:
        return False

    return (
        set(
            shard[
                "external_index"
            ].astype(int)
        )
        == set(
            map(
                int,
                expected_indices,
            )
        )
    )


class ExternalVisualDataset(Dataset):
    """
    decode and sample FakeAVCeleb frames for the saved visual temporal model.

    each video is sampled uniformly at num_visual_frames positions. frames use
    the same imagenet preprocessing as the source ConvNeXt model.

    if a video cannot be decoded, a zero frame is returned and decode_ok=0. the
    evaluation later reports/guards decode coverage so failures are not silently
    treated as valid external examples.

    returns:
        sampled frame tensor
        external_index
        decode flag
        reported total frame count
    """

    def __init__(
        self,
        dataframe,
        num_visual_frames,
        image_size,
    ):
        self.dataframe = (
            dataframe.reset_index(
                drop=True
            )
        )

        self.num_visual_frames = (
            num_visual_frames
        )

        self.image_size = (
            image_size
        )

        self.transform = (
            transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize(
                    (
                        image_size,
                        image_size,
                    )
                ),
                transforms.ToTensor(),
                transforms.Normalize(
                    [
                        0.485,
                        0.456,
                        0.406,
                    ],
                    [
                        0.229,
                        0.224,
                        0.225,
                    ],
                ),
            ])
        )

    def __len__(
        self,
    ):
        return len(
            self.dataframe
        )

    def __getitem__(
        self,
        index,
    ):
        row = (
            self.dataframe
            .iloc[index]
        )

        capture = cv2.VideoCapture(
            str(row.path)
        )

        frames = []
        decode_ok = (
            capture.isOpened()
        )

        total_frames = 0

        if decode_ok:
            total_frames = int(
                capture.get(
                    cv2.CAP_PROP_FRAME_COUNT
                )
            )

            if total_frames > 0:
                targets = np.linspace(
                    0,
                    total_frames - 1,
                    self.num_visual_frames,
                ).astype(int)

                target_set = set(
                    targets.tolist()
                )

                frame_index = 0

                while True:
                    success, frame = (
                        capture.read()
                    )

                    if not success:
                        break

                    if frame_index in target_set:
                        frame = cv2.cvtColor(
                            frame,
                            cv2.COLOR_BGR2RGB,
                        )

                        frames.append(
                            self.transform(
                                frame
                            )
                        )

                        if (
                            len(frames)
                            == self.num_visual_frames
                        ):
                            break

                    frame_index += 1

        capture.release()

        if not frames:
            decode_ok = False

            frames = [
                torch.zeros(
                    3,
                    self.image_size,
                    self.image_size,
                )
            ]

        # keep a fixed 64-position tensor even if the final decode positions fail
        while (
            len(frames)
            < self.num_visual_frames
        ):
            frames.append(
                frames[-1].clone()
            )

        return (
            torch.stack(frames),
            int(
                row.external_index
            ),
            int(
                decode_ok
            ),
            int(
                total_frames
            ),
        )


@torch.no_grad()
def extract_visual_scores(
    dataframe,
    visual_encoder,
    visual_head,
    device,
    score_shards_dir,
    results_dir,
    manifest_hash,
    num_visual_frames,
    image_size,
    visual_batch_size,
    num_workers,
    prefetch_factor,
    pin_memory,
    chunk_size,
):
    """
    generate/reuse visual temporal scores for the external evaluation manifest.

    the manifest is processed in chunks so a long cross-dataset run can resume.
    every chunk writes one csv under score_shards/<manifest_hash>/visual. an
    existing chunk is reused only when its external_index set exactly matches.

    the source visual encoder/head are used without retraining. the 64 temporal
    probabilities are pooled into max/mean/top-k clip scores for later external
    evaluation.

    returns:
        dataframe with one row per external video and its visual pooled scores
    """
    output_dir = (
        Path(score_shards_dir)
        / manifest_hash
        / "visual"
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    for start in range(
        0,
        len(dataframe),
        chunk_size,
    ):
        chunk = (
            dataframe.iloc[
                start:start + chunk_size
            ]
            .copy()
            .reset_index(drop=True)
        )

        part_path = (
            output_dir
            / f"part_{start:06d}.csv"
        )

        expected_indices = (
            chunk["external_index"]
            .astype(int)
            .tolist()
        )

        if shard_is_complete(
            part_path,
            expected_indices,
        ):
            continue

        if part_path.exists():
            part_path.unlink()

        loader = DataLoader(
            ExternalVisualDataset(
                chunk,
                num_visual_frames=num_visual_frames,
                image_size=image_size,
            ),
            batch_size=visual_batch_size,
            shuffle=False,
            num_workers=num_workers,
            prefetch_factor=(
                prefetch_factor
                if num_workers
                else None
            ),
            pin_memory=pin_memory,
        )

        rows = []

        for (
            frames,
            indices,
            decode_flags,
            total_frames,
        ) in tqdm(
            loader,
            desc=f"visual {start}",
            leave=False,
        ):
            frames = frames.to(
                device
            )

            probabilities = (
                torch.sigmoid(
                    visual_head(
                        visual_encoder(
                            frames
                        )
                    )
                )
                .cpu()
                .numpy()
            )

            for (
                item_index,
                external_index,
            ) in enumerate(
                indices.numpy()
            ):
                pooled = pool_scores(
                    probabilities[
                        item_index
                    ]
                )

                row = {
                    "external_index": int(
                        external_index
                    ),
                    "visual_decode_ok": int(
                        decode_flags[
                            item_index
                        ]
                    ),
                    "visual_frame_count": int(
                        total_frames[
                            item_index
                        ]
                    ),
                    "visual_positive_rate_05": float(
                        (
                            probabilities[
                                item_index
                            ]
                            >= 0.5
                        ).mean()
                    ),
                }

                row.update({
                    f"visual_{key}": value
                    for key, value in pooled.items()
                })

                rows.append(
                    row
                )

            del (
                frames,
                probabilities,
            )

            gc.collect()

        pd.DataFrame(
            rows
        ).to_csv(
            part_path,
            index=False,
        )

    part_files = sorted(
        output_dir.glob(
            "part_*.csv"
        )
    )

    if not part_files:
        raise RuntimeError(
            "No visual score shards were created."
        )

    result = (
        pd.concat(
            [
                pd.read_csv(path)
                for path in part_files
            ],
            ignore_index=True,
        )
        .drop_duplicates(
            "external_index"
        )
        .sort_values(
            "external_index"
        )
    )

    if (
        set(
            result["external_index"]
            .astype(int)
        )
        != set(
            dataframe[
                "external_index"
            ]
            .astype(int)
        )
    ):
        raise RuntimeError(
            "Visual score coverage does not match the evaluation manifest."
        )

    result.to_csv(
        Path(results_dir)
        / "fakeavceleb_visual_scores.csv",
        index=False,
    )

    return result


def probe_audio(
    path,
    ffprobe_bin,
):
    """
    inspect the first audio stream in one FakeAVCeleb video with ffprobe.

    returns:
        first audio stream dictionary, or None
        short error string
    """
    command = [
        ffprobe_bin,
        "-v",
        "error",
        "-select_streams",
        "a:0",
        "-show_entries",
        "stream=index,codec_name,sample_rate,channels",
        "-of",
        "json",
        str(path),
    ]

    process = subprocess.run(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    try:
        streams = json.loads(
            process.stdout
        ).get(
            "streams",
            [],
        )

    except json.JSONDecodeError as error:
        return (
            None,
            f"Invalid ffprobe JSON: {error}",
        )

    if not streams:
        return (
            None,
            "No audio stream",
        )

    return (
        streams[0],
        "",
    )


def decode_audio(
    path,
    ffmpeg_bin,
    ffprobe_bin,
    audio_sample_rate,
):
    """
    decode FakeAVCeleb audio to 16 khz mono signed-16-bit pcm.

    unlike the AV-Deepfake1M++ repair path, FakeAVCeleb contains ordinary
    compressed media audio. ffmpeg therefore decodes the stream explicitly to
    pcm_s16le rather than stream-copying compressed bytes and treating them as
    int16 samples.

    returns:
        float32 waveform
        decode_ok flag
        error string
        source codec name
    """
    stream, probe_error = probe_audio(
        path,
        ffprobe_bin=ffprobe_bin,
    )

    if stream is None:
        return (
            np.zeros(
                audio_sample_rate,
                dtype=np.float32,
            ),
            False,
            probe_error,
            None,
        )

    stream_index = int(
        stream.get(
            "index",
            1,
        )
    )

    codec_name = str(
        stream.get(
            "codec_name",
            "unknown",
        )
    )

    command = [
        ffmpeg_bin,
        "-nostdin",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(path),
        "-map",
        f"0:{stream_index}",
        "-vn",
        "-acodec",
        "pcm_s16le",
        "-ac",
        "1",
        "-ar",
        str(audio_sample_rate),
        "-f",
        "s16le",
        "pipe:1",
    ]

    process = subprocess.run(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    if (
        process.returncode != 0
        or not process.stdout
    ):
        error = (
            process.stderr.decode(
                errors="ignore"
            )[-2_500:]
        )

        return (
            np.zeros(
                audio_sample_rate,
                dtype=np.float32,
            ),
            False,
            error,
            codec_name,
        )

    waveform = (
        np.frombuffer(
            process.stdout,
            dtype=np.int16,
        )
        .astype(np.float32)
        / 32768.0
    )

    if (
        len(waveform) == 0
        or not np.isfinite(
            waveform
        ).all()
    ):
        return (
            np.zeros(
                audio_sample_rate,
                dtype=np.float32,
            ),
            False,
            "Empty or invalid waveform",
            codec_name,
        )

    return (
        waveform,
        True,
        "",
        codec_name,
    )


def pad_crop(
    waveform,
    centre,
    number_of_samples,
):
    """
    take a fixed-size waveform window around one centre sample.

    positions outside the decoded clip are zero padded so the windows near the
    beginning/end have the same length as all other windows.
    """
    start = int(
        centre
        - number_of_samples // 2
    )

    end = (
        start
        + number_of_samples
    )

    output = np.zeros(
        number_of_samples,
        dtype=np.float32,
    )

    source_start = max(
        0,
        start,
    )

    source_end = min(
        len(waveform),
        end,
    )

    if (
        source_end
        > source_start
    ):
        output[
            source_start - start:
            source_end - start
        ] = waveform[
            source_start:
            source_end
        ]

    return output


def logstft_stats(
    waveform,
    n_fft,
    hop_length,
    win_length,
):
    """
    convert one waveform window into the 402-d audio representation.

    with n_fft=400 there are 201 frequency bins. after log(1 + magnitude), mean
    and standard deviation are calculated across STFT time for every frequency
    bin:

        201 means + 201 standard deviations = 402 features
    """
    waveform_tensor = torch.tensor(
        waveform,
        dtype=torch.float32,
    )

    spectrum = torch.stft(
        waveform_tensor,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        window=torch.hann_window(
            win_length
        ),
        return_complex=True,
        center=True,
    )

    magnitude = torch.log1p(
        torch.abs(
            spectrum
        )
    )

    return torch.cat([
        magnitude.mean(
            dim=1
        ),
        magnitude.std(
            dim=1
        ),
    ])


def audio_features(
    waveform,
    audio_sample_rate,
    num_audio_windows,
    audio_window_seconds,
    n_fft,
    hop_length,
    win_length,
):
    """
    create the 64 x 402 temporal audio features for one external video.

    64 centre times are spaced uniformly across the decoded duration. a fixed
    0.96-second waveform region is taken around each centre and converted with
    the same log-STFT summary used by the source audio experiment.

    returns:
        [64, 402] feature tensor
        decoded audio duration in seconds
    """
    duration = max(
        len(waveform)
        / audio_sample_rate,
        1
        / audio_sample_rate,
    )

    centres = (
        np.linspace(
            0,
            duration,
            num_audio_windows,
            endpoint=False,
        )
        + (
            duration
            / num_audio_windows
        )
        / 2
    )

    window_samples = max(
        int(
            audio_window_seconds
            * audio_sample_rate
        ),
        win_length,
    )

    features = torch.stack([
        logstft_stats(
            pad_crop(
                waveform,
                int(
                    centre
                    * audio_sample_rate
                ),
                window_samples,
            ),
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
        )
        for centre in centres
    ])

    return (
        features,
        duration,
    )


class ExternalAudioDataset(Dataset):
    """
    decode one FakeAVCeleb video's audio and build its temporal features.

    every item uses the same 16 khz / 64-window / log-STFT feature design as the
    source audio model. the codec name and decode flag are also returned because
    codec/decode behaviour is important to the external evaluation audit.
    """

    def __init__(
        self,
        dataframe,
        ffmpeg_bin,
        ffprobe_bin,
        audio_sample_rate,
        num_audio_windows,
        audio_window_seconds,
        n_fft,
        hop_length,
        win_length,
    ):
        self.dataframe = (
            dataframe.reset_index(
                drop=True
            )
        )

        self.ffmpeg_bin = (
            ffmpeg_bin
        )

        self.ffprobe_bin = (
            ffprobe_bin
        )

        self.audio_sample_rate = (
            audio_sample_rate
        )

        self.num_audio_windows = (
            num_audio_windows
        )

        self.audio_window_seconds = (
            audio_window_seconds
        )

        self.n_fft = n_fft
        self.hop_length = (
            hop_length
        )
        self.win_length = (
            win_length
        )

    def __len__(
        self,
    ):
        return len(
            self.dataframe
        )

    def __getitem__(
        self,
        index,
    ):
        row = (
            self.dataframe
            .iloc[index]
        )

        (
            waveform,
            decode_ok,
            error,
            codec_name,
        ) = decode_audio(
            row.path,
            ffmpeg_bin=self.ffmpeg_bin,
            ffprobe_bin=self.ffprobe_bin,
            audio_sample_rate=self.audio_sample_rate,
        )

        (
            features,
            duration,
        ) = audio_features(
            waveform,
            audio_sample_rate=self.audio_sample_rate,
            num_audio_windows=self.num_audio_windows,
            audio_window_seconds=self.audio_window_seconds,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
        )

        return (
            features,
            int(
                row.external_index
            ),
            int(
                decode_ok
            ),
            float(
                duration
            ),
            str(
                codec_name
                or "unknown"
            ),
        )


@torch.no_grad()
def extract_audio_scores(
    dataframe,
    audio_head,
    device,
    score_shards_dir,
    results_dir,
    manifest_hash,
    ffmpeg_bin,
    ffprobe_bin,
    audio_sample_rate,
    num_audio_windows,
    audio_window_seconds,
    n_fft,
    hop_length,
    win_length,
    audio_batch_size,
    num_workers,
    prefetch_factor,
    chunk_size,
):
    """
    generate/reuse external audio scores in resumable csv chunks.

    FakeAVCeleb audio is decoded, converted into the 64 x 402 source-compatible
    features, passed through the saved audio temporal head and pooled to clip
    scores.

    each chunk is only reused when the external_index set exactly matches the
    current manifest chunk. the final combined score table must cover every
    external_index in the active evaluation manifest.
    """
    output_dir = (
        Path(score_shards_dir)
        / manifest_hash
        / "audio"
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    for start in range(
        0,
        len(dataframe),
        chunk_size,
    ):
        chunk = (
            dataframe.iloc[
                start:start + chunk_size
            ]
            .copy()
            .reset_index(drop=True)
        )

        part_path = (
            output_dir
            / f"part_{start:06d}.csv"
        )

        expected_indices = (
            chunk["external_index"]
            .astype(int)
            .tolist()
        )

        if shard_is_complete(
            part_path,
            expected_indices,
        ):
            continue

        if part_path.exists():
            part_path.unlink()

        loader = DataLoader(
            ExternalAudioDataset(
                chunk,
                ffmpeg_bin=ffmpeg_bin,
                ffprobe_bin=ffprobe_bin,
                audio_sample_rate=audio_sample_rate,
                num_audio_windows=num_audio_windows,
                audio_window_seconds=audio_window_seconds,
                n_fft=n_fft,
                hop_length=hop_length,
                win_length=win_length,
            ),
            batch_size=audio_batch_size,
            shuffle=False,
            num_workers=num_workers,
            prefetch_factor=(
                prefetch_factor
                if num_workers
                else None
            ),
            pin_memory=False,
        )

        rows = []

        for (
            features,
            indices,
            decode_flags,
            durations,
            codecs,
        ) in tqdm(
            loader,
            desc=f"audio {start}",
            leave=False,
        ):
            probabilities = (
                torch.sigmoid(
                    audio_head(
                        features.float()
                        .to(device)
                    )
                )
                .cpu()
                .numpy()
            )

            for (
                item_index,
                external_index,
            ) in enumerate(
                indices.numpy()
            ):
                pooled = pool_scores(
                    probabilities[
                        item_index
                    ]
                )

                row = {
                    "external_index": int(
                        external_index
                    ),
                    "audio_decode_ok": int(
                        decode_flags[
                            item_index
                        ]
                    ),
                    "audio_duration": float(
                        durations[
                            item_index
                        ]
                    ),
                    "audio_codec": str(
                        codecs[
                            item_index
                        ]
                    ),
                    "audio_positive_rate_05": float(
                        (
                            probabilities[
                                item_index
                            ]
                            >= 0.5
                        ).mean()
                    ),
                }

                row.update({
                    f"audio_{key}": value
                    for key, value in pooled.items()
                })

                rows.append(
                    row
                )

            del (
                features,
                probabilities,
            )

            gc.collect()

        pd.DataFrame(
            rows
        ).to_csv(
            part_path,
            index=False,
        )

    part_files = sorted(
        output_dir.glob(
            "part_*.csv"
        )
    )

    if not part_files:
        raise RuntimeError(
            "No audio score shards were created."
        )

    result = (
        pd.concat(
            [
                pd.read_csv(path)
                for path in part_files
            ],
            ignore_index=True,
        )
        .drop_duplicates(
            "external_index"
        )
        .sort_values(
            "external_index"
        )
    )

    if (
        set(
            result["external_index"]
            .astype(int)
        )
        != set(
            dataframe[
                "external_index"
            ]
            .astype(int)
        )
    ):
        raise RuntimeError(
            "Audio score coverage does not match the evaluation manifest."
        )

    result.to_csv(
        Path(results_dir)
        / "fakeavceleb_audio_scores.csv",
        index=False,
    )

    return result
