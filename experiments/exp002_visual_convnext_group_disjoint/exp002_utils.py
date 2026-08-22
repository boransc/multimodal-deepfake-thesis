# helper code used by exp002
# kept together in one file because these bits are all specific to this experiment

from pathlib import Path

import cv2
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
from torch.utils.data import Dataset, DataLoader
from torchvision.models import convnext_base, ConvNeXt_Base_Weights
from tqdm.notebook import tqdm

cv2.setNumThreads(0)


def choose_manifest_dir(split_manifest_dir, candidate_manifest_dirs):
    """
    find the folder containing the exact saved exp002 split manifests.

    the final split is part of the experiment, so this does not generate a new
    train/val/test split. if split_manifest_dir is given manually it checks that
    folder first. otherwise it looks through the known candidate locations.

    exactly one candidate folder has to be found. if none or multiple are found
    the function stops instead of guessing which manifests should be used.
    """
    if split_manifest_dir is not None:
        directory = Path(split_manifest_dir)
        if not directory.is_dir():
            raise FileNotFoundError(
                f"Configured SPLIT_MANIFEST_DIR does not exist: {directory}"
            )
        return directory

    existing = [Path(d) for d in candidate_manifest_dirs if Path(d).is_dir()]

    if len(existing) == 1:
        return existing[0]

    if len(existing) == 0:
        raise FileNotFoundError(
            "Could not find the archived Exp002 manifest directory. "
            "Set SPLIT_MANIFEST_DIR manually to the folder containing the exact "
            "final train/val/test CSV files."
        )

    raise RuntimeError(
        "Multiple candidate manifest directories were found. Set "
        f"SPLIT_MANIFEST_DIR manually. Candidates: {existing}"
    )


def find_one_csv(directory, split):
    """
    find one csv for a requested split inside the saved manifest folder.

    filenames containing things such as predictions, metrics or training history
    are ignored so they cannot accidentally be treated as a split manifest.
    this also deliberately requires exactly one match rather than silently
    choosing between multiple possible files.
    """
    directory = Path(directory)

    patterns = {
        "train": ["*train*.csv"],
        "val": ["*val*.csv", "*validation*.csv"],
        "test": ["*test*.csv"],
    }[split]

    found = []
    for pattern in patterns:
        found.extend(directory.glob(pattern))

    found = sorted(set(
        path for path in found
        if all(
            token not in path.name.lower()
            for token in ["prediction", "metric", "history", "condition"]
        )
    ))

    if len(found) != 1:
        raise RuntimeError(
            f"Expected exactly one {split} split CSV under {directory}, "
            f"found {len(found)}: {[p.name for p in found]}"
        )

    return found[0]


class VideoFrameDataset(Dataset):
    """
    dataset used to turn each exp002 video into 16 sampled frames.

    the dataframe already contains the exact saved split rows. for each row the
    video is decoded with opencv and frame positions are spread uniformly from
    the beginning to the end of the clip.

    each decoded frame is:
    - converted from bgr to rgb
    - resized to the requested image size
    - converted to a float tensor in [0, 1]
    - normalised using imagenet mean/std values

    if fewer frames are decoded than requested, the last valid frame is repeated
    so every video still returns the same tensor shape.

    returns:
        frames: [num_frames, 3, image_size, image_size]
        label: binary real/fake label
        idx: row index from the saved split dataframe
    """

    def __init__(
        self,
        frame,
        num_frames,
        image_size,
        path_col="path",
        label_col="binary_label",
    ):
        self.df = frame.reset_index(drop=True)
        self.num_frames = int(num_frames)
        self.image_size = int(image_size)
        self.path_col = path_col
        self.label_col = label_col

        self.mean = torch.tensor(
            [0.485, 0.456, 0.406], dtype=torch.float32
        ).view(3, 1, 1)

        self.std = torch.tensor(
            [0.229, 0.224, 0.225], dtype=torch.float32
        ).view(3, 1, 1)

    def __len__(self):
        return len(self.df)

    def _preprocess(self, frame_bgr):
        """convert one decoded opencv frame into the normalised model input."""
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

        frame_rgb = cv2.resize(
            frame_rgb,
            (self.image_size, self.image_size),
            interpolation=cv2.INTER_AREA,
        )

        x = torch.from_numpy(frame_rgb).permute(2, 0, 1).float().div_(255.0)
        return (x - self.mean) / self.std

    def _sample_frames(self, path):
        """
        decode the video and collect the uniformly spaced positions used by exp002.

        the full clip is read sequentially and frames whose indices appear in the
        linspace sample are kept. this is the frame-sampling behaviour used by
        this reconstructed exp002 notebook, so it is kept unchanged here.
        """
        cap = cv2.VideoCapture(str(path))

        if not cap.isOpened():
            raise RuntimeError(f"Could not open video: {path}")

        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        if total <= 0:
            cap.release()
            raise RuntimeError(f"Video has no reported frames: {path}")

        wanted = np.linspace(
            0,
            total - 1,
            self.num_frames,
        ).astype(np.int64)

        wanted_set = set(wanted.tolist())

        frames = []
        i = 0

        while True:
            ok, frame = cap.read()
            if not ok:
                break

            if i in wanted_set:
                frames.append(self._preprocess(frame))

                if len(frames) == self.num_frames:
                    break

            i += 1

        cap.release()

        if not frames:
            raise RuntimeError(f"No frames decoded from: {path}")

        # pad with the final valid frame if decoding returned too few positions
        while len(frames) < self.num_frames:
            frames.append(frames[-1].clone())

        return torch.stack(frames, dim=0)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        frames = self._sample_frames(row[self.path_col])
        label = int(row[self.label_col])

        return (
            frames,
            torch.tensor(label, dtype=torch.long),
            torch.tensor(idx, dtype=torch.long),
        )


class ConvNeXtBaseClipFeatures(nn.Module):
    """
    frozen visual feature extractor used by exp002.

    imagenet-pretrained convnext-base normally ends with a classifier. the last
    classifier layer is replaced with identity so each frame instead produces a
    1024-dimensional representation.

    a video contains several sampled frames, so the batch and frame dimensions
    are flattened before convnext and restored afterwards. the 16 frame features
    are then mean-pooled to produce one 1024-d vector for the whole clip.

    all convnext parameters are frozen. the later layernorm/linear classifier is
    the only part trained for the exp002 binary classification task.
    """

    def __init__(self):
        super().__init__()

        weights = ConvNeXt_Base_Weights.IMAGENET1K_V1
        self.backbone = convnext_base(weights=weights)
        self.backbone.classifier[2] = nn.Identity()

        for parameter in self.backbone.parameters():
            parameter.requires_grad = False

    def forward(self, frames):
        # frames: [batch, sampled frames, channels, height, width]
        b, t, c, h, w = frames.shape

        flat = frames.reshape(b * t, c, h, w)
        features = self.backbone(flat)             # [B*T, 1024]
        features = features.reshape(b, t, -1)      # [B, T, 1024]

        # one clip representation from the sampled frame features
        return features.mean(dim=1)                # [B, 1024]


def make_feature_loader(
    frame,
    num_frames,
    image_size,
    feature_batch_size,
    num_workers,
    pin_memory,
    prefetch_factor,
    path_col="path",
    label_col="binary_label",
):
    """
    create the dataloader used for frozen convnext feature extraction.

    shuffling stays disabled because these features are just being cached in the
    same order as the saved split manifest. the original row index is also kept
    by VideoFrameDataset so the cache can be traced back to the manifest row.
    """
    dataset = VideoFrameDataset(
        frame=frame,
        num_frames=num_frames,
        image_size=image_size,
        path_col=path_col,
        label_col=label_col,
    )

    kwargs = dict(
        dataset=dataset,
        batch_size=feature_batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=False,
    )

    if num_workers > 0:
        kwargs["prefetch_factor"] = prefetch_factor
        kwargs["timeout"] = 240

    return DataLoader(**kwargs)


@torch.inference_mode()
def extract_split_features(
    split_name,
    frame,
    feature_extractor,
    feature_dir,
    device,
    num_frames,
    image_size,
    feature_dim,
    feature_batch_size,
    num_workers,
    pin_memory,
    prefetch_factor,
    path_col="path",
    label_col="binary_label",
    overwrite=False,
):
    """
    extract and cache one frozen convnext feature vector for every video.

    this is the expensive part of exp002 because it has to decode each video and
    run all sampled frames through convnext-base. once the split cache exists,
    classifier training can use the saved 1024-d vectors instead of repeatedly
    decoding the raw videos.

    if a cache already exists and has the expected number of rows it is reused.
    otherwise the split is processed batch by batch. mixed precision is used for
    the convnext forward pass on cuda, then the saved feature tensor is converted
    back to float32 on cpu.

    the saved dictionary contains:
        X: [number of clips, 1024] feature matrix
        y: binary labels
        idx: original dataframe row indices
        num_frames: number of sampled video frames
        feature_dim: size of the pooled visual representation
    """
    feature_dir = Path(feature_dir)
    cache_path = feature_dir / f"{split_name}_clip_features.pt"

    if cache_path.exists() and not overwrite:
        payload = torch.load(cache_path, map_location="cpu")

        if len(payload["y"]) == len(frame):
            print(f"Using existing {split_name} cache:", cache_path)
            return payload

        print("Existing cache row count does not match; regenerating.")

    loader = make_feature_loader(
        frame=frame,
        num_frames=num_frames,
        image_size=image_size,
        feature_batch_size=feature_batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        prefetch_factor=prefetch_factor,
        path_col=path_col,
        label_col=label_col,
    )

    xs, ys, idxs = [], [], []

    feature_extractor.eval()

    for frames, labels, indices in tqdm(loader, desc=f"Extract {split_name}"):
        frames = frames.to(device, non_blocking=False)

        with torch.autocast(
            device_type="cuda",
            dtype=torch.float16,
            enabled=(device.type == "cuda"),
        ):
            features = feature_extractor(frames)

        xs.append(features.float().cpu())
        ys.append(labels.cpu())
        idxs.append(indices.cpu())

    payload = {
        "X": torch.cat(xs, dim=0),
        "y": torch.cat(ys, dim=0),
        "idx": torch.cat(idxs, dim=0),
        "num_frames": num_frames,
        "feature_dim": feature_dim,
    }

    assert payload["X"].shape == (len(frame), feature_dim)
    assert len(payload["y"]) == len(frame)

    torch.save(payload, cache_path)
    print("Saved:", cache_path, tuple(payload["X"].shape))

    return payload


class ClipLinearClassifier(nn.Module):
    """
    small classifier trained on top of the frozen 1024-d clip features.

    the head is deliberately simple:
        LayerNorm(1024)
        Linear(1024 -> 2)

    the two outputs correspond to the real/fake classes. convnext itself is not
    updated during this training stage because its clip features were already
    extracted and cached.
    """

    def __init__(self, dim=1024):
        super().__init__()

        # keep the same attribute/layout used in the reconstructed notebook
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, 2),
        )

    def forward(self, x):
        return self.net(x)


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    """
    evaluate the clip classifier and return both metrics and raw predictions.

    no weights are updated here. for every clip the function stores the hard
    predicted class and the softmax probability of class 1 (fake). the fake
    probability is used for roc auc while the hard class is used for the other
    classification metrics.

    returns:
        metrics: loss, accuracy, balanced accuracy, f1, precision, recall, auc
        y_true: ground-truth labels
        y_pred: predicted binary classes
        y_prob: probability assigned to the fake class
    """
    model.eval()

    total_loss = 0.0
    y_true, y_pred, y_prob = [], [], []

    for X, y in loader:
        X = X.to(device)
        y = y.to(device)

        logits = model(X)
        loss = criterion(logits, y)

        # class 1 is fake
        probs = torch.softmax(logits, dim=1)[:, 1]
        preds = logits.argmax(dim=1)

        total_loss += loss.item() * len(y)

        y_true.extend(y.cpu().numpy())
        y_pred.extend(preds.cpu().numpy())
        y_prob.extend(probs.cpu().numpy())

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    y_prob = np.asarray(y_prob)

    metrics = {
        "loss": total_loss / len(loader.dataset),
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "auc": roc_auc_score(y_true, y_prob),
    }

    return metrics, y_true, y_pred, y_prob


def build_archive_index(run_dir):
    """
    make a small csv index of the files produced inside the exp002 run folder.

    this was used for packaging/auditing the reproduction run. it records each
    file path relative to the run directory together with its size in mb, saves
    the table as archive_index.csv, and returns the dataframe for display.
    """
    run_dir = Path(run_dir)

    rows = []

    for path in run_dir.rglob("*"):
        if path.is_file():
            rows.append({
                "relative_path": path.relative_to(run_dir).as_posix(),
                "size_mb": path.stat().st_size / 1024**2,
            })

    out = pd.DataFrame(rows).sort_values("relative_path")
    out.to_csv(run_dir / "archive_index.csv", index=False)

    return out
