# helper code used by exp003
# kept in one file because the visual feature / classifier helpers are all part of the same pipeline

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
    find the folder containing the exact saved exp003 split manifests.

    this does not create a new split. if a split directory is supplied manually
    it checks that first, otherwise it searches the known archived locations.

    exactly one folder has to be found. if none or more than one are found the
    function stops instead of guessing, since using a different split would mean
    this is no longer the same exp003 setup.
    """
    if split_manifest_dir is not None:
        directory = Path(split_manifest_dir)

        if not directory.is_dir():
            raise FileNotFoundError(
                f"Configured SPLIT_MANIFEST_DIR does not exist: {directory}"
            )

        return directory

    existing = [
        Path(directory)
        for directory in candidate_manifest_dirs
        if Path(directory).is_dir()
    ]

    if len(existing) == 1:
        return existing[0]

    if len(existing) == 0:
        raise FileNotFoundError(
            "Could not find the archived Exp003 manifest directory. "
            "Set SPLIT_MANIFEST_DIR manually to the folder containing the exact "
            "final train/val/test CSV files."
        )

    raise RuntimeError(
        "Multiple candidate manifest directories were found. Set "
        f"SPLIT_MANIFEST_DIR manually. Candidates: {existing}"
    )


def find_one_csv(directory, split):
    """
    find the saved csv for one train / val / test split.

    the search ignores files with names such as predictions, metrics, history or
    condition outputs so they cannot accidentally be treated as split manifests.

    it deliberately requires one exact match rather than silently picking the
    first csv it finds.
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
    turn each video in an exp003 split into the sampled visual frames.

    the dataframe already contains the exact archived split rows. for each row,
    opencv reads the video and frame positions are sampled uniformly across the
    complete clip.

    every selected frame is:
    - converted from bgr to rgb
    - resized to image_size
    - converted to a float tensor in [0, 1]
    - normalised using imagenet mean/std values

    exp003 uses 64 sampled frames when num_frames=64 is passed from the notebook.

    returns:
        frames: [num_frames, 3, image_size, image_size]
        label: binary real/fake label
        idx: row index inside the saved split dataframe
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
        """
        convert one opencv frame into the tensor format expected by convnext.

        this keeps the preprocessing used by the reconstructed exp003 notebook:
        rgb conversion, opencv resize, scaling to [0,1], then imagenet
        normalisation.
        """
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
        decode one video and collect the requested uniformly spaced positions.

        linspace gives the target frame indices from the start to the end of the
        video. the clip is read sequentially and only those positions are kept.

        if decoding gives fewer valid frames than requested, the last valid frame
        is repeated so every sample still has a fixed shape.
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

        # keep every item at exactly num_frames so a batch can be stacked
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
    frozen convnext-base feature extractor used for the exp003 visual clips.

    the original imagenet classification layer is replaced with identity, so
    each sampled frame gives a 1024-dimensional representation instead of an
    imagenet class prediction.

    the input shape is [batch, frames, channels, height, width]. batch and frame
    dimensions are flattened for convnext, restored afterwards, and then the 64
    frame representations are mean pooled to one 1024-d vector per video.

    all convnext parameters are frozen. only the small classifier trained later
    in the notebook is updated.
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
        features = self.backbone(flat)        # [B*T, 1024]
        features = features.reshape(b, t, -1) # [B, T, 1024]

        # one clip representation made by averaging the sampled frame features
        return features.mean(dim=1)           # [B, 1024]


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
    build the dataloader used only for frozen feature extraction.

    shuffling is disabled because the cached features should stay in the same
    order as the saved split manifest. the dataset also returns its row index so
    the cache can still be traced back to the original manifest row.
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
    extract and cache the frozen visual representation for one exp003 split.

    this is the expensive stage because each raw video has to be decoded and its
    sampled frames passed through convnext-base. once the cache exists, the
    classifier can train on the saved clip vectors without decoding videos again.

    if a cache exists and has the expected number of rows it is reused. otherwise
    the split is processed batch by batch. autocast is used for the convnext
    forward pass on cuda and the final saved feature matrix is moved back to
    float32 on cpu.

    saved payload:
        X: [number of clips, 1024]
        y: binary labels
        idx: original dataframe row indices
        num_frames: sampled frame count used for the clip
        feature_dim: size of each pooled clip representation
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
    classifier trained on top of the frozen 1024-d exp003 clip features.

    the head stays deliberately small:
        LayerNorm(1024)
        Linear(1024 -> 2)

    convnext is not updated here. this stage only learns how to separate real
    and fake using the cached visual representation.
    """

    def __init__(self, dim=1024):
        super().__init__()

        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, 2),
        )

    def forward(self, x):
        return self.net(x)


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    """
    evaluate the clip classifier without changing any model weights.

    for each clip this stores:
    - ground-truth label
    - predicted class
    - softmax probability for class 1 (fake)

    the probability is used for roc auc, while the hard predictions are used for
    accuracy, balanced accuracy, f1, precision and recall.

    returns the metric dictionary plus the raw labels/predictions/probabilities,
    which are later used for the condition analysis and plots.
    """
    model.eval()

    total_loss = 0.0
    y_true, y_pred, y_prob = [], [], []

    for X, y in loader:
        X = X.to(device)
        y = y.to(device)

        logits = model(X)
        loss = criterion(logits, y)

        # class 1 is the fake class
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
    make a small csv index of the files produced inside the exp003 run folder.

    this was only used for packaging/auditing the reconstruction. it records the
    path of every file relative to the run directory together with its size in mb,
    saves archive_index.csv and returns the dataframe so it can be displayed.
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
