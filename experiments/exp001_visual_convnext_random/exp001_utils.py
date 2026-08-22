# helper code used by exp001
# most of the longer reusable bits are kept here so the notebook is easier to follow

import time
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import psutil
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
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.models import ConvNeXt_Tiny_Weights, convnext_tiny
from tqdm.auto import tqdm

# opencv was only being used for frame reading, so i stopped it creating extra cpu threads
cv2.setNumThreads(0)

# pynvml was only used for the gpu monitoring during training
try:
    import pynvml

    pynvml.nvmlInit()
    NVML_AVAILABLE = True
    GPU_HANDLE = pynvml.nvmlDeviceGetHandleByIndex(0)
except Exception:
    NVML_AVAILABLE = False
    GPU_HANDLE = None


class VideoFrameDataset(Dataset):
    """
    dataset used for the visual baseline.

    each row in the csv points to one video and its binary real/fake label.
    when a sample is requested, the video is opened with opencv and a fixed
    number of frame positions are sampled across the full clip. the frames
    are then converted to rgb and normalised in the same way as the
    imagenet-pretrained convnext model expects.

    the returned frame tensor has shape:
        [num_frames, 3, image_size, image_size]

    the label is returned as a single integer:
        0 = real
        1 = fake
    """

    def __init__(self, csv_path, path_col, label_col, num_frames=16, image_size=224):
        self.df = pd.read_csv(csv_path)
        self.path_col = path_col
        self.label_col = label_col
        self.num_frames = num_frames

        # same imagenet preprocessing expected by the pretrained convnext weights
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

    def __len__(self):
        return len(self.df)

    def _sample_frames(self, video_path):
        """
        sample the fixed number of frames used to represent one video.

        the frame indices are spread from the start to the end of the clip
        using linspace. if an individual frame cannot be decoded it is skipped,
        and if this leaves too few frames the last valid frame is repeated so
        that every example still has the same tensor shape.
        """
        cap = cv2.VideoCapture(str(video_path))

        if not cap.isOpened():
            raise RuntimeError(f"Could not open video: {video_path}")

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        if total_frames <= 0:
            cap.release()
            raise RuntimeError(f"No frames found: {video_path}")

        indices = np.linspace(0, total_frames - 1, self.num_frames).astype(int)
        frames = []

        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
            ret, frame = cap.read()

            # if one position fails to decode i skip it, then pad later if needed
            if not ret:
                continue

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = self.transform(frame)
            frames.append(frame)

        cap.release()

        if len(frames) == 0:
            raise RuntimeError(f"No frames sampled: {video_path}")

        # keeps every sample at exactly num_frames so batches can be stacked
        while len(frames) < self.num_frames:
            frames.append(frames[-1])

        return torch.stack(frames)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        frames = self._sample_frames(row[self.path_col])
        label = int(row[self.label_col])

        return frames, torch.tensor(label, dtype=torch.long)


class FrameAverageConvNeXt(nn.Module):
    """
    clip-level visual classifier used in exp001.

    convnext-tiny processes the sampled frames independently. the batch and
    frame dimensions are temporarily flattened so the normal image model can
    process all frames, then the two-class logits are reshaped back into
    [batch, frames, classes].

    the frame logits are averaged across the sampled positions to give one
    real/fake prediction for the whole video. there is no separate temporal
    model here, just mean aggregation across frame predictions.
    """

    def __init__(self, num_classes=2):
        super().__init__()

        weights = ConvNeXt_Tiny_Weights.IMAGENET1K_V1
        self.backbone = convnext_tiny(weights=weights)

        # replace the original imagenet classifier with real/fake output
        in_features = self.backbone.classifier[2].in_features
        self.backbone.classifier[2] = nn.Linear(in_features, num_classes)

    def forward(self, x):
        # x starts as [batch, frames, channels, height, width]
        B, T, C, H, W = x.shape

        # flatten batch and time so convnext can process frames normally
        x = x.reshape(B * T, C, H, W)

        frame_logits = self.backbone(x)          # [B*T, 2]
        frame_logits = frame_logits.reshape(B, T, -1)

        # this is the temporal aggregation used in exp001
        video_logits = frame_logits.mean(dim=1)  # [B, 2]

        return video_logits


def get_hardware_snapshot(device="cuda"):
    """
    collect one hardware reading during the run.

    this was only for monitoring the jupyter/gpu environment while training.
    it records cpu and ram usage every time it is called, then adds pytorch gpu
    memory information when cuda is being used. if pynvml is available it also
    records gpu utilisation, memory use, temperature and power.

    returns a normal dictionary so the notebook can keep appending the readings
    and eventually save them as a csv.
    """
    snapshot = {
        "timestamp": time.time(),
        "cpu_percent": psutil.cpu_percent(interval=None),
        "ram_used_gb": psutil.virtual_memory().used / 1024**3,
        "ram_total_gb": psutil.virtual_memory().total / 1024**3,
        "ram_percent": psutil.virtual_memory().percent,
    }

    # pytorch memory stats only make sense when training on cuda
    if device == "cuda" and torch.cuda.is_available():
        snapshot["torch_gpu_allocated_gb"] = torch.cuda.memory_allocated() / 1024**3
        snapshot["torch_gpu_reserved_gb"] = torch.cuda.memory_reserved() / 1024**3
        snapshot["torch_gpu_max_allocated_gb"] = torch.cuda.max_memory_allocated() / 1024**3
    else:
        snapshot["torch_gpu_allocated_gb"] = None
        snapshot["torch_gpu_reserved_gb"] = None
        snapshot["torch_gpu_max_allocated_gb"] = None

    # nvml gives extra gpu usage / temperature / power information when available
    if NVML_AVAILABLE:
        mem = pynvml.nvmlDeviceGetMemoryInfo(GPU_HANDLE)
        util = pynvml.nvmlDeviceGetUtilizationRates(GPU_HANDLE)

        snapshot["nvml_gpu_used_gb"] = mem.used / 1024**3
        snapshot["nvml_gpu_total_gb"] = mem.total / 1024**3
        snapshot["nvml_gpu_util_percent"] = util.gpu
        snapshot["nvml_gpu_mem_util_percent"] = util.memory

        try:
            snapshot["gpu_temp_c"] = pynvml.nvmlDeviceGetTemperature(
                GPU_HANDLE,
                pynvml.NVML_TEMPERATURE_GPU
            )
        except Exception:
            snapshot["gpu_temp_c"] = None

        try:
            snapshot["gpu_power_w"] = pynvml.nvmlDeviceGetPowerUsage(GPU_HANDLE) / 1000
        except Exception:
            snapshot["gpu_power_w"] = None
    else:
        snapshot["nvml_gpu_used_gb"] = None
        snapshot["nvml_gpu_total_gb"] = None
        snapshot["nvml_gpu_util_percent"] = None
        snapshot["nvml_gpu_mem_util_percent"] = None
        snapshot["gpu_temp_c"] = None
        snapshot["gpu_power_w"] = None

    return snapshot


def train_one_epoch(
    model,
    loader,
    optimizer,
    criterion,
    scaler,
    device,
    epoch=None,
    total_epochs=None,
    run_name=None,
    results_dir=None,
    hardware_log_rows=None,
    log_every=25,
):
    """
    run one training epoch and return the main training metrics.

    for each batch this:
    - moves the sampled video frames and labels to the selected device
    - runs the model under mixed precision when cuda is available
    - calculates cross-entropy loss
    - backpropagates with GradScaler and updates the optimiser
    - keeps the predictions/labels so epoch-level classification metrics can
      be calculated at the end
    - periodically updates tqdm and records hardware usage

    the function returns a dictionary containing loss, accuracy, balanced
    accuracy, f1, precision, recall and the total epoch time. the notebook
    adds these values to the training history after each epoch.
    """
    model.train()

    total_loss = 0.0
    all_preds = []
    all_labels = []

    desc = f"Train {epoch}/{total_epochs}" if epoch is not None else "Train"

    progress_bar = tqdm(
        loader,
        desc=desc,
        total=len(loader),
        leave=True,
        dynamic_ncols=True,
        mininterval=1.0,
    )

    epoch_start_time = time.time()

    for batch_idx, (frames, labels) in enumerate(progress_bar):
        batch_start_time = time.time()

        frames = frames.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        # mixed precision reduces gpu memory use and speeds up the forward/backward pass
        with torch.amp.autocast("cuda", enabled=(device == "cuda")):
            logits = model(frames)
            loss = criterion(logits, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        batch_time = time.time() - batch_start_time

        total_loss += loss.item() * labels.size(0)

        preds = logits.argmax(dim=1)
        all_preds.extend(preds.detach().cpu().numpy())
        all_labels.extend(labels.detach().cpu().numpy())

        # no need to recalculate the progress metrics on every single batch
        if batch_idx % log_every == 0 or batch_idx == len(loader) - 1:
            running_loss = total_loss / max(1, len(all_labels))
            running_acc = accuracy_score(all_labels, all_preds)
            running_f1 = f1_score(all_labels, all_preds, zero_division=0)

            gpu_allocated = (
                torch.cuda.memory_allocated() / 1024**3
                if device == "cuda" else 0
            )

            progress_bar.set_postfix_str(
                f"loss={running_loss:.4f} | "
                f"acc={running_acc:.4f} | "
                f"f1={running_f1:.4f} | "
                f"gpu={gpu_allocated:.1f}GB | "
                f"batch={batch_time:.2f}s"
            )

        # save occasional hardware readings alongside the training log
        if hardware_log_rows is not None and batch_idx % log_every == 0:
            hw = get_hardware_snapshot(device=device)

            hw.update({
                "run_name": run_name,
                "phase": "train",
                "epoch": epoch,
                "batch_idx": batch_idx,
                "batch_time_sec": batch_time,
                "running_loss": total_loss / max(1, len(all_labels)),
            })

            hardware_log_rows.append(hw)

            if results_dir is not None and run_name is not None:
                pd.DataFrame(hardware_log_rows).to_csv(
                    Path(results_dir) / f"{run_name}_hardware_log.csv",
                    index=False
                )

    epoch_time = time.time() - epoch_start_time

    # return the epoch summary so the notebook can save it in the history csv
    return {
        "loss": total_loss / len(loader.dataset),
        "acc": accuracy_score(all_labels, all_preds),
        "balanced_acc": balanced_accuracy_score(all_labels, all_preds),
        "f1": f1_score(all_labels, all_preds, zero_division=0),
        "precision": precision_score(all_labels, all_preds, zero_division=0),
        "recall": recall_score(all_labels, all_preds, zero_division=0),
        "epoch_time_sec": epoch_time,
    }


@torch.no_grad()
def evaluate(
    model,
    loader,
    criterion,
    device,
    split_name="Val",
    return_outputs=False,
    run_name=None,
    results_dir=None,
    hardware_log_rows=None,
    epoch=None,
    log_every=25,
):
    """
    evaluate the model without updating any weights.

    this follows the same basic batch structure as training but gradients are
    disabled. for every video it stores:
    - the predicted class
    - the ground-truth class
    - the softmax probability for class 1 (fake)

    those are then used to calculate the main classification metrics. roc auc
    uses the fake probability rather than the hard class prediction.

    if return_outputs is true, the labels, predictions and probabilities are
    returned as well as the metric dictionary. this is what the final test
    evaluation uses for the confusion matrix, roc curve and condition analysis.
    """
    model.eval()

    total_loss = 0.0
    all_preds = []
    all_labels = []
    all_probs = []

    progress_bar = tqdm(
        loader,
        desc=split_name,
        total=len(loader),
        leave=True,
        dynamic_ncols=True,
        mininterval=1.0,
    )

    eval_start_time = time.time()

    for batch_idx, (frames, labels) in enumerate(progress_bar):
        batch_start_time = time.time()

        frames = frames.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        with torch.amp.autocast("cuda", enabled=(device == "cuda")):
            logits = model(frames)
            loss = criterion(logits, labels)

        batch_time = time.time() - batch_start_time

        # class 1 is the fake class, so this is the score later used for roc auc
        probs = torch.softmax(logits, dim=1)[:, 1]
        preds = logits.argmax(dim=1)

        total_loss += loss.item() * labels.size(0)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        all_probs.extend(probs.cpu().numpy())

        if batch_idx % log_every == 0 or batch_idx == len(loader) - 1:
            running_loss = total_loss / max(1, len(all_labels))
            running_acc = accuracy_score(all_labels, all_preds)
            running_f1 = f1_score(all_labels, all_preds, zero_division=0)

            progress_bar.set_postfix_str(
                f"loss={running_loss:.4f} | "
                f"acc={running_acc:.4f} | "
                f"f1={running_f1:.4f} | "
                f"batch={batch_time:.2f}s"
            )

        if hardware_log_rows is not None and batch_idx % log_every == 0:
            hw = get_hardware_snapshot(device=device)

            hw.update({
                "run_name": run_name,
                "phase": split_name,
                "epoch": epoch,
                "batch_idx": batch_idx,
                "batch_time_sec": batch_time,
                "running_loss": total_loss / max(1, len(all_labels)),
            })

            hardware_log_rows.append(hw)

            if results_dir is not None and run_name is not None:
                pd.DataFrame(hardware_log_rows).to_csv(
                    Path(results_dir) / f"{run_name}_hardware_log.csv",
                    index=False
                )

    eval_time = time.time() - eval_start_time

    metrics = {
        "loss": total_loss / len(loader.dataset),
        "acc": accuracy_score(all_labels, all_preds),
        "balanced_acc": balanced_accuracy_score(all_labels, all_preds),
        "f1": f1_score(all_labels, all_preds, zero_division=0),
        "precision": precision_score(all_labels, all_preds, zero_division=0),
        "recall": recall_score(all_labels, all_preds, zero_division=0),
        "eval_time_sec": eval_time,
    }

    # auc can fail if a split somehow contains only one class
    try:
        metrics["auc"] = roc_auc_score(all_labels, all_probs)
    except ValueError:
        metrics["auc"] = None

    # test evaluation asks for the raw outputs as well as the metric summary
    if return_outputs:
        return metrics, all_labels, all_preds, all_probs

    return metrics


def plot_training_history(history_df, save_dir, run_name):
    """
    plot the main training/validation curves saved during exp001.

    this makes separate figures for:
    - train/validation loss
    - train/validation f1 and balanced accuracy
    - validation precision and recall
    - validation auc, when auc is available

    each plot is both displayed in the notebook and written to the experiment
    results directory so it can be inspected later without rerunning training.
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # loss
    plt.figure(figsize=(8, 5))
    plt.plot(history_df["epoch"], history_df["train_loss"], marker="o", label="Train loss")
    plt.plot(history_df["epoch"], history_df["val_loss"], marker="o", label="Val loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"{run_name}: Loss")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_dir / f"{run_name}_loss.png", dpi=200)
    plt.show()

    # f1 + balanced accuracy
    plt.figure(figsize=(8, 5))
    plt.plot(history_df["epoch"], history_df["train_f1"], marker="o", label="Train F1")
    plt.plot(history_df["epoch"], history_df["val_f1"], marker="o", label="Val F1")
    plt.plot(history_df["epoch"], history_df["train_balanced_acc"], marker="o", label="Train balanced acc")
    plt.plot(history_df["epoch"], history_df["val_balanced_acc"], marker="o", label="Val balanced acc")
    plt.xlabel("Epoch")
    plt.ylabel("Score")
    plt.title(f"{run_name}: F1 and balanced accuracy")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_dir / f"{run_name}_f1_balanced_acc.png", dpi=200)
    plt.show()

    # precision + recall
    plt.figure(figsize=(8, 5))
    plt.plot(history_df["epoch"], history_df["val_precision"], marker="o", label="Val precision")
    plt.plot(history_df["epoch"], history_df["val_recall"], marker="o", label="Val recall")
    plt.xlabel("Epoch")
    plt.ylabel("Score")
    plt.title(f"{run_name}: Validation precision and recall")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_dir / f"{run_name}_precision_recall.png", dpi=200)
    plt.show()

    # only draw auc if the evaluation actually produced it
    if "val_auc" in history_df.columns and history_df["val_auc"].notna().any():
        plt.figure(figsize=(8, 5))
        plt.plot(history_df["epoch"], history_df["val_auc"], marker="o", label="Val AUC")
        plt.xlabel("Epoch")
        plt.ylabel("AUC")
        plt.title(f"{run_name}: Validation AUC")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(save_dir / f"{run_name}_auc.png", dpi=200)
        plt.show()
