import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path

def plot_training_history(history_df, save_dir, run_name):
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # Loss curve
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

    # Main performance curve
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

    # Precision/recall curve
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

    # AUC curve, if available
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


"""
torch_gpu_allocated_gb = memory PyTorch tensors are actively using
torch_gpu_reserved_gb  = memory PyTorch has reserved/cached
nvml_gpu_used_gb       = what nvidia-smi sees as total used GPU memory
"""

def get_hardware_snapshot(device="cuda"):
    snapshot = {
        "timestamp": time.time(),
        "cpu_percent": psutil.cpu_percent(interval=None),
        "ram_used_gb": psutil.virtual_memory().used / 1024**3,
        "ram_total_gb": psutil.virtual_memory().total / 1024**3,
        "ram_percent": psutil.virtual_memory().percent,
    }

    if device == "cuda" and torch.cuda.is_available():
        snapshot["torch_gpu_allocated_gb"] = torch.cuda.memory_allocated() / 1024**3
        snapshot["torch_gpu_reserved_gb"] = torch.cuda.memory_reserved() / 1024**3
        snapshot["torch_gpu_max_allocated_gb"] = torch.cuda.max_memory_allocated() / 1024**3
    else:
        snapshot["torch_gpu_allocated_gb"] = None
        snapshot["torch_gpu_reserved_gb"] = None
        snapshot["torch_gpu_max_allocated_gb"] = None

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