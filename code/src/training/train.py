"""
Role 4 - Training Pipeline (v2)
=================================

Shared training loop: optimizer + Dice/CE loss + FULL-VOLUME validation
(sliding-window inference, stitched via reconstruct_from_patches) +
checkpointing + early stopping + reproducible seeding.

Runs against a DUMMY model by default so it works standalone. Roles 2/3
plug their real models in by implementing the same forward() contract --
see "SWAPPING IN A REAL MODEL" below. Nothing else in this file should
need to change for either the baseline or the comparative model.

USAGE (from code/ directory, with venv active):
    python -m src.training.train --config src/training/train_config.yaml

For a quick local sanity check against a handful of downloaded cases,
point --config at a config that references a trimmed local split file
(e.g. split_local_test.json) instead of the frozen split_v1.json.
"""

import argparse
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
import matplotlib
matplotlib.use("Agg")  # non-interactive backend, safe for headless/HPC runs
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

from src.preprocessing_data_preparation.dataset import FeTADataset, reconstruct_from_patches

NUM_CLASSES = 8  # 7 tissue classes + background (per CONTEXT.md / dataset contract)


# ---------------------------------------------------------------------------
# SWAPPING IN A REAL MODEL
# ---------------------------------------------------------------------------
# Replace DummyModel below with an import of the real model, e.g.:
#   from src.models.baseline import BaselineUNet as Model
#   from src.models.comparative import UncertaintyUNet as Model
#
# Required contract:
#   input:  (B, 1, D, H, W) float32
#   output (baseline):     (B, num_classes, D, H, W) logits
#   output (comparative):  (logits, uncertainty) tuple, both (B, ..., D, H, W)
#
# If the comparative model returns a tuple, set MODEL_RETURNS_UNCERTAINTY =
# True below -- compute_loss() and run_epoch() already branch on this flag.
# ---------------------------------------------------------------------------
class DummyModel(nn.Module):
    def __init__(self, num_classes: int = NUM_CLASSES):
        super().__init__()
        self.conv = nn.Conv3d(1, num_classes, kernel_size=3, padding=1)

    def forward(self, x):
        return self.conv(x)  # (B, num_classes, D, H, W) logits


MODEL_RETURNS_UNCERTAINTY = False  # flip to True once Role 3's model is wired in


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------------
# Loss: Dice + CE, as specified in the shared research plan.
# ---------------------------------------------------------------------------
def dice_loss(logits: torch.Tensor, target: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    probs = F.softmax(logits, dim=1)
    target_onehot = F.one_hot(target, num_classes=logits.shape[1]).permute(0, 4, 1, 2, 3).float()
    dims = (0, 2, 3, 4)
    intersection = torch.sum(probs * target_onehot, dims)
    cardinality = torch.sum(probs + target_onehot, dims)
    dice_per_class = (2.0 * intersection + eps) / (cardinality + eps)
    return 1.0 - dice_per_class.mean()


def compute_loss(
    output,
    target: torch.Tensor,
    uncertainty_loss_weight: float = 0.0,
    class_weights: torch.Tensor = None,
) -> torch.Tensor:
    if MODEL_RETURNS_UNCERTAINTY:
        logits, uncertainty = output
        ce = F.cross_entropy(logits, target, weight=class_weights)
        dsc = dice_loss(logits, target)
        # Placeholder calibration term (negative log-likelihood under predicted
        # variance). Role 3 owns the real formulation -- this is a stand-in so
        # the loop has somewhere to plug the weighted term in once it exists.
        calibration_term = uncertainty.mean() * 0.0
        return ce + dsc + uncertainty_loss_weight * calibration_term
    else:
        logits = output
        ce = F.cross_entropy(logits, target, weight=class_weights)
        dsc = dice_loss(logits, target)
        return ce + dsc


@torch.no_grad()
def mean_dice(logits: torch.Tensor, target: torch.Tensor, eps: float = 1e-5) -> float:
    preds = torch.argmax(logits, dim=1)
    dices = []
    for c in range(1, logits.shape[1]):  # skip background class 0
        pred_c = (preds == c).float()
        target_c = (target == c).float()
        intersection = (pred_c * target_c).sum()
        union = pred_c.sum() + target_c.sum()
        if union == 0:
            continue  # class absent in this case; skip rather than penalize
        dices.append(((2.0 * intersection + eps) / (union + eps)).item())
    return sum(dices) / len(dices) if dices else 0.0


# ---------------------------------------------------------------------------
# Sliding-window full-volume inference (for eval-mode validation/testing).
# Eval-mode volumes vary in size, so we tile at the training patch size and
# stitch back together via the shared reconstruct_from_patches utility, per
# the data_preprocessing_preparation contract -- this keeps baseline,
# comparative, and evaluation all reconstructing volumes the same way.
# ---------------------------------------------------------------------------
def sliding_window_coords(volume_shape, patch_size, overlap=0.5):
    stride = [max(1, int(p * (1 - overlap))) for p in patch_size]
    axis_starts = []
    for v, p, s in zip(volume_shape, patch_size, stride):
        starts = list(range(0, max(v - p, 0) + 1, s))
        if not starts or starts[-1] != v - p:
            starts.append(max(v - p, 0))
        axis_starts.append(starts)
    d_starts, h_starts, w_starts = axis_starts

    slices = []
    for d0 in d_starts:
        for h0 in h_starts:
            for w0 in w_starts:
                slices.append((
                    slice(d0, d0 + patch_size[0]),
                    slice(h0, h0 + patch_size[1]),
                    slice(w0, w0 + patch_size[2]),
                ))
    return slices


@torch.no_grad()
def sliding_window_inference(model, image, patch_size, device):
    """image: (1, D, H, W) float32 tensor (single case, no batch dim)."""
    _, D, H, W = image.shape
    coords = sliding_window_coords((D, H, W), patch_size)

    patches, patch_coords = [], []
    for dz, dy, dx in coords:
        patch = image[:, dz, dy, dx].unsqueeze(0).to(device)  # (1, 1, pd, ph, pw)
        output = model(patch)
        logits = output[0] if MODEL_RETURNS_UNCERTAINTY else output
        patches.append(logits.squeeze(0).cpu())  # (num_classes, pd, ph, pw)
        patch_coords.append((dz, dy, dx))

    full_logits = reconstruct_from_patches(
        patches, patch_coords, full_volume_shape=(D, H, W), aggregation="gaussian"
    )
    return full_logits.unsqueeze(0)  # (1, num_classes, D, H, W)


# ---------------------------------------------------------------------------
# Train / validate
# ---------------------------------------------------------------------------
def train_one_epoch(model, loader, optimizer, device, uncertainty_loss_weight, class_weights):
    model.train()
    total_loss, total_dice, n_batches = 0.0, 0.0, 0

    for image, label in loader:
        image, label = image.to(device), label.to(device)

        optimizer.zero_grad()
        output = model(image)
        loss = compute_loss(output, label, uncertainty_loss_weight, class_weights)
        loss.backward()
        optimizer.step()

        logits = output[0] if MODEL_RETURNS_UNCERTAINTY else output
        total_loss += loss.item()
        total_dice += mean_dice(logits, label)
        n_batches += 1

    return total_loss / n_batches, total_dice / n_batches


@torch.no_grad()
def validate_full_volume(model, eval_ds, device, patch_size, class_weights):
    """Full-volume validation via sliding-window inference, per case."""
    model.eval()
    total_loss, total_dice, n_cases = 0.0, 0.0, 0

    for i in range(len(eval_ds)):
        image, label, meta = eval_ds[i]
        label = label.unsqueeze(0).to(device)  # (1, D, H, W)

        logits = sliding_window_inference(model, image, patch_size, device).to(device)
        loss = F.cross_entropy(logits, label, weight=class_weights) + dice_loss(logits, label)

        total_loss += loss.item()
        total_dice += mean_dice(logits, label)
        n_cases += 1

    return total_loss / n_cases, total_dice / n_cases


def plot_training_history(history: dict, out_dir: Path, run_name: str = "run"):
    """Saves the raw history as JSON (for later re-plotting/comparison) and a
    loss/Dice curve PNG. history is a dict of equal-length lists, e.g.
    {"epoch": [...], "train_loss": [...], "val_loss": [...],
     "train_dice": [...], "val_dice": [...], "lr": [...]}.
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    import json
    with open(out_dir / f"{run_name}_history.json", "w") as f:
        json.dump(history, f, indent=2)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].plot(history["epoch"], history["train_loss"], label="train_loss")
    axes[0].plot(history["epoch"], history["val_loss"], label="val_loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss (Dice + CE)")
    axes[0].set_title(f"{run_name}: Loss")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    axes[1].plot(history["epoch"], history["train_dice"], label="train_dice")
    axes[1].plot(history["epoch"], history["val_dice"], label="val_dice")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Mean Dice")
    axes[1].set_title(f"{run_name}: Dice")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_dir / f"{run_name}_history.png", dpi=150)
    plt.close(fig)
    print(f"Saved training history to {out_dir / (run_name + '_history.png')}")


def train(config: dict):
    set_seed(config.get("seed", 42))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    code_root = Path(config["code_root"])
    split_filename = config.get("split_filename", "split_v1.json")
    split_path = code_root / "src/data/splits" / split_filename
    data_config_path = code_root / "src/data/config.yaml"

    train_ds = FeTADataset(split_path, data_config_path, split_name="train", mode="train", seed=0)
    val_ds_eval = FeTADataset(split_path, data_config_path, split_name="val", mode="eval")

    train_loader = DataLoader(
        train_ds,
        batch_size=config["batch_size"],
        shuffle=True,
        num_workers=config.get("num_workers", 0),
    )

    patch_size = tuple(config.get("patch_size", [128, 128, 128]))

    from src.models.baseline import BaselineUNet
    model = BaselineUNet(in_channels=1, num_classes=NUM_CLASSES).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"])
    uncertainty_loss_weight = config.get("uncertainty_loss_weight", 0.0)

    # Class weighting -- off by default (class_weights: null in config), since
    # the dummy model doesn't need it. Real models almost certainly will: the
    # 7 tissue classes are heavily imbalanced (see README.md volume stats),
    # so an unweighted loss risks the model collapsing to mostly-WM/background
    # predictions. Provide 8 floats (background + 7 tissues) in the config to
    # turn this on, in the same class-index order as the dataset labels.
    class_weights_cfg = config.get("class_weights")
    class_weights = (
        torch.tensor(class_weights_cfg, dtype=torch.float32, device=device)
        if class_weights_cfg is not None
        else None
    )
    if class_weights is not None:
        print(f"Using class weights: {class_weights_cfg}")

    # LR scheduler -- off by default (lr_scheduler: null / absent in config).
    # ReduceLROnPlateau watches validation Dice (mode="max") and cuts the LR
    # when it stalls, which tends to help once real models are training for
    # longer and plateau naturally partway through.
    scheduler = None
    if config.get("lr_scheduler", False):
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="max",
            factor=config.get("lr_scheduler_factor", 0.5),
            patience=config.get("lr_scheduler_patience", 5),
        )

    ckpt_dir = Path(config["checkpoint_dir"])
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    best_val_dice = -1.0
    epochs_no_improve = 0
    history = {"epoch": [], "train_loss": [], "val_loss": [], "train_dice": [], "val_dice": [], "lr": []}

    for epoch in range(1, config["max_epochs"] + 1):
        start = time.time()
        train_loss, train_dice = train_one_epoch(
            model, train_loader, optimizer, device, uncertainty_loss_weight, class_weights
        )
        val_loss, val_dice = validate_full_volume(model, val_ds_eval, device, patch_size, class_weights)
        elapsed = time.time() - start

        current_lr = optimizer.param_groups[0]["lr"]
        print(
            f"Epoch {epoch:03d} | "
            f"train_loss={train_loss:.4f} train_dice={train_dice:.4f} | "
            f"val_loss={val_loss:.4f} val_dice={val_dice:.4f} | "
            f"lr={current_lr:.2e} | "
            f"{elapsed:.1f}s"
        )

        history["epoch"].append(epoch)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_dice"].append(train_dice)
        history["val_dice"].append(val_dice)
        history["lr"].append(current_lr)

        if scheduler is not None:
            prev_lr = current_lr
            scheduler.step(val_dice)
            new_lr = optimizer.param_groups[0]["lr"]
            if new_lr != prev_lr:
                print(f"  -> LR reduced: {prev_lr:.2e} -> {new_lr:.2e}")

        if val_dice > best_val_dice:
            best_val_dice = val_dice
            epochs_no_improve = 0
            torch.save(model.state_dict(), ckpt_dir / "best_model.pt")
            print(f"  -> new best val_dice={val_dice:.4f}, checkpoint saved")
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= config["early_stopping_patience"]:
            print(f"Early stopping at epoch {epoch} (no improvement for {config['early_stopping_patience']} epochs)")
            break

    print(f"Training complete. Best val_dice={best_val_dice:.4f}")

    run_name = config.get("run_name", "run")
    plot_training_history(history, ckpt_dir, run_name=run_name)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to training config yaml")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    train(config)


if __name__ == "__main__":
    main()