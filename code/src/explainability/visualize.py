"""
Explainability & Clinical Presentation
=======================================

Slice-level overlays of ground truth vs. prediction, plus uncertainty
heatmaps for the comparative model, focused on eCSF/GM boundary disagreement.

`--model-type comparative` runs UncertaintyUNet through MC-Dropout
sliding-window inference (see mc_dropout_sliding_window_inference) and
produces a 4-panel figure. `pick_informative_slice` is a heuristic slice
picker.

USAGE (from code/ directory, with venv active):
    python -m src.explainability.visualize \
        --checkpoint src/training/checkpoints/best_model.pt \
        --split test --case-index 0 --out-dir src/explainability/figures

    # Once a trained comparative checkpoint exists:
    python -m src.explainability.visualize \
        --checkpoint src/training/checkpoints/comparative_best.pt \
        --model-type comparative --refine \
        --split test --case-index 0 --out-dir src/explainability/figures
"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # non-interactive backend, safe for headless/HPC runs
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.preprocessing_data_preparation.dataset import FeTADataset, reconstruct_from_patches
from src.training.train import NUM_CLASSES, sliding_window_inference, sliding_window_coords

# Label indices per CONTEXT.md.
CLASS_NAMES = {
    0: "background",
    1: "eCSF",
    2: "GM",
    3: "WM",
    4: "Ventricles",
    5: "Cerebellum",
    6: "dGM",
    7: "Brainstem",
}


def load_model(checkpoint_path: Path, device: torch.device, model_type: str = "baseline"):
    """Loads BaselineUNet or UncertaintyUNet."""
    if model_type == "comparative":
        from src.models.uncertainty_unet import UncertaintyUNet as Model
    else:
        from src.models.baseline import BaselineUNet as Model

    model = Model(in_channels=1, num_classes=NUM_CLASSES).to(device)
    state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    return model


def mc_dropout_sliding_window_inference(model, image, patch_size, device, n_samples=10):
    """MC-Dropout counterpart to sliding_window_inference, for models
    returning (logits, uncertainty_head_output).

    Runs n dropout-enabled forward passes per patch, averages class
    probabilities, computes predictive entropy, and stitches both via
    reconstruct_from_patches.

    image: (1, D, H, W). Returns (pred_labels, entropy), both (D, H, W).
    """
    _, D, H, W = image.shape
    coords = sliding_window_coords((D, H, W), patch_size)

    model.eval()
    for module in model.modules():
        if isinstance(module, nn.Dropout3d):
            module.train()

    patches, patch_coords = [], []
    with torch.no_grad():
        for dz, dy, dx in coords:
            raw = image[:, dz, dy, dx]
            # Pad to full patch size if this slice hits the volume boundary.
            pad = []
            for dim_size, p in zip(raw.shape[1:], patch_size):
                pad = [0, p - dim_size] + pad
            raw = F.pad(raw, pad)
            patch = raw.unsqueeze(0).to(device)  # (1, 1, pd, ph, pw)

            probs = []
            for _ in range(n_samples):
                logits, _ = model(patch)
                probs.append(torch.softmax(logits, dim=1))
            mean_prob = torch.stack(probs, dim=0).mean(dim=0).squeeze(0)  # (C, pd, ph, pw)

            entropy = -torch.sum(mean_prob * torch.log(mean_prob + 1e-8), dim=0, keepdim=True)
            entropy = entropy / torch.log(
                torch.tensor(float(mean_prob.shape[0]), device=mean_prob.device)
            )

            combined = torch.cat([mean_prob, entropy], dim=0)  # (C+1, pd, ph, pw)

            # Crop back to the original (possibly smaller) slice size.
            orig_d = min(dz.stop, D) - dz.start
            orig_h = min(dy.stop, H) - dy.start
            orig_w = min(dx.stop, W) - dx.start
            combined = combined[:, :orig_d, :orig_h, :orig_w]

            patches.append(combined.cpu())
            patch_coords.append((
                slice(dz.start, dz.start + orig_d),
                slice(dy.start, dy.start + orig_h),
                slice(dx.start, dx.start + orig_w),
            ))

    stitched = reconstruct_from_patches(
        patches, patch_coords, full_volume_shape=(D, H, W), aggregation="gaussian"
    )
    mean_prob_full, entropy_full = stitched[:-1], stitched[-1]
    pred_labels = torch.argmax(mean_prob_full, dim=0)
    return pred_labels, entropy_full


@torch.no_grad()
def run_inference(
    model,
    image,
    patch_size,
    device,
    model_type: str = "baseline",
    mc_samples: int = 10,
    refine: bool = False,
    refine_threshold: float = 0.5,
):
    """Returns (pred_labels, uncertainty_or_None) for one full eval-mode
    volume. image: (1, D, H, W)."""
    if model_type == "comparative":
        pred_labels, uncertainty = mc_dropout_sliding_window_inference(
            model, image, patch_size, device, n_samples=mc_samples
        )
        if refine:
            from src.models.uncertainty_unet import refine_prediction

            pred_labels = refine_prediction(
                pred_labels.unsqueeze(0), uncertainty.unsqueeze(0), threshold=refine_threshold
            ).squeeze(0)
        return pred_labels.cpu().numpy(), uncertainty.cpu().numpy()

    logits = sliding_window_inference(model, image, patch_size, device)
    pred_labels = torch.argmax(logits, dim=1).squeeze(0).cpu().numpy()
    return pred_labels, None


def pick_informative_slice(label_volume: np.ndarray, classes=(1, 2)) -> int:
    """Picks the axial slice with the most combined voxels of the given
    classes (default eCSF/GM)."""
    mask = np.isin(label_volume, classes)
    counts = mask.sum(axis=(1, 2))
    return int(np.argmax(counts))


def overlay_slice(ax, image_slice: np.ndarray, label_slice: np.ndarray, title: str, alpha: float = 0.4):
    ax.imshow(image_slice, cmap="gray")
    masked = np.ma.masked_where(label_slice == 0, label_slice)
    ax.imshow(masked, cmap="tab10", alpha=alpha, vmin=0, vmax=9)
    ax.set_title(title, fontsize=10)
    ax.axis("off")


def plot_case(image, label, pred: np.ndarray, subject_id: str, out_dir: Path, uncertainty: np.ndarray | None = None) -> Path:
    """Saves a GT-vs-prediction (vs-uncertainty, if available) slice figure
    for one case, centered on its most eCSF/GM-heavy axial slice."""
    label_np = label.numpy() if torch.is_tensor(label) else label
    image_np = (image.squeeze(0).numpy() if torch.is_tensor(image) else image.squeeze(0))

    z = pick_informative_slice(label_np)
    n_panels = 3 if uncertainty is None else 4
    fig, axes = plt.subplots(1, n_panels, figsize=(5 * n_panels, 5))

    axes[0].imshow(image_np[z], cmap="gray")
    axes[0].set_title(f"{subject_id} - T2w (z={z})", fontsize=10)
    axes[0].axis("off")

    overlay_slice(axes[1], image_np[z], label_np[z], "Ground truth")
    overlay_slice(axes[2], image_np[z], pred[z], "Prediction")

    if uncertainty is not None:
        im = axes[3].imshow(uncertainty[z], cmap="magma")
        axes[3].set_title("Uncertainty", fontsize=10)
        axes[3].axis("off")
        fig.colorbar(im, ax=axes[3], fraction=0.046)

    fig.tight_layout()
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{subject_id}_slice{z}.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved {out_path}")
    return out_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--split", type=str, default="test", choices=["train", "val", "test"])
    parser.add_argument("--case-index", type=int, default=0)
    parser.add_argument("--out-dir", type=str, default="src/explainability/figures")
    parser.add_argument("--patch-size", type=int, nargs=3, default=[128, 128, 128])
    parser.add_argument(
        "--model-type", type=str, default="baseline", choices=["baseline", "comparative"]
    )
    parser.add_argument(
        "--mc-samples", type=int, default=10, help="MC-Dropout forward passes per patch (comparative model only)"
    )
    parser.add_argument(
        "--refine", action="store_true", help="Apply uncertainty-guided boundary refinement (comparative model only)"
    )
    parser.add_argument("--refine-threshold", type=float, default=0.5)
    args = parser.parse_args()

    # torch==2.2.2 doesn't support Conv3D on MPS, so 3D models like
    # BaselineUNet must run on CPU on Apple Silicon.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    code_root = Path(__file__).resolve().parents[2]
    split_path = code_root / "src/data/splits/split_v1.json"
    config_path = code_root / "src/data/config.yaml"

    eval_ds = FeTADataset(split_path, config_path, split_name=args.split, mode="eval")
    image, label, meta = eval_ds[args.case_index]

    model = load_model(Path(args.checkpoint), device, model_type=args.model_type)
    pred, uncertainty = run_inference(
        model,
        image,
        tuple(args.patch_size),
        device,
        model_type=args.model_type,
        mc_samples=args.mc_samples,
        refine=args.refine,
        refine_threshold=args.refine_threshold,
    )

    plot_case(image, label, pred, meta["subject_id"], Path(args.out_dir), uncertainty=uncertainty)


if __name__ == "__main__":
    main()
