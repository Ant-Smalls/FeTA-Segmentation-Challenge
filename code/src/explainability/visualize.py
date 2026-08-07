"""
Role 6 - Explainability & Clinical Presentation
==================================================

Turns model output into a clinically legible story: slice-level overlays of
ground truth vs. prediction (and, once Role 3 lands, uncertainty heatmaps),
tied to specific test-split cases and named clinical phenomena (eCSF/GM
boundary disagreement in particular -- see CONTEXT.md).

STATUS: scaffolding only. Roles 3 (comparative uncertainty model) and 5
(evaluation / case selection) haven't landed yet -- this currently runs
end-to-end against the Role 2 baseline checkpoint so the plotting path is
proven out ahead of time. See README.md for how to swap in the comparative
model and Role 5's case selection once they exist.

USAGE (from code/ directory, with venv active):
    python -m src.explainability.visualize \
        --checkpoint src/training/checkpoints/best_model.pt \
        --split test --case-index 0 --out-dir src/explainability/figures
"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # non-interactive backend, safe for headless/HPC runs
import matplotlib.pyplot as plt
import numpy as np
import torch

from src.preprocessing_data_preparation.dataset import FeTADataset
from src.training.train import NUM_CLASSES, sliding_window_inference

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


def load_model(checkpoint_path: Path, device: torch.device):
    """Loads the Role 2 baseline. Swap for
    `from src.models.comparative import UncertaintyUNet as Model` once
    Role 3 lands -- see README.md for the MODEL_RETURNS_UNCERTAINTY gotcha
    that comes with it."""
    from src.models.baseline import BaselineUNet

    model = BaselineUNet(in_channels=1, num_classes=NUM_CLASSES).to(device)
    state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    return model


@torch.no_grad()
def run_inference(model, image, patch_size, device):
    """Returns (pred_labels, uncertainty_or_None) for one full eval-mode
    volume. image: (1, D, H, W)."""
    logits = sliding_window_inference(model, image, patch_size, device)
    pred_labels = torch.argmax(logits, dim=1).squeeze(0).cpu().numpy()
    return pred_labels, None  # uncertainty slot filled in once Role 3 lands


def pick_informative_slice(label_volume: np.ndarray, classes=(1, 2)) -> int:
    """Picks the axial slice with the most combined voxels of the given
    classes (default eCSF/GM, the boundary this option cares about most) --
    a heuristic stand-in for Role 5's per-case Dice-driven case selection."""
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
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    code_root = Path(__file__).resolve().parents[2]
    split_path = code_root / "src/data/splits/split_v1.json"
    config_path = code_root / "src/data/config.yaml"

    eval_ds = FeTADataset(split_path, config_path, split_name=args.split, mode="eval")
    image, label, meta = eval_ds[args.case_index]

    model = load_model(Path(args.checkpoint), device)
    pred, uncertainty = run_inference(model, image, tuple(args.patch_size), device)

    plot_case(image, label, pred, meta["subject_id"], Path(args.out_dir), uncertainty=uncertainty)


if __name__ == "__main__":
    main()
