"""
Role 4 - Refinement Threshold Sweep
=====================================

Confirmed pipeline (per Role 3, see team thread):
    UncertaintyUNet -> MC Dropout (mean prediction + predictive entropy,
    both via mc_dropout_predict) -> refine_prediction(prediction,
    uncertainty, threshold)

The uncertainty_head trained via the NLL loss in train.py is NOT used
here -- it only shapes training. Refinement runs entirely on MC Dropout's
predictive entropy, which is already normalised to [0, 1], so threshold
values in that same range are meaningful.

This script:
  1. Loads a trained comparative checkpoint.
  2. Runs sliding-window inference over each VALIDATION volume, with MC
     Dropout active at each patch (multiple stochastic forward passes,
     averaged to a per-patch mean probability), stitched into a full-volume
     mean-probability map via the shared reconstruct_from_patches utility.
  3. Computes predictive entropy from the stitched full-volume mean
     probability (not stitched per-patch -- avoids entropy discontinuities
     at patch boundaries from a nonlinear function of a linearly-stitched
     quantity).
  4. Sweeps refine_prediction() over a list of thresholds, scoring each
     against validation Dice on eCSF / GM / dGM (the classes this
     mechanism targets) and the overall 7-class mean.
  5. Saves a results table + plot, and reports the threshold that
     maximises mean Dice on the three target classes.

VALIDATION ONLY -- do not run this against the test split. Per the
project's tuning rule, the refinement threshold must be selected without
looking at test data.

USAGE (from code/ directory, with venv active):
    python -m src.training.tune_refinement_threshold \
        --config src/training/train_config_comparative.yaml \
        --checkpoint src/training/checkpoints_comparative/best_model.pt \
        --thresholds 0.3,0.4,0.5,0.6,0.7 \
        --n-mc-samples 10
"""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import yaml

from src.preprocessing_data_preparation.dataset import FeTADataset, reconstruct_from_patches
from src.models.uncertainty_unet import UncertaintyUNet, refine_prediction
from src.training.train import sliding_window_coords

NUM_CLASSES = 8

# Class indices per CONTEXT.md: 1=eCSF, 2=GM, 3=WM, 4=Ventricles,
# 5=Cerebellum, 6=dGM, 7=Brainstem. These three are where refinement is
# meant to help most.
TARGET_CLASSES = {"eCSF": 1, "GM": 2, "dGM": 6}


@torch.no_grad()
def mc_dropout_sliding_window(model, image, patch_size, device, n_mc_samples=10):
    """Sliding-window inference with MC Dropout active per patch.

    Returns:
        mean_probability: (num_classes, D, H, W) -- stitched full volume
        prediction: (D, H, W) -- argmax of mean_probability
        uncertainty: (D, H, W) -- predictive entropy of mean_probability,
                                  normalised to [0, 1]
    """
    model.eval()
    for module in model.modules():
        if isinstance(module, torch.nn.Dropout3d):
            module.train()  # keep MC Dropout active during inference

    _, D, H, W = image.shape
    coords = sliding_window_coords((D, H, W), patch_size)

    prob_patches, patch_coords = [], []
    for dz, dy, dx in coords:
        raw = image[:, dz, dy, dx]
        pad = []
        for dim_size, p in zip(raw.shape[1:], patch_size):
            pad = [0, p - dim_size] + pad
        raw = F.pad(raw, pad)
        patch = raw.unsqueeze(0).to(device)  # (1, 1, pd, ph, pw)

        sample_probs = []
        for _ in range(n_mc_samples):
            logits, _ = model(patch)
            sample_probs.append(F.softmax(logits, dim=1))
        mean_prob_patch = torch.stack(sample_probs, dim=0).mean(dim=0).squeeze(0)  # (C, pd, ph, pw)

        orig_d = min(dz.stop, D) - dz.start
        orig_h = min(dy.stop, H) - dy.start
        orig_w = min(dx.stop, W) - dx.start
        mean_prob_patch = mean_prob_patch[:, :orig_d, :orig_h, :orig_w]

        prob_patches.append(mean_prob_patch.cpu())
        patch_coords.append((
            slice(dz.start, dz.start + orig_d),
            slice(dy.start, dy.start + orig_h),
            slice(dx.start, dx.start + orig_w),
        ))

    mean_probability = reconstruct_from_patches(
        prob_patches, patch_coords, full_volume_shape=(D, H, W), aggregation="gaussian"
    )  # (C, D, H, W)

    prediction = torch.argmax(mean_probability, dim=0)  # (D, H, W)

    entropy = -torch.sum(mean_probability * torch.log(mean_probability + 1e-8), dim=0)
    uncertainty = entropy / torch.log(torch.tensor(float(mean_probability.shape[0])))

    return mean_probability, prediction, uncertainty


def per_class_dice(prediction: torch.Tensor, target: torch.Tensor, class_idx: int, eps: float = 1e-5) -> float:
    pred_c = (prediction == class_idx).float()
    target_c = (target == class_idx).float()
    intersection = (pred_c * target_c).sum()
    union = pred_c.sum() + target_c.sum()
    if union == 0:
        return None  # class absent in this case
    return ((2.0 * intersection + eps) / (union + eps)).item()


def run_sweep(config: dict, checkpoint_path: str, thresholds: list, n_mc_samples: int):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    code_root = Path(config["code_root"])
    split_path = code_root / "src/data/splits" / config.get("split_filename", "split_v1.json")
    data_config_path = code_root / "src/data/config.yaml"
    patch_size = tuple(config.get("patch_size", [128, 128, 128]))

    val_ds = FeTADataset(split_path, data_config_path, split_name="val", mode="eval")

    model = UncertaintyUNet(in_channels=1, num_classes=NUM_CLASSES).to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    print(f"Loaded checkpoint: {checkpoint_path}")

    # Run MC Dropout inference once per case, reuse across all thresholds --
    # refinement is a cheap post-hoc rule, no need to re-run the model per
    # threshold value.
    print(f"Running MC Dropout sliding-window inference on {len(val_ds)} validation cases "
          f"({n_mc_samples} samples/patch)...")
    case_data = []
    for i in range(len(val_ds)):
        image, label, meta = val_ds[i]
        _, prediction, uncertainty = mc_dropout_sliding_window(
            model, image, patch_size, device, n_mc_samples
        )
        case_data.append((prediction, uncertainty, label, meta.get("subject_id", f"case_{i}")))
        print(f"  [{i + 1}/{len(val_ds)}] {meta.get('subject_id', '?')} done")

    results = []
    for threshold in thresholds:
        per_case_target_dice, per_case_overall_dice = [], []
        for prediction, uncertainty, label, subject_id in case_data:
            pred_batched = prediction.unsqueeze(0)
            unc_batched = uncertainty.unsqueeze(0)
            refined = refine_prediction(pred_batched, unc_batched, threshold=threshold, enabled=True).squeeze(0)

            target_dices = []
            for class_idx in TARGET_CLASSES.values():
                d = per_class_dice(refined, label, class_idx)
                if d is not None:
                    target_dices.append(d)
            if target_dices:
                per_case_target_dice.append(sum(target_dices) / len(target_dices))

            overall_dices = []
            for class_idx in range(1, NUM_CLASSES):
                d = per_class_dice(refined, label, class_idx)
                if d is not None:
                    overall_dices.append(d)
            if overall_dices:
                per_case_overall_dice.append(sum(overall_dices) / len(overall_dices))

        mean_target = sum(per_case_target_dice) / len(per_case_target_dice)
        mean_overall = sum(per_case_overall_dice) / len(per_case_overall_dice)
        results.append({"threshold": threshold, "target_dice_eCSF_GM_dGM": mean_target, "overall_dice": mean_overall})
        print(f"threshold={threshold:.2f} | target_dice(eCSF/GM/dGM)={mean_target:.4f} | overall_dice={mean_overall:.4f}")

    # No-refinement reference point (threshold effectively disables refinement).
    per_case_target_dice, per_case_overall_dice = [], []
    for prediction, uncertainty, label, subject_id in case_data:
        target_dices = [d for c in TARGET_CLASSES.values() if (d := per_class_dice(prediction, label, c)) is not None]
        overall_dices = [d for c in range(1, NUM_CLASSES) if (d := per_class_dice(prediction, label, c)) is not None]
        if target_dices:
            per_case_target_dice.append(sum(target_dices) / len(target_dices))
        if overall_dices:
            per_case_overall_dice.append(sum(overall_dices) / len(overall_dices))
    no_refine_target = sum(per_case_target_dice) / len(per_case_target_dice)
    no_refine_overall = sum(per_case_overall_dice) / len(per_case_overall_dice)
    print(f"no refinement (reference) | target_dice(eCSF/GM/dGM)={no_refine_target:.4f} | overall_dice={no_refine_overall:.4f}")

    best = max(results, key=lambda r: r["target_dice_eCSF_GM_dGM"])
    print(f"\nSelected threshold: {best['threshold']} "
          f"(target_dice={best['target_dice_eCSF_GM_dGM']:.4f} vs. no-refinement {no_refine_target:.4f})")

    return results, {"threshold": None, "target_dice_eCSF_GM_dGM": no_refine_target, "overall_dice": no_refine_overall}, best


def save_outputs(results, no_refine, best, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(out_dir / "refinement_threshold_sweep.json", "w") as f:
        json.dump({"results": results, "no_refinement_reference": no_refine, "selected": best}, f, indent=2)

    thresholds = [r["threshold"] for r in results]
    target_dice = [r["target_dice_eCSF_GM_dGM"] for r in results]
    overall_dice = [r["overall_dice"] for r in results]

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(thresholds, target_dice, marker="o", label="eCSF/GM/dGM mean Dice")
    ax.plot(thresholds, overall_dice, marker="o", label="Overall (7-class) mean Dice")
    ax.axhline(no_refine["target_dice_eCSF_GM_dGM"], linestyle="--", color="gray",
               label="No refinement (target classes)")
    ax.axvline(best["threshold"], linestyle=":", color="red", label=f"Selected threshold = {best['threshold']}")
    ax.set_xlabel("Refinement threshold")
    ax.set_ylabel("Mean Dice (validation)")
    ax.set_title("Refinement threshold sweep (validation set)")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "refinement_threshold_sweep.png", dpi=150)
    plt.close(fig)

    print(f"Saved sweep results to {out_dir / 'refinement_threshold_sweep.json'} "
          f"and {out_dir / 'refinement_threshold_sweep.png'}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to the comparative model's training config yaml")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to the trained comparative model checkpoint (.pt)")
    parser.add_argument("--thresholds", type=str, default="0.3,0.4,0.5,0.6,0.7",
                         help="Comma-separated list of thresholds to sweep")
    parser.add_argument("--n-mc-samples", type=int, default=10, help="MC Dropout forward passes per patch")
    parser.add_argument("--out-dir", type=str, default="src/training/tuning_results",
                         help="Where to save the sweep JSON/plot")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    thresholds = [float(t) for t in args.thresholds.split(",")]

    results, no_refine, best = run_sweep(config, args.checkpoint, thresholds, args.n_mc_samples)
    save_outputs(results, no_refine, best, Path(args.out_dir))


if __name__ == "__main__":
    main()