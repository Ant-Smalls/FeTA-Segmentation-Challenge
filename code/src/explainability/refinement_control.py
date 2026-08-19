"""
Separates the two components of uncertainty-guided refinement.

Refinement replaces high-uncertainty boundary voxels with the majority class of their
3x3x3 neighbourhood, so a gain can come either from the uncertainty gate selecting the
right voxels or from the smoothing filter alone. Four conditions isolate the two:

  baseline              - baseline prediction as-is
  baseline_smoothed     - the same majority filter over every boundary voxel, ungated
  comparative           - comparative prediction as-is
  comparative_refined   - the majority filter gated on MC-Dropout uncertainty

An equal improvement in both arms attributes the gain to the smoothing.

Metrics span Dice, mean surface distance, HD95, HD100 and connected-component count,
because removing small isolated components barely moves Dice or HD95 while moving
HD100 and component count considerably.

USAGE (from code/ directory, with venv active):
    python -m src.explainability.refinement_control \
        --baseline-checkpoint src/training/checkpoints/best_model.pt \
        --comparative-checkpoint src/training/checkpoints_comparative/best_model_comp.pt \
        --split test --out-dir src/explainability/analysis
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from scipy.ndimage import distance_transform_edt, label as connected_components

from src.models.uncertainty_unet import refine_prediction
from src.preprocessing_data_preparation.dataset import FeTADataset
from src.explainability.visualize import CLASS_NAMES, load_model, run_inference
from src.training.train import NUM_CLASSES, sliding_window_inference


METRICS = ("mean_dice", "mean_surface_distance_mm", "hd95_mm", "hd100_mm", "n_components")


def surface_mask(binary: np.ndarray) -> np.ndarray:
    """Voxels inside `binary` that touch a voxel outside it (6-connectivity)."""
    surface = np.zeros(binary.shape, dtype=bool)
    for axis in range(3):
        differs = np.diff(binary.astype(np.int8), axis=axis) != 0
        lo = [slice(None)] * 3
        hi = [slice(None)] * 3
        lo[axis] = slice(None, -1)
        hi[axis] = slice(1, None)
        surface[tuple(lo)] |= differs
        surface[tuple(hi)] |= differs
    return surface & binary


def surface_distances(pred_binary: np.ndarray, gt_binary: np.ndarray, spacing) -> np.ndarray:
    """Symmetric set of surface-to-surface distances in mm."""
    pred_surface = surface_mask(pred_binary)
    gt_surface = surface_mask(gt_binary)
    if not pred_surface.any() or not gt_surface.any():
        return np.array([])
    to_gt = distance_transform_edt(~gt_surface, sampling=spacing)[pred_surface]
    to_pred = distance_transform_edt(~pred_surface, sampling=spacing)[gt_surface]
    return np.concatenate([to_gt, to_pred])


def score_prediction(pred: np.ndarray, label: np.ndarray, spacing) -> dict:
    """Per-class Dice, surface distances and component count."""
    per_class = {}
    for class_index in range(1, NUM_CLASSES):
        pred_binary = pred == class_index
        gt_binary = label == class_index
        if not gt_binary.any():
            continue

        intersection = float(np.logical_and(pred_binary, gt_binary).sum())
        denominator = float(pred_binary.sum() + gt_binary.sum())
        dice = 2.0 * intersection / denominator if denominator else float("nan")

        distances = surface_distances(pred_binary, gt_binary, spacing)
        # HD95 discards the top 5% of distances, so a few stray voxels never reach it;
        # HD100 and the component count are what respond to isolated islands.
        per_class[CLASS_NAMES[class_index]] = {
            "dice": dice,
            "mean_surface_distance_mm": float(distances.mean()) if distances.size else float("nan"),
            "hd95_mm": float(np.percentile(distances, 95)) if distances.size else float("nan"),
            "hd100_mm": float(distances.max()) if distances.size else float("nan"),
            "n_components": int(connected_components(pred_binary)[1]),
        }

    def average(metric):
        values = [v[metric] for v in per_class.values() if not np.isnan(v[metric])]
        return float(np.mean(values)) if values else float("nan")

    return {
        "per_class": per_class,
        "mean_dice": average("dice"),
        "mean_surface_distance_mm": average("mean_surface_distance_mm"),
        "hd95_mm": average("hd95_mm"),
        "hd100_mm": average("hd100_mm"),
        "n_components": average("n_components"),
    }


def majority_smooth(pred: np.ndarray) -> np.ndarray:
    """The refinement filter applied to every boundary voxel.

    An all-ones uncertainty map passes refine_prediction's threshold test everywhere,
    leaving only the smoothing component.
    """
    prediction = torch.from_numpy(pred).long().unsqueeze(0)
    uncertainty = torch.ones_like(prediction, dtype=torch.float32)
    return refine_prediction(prediction, uncertainty, threshold=0.5).squeeze(0).numpy()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-checkpoint", type=str, required=True)
    parser.add_argument("--comparative-checkpoint", type=str, required=True)
    parser.add_argument("--split", type=str, default="test", choices=["train", "val", "test"])
    parser.add_argument("--out-dir", type=str, default="src/explainability/analysis")
    parser.add_argument("--patch-size", type=int, nargs=3, default=[128, 128, 128])
    parser.add_argument("--mc-samples", type=int, default=10)
    parser.add_argument("--refine-threshold", type=float, default=0.5)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    patch_size = tuple(args.patch_size)

    code_root = Path(__file__).resolve().parents[2]
    eval_ds = FeTADataset(
        code_root / "src/data/splits/split_v1.json",
        code_root / "src/data/config.yaml",
        split_name=args.split,
        mode="eval",
    )
    n_cases = len(eval_ds) if args.limit is None else min(args.limit, len(eval_ds))

    baseline_model = load_model(Path(args.baseline_checkpoint), device, model_type="baseline")
    comparative_model = load_model(Path(args.comparative_checkpoint), device, model_type="comparative")

    conditions = ["baseline", "baseline_smoothed", "comparative", "comparative_refined"]
    per_case = []

    for index in range(n_cases):
        image, label, meta = eval_ds[index]
        label_np = label.numpy()
        spacing = tuple(meta["spacing"])
        print(f"[{index + 1}/{n_cases}] {meta['subject_id']}", flush=True)

        with torch.no_grad():
            logits = sliding_window_inference(baseline_model, image, patch_size, device)
        baseline_pred = torch.argmax(logits, dim=1).squeeze(0).cpu().numpy()

        comparative_pred, uncertainty = run_inference(
            comparative_model, image, patch_size, device,
            model_type="comparative", mc_samples=args.mc_samples, refine=False,
        )
        refined_pred = refine_prediction(
            torch.from_numpy(comparative_pred).long().unsqueeze(0),
            torch.from_numpy(uncertainty).float().unsqueeze(0),
            threshold=args.refine_threshold,
        ).squeeze(0).numpy()

        predictions = {
            "baseline": baseline_pred,
            "baseline_smoothed": majority_smooth(baseline_pred),
            "comparative": comparative_pred,
            "comparative_refined": refined_pred,
        }

        case = {"subject_id": meta["subject_id"]}
        for name, pred in predictions.items():
            case[name] = score_prediction(pred, label_np, spacing)
        per_case.append(case)

        for name in conditions:
            print(
                f"    {name:20} dice={case[name]['mean_dice']:.4f} "
                f"msd={case[name]['mean_surface_distance_mm']:.2f}mm "
                f"hd95={case[name]['hd95_mm']:.2f}mm "
                f"hd100={case[name]['hd100_mm']:.2f}mm "
                f"components={case[name]['n_components']:.1f}",
                flush=True,
            )

    def across(condition, metric):
        values = [c[condition][metric] for c in per_case if not np.isnan(c[condition][metric])]
        return float(np.mean(values)) if values else float("nan")

    summary = {
        condition: {metric: across(condition, metric)
                    for metric in METRICS}
        for condition in conditions
    }

    def relative_change(before, after, metric):
        a, b = summary[before][metric], summary[after][metric]
        return float((b - a) / a * 100.0) if a else float("nan")

    summary["smoothing_effect_on_baseline_pct"] = {
        metric: relative_change("baseline", "baseline_smoothed", metric) for metric in METRICS
    }
    summary["refinement_effect_on_comparative_pct"] = {
        metric: relative_change("comparative", "comparative_refined", metric) for metric in METRICS
    }

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "refinement_control.json"
    out_path.write_text(json.dumps(
        {"split": args.split, "n_subjects": len(per_case),
         "summary": summary, "per_case": per_case}, indent=2))
    print(f"\nSaved {out_path}\n")

    for condition in conditions:
        s = summary[condition]
        print(f"{condition:22} dice={s['mean_dice']:.4f} "
              f"msd={s['mean_surface_distance_mm']:.2f}mm hd95={s['hd95_mm']:.2f}mm "
              f"hd100={s['hd100_mm']:.2f}mm components={s['n_components']:.1f}")
    print("\nchange from smoothing the baseline (%):   ",
          {k: round(v, 2) for k, v in summary["smoothing_effect_on_baseline_pct"].items()})
    print("change from refining the comparative (%): ",
          {k: round(v, 2) for k, v in summary["refinement_effect_on_comparative_pct"].items()})


if __name__ == "__main__":
    main()
