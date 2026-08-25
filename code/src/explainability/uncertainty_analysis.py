"""
Spatial validation of MC-Dropout uncertainty maps.

Measured over a split:
  - mean uncertainty binned by distance to the nearest ground-truth tissue boundary
  - mean uncertainty at correct vs. incorrect voxels, and the AUROC of uncertainty as
    an error detector, both overall and within each distance bin
  - mean uncertainty per ground-truth class

Statistics cover voxels where ground truth or prediction is non-background, and are
aggregated per subject rather than pooled over voxels.

USAGE (from code/ directory, with venv active):
    python -m src.explainability.uncertainty_analysis \
        --checkpoint src/training/checkpoints_comparative/best_model_comp.pt \
        --split test --out-dir src/explainability/analysis
"""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # non-interactive backend, safe for headless/HPC runs
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import distance_transform_edt
from scipy.stats import rankdata, wilcoxon

from src.preprocessing_data_preparation.dataset import FeTADataset
from src.explainability.visualize import CLASS_NAMES, load_model, run_inference

# Upper edges in mm; the final bin catches everything beyond the last edge.
DISTANCE_BIN_EDGES = [1.0, 2.0, 3.0, 5.0, 10.0]

# Minimum voxels for a subject to contribute to a distance bin, and minimum
# contributing subjects for the bin to be aggregated and plotted.
MIN_BIN_VOXELS = 1000
MIN_BIN_SUBJECTS = 3


def distance_bin_labels(edges=DISTANCE_BIN_EDGES):
    labels = [f"<{edges[0]:g}"]
    labels += [f"{lo:g}-{hi:g}" for lo, hi in zip(edges, edges[1:])]
    labels.append(f">={edges[-1]:g}")
    return labels


def boundary_mask(label: np.ndarray) -> np.ndarray:
    """Voxels whose label differs from at least one 6-connected neighbour."""
    boundary = np.zeros(label.shape, dtype=bool)
    for axis in range(3):
        differs = np.diff(label, axis=axis) != 0
        lo = [slice(None)] * 3
        hi = [slice(None)] * 3
        lo[axis] = slice(None, -1)
        hi[axis] = slice(1, None)
        boundary[tuple(lo)] |= differs
        boundary[tuple(hi)] |= differs
    return boundary


def auroc(scores: np.ndarray, positives: np.ndarray) -> float:
    """Rank-based AUROC of `scores` separating `positives` from the rest.

    Mann-Whitney U normalised to [0, 1], ties resolved by average rank. Returns
    nan when either group is empty.
    """
    n_pos = int(positives.sum())
    n_neg = int(positives.size - n_pos)
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    ranks = rankdata(scores)
    return float((ranks[positives].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def analyse_case(label: np.ndarray, pred: np.ndarray, uncertainty: np.ndarray, spacing) -> dict:
    """All three statistics for one volume."""
    region = (label > 0) | (pred > 0)

    distance = distance_transform_edt(~boundary_mask(label), sampling=spacing)
    bin_index = np.digitize(distance[region], DISTANCE_BIN_EDGES)
    region_uncertainty = uncertainty[region]

    error = (pred != label)[region]

    by_distance = []
    for i, name in enumerate(distance_bin_labels()):
        in_bin = bin_index == i
        selected = region_uncertainty[in_bin]
        by_distance.append({
            "bin_mm": name,
            "n_voxels": int(selected.size),
            "mean_uncertainty": float(selected.mean()) if selected.size else float("nan"),
            # Boundary proximity is near-constant within a bin.
            "error_detection_auroc": auroc(selected, error[in_bin]) if selected.size else float("nan"),
            "error_rate": float(error[in_bin].mean()) if selected.size else float("nan"),
        })

    correct_mean = float(region_uncertainty[~error].mean()) if (~error).any() else float("nan")
    error_mean = float(region_uncertainty[error].mean()) if error.any() else float("nan")

    by_class = {}
    for class_index, class_name in CLASS_NAMES.items():
        selected = uncertainty[label == class_index]
        if selected.size:
            by_class[class_name] = float(selected.mean())

    return {
        "n_region_voxels": int(region.sum()),
        "error_rate": float(error.mean()),
        "uncertainty_by_boundary_distance": by_distance,
        "mean_uncertainty_correct": correct_mean,
        "mean_uncertainty_error": error_mean,
        "error_detection_auroc": auroc(region_uncertainty, error),
        "mean_uncertainty_by_gt_class": by_class,
    }


def aggregate(per_case: list[dict]) -> dict:
    """Subject-level means and standard deviations across cases."""
    def summarise(values):
        values = np.array([v for v in values if not np.isnan(v)], dtype=float)
        if values.size == 0:
            return {"mean": float("nan"), "std": float("nan"), "n_subjects": 0}
        return {
            "mean": float(values.mean()),
            "std": float(values.std(ddof=1)) if values.size > 1 else 0.0,
            "n_subjects": int(values.size),
        }

    by_distance = []
    for i, name in enumerate(distance_bin_labels()):
        # Sparse bins are dropped per subject, then entirely if too few remain.
        populated = [
            c["uncertainty_by_boundary_distance"][i]
            for c in per_case
            if c["uncertainty_by_boundary_distance"][i]["n_voxels"] >= MIN_BIN_VOXELS
        ]
        entry = {"bin_mm": name, **summarise([b["mean_uncertainty"] for b in populated])}
        entry["within_bin_auroc"] = summarise(
            [b.get("error_detection_auroc", float("nan")) for b in populated]
        )
        if entry["n_subjects"] < MIN_BIN_SUBJECTS:
            entry.update({"mean": float("nan"), "std": float("nan")})
            entry["within_bin_auroc"].update({"mean": float("nan"), "std": float("nan")})
        by_distance.append(entry)

    class_names = sorted({n for c in per_case for n in c["mean_uncertainty_by_gt_class"]})
    by_class = {
        name: summarise([c["mean_uncertainty_by_gt_class"].get(name, float("nan")) for c in per_case])
        for name in class_names
    }

    # Paired per subject: each subject's correct-voxel mean against its own error mean.
    correct = np.array([c["mean_uncertainty_correct"] for c in per_case], dtype=float)
    error = np.array([c["mean_uncertainty_error"] for c in per_case], dtype=float)
    paired = ~(np.isnan(correct) | np.isnan(error))
    if paired.sum() >= 3:
        statistic, pvalue = wilcoxon(error[paired], correct[paired])
        paired_test = {
            "statistic": float(statistic),
            "pvalue": float(pvalue),
            "n_pairs": int(paired.sum()),
            "n_subjects_error_higher": int((error[paired] > correct[paired]).sum()),
        }
    else:
        paired_test = {"statistic": float("nan"), "pvalue": float("nan"),
                       "n_pairs": int(paired.sum()), "n_subjects_error_higher": 0}

    return {
        "n_subjects": len(per_case),
        "uncertainty_by_boundary_distance": by_distance,
        "mean_uncertainty_correct": summarise([c["mean_uncertainty_correct"] for c in per_case]),
        "mean_uncertainty_error": summarise([c["mean_uncertainty_error"] for c in per_case]),
        "correct_vs_error_wilcoxon": paired_test,
        "error_detection_auroc": summarise([c["error_detection_auroc"] for c in per_case]),
        "mean_uncertainty_by_gt_class": by_class,
    }


def plot_summary(summary: dict, per_case: list[dict], out_dir: Path) -> Path:
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))

    bins = [b for b in summary["uncertainty_by_boundary_distance"] if not np.isnan(b["mean"])]
    x = np.arange(len(bins))
    means = [b["mean"] for b in bins]
    stds = [b["std"] for b in bins]
    axes[0].errorbar(x, means, yerr=stds, marker="o", capsize=4)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels([b["bin_mm"] for b in bins])
    axes[0].set_xlabel("Distance to nearest GT tissue boundary (mm)")
    axes[0].set_ylabel("Mean uncertainty")
    axes[0].set_title("Uncertainty vs. boundary distance")

    correct = [c["mean_uncertainty_correct"] for c in per_case]
    errors = [c["mean_uncertainty_error"] for c in per_case]
    for i, (a, b) in enumerate(zip(correct, errors)):
        axes[1].plot([0, 1], [a, b], marker="o", color="gray", alpha=0.6,
                     label="subject" if i == 0 else None)
    axes[1].set_xticks([0, 1])
    axes[1].set_xticklabels(["correct", "error"])
    axes[1].set_ylabel("Mean uncertainty")
    axes[1].set_title("Uncertainty at correct vs. incorrect voxels")

    by_class = summary["mean_uncertainty_by_gt_class"]
    names = list(by_class)
    y = np.arange(len(names))
    axes[2].barh(y, [by_class[n]["mean"] for n in names],
                 xerr=[by_class[n]["std"] for n in names], capsize=3)
    axes[2].set_yticks(y)
    axes[2].set_yticklabels(names)
    axes[2].set_xlabel("Mean uncertainty")
    axes[2].set_title("Uncertainty by ground-truth class")

    fig.tight_layout()
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "uncertainty_spatial_analysis.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved {out_path}")
    return out_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str)
    parser.add_argument(
        "--from-json", type=str,
        help="Re-aggregate and re-plot a saved report instead of re-running inference"
    )
    parser.add_argument("--split", type=str, default="test", choices=["train", "val", "test"])
    parser.add_argument("--out-dir", type=str, default="src/explainability/analysis")
    parser.add_argument("--patch-size", type=int, nargs=3, default=[128, 128, 128])
    parser.add_argument("--mc-samples", type=int, default=10)
    parser.add_argument(
        "--refine", action="store_true",
        help="Analyse the refined prediction instead of the model's raw output"
    )
    parser.add_argument("--refine-threshold", type=float, default=0.5)
    parser.add_argument("--limit", type=int, default=None, help="Analyse only the first N cases")
    args = parser.parse_args()

    if args.from_json:
        source = json.loads(Path(args.from_json).read_text())
        per_case = source["per_case"]
        summary = aggregate(per_case)
        report = {**source, "summary": summary, "per_case": per_case}
        out_dir = Path(args.out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "uncertainty_spatial_analysis.json").write_text(json.dumps(report, indent=2))
        plot_summary(summary, per_case, out_dir)
        return

    if not args.checkpoint:
        parser.error("--checkpoint is required unless --from-json is given")

    import torch
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    code_root = Path(__file__).resolve().parents[2]
    eval_ds = FeTADataset(
        code_root / "src/data/splits/split_v1.json",
        code_root / "src/data/config.yaml",
        split_name=args.split,
        mode="eval",
    )
    n_cases = len(eval_ds) if args.limit is None else min(args.limit, len(eval_ds))

    model = load_model(Path(args.checkpoint), device, model_type="comparative")

    per_case = []
    for index in range(n_cases):
        image, label, meta = eval_ds[index]
        subject_id = meta["subject_id"]
        print(f"[{index + 1}/{n_cases}] {subject_id}", flush=True)

        pred, uncertainty = run_inference(
            model, image, tuple(args.patch_size), device,
            model_type="comparative",
            mc_samples=args.mc_samples,
            refine=args.refine,
            refine_threshold=args.refine_threshold,
        )

        result = analyse_case(label.numpy(), pred, uncertainty, tuple(meta["spacing"]))
        result["subject_id"] = subject_id
        per_case.append(result)
        print(
            f"    AUROC={result['error_detection_auroc']:.3f} "
            f"correct={result['mean_uncertainty_correct']:.4f} "
            f"error={result['mean_uncertainty_error']:.4f}",
            flush=True,
        )

    summary = aggregate(per_case)
    report = {
        "split": args.split,
        "checkpoint": str(args.checkpoint),
        "mc_samples": args.mc_samples,
        "refined": args.refine,
        "summary": summary,
        "per_case": per_case,
    }

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "uncertainty_spatial_analysis.json"
    out_path.write_text(json.dumps(report, indent=2))
    print(f"Saved {out_path}")

    plot_summary(summary, per_case, out_dir)

    auroc_summary = summary["error_detection_auroc"]
    print(
        f"\nError-detection AUROC across {auroc_summary['n_subjects']} subjects: "
        f"{auroc_summary['mean']:.3f} +/- {auroc_summary['std']:.3f}"
    )


if __name__ == "__main__":
    main()
