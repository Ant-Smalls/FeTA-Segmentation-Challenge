# my part (role 5) - runs inference, computes the metrics, does calibration
# + refinement, spits out the report. see below for how to actually run it

# from code/ (venv on):
#   python -m src.evaluation.full_evaluation \
#       --baseline-checkpoint checkpoints/best_model_baseline.pt \
#       --comparative-checkpoint checkpoints/best_model_comparative.pt \
#       --refinement-threshold 0.5
# (comparative checkpoint optional, just does baseline dice/ED if left out)

from __future__ import annotations

import argparse
import csv
import json
import time
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Protocol, runtime_checkable

import matplotlib

matplotlib.use("Agg")  # HPC has no display, forgot this the first time and it just hung
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import yaml
from scipy.stats import false_discovery_control, spearmanr, wilcoxon
from skimage.measure import euler_number

from ..preprocessing_data_preparation.dataset import FeTADataset, reconstruct_from_patches

NUM_CLASSES = 8  # 0=background, 1=eCSF, 2=GM, 3=WM, 4=Ventricles, 5=Cerebellum, 6=dGM, 7=Brainstem

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

EvalCase = tuple[torch.Tensor, torch.Tensor, dict]  # image, label, meta (eval-mode FeTADataset)
PatchCoords = tuple[slice, slice, slice]


# using Protocol here so roles 2/3's models just need to match the shape,
# means I can test this whole file with fake/dummy models before their
# checkpoints actually exist

DEFAULT_STRIDE_FRACTION = 0.5  # config.yaml doesn't define a stride, nnU-Net default is 0.5 so going with that


@runtime_checkable
class SegmentationModel(Protocol):
    # minimum interface a model needs for eval

    def predict_patch(self, patch: torch.Tensor) -> torch.Tensor:
        # (1,D,H,W) float32 in, (NUM_CLASSES,D,H,W) probs out
        ...


@runtime_checkable
class UncertaintyModel(SegmentationModel, Protocol):
    # comparative model - on top of predict_patch it also gives per-voxel uncertainty + can refine

    def predict_patch_with_uncertainty(self, patch: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        ...

    def refine(
        self,
        probabilities: torch.Tensor,
        uncertainty: torch.Tensor,
        threshold: float,
        enabled: bool = True,
    ) -> torch.Tensor:
        # NOTE: enabled=False needs to give back EXACTLY to_hard_prediction(probabilities),
        # i rely on this later to sanity check the refine() implementations
        ...


def default_stride(
    patch_size: tuple[int, int, int], stride_fraction: float = DEFAULT_STRIDE_FRACTION
) -> tuple[int, int, int]:
    # fraction=1.0 -> no overlap between tiles (quicker, useful on cpu). 0.5 is nnU-Net's usual choice.
    # either way you still get full coverage of the volume
    return tuple(max(1, int(p * stride_fraction)) for p in patch_size)


def sliding_window_patch_coords(
    volume_shape: tuple[int, int, int],
    patch_size: tuple[int, int, int],
    stride: tuple[int, int, int],
) -> list[PatchCoords]:
    # last tile in each dim gets shifted back so it lines up with the edge instead of
    # overshooting - otherwise you'd need to pad more than necessary
    starts_per_dim = []
    for dim_size, p, s in zip(volume_shape, patch_size, stride):
        if dim_size < p:
            raise ValueError(f"volume_shape dim {dim_size} smaller than patch_size dim {p}; pad first.")
        starts = list(range(0, dim_size - p + 1, s))
        if starts[-1] != dim_size - p:
            starts.append(dim_size - p)
        starts_per_dim.append(starts)

    coords = []
    for d in starts_per_dim[0]:
        for h in starts_per_dim[1]:
            for w in starts_per_dim[2]:
                coords.append((slice(d, d + patch_size[0]), slice(h, h + patch_size[1]), slice(w, w + patch_size[2])))
    return coords


def pad_to_patch_size(
    image: torch.Tensor, patch_size: tuple[int, int, int]
) -> tuple[torch.Tensor, PatchCoords]:
    # pad with the volume's min value, NOT zero - after z-score norm zero isn't
    # actually background so padding with it would be wrong. also returns the
    # crop slices so we can undo the padding later
    spatial_shape = tuple(image.shape[1:])
    crop_slices = tuple(slice(0, s) for s in spatial_shape)
    pad_widths = [max(0, p - s) for s, p in zip(spatial_shape, patch_size)]

    if all(w == 0 for w in pad_widths):
        return image, crop_slices

    pad_value = float(image.min())
    # careful - F.pad wants the pad tuple last-dim-first (W,W,H,H,D,D), not D,H,W order like everything else
    torch_pad = (0, pad_widths[2], 0, pad_widths[1], 0, pad_widths[0])
    padded = F.pad(image, torch_pad, mode="constant", value=pad_value)
    return padded, crop_slices


def to_hard_prediction(probabilities: torch.Tensor) -> torch.Tensor:
    return torch.argmax(probabilities, dim=0).to(torch.int64)


def run_sliding_window_inference(
    model: SegmentationModel,
    image: torch.Tensor,
    patch_size: tuple[int, int, int],
    stride: tuple[int, int, int] | None,
    aggregation: str,
) -> torch.Tensor:
    stride = stride if stride is not None else default_stride(patch_size)
    padded_image, crop_slices = pad_to_patch_size(image, patch_size)
    padded_shape = tuple(padded_image.shape[1:])
    coords = sliding_window_patch_coords(padded_shape, patch_size, stride)

    patch_predictions = [model.predict_patch(padded_image[(slice(None), *c)]) for c in coords]
    stitched = reconstruct_from_patches(patch_predictions, coords, padded_shape, aggregation=aggregation)
    return stitched[(slice(None), *crop_slices)]  # crop back off the padding we added


def run_sliding_window_inference_with_uncertainty(
    model: UncertaintyModel,
    image: torch.Tensor,
    patch_size: tuple[int, int, int],
    stride: tuple[int, int, int] | None,
    aggregation: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    # same idea as run_sliding_window_inference but also stitches the uncertainty map
    stride = stride if stride is not None else default_stride(patch_size)
    padded_image, crop_slices = pad_to_patch_size(image, patch_size)
    padded_shape = tuple(padded_image.shape[1:])
    coords = sliding_window_patch_coords(padded_shape, patch_size, stride)

    prob_patches: list[torch.Tensor] = []
    uncertainty_patches: list[torch.Tensor] = []
    for c in coords:
        probs, uncertainty = model.predict_patch_with_uncertainty(padded_image[(slice(None), *c)])
        prob_patches.append(probs)
        uncertainty_patches.append(uncertainty.unsqueeze(0))

    stitched_probs = reconstruct_from_patches(prob_patches, coords, padded_shape, aggregation=aggregation)
    stitched_uncertainty = reconstruct_from_patches(uncertainty_patches, coords, padded_shape, aggregation=aggregation)

    return (
        stitched_probs[(slice(None), *crop_slices)],
        stitched_uncertainty[(0, *crop_slices)],
    )


# both of these expect hard (D,H,W) label arrays, not probabilities

EULER_CONNECTIVITY = 1  # doesn't really matter which connectivity as long as pred + gt use the same one, it cancels out


def dice_per_class(
    prediction: np.ndarray, ground_truth: np.ndarray, num_classes: int = NUM_CLASSES
) -> dict[int, float]:
    # NaN when the class isn't in the gt at all - dice isn't really defined there,
    # calling it 0 would be misleading (pulls down the mean for no reason)
    scores: dict[int, float] = {}
    for class_id in range(num_classes):
        pred_mask = prediction == class_id
        gt_mask = ground_truth == class_id
        gt_voxels = int(gt_mask.sum())

        if gt_voxels == 0:
            scores[class_id] = float("nan")
            continue

        intersection = int(np.logical_and(pred_mask, gt_mask).sum())
        denom = int(pred_mask.sum()) + gt_voxels
        scores[class_id] = 2.0 * intersection / denom

    return scores


def mean_dice(per_class_dice: dict[int, float], exclude_background: bool = True) -> float:
    # background dominates the volume so leaving it in would hide how bad the
    # actual foreground segmentation is - excluded by default for that reason
    values = [v for k, v in per_class_dice.items() if not (exclude_background and k == 0)]
    return float(np.nanmean(values))


def euler_characteristic_difference(
    prediction: np.ndarray,
    ground_truth: np.ndarray,
    class_id: int,
    connectivity: int = EULER_CONNECTIVITY,
) -> int:
    pred_euler = euler_number(prediction == class_id, connectivity=connectivity)
    gt_euler = euler_number(ground_truth == class_id, connectivity=connectivity)
    return abs(int(pred_euler) - int(gt_euler))


def euler_characteristic_difference_per_class(
    prediction: np.ndarray,
    ground_truth: np.ndarray,
    class_ids: tuple[int, ...] = tuple(range(1, NUM_CLASSES)),
    connectivity: int = EULER_CONNECTIVITY,
) -> dict[int, int]:
    # loops euler_characteristic_difference over every class (defaults to all fg classes)
    return {
        class_id: euler_characteristic_difference(prediction, ground_truth, class_id, connectivity)
        for class_id in class_ids
    }


# important: run_full_evaluation always does calibration BEFORE refinement, since
# the refinement numbers are basically meaningless if the uncertainty is garbage
# (see supervisor meeting notes 03/14 - this took a while to convince myself of)

DEFAULT_N_BINS = 10

# using spearman rho instead of a hard ECE cutoff since we just need uncertainty to
# be monotonic w/ error, doesn't need to literally be a calibrated probability
SPEARMAN_RHO_GATE = 0.2
SPEARMAN_ALPHA = 0.05


@dataclass
class CalibrationSummary:
    spearman_rho: float
    spearman_pvalue: float
    ece_style_score: float
    n_bins: int
    n_voxels: int
    passes_gate: bool
    caveat: str | None


def compute_voxel_errors(prediction: np.ndarray, ground_truth: np.ndarray) -> np.ndarray:
    # this always runs on the PRE-refinement prediction, don't move it after refine()
    return (prediction != ground_truth).astype(np.float64)


def bin_uncertainty_vs_error(
    uncertainty: np.ndarray, error: np.ndarray, n_bins: int = DEFAULT_N_BINS
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    # quantile bins (equal voxel count each) rather than equal-width - most voxels
    # sit at low uncertainty so equal-width bins would leave half of them empty
    order = np.argsort(uncertainty)
    uncertainty_groups = np.array_split(uncertainty[order], n_bins)
    error_groups = np.array_split(error[order], n_bins)

    bin_mean_uncertainty = np.array([g.mean() for g in uncertainty_groups])
    bin_error_rate = np.array([g.mean() for g in error_groups])
    bin_voxel_count = np.array([g.size for g in uncertainty_groups])
    return bin_mean_uncertainty, bin_error_rate, bin_voxel_count


def compute_calibration_summary(
    uncertainty: np.ndarray, error: np.ndarray, n_bins: int = DEFAULT_N_BINS
) -> CalibrationSummary:
    # note the pass/fail decision is based on rho, the ece-style score below is
    # just for the plot / reporting, doesn't affect the gate
    rho, pvalue = spearmanr(uncertainty, error)
    bin_mean_uncertainty, bin_error_rate, bin_voxel_count = bin_uncertainty_vs_error(uncertainty, error, n_bins)

    unc_min, unc_max = float(uncertainty.min()), float(uncertainty.max())
    if unc_max > unc_min:
        normalized_bin_uncertainty = (bin_mean_uncertainty - unc_min) / (unc_max - unc_min)
    else:
        normalized_bin_uncertainty = np.zeros_like(bin_mean_uncertainty)
    weights = bin_voxel_count / bin_voxel_count.sum()
    ece_style_score = float(np.sum(weights * np.abs(normalized_bin_uncertainty - bin_error_rate)))

    passes_gate = bool(rho > SPEARMAN_RHO_GATE and pvalue < SPEARMAN_ALPHA)
    caveat = None
    if not passes_gate:
        caveat = (
            f"Uncertainty-error calibration check FAILED (Spearman rho={rho:.3f}, p={pvalue:.3g}, "
            f"gate requires rho > {SPEARMAN_RHO_GATE} and p < {SPEARMAN_ALPHA}). Predicted uncertainty "
            "does not reliably flag real error; refinement results should be treated as exploratory "
            "only, not evidence that uncertainty-guided refinement improves boundary accuracy."
        )

    return CalibrationSummary(
        spearman_rho=float(rho),
        spearman_pvalue=float(pvalue),
        ece_style_score=ece_style_score,
        n_bins=n_bins,
        n_voxels=int(uncertainty.size),
        passes_gate=passes_gate,
        caveat=caveat,
    )


def plot_reliability_diagram(
    bin_mean_uncertainty: np.ndarray,
    bin_error_rate: np.ndarray,
    bin_voxel_count: np.ndarray,
    summary: CalibrationSummary,
    output_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(6, 6))

    sizes = 200 * bin_voxel_count / bin_voxel_count.max()
    ax.scatter(bin_mean_uncertainty, bin_error_rate, s=sizes, alpha=0.75, edgecolors="black", zorder=3)
    ax.plot(bin_mean_uncertainty, bin_error_rate, linestyle="--", alpha=0.5, zorder=2)

    ax.set_xlabel("Mean predicted uncertainty (bin)")
    ax.set_ylabel("Observed voxel error rate (bin)")
    verdict = "PASSES" if summary.passes_gate else "FAILS"
    ax.set_title(
        f"Reliability diagram ({summary.n_voxels} voxels, {summary.n_bins} bins)\n"
        f"Spearman rho={summary.spearman_rho:.3f}, p={summary.spearman_pvalue:.3g} -- calibration {verdict}"
    )
    ax.grid(True, alpha=0.3)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


# enabled=False HAS to be a true no-op or the before/after comparison is meaningless -
# compare_before_after_refinement checks this and blows up loudly if it's not true

REFINEMENT_CLASSES = (1, 2, 6)  # eCSF, GM, dGM - the classes refinement actually targets


@dataclass
class RefinementComparison:
    per_case: list[dict]
    mean_dice_before: dict[int, float]
    mean_dice_after: dict[int, float]
    mean_ed_before: dict[int, float]
    mean_ed_after: dict[int, float]


def compare_before_after_refinement(
    subject_id: str,
    probabilities: torch.Tensor,
    uncertainty: torch.Tensor,
    ground_truth: torch.Tensor,
    model: UncertaintyModel,
    threshold: float,
    classes: tuple[int, ...] = REFINEMENT_CLASSES,
) -> dict:
    before_hard = to_hard_prediction(probabilities)

    # sanity check - if disabling refinement doesn't reproduce the plain prediction
    # then something's wrong w/ the model's refine() and before/after numbers are junk
    reproduced = model.refine(probabilities, uncertainty, threshold, enabled=False)
    if not torch.equal(reproduced, before_hard):
        raise ValueError(
            f"{subject_id}: model.refine(..., enabled=False) did not reproduce the pre-refinement "
            "prediction exactly. Refinement must be a true no-op when disabled (role 3's ablation "
            "contract) -- this model violates it, so before/after numbers would not be meaningful."
        )

    after_hard = model.refine(probabilities, uncertainty, threshold, enabled=True)

    before_np = before_hard.detach().cpu().numpy()
    after_np = after_hard.detach().cpu().numpy()
    gt_np = ground_truth.detach().cpu().numpy()

    dice_before_all = dice_per_class(before_np, gt_np)
    dice_after_all = dice_per_class(after_np, gt_np)
    ed_before = euler_characteristic_difference_per_class(before_np, gt_np, class_ids=classes)
    ed_after = euler_characteristic_difference_per_class(after_np, gt_np, class_ids=classes)

    return {
        "subject_id": subject_id,
        "dice_before": {c: dice_before_all[c] for c in classes},
        "dice_after": {c: dice_after_all[c] for c in classes},
        "ed_before": ed_before,
        "ed_after": ed_after,
    }


def aggregate_refinement_comparison(
    per_case: list[dict], classes: tuple[int, ...] = REFINEMENT_CLASSES
) -> RefinementComparison:
    def mean_over_cases(key: str, class_id: int) -> float:
        return float(np.nanmean([case[key][class_id] for case in per_case]))

    mean_dice_before = {c: mean_over_cases("dice_before", c) for c in classes}
    mean_dice_after = {c: mean_over_cases("dice_after", c) for c in classes}
    mean_ed_before = {c: mean_over_cases("ed_before", c) for c in classes}
    mean_ed_after = {c: mean_over_cases("ed_after", c) for c in classes}

    return RefinementComparison(
        per_case=per_case,
        mean_dice_before=mean_dice_before,
        mean_dice_after=mean_dice_after,
        mean_ed_before=mean_ed_before,
        mean_ed_after=mean_ed_after,
    )


# mc dropout stuff below - only for the validation split, this is a model
# diagnostic not something we report as a test metric

def run_mc_dropout_sliding_window_inference(
    net: torch.nn.Module,
    image: torch.Tensor,
    patch_size: tuple[int, int, int],
    stride: tuple[int, int, int] | None,
    aggregation: str,
    n_mc_samples: int = 10,
) -> tuple[torch.Tensor, torch.Tensor]:
    # entropy gets computed AFTER stitching the full volume back together (on the
    # mean prob), not per-patch first - doing it per patch left visible seams
    # between tiles when I first tried it
    net.eval()
    for module in net.modules():
        if isinstance(module, torch.nn.Dropout3d):
            module.train()  # keep MC Dropout active; everything else (BatchNorm etc.) stays in eval mode

    stride = stride if stride is not None else default_stride(patch_size)
    padded_image, crop_slices = pad_to_patch_size(image, patch_size)
    padded_shape = tuple(padded_image.shape[1:])
    coords = sliding_window_patch_coords(padded_shape, patch_size, stride)

    prob_patches = []
    with torch.no_grad():
        for c in coords:
            patch = padded_image[(slice(None), *c)].unsqueeze(0)  # (1, 1, pd, ph, pw)
            samples = []
            for _ in range(n_mc_samples):
                logits, _ = net(patch)
                samples.append(torch.softmax(logits, dim=1))
            prob_patches.append(torch.stack(samples, dim=0).mean(dim=0).squeeze(0))  # (C, pd, ph, pw)

    stitched = reconstruct_from_patches(prob_patches, coords, padded_shape, aggregation=aggregation)
    mean_probability = stitched[(slice(None), *crop_slices)]

    entropy = -(mean_probability * torch.log(mean_probability.clamp_min(1e-8))).sum(dim=0)
    uncertainty = entropy / np.log(mean_probability.shape[0])
    return mean_probability, uncertainty


@dataclass
class ValidationUncertaintyMapReport:
    per_case: list[dict]  # subject_id, mean_entropy, std_entropy, spearman_rho, spearman_pvalue, n_voxels
    pooled_spearman_rho: float
    pooled_spearman_pvalue: float
    n_mc_samples: int
    n_cases: int


def run_validation_uncertainty_maps(
    dataset: Iterable[EvalCase],
    comparative_net: torch.nn.Module,
    patch_size: tuple[int, int, int],
    stride: tuple[int, int, int] | None,
    aggregation: str,
    n_mc_samples: int = 10,
    maps_dir: Path | None = None,
    verbose: bool = True,
    skip_existing: bool = True,
) -> ValidationUncertaintyMapReport:
    # skip_existing = reload saved .npy maps instead of rerunning MC-dropout inference.
    # added this after losing like 40 min of compute to a crash 3/4 of the way through :(
    if maps_dir is not None:
        maps_dir = Path(maps_dir)
        maps_dir.mkdir(parents=True, exist_ok=True)

    per_case: list[dict] = []
    pooled_uncertainty: list[np.ndarray] = []
    pooled_error: list[np.ndarray] = []

    for case_index, (image, label, meta) in enumerate(dataset):
        case_start = time.time()
        subject_id = meta["subject_id"]
        gt_np = _as_numpy(label)

        entropy_path = maps_dir / f"{subject_id}_entropy.npy" if maps_dir is not None else None
        error_path = maps_dir / f"{subject_id}_error.npy" if maps_dir is not None else None
        reuse_existing = (
            skip_existing and entropy_path is not None and entropy_path.exists() and error_path.exists()
        )

        if reuse_existing:
            unc_np = np.load(entropy_path)
            error = np.load(error_path)
            if verbose:
                print(f"[{case_index + 1}] {subject_id}: found existing entropy/error maps, skipping MC-dropout inference")
        else:
            mean_probability, uncertainty = run_mc_dropout_sliding_window_inference(
                comparative_net, image, patch_size, stride, aggregation, n_mc_samples
            )
            hard = to_hard_prediction(mean_probability)
            unc_np = _as_numpy(uncertainty)
            error = compute_voxel_errors(_as_numpy(hard), gt_np)

            if maps_dir is not None:
                np.save(entropy_path, unc_np)
                np.save(error_path, error.astype(np.uint8))
                plot_uncertainty_map(
                    _as_numpy(image)[0], gt_np, _as_numpy(hard), unc_np, subject_id, maps_dir / f"{subject_id}_map.png"
                )

            if verbose:
                print(f"[{case_index + 1}] {subject_id}: MC-dropout entropy done ({time.time() - case_start:.1f}s)")

        foreground_mask = gt_np != 0
        rho, pvalue = spearmanr(unc_np[foreground_mask], error[foreground_mask])
        per_case.append(
            {
                "subject_id": subject_id,
                "mean_entropy": float(unc_np[foreground_mask].mean()),
                "std_entropy": float(unc_np[foreground_mask].std()),
                "spearman_rho": float(rho),
                "spearman_pvalue": float(pvalue),
                "n_voxels": int(foreground_mask.sum()),
            }
        )
        pooled_uncertainty.append(unc_np[foreground_mask])
        pooled_error.append(error[foreground_mask])

    pooled_rho, pooled_pvalue = spearmanr(np.concatenate(pooled_uncertainty), np.concatenate(pooled_error))

    return ValidationUncertaintyMapReport(
        per_case=per_case,
        pooled_spearman_rho=float(pooled_rho),
        pooled_spearman_pvalue=float(pooled_pvalue),
        n_mc_samples=n_mc_samples,
        n_cases=len(per_case),
    )


def plot_uncertainty_map(
    image: np.ndarray, ground_truth: np.ndarray, prediction: np.ndarray, uncertainty: np.ndarray,
    subject_id: str, output_path: Path,
) -> None:
    # just grabbing the middle slice for a quick visual, not doing anything fancy w/ MIP or whatever
    mid = image.shape[0] // 2
    fig, axes = plt.subplots(1, 4, figsize=(16, 4.5))
    axes[0].imshow(image[mid], cmap="gray")
    axes[0].set_title(f"{subject_id} -- T2w")
    axes[1].imshow(ground_truth[mid], cmap="tab10", vmin=0, vmax=NUM_CLASSES - 1)
    axes[1].set_title("Ground truth")
    axes[2].imshow(prediction[mid], cmap="tab10", vmin=0, vmax=NUM_CLASSES - 1)
    axes[2].set_title("Prediction (MC-dropout mean)")
    im = axes[3].imshow(uncertainty[mid], cmap="magma", vmin=0, vmax=1)
    axes[3].set_title("Predictive entropy (MC dropout)")
    for ax in axes:
        ax.axis("off")
    fig.colorbar(im, ax=axes[3], fraction=0.046, pad=0.04)
    fig.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def write_validation_uncertainty_report(report: ValidationUncertaintyMapReport, output_dir: Path) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(output_dir / "validation_uncertainty_report.json", "w") as f:
        json.dump(
            {
                "n_cases": report.n_cases,
                "n_mc_samples": report.n_mc_samples,
                "pooled_spearman_rho": report.pooled_spearman_rho,
                "pooled_spearman_pvalue": report.pooled_spearman_pvalue,
                "per_case": report.per_case,
            },
            f,
            indent=2,
        )

    fieldnames = ["subject_id", "mean_entropy", "std_entropy", "spearman_rho", "spearman_pvalue", "n_voxels"]
    with open(output_dir / "validation_uncertainty_per_case.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for case in report.per_case:
            writer.writerow(case)


# order in run_full_evaluation matters: baseline -> comparative -> calibration -> refinement.
# comparative_model can be None (baseline-only run, e.g. before role 3's checkpoint exists)

@dataclass
class EvaluationReport:
    baseline_per_case: list[dict]
    comparative_per_case: list[dict] | None
    baseline_mean_dice_per_class: dict[int, float]
    comparative_mean_dice_per_class: dict[int, float] | None
    baseline_mean_ed_per_class: dict[int, float]
    comparative_mean_ed_per_class: dict[int, float] | None
    baseline_mean_dice_foreground: float
    comparative_mean_dice_foreground: float | None
    calibration: CalibrationSummary | None
    calibration_bin_mean_uncertainty: list[float] | None
    calibration_bin_error_rate: list[float] | None
    calibration_bin_voxel_count: list[int] | None
    refinement_comparison: RefinementComparison | None
    threshold: float | None
    patch_size: tuple[int, int, int]
    aggregation: str
    n_test_cases: int


def _as_numpy(tensor: torch.Tensor) -> np.ndarray:
    return tensor.detach().cpu().numpy()


def _aggregate_mean_per_class(per_case: list[dict], key: str) -> dict[int, float]:
    class_ids = per_case[0][key].keys()
    return {c: float(np.nanmean([case[key][c] for case in per_case])) for c in class_ids}


def run_full_evaluation(
    dataset: Iterable[EvalCase],
    baseline_model: SegmentationModel,
    comparative_model: UncertaintyModel | None,
    patch_size: tuple[int, int, int],
    stride: tuple[int, int, int] | None,
    aggregation: str,
    refinement_threshold: float | None,
    verbose: bool = True,
) -> EvaluationReport:
    # dataset gives (image, label, meta) - either the real eval-mode FeTADataset or
    # fake tuples in the unit tests. comparative_model=None -> baseline only
    baseline_per_case: list[dict] = []
    comparative_per_case: list[dict] | None = [] if comparative_model is not None else None
    pooled_uncertainty: list[np.ndarray] = []
    pooled_error: list[np.ndarray] = []
    # keeping these around so we don't have to re-run the (slow) sliding window
    # inference again after calibration - just reuse what we already computed
    comparative_cache: list[tuple[str, torch.Tensor, torch.Tensor, torch.Tensor]] = []

    for case_index, (image, label, meta) in enumerate(dataset):
        case_start = time.time()
        subject_id = meta["subject_id"]
        ground_truth_np = _as_numpy(label)

        baseline_probs = run_sliding_window_inference(baseline_model, image, patch_size, stride, aggregation)
        baseline_hard = _as_numpy(to_hard_prediction(baseline_probs))
        baseline_per_case.append(
            {
                "subject_id": subject_id,
                "dice": dice_per_class(baseline_hard, ground_truth_np),
                "ed": euler_characteristic_difference_per_class(baseline_hard, ground_truth_np),
            }
        )

        if comparative_model is None:
            if verbose:
                print(f"[{case_index + 1}] {subject_id}: baseline done ({time.time() - case_start:.1f}s)")
            continue

        comparative_probs, comparative_uncertainty = run_sliding_window_inference_with_uncertainty(
            comparative_model, image, patch_size, stride, aggregation
        )
        comparative_hard = _as_numpy(to_hard_prediction(comparative_probs))
        comparative_per_case.append(
            {
                "subject_id": subject_id,
                "dice": dice_per_class(comparative_hard, ground_truth_np),
                "ed": euler_characteristic_difference_per_class(comparative_hard, ground_truth_np),
            }
        )

        foreground_mask = ground_truth_np != 0
        errors = compute_voxel_errors(comparative_hard, ground_truth_np)
        pooled_uncertainty.append(_as_numpy(comparative_uncertainty)[foreground_mask])
        pooled_error.append(errors[foreground_mask])

        comparative_cache.append((subject_id, comparative_probs, comparative_uncertainty, label))

        if verbose:
            print(
                f"[{case_index + 1}] {subject_id}: baseline+comparative done ({time.time() - case_start:.1f}s)"
            )

    if verbose and comparative_model is not None and comparative_cache:
        print("Computing calibration + before/after refinement...")

    baseline_mean_dice_per_class = _aggregate_mean_per_class(baseline_per_case, "dice")
    baseline_mean_ed_per_class = _aggregate_mean_per_class(baseline_per_case, "ed")
    baseline_mean_dice_foreground = mean_dice(baseline_mean_dice_per_class)

    if comparative_model is None:
        return EvaluationReport(
            baseline_per_case=baseline_per_case,
            comparative_per_case=None,
            baseline_mean_dice_per_class=baseline_mean_dice_per_class,
            comparative_mean_dice_per_class=None,
            baseline_mean_ed_per_class=baseline_mean_ed_per_class,
            comparative_mean_ed_per_class=None,
            baseline_mean_dice_foreground=baseline_mean_dice_foreground,
            comparative_mean_dice_foreground=None,
            calibration=None,
            calibration_bin_mean_uncertainty=None,
            calibration_bin_error_rate=None,
            calibration_bin_voxel_count=None,
            refinement_comparison=None,
            threshold=None,
            patch_size=tuple(patch_size),
            aggregation=aggregation,
            n_test_cases=len(baseline_per_case),
        )

    # pool voxels from every case together for calibration (before refinement touches anything)
    all_uncertainty = np.concatenate(pooled_uncertainty)
    all_error = np.concatenate(pooled_error)
    calibration_summary = compute_calibration_summary(all_uncertainty, all_error)
    bin_mean_uncertainty, bin_error_rate, bin_voxel_count = bin_uncertainty_vs_error(all_uncertainty, all_error)

    refinement_per_case = [
        compare_before_after_refinement(subject_id, probs, uncertainty, label, comparative_model, refinement_threshold)
        for subject_id, probs, uncertainty, label in comparative_cache
    ]
    refinement_comparison = aggregate_refinement_comparison(refinement_per_case)

    comparative_mean_dice_per_class = _aggregate_mean_per_class(comparative_per_case, "dice")

    return EvaluationReport(
        baseline_per_case=baseline_per_case,
        comparative_per_case=comparative_per_case,
        baseline_mean_dice_per_class=baseline_mean_dice_per_class,
        comparative_mean_dice_per_class=comparative_mean_dice_per_class,
        baseline_mean_ed_per_class=baseline_mean_ed_per_class,
        comparative_mean_ed_per_class=_aggregate_mean_per_class(comparative_per_case, "ed"),
        baseline_mean_dice_foreground=baseline_mean_dice_foreground,
        comparative_mean_dice_foreground=mean_dice(comparative_mean_dice_per_class),
        calibration=calibration_summary,
        calibration_bin_mean_uncertainty=bin_mean_uncertainty.tolist(),
        calibration_bin_error_rate=bin_error_rate.tolist(),
        calibration_bin_voxel_count=bin_voxel_count.tolist(),
        refinement_comparison=refinement_comparison,
        threshold=refinement_threshold,
        patch_size=tuple(patch_size),
        aggregation=aggregation,
        n_test_cases=len(baseline_per_case),
    )


def _report_to_json_dict(report: EvaluationReport) -> dict:
    result: dict = {
        "n_test_cases": report.n_test_cases,
        "patch_size": list(report.patch_size),
        "aggregation": report.aggregation,
        "refinement_threshold": report.threshold,
        "baseline": {
            "mean_dice_per_class": report.baseline_mean_dice_per_class,
            "mean_dice_foreground": report.baseline_mean_dice_foreground,
            "mean_ed_per_class": report.baseline_mean_ed_per_class,
        },
    }

    if report.comparative_mean_dice_foreground is None:
        result["comparative"] = None
        result["calibration"] = None
        result["refinement_comparison"] = None
        return result

    # write calibration in before the refinement dict below, since it's what gates trusting those numbers
    refinement_dict = asdict(report.refinement_comparison)
    if report.calibration.caveat is not None:
        refinement_dict["calibration_caveat"] = report.calibration.caveat

    result["comparative"] = {
        "mean_dice_per_class": report.comparative_mean_dice_per_class,
        "mean_dice_foreground": report.comparative_mean_dice_foreground,
        "mean_ed_per_class": report.comparative_mean_ed_per_class,
    }
    result["calibration"] = asdict(report.calibration)
    result["refinement_comparison"] = refinement_dict
    return result


def _write_per_case_csv(report: EvaluationReport, path: Path) -> None:
    fieldnames = ["subject_id", "model", "class_id", "class_name", "dice", "ed"]
    model_case_pairs = [("baseline", report.baseline_per_case)]
    if report.comparative_per_case is not None:
        model_case_pairs.append(("comparative", report.comparative_per_case))

    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for model_name, cases in model_case_pairs:
            for case in cases:
                for class_id, dice_value in case["dice"].items():
                    writer.writerow(
                        {
                            "subject_id": case["subject_id"],
                            "model": model_name,
                            "class_id": class_id,
                            "class_name": CLASS_NAMES.get(class_id, str(class_id)),
                            "dice": dice_value,
                            "ed": case["ed"].get(class_id, ""),
                        }
                    )


def _write_refinement_csv(report: EvaluationReport, path: Path) -> None:
    fieldnames = ["subject_id", "class_id", "class_name", "dice_before", "dice_after", "ed_before", "ed_after"]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for case in report.refinement_comparison.per_case:
            for class_id in REFINEMENT_CLASSES:
                writer.writerow(
                    {
                        "subject_id": case["subject_id"],
                        "class_id": class_id,
                        "class_name": CLASS_NAMES[class_id],
                        "dice_before": case["dice_before"][class_id],
                        "dice_after": case["dice_after"][class_id],
                        "ed_before": case["ed_before"][class_id],
                        "ed_after": case["ed_after"][class_id],
                    }
                )


def render_console_summary(report: EvaluationReport) -> str:
    lines = [
        (
            f"Evaluation report -- {report.n_test_cases} test cases, "
            f"patch_size={report.patch_size}, aggregation={report.aggregation!r}"
        ),
        "",
        f"Baseline mean Dice (foreground):     {report.baseline_mean_dice_foreground:.4f}",
    ]

    if report.comparative_mean_dice_foreground is None:
        lines += [
            "",
            (
                "No comparative/uncertainty checkpoint provided -- baseline-only evaluation. "
                "Calibration and before/after-refinement sections are skipped."
            ),
        ]
        return "\n".join(lines)

    c = report.calibration
    rc = report.refinement_comparison
    lines += [
        f"Comparative mean Dice (foreground):  {report.comparative_mean_dice_foreground:.4f}",
        "",
        "=== Calibration (review before trusting the refinement numbers below) ===",
        (
            f"Spearman rho={c.spearman_rho:.3f} (p={c.spearman_pvalue:.3g}), "
            f"ECE-style score={c.ece_style_score:.4f}, n_voxels={c.n_voxels}"
        ),
        f"Gate: {'PASSES' if c.passes_gate else 'FAILS'}",
    ]
    if c.caveat:
        lines.append(f"CAVEAT: {c.caveat}")
    lines += [
        "",
        f"=== Before/after refinement (eCSF/GM/dGM), threshold={report.threshold} ===",
    ]
    if c.caveat:
        lines.append(f"CAVEAT (inherited from calibration failure above): {c.caveat}")
    for class_id in REFINEMENT_CLASSES:
        lines.append(
            f"{CLASS_NAMES[class_id]:>10}: "
            f"Dice {rc.mean_dice_before[class_id]:.4f} -> {rc.mean_dice_after[class_id]:.4f}, "
            f"ED {rc.mean_ed_before[class_id]:.2f} -> {rc.mean_ed_after[class_id]:.2f}"
        )
    return "\n".join(lines)


def write_report(report: EvaluationReport, output_dir: Path) -> str:
    # refinement csv + reliability plot only get written for comparative runs obviously
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(output_dir / "report.json", "w") as f:
        json.dump(_report_to_json_dict(report), f, indent=2)

    _write_per_case_csv(report, output_dir / "report_per_case.csv")

    if report.calibration is not None:
        _write_refinement_csv(report, output_dir / "report_refinement.csv")
        plot_reliability_diagram(
            np.array(report.calibration_bin_mean_uncertainty),
            np.array(report.calibration_bin_error_rate),
            np.array(report.calibration_bin_voxel_count),
            report.calibration,
            output_dir / "reliability_plot.png",
        )

    summary = render_console_summary(report)
    (output_dir / "report_summary.txt").write_text(summary)
    return summary


# 8 comparisons total (pre-specified, not p-hacked after the fact) - correcting
# with benjamini-hochberg since we're running that many tests at once

@dataclass
class SignificanceTests:
    comparisons: list[dict]
    alpha: float


def compute_significance_tests(report: EvaluationReport, alpha: float = 0.05) -> SignificanceTests:
    # paired wilcoxon for each: baseline vs comparative dice, before/after refinement
    # dice+ED per class, and the pooled target-class dice. all p-values get FDR corrected together
    if report.comparative_per_case is None or report.refinement_comparison is None:
        raise ValueError(
            "compute_significance_tests requires a comparative-model run -- "
            "report.comparative_per_case and report.refinement_comparison must both be present."
        )

    baseline_by_subject = {c["subject_id"]: c for c in report.baseline_per_case}
    comparative_by_subject = {c["subject_id"]: c for c in report.comparative_per_case}
    subjects = [c["subject_id"] for c in report.baseline_per_case]

    comparisons: list[tuple[str, np.ndarray, np.ndarray]] = []

    baseline_fg_dice = np.array([mean_dice(baseline_by_subject[s]["dice"]) for s in subjects])
    comparative_fg_dice = np.array([mean_dice(comparative_by_subject[s]["dice"]) for s in subjects])
    comparisons.append(("baseline_vs_comparative_mean_foreground_dice", baseline_fg_dice, comparative_fg_dice))

    per_case = report.refinement_comparison.per_case
    for class_id in REFINEMENT_CLASSES:
        before = np.array([c["dice_before"][class_id] for c in per_case])
        after = np.array([c["dice_after"][class_id] for c in per_case])
        comparisons.append((f"refinement_dice_{CLASS_NAMES[class_id]}", before, after))

    target_before = np.array([np.mean([c["dice_before"][cid] for cid in REFINEMENT_CLASSES]) for c in per_case])
    target_after = np.array([np.mean([c["dice_after"][cid] for cid in REFINEMENT_CLASSES]) for c in per_case])
    comparisons.append(("refinement_dice_target_mean", target_before, target_after))

    for class_id in REFINEMENT_CLASSES:
        before = np.array([c["ed_before"][class_id] for c in per_case])
        after = np.array([c["ed_after"][class_id] for c in per_case])
        comparisons.append((f"refinement_ed_{CLASS_NAMES[class_id]}", before, after))

    statistics: list[float] = []
    raw_pvalues: list[float] = []
    notes: list[str | None] = []
    for _name, before, after in comparisons:
        try:
            stat, pvalue = wilcoxon(before, after)
            statistics.append(float(stat))
            raw_pvalues.append(float(pvalue))
            notes.append(None)
        except ValueError as e:
            # wilcoxon throws if before==after for literally every case, just skip it then
            statistics.append(float("nan"))
            raw_pvalues.append(float("nan"))
            notes.append(str(e))

    # only correct over comparisons that actually gave us a p-value, obviously can't FDR a NaN
    valid_idx = [i for i, p in enumerate(raw_pvalues) if not np.isnan(p)]
    adjusted_pvalues = [float("nan")] * len(raw_pvalues)
    if valid_idx:
        corrected = false_discovery_control([raw_pvalues[i] for i in valid_idx], method="bh")
        for i, p_fdr in zip(valid_idx, corrected):
            adjusted_pvalues[i] = float(p_fdr)

    results = []
    for (name, before, _after), stat, p_raw, p_fdr, note in zip(
        comparisons, statistics, raw_pvalues, adjusted_pvalues, notes
    ):
        results.append(
            {
                "comparison": name,
                "n": len(before),
                "statistic": stat,
                "pvalue": p_raw,
                "pvalue_fdr_bh": p_fdr,
                "significant_raw": bool(p_raw < alpha) if not np.isnan(p_raw) else False,
                "significant_fdr": bool(p_fdr < alpha) if not np.isnan(p_fdr) else False,
                "note": note,
            }
        )

    return SignificanceTests(comparisons=results, alpha=alpha)


def render_significance_summary(tests: SignificanceTests) -> str:
    lines = [f"Significance tests -- paired Wilcoxon signed-rank, Benjamini-Hochberg FDR (alpha={tests.alpha})", ""]
    for r in tests.comparisons:
        if np.isnan(r["pvalue"]):
            flag = f"undefined ({r['note']})"
        elif r["significant_fdr"]:
            flag = "significant (FDR)"
        elif r["significant_raw"]:
            flag = "significant (raw only)"
        else:
            flag = "n.s."
        lines.append(f"{r['comparison']:<45} n={r['n']:>2}  p={r['pvalue']:.4g}  p_fdr={r['pvalue_fdr_bh']:.4g}  {flag}")
    return "\n".join(lines)


def write_significance_tests(tests: SignificanceTests, output_dir: Path) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(output_dir / "significance_tests.json", "w") as f:
        json.dump({"alpha": tests.alpha, "comparisons": tests.comparisons}, f, indent=2)

    (output_dir / "significance_tests.txt").write_text(render_significance_summary(tests))


# importing models inside the functions below, not at the top - so this file still
# imports fine even if role 2/3 are mid-refactor on their model files

def load_baseline_model(checkpoint_path: Path) -> SegmentationModel:
    # wraps role 2's BaselineUNet so it matches the SegmentationModel protocol
    from ..models.baseline import BaselineUNet

    net = BaselineUNet(in_channels=1, num_classes=NUM_CLASSES)
    state_dict = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    load_result = net.load_state_dict(state_dict, strict=True)
    net.eval()
    print(f"Loaded baseline checkpoint {checkpoint_path.name}: {load_result}")

    class _BaselineAdapter:
        def __init__(self, net: torch.nn.Module) -> None:
            self.net = net

        def predict_patch(self, patch: torch.Tensor) -> torch.Tensor:
            with torch.inference_mode():
                logits = self.net(patch.unsqueeze(0))  # add/remove the batch dim
            return torch.softmax(logits.squeeze(0), dim=0)

    return _BaselineAdapter(net)


def load_comparative_model(checkpoint_path: Path) -> UncertaintyModel:
    # wraps role 3's UncertaintyUNet. heads up: on this checkpoint the uncertainty
    # head itself is untrained (its loss term was just a placeholder during training)
    # and forward() never turns dropout on, so I'm using single-pass softmax entropy
    # as the uncertainty signal here instead of whatever the head outputs.
    # refine() just calls role 3's refine_prediction - enabled=False is a real no-op there.
    from ..models.uncertainty_unet import UncertaintyUNet, refine_prediction

    net = UncertaintyUNet(in_channels=1, num_classes=NUM_CLASSES)
    state_dict = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    load_result = net.load_state_dict(state_dict, strict=True)
    net.eval()
    print(f"Loaded comparative checkpoint {checkpoint_path.name}: {load_result}")

    class _ComparativeAdapter:
        def __init__(self, net: torch.nn.Module) -> None:
            self.net = net

        def predict_patch(self, patch: torch.Tensor) -> torch.Tensor:
            with torch.inference_mode():
                logits, _ = self.net(patch.unsqueeze(0))
            return torch.softmax(logits.squeeze(0), dim=0)

        def predict_patch_with_uncertainty(self, patch: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            with torch.inference_mode():
                logits, _ = self.net(patch.unsqueeze(0))
            probs = torch.softmax(logits.squeeze(0), dim=0)
            entropy = -(probs * torch.log(probs.clamp_min(1e-8))).sum(dim=0) / np.log(probs.shape[0])
            return probs, entropy

        def refine(
            self,
            probabilities: torch.Tensor,
            uncertainty: torch.Tensor,
            threshold: float,
            enabled: bool = True,
        ) -> torch.Tensor:
            hard = to_hard_prediction(probabilities)
            refined = refine_prediction(
                hard.unsqueeze(0), uncertainty.unsqueeze(0), threshold=threshold, enabled=enabled
            )
            return refined.squeeze(0).to(torch.int64)

    return _ComparativeAdapter(net)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate the baseline (and, if available, comparative) model on the frozen "
            "test split. Comparative is optional -- omit it for a baseline-only Dice/ED run."
        )
    )
    parser.add_argument("--split-path", type=Path, default=Path("src/data/splits/split_v1.json"))
    parser.add_argument("--config-path", type=Path, default=Path("src/data/config.yaml"))
    parser.add_argument("--baseline-checkpoint", type=Path, required=True)
    parser.add_argument(
        "--comparative-checkpoint",
        type=Path,
        default=None,
        help="Optional. Omit to run a baseline-only evaluation (no comparative model exists yet).",
    )
    parser.add_argument(
        "--refinement-threshold",
        type=float,
        default=None,
        help="Required if --comparative-checkpoint is given; selected by role 4 on the validation split.",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/full_evaluation"))
    parser.add_argument(
        "--max-cases",
        type=int,
        default=None,
        help="Limit to the first N test cases (debugging/smoke-testing only -- full 128^3 "
        "sliding-window inference on CPU is slow; omit for a real evaluation run).",
    )
    parser.add_argument(
        "--stride-fraction",
        type=float,
        default=1.0,
        help="Sliding-window tile stride as a fraction of patch_size. 1.0 (default here) = no "
        "intentional tile overlap -- still full voxel coverage, just fewer redundant tiles for "
        "volumes needing 3+ tiles per axis (real speedup on CPU, see default_stride's docstring). "
        "0.5 matches the nnU-Net convention (more tile-seam blending, slower).",
    )
    parser.add_argument(
        "--mc-dropout-validation-maps",
        action="store_true",
        help="Also run MC-dropout predictive-entropy uncertainty maps on the VALIDATION split (requires "
        "--comparative-checkpoint). Distinct from the test-split calibration section: this uses "
        "n-mc-samples stochastic forward passes per patch (real disagreement across samples), not a "
        "single deterministic pass. Slow -- roughly n_mc_samples x a single-pass case.",
    )
    parser.add_argument(
        "--n-mc-samples",
        type=int,
        default=10,
        help="Stochastic forward passes per patch for --mc-dropout-validation-maps.",
    )
    args = parser.parse_args(argv)

    if args.comparative_checkpoint is not None and args.refinement_threshold is None:
        parser.error("--refinement-threshold is required when --comparative-checkpoint is given.")

    return args


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)

    with open(args.config_path) as f:
        config = yaml.safe_load(f)
    patch_size = tuple(config["patch_sampling"]["patch_size"])
    aggregation = config["eval"]["patch_aggregation"]
    stride = default_stride(patch_size, args.stride_fraction)

    test_dataset: Iterable[EvalCase] = FeTADataset(args.split_path, args.config_path, split_name="test", mode="eval")
    if args.max_cases is not None:
        test_dataset = [test_dataset[i] for i in range(min(args.max_cases, len(test_dataset)))]

    baseline_model = load_baseline_model(args.baseline_checkpoint)
    comparative_model = (
        load_comparative_model(args.comparative_checkpoint) if args.comparative_checkpoint is not None else None
    )

    print(f"patch_size={patch_size} stride={stride} (stride_fraction={args.stride_fraction}) aggregation={aggregation!r}")

    t_start = time.time()
    result = run_full_evaluation(
        test_dataset, baseline_model, comparative_model, patch_size, stride, aggregation, args.refinement_threshold
    )
    elapsed = time.time() - t_start

    summary = write_report(result, args.output_dir)
    print(summary)
    print(f"\nTotal evaluation time: {elapsed:.1f}s")

    if comparative_model is not None:
        try:
            tests = compute_significance_tests(result)
            write_significance_tests(tests, args.output_dir)
            print(f"\n{render_significance_summary(tests)}")
        except ValueError as e:
            # probably just --max-cases being small for a smoke test, not enough pairs for wilcoxon
            print(f"\nSkipped significance testing: {e}")

    if args.mc_dropout_validation_maps:
        if comparative_model is None:
            parser_error = "--mc-dropout-validation-maps requires --comparative-checkpoint"
            raise SystemExit(parser_error)
        val_dataset = FeTADataset(args.split_path, args.config_path, split_name="val", mode="eval")
        if args.max_cases is not None:
            val_dataset = [val_dataset[i] for i in range(min(args.max_cases, len(val_dataset)))]

        print(f"\nRunning MC-dropout validation uncertainty maps ({args.n_mc_samples} samples/patch)...")
        t_start = time.time()
        unc_report = run_validation_uncertainty_maps(
            val_dataset,
            comparative_model.net,
            patch_size,
            stride,
            aggregation,
            n_mc_samples=args.n_mc_samples,
            maps_dir=args.output_dir / "validation_uncertainty_maps",
        )
        write_validation_uncertainty_report(unc_report, args.output_dir)
        print(
            f"Validation uncertainty maps: pooled Spearman rho={unc_report.pooled_spearman_rho:.3f} "
            f"(p={unc_report.pooled_spearman_pvalue:.3g}), {unc_report.n_cases} cases, "
            f"{time.time() - t_start:.1f}s"
        )


if __name__ == "__main__":
    main()
