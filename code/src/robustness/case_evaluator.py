"""
Unified robustness evaluation for one 3D segmentation case.

This module combines:
- Dice
- HD95
- ASSD
- uncertainty-based error detection
- boundary-specific uncertainty analysis

It does not perform model inference. It accepts already generated
predictions, ground truth labels, and optional uncertainty maps.
"""

from __future__ import annotations

from dataclasses import asdict
from typing import Optional

import numpy as np

from src.robustness.surface_metrics import hd95, assd
from src.robustness.error_detection import (
    evaluate_error_detection,
)
from src.robustness.boundary_analysis import (
    boundary_error_groups,
    summarize_boundary_groups,
)


DEFAULT_CLASS_NAMES = {
    1: "eCSF",
    2: "GM",
    3: "WM",
    4: "Ventricles",
    5: "Cerebellum",
    6: "dGM",
    7: "Brainstem",
}


def dice_score(
    prediction: np.ndarray,
    target: np.ndarray,
) -> float:
    """
    Dice similarity coefficient for binary masks.

    Returns:
        1.0 if both masks are empty.
    """
    prediction = np.asarray(
        prediction,
        dtype=bool,
    )

    target = np.asarray(
        target,
        dtype=bool,
    )

    intersection = np.logical_and(
        prediction,
        target,
    ).sum()

    denominator = (
        prediction.sum()
        + target.sum()
    )

    if denominator == 0:
        return 1.0

    return float(
        2.0 * intersection / denominator
    )


def evaluate_segmentation_case(
    prediction: np.ndarray,
    target: np.ndarray,
    spacing=(0.5, 0.5, 0.5),
    class_names: Optional[dict[int, str]] = None,
) -> dict:
    """
    Compute per-class Dice, HD95 and ASSD.

    Parameters
    ----------
    prediction:
        Integer predicted segmentation, shape (D, H, W).

    target:
        Integer ground-truth segmentation, shape (D, H, W).

    spacing:
        Physical voxel spacing in millimetres.

    class_names:
        Mapping from class ID to display name.
    """
    prediction = np.asarray(prediction)
    target = np.asarray(target)

    if prediction.shape != target.shape:
        raise ValueError(
            f"Shape mismatch: "
            f"{prediction.shape} vs {target.shape}"
        )

    if class_names is None:
        class_names = DEFAULT_CLASS_NAMES

    per_class = {}

    for class_id, class_name in class_names.items():

        pred_mask = prediction == class_id
        target_mask = target == class_id

        per_class[class_name] = {
            "class_id": class_id,
            "dice": dice_score(
                pred_mask,
                target_mask,
            ),
            "hd95_mm": hd95(
                pred_mask,
                target_mask,
                spacing=spacing,
            ),
            "assd_mm": assd(
                pred_mask,
                target_mask,
                spacing=spacing,
            ),
            "prediction_voxels": int(
                pred_mask.sum()
            ),
            "target_voxels": int(
                target_mask.sum()
            ),
        }

    finite_dice = [
        values["dice"]
        for values in per_class.values()
        if np.isfinite(values["dice"])
    ]

    finite_hd95 = [
        values["hd95_mm"]
        for values in per_class.values()
        if np.isfinite(values["hd95_mm"])
    ]

    finite_assd = [
        values["assd_mm"]
        for values in per_class.values()
        if np.isfinite(values["assd_mm"])
    ]

    return {
        "mean_foreground_dice": float(
            np.mean(finite_dice)
        ),
        "mean_hd95_mm": (
            float(np.mean(finite_hd95))
            if finite_hd95
            else float("nan")
        ),
        "mean_assd_mm": (
            float(np.mean(finite_assd))
            if finite_assd
            else float("nan")
        ),
        "per_class": per_class,
    }


def evaluate_uncertainty_case(
    prediction: np.ndarray,
    target: np.ndarray,
    uncertainty: np.ndarray,
    boundary_width_mm: float = 1.0,
    spacing=(0.5, 0.5, 0.5),
) -> dict:
    """
    Evaluate one voxel-wise uncertainty signal.

    Produces:
    - foreground-relevant error-detection statistics
    - boundary-group uncertainty statistics
    """
    prediction = np.asarray(prediction)
    target = np.asarray(target)
    uncertainty = np.asarray(
        uncertainty,
        dtype=float,
    )

    if prediction.shape != target.shape:
        raise ValueError(
            "Prediction and target shapes differ."
        )

    if uncertainty.shape != target.shape:
        raise ValueError(
            "Uncertainty and target shapes differ."
        )

    error = (
        prediction != target
    ).astype(float)

    # Exclude voxels where prediction and reference
    # are both background.
    foreground_relevant = ~(
        (prediction == 0)
        & (target == 0)
    )

    uncertainty_fg = uncertainty[
        foreground_relevant
    ]

    error_fg = error[
        foreground_relevant
    ]

    detection = evaluate_error_detection(
        uncertainty_fg,
        error_fg,
    )

    groups = boundary_error_groups(
        prediction=prediction,
        target=target,
        uncertainty=uncertainty,
        width_mm=boundary_width_mm,
        spacing=spacing,
        exclude_joint_background=True,
    )

    group_summary = summarize_boundary_groups(
        groups
    )

    correct_boundary = groups[
        "correct_boundary"
    ]

    incorrect_boundary = groups[
        "incorrect_boundary"
    ]

    if (
        len(correct_boundary) > 0
        and len(incorrect_boundary) > 0
    ):
        boundary_difference = float(
            np.median(
                incorrect_boundary
            )
            - np.median(
                correct_boundary
            )
        )
    else:
        boundary_difference = float(
            "nan"
        )

    return {
        "foreground_error_detection": asdict(
            detection
        ),
        "boundary_width_mm": float(
            boundary_width_mm
        ),
        "boundary_groups": group_summary,
        "incorrect_minus_correct_boundary_median": (
            boundary_difference
        ),
    }


def evaluate_case(
    subject_id: str,
    prediction: np.ndarray,
    target: np.ndarray,
    spacing=(0.5, 0.5, 0.5),
    uncertainty_maps: Optional[
        dict[str, np.ndarray]
    ] = None,
    boundary_width_mm: float = 1.0,
) -> dict:
    """
    Unified entry point for one subject.

    uncertainty_maps may contain any named uncertainty signals, e.g.:

        {
            "single_entropy": array,
            "mc_predictive_entropy": array,
            "mc_mutual_information": array,
            "learned_head": array,
        }
    """
    result = {
        "subject_id": subject_id,
        "spacing_mm": list(spacing),
        "segmentation": (
            evaluate_segmentation_case(
                prediction,
                target,
                spacing=spacing,
            )
        ),
        "uncertainty": {},
    }

    if uncertainty_maps:

        for name, uncertainty in (
            uncertainty_maps.items()
        ):

            result["uncertainty"][name] = (
                evaluate_uncertainty_case(
                    prediction=prediction,
                    target=target,
                    uncertainty=uncertainty,
                    boundary_width_mm=(
                        boundary_width_mm
                    ),
                    spacing=spacing,
                )
            )

    return result


if __name__ == "__main__":

    print("=" * 70)
    print("CASE EVALUATOR SANITY TEST")
    print("=" * 70)

    spacing = (
        0.5,
        0.5,
        0.5,
    )

    # ----------------------------------------------------------
    # Synthetic ground truth
    # ----------------------------------------------------------

    target = np.zeros(
        (32, 32, 32),
        dtype=np.int64,
    )

    target[
        8:24,
        8:24,
        8:24,
    ] = 1

    # ----------------------------------------------------------
    # Prediction with a small boundary error
    # ----------------------------------------------------------

    prediction = target.copy()

    prediction[
        8,
        10:20,
        10:20,
    ] = 0

    # ----------------------------------------------------------
    # Synthetic uncertainty:
    # low everywhere, high where the prediction is wrong.
    # ----------------------------------------------------------

    uncertainty = np.full(
        target.shape,
        0.05,
        dtype=float,
    )

    uncertainty[
        prediction != target
    ] = 0.90

    result = evaluate_case(
        subject_id="synthetic",
        prediction=prediction,
        target=target,
        spacing=spacing,
        uncertainty_maps={
            "synthetic_uncertainty": (
                uncertainty
            )
        },
        boundary_width_mm=1.0,
    )

    ecsf = result[
        "segmentation"
    ]["per_class"]["eCSF"]

    uncertainty_result = result[
        "uncertainty"
    ]["synthetic_uncertainty"]

    print()
    print("Segmentation")
    print(
        f"   eCSF Dice: "
        f"{ecsf['dice']:.6f}"
    )
    print(
        f"   eCSF HD95: "
        f"{ecsf['hd95_mm']:.6f} mm"
    )
    print(
        f"   eCSF ASSD: "
        f"{ecsf['assd_mm']:.6f} mm"
    )

    detection = uncertainty_result[
        "foreground_error_detection"
    ]

    print()
    print("Error detection")
    print(
        f"   AUROC: "
        f"{detection['auroc']:.6f}"
    )
    print(
        f"   Average precision: "
        f"{detection['average_precision']:.6f}"
    )

    boundary_delta = uncertainty_result[
        "incorrect_minus_correct_boundary_median"
    ]

    print()
    print("Boundary discrimination")
    print(
        f"   Incorrect minus correct "
        f"boundary median uncertainty: "
        f"{boundary_delta:.6f}"
    )

    assert ecsf["dice"] < 1.0
    assert detection["auroc"] > 0.9
    assert boundary_delta > 0

    print()
    print("CASE EVALUATOR TEST PASSED")
