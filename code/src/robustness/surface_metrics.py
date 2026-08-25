"""
Surface-distance metrics for 3D segmentation.

All distances are returned in physical units determined by `spacing`.
For the frozen FeTA evaluation pipeline this will normally be
(0.5, 0.5, 0.5) mm.

Metrics
-------
HD95:
    95th percentile symmetric Hausdorff distance.
    Lower is better.

ASSD:
    Average symmetric surface distance.
    Lower is better.
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import (
    binary_erosion,
    distance_transform_edt,
    generate_binary_structure,
)


def _surface(mask: np.ndarray) -> np.ndarray:
    """Return a one-voxel-wide 3D surface."""
    mask = np.asarray(mask, dtype=bool)

    structure = generate_binary_structure(3, 1)

    eroded = binary_erosion(
        mask,
        structure=structure,
        border_value=0,
    )

    return mask & ~eroded


def symmetric_surface_distances(
    prediction: np.ndarray,
    target: np.ndarray,
    spacing=(1.0, 1.0, 1.0),
) -> np.ndarray:
    """
    Compute bidirectional distances between prediction and target surfaces.
    """
    prediction = np.asarray(prediction, dtype=bool)
    target = np.asarray(target, dtype=bool)

    if prediction.shape != target.shape:
        raise ValueError(
            f"Shape mismatch: {prediction.shape} vs {target.shape}"
        )

    pred_nonempty = prediction.any()
    target_nonempty = target.any()

    # Both absent -> perfect agreement for this class.
    if not pred_nonempty and not target_nonempty:
        return np.array([0.0])

    # One absent -> surface distance is undefined/infinite.
    if pred_nonempty != target_nonempty:
        return np.array([np.inf])

    pred_surface = _surface(prediction)
    target_surface = _surface(target)

    # EDT gives distance to the nearest zero voxel.
    # In ~surface, surface voxels are zero.
    distance_to_target = distance_transform_edt(
        ~target_surface,
        sampling=spacing,
    )

    distance_to_prediction = distance_transform_edt(
        ~pred_surface,
        sampling=spacing,
    )

    pred_to_target = distance_to_target[pred_surface]
    target_to_pred = distance_to_prediction[target_surface]

    return np.concatenate(
        [pred_to_target, target_to_pred]
    )


def hd95(
    prediction: np.ndarray,
    target: np.ndarray,
    spacing=(1.0, 1.0, 1.0),
) -> float:
    """95th-percentile symmetric Hausdorff distance."""
    distances = symmetric_surface_distances(
        prediction,
        target,
        spacing,
    )

    if np.isinf(distances).any():
        return float("inf")

    return float(np.percentile(distances, 95))


def assd(
    prediction: np.ndarray,
    target: np.ndarray,
    spacing=(1.0, 1.0, 1.0),
) -> float:
    """Average symmetric surface distance."""
    distances = symmetric_surface_distances(
        prediction,
        target,
        spacing,
    )

    if np.isinf(distances).any():
        return float("inf")

    return float(np.mean(distances))


if __name__ == "__main__":

    # -------------------------------------------------------
    # Synthetic sanity test
    # -------------------------------------------------------

    gt = np.zeros((32, 32, 32), dtype=bool)
    pred = np.zeros_like(gt)

    # Identical cubes
    gt[8:24, 8:24, 8:24] = True
    pred[8:24, 8:24, 8:24] = True

    spacing = (0.5, 0.5, 0.5)

    print("=== IDENTICAL MASKS ===")
    print("HD95:", hd95(pred, gt, spacing), "mm")
    print("ASSD:", assd(pred, gt, spacing), "mm")

    assert hd95(pred, gt, spacing) == 0.0
    assert assd(pred, gt, spacing) == 0.0

    # Shift prediction by one voxel along one axis.
    pred_shifted = np.zeros_like(gt)
    pred_shifted[9:25, 8:24, 8:24] = True

    shifted_hd95 = hd95(
        pred_shifted,
        gt,
        spacing,
    )

    shifted_assd = assd(
        pred_shifted,
        gt,
        spacing,
    )

    print()
    print("=== ONE-VOXEL SHIFT ===")
    print("HD95:", shifted_hd95, "mm")
    print("ASSD:", shifted_assd, "mm")

    assert shifted_hd95 > 0
    assert shifted_assd > 0

    print()
    print("SURFACE METRIC TESTS PASSED ")
