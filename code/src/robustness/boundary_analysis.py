"""
Boundary-focused utilities for 3D segmentation uncertainty analysis.

Purpose
-------
Distinguish uncertainty that merely detects anatomical edges from uncertainty
that specifically identifies erroneous predictions near those edges.

The core comparison is:

    correct interior
    correct boundary
    incorrect boundary

If uncertainty is merely an edge detector, correct and incorrect boundary
voxels should have similarly elevated uncertainty.

If uncertainty carries useful error information, incorrect boundary voxels
should exhibit greater uncertainty than correct boundary voxels.
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import (
    binary_dilation,
    binary_erosion,
    distance_transform_edt,
    generate_binary_structure,
)
from scipy.stats import mannwhitneyu


def multiclass_boundary(
    segmentation: np.ndarray,
) -> np.ndarray:
    """
    Identify voxels adjacent to a class transition.

    Parameters
    ----------
    segmentation:
        Integer label map shaped (D, H, W).

    Returns
    -------
    Boolean array shaped (D, H, W).
    """
    segmentation = np.asarray(segmentation)

    if segmentation.ndim != 3:
        raise ValueError(
            f"Expected 3D segmentation, got shape {segmentation.shape}"
        )

    boundary = np.zeros_like(segmentation, dtype=bool)

    # Compare neighbouring voxels independently along each spatial axis.
    for axis in range(3):

        slicer_a = [slice(None)] * 3
        slicer_b = [slice(None)] * 3

        slicer_a[axis] = slice(1, None)
        slicer_b[axis] = slice(None, -1)

        a = tuple(slicer_a)
        b = tuple(slicer_b)

        different = segmentation[a] != segmentation[b]

        boundary[a] |= different
        boundary[b] |= different

    return boundary


def boundary_band(
    segmentation: np.ndarray,
    width_mm: float,
    spacing=(1.0, 1.0, 1.0),
) -> np.ndarray:
    """
    Construct a physical-distance band around multiclass boundaries.

    Parameters
    ----------
    segmentation:
        Integer label map shaped (D, H, W).

    width_mm:
        Maximum physical distance from a class transition.

    spacing:
        Voxel spacing in millimetres.

    Returns
    -------
    Boolean mask where True denotes a voxel within width_mm of a boundary.
    """
    if width_mm < 0:
        raise ValueError("width_mm must be non-negative.")

    boundary = multiclass_boundary(segmentation)

    if width_mm == 0:
        return boundary

    distance = distance_transform_edt(
        ~boundary,
        sampling=spacing,
    )

    return distance <= width_mm


def binary_class_boundary(
    segmentation: np.ndarray,
    class_id: int,
) -> np.ndarray:
    """
    Return the one-voxel boundary of one tissue class.
    """
    mask = np.asarray(segmentation) == class_id

    if not mask.any():
        return np.zeros_like(mask, dtype=bool)

    structure = generate_binary_structure(3, 1)

    inner = mask & ~binary_erosion(
        mask,
        structure=structure,
        border_value=0,
    )

    outer = binary_dilation(
        mask,
        structure=structure,
    ) & ~mask

    return inner | outer


def class_boundary_band(
    segmentation: np.ndarray,
    class_id: int,
    width_mm: float,
    spacing=(1.0, 1.0, 1.0),
) -> np.ndarray:
    """
    Construct a physical-distance band around one tissue class boundary.
    """
    boundary = binary_class_boundary(
        segmentation,
        class_id,
    )

    if not boundary.any():
        return boundary

    if width_mm == 0:
        return boundary

    distance = distance_transform_edt(
        ~boundary,
        sampling=spacing,
    )

    return distance <= width_mm


def boundary_error_groups(
    prediction: np.ndarray,
    target: np.ndarray,
    uncertainty: np.ndarray,
    width_mm: float = 1.0,
    spacing=(1.0, 1.0, 1.0),
    exclude_joint_background: bool = True,
) -> dict[str, np.ndarray]:
    """
    Divide uncertainty values into clinically useful groups.

    Groups
    ------
    correct_interior:
        Correct prediction outside the target boundary band.

    correct_boundary:
        Correct prediction within the target boundary band.

    incorrect_boundary:
        Incorrect prediction within the target boundary band.

    incorrect_interior:
        Incorrect prediction outside the target boundary band.
    """
    prediction = np.asarray(prediction)
    target = np.asarray(target)
    uncertainty = np.asarray(uncertainty, dtype=float)

    if prediction.shape != target.shape:
        raise ValueError(
            f"Prediction/target shape mismatch: "
            f"{prediction.shape} vs {target.shape}"
        )

    if uncertainty.shape != target.shape:
        raise ValueError(
            f"Uncertainty/target shape mismatch: "
            f"{uncertainty.shape} vs {target.shape}"
        )

    band = boundary_band(
        target,
        width_mm=width_mm,
        spacing=spacing,
    )

    correct = prediction == target
    incorrect = ~correct

    valid = np.isfinite(uncertainty)

    if exclude_joint_background:
        valid &= ~(
            (prediction == 0)
            & (target == 0)
        )

    groups = {
        "correct_interior": uncertainty[
            valid & correct & ~band
        ],
        "correct_boundary": uncertainty[
            valid & correct & band
        ],
        "incorrect_boundary": uncertainty[
            valid & incorrect & band
        ],
        "incorrect_interior": uncertainty[
            valid & incorrect & ~band
        ],
    }

    return groups


def summarize_boundary_groups(
    groups: dict[str, np.ndarray],
) -> dict[str, dict]:
    """
    Return descriptive statistics for each boundary/error group.
    """
    output = {}

    for name, values in groups.items():

        values = np.asarray(values, dtype=float)

        if len(values) == 0:
            output[name] = {
                "n": 0,
                "mean": float("nan"),
                "median": float("nan"),
                "q25": float("nan"),
                "q75": float("nan"),
            }
            continue

        output[name] = {
            "n": int(len(values)),
            "mean": float(np.mean(values)),
            "median": float(np.median(values)),
            "q25": float(np.percentile(values, 25)),
            "q75": float(np.percentile(values, 75)),
        }

    return output


def compare_correct_vs_incorrect_boundary(
    groups: dict[str, np.ndarray],
) -> dict:
    """
    Compare uncertainty at correct and incorrect boundary voxels.

    Uses a two-sided Mann-Whitney U test because uncertainty distributions
    are generally non-Gaussian and may contain very large voxel counts.
    """
    correct = np.asarray(
        groups["correct_boundary"],
        dtype=float,
    )

    incorrect = np.asarray(
        groups["incorrect_boundary"],
        dtype=float,
    )

    if len(correct) == 0 or len(incorrect) == 0:
        return {
            "u_statistic": float("nan"),
            "p_value": float("nan"),
            "median_correct": float("nan"),
            "median_incorrect": float("nan"),
            "median_difference": float("nan"),
        }

    result = mannwhitneyu(
        incorrect,
        correct,
        alternative="two-sided",
    )

    return {
        "u_statistic": float(result.statistic),
        "p_value": float(result.pvalue),
        "median_correct": float(np.median(correct)),
        "median_incorrect": float(np.median(incorrect)),
        "median_difference": float(
            np.median(incorrect)
            - np.median(correct)
        ),
    }


if __name__ == "__main__":

    print("=" * 70)
    print("BOUNDARY ANALYSIS SANITY TESTS")
    print("=" * 70)

    spacing = (0.5, 0.5, 0.5)

    # ------------------------------------------------------------
    # Synthetic target:
    # central cube of class 1 inside background.
    # ------------------------------------------------------------

    target = np.zeros(
        (32, 32, 32),
        dtype=np.int64,
    )

    target[8:24, 8:24, 8:24] = 1

    boundary = multiclass_boundary(target)

    band_05 = boundary_band(
        target,
        width_mm=0.5,
        spacing=spacing,
    )

    band_10 = boundary_band(
        target,
        width_mm=1.0,
        spacing=spacing,
    )

    print()
    print("1. Boundary-band size")
    print(f"   Exact boundary voxels: {boundary.sum()}")
    print(f"   0.5 mm band voxels:    {band_05.sum()}")
    print(f"   1.0 mm band voxels:    {band_10.sum()}")

    assert boundary.sum() > 0
    assert band_05.sum() >= boundary.sum()
    assert band_10.sum() >= band_05.sum()

    # ------------------------------------------------------------
    # Synthetic prediction with deliberate boundary errors.
    # ------------------------------------------------------------

    prediction = target.copy()

    # Shift part of one cube face inward.
    prediction[8, 10:20, 10:20] = 0

    # Create uncertainty:
    # low interior,
    # medium correct boundary,
    # high incorrect boundary.
    uncertainty = np.full(
        target.shape,
        0.05,
        dtype=float,
    )

    band = boundary_band(
        target,
        width_mm=1.0,
        spacing=spacing,
    )

    uncertainty[band] = 0.40

    error = prediction != target
    uncertainty[error & band] = 0.90

    groups = boundary_error_groups(
        prediction,
        target,
        uncertainty,
        width_mm=1.0,
        spacing=spacing,
        exclude_joint_background=True,
    )

    summary = summarize_boundary_groups(groups)

    print()
    print("2. Boundary/error uncertainty groups")

    for name, stats in summary.items():
        print(
            f"   {name:20s}: "
            f"n={stats['n']:5d}, "
            f"median={stats['median']:.3f}"
        )

    assert (
        summary["incorrect_boundary"]["median"]
        >
        summary["correct_boundary"]["median"]
    )

    assert (
        summary["correct_boundary"]["median"]
        >
        summary["correct_interior"]["median"]
    )

    comparison = compare_correct_vs_incorrect_boundary(
        groups
    )

    print()
    print("3. Correct vs incorrect boundary")
    print(
        f"   Median correct boundary uncertainty:   "
        f"{comparison['median_correct']:.3f}"
    )
    print(
        f"   Median incorrect boundary uncertainty: "
        f"{comparison['median_incorrect']:.3f}"
    )
    print(
        f"   Median difference:                     "
        f"{comparison['median_difference']:.3f}"
    )

    print()
    print("BOUNDARY ANALYSIS TESTS PASSED")
