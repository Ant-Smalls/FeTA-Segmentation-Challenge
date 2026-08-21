"""
Evaluation utilities for voxel-wise segmentation uncertainty.

The central question is whether a voxel assigned higher uncertainty is more
likely to be incorrectly segmented.

Metrics
-------
AUROC
    Measures ranking quality across correct and incorrect voxels.
    0.5 = random ranking, 1.0 = perfect ranking.

AUPRC / Average Precision
    Particularly useful because segmentation errors may represent a small
    fraction of all voxels.

Spearman correlation
    Measures monotonic association between uncertainty and binary error.

Uncertainty bins
    Groups voxels by uncertainty and reports the empirical error rate in each
    bin. A useful uncertainty signal should generally show increasing error
    rates as uncertainty increases.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.stats import rankdata, spearmanr


@dataclass
class ErrorDetectionResult:
    n_voxels: int
    n_errors: int
    error_rate: float
    auroc: float
    average_precision: float
    spearman_rho: float
    spearman_p: float


def _prepare_arrays(
    uncertainty: np.ndarray,
    error_mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Flatten and validate uncertainty and binary error arrays.
    """
    uncertainty = np.asarray(uncertainty, dtype=float)
    error_mask = np.asarray(error_mask)

    if uncertainty.shape != error_mask.shape:
        raise ValueError(
            f"Shape mismatch: uncertainty={uncertainty.shape}, "
            f"error_mask={error_mask.shape}"
        )

    uncertainty = uncertainty.ravel()
    error_mask = error_mask.ravel()

    valid = np.isfinite(uncertainty) & np.isfinite(error_mask)

    uncertainty = uncertainty[valid]
    error_mask = error_mask[valid]

    if len(uncertainty) == 0:
        raise ValueError("No finite uncertainty values remain.")

    unique_errors = np.unique(error_mask)

    if not np.all(np.isin(unique_errors, [0, 1, False, True])):
        raise ValueError(
            "error_mask must contain only binary values 0/1 or False/True."
        )

    error_mask = error_mask.astype(np.int64)

    return uncertainty, error_mask


def binary_auroc(
    scores: np.ndarray,
    labels: np.ndarray,
) -> float:
    """
    Compute AUROC using the Mann-Whitney rank formulation.

    Higher score must indicate greater likelihood of label=1.
    """
    scores, labels = _prepare_arrays(scores, labels)

    n_positive = int(labels.sum())
    n_negative = len(labels) - n_positive

    if n_positive == 0 or n_negative == 0:
        return float("nan")

    ranks = rankdata(scores, method="average")

    positive_rank_sum = ranks[labels == 1].sum()

    u_statistic = (
        positive_rank_sum
        - n_positive * (n_positive + 1) / 2
    )

    auroc = u_statistic / (n_positive * n_negative)

    return float(auroc)


def average_precision(
    scores: np.ndarray,
    labels: np.ndarray,
) -> float:
    """
    Compute average precision from ranked uncertainty scores.

    This is the area under the precision-recall curve using the standard
    step-wise average-precision formulation.
    """
    scores, labels = _prepare_arrays(scores, labels)

    n_positive = int(labels.sum())

    if n_positive == 0:
        return float("nan")

    order = np.argsort(-scores, kind="mergesort")

    sorted_labels = labels[order]

    true_positives = np.cumsum(sorted_labels)
    ranks = np.arange(1, len(sorted_labels) + 1)

    precision = true_positives / ranks

    ap = precision[sorted_labels == 1].sum() / n_positive

    return float(ap)


def evaluate_error_detection(
    uncertainty: np.ndarray,
    error_mask: np.ndarray,
) -> ErrorDetectionResult:
    """
    Evaluate how well uncertainty detects segmentation errors.
    """
    uncertainty, error_mask = _prepare_arrays(
        uncertainty,
        error_mask,
    )

    n_errors = int(error_mask.sum())

    rho, p = spearmanr(
        uncertainty,
        error_mask,
    )

    return ErrorDetectionResult(
        n_voxels=len(error_mask),
        n_errors=n_errors,
        error_rate=float(error_mask.mean()),
        auroc=binary_auroc(
            uncertainty,
            error_mask,
        ),
        average_precision=average_precision(
            uncertainty,
            error_mask,
        ),
        spearman_rho=float(rho),
        spearman_p=float(p),
    )


def uncertainty_bins(
    uncertainty: np.ndarray,
    error_mask: np.ndarray,
    n_bins: int = 10,
) -> list[dict]:
    """
    Divide voxels into equal-count uncertainty bins.

    Equal-count bins are preferable here to fixed-width bins because medical
    segmentation uncertainty distributions are often highly skewed.

    Returns one dictionary per bin containing:
        uncertainty_min
        uncertainty_max
        uncertainty_mean
        error_rate
        n_voxels
        n_errors
    """
    if n_bins < 2:
        raise ValueError("n_bins must be at least 2.")

    uncertainty, error_mask = _prepare_arrays(
        uncertainty,
        error_mask,
    )

    order = np.argsort(uncertainty)

    uncertainty = uncertainty[order]
    error_mask = error_mask[order]

    index_bins = np.array_split(
        np.arange(len(uncertainty)),
        n_bins,
    )

    results = []

    for bin_number, indices in enumerate(index_bins, start=1):
        if len(indices) == 0:
            continue

        u = uncertainty[indices]
        e = error_mask[indices]

        results.append(
            {
                "bin": bin_number,
                "uncertainty_min": float(u.min()),
                "uncertainty_max": float(u.max()),
                "uncertainty_mean": float(u.mean()),
                "error_rate": float(e.mean()),
                "n_voxels": int(len(e)),
                "n_errors": int(e.sum()),
            }
        )

    return results


def prediction_error_mask(
    prediction: np.ndarray,
    target: np.ndarray,
    ignore_background: bool = False,
) -> np.ndarray:
    """
    Return a binary mask where 1 indicates an incorrect predicted class.

    If ignore_background=True, voxels where both prediction and target are
    background are excluded by returning NaN. This option is useful when
    evaluating foreground segmentation uncertainty without allowing the
    very large correctly classified background to dominate the analysis.
    """
    prediction = np.asarray(prediction)
    target = np.asarray(target)

    if prediction.shape != target.shape:
        raise ValueError(
            f"Shape mismatch: prediction={prediction.shape}, "
            f"target={target.shape}"
        )

    error = (prediction != target).astype(float)

    if ignore_background:
        jointly_background = (
            (prediction == 0)
            & (target == 0)
        )

        error[jointly_background] = np.nan

    return error


if __name__ == "__main__":

    print("=" * 70)
    print("ERROR-DETECTION METRIC SANITY TESTS")
    print("=" * 70)

    # ------------------------------------------------------------
    # Test 1: Perfect uncertainty ranking.
    # Errors receive the highest uncertainty values.
    # ------------------------------------------------------------

    errors = np.array(
        [0, 0, 0, 0, 1, 1, 1, 1],
        dtype=int,
    )

    perfect_uncertainty = np.array(
        [0.1, 0.2, 0.3, 0.4, 0.6, 0.7, 0.8, 0.9],
        dtype=float,
    )

    result = evaluate_error_detection(
        perfect_uncertainty,
        errors,
    )

    print()
    print("1. Perfect ranking")
    print(f"   AUROC:             {result.auroc:.6f}")
    print(f"   Average precision: {result.average_precision:.6f}")
    print(f"   Spearman rho:      {result.spearman_rho:.6f}")

    assert np.isclose(result.auroc, 1.0)
    assert np.isclose(result.average_precision, 1.0)


    # ------------------------------------------------------------
    # Test 2: Perfectly reversed ranking.
    # ------------------------------------------------------------

    reversed_uncertainty = perfect_uncertainty[::-1]

    result_reversed = evaluate_error_detection(
        reversed_uncertainty,
        errors,
    )

    print()
    print("2. Reversed ranking")
    print(f"   AUROC:             {result_reversed.auroc:.6f}")
    print(
        f"   Average precision: "
        f"{result_reversed.average_precision:.6f}"
    )
    print(
        f"   Spearman rho:      "
        f"{result_reversed.spearman_rho:.6f}"
    )

    assert np.isclose(result_reversed.auroc, 0.0)


    # ------------------------------------------------------------
    # Test 3: Equal-count uncertainty bins.
    # ------------------------------------------------------------

    bins = uncertainty_bins(
        perfect_uncertainty,
        errors,
        n_bins=4,
    )

    print()
    print("3. Uncertainty bins")

    for row in bins:
        print(
            f"   Bin {row['bin']}: "
            f"mean uncertainty={row['uncertainty_mean']:.3f}, "
            f"error rate={row['error_rate']:.3f}"
        )

    error_rates = [
        row["error_rate"]
        for row in bins
    ]

    assert error_rates[-1] > error_rates[0]


    # ------------------------------------------------------------
    # Test 4: Prediction error mask.
    # ------------------------------------------------------------

    target = np.array([0, 1, 1, 2, 2])
    prediction = np.array([0, 1, 2, 2, 1])

    mask = prediction_error_mask(
        prediction,
        target,
    )

    expected = np.array(
        [0, 0, 1, 0, 1],
        dtype=float,
    )

    assert np.array_equal(mask, expected)

    print()
    print("ERROR-DETECTION METRIC TESTS PASSED")
