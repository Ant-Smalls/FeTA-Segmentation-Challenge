"""
Uncertainty metrics for MC-Dropout segmentation.

This module separates three concepts that should not be conflated:

1. Predictive entropy
   Uncertainty of the mean class-probability distribution.

2. Expected entropy
   Mean entropy of individual stochastic predictions.

3. Mutual information
   Predictive entropy - expected entropy.
   Measures disagreement between stochastic model samples and is commonly
   interpreted as an epistemic/model-uncertainty component.

All entropy values are normalised by log(C), where C is the number of classes,
so values lie approximately in [0, 1].
"""

from __future__ import annotations

import torch


EPS = 1e-8


def _entropy(probabilities: torch.Tensor, class_dim: int) -> torch.Tensor:
    """
    Shannon entropy normalised by log(number of classes).
    """
    probabilities = probabilities.clamp_min(EPS)

    entropy = -torch.sum(
        probabilities * torch.log(probabilities),
        dim=class_dim,
    )

    n_classes = probabilities.shape[class_dim]

    normaliser = torch.log(
        torch.tensor(
            float(n_classes),
            dtype=probabilities.dtype,
            device=probabilities.device,
        )
    )

    return entropy / normaliser


def predictive_entropy(
    probability_samples: torch.Tensor,
) -> torch.Tensor:
    """
    Predictive entropy from MC probability samples.

    Parameters
    ----------
    probability_samples:
        Tensor shaped:
            (T, B, C, D, H, W)

        T = number of MC samples
        C = number of classes

    Returns
    -------
    Tensor:
        (B, D, H, W)

    Interpretation
    --------------
    Total uncertainty in the mean predictive distribution.
    Includes both class ambiguity and stochastic model disagreement.
    """
    mean_probability = probability_samples.mean(dim=0)

    return _entropy(
        mean_probability,
        class_dim=1,
    )


def expected_entropy(
    probability_samples: torch.Tensor,
) -> torch.Tensor:
    """
    Mean entropy of the individual MC predictions.

    Returns:
        (B, D, H, W)

    High expected entropy means individual stochastic models are themselves
    uncertain between classes.
    """
    sample_entropies = _entropy(
        probability_samples,
        class_dim=2,
    )

    return sample_entropies.mean(dim=0)


def mutual_information(
    probability_samples: torch.Tensor,
) -> torch.Tensor:
    """
    MC-Dropout mutual information.

        MI = predictive entropy - expected entropy

    Returns:
        (B, D, H, W)

    Large MI means the stochastic model samples disagree with one another,
    providing a more specific measure of epistemic/model uncertainty.
    """
    mi = (
        predictive_entropy(probability_samples)
        - expected_entropy(probability_samples)
    )

    # Tiny negative values can occur because of floating-point precision.
    return mi.clamp_min(0.0)


def mean_probability_variance(
    probability_samples: torch.Tensor,
) -> torch.Tensor:
    """
    Mean variance across classes for MC probability samples.

    Returns:
        (B, D, H, W)
    """
    class_variance = probability_samples.var(
        dim=0,
        unbiased=False,
    )

    return class_variance.mean(dim=1)


if __name__ == "__main__":

    print("=" * 70)
    print("UNCERTAINTY METRIC SANITY TESTS")
    print("=" * 70)

    # ------------------------------------------------------------
    # CASE 1
    # All stochastic models confidently agree.
    # ------------------------------------------------------------

    confident_agreement = torch.tensor(
        [
            [[[[[1.0]]], [[[0.0]]]]],
            [[[[[1.0]]], [[[0.0]]]]],
            [[[[[1.0]]], [[[0.0]]]]],
            [[[[[1.0]]], [[[0.0]]]]],
        ],
        dtype=torch.float32,
    )

    pe = predictive_entropy(confident_agreement).item()
    ee = expected_entropy(confident_agreement).item()
    mi = mutual_information(confident_agreement).item()

    print()
    print("1. Confident agreement")
    print(f"   Predictive entropy: {pe:.6f}")
    print(f"   Expected entropy:   {ee:.6f}")
    print(f"   Mutual information: {mi:.6f}")

    # ------------------------------------------------------------
    # CASE 2
    # All stochastic models give the SAME ambiguous prediction.
    #
    # This is important:
    # predictive entropy is high, but MC models do not disagree.
    # ------------------------------------------------------------

    ambiguous_agreement = torch.tensor(
        [
            [[[[[0.5]]], [[[0.5]]]]],
            [[[[[0.5]]], [[[0.5]]]]],
            [[[[[0.5]]], [[[0.5]]]]],
            [[[[[0.5]]], [[[0.5]]]]],
        ],
        dtype=torch.float32,
    )

    pe2 = predictive_entropy(ambiguous_agreement).item()
    ee2 = expected_entropy(ambiguous_agreement).item()
    mi2 = mutual_information(ambiguous_agreement).item()

    print()
    print("2. Ambiguous but unanimous")
    print(f"   Predictive entropy: {pe2:.6f}")
    print(f"   Expected entropy:   {ee2:.6f}")
    print(f"   Mutual information: {mi2:.6f}")

    # ------------------------------------------------------------
    # CASE 3
    # Individual stochastic models are confident,
    # but half choose class 0 and half class 1.
    # ------------------------------------------------------------

    confident_disagreement = torch.tensor(
        [
            [[[[[1.0]]], [[[0.0]]]]],
            [[[[[1.0]]], [[[0.0]]]]],
            [[[[[0.0]]], [[[1.0]]]]],
            [[[[[0.0]]], [[[1.0]]]]],
        ],
        dtype=torch.float32,
    )

    pe3 = predictive_entropy(confident_disagreement).item()
    ee3 = expected_entropy(confident_disagreement).item()
    mi3 = mutual_information(confident_disagreement).item()

    print()
    print("3. Confident disagreement")
    print(f"   Predictive entropy: {pe3:.6f}")
    print(f"   Expected entropy:   {ee3:.6f}")
    print(f"   Mutual information: {mi3:.6f}")

    # ------------------------------------------------------------
    # Assertions
    # ------------------------------------------------------------

    assert pe < 1e-5
    assert mi < 1e-5

    assert pe2 > 0.99
    assert ee2 > 0.99
    assert mi2 < 1e-5

    assert pe3 > 0.99
    assert ee3 < 1e-5
    assert mi3 > 0.99

    print()
    print("UNCERTAINTY METRIC TESTS PASSED ")
