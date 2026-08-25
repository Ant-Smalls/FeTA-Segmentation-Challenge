# Uncertainty Pipeline Audit

## Purpose

The current comparative FeTA pipeline contains multiple uncertainty signals.
These must be distinguished explicitly because they are generated differently,
have different interpretations, and are used for different parts of the
experiment.

## 1. Learned uncertainty head

`UncertaintyUNet` contains a single-channel uncertainty head branching from the
final decoder representation.

During training, its output is interpreted as voxel-wise log-variance and is
optimised using a heteroscedastic negative log-likelihood term:

    L = L_CE + L_Dice + w_u * L_NLL

with:

    uncertainty_loss_weight = 0.1

This output is therefore a learned heteroscedastic uncertainty signal.

The final evaluation currently reports a direct two-case correlation between
this raw uncertainty-head output and segmentation error of approximately:

    Spearman rho = 0.144

This is substantially weaker than the main calibration signal used elsewhere in
the evaluation.

## 2. Single-pass predictive entropy

The current tuned evaluation report states that the headline calibration result:

    Spearman rho = 0.417

is based on softmax predictive entropy rather than the learned uncertainty head.

This signal can be produced from a single segmentation probability map:

    p = softmax(logits)

    H(p) = -sum_c p_c log(p_c)

after normalisation by log(C).

This measures ambiguity in the predicted class distribution but does not require
Monte Carlo Dropout.

## 3. MC-Dropout predictive entropy

Refinement threshold tuning uses Monte Carlo Dropout.

For each patch:

1. Dropout is enabled during inference.
2. The model is run multiple times.
3. Softmax probabilities are averaged.
4. Mean probability maps are stitched into the full volume.
5. Predictive entropy is computed from the stitched mean probabilities.
6. Voxels above the validation-selected uncertainty threshold are eligible for
   boundary refinement.

The current configuration uses 10 stochastic passes and a selected threshold of
0.5.

The learned uncertainty head is not used by this refinement procedure.

## 4. MC-Dropout disagreement metrics

The current pipeline does not separately report model disagreement across MC
samples.

The robustness analysis therefore introduces:

- predictive entropy;
- expected entropy;
- mutual information;
- probability variance.

Mutual information is particularly useful because it separates stochastic model
disagreement from ambiguity that is present in every individual prediction.

## 5. Explainability inconsistency

The refinement-threshold implementation calculates predictive entropy after
full-volume probability reconstruction:

    patch probabilities
        -> mean patch probabilities
        -> probability stitching
        -> full-volume predictive entropy

The current explainability implementation instead calculates entropy at the
patch level and then stitches the entropy maps:

    patch probabilities
        -> mean patch probabilities
        -> patch entropy
        -> entropy stitching

Because entropy is a nonlinear operation, these two procedures are not
mathematically equivalent.

For consistency, explainability figures should use the same full-volume
uncertainty computation as threshold tuning and final evaluation.

## 6. Recommended terminology

The final report should distinguish:

- **learned heteroscedastic uncertainty**:
  output of the dedicated uncertainty head;

- **single-pass predictive entropy**:
  entropy of one deterministic softmax prediction;

- **MC predictive entropy**:
  entropy of the mean probability distribution across MC-Dropout samples;

- **MC mutual information**:
  disagreement-based epistemic uncertainty across stochastic samples.

The generic term "uncertainty" should only be used when the specific signal has
already been defined in context.

## 7. Required final comparison

Once voxel-level predictions are available, compare all available uncertainty
signals using the same test subjects and the same masks:

| Signal | Error AUROC | Error AUPRC | Spearman rho | Boundary discrimination |
|---|---:|---:|---:|---:|
| Single-pass predictive entropy | | | | |
| MC predictive entropy | | | | |
| MC mutual information | | | | |
| Learned uncertainty head | | | | |

The analysis should be performed per subject first and then summarised across
subjects to avoid treating millions of spatially correlated voxels as
independent observations.
