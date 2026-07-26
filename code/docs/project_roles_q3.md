# Option 3: Uncertainty-Guided Boundary Refinement — Role Definitions

Research question (full context in `[initial_research_plan.md](./initial_research_plan.md)`): does a model that predicts its own voxel-wise uncertainty reliably flag the boundaries hardest to segment (eCSF, GM, dGM), and can that signal drive a targeted refinement step that improves boundary accuracy?

This doc defines the minimum bar and expected outcomes for the 5 remaining roles.

## Shared foundation (already frozen, don't re-derive)

Full contract in `[data_preprocessing_preparation.md](./data_preprocessing_preparation.md)` — read it before starting. In short:

- `FeTADataset` is the only sanctioned way to load data. Train mode yields `(image, label)` patch pairs; eval mode yields `(image, label, meta)` full volumes. Tensor shapes/dtypes are fixed.
- Every case is already cropped, resampled to a common spacing, and z-score normalized. All modeling, training, and evaluation happens in this preprocessed space.
- The train/val/test split (`split_v1.json`) and all pipeline parameters (`config.yaml`) are frozen. Propose changes to the preprocessing role rather than forking them.
- Eval-mode volumes vary in size (crop-dependent), so any role running inference on full volumes needs a sliding-window pass at the training patch size, stitched back with `dataset.reconstruct_from_patches`. Use `config.yaml`'s `eval.patch_aggregation` setting for that stitching so baseline, comparative, and evaluation all reconstruct volumes the same way.

## Role 2: Baseline Models

**Goal.** The shared 3D U-Net trained with Dice+CE only, single output head, no uncertainty — the "confident but possibly wrong" reference every other result is measured against.

**Consumes.** `FeTADataset` in train mode for training, eval mode for inference.

**Produces.** A trained checkpoint plus an inference entry point that takes a preprocessed volume (or patch) and returns per-voxel class probabilities/logits, using the same volume-stitching convention as everyone else so evaluation can call baseline and comparative interchangeably.

**On track when:** training converges to a stable validation Dice using the frozen split with no leakage across splits, and the inference interface works standalone on any case without needing the comparative model's code.

## Role 3: Comparative Models (Uncertainty Head)

**Goal.** Same backbone as the baseline plus one small uncertainty output (heteroscedastic/aleatoric variance head, or Monte Carlo Dropout at inference — either is good approach so maybe try both), and a lightweight, rule-based refinement step that adjusts predictions in high-uncertainty, near-boundary voxels.

**Consumes.** Same `FeTADataset` interface as baseline. Should stay architecturally identical to the baseline apart from the uncertainty mechanism, so the comparison stays a clean ablation.

**Produces.** A trained checkpoint whose inference call returns both a class prediction and a voxel-wise uncertainty map (same spatial shape as the prediction), plus a refinement function that takes prediction + uncertainty + a threshold and returns a refined prediction. Refinement must be toggle-able independently of the model itself.

**On track when:** the model is drop-in compatible with baseline's inference interface (evaluation shouldn't need special-casing), the uncertainty map is well-defined and bounded, and refinement can be turned off to recover the pre-refinement prediction exactly (needed for the with/without ablation).

## Role 4: Training Pipeline & Hyperparameter Tuning

**Goal.** One shared training loop (optimizer, LR schedule, early stopping, checkpointing) used identically by roles 2 and 3, so only their model/loss differs — plus the two option-specific tuning axes: the uncertainty loss weight (variance-head route only) and the refinement threshold.

**Consumes.** Baseline and comparative model code from roles 2/3; `FeTADataset` train/val modes for the training and tuning loop itself.

**Produces.** A training configuration/budget both models run under, and a chosen refinement threshold (and uncertainty loss weight, if applicable) selected against the validation split, with the selection criteria documented.

**On track when:** baseline and comparative are trained under an identical epoch/optimizer budget with only the deliberate option-specific differences, and the refinement threshold is chosen on validation data only — never on test.

## Role 5: Evaluation & Metrics

**Goal.** Dice, per-class Dice (watch eCSF/GM/dGM/brainstem), and Euler characteristic difference (ED) for baseline vs. comparative on the test split, plus the calibration check this option's whole premise depends on.

**Consumes.** Both trained models' inference interfaces (roles 2/3), `FeTADataset` eval mode for the test split, `reconstruct_from_patches` for stitching predictions.

**Produces.** A metrics report (Dice/ED, baseline vs. comparative, same test cases for both), a reliability plot of predicted uncertainty vs. observed voxel-wise error, and a before/after-refinement Dice/ED comparison focused on eCSF/GM/dGM.

**On track when:** the calibration/reliability check is produced and reviewed *before* the before/after-refinement numbers are treated as meaningful — if uncertainty doesn't correlate with real error, refinement results need that caveat attached, not to be presented at face value.

## Role 6: Explainability & Clinical Presentation

**Goal.** Turn the uncertainty + refinement results into a clinically legible story: do high-uncertainty regions line up with boundaries radiologists themselves disagree on (eCSF/GM in particular)?

**Consumes.** Comparative model's uncertainty maps and refined predictions (role 3), evaluation's per-case results (role 5) to pick representative cases, `meta["subject_id"]` from eval mode to trace visuals back to source cases.

**Produces.** Slice-level visuals overlaying uncertainty heatmaps against ground truth and prediction, and where relevant, before/after-refinement comparisons, framed around a specific documented clinical ambiguity rather than a generic heatmap.

**On track when:** every visual is traceable to a real test-split case and its interpretation is tied to a named anatomical/clinical phenomenon (e.g. eCSF/GM boundary disagreement), not just "uncertainty is high here."

## Dependency chain

```
Preprocessing (frozen)
   |
   +--> Role 2: Baseline  ----\
   |                           +--> Role 4: Training pipeline drives both
   +--> Role 3: Comparative --/                |
                                                v
                                      Role 5: Evaluation & calibration
                                                |
                                                v
                                 Role 6: Explainability & clinical presentation
```

