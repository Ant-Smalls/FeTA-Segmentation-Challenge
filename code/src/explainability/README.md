# Explainability & Clinical Presentation

## Purpose

Do high-uncertainty regions line up with tissue boundaries that are
ambiguous in the ground truth (eCSF/GM in particular)?

```
code/src/explainability/
  __init__.py
  visualize.py   # slice-overlay + uncertainty-heatmap plotting
  figures/       # generated, gitignored
```

## Consumes

- `FeTADataset` eval mode (`code/src/preprocessing_data_preparation/dataset.py`)
  for full preprocessed volumes + `meta["subject_id"]`.
- `src.models.uncertainty_unet.UncertaintyUNet`. `forward()` returns
  `(logits, uncertainty_head_output)`, but the map fed to
  `refine_prediction()` is the MC-Dropout predictive entropy, not the raw
  head output — the head is training-only, used as a loss term.
- Per-case Dice/ED results for case selection. Not wired in yet.

## Status

`visualize.py` supports both models via `--model-type {baseline,comparative}`:

- `baseline` (default): `src.training.train.sliding_window_inference`,
  `uncertainty=None`, 3-panel figure.
- `comparative`: `UncertaintyUNet` via `mc_dropout_sliding_window_inference()`
  (in `visualize.py`) — `sliding_window_inference` discards the uncertainty
  head output even with `MODEL_RETURNS_UNCERTAINTY=True`, so this instead
  runs dropout-enabled forward passes and stitches mean class-probabilities
  + entropy via `reconstruct_from_patches`. `--refine` applies
  `refine_prediction()` on top. `plot_case()` handles 3-panel vs. 4-panel.

Verified against a trained comparative checkpoint
(`checkpoints_comparative/best_model_comp.pt`) across 5 test cases; output
figures in `figures/`.

`pick_informative_slice`'s heuristic (most eCSF/GM voxels) is a placeholder
until per-case selection is wired in.

## Clinical framing

eCSF (label 1) and GM (label 2) are the primary target — their boundary is
where ground-truth annotations are most ambiguous, which is what a
well-calibrated uncertainty map should flag.
