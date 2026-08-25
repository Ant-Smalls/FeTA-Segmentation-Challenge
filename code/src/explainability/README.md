# Explainability & Clinical Presentation

## Purpose

Do high-uncertainty regions line up with tissue boundaries that are
ambiguous in the ground truth (eCSF/GM in particular)?

```
code/src/explainability/
  __init__.py
  visualize.py               # slice-overlay + uncertainty-heatmap plotting
  uncertainty_analysis.py    # does uncertainty land on boundaries and on errors
  refinement_control.py      # is the refinement gain from uncertainty or from smoothing
  run_visualize.sh           # SLURM: figures, refined and unrefined
  run_analysis.sh            # SLURM: both analyses
  run_control.sh             # SLURM: refinement control only
  figures*/                  # generated, gitignored
  analysis/                  # generated JSON + summary figure
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

## Quantitative results

`uncertainty_analysis.py` measures whether the heatmaps mean what the figures
suggest, over all 12 test cases. Written to
`analysis/uncertainty_spatial_analysis.json`.

- Mean uncertainty falls with distance from the nearest ground-truth tissue
  boundary: 0.432 within 1mm, 0.205 at 5-10mm.
- Mean uncertainty is 0.311 at correct voxels and 0.524 at incorrect ones;
  higher in 12/12 subjects (Wilcoxon p=0.0005).
- Uncertainty as an error detector: AUROC 0.838 +/- 0.017 overall. Within a
  fixed distance band, where boundary proximity cannot explain the separation,
  AUROC is 0.736 (<1mm) rising to 0.991 (5-10mm).
- Three subjects (sub-020, sub-025, sub-017) have roughly double the error
  rate of the rest; AUROC holds at 0.806-0.857 on those.

`refinement_control.py` compares baseline, baseline with ungated smoothing,
comparative, and comparative with uncertainty-gated refinement. Written to
`analysis/refinement_control.json`.

- Ungated smoothing removes 71% of connected components from the baseline;
  uncertainty-gated refinement removes 52% from the comparative model. The
  per-subject difference is not significant (Wilcoxon p=0.11).
- Against the baseline with the same smoothing applied, the refined
  comparative model is no better on Dice (0.7812 vs 0.7791) and worse on mean
  surface distance, HD95, HD100 and component count.

## Clinical framing

eCSF (label 1) and GM (label 2) are the primary target — their boundary is
where ground-truth annotations are most ambiguous, which is what a
well-calibrated uncertainty map should flag.
