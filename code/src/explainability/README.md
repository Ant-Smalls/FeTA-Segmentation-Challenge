# Explainability & Clinical Presentation (Role 6)

## What this role owns

Turning the uncertainty + refinement results into a clinically legible
story: do high-uncertainty regions line up with boundaries radiologists
themselves disagree on (eCSF/GM in particular)? Full contract in
`code/docs/project_roles_q3.md`.

```
code/src/explainability/
  __init__.py
  visualize.py   # slice-overlay + uncertainty-heatmap plotting
  figures/       # generated, gitignored
```

## Consumes

- `FeTADataset` eval mode (`code/src/preprocessing_data_preparation/dataset.py`)
  for full preprocessed volumes + `meta["subject_id"]`.
- Role 3's comparative model: inference must return `(logits, uncertainty)`,
  same spatial shape, per the interface contract in
  `code/src/training/README.md`. Not implemented yet — see STATUS below.
- Role 5's per-case Dice/ED results, for picking representative
  (best/worst/most-disagreement) cases instead of a heuristic slice pick.
  Not implemented yet.

## STATUS: scaffolding only

Roles 3 (comparative uncertainty model) and 5 (evaluation/case selection)
haven't landed as of this writing. `visualize.py` currently runs end-to-end
against the Role 2 **baseline** checkpoint only, so the plotting path
(overlay rendering, slice picking, figure layout) is proven out ahead of
time rather than blocked on them. `run_inference()` always returns
`uncertainty=None` for now; `plot_case()` already handles the `None` case
(3-panel: image/GT/prediction) vs. the eventual 4-panel version once an
uncertainty map is available.

**To swap in the comparative model once Role 3 lands:**
1. In `load_model()`, replace the `BaselineUNet` import with
   `from src.models.comparative import UncertaintyUNet as Model`.
2. `run_inference()` calls `src.training.train.sliding_window_inference`,
   which branches on the *module-level* flag
   `src.training.train.MODEL_RETURNS_UNCERTAINTY` — set that to `True`
   before calling inference, or the uncertainty output will be silently
   dropped (it's a shared global from another role's module, not a
   parameter, so this is easy to miss).
3. Replace `pick_informative_slice`'s heuristic (most eCSF/GM voxels) with
   Role 5's actual case selection once its metrics report exists.

## Clinical framing

Every figure should tie back to a named, documented clinical phenomenon —
not a generic heatmap. eCSF (label 1) and GM (label 2) are the primary
target per `CONTEXT.md` and the research question in
`code/docs/project_roles_q3.md`: their boundary is where radiologists
themselves disagree, which is exactly what a well-calibrated uncertainty
map should flag.
