# Training Pipeline & Hyperparameter Tuning (Role 4)

## What this role owns

The shared training loop used identically by the baseline model (Role 2) and
the comparative uncertainty model (Role 3), so the only difference between
their two results is the model/loss itself -- not how they were trained.
Also owns the two Option-3-specific tuning axes: the uncertainty loss weight
(variance-head route only) and the refinement threshold.

```
code/src/training/
  __init__.py
  train.py                       # the shared training loop
  train_config.yaml              # real training config (points at the frozen split_v1.json)
  train_config_local_test.yaml   # local sanity-check config (points at a trimmed local split)
  checkpoints/                   # generated, gitignored
  checkpoints_local_test/        # generated, gitignored
```

## Using `train.py`

```bash
# from the code/ directory, with the venv active
python -m src.training.train --config src/training/train_config.yaml
```

Requires `matplotlib` in addition to what's in `requirements.txt` (used for
the training history plot) -- `pip install matplotlib` if you hit a
`ModuleNotFoundError` on first run.

**Tensor contract in:** consumes `FeTADataset` exactly per the frozen
contract in `data_preprocessing_preparation.md` -- train mode `(image,
label)` patch pairs for training, eval mode `(image, label, meta)` full
volumes for validation.

**What the loop does, in order:** load train/val data -> train one epoch
(Dice+CE loss, Adam optimizer) -> validate on the *full* val volumes via
sliding-window inference, stitched back together with the shared
`reconstruct_from_patches` utility (same aggregation setting,
`config.yaml`'s `eval.patch_aggregation`, that Role 5 uses -- so training-time
validation Dice and Role 5's final test Dice are computed the same way) ->
checkpoint if val Dice improved -> early stop if it hasn't improved for
`early_stopping_patience` epochs.

**Early stopping:** on by default, controlled by `early_stopping_patience`
in the config. Training halts once `val_dice` hasn't set a new best for that
many consecutive epochs.

**Training history:** every run saves `{run_name}_history.json` (raw
per-epoch loss/Dice/LR values, for later re-plotting or cross-run
comparison) and `{run_name}_history.png` (loss + Dice curves, train vs. val)
into `checkpoint_dir`. Set `run_name` in the config per run (e.g.
`"baseline"`, `"comparative"`) so files from different runs don't overwrite
each other.

## Swapping in a real model

`train.py` runs against a `DummyModel` (a single 3D conv layer) by default,
so the loop is fully testable without waiting on Roles 2/3. To plug in a
real model, replace the `DummyModel` import/instantiation with the real one:

```python
from src.models.baseline import BaselineUNet as Model        # Role 2
# or
from src.models.comparative import UncertaintyUNet as Model  # Role 3
```

**Required interface contract**, so nothing else in `train.py` needs to
change:

| | Input | Output |
|---|---|---|
| Baseline (Role 2) | `(B, 1, D, H, W)` float32 | `(B, num_classes, D, H, W)` logits |
| Comparative (Role 3) | `(B, 1, D, H, W)` float32 | `(logits, uncertainty)` tuple, both `(B, ..., D, H, W)` |

If the model returns the `(logits, uncertainty)` tuple, set
`MODEL_RETURNS_UNCERTAINTY = True` at the top of `train.py`. `compute_loss()`
and the sliding-window inference path both already branch on this flag, so
no other code changes should be needed.

## The uncertainty loss term (placeholder, Role 3 to finalize)

`compute_loss()` has a stubbed-out calibration term:

```python
calibration_term = uncertainty.mean() * 0.0   # placeholder, cancels to zero
return ce + dsc + uncertainty_loss_weight * calibration_term
```

This is intentionally inert until Role 3 shares the real formulation (most
likely a negative log-likelihood term under the predicted variance, per the
research plan). Once that's ready, replace the placeholder line -- the
`uncertainty_loss_weight` config value and the surrounding wiring are
already in place.

## Config files

`train_config.yaml` (real runs) and `train_config_local_test.yaml` (local
sanity checks against a handful of downloaded cases) share the same fields:

| Field | Meaning |
|---|---|
| `code_root` | `.` if running from inside `code/`, `code` if running from the repo root |
| `split_filename` | Which split file to load from `src/data/splits/`. Real runs: `split_v1.json` (frozen, never edit). Local checks: a trimmed local copy, see below. |
| `seed` | Passed to `set_seed()` for reproducible runs -- important once baseline vs. comparative are being compared. |
| `batch_size`, `learning_rate`, `max_epochs`, `early_stopping_patience` | Standard training loop settings. |
| `num_workers` | `DataLoader` worker count. Defaults to `0` -- Windows can be flaky with >0; raise if stable on your machine. |
| `patch_size` | Must match `FeTADataset`'s training patch size (`128, 128, 128` per the nnU-Net default in the frozen `config.yaml`). |
| `uncertainty_loss_weight` | Only used once `MODEL_RETURNS_UNCERTAINTY = True`. |
| `lr_scheduler`, `lr_scheduler_factor`, `lr_scheduler_patience` | Optional `ReduceLROnPlateau` on val Dice. Off by default -- turn on for longer real-model runs where a plateau is expected. |
| `class_weights` | Optional per-class weights for the CE loss (8 floats: background + 7 tissues, in class-index order). Off by default (`null`). Worth enabling once real models train, since the tissue classes are heavily imbalanced -- see the Label Information table in the repo's top-level `README.md` (eCSF/GM/dGM/brainstem are all much smaller volumes than WM). |

**Local sanity-check split:** copy `split_v1.json` to
`splits/split_local_test.json`, trim its `train`/`val` lists down to
whichever subject IDs you've actually downloaded locally (check
`rec_type` per subject -- it's fixed per case, not a free choice), and point
`train_config_local_test.yaml`'s `split_filename` at it.
**Never edit or commit the real `split_v1.json`, and never commit
`split_local_test.json`** -- both are enforced by `.gitignore`.

## Hyperparameter tuning (once Role 3's model exists)

Two axes, both selected on the **validation split only, never test**:

1. **Uncertainty loss weight** (variance-head route only) -- sweep a few
   values, select using validation Dice + calibration quality jointly.
2. **Refinement threshold** -- how uncertain a voxel must be before the
   refinement rule intervenes. Tune specifically against validation Dice/ED
   on eCSF/GM/dGM, since that's where the mechanism is meant to help most.

Document the selection criteria for both alongside whatever values are
chosen -- Role 5 and the final report need the reasoning, not just the
number.

## Status

- [x] Shared training loop (optimizer, Dice+CE loss, early stopping, checkpointing)
- [x] Full-volume validation via sliding-window inference + shared stitching utility
- [x] Reproducible seeding
- [x] Verified end-to-end locally against a small downloaded subset
- [ ] Baseline model plugged in (blocked on Role 2)
- [ ] Comparative model plugged in (blocked on Role 3)
- [ ] Uncertainty loss weight sweep (blocked on Role 3)
- [ ] Refinement threshold sweep (blocked on Role 3)