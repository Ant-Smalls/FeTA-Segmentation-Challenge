# Data Preprocessing & Transformation —

## What this role owns

Everything upstream of the model: discovering/QC-ing the raw NIfTI cases, generating the train/val/test split, and the `FeTADataset` class (load → crop → resample → normalize → patch-sample/augment).

```
code/src/data/
  config.yaml            # single shared config every role reads from
  splits/split_v1.json   # frozen split + QC log (generated, checked into git)
  mri_gz/                # raw sub-XXX_rec-{mial|irtk}_{T2w,dseg}.nii.gz (not checked in, HPC-only)

code/src/preprocessing_data_preparation/
  qc_and_split.py            # discover cases, run QC, write split_v1.json
  preprocessing.py           # crop/resample/normalize/augment transforms
  dataset.py                 # FeTADataset + reconstruct_from_patches
  validate_preprocessing.py  # full-scale preprocessing sanity check
  run_qc_stratify.sh         # HPC SLURM script for qc & stratify
  run_preprocessing.sh       # HPC SLURM script for preprocessing pipeline (after qc run)
```

## The frozen split (`split_v1.json`)

QC excluded **1/80** cases: `sub-022` (simultaneous eCSF/GM/brainstem depletion — three unrelated tissue compartments all critically small at once, unlike genuine gestational-age variation which shrinks the whole brain roughly proportionally). The remaining **79** cases are split, stratified by the `rec-mial`/`rec-irtk` site proxy, into:

| Split | N  |
|-------|----|
| train | 55 |
| val   | 12 |
| test  | 12 |

Seed `42`, fractions `70/15/15`. `split_v1.json` also carries the full QC log (pass/fail + reason for all 80 cases) and `train_median_spacing_mm` (computed from the 55 train cases only, to avoid leaking split info into preprocessing). Re-running `qc_and_split.py` against the same data is deterministic and will reproduce this file exactly.

## The frozen config (`config.yaml`)

All static pipeline parameters live here — QC thresholds, target spacing, patch size, augmentation probabilities, split fractions, eval-space/aggregation choice. Every value is final; `target_spacing_mm` is now filled in as `[0.5, 0.5, 0.5]`, computed from the real train split and matching the FeTA README's reported ~0.5mm isotropic. Read comments inline for the reasoning behind each value.

## Using `FeTADataset`

```python
from pathlib import Path
from src.preprocessing_data_preparation.dataset import FeTADataset

code_root = Path("code")
split_path = code_root / "src/data/splits/split_v1.json"
config_path = code_root / "src/data/config.yaml"

train_ds = FeTADataset(split_path, config_path, split_name="train", mode="train", seed=0)
image, label = train_ds[i]                      # (1, D, H, W) float32, (D, H, W) int64 patch

eval_ds = FeTADataset(split_path, config_path, split_name="test", mode="eval")
image, label, meta = eval_ds[i]                  # full preprocessed volume + {subject_id, affine, spacing}
```

**Tensor contract (frozen, both modes):** image `float32` shaped `(1, D, H, W)`; label `int64` shaped `(D, H, W)` with raw class indices `0`-`7` (0 = background). Nobody downstream should need to reshape, retype, or reinterpret these.

**Train mode** returns a `(image, label)` 2-tuple: one `128³` patch per case per `__getitem__` call (nnU-Net default patch size), foreground-oversampled with probability `0.33` (config-driven), plus light augmentation (flips/rotation/noise, all config-driven probabilities). `__len__` == number of cases in the split, not number of patches — training loops should run enough epochs, or wrap with a repeated sampler, to see enough patches per case.

**Eval mode** returns a `(image, label, meta)` 3-tuple: the *full* preprocessed volume (no patching), plus `meta = {subject_id, affine, spacing}` for anything that needs to trace a prediction back to its source case. Since eval-mode volumes aren't a fixed size, evaluation/inference code that needs patches (e.g. sliding-window inference) should tile them itself and use `reconstruct_from_patches` (below) to stitch predictions back.

## Preprocessing pipeline

Order, applied identically in train and eval mode via `preprocessing.preprocess_case`: **load → crop to non-zero foreground bbox → resample to `[0.5, 0.5, 0.5]`mm isotropic (only if a case's native spacing is outside tolerance; most already match) → per-volume z-score normalize.** Augmentation (train-mode only) happens *after* this, at the patch level, inside `FeTADataset.__getitem__`.

**Evaluation happens in this preprocessed (crop+resample) space, never mapped back to original resolution** — Dice/ED/uncertainty-calibration should all be computed here. If need original-resolution outputs for the clinical-presentation role, need to invert crop+resample using `meta["affine"]`/`meta["spacing"]` — this role doesn't currently provide that inverse.

## Stitching patches back together

`dataset.reconstruct_from_patches(patches, patch_coords, full_volume_shape, aggregation)` takes a list of per-patch prediction tensors `(C, *patch_shape)` plus their coordinate slices and returns one full-volume tensor `(C, *full_volume_shape)`, blending overlaps by mean or Gaussian (edge-downweighted) averaging. This exists specifically so the uncertainty role (3) and evaluation role (5) don't each reinvent slightly different stitching logic. `config.yaml`'s `eval.patch_aggregation` (`"gaussian"`) is the recommended default — pass it explicitly.

## Running the pipeline on the HPC

Two SLURM scripts, run in order:

1. `run_qc_stratify.sh` — discovers all cases in `mri_gz/`, runs QC, writes `split_v1.json`. Only needs re-running if the raw data changes.
2. `run_preprocessing.sh` — runs `validate_preprocessing.py`, which preprocesses every case in `split_v1.json` and exercises `FeTADataset` in every split/mode, timing each case and reporting peak memory. Exits non-zero on any failure, safe to gate a downstream job on.

Both expect a `venv` at `code/venv` built from `code/requirements.txt`.
