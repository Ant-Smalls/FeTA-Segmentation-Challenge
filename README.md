# FeTA Fetal Brain Segmentation — Segmentation Challenge

## Project goal

Automatic multi-tissue segmentation of fetal brain MRI (FeTA dataset, 80 cases, 7 tissue classes) using a shared 3D U-Net backbone.

The experiment compares two models on the same data, split, and training loop:

- **Baseline model** — one segmentation head; Dice + cross-entropy only.
- **Comparative model** — the same backbone, plus a predicted-variance head and uncertainty-guided boundary refinement.

## Repo layout

```
code/
  requirements.txt                     # pinned deps
  src/
    data/
      config.yaml                      # shared pipeline config (frozen)
      splits/split_v1.json             # frozen 55/12/12 split + QC log
      mri_gz/                          # raw *_T2w.nii.gz / *_dseg.nii.gz (HPC-only)
    preprocessing_data_preparation/    # QC, split, crop/resample/normalize, FeTADataset
    models/                            # BaselineUNet, UncertaintyUNet + refine_prediction
    training/                          # shared train loop + refinement-threshold sweep
    explainability/                    # slice overlays, spatial analysis, refinement control
    robustness/                        # audits of frozen test-set results
    tuned_evaluation_outputs/          # frozen held-out evaluation reports
```



## Pipeline

Run all Python commands from `code/` with the venv active. HPC setup (clone, data, venv) is below.

`FeTADataset` is the data contract: train mode returns `(image, label)` patches; eval mode returns `(image, label, meta)` full volumes. Image is `(1, D, H, W)` float32; label is `(D, H, W)` int64, classes 0–7.

### 1. Quality control and stratified split

Already done. 79/80 cases passed QC (`sub-022` excluded). Split is 55 train / 12 val / 12 test, stratified by `rec_type` (mial vs irtk). Frozen in `code/src/data/splits/split_v1.json`. Re-run only if the raw data changes:

```bash
sbatch src/preprocessing_data_preparation/run_qc_stratify.sh
```



### 2. Crop, resample, normalize

`FeTADataset` applies this per case (foreground crop, 0.5 mm isotropic, per-volume z-score). Train mode samples a `128³` patch with light augmentation. Eval is in preprocessed space.

Already validated on all 79 cases. Re-run only if the raw data changes:

```bash
sbatch src/preprocessing_data_preparation/run_preprocessing.sh
```



### 3. Shared 3D U-Net — baseline vs comparative

Same loop, same hyperparameters. Config `model_type` selects the model.

```bash
python -m src.training.train --config src/training/train_config_baseline.yaml
python -m src.training.train --config src/training/train_config_comparative.yaml
```

Comparative SLURM job: edit `CODE_DIR` in `src/training/run_train_comparative.sh`, then `sbatch` it. Checkpoints (gitignored):

- `src/training/checkpoints_baseline/best_model.pt`
- `src/training/checkpoints_comparative/best_model.pt`

Trained weights also live on the shared [Google Drive](https://drive.google.com/drive/folders/1_3NFIKvh4PCQ1G0voYH4MGWHbudMnHYX?usp=sharing).

### 4. Uncertainty-guided refinement (comparative only)

The variance head is a training loss term. Refinement uses MC-Dropout predictive entropy and a local majority vote on high-uncertainty boundary voxels. Threshold is selected on **validation only** (current value: 0.5). Comparative training runs this sweep when `run_refinement_sweep: true`. Standalone:

```bash
python -m src.training.tune_refinement_threshold \
    --config src/training/train_config_comparative.yaml \
    --checkpoint src/training/checkpoints_comparative/best_model.pt
```



### 5. Held-out evaluation

Test-set reports are frozen in `[code/src/tuned_evaluation_outputs/](code/src/tuned_evaluation_outputs/)`.

### 6. Explainability and robustness

```bash
sbatch src/explainability/run_visualize.sh
sbatch src/explainability/run_analysis.sh
sbatch src/explainability/run_control.sh
```

Those scripts expect named checkpoints (`best_model.pt` / `best_model_comp.pt`).

Robustness reads the frozen evaluation CSVs (no retrain). From `code/`:

```bash
python src/robustness/01_existing_results_audit.py
```

More detail: `[code/src/training/README.md](code/src/training/README.md)`, `[code/src/explainability/README.md](code/src/explainability/README.md)`.

