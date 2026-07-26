# FeTA Fetal Brain Segmentation — Segmentation Challenge

## Project goal

Automatic multi-tissue segmentation of fetal brain MRI (FeTA dataset, 80 cases, 7 tissue classes) using a shared 3D U-Net backbone. Full background and the three candidate research directions we considered are in [`code/docs/initial_research_plan.md`](code/docs/initial_research_plan.md).

**We chose Option 3: Uncertainty-Guided Boundary Refinement** — does a model that predicts its own voxel-wise uncertainty reliably flag the boundaries hardest to segment (eCSF, GM, dGM), and can that signal drive a targeted refinement step that improves boundary accuracy? See [`code/docs/project_roles_q3.md`](code/docs/project_roles_q3.md) for what each of the 6 team roles owns and delivers under this option.

Shared domain language (dataset terms, anatomy, metrics) is defined once in [`CONTEXT.md`](CONTEXT.md) — use it consistently in code, docs, and discussion.

## Repo layout

```
code/
  docs/                                # role definitions, research plan
  requirements.txt                     # pinned deps for the whole codebase
  src/
    data/
      config.yaml                      # shared pipeline config (frozen, checked in)
      splits/split_v1.json             # frozen train/val/test split + QC log (checked in)
      mri_gz/                          # raw *_T2w.nii.gz / *_dseg.nii.gz (HPC-only, NOT checked in)
    preprocessing_data_preparation/    # QC, split generation, FeTADataset (done — see below)
    models/                            # baseline / comparative model code (role 2/3, not yet started)
literature/                            # paper summaries backing the research plan
assignment_details/                    # FeTA challenge README, rubric, example notebooks
```

## HPC setup

Steps 1-5 are one-time setup every team member needs to do on their own HPC account. Steps 6-7 (QC/split + preprocessing validation) have **already been run** against the real 80-case dataset — their outputs are committed to the repo, so you don't need to redo them. They're listed anyway so you know what already happened and can re-run them if the raw data ever changes.

1. **Clone the repo.**

   ```bash
   git clone git@github.com:Ant-Smalls/FeTA-Segmentation-Challenge.git
   cd FeTA-Segmentation-Challenge
   ```

2. **Copy the code to the HPC.**

   ```bash
   scp -r /path/to/FeTA-Segmentation-Challenge/code <userid>@login.ucd.ie:/home/people/<userid>/scratch/segmentation-assignment/
   ```

3. **Create the data folder and get the dataset onto the HPC.** `code/src/data/mri_gz/` isn't checked into git (raw MRI data is too large) — download the dataset zip from the shared Google Drive, then upload it:

   ```bash
   ssh <userid>@login.ucd.ie 'mkdir -p /home/people/<userid>/scratch/segmentation-assignment/code/src/data'
   scp /path/to/downloaded_mri_gz.zip <userid>@login.ucd.ie:/home/people/<userid>/scratch/segmentation-assignment/code/src/data/
   ```

4. **Unzip on the HPC.** This should produce `code/src/data/mri_gz/` containing all 80 cases as `sub-XXX_rec-{mial|irtk}_{T2w,dseg}.nii.gz` pairs.

   ```bash
   cd /home/people/<userid>/scratch/segmentation-assignment/code/src/data
   unzip mri_gz.zip
   ```

5. **Create a virtual environment in `code/` and install dependencies.**

   ```bash
   cd /home/people/<userid>/scratch/segmentation-assignment/code
   python3 -m venv venv && source venv/bin/activate
   pip install -r requirements.txt
   ```

6. ~~Run QC + generate the split (`run_qc_stratify.sh`), fill in `config.yaml`.~~ **Already done.** `run_qc_stratify.sh` ran against all 80 real cases: 79 passed, 1 excluded (`sub-022`, simultaneous eCSF/GM/brainstem depletion), split into 55 train / 12 val / 12 test. Results are frozen in `code/src/data/splits/split_v1.json` (checked into git). `config.yaml`'s one HPC-dependent value, `preprocessing.target_spacing_mm`, is filled in as `[0.5, 0.5, 0.5]` (computed from the real train split). Only re-run this if the raw dataset changes.

7. ~~Run preprocessing (`run_preprocessing.sh`).~~ **Already done.** This validated the full crop/resample/normalize pipeline and `FeTADataset` against all 79 QC-passed cases: every case preprocessed successfully (0.35-2.44s/case, 962MB peak memory), confirming the pipeline is safe to build on at real data scale. No output artifact beyond the passing log — this is a sanity check, not a data-generating step.

Full detail on what preprocessing produces and how to consume it (`FeTADataset`, tensor contracts, `config.yaml`) is in [`code/docs/data_preprocessing_preparation.md`](code/docs/data_preprocessing_preparation.md) — start there before building the baseline/comparative models, training loop, or evaluation.
