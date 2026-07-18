# FeTA Fetal Brain Segmentation — Initial Research Plan

This document lays out three candidate research directions for the class project. Each option targets a distinct gap left open by the FeTA 2022/2024 winning solutions and the foundational reconstruction papers, scoped to what a masters-level team can realistically execute on our HPC allocation in one semester. All three share the same data, split, backbone, and evaluation protocol below, so the choice between them comes down to which single mechanism you most want to investigate — not which team has more resources.

Deliberately out of scope: exact/adaptive Total Variation reconstruction and 4D spatiotemporal atlas construction. Both are covered well by our literature review as *problem* framing, but as *solutions* they add heavy mathematical-optimization machinery unrelated to the deep-learning skills this course is assessing, and neither is a novel angle relative to the papers we already reviewed.

## Shared Foundations (apply to all three options)

**Data reality.** Our data drop contains exactly two file types per case: `sub-XXX_rec-{mial|irtk}_T2w.nii.gz` (the reconstructed image) and the matching `_dseg.nii.gz` (7 tissue labels + background). There are no `.json` sidecar or `participants.tsv` metadata files, so gestational age, pathology status, and source institution are all **unknown** for our 80 cases — the full FeTA dataset ships this metadata, but our specific delivery does not include it. No option below depends on metadata we don't have; where an idea from the literature assumes it (e.g. GA-conditioning, pathology-targeted augmentation), we've noted the fallback.

**Split.** Random, fixed, subject-level split of the 80 cases: 56 train / 12 val / 12 test (70/15/15). We can't stratify by GA or pathology since we don't have those labels. Reuse the exact same split across all three options for a fair comparison. Given how small 12 test cases is, treat single-split numbers as noisy — reporting mean ± std across a couple of alternate random splits is a good stretch goal once we know per-run training time, but is not the initial baseline (see compute policy below).

**Compute / model-count policy.** Default to training **one model per experiment**. Both prior FeTA winners used ensembles of several models plus a second post-processing network — expensive, and not something we can budget for until we know per-run training time on the HPC. Ensembling or multi-model training is only added to an option if a single model demonstrably can't answer that option's research question, not assumed upfront.

**Shared backbone.** A single 3D U-Net (encoder-decoder, skip connections), trained on foreground-oversampled 3D patches rather than full 256³ volumes (memory). This backbone is identical across all three options — each option's novelty is layered on top of it (a loss term, a data pipeline, or an output head), not a reinvented architecture each time. This keeps the three options genuinely comparable and satisfies the assignment's "novelty in architecture" criterion once, deliberately, per option.

**Shared preprocessing baseline.** Per-volume z-score normalization; crop to the non-zero foreground bounding box (no separate skull-stripping model needed — FeTA's SRR pipeline already removes maternal/background tissue before labeling); verify voxel spacing consistency across cases at load time and resample any outliers to the training-set median spacing (~0.5 mm isotropic per the dataset README — most cases should already match). We assume no further bias-field correction is needed since these are already SRR-reconstructed images; flag this for a quick visual sanity check early on.

**Assignment-rubric mapping.** A `FeTADataset(Dataset)` class (`__init__` for file paths/split/transforms, `__len__` for patch count, `__getitem__` returning an image/label patch pair) and a per-option `Model(nn.Module)` class (`__init__`/`forward`) satisfy the dataloader and model-class requirements directly. Each option's training loop uses a standard Adam/AdamW optimizer, a combined Dice+CE loss (plus the option-specific term where relevant), early stopping on validation Dice, and a defined hyperparameter-tuning axis (called out per option below) — covering the training-process rubric item without needing to repeat it three times below.

**Evaluation.** Mean Dice + per-class Dice (watch eCSF, GM, dGM, brainstem — the classes every paper we reviewed flags as hardest), and the Euler characteristic difference (ED) per class, the topology metric introduced in FeTA 2024. Using ED for all three options — not just the ones explicitly about topology — gives every option a topology baseline for comparison.

---



## Option 1: Topology-in-the-Loss

**Research question.** Can a single 3D U-Net trained with an explicit topology-aware loss term produce anatomically plausible predictions (fewer holes, fewer disconnected blobs) comparable to what a *separate* post-hoc Denoising Autoencoder (DAE) achieved for both FeTA winners — without training a second network?

**Novel approach & why it's novel.** Add a differentiable topology-consistency term (e.g. a soft clDice-style skeleton-overlap penalty, or a boundary/surface-distance penalty as a lower-risk fallback) to the standard Dice+CE loss, weighted toward the topologically fragile classes (eCSF, GM, dGM). Both the FIT (2022) and cesne-digair (2024) winners treated topology purely as a *post-processing* problem — train the segmentation network, then separately train and run a DAE to patch holes afterward. This option tests whether that correction can instead be learned end-to-end inside one network, which the FeTA 2024 organizers explicitly named as unexplored future work.

**Experimental setup.** Train a Dice+CE-only baseline and a Dice+CE+topology-term treatment on the identical split/backbone. The whole option is essentially one clean ablation: same everything, one loss term added.

**Architecture guidance.** Shared single 3D U-Net, unchanged. No second network, no ensembling — all novelty is in the loss function.

**Training strategy.** Curriculum weighting: train with Dice+CE alone for the first portion of training (topology losses tend to be unstable while predictions are still noisy), then ramp up the topology term's weight. Hyperparameter-tuning axis: the topology loss weight (and its ramp-up schedule).

**Preprocessing pipeline.** Shared baseline only. If the chosen topology loss needs a soft-skeleton or boundary map, compute it on the fly with a small fixed convolutional approximation rather than precomputing a separate dataset.

**Key risks / assumptions.** Differentiable topology losses can have weak or noisy gradients early in training. Fallback if unstable: drop to a simpler boundary-weighted Dice or surface-distance loss, which is a smaller step from the shared baseline but still tests the same core idea (loss-level vs. post-hoc topology correction).

---



## Option 2: Augmentation-Only Generalization Attack

**Research question.** How much of the robustness that prior winners reported actually comes from data augmentation alone, versus the ensembling and DAE stages they bundled alongside it? Can a *single* U-Net, trained only with aggressive appearance + anatomy augmentation, close a meaningful chunk of that gap?

**Novel approach & why it's novel.** Combine (a) appearance augmentation — synthetic bias-field, contrast, and gamma jitter as a lightweight stand-in for the GIN/style-transfer tricks FIT used to simulate scanner/protocol variation — with (b) anatomy augmentation via deformable (B-spline or SyN-style) warps applied between randomly paired training scans, increasing shape diversity the way cesne-digair did for pathological anatomy. Then evaluate on both the normal held-out split and a deliberately "hard" synthetic validation set (extra-strong simulated bias field/noise/contrast shift), so the *gap* between the two is a direct measure of what augmentation bought us. Neither winning team isolated augmentation's contribution — they always combined it with ensembling and a DAE — so nobody actually knows its standalone value. That isolation is the novel contribution here.

**Pathology-metadata fallback.** cesne-digair's anatomy augmentation specifically warped *healthy* scans toward *known-pathological* shapes, which requires a pathology label we don't have. Default plan: apply deformable warps between **randomly paired** training cases regardless of any pathology status — still increases shape diversity, just untargeted. 

**Architecture guidance.** Shared single 3D U-Net, unchanged. All novelty lives in the data pipeline.

**Training strategy.** Standard Dice+CE loss. Hyperparameter-tuning axis: augmentation strength/probability schedule — ablate at least two or three strengths (e.g. off / moderate / aggressive) to show the effect isn't just noise.

**Preprocessing pipeline.** Shared baseline plus an augmentation stage inserted at data-loading time (inside `__getitem__` or a transform pipeline called from it): random bias-field simulation, contrast/gamma jitter, and occasional deformable warp sourced from another random training case's shape.

**Key risks / assumptions.** Our 80 cases may come from a limited number of institutions (unknown without metadata), so true cross-center generalization can't be directly measured — the synthetic "hard" validation set is a proxy for that, not a substitute. Untargeted anatomy augmentation (the default plan) is a weaker signal than pathology-targeted warping; call this out explicitly if pursuing this option.

---



## Option 3: Uncertainty-Guided Boundary Refinement

**Research question.** Does a model that predicts its own voxel-wise uncertainty reliably flag the boundaries that are hardest to segment (eCSF, GM, dGM), and can that signal drive a small, targeted refinement step that improves boundary accuracy — without training a full second-network DAE?

**Novel approach & why it's novel.** Extend the shared U-Net with a second, small output head predicting per-voxel uncertainty (heteroscedastic/aleatoric uncertainty via a predicted-variance term in the loss, or Monte Carlo Dropout at inference for a cheaper epistemic-uncertainty estimate). Then apply a lightweight, rule-based refinement — a small local soft-relabeling step, in the spirit of the partial-volume correction from the foundational reconstruction papers — only inside voxels the model flags as high-uncertainty and near a class boundary. This directly operationalizes the FeTA 2024 organizers' "integrate uncertainty quantification" recommendation, fused with a scaled-down version of a much older partial-voluming correction idea — a combination none of the reviewed papers actually tried together.

**Architecture guidance.** Shared single 3D U-Net plus one small uncertainty head branching off the final decoder layer (minimal extra parameters). No ensembling, no second network — the "cleanup" step is a rule, not a trained model.

**Training strategy.** Composite loss: Dice+CE (segmentation) plus a calibration-aware term (negative log-likelihood under the predicted variance) — or, for the cheaper route, plain Dice+CE with Monte Carlo Dropout enabled only at inference (still one trained model, just multiple forward passes at test time). Hyperparameter-tuning axis: the uncertainty loss weight (if using the variance-head route) and the refinement threshold — tune the latter specifically on the validation set.

**Preprocessing pipeline.** Shared baseline only; refinement is a post-inference step, no extra preprocessing.

**Key risks / assumptions.** The refinement step's value is entirely gated on whether predicted uncertainty actually correlates with real errors — check this early (a reliability plot: predicted uncertainty vs. observed error) before investing in the refinement rule itself. If calibration is poor, the segmentation-plus-uncertainty-head model is still a valid, still-reasonably-novel submission on its own; the refinement step becomes optional rather than load-bearing.

---



## Choosing between the three


|                                        | Option 1: Topology-in-the-Loss                      | Option 2: Augmentation-Only                                                            | Option 3: Uncertainty-Guided Refinement                                                   |
| -------------------------------------- | --------------------------------------------------- | -------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------- |
| Where the novelty lives                | Loss function                                       | Data/training pipeline                                                                 | Output head + lightweight post-processing rule                                            |
| Extra moving parts vs. shared baseline | One new loss term                                   | One augmentation pipeline                                                              | One extra output head + a rule-based step                                                 |
| Main implementation risk               | Topology loss gradient instability                  | Untargeted anatomy augmentation is a weaker signal without pathology labels            | Refinement is only as good as uncertainty calibration                                     |
| Directly answers                       | 2024 organizers' "topology in the loss" future work | The un-tested question of how much augmentation alone (vs. ensembling/DAE) contributes | 2024 organizers' "uncertainty quantification" future work + classic partial-voluming idea |
| Clean ablation story for the report?   | Yes — one loss term on/off                          | Yes — augmentation strength on/off/aggressive, plus normal-vs-hard validation gap      | Yes — with/without refinement, uncertainty-error correlation                              |


All three are single-model, single-backbone, same-split, same-metrics — so whichever is chosen, the "Shared Foundations" section above is the starting implementation regardless, and only the option-specific piece needs to be decided as a team.