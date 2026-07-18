# FeTA Fetal Brain Segmentation — Project Roles

Six roles, carried over from the team's MRNet project structure (data preprocessing & transformation, baseline models, comparative models, training pipeline & hyperparameter tuning, evaluation & metrics, explainability & clinical presentation) because that split worked well against the same assignment rubric. The roles are identical regardless of which of the [three research options](./initial_research_plan.md) the team picks — what changes is *what each role actually does*, broken out below.

One naming note carried over from MRNet, sharpened for this project: **"Baseline model"** here specifically means *the ablation control for that option* — the shared U-Net trained without the option's novel mechanism — not a generic "simple" architecture. **"Comparative model"** means the treatment condition with the novel mechanism turned on. Every option in this project is structured as a clean baseline-vs-comparative ablation, so this pairing maps unusually well here: whoever owns "baseline" and whoever owns "comparative" are effectively running the same experiment with one thing switched on or off, and must coordinate closely on shared settings (split, optimizer, epoch budget) so the comparison stays fair.

## The 6 roles, in general

1. **Data preprocessing & transformation** — owns the `FeTADataset` class and everything upstream of the model: loading, normalization, cropping/resampling, patch sampling, and (where relevant) augmentation.
2. **Baseline models** — owns training the shared 3D U-Net with plain Dice+CE loss and no option-specific novelty; the yardstick everything else is measured against.
3. **Comparative models** — owns implementing and training the option-specific novel mechanism (the extra loss term, the augmented-data model, or the extra output head).
4. **Training pipeline & hyperparameter tuning** — owns the shared training loop (optimizer, LR schedule, early stopping, checkpointing) used by both baseline and comparative models, and runs the option's specific hyperparameter sweep.
5. **Evaluation & metrics** — owns computing Dice, per-class Dice, and Euler characteristic difference (ED) on the test split for both models, plus any option-specific evaluation, and the statistical comparison between them.
6. **Explainability & clinical presentation** — owns turning results into a clinically legible story: slice overlays, failure-case visuals, and framing tied to real clinical ambiguity (e.g. eCSF/GM boundary disagreement) for the "clinical feasibility" rubric criterion.

---

## Option 1: Topology-in-the-Loss

**1. Data preprocessing & transformation.** Standard shared pipeline (z-score normalization, foreground crop, foreground-oversampled patch sampling). Additionally owns implementing the soft-skeletonization / boundary-map utility the topology loss needs (e.g. a small fixed-kernel erosion-based skeleton approximation for clDice), built once and reused both inside the training loss and later inside evaluation's topology visualizations, so the definition of "skeleton" is consistent everywhere it's used.

**2. Baseline models.** Trains the shared 3D U-Net with Dice+CE only — no topology term. This is the direct stand-in for "current practice without a cleanup network." If time allows, this role can also train a simple post-hoc DAE on this baseline's outputs as a secondary reference point, showing where the field's existing post-processing approach lands relative to the loss-level approach in role 3.

**3. Comparative models.** Implements the topology-aware loss term itself (clDice-style skeleton-overlap penalty, with a boundary/surface-distance loss as a documented fallback if clDice proves unstable), wires it into the model's forward/loss computation with a configurable per-class weight (heavier for eCSF, GM, dGM), and implements the curriculum schedule that ramps the topology weight up only after the segmentation head has stabilized on Dice+CE.

**4. Training pipeline & hyperparameter tuning.** Runs both the baseline and topology-loss models under identical optimizer, LR schedule, and early-stopping settings so only the loss differs. Owns the hyperparameter sweep over topology loss weight and ramp-up length (e.g. off / low / medium / high), selecting the final configuration using validation Dice *and* ED jointly, since a heavy topology weight could trade raw overlap accuracy for topological correctness.

**5. Evaluation & metrics.** Computes Dice/ED per class for baseline vs. topology-loss model on the held-out test split; the headline result is the ED improvement achieved without a second network, directly comparable to the ~50% ED gain cesne-digair's DAE achieved. Also produces a plain-language "hole/disconnected-component count per class" table as a more literal companion to ED for the report.

**6. Explainability & clinical presentation.** Produces side-by-side slice visualizations showing specific holes or disconnected blobs in the baseline's GM/eCSF predictions that the topology-loss model closes, framed as "fewer anatomically impossible shapes a radiologist would otherwise have to manually correct."

---

## Option 2: Augmentation-Only Generalization Attack

**1. Data preprocessing & transformation.** This role carries the most weight in this option. Owns the full augmentation pipeline inside the `FeTADataset`/transform layer: synthetic bias-field simulation, contrast/gamma jitter, and deformable (B-spline/SyN-style) warps between randomly paired training cases, all behind a configurable strength parameter. Also builds a separate, held-out "hard" synthetic validation set (stronger versions of the same corruptions) used only for evaluation — never seen during training by any model.

**2. Baseline models.** Trains the shared 3D U-Net on the shared preprocessing baseline plus only default nnU-Net-style augmentation (flips/rotation/basic noise) — i.e. no appearance or anatomy augmentation from role 1. This is the "no augmentation" reference point.

**3. Comparative models.** Trains the architecturally identical U-Net on the augmented pipeline from role 1, at two or three strength settings (e.g. off / moderate / aggressive) so the effect is clearly attributable to augmentation strength and not noise. Since the architecture doesn't change, this role's real work is verifying the augmented data pipeline is wired into training correctly and training each strength variant under the same budget as the baseline.

**4. Training pipeline & hyperparameter tuning.** Owns the shared training loop and ensures the baseline and every augmentation-strength variant see the same number of epochs/steps, so augmentation policy is the only thing that differs. The hyperparameter-tuning axis here *is* the augmentation strength sweep itself, run in coordination with role 3.

**5. Evaluation & metrics.** Evaluates every trained variant (baseline + each augmentation strength) on both the normal test split and the "hard" synthetic set from role 1. The headline result is the *gap* between normal-set and hard-set performance for each variant — a shrinking gap as augmentation strength increases is the direct evidence answering the research question, not the hard-set score in isolation.

**6. Explainability & clinical presentation.** Visualizes specific cases where the baseline fails badly on the "hard" synthetic set (e.g. under a simulated bias field) while the augmented model holds up, framed as robustness to the scanner/protocol variability a real multi-institution deployment would face.

---

## Option 3: Uncertainty-Guided Boundary Refinement

**1. Data preprocessing & transformation.** Standard shared pipeline only — no option-specific preprocessing burden, since refinement happens post-inference. This role can additionally help role 6 extract the specific hard-boundary slices (eCSF/GM/dGM regions) used for the uncertainty visualizations.

**2. Baseline models.** Trains the shared 3D U-Net with Dice+CE only, single output head, no uncertainty estimation — the "confident but possibly wrong" reference model.

**3. Comparative models.** Implements the second output head (a predicted-variance term for heteroscedastic/aleatoric uncertainty, or wires up Monte Carlo Dropout for a cheaper epistemic-uncertainty estimate at inference) and the composite loss if using the variance-head route. Also implements the rule-based boundary-refinement step (local soft-relabeling in high-uncertainty voxels near class boundaries) as a clearly separable post-inference function that can be toggled on/off for the ablation.

**4. Training pipeline & hyperparameter tuning.** Owns the shared training loop plus two option-specific tunable axes: the uncertainty loss weight (variance-head route only) and, more importantly, the refinement threshold — how uncertain a voxel must be before the rule intervenes — tuned specifically on the validation set since it directly trades off refinement aggressiveness against risk of introducing new errors.

**5. Evaluation & metrics.** In addition to Dice/ED, owns the calibration check: a reliability plot of predicted uncertainty vs. observed voxel-wise error, which must look reasonable *before* the refinement step's results are treated as meaningful. Also reports before/after-refinement Dice/ED specifically on eCSF, GM, and dGM, since that's where the mechanism is supposed to help most.

**6. Explainability & clinical presentation.** Produces uncertainty heatmaps overlaid on MRI slices next to ground truth and prediction, showing whether high-uncertainty regions line up with boundaries radiologists themselves disagree on (eCSF/GM in particular) — directly tying the model's self-reported confidence to a documented clinical phenomenon, a strong basis for the "clinical feasibility" narrative.

---

## Cross-option note for whoever ends up choosing

Roles 1 (data) and 4 (training pipeline) are the most option-dependent in scope — Option 2 makes role 1 the heaviest lift of any role in any option, while Option 1 and Option 3 keep role 1 close to the shared baseline. Roles 2 and 5 are the most *consistent* across options — "train the plain baseline" and "compute Dice/ED and compare" look almost identical no matter which option is picked, which makes them a safe assignment for team members who want to ramp up on the codebase before committing to the option's specific novel mechanism.
