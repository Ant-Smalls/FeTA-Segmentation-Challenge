# FeTA Fetal Brain Segmentation Challenge

This context defines the domain language for the FeTA (Fetal Tissue Annotation and Segmentation) challenge, which focuses on automatic multi-tissue segmentation of fetal brain MRI to support diagnosis of congenital diseases and study of fetal neurodevelopment.

## Language

### Challenge & Dataset

**FeTA Dataset**:
A multi-center dataset of T2-weighted fetal brain MRI scans with manual segmentation labels for 7 brain tissue types, designed to test cross-center generalizability of segmentation algorithms.
*Avoid*: FeTA data, fetal dataset

**Gestational Age (GA)**:
The developmental age of the fetus measured in weeks from conception, ranging from 19 to 39 weeks in the dataset; a critical variable because fetal brain anatomy changes dramatically across this span.
*Avoid*: Fetal age, weeks, developmental stage

**Cross-center generalizability**:
The ability of a segmentation model to perform accurately on scans from medical institutions not seen during training, accounting for variations in scanner hardware, imaging protocols, and patient populations.
*Avoid*: Multi-center performance, domain transfer, hospital generalization

**SRR (Super-Resolution Reconstruction)**:
The process of reconstructing a single high-resolution 3D volume from multiple motion-corrupted 2D slice stacks acquired during fetal MRI scanning.
*Avoid*: 3D reconstruction, super-resolution, volume reconstruction

### Anatomical Structures

**External Cerebrospinal Fluid (eCSF)**:
The fluid-filled space surrounding the outer surface of the fetal brain, appearing as the largest volume structure in early gestational ages (Label 1).
*Avoid*: Outer CSF, subarachnoid space

**Grey Matter (GM)**:
The cortical tissue forming the outer layer of the cerebral hemispheres, appearing thin and ribbon-like in fetal brains (Label 2).
*Avoid*: Cortex, cortical grey matter

**White Matter (WM)**:
The inner brain tissue composed of myelinated nerve fibers, occupying the largest volume in later gestational ages (Label 3).
*Avoid*: Cerebral white matter

**Ventricles**:
The fluid-filled cavities within the brain; pathological enlargement (ventriculomegaly) is a key diagnostic marker for fetal abnormalities (Label 4).
*Avoid*: Ventricular system, CSF spaces

**Cerebellum**:
The hindbrain structure responsible for motor coordination, located at the back and base of the brain (Label 5).
*Avoid*: Cerebellar hemispheres

**Deep Grey Matter (dGM)**:
Subcortical nuclei including the basal ganglia and thalamus, appearing as small, compact structures with low contrast against surrounding tissue (Label 6).
*Avoid*: Basal ganglia, subcortical grey matter

**Brainstem**:
The structure connecting the brain to the spinal cord, including the midbrain, pons, and medulla (Label 7).
*Avoid*: Brain stem, hindbrain stem

### Image Quality & Acquisition Issues

**Motion artifacts**:
Blurring and misalignment in MRI images caused by unpredictable fetal and maternal movement during the multi-minute scanning session, the primary challenge in fetal MRI.
*Avoid*: Movement artifacts, motion corruption

**Bias field**:
Spatially varying intensity inhomogeneity in MRI images caused by the fetus changing position relative to the scanner's magnetic field during acquisition.
*Avoid*: Intensity inhomogeneity, RF inhomogeneity, bias artifacts

**Outlier slices**:
Individual 2D slices that are so severely corrupted by motion that they must be entirely excluded from the reconstruction rather than down-weighted.
*Avoid*: Corrupted slices, bad slices, motion-corrupted frames

**Partial voluming**:
The phenomenon where a single MRI voxel contains a mixture of multiple tissue types due to limited scanner resolution, causing ambiguous boundaries especially for thin structures.
*Avoid*: Partial volume effect, mixed voxels

**Low-resolution (LR) stacks**:
The raw 2D slice acquisitions obtained during fetal MRI scanning, typically 3 to 6 orthogonal stacks per subject.
*Avoid*: 2D slices, acquisition stacks, slice stacks

### Reconstruction Techniques

**Total Variation (TV)**:
A convex optimization regularization method that penalizes the total amount of gradient in an image while preserving sharp edges, used in SRR to prevent blurring of tissue boundaries.
*Avoid*: TV regularization, total variation regularization

**Exact TV**:
A Total Variation optimization approach that solves the TV problem without smoothing approximations, yielding sharper edges than epsilon-TV methods.
*Avoid*: Unsmoothed TV, precise TV

**PDHG (Primal-Dual Hybrid Gradient)**:
An accelerated optimization algorithm that solves the exact TV problem with quadratic convergence rate, significantly faster than traditional gradient descent.
*Avoid*: Primal-dual method, hybrid gradient algorithm

**Adaptive regularization**:
A data-driven strategy that automatically selects the optimal regularization weight (lambda) for each individual subject's scan based on image quality metrics like PSNR.
*Avoid*: Subject-specific regularization, dynamic lambda tuning

**Slice-to-volume registration**:
The process of aligning each 2D slice to a 3D reference volume to correct for motion, typically using 6 degrees-of-freedom rigid transformation.
*Avoid*: Motion correction, registration

**EM (Expectation-Maximization) outlier rejection**:
A statistical framework that classifies each slice or voxel as either an inlier (modeled as Gaussian) or outlier (modeled as uniform distribution) to enable complete exclusion of corrupted data.
*Avoid*: EM robust statistics, statistical outlier removal

**Intensity matching**:
Correcting for slice-dependent scaling factors and differential bias fields during reconstruction to ensure uniform contrast across the final 3D volume.
*Avoid*: Intensity normalization, intensity correction

### Segmentation Approaches & Architectures

**nnU-Net**:
A self-configuring 3D U-Net CNN framework that automatically adapts its architecture and training strategy to the specific properties of a medical imaging dataset, the foundation of winning FeTA solutions.
*Avoid*: nnUNet, no-new-Net, self-configuring U-Net

**3D U-Net**:
A convolutional neural network architecture with encoder-decoder structure and skip connections, operating on volumetric 3D data for semantic segmentation.
*Avoid*: U-Net, volumetric U-Net

**Baseline model**:
The control 3D U-Net: one segmentation head, Dice plus cross-entropy only, no uncertainty signal and no refinement.
*Avoid*: control network, Role 2 model, confident model

**Comparative model**:
The treatment 3D U-Net: the same backbone as the baseline model, plus a predicted-variance head and an optional refinement rule.
*Avoid*: UncertaintyUNet, treatment model, uncertainty model

**Predicted variance**:
A per-voxel aleatoric value from the 1-channel training head. It is a learned log-variance of label noise, not disagreement with the manual label.
*Avoid*: uncertainty head output, aleatoric map, variance map

**Predictive entropy**:
A per-voxel epistemic value computed at inference from Monte Carlo Dropout. It is high when repeated stochastic forward passes disagree on the class. It is not a distance or entropy between the prediction and the ground-truth boundary.
*Avoid*: MC entropy, dropout uncertainty, model disagreement

**Uncertainty map**:
The voxel-wise signal used at inference for refinement and explainability. In this project it means predictive entropy unless stated otherwise.
*Avoid*: uncertainty, unsure map

**Refinement**:
A reversible rule: a local majority vote on voxels that are both a class boundary and above an uncertainty threshold. The goal is better overlap with the manual labels, not lower predictive entropy. It is not a second trained network.
*Avoid*: cleanup, tidy step, post-processing network, DAE

**Calibration check**:
The test of whether high predictive entropy coincides with voxels the model gets wrong versus the manual labels.
*Avoid*: reliability plot, uncertainty-error correlation

**Denoising Autoencoder (DAE)**:
A post-processing neural network trained to fix topological errors (holes, disconnections) in initial segmentation predictions by learning to map noisy outputs to clean anatomically plausible masks.
*Avoid*: Denoiser, cleanup network, post-processing network

**Topology-aware loss**:
A loss term added during training that penalizes anatomically implausible predictions (holes, disconnected components) directly, as an alternative to fixing them with a separate post-hoc DAE.
*Avoid*: Topology loss, shape-aware loss, clDice loss

**Ensemble**:
Combining predictions from multiple independently trained models by averaging their outputs to improve robustness and reduce prediction variance on unseen data.
*Avoid*: Model averaging, ensemble voting

**Multi-atlas segmentation**:
Propagating manual labels from multiple template atlases to a target image via deformable registration and fusing the warped labels using voting or probabilistic fusion.
*Avoid*: Atlas-based segmentation, label fusion

**4D spatiotemporal atlas**:
A dynamic brain template that continuously varies across gestational age, constructed using kernel regression in time and diffeomorphic registration in space.
*Avoid*: Temporal atlas, age-specific atlas, dynamic atlas

**STAPLE (Simultaneous Truth and Performance Level Estimation)**:
A probabilistic label fusion algorithm that weights different segmentations based on their estimated local quality to predict the true underlying segmentation.
*Avoid*: Probabilistic fusion, weighted label fusion

### Evaluation Metrics

**Dice Similarity Coefficient (DSC)**:
The primary overlap metric measuring segmentation accuracy, calculated as 2×(intersection)/(sum of volumes), ranging from 0 (no overlap) to 1 (perfect overlap).
*Avoid*: Dice score, Dice coefficient, F1 score

**Euler characteristic difference (ED)**:
A topology metric introduced in FeTA 2024 that quantifies the number of holes and disconnected components in a segmentation, measuring anatomical plausibility rather than just overlap.
*Avoid*: Topological error, topology score

**PSNR (Peak Signal-to-Noise Ratio)**:
An image quality metric used to automatically tune reconstruction regularization parameters by comparing reconstructed volumes against original slices.
*Avoid*: Signal-to-noise ratio, reconstruction quality

### Data Augmentation Strategies

**GIN (Global Intensity Non-linearity)**:
An augmentation technique using random neural networks to drastically alter image contrast and brightness, forcing models to learn anatomy rather than appearance.
*Avoid*: Intensity augmentation, contrast augmentation

**Random style augmentation**:
Applying the visual style of unrelated images (e.g., from ImageNet) to medical scans to simulate the appearance variations across different MRI scanners and protocols.
*Avoid*: Style transfer augmentation, appearance augmentation

**Deformable registration augmentation**:
Warping healthy brain scans to match the shape of pathological brains using non-linear registration (e.g., SyN), artificially generating diverse diseased anatomy for training.
*Avoid*: Anatomical augmentation, shape warping, SyN augmentation

**Motion artifact simulation**:
Injecting synthetic motion corruption into clean training images to teach models robustness to the motion artifacts present in real fetal MRI.
*Avoid*: Simulated motion, artificial motion

### Preprocessing & Quality Control

**Skull-stripping**:
Removing non-brain tissue (skull, scalp, maternal tissue) from fetal MRI to focus segmentation on brain structures only.
*Avoid*: Brain extraction, skull removal

**N4 bias field correction**:
A standard preprocessing algorithm that estimates and removes smooth intensity inhomogeneities from MRI scans.
*Avoid*: Bias correction, N4 normalization

**Z-score normalization**:
Standardizing image intensities to have zero mean and unit variance using per-image statistics.
*Avoid*: Intensity standardization, intensity normalization

**Leave-one-out evaluation**:
A validation strategy for reconstruction algorithms where one LR stack is excluded, the volume is reconstructed from remaining stacks, and error is measured against the excluded stack.
*Avoid*: LOO validation, stack exclusion validation