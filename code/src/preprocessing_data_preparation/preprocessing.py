"""Per-case crop / resample / normalize / augment transforms

- Order: load -> crop to non-zero foreground bbox -> resample spacing outliers
  to the train-split-only median spacing (config.yaml) -> per-volume z-score normalize.
- Evaluation happens in this preprocessed (crop+resample) space, not mapped
  back to original resolution.
- ``preprocess_case`` is the single shared code path both FeTADataset modes
  call.
- Light augmentation (flips/rotation/noise) is train-mode-only and applied at
  the patch level by dataset.py.
"""

from __future__ import annotations

from pathlib import Path

import nibabel as nib
import numpy as np
from scipy.ndimage import rotate as ndi_rotate
from scipy.ndimage import zoom as ndi_zoom

_SPACING_TOLERANCE_MM = 1e-3


def load_case(t2w_path: Path, dseg_path: Path) -> tuple[np.ndarray, np.ndarray, dict]:
    """Load a T2w volume + its dseg label map, returning arrays plus metadata
    (affine, spacing, original_shape)."""

    t2w_img = nib.load(t2w_path)
    dseg_img = nib.load(dseg_path)

    if t2w_img.shape != dseg_img.shape:
        raise ValueError(
            f"T2w/dseg shape mismatch: {Path(t2w_path).name}={t2w_img.shape} "
            f"{Path(dseg_path).name}={dseg_img.shape}"
        )

    image = np.asarray(t2w_img.get_fdata(), dtype=np.float32)
    label = np.asarray(dseg_img.get_fdata()).astype(np.int16)

    meta = {
        "affine": t2w_img.affine,
        "spacing": tuple(float(s) for s in t2w_img.header.get_zooms()[:3]),
        "original_shape": t2w_img.shape,
    }
    return image, label, meta


def crop_to_foreground_bbox(
    image: np.ndarray, label: np.ndarray
) -> tuple[np.ndarray, np.ndarray, tuple[slice, slice, slice]]:
    """Crop image+label to the non-zero foreground bounding box. Returns the
    bbox slices."""

    foreground = image > 0
    if not foreground.any():
        raise ValueError("Image has no non-zero foreground voxels; cannot crop.")

    coords = np.argwhere(foreground)
    mins = coords.min(axis=0)
    maxs = coords.max(axis=0) + 1
    bbox = tuple(slice(int(lo), int(hi)) for lo, hi in zip(mins, maxs))

    return image[bbox], label[bbox], bbox


def resample_to_spacing(
    image: np.ndarray,
    label: np.ndarray,
    current_spacing: tuple[float, float, float],
    target_spacing: tuple[float, float, float] | None,
) -> tuple[np.ndarray, np.ndarray]:
    """Resample image (linear) and label (nearest-neighbor) to target spacing.

    ``target_spacing=None`` (config.yaml's pre-Phase-5 default) and spacing
    already within tolerance of the target both no-op -- true no-op for the
    majority of cases already close to the train-split median spacing.
    """

    if target_spacing is None:
        return image, label

    if all(abs(c - t) <= _SPACING_TOLERANCE_MM for c, t in zip(current_spacing, target_spacing)):
        return image, label

    zoom_factors = tuple(c / t for c, t in zip(current_spacing, target_spacing))
    resampled_image = ndi_zoom(image, zoom_factors, order=1)
    resampled_label = ndi_zoom(label, zoom_factors, order=0)
    return resampled_image.astype(np.float32), resampled_label.astype(label.dtype)


def zscore_normalize(image: np.ndarray) -> np.ndarray:
    """Per-volume z-score normalization (zero mean, unit variance)."""

    mean = float(image.mean())
    std = float(image.std())
    if std == 0.0:
        raise ValueError("Cannot z-score normalize a constant image (std=0).")
    return ((image - mean) / std).astype(np.float32)


def apply_light_augmentation(
    image: np.ndarray, label: np.ndarray, rng: np.random.Generator, aug_config: dict
) -> tuple[np.ndarray, np.ndarray]:
    """Random flips / small rotations / mild noise, applied identically for
    baseline and comparative model training(dataset.py calls this on
    the sampled patch, not the full volume)."""

    for axis in range(image.ndim):
        if rng.random() < aug_config["flip_prob"]:
            image = np.flip(image, axis=axis)
            label = np.flip(label, axis=axis)

    if rng.random() < aug_config["rotate_prob"]:
        axes = tuple(int(a) for a in rng.choice(3, size=2, replace=False))
        angle = float(rng.uniform(-aug_config["rotate_max_degrees"], aug_config["rotate_max_degrees"]))
        image = ndi_rotate(image, angle, axes=axes, reshape=False, order=1, mode="nearest")
        label = ndi_rotate(label, angle, axes=axes, reshape=False, order=0, mode="nearest")

    if rng.random() < aug_config["noise_prob"]:
        image = image + rng.normal(0.0, aug_config["noise_std"], size=image.shape)

    return np.ascontiguousarray(image, dtype=np.float32), np.ascontiguousarray(label, dtype=label.dtype)


def preprocess_case(
    t2w_path: Path,
    dseg_path: Path,
    target_spacing: tuple[float, float, float] | None,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Full per-case pipeline: load -> crop -> resample -> normalize."""

    image, label, meta = load_case(t2w_path, dseg_path)
    image, label, bbox = crop_to_foreground_bbox(image, label)
    image, label = resample_to_spacing(image, label, meta["spacing"], target_spacing)
    image = zscore_normalize(image)

    meta = {**meta, "crop_bbox": bbox}
    return image, label, meta
