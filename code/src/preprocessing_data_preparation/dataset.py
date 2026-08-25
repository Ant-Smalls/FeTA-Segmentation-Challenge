"""FeTADataset: the frozen contract every other role builds against

- Single class, ``mode='train'|'eval'``. Both modes share the exact same
  load/crop/resample/normalize code path (preprocessing.preprocess_case) --
  only the last step differs: train samples+augments a patch, eval returns
  the full preprocessed volume.
- Train mode ``__getitem__`` -> ``(image, label)`` 2-tuple. Eval mode ->
  ``(image, label, meta)`` 3-tuple, ``meta = {subject_id, affine, spacing}``.
- Tensor contract (frozen, both modes): image float32 shaped (1, D, H, W);
  label int64 shaped (D, H, W) with raw class indices 0-7.
- Train mode: one random patch sampled per case per __getitem__ call;
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

import numpy as np
import torch
import yaml
from torch.utils.data import Dataset

from .preprocessing import apply_light_augmentation, preprocess_case
from .qc_and_split import case_paths


class FeTADataset(Dataset):
    """Dataset over FeTA cases, in either patch-sampling ('train') or
    full-volume ('eval') mode."""

    def __init__(
        self,
        split_path: Path,
        config_path: Path,
        split_name: Literal["train", "val", "test"],
        mode: Literal["train", "eval"],
        data_root: Path | None = None,
        seed: int | None = None,
    ) -> None:
        with open(split_path) as f:
            split_artifact = json.load(f)
        with open(config_path) as f:
            self.config = yaml.safe_load(f)

        self.mode = mode
        self.split_name = split_name
        self.cases = split_artifact["assignments"][split_name]
        if not self.cases:
            raise ValueError(f"Split '{split_name}' has zero cases in {split_path}.")

        code_root = Path(config_path).resolve().parents[2]
        self.data_root = Path(data_root) if data_root is not None else code_root / self.config["paths"]["data_root"]

        target_spacing = self.config["preprocessing"]["target_spacing_mm"]
        self.target_spacing = tuple(target_spacing) if target_spacing is not None else None

        self.patch_size = tuple(self.config["patch_sampling"]["patch_size"])
        self.foreground_oversample_prob = self.config["patch_sampling"]["foreground_oversample_prob"]
        self.aug_config = self.config["augmentation"]

        self.rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return len(self.cases)

    def _preprocessed_case(self, index: int):
        entry = self.cases[index]
        t2w_path, dseg_path = case_paths(self.data_root, entry["subject_id"], entry["rec_type"])
        image, label, meta = preprocess_case(t2w_path, dseg_path, self.target_spacing)
        return entry, image, label, meta

    def __getitem__(self, index: int):
        entry, image, label, meta = self._preprocessed_case(index)

        if self.mode == "train":
            image, label = sample_foreground_oversampled_patch(
                image, label, self.patch_size, self.foreground_oversample_prob, self.rng
            )
            if self.aug_config["enabled_in_train_mode_only"]:
                image, label = apply_light_augmentation(image, label, self.rng, self.aug_config)
            image_t = torch.from_numpy(np.ascontiguousarray(image, dtype=np.float32)).unsqueeze(0)
            label_t = torch.from_numpy(np.ascontiguousarray(label, dtype=np.int64))
            return image_t, label_t

        image_t = torch.from_numpy(np.ascontiguousarray(image, dtype=np.float32)).unsqueeze(0)
        label_t = torch.from_numpy(np.ascontiguousarray(label, dtype=np.int64))
        eval_meta = {"subject_id": entry["subject_id"], "affine": meta["affine"], "spacing": meta["spacing"]}
        return image_t, label_t, eval_meta


def sample_foreground_oversampled_patch(
    image: np.ndarray,
    label: np.ndarray,
    patch_size: tuple[int, int, int],
    foreground_oversample_prob: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Sample one train-mode patch, forcing foreground-of-a-random-class
    centering with probability ``foreground_oversample_prob`` (nnU-Net
    convention default: ~0.33), otherwise a fully random-location patch."""

    image, label = _pad_to_at_least(image, label, patch_size)
    shape = image.shape

    if rng.random() < foreground_oversample_prob:
        foreground_classes = [c for c in np.unique(label) if c != 0]
    else:
        foreground_classes = []

    if foreground_classes:
        chosen_class = foreground_classes[rng.integers(len(foreground_classes))]
        candidates = np.argwhere(label == chosen_class)
        center = candidates[rng.integers(len(candidates))]
        starts = [int(np.clip(c - p // 2, 0, s - p)) for c, p, s in zip(center, patch_size, shape)]
    else:
        starts = [int(rng.integers(0, s - p + 1)) for s, p in zip(shape, patch_size)]

    slices = tuple(slice(start, start + p) for start, p in zip(starts, patch_size))
    return image[slices], label[slices]


def _pad_to_at_least(
    image: np.ndarray, label: np.ndarray, patch_size: tuple[int, int, int]
) -> tuple[np.ndarray, np.ndarray]:
    """Pad small (post-crop) volumes up to at least patch_size. Image padded with
    its own minimum value; label padded with background (0)."""

    pad_widths = [(0, max(0, p - s)) for s, p in zip(image.shape, patch_size)]
    if all(w == (0, 0) for w in pad_widths):
        return image, label
    image = np.pad(image, pad_widths, mode="constant", constant_values=float(image.min()))
    label = np.pad(label, pad_widths, mode="constant", constant_values=0)
    return image, label


def reconstruct_from_patches(
    patches: list[torch.Tensor],
    patch_coords: list[tuple[slice, slice, slice]],
    full_volume_shape: tuple[int, int, int],
    aggregation: Literal["mean", "gaussian"] = "gaussian",
) -> torch.Tensor:
    """Stitch overlapping sliding-window patch predictions (each shaped
    (C, *patch_shape)) back into one full preprocessed-space volume shaped
    (C, *full_volume_shape). Used by uncertainty averaging over
    overlaps and Dice/ED scoring."""

    if len(patches) != len(patch_coords):
        raise ValueError("patches and patch_coords must have the same length.")
    if not patches:
        raise ValueError("Cannot reconstruct from an empty patch list.")

    num_channels = patches[0].shape[0]
    dtype = patches[0].dtype
    device = patches[0].device

    accumulator = torch.zeros((num_channels, *full_volume_shape), dtype=dtype, device=device)
    weight_map = torch.zeros(full_volume_shape, dtype=dtype, device=device)

    for patch, coords in zip(patches, patch_coords):
        patch_shape = tuple(s.stop - s.start for s in coords)
        weight = _patch_weight(patch_shape, aggregation, dtype=dtype, device=device)
        accumulator[(slice(None), *coords)] += patch * weight
        weight_map[coords] += weight

    if torch.any(weight_map == 0):
        raise ValueError("Some voxels in full_volume_shape were never covered by a patch.")

    return accumulator / weight_map.unsqueeze(0)


def _patch_weight(
    patch_shape: tuple[int, int, int], aggregation: str, dtype: torch.dtype, device: torch.device
) -> torch.Tensor:
    if aggregation == "mean":
        return torch.ones(patch_shape, dtype=dtype, device=device)

    if aggregation == "gaussian":
        # Separable Gaussian peaking at the patch center, downweighting patch edges
        def gauss(n: int) -> torch.Tensor:
            coord = torch.arange(n, dtype=dtype, device=device) - (n - 1) / 2
            return torch.exp(-0.5 * (coord / (n / 4)) ** 2)

        d, h, w = patch_shape
        gd, gh, gw = gauss(d), gauss(h), gauss(w)
        return gd.view(d, 1, 1) * gh.view(1, h, 1) * gw.view(1, 1, w)

    raise ValueError(f"Unknown aggregation mode: {aggregation}")
