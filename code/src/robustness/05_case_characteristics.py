"""
Ground-truth and image-characteristic analysis for the six MIAL test cases.

Compares the three severe segmentation failures:
    sub-017, sub-020, sub-025

against the three higher-performing MIAL test cases:
    sub-034, sub-038, sub-039

This analysis uses only the original FeTA images and reference segmentations.
It does not use model predictions and therefore cannot introduce model-specific
bias into the case-characterisation step.
"""

from __future__ import annotations

import csv
from pathlib import Path

import nibabel as nib
import numpy as np
from scipy.ndimage import label as connected_components
from skimage.measure import euler_number, marching_cubes, mesh_surface_area


ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "src/data/mri_gz"
OUT = ROOT / "src/robustness/outputs"
OUT.mkdir(parents=True, exist_ok=True)

SEVERE = {"sub-017", "sub-020", "sub-025"}
CONTROL = {"sub-034", "sub-038", "sub-039"}

SUBJECTS = sorted(SEVERE | CONTROL)

CLASS_NAMES = {
    1: "eCSF",
    2: "GM",
    3: "WM",
    4: "Ventricles",
    5: "Cerebellum",
    6: "dGM",
    7: "Brainstem",
}


def load_case(subject_id: str):
    number = subject_id.split("-")[1]

    image_path = DATA / f"sub-{number}_rec-mial_T2w.nii.gz"
    label_path = DATA / f"sub-{number}_rec-mial_dseg.nii.gz"

    image_nii = nib.load(image_path)
    label_nii = nib.load(label_path)

    image = np.asarray(image_nii.dataobj, dtype=np.float32)
    segmentation = np.rint(
        np.asarray(label_nii.dataobj)
    ).astype(np.int16)

    spacing = tuple(
        float(x)
        for x in image_nii.header.get_zooms()[:3]
    )

    return image, segmentation, spacing


def component_count(mask: np.ndarray) -> int:
    if not mask.any():
        return 0

    _, n_components = connected_components(mask)

    return int(n_components)


def surface_area_mm2(
    mask: np.ndarray,
    spacing: tuple[float, float, float],
) -> float:
    """
    Estimate surface area using marching cubes.

    Returns NaN for an empty or degenerate mask.
    """
    mask = np.asarray(mask, dtype=np.uint8)

    if mask.sum() < 8:
        return float("nan")

    try:
        vertices, faces, _, _ = marching_cubes(
            mask,
            level=0.5,
            spacing=spacing,
        )

        return float(
            mesh_surface_area(
                vertices,
                faces,
            )
        )

    except (ValueError, RuntimeError):
        return float("nan")


def bounding_box_shape(mask: np.ndarray):
    coordinates = np.argwhere(mask)

    if len(coordinates) == 0:
        return (0, 0, 0)

    mins = coordinates.min(axis=0)
    maxs = coordinates.max(axis=0)

    return tuple(
        int(x)
        for x in (
            maxs - mins + 1
        )
    )


case_rows = []
tissue_rows = []


for subject in SUBJECTS:

    image, segmentation, spacing = load_case(subject)

    group = (
        "severe"
        if subject in SEVERE
        else "control"
    )

    voxel_volume_mm3 = float(
        np.prod(spacing)
    )

    foreground = segmentation > 0

    total_brain_voxels = int(
        foreground.sum()
    )

    total_brain_volume_cm3 = (
        total_brain_voxels
        * voxel_volume_mm3
        / 1000.0
    )

    bbox = bounding_box_shape(
        foreground
    )

    foreground_intensities = image[
        foreground
    ]

    intensity_median = float(
        np.median(
            foreground_intensities
        )
    )

    intensity_q25 = float(
        np.percentile(
            foreground_intensities,
            25,
        )
    )

    intensity_q75 = float(
        np.percentile(
            foreground_intensities,
            75,
        )
    )

    intensity_iqr = (
        intensity_q75
        - intensity_q25
    )

    intensity_p01 = float(
        np.percentile(
            foreground_intensities,
            1,
        )
    )

    intensity_p99 = float(
        np.percentile(
            foreground_intensities,
            99,
        )
    )

    case_rows.append(
        {
            "subject_id": subject,
            "group": group,
            "spacing_x_mm": spacing[0],
            "spacing_y_mm": spacing[1],
            "spacing_z_mm": spacing[2],
            "voxel_volume_mm3": voxel_volume_mm3,
            "image_dim_x": image.shape[0],
            "image_dim_y": image.shape[1],
            "image_dim_z": image.shape[2],
            "brain_bbox_x": bbox[0],
            "brain_bbox_y": bbox[1],
            "brain_bbox_z": bbox[2],
            "total_brain_volume_cm3": (
                total_brain_volume_cm3
            ),
            "foreground_intensity_median": (
                intensity_median
            ),
            "foreground_intensity_iqr": (
                intensity_iqr
            ),
            "foreground_intensity_p01": (
                intensity_p01
            ),
            "foreground_intensity_p99": (
                intensity_p99
            ),
        }
    )

    for class_id, class_name in CLASS_NAMES.items():

        mask = (
            segmentation == class_id
        )

        voxels = int(
            mask.sum()
        )

        volume_cm3 = (
            voxels
            * voxel_volume_mm3
            / 1000.0
        )

        proportion = (
            volume_cm3
            / total_brain_volume_cm3
            if total_brain_volume_cm3 > 0
            else float("nan")
        )

        components = component_count(
            mask
        )

        # connectivity=1 corresponds to face-connected 3D neighbourhood.
        euler = (
            int(
                euler_number(
                    mask,
                    connectivity=1,
                )
            )
            if mask.any()
            else 0
        )

        area_mm2 = surface_area_mm2(
            mask,
            spacing,
        )

        volume_mm3 = (
            volume_cm3 * 1000.0
        )

        surface_to_volume = (
            area_mm2 / volume_mm3
            if (
                volume_mm3 > 0
                and np.isfinite(area_mm2)
            )
            else float("nan")
        )

        tissue_rows.append(
            {
                "subject_id": subject,
                "group": group,
                "class_id": class_id,
                "class_name": class_name,
                "volume_cm3": volume_cm3,
                "fraction_of_brain": proportion,
                "connected_components": components,
                "euler_number": euler,
                "surface_area_mm2": area_mm2,
                "surface_to_volume_mm_inv": (
                    surface_to_volume
                ),
            }
        )


case_path = (
    OUT
    / "mial_case_characteristics.csv"
)

with open(
    case_path,
    "w",
    newline="",
) as f:

    writer = csv.DictWriter(
        f,
        fieldnames=list(
            case_rows[0].keys()
        ),
    )

    writer.writeheader()
    writer.writerows(
        case_rows
    )


tissue_path = (
    OUT
    / "mial_tissue_characteristics.csv"
)

with open(
    tissue_path,
    "w",
    newline="",
) as f:

    writer = csv.DictWriter(
        f,
        fieldnames=list(
            tissue_rows[0].keys()
        ),
    )

    writer.writeheader()
    writer.writerows(
        tissue_rows
    )


print("=" * 78)
print("MIAL CASE CHARACTERISTICS")
print("=" * 78)

for row in case_rows:

    print()
    print(
        f"{row['subject_id']} "
        f"({row['group'].upper()})"
    )

    print(
        f"  spacing:       "
        f"{row['spacing_x_mm']:.4f} mm"
    )

    print(
        f"  brain volume:  "
        f"{row['total_brain_volume_cm3']:.2f} cm3"
    )

    print(
        f"  brain bbox:    "
        f"{row['brain_bbox_x']} x "
        f"{row['brain_bbox_y']} x "
        f"{row['brain_bbox_z']} voxels"
    )

    print(
        f"  intensity:     "
        f"median={row['foreground_intensity_median']:.1f}, "
        f"IQR={row['foreground_intensity_iqr']:.1f}"
    )


print()
print("=" * 78)
print("GROUND-TRUTH eCSF CHARACTERISTICS")
print("=" * 78)

ecsf_rows = [
    r
    for r in tissue_rows
    if r["class_name"] == "eCSF"
]

for row in ecsf_rows:

    print()
    print(
        f"{row['subject_id']} "
        f"({row['group'].upper()})"
    )

    print(
        f"  volume:             "
        f"{row['volume_cm3']:.3f} cm3"
    )

    print(
        f"  brain fraction:     "
        f"{100 * row['fraction_of_brain']:.2f}%"
    )

    print(
        f"  connected comps:    "
        f"{row['connected_components']}"
    )

    print(
        f"  Euler number:       "
        f"{row['euler_number']}"
    )

    print(
        f"  surface area:       "
        f"{row['surface_area_mm2']:.1f} mm2"
    )

    print(
        f"  surface/volume:     "
        f"{row['surface_to_volume_mm_inv']:.4f} mm^-1"
    )


print()
print("=" * 78)
print("SEVERE VS CONTROL GROUP SUMMARY")
print("=" * 78)

for class_name in [
    "eCSF",
    "GM",
    "Cerebellum",
    "Brainstem",
]:

    relevant = [
        r
        for r in tissue_rows
        if r["class_name"] == class_name
    ]

    print()
    print(class_name)

    for group in [
        "severe",
        "control",
    ]:

        group_rows = [
            r
            for r in relevant
            if r["group"] == group
        ]

        volumes = np.array(
            [
                r["volume_cm3"]
                for r in group_rows
            ]
        )

        components = np.array(
            [
                r["connected_components"]
                for r in group_rows
            ]
        )

        eulers = np.array(
            [
                r["euler_number"]
                for r in group_rows
            ]
        )

        complexity = np.array(
            [
                r["surface_to_volume_mm_inv"]
                for r in group_rows
            ]
        )

        print(
            f"  {group:7s}: "
            f"volume median={np.median(volumes):.3f} cm3, "
            f"components median={np.median(components):.1f}, "
            f"Euler median={np.median(eulers):.1f}, "
            f"S/V median={np.nanmedian(complexity):.4f}"
        )


print()
print("Saved:")
print(f"  {case_path}")
print(f"  {tissue_path}")
