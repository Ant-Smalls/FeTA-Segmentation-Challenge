"""
Cohort context for the severe FeTA test-set failure cases.

Purpose
-------
Determine whether sub-017, sub-020 and sub-025 are extreme relative to the
full usable 79-case cohort with respect to the ground-truth characteristics
already implicated by the targeted six-case failure analysis.

This is an exploratory contextual analysis, not a population-level
hypothesis-testing exercise.

Features
--------
- total labelled brain volume
- eCSF volume
- eCSF fraction of total labelled brain
- eCSF connected-component count
- eCSF Euler number
- eCSF surface-to-volume ratio

The existing frozen split_v1.json is used exactly as committed. The QC split
is not regenerated.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
from scipy.ndimage import label as connected_components
from skimage.measure import (
    euler_number,
    marching_cubes,
    mesh_surface_area,
)


ROOT = Path(__file__).resolve().parents[2]

DATA = ROOT / "src/data/mri_gz"
SPLIT_PATH = ROOT / "src/data/splits/split_v1.json"
OUT = ROOT / "src/robustness/outputs"

OUT.mkdir(parents=True, exist_ok=True)

SEVERE = {
    "sub-017",
    "sub-020",
    "sub-025",
}

ECSF_CLASS_ID = 1


def load_frozen_cohort():
    """
    Read the exact usable cohort from split_v1.json.

    Returns one record per subject with:
        subject_id
        split
        rec_type
    """
    with open(SPLIT_PATH) as f:
        split = json.load(f)

    subjects = []

    for split_name, entries in split["assignments"].items():
        for entry in entries:
            subjects.append(
                {
                    "subject_id": entry["subject_id"],
                    "split": split_name,
                    "rec_type": entry["rec_type"],
                }
            )

    return sorted(
        subjects,
        key=lambda row: row["subject_id"],
    )


def segmentation_path(
    subject_id: str,
    rec_type: str,
) -> Path:
    """
    Construct the raw FeTA segmentation path.
    """
    return (
        DATA
        / f"{subject_id}_rec-{rec_type}_dseg.nii.gz"
    )


def component_count(mask: np.ndarray) -> int:
    if not mask.any():
        return 0

    _, n = connected_components(mask)

    return int(n)


def surface_area_mm2(
    mask: np.ndarray,
    spacing,
) -> float:
    """
    Estimate physical surface area using marching cubes.
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


def empirical_percentile(
    value: float,
    cohort_values,
) -> float:
    """
    Percentage of cohort observations less than or equal to value.

    Example:
        percentile = 2 means the value lies near the low end.
        percentile = 98 means the value lies near the high end.
    """
    values = np.asarray(
        cohort_values,
        dtype=float,
    )

    values = values[
        np.isfinite(values)
    ]

    return float(
        100.0
        * np.mean(values <= value)
    )


cohort = load_frozen_cohort()

print(
    f"Frozen usable cohort: {len(cohort)} subjects"
)

if len(cohort) != 79:
    raise RuntimeError(
        f"Expected 79 usable subjects, found {len(cohort)}."
    )


rows = []


for index, subject in enumerate(
    cohort,
    start=1,
):

    subject_id = subject["subject_id"]
    rec_type = subject["rec_type"]

    path = segmentation_path(
        subject_id,
        rec_type,
    )

    if not path.exists():
        raise FileNotFoundError(
            f"Missing segmentation: {path}"
        )

    nii = nib.load(path)

    segmentation = np.rint(
        np.asarray(nii.dataobj)
    ).astype(np.int16)

    spacing = tuple(
        float(x)
        for x in nii.header.get_zooms()[:3]
    )

    voxel_volume_mm3 = float(
        np.prod(spacing)
    )

    brain_mask = segmentation > 0
    ecsf_mask = segmentation == ECSF_CLASS_ID

    brain_voxels = int(
        brain_mask.sum()
    )

    ecsf_voxels = int(
        ecsf_mask.sum()
    )

    brain_volume_cm3 = (
        brain_voxels
        * voxel_volume_mm3
        / 1000.0
    )

    ecsf_volume_cm3 = (
        ecsf_voxels
        * voxel_volume_mm3
        / 1000.0
    )

    ecsf_fraction = (
        ecsf_volume_cm3
        / brain_volume_cm3
        if brain_volume_cm3 > 0
        else float("nan")
    )

    n_components = component_count(
        ecsf_mask
    )

    ecsf_euler = (
        int(
            euler_number(
                ecsf_mask,
                connectivity=1,
            )
        )
        if ecsf_mask.any()
        else 0
    )

    area_mm2 = surface_area_mm2(
        ecsf_mask,
        spacing,
    )

    ecsf_volume_mm3 = (
        ecsf_volume_cm3
        * 1000.0
    )

    surface_to_volume = (
        area_mm2 / ecsf_volume_mm3
        if (
            ecsf_volume_mm3 > 0
            and np.isfinite(area_mm2)
        )
        else float("nan")
    )

    rows.append(
        {
            "subject_id": subject_id,
            "split": subject["split"],
            "rec_type": rec_type,
            "is_severe_failure": (
                subject_id in SEVERE
            ),
            "spacing_mm": spacing[0],
            "brain_volume_cm3": brain_volume_cm3,
            "ecsf_volume_cm3": ecsf_volume_cm3,
            "ecsf_fraction_of_brain": ecsf_fraction,
            "ecsf_connected_components": n_components,
            "ecsf_euler_number": ecsf_euler,
            "ecsf_surface_area_mm2": area_mm2,
            "ecsf_surface_to_volume_mm_inv": (
                surface_to_volume
            ),
        }
    )

    print(
        f"[{index:02d}/{len(cohort)}] "
        f"{subject_id} complete"
    )


# ---------------------------------------------------------------------
# Save cohort-level feature table
# ---------------------------------------------------------------------

cohort_path = (
    OUT
    / "cohort_ecsf_context.csv"
)

with open(
    cohort_path,
    "w",
    newline="",
) as f:

    writer = csv.DictWriter(
        f,
        fieldnames=list(
            rows[0].keys()
        ),
    )

    writer.writeheader()
    writer.writerows(rows)


# ---------------------------------------------------------------------
# Percentile context for severe cases
# ---------------------------------------------------------------------

FEATURES = {
    "brain_volume_cm3": (
        "Total brain volume",
        "cm3",
    ),
    "ecsf_volume_cm3": (
        "eCSF volume",
        "cm3",
    ),
    "ecsf_fraction_of_brain": (
        "eCSF fraction of brain",
        "fraction",
    ),
    "ecsf_connected_components": (
        "eCSF connected components",
        "count",
    ),
    "ecsf_euler_number": (
        "eCSF Euler number",
        "count",
    ),
    "ecsf_surface_to_volume_mm_inv": (
        "eCSF surface-to-volume ratio",
        "mm^-1",
    ),
}


percentile_rows = []


print()
print("=" * 82)
print("SEVERE-CASE COHORT CONTEXT")
print("=" * 82)


for subject_id in sorted(SEVERE):

    subject_row = next(
        row
        for row in rows
        if row["subject_id"] == subject_id
    )

    print()
    print(subject_id)
    print("-" * 82)

    for feature, (
        display_name,
        unit,
    ) in FEATURES.items():

        value = float(
            subject_row[feature]
        )

        cohort_values = [
            float(row[feature])
            for row in rows
        ]

        percentile = empirical_percentile(
            value,
            cohort_values,
        )

        percentile_rows.append(
            {
                "subject_id": subject_id,
                "feature": feature,
                "display_name": display_name,
                "value": value,
                "unit": unit,
                "percentile": percentile,
            }
        )

        if unit == "fraction":
            display_value = (
                f"{100 * value:.2f}%"
            )
        else:
            display_value = (
                f"{value:.4f}"
            )

        print(
            f"{display_name:31s} "
            f"{display_value:>12s} "
            f"percentile={percentile:5.1f}"
        )


percentile_path = (
    OUT
    / "severe_case_cohort_percentiles.csv"
)

with open(
    percentile_path,
    "w",
    newline="",
) as f:

    writer = csv.DictWriter(
        f,
        fieldnames=list(
            percentile_rows[0].keys()
        ),
    )

    writer.writeheader()
    writer.writerows(
        percentile_rows
    )


# ---------------------------------------------------------------------
# Figure 1:
# eCSF volume vs fragmentation
# ---------------------------------------------------------------------

fig, ax = plt.subplots(
    figsize=(7.5, 5.5)
)

for row in rows:

    marker = (
        "X"
        if row["subject_id"] in SEVERE
        else "o"
    )

    size = (
        100
        if row["subject_id"] in SEVERE
        else 30
    )

    ax.scatter(
        row["ecsf_volume_cm3"],
        row["ecsf_connected_components"],
        marker=marker,
        s=size,
        alpha=0.75,
    )


for row in rows:

    if row["subject_id"] in SEVERE:

        ax.annotate(
            row["subject_id"],
            (
                row["ecsf_volume_cm3"],
                row[
                    "ecsf_connected_components"
                ],
            ),
            xytext=(6, 6),
            textcoords="offset points",
            fontsize=9,
        )


ax.set_xlabel(
    "Ground-truth eCSF volume (cm3)"
)

ax.set_ylabel(
    "Ground-truth eCSF connected components"
)

ax.set_title(
    "eCSF volume and fragmentation across the usable cohort"
)

ax.grid(
    alpha=0.2
)

fig.tight_layout()

figure_1 = (
    OUT
    / "fig_cohort_ecsf_volume_components.png"
)

fig.savefig(
    figure_1,
    dpi=300,
    bbox_inches="tight",
)

plt.close(fig)


# ---------------------------------------------------------------------
# Figure 2:
# eCSF fraction vs surface complexity
# ---------------------------------------------------------------------

fig, ax = plt.subplots(
    figsize=(7.5, 5.5)
)

for row in rows:

    marker = (
        "X"
        if row["subject_id"] in SEVERE
        else "o"
    )

    size = (
        100
        if row["subject_id"] in SEVERE
        else 30
    )

    ax.scatter(
        100
        * row[
            "ecsf_fraction_of_brain"
        ],
        row[
            "ecsf_surface_to_volume_mm_inv"
        ],
        marker=marker,
        s=size,
        alpha=0.75,
    )


for row in rows:

    if row["subject_id"] in SEVERE:

        ax.annotate(
            row["subject_id"],
            (
                100
                * row[
                    "ecsf_fraction_of_brain"
                ],
                row[
                    "ecsf_surface_to_volume_mm_inv"
                ],
            ),
            xytext=(6, 6),
            textcoords="offset points",
            fontsize=9,
        )


ax.set_xlabel(
    "Ground-truth eCSF fraction of labelled brain (%)"
)

ax.set_ylabel(
    "Ground-truth eCSF surface-to-volume ratio (mm$^{-1}$)"
)

ax.set_title(
    "eCSF relative size and structural complexity"
)

ax.grid(
    alpha=0.2
)

fig.tight_layout()

figure_2 = (
    OUT
    / "fig_cohort_ecsf_fraction_complexity.png"
)

fig.savefig(
    figure_2,
    dpi=300,
    bbox_inches="tight",
)

plt.close(fig)


print()
print("Saved:")
print(f"  {cohort_path}")
print(f"  {percentile_path}")
print(f"  {figure_1}")
print(f"  {figure_2}")
