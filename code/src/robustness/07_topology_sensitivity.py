"""
Sensitivity analysis for the eCSF fragmentation finding.

Purpose
-------
Test whether the severe failure cases remain topologically unusual when:

1. connected-component connectivity is changed from 6-connected to
   18-connected or 26-connected;
2. very small components are ignored.

This determines whether the extreme component counts are driven mainly by
tiny isolated label fragments or represent more persistent structural
fragmentation.

No model predictions are used.
"""

from __future__ import annotations

import csv
from pathlib import Path

import nibabel as nib
import numpy as np
from scipy.ndimage import (
    generate_binary_structure,
    label,
)


ROOT = Path(__file__).resolve().parents[2]

DATA = ROOT / "src/data/mri_gz"

COHORT_PATH = (
    ROOT
    / "src/robustness/outputs/cohort_ecsf_context.csv"
)

OUT = ROOT / "src/robustness/outputs"

SEVERE = {
    "sub-017",
    "sub-020",
    "sub-025",
}

ECSF_CLASS_ID = 1


def empirical_percentile(
    value: float,
    values,
) -> float:
    values = np.asarray(
        values,
        dtype=float,
    )

    values = values[
        np.isfinite(values)
    ]

    return float(
        100.0 * np.mean(values <= value)
    )


def component_statistics(
    mask: np.ndarray,
    connectivity: int,
):
    """
    Return component sizes for a binary 3D mask.

    connectivity:
        1 -> 6-connected
        2 -> 18-connected
        3 -> 26-connected
    """
    structure = generate_binary_structure(
        rank=3,
        connectivity=connectivity,
    )

    labelled, n_components = label(
        mask,
        structure=structure,
    )

    if n_components == 0:
        return np.array(
            [],
            dtype=np.int64,
        )

    sizes = np.bincount(
        labelled.ravel()
    )[1:]

    return sizes.astype(
        np.int64
    )


def count_at_least(
    component_sizes: np.ndarray,
    minimum_size: int,
) -> int:
    return int(
        np.sum(
            component_sizes >= minimum_size
        )
    )


def fraction_in_small_components(
    component_sizes: np.ndarray,
    threshold: int,
) -> float:
    """
    Fraction of all foreground eCSF voxels belonging to components
    smaller than threshold voxels.
    """
    total = int(
        component_sizes.sum()
    )

    if total == 0:
        return float("nan")

    small = int(
        component_sizes[
            component_sizes < threshold
        ].sum()
    )

    return small / total


def largest_component_fraction(
    component_sizes: np.ndarray,
) -> float:
    if len(component_sizes) == 0:
        return float("nan")

    total = component_sizes.sum()

    return float(
        component_sizes.max()
        / total
    )


# ---------------------------------------------------------------------
# Read the existing cohort table
# ---------------------------------------------------------------------

with open(COHORT_PATH) as f:
    cohort = list(
        csv.DictReader(f)
    )


rows = []


for index, subject in enumerate(
    cohort,
    start=1,
):

    subject_id = subject[
        "subject_id"
    ]

    rec_type = subject[
        "rec_type"
    ]

    segmentation_path = (
        DATA
        / f"{subject_id}_rec-{rec_type}_dseg.nii.gz"
    )

    nii = nib.load(
        segmentation_path
    )

    segmentation = np.rint(
        np.asarray(nii.dataobj)
    ).astype(np.int16)

    ecsf = (
        segmentation
        == ECSF_CLASS_ID
    )

    sizes_6 = component_statistics(
        ecsf,
        connectivity=1,
    )

    sizes_18 = component_statistics(
        ecsf,
        connectivity=2,
    )

    sizes_26 = component_statistics(
        ecsf,
        connectivity=3,
    )

    row = {
        "subject_id": subject_id,
        "split": subject["split"],
        "rec_type": rec_type,
        "is_severe_failure": (
            subject_id in SEVERE
        ),

        "components_6_all": (
            len(sizes_6)
        ),
        "components_18_all": (
            len(sizes_18)
        ),
        "components_26_all": (
            len(sizes_26)
        ),

        "components_6_ge5vox": (
            count_at_least(
                sizes_6,
                5,
            )
        ),
        "components_6_ge10vox": (
            count_at_least(
                sizes_6,
                10,
            )
        ),
        "components_26_ge10vox": (
            count_at_least(
                sizes_26,
                10,
            )
        ),

        "fraction_voxels_in_lt5vox_components_6": (
            fraction_in_small_components(
                sizes_6,
                5,
            )
        ),

        "fraction_voxels_in_lt10vox_components_6": (
            fraction_in_small_components(
                sizes_6,
                10,
            )
        ),

        "largest_component_fraction_6": (
            largest_component_fraction(
                sizes_6
            )
        ),

        "largest_component_fraction_26": (
            largest_component_fraction(
                sizes_26
            )
        ),
    }

    rows.append(row)

    print(
        f"[{index:02d}/{len(cohort)}] "
        f"{subject_id} complete"
    )


# ---------------------------------------------------------------------
# Save full cohort results
# ---------------------------------------------------------------------

output_path = (
    OUT
    / "ecsf_topology_sensitivity.csv"
)

with open(
    output_path,
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
# Severe-case percentile context
# ---------------------------------------------------------------------

COUNT_METRICS = [
    (
        "components_6_all",
        "Components: 6-connected, all",
    ),
    (
        "components_18_all",
        "Components: 18-connected, all",
    ),
    (
        "components_26_all",
        "Components: 26-connected, all",
    ),
    (
        "components_6_ge5vox",
        "Components: 6-connected, >=5 voxels",
    ),
    (
        "components_6_ge10vox",
        "Components: 6-connected, >=10 voxels",
    ),
    (
        "components_26_ge10vox",
        "Components: 26-connected, >=10 voxels",
    ),
]


print()
print("=" * 96)
print("eCSF TOPOLOGY SENSITIVITY")
print("=" * 96)


for subject_id in sorted(SEVERE):

    row = next(
        r
        for r in rows
        if r["subject_id"] == subject_id
    )

    print()
    print(subject_id)
    print("-" * 96)

    for metric, display_name in COUNT_METRICS:

        value = float(
            row[metric]
        )

        cohort_values = [
            float(r[metric])
            for r in rows
        ]

        percentile = empirical_percentile(
            value,
            cohort_values,
        )

        print(
            f"{display_name:43s} "
            f"{value:7.0f} "
            f"percentile={percentile:5.1f}"
        )

    print()
    print(
        "Fraction of eCSF voxels in "
        f"<5-voxel components:  "
        f"{100 * float(row['fraction_voxels_in_lt5vox_components_6']):.2f}%"
    )

    print(
        "Fraction of eCSF voxels in "
        f"<10-voxel components: "
        f"{100 * float(row['fraction_voxels_in_lt10vox_components_6']):.2f}%"
    )

    print(
        "Largest component fraction "
        f"(6-connected):  "
        f"{100 * float(row['largest_component_fraction_6']):.2f}%"
    )

    print(
        "Largest component fraction "
        f"(26-connected): "
        f"{100 * float(row['largest_component_fraction_26']):.2f}%"
    )


print()
print("=" * 96)
print("INTERPRETATION CHECK")
print("=" * 96)
print(
    "If the severe subjects remain at high cohort percentiles "
    "after using 26-connectivity and after excluding components "
    "smaller than 5-10 voxels, the fragmentation finding is robust "
    "to connectivity choice and tiny isolated islands."
)

print()
print(
    "If their rankings collapse after these changes, the original "
    "extreme component counts are primarily driven by very small "
    "annotation-scale fragments."
)

print()
print(f"Saved: {output_path}")
