"""
Audit the three severe eCSF topology failure cases:
sub-017, sub-020 and sub-025.

Questions:
1. Are these globally poor segmentation cases?
2. Which tissues account for the performance failure?
3. Do baseline and comparative models fail on the same tissues?
4. How different are these subjects from the remaining MIAL cases?
"""

from pathlib import Path
import csv
import json
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[2]

PER_CASE = ROOT / "src/tuned_evaluation_outputs/report_per_case.csv"
SPLIT = ROOT / "src/data/splits/split_v1.json"
OUT = ROOT / "src/robustness/outputs"

OUT.mkdir(parents=True, exist_ok=True)

SEVERE = {"sub-017", "sub-020", "sub-025"}


# ------------------------------------------------------------
# Reconstruction labels
# ------------------------------------------------------------

with open(SPLIT) as f:
    split = json.load(f)

rec_type = {}

for split_name, entries in split["assignments"].items():
    for entry in entries:
        rec_type[entry["subject_id"]] = entry["rec_type"]


# ------------------------------------------------------------
# Load evaluation rows
# ------------------------------------------------------------

with open(PER_CASE, newline="") as f:
    rows = list(csv.DictReader(f))

print("Columns in report_per_case.csv:")
print(list(rows[0].keys()))
print()


# Keep foreground only
rows = [
    r for r in rows
    if int(r["class_id"]) != 0
]


# ------------------------------------------------------------
# Organise:
# subject -> model -> class -> dice
# ------------------------------------------------------------

data = defaultdict(lambda: defaultdict(dict))

for r in rows:
    subject = r["subject_id"]
    model = r["model"]
    class_name = r["class_name"]

    data[subject][model][class_name] = float(r["dice"])


class_names = sorted(
    {
        r["class_name"]
        for r in rows
    }
)


# ------------------------------------------------------------
# Print severe cases
# ------------------------------------------------------------

print("=" * 78)
print("SEVERE FAILURE CASES")
print("=" * 78)

for subject in sorted(SEVERE):

    print()
    print(subject, f"({rec_type[subject].upper()})")
    print("-" * 78)

    for model in ["baseline", "comparative"]:

        scores = data[subject][model]

        mean_dice = np.mean(
            [scores[c] for c in class_names]
        )

        print(f"{model.upper():12s} mean foreground Dice = {mean_dice:.4f}")

        for c in class_names:
            print(
                f"    {c:20s}: {scores[c]:.4f}"
            )


# ------------------------------------------------------------
# Compare severe MIAL vs remaining MIAL
# ------------------------------------------------------------

mial_subjects = sorted(
    {
        r["subject_id"]
        for r in rows
        if rec_type[r["subject_id"]] == "mial"
    }
)

normal_mial = [
    s for s in mial_subjects
    if s not in SEVERE
]


print()
print("=" * 78)
print("SEVERE MIAL VS OTHER MIAL")
print("=" * 78)

print("Severe MIAL:", sorted(SEVERE))
print("Other MIAL: ", normal_mial)


comparison_rows = []

for model in ["baseline", "comparative"]:

    print()
    print(model.upper())
    print("-" * 78)

    for class_name in class_names:

        severe_scores = np.array(
            [
                data[s][model][class_name]
                for s in SEVERE
            ]
        )

        normal_scores = np.array(
            [
                data[s][model][class_name]
                for s in normal_mial
            ]
        )

        severe_mean = severe_scores.mean()
        normal_mean = normal_scores.mean()

        difference = severe_mean - normal_mean

        comparison_rows.append(
            {
                "model": model,
                "class_name": class_name,
                "severe_mial_mean_dice": severe_mean,
                "other_mial_mean_dice": normal_mean,
                "difference": difference,
            }
        )

        print(
            f"{class_name:20s} "
            f"severe={severe_mean:.4f} "
            f"other={normal_mean:.4f} "
            f"delta={difference:+.4f}"
        )


# ------------------------------------------------------------
# Save comparison CSV
# ------------------------------------------------------------

csv_path = OUT / "failure_case_class_comparison.csv"

with open(csv_path, "w", newline="") as f:

    writer = csv.DictWriter(
        f,
        fieldnames=list(comparison_rows[0].keys()),
    )

    writer.writeheader()
    writer.writerows(comparison_rows)

print()
print(f"Saved {csv_path}")


# ------------------------------------------------------------
# Figure:
# per-class Dice for all MIAL cases
# ------------------------------------------------------------

subjects = sorted(
    mial_subjects,
    key=lambda s: (
        s not in SEVERE,
        s,
    )
)

for model in ["baseline", "comparative"]:

    matrix = np.array(
        [
            [
                data[s][model][c]
                for c in class_names
            ]
            for s in subjects
        ]
    )

    fig, ax = plt.subplots(figsize=(10, 5))

    im = ax.imshow(
        matrix,
        aspect="auto",
        vmin=0,
        vmax=1,
    )

    ax.set_xticks(
        np.arange(len(class_names))
    )

    ax.set_xticklabels(
        class_names,
        rotation=45,
        ha="right",
    )

    labels = []

    for s in subjects:

        marker = " *" if s in SEVERE else ""

        labels.append(
            f"{s}{marker}"
        )

    ax.set_yticks(
        np.arange(len(subjects))
    )

    ax.set_yticklabels(labels)

    ax.set_xlabel("Tissue class")
    ax.set_ylabel("MIAL test subject")

    ax.set_title(
        f"Per-class Dice across MIAL subjects — {model}"
    )

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Dice")

    fig.tight_layout()

    path = OUT / f"fig_mial_failure_heatmap_{model}.png"

    fig.savefig(
        path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)

    print(f"Saved {path}")
