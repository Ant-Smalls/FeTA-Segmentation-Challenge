"""
Robustness audit of the final tuned FeTA evaluation outputs.

This script does NOT retrain or re-evaluate either model.
It analyses the frozen test-set results already committed under
src/tuned_evaluation_outputs/.

Questions:
1. Are baseline and comparative Dice meaningfully different?
2. Does performance differ between MIAL and IRTK reconstruction subgroups?
3. Is uncertainty-guided refinement consistently beneficial across subjects?
4. Is the reported mean ED reduction representative of the typical subject?

All analyses are paired at subject level where appropriate.
"""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean, median, stdev

import numpy as np
from scipy.stats import mannwhitneyu, wilcoxon


ROOT = Path(__file__).resolve().parents[2]

SPLIT_PATH = ROOT / "src/data/splits/split_v1.json"
PER_CASE_PATH = ROOT / "src/tuned_evaluation_outputs/report_per_case.csv"
REFINEMENT_PATH = ROOT / "src/tuned_evaluation_outputs/report_refinement.csv"

OUTPUT_DIR = ROOT / "src/robustness/outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def bootstrap_mean_ci(values, n_boot=20000, seed=42):
    """Non-parametric bootstrap 95% CI for the mean."""
    values = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)

    samples = rng.choice(
        values,
        size=(n_boot, len(values)),
        replace=True,
    )

    boot_means = samples.mean(axis=1)
    lo, hi = np.percentile(boot_means, [2.5, 97.5])

    return float(lo), float(hi)


def safe_wilcoxon(deltas):
    """
    Two-sided paired Wilcoxon test on paired differences.

    Returns None when too few non-zero differences are available for a
    meaningful calculation.
    """
    deltas = np.asarray(deltas, dtype=float)
    non_zero = deltas[np.abs(deltas) > 1e-12]

    if len(non_zero) < 5:
        return None

    return float(
        wilcoxon(
            deltas,
            alternative="two-sided",
            zero_method="wilcox",
        ).pvalue
    )


# ---------------------------------------------------------------------
# Load frozen split and reconstruction labels
# ---------------------------------------------------------------------

with open(SPLIT_PATH) as f:
    split = json.load(f)

rec_type = {}

for split_name, subjects in split["assignments"].items():
    for item in subjects:
        rec_type[item["subject_id"]] = item["rec_type"]


# ---------------------------------------------------------------------
# Load baseline/comparative per-case metrics
# ---------------------------------------------------------------------

with open(PER_CASE_PATH, newline="") as f:
    per_case_rows = list(csv.DictReader(f))


subject_model_dice = defaultdict(lambda: defaultdict(list))

for row in per_case_rows:

    class_id = int(row["class_id"])

    # Exclude background from foreground Dice.
    if class_id == 0:
        continue

    subject = row["subject_id"]
    model = row["model"]

    subject_model_dice[subject][model].append(float(row["dice"]))


subject_summary = []

for subject in sorted(subject_model_dice):

    baseline = mean(subject_model_dice[subject]["baseline"])
    comparative = mean(subject_model_dice[subject]["comparative"])

    subject_summary.append(
        {
            "subject_id": subject,
            "rec_type": rec_type[subject],
            "baseline_mean_dice": baseline,
            "comparative_mean_dice": comparative,
            "delta_comparative_minus_baseline": comparative - baseline,
        }
    )


# Write reusable subject-level table.
summary_csv = OUTPUT_DIR / "subject_level_model_comparison.csv"

with open(summary_csv, "w", newline="") as f:

    fieldnames = list(subject_summary[0].keys())

    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(subject_summary)


# ---------------------------------------------------------------------
# Overall paired baseline/comparative comparison
# ---------------------------------------------------------------------

baseline_values = np.array(
    [r["baseline_mean_dice"] for r in subject_summary]
)

comparative_values = np.array(
    [r["comparative_mean_dice"] for r in subject_summary]
)

model_delta = comparative_values - baseline_values

model_delta_ci = bootstrap_mean_ci(model_delta)

model_wilcoxon = safe_wilcoxon(model_delta)


# ---------------------------------------------------------------------
# Reconstruction subgroup analysis
# ---------------------------------------------------------------------

subgroup_results = {}

for reconstruction in ["mial", "irtk"]:

    group = [
        r
        for r in subject_summary
        if r["rec_type"] == reconstruction
    ]

    baseline_group = np.array(
        [r["baseline_mean_dice"] for r in group]
    )

    comparative_group = np.array(
        [r["comparative_mean_dice"] for r in group]
    )

    subgroup_results[reconstruction] = {
        "n": len(group),
        "baseline_mean": float(baseline_group.mean()),
        "baseline_median": float(np.median(baseline_group)),
        "baseline_sd": float(baseline_group.std(ddof=1)),
        "comparative_mean": float(comparative_group.mean()),
        "comparative_median": float(np.median(comparative_group)),
        "comparative_sd": float(comparative_group.std(ddof=1)),
    }


for model_name in ["baseline", "comparative"]:

    key = f"{model_name}_mean_dice"

    mial = np.array(
        [
            r[key]
            for r in subject_summary
            if r["rec_type"] == "mial"
        ]
    )

    irtk = np.array(
        [
            r[key]
            for r in subject_summary
            if r["rec_type"] == "irtk"
        ]
    )

    test = mannwhitneyu(
        mial,
        irtk,
        alternative="two-sided",
    )

    subgroup_results[f"{model_name}_mial_vs_irtk_p"] = float(test.pvalue)


# ---------------------------------------------------------------------
# Refinement analysis
# ---------------------------------------------------------------------

with open(REFINEMENT_PATH, newline="") as f:
    refinement_rows = list(csv.DictReader(f))


refinement_by_class = defaultdict(list)

for row in refinement_rows:

    refinement_by_class[row["class_name"]].append(
        {
            "subject_id": row["subject_id"],
            "rec_type": rec_type[row["subject_id"]],
            "dice_before": float(row["dice_before"]),
            "dice_after": float(row["dice_after"]),
            "ed_before": float(row["ed_before"]),
            "ed_after": float(row["ed_after"]),
        }
    )


refinement_summary = {}

for class_name, rows in refinement_by_class.items():

    dice_before = np.array([r["dice_before"] for r in rows])
    dice_after = np.array([r["dice_after"] for r in rows])

    ed_before = np.array([r["ed_before"] for r in rows])
    ed_after = np.array([r["ed_after"] for r in rows])

    dice_delta = dice_after - dice_before

    # Positive = topology improvement.
    ed_improvement = ed_before - ed_after

    refinement_summary[class_name] = {
        "n": len(rows),

        "dice_before_mean": float(dice_before.mean()),
        "dice_after_mean": float(dice_after.mean()),
        "dice_delta_mean": float(dice_delta.mean()),
        "dice_delta_median": float(np.median(dice_delta)),
        "dice_improved_n": int(np.sum(dice_delta > 0)),
        "dice_worsened_n": int(np.sum(dice_delta < 0)),
        "dice_delta_ci95": bootstrap_mean_ci(dice_delta),
        "dice_wilcoxon_p": safe_wilcoxon(dice_delta),

        "ed_before_mean": float(ed_before.mean()),
        "ed_after_mean": float(ed_after.mean()),
        "ed_before_median": float(np.median(ed_before)),
        "ed_after_median": float(np.median(ed_after)),
        "ed_mean_improvement": float(ed_improvement.mean()),
        "ed_median_improvement": float(np.median(ed_improvement)),
        "ed_improved_n": int(np.sum(ed_improvement > 0)),
        "ed_worsened_n": int(np.sum(ed_improvement < 0)),
        "ed_unchanged_n": int(np.sum(ed_improvement == 0)),
        "ed_wilcoxon_p": safe_wilcoxon(ed_improvement),
    }

    if ed_before.mean() != 0:
        refinement_summary[class_name]["ed_relative_mean_reduction"] = float(
            (ed_before.mean() - ed_after.mean())
            / ed_before.mean()
        )


# ---------------------------------------------------------------------
# Subject-level eCSF refinement table
# ---------------------------------------------------------------------

ecsf_rows = refinement_by_class["eCSF"]

ecsf_detail_path = OUTPUT_DIR / "ecsf_refinement_by_subject.csv"

with open(ecsf_detail_path, "w", newline="") as f:

    fieldnames = [
        "subject_id",
        "rec_type",
        "dice_before",
        "dice_after",
        "dice_delta",
        "ed_before",
        "ed_after",
        "ed_improvement",
    ]

    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()

    for r in sorted(ecsf_rows, key=lambda x: x["subject_id"]):

        writer.writerow(
            {
                **r,
                "dice_delta": r["dice_after"] - r["dice_before"],
                "ed_improvement": r["ed_before"] - r["ed_after"],
            }
        )


# ---------------------------------------------------------------------
# Print readable report
# ---------------------------------------------------------------------

print()
print("=" * 72)
print("FeTA FINAL RESULTS — ROBUSTNESS AUDIT")
print("=" * 72)

print()
print("1. BASELINE VS COMPARATIVE")
print("-" * 72)

print(f"Baseline mean foreground Dice:     {baseline_values.mean():.4f}")
print(f"Comparative mean foreground Dice:  {comparative_values.mean():.4f}")
print(
    f"Mean paired difference:            "
    f"{model_delta.mean():+.4f}"
)
print(
    f"95% bootstrap CI for difference:   "
    f"[{model_delta_ci[0]:+.4f}, {model_delta_ci[1]:+.4f}]"
)
print(
    f"Median paired difference:          "
    f"{np.median(model_delta):+.4f}"
)

if model_wilcoxon is not None:
    print(f"Paired Wilcoxon p:                 {model_wilcoxon:.4f}")


print()
print("2. RECONSTRUCTION SUBGROUP")
print("-" * 72)

for reconstruction in ["irtk", "mial"]:

    r = subgroup_results[reconstruction]

    print(
        f"{reconstruction.upper():4s} (n={r['n']}): "
        f"baseline={r['baseline_mean']:.4f} "
        f"(SD={r['baseline_sd']:.4f}), "
        f"comparative={r['comparative_mean']:.4f} "
        f"(SD={r['comparative_sd']:.4f})"
    )

print()
print(
    "Exploratory MIAL-vs-IRTK Mann-Whitney p "
    f"(baseline):    "
    f"{subgroup_results['baseline_mial_vs_irtk_p']:.4f}"
)

print(
    "Exploratory MIAL-vs-IRTK Mann-Whitney p "
    f"(comparative): "
    f"{subgroup_results['comparative_mial_vs_irtk_p']:.4f}"
)

print()
print(
    "NOTE: Reconstruction subgroup tests are exploratory. "
    "rec_type is not a randomized exposure and may be confounded "
    "by case characteristics."
)


print()
print("3. REFINEMENT")
print("-" * 72)

for class_name in ["eCSF", "GM", "dGM"]:

    r = refinement_summary[class_name]

    print()
    print(class_name)

    print(
        f"  Dice: {r['dice_before_mean']:.4f} -> "
        f"{r['dice_after_mean']:.4f}"
    )

    print(
        f"  Mean Dice delta: {r['dice_delta_mean']:+.6f} "
        f"(improved {r['dice_improved_n']}/{r['n']})"
    )

    print(
        f"  Dice delta 95% bootstrap CI: "
        f"[{r['dice_delta_ci95'][0]:+.6f}, "
        f"{r['dice_delta_ci95'][1]:+.6f}]"
    )

    if r["dice_wilcoxon_p"] is not None:
        print(
            f"  Dice paired Wilcoxon p: "
            f"{r['dice_wilcoxon_p']:.6f}"
        )

    print(
        f"  ED mean:   {r['ed_before_mean']:.2f} -> "
        f"{r['ed_after_mean']:.2f}"
    )

    print(
        f"  ED median: {r['ed_before_median']:.2f} -> "
        f"{r['ed_after_median']:.2f}"
    )

    print(
        f"  ED improved/worsened/unchanged: "
        f"{r['ed_improved_n']}/"
        f"{r['ed_worsened_n']}/"
        f"{r['ed_unchanged_n']}"
    )

    if "ed_relative_mean_reduction" in r:
        print(
            f"  Relative change in mean ED: "
            f"{100*r['ed_relative_mean_reduction']:+.1f}% reduction"
        )

    if r["ed_wilcoxon_p"] is not None:
        print(
            f"  ED paired Wilcoxon p: "
            f"{r['ed_wilcoxon_p']:.6f}"
        )


print()
print("4. eCSF SUBJECT-LEVEL ED CHANGES")
print("-" * 72)

ecsf_sorted = sorted(
    ecsf_rows,
    key=lambda r: r["ed_before"] - r["ed_after"],
)

for r in ecsf_sorted:

    improvement = r["ed_before"] - r["ed_after"]

    print(
        f"{r['subject_id']} "
        f"({r['rec_type'].upper()}): "
        f"{r['ed_before']:.0f} -> {r['ed_after']:.0f} "
        f"(improvement={improvement:+.0f})"
    )


print()
print("Saved:")
print(f"  {summary_csv}")
print(f"  {ecsf_detail_path}")
print()
