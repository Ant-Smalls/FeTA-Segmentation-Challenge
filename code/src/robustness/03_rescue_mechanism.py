"""
Test whether uncertainty-guided refinement behaves primarily as a
rescue mechanism for severe eCSF topology failures.
"""

from pathlib import Path
import csv

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import spearmanr, wilcoxon


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "src/robustness/outputs"
DATA = OUT / "ecsf_refinement_by_subject.csv"


with open(DATA, newline="") as f:
    rows = list(csv.DictReader(f))

for r in rows:
    for key in [
        "dice_before",
        "dice_after",
        "dice_delta",
        "ed_before",
        "ed_after",
        "ed_improvement",
    ]:
        r[key] = float(r[key])


# ============================================================
# 1. Does worse initial topology predict greater improvement?
# ============================================================

ed_before = np.array([r["ed_before"] for r in rows])
ed_improvement = np.array([r["ed_improvement"] for r in rows])

rho, p = spearmanr(ed_before, ed_improvement)

print("=" * 72)
print("RESCUE-MECHANISM ANALYSIS")
print("=" * 72)

print()
print("1. INITIAL SEVERITY VS REFINEMENT BENEFIT")
print("-" * 72)
print(f"Spearman rho: {rho:.4f}")
print(f"p-value:      {p:.4f}")


# ============================================================
# 2. Reconstruction-specific response
# ============================================================

print()
print("2. REFINEMENT BY RECONSTRUCTION GROUP")
print("-" * 72)

for rec in ["mial", "irtk"]:

    group = [r for r in rows if r["rec_type"] == rec]

    before = np.array([r["ed_before"] for r in group])
    after = np.array([r["ed_after"] for r in group])
    improvement = before - after

    relative_change = (
        (before.mean() - after.mean())
        / before.mean()
    )

    print()
    print(rec.upper())
    print(f"  n:                    {len(group)}")
    print(f"  Mean ED before:       {before.mean():.2f}")
    print(f"  Mean ED after:        {after.mean():.2f}")
    print(
        f"  Relative mean change: "
        f"{100 * relative_change:+.1f}% improvement"
    )
    print(
        f"  Improved/worsened:    "
        f"{np.sum(improvement > 0)}/"
        f"{np.sum(improvement < 0)}"
    )

    dice_delta = np.array([r["dice_delta"] for r in group])

    print(
        f"  Mean eCSF Dice delta: {dice_delta.mean():+.6f}"
    )
    print(
        f"  Dice improved:        "
        f"{np.sum(dice_delta > 0)}/{len(group)}"
    )


# ============================================================
# 3. Outlier-sensitivity analysis
# ============================================================

print()
print("3. OUTLIER-SENSITIVITY ANALYSIS")
print("-" * 72)

ordered = sorted(
    rows,
    key=lambda r: r["ed_before"],
    reverse=True,
)

outlier_table = []

for remove_n in range(0, 4):

    retained = ordered[remove_n:]

    before = np.array([r["ed_before"] for r in retained])
    after = np.array([r["ed_after"] for r in retained])
    improvement = before - after

    relative_reduction = (
        (before.mean() - after.mean())
        / before.mean()
    )

    try:
        p_w = wilcoxon(improvement).pvalue
    except ValueError:
        p_w = np.nan

    removed = [
        r["subject_id"]
        for r in ordered[:remove_n]
    ]

    result = {
        "removed_n": remove_n,
        "removed_subjects": ",".join(removed) if removed else "none",
        "n_remaining": len(retained),
        "mean_ed_before": before.mean(),
        "mean_ed_after": after.mean(),
        "median_ed_before": np.median(before),
        "median_ed_after": np.median(after),
        "relative_mean_reduction": relative_reduction,
        "improved_n": int(np.sum(improvement > 0)),
        "worsened_n": int(np.sum(improvement < 0)),
        "wilcoxon_p": p_w,
    }

    outlier_table.append(result)

    print()
    print(f"Remove worst {remove_n}: {result['removed_subjects']}")
    print(
        f"  n = {len(retained)}, "
        f"mean ED {before.mean():.2f} -> {after.mean():.2f}"
    )
    print(
        f"  Relative mean reduction: "
        f"{100 * relative_reduction:+.1f}%"
    )
    print(
        f"  Improved/worsened: "
        f"{result['improved_n']}/{result['worsened_n']}"
    )


# Save table
table_path = OUT / "ecsf_outlier_sensitivity.csv"

with open(table_path, "w", newline="") as f:
    fieldnames = list(outlier_table[0].keys())
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(outlier_table)

print()
print(f"Saved {table_path}")


# ============================================================
# 4. Figure: severity vs refinement benefit
# ============================================================

fig, ax = plt.subplots(figsize=(7.5, 5.5))

for rec, marker in [("mial", "o"), ("irtk", "s")]:

    group = [
        r for r in rows
        if r["rec_type"] == rec
    ]

    x = [r["ed_before"] for r in group]
    y = [r["ed_improvement"] for r in group]

    ax.scatter(
        x,
        y,
        marker=marker,
        s=70,
        label=rec.upper(),
    )


# Mark zero benefit
ax.axhline(0, linewidth=1)

# Annotate severe cases
for r in rows:
    if r["subject_id"] in {"sub-017", "sub-020", "sub-025"}:
        ax.annotate(
            r["subject_id"],
            (r["ed_before"], r["ed_improvement"]),
            xytext=(6, 6),
            textcoords="offset points",
            fontsize=9,
        )

ax.set_xlabel("eCSF Euler difference before refinement")
ax.set_ylabel("Improvement in eCSF Euler difference")
ax.set_title(
    "Does refinement preferentially rescue severe topology failures?"
)

ax.legend(frameon=False)
ax.grid(alpha=0.2)

fig.tight_layout()

figure_path = OUT / "fig_ed_severity_vs_refinement_benefit.png"
fig.savefig(figure_path, dpi=300, bbox_inches="tight")
plt.close(fig)

print(f"Saved {figure_path}")
