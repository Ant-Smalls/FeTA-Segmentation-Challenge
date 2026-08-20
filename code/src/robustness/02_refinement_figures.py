from pathlib import Path
import csv

import numpy as np
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "src/robustness/outputs"
OUT.mkdir(parents=True, exist_ok=True)

ECSF_PATH = OUT / "ecsf_refinement_by_subject.csv"
MODEL_PATH = OUT / "subject_level_model_comparison.csv"


# ============================================================
# Load data
# ============================================================

with open(ECSF_PATH, newline="") as f:
    ecsf = list(csv.DictReader(f))

for r in ecsf:
    for key in [
        "dice_before",
        "dice_after",
        "dice_delta",
        "ed_before",
        "ed_after",
        "ed_improvement",
    ]:
        r[key] = float(r[key])


with open(MODEL_PATH, newline="") as f:
    model_rows = list(csv.DictReader(f))

for r in model_rows:
    for key in [
        "baseline_mean_dice",
        "comparative_mean_dice",
        "delta_comparative_minus_baseline",
    ]:
        r[key] = float(r[key])


# ============================================================
# FIGURE 1
# Subject-level eCSF Euler difference before/after refinement
# ============================================================

rows = sorted(ecsf, key=lambda r: r["ed_before"])

subjects = [r["subject_id"] for r in rows]
before = np.array([r["ed_before"] for r in rows])
after = np.array([r["ed_after"] for r in rows])

x = np.arange(len(rows))

fig, ax = plt.subplots(figsize=(10, 5.5))

# Connecting line for every subject
for i, r in enumerate(rows):
    ax.plot(
        [i - 0.13, i + 0.13],
        [r["ed_before"], r["ed_after"]],
        linewidth=1.2,
        alpha=0.75,
    )

ax.scatter(
    x - 0.13,
    before,
    marker="o",
    s=55,
    label="Before refinement",
    zorder=3,
)

ax.scatter(
    x + 0.13,
    after,
    marker="x",
    s=65,
    label="After refinement",
    zorder=3,
)

ax.set_yscale("log")

labels = [
    f"{r['subject_id']}\n{r['rec_type'].upper()}"
    for r in rows
]

ax.set_xticks(x)
ax.set_xticklabels(labels, rotation=45, ha="right")

ax.set_ylabel("eCSF Euler difference (log scale)")
ax.set_xlabel("Test subject")
ax.set_title(
    "Subject-level eCSF topology before and after uncertainty-guided refinement"
)

ax.legend(frameon=False)
ax.grid(axis="y", alpha=0.2)

fig.tight_layout()

path = OUT / "fig_ecsf_ed_subject_level.png"
fig.savefig(path, dpi=300, bbox_inches="tight")
plt.close(fig)

print(f"Saved {path}")


# ============================================================
# FIGURE 2
# Small but consistent eCSF Dice improvement
# ============================================================

rows = sorted(ecsf, key=lambda r: r["dice_delta"])

delta = np.array([r["dice_delta"] for r in rows])
x = np.arange(len(rows))

fig, ax = plt.subplots(figsize=(9, 5))

for rec, marker in [("mial", "o"), ("irtk", "s")]:
    indices = [
        i for i, r in enumerate(rows)
        if r["rec_type"] == rec
    ]

    ax.scatter(
        indices,
        delta[indices],
        marker=marker,
        s=65,
        label=rec.upper(),
    )

ax.axhline(0, linewidth=1)

ax.set_xticks(x)
ax.set_xticklabels(
    [r["subject_id"] for r in rows],
    rotation=45,
    ha="right",
)

ax.set_ylabel("Change in eCSF Dice after refinement")
ax.set_xlabel("Test subject")
ax.set_title(
    "eCSF overlap change after uncertainty-guided refinement"
)

ax.legend(frameon=False)
ax.grid(axis="y", alpha=0.2)

fig.tight_layout()

path = OUT / "fig_ecsf_dice_delta.png"
fig.savefig(path, dpi=300, bbox_inches="tight")
plt.close(fig)

print(f"Saved {path}")


# ============================================================
# FIGURE 3
# Baseline vs comparative by reconstruction subgroup
# ============================================================

fig, ax = plt.subplots(figsize=(7.5, 5.5))

positions = {
    "irtk": (0, 1),
    "mial": (3, 4),
}

for rec in ["irtk", "mial"]:

    group = [
        r for r in model_rows
        if r["rec_type"] == rec
    ]

    x_base, x_comp = positions[rec]

    for r in group:

        y_base = r["baseline_mean_dice"]
        y_comp = r["comparative_mean_dice"]

        ax.plot(
            [x_base, x_comp],
            [y_base, y_comp],
            alpha=0.6,
            linewidth=1,
        )

        ax.scatter(
            x_base,
            y_base,
            marker="o",
            s=45,
            zorder=3,
        )

        ax.scatter(
            x_comp,
            y_comp,
            marker="x",
            s=55,
            zorder=3,
        )


# Add group means
for rec in ["irtk", "mial"]:

    group = [
        r for r in model_rows
        if r["rec_type"] == rec
    ]

    x_base, x_comp = positions[rec]

    base_mean = np.mean(
        [r["baseline_mean_dice"] for r in group]
    )

    comp_mean = np.mean(
        [r["comparative_mean_dice"] for r in group]
    )

    ax.scatter(
        [x_base, x_comp],
        [base_mean, comp_mean],
        marker="_",
        s=500,
        linewidth=3,
        zorder=5,
    )


ax.set_xticks([0, 1, 3, 4])

ax.set_xticklabels(
    [
        "IRTK\nBaseline",
        "IRTK\nComparative",
        "MIAL\nBaseline",
        "MIAL\nComparative",
    ]
)

ax.set_ylabel("Mean foreground Dice")
ax.set_title(
    "Model performance by reconstruction subgroup"
)

ax.grid(axis="y", alpha=0.2)

fig.tight_layout()

path = OUT / "fig_reconstruction_subgroup_dice.png"
fig.savefig(path, dpi=300, bbox_inches="tight")
plt.close(fig)

print(f"Saved {path}")
