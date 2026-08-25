"""QC gate + stratified split generation

- QC checklist run per case, exclusions logged with a reason:
  (a) T2w and dseg load without error and share shape/affine/spacing,
  (b) dseg contains only labels 0-7 with all 7 tissue classes present and
  non-degenerate,
  (c) per-class foreground volume within a generous, per-class multiple of the
  FeTA README's reported median/max, (d) spacing within the dataset's
  reported (~0.43-1.0mm) range, (e) image isn't blank/constant/corrupted

- Volume-plausibility multipliers are per-class, not a single flat number:
  recalibrated against the first full 80-case HPC run (flat multiplier=3.0
  excluded 19/80 = 23.75%, almost all low-side eCSF/dGM/cerebellum/brainstem
  flags spanning a smooth low-to-normal continuum, not a bimodal
  corrupt-vs-clean split, and none also flagged degenerate_class_X -- i.e.
  real but GA-small tissue, not missing tissue). See config.yaml's
  qc.volume_plausibility_multiplier for the tuned per-class values. The one
  case that stayed excluded even at the loosened multipliers (simultaneous
  eCSF/GM/brainstem depletion, not just one class) is worth a manual look
  rather than further loosening thresholds around it.

- Subject IDs are deterministic: sub-001..sub-040 are rec-mial, sub-041..sub-080
  are rec-irtk. This is a site/institution proxy label, the split is stratified across it (56/12/12 train/val/test, proportional mial/irtk
  in each)
- Output is a single frozen artifact: code/src/data/splits/split_v1.json
  (seed, per-split subject ID lists with rec type, QC exclusion log). File is reloaded downstream.

"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path

import nibabel as nib
import numpy as np


# FeTA README label-statistics table (computed over the full 120-case set, NOT
# 80-case subset. The
# README's "mean" column is inconsistent with its own "median"/"max" columns
# (e.g. eCSF mean=0.83cm3 vs median=60.79cm3) so plausibility bounds below use
# median/max, not mean.
FETA_LABEL_VOLUME_STATS_CM3 = {
    1: {"name": "eCSF", "mean": 0.83, "median": 60.79, "max": 197.17},
    2: {"name": "GM", "mean": 3.93, "median": 29.12, "max": 93.88},
    3: {"name": "WM", "mean": 13.89, "median": 61.5, "max": 150.14},
    4: {"name": "Ventricles", "mean": 2.42, "median": 8.04, "max": 127.41},
    5: {"name": "Cerebellum", "mean": 0.74, "median": 3.82, "max": 15.91},
    6: {"name": "dGM", "mean": 0.54, "median": 7.91, "max": 21.14},
    7: {"name": "Brainstem", "mean": 0.43, "median": 3.25, "max": 8.04},
}

SPACING_MIN_MM = 0.42
SPACING_MAX_MM = 1.0

DEFAULT_AFFINE_ATOL = 1.0e-3

# rec-mial: sub-001..sub-040. rec-irtk: sub-041..sub-080
MIAL_SUBJECT_RANGE = (1, 40)
IRTK_SUBJECT_RANGE = (41, 80)


def _assert_rec_type_matches_expected_range(subject_id: str, subject_num: int, rec_type: str) -> None:
    """The mial/irtk boundary is a site/institution proxy label the whole
    stratified-split design depends on."""

    expected_range = MIAL_SUBJECT_RANGE if rec_type == "mial" else IRTK_SUBJECT_RANGE
    if not (expected_range[0] <= subject_num <= expected_range[1]):
        raise ValueError(
            f"{subject_id} is rec-{rec_type} but outside the expected range "
            f"{expected_range} for that rec_type -- the mial/irtk site-proxy "
            f"convention this split relies on may not hold for this data drop."
        )

_T2W_RE = re.compile(r"^sub-(\d{3})_rec-(mial|irtk)_T2w\.nii\.gz$")
_DSEG_RE = re.compile(r"^sub-(\d{3})_rec-(mial|irtk)_dseg\.nii\.gz$")


@dataclass
class QCResult:
    """Outcome of running the QC checklist against a single case."""

    subject_id: str
    rec_type: str
    passed: bool
    failure_reasons: list[str] = field(default_factory=list)


@dataclass
class CaseFiles:
    """Resolved file paths for one case's T2w image and dseg label map."""

    subject_id: str
    rec_type: str
    t2w_path: Path
    dseg_path: Path


def case_paths(data_root: Path, subject_id: str, rec_type: str) -> tuple[Path, Path]:
    """Resolve a case's T2w/dseg paths from data_root + the naming convention."""

    stem = f"{subject_id}_rec-{rec_type}"
    data_root = Path(data_root)
    return data_root / f"{stem}_T2w.nii.gz", data_root / f"{stem}_dseg.nii.gz"


def discover_cases(data_root: Path) -> list[CaseFiles]:
    """Scan ``data_root`` for sub-XXX_rec-{mial|irtk}_{T2w,dseg}.nii.gz pairs.

    Raises rather than silently skipping if a T2w file has no matching dseg
    file, or vice versa.
    """

    data_root = Path(data_root)
    cases: list[CaseFiles] = []
    seen_subjects: set[str] = set()

    for t2w_path in sorted(data_root.glob("sub-*_rec-*_T2w.nii.gz")):
        match = _T2W_RE.match(t2w_path.name)
        if match is None:
            raise ValueError(f"Unrecognized T2w filename pattern: {t2w_path.name}")
        subject_num, rec_type = match.groups()
        subject_id = f"sub-{subject_num}"
        _assert_rec_type_matches_expected_range(subject_id, int(subject_num), rec_type)
        dseg_path = t2w_path.with_name(t2w_path.name.replace("_T2w.nii.gz", "_dseg.nii.gz"))
        if not dseg_path.exists():
            raise FileNotFoundError(
                f"{subject_id} ({rec_type}): T2w found but no matching dseg at {dseg_path}"
            )
        cases.append(CaseFiles(subject_id, rec_type, t2w_path, dseg_path))
        seen_subjects.add(subject_id)

    for dseg_path in sorted(data_root.glob("sub-*_rec-*_dseg.nii.gz")):
        match = _DSEG_RE.match(dseg_path.name)
        if match is None:
            raise ValueError(f"Unrecognized dseg filename pattern: {dseg_path.name}")
        subject_id = f"sub-{match.group(1)}"
        if subject_id not in seen_subjects:
            raise FileNotFoundError(f"{subject_id}: dseg found but no matching T2w at {dseg_path}")

    return cases


def _class_volume_multiplier(qc_config: dict, label_id: int) -> float:
    """Resolve the volume-plausibility multiplier for ``label_id``. Accepts
    either the per-class dict form (config.yaml's default, with a "default"
    fallback key) or a single flat number, for callers that don't need the
    per-class distinction."""

    multiplier_cfg = qc_config.get("volume_plausibility_multiplier", 3.0)
    if isinstance(multiplier_cfg, dict):
        return float(multiplier_cfg.get(label_id, multiplier_cfg["default"]))
    return float(multiplier_cfg)


def run_qc_checks(case: CaseFiles, qc_config: dict) -> QCResult:
    """Run the 5-check QC checklist against one case. Never raises: load
    failures and every other check become a logged failure instead."""

    reasons: list[str] = []

    try:
        t2w_img = nib.load(case.t2w_path)
        dseg_img = nib.load(case.dseg_path)
    except Exception as exc:  
        return QCResult(case.subject_id, case.rec_type, passed=False, failure_reasons=[f"failed_to_load: {exc}"])

    # T2w/dseg agreement
    if t2w_img.shape != dseg_img.shape:
        reasons.append(f"shape_mismatch: T2w={t2w_img.shape} dseg={dseg_img.shape}")
    affine_atol = qc_config.get("affine_atol", DEFAULT_AFFINE_ATOL)
    if not np.allclose(t2w_img.affine, dseg_img.affine, atol=affine_atol):
        reasons.append("affine_mismatch")

    # spacing range
    spacing = tuple(float(s) for s in t2w_img.header.get_zooms()[:3])
    spacing_min = qc_config.get("spacing_min_mm", SPACING_MIN_MM)
    spacing_max = qc_config.get("spacing_max_mm", SPACING_MAX_MM)
    if not all(spacing_min - 1e-6 <= s <= spacing_max + 1e-6 for s in spacing):
        reasons.append(f"spacing_out_of_range: {spacing}")

    dseg = np.asarray(dseg_img.get_fdata()).astype(np.int32)
    unique_labels = set(np.unique(dseg).tolist())

    # label identity + presence + non-degeneracy
    unexpected = sorted(unique_labels - set(range(8)))
    if unexpected:
        reasons.append(f"unexpected_labels: {unexpected}")

    min_voxels = qc_config.get("min_voxels_per_class", 10)
    present_classes = set()
    for label_id in range(1, 8):
        count = int(np.count_nonzero(dseg == label_id))
        if count >= min_voxels:
            present_classes.add(label_id)
        elif count > 0:
            reasons.append(f"degenerate_class_{label_id}: only {count} voxels")

    min_labels_present = qc_config.get("min_labels_present", 7)
    if len(present_classes) < min_labels_present:
        missing = sorted(set(range(1, 8)) - present_classes)
        reasons.append(f"insufficient_classes_present ({len(present_classes)}/7): missing {missing}")

    # per-class volume plausibility (median/max reference)
    voxel_vol_mm3 = float(np.prod(spacing))
    for label_id, stats in FETA_LABEL_VOLUME_STATS_CM3.items():
        if label_id not in present_classes:
            continue
        multiplier = _class_volume_multiplier(qc_config, label_id)
        volume_cm3 = int(np.count_nonzero(dseg == label_id)) * voxel_vol_mm3 / 1000.0
        lower = stats["median"] / multiplier
        upper = stats["max"] * multiplier
        if not (lower <= volume_cm3 <= upper):
            reasons.append(
                f"implausible_volume_class_{label_id}: {volume_cm3:.2f}cm3 not in [{lower:.2f}, {upper:.2f}]"
            )

    # blank/constant image
    t2w = np.asarray(t2w_img.get_fdata())
    if float(np.std(t2w)) == 0.0:
        reasons.append("blank_or_constant_image")

    return QCResult(case.subject_id, case.rec_type, passed=not reasons, failure_reasons=reasons)


def generate_stratified_split(
    qc_passed_cases: list[CaseFiles],
    seed: int,
    train_frac: float = 0.70,
    val_frac: float = 0.15,
    test_frac: float = 0.15,
) -> dict:
    """Produce a seeded, mial/irtk-stratified split.

    Groups are split proportionally within each rec_type so both sites are
    represented in every split at any N (56/12/12 exactly at the real N=80)"""

    if abs((train_frac + val_frac + test_frac) - 1.0) > 1e-6:
        raise ValueError("train/val/test fractions must sum to 1.0")

    groups: dict[str, list[CaseFiles]] = {}
    for case in qc_passed_cases:
        groups.setdefault(case.rec_type, []).append(case)

    assignments: dict[str, list[dict]] = {"train": [], "val": [], "test": []}
    rng = np.random.default_rng(seed)

    for rec_type in sorted(groups):
        ordered = sorted(groups[rec_type], key=lambda c: c.subject_id)
        shuffled = [ordered[i] for i in rng.permutation(len(ordered))]

        n = len(shuffled)
        n_train = round(n * train_frac)
        n_val = round(n * val_frac)
        n_test = n - n_train - n_val
        if n_test < 0:
            n_train, n_val, n_test = n, 0, 0

        split_labels = ["train"] * n_train + ["val"] * n_val + ["test"] * n_test
        for case, split_name in zip(shuffled, split_labels):
            assignments[split_name].append({"subject_id": case.subject_id, "rec_type": case.rec_type})

    total = len(qc_passed_cases)
    for split_name in ("val", "test"):
        if not assignments[split_name] and total >= 10:
            print(f"WARNING: '{split_name}' split is empty across {total} QC-passed cases")

    return {
        "seed": seed,
        "fractions": {"train": train_frac, "val": val_frac, "test": test_frac},
        "assignments": assignments,
    }


def compute_median_spacing(cases: list[CaseFiles]) -> tuple[float, float, float]:
    """Per-axis median voxel spacing across ``cases``. Must be called with
    TRAIN cases only -- this becomes config.yaml's preprocessing.target_spacing_mm.
    Reads only the NIfTI header."""

    spacings = [nib.load(case.t2w_path).header.get_zooms()[:3] for case in cases]
    return tuple(float(np.median(axis_spacings)) for axis_spacings in zip(*spacings))


def write_split_artifact(split: dict, qc_log: list[QCResult], output_path: Path) -> None:
    """Write the frozen split_v1.json artifact (seed, split lists, QC log)."""

    artifact = {**split, "qc_log": [asdict(result) for result in qc_log]}
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(artifact, indent=2))


def run_pipeline(data_root: Path, config: dict, output_path: Path) -> dict:
    """Discover -> QC -> stratified split -> write frozen artifact. Returns
    the split dict (callers/tests can inspect it without re-reading the file)."""

    cases = discover_cases(data_root)
    qc_log = [run_qc_checks(case, config["qc"]) for case in cases]
    passed = [case for case, result in zip(cases, qc_log) if result.passed]

    excluded = [r for r in qc_log if not r.passed]
    if excluded:
        print(f"QC excluded {len(excluded)}/{len(cases)} case(s):")
        for r in excluded:
            print(f"  {r.subject_id} ({r.rec_type}): {'; '.join(r.failure_reasons)}")

    split_cfg = config["split"]
    split = generate_stratified_split(
        passed,
        seed=split_cfg["seed"],
        train_frac=split_cfg["train_frac"],
        val_frac=split_cfg["val_frac"],
        test_frac=split_cfg["test_frac"],
    )

    train_ids = {(c["subject_id"], c["rec_type"]) for c in split["assignments"]["train"]}
    train_cases = [c for c in passed if (c.subject_id, c.rec_type) in train_ids]
    median_spacing = compute_median_spacing(train_cases) if train_cases else None
    split["train_median_spacing_mm"] = median_spacing

    write_split_artifact(split, qc_log, output_path)

    if median_spacing is not None:
        print(
            f"\nTrain-split median spacing: {median_spacing} -- copy into "
            f"config.yaml's preprocessing.target_spacing_mm manually"
        )

    return split


if __name__ == "__main__":
    import yaml

    code_root = Path(__file__).resolve().parents[2]  # preprocessing_data_preparation -> src -> code
    with open(code_root / "src" / "data" / "config.yaml") as f:
        config = yaml.safe_load(f)

    run_pipeline(
        data_root=code_root / config["paths"]["data_root"],
        config=config,
        output_path=code_root / config["paths"]["split_file"],
    )
