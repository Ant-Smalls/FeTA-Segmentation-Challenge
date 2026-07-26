"""Full-scale preprocessing validation across the entire split (Phase 6)

- Runs preprocess_case (load -> crop -> resample -> normalize) against every
  case in split_v1.json (train+val+test -- QC already ran in qc_and_split.py,
  this validates the transform pipeline itself at real data scale), timing
  each case and logging failures instead of crashing the whole run.
- Also exercises FeTADataset in both 'train' mode (patch sampling +
  augmentation) and 'eval' mode (full-volume) with one __getitem__ call per
  split, to catch contract problems a preprocess_case-only sweep wouldn't
  (shape/dtype, patch padding on small volumes).
- Prints a timing/memory summary at the end.
"""

from __future__ import annotations

import json
import resource
import time
from pathlib import Path

import yaml

from .dataset import FeTADataset
from .preprocessing import preprocess_case
from .qc_and_split import case_paths


def _peak_memory_mb() -> float:
    """Peak resident set size for this process so far, in MB. ru_maxrss is
    KB on Linux (the HPC target); macOS reports bytes instead, which just
    means a harmless overestimate for local runs."""

    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


def validate_case(
    subject_id: str, rec_type: str, data_root: Path, target_spacing: tuple[float, float, float] | None
) -> tuple[float, str | None]:
    """Time one preprocess_case call. Returns (elapsed_seconds, error_or_None)."""

    t2w_path, dseg_path = case_paths(data_root, subject_id, rec_type)
    start = time.perf_counter()
    try:
        preprocess_case(t2w_path, dseg_path, target_spacing)
    except Exception as exc:  # a single bad case must not abort the sweep
        return time.perf_counter() - start, f"{type(exc).__name__}: {exc}"
    return time.perf_counter() - start, None


def validate_dataset_getitem(split_path: Path, config_path: Path) -> list[str]:
    """One __getitem__ call per split/mode combination as a contract sanity
    check -- validate_case's sweep above already covers every case, this
    just confirms FeTADataset's patch sampling, padding, and augmentation
    don't blow up at real data scale."""

    errors: list[str] = []
    for split_name in ("train", "val", "test"):
        for mode in ("train", "eval"):
            try:
                dataset = FeTADataset(split_path, config_path, split_name=split_name, mode=mode, seed=0)
                _ = dataset[0]
            except Exception as exc:
                errors.append(f"{split_name}/{mode}: {type(exc).__name__}: {exc}")
    return errors


def run_validation(data_root: Path, split_path: Path, config_path: Path) -> None:
    """Preprocess every case in split_v1.json, reporting failures and
    timing/memory stats."""

    with open(split_path) as f:
        split_artifact = json.load(f)
    with open(config_path) as f:
        config = yaml.safe_load(f)

    target_spacing = config["preprocessing"]["target_spacing_mm"]
    target_spacing = tuple(target_spacing) if target_spacing is not None else None

    cases = [
        (entry["subject_id"], entry["rec_type"], split_name)
        for split_name, entries in split_artifact["assignments"].items()
        for entry in entries
    ]

    print(f"Validating preprocessing for {len(cases)} case(s) across train/val/test...")

    durations: list[float] = []
    failures: list[str] = []
    for subject_id, rec_type, split_name in cases:
        elapsed, error = validate_case(subject_id, rec_type, data_root, target_spacing)
        durations.append(elapsed)
        if error is not None:
            failures.append(f"{subject_id} ({rec_type}, {split_name}): {error}")

    getitem_errors = validate_dataset_getitem(split_path, config_path)

    print(f"\n{len(cases) - len(failures)}/{len(cases)} case(s) preprocessed successfully.")
    if durations:
        sorted_durations = sorted(durations)
        print(
            f"Per-case time (s): min={sorted_durations[0]:.2f} "
            f"median={sorted_durations[len(sorted_durations) // 2]:.2f} max={sorted_durations[-1]:.2f}"
        )
    print(f"Peak memory so far: {_peak_memory_mb():.0f} MB")

    if failures:
        print(f"\n{len(failures)} preprocess_case failure(s):")
        for failure in failures:
            print(f"  {failure}")

    if getitem_errors:
        print(f"\n{len(getitem_errors)} FeTADataset.__getitem__ failure(s):")
        for error in getitem_errors:
            print(f"  {error}")

    if failures or getitem_errors:
        raise SystemExit(1)

    print("\nPreprocessing validation passed for every case in every split/mode.")


if __name__ == "__main__":
    code_root = Path(__file__).resolve().parents[2]  # preprocessing_data_preparation -> src -> code
    config_path = code_root / "src" / "data" / "config.yaml"
    with open(config_path) as f:
        config = yaml.safe_load(f)

    run_validation(
        data_root=code_root / config["paths"]["data_root"],
        split_path=code_root / config["paths"]["split_file"],
        config_path=config_path,
    )
