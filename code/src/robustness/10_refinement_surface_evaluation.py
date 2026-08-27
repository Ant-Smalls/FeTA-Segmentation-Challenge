from pathlib import Path
import argparse
import csv
import time

import numpy as np
import torch
import torch.nn.functional as F

from src.preprocessing_data_preparation.dataset import (
    FeTADataset,
    reconstruct_from_patches,
)
from src.training.train import build_model, sliding_window_coords
from src.models.uncertainty_unet import refine_prediction
from src.robustness.surface_metrics import hd95, assd


PATCH_SIZE = (128, 128, 128)
N_MC_SAMPLES = 10
THRESHOLD = 0.5

TARGET_CLASSES = {
    1: "eCSF",
    2: "GM",
    6: "dGM",
}

OUTPUT_DIR = Path(
    "src/robustness/outputs/refinement_surface"
)

RESULTS_CSV = (
    OUTPUT_DIR / "refinement_surface_per_case.csv"
)

SUMMARY_CSV = (
    OUTPUT_DIR / "refinement_surface_summary.csv"
)



@torch.no_grad()
def mc_dropout_sliding_window(
    model,
    image,
    patch_size,
    device,
    n_mc_samples=10,
):
    """
    MC Dropout inference using the tuned test-time tiling convention:
    full-patch stride (overlap=0.0), Gaussian probability stitching.
    """

    model.eval()

    for module in model.modules():
        if isinstance(module, torch.nn.Dropout3d):
            module.train()

    _, depth, height, width = image.shape

    coords = sliding_window_coords(
        (depth, height, width),
        patch_size,
        overlap=0.0,
    )

    prob_patches = []
    patch_coords = []

    print(f"MC tiles: {len(coords)}")

    for tile_index, (dz, dy, dx) in enumerate(
        coords,
        start=1,
    ):
        raw = image[:, dz, dy, dx]

        pad = []
        for dim_size, patch_dim in zip(
            raw.shape[1:],
            patch_size,
        ):
            pad = [0, patch_dim - dim_size] + pad

        raw = F.pad(raw, pad)
        patch = raw.unsqueeze(0).to(device)

        sample_probs = []

        for _ in range(n_mc_samples):
            logits, _ = model(patch)

            sample_probs.append(
                F.softmax(logits, dim=1)
            )

        mean_prob_patch = (
            torch.stack(sample_probs, dim=0)
            .mean(dim=0)
            .squeeze(0)
        )

        original_depth = (
            min(dz.stop, depth) - dz.start
        )
        original_height = (
            min(dy.stop, height) - dy.start
        )
        original_width = (
            min(dx.stop, width) - dx.start
        )

        mean_prob_patch = mean_prob_patch[
            :,
            :original_depth,
            :original_height,
            :original_width,
        ]

        prob_patches.append(
            mean_prob_patch.cpu()
        )

        patch_coords.append(
            (
                slice(
                    dz.start,
                    dz.start + original_depth,
                ),
                slice(
                    dy.start,
                    dy.start + original_height,
                ),
                slice(
                    dx.start,
                    dx.start + original_width,
                ),
            )
        )

        print(
            f"  tile {tile_index}/{len(coords)}",
            end="\r",
            flush=True,
        )

    print(" " * 40, end="\r")

    mean_probability = reconstruct_from_patches(
        prob_patches,
        patch_coords,
        full_volume_shape=(
            depth,
            height,
            width,
        ),
        aggregation="gaussian",
    )

    prediction = torch.argmax(
        mean_probability,
        dim=0,
    )

    entropy = -torch.sum(
        mean_probability
        * torch.log(mean_probability + 1e-8),
        dim=0,
    )

    uncertainty = entropy / torch.log(
        torch.tensor(
            float(mean_probability.shape[0])
        )
    )

    return (
        mean_probability,
        prediction,
        uncertainty,
    )


def dice_score(prediction, target, class_id):
    pred = prediction == class_id
    gt = target == class_id

    denominator = pred.sum() + gt.sum()

    if denominator == 0:
        return 1.0

    intersection = np.logical_and(
        pred,
        gt,
    ).sum()

    return float(
        2.0 * intersection / denominator
    )


def evaluate(
    subject_id,
    prediction_before,
    prediction_after,
    target,
    spacing,
):
    rows = []

    for class_id, class_name in TARGET_CLASSES.items():
        gt_mask = target == class_id

        before_mask = (
            prediction_before == class_id
        )

        after_mask = (
            prediction_after == class_id
        )

        before_dice = dice_score(
            prediction_before,
            target,
            class_id,
        )

        after_dice = dice_score(
            prediction_after,
            target,
            class_id,
        )

        before_hd95 = hd95(
            before_mask,
            gt_mask,
            spacing=spacing,
        )

        after_hd95 = hd95(
            after_mask,
            gt_mask,
            spacing=spacing,
        )

        before_assd = assd(
            before_mask,
            gt_mask,
            spacing=spacing,
        )

        after_assd = assd(
            after_mask,
            gt_mask,
            spacing=spacing,
        )

        rows.append(
            {
                "subject_id": subject_id,
                "class_id": class_id,
                "class_name": class_name,
                "dice_before": before_dice,
                "dice_after": after_dice,
                "dice_delta": (
                    after_dice - before_dice
                ),
                "hd95_before_mm": before_hd95,
                "hd95_after_mm": after_hd95,
                "hd95_delta_mm": (
                    after_hd95 - before_hd95
                ),
                "assd_before_mm": before_assd,
                "assd_after_mm": after_assd,
                "assd_delta_mm": (
                    after_assd - before_assd
                ),
            }
        )

    return rows


def write_results(rows):
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    fields = [
        "subject_id",
        "class_id",
        "class_name",
        "dice_before",
        "dice_after",
        "dice_delta",
        "hd95_before_mm",
        "hd95_after_mm",
        "hd95_delta_mm",
        "assd_before_mm",
        "assd_after_mm",
        "assd_delta_mm",
    ]

    with RESULTS_CSV.open(
        "w",
        newline="",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=fields,
        )

        writer.writeheader()
        writer.writerows(rows)


def write_summary(rows):
    summary = []

    for class_id, class_name in TARGET_CLASSES.items():
        subset = [
            row
            for row in rows
            if row["class_id"] == class_id
        ]

        if not subset:
            continue

        dice_before = np.array(
            [r["dice_before"] for r in subset]
        )

        dice_after = np.array(
            [r["dice_after"] for r in subset]
        )

        hd95_before = np.array(
            [r["hd95_before_mm"] for r in subset]
        )

        hd95_after = np.array(
            [r["hd95_after_mm"] for r in subset]
        )

        assd_before = np.array(
            [r["assd_before_mm"] for r in subset]
        )

        assd_after = np.array(
            [r["assd_after_mm"] for r in subset]
        )

        summary.append(
            {
                "class_id": class_id,
                "class_name": class_name,
                "n_cases": len(subset),

                "mean_dice_before":
                    np.nanmean(dice_before),
                "mean_dice_after":
                    np.nanmean(dice_after),
                "mean_dice_delta":
                    np.nanmean(
                        dice_after - dice_before
                    ),

                "mean_hd95_before_mm":
                    np.nanmean(hd95_before),
                "mean_hd95_after_mm":
                    np.nanmean(hd95_after),
                "mean_hd95_delta_mm":
                    np.nanmean(
                        hd95_after - hd95_before
                    ),

                "median_hd95_before_mm":
                    np.nanmedian(hd95_before),
                "median_hd95_after_mm":
                    np.nanmedian(hd95_after),

                "mean_assd_before_mm":
                    np.nanmean(assd_before),
                "mean_assd_after_mm":
                    np.nanmean(assd_after),
                "mean_assd_delta_mm":
                    np.nanmean(
                        assd_after - assd_before
                    ),

                "median_assd_before_mm":
                    np.nanmedian(assd_before),
                "median_assd_after_mm":
                    np.nanmedian(assd_after),
            }
        )

    if not summary:
        return

    fields = list(summary[0].keys())

    with SUMMARY_CSV.open(
        "w",
        newline="",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=fields,
        )

        writer.writeheader()
        writer.writerows(summary)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--subject",
        default=None,
        help=(
            "Run one subject only, e.g. sub-079. "
            "Omit to run all test subjects."
        ),
    )

    args = parser.parse_args()

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # Match the team's CPU evaluation environment.
    device = torch.device("cpu")

    torch.manual_seed(42)
    np.random.seed(42)

    print("=" * 72)
    print("REFINEMENT SURFACE EVALUATION")
    print("=" * 72)
    print(f"Device: {device}")
    print(f"Patch size: {PATCH_SIZE}")
    print("MC sliding window: team's existing implementation")
    print("MC overlap: 0.0")
    print(f"MC samples: {N_MC_SAMPLES}")
    print(f"Refinement threshold: {THRESHOLD}")
    print()

    dataset = FeTADataset(
        Path("src/data/splits/split_v1.json"),
        Path("src/data/config.yaml"),
        split_name="test",
        mode="eval",
    )

    model, _ = build_model(
        "comparative",
        device,
    )

    state = torch.load(
        Path(
            "src/training/checkpoints_comparative/"
            "best_model.pt"
        ),
        map_location=device,
    )

    model.load_state_dict(
        state,
        strict=True,
    )

    model.eval()

    all_rows = []

    selected = []

    for index in range(len(dataset)):
        image, label, meta = dataset[index]

        if (
            args.subject is None
            or meta["subject_id"] == args.subject
        ):
            selected.append(
                (index, image, label, meta)
            )

    if not selected:
        raise RuntimeError(
            f"Subject not found: {args.subject}"
        )

    total_start = time.time()

    for run_index, (
        dataset_index,
        image,
        label,
        meta,
    ) in enumerate(selected, start=1):

        subject_id = meta["subject_id"]

        spacing = tuple(
            float(x)
            for x in meta["spacing"]
        )

        target = (
            label.cpu()
            .numpy()
            .astype(np.uint8)
        )

        save_path = (
            OUTPUT_DIR /
            f"{subject_id}_refinement.npz"
        )

        print("=" * 72)
        print(
            f"{run_index}/{len(selected)} "
            f"{subject_id}"
        )
        print("=" * 72)

        start = time.time()

        if save_path.exists():
            print(
                "Existing refinement output found; "
                "loading."
            )

            saved = np.load(save_path)

            prediction_before = saved[
                "prediction_before"
            ]

            prediction_after = saved[
                "prediction_after"
            ]

        else:
            # Make each subject reproducible independently.
            torch.manual_seed(
                42 + dataset_index
            )

            print(
                "Running 10-pass MC Dropout inference..."
            )

            (
                mean_probability,
                prediction,
                uncertainty,
            ) = mc_dropout_sliding_window(
                model,
                image,
                PATCH_SIZE,
                device,
                n_mc_samples=N_MC_SAMPLES,
            )

            print("Applying refinement...")

            refined = refine_prediction(
                prediction.unsqueeze(0),
                uncertainty.unsqueeze(0),
                threshold=THRESHOLD,
                enabled=True,
            ).squeeze(0)

            prediction_before = (
                prediction.cpu()
                .numpy()
                .astype(np.uint8)
            )

            prediction_after = (
                refined.cpu()
                .numpy()
                .astype(np.uint8)
            )

            uncertainty_np = (
                uncertainty.cpu()
                .numpy()
                .astype(np.float32)
            )

            changed = (
                prediction_before
                != prediction_after
            )

            changed_fraction = float(
                changed.mean()
            )

            print(
                "Fraction of all voxels changed: "
                f"{changed_fraction:.6f}"
            )

            np.savez_compressed(
                save_path,
                prediction_before=prediction_before,
                prediction_after=prediction_after,
                uncertainty=uncertainty_np,
                target=target,
                spacing=np.asarray(spacing),
            )

            print(
                f"Saved: {save_path}"
            )

        case_rows = evaluate(
            subject_id,
            prediction_before,
            prediction_after,
            target,
            spacing,
        )

        all_rows.extend(case_rows)

        # Progressive outputs.
        write_results(all_rows)
        write_summary(all_rows)

        print()
        print(
            f"{'Class':<12}"
            f"{'Dice before':>14}"
            f"{'Dice after':>14}"
            f"{'HD95 Δ':>12}"
            f"{'ASSD Δ':>12}"
        )

        for row in case_rows:
            print(
                f"{row['class_name']:<12}"
                f"{row['dice_before']:>14.6f}"
                f"{row['dice_after']:>14.6f}"
                f"{row['hd95_delta_mm']:>+12.4f}"
                f"{row['assd_delta_mm']:>+12.4f}"
            )

        elapsed = (
            time.time() - start
        ) / 60

        print(
            f"\nCase time: {elapsed:.1f} min"
        )
        print()

    total_elapsed = (
        time.time() - total_start
    ) / 60

    print("=" * 72)
    print("COMPLETE")
    print("=" * 72)
    print(
        f"Total time: {total_elapsed:.1f} min"
    )
    print(
        f"Results: {RESULTS_CSV}"
    )
    print(
        f"Summary: {SUMMARY_CSV}"
    )


if __name__ == "__main__":
    main()
