from pathlib import Path
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
from src.robustness.surface_metrics import (
    hd95,
    assd,
)


PATCH_SIZE = (128, 128, 128)
OVERLAP = 0.0

OUTPUT_DIR = Path("src/robustness/outputs/final_predictions")
RESULTS_CSV = Path("src/robustness/outputs/surface_metrics_per_case.csv")
SUMMARY_CSV = Path("src/robustness/outputs/surface_metrics_summary.csv")

CLASS_NAMES = {
    1: "eCSF",
    2: "GM",
    3: "WM",
    4: "Ventricles",
    5: "Cerebellum",
    6: "dGM",
    7: "Brainstem",
}


def dice_score(prediction, target, class_id):
    pred = prediction == class_id
    gt = target == class_id

    denominator = pred.sum() + gt.sum()

    if denominator == 0:
        return 1.0

    intersection = np.logical_and(pred, gt).sum()
    return float(2.0 * intersection / denominator)


@torch.no_grad()
def sliding_window_full_stride(model, image, model_type, device):
    _, depth, height, width = image.shape

    coords = sliding_window_coords(
        (depth, height, width),
        PATCH_SIZE,
        overlap=OVERLAP,
    )

    patches = []
    patch_coords = []

    for tile_index, (dz, dy, dx) in enumerate(coords, start=1):
        raw = image[:, dz, dy, dx]

        pad = []

        for dim_size, patch_dim in zip(raw.shape[1:], PATCH_SIZE):
            pad = [0, patch_dim - dim_size] + pad

        raw = F.pad(raw, pad)
        patch = raw.unsqueeze(0).to(device)

        output = model(patch)

        if model_type == "comparative":
            logits = output[0]
        else:
            logits = output

        logits = logits.squeeze(0)

        original_depth = min(dz.stop, depth) - dz.start
        original_height = min(dy.stop, height) - dy.start
        original_width = min(dx.stop, width) - dx.start

        logits = logits[
            :,
            :original_depth,
            :original_height,
            :original_width,
        ]

        patches.append(logits.cpu())

        patch_coords.append(
            (
                slice(dz.start, dz.start + original_depth),
                slice(dy.start, dy.start + original_height),
                slice(dx.start, dx.start + original_width),
            )
        )

        print(
            f"      tile {tile_index}/{len(coords)}",
            end="\r",
            flush=True,
        )

    print(" " * 50, end="\r")

    full_logits = reconstruct_from_patches(
        patches,
        patch_coords,
        full_volume_shape=(depth, height, width),
        aggregation="gaussian",
    )

    return full_logits.argmax(dim=0).numpy().astype(np.uint8)


def load_model(model_type, checkpoint, device):
    model, _ = build_model(model_type, device)

    state = torch.load(
        checkpoint,
        map_location=device,
    )

    model.load_state_dict(state, strict=True)
    model.eval()

    return model


def evaluate_prediction(
    subject_id,
    model_type,
    prediction,
    target,
    spacing,
):
    rows = []

    for class_id, class_name in CLASS_NAMES.items():
        pred_mask = prediction == class_id
        gt_mask = target == class_id

        dice = dice_score(
            prediction,
            target,
            class_id,
        )

        hd95_value = hd95(
            pred_mask,
            gt_mask,
            spacing=spacing,
        )

        assd_value = assd(
            pred_mask,
            gt_mask,
            spacing=spacing,
        )

        rows.append(
            {
                "subject_id": subject_id,
                "model": model_type,
                "class_id": class_id,
                "class_name": class_name,
                "dice": dice,
                "hd95_mm": hd95_value,
                "assd_mm": assd_value,
            }
        )

    return rows


def write_results(rows):
    RESULTS_CSV.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    fields = [
        "subject_id",
        "model",
        "class_id",
        "class_name",
        "dice",
        "hd95_mm",
        "assd_mm",
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
    summary_rows = []

    for model_type in ["baseline", "comparative"]:
        model_rows = [
            row
            for row in rows
            if row["model"] == model_type
        ]

        for class_id, class_name in CLASS_NAMES.items():
            class_rows = [
                row
                for row in model_rows
                if row["class_id"] == class_id
            ]

            dice_values = np.array(
                [row["dice"] for row in class_rows],
                dtype=float,
            )

            hd95_values = np.array(
                [row["hd95_mm"] for row in class_rows],
                dtype=float,
            )

            assd_values = np.array(
                [row["assd_mm"] for row in class_rows],
                dtype=float,
            )

            summary_rows.append(
                {
                    "model": model_type,
                    "class_id": class_id,
                    "class_name": class_name,
                    "mean_dice": np.nanmean(dice_values),
                    "median_dice": np.nanmedian(dice_values),
                    "mean_hd95_mm": np.nanmean(hd95_values),
                    "median_hd95_mm": np.nanmedian(hd95_values),
                    "mean_assd_mm": np.nanmean(assd_values),
                    "median_assd_mm": np.nanmedian(assd_values),
                }
            )

    fields = [
        "model",
        "class_id",
        "class_name",
        "mean_dice",
        "median_dice",
        "mean_hd95_mm",
        "median_hd95_mm",
        "mean_assd_mm",
        "median_assd_mm",
    ]

    with SUMMARY_CSV.open(
        "w",
        newline="",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=fields,
        )

        writer.writeheader()
        writer.writerows(summary_rows)


def main():
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    device = torch.device("cpu")

    print("=" * 72)
    print("FINAL SURFACE EVALUATION")
    print("=" * 72)
    print(f"Device: {device}")
    print(f"Patch size: {PATCH_SIZE}")
    print(f"Overlap: {OVERLAP}")
    print("Aggregation: gaussian")
    print()

    dataset = FeTADataset(
        Path("src/data/splits/split_v1.json"),
        Path("src/data/config.yaml"),
        split_name="test",
        mode="eval",
    )

    baseline = load_model(
        "baseline",
        Path(
            "src/training/checkpoints_baseline/"
            "best_model.pt"
        ),
        device,
    )

    comparative = load_model(
        "comparative",
        Path(
            "src/training/checkpoints_comparative/"
            "best_model.pt"
        ),
        device,
    )

    all_rows = []

    print(f"Test subjects: {len(dataset)}")
    print()

    start_total = time.time()

    for case_index in range(len(dataset)):
        image, label, meta = dataset[case_index]

        subject_id = meta["subject_id"]
        spacing = tuple(float(x) for x in meta["spacing"])

        target = (
            label.cpu()
            .numpy()
            .astype(np.uint8)
        )

        print("=" * 72)
        print(
            f"{case_index + 1}/{len(dataset)} "
            f"{subject_id}"
        )
        print("=" * 72)
        print(f"Shape: {tuple(target.shape)}")
        print(f"Spacing: {spacing}")

        case_start = time.time()

        prediction_file = (
            OUTPUT_DIR /
            f"{subject_id}_predictions.npz"
        )

        if prediction_file.exists():
            print(
                "  Existing predictions found; "
                "loading instead of rerunning inference."
            )

            saved = np.load(prediction_file)

            baseline_prediction = saved[
                "baseline"
            ]

            comparative_prediction = saved[
                "comparative"
            ]

        else:
            print("  Baseline inference")

            baseline_prediction = (
                sliding_window_full_stride(
                    baseline,
                    image,
                    "baseline",
                    device,
                )
            )

            print("  Comparative inference")

            comparative_prediction = (
                sliding_window_full_stride(
                    comparative,
                    image,
                    "comparative",
                    device,
                )
            )

            np.savez_compressed(
                prediction_file,
                baseline=baseline_prediction,
                comparative=comparative_prediction,
                target=target,
                spacing=np.asarray(spacing),
            )

            print(
                f"  Saved: {prediction_file}"
            )

        print("  Computing surface metrics")

        baseline_rows = evaluate_prediction(
            subject_id,
            "baseline",
            baseline_prediction,
            target,
            spacing,
        )

        comparative_rows = evaluate_prediction(
            subject_id,
            "comparative",
            comparative_prediction,
            target,
            spacing,
        )

        all_rows.extend(baseline_rows)
        all_rows.extend(comparative_rows)

        # Save progressively after every subject.
        write_results(all_rows)
        write_summary(all_rows)

        baseline_fg = np.mean(
            [row["dice"] for row in baseline_rows]
        )

        comparative_fg = np.mean(
            [row["dice"] for row in comparative_rows]
        )

        elapsed = (
            time.time() - case_start
        ) / 60.0

        print(
            f"  Baseline mean FG Dice:    "
            f"{baseline_fg:.4f}"
        )

        print(
            f"  Comparative mean FG Dice: "
            f"{comparative_fg:.4f}"
        )

        print(
            f"  Case time: {elapsed:.1f} min"
        )
        print()

    write_results(all_rows)
    write_summary(all_rows)

    total_minutes = (
        time.time() - start_total
    ) / 60.0

    print("=" * 72)
    print("COMPLETE")
    print("=" * 72)
    print(
        f"Total time: {total_minutes:.1f} min"
    )

    print(
        f"Per-case results: {RESULTS_CSV}"
    )

    print(
        f"Summary: {SUMMARY_CSV}"
    )

    print(
        f"Saved predictions: {OUTPUT_DIR}"
    )


if __name__ == "__main__":
    main()
