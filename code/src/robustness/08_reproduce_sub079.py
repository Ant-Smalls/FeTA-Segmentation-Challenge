from pathlib import Path

import torch
import torch.nn.functional as F

from src.preprocessing_data_preparation.dataset import (
    FeTADataset,
    reconstruct_from_patches,
)
from src.training.train import build_model, sliding_window_coords


SUBJECT = "sub-079"
PATCH_SIZE = (128, 128, 128)

REFERENCE = {
    "baseline": {
        1: 0.8988650340078321,
        2: 0.7828616230594365,
        3: 0.9264044059328,
        4: 0.822862281731658,
        5: 0.9105169269405937,
        6: 0.8554206638933696,
        7: 0.8367739488513221,
    },
    "comparative": {
        1: 0.8890960833909343,
        2: 0.7895077540392166,
        3: 0.9261316144614143,
        4: 0.8367925581395349,
        5: 0.8968408849386532,
        6: 0.8495022965558643,
        7: 0.8278152331057386,
    },
}

CLASS_NAMES = {
    1: "eCSF",
    2: "GM",
    3: "WM",
    4: "Ventricles",
    5: "Cerebellum",
    6: "dGM",
    7: "Brainstem",
}


def dice_score(pred, target, class_id):
    p = pred == class_id
    t = target == class_id

    intersection = (p & t).sum().item()
    denominator = p.sum().item() + t.sum().item()

    if denominator == 0:
        return 1.0

    return 2.0 * intersection / denominator


@torch.no_grad()
def infer_full_stride(model, image, model_type, device):
    """
    128^3 sliding-window inference using full-patch stride:
    overlap=0.0, with the repo's Gaussian reconstruction.
    """
    _, D, H, W = image.shape

    coords = sliding_window_coords(
        (D, H, W),
        PATCH_SIZE,
        overlap=0.0,
    )

    patches = []
    patch_coords = []

    print(f"Volume shape: {(D, H, W)}")
    print(f"Number of tiles: {len(coords)}")

    for i, (dz, dy, dx) in enumerate(coords, start=1):
        raw = image[:, dz, dy, dx]

        pad = []
        for dim_size, p in zip(raw.shape[1:], PATCH_SIZE):
            pad = [0, p - dim_size] + pad

        raw = F.pad(raw, pad)
        patch = raw.unsqueeze(0).to(device)

        output = model(patch)

        if model_type == "comparative":
            logits = output[0]
        else:
            logits = output

        logits = logits.squeeze(0)

        orig_d = min(dz.stop, D) - dz.start
        orig_h = min(dy.stop, H) - dy.start
        orig_w = min(dx.stop, W) - dx.start

        logits = logits[:, :orig_d, :orig_h, :orig_w]

        patches.append(logits.cpu())

        patch_coords.append(
            (
                slice(dz.start, dz.start + orig_d),
                slice(dy.start, dy.start + orig_h),
                slice(dx.start, dx.start + orig_w),
            )
        )

        print(f"  tile {i}/{len(coords)} complete")

    full_logits = reconstruct_from_patches(
        patches,
        patch_coords,
        full_volume_shape=(D, H, W),
        aggregation="gaussian",
    )

    return full_logits.argmax(dim=0)


def find_subject(dataset, subject_id):
    for i in range(len(dataset)):
        image, label, meta = dataset[i]

        if meta["subject_id"] == subject_id:
            return image, label, meta

    raise RuntimeError(f"{subject_id} not found in test split")


def evaluate_model(model_type, checkpoint, image, label, device):
    print()
    print("=" * 72)
    print(model_type.upper())
    print("=" * 72)

    model, _ = build_model(model_type, device)

    state = torch.load(
        checkpoint,
        map_location=device,
    )

    model.load_state_dict(state, strict=True)
    model.eval()

    prediction = infer_full_stride(
        model,
        image,
        model_type,
        device,
    )

    print()
    print(
        f"{'Class':<14}"
        f"{'Reproduced':>14}"
        f"{'Reference':>14}"
        f"{'Difference':>14}"
    )

    differences = []

    for class_id in range(1, 8):
        reproduced = dice_score(
            prediction,
            label,
            class_id,
        )

        reference = REFERENCE[model_type][class_id]
        difference = reproduced - reference
        differences.append(abs(difference))

        print(
            f"{CLASS_NAMES[class_id]:<14}"
            f"{reproduced:>14.6f}"
            f"{reference:>14.6f}"
            f"{difference:>+14.6f}"
        )

    print()
    print(f"Maximum absolute difference: {max(differences):.8f}")

    return prediction


def main():
    # CPU is deliberate for the provenance test.
    device = torch.device("cpu")

    split_path = Path("src/data/splits/split_v1.json")
    config_path = Path("src/data/config.yaml")

    dataset = FeTADataset(
        split_path,
        config_path,
        split_name="test",
        mode="eval",
    )

    image, label, meta = find_subject(
        dataset,
        SUBJECT,
    )

    print("=" * 72)
    print("SUBJECT")
    print("=" * 72)
    print(f"Subject: {meta['subject_id']}")
    print(f"Image shape: {tuple(image.shape)}")
    print(f"Label shape: {tuple(label.shape)}")
    print(f"Spacing: {meta['spacing']}")

    evaluate_model(
        "baseline",
        Path("src/training/checkpoints_baseline/best_model.pt"),
        image,
        label,
        device,
    )

    evaluate_model(
        "comparative",
        Path("src/training/checkpoints_comparative/best_model.pt"),
        image,
        label,
        device,
    )


if __name__ == "__main__":
    main()
