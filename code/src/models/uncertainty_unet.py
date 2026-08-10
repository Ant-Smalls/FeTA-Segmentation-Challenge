"""
Role 3 - Comparative Uncertainty Model for FeTA segmentation.

This model uses the same 3D U-Net backbone as the baseline model,
with an additional uncertainty head.

Input:
    (B, 1, D, H, W) float32

Output:
    logits:
        (B, num_classes, D, H, W)
        8-class segmentation logits; no softmax.

    uncertainty:
        (B, 1, D, H, W)
        Learned voxel-wise uncertainty output.

Architecture:
    Encoder: 4 levels
    Bottleneck
    Decoder: 4 levels with skip connections
    Segmentation head: Conv3d(32, num_classes, 1)
    Uncertainty head: Conv3d(32, 1, 1)

Role 3 also provides:
    - Monte Carlo Dropout uncertainty estimation
    - Uncertainty-guided boundary refinement
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    """Two consecutive Conv3d -> BatchNorm -> ReLU operations."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class Encoder(nn.Module):
    """Downsampling path. Returns list of skip features + bottleneck input."""

    def __init__(self, in_channels: int, features: list):
        super().__init__()
        self.blocks = nn.ModuleList()
        self.pools = nn.ModuleList()

        ch = in_channels
        for f in features:
            self.blocks.append(ConvBlock(ch, f))
            self.pools.append(nn.MaxPool3d(kernel_size=2, stride=2))
            ch = f

    def forward(self, x):
        skips = []
        for block, pool in zip(self.blocks, self.pools):
            x = block(x)
            skips.append(x)
            x = pool(x)
        return x, skips


class Decoder(nn.Module):
    """Upsampling path. Consumes skip features from encoder."""

    def __init__(self, features: list):
        super().__init__()
        self.ups = nn.ModuleList()
        self.blocks = nn.ModuleList()

        # features passed in reverse: [512, 256, 128, 64, 32]
        for i in range(len(features) - 1):
            self.ups.append(
                nn.ConvTranspose3d(features[i], features[i + 1], kernel_size=2, stride=2)
            )
            self.blocks.append(ConvBlock(features[i], features[i + 1]))

    def forward(self, x, skips):
        for up, block, skip in zip(self.ups, self.blocks, skips):
            x = up(x)
            # Handle size mismatch from odd-dimension inputs
            if x.shape != skip.shape:
                x = F.interpolate(x, size=skip.shape[2:])
            x = torch.cat([skip, x], dim=1)
            x = block(x)
        return x


class UncertaintyUNet(nn.Module):
    """
    Role 3 comparative uncertainty U-Net for FeTA fetal brain segmentation.

Uses the same U-Net backbone as the baseline model and adds a
single-channel uncertainty head.
    """

    def __init__(
        self,
        in_channels: int = 1,
        num_classes: int = 8,
        features: list = [32, 64, 128, 256],
    ):
        super().__init__()

        self.encoder = Encoder(in_channels, features)
        self.bottleneck = ConvBlock(features[-1], features[-1] * 2)

        # Decoder features go from bottleneck down to first encoder level
        decoder_features = [features[-1] * 2] + list(reversed(features))
        self.decoder = Decoder(decoder_features)
        self.mc_dropout = nn.Dropout3d(p=0.2)

        self.head = nn.Conv3d(features[0], num_classes, kernel_size=1)
        self.uncertainty_head = nn.Conv3d( features[0], 1, kernel_size=1)

    def forward(self, x):
        x, skips = self.encoder(x)
        x = self.bottleneck(x)
        x = self.decoder(x, list(reversed(skips)))

        logits = self.head(x)
        uncertainty = self.uncertainty_head(x)

        return logits, uncertainty


def mc_dropout_predict(model, x, n_samples=10):
    """
    Monte Carlo Dropout inference.

    Returns:
        prediction: (B, D, H, W)
        uncertainty: (B, D, H, W)
    """
    model.eval()

    # Keep only Dropout3d layers active during inference.
    for module in model.modules():
        if isinstance(module, nn.Dropout3d):
            module.train()

    predictions = []

    with torch.no_grad():
        for _ in range(n_samples):
            logits, _ = model(x)
            probabilities = torch.softmax(logits, dim=1)
            predictions.append(probabilities)

    predictions = torch.stack(predictions, dim=0)

    mean_probability = predictions.mean(dim=0)

    prediction = torch.argmax(mean_probability, dim=1)

    # Predictive entropy.
    entropy = -torch.sum(
        mean_probability * torch.log(mean_probability + 1e-8),
        dim=1,
    )

    # Normalize entropy to [0, 1].
    uncertainty = entropy / torch.log(
        torch.tensor(
            mean_probability.shape[1],
            dtype=mean_probability.dtype,
            device=mean_probability.device,
        )
    )

    return prediction, uncertainty

def refine_prediction(prediction, uncertainty, threshold=0.5, enabled=True):
    """
    Rule-based refinement of uncertain boundary voxels.

    Args:
        prediction:  (B, D, H, W) predicted class labels
        uncertainty: (B, D, H, W) uncertainty values in [0, 1]
        threshold:   uncertainty threshold
        enabled:     whether refinement is applied

    Returns:
        Refined prediction with the same shape as prediction.
    """

    # Refinement can be switched off completely.
    if not enabled:
        return prediction

    refined = prediction.clone()

    # Identify voxels where a neighbouring voxel has a different class.
    boundary = torch.zeros_like(prediction, dtype=torch.bool)

    boundary[:, 1:, :, :] |= prediction[:, 1:, :, :] != prediction[:, :-1, :, :]
    boundary[:, :-1, :, :] |= prediction[:, :-1, :, :] != prediction[:, 1:, :, :]
    boundary[:, :, 1:, :] |= prediction[:, :, 1:, :] != prediction[:, :, :-1, :]
    boundary[:, :, :-1, :] |= prediction[:, :, :-1, :] != prediction[:, :, 1:, :]
    boundary[:, :, :, 1:] |= prediction[:, :, :, 1:] != prediction[:, :, :, :-1]
    boundary[:, :, :, :-1] |= prediction[:, :, :, :-1] != prediction[:, :, :, 1:]

    # Only refine high-uncertainty boundary voxels.
    uncertain_boundary = boundary & (uncertainty >= threshold)

    # One-hot encode the segmentation.
    num_classes = int(prediction.max().item()) + 1

    one_hot = F.one_hot(
        prediction.long(),
        num_classes=num_classes
    ).permute(0, 4, 1, 2, 3).float()

    # Count neighbouring class labels in a 3x3x3 neighbourhood.
    kernel = torch.ones(
        (num_classes, 1, 3, 3, 3),
        device=prediction.device
    )

    neighbour_counts = F.conv3d(
        one_hot,
        kernel,
        padding=1,
        groups=num_classes
    )

    # Choose the local majority class.
    local_majority = neighbour_counts.argmax(dim=1)

    # Apply refinement only at uncertain boundary voxels.
    refined[uncertain_boundary] = local_majority[uncertain_boundary]

    return refined