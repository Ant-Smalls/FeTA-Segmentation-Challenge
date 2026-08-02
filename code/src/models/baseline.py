"""
Role 2 - Baseline 3D U-Net
===========================
Standard encoder-decoder with skip connections.

Input:  (B, 1, D, H, W)   float32
Output: (B, num_classes, D, H, W)  logits (no softmax — loss functions handle that)

Architecture:
  Encoder: 4 levels, each = [ConvBlock -> MaxPool3d]
  Bottleneck: ConvBlock at lowest resolution
  Decoder: 4 levels, each = [Upsample -> concat skip -> ConvBlock]
  Head: Conv3d(32, num_classes, 1)

Feature map sizes (128^3 input):
  enc1: (B, 32,  128, 128, 128)
  enc2: (B, 64,   64,  64,  64)
  enc3: (B, 128,  32,  32,  32)
  enc4: (B, 256,  16,  16,  16)
  bottleneck: (B, 512, 8,   8,   8)
  dec4: (B, 256,  16,  16,  16)
  dec3: (B, 128,  32,  32,  32)
  dec2: (B, 64,   64,  64,  64)
  dec1: (B, 32,  128, 128, 128)
  out:  (B, num_classes, 128, 128, 128)
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


class BaselineUNet(nn.Module):
    """
    3D U-Net baseline for FeTA fetal brain segmentation.

    Args:
        in_channels:  always 1 for FeTA T2w MRI
        num_classes:  8 (background + 7 tissue labels)
        features:     channel progression per encoder level
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

        self.head = nn.Conv3d(features[0], num_classes, kernel_size=1)

    def forward(self, x):
        x, skips = self.encoder(x)
        x = self.bottleneck(x)
        x = self.decoder(x, list(reversed(skips)))
        return self.head(x)