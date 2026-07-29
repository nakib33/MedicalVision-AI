"""Shared Model: Custom CNN — a configurable CNN for small/medium datasets."""
import torch.nn as nn
import torch.nn.functional as F


class CustomCNN(nn.Module):
    """A flexible CNN for medical image classification.

    Architecture:
        ConvBlock × N → AdaptiveAvgPool → Dropout → Linear

    Default config (224×224 input): 4 conv blocks with doubling filters.
    """

    def __init__(self, num_classes: int, in_channels: int = 3,
                 base_filters: int = 32, num_blocks: int = 4,
                 dropout: float = 0.5):
        super().__init__()

        layers = []
        filters = base_filters
        for i in range(num_blocks):
            layers.append(ConvBlock(in_channels, filters,
                                    kernel_size=3 if i > 0 else 7,
                                    pool=(i > 0)))  # pool from block 2 onward
            in_channels = filters
            filters = min(filters * 2, 512)

        self.features = nn.Sequential(*layers)
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(in_channels, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = self.classifier(x)
        return x


class ConvBlock(nn.Module):
    """Conv2D → BatchNorm → ReLU → (optional MaxPool)."""

    def __init__(self, in_ch: int, out_ch: int, kernel_size: int = 3,
                 pool: bool = True):
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size, padding=padding)
        self.bn = nn.BatchNorm2d(out_ch)
        self.pool = nn.MaxPool2d(2) if pool else nn.Identity()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = F.relu(x, inplace=True)
        x = self.pool(x)
        return x
