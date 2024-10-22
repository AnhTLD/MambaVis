# pylint: disable=not-callable, too-many-ancestors
from typing import Any

import torch
from torch import nn
from torch.nn.functional import gelu

from hust_bearing.models.classifier import Classifier
from hust_bearing.models.conv_mamba.vision_mamba import VisionMamba


class ConvMamba(Classifier):
    def __init__(self, num_classes: int) -> None:
        super().__init__(num_classes)
        self.conv = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3, padding=1)
        self.batch_norm = nn.BatchNorm2d(num_features=64)
        self.pool = nn.MaxPool2d(kernel_size=2)
        self.mixer = VisionMamba(
            img_size=16,
            patch_size=4,
            depth=8,
            embed_dim=32,
            channels=64,
            num_classes=num_classes,
        )

    def forward(self, *args: Any, **kwargs: Any) -> torch.Tensor:
        inputs, *_ = args
        conv = self.batch_norm(gelu(self.conv(inputs)))
        pool = self.pool(conv)
        return self.mixer(pool)
