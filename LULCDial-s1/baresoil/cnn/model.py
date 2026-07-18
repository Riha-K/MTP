"""ResNet-18/50 with 14-way multi-label head."""

from __future__ import annotations

import torch
import torch.nn as nn
from torchvision import models


def build_resnet(arch: str = "resnet18", num_classes: int = 14, pretrained: bool = True) -> nn.Module:
    if arch == "resnet18":
        try:
            weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
            model = models.resnet18(weights=weights)
        except AttributeError:
            model = models.resnet18(pretrained=pretrained)
    elif arch == "resnet50":
        try:
            weights = models.ResNet50_Weights.IMAGENET1K_V1 if pretrained else None
            model = models.resnet50(weights=weights)
        except AttributeError:
            model = models.resnet50(pretrained=pretrained)
    else:
        raise ValueError(f"Unsupported arch: {arch}")

    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    return model


def load_checkpoint(path: str, device: torch.device, arch: str = "resnet18", num_classes: int = 14) -> nn.Module:
    ckpt = torch.load(path, map_location=device)
    saved_arch = ckpt.get("arch", arch)
    model = build_resnet(arch=saved_arch, num_classes=num_classes, pretrained=False)
    model.load_state_dict(ckpt["model"])
    model.to(device)
    model.eval()
    return model
