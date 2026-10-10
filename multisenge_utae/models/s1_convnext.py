"""S1 ConvNeXt-Tiny student (VV+VH) for Stage 1 CMU.

The map is the 32×32 stage. That is the same grid as the U-TAE spatial
bottleneck on a 256 patch, so Stage 1 does not resize the teacher.
GELU inside the ConvNeXt blocks stays as in the ImageNet weights.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class S1ConvNeXtTiny(nn.Module):
  """ConvNeXt-Tiny stem through the 32×32 stage. in_chans=2."""

  def __init__(self, in_chans: int = 2, image_size: int = 256, pretrained: bool = True):
    super().__init__()
    from torchvision.models import ConvNeXt_Tiny_Weights, convnext_tiny

    if in_chans != 2:
      raise ValueError(f"expected 2 SAR channels, got {in_chans}")
    self.in_chans = in_chans
    self.image_size = int(image_size)
    weights = ConvNeXt_Tiny_Weights.IMAGENET1K_V1 if pretrained else None
    full = convnext_tiny(weights=weights)
    # features: 0 stem, 1 stage 64×64, 2 down, 3 stage 32×32. Later stages are unused.
    self.features = nn.Sequential(*[full.features[i] for i in range(4)])
    self.embed_dim = 192
    self._replace_stem(in_chans)
    del full

  def _replace_stem(self, in_chans: int) -> None:
    stem = self.features[0]
    old = stem[0] if isinstance(stem, nn.Sequential) else stem
    if not isinstance(old, nn.Conv2d) or old.in_channels != 3:
      raise RuntimeError(f"expected an RGB ConvNeXt stem, got {type(old)}")
    new = nn.Conv2d(
        in_chans,
        old.out_channels,
        kernel_size=old.kernel_size,
        stride=old.stride,
        padding=old.padding,
        bias=old.bias is not None,
    )
    with torch.no_grad():
      rgb = old.weight.detach()
      new.weight.copy_(rgb.mean(dim=1, keepdim=True).repeat(1, in_chans, 1, 1))
      if new.bias is not None and old.bias is not None:
        new.bias.copy_(old.bias)
    if isinstance(stem, nn.Sequential):
      stem[0] = new
    else:
      self.features[0] = new

  def forward_tokens(self, x: torch.Tensor) -> torch.Tensor:
    """x: B,2,H,W → B,192,H/8,W/8."""
    if x.shape[1] != self.in_chans:
      raise ValueError(f"expected {self.in_chans} channels, got {x.shape[1]}")
    if x.shape[-2] != self.image_size or x.shape[-1] != self.image_size:
      raise ValueError(f"expected {self.image_size}px, got {tuple(x.shape[-2:])}")
    for layer in self.features:
      x = layer(x)
    side = self.image_size // 8
    if x.shape[1] != self.embed_dim or x.shape[-2] != side or x.shape[-1] != side:
      raise RuntimeError(f"ConvNeXt map {tuple(x.shape)} is not {(self.embed_dim, side, side)}")
    return x
