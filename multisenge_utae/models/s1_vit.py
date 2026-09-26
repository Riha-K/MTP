"""S1 ViT-B/16 student (VV+VH) for Stage 1 CMU."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class S1ViTB16(nn.Module):
  """ViT-B/16 with 2-channel patch embed; returns CLS features (768-d)."""

  def __init__(self, in_chans: int = 2, image_size: int = 256):
    super().__init__()
    from torchvision.models import vit_b_16

    self.in_chans = in_chans
    self.requested_size = int(image_size)
    self.vit = None
    self.image_size = 224
    try:
      self.vit = vit_b_16(weights=None, image_size=self.requested_size)
      self.image_size = self.requested_size
    except TypeError:
      self.vit = vit_b_16(weights=None)
      self.image_size = 224

    old = self.vit.conv_proj
    self.vit.conv_proj = nn.Conv2d(
        in_chans,
        old.out_channels,
        kernel_size=old.kernel_size,
        stride=old.stride,
        bias=old.bias is not None,
    )
    nn.init.kaiming_normal_(self.vit.conv_proj.weight, mode="fan_out", nonlinearity="relu")
    if self.vit.conv_proj.bias is not None:
      nn.init.zeros_(self.vit.conv_proj.bias)
    self.vit.heads = nn.Identity()
    self.embed_dim = int(getattr(self.vit, "hidden_dim", 768))

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    """x: B,2,H,W → B,768 (CLS)."""
    if x.shape[1] != self.in_chans:
      raise ValueError(f"expected {self.in_chans} channels, got {x.shape[1]}")
    if x.shape[-2] != self.image_size or x.shape[-1] != self.image_size:
      x = F.interpolate(x, size=(self.image_size, self.image_size), mode="bilinear", align_corners=False)
    return self.vit(x)
