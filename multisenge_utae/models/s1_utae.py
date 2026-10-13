"""S1 U-TAE student (VV+VH) for Stage 1 CMU.

Spatial encoder only, same widths as the S2 teacher. On a 256 patch the map
is 128 channels at 32×32, so Stage 1 does not resize the teacher. ReLU matches
the finished S2 teacher. There are no ImageNet weights.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from multisenge_utae.models.utae import UTAE


class S1UTAE(nn.Module):
  """U-TAE spatial encoder. in_chans=2. L-TAE and the decoder stay frozen."""

  def __init__(self, in_chans: int = 2, activation: str = "relu"):
    super().__init__()
    if in_chans != 2:
      raise ValueError(f"expected 2 SAR channels, got {in_chans}")
    if activation != "relu":
      raise ValueError(f"this student is the ReLU U-TAE, got {activation}")
    self.in_chans = in_chans
    self.activation = activation
    self.net = UTAE(input_dim=in_chans, num_classes=10, activation=activation)
    self.embed_dim = int(self.net.encoder_widths[-1])
    for mod in (
        self.net.temporal_encoder,
        self.net.temporal_aggregator,
        *self.net.up_blocks,
        self.net.out_conv,
    ):
      for p in mod.parameters():
        p.requires_grad = False

  def forward_tokens(self, x: torch.Tensor) -> torch.Tensor:
    """x: B,2,H,W → B,128,H/8,W/8."""
    if x.shape[1] != self.in_chans:
      raise ValueError(f"expected {self.in_chans} channels, got {x.shape[1]}")
    feat = self.net.encode_spatial_bottleneck(x)
    if feat.dim() == 5:
      feat = feat.squeeze(1)
    side = x.shape[-1] // 8
    if feat.shape[1] != self.embed_dim or feat.shape[-2] != side or feat.shape[-1] != side:
      raise RuntimeError(f"U-TAE map {tuple(feat.shape)} is not {(self.embed_dim, side, side)}")
    return feat
