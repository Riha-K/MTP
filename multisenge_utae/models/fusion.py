"""Modality fusion blocks for MA-UTAE (P1)."""

from __future__ import annotations

import torch
import torch.nn as nn


class ConcatFusion(nn.Module):
  """Stack S2/S1 features and mix with 1x1 conv -> out_channels."""

  def __init__(self, in_channels: int, out_channels: int | None = None):
    super().__init__()
    out_channels = in_channels if out_channels is None else out_channels
    self.proj = nn.Sequential(
        nn.Conv2d(2 * in_channels, out_channels, kernel_size=1, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
    )

  def forward(self, f_s2: torch.Tensor, f_s1: torch.Tensor) -> torch.Tensor:
    return self.proj(torch.cat([f_s2, f_s1], dim=1))


class GatedFusion(nn.Module):
  """Soft modality gate: out = g⊙f_s2 + (1-g)⊙f_s1 (sir A5 / P1 main)."""

  def __init__(self, channels: int):
    super().__init__()
    self.gate = nn.Sequential(
        nn.Conv2d(2 * channels, channels, kernel_size=1, bias=True),
        nn.Sigmoid(),
    )

  def forward(self, f_s2: torch.Tensor, f_s1: torch.Tensor) -> torch.Tensor:
    g = self.gate(torch.cat([f_s2, f_s1], dim=1))
    return g * f_s2 + (1.0 - g) * f_s1


def build_fusion(kind: str, channels: int) -> nn.Module:
  kind = kind.lower().strip()
  if kind in ("concat", "cat", "1x1"):
    return ConcatFusion(channels)
  if kind in ("gated", "gate"):
    return GatedFusion(channels)
  raise ValueError(f"unknown fusion kind={kind!r}; use concat|gated")
