"""Novelty Task H — hierarchical / agent heads (A1 UF-vs-rest, A2 Dense-vs-Sparse).

Skip water agent (10c-only, optional). Do not call this "P2" — clashes with U-TAE P3/P4/P5.

Training masks are 0-indexed (dataset remaps paper class c → c-1); ignore=255.
Paper UF 1..5 → code 0..4; Dense=0; Sparse=1.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def coarse_uf_target(mask: torch.Tensor) -> torch.Tensor:
  """Map fine mask -> {0: rest, 1: urban fabric (0-4)}; keep 255 ignore."""
  out = torch.zeros_like(mask)
  uf = (mask >= 0) & (mask <= 4) & (mask != 255)
  out[uf] = 1
  out[mask == 255] = 255
  return out


def dense_sparse_target(mask: torch.Tensor) -> torch.Tensor:
  """Map -> {0: Dense(code 0), 1: Sparse(code 1)}; ignore elsewhere (+255)."""
  out = torch.full_like(mask, 255)
  out[mask == 0] = 0
  out[mask == 1] = 1
  return out


class HierarchicalHeads(nn.Module):
  """Extra 1x1 heads on decoder features (before / beside fine out_conv).

  Attach to MA-UTAE or U-TAE by wrapping, or call from a wrapper model.
  """

  def __init__(self, in_channels: int, use_a1: bool = True, use_a2: bool = True):
    super().__init__()
    self.use_a1 = use_a1
    self.use_a2 = use_a2
    self.a1 = nn.Conv2d(in_channels, 2, kernel_size=1) if use_a1 else None
    self.a2 = nn.Conv2d(in_channels, 2, kernel_size=1) if use_a2 else None

  def forward(self, feat: torch.Tensor) -> dict[str, torch.Tensor]:
    out: dict[str, torch.Tensor] = {}
    if self.a1 is not None:
      out["a1"] = self.a1(feat)
    if self.a2 is not None:
      out["a2"] = self.a2(feat)
    return out


def hierarchical_loss(
    fine_logits: torch.Tensor,
    mask: torch.Tensor,
    aux: dict[str, torch.Tensor],
    fine_criterion: nn.Module,
    lambda_a1: float = 0.3,
    lambda_a2: float = 0.2,
) -> tuple[torch.Tensor, dict[str, float]]:
  """L = L_fine + λ1 L_A1 + λ2 L_A2 (A2 only on Dense/Sparse pixels)."""
  loss_fine = fine_criterion(fine_logits, mask)
  parts = {"fine": float(loss_fine.detach())}
  total = loss_fine

  if "a1" in aux and lambda_a1 > 0:
    t1 = coarse_uf_target(mask)
    loss_a1 = F.cross_entropy(aux["a1"], t1, ignore_index=255)
    total = total + lambda_a1 * loss_a1
    parts["a1"] = float(loss_a1.detach())

  if "a2" in aux and lambda_a2 > 0:
    t2 = dense_sparse_target(mask)
    # Only Dense/Sparse pixels contribute (others already 255).
    if (t2 != 255).any():
      loss_a2 = F.cross_entropy(aux["a2"], t2, ignore_index=255)
      total = total + lambda_a2 * loss_a2
      parts["a2"] = float(loss_a2.detach())
    else:
      parts["a2"] = 0.0

  parts["total"] = float(total.detach())
  return total, parts
