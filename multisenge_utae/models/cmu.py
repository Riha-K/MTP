"""CMU helpers: projectors + InfoNCE (Stage 1)."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ProjHead(nn.Module):
  """MLP projector for contrastive CMU."""

  def __init__(self, in_dim: int, proj_dim: int = 256, hidden: int | None = None):
    super().__init__()
    h = int(hidden) if hidden is not None else max(in_dim, proj_dim)
    self.net = nn.Sequential(
        nn.Linear(in_dim, h),
        nn.ReLU(inplace=True),
        nn.Linear(h, proj_dim),
    )

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    return self.net(x)


def global_pool(feat: torch.Tensor) -> torch.Tensor:
  """B,C,H,W or B,T,C,H,W → B,C (mean over spatial; squeeze T if 1)."""
  if feat.dim() == 5:
    if feat.shape[1] != 1:
      raise ValueError(f"expected T=1 for CMU pool, got T={feat.shape[1]}")
    feat = feat.squeeze(1)
  return feat.mean(dim=(-2, -1))


def info_nce(student: torch.Tensor, teacher: torch.Tensor, temperature: float = 0.07) -> torch.Tensor:
  """Symmetric-style InfoNCE with student queries vs teacher keys (batch negatives)."""
  q = F.normalize(student, dim=-1)
  k = F.normalize(teacher, dim=-1)
  logits = q @ k.t() / max(float(temperature), 1e-6)
  labels = torch.arange(q.shape[0], device=q.device)
  return F.cross_entropy(logits, labels)


@torch.no_grad()
def retrieval_acc(student: torch.Tensor, teacher: torch.Tensor) -> float:
  q = F.normalize(student, dim=-1)
  k = F.normalize(teacher, dim=-1)
  pred = (q @ k.t()).argmax(dim=1)
  labels = torch.arange(q.shape[0], device=q.device)
  return float((pred == labels).float().mean().item())
