"""CMU helpers: projectors + InfoNCE (Stage 1)."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ProjHead(nn.Module):
  """128/768 to 256. LayerNorm keeps the hidden units alive.

  A dead ReLU projector maps every sample to one vector. InfoNCE then ties,
  the loss locks at ln(N), and the gradient stays zero.
  """

  def __init__(self, in_dim: int, proj_dim: int = 256, hidden: int | None = None):
    super().__init__()
    h = int(hidden) if hidden is not None else max(in_dim, proj_dim)
    self.net = nn.Sequential(
        nn.Linear(in_dim, h),
        nn.LayerNorm(h),
        nn.GELU(),
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


def project_map(proj: ProjHead, feat: torch.Tensor) -> torch.Tensor:
  """N,C,H,W or N,1,C,H,W -> N,L,D with L=H*W."""
  if feat.dim() == 5:
    if feat.shape[1] != 1:
      raise ValueError(f"expected T=1 for CMU map, got T={feat.shape[1]}")
    feat = feat.squeeze(1)
  n, _c, h, w = feat.shape
  x = feat.permute(0, 2, 3, 1).reshape(n * h * w, feat.shape[1])
  return proj(x).view(n, h * w, -1)


def spatial_info_nce(student: torch.Tensor, teacher: torch.Tensor, temperature: float = 0.07) -> torch.Tensor:
  """InfoNCE at each spatial site. student/teacher are N,L,D.

  Positive is the same batch index. Negatives are the other samples.
  Chance loss stays ln(N), not ln(N*L), so a pooled collapse cannot hide here:
  each location has its own teacher target.
  """
  if student.shape[:2] != teacher.shape[:2]:
    raise ValueError(f"spatial pair mismatch student={tuple(student.shape)} teacher={tuple(teacher.shape)}")
  q = F.normalize(student, dim=-1)
  k = F.normalize(teacher, dim=-1)
  n, length, _ = q.shape
  logits = torch.einsum("nld,mld->nlm", q, k) / max(float(temperature), 1e-6)
  logits = logits.reshape(n * length, n)
  labels = torch.arange(n, device=q.device).view(n, 1).expand(n, length).reshape(-1)
  return F.cross_entropy(logits, labels)


@torch.no_grad()
def spatial_retrieval_acc(student: torch.Tensor, teacher: torch.Tensor) -> float:
  q = F.normalize(student, dim=-1)
  k = F.normalize(teacher, dim=-1)
  pred = torch.einsum("nld,mld->nlm", q, k).argmax(dim=-1)
  labels = torch.arange(q.shape[0], device=q.device).view(-1, 1)
  return float((pred == labels).float().mean().item())


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
