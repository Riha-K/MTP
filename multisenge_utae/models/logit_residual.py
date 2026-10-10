"""10-class logit residual: frozen ReLU S2 U-TAE plus a SAR correction.

The photo-only model is loaded whole and frozen. The Stage 1 SAR ViT is frozen.
A small head reads the SAR token maps and adds a correction to the photo logits.
The last convolution of that head is zero, so the correction is zero at the start
and the first prediction matches the photo-only model.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from multisenge_utae.models.s1_vit import S1ViTB16
from multisenge_utae.models.utae import UTAE


class SarLogitCorrector(nn.Module):
  """Token map B,C,h,w → class residual B,K,H,W. Last conv starts at zero."""

  def __init__(self, in_ch: int, num_classes: int, hid: int = 128):
    super().__init__()
    groups = min(32, hid)
    self.mix = nn.Sequential(
        nn.Conv2d(in_ch, hid, kernel_size=1, bias=False),
        nn.GroupNorm(groups, hid),
        nn.ReLU(inplace=True),
        nn.Conv2d(hid, hid, kernel_size=3, padding=1, padding_mode="reflect", bias=False),
        nn.GroupNorm(groups, hid),
        nn.ReLU(inplace=True),
    )
    self.out = nn.Conv2d(hid, num_classes, kernel_size=1)
    nn.init.zeros_(self.out.weight)
    nn.init.zeros_(self.out.bias)

  def forward(self, feat: torch.Tensor, size_hw: tuple[int, int]) -> torch.Tensor:
    x = self.mix(feat)
    if x.shape[-2] != size_hw[0] or x.shape[-1] != size_hw[1]:
      x = F.interpolate(x, size=size_hw, mode="bilinear", align_corners=False)
    return self.out(x)

  def last_conv_max_abs(self) -> float:
    w = float(self.out.weight.detach().abs().max())
    b = float(self.out.bias.detach().abs().max()) if self.out.bias is not None else 0.0
    return max(w, b)


class LogitResidual(nn.Module):
  """logits = frozen S2 U-TAE(photo) + SAR correction."""

  def __init__(
      self,
      num_classes: int = 10,
      s2_dim: int = 10,
      s1_dim: int = 2,
      vit_image_size: int = 256,
      pad_value: float = 0.0,
  ):
    super().__init__()
    self.s2_dim = s2_dim
    self.s1_dim = s1_dim
    self.pad_value = pad_value
    self.s2 = UTAE(input_dim=s2_dim, num_classes=num_classes, activation="relu")
    self.s1_vit = S1ViTB16(in_chans=s1_dim, image_size=vit_image_size, pretrained=False)
    self.corrector = SarLogitCorrector(self.s1_vit.embed_dim, num_classes)
    self.freeze_backbones()

  def freeze_backbones(self) -> None:
    for mod in (self.s2, self.s1_vit):
      for p in mod.parameters():
        p.requires_grad = False
      mod.eval()

  def train(self, mode: bool = True):
    super().train(mode)
    self.s2.eval()
    self.s1_vit.eval()
    return self

  def _split(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    return x[:, :, : self.s2_dim], x[:, :, self.s2_dim : self.s2_dim + self.s1_dim]

  def _pad_mask(self, frames: torch.Tensor) -> torch.Tensor:
    return (frames == self.pad_value).all(dim=-1).all(dim=-1).all(dim=-1)

  def _sar_feat(self, s1: torch.Tensor) -> torch.Tensor:
    """s1 B,T,2,H,W → masked mean of ViT token maps, B,768,h,w."""
    b, t, c, h, w = s1.shape
    pad = self._pad_mask(s1)
    flat = s1.reshape(b * t, c, h, w)
    with torch.no_grad():
      tokens = self.s1_vit.forward_tokens(flat)
    tokens = tokens.view(b, t, tokens.shape[1], tokens.shape[2], tokens.shape[3])
    weight = (~pad).to(dtype=tokens.dtype).view(b, t, 1, 1, 1)
    denom = weight.sum(dim=1).clamp(min=1.0)
    return (tokens * weight).sum(dim=1) / denom

  def forward(self, x: torch.Tensor, batch_positions=None) -> torch.Tensor:
    s2, s1 = self._split(x)
    with torch.no_grad():
      base = self.s2(s2, batch_positions=batch_positions)
    feat = self._sar_feat(s1)
    delta = self.corrector(feat, size_hw=(base.shape[-2], base.shape[-1]))
    return base + delta

  def load_s2(self, ckpt_path, map_location="cpu") -> None:
    try:
      ckpt = torch.load(ckpt_path, map_location=map_location, weights_only=False)
    except TypeError:
      ckpt = torch.load(ckpt_path, map_location=map_location)
    state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    self.s2.load_state_dict(state, strict=True)
    self.freeze_backbones()
    print(f"loaded frozen S2 U-TAE from {ckpt_path}")

  def load_student_vit(self, ckpt_path, map_location="cpu") -> None:
    try:
      ckpt = torch.load(ckpt_path, map_location=map_location, weights_only=False)
    except TypeError:
      ckpt = torch.load(ckpt_path, map_location=map_location)
    state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    if isinstance(ckpt, dict) and "student" in ckpt and "model" not in ckpt:
      state = ckpt["student"]
    missing, unexpected = self.s1_vit.load_state_dict(state, strict=False)
    if missing or unexpected:
      raise RuntimeError(f"S1 ViT load mismatch missing={missing} unexpected={unexpected}")
    self.freeze_backbones()
    print(f"loaded frozen S1 ViT from {ckpt_path}")
