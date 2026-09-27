"""S1 ViT-B/16 student (VV+VH) for Stage 1 CMU and Stage 2 spatial maps."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def _interpolate_pos_embed(pos: torch.Tensor, grid: int) -> torch.Tensor:
  """Resize ViT patch position embeddings from the ImageNet 14×14 grid."""
  cls_tok, patch = pos[:, :1], pos[:, 1:]
  dim = patch.shape[-1]
  old = int(patch.shape[1] ** 0.5)
  if old * old != patch.shape[1]:
    raise RuntimeError(f"position grid is not square: {patch.shape[1]} tokens")
  if old == grid:
    return pos
  patch = patch.reshape(1, old, old, dim).permute(0, 3, 1, 2).float()
  patch = F.interpolate(patch, size=(grid, grid), mode="bicubic", align_corners=False)
  patch = patch.permute(0, 2, 3, 1).reshape(1, grid * grid, dim)
  return torch.cat([cls_tok, patch.to(dtype=pos.dtype)], dim=1)


class S1ViTB16(nn.Module):
  """ViT-B/16 with 2-channel patch embed.

  - forward: CLS (768-d) for CMU InfoNCE
  - forward_tokens: patch-token map B,768,Gh,Gw for Stage 2 adapter
  """

  def __init__(self, in_chans: int = 2, image_size: int = 256, pretrained: bool = True):
    super().__init__()
    from torchvision.models import ViT_B_16_Weights, vit_b_16

    self.in_chans = in_chans
    self.requested_size = int(image_size)
    weights = ViT_B_16_Weights.IMAGENET1K_V1 if pretrained else None
    # ImageNet checkpoint is 224. Build at 224, then stretch positions if needed.
    try:
      self.vit = vit_b_16(weights=weights)
    except TypeError:
      self.vit = vit_b_16(pretrained=bool(pretrained))
    self.image_size = 224
    self.patch_size = int(getattr(self.vit, "patch_size", 16))

    old = self.vit.conv_proj
    rgb = old.weight.detach()
    if rgb.shape[1] != 3:
      raise RuntimeError(f"expected RGB stem, got {rgb.shape[1]} input channels")
    stem = nn.Conv2d(
        in_chans,
        old.out_channels,
        kernel_size=old.kernel_size,
        stride=old.stride,
        bias=old.bias is not None,
    )
    with torch.no_grad():
      stem.weight.copy_(rgb.mean(dim=1, keepdim=True).repeat(1, in_chans, 1, 1))
      if stem.bias is not None and old.bias is not None:
        stem.bias.copy_(old.bias)
      elif stem.bias is not None:
        stem.bias.zero_()
    self.vit.conv_proj = stem
    self.vit.heads = nn.Identity()

    grid = self.requested_size // self.patch_size
    if self.requested_size % self.patch_size != 0:
      raise ValueError(f"image_size {self.requested_size} is not divisible by patch {self.patch_size}")
    if grid * self.patch_size != 224:
      pos = _interpolate_pos_embed(self.vit.encoder.pos_embedding.detach(), grid)
      self.vit.encoder.pos_embedding = nn.Parameter(pos)
      self.image_size = self.requested_size
    self.embed_dim = int(getattr(self.vit, "hidden_dim", 768))

  def _maybe_resize(self, x: torch.Tensor) -> torch.Tensor:
    if x.shape[1] != self.in_chans:
      raise ValueError(f"expected {self.in_chans} channels, got {x.shape[1]}")
    if x.shape[-2] != self.image_size or x.shape[-1] != self.image_size:
      x = F.interpolate(x, size=(self.image_size, self.image_size), mode="bilinear", align_corners=False)
    return x

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    """x: B,2,H,W → B,768 (CLS)."""
    return self.vit(self._maybe_resize(x))

  def forward_tokens(self, x: torch.Tensor) -> torch.Tensor:
    """x: B,2,H,W → B,768,Gh,Gw (patch tokens, no CLS)."""
    x = self._maybe_resize(x)
    tokens = self.vit._process_input(x)
    n = tokens.shape[0]
    batch_class_token = self.vit.class_token.expand(n, -1, -1)
    tokens = torch.cat([batch_class_token, tokens], dim=1)
    tokens = self.vit.encoder(tokens)
    patches = tokens[:, 1:, :]
    gh = gw = self.image_size // self.patch_size
    if patches.shape[1] != gh * gw:
      side = int(patches.shape[1] ** 0.5)
      gh = gw = side
    return patches.reshape(n, gh, gw, self.embed_dim).permute(0, 3, 1, 2).contiguous()
