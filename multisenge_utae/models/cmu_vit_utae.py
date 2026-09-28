"""CMU-ViT U-TAE Stage 2: S1 ViT (CMU'd) + S2 CNN encoder, L-TAE both, CONCAT fuse.

Plan §8.4: bottleneck fusion only; decoder uses S2 skips (no S1 skips).
P4: freeze the loaded S2 encoder, S2 L-TAE, and CMU ViT.
S1 L-TAE is new, so it trains with the adapter, fusion, and decoder.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from multisenge_utae.models.fusion import build_fusion
from multisenge_utae.models.ltae import LTAE2d
from multisenge_utae.models.ma_utae import _ModalityEncoder
from multisenge_utae.models.s1_vit import S1ViTB16
from multisenge_utae.models.utae import ConvBlock, TemporalAggregator, UpConvBlock


class BottleneckAdapter(nn.Module):
  """ViT token map → S2 bottleneck channels (+ optional spatial resize in forward)."""

  def __init__(self, in_ch: int, out_ch: int):
    super().__init__()
    self.proj = nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False),
        nn.GroupNorm(num_groups=min(32, out_ch), num_channels=out_ch),
        nn.ReLU(inplace=True),
    )

  def forward(self, x: torch.Tensor, size_hw: tuple[int, int] | None = None) -> torch.Tensor:
    x = self.proj(x)
    if size_hw is not None and (x.shape[-2] != size_hw[0] or x.shape[-1] != size_hw[1]):
      x = F.interpolate(x, size=size_hw, mode="bilinear", align_corners=False)
    return x


class CMUViTUTAE(nn.Module):
  """Dual-stream seg: S2 CNN + S1 CMU-ViT → L-TAEs → CONCAT bottleneck → decoder."""

  def __init__(
      self,
      num_classes: int,
      s2_dim: int = 10,
      s1_dim: int = 2,
      fusion: str = "concat",
      encoder_widths=None,
      decoder_widths=None,
      str_conv_k: int = 4,
      str_conv_s: int = 2,
      str_conv_p: int = 1,
      agg_mode: str = "att_group",
      encoder_norm: str = "group",
      n_head: int = 16,
      d_model: int = 256,
      d_k: int = 4,
      pad_value: float = 0.0,
      padding_mode: str = "reflect",
      vit_image_size: int = 256,
  ):
    super().__init__()
    if encoder_widths is None:
      encoder_widths = [64, 64, 64, 128]
    if decoder_widths is None:
      decoder_widths = [32, 32, 64, 128]
    assert len(encoder_widths) == len(decoder_widths)
    assert encoder_widths[-1] == decoder_widths[-1]

    self.s2_dim = s2_dim
    self.s1_dim = s1_dim
    self.n_stages = len(encoder_widths)
    self.encoder_widths = encoder_widths
    self.decoder_widths = decoder_widths
    self.pad_value = pad_value
    self.fusion_kind = fusion

    self.encoder_s2 = _ModalityEncoder(
        input_dim=s2_dim,
        encoder_widths=encoder_widths,
        str_conv_k=str_conv_k,
        str_conv_s=str_conv_s,
        str_conv_p=str_conv_p,
        pad_value=pad_value,
        encoder_norm=encoder_norm,
        padding_mode=padding_mode,
    )
    # Stage 2 loads the CMU student checkpoint; skip the ImageNet download here.
    self.s1_vit = S1ViTB16(in_chans=s1_dim, image_size=vit_image_size, pretrained=False)
    self.s1_adapter = BottleneckAdapter(self.s1_vit.embed_dim, encoder_widths[-1])

    self.temporal_s2 = LTAE2d(
        in_channels=encoder_widths[-1],
        d_model=d_model,
        n_head=n_head,
        mlp=[d_model, encoder_widths[-1]],
        return_att=True,
        d_k=d_k,
    )
    self.temporal_s1 = LTAE2d(
        in_channels=encoder_widths[-1],
        d_model=d_model,
        n_head=n_head,
        mlp=[d_model, encoder_widths[-1]],
        return_att=True,
        d_k=d_k,
    )
    self.fuse_bottleneck = build_fusion(fusion, encoder_widths[-1])
    self.temporal_aggregator = TemporalAggregator(mode=agg_mode)
    self.up_blocks = nn.ModuleList(
        UpConvBlock(
            d_in=decoder_widths[i],
            d_out=decoder_widths[i - 1],
            d_skip=encoder_widths[i - 1],
            k=str_conv_k,
            s=str_conv_s,
            p=str_conv_p,
            padding_mode=padding_mode,
        )
        for i in range(self.n_stages - 1, 0, -1)
    )
    self.out_conv = ConvBlock(nkernels=[decoder_widths[0], 32, num_classes], padding_mode=padding_mode)
    self._frozen_eval: list[nn.Module] = []

  def _split(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    return x[:, :, : self.s2_dim], x[:, :, self.s2_dim : self.s2_dim + self.s1_dim]

  def _pad_mask(self, input: torch.Tensor) -> torch.Tensor:
    return (input == self.pad_value).all(dim=-1).all(dim=-1).all(dim=-1)

  def encode_s1_dates(self, s1: torch.Tensor, size_hw: tuple[int, int]) -> torch.Tensor:
    """s1: B,T,2,H,W → B,T,C_b,h,w (adapter to S2 bottleneck grid)."""
    b, t, c, h, w = s1.shape
    flat = s1.reshape(b * t, c, h, w)
    tokens = self.s1_vit.forward_tokens(flat)
    adapted = self.s1_adapter(tokens, size_hw=size_hw)
    cb, hb, wb = adapted.shape[1], adapted.shape[2], adapted.shape[3]
    return adapted.view(b, t, cb, hb, wb)

  def forward(self, input: torch.Tensor, batch_positions=None, return_att: bool = False):
    s2, s1 = self._split(input)
    pad_s2 = self._pad_mask(s2)
    pad_s1 = self._pad_mask(s1)

    maps_s2 = self.encoder_s2(s2)
    bn_s2 = maps_s2[-1]  # B,T,C,h,w
    size_hw = (bn_s2.shape[-2], bn_s2.shape[-1])
    maps_s1_bn = self.encode_s1_dates(s1, size_hw=size_hw)

    out_s2, att_s2 = self.temporal_s2(bn_s2, batch_positions=batch_positions, pad_mask=pad_s2)
    out_s1, att_s1 = self.temporal_s1(maps_s1_bn, batch_positions=batch_positions, pad_mask=pad_s1)
    out = self.fuse_bottleneck(out_s2, out_s1)

    # §8.4: S2 skips only
    for i in range(self.n_stages - 1):
      skip = self.temporal_aggregator(maps_s2[-(i + 2)], pad_mask=pad_s2, attn_mask=att_s2)
      out = self.up_blocks[i](out, skip)

    logits = self.out_conv(out)
    if return_att:
      return logits, {"att_s2": att_s2, "att_s1": att_s1}
    return logits

  def load_s2_utae(self, ckpt_path, map_location="cpu") -> None:
    """Copy the trained S2 U-TAE encoder and its L-TAE (not the class head)."""
    try:
      ckpt = torch.load(ckpt_path, map_location=map_location, weights_only=False)
    except TypeError:
      ckpt = torch.load(ckpt_path, map_location=map_location)
    state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    enc_state = {
        k: v for k, v in state.items() if k.startswith("in_conv.") or k.startswith("down_blocks.")
    }
    ltae_state = {
        k[len("temporal_encoder.") :]: v
        for k, v in state.items()
        if k.startswith("temporal_encoder.")
    }
    if not enc_state or not ltae_state:
      raise RuntimeError(f"S2 checkpoint has no encoder/L-TAE weights: {ckpt_path}")
    self.encoder_s2.load_state_dict(enc_state, strict=True)
    self.temporal_s2.load_state_dict(ltae_state, strict=True)
    print(f"loaded S2 encoder+L-TAE from {ckpt_path} keys={len(enc_state)}+{len(ltae_state)}")

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
      raise RuntimeError(
          f"S1 ViT load mismatch missing={missing} unexpected={unexpected}"
      )
    print(f"loaded S1 ViT from {ckpt_path}")

  def train(self, mode: bool = True):
    super().train(mode)
    if mode:
      for mod in self._frozen_eval:
        mod.eval()
    return self

  def set_train_mode(self, mode: str) -> None:
    """P4 head: freeze loaded S2 encoder, S2 L-TAE, and CMU ViT.
    S1 L-TAE has no checkpoint, so it trains with the adapter, fusion, and decoder.
    P5 full: train all.
    """
    if mode == "full":
      for p in self.parameters():
        p.requires_grad = True
      self._frozen_eval = []
      return
    if mode != "head":
      raise ValueError(f"unknown train mode {mode}")
    freeze = [self.encoder_s2, self.s1_vit, self.temporal_s2, self.temporal_aggregator]
    for mod in freeze:
      for p in mod.parameters():
        p.requires_grad = False
    train = [self.temporal_s1, self.s1_adapter, self.fuse_bottleneck, self.up_blocks, self.out_conv]
    for mod in train:
      for p in mod.parameters():
        p.requires_grad = True
    self._frozen_eval = freeze
