"""MA-UTAE: dual-stream modality-aware U-TAE (novelty Task M).

S2 (10ch) and S1 (2ch) get separate encoders + L-TAEs, fuse, then one decoder.
Fusion: concat+1x1 or gated (default for publishable claim).

Training phases reuse breast-style names: P4=head, P5=full (see train_ma.py).
Do not call this "P1" — that clashes with U-TAE P3/P4/P5.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from multisenge_utae.heads import HierarchicalHeads
from multisenge_utae.models.fusion import build_fusion
from multisenge_utae.models.ltae import LTAE2d
from multisenge_utae.models.utae import (
    ConvBlock,
    DownConvBlock,
    TemporalAggregator,
    UpConvBlock,
)


class _ModalityEncoder(nn.Module):
  def __init__(
      self,
      input_dim: int,
      encoder_widths: list[int],
      str_conv_k: int,
      str_conv_s: int,
      str_conv_p: int,
      pad_value: float,
      encoder_norm: str,
      padding_mode: str,
  ):
    super().__init__()
    self.n_stages = len(encoder_widths)
    self.in_conv = ConvBlock(
        nkernels=[input_dim, encoder_widths[0], encoder_widths[0]],
        pad_value=pad_value,
        norm=encoder_norm,
        padding_mode=padding_mode,
    )
    self.down_blocks = nn.ModuleList(
        DownConvBlock(
            d_in=encoder_widths[i],
            d_out=encoder_widths[i + 1],
            k=str_conv_k,
            s=str_conv_s,
            p=str_conv_p,
            pad_value=pad_value,
            norm=encoder_norm,
            padding_mode=padding_mode,
        )
        for i in range(self.n_stages - 1)
    )

  def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
    out = self.in_conv.smart_forward(x)
    maps = [out]
    for i in range(self.n_stages - 1):
      out = self.down_blocks[i].smart_forward(maps[-1])
      maps.append(out)
    return maps


class MAUTAE(nn.Module):
  """Dual-stream U-TAE with late bottleneck (+ skip) fusion."""

  def __init__(
      self,
      num_classes: int,
      s2_dim: int = 10,
      s1_dim: int = 2,
      fusion: str = "gated",
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
      fuse_skips: bool = True,
      use_a1: bool = False,
      use_a2: bool = False,
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
    self.fuse_skips = fuse_skips
    self.use_a1 = use_a1
    self.use_a2 = use_a2

    enc_kwargs = dict(
        encoder_widths=encoder_widths,
        str_conv_k=str_conv_k,
        str_conv_s=str_conv_s,
        str_conv_p=str_conv_p,
        pad_value=pad_value,
        encoder_norm=encoder_norm,
        padding_mode=padding_mode,
    )
    self.encoder_s2 = _ModalityEncoder(input_dim=s2_dim, **enc_kwargs)
    self.encoder_s1 = _ModalityEncoder(input_dim=s1_dim, **enc_kwargs)

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
    self.skip_fusions = nn.ModuleList(
        build_fusion(fusion, encoder_widths[i]) for i in range(self.n_stages - 1)
    ) if fuse_skips else None

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
    self.aux_heads = None
    if use_a1 or use_a2:
      self.aux_heads = HierarchicalHeads(decoder_widths[0], use_a1=use_a1, use_a2=use_a2)

  def _split(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """x: B,T,12,H,W with S2 then S1 (see data.stack_modalities)."""
    return x[:, :, : self.s2_dim], x[:, :, self.s2_dim : self.s2_dim + self.s1_dim]

  def _pad_mask(self, input: torch.Tensor) -> torch.Tensor:
    return (input == self.pad_value).all(dim=-1).all(dim=-1).all(dim=-1)

  def forward(self, input: torch.Tensor, batch_positions=None, return_att: bool = False):
    s2, s1 = self._split(input)
    pad_s2 = self._pad_mask(s2)
    pad_s1 = self._pad_mask(s1)

    maps_s2 = self.encoder_s2(s2)
    maps_s1 = self.encoder_s1(s1)

    out_s2, att_s2 = self.temporal_s2(maps_s2[-1], batch_positions=batch_positions, pad_mask=pad_s2)
    out_s1, att_s1 = self.temporal_s1(maps_s1[-1], batch_positions=batch_positions, pad_mask=pad_s1)
    out = self.fuse_bottleneck(out_s2, out_s1)

    for i in range(self.n_stages - 1):
      skip_s2 = self.temporal_aggregator(maps_s2[-(i + 2)], pad_mask=pad_s2, attn_mask=att_s2)
      if self.fuse_skips:
        skip_s1 = self.temporal_aggregator(maps_s1[-(i + 2)], pad_mask=pad_s1, attn_mask=att_s1)
        skip = self.skip_fusions[-(i + 1)](skip_s2, skip_s1)
      else:
        skip = skip_s2
      out = self.up_blocks[i](out, skip)

    aux = self.aux_heads(out) if self.aux_heads is not None else {}
    logits = self.out_conv(out)
    if return_att:
      return logits, aux, {"att_s2": att_s2, "att_s1": att_s1}
    if self.aux_heads is not None:
      return logits, aux
    return logits

  def set_train_mode(self, mode: str) -> None:
    """mode: 'full' | 'head' (freeze dual encoders + L-TAEs; train fusion/decoder/head)."""
    if mode == "full":
      for p in self.parameters():
        p.requires_grad = True
      return
    if mode != "head":
      raise ValueError(f"unknown train mode {mode}")
    freeze = [self.encoder_s2, self.encoder_s1, self.temporal_s2, self.temporal_s1, self.temporal_aggregator]
    for mod in freeze:
      for p in mod.parameters():
        p.requires_grad = False
    train = [self.fuse_bottleneck, self.up_blocks, self.out_conv]
    if self.skip_fusions is not None:
      train.append(self.skip_fusions)
    if self.aux_heads is not None:
      train.append(self.aux_heads)
    for mod in train:
      for p in mod.parameters():
        p.requires_grad = True
