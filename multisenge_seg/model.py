"""ConvLSTM + naive Inception → VGG-16 U-Net (Wenger et al. RS 2023).

Hyperparams: ConvLSTM 3×3 / 32 filters; naive Inception 1×1/3×3/5×5 + pool;
U-Net encoder follows VGG-16 block depths (64-128-256-512-512), trained from
scratch on fused features (not ImageNet RGB weights).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvLSTMCell(nn.Module):
    def __init__(self, in_ch: int, hidden_ch: int, kernel_size: int = 3) -> None:
        super().__init__()
        padding = kernel_size // 2
        self.hidden_ch = hidden_ch
        self.gates = nn.Conv2d(in_ch + hidden_ch, 4 * hidden_ch, kernel_size, padding=padding)

    def forward(self, x, state):
        h, c = state
        gates = self.gates(torch.cat([x, h], dim=1))
        i, f, o, g = torch.chunk(gates, 4, dim=1)
        i, f, o, g = torch.sigmoid(i), torch.sigmoid(f), torch.sigmoid(o), torch.tanh(g)
        c = f * c + i * g
        h = o * torch.tanh(c)
        return h, c


class ConvLSTMEncoder(nn.Module):
    """Encode a length-T sequence (B,T,C,H,W) → (B, hidden, H, W)."""

    def __init__(self, in_ch: int, hidden_ch: int = 32, kernel_size: int = 3) -> None:
        super().__init__()
        self.cell = ConvLSTMCell(in_ch, hidden_ch, kernel_size)
        self.hidden_ch = hidden_ch

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, t, _, h, w = x.shape
        h_t = x.new_zeros(b, self.hidden_ch, h, w)
        c_t = x.new_zeros(b, self.hidden_ch, h, w)
        for ti in range(t):
            h_t, c_t = self.cell(x[:, ti], (h_t, c_t))
        return h_t


class NaiveInception(nn.Module):
    """1x1 / 3x3 / 5x5 + max-pool branch, concat (paper-style naive module)."""

    def __init__(self, in_ch: int, branch_ch: int = 32) -> None:
        super().__init__()
        self.b1 = nn.Sequential(nn.Conv2d(in_ch, branch_ch, 1), nn.ReLU(inplace=True))
        self.b3 = nn.Sequential(nn.Conv2d(in_ch, branch_ch, 3, padding=1), nn.ReLU(inplace=True))
        self.b5 = nn.Sequential(nn.Conv2d(in_ch, branch_ch, 5, padding=2), nn.ReLU(inplace=True))
        self.pool = nn.Sequential(
            nn.MaxPool2d(3, stride=1, padding=1),
            nn.Conv2d(in_ch, branch_ch, 1),
            nn.ReLU(inplace=True),
        )
        self.out_ch = branch_ch * 4

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.cat([self.b1(x), self.b3(x), self.b5(x), self.pool(x)], dim=1)


def _vgg_block(in_ch: int, out_ch: int, n_convs: int) -> nn.Sequential:
    layers: list[nn.Module] = []
    c = in_ch
    for _ in range(n_convs):
        layers += [nn.Conv2d(c, out_ch, 3, padding=1), nn.ReLU(inplace=True)]
        c = out_ch
    return nn.Sequential(*layers)


class VGG16UNet(nn.Module):
    """U-Net with VGG-16-style encoder (paper backbone), arbitrary in_ch.

    Encoder depths match VGG-16: 2 / 2 / 3 / 3 / 3 convs at 64 / 128 / 256 / 512 / 512.
    Trained from scratch — ImageNet RGB weights do not apply to fused feature maps.
    """

    def __init__(self, in_ch: int, num_classes: int) -> None:
        super().__init__()
        self.enc1 = _vgg_block(in_ch, 64, 2)   # /1
        self.pool1 = nn.MaxPool2d(2)
        self.enc2 = _vgg_block(64, 128, 2)      # /2
        self.pool2 = nn.MaxPool2d(2)
        self.enc3 = _vgg_block(128, 256, 3)     # /4
        self.pool3 = nn.MaxPool2d(2)
        self.enc4 = _vgg_block(256, 512, 3)     # /8
        self.pool4 = nn.MaxPool2d(2)
        self.enc5 = _vgg_block(512, 512, 3)     # /16

        self.up4 = nn.ConvTranspose2d(512, 512, 2, stride=2)
        self.dec4 = _vgg_block(512 + 512, 512, 2)
        self.up3 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.dec3 = _vgg_block(256 + 256, 256, 2)
        self.up2 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec2 = _vgg_block(128 + 128, 128, 2)
        self.up1 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec1 = _vgg_block(64 + 64, 64, 2)
        self.head = nn.Conv2d(64, num_classes, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        e3 = self.enc3(self.pool2(e2))
        e4 = self.enc4(self.pool3(e3))
        e5 = self.enc5(self.pool4(e4))

        d4 = self.up4(e5)
        d4 = self._match(d4, e4)
        d4 = self.dec4(torch.cat([d4, e4], dim=1))
        d3 = self.up3(d4)
        d3 = self._match(d3, e3)
        d3 = self.dec3(torch.cat([d3, e3], dim=1))
        d2 = self.up2(d3)
        d2 = self._match(d2, e2)
        d2 = self.dec2(torch.cat([d2, e2], dim=1))
        d1 = self.up1(d2)
        d1 = self._match(d1, e1)
        d1 = self.dec1(torch.cat([d1, e1], dim=1))
        return self.head(d1)

    @staticmethod
    def _match(up: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        if up.shape[-2:] != skip.shape[-2:]:
            up = F.interpolate(up, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        return up


class ConvLSTMInceptionS1S2(nn.Module):
    def __init__(
        self,
        s1_ch: int = 2,
        s2_ch: int = 10,
        hidden: int = 32,
        num_classes: int = 6,
    ) -> None:
        super().__init__()
        self.s1_lstm = ConvLSTMEncoder(s1_ch, hidden)
        self.s2_lstm = ConvLSTMEncoder(s2_ch, hidden)
        self.inception = NaiveInception(s1_ch + s2_ch, branch_ch=hidden)
        fuse_ch = hidden + hidden + self.inception.out_ch
        self.unet = VGG16UNet(fuse_ch, num_classes)
        self.num_classes = num_classes

    def forward(self, s1: torch.Tensor, s2: torch.Tensor) -> torch.Tensor:
        """
        s1, s2: (B, T, C, H, W)
        returns logits (B, num_classes, H, W) for classes mapped as 0..K-1 at train time
        """
        f1 = self.s1_lstm(s1)
        f2 = self.s2_lstm(s2)
        x0 = torch.cat([s1[:, 0], s2[:, 0]], dim=1)
        fi = self.inception(x0)
        fused = torch.cat([f1, f2, fi], dim=1)
        return self.unet(fused)


def smoke_forward(device: str = "cpu") -> tuple[torch.Tensor, tuple[int, ...]]:
    """Random tensors forward — verifies shapes without rasters."""
    model = ConvLSTMInceptionS1S2(s1_ch=2, s2_ch=10, num_classes=6).to(device)
    s1 = torch.randn(1, 4, 2, 64, 64, device=device)
    s2 = torch.randn(1, 4, 10, 64, 64, device=device)
    logits = model(s1, s2)
    return logits, tuple(logits.shape)
