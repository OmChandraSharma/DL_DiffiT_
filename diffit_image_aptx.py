#!/usr/bin/env python3
"""
diffit_image_aptx.py — Image-space DiffiT model for CIFAR-10,
                        with SiLU replaced by APTx activation function.

APTx (Alpha Plus Tanh Times) from:
  "APTx: better activation function than MISH, SWISH, and ReLU's variants"
  Ravin Kumar (2022).  https://arxiv.org/abs/2209.06119

APTx formula (α=1, β=1, γ=½):
    ψ(x) = (1 + tanh(x)) * x / 2

Derivative:
    ψ'(x) = (1 + tanh(x) + x * sech²(x)) / 2

Replaces SiLU in:
  - DiffiTResBlock conv branch  (Eq. 9 of DiffiT paper)
  - TimestepMLP hidden activation

GELU (transformer MLP) and Softmax (attention) are unchanged —
they are standard transformer components not targeted by this change.
"""

import math
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import trunc_normal_


# ---------------------------------------------------------------------------
# APTx Activation  (α=1, β=1, γ=½)
# ψ(x) = (1 + tanh(x)) * x / 2
# ---------------------------------------------------------------------------

class APTx(nn.Module):
    """
    APTx activation function (Kumar 2022).

    Default parameters α=1, β=1, γ=½ make it behave similar to MISH
    while requiring fewer mathematical operations in both forward and
    backward passes.

    Can optionally be used with learnable parameters by setting
    learnable=True, which initialises α, β, γ as nn.Parameters.
    """

    def __init__(
        self,
        alpha: float = 1.0,
        beta:  float = 1.0,
        gamma: float = 0.5,
        learnable: bool = False,
    ):
        super().__init__()
        if learnable:
            self.alpha = nn.Parameter(torch.tensor(alpha))
            self.beta  = nn.Parameter(torch.tensor(beta))
            self.gamma = nn.Parameter(torch.tensor(gamma))
        else:
            self.register_buffer("alpha", torch.tensor(alpha))
            self.register_buffer("beta",  torch.tensor(beta))
            self.register_buffer("gamma", torch.tensor(gamma))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # ψ(x) = (α + tanh(β·x)) * γ·x
        return (self.alpha + torch.tanh(self.beta * x)) * (self.gamma * x)

    def extra_repr(self) -> str:
        return (f"alpha={self.alpha.item():.3f}, "
                f"beta={self.beta.item():.3f}, "
                f"gamma={self.gamma.item():.3f}")


def aptx(x: torch.Tensor,
         alpha: float = 1.0,
         beta:  float = 1.0,
         gamma: float = 0.5) -> torch.Tensor:
    """Functional form of APTx for use inside forward() calls."""
    return (alpha + torch.tanh(beta * x)) * (gamma * x)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class SinusoidalTimestep(nn.Module):
    """
    Maps scalar noise level σ → (B, embed_dim) sinusoidal embedding.
    Follows EDM convention: embed log(σ)/4.
    """
    def __init__(self, embed_dim: int):
        super().__init__()
        assert embed_dim % 2 == 0
        half = embed_dim // 2
        freqs = torch.exp(
            -math.log(10000) * torch.arange(half, dtype=torch.float32) / half
        )
        self.register_buffer("freqs", freqs)

    def forward(self, log_sigma: torch.Tensor) -> torch.Tensor:
        x = log_sigma[:, None] * self.freqs[None]
        return torch.cat([torch.sin(x), torch.cos(x)], dim=-1)


class TimestepMLP(nn.Module):
    """
    Small 2-layer MLP with APTx activation to produce time token.
    Original used SiLU (=Swish); replaced with APTx per this ablation.
    """
    def __init__(self, in_dim: int, hidden_dim: int):
        super().__init__()
        self.fc1  = nn.Linear(in_dim, hidden_dim)
        self.act  = APTx()          # ← APTx replaces SiLU here
        self.fc2  = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.act(self.fc1(x)))


# ---------------------------------------------------------------------------
# Relative-position bias (Swin-style)
# ---------------------------------------------------------------------------

class RelativePositionBias(nn.Module):
    def __init__(self, window_size: int, num_heads: int):
        super().__init__()
        ws = window_size
        self.num_heads = num_heads
        self.table = nn.Parameter(
            torch.zeros((2 * ws - 1) * (2 * ws - 1), num_heads)
        )
        trunc_normal_(self.table, std=0.02)

        coords = torch.stack(
            torch.meshgrid(torch.arange(ws), torch.arange(ws), indexing="ij")
        )
        flat = coords.flatten(1)
        rel  = flat[:, :, None] - flat[:, None, :]
        rel  = rel.permute(1, 2, 0).contiguous()
        rel[:, :, 0] += ws - 1
        rel[:, :, 1] += ws - 1
        rel[:, :, 0] *= 2 * ws - 1
        self.register_buffer("index", rel.sum(-1))

    def forward(self) -> torch.Tensor:
        N = self.index.shape[0]
        bias = self.table[self.index.view(-1)].view(N, N, self.num_heads)
        return bias.permute(2, 0, 1).contiguous()


# ---------------------------------------------------------------------------
# TMSA — Time-dependent Multi-head Self-Attention (window-based)
# Softmax unchanged — standard attention component
# ---------------------------------------------------------------------------

class TMSA(nn.Module):
    def __init__(
        self,
        dim: int,
        temb_dim: int,
        num_heads: int,
        window_size: int,
        qkv_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ):
        super().__init__()
        self.num_heads   = num_heads
        self.head_dim    = dim // num_heads
        self.scale       = self.head_dim ** -0.5
        self.window_size = window_size

        self.qkv   = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.qkv_t = nn.Linear(temb_dim, dim * 3, bias=False)

        self.proj      = nn.Linear(dim, dim)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj_drop = nn.Dropout(proj_drop)
        self.softmax   = nn.Softmax(dim=-1)   # unchanged

        self.rpb = RelativePositionBias(window_size, num_heads)

    def forward(self, x: torch.Tensor, temb: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        H = self.num_heads
        D = self.head_dim

        qkv_s = self.qkv(x)
        qkv_t = self.qkv_t(temb).unsqueeze(1)

        qkv = (qkv_s + qkv_t).reshape(B, N, 3, H, D).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)

        attn = (q * self.scale) @ k.transpose(-2, -1)
        attn = attn + self.rpb()
        attn = self.attn_drop(self.softmax(attn))

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        return self.proj_drop(self.proj(x))


# ---------------------------------------------------------------------------
# DiffiT Transformer block  (Eq. 7-8)
# GELU in MLP unchanged — standard transformer component
# ---------------------------------------------------------------------------

class DiffiTTransformerBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        temb_dim: int,
        num_heads: int,
        window_size: int,
        mlp_ratio: float = 4.0,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.tmsa  = TMSA(dim, temb_dim, num_heads, window_size)
        self.norm2 = nn.LayerNorm(dim)
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),              # unchanged
            nn.Linear(hidden, dim),
        )

    def forward(self, x: torch.Tensor, temb: torch.Tensor) -> torch.Tensor:
        x = x + self.tmsa(self.norm1(x), temb)
        x = x + self.mlp(self.norm2(x))
        return x


# ---------------------------------------------------------------------------
# DiffiT ResBlock  (Eq. 9-10)
# APTx replaces SiLU in the conv branch
# ---------------------------------------------------------------------------

class DiffiTResBlock(nn.Module):
    """
    Conv branch: GN → APTx → Conv3×3  (Eq. 9, SiLU replaced by APTx)
    Then: DiffiT Transformer block + residual  (Eq. 10)
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        temb_dim: int,
        num_heads: int,
        window_size: int,
        mlp_ratio: float = 4.0,
        groups: int = 8,
    ):
        super().__init__()
        self.window_size = window_size

        self.gn   = nn.GroupNorm(groups, in_channels)
        self.act  = APTx()          # ← APTx replaces F.silu here
        self.conv = nn.Conv2d(in_channels, out_channels, 3, padding=1)

        self.transformer = DiffiTTransformerBlock(
            dim=out_channels,
            temb_dim=temb_dim,
            num_heads=num_heads,
            window_size=window_size,
            mlp_ratio=mlp_ratio,
        )

        self.skip = (
            nn.Conv2d(in_channels, out_channels, 1, bias=False)
            if in_channels != out_channels else nn.Identity()
        )

    def forward(self, x: torch.Tensor, temb: torch.Tensor) -> torch.Tensor:
        h = self.conv(self.act(self.gn(x)))              # APTx instead of SiLU

        B, C, H, W = h.shape
        ws = self.window_size
        assert H % ws == 0 and W % ws == 0

        h_win = h.reshape(B, C, H // ws, ws, W // ws, ws)
        h_win = h_win.permute(0, 2, 4, 3, 5, 1).contiguous()
        nH, nW = H // ws, W // ws
        h_win = h_win.reshape(B * nH * nW, ws * ws, C)

        temb_rep = temb.unsqueeze(1).expand(B, nH * nW, -1).reshape(B * nH * nW, -1)
        h_win = self.transformer(h_win, temb_rep)

        h = h_win.reshape(B, nH, nW, ws, ws, C)
        h = h.permute(0, 5, 1, 3, 2, 4).contiguous()
        h = h.reshape(B, C, H, W)

        return h + self.skip(x)


# ---------------------------------------------------------------------------
# Image-space DiffiT U-Net  (Fig. 4 / Appendix H.1)
# ---------------------------------------------------------------------------

class DiffiTImageUNet(nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        base_channels: int = 128,
        channel_mult: tuple = (1, 2, 2),
        num_blocks: tuple = (2, 2, 2),
        num_heads: int = 4,
        window_size: int = 4,
        temb_dim: int = 512,
        mlp_ratio: float = 4.0,
    ):
        super().__init__()
        assert len(channel_mult) == len(num_blocks)
        self.num_stages = len(channel_mult)
        channels = [base_channels * m for m in channel_mult]

        # Time embedding: sinusoidal → APTx MLP
        self.temb_proj = nn.Sequential(
            SinusoidalTimestep(temb_dim),
            TimestepMLP(temb_dim, temb_dim),   # APTx inside
        )

        self.tokenizer = nn.Conv2d(in_channels, channels[0], 3, padding=1)

        # ---- Encoder -------------------------------------------------------
        self.enc_blocks: nn.ModuleList = nn.ModuleList()
        self.enc_downs:  nn.ModuleList = nn.ModuleList()

        for i in range(self.num_stages):
            c_in = channels[i]
            stage = nn.ModuleList([
                DiffiTResBlock(c_in, c_in, temb_dim, num_heads, window_size, mlp_ratio)
                for _ in range(num_blocks[i])
            ])
            self.enc_blocks.append(stage)
            if i < self.num_stages - 1:
                self.enc_downs.append(
                    nn.Conv2d(channels[i], channels[i + 1], 3, stride=2, padding=1)
                )

        # ---- Decoder -------------------------------------------------------
        self.dec_blocks: nn.ModuleList = nn.ModuleList()
        self.dec_ups:    nn.ModuleList = nn.ModuleList()

        for i in reversed(range(self.num_stages)):
            c = channels[i]
            fuse_ch = c * 2 if i < self.num_stages - 1 else c

            stage_blocks = nn.ModuleList()
            for j in range(num_blocks[i]):
                in_c = fuse_ch if j == 0 and i < self.num_stages - 1 else c
                stage_blocks.append(
                    DiffiTResBlock(in_c, c, temb_dim, num_heads, window_size, mlp_ratio)
                )
            self.dec_blocks.append(stage_blocks)

            if i > 0:
                self.dec_ups.append(
                    nn.ConvTranspose2d(channels[i], channels[i - 1], 4, stride=2, padding=1)
                )

        self.head = nn.Conv2d(channels[0], in_channels, 3, padding=1)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)):
                nn.init.kaiming_normal_(m.weight, nonlinearity="linear")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.GroupNorm, nn.LayerNorm)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, x: torch.Tensor, temb_input: torch.Tensor) -> torch.Tensor:
        temb = self.temb_proj(temb_input)
        h = self.tokenizer(x)

        skips: List[torch.Tensor] = []
        for i in range(self.num_stages):
            for blk in self.enc_blocks[i]:
                h = blk(h, temb)
            skips.append(h)
            if i < self.num_stages - 1:
                h = self.enc_downs[i](h)

        up_idx = 0
        for j, stage_blocks in enumerate(self.dec_blocks):
            stage_idx = self.num_stages - 1 - j
            if j > 0:
                h = torch.cat([h, skips[stage_idx]], dim=1)
            for blk in stage_blocks:
                h = blk(h, temb)
            if stage_idx > 0:
                h = self.dec_ups[up_idx](h)
                up_idx += 1

        return self.head(h)


# ---------------------------------------------------------------------------
# EDM Preconditioning  (Karras et al. 2022, Algorithm 2)
# ---------------------------------------------------------------------------

class EDMPrecond(nn.Module):
    def __init__(self, model: DiffiTImageUNet, sigma_data: float = 0.5):
        super().__init__()
        self.inner_model = model
        self.sigma_data  = sigma_data

    def forward(self, y: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
        s = sigma.view(-1, 1, 1, 1)
        c_skip  = self.sigma_data ** 2 / (s ** 2 + self.sigma_data ** 2)
        c_out   = s * self.sigma_data / (s ** 2 + self.sigma_data ** 2).sqrt()
        c_in    = 1.0 / (s ** 2 + self.sigma_data ** 2).sqrt()
        c_noise = sigma.log() / 4.0
        F_out = self.inner_model(c_in * y, c_noise)
        return c_skip * y + c_out * F_out

    @torch.no_grad()
    def round_sigma(self, sigma):
        return torch.as_tensor(sigma)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def build_cifar10_model(sigma_data: float = 0.5) -> EDMPrecond:
    """
    CIFAR-10 image-space DiffiT with APTx activation (paper Appendix H.1 + I.1).
    Identical architecture to original except SiLU → APTx in:
      - DiffiTResBlock conv branch
      - TimestepMLP hidden layer
    """
    unet = DiffiTImageUNet(
        in_channels=3,
        base_channels=128,
        channel_mult=(1, 2, 2),
        num_blocks=(2, 2, 2),
        num_heads=4,
        window_size=4,
        temb_dim=512,
        mlp_ratio=4.0,
    )
    return EDMPrecond(unet, sigma_data=sigma_data)