#!/usr/bin/env python3
"""
diffit_image_naa.py — Image-space DiffiT with Noise-Aware Attention (NAA).

Instead of injecting time via learned token projections (TMSA), NAA directly
conditions the attention mechanism on the raw noise level σ:

  1. A noise gate g(σ) ∈ (0,1) is computed from σ via a small MLP.
     At high σ (coarse denoising) → g→1 → broad attention (low temperature).
     At low  σ (fine detail)      → g→0 → sharp attention (high temperature).

  2. The attention temperature is modulated:
        scaled_logits = QK^T / (sqrt(d) * (1 + g(σ)))
     High σ → denominator large → flatter softmax → broader attention.
     Low  σ → denominator ~1    → peaked softmax  → sharper attention.

  3. A per-head noise bias b(σ) is added to the logits, learned per head,
     allowing each head to independently decide how to respond to noise level.

  4. QKV still receive spatial + temporal signal via the time embedding,
     but the *attention pattern itself* is now explicitly noise-aware.

This is complementary to TMSA — TMSA adapts what the model looks at (QKV),
NAA adapts *how broadly* it looks (the attention distribution shape).

Architecture changes vs diffit_image.py:
  - TMSA           → NoiseAwareAttention (NAA)
  - NoiseLevelMLP  : new small module, σ → gate g(σ) and bias b(σ)
  - Everything else unchanged (ResBlock, UNet, EDM preconditioning)
"""

import math
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import trunc_normal_


# ---------------------------------------------------------------------------
# Noise Level MLP — σ → (gate, per-head bias)
# ---------------------------------------------------------------------------

class NoiseLevelMLP(nn.Module):
    """
    Maps raw noise level σ → noise gate g(σ) and per-head attention bias b(σ).

    g(σ) ∈ (0, 1):
        Sigmoid output — controls attention temperature modulation.
        High σ → g→1 → broader attention.
        Low  σ → g→0 → sharper attention.

    b(σ) ∈ R^num_heads:
        Additive bias added to attention logits per head.
        Allows each head to independently shift its attention distribution
        based on the noise level.
    """

    def __init__(self, num_heads: int, hidden_dim: int = 64):
        super().__init__()
        self.num_heads = num_heads

        # Shared trunk: log(σ) → hidden
        self.trunk = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
        )

        # Gate head: hidden → scalar g(σ) ∈ (0,1)
        self.gate_head = nn.Sequential(
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )

        # Bias head: hidden → (num_heads,) additive logit bias
        self.bias_head = nn.Linear(hidden_dim, num_heads)

        self._init_weights()

    def _init_weights(self):
        # Init bias head to zero so training starts from unbiased attention
        nn.init.zeros_(self.bias_head.weight)
        nn.init.zeros_(self.bias_head.bias)

    def forward(self, sigma: torch.Tensor):
        """
        sigma : (B,)  raw noise level
        Returns:
            gate : (B, 1, 1, 1)   for temperature scaling
            bias : (B, H, 1, 1)   additive logit bias per head
        """
        # Use log(σ) as input — spans a more uniform range than raw σ
        log_sigma = sigma.clamp(min=1e-6).log().unsqueeze(-1)  # (B, 1)

        h = self.trunk(log_sigma)                               # (B, hidden)
        gate = self.gate_head(h)                                # (B, 1)
        bias = self.bias_head(h)                                # (B, num_heads)

        gate = gate.view(-1, 1, 1, 1)                          # broadcast over H,N,N
        bias = bias.view(-1, self.num_heads, 1, 1)             # broadcast over N,N

        return gate, bias


# ---------------------------------------------------------------------------
# Noise-Aware Attention (NAA)
# ---------------------------------------------------------------------------

class NoiseAwareAttention(nn.Module):
    """
    Window-based Noise-Aware Multi-head Self-Attention.

    Replaces TMSA. Key differences:
      - QKV are still modulated by the time embedding (same as TMSA)
        to preserve temporal signal in feature space.
      - The attention *distribution* is additionally conditioned on σ directly:
          * Temperature scaling: divide logits by (1 + g(σ))
          * Per-head bias:       add b(σ) to logits before softmax
      - g(σ) and b(σ) come from NoiseLevelMLP.
    """

    def __init__(
        self,
        dim: int,
        temb_dim: int,
        num_heads: int,
        window_size: int,
        noise_mlp_hidden: int = 64,
        qkv_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ):
        super().__init__()
        self.num_heads   = num_heads
        self.head_dim    = dim // num_heads
        self.scale       = self.head_dim ** -0.5
        self.window_size = window_size

        # QKV projections (spatial + temporal, same as TMSA)
        self.qkv   = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.qkv_t = nn.Linear(temb_dim, dim * 3, bias=False)

        # Noise-level conditioner — new module
        self.noise_mlp = NoiseLevelMLP(num_heads, hidden_dim=noise_mlp_hidden)

        self.proj      = nn.Linear(dim, dim)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj_drop = nn.Dropout(proj_drop)
        self.softmax   = nn.Softmax(dim=-1)

        # Relative position bias (unchanged)
        self.rpb = RelativePositionBias(window_size, num_heads)

    def forward(
        self,
        x: torch.Tensor,       # (B, N, C)
        temb: torch.Tensor,    # (B, temb_dim)
        sigma: torch.Tensor,   # (B,) raw noise level
    ) -> torch.Tensor:
        B, N, C = x.shape
        H = self.num_heads
        D = self.head_dim

        # ── QKV from spatial + temporal (same as TMSA) ───────────────────
        qkv_s = self.qkv(x)
        qkv_t = self.qkv_t(temb).unsqueeze(1)
        qkv = (qkv_s + qkv_t).reshape(B, N, 3, H, D).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)                        # (B, H, N, D)

        # ── Raw attention logits ─────────────────────────────────────────
        attn = (q * self.scale) @ k.transpose(-2, -1)  # (B, H, N, N)
        attn = attn + self.rpb()                        # relative pos bias

        # ── Noise-aware modulation ───────────────────────────────────────
        gate, noise_bias = self.noise_mlp(sigma)        # (B,1,1,1), (B,H,1,1)

        # Temperature scaling: high σ → gate→1 → divide by up to 2 → flatter
        attn = attn / (1.0 + gate)

        # Per-head additive bias: shift attention distribution per head
        attn = attn + noise_bias

        # ── Softmax + output ─────────────────────────────────────────────
        attn = self.attn_drop(self.softmax(attn))
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        return self.proj_drop(self.proj(x))


# ---------------------------------------------------------------------------
# Helpers (unchanged)
# ---------------------------------------------------------------------------

class SinusoidalTimestep(nn.Module):
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
    def __init__(self, in_dim: int, hidden_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


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
# DiffiT Transformer block — now passes sigma through to NAA
# ---------------------------------------------------------------------------

class DiffiTTransformerBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        temb_dim: int,
        num_heads: int,
        window_size: int,
        mlp_ratio: float = 4.0,
        noise_mlp_hidden: int = 64,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.naa   = NoiseAwareAttention(   # ← NAA replaces TMSA
            dim, temb_dim, num_heads, window_size,
            noise_mlp_hidden=noise_mlp_hidden,
        )
        self.norm2 = nn.LayerNorm(dim)
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, dim),
        )

    def forward(
        self,
        x: torch.Tensor,
        temb: torch.Tensor,
        sigma: torch.Tensor,   # ← sigma now threaded through
    ) -> torch.Tensor:
        x = x + self.naa(self.norm1(x), temb, sigma)
        x = x + self.mlp(self.norm2(x))
        return x


# ---------------------------------------------------------------------------
# DiffiT ResBlock — threads sigma through to transformer block
# ---------------------------------------------------------------------------

class DiffiTResBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        temb_dim: int,
        num_heads: int,
        window_size: int,
        mlp_ratio: float = 4.0,
        groups: int = 8,
        noise_mlp_hidden: int = 64,
    ):
        super().__init__()
        self.window_size = window_size

        self.gn   = nn.GroupNorm(groups, in_channels)
        self.conv = nn.Conv2d(in_channels, out_channels, 3, padding=1)

        self.transformer = DiffiTTransformerBlock(
            dim=out_channels,
            temb_dim=temb_dim,
            num_heads=num_heads,
            window_size=window_size,
            mlp_ratio=mlp_ratio,
            noise_mlp_hidden=noise_mlp_hidden,
        )

        self.skip = (
            nn.Conv2d(in_channels, out_channels, 1, bias=False)
            if in_channels != out_channels else nn.Identity()
        )

    def forward(
        self,
        x: torch.Tensor,
        temb: torch.Tensor,
        sigma: torch.Tensor,   # ← raw σ passed directly
    ) -> torch.Tensor:
        h = self.conv(F.silu(self.gn(x)))

        B, C, H, W = h.shape
        ws = self.window_size
        assert H % ws == 0 and W % ws == 0

        h_win = h.reshape(B, C, H // ws, ws, W // ws, ws)
        h_win = h_win.permute(0, 2, 4, 3, 5, 1).contiguous()
        nH, nW = H // ws, W // ws
        h_win = h_win.reshape(B * nH * nW, ws * ws, C)

        # Repeat temb and sigma for each window
        temb_rep  = temb.unsqueeze(1).expand(B, nH * nW, -1).reshape(B * nH * nW, -1)
        sigma_rep = sigma.unsqueeze(1).expand(B, nH * nW).reshape(B * nH * nW)

        h_win = self.transformer(h_win, temb_rep, sigma_rep)

        h = h_win.reshape(B, nH, nW, ws, ws, C)
        h = h.permute(0, 5, 1, 3, 2, 4).contiguous()
        h = h.reshape(B, C, H, W)

        return h + self.skip(x)


# ---------------------------------------------------------------------------
# Image-space DiffiT U-Net
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
        noise_mlp_hidden: int = 64,
    ):
        super().__init__()
        assert len(channel_mult) == len(num_blocks)
        self.num_stages = len(channel_mult)
        channels = [base_channels * m for m in channel_mult]

        self.temb_proj = nn.Sequential(
            SinusoidalTimestep(temb_dim),
            TimestepMLP(temb_dim, temb_dim),
        )

        self.tokenizer = nn.Conv2d(in_channels, channels[0], 3, padding=1)

        # ---- Encoder -------------------------------------------------------
        self.enc_blocks: nn.ModuleList = nn.ModuleList()
        self.enc_downs:  nn.ModuleList = nn.ModuleList()

        for i in range(self.num_stages):
            c = channels[i]
            stage = nn.ModuleList([
                DiffiTResBlock(c, c, temb_dim, num_heads, window_size,
                               mlp_ratio, noise_mlp_hidden=noise_mlp_hidden)
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
                    DiffiTResBlock(in_c, c, temb_dim, num_heads, window_size,
                                   mlp_ratio, noise_mlp_hidden=noise_mlp_hidden)
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

    def forward(
        self,
        x: torch.Tensor,
        temb_input: torch.Tensor,   # (B,)  log(σ)/4  from EDM precond
        sigma: torch.Tensor,        # (B,)  raw σ      passed separately
    ) -> torch.Tensor:
        temb = self.temb_proj(temb_input)
        h = self.tokenizer(x)

        skips: List[torch.Tensor] = []
        for i in range(self.num_stages):
            for blk in self.enc_blocks[i]:
                h = blk(h, temb, sigma)
            skips.append(h)
            if i < self.num_stages - 1:
                h = self.enc_downs[i](h)

        up_idx = 0
        for j, stage_blocks in enumerate(self.dec_blocks):
            stage_idx = self.num_stages - 1 - j
            if j > 0:
                h = torch.cat([h, skips[stage_idx]], dim=1)
            for blk in stage_blocks:
                h = blk(h, temb, sigma)
            if stage_idx > 0:
                h = self.dec_ups[up_idx](h)
                up_idx += 1

        return self.head(h)


# ---------------------------------------------------------------------------
# EDM Preconditioning — now passes raw sigma to inner model as well
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

        # Pass both c_noise (for time embedding) and raw sigma (for NAA)
        F_out = self.inner_model(c_in * y, c_noise, sigma)
        return c_skip * y + c_out * F_out

    @torch.no_grad()
    def round_sigma(self, sigma):
        return torch.as_tensor(sigma)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def build_cifar10_model(sigma_data: float = 0.5) -> EDMPrecond:
    """
    CIFAR-10 image-space DiffiT with Noise-Aware Attention (NAA).
    TMSA replaced by NAA which directly conditions attention on raw σ.
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
        noise_mlp_hidden=64,
    )
    return EDMPrecond(unet, sigma_data=sigma_data)