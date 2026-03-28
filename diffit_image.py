#!/usr/bin/env python3
"""
diffit_image.py — Image-space DiffiT model for CIFAR-10.

Implements the U-Net–style encoder-decoder architecture described in
Section 3 / Fig. 4 and Appendix H.1 / I.1 of:

  "DiffiT: Diffusion Vision Transformers for Image Generation"
  Hatamizadeh et al., NVIDIA (2023).

Paper specs for CIFAR-10 (32×32, unconditional, image-space):
  - 3 resolution stages: 32 → 16 → 8
  - L1=L2=L3 = 2 DiffiT ResBlocks per stage (encoder & decoder each)
  - Channel dims: 128 → 256 → 256  (only first→second transition changes channels)
  - Window-based TMSA with local window size = 4 at every stage
  - Convolutional tokenizer (in) and head (out)
  - Skip connections between matching encoder/decoder stages
  - EDM-compatible output (no learn_sigma; direct noise prediction)

EDM framework compatibility:
  - Input : (B, 3, 32, 32) noisy image x, scalar σ (noise level)
  - Output: (B, 3, 32, 32) denoised estimate D(x; σ)
  - Preconditioning follows Karras et al. EDM (Algorithm 2)
"""

import math
from typing import Optional, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import trunc_normal_


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
        # log_sigma: (B,)  — pass log(σ)/4 as in EDM
        x = log_sigma[:, None] * self.freqs[None]          # (B, half)
        return torch.cat([torch.sin(x), torch.cos(x)], dim=-1)  # (B, embed_dim)


class TimestepMLP(nn.Module):
    """Small 2-layer MLP with SiLU (= Swish) to produce time token."""
    def __init__(self, in_dim: int, hidden_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ---------------------------------------------------------------------------
# Relative-position bias (Swin-style), window size w×w
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
        )                                               # (2, ws, ws)
        flat = coords.flatten(1)                        # (2, ws*ws)
        rel  = flat[:, :, None] - flat[:, None, :]     # (2, ws*ws, ws*ws)
        rel  = rel.permute(1, 2, 0).contiguous()        # (ws*ws, ws*ws, 2)
        rel[:, :, 0] += ws - 1
        rel[:, :, 1] += ws - 1
        rel[:, :, 0] *= 2 * ws - 1
        self.register_buffer("index", rel.sum(-1))      # (ws*ws, ws*ws)

    def forward(self) -> torch.Tensor:
        """Returns (num_heads, N, N) bias to add before softmax."""
        N = self.index.shape[0]
        bias = self.table[self.index.view(-1)].view(N, N, self.num_heads)
        return bias.permute(2, 0, 1).contiguous()       # (H, N, N)


# ---------------------------------------------------------------------------
# TMSA — Time-dependent Multi-head Self-Attention (window-based)
# Paper eq. 3-6 and Section 3.2
# ---------------------------------------------------------------------------

class TMSA(nn.Module):
    """
    Window-based Time-dependent Multi-head Self-Attention.

    The time token x_t modulates q/k/v via separate linear projections
    (W_qt, W_kt, W_vt) that are added to the spatial projections before
    reshaping into heads. This is the TMSA formulation from the paper.
    """

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
        self.num_heads = num_heads
        self.head_dim  = dim // num_heads
        self.scale     = self.head_dim ** -0.5
        self.window_size = window_size

        # Spatial QKV projection
        self.qkv  = nn.Linear(dim, dim * 3, bias=qkv_bias)
        # Temporal QKV projection (time token → same size)
        self.qkv_t = nn.Linear(temb_dim, dim * 3, bias=False)

        self.proj      = nn.Linear(dim, dim)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj_drop = nn.Dropout(proj_drop)
        self.softmax   = nn.Softmax(dim=-1)

        self.rpb = RelativePositionBias(window_size, num_heads)

    def forward(self, x: torch.Tensor, temb: torch.Tensor) -> torch.Tensor:
        """
        x    : (B, N, C)  spatial tokens (N = window_size**2)
        temb : (B, temb_dim)  time embedding
        """
        B, N, C = x.shape
        H = self.num_heads
        D = self.head_dim

        # q_s, k_s, v_s  from spatial tokens
        qkv_s = self.qkv(x)                              # (B, N, 3C)
        # q_t, k_t, v_t  from time token (broadcast over N)
        qkv_t = self.qkv_t(temb).unsqueeze(1)            # (B, 1, 3C)

        # Add temporal component (Eq. 3-5 in paper)
        qkv = (qkv_s + qkv_t).reshape(B, N, 3, H, D).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)                          # each (B, H, N, D)

        attn = (q * self.scale) @ k.transpose(-2, -1)    # (B, H, N, N)
        attn = attn + self.rpb()                          # relative pos bias
        attn = self.attn_drop(self.softmax(attn))

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        return self.proj_drop(self.proj(x))


# ---------------------------------------------------------------------------
# DiffiT Transformer block  (Eq. 7-8 in paper)
# ---------------------------------------------------------------------------

class DiffiTTransformerBlock(nn.Module):
    """LN → TMSA → residual; LN → MLP → residual."""

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
            nn.GELU(),
            nn.Linear(hidden, dim),
        )

    def forward(self, x: torch.Tensor, temb: torch.Tensor) -> torch.Tensor:
        x = x + self.tmsa(self.norm1(x), temb)
        x = x + self.mlp(self.norm2(x))
        return x


# ---------------------------------------------------------------------------
# DiffiT ResBlock  (Eq. 9-10 in paper)
#   Conv3×3(Swish(GN(x)))  →  DiffiT-Block  +  residual
# ---------------------------------------------------------------------------

class DiffiTResBlock(nn.Module):
    """
    Residual cell combining Conv3×3 + DiffiT Transformer block.

    The conv branch applies GN → Swish → Conv3×3 to the input,
    then the transformer block processes the result with TMSA,
    and the output is added back to the original input via a
    learnable channel-matching projection when dims differ.
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
        self.in_channels  = in_channels
        self.out_channels = out_channels
        self.window_size  = window_size

        # Conv branch: GN → Swish → Conv3×3  (Eq. 9)
        self.gn   = nn.GroupNorm(groups, in_channels)
        self.conv = nn.Conv2d(in_channels, out_channels, 3, padding=1)

        # DiffiT Transformer block  (Eq. 10)
        self.transformer = DiffiTTransformerBlock(
            dim=out_channels,
            temb_dim=temb_dim,
            num_heads=num_heads,
            window_size=window_size,
            mlp_ratio=mlp_ratio,
        )

        # Skip projection when channel dims differ
        self.skip = (
            nn.Conv2d(in_channels, out_channels, 1, bias=False)
            if in_channels != out_channels else nn.Identity()
        )

    def forward(self, x: torch.Tensor, temb: torch.Tensor) -> torch.Tensor:
        # Conv branch
        h = self.conv(F.silu(self.gn(x)))               # (B, C_out, H, W)
        B, C, H, W = h.shape

        # Partition into non-overlapping windows for TMSA
        ws = self.window_size
        assert H % ws == 0 and W % ws == 0, \
            f"Feature map {H}×{W} not divisible by window size {ws}"

        # (B, C, H, W) → (B*nW, ws*ws, C)
        h_win = h.reshape(B, C, H // ws, ws, W // ws, ws)
        h_win = h_win.permute(0, 2, 4, 3, 5, 1).contiguous()  # (B, nH, nW, ws, ws, C)
        nH, nW = H // ws, W // ws
        h_win = h_win.reshape(B * nH * nW, ws * ws, C)

        # Time token is broadcast to all windows
        temb_rep = temb.unsqueeze(1).expand(B, nH * nW, -1).reshape(B * nH * nW, -1)
        h_win = self.transformer(h_win, temb_rep)               # (B*nW, ws*ws, C)

        # Reassemble windows → feature map
        h = h_win.reshape(B, nH, nW, ws, ws, C)
        h = h.permute(0, 5, 1, 3, 2, 4).contiguous()           # (B, C, H, W)
        h = h.reshape(B, C, H, W)

        return h + self.skip(x)


# ---------------------------------------------------------------------------
# Image-space DiffiT U-Net  (Fig. 4 / Appendix H.1)
# ---------------------------------------------------------------------------

class DiffiTImageUNet(nn.Module):
    """
    Image-space DiffiT for CIFAR-10 (32×32, unconditional).

    Architecture (3 stages, matching Appendix H.1 for 32×32):
      Encoder:
        Tokenizer:  Conv3×3 → 128 ch   [32×32×128]
        Stage 1:    L1 DiffiT ResBlocks  [32×32×128]
        Downsample: Conv stride-2       [16×16×128]
        Stage 2:    L2 DiffiT ResBlocks  [16×16×256]
        Downsample: Conv stride-2       [8×8×256]
        Stage 3:    L3 DiffiT ResBlocks  [8×8×256]
      Decoder (symmetric with skip connections):
        Stage 3:    L3 DiffiT ResBlocks  [8×8×256]
        Upsample:   ConvTranspose2×     [16×16×256]
        Stage 2:    L2 DiffiT ResBlocks  [16×16×256]
        Upsample:   ConvTranspose2×     [32×32×128]
        Stage 1:    L1 DiffiT ResBlocks  [32×32×128]
        Head:       Conv3×3 → 3 ch      [32×32×3]

    EDM preconditioning (Karras et al. 2022, Algorithm 2) wraps this net:
      see EDMPrecond below.
    """

    def __init__(
        self,
        in_channels: int = 3,
        base_channels: int = 128,
        channel_mult: tuple = (1, 2, 2),   # → 128, 256, 256
        num_blocks: tuple = (2, 2, 2),     # L1, L2, L3 per stage
        num_heads: int = 4,
        window_size: int = 4,              # paper: window size 4 for CIFAR-10
        temb_dim: int = 512,               # paper Table 3: time dim 512
        mlp_ratio: float = 4.0,
    ):
        super().__init__()
        assert len(channel_mult) == len(num_blocks)
        self.num_stages = len(channel_mult)
        channels = [base_channels * m for m in channel_mult]

        # Time embedding: sinusoidal → 2-layer MLP
        self.temb_proj = nn.Sequential(
            SinusoidalTimestep(temb_dim),
            TimestepMLP(temb_dim, temb_dim),
        )

        # Tokenizer (input projection)
        self.tokenizer = nn.Conv2d(in_channels, channels[0], 3, padding=1)

        # ---- Encoder -------------------------------------------------------
        self.enc_blocks: nn.ModuleList = nn.ModuleList()
        self.enc_downs:  nn.ModuleList = nn.ModuleList()

        for i in range(self.num_stages):
            c_in  = channels[i]
            stage = nn.ModuleList([
                DiffiTResBlock(
                    in_channels=c_in,
                    out_channels=c_in,
                    temb_dim=temb_dim,
                    num_heads=num_heads,
                    window_size=window_size,
                    mlp_ratio=mlp_ratio,
                )
                for _ in range(num_blocks[i])
            ])
            self.enc_blocks.append(stage)

            if i < self.num_stages - 1:
                # Convolutional downsampling + channel projection
                self.enc_downs.append(
                    nn.Conv2d(channels[i], channels[i + 1], 3, stride=2, padding=1)
                )

        # ---- Decoder -------------------------------------------------------
        self.dec_blocks: nn.ModuleList = nn.ModuleList()
        self.dec_ups:    nn.ModuleList = nn.ModuleList()

        for i in reversed(range(self.num_stages)):
            c = channels[i]
            # Skip connection doubles the channels at input of each decoder stage
            # (except the bottleneck which has no skip yet)
            c_skip = c if i == self.num_stages - 1 else c
            # First block fuses skip: in_channels = c (skip) + c (from upsample)
            fuse_ch = c * 2 if i < self.num_stages - 1 else c

            stage_blocks = nn.ModuleList()
            for j in range(num_blocks[i]):
                in_c = fuse_ch if j == 0 and i < self.num_stages - 1 else c
                stage_blocks.append(
                    DiffiTResBlock(
                        in_channels=in_c,
                        out_channels=c,
                        temb_dim=temb_dim,
                        num_heads=num_heads,
                        window_size=window_size,
                        mlp_ratio=mlp_ratio,
                    )
                )
            self.dec_blocks.append(stage_blocks)

            if i > 0:
                # Convolutional upsampling + channel projection
                self.dec_ups.append(
                    nn.ConvTranspose2d(channels[i], channels[i - 1], 4, stride=2, padding=1)
                )

        # Head
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
        # Zero-init head so output starts near zero
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, x: torch.Tensor, temb_input: torch.Tensor) -> torch.Tensor:
        """
        x          : (B, C_in, H, W)  noisy image (already scaled by EDM)
        temb_input : (B,)  log(σ)/4  (EDM convention) or raw σ for manual use
        """
        # Time embedding
        temb = self.temb_proj(temb_input)               # (B, temb_dim)

        # Tokenize
        h = self.tokenizer(x)                            # (B, ch[0], H, W)

        # Encoder — save skip connections
        skips: List[torch.Tensor] = []
        for i in range(self.num_stages):
            for blk in self.enc_blocks[i]:
                h = blk(h, temb)
            skips.append(h)
            if i < self.num_stages - 1:
                h = self.enc_downs[i](h)

        # Decoder — stages in reverse
        # dec_blocks[0] = deepest stage, dec_ups[0] = deepest upsample
        up_idx = 0
        for j, stage_blocks in enumerate(self.dec_blocks):
            stage_idx = self.num_stages - 1 - j          # which resolution
            if j > 0:
                # Concatenate skip from matching encoder stage
                h = torch.cat([h, skips[stage_idx]], dim=1)
            for blk in stage_blocks:
                h = blk(h, temb)
            if stage_idx > 0:
                h = self.dec_ups[up_idx](h)
                up_idx += 1

        return self.head(h)


# ---------------------------------------------------------------------------
# EDM Preconditioning  (Karras et al. 2022, Algorithm 2)
# Wraps the raw U-Net so the network sees normalized inputs and learns
# a well-conditioned target at every noise level.
# ---------------------------------------------------------------------------

class EDMPrecond(nn.Module):
    """
    EDM-style preconditioning that wraps DiffiTImageUNet.

    Given noisy image y = x + n  where n ~ N(0, σ²I):
      c_skip(σ)  = σ_data² / (σ² + σ_data²)
      c_out(σ)   = σ · σ_data / sqrt(σ² + σ_data²)
      c_in(σ)    = 1 / sqrt(σ² + σ_data²)
      c_noise(σ) = log(σ) / 4

    D(y; σ) = c_skip · y  +  c_out · F(c_in · y,  c_noise)

    where F is the raw network (DiffiTImageUNet).
    """

    def __init__(
        self,
        model: DiffiTImageUNet,
        sigma_data: float = 0.5,   # EDM default for CIFAR-10
    ):
        super().__init__()
        self.inner_model = model
        self.sigma_data  = sigma_data

    def forward(self, y: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
        """
        y     : (B, C, H, W)  noisy image
        sigma : (B,)          noise level σ for each sample
        Returns denoised estimate D(y; σ).
        """
        s = sigma.view(-1, 1, 1, 1)                     # broadcast shape

        c_skip  = self.sigma_data ** 2 / (s ** 2 + self.sigma_data ** 2)
        c_out   = s * self.sigma_data / (s ** 2 + self.sigma_data ** 2).sqrt()
        c_in    = 1.0 / (s ** 2 + self.sigma_data ** 2).sqrt()
        c_noise = (sigma.log() / 4.0)                   # (B,) for temb

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
    Construct the image-space DiffiT model for CIFAR-10 exactly as described
    in the paper (Appendix H.1 + I.1):
      - 3 stages (32→16→8), 2 blocks each
      - base_channels = 128, channel_mult = (1, 2, 2)
      - window size = 4, num_heads = 4
      - time embedding dim = 512
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