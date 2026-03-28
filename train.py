#!/usr/bin/env python3
"""
train.py — Train image-space DiffiT on CIFAR-10 using the EDM framework.

Faithfully implements the paper's training configuration
(Appendix I.1 of "DiffiT: Diffusion Vision Transformers for Image Generation"):
  - EDM noise conditioning & loss weighting  (Karras et al. NeurIPS 2022)
  - Adam, lr = 1e-3, batch = 512, 200 000 iterations
  - Unconditional generation (no class labels)
  - 8× A100 GPUs via torchrun DDP
  - EMA decay = 0.9999

Usage:
    torchrun --nproc_per_node=8 train.py            # paper config
    torchrun --nproc_per_node=1 train.py --gpus 1  # single-GPU debug
"""

import argparse
import copy
import logging
import os
from collections import OrderedDict

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchvision import datasets, transforms
from torchvision.utils import save_image

# Local import — diffit_image.py must be in the same directory (or on PYTHONPATH)
from diffit_image_naa import build_cifar10_model

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def setup_logging(rank: int, log_dir: str) -> None:
    handlers = [logging.StreamHandler()]
    if rank == 0:
        os.makedirs(log_dir, exist_ok=True)
        handlers.append(logging.FileHandler(os.path.join(log_dir, "train.log")))
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s][rank %(process)d] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=handlers,
        force=True,
    )

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# EMA
# ---------------------------------------------------------------------------

@torch.no_grad()
def update_ema(ema: torch.nn.Module, model: torch.nn.Module, decay: float) -> None:
    ema_p = OrderedDict(ema.named_parameters())
    mdl_p = OrderedDict(model.named_parameters())
    for name, param in mdl_p.items():
        ema_p[name].mul_(decay).add_(param.data, alpha=1.0 - decay)


def requires_grad(model: torch.nn.Module, flag: bool) -> None:
    for p in model.parameters():
        p.requires_grad = flag


# ---------------------------------------------------------------------------
# EDM noise distribution & loss  (Karras et al. 2022, Section 5)
# ---------------------------------------------------------------------------

class EDMLoss:
    """
    EDM training loss (Algorithm 1 in Karras et al. 2022).

    σ is sampled from ln N(P_mean, P_std²):
        ln σ ~ N(P_mean, P_std²)

    Loss weight λ(σ) = (σ² + σ_data²) / (σ · σ_data)²

    L = E_σ E_y E_n [ λ(σ) · ‖ D(y+n; σ) − y ‖² ]
    where y is a clean image and n ~ N(0, σ²I).

    Paper uses EDM defaults for CIFAR-10:
        P_mean = -1.2,  P_std = 1.2,  σ_data = 0.5
    """

    def __init__(
        self,
        P_mean:    float = -1.2,
        P_std:     float =  1.2,
        sigma_data: float = 0.5,
    ):
        self.P_mean     = P_mean
        self.P_std      = P_std
        self.sigma_data = sigma_data

    def __call__(
        self,
        model: torch.nn.Module,
        images: torch.Tensor,
    ) -> torch.Tensor:
        """
        images : (B, 3, H, W)  clean images in [-1, 1]
        Returns scalar mean loss.
        """
        B = images.shape[0]
        device = images.device

        # Sample noise levels: ln σ ~ N(P_mean, P_std²)
        rnd_normal = torch.randn(B, device=device)
        sigma = (rnd_normal * self.P_std + self.P_mean).exp()  # (B,)

        # Loss weight λ(σ)
        weight = (sigma ** 2 + self.sigma_data ** 2) / (sigma * self.sigma_data) ** 2

        # Corrupt images: y = x + n,  n ~ N(0, σ²I)
        noise = torch.randn_like(images)
        noisy = images + sigma.view(-1, 1, 1, 1) * noise

        # Forward pass → denoised estimate D(y; σ)
        denoised = model(noisy, sigma)

        # MSE loss weighted by λ(σ)
        loss = weight.view(-1, 1, 1, 1) * (denoised - images) ** 2
        return loss.mean()


# ---------------------------------------------------------------------------
# EDM deterministic sampler (Karras et al. Algorithm 1, stochastic=False)
# Paper: 18 steps for CIFAR-10 evaluation
# ---------------------------------------------------------------------------

@torch.no_grad()
def edm_sample(
    model: torch.nn.Module,
    batch_size: int,
    img_channels: int,
    img_size: int,
    device: torch.device,
    num_steps: int = 18,
    sigma_min: float = 0.002,
    sigma_max: float = 80.0,
    rho: float = 7.0,
    S_churn: float = 0.0,   # deterministic: S_churn = 0
    sigma_data: float = 0.5,
) -> torch.Tensor:
    """
    EDM deterministic ODE sampler (Heun 2nd-order).
    Returns (B, C, H, W) images in [-1, 1].
    """
    # Build σ schedule: Eq. 5 in Karras et al.
    step_indices = torch.arange(num_steps, device=device, dtype=torch.float64)
    t_steps = (
        sigma_max ** (1 / rho)
        + step_indices / (num_steps - 1) * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))
    ) ** rho
    t_steps = torch.cat([t_steps, torch.zeros_like(t_steps[:1])])  # append σ=0

    # Start from pure noise: x_T ~ N(0, σ_max²·I)
    x = torch.randn(batch_size, img_channels, img_size, img_size, device=device,
                    dtype=torch.float64) * t_steps[0]

    model.eval()
    for i, (t_cur, t_next) in enumerate(zip(t_steps[:-1], t_steps[1:])):
        x_cur = x

        # Stochastic churn (0 for deterministic)
        gamma = min(S_churn / num_steps, 2 ** 0.5 - 1)
        t_hat = t_cur + gamma * t_cur
        if gamma > 0:
            eps = torch.randn_like(x_cur) * (t_hat ** 2 - t_cur ** 2).sqrt()
            x_cur = x_cur + eps

        # First-order step
        sigma_batch = t_hat.expand(batch_size).float()
        d_cur = (x_cur - model(x_cur.float(), sigma_batch).double()) / t_hat

        x_next = x_cur + (t_next - t_hat) * d_cur

        # Second-order correction (Heun)
        if i < num_steps - 1:
            sigma_batch2 = t_next.expand(batch_size).float()
            d_next = (x_next - model(x_next.float(), sigma_batch2).double()) / t_next
            x_next = x_cur + (t_next - t_hat) * 0.5 * (d_cur + d_next)

        x = x_next

    return x.float().clamp(-1, 1)


# ---------------------------------------------------------------------------
# CIFAR-10 DataLoader
# ---------------------------------------------------------------------------

def build_loader(
    data_root: str,
    batch_size: int,
    num_workers: int,
    rank: int,
    world_size: int,
) -> DataLoader:
    """
    CIFAR-10 with horizontal flip augmentation, normalised to [-1, 1].
    Paper trains unconditionally so labels are discarded in the training loop.
    """
    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),  # [0,1]→[-1,1]
    ])
    dataset = datasets.CIFAR10(
        root=data_root, train=True, transform=transform, download=True
    )
    sampler = DistributedSampler(
        dataset, num_replicas=world_size, rank=rank, shuffle=True, drop_last=True
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
        persistent_workers=(num_workers > 0),
    )


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------

def save_checkpoint(
    out_dir: str,
    step: int,
    model: torch.nn.Module,    # DDP-wrapped
    ema: torch.nn.Module,
    opt: torch.optim.Optimizer,
) -> None:
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"ckpt_{step:07d}.pt")
    torch.save({
        "step":  step,
        "model": model.module.state_dict(),
        "ema":   ema.state_dict(),
        "opt":   opt.state_dict(),
    }, path)
    logger.info("Checkpoint saved → %s", path)


def load_checkpoint(
    path: str,
    model: torch.nn.Module,   # DDP-wrapped
    ema: torch.nn.Module,
    opt: torch.optim.Optimizer,
) -> int:
    state = torch.load(path, map_location="cpu")
    model.module.load_state_dict(state["model"])
    ema.load_state_dict(state["ema"])
    opt.load_state_dict(state["opt"])
    logger.info("Resumed from %s at step %d", path, state["step"])
    return state["step"]


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Train image-space DiffiT on CIFAR-10 (EDM framework)"
    )

    # I/O
    p.add_argument("--data-root",    default="./data")
    p.add_argument("--results-dir",  default="./results/diffit_cifar10")
    p.add_argument("--resume",       default=None,
                   help="Path to a .pt checkpoint to resume from")

    # Training  — paper: Adam, lr=1e-3, batch=512, 200k iters
    p.add_argument("--total-kimg",   type=int,   default=200,
                   help="Total training images in thousands "
                        "(200 = 200k iters × bs512 = 102.4M images). "
                        "Paper trains for 200k iterations.")
    p.add_argument("--global-batch", type=int,   default=512,
                   help="Global batch size across all GPUs. Paper: 512")
    p.add_argument("--lr",           type=float, default=1e-3,
                   help="Adam learning rate. Paper: 1e-3 for CIFAR-10")
    p.add_argument("--beta1",        type=float, default=0.9)
    p.add_argument("--beta2",        type=float, default=0.999)
    p.add_argument("--eps",          type=float, default=1e-8)
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--grad-clip",    type=float, default=1.0,
                   help="Max gradient norm (0 = disabled)")
    p.add_argument("--ema-decay",    type=float, default=0.9999,
                   help="EMA decay. Paper latent space: 0.9999")

    # EDM loss hyper-parameters  (Karras et al. 2022 CIFAR-10 defaults)
    p.add_argument("--P-mean",       type=float, default=-1.2)
    p.add_argument("--P-std",        type=float, default=1.2)
    p.add_argument("--sigma-data",   type=float, default=0.5)

    # Model
    p.add_argument("--base-channels",  type=int,   default=128)
    p.add_argument("--channel-mult",   type=int,   nargs="+", default=[1, 2, 2])
    p.add_argument("--num-blocks",     type=int,   nargs="+", default=[2, 2, 2])
    p.add_argument("--num-heads",      type=int,   default=4)
    p.add_argument("--window-size",    type=int,   default=4,
                   help="TMSA local window size. Paper: 4 for CIFAR-10")
    p.add_argument("--temb-dim",       type=int,   default=512,
                   help="Time embedding dim. Paper Table 3: 512")

    # AMP
    p.add_argument("--fp16", action="store_true", default=False)

    # Logging / checkpointing
    p.add_argument("--log-every",    type=int, default=100)
    p.add_argument("--ckpt-kimg",    type=int, default=10,
                   help="Save checkpoint every N thousand images")
    p.add_argument("--sample-kimg",  type=int, default=5,
                   help="Save sample grid every N thousand images")
    p.add_argument("--num-workers",  type=int, default=4)

    # Sampling (used for periodic eval grids)
    p.add_argument("--sample-steps", type=int,   default=18,
                   help="EDM deterministic sampler steps. Paper: 18 for CIFAR-10")
    p.add_argument("--sample-n",     type=int,   default=64,
                   help="Number of samples to generate in eval grids")

    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    # ------------------------------------------------------------------ dist
    assert torch.cuda.is_available(), "CUDA required"
    dist.init_process_group(backend="nccl")
    rank       = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = rank % torch.cuda.device_count()
    device     = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)

    # Performance flags
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32       = True
    torch.backends.cudnn.benchmark        = True

    setup_logging(rank, args.results_dir)
    if rank == 0:
        logger.info("=" * 70)
        logger.info("Image-space DiffiT  |  CIFAR-10  |  EDM framework")
        logger.info("=" * 70)
        logger.info("Args: %s", args)
        logger.info("World size: %d", world_size)

    # ---------------------------------------------------- per-GPU batch size
    assert args.global_batch % world_size == 0, \
        "global_batch must be divisible by world_size"
    local_batch = args.global_batch // world_size

    # ----------------------------------------------------------------- model
    model = build_cifar10_model(sigma_data=args.sigma_data).to(device)

    ema = copy.deepcopy(model).to(device)
    requires_grad(ema, False)
    ema.eval()

    model = DDP(model, device_ids=[local_rank])

    if rank == 0:
        n_params = sum(p.numel() for p in model.parameters())
        logger.info("DiffiT image-space params: %.2f M", n_params / 1e6)

    # --------------------------------------------------------- loss & optim
    edm_loss = EDMLoss(
        P_mean=args.P_mean,
        P_std=args.P_std,
        sigma_data=args.sigma_data,
    )

    opt = torch.optim.Adam(
        model.parameters(),
        lr=args.lr,
        betas=(args.beta1, args.beta2),
        eps=args.eps,
        weight_decay=args.weight_decay,
    )
    scaler = torch.cuda.amp.GradScaler(enabled=args.fp16)

    # -------------------------------------------------------------- dataset
    loader = build_loader(
        data_root=args.data_root,
        batch_size=local_batch,
        num_workers=args.num_workers,
        rank=rank,
        world_size=world_size,
    )

    # -------------------------------------------------- compute step budget
    # total_kimg * 1000 images / global_batch_size = total iterations
    total_steps = (args.total_kimg * 1000) // args.global_batch
    ckpt_every  = (args.ckpt_kimg   * 1000) // args.global_batch
    sample_every = (args.sample_kimg * 1000) // args.global_batch

    if rank == 0:
        logger.info(
            "Training plan: %d kimg  (%d iterations at global_batch=%d)",
            args.total_kimg, total_steps, args.global_batch,
        )

    # --------------------------------------------------------- optional resume
    start_step = 0
    if args.resume:
        start_step = load_checkpoint(args.resume, model, ema, opt)

    # ---------------------------------------------------------------- train
    model.train()
    step          = start_step
    running_loss  = 0.0
    epoch         = 0
    loader_iter   = iter(loader)

    while step < total_steps:
        # --- next batch (cycle through dataset epochs) ---
        try:
            images, _ = next(loader_iter)   # labels discarded (unconditional)
        except StopIteration:
            epoch += 1
            loader.sampler.set_epoch(epoch)
            loader_iter = iter(loader)
            images, _ = next(loader_iter)

        images = images.to(device, non_blocking=True)   # (B, 3, 32, 32) ∈ [-1,1]

        # --- forward + loss ---
        with torch.cuda.amp.autocast(enabled=args.fp16):
            loss = edm_loss(model, images)

        # --- backward ---
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()

        if args.grad_clip > 0:
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)

        scaler.step(opt)
        scaler.update()

        # --- EMA update ---
        update_ema(ema, model.module, decay=args.ema_decay)

        running_loss += loss.item()
        step += 1

        # --- logging ---
        if rank == 0 and step % args.log_every == 0:
            avg = running_loss / args.log_every
            running_loss = 0.0
            imgs_seen = step * args.global_batch // 1000
            logger.info(
                "step %7d / %d  |  kimg %5d  |  loss %.5f",
                step, total_steps, imgs_seen, avg,
            )

        # --- checkpoint ---
        if rank == 0 and step % ckpt_every == 0:
            save_checkpoint(args.results_dir, step, model, ema, opt)

        # --- sample grid (EMA model, rank-0 only) ---
        if rank == 0 and step % sample_every == 0:
            _save_sample_grid(
                ema=ema,
                step=step,
                out_dir=args.results_dir,
                n=args.sample_n,
                num_steps=args.sample_steps,
                sigma_data=args.sigma_data,
                device=device,
            )
            model.train()

    # Final checkpoint
    if rank == 0:
        save_checkpoint(args.results_dir, step, model, ema, opt)
        logger.info("Training complete. Final step: %d", step)

    dist.destroy_process_group()


# ---------------------------------------------------------------------------
# Sample grid helper
# ---------------------------------------------------------------------------

@torch.no_grad()
def _save_sample_grid(
    ema: torch.nn.Module,
    step: int,
    out_dir: str,
    n: int,
    num_steps: int,
    sigma_data: float,
    device: torch.device,
) -> None:
    samples = edm_sample(
        model=ema,
        batch_size=n,
        img_channels=3,
        img_size=32,
        device=device,
        num_steps=num_steps,
        sigma_data=sigma_data,
    )
    # [-1, 1] → [0, 1]
    samples = (samples * 0.5 + 0.5).clamp(0, 1)
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"samples_{step:07d}.png")
    save_image(samples, path, nrow=int(n ** 0.5))
    logger.info("Sample grid saved → %s", path)


if __name__ == "__main__":
    main()