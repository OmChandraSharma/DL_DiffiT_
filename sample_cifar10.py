#!/usr/bin/env python3
"""
sample_cifar10.py — Generate samples from a trained image-space DiffiT
                    (CIFAR-10, EDM framework) and save as .npz for FID eval.

Loads the EMA weights saved by train.py and runs the EDM deterministic
sampler (18 steps, as per Appendix I.1 of the paper).

Output .npz shape: (N, 32, 32, 3)  uint8  — compatible with evaluator.py
"""

import argparse
import os

import numpy as np
import torch
import torch.distributed as dist
from tqdm import tqdm

from diffit_image_naa import build_cifar10_model


# ---------------------------------------------------------------------------
# EDM deterministic sampler  (copied from train.py — no train.py import
# needed so this script is fully self-contained)
# ---------------------------------------------------------------------------

@torch.no_grad()
def edm_sample(
    model: torch.nn.Module,
    batch_size: int,
    device: torch.device,
    num_steps: int = 18,
    sigma_min: float = 0.002,
    sigma_max: float = 80.0,
    rho: float = 7.0,
    sigma_data: float = 0.5,
) -> torch.Tensor:
    """
    EDM deterministic Heun ODE sampler.
    Returns (B, 3, 32, 32) float32 in [-1, 1].
    """
    # σ schedule (Eq. 5, Karras et al.)
    step_indices = torch.arange(num_steps, device=device, dtype=torch.float64)
    t_steps = (
        sigma_max ** (1 / rho)
        + step_indices / (num_steps - 1)
        * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))
    ) ** rho
    t_steps = torch.cat([t_steps, torch.zeros_like(t_steps[:1])])  # append σ=0

    # Start from pure noise x_T ~ N(0, σ_max² I)
    x = torch.randn(batch_size, 3, 32, 32, device=device, dtype=torch.float64) * t_steps[0]

    model.eval()
    for i, (t_cur, t_next) in enumerate(zip(t_steps[:-1], t_steps[1:])):
        x_cur = x
        t_hat = t_cur  # S_churn=0 → deterministic

        # First-order step
        sigma_b = t_hat.expand(batch_size).float()
        d_cur = (x_cur - model(x_cur.float(), sigma_b).double()) / t_hat

        x_next = x_cur + (t_next - t_hat) * d_cur

        # Second-order Heun correction
        if i < num_steps - 1:
            sigma_b2 = t_next.expand(batch_size).float()
            d_next = (x_next - model(x_next.float(), sigma_b2).double()) / t_next
            x_next = x_cur + (t_next - t_hat) * 0.5 * (d_cur + d_next)

        x = x_next

    return x.float().clamp(-1.0, 1.0)


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Sample from trained CIFAR-10 image-space DiffiT"
    )
    p.add_argument(
        "--ckpt",
        required=True,
        help="Path to checkpoint .pt saved by train.py "
             "(e.g. results/diffit_cifar10/ckpt_0200000.pt)",
    )
    p.add_argument(
        "--log_dir",
        default="./log_dir/cifar10",
        help="Output directory — .npz is saved here",
    )
    p.add_argument(
        "--num_samples", type=int, default=50000,
        help="Total samples to generate (default: 50000 for FID-50K)",
    )
    p.add_argument(
        "--batch_size", type=int, default=256,
        help="Samples per forward pass (lower if OOM)",
    )
    p.add_argument(
        "--num_steps", type=int, default=18,
        help="EDM sampler steps (paper: 18 for CIFAR-10)",
    )
    p.add_argument(
        "--sigma_data", type=float, default=0.5,
        help="EDM sigma_data (paper default: 0.5)",
    )
    p.add_argument(
        "--seed", type=int, default=42,
    )
    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    torch.manual_seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.benchmark = True

    os.makedirs(args.log_dir, exist_ok=True)

    # ── Load model ──────────────────────────────────────────────────────────
    print(f"Loading checkpoint: {args.ckpt}")
    state = torch.load(args.ckpt, map_location="cpu")

    model = build_cifar10_model(sigma_data=args.sigma_data).to(device)

    # train.py saves EMA weights under key "ema"
    model.load_state_dict(state["ema"])
    model.eval()

    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    step_saved = state.get("step", "?")
    print(f"  Loaded EMA weights  |  {n_params:.2f}M params  |  step {step_saved}")

    # ── Sample ──────────────────────────────────────────────────────────────
    all_samples = []
    n_done = 0
    pbar = tqdm(total=args.num_samples, unit="img",
                bar_format="{percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} "
                           "[{elapsed}<{remaining}, {rate_fmt}]")

    while n_done < args.num_samples:
        bs = min(args.batch_size, args.num_samples - n_done)

        with torch.no_grad():
            samples = edm_sample(
                model=model,
                batch_size=bs,
                device=device,
                num_steps=args.num_steps,
                sigma_data=args.sigma_data,
            )

        # [-1, 1] → [0, 255] uint8,  shape (B, 3, 32, 32) → (B, 32, 32, 3)
        samples = ((samples * 0.5 + 0.5) * 255).clamp(0, 255).byte()
        samples = samples.permute(0, 2, 3, 1).cpu().numpy()   # (B, 32, 32, 3)
        all_samples.append(samples)

        n_done += bs
        pbar.update(bs)

    pbar.close()

    # ── Save .npz ────────────────────────────────────────────────────────────
    arr = np.concatenate(all_samples, axis=0)[: args.num_samples]   # (N, 32, 32, 3)
    shape_str = "x".join(str(s) for s in arr.shape)
    out_path = os.path.join(args.log_dir, f"naa10kiter_samples_{shape_str}.npz")
    np.savez(out_path, arr_0=arr)

    print(f"\nSaved {arr.shape[0]} samples → {out_path}")
    print(f"Shape: {arr.shape}  dtype: {arr.dtype}")


if __name__ == "__main__":
    main()