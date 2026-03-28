# DiffiT — Diffusion Vision Transformers for Image Generation

Implementation of **DiffiT** (Hatamizadeh et al., NVIDIA 2023) for image-space generation on CIFAR-10, with two novel activation/attention variants: **APTx** and **Noise-Aware Attention (NAA)**.

> Paper: *DiffiT: Diffusion Vision Transformers for Image Generation*
> Framework: EDM (Karras et al., NeurIPS 2022)

---

## Repository Structure

```
DL_DiffiT_/
├── diffit/                  # Latent-space DiffiT (ImageNet-256/512)
├── diffit_image.py          # Image-space DiffiT model — baseline (SiLU)
├── diffit_image_aptx.py     # Image-space DiffiT — APTx activation variant
├── diffit_image_naa.py      # Image-space DiffiT — Noise-Aware Attention variant
├── train.py                 # Training script (EDM framework, CIFAR-10)
├── train_cifar10.sh         # Shell launcher for train.py
├── sample.py                # Sampling script (latent ImageNet model)
├── sample_cifar10.py        # Sampling script (image-space CIFAR-10 model)
├── sample_cifar10.sh        # Shell launcher for sample_cifar10.py
├── run_sample.sh            # Shell launcher for latent sample.py
├── evaluator.py             # FID / IS / sFID / Precision / Recall evaluator
├── eval_run.sh              # Shell launcher for evaluator.py
├── requirement.txt          # Python dependencies
└── README.md                # This file
```

---

## File Descriptions

### `diffit_image.py` — Baseline Image-Space Model

The core U-Net–style DiffiT architecture for CIFAR-10 (32×32, unconditional).

**Architecture:**
- 3 resolution stages: 32×32 → 16×16 → 8×8 (encoder) and reverse (decoder)
- Each stage: 2 `DiffiTResBlock`s with skip connections
- Channel dims: 128 → 256 → 256
- **TMSA** (Time-dependent Multi-head Self-Attention): time token modulates Q, K, V via separate linear projections `W_qt, W_kt, W_vt`
- Window-based local attention, window size = 4
- **SiLU** (Swish) activation in conv branches and time MLP
- EDM preconditioning wraps the UNet (`EDMPrecond`)

**Key classes:**
| Class | Role |
|---|---|
| `SinusoidalTimestep` | Maps σ → sinusoidal embedding |
| `TimestepMLP` | 2-layer MLP, produces time token |
| `RelativePositionBias` | Swin-style relative position bias |
| `TMSA` | Time-dependent multi-head self-attention |
| `DiffiTTransformerBlock` | LN → TMSA → residual; LN → MLP → residual |
| `DiffiTResBlock` | GN → SiLU → Conv3×3 → DiffiT block + skip |
| `DiffiTImageUNet` | Full encoder-decoder U-Net |
| `EDMPrecond` | EDM preconditioning wrapper (c_skip, c_out, c_in, c_noise) |
| `build_cifar10_model()` | Factory function |

---

### `diffit_image_aptx.py` — APTx Activation Variant

Identical to `diffit_image.py` except **SiLU is replaced by APTx** in:
- `DiffiTResBlock` conv branch: `GN → APTx → Conv3×3`
- `TimestepMLP` hidden activation

**APTx** (Kumar 2022, *APTx: better activation function than MISH, SWISH, and ReLU's variants*):

```
ψ(x) = (1 + tanh(x)) * x/2          (α=1, β=1, γ=½)
ψ'(x) = (1 + tanh(x) + x·sech²(x)) / 2
```

Behaves similarly to MISH but requires fewer floating-point operations in both forward and backward passes. Bounded below, unbounded above — no dying neuron problem.

**New classes:**
| Class | Role |
|---|---|
| `APTx` | Module form, optionally learnable α, β, γ |
| `aptx()` | Functional form for inline use |

GELU (transformer MLP) and Softmax (attention) are **unchanged**.

**Observed training speed:** ~0.84s/step — negligible overhead vs SiLU (0.83s/step).

---

### `diffit_image_naa.py` — Noise-Aware Attention Variant

Replaces TMSA with **Noise-Aware Attention (NAA)** — a novel attention mechanism that directly conditions the attention *distribution shape* on the raw noise level σ.

**Motivation:** At high σ (early denoising), the network should attend broadly to capture global structure. At low σ (fine detail), attention should be sharp and local. TMSA achieves this implicitly via the time token; NAA makes it explicit and direct.

**NAA mechanism:**

A `NoiseLevelMLP` maps raw σ → two outputs:
- **Gate** `g(σ) ∈ (0,1)`: controls attention temperature
- **Per-head bias** `b(σ) ∈ R^H`: shifts each head's attention distribution

Attention logits are modulated as:
```
attn = QK^T / (√d · (1 + g(σ)))  +  b(σ)
```

- High σ → g→1 → divide by ~2 → flat softmax → **broad attention**
- Low σ → g→0 → divide by ~1 → peaked softmax → **sharp attention**

QKV still receive the time embedding (same as TMSA) — the time token controls *what* features are extracted; NAA controls *how broadly* they are attended to. These are complementary.

**New classes:**
| Class | Role |
|---|---|
| `NoiseLevelMLP` | σ → gate g(σ) and per-head bias b(σ) |
| `NoiseAwareAttention` | NAA replacing TMSA |

**Observed training speed:** ~0.80s/step — slightly faster than baseline.

---

### `train.py` — Training Script

Trains the image-space DiffiT on CIFAR-10 using the EDM framework.

**Paper configuration (Appendix I.1):**
- Optimizer: Adam, lr=1e-3, β=(0.9, 0.999)
- Batch size: 512, 200k iterations
- EMA decay: 0.9999
- EDM loss: weighted MSE with λ(σ) = (σ²+σ_data²)/(σ·σ_data)²
- Noise schedule: ln σ ~ N(-1.2, 1.2²)

**Key components:**
| Component | Description |
|---|---|
| `EDMLoss` | EDM training objective (Karras et al. Algorithm 1) |
| `edm_sample()` | EDM deterministic Heun ODE sampler, 18 steps |
| `build_loader()` | CIFAR-10 DataLoader with horizontal flip, DDP sampler |
| `update_ema()` | EMA weight update |
| `save_checkpoint()` | Saves model, EMA, optimizer state |
| `load_checkpoint()` | Resumes from saved checkpoint |

**Usage:**
```bash
bash train_cifar10.sh
```

To switch model variant, change the import in `train.py`:
```python
from diffit_image import build_cifar10_model       # baseline SiLU
from diffit_image_aptx import build_cifar10_model  # APTx variant
from diffit_image_naa import build_cifar10_model   # NAA variant
```

---

### `train_cifar10.sh` — Training Launcher

Shell script that configures and launches `train.py` via `torchrun`.

**Key parameters to edit:**
```bash
RESULTS_DIR="./results/diffit_cifar10"   # output directory
TOTAL_KIMG=102400                         # training budget (200k iters = 102400 kimg)
RESUME=""                                 # path to checkpoint to resume from
CUDA_VISIBLE_DEVICES=0                    # GPU selection
```

---

### `sample_cifar10.py` — CIFAR-10 Sampler

Generates 50k samples from a trained CIFAR-10 checkpoint for FID evaluation.

- Loads EMA weights from checkpoint saved by `train.py`
- Runs EDM deterministic Heun sampler (18 steps)
- Saves `samples_50000x32x32x3.npz` (uint8, shape N×H×W×C) — compatible with `evaluator.py`

**Usage:**
```bash
bash sample_cifar10.sh
```

---

### `sample_cifar10.sh` — Sampling Launcher

```bash
CKPT="./results/diffit_cifar10/ckpt_0200000.pt"
LOG_DIR="./log_dir/cifar10_1"
NUM_SAMPLES=50000
BATCH_SIZE=256
NUM_STEPS=18
```

---

### `sample.py` — Latent ImageNet Sampler

Samples from the **latent-space DiffiT** model (ImageNet-256/512). Uses:
- `stabilityai/sd-vae-ft-ema` VAE decoder
- DDPM/DDIM sampler (250 steps)
- Class-conditional generation with classifier-free guidance

**Not for CIFAR-10** — this is a separate pipeline for the pretrained ImageNet model.

**Usage:**
```bash
bash run_sample.sh
```

---

### `evaluator.py` — FID / IS / sFID Evaluator

Computes image generation metrics using InceptionV3 features.

**Metrics computed:**
| Metric | Description |
|---|---|
| FID | Fréchet Inception Distance (lower is better) |
| IS | Inception Score (higher is better) |
| sFID | Spatial FID using mixed_6/conv features |
| Precision | Fraction of generated samples in real manifold |
| Recall | Fraction of real samples covered by generated manifold |

**Usage:**
```bash
# Run on CPU (avoids CuDNN version mismatch issues)
CUDA_VISIBLE_DEVICES="" python evaluator.py \
    ./cifar10_ref.npz \
    ./log_dir/cifar10_1/samples_50000x32x32x3.npz \
    --log_dir ./log_dir/cifar10_1
```

The reference `.npz` must contain raw images under key `arr_0`. Generate it from CIFAR-10 test set:
```bash
python -c "
import numpy as np
from torchvision.datasets import CIFAR10
import torchvision.transforms as T
ds = CIFAR10('./data', train=False, download=True, transform=T.ToTensor())
imgs = np.stack([(np.array(ds[i][0].permute(1,2,0))*255).astype('uint8') for i in range(len(ds))])
np.savez('./cifar10_ref.npz', arr_0=imgs)
"
```

---

### `eval_run.sh` — Evaluator Launcher

```bash
CUDA_VISIBLE_DEVICES="" python evaluator.py \
    ./cifar10_ref.npz \
    ./log_dir/cifar10_1/samples_50000x32x32x3.npz \
    --log_dir ./log_dir/cifar10_1
```

Edit the sample batch path to match your output file.

---

## Model Variants — Comparison

| Variant | File | Activation | Attention | Speed |
|---|---|---|---|---|
| Baseline | `diffit_image.py` | SiLU | TMSA (time token) | 0.83s/step |
| APTx | `diffit_image_aptx.py` | APTx | TMSA (time token) | 0.84s/step |
| NAA | `diffit_image_naa.py` | SiLU | Noise-Aware (σ direct) | 0.80s/step |

---

## Full Pipeline

```
1. Train
   bash train_cifar10.sh
   → results/diffit_cifar10/ckpt_*.pt

2. Sample
   bash sample_cifar10.sh
   → log_dir/cifar10_1/samples_50000x32x32x3.npz

3. Build reference stats (once)
   python -c "... cifar10_ref.npz ..."

4. Evaluate
   bash eval_run.sh
   → log_dir/cifar10_1/evaluation_*.log
```

---

## Paper Results (ImageNet-256, Latent DiffiT)

| Model | FID ↓ | IS ↑ | Precision ↑ | Recall ↑ |
|---|---|---|---|---|
| DiT-XL/2-G | 2.27 | 278.24 | 0.83 | 0.57 |
| MDT-G | 1.79 | 283.01 | 0.81 | 0.61 |
| **DiffiT** | **1.73** | 276.49 | 0.80 | **0.62** |

---

## Dependencies

```bash
pip install -r requirement.txt
```

Core requirements: `torch`, `torchvision`, `timm`, `diffusers`, `numpy`, `tqdm`, `tensorflow` (for evaluator), `scipy`, `requests`

---

## Citation

```bibtex
@article{hatamizadeh2023diffit,
  title={DiffiT: Diffusion Vision Transformers for Image Generation},
  author={Hatamizadeh, Ali and Song, Jiaming and Liu, Guilin and Kautz, Jan and Vahdat, Arash},
  journal={arXiv preprint arXiv:2312.02139},
  year={2023}
}
```