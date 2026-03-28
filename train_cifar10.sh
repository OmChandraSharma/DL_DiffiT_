#!/usr/bin/env bash
# =============================================================================
# train_cifar10.sh — Launch image-space DiffiT on CIFAR-10 (EDM framework)
#
# Faithfully reproduces Appendix I.1 of:
#   "DiffiT: Diffusion Vision Transformers for Image Generation"
#   Hatamizadeh et al., NVIDIA (2023)
#
# Paper configuration:
#   - EDM framework (Karras et al. 2022)
#   - Adam, lr=1e-3, batch=512, 200k iterations (unconditional)
#   - 3 stages (32→16→8), 2 blocks/stage, window size=4
#   - Time embedding dim=512  (Table 3 optimal)
#   - EDM deterministic sampler, 18 steps for evaluation
#   - 8× NVIDIA A100 GPUs
#   - EMA decay 0.9999
#
# Modes (set MODE below):
#   single  — 1 GPU, smoke-test / quick ablation
#   multi   — N GPUs on one node (torchrun), recommended
#   slurm   — SLURM multi-node via sbatch
# =============================================================================

set -euo pipefail

# --------------------------------------------------------------------------- #
# ▶  EDIT THESE                                                               #
# --------------------------------------------------------------------------- #
MODE="single"              # Forced to single
NUM_GPUS=1                 # Only using 1 GPU
CUDA_VISIBLE_DEVICES=0     # Explicitly target the first GPU
export CUDA_VISIBLE_DEVICES
#MODE="single"          # single | multi | slurm

#NUM_GPUS=1          # GPUs per node   (multi / slurm)
#NUM_NODES=1           # Nodes           (slurm only)
MASTER_PORT=29500

# Paths — adjust to your cluster
VENV_PATH="venv"                              # virtualenv root (activate script)
RESULTS_DIR="./results/diffit_cifar10_naa"
DATA_ROOT="./data"

# Optional: CUDNN path (adjust or remove if not needed on your system)
# CUDNN_LIB="/home/omsharma07/rudra/repvit-project/.venv/lib/python3.10/site-packages/nvidia/cudnn/lib"

# --------------------------------------------------------------------------- #
# Model  (Appendix H.1 + I.1, CIFAR-10 image-space)
# --------------------------------------------------------------------------- #

BASE_CHANNELS=128
CHANNEL_MULT="1 2 2"       # → 128, 256, 256 channel dims
NUM_BLOCKS="2 2 2"         # L1=L2=L3=2 DiffiT ResBlocks per stage
NUM_HEADS=4
WINDOW_SIZE=4              # paper: window size 4 for CIFAR-10 (Fig. 8)
TEMB_DIM=512               # paper Table 3 optimal: time dim = 512

# --------------------------------------------------------------------------- #
# EDM loss hyper-parameters  (Karras et al. 2022 CIFAR-10 defaults)
# --------------------------------------------------------------------------- #

P_MEAN=-1.2
P_STD=1.2
SIGMA_DATA=0.5

# --------------------------------------------------------------------------- #
# Training  (Appendix I.1)
# --------------------------------------------------------------------------- #

GLOBAL_BATCH=512           # paper: batch size 512
LR=1e-3                    # paper: lr = 1e-3 for CIFAR-10
BETA1=0.9
BETA2=0.999
EPS_ADAM=1e-8
WEIGHT_DECAY=0.0
GRAD_CLIP=1.0
EMA_DECAY=0.9999           # paper latent space (I.2): 0.9999

# total_kimg = total_iters * global_batch / 1000
# Paper: 200k iterations × 512 = 102 400k images = 102400 kimg
# Set TOTAL_KIMG=102400 for a full paper run.
# Set TOTAL_KIMG=512 for a quick 1000-iteration smoke-test.
TOTAL_KIMG=51200

# AMP (off by default; paper trains fp32 on A100s)
FP16=""                    # set to "--fp16" to enable

# --------------------------------------------------------------------------- #
# Sampling  (Appendix I.1)
# --------------------------------------------------------------------------- #

SAMPLE_STEPS=18            # paper: 18 EDM deterministic steps for CIFAR-10
SAMPLE_N=64                # images in each eval grid

# --------------------------------------------------------------------------- #
# Logging / checkpointing cadence
# --------------------------------------------------------------------------- #

LOG_EVERY=100              # log loss every N steps
CKPT_KIMG=1000             # checkpoint every 1000 kimg  (~976 iters at bs512)
SAMPLE_KIMG=500            # sample grid every 500 kimg  (~488 iters at bs512)
NUM_WORKERS=4

# --------------------------------------------------------------------------- #
# Optional resume
# --------------------------------------------------------------------------- #

RESUME="./results/diffit_cifar10_naa/ckpt_0011699.pt"   # e.g. "./results/diffit_cifar10/ckpt_0050000.pt"

# --------------------------------------------------------------------------- #
# Build argument list
# --------------------------------------------------------------------------- #

ARGS=(
    --data-root      "${DATA_ROOT}"
    --results-dir    "${RESULTS_DIR}"
    --base-channels  "${BASE_CHANNELS}"
    --channel-mult   ${CHANNEL_MULT}
    --num-blocks     ${NUM_BLOCKS}
    --num-heads      "${NUM_HEADS}"
    --window-size    "${WINDOW_SIZE}"
    --temb-dim       "${TEMB_DIM}"
    --P-mean         "${P_MEAN}"
    --P-std          "${P_STD}"
    --sigma-data     "${SIGMA_DATA}"
    --total-kimg     "${TOTAL_KIMG}"
    --global-batch   "${GLOBAL_BATCH}"
    --lr             "${LR}"
    --beta1          "${BETA1}"
    --beta2          "${BETA2}"
    --eps            "${EPS_ADAM}"
    --weight-decay   "${WEIGHT_DECAY}"
    --grad-clip      "${GRAD_CLIP}"
    --ema-decay      "${EMA_DECAY}"
    --log-every      "${LOG_EVERY}"
    --ckpt-kimg      "${CKPT_KIMG}"
    --sample-kimg    "${SAMPLE_KIMG}"
    --num-workers    "${NUM_WORKERS}"
    --sample-steps   "${SAMPLE_STEPS}"
    --sample-n       "${SAMPLE_N}"
)

[[ -n "${FP16}"   ]] && ARGS+=(${FP16})
[[ -n "${RESUME}" ]] && ARGS+=(--resume "${RESUME}")

# --------------------------------------------------------------------------- #
# Activate environment
# --------------------------------------------------------------------------- #

source "${VENV_PATH}/bin/activate"

# Uncomment below if your cluster needs explicit CUDNN path
# export LD_LIBRARY_PATH="${CUDNN_LIB}:${LD_LIBRARY_PATH:-}"

mkdir -p "${RESULTS_DIR}"

echo "======================================================================"
echo "  Image-space DiffiT | CIFAR-10 | EDM framework"
echo "  MODE         : ${MODE}"
echo "  TOTAL KIMG   : ${TOTAL_KIMG}  (~$(( TOTAL_KIMG * 1000 / GLOBAL_BATCH )) iters)"
echo "  GLOBAL BATCH : ${GLOBAL_BATCH}"
echo "  LR           : ${LR}"
echo "  RESULTS DIR  : ${RESULTS_DIR}"
echo "======================================================================"

# --------------------------------------------------------------------------- #
# Launch
# --------------------------------------------------------------------------- #

case "${MODE}" in

  # ----------------------------------------------------------------- single
  single)
    echo "Single-GPU mode"
    export CUDA_VISIBLE_DEVICES=0
    torchrun \
        --standalone \
        --nproc_per_node=1 \
        --master_addr=127.0.0.1 \
        --master_port="${MASTER_PORT}" \
        train.py "${ARGS[@]}"
    ;;

  # ------------------------------------------------------------------ multi
  multi)
    echo "Multi-GPU mode  (${NUM_GPUS} GPUs)"
    torchrun \
        --standalone \
        --nproc_per_node="${NUM_GPUS}" \
        --master_addr=127.0.0.1 \
        --master_port="${MASTER_PORT}" \
        train.py "${ARGS[@]}"
    ;;

  # ------------------------------------------------------------------ slurm
  slurm)
    echo "SLURM mode  (${NUM_NODES} × ${NUM_GPUS} GPUs)"
    sbatch <<EOF
#!/usr/bin/env bash
#SBATCH --job-name=diffit_cifar10
#SBATCH --nodes=${NUM_NODES}
#SBATCH --ntasks-per-node=${NUM_GPUS}
#SBATCH --gres=gpu:${NUM_GPUS}
#SBATCH --cpus-per-task=${NUM_WORKERS}
#SBATCH --time=72:00:00
#SBATCH --output=${RESULTS_DIR}/slurm_%j.log

source "${VENV_PATH}/bin/activate"

export MASTER_ADDR=\$(scontrol show hostnames \$SLURM_JOB_NODELIST | head -n 1)
export MASTER_PORT=${MASTER_PORT}

srun torchrun \
    --nnodes="${NUM_NODES}" \
    --nproc_per_node="${NUM_GPUS}" \
    --rdzv_id=\$SLURM_JOB_ID \
    --rdzv_backend=c10d \
    --rdzv_endpoint=\${MASTER_ADDR}:\${MASTER_PORT} \
    train.py ${ARGS[@]}
EOF
    ;;

  *)
    echo "ERROR: Unknown MODE '${MODE}'. Choose: single | multi | slurm"
    exit 1
    ;;
esac