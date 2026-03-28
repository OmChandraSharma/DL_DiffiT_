#!/usr/bin/env bash
# =============================================================================
# sample_cifar10.sh — Generate 50k samples from trained image-space DiffiT
#                     (CIFAR-10, EDM framework) for FID evaluation.
# =============================================================================

set -euo pipefail

# --------------------------------------------------------------------------- #
# ▶  EDIT THESE                                                               #
# --------------------------------------------------------------------------- #

VENV_PATH="venv"
CKPT="./results/diffit_cifar10_naa/ckpt_0011699.pt"   # path to your checkpoint
LOG_DIR="./log_dir/cifar10_1"                      # where .npz will be saved

export CUDA_VISIBLE_DEVICES=0
export LD_LIBRARY_PATH=/home/omsharma07/rudra/repvit-project/.venv/lib/python3.10/site-packages/nvidia/cudnn/lib:${LD_LIBRARY_PATH:-}

# --------------------------------------------------------------------------- #
# Sampling config  (Appendix I.1)
# --------------------------------------------------------------------------- #

NUM_SAMPLES=50000   # FID-50K
BATCH_SIZE=256      # reduce to 128 if OOM
NUM_STEPS=18        # paper: 18 EDM deterministic steps for CIFAR-10
SIGMA_DATA=0.5
SEED=42

# --------------------------------------------------------------------------- #
# Launch
# --------------------------------------------------------------------------- #

source "${VENV_PATH}/bin/activate"

mkdir -p "${LOG_DIR}"

echo "======================================================================"
echo "  Image-space DiffiT(Noise Aware) CIFAR-10 — Sampling"
echo "  Checkpoint  : ${CKPT}"
echo "  Num samples : ${NUM_SAMPLES}"
echo "  Batch size  : ${BATCH_SIZE}"
echo "  EDM steps   : ${NUM_STEPS}"
echo "  Output dir  : ${LOG_DIR}"
echo "======================================================================"

python sample_cifar10.py \
    --ckpt          "${CKPT}"        \
    --log_dir       "${LOG_DIR}"     \
    --num_samples   "${NUM_SAMPLES}" \
    --batch_size    "${BATCH_SIZE}"  \
    --num_steps     "${NUM_STEPS}"   \
    --sigma_data    "${SIGMA_DATA}"  \
    --seed          "${SEED}"

echo ""
echo "======================================================================"
echo "  Sampling complete."
echo "  Now run the evaluator:"
echo ""
echo "  CUDA_VISIBLE_DEVICES=\"\" python evaluator.py \\"
echo "      ./VIRTUAL_cifar10_32x32.npz \\"
echo "      ${LOG_DIR}/samples_50000x32x32x3.npz \\"
echo "      --log_dir ${LOG_DIR}"
echo "======================================================================"