#!/bin/bash

source venv/bin/activate
export LD_LIBRARY_PATH=/home/omsharma07/rudra/repvit-project/.venv/lib/python3.10/site-packages/nvidia/cudnn/lib:$LD_LIBRARY_PATH
export CUDA_VISIBLE_DEVICES=0

torchrun \
    --nproc_per_node=1 \
    --master_addr=127.0.0.1 \
    --master_port=6000 \
    sample.py \
    --log_dir ./log_dir/256_1 \
    --cfg_scale 4.6 \
    --model_path ./diffit_256.safetensors \
    --image_size 256 \
    --model Diffit \
    --num_sampling_steps 250 \
    --num_samples 50000 \
    --batch_size 64 \
    --cfg_cond True \
    --class_cond True \
    --vae_decoder ema