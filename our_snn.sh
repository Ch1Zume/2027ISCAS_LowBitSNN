#!/usr/bin/env bash
set -euo pipefail

# ── 纯 BP 训练 ──
# python our_snn.py train \
#     --T 10 \
#     --data-dir ./work/datasets \
#     -b 32 \
#     --num-workers 8 \
#     --opt adam \
#     --lr 1e-3 \
#     --momentum 0.9 \
#     --epochs 10 \
#     --amp bf16 \
#     --seed 42 \
#     --save-path best_snn.pth

# ── BP + STDP 混合训练 ──
python our_snn.py train \
    --T 10 \
    --data-dir ./work/datasets \
    -b 32 \
    --num-workers 8 \
    --opt adam \
    --lr 1e-2 \
    --momentum 0.9 \
    --epochs 50 \
    --amp bf16 \
    --seed 42 \
    --save-path best_snn_stdp.pth \
    --stdp \
    --lr-stdp 0.01 \
    --stdp-scale 1.0

# ── 评估 ──
# python our_snn.py eval \
#     --T 10 \
#     --data-dir ./work/datasets \
#     -b 32 \
#     --num-workers 8 \
#     --amp bf16 \
#     --checkpoint best_snn_stdp.pth
