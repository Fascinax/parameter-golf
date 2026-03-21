#!/bin/bash
# =============================================================================
# RunPod One-Time Setup
# =============================================================================
# Usage: After SSH into your RunPod, run:
#   bash runpod_setup.sh
#
# Prerequisites: RunPod PyTorch template (e.g. runpod/pytorch:2.4.0-py3.11-cuda12.4.1)
# Recommended: A100 80GB ($1.10/h) or H100 ($3.50/h)
#              Multi-GPU pod for parallel experiments (e.g. 2xA100, 4xA100)
# =============================================================================

set -euo pipefail

REPO_URL="https://github.com/Fascinax/parameter-golf.git"
BRANCH="autoresearch/improve-bpb"
WORKDIR="/workspace/parameter-golf"

echo "=== [1/5] Cloning repo ==="
if [ -d "$WORKDIR" ]; then
    echo "Repo already exists, pulling latest..."
    cd "$WORKDIR"
    git fetch origin
    git checkout "$BRANCH"
    git pull origin "$BRANCH"
else
    git clone --branch "$BRANCH" "$REPO_URL" "$WORKDIR"
    cd "$WORKDIR"
fi

echo "=== [2/5] Installing Python dependencies ==="
pip install -q sentencepiece zstandard numpy

echo "=== [3/5] Downloading dataset ==="
if [ -d "$WORKDIR/data/datasets/fineweb10B_sp1024" ] && \
   [ -f "$WORKDIR/data/datasets/fineweb10B_sp1024/fineweb_val_000000.bin" ]; then
    echo "Dataset already present, skipping download."
else
    python3 data/cached_challenge_fineweb.py --variant sp1024 --train-shards 1
fi

echo "=== [4/5] Verifying GPU ==="
python3 -c "
import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
n = torch.cuda.device_count()
print(f'GPU count: {n}')
for i in range(n):
    props = torch.cuda.get_device_properties(i)
    print(f'  GPU {i}: {props.name} — {props.total_mem // 1024**2} MiB')
"

echo "=== [5/5] Quick smoke test (5 steps) ==="
ITERATIONS=5 TRAIN_BATCH_TOKENS=65536 VAL_LOSS_EVERY=5 WARMUP_STEPS=2 \
MAX_WALLCLOCK_SECONDS=0 TRAIN_LOG_EVERY=1 \
python3 train_gpt.py 2>&1 | tail -20

echo ""
echo "=== Setup complete! ==="
echo "GPU count: $(python3 -c 'import torch; print(torch.cuda.device_count())')"
echo ""
echo "To run the autoresearch loop:"
echo "  python3 runpod_autoresearch.py"
echo ""
echo "To run with parallel experiments (uses all available GPUs):"
echo "  python3 runpod_autoresearch.py --parallel"
