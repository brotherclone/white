#!/bin/bash
# Train on RunPod GPU
# Simple wrapper for python train.py with sensible defaults

set -e

echo "🚀 STARTING TRAINING ON RUNPOD"
echo "======================================"

# Check we're in the right place
if [ ! -f "train.py" ]; then
    echo "❌ ERROR: train.py not found. Are you in the training directory?"
    exit 1
fi

# Check for data
if [ ! -f "/workspace/data/base_manifest_db.parquet" ]; then
    echo "❌ ERROR: Training data not found at /workspace/data/base_manifest_db.parquet"
    echo "   Run setup.sh first and upload your data."
    exit 1
fi

# Config file (default to GPU config)
CONFIG="${1:-config.yml}"

if [ ! -f "$CONFIG" ]; then
    echo "❌ ERROR: Config not found: $CONFIG"
    exit 1
fi

echo "⚙️  Using config: $CONFIG"

# Check GPU
echo ""
echo "🎮 GPU Status:"
nvidia-smi --query-gpu=index,name,memory.free,utilization.gpu --format=csv,noheader
echo ""

# Confirm start
read -p "Start training? [Y/n] " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]*$ ]] && [[ ! -z $REPLY ]]; then
    echo "Cancelled"
    exit 0
fi

# Start training
echo ""
echo "🔥 Training starting..."
echo "======================================"

python train.py --config "$CONFIG"

# Done
EXIT_CODE=$?
echo ""
echo "======================================"
if [ $EXIT_CODE -eq 0 ]; then
    echo "✅ TRAINING COMPLETED"
    echo ""
    echo "📊 Outputs saved to: ./output/"
    echo "📈 View history: cat ./output/history.json"
    echo "🔍 Best checkpoint: ./output/checkpoint_best.pt"
else
    echo "❌ TRAINING FAILED (exit code: $EXIT_CODE)"
fi
echo "======================================"

exit $EXIT_CODE