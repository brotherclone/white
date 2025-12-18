#!/bin/bash
# RunPod Setup Script
# Prepares environment and data for training

set -e

echo "🚀 RAINBOW PIPELINE - RUNPOD SETUP"
echo "======================================"

# Check we're on RunPod
if [ ! -d "/workspace" ]; then
    echo "❌ ERROR: Not running on RunPod (no /workspace directory)"
    exit 1
fi

cd /workspace

# Install/update training code
echo ""
echo "📦 Installing training code..."
if [ -d "rainbow-training" ]; then
    echo "  Updating existing code..."
    cd rainbow-training
    git pull
else
    echo "  Cloning repository..."
    # Replace with your actual repo URL
    git clone https://github.com/yourusername/rainbow-training.git
    cd rainbow-training
fi

# Install dependencies
echo ""
echo "📚 Installing dependencies..."
pip install -q -r requirements.txt

# Check for data
echo ""
echo "📊 Checking for training data..."
if [ ! -f "/workspace/data/base_manifest_db.parquet" ]; then
    echo "⚠️  WARNING: Training data not found!"
    echo ""
    echo "Please upload data to /workspace/data/"
    echo "  1. Use RunPod web interface file upload"
    echo "  2. Or use rsync: rsync -avz /local/data/ root@<pod-ip>:/workspace/data/"
    echo "  3. Or use SCP: scp base_manifest_db.parquet root@<pod-ip>:/workspace/data/"
    echo ""
    echo "After uploading data, run: bash train.sh"
    exit 1
fi

echo "✓ Found training data"

# Check GPU
echo ""
echo "🎮 Checking GPU..."
if ! nvidia-smi &> /dev/null; then
    echo "❌ ERROR: No GPU detected!"
    exit 1
fi

nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader
echo "✓ GPU detected"

# Setup W&B (if API key provided)
if [ ! -z "$WANDB_API_KEY" ]; then
    echo ""
    echo "📈 Setting up Weights & Biases..."
    wandb login "$WANDB_API_KEY"
    echo "✓ W&B configured"
else
    echo ""
    echo "⚠️  W&B API key not set. Logging disabled."
    echo "   To enable: export WANDB_API_KEY=your_key"
fi

echo ""
echo "======================================"
echo "✅ SETUP COMPLETE"
echo "======================================"
echo ""
echo "To start training:"
echo "  bash train.sh"
echo ""
echo "To test setup:"
echo "  python test_setup.py --manifest /workspace/data/base_manifest_db.parquet"