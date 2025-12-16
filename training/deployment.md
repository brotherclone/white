                                         # Rainbow Pipeline: RunPod Deployment Guide

Clean, simple workflow: develop locally, train on GPU.

## Overview

Your main White project uses **Python 3.13** and stays clean. Training infrastructure uses **Python 3.11** and runs in a RunPod container. No version conflicts, no local testing complexity.

```
LOCAL (Mac)                RUNPOD (GPU)
───────────────            ────────────────
• Write code               • Test (2 min, $0.01)
• Python 3.13 OK           • Train (60 min, $0.30)
• No dependencies          • Python 3.11 + CUDA
• Just upload              • Full pipeline
```

---

## Quick Start

### 1. Create RunPod Instance

**Go to:** https://runpod.io

**Create Pod:**
- **Template:** PyTorch 2.1.1 (or Community/Runpod Pytorch)
- **GPU:** RTX 4090 or A40 (best price/performance)
  - 4090: ~$0.34/hr (fast)
  - A40: ~$0.29/hr (good value)
- **Disk:** 50GB (for models + data)
- **Region:** Whatever's cheapest available

**Start Pod** and note the SSH connection string.

### 2. Upload Code & Data

```bash
# From your Mac
cd /Volumes/LucidNonsense/White

# Upload training code
rsync -avz --exclude='output*' --exclude='__pycache__' \
  training/ root@<pod-ip>:/workspace/training/

# Upload training data
rsync -avz data/base_manifest_db.parquet \
  root@<pod-ip>:/workspace/data/
```

**Time:** ~5-10 minutes depending on connection

### 3. Setup RunPod Environment

```bash
# SSH into the pod
ssh root@<pod-ip>

# Run setup script
cd /workspace/training
bash setup.sh

# (Optional) Set W&B API key for experiment tracking
export WANDB_API_KEY=your_key_here
```

**Time:** ~2 minutes (installs dependencies, checks GPU)

### 4. Test the Pipeline

```bash
# Verify everything works before paying for training
python test_setup.py --manifest /workspace/data/base_manifest_db.parquet

# Expected output:
# ✓ PASS Data Loading
# ✓ PASS DataLoader  
# ✓ PASS Model
# ✓ PASS Training Step
# 🎉 All tests passed! Ready to train.
```

**Time:** ~2 minutes  
**Cost:** ~$0.01  

**If tests fail:** Debug here (cheap!), fix code locally, re-upload, test again.

### 5. Start Training

```bash
# Use the clean wrapper script
bash train.sh

# Or run directly
python train.py --config config.yml
```

**For long runs, use tmux:**
```bash
# Start tmux session
tmux new -s training

# Run training
bash train.sh

# Detach: Ctrl+B, then D
# Reattach later: tmux attach -t training
```

**Time:** ~30-60 minutes for 20 epochs  
**Cost:** ~$0.30 on A40

### 6. Monitor Progress

**Option 1: Weights & Biases** (if enabled)
- Dashboard: https://wandb.ai/your-username/rainbow-pipeline
- Real-time metrics, GPU usage, system stats

**Option 2: Direct monitoring**
```bash
# Watch GPU
watch -n 1 nvidia-smi

# View training progress (in another tmux pane/window)
tail -f /workspace/output/training.log

# Check current metrics
cat /workspace/output/history.json | python -m json.tool
```

### 7. Download Results

```bash
# From your Mac, after training completes
rsync -avz root@<pod-ip>:/workspace/output/ \
  /Volumes/LucidNonsense/White/training/output/

# Just get the best checkpoint
scp root@<pod-ip>:/workspace/output/checkpoint_best.pt \
  /Volumes/LucidNonsense/White/training/output/
```

### 8. Terminate Pod

**Don't forget!** Stop the pod when done to avoid hourly charges.

In RunPod dashboard: **Stop Pod** or **Terminate Pod**

---

## Configuration

Single config file: **config.yml**

Key settings for RunPod:

```yaml
data:
  manifest_path: "/workspace/data/base_manifest_db.parquet"

training:
  batch_size: 32        # Large batch for GPU
  epochs: 20            # Full training
  device: "cuda"        # GPU acceleration
  mixed_precision: true # Faster training

logging:
  output_dir: "/workspace/output"  # Persistent storage
  wandb:
    enabled: true       # Remote monitoring
```

To customize, edit locally and re-upload before training.

---

## Cost Breakdown

| Task | Time | Cost (A40 @ $0.29/hr) |
|------|------|----------------------|
| Setup | 2 min | $0.01 |
| Testing | 2 min | $0.01 |
| Training (20 epochs) | 60 min | $0.30 |
| **Total** | **~70 min** | **~$0.35** |

**Tips to save money:**
- ✅ Test locally before uploading (code checks)
- ✅ Use test_setup.py before training
- ✅ Use tmux (disconnect without stopping)
- ✅ Terminate pod immediately after downloading results
- ❌ Don't leave pod running idle
- ❌ Don't debug interactively on GPU

---

## Workflow Summary

```
┌─────────────────┐
│ Mac: Write Code │  Edit train.py, models/, etc.
│ (Python 3.13 OK)│  No dependencies needed
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Upload to RunPod│  rsync training/ and data/
│                 │  (5-10 minutes)
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ RunPod: Setup   │  bash setup.sh
│                 │  (2 minutes, $0.01)
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ RunPod: Test    │  python test_setup.py
│                 │  (2 minutes, $0.01)
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ RunPod: Train   │  bash train.sh
│                 │  (60 minutes, $0.30)
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Download Results│  rsync output/ back
│                 │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Terminate Pod   │  Stop charges!
│                 │
└─────────────────┘
```

---

## Troubleshooting

### "No such file or directory" errors
- Manifest path in config.yml must be `/workspace/data/base_manifest_db.parquet`
- Check file uploaded: `ls -lh /workspace/data/`

### Out of memory
- Reduce `batch_size` in config.yml (try 16 or 8)
- Increase `gradient_accumulation_steps` to maintain effective batch size

### Training too slow
- Check GPU usage: `nvidia-smi` (should be 90%+)
- Verify `mixed_precision: true` in config
- Increase `batch_size` if memory allows

### Tests fail
- Check Python version: `python --version` (should be 3.10 or 3.11)
- Reinstall dependencies: `pip install -r requirements.txt`
- Verify GPU: `nvidia-smi`

### Can't connect to pod
- Check pod is running in RunPod dashboard
- Verify SSH key added to RunPod account
- Try web terminal in RunPod interface

---

## Tips & Tricks

### Development Workflow

1. **Write code locally** (Python 3.13, no dependencies)
2. **Test syntax locally** (`python -m py_compile train.py`)
3. **Upload to RunPod** (rsync is incremental, fast)
4. **Test on GPU** (2 min, catches GPU-specific issues)
5. **Train** (60 min, actual work)
6. **Iterate** (download results, adjust config, repeat)

### Efficient Iteration

```bash
# One-liner: upload and test
rsync -avz training/ root@<pod-ip>:/workspace/training/ && \
  ssh root@<pod-ip> "cd /workspace/training && python test_setup.py --manifest /workspace/data/base_manifest_db.parquet"
```

### Using tmux Effectively

```bash
# Start training in tmux
tmux new -s training
bash train.sh

# Detach (training continues): Ctrl+B, then D
# Close terminal, go to bed, etc.

# Later: reattach
ssh root@<pod-ip>
tmux attach -t training

# Split panes to monitor
# Ctrl+B, then " (horizontal split)
# Ctrl+B, then % (vertical split)
# In second pane: watch -n 1 nvidia-smi
```

### Saving Intermediate Checkpoints

If training crashes or you want to resume later:

```bash
# Download latest checkpoint during training
scp root@<pod-ip>:/workspace/output/checkpoint_latest.pt \
  ./checkpoint_backup.pt

# Resume training (requires adding resume logic to train.py)
# For now: checkpoints saved every epoch automatically
```

---

## What's Next

Once this baseline training works:

**Phase 2:** Multi-class classification
- Target: `rebracketing_type` (spatial/perceptual/causal/temporal)
- Handle severe class imbalance (88% spatial)
- Use oversampling or class weights

**Phase 3:** Add audio encoder
- Wav2Vec2 for audio features
- Cross-attention fusion with text
- Requires uploading audio files (~10GB)

**Phase 4:** Add MIDI encoder
- Custom transformer for MIDI sequences
- Full multimodal fusion
- Complete the INFORMATION → TIME → SPACE pipeline

**Phase 5:** Temporal segmentation
- Segment-level predictions
- Time-aware modeling
- Use the temporal segmentation examples

---

## No More Shell Script Chaos

One training workflow:
- `setup.sh` - Install dependencies, verify environment
- `train.sh` - Run training with GPU config
- `test_setup.py` - Verify everything works

That's it. Clean, simple, professional. 🎯