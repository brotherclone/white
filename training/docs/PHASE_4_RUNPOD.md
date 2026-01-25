# Phase 4 Regression Training - RunPod Setup Guide

## Quick Start

```bash
# 1. Upload your data to /workspace/data/
#    - base_manifest_db.parquet (required)
#    - training_data_embedded.parquet (optional, 69GB)

# 2. Upload runpod_train_phase4.py to /workspace/

# 3. Run training
python /workspace/runpod_train_phase4.py

# 4. Get your trained model
#    Output: /workspace/output/phase4_best.pt
```

## Detailed Setup

### 1. Create RunPod Instance

**Recommended GPU:** RTX 4090 or A100

**Minimum specs:**
- 24GB VRAM
- 50GB storage
- PyTorch template

### 2. Upload Data

Your data should be in `/workspace/data/`:

```
/workspace/data/
├── base_manifest_db.parquet       # Required (892 segments, ~4MB)
└── training_data_embedded.parquet # Optional (69GB with embeddings)
```

**Option A: With embeddings (faster training)**
- Upload the 69GB embedded parquet
- Training will use pre-computed embeddings

**Option B: Without embeddings (slower but works)**
- Just upload base_manifest_db.parquet
- Embeddings computed on-the-fly

### 3. Upload Training Script

Upload `runpod_train_phase4.py` to `/workspace/`

### 4. Run Training

```bash
# SSH into your RunPod instance
ssh root@<your-runpod-ip> -i ~/.ssh/id_runpod

# Navigate to workspace
cd /workspace

# Run training (with dependency installation)
python runpod_train_phase4.py

# OR skip deps if already installed
python runpod_train_phase4.py --skip-deps
```

## What Happens

The script will:

1. ✅ Install dependencies (torch, transformers, wandb, etc.)
2. ✅ Check GPU availability
3. ✅ Verify data files exist
4. ✅ Create training directory: `/workspace/phase4_training/`
5. ✅ Generate training script with soft target generation
6. ✅ Train regression model for 50 epochs (or early stopping)
7. ✅ Save best model to `/workspace/output/phase4_best.pt`
8. ✅ Log to Weights & Biases

## Training Configuration

The config is in `/workspace/phase4_training/config_regression.yml`:

```yaml
# Key settings:
data:
  train_split: 0.8
  label_smoothing: 0.1

model:
  regression_head:
    outputs:
      temporal: 3      # Past/Present/Future
      spatial: 3       # Thing/Place/Person
      ontological: 3   # Imagined/Forgotten/Known
      confidence: 1    # Chromatic confidence

training:
  batch_size: 32
  learning_rate: 0.00002
  epochs: 50
  early_stopping:
    patience: 10
```

## Expected Training Time

**With embeddings (fast):**
- ~2-3 minutes per epoch
- ~30-50 minutes total

**Without embeddings (slow):**
- ~10-15 minutes per epoch  
- ~2-3 hours total

## Expected Results

Based on Phase 8 classification achieving 100% accuracy, Phase 4 should achieve:

- **Temporal mode accuracy**: ~95-100%
- **Spatial mode accuracy**: ~95-100%
- **Ontological mode accuracy**: ~95-100%
- **Album prediction accuracy**: ~95-100%

## Monitoring Training

**Weights & Biases:**
- Project: `rainbow-pipeline-regression`
- Metrics tracked:
  - Train/val loss
  - Per-dimension MAE
  - Per-dimension mode accuracy
  - Confidence MAE
  - Loss components

**Command line:**
```bash
# Watch training progress
tail -f /workspace/phase4_training/train.log
```

## Download Trained Model

```bash
# From your local machine
scp root@<runpod-ip>:/workspace/output/phase4_best.pt ./

# Or use RunPod's file browser
```

## Troubleshooting

### "CUDA not available"
- Make sure you selected a GPU template
- Check PyTorch version: `python -c "import torch; print(torch.cuda.is_available())"`

### "Data file not found"
- Verify data location: `ls -lh /workspace/data/`
- Make sure you uploaded to correct path

### "Out of memory"
- Reduce batch_size in config (try 16 or 8)
- Use gradient accumulation
- Close other programs using VRAM

### "Training too slow"
- Upload the 69GB embedded parquet for faster training
- Use a bigger GPU (A100 > RTX 4090 > RTX 3090)

## After Training

Once you have `phase4_best.pt`:

1. **Test locally** with `concept_validator.py`
2. **Integrate into LangGraph** workflow
3. **Validate White Agent concepts** automatically

## Cost Estimate

RunPod pricing (approximate):

- **RTX 4090**: ~$0.40/hour
- **A100 40GB**: ~$1.00/hour

Total cost:
- With embeddings: ~$0.50-1.00 (1-2 hours)
- Without embeddings: ~$1.00-3.00 (2-3 hours)

## Next Steps

After training completes:

1. Download `phase4_best.pt`
2. Test with `concept_validator.py`
3. Integrate into White Agent workflow
4. Generate validated concepts!

---

**Questions?** Check the code comments in `runpod_train_phase4.py`