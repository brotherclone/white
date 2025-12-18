# Rainbow Pipeline Training

Clean training pipeline for rebracketing classification. Designed for RunPod GPU deployment.

## Structure

```
training/
├── core/
│   └── pipeline.py          # Data loading and preprocessing
├── models/
│   ├── text_encoder.py      # DeBERTa wrapper
│   └── classifier.py        # Binary classifier head
├── config.yml               # Training configuration (RunPod)
├── train.py                 # Main training script
├── test_setup.py           # Verify setup before training
├── setup.sh                # RunPod environment setup
├── train.sh                # RunPod training wrapper
├── Dockerfile.runpod       # GPU container definition
├── DEPLOYMENT.md           # Full deployment workflow
└── requirements.txt         # Dependencies (Python 3.11)
```

## Quick Start

**See [DEPLOYMENT.md](DEPLOYMENT.md) for complete workflow.**

```bash
# 1. Upload to RunPod
rsync -avz training/ root@<pod-ip>:/workspace/training/
rsync -avz data/base_manifest_db.parquet root@<pod-ip>:/workspace/data/

# 2. SSH and setup
ssh root@<pod-ip>
cd /workspace/training
bash setup.sh

# 3. Test
python test_setup.py --manifest /workspace/data/base_manifest_db.parquet

# 4. Train
bash train.sh
```

## Configuration

Edit `config.yml` to customize:

- **Model:** Text encoder, pooling method, freeze layers
- **Data:** Train/val split, filtering, target column
- **Training:** Batch size, learning rate, epochs, early stopping
- **Logging:** Output directory, W&B integration

## Current Phase

**Phase 1:** Text-only binary classification
- **Task:** Predict `has_rebracketing_markers` (True/False)
- **Input:** Concept text only
- **Model:** DeBERTa-v3-base + MLP classifier
- **Goal:** Establish baseline, verify training pipeline

## Why RunPod-Only?

The training code requires Python 3.11 and GPU dependencies. Your main White project uses Python 3.13 and doesn't need ML dependencies. Keeping them separate avoids version conflicts.

**Cost:** ~$0.35 for complete training run (setup + test + 20 epochs)

## Next Phases

**Phase 2:** Multi-class classification
- Predict `rebracketing_type` (spatial/perceptual/causal/temporal)
- Handle class imbalance with oversampling/weighting

**Phase 3:** Multimodal
- Add audio encoder (Wav2Vec2)
- Add MIDI encoder (custom transformer)
- Cross-attention fusion

**Phase 4:** Regression
- Predict `rebracketing_intensity` (continuous)
- Helps with more nuanced understanding

**Phase 5:** Temporal segmentation
- Segment-level predictions
- Time-aware modeling

## Testing Individual Components

These work in the RunPod environment:

```bash
# Test data pipeline
cd core && python pipeline.py

# Test text encoder
cd models && python text_encoder.py

# Test classifier
cd models && python classifier.py
```

## Troubleshooting

### CUDA out of memory
- Reduce `batch_size` in config.yml
- Increase `gradient_accumulation_steps`
- Disable `mixed_precision`

### Training too slow
- Increase `batch_size` if memory allows
- Ensure `mixed_precision: true`
- Check GPU utilization: `nvidia-smi`

### Poor convergence
- Increase `learning_rate`
- Disable `freeze_layers`
- Increase `epochs`
- Check class imbalance handling

## Class Imbalance Handling

Current dataset: ~95% True, ~5% False for `has_rebracketing_markers`

The pipeline automatically:
1. Computes `pos_weight` for BCEWithLogitsLoss
2. Upweights minority class in loss function

If still imbalanced, try:
- Oversample minority class
- Undersample majority class
- Use focal loss instead of BCE