# Multi-Task Learning Strategies

This document describes the multi-task learning framework for combining classification and regression in the Rainbow Pipeline.

## Overview

The Rainbow Pipeline supports joint training of:
1. **Classification**: Discrete album assignment (9 classes)
2. **Regression**: Continuous ontological scores (10 outputs)

Multi-task learning enables shared representation learning while optimizing for both objectives.

## Architecture

### Shared Encoder
The text encoder (e.g., DeBERTa) produces embeddings shared across tasks:

```
Text → Encoder → Shared Embeddings (768-dim)
                      ↓
              ┌───────┴───────┐
              ↓               ↓
    Classification Head   Regression Head
              ↓               ↓
    Album Logits (9)    Ontological Scores (10)
```

### MultiTaskRainbowModel

```python
from models import MultiTaskRainbowModel

model = MultiTaskRainbowModel(
    text_encoder=encoder,
    num_classes=9,
    regression_targets=10,
    shared_dim=768,
)
```

## Loss Weighting Strategies

The combined loss balances classification and regression:

```
L_total = α * L_classification + β * L_regression
```

### Fixed Weighting
Static weights throughout training.

```yaml
multitask:
  strategy: fixed
  classification_weight: 1.0
  regression_weight: 0.5
```

**When to use**: Baseline, when task importance is known a priori.

### Uncertainty Weighting
Learn task weights based on homoscedastic uncertainty (Kendall et al., 2018).

```
L = (1/2σ₁²) * L₁ + (1/2σ₂²) * L₂ + log(σ₁) + log(σ₂)
```

```yaml
multitask:
  strategy: uncertainty
  initial_log_var_cls: 0.0
  initial_log_var_reg: 0.0
```

**When to use**: When optimal weighting is unknown, tasks have different scales.

### Gradient Normalization (GradNorm)
Dynamically adjust weights to balance gradient magnitudes.

```yaml
multitask:
  strategy: gradnorm
  alpha: 1.5  # Asymmetry parameter
```

**When to use**: When tasks have imbalanced gradient scales.

### Dynamic Weight Average (DWA)
Weight based on relative loss change rate.

```yaml
multitask:
  strategy: dwa
  temperature: 2.0
```

**When to use**: When tasks converge at different rates.

## Loss Functions

### Classification Loss
Standard cross-entropy for album prediction:

```python
L_cls = CrossEntropyLoss(logits, album_labels)
```

### Regression Loss Options

#### MSE Loss
Mean squared error for continuous targets:

```python
L_reg = MSELoss(predictions, targets)
```

**Best for**: When all errors should be treated equally.

#### Huber Loss
Robust to outliers, combines MSE and MAE:

```python
L_reg = HuberLoss(predictions, targets, delta=1.0)
```

**Best for**: When data contains outliers.

#### KL Divergence Loss
For probability distributions:

```python
L_reg = KLDivLoss(log_softmax(predictions), targets)
```

**Best for**: When targets are probability distributions.

### Ontological Regression Loss
Combined loss for all three dimensions:

```python
from core import OntologicalRegressionLoss

loss_fn = OntologicalRegressionLoss(
    temporal_weight=1.0,
    spatial_weight=1.0,
    ontological_weight=1.0,
    confidence_weight=0.5,
    distribution_loss="kl",  # or "mse", "jsd"
)
```

## Sequential Training

Train tasks in phases for better convergence:

### Phase 1: Classification Only
```python
trainer = SequentialTrainer(
    model=model,
    phase1_epochs=5,
    phase1_tasks=["classification"],
)
```

### Phase 2: Joint Training
```python
trainer.train_phase2(
    epochs=10,
    tasks=["classification", "regression"],
)
```

### Phase 3: Regression Fine-tuning
```python
trainer.train_phase3(
    epochs=5,
    tasks=["regression"],
    freeze_encoder=True,
)
```

## Task-Specific Learning Rates

Different learning rates per component:

```yaml
training:
  optimizer:
    encoder_lr: 1e-5
    classification_head_lr: 1e-4
    regression_head_lr: 1e-4
```

Implementation using parameter groups:

```python
optimizer = AdamW([
    {"params": model.encoder.parameters(), "lr": 1e-5},
    {"params": model.classification_head.parameters(), "lr": 1e-4},
    {"params": model.regression_head.parameters(), "lr": 1e-4},
])
```

## Gradient Accumulation

For effective larger batch sizes:

```yaml
training:
  batch_size: 16
  gradient_accumulation_steps: 4  # Effective batch = 64
```

## Task Balancing Strategies

### Loss Scaling
Scale losses to similar magnitudes before weighting:

```python
# Normalize by running average
cls_loss_scaled = cls_loss / cls_loss_ema
reg_loss_scaled = reg_loss / reg_loss_ema
```

### Gradient Clipping
Prevent one task from dominating:

```yaml
training:
  max_grad_norm: 1.0
  per_task_grad_clip: true
```

### Task Dropout
Randomly skip tasks during training:

```yaml
multitask:
  task_dropout: 0.1  # 10% chance to skip each task
```

## Monitoring Multi-Task Training

### Key Metrics to Track
1. Per-task loss curves
2. Gradient magnitudes per task
3. Weight evolution (for dynamic strategies)
4. Task-specific validation metrics

### Early Stopping
Stop based on combined metric:

```python
combined_metric = (
    0.5 * album_accuracy +
    0.3 * (1 - temporal_mae) +
    0.2 * (1 - ontological_mae)
)
```

## Best Practices

1. **Start with fixed weights** to establish baselines
2. **Monitor gradient ratios** between tasks
3. **Use uncertainty weighting** when task scales differ
4. **Consider sequential training** if joint training struggles
5. **Validate on both tasks** to detect task interference

## Example Configuration

```yaml
model:
  text_encoder:
    model_name: microsoft/deberta-v3-base

  classifier:
    num_classes: 9
    hidden_dims: [512, 256]
    dropout: 0.3

  regression_head:
    num_targets: 10
    hidden_dims: [256, 128]
    output_activation: null  # Softmax applied per-dimension
    predict_uncertainty: true

multitask:
  enabled: true
  strategy: uncertainty
  classification_weight: 1.0
  regression_weight: 1.0

training:
  epochs: 20
  batch_size: 32
  learning_rate: 2e-5
  encoder_lr: 1e-5
  warmup_ratio: 0.1

  regression_loss:
    type: ontological
    temporal_weight: 1.0
    spatial_weight: 1.0
    ontological_weight: 1.0
    confidence_weight: 0.5
```

## Troubleshooting

| Issue | Possible Cause | Solution |
|-------|---------------|----------|
| Classification accuracy drops | Regression dominates gradients | Increase classification weight |
| Regression MAE plateaus | Insufficient capacity | Add hidden layers |
| Both tasks underperform | Task interference | Try sequential training |
| Unstable training | Gradient scale mismatch | Use gradient normalization |
| Slow convergence | Learning rate too low | Use task-specific LRs |
