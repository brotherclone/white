# Uncertainty Estimation

This document describes uncertainty estimation approaches for Rainbow Pipeline predictions.

## Overview

Uncertainty quantification helps identify:
- Low-confidence predictions requiring human review
- Hybrid/liminal concepts between modes
- Black Album candidates (diffuse across all dimensions)
- Out-of-distribution inputs

## Types of Uncertainty

### Aleatoric Uncertainty
**Data uncertainty** - inherent noise in the data that cannot be reduced with more training.

- Caused by: Label ambiguity, concept liminality, annotation disagreement
- Example: A concept genuinely between "Past" and "Present"
- Cannot be reduced by: More training data

### Epistemic Uncertainty
**Model uncertainty** - uncertainty due to limited training data or model capacity.

- Caused by: Insufficient training examples, novel input patterns
- Example: Concept type not seen during training
- Can be reduced by: More training data, larger models

## Estimation Methods

### 1. Monte Carlo Dropout

Apply dropout at inference time and aggregate multiple forward passes.

```python
from models import MCDropout

# Wrap existing model
mc_model = MCDropout(
    model=trained_model,
    dropout_rate=0.1,
    n_samples=10,
)

# Get prediction with uncertainty
output = mc_model(input_ids, attention_mask)
prediction = output.mean
uncertainty = output.epistemic_uncertainty
```

**How it works**:
1. Enable dropout during inference
2. Run N forward passes
3. Mean of outputs = prediction
4. Variance of outputs = epistemic uncertainty

**Advantages**:
- Simple to implement
- No architecture changes needed
- Works with any model

**Disadvantages**:
- N times slower at inference
- Only captures epistemic uncertainty

**Recommended settings**:
```yaml
uncertainty:
  method: mc_dropout
  dropout_rate: 0.1
  n_samples: 10
```

### 2. Ensemble Methods

Train multiple models and measure prediction disagreement.

```python
from models import EnsemblePredictor

# Load multiple trained models
ensemble = EnsemblePredictor.from_checkpoints(
    model_class=MultiTaskRainbowModel,
    checkpoint_paths=[
        "checkpoints/model_seed1.pt",
        "checkpoints/model_seed2.pt",
        "checkpoints/model_seed3.pt",
    ],
    model_kwargs={"num_classes": 9},
)

# Get ensemble prediction
output = ensemble(input_ids, attention_mask)
prediction = output.mean
uncertainty = output.epistemic_uncertainty
```

**How it works**:
1. Train M models with different initializations
2. At inference, get predictions from all models
3. Mean = ensemble prediction
4. Variance = epistemic uncertainty

**Advantages**:
- More robust uncertainty estimates
- Better calibrated than single models
- Captures diverse failure modes

**Disadvantages**:
- M times training cost
- M times storage requirements
- M times inference cost (can be parallelized)

**Recommended settings**:
```yaml
uncertainty:
  method: ensemble
  n_models: 5
  aggregation: mean  # or median
```

### 3. Evidential Deep Learning

Predict distribution parameters instead of point estimates.

```python
from models import EvidentialHead, EvidentialLoss

# Replace standard head with evidential head
evidential_head = EvidentialHead(
    input_dim=768,
    num_classes=3,
    task="classification",
)

# Training with evidential loss
loss_fn = EvidentialLoss(
    lambda_reg=0.1,
    annealing_epochs=10,
)

# Forward pass returns distribution parameters
output = evidential_head(embeddings)
probs = output.mean  # Expected probabilities
aleatoric = output.aleatoric_uncertainty
epistemic = output.epistemic_uncertainty
```

**How it works**:
For classification, predicts Dirichlet concentration parameters α:
- Expected probability: p = α / Σα
- Epistemic uncertainty: K / Σα (inverse of total evidence)
- Aleatoric uncertainty: entropy of expected categorical

**Advantages**:
- Single forward pass
- Separates aleatoric and epistemic uncertainty
- Theoretically grounded

**Disadvantages**:
- Requires architecture changes
- Special loss function needed
- Can be harder to train

**Recommended settings**:
```yaml
uncertainty:
  method: evidential
  lambda_reg: 0.1
  annealing_epochs: 10
```

### 4. Direct Variance Prediction

Train the model to directly predict variance alongside mean.

```python
from models import RainbowTableRegressionHead

head = RainbowTableRegressionHead(
    input_dim=768,
    predict_uncertainty=True,
)

output = head(embeddings)
temporal_mean = output.temporal_scores
temporal_var = output.temporal_uncertainty
```

**How it works**:
- Add variance output heads parallel to mean heads
- Train with negative log-likelihood loss
- Variance captures aleatoric uncertainty

**Advantages**:
- Single forward pass
- Captures heteroscedastic (input-dependent) uncertainty
- No architectural overhead

**Disadvantages**:
- Only captures aleatoric uncertainty
- Requires careful loss formulation

## Ontological Evidential Head

Specialized evidential head for Rainbow Table dimensions:

```python
from models import OntologicalEvidentialHead

head = OntologicalEvidentialHead(
    input_dim=768,
    hidden_dim=256,
)

outputs = head(embeddings)
# outputs["temporal"].mean  # Temporal probabilities
# outputs["temporal"].epistemic_uncertainty
# outputs["spatial"].aleatoric_uncertainty
# etc.
```

## Calibration

Well-calibrated uncertainty means predicted confidence matches actual accuracy.

### Expected Calibration Error (ECE)

```python
from models import compute_calibration_error

metrics = compute_calibration_error(
    predictions=model_outputs,
    targets=true_labels,
    uncertainties=predicted_uncertainties,
    n_bins=10,
)

print(f"ECE: {metrics['ece']:.3f}")
print(f"MCE: {metrics['mce']:.3f}")
```

### Temperature Scaling
Post-hoc calibration by scaling logits:

```python
# Find optimal temperature on validation set
calibrated_logits = logits / temperature
```

## Uncertainty Thresholds

### For Validation Gates

```yaml
validation:
  confidence_threshold: 0.7    # Below = REJECT
  dominant_threshold: 0.6      # Score for dominant mode
  hybrid_threshold: 0.15       # Margin for hybrid detection
  diffuse_threshold: 0.2       # Max deviation from uniform
  uncertainty_threshold: 0.8   # Max acceptable uncertainty
```

### Decision Logic

```python
if epistemic_uncertainty > 0.5:
    decision = "REJECT"  # Model is unsure
    reason = "high_epistemic_uncertainty"
elif aleatoric_uncertainty > 0.3:
    decision = "ACCEPT_HYBRID"  # Concept is genuinely liminal
    reason = "liminal_concept"
elif confidence < threshold:
    decision = "REJECT"
    reason = "low_confidence"
else:
    decision = "ACCEPT"
```

## Visualization

### Prediction Intervals

```python
from visualization import plot_prediction_intervals

fig = plot_prediction_intervals(
    predictions=means,
    uncertainties=stds,
    true_values=targets,
    confidence=0.95,
)
```

### Calibration Curves

```python
from visualization import plot_calibration_curve

fig = plot_calibration_curve(
    predicted_probs=probs,
    true_labels=labels,
    n_bins=10,
)
```

## Best Practices

1. **Use MC Dropout for quick experiments** - minimal changes needed
2. **Use ensembles for production** - best uncertainty quality
3. **Use evidential for research** - separates uncertainty types
4. **Always calibrate** - raw uncertainties often miscalibrated
5. **Validate uncertainty quality** - check calibration metrics
6. **Set thresholds on validation data** - don't use training data

## Example: Full Uncertainty Pipeline

```python
from models import MCDropout, compute_calibration_error
from validation import ConceptValidator

# 1. Wrap model with MC Dropout
mc_model = MCDropout(trained_model, n_samples=10)

# 2. Get predictions with uncertainty
output = mc_model(input_ids, attention_mask)

# 3. Check calibration
cal_metrics = compute_calibration_error(
    output.mean, targets, output.epistemic_uncertainty
)
print(f"ECE: {cal_metrics['ece']:.3f}")

# 4. Use in validation
validator = ConceptValidator(
    model_path="model.pt",
    uncertainty_threshold=0.5,
)

result = validator.validate_concept("A remembered dream")
if result.uncertainty_estimates["epistemic"] > 0.5:
    print("High model uncertainty - needs review")
```

## Method Comparison

| Method | Speed | Aleatoric | Epistemic | Calibration | Complexity |
|--------|-------|-----------|-----------|-------------|------------|
| MC Dropout | Slow | No | Yes | Medium | Low |
| Ensemble | Slow | No | Yes | Good | Medium |
| Evidential | Fast | Yes | Yes | Varies | High |
| Direct Var | Fast | Yes | No | Medium | Low |

## Recommended Configuration

```yaml
model:
  regression_head:
    predict_uncertainty: true

uncertainty:
  method: mc_dropout
  n_samples: 10
  dropout_rate: 0.1

  # Evidential (alternative)
  # method: evidential
  # lambda_reg: 0.1

validation:
  uncertainty_threshold: 0.5
  flag_high_uncertainty: true
```
