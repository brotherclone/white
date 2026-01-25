# Continuous Rebracketing Metrics

This document describes the regression metrics used for Rainbow Table ontological prediction.

## Overview

The Rainbow Pipeline predicts continuous scores across three ontological dimensions, enabling nuanced concept placement beyond discrete album assignments. Each dimension outputs a probability distribution over three modes.

## Ontological Dimensions

### Temporal Dimension
Describes when a concept exists relative to the perceiver's present.

| Mode | Index | Description |
|------|-------|-------------|
| Past | 0 | Memory, nostalgia, historical reference |
| Present | 1 | Current moment, immediate experience |
| Future | 2 | Anticipation, prediction, aspiration |

### Spatial Dimension
Describes the type of entity the concept represents.

| Mode | Index | Description |
|------|-------|-------------|
| Thing | 0 | Object, artifact, abstract concept |
| Place | 1 | Location, environment, setting |
| Person | 2 | Individual, character, identity |

### Ontological Dimension
Describes the epistemic status of the concept.

| Mode | Index | Description |
|------|-------|-------------|
| Imagined | 0 | Fictional, hypothetical, dreamed |
| Forgotten | 1 | Once known but lost, fading memory |
| Known | 2 | Factual, verified, certain |

## Output Format

The model outputs 10 continuous values:

```
[temporal_past, temporal_present, temporal_future,    # 3 values, softmax
 spatial_thing, spatial_place, spatial_person,        # 3 values, softmax
 onto_imagined, onto_forgotten, onto_known,           # 3 values, softmax
 chromatic_confidence]                                 # 1 value, sigmoid
```

Each dimension's three values sum to 1.0 (probability distribution).

## Regression Metrics

### Per-Target Metrics

#### Mean Absolute Error (MAE)
Average absolute difference between predicted and true values.

```
MAE = (1/n) * Σ|y_pred - y_true|
```

- Range: [0, ∞)
- Lower is better
- Interpretable in original units

#### Root Mean Squared Error (RMSE)
Square root of average squared errors. Penalizes large errors more heavily.

```
RMSE = sqrt((1/n) * Σ(y_pred - y_true)²)
```

- Range: [0, ∞)
- Lower is better
- More sensitive to outliers than MAE

#### R² (Coefficient of Determination)
Proportion of variance explained by the model.

```
R² = 1 - (SS_res / SS_tot)
```

- Range: (-∞, 1]
- Higher is better
- R² = 1 means perfect prediction
- R² = 0 means predicting the mean
- R² < 0 means worse than predicting the mean

### Correlation Metrics

#### Pearson Correlation
Measures linear relationship between predicted and true values.

```
r = cov(y_pred, y_true) / (std(y_pred) * std(y_true))
```

- Range: [-1, 1]
- |r| closer to 1 is better
- Sensitive to outliers

#### Spearman Correlation
Measures monotonic relationship using ranks. More robust to outliers.

```
ρ = Pearson(rank(y_pred), rank(y_true))
```

- Range: [-1, 1]
- |ρ| closer to 1 is better

### Distribution Metrics

#### Jensen-Shannon Divergence (JSD)
Symmetric measure of similarity between probability distributions.

```
JSD(P||Q) = 0.5 * KL(P||M) + 0.5 * KL(Q||M)
where M = 0.5 * (P + Q)
```

- Range: [0, 1] when using log base 2
- Lower is better
- JSD = 0 means identical distributions

#### Total Variation Distance
Maximum difference between probabilities across all events.

```
TV(P, Q) = 0.5 * Σ|P(x) - Q(x)|
```

- Range: [0, 1]
- Lower is better

#### Hellinger Distance
Geometric mean-based distance between distributions.

```
H(P, Q) = sqrt(0.5 * Σ(sqrt(P(x)) - sqrt(Q(x)))²)
```

- Range: [0, 1]
- Lower is better

## Calibration Metrics

### Expected Calibration Error (ECE)
Measures how well predicted probabilities match observed frequencies.

```
ECE = Σ (|B_m| / n) * |accuracy(B_m) - confidence(B_m)|
```

- Range: [0, 1]
- Lower is better
- ECE = 0 means perfectly calibrated

### Maximum Calibration Error (MCE)
Worst-case calibration error across all bins.

```
MCE = max_m |accuracy(B_m) - confidence(B_m)|
```

- Range: [0, 1]
- Lower is better

## Hybrid State Metrics

### State Classification
Each dimension is classified into one of three states:

| State | Condition | Meaning |
|-------|-----------|---------|
| Dominant | top_score ≥ 0.6 | Clear mode assignment |
| Hybrid | top - second ≤ 0.15 | Liminal/transitional |
| Diffuse | max_dev from uniform ≤ 0.2 | Black Album candidate |

### Hybrid Rate
Percentage of predictions with at least one hybrid dimension.

### Diffuse Rate
Percentage of predictions with all dimensions diffuse (Black Album).

## Album Prediction Metrics

### Album Accuracy
Percentage of correct album predictions.

### Album F1 Score
Harmonic mean of precision and recall, per album.

```
F1 = 2 * (precision * recall) / (precision + recall)
```

### Macro F1
Unweighted average of per-album F1 scores.

### Weighted F1
Support-weighted average of per-album F1 scores.

## Transmigration Metrics

### Transmigration Distance
Euclidean distance between ontological states.

```
d = sqrt(d_temporal² + d_spatial² + d_ontological²)
```

Where each dimension distance uses L2 norm between probability vectors.

### Feasibility Score
Inverse of distance scaled by confidence.

```
feasibility = confidence / (1 + distance)
```

## Recommended Metric Selection

| Use Case | Primary Metrics |
|----------|-----------------|
| Training loss | JSD, KL Divergence |
| Validation | MAE, R², Album Accuracy |
| Production monitoring | ECE, Hybrid Rate |
| Research comparison | Macro F1, Spearman ρ |

## Example Output

```python
{
    "temporal": {
        "mae": 0.082,
        "rmse": 0.115,
        "r2": 0.891,
        "jsd": 0.034
    },
    "spatial": {
        "mae": 0.095,
        "rmse": 0.128,
        "r2": 0.867,
        "jsd": 0.041
    },
    "ontological": {
        "mae": 0.078,
        "rmse": 0.102,
        "r2": 0.903,
        "jsd": 0.029
    },
    "overall": {
        "album_accuracy": 0.847,
        "macro_f1": 0.812,
        "hybrid_rate": 0.156,
        "ece": 0.043
    }
}
```
