# Example Configurations

This document provides example configurations for regression and multi-task training in the Rainbow Pipeline.

## Basic Regression Training

Minimal configuration for ontological regression:

```yaml
# config_regression_basic.yml

model:
  text_encoder:
    model_name: microsoft/deberta-v3-base
    freeze: false

  regression_head:
    num_targets: 10
    hidden_dims: [256, 128]
    dropout: 0.3
    output_activation: null
    predict_uncertainty: false

training:
  epochs: 20
  batch_size: 32
  learning_rate: 2e-5
  weight_decay: 0.01
  warmup_ratio: 0.1

  regression_loss:
    type: mse

evaluation:
  regression_metrics:
    - mae
    - rmse
    - r2

data:
  train_path: data/train.parquet
  val_path: data/val.parquet
  text_column: text
  target_columns:
    - temporal_past
    - temporal_present
    - temporal_future
    - spatial_thing
    - spatial_place
    - spatial_person
    - onto_imagined
    - onto_forgotten
    - onto_known
    - confidence
```

## Multi-Task Training

Joint classification and regression:

```yaml
# config_multitask.yml

model:
  text_encoder:
    model_name: microsoft/deberta-v3-base
    freeze: false

  classifier:
    num_classes: 9
    hidden_dims: [512, 256]
    dropout: 0.3

  regression_head:
    num_targets: 10
    hidden_dims: [256, 128]
    dropout: 0.3
    predict_uncertainty: true

multitask:
  enabled: true
  strategy: uncertainty  # fixed, uncertainty, gradnorm, dwa
  classification_weight: 1.0
  regression_weight: 1.0

training:
  epochs: 25
  batch_size: 32

  # Task-specific learning rates
  encoder_lr: 1e-5
  classification_head_lr: 1e-4
  regression_head_lr: 1e-4

  warmup_ratio: 0.1
  weight_decay: 0.01
  max_grad_norm: 1.0

  classification_loss:
    type: cross_entropy
    label_smoothing: 0.1

  regression_loss:
    type: ontological
    temporal_weight: 1.0
    spatial_weight: 1.0
    ontological_weight: 1.0
    confidence_weight: 0.5
    distribution_loss: kl

evaluation:
  classification_metrics:
    - accuracy
    - f1_macro
    - f1_weighted
  regression_metrics:
    - mae
    - rmse
    - r2
    - jsd
  calibration_metrics:
    - ece
    - mce

data:
  train_path: data/train.parquet
  val_path: data/val.parquet
  text_column: text
  label_column: album
  target_columns:
    - temporal_past
    - temporal_present
    - temporal_future
    - spatial_thing
    - spatial_place
    - spatial_person
    - onto_imagined
    - onto_forgotten
    - onto_known
    - confidence
```

## Ontological Regression with Soft Targets

Training with label-smoothed soft targets:

```yaml
# config_soft_targets.yml

model:
  text_encoder:
    model_name: microsoft/deberta-v3-base

  regression_head:
    num_targets: 10
    hidden_dims: [256, 128]
    dropout: 0.3

soft_targets:
  enabled: true
  label_smoothing: 0.1

  temporal_context:
    enabled: true
    window_size: 3
    weight: 0.3

  black_album:
    confidence: 0.0
    uniform_targets: true

training:
  epochs: 20
  batch_size: 32
  learning_rate: 2e-5

  regression_loss:
    type: kl_divergence
    reduction: batchmean

data:
  train_path: data/train.parquet
  val_path: data/val.parquet
  text_column: text

  # Discrete labels to convert to soft targets
  label_columns:
    temporal: temporal_mode
    spatial: spatial_mode
    ontological: ontological_mode

  track_id_column: track_id  # For temporal context
```

## Sequential Training

Phase-based training for complex multi-task:

```yaml
# config_sequential.yml

model:
  text_encoder:
    model_name: microsoft/deberta-v3-base

  classifier:
    num_classes: 9
    hidden_dims: [512, 256]

  regression_head:
    num_targets: 10
    hidden_dims: [256, 128]
    predict_uncertainty: true

sequential_training:
  enabled: true

  phase1:
    name: classification_only
    epochs: 5
    tasks: [classification]
    learning_rate: 2e-5
    freeze_encoder: false

  phase2:
    name: joint_training
    epochs: 15
    tasks: [classification, regression]
    learning_rate: 1e-5
    classification_weight: 0.7
    regression_weight: 0.3

  phase3:
    name: regression_finetune
    epochs: 5
    tasks: [regression]
    learning_rate: 5e-6
    freeze_encoder: true
    freeze_classifier: true

training:
  batch_size: 32
  weight_decay: 0.01
  max_grad_norm: 1.0
```

## Evidential Deep Learning

Uncertainty-aware training with evidential heads:

```yaml
# config_evidential.yml

model:
  text_encoder:
    model_name: microsoft/deberta-v3-base

  evidential_head:
    enabled: true
    hidden_dim: 256

    # Separate evidential heads per dimension
    temporal:
      num_classes: 3
    spatial:
      num_classes: 3
    ontological:
      num_classes: 3

training:
  epochs: 25
  batch_size: 32
  learning_rate: 2e-5

  evidential_loss:
    lambda_reg: 0.1
    annealing_epochs: 10

uncertainty:
  method: evidential
  calibrate: true

evaluation:
  uncertainty_metrics:
    - ece
    - mce
    - coverage_at_90
```

## White Agent Integration

Configuration for validation API:

```yaml
# config_validation.yml

model:
  checkpoint_path: checkpoints/best_model.pt

  text_encoder:
    model_name: microsoft/deberta-v3-base

  regression_head:
    num_targets: 10
    predict_uncertainty: true

validation:
  # Thresholds
  confidence_threshold: 0.7
  dominant_threshold: 0.6
  hybrid_threshold: 0.15
  diffuse_threshold: 0.2
  uncertainty_threshold: 0.8

  # Validation gates
  gates:
    accept:
      min_confidence: 0.7
      all_dominant: true
    accept_hybrid:
      min_confidence: 0.5
      max_hybrid_dimensions: 2
    accept_black:
      max_confidence: 0.3
      all_diffuse: true
    reject:
      default: true

  # Cache settings
  cache:
    enabled: true
    ttl_seconds: 3600
    max_size: 10000

api:
  host: 0.0.0.0
  port: 8000
  workers: 4

album_mapping:
  Red: [past, thing, forgotten]
  Orange: [past, thing, imagined]
  Yellow: [present, place, imagined]
  Green: [present, person, known]
  Blue: [future, place, known]
  Indigo: [present, person, forgotten]
  Violet: [past, person, known]
  White: [present, person, known]
  Black: null  # Uniform distribution
```

## Production Configuration

Optimized for inference:

```yaml
# config_production.yml

model:
  checkpoint_path: checkpoints/production_v2.pt

  text_encoder:
    model_name: microsoft/deberta-v3-base

  regression_head:
    num_targets: 10
    predict_uncertainty: true

inference:
  device: cuda
  batch_size: 64
  max_length: 512

  # Optimizations
  fp16: true
  torch_compile: true

  # MC Dropout for uncertainty
  mc_dropout:
    enabled: true
    n_samples: 5
    dropout_rate: 0.1

validation:
  confidence_threshold: 0.7
  uncertainty_threshold: 0.5

  cache:
    enabled: true
    backend: redis
    ttl_seconds: 3600

api:
  host: 0.0.0.0
  port: 8000
  workers: 8
  timeout: 30

logging:
  level: INFO
  format: json

monitoring:
  prometheus:
    enabled: true
    port: 9090

  metrics:
    - request_latency
    - prediction_confidence
    - uncertainty_distribution
    - album_distribution
```

## Development Configuration

For local development and debugging:

```yaml
# config_dev.yml

model:
  text_encoder:
    model_name: microsoft/deberta-v3-small  # Smaller model
    freeze: true  # Faster iteration

  regression_head:
    num_targets: 10
    hidden_dims: [128]  # Smaller
    dropout: 0.1

training:
  epochs: 3
  batch_size: 8
  learning_rate: 1e-4

  # Fast iteration
  val_check_interval: 0.5
  limit_train_batches: 100
  limit_val_batches: 20

data:
  train_path: data/train_sample.parquet  # Subset
  val_path: data/val_sample.parquet

logging:
  level: DEBUG

debugging:
  detect_anomaly: true
  profiler: simple
```

## Hyperparameter Search

Configuration for hyperparameter optimization:

```yaml
# config_hparam_search.yml

model:
  text_encoder:
    model_name: microsoft/deberta-v3-base

  regression_head:
    num_targets: 10
    hidden_dims: ${hidden_dims}
    dropout: ${dropout}

training:
  epochs: 10
  batch_size: ${batch_size}
  learning_rate: ${learning_rate}

  regression_loss:
    type: ${loss_type}

# Optuna search space
hparam_search:
  sampler: tpe
  n_trials: 50

  parameters:
    hidden_dims:
      type: categorical
      values: [[256, 128], [512, 256], [256]]
    dropout:
      type: float
      low: 0.1
      high: 0.5
    batch_size:
      type: categorical
      values: [16, 32, 64]
    learning_rate:
      type: loguniform
      low: 1e-6
      high: 1e-4
    loss_type:
      type: categorical
      values: [mse, huber, kl_divergence]

  objective:
    metric: val_mae
    direction: minimize
```

## Command Line Usage

```bash
# Basic training
python train.py --config config_multitask.yml

# Override specific values
python train.py --config config_multitask.yml \
  --training.epochs=30 \
  --training.batch_size=64

# Multi-GPU training
torchrun --nproc_per_node=4 train.py --config config_multitask.yml

# Evaluation only
python evaluate.py --config config_multitask.yml \
  --checkpoint checkpoints/best_model.pt

# Start validation API
python -m validation.api --config config_validation.yml
```
