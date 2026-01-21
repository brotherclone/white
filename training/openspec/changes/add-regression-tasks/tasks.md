# Implementation Tasks

## 1. Regression Head Architecture
- [ ] 1.1 Create `RegressionHead` module with linear output layer
- [ ] 1.2 Implement support for multiple continuous targets (intensity, fluidity, complexity)
- [ ] 1.3 Add output activation options (sigmoid for bounded [0,1], none for unbounded)
- [ ] 1.4 Implement uncertainty estimation head (variance prediction)

## 2. Multi-Task Learning Framework
- [ ] 2.1 Create `MultiTaskModel` combining shared encoder with classification and regression heads
- [ ] 2.2 Implement combined loss: alpha * classification_loss + beta * regression_loss
- [ ] 2.3 Add dynamic loss weighting strategies (uncertainty weighting, gradient normalization)
- [ ] 2.4 Implement task-specific learning rates if needed
- [ ] 2.5 Add option for sequential training (classification first, then regression)

## 3. Loss Functions
- [ ] 3.1 Implement MSE loss for regression targets
- [ ] 3.2 Implement Huber loss (robust to outliers)
- [ ] 3.3 Implement smooth L1 loss
- [ ] 3.4 Add per-target loss weighting
- [ ] 3.5 Implement combined classification + regression loss computation

## 4. Regression Evaluation Metrics
- [ ] 4.1 Implement Mean Absolute Error (MAE)
- [ ] 4.2 Implement Root Mean Squared Error (RMSE)
- [ ] 4.3 Implement R² (coefficient of determination)
- [ ] 4.4 Implement Pearson and Spearman correlation coefficients
- [ ] 4.5 Add per-target metric computation and logging

## 5. Dataset Extensions
- [ ] 5.1 Add continuous target columns to parquet schema (intensity, fluidity, complexity)
- [ ] 5.2 Implement normalization for regression targets
- [ ] 5.3 Handle missing continuous annotations gracefully
- [ ] 5.4 Add data validation for continuous targets (range checks)
- [ ] 5.5 Implement target distribution visualization

## 6. Uncertainty Estimation
- [ ] 6.1 Implement ensemble-based uncertainty (train multiple models)
- [ ] 6.2 Implement evidential deep learning (predict distribution parameters)
- [ ] 6.3 Implement Monte Carlo dropout for uncertainty at inference
- [ ] 6.4 Add calibration metrics for uncertainty estimates
- [ ] 6.5 Visualize prediction intervals vs true values

## 7. Configuration Schema
- [ ] 7.1 Add `model.regression_head` section (targets, activation, hidden_layers)
- [ ] 7.2 Add `model.multitask` section (loss_weights, strategy, dynamic_weighting)
- [ ] 7.3 Add `training.regression_loss` config (type: mse, huber, smooth_l1)
- [ ] 7.4 Add `evaluation.regression_metrics` list of metrics to compute

## 8. Testing & Validation
- [ ] 8.1 Write unit tests for regression head forward pass
- [ ] 8.2 Write unit tests for multi-task model
- [ ] 8.3 Test combined loss computation
- [ ] 8.4 Validate regression metrics against known examples
- [ ] 8.5 Run training and verify regression convergence
- [ ] 8.6 Compare multi-task vs single-task performance

## 9. Documentation
- [ ] 9.1 Document continuous rebracketing metrics and their meaning
- [ ] 9.2 Document multi-task learning strategies and loss weighting
- [ ] 9.3 Document uncertainty estimation approaches
- [ ] 9.4 Add example configuration for regression and multi-task training
