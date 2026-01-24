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
- [ ] 5.6 Derive continuous targets from discrete Rainbow Table labels
  - [ ] 5.6.1 Convert discrete labels to soft targets (Past=1.0 → [1.0, 0.0, 0.0])
  - [ ] 5.6.2 Add label smoothing for less confident segments (Past → [0.9, 0.05, 0.05])
  - [ ] 5.6.3 Use temporal context for smoothing (if prev=Past, curr=Present, next=Past → curr=[0.3, 0.4, 0.3])
  - [ ] 5.6.4 Implement configurable label smoothing: (1-alpha)*one_hot + alpha*uniform
  - [ ] 5.6.5 Add human-in-the-loop refinement for ambiguous segments
  - [ ] 5.6.6 Create validation for soft targets (sum to 1.0, valid ranges)
  - [ ] 5.6.7 Add special handling for Black Album (uniform distributions [0.33, 0.33, 0.33])

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

## 10. White Agent Integration API
- [ ] 10.1 Create `ConceptValidator` class wrapping regression model
- [ ] 10.2 Implement `validate_concept(text: str) -> ValidationResult` method
- [ ] 10.3 Create `ValidationResult` dataclass with all ontological scores and flags
- [ ] 10.4 Implement validation gate logic (accept/reject/hybrid decisions)
- [ ] 10.5 Add hybrid state detection (threshold-based flagging)
- [ ] 10.6 Implement transmigration distance computation
- [ ] 10.7 Add album prediction from ontological scores
- [ ] 10.8 Create FastAPI endpoint for validation requests
- [ ] 10.9 Implement batch validation for multiple concepts
- [ ] 10.10 Add validation result caching with TTL
- [ ] 10.11 Create actionable suggestion generator for rejected concepts
- [ ] 10.12 Write integration tests with LangGraph workflow

## 11. Soft Target Generation
- [ ] 11.1 Implement one-hot encoding for discrete Rainbow Table labels
- [ ] 11.2 Add label smoothing with configurable alpha
- [ ] 11.3 Implement temporal context smoothing using surrounding segments
- [ ] 11.4 Add special handling for Black Album (uniform distributions)
- [ ] 11.5 Create human-in-the-loop annotation interface
- [ ] 11.6 Implement target validation (check sums to 1.0, ranges)
- [ ] 11.7 Add uncertainty weighting for ambiguous segments
- [ ] 11.8 Create visualization of soft vs hard targets

## 12. Transmigration Analysis
- [ ] 12.1 Implement distance computation between ontological states
- [ ] 12.2 Add transmigration vector calculation (delta in each dimension)
- [ ] 12.3 Create intermediate state generator for long transmigrations
- [ ] 12.4 Implement dimension priority ranking for transmigration planning
- [ ] 12.5 Add feasibility assessment (distance thresholds)
- [ ] 12.6 Create minimal edit suggestion generator
- [ ] 12.7 Add visualization of transmigration paths in embedding space

## 13. Album Prediction
- [ ] 13.1 Implement album assignment from argmax of dimensions
- [ ] 13.2 Add probability distribution over all albums
- [ ] 13.3 Implement tie-breaking logic for similar probabilities
- [ ] 13.4 Add confidence thresholding for album assignment
- [ ] 13.5 Create album confusion matrix for evaluation

## 14. Rainbow Table Ontological Regression
- [ ] 14.1 Implement temporal mode prediction head (3 softmax outputs)
- [ ] 14.2 Implement spatial mode prediction head (3 softmax outputs)
- [ ] 14.3 Implement ontological mode prediction head (3 softmax outputs)
- [ ] 14.4 Implement chromatic confidence prediction head (1 sigmoid output)
- [ ] 14.5 Create combined 10-output regression architecture
- [ ] 14.6 Add per-dimension softmax activation layers
- [ ] 14.7 Implement hybrid state detection logic
- [ ] 14.8 Add diffuse state detection and Black Album flagging

## 15. Training Data Validation for Rainbow Table
- [ ] 15.1 Implement target completeness checker (verify all 9 ontological targets)
- [ ] 15.2 Add target distribution histograms per dimension
- [ ] 15.3 Create consistency validator (discrete labels align with continuous scores)
- [ ] 15.4 Add album balance reporter
- [ ] 15.5 Implement automated data quality warnings

