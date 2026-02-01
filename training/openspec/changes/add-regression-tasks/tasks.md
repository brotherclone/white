# Implementation Tasks

## 1. Regression Head Architecture
- [x] 1.1 Create `RegressionHead` module with linear output layer
- [x] 1.2 Implement support for multiple continuous targets (intensity, fluidity, complexity)
- [x] 1.3 Add output activation options (sigmoid for bounded [0,1], none for unbounded)
- [x] 1.4 Implement uncertainty estimation head (variance prediction)

## 2. Multi-Task Learning Framework
- [x] 2.1 Create `MultiTaskModel` combining shared encoder with classification and regression heads
- [x] 2.2 Implement combined loss: alpha * classification_loss + beta * regression_loss
- [x] 2.3 Add dynamic loss weighting strategies (uncertainty weighting, gradient normalization)
- [x] 2.4 Implement task-specific learning rates if needed
- [x] 2.5 Add option for sequential training (classification first, then regression)

## 3. Loss Functions
- [x] 3.1 Implement MSE loss for regression targets
- [x] 3.2 Implement Huber loss (robust to outliers)
- [x] 3.3 Implement smooth L1 loss
- [x] 3.4 Add per-target loss weighting
- [x] 3.5 Implement combined classification + regression loss computation

## 4. Regression Evaluation Metrics
- [x] 4.1 Implement Mean Absolute Error (MAE)
- [x] 4.2 Implement Root Mean Squared Error (RMSE)
- [x] 4.3 Implement R² (coefficient of determination)
- [x] 4.4 Implement Pearson and Spearman correlation coefficients
- [x] 4.5 Add per-target metric computation and logging

## 5. Dataset Extensions
- [x] 5.1 Add continuous target columns to parquet schema (intensity, fluidity, complexity)
      NOTE: Implemented via on-the-fly generation from discrete labels instead of schema changes.
      Column auto-detection supports: rainbow_color_temporal_mode, rainbow_color_objectional_mode, rainbow_color_ontological_mode
- [x] 5.2 Implement normalization for regression targets
- [x] 5.3 Handle missing continuous annotations gracefully
- [x] 5.4 Add data validation for continuous targets (range checks)
- [x] 5.5 Implement target distribution visualization
- [x] 5.6 Derive continuous targets from discrete Rainbow Table labels
  - [x] 5.6.1 Convert discrete labels to soft targets (Past=1.0 → [1.0, 0.0, 0.0])
  - [x] 5.6.2 Add label smoothing for less confident segments (Past → [0.9, 0.05, 0.05])
  - [x] 5.6.3 Use temporal context for smoothing (if prev=Past, curr=Present, next=Past → curr=[0.3, 0.4, 0.3])
  - [x] 5.6.4 Implement configurable label smoothing: (1-alpha)*one_hot + alpha*uniform
  - [x] 5.6.5 Add human-in-the-loop refinement for ambiguous segments
  - [x] 5.6.6 Create validation for soft targets (sum to 1.0, valid ranges)
  - [x] 5.6.7 Add special handling for Black Album (uniform distributions [0.33, 0.33, 0.33])

## 6. Uncertainty Estimation
- [x] 6.1 Implement ensemble-based uncertainty (train multiple models)
- [x] 6.2 Implement evidential deep learning (predict distribution parameters)
- [x] 6.3 Implement Monte Carlo dropout for uncertainty at inference
- [x] 6.4 Add calibration metrics for uncertainty estimates
- [x] 6.5 Visualize prediction intervals vs true values

## 7. Configuration Schema
- [x] 7.1 Add `model.regression_head` section (targets, activation, hidden_layers)
- [x] 7.2 Add `model.multitask` section (loss_weights, strategy, dynamic_weighting)
- [x] 7.3 Add `training.regression_loss` config (type: mse, huber, smooth_l1)
- [x] 7.4 Add `evaluation.regression_metrics` list of metrics to compute

## 8. Testing & Validation
- [x] 8.1 Write unit tests for regression head forward pass
- [x] 8.2 Write unit tests for multi-task model
- [x] 8.3 Test combined loss computation
- [x] 8.4 Validate regression metrics against known examples
- [x] 8.5 Run training and verify regression convergence
      COMPLETED: RunPod training validated with excellent results:
      - Temporal Mode Accuracy: 98.4%
      - Spatial Mode Accuracy: 98.4%
      - Ontological Mode Accuracy: 98.5%
      - Album Accuracy: 96.9%
      - Val Loss: 0.00033
      Checkpoint saved: regression_validation_best.pt
- [x] 8.6 Compare multi-task vs single-task performance
      COMPLETED: Multi-task shows task interference - ontological accuracy drops from 98.5% to 56.3%
      Recommendation: Use single-task models (both at ceiling performance individually)

## 9. Documentation
- [x] 9.1 Document continuous rebracketing metrics and their meaning
- [x] 9.2 Document multi-task learning strategies and loss weighting
- [x] 9.3 Document uncertainty estimation approaches
- [x] 9.4 Add example configuration for regression and multi-task training

## 10. White Agent Integration API
- [x] 10.1 Create `ConceptValidator` class wrapping regression model
- [x] 10.2 Implement `validate_concept(text: str) -> ValidationResult` method
- [x] 10.3 Create `ValidationResult` dataclass with all ontological scores and flags
- [x] 10.4 Implement validation gate logic (accept/reject/hybrid decisions)
- [x] 10.5 Add hybrid state detection (threshold-based flagging)
- [x] 10.6 Implement transmigration distance computation
- [x] 10.7 Add album prediction from ontological scores
- [x] 10.8 Create FastAPI endpoint for validation requests
- [x] 10.9 Implement batch validation for multiple concepts
- [x] 10.10 Add validation result caching with TTL
- [x] 10.11 Create actionable suggestion generator for rejected concepts
- [x] 10.12 Write integration tests with LangGraph workflow
      COMPLETED: 19 integration tests in tests/integration/test_langgraph_workflow_integration.py
      - Tests ConceptValidator within LangGraph StateGraph workflows
      - Tests routing based on validation status (accept/reject/refine)
      - Tests album-based conditional branching
      - Tests batch validation, caching, edge cases

## 11. Soft Target Generation
- [x] 11.1 Implement one-hot encoding for discrete Rainbow Table labels
- [x] 11.2 Add label smoothing with configurable alpha
- [x] 11.3 Implement temporal context smoothing using surrounding segments
- [x] 11.4 Add special handling for Black Album (uniform distributions)
- [x] 11.5 Create human-in-the-loop annotation interface
- [x] 11.6 Implement target validation (check sums to 1.0, ranges)
- [x] 11.7 Add uncertainty weighting for ambiguous segments
- [x] 11.8 Create visualization of soft vs hard targets

## 12. Transmigration Analysis
- [x] 12.1 Implement distance computation between ontological states
- [x] 12.2 Add transmigration vector calculation (delta in each dimension)
- [x] 12.3 Create intermediate state generator for long transmigrations
- [x] 12.4 Implement dimension priority ranking for transmigration planning
- [x] 12.5 Add feasibility assessment (distance thresholds)
- [x] 12.6 Create minimal edit suggestion generator
- [x] 12.7 Add visualization of transmigration paths in embedding space

## 13. Album Prediction
- [x] 13.1 Implement album assignment from argmax of dimensions
- [x] 13.2 Add probability distribution over all albums
- [x] 13.3 Implement tie-breaking logic for similar probabilities
- [x] 13.4 Add confidence thresholding for album assignment
- [x] 13.5 Create album confusion matrix for evaluation

## 14. Rainbow Table Ontological Regression
- [x] 14.1 Implement temporal mode prediction head (3 softmax outputs)
- [x] 14.2 Implement spatial mode prediction head (3 softmax outputs)
- [x] 14.3 Implement ontological mode prediction head (3 softmax outputs)
- [x] 14.4 Implement chromatic confidence prediction head (1 sigmoid output)
- [x] 14.5 Create combined 10-output regression architecture
- [x] 14.6 Add per-dimension softmax activation layers
- [x] 14.7 Implement hybrid state detection logic
- [x] 14.8 Add diffuse state detection and Black Album flagging

## 15. Training Data Validation for Rainbow Table
- [x] 15.1 Implement target completeness checker (verify all 9 ontological targets)
- [x] 15.2 Add target distribution histograms per dimension
- [x] 15.3 Create consistency validator (discrete labels align with continuous scores)
- [x] 15.4 Add album balance reporter
- [x] 15.5 Implement automated data quality warnings

