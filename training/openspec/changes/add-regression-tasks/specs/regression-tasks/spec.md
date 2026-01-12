# Regression Tasks

## ADDED Requirements

### Requirement: Regression Head Architecture
The system SHALL provide a neural network head that predicts continuous rebracketing metrics from learned embeddings.

#### Scenario: Single continuous target prediction
- **WHEN** the regression head receives an embedding vector
- **THEN** it outputs a single continuous value (e.g., intensity score)

#### Scenario: Multiple continuous targets
- **WHEN** configured for multiple regression targets
- **THEN** the head outputs a vector with one value per target

#### Scenario: Bounded output with sigmoid
- **WHEN** target values are bounded (e.g., [0, 1] for intensity)
- **THEN** sigmoid activation constrains outputs to valid range

#### Scenario: Unbounded output
- **WHEN** target values are unbounded (e.g., temporal complexity score)
- **THEN** no activation is applied to the output layer

### Requirement: Multi-Task Learning Framework
The system SHALL support simultaneous training of classification and regression tasks sharing a common encoder.

#### Scenario: Shared encoder with dual heads
- **WHEN** multi-task model is configured
- **THEN** embeddings from shared encoder feed both classification and regression heads

#### Scenario: Combined loss computation
- **WHEN** computing total loss
- **THEN** classification and regression losses are weighted and summed

#### Scenario: Task-specific gradient flow
- **WHEN** backpropagating multi-task loss
- **THEN** gradients from both tasks update the shared encoder

#### Scenario: Independent head training
- **WHEN** tasks have very different convergence rates
- **THEN** optional separate optimizers for each head can be configured

### Requirement: Regression Loss Functions
The system SHALL provide multiple loss functions suitable for continuous target prediction.

#### Scenario: MSE loss for standard regression
- **WHEN** regression loss type is "mse"
- **THEN** Mean Squared Error is computed between predictions and targets

#### Scenario: Huber loss for outlier robustness
- **WHEN** regression loss type is "huber"
- **THEN** Huber loss (smooth combination of L1 and L2) is used

#### Scenario: Smooth L1 loss
- **WHEN** regression loss type is "smooth_l1"
- **THEN** Smooth L1 loss reduces sensitivity to outliers

#### Scenario: Per-target loss weighting
- **WHEN** different targets have different importance
- **THEN** per-target weights scale individual loss contributions

### Requirement: Multi-Task Loss Weighting
The system SHALL balance classification and regression losses through configurable weighting strategies.

#### Scenario: Fixed loss weights
- **WHEN** loss weights alpha and beta are manually configured
- **THEN** combined_loss = alpha * classification_loss + beta * regression_loss

#### Scenario: Uncertainty-based weighting
- **WHEN** dynamic weighting is enabled
- **THEN** task uncertainties adaptively adjust loss weights during training

#### Scenario: Gradient normalization weighting
- **WHEN** gradient-based weighting is enabled
- **THEN** loss weights balance gradient magnitudes across tasks

#### Scenario: Equal weighting baseline
- **WHEN** no weighting strategy is specified
- **THEN** both losses are weighted equally (alpha = beta = 1.0)

### Requirement: Regression Evaluation Metrics
The system SHALL compute comprehensive metrics to assess continuous prediction quality.

#### Scenario: Mean Absolute Error
- **WHEN** computing MAE
- **THEN** the average absolute difference between predictions and targets is reported

#### Scenario: Root Mean Squared Error
- **WHEN** computing RMSE
- **THEN** the square root of mean squared errors is reported

#### Scenario: R² coefficient of determination
- **WHEN** computing R²
- **THEN** the proportion of variance explained by the model is reported

#### Scenario: Correlation coefficients
- **WHEN** computing correlation metrics
- **THEN** Pearson and Spearman correlations measure linear and monotonic relationships

#### Scenario: Per-target metrics
- **WHEN** multiple regression targets exist
- **THEN** metrics are computed independently for each target

### Requirement: Continuous Target Annotation
The system SHALL load and preprocess continuous rebracketing metrics from training data.

#### Scenario: Load intensity scores
- **WHEN** dataset includes rebracketing_intensity column
- **THEN** continuous values in [0, 1] are loaded as regression targets

#### Scenario: Load boundary fluidity scores
- **WHEN** dataset includes boundary_fluidity column
- **THEN** continuous values representing fluidity are loaded

#### Scenario: Load temporal complexity scores
- **WHEN** dataset includes temporal_complexity column
- **THEN** continuous unbounded values are loaded

#### Scenario: Missing target handling
- **WHEN** a segment lacks continuous annotations
- **THEN** it is either excluded or imputed based on configuration

#### Scenario: Target normalization
- **WHEN** targets have different scales
- **THEN** standardization or min-max normalization is applied

### Requirement: Uncertainty Estimation
The system SHALL quantify prediction uncertainty for continuous targets to support decision-making.

#### Scenario: Ensemble-based uncertainty
- **WHEN** multiple models are trained with different initializations
- **THEN** prediction variance across the ensemble estimates uncertainty

#### Scenario: Monte Carlo dropout uncertainty
- **WHEN** running inference with dropout enabled
- **THEN** multiple forward passes with dropout provide uncertainty estimates

#### Scenario: Evidential deep learning
- **WHEN** the model predicts distribution parameters
- **THEN** aleatoric and epistemic uncertainty are separated

#### Scenario: Calibration assessment
- **WHEN** uncertainty estimates are evaluated
- **THEN** calibration curves show if confidence matches actual accuracy

### Requirement: Prediction Interval Estimation
The system SHALL provide confidence intervals around continuous predictions.

#### Scenario: 95% prediction interval
- **WHEN** uncertainty is quantified
- **THEN** a 95% interval is computed around each prediction

#### Scenario: Interval visualization
- **WHEN** generating prediction plots
- **THEN** intervals are shown as shaded regions around point estimates

#### Scenario: Interval coverage evaluation
- **WHEN** assessing interval quality
- **THEN** empirical coverage (% of targets within intervals) is measured

### Requirement: Sequential Training Strategy
The system SHALL support sequential task training when simultaneous multi-task learning is unstable.

#### Scenario: Classification first
- **WHEN** sequential training is enabled
- **THEN** classification head is trained to convergence before enabling regression

#### Scenario: Regression second
- **WHEN** classification converges
- **THEN** regression head is trained while keeping classification head frozen or fine-tuning

#### Scenario: Joint fine-tuning
- **WHEN** both tasks have individually converged
- **THEN** optional joint fine-tuning with both losses can refine the shared encoder

### Requirement: Regression Configuration Schema
The system SHALL extend configuration to support regression-specific parameters.

#### Scenario: Regression targets specification
- **WHEN** config.model.regression_head.targets is set to ["intensity", "fluidity", "complexity"]
- **THEN** the regression head outputs three continuous values

#### Scenario: Activation function selection
- **WHEN** config.model.regression_head.activation is "sigmoid"
- **THEN** sigmoid activation is applied to outputs

#### Scenario: Multi-task loss weights
- **WHEN** config.model.multitask.loss_weights is {"classification": 1.0, "regression": 0.5}
- **THEN** regression loss is weighted at half the classification loss

#### Scenario: Regression loss type
- **WHEN** config.training.regression_loss is "huber"
- **THEN** Huber loss is used instead of MSE

### Requirement: Target Distribution Analysis
The system SHALL provide tools to analyze the distribution of continuous targets in the dataset.

#### Scenario: Histogram visualization
- **WHEN** analyzing target distributions
- **THEN** histograms show the frequency of different intensity/fluidity/complexity values

#### Scenario: Outlier detection
- **WHEN** targets are loaded
- **THEN** values beyond specified percentiles are flagged as potential outliers

#### Scenario: Correlation analysis
- **WHEN** multiple continuous targets exist
- **THEN** correlation matrices show relationships between targets

### Requirement: Gradient Monitoring for Multi-Task Learning
The system SHALL monitor gradients from each task to detect training imbalances.

#### Scenario: Per-task gradient norms
- **WHEN** backward pass completes
- **THEN** gradient norms from classification and regression losses are logged separately

#### Scenario: Gradient imbalance warning
- **WHEN** one task dominates gradients (e.g., 10x larger)
- **THEN** a warning suggests adjusting loss weights

#### Scenario: Gradient clipping per-task
- **WHEN** gradient clipping is enabled
- **THEN** it can be applied independently to each task's gradients
