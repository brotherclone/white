# Multi-Class Rebracketing Classifier

## ADDED Requirements

### Requirement: Multi-Class Classification Architecture
The system SHALL provide a neural network architecture that classifies segments into specific rebracketing type categories (spatial, temporal, causal, perceptual, memory, etc.) using text embeddings and softmax output.

#### Scenario: Single-label classification
- **WHEN** a text embedding is fed to the multiclass classifier
- **THEN** a probability distribution over all rebracketing types is returned with values summing to 1.0

#### Scenario: Multi-label classification mode
- **WHEN** the model is configured for multi-label mode and a segment has multiple rebracketing types
- **THEN** independent sigmoid activations return probabilities for each type independently

#### Scenario: Model dimension compatibility
- **WHEN** the classifier is initialized with input_dim matching the text encoder output
- **THEN** the forward pass completes without dimension mismatch errors

### Requirement: Class Weighting for Imbalanced Data
The system SHALL handle class imbalance through configurable class weighting strategies to prevent bias toward common rebracketing types.

#### Scenario: Balanced class weights
- **WHEN** class_weights is set to "balanced"
- **THEN** weights are computed as inverse class frequency to equalize learning signals

#### Scenario: Manual class weights
- **WHEN** class_weights is provided as a list of floats
- **THEN** the specified weights are applied to the loss function for each class

#### Scenario: No class weighting
- **WHEN** class_weights is set to None
- **THEN** all classes are weighted equally regardless of frequency

### Requirement: Loss Function Selection
The system SHALL use appropriate loss functions based on classification mode (single-label vs multi-label).

#### Scenario: Single-label with CrossEntropyLoss
- **WHEN** training in single-label mode
- **THEN** CrossEntropyLoss combines LogSoftmax and NLLLoss for efficient training

#### Scenario: Multi-label with BCEWithLogitsLoss
- **WHEN** training in multi-label mode
- **THEN** BCEWithLogitsLoss with independent sigmoid activations allows multiple active classes

#### Scenario: Class weights applied to loss
- **WHEN** class weights are configured
- **THEN** the loss function incorporates weights to adjust per-class gradients

### Requirement: Per-Class Evaluation Metrics
The system SHALL compute and report detailed per-class performance metrics to identify which rebracketing types are learned effectively.

#### Scenario: Per-class F1 scores
- **WHEN** evaluation is run on validation data
- **THEN** F1 score is computed independently for each rebracketing type

#### Scenario: Confusion matrix generation
- **WHEN** predictions are collected across the validation set
- **THEN** a confusion matrix shows which types are confused with each other

#### Scenario: Macro vs micro averaging
- **WHEN** aggregate metrics are computed
- **THEN** both macro-averaged (unweighted class average) and micro-averaged (weighted by frequency) metrics are reported

#### Scenario: Top-k accuracy for multi-label
- **WHEN** evaluating multi-label predictions
- **THEN** top-k accuracy measures if correct labels appear in the k highest probability predictions

### Requirement: Rebracketing Type Taxonomy
The system SHALL map conceptual segments to a defined taxonomy of rebracketing types based on training data annotations.

#### Scenario: Taxonomy enumeration
- **WHEN** the model is configured
- **THEN** the number of classes matches the rebracketing type taxonomy (e.g., 8 types: spatial, temporal, causal, perceptual, memory, ontological, narrative, identity)

#### Scenario: Label encoding
- **WHEN** text labels for rebracketing types are provided
- **THEN** they are mapped to integer indices for model training

#### Scenario: Unknown type handling
- **WHEN** a segment has an unrecognized rebracketing type
- **THEN** it is either mapped to an "other" class or flagged as an error

### Requirement: Training Configuration
The system SHALL extend the configuration schema to support multiclass-specific parameters.

#### Scenario: Classifier type selection
- **WHEN** config.model.classifier.type is set to "multiclass"
- **THEN** the MultiClassRebracketingClassifier is instantiated instead of the binary classifier

#### Scenario: Number of classes configuration
- **WHEN** config.model.classifier.num_classes is set to 8
- **THEN** the output layer has 8 neurons corresponding to the rebracketing types

#### Scenario: Multi-label flag
- **WHEN** config.model.classifier.multi_label is True
- **THEN** the model uses sigmoid activations and BCE loss instead of softmax and CrossEntropy

### Requirement: Interpretability and Diagnosis
The system SHALL provide tools to understand model predictions and identify failure modes.

#### Scenario: Confusion matrix visualization
- **WHEN** a confusion matrix is generated
- **THEN** it is saved as a heatmap image showing prediction patterns

#### Scenario: Misclassification analysis
- **WHEN** validation completes
- **THEN** the most confused class pairs are reported with example segments

#### Scenario: Class-wise confidence distribution
- **WHEN** analyzing model predictions
- **THEN** the confidence distribution for each class is visualized to detect overconfident or uncertain predictions

### Requirement: Data Pipeline Integration
The system SHALL integrate with existing data loading infrastructure while adding multiclass label support.

#### Scenario: Parquet column for rebracketing type
- **WHEN** loading training data from parquet files
- **THEN** the rebracketing_type column is read and encoded as class indices

#### Scenario: Class distribution logging
- **WHEN** data loading begins
- **THEN** the distribution of rebracketing types is logged to help identify imbalances

#### Scenario: Stratified splitting
- **WHEN** creating train/validation splits
- **THEN** splits are stratified by rebracketing type to ensure balanced representation

### Requirement: Backward Compatibility
The system SHALL maintain compatibility with Phase 1 binary classification while adding multiclass capabilities.

#### Scenario: Binary mode still supported
- **WHEN** config.model.classifier.type is set to "binary"
- **THEN** the original binary classifier is used without changes

#### Scenario: Shared encoder weights
- **WHEN** switching between binary and multiclass modes
- **THEN** the text encoder weights can be reused for transfer learning

#### Scenario: Config validation
- **WHEN** an invalid classifier type is specified
- **THEN** a clear error message guides the user to valid options
