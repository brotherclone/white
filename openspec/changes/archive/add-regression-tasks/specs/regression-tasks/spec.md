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

### Requirement: Rainbow Table Ontological Regression Targets
The system SHALL predict continuous scores for temporal, spatial, and ontological modes that define the Rainbow Table framework.

#### Scenario: Temporal mode distribution
- **WHEN** predicting temporal orientation of a concept/segment
- **THEN** three softmax values sum to 1.0: [past_score, present_score, future_score]
- **AND** values represent probability distribution over temporal modes

#### Scenario: Spatial mode distribution
- **WHEN** predicting spatial focus of a concept/segment
- **THEN** three softmax values sum to 1.0: [thing_score, place_score, person_score]
- **AND** values represent probability distribution over spatial modes

#### Scenario: Ontological mode distribution
- **WHEN** predicting reality status of a concept/segment
- **THEN** three softmax values sum to 1.0: [imagined_score, forgotten_score, known_score]
- **AND** values represent probability distribution over ontological modes

#### Scenario: Chromatic confidence score
- **WHEN** predicting album assignment certainty
- **THEN** single sigmoid value in [0,1] indicates how strongly concept fits any single mode
- **AND** low confidence (<0.5) suggests hybrid or transitional state

#### Scenario: Combined ontological state
- **WHEN** all three distributions are predicted
- **THEN** argmax of each gives discrete mode (e.g., Past_Thing_Imagined)
- **AND** confidence scores indicate certainty of assignment

#### Scenario: Model output structure
- **WHEN** regression head produces predictions
- **THEN** output contains temporal [3], spatial [3], ontological [3], confidence [1] tensors
- **AND** total of 10 output values per prediction

### Requirement: Hybrid State Detection
The system SHALL identify concepts that straddle multiple ontological modes, indicating liminal or transitional states.

#### Scenario: Balanced hybrid (liminal state)
- **WHEN** top two scores in any dimension are within 0.15 of each other
- **THEN** concept is flagged as "hybrid_{dimension}" (e.g., "hybrid_temporal")
- **AND** both modes are reported in results

#### Scenario: Dominant mode (clear assignment)
- **WHEN** top score in a dimension exceeds 0.6
- **THEN** dimension is flagged as "dominant_{mode}" (e.g., "dominant_past")
- **AND** concept has clear orientation in that dimension

#### Scenario: Diffuse state (unclear assignment)
- **WHEN** all three scores in a dimension are within 0.2 of each other (≈0.33 each)
- **THEN** dimension is flagged as "diffuse_{dimension}"
- **AND** concept lacks clear orientation in that dimension

#### Scenario: Triple diffuse (Black Album candidate)
- **WHEN** all three dimensions (temporal, spatial, ontological) are diffuse
- **THEN** concept is flagged as "black_album_candidate"
- **AND** suggested for None_None_None mode (chaos/void)

#### Scenario: Partial hybrid
- **WHEN** one dimension is dominant but others are hybrid/diffuse
- **THEN** concept is flagged as "partial_hybrid"
- **AND** dominant dimension determines primary classification

#### Scenario: Hybrid threshold configuration
- **WHEN** hybrid_threshold parameter is set
- **THEN** difference between top two scores must be < threshold to flag hybrid
- **AND** default threshold is 0.15

### Requirement: Transmigration Distance Computation
The system SHALL measure the conceptual distance between ontological states to quantify transmigration difficulty.

#### Scenario: Single dimension shift distance
- **WHEN** computing distance between modes in one dimension
- **THEN** L2 norm of score vectors is computed
- **AND** Past[1,0,0] → Present[0,1,0] yields distance ≈ 1.414

#### Scenario: Multi-dimensional shift distance
- **WHEN** moving from one combined state to another (e.g., Past_Thing_Imagined → Future_Place_Known)
- **THEN** total_distance = sqrt(temporal_distance² + spatial_distance² + ontological_distance²)
- **AND** each dimension contributes to total transmigration effort

#### Scenario: Minimum transmigration path
- **WHEN** given source and target ontological states
- **THEN** identify which dimensions require most change
- **AND** report dimensions in order of required shift magnitude

#### Scenario: Transmigration feasibility assessment
- **WHEN** distance exceeds threshold (default: 2.0)
- **THEN** flag as "difficult_transmigration"
- **AND** suggest intermediate transitional states

#### Scenario: Within-album distance
- **WHEN** both states belong to same album (e.g., two Orange Album concepts)
- **THEN** distance should be < 1.0
- **AND** indicates concepts are ontologically similar

#### Scenario: Cross-album distance
- **WHEN** states belong to different albums
- **THEN** distance typically > 1.0
- **AND** indicates significant ontological shift required

### Requirement: Concept Validation Gates
The system SHALL provide automated accept/reject decisions for generated concepts based on ontological coherence.

#### Scenario: High confidence concept (accept)
- **WHEN** chromatic_confidence > 0.7 AND top mode in each dimension > 0.6
- **THEN** validation_status = "ACCEPT"
- **AND** concept has clear, confident ontological position

#### Scenario: Hybrid concept (conditional accept)
- **WHEN** concept flagged as hybrid in ≤2 dimensions AND confidence > 0.5
- **THEN** validation_status = "ACCEPT_HYBRID"
- **AND** suggest album based on dominant dimension(s)

#### Scenario: Diffuse concept (reject for regeneration)
- **WHEN** ≥2 dimensions are diffuse OR confidence < 0.4
- **THEN** validation_status = "REJECT"
- **AND** reason = "diffuse_ontology"

#### Scenario: Out-of-distribution concept (reject)
- **WHEN** uncertainty estimate > 0.8
- **THEN** validation_status = "REJECT"
- **AND** reason = "ood_detection" (outside training distribution)

#### Scenario: Black Album acceptance
- **WHEN** all three dimensions diffuse AND confidence < 0.3
- **THEN** validation_status = "ACCEPT_BLACK"
- **AND** suggest Black Album (None_None_None mode)

#### Scenario: Validation with actionable suggestions
- **WHEN** concept rejected
- **THEN** provide specific suggestions for improvement
- **AND** suggest which scores to increase/decrease (e.g., "increase past_score by 0.3")

#### Scenario: Configurable validation thresholds
- **WHEN** validation thresholds are configured
- **THEN** accept/reject logic uses custom thresholds
- **AND** defaults: confidence_threshold=0.7, dominant_threshold=0.6, diffuse_threshold=0.2

### Requirement: White Agent Integration API
The system SHALL provide a structured API for validating concepts generated by the White Agent workflow.

#### Scenario: Concept validation request
- **WHEN** White Agent submits concept text for validation
- **THEN** system extracts features and runs regression model
- **AND** returns ValidationResult with full ontological analysis

#### Scenario: Validation result structure
- **WHEN** concept validation completes
- **THEN** return structured result containing temporal_scores, spatial_scores, ontological_scores, chromatic_confidence, predicted_album, predicted_mode, validation_status, hybrid_flags, uncertainty_estimates, transmigration_distances, and suggestions

#### Scenario: Batch validation
- **WHEN** multiple concepts submitted in single request
- **THEN** process all concepts efficiently
- **AND** return List[ValidationResult] in same order

#### Scenario: Validation caching
- **WHEN** identical concept text submitted multiple times
- **THEN** return cached result
- **AND** cache expires after configurable TTL (default: 1 hour)

#### Scenario: LangGraph integration
- **WHEN** called from LangGraph workflow node
- **THEN** API accepts concept as string, returns ValidationResult
- **AND** workflow can branch based on validation_status

#### Scenario: Real-time validation endpoint
- **WHEN** FastAPI endpoint receives POST request
- **THEN** process concept and return JSON response
- **AND** response time < 500ms for single concept

### Requirement: Soft Target Derivation from Discrete Labels
The system SHALL convert discrete Rainbow Table labels into continuous regression targets for training.

#### Scenario: One-hot encoding for dominant modes
- **WHEN** segment has discrete label "Past"
- **THEN** convert to temporal target [1.0, 0.0, 0.0]
- **AND** similarly for all nine modes across three dimensions

#### Scenario: Label smoothing for less confident segments
- **WHEN** label_smoothing parameter is set (e.g., 0.1)
- **THEN** smooth one-hot: [1.0, 0.0, 0.0] → [0.9, 0.05, 0.05]
- **AND** prevents overconfident predictions

#### Scenario: Temporal context smoothing
- **WHEN** segment surrounded by different modes (e.g., prev=Past, curr=Present, next=Past)
- **THEN** adjust current target: [0.0, 1.0, 0.0] → [0.3, 0.4, 0.3]
- **AND** accounts for transitional nature

#### Scenario: Black Album (None) handling
- **WHEN** segment labeled as None_None_None (Black Album)
- **THEN** all dimensions get uniform distribution [0.33, 0.33, 0.33]
- **AND** chromatic_confidence target = 0.0

#### Scenario: Human-in-the-loop refinement
- **WHEN** human annotates segment as "mostly Past, slightly Present"
- **THEN** use provided distribution (e.g., [0.7, 0.3, 0.0])
- **AND** override automatic soft target generation

#### Scenario: Uncertainty-weighted targets
- **WHEN** segment has high annotation uncertainty
- **THEN** target distribution spread more evenly
- **AND** loss weighting reduced for uncertain examples

#### Scenario: Cross-validation of soft targets
- **WHEN** soft targets are generated
- **THEN** validate that distributions sum to 1.0
- **AND** verify consistency across temporal sequences

### Requirement: Album Assignment Prediction
The system SHALL predict which Rainbow Table album a concept belongs to based on ontological scores.

#### Scenario: Direct album prediction
- **WHEN** computing album assignment
- **THEN** determine mode from argmax of each dimension
- **AND** map combined mode to album (e.g., Past_Thing_Imagined → Orange)

#### Scenario: Album probability distribution
- **WHEN** computing probabilities for all albums
- **THEN** multiply dimension probabilities: P(Orange) = P(Past) * P(Thing) * P(Imagined)
- **AND** return distribution over all albums

#### Scenario: Album confidence threshold
- **WHEN** top album probability < threshold (default: 0.3)
- **THEN** flag as "uncertain_album"
- **AND** consider Black Album or request regeneration

#### Scenario: Tie-breaking for equal probabilities
- **WHEN** two albums have similar probabilities (within 0.1)
- **THEN** use chromatic_confidence as tiebreaker
- **AND** report both albums as potential matches

### Requirement: Transmigration Guidance
The system SHALL provide specific guidance for moving concepts between ontological modes.

#### Scenario: Transmigration vector computation
- **WHEN** given source and target modes
- **THEN** compute delta vectors in each dimension
- **AND** provide magnitude and direction of required shift

#### Scenario: Step-by-step transmigration plan
- **WHEN** transmigration distance > 2.0
- **THEN** suggest intermediate states
- **AND** provide sequence: source → intermediate(s) → target

#### Scenario: Dimension priority for transmigration
- **WHEN** planning transmigration
- **THEN** rank dimensions by required change magnitude
- **AND** suggest tackling largest shifts first

#### Scenario: Feasibility warning
- **WHEN** transmigration requires changes in all three dimensions
- **THEN** warn that transformation may be difficult
- **AND** suggest starting from different source concept

#### Scenario: Minimal edit suggestions
- **WHEN** concept almost fits target mode (distance < 0.5 in one dimension)
- **THEN** suggest minimal textual edits
- **AND** provide specific score targets (e.g., "increase past_score from 0.55 to 0.70")

### Requirement: Training Data Validation
The system SHALL validate that training data has appropriate continuous targets for all Rainbow Table segments.

#### Scenario: Target completeness check
- **WHEN** loading training dataset
- **THEN** verify all nine ontological targets exist for each segment
- **AND** report missing targets or invalid ranges

#### Scenario: Target distribution analysis
- **WHEN** analyzing training dataset
- **THEN** compute histograms of each score dimension
- **AND** verify adequate coverage of score ranges

#### Scenario: Target consistency validation
- **WHEN** segment has discrete label and continuous scores
- **THEN** verify scores align with label (e.g., Past label → past_score highest)
- **AND** flag inconsistencies for review

#### Scenario: Album balance in training data
- **WHEN** training dataset is loaded
- **THEN** report distribution of segments per album
- **AND** warn if any album severely underrepresented

