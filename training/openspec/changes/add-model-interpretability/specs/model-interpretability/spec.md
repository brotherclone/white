# Model Interpretability

## ADDED Requirements

### Requirement: Attention Visualization
The system SHALL extract and visualize attention weights from transformer-based models.

#### Scenario: Text attention heatmap
- **WHEN** visualizing text attention
- **THEN** a heatmap shows which words attend to which other words

#### Scenario: Cross-modal attention patterns
- **WHEN** visualizing audio-text cross-attention
- **THEN** patterns show which audio regions align with which text tokens

#### Scenario: Interactive attention exploration
- **WHEN** using interactive tools
- **THEN** users can hover over tokens to see attention distributions

### Requirement: Embedding Space Projection
The system SHALL project high-dimensional embeddings to 2D/3D for visualization and analysis.

#### Scenario: TSNE projection
- **WHEN** using TSNE
- **THEN** embeddings are projected preserving local structure

#### Scenario: UMAP projection
- **WHEN** using UMAP
- **THEN** embeddings are projected preserving global structure

#### Scenario: Color by chromatic mode
- **WHEN** visualizing embeddings
- **THEN** points are colored by chromatic mode to reveal clustering

#### Scenario: Cluster analysis
- **WHEN** analyzing projected embeddings
- **THEN** chromatic modes and rebracketing types are assessed for separation

### Requirement: Feature Attribution
The system SHALL identify which input features most influence predictions.

#### Scenario: Integrated Gradients attribution
- **WHEN** using Integrated Gradients
- **THEN** attribution scores show importance of each input feature

#### Scenario: SHAP values
- **WHEN** using SHAP
- **THEN** Shapley values fairly distribute prediction contribution across features

#### Scenario: Word-level attribution
- **WHEN** attributing text predictions
- **THEN** each word receives an importance score

#### Scenario: Audio region attribution
- **WHEN** attributing audio predictions
- **THEN** time regions in waveforms receive importance scores

### Requirement: Counterfactual Explanations
The system SHALL generate minimal input changes that flip model predictions.

#### Scenario: Minimal edit search
- **WHEN** finding counterfactuals
- **THEN** the smallest change flipping prediction is identified

#### Scenario: Decision boundary visualization
- **WHEN** analyzing counterfactuals
- **THEN** decision boundaries between classes are revealed

### Requirement: Chromatic Geometry Analysis
The system SHALL analyze the geometric structure of chromatic embeddings.

#### Scenario: Pairwise distance matrix
- **WHEN** computing distances between chromatic modes
- **THEN** a matrix shows separation between modes

#### Scenario: Linear trajectory test
- **WHEN** testing for BLACK → WHITE progression
- **THEN** embedding positions are tested for linear arrangement

#### Scenario: Emergent structure discovery
- **WHEN** analyzing geometry
- **THEN** unexpected patterns or clusterings are identified

### Requirement: Layer-Wise Representation Analysis
The system SHALL analyze how representations evolve across model layers.

#### Scenario: Extract layer activations
- **WHEN** running inference
- **THEN** activations from each layer are captured

#### Scenario: Layer-wise visualization
- **WHEN** visualizing representations
- **THEN** changes across layers reveal learning progression

### Requirement: Interpretability Configuration
The system SHALL provide configuration for interpretability analysis.

#### Scenario: Enable interpretability
- **WHEN** config.interpretability.enabled is True
- **THEN** interpretability tools are run during evaluation

#### Scenario: Method selection
- **WHEN** config.interpretability.methods includes ["attention", "embedding", "attribution"]
- **THEN** specified methods are applied

### Requirement: Visualization Output
The system SHALL generate and save interpretability visualizations.

#### Scenario: Save heatmaps
- **WHEN** generating attention heatmaps
- **THEN** images are saved to configured output directory

#### Scenario: Interactive dashboard
- **WHEN** generating interactive visualizations
- **THEN** HTML dashboards with Plotly are created

#### Scenario: Export embeddings
- **WHEN** analyzing embedding space
- **THEN** projected coordinates are exported for external tools
