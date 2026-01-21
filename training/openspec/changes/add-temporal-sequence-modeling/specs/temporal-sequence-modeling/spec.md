# Temporal Sequence Modeling

## ADDED Requirements

### Requirement: Temporal Dataset with Context Windows
The system SHALL load segments with temporal context including previous and next segments to enable sequence modeling.

#### Scenario: Load current segment with context
- **WHEN** loading segment at index N with context_window=3
- **THEN** segments [N-3, N-2, N-1, N, N+1, N+2, N+3] are returned

#### Scenario: Handle album boundaries
- **WHEN** a segment is at the start of an album with insufficient previous segments
- **THEN** padding or truncated context is provided without crossing album boundaries

#### Scenario: Handle album end
- **WHEN** a segment is at the end of an album with insufficient next segments
- **THEN** padding or truncated context is provided

#### Scenario: Validate temporal ordering
- **WHEN** loading a sequence of segments
- **THEN** timestamps are verified to be in ascending order

### Requirement: LSTM Sequence Modeling
The system SHALL provide LSTM-based models for learning temporal dependencies across segment sequences.

#### Scenario: Bidirectional LSTM
- **WHEN** configured for bidirectional LSTM
- **THEN** the model processes sequences forward and backward to capture past and future context

#### Scenario: Stacked LSTM layers
- **WHEN** multiple LSTM layers are configured
- **THEN** deep temporal representations are learned

#### Scenario: Sequence output modes
- **WHEN** returning sequence predictions
- **THEN** options include last-step output, mean-pooled output, or all-step outputs

### Requirement: Temporal Transformer Architecture
The system SHALL provide transformer-based models with temporal positional encoding.

#### Scenario: Actual temporal distance encoding
- **WHEN** segments have irregular time intervals
- **THEN** positional encodings use actual time distances rather than sequence positions

#### Scenario: Self-attention over sequences
- **WHEN** processing a segment sequence
- **THEN** self-attention captures dependencies between any pair of segments

#### Scenario: Learnable positional encodings
- **WHEN** configured for learnable encodings
- **THEN** positional encoding parameters are optimized during training

### Requirement: Temporal Convolutional Network (TCN)
The system SHALL provide TCN models with dilated causal convolutions for long-range dependencies.

#### Scenario: Dilated convolutions
- **WHEN** TCN processes sequences
- **THEN** exponentially increasing dilation factors capture long-range patterns efficiently

#### Scenario: Causal convolutions
- **WHEN** predicting future segments
- **THEN** causal masking prevents information leakage from future to past

#### Scenario: Residual connections
- **WHEN** TCN has many layers
- **THEN** residual connections enable gradient flow and deeper architectures

### Requirement: Transition Prediction
The system SHALL predict rebracketing changes between consecutive segments.

#### Scenario: Intensity delta prediction
- **WHEN** given segment N and N+1
- **THEN** the model predicts delta_intensity = intensity(N+1) - intensity(N)

#### Scenario: Type transition prediction
- **WHEN** segment N has rebracketing type A
- **THEN** the model predicts the probability distribution over types for segment N+1

#### Scenario: Transition abruptness scoring
- **WHEN** analyzing transitions
- **THEN** an abruptness score quantifies how sudden or smooth the rebracketing change is

### Requirement: Temporal Positional Encoding
The system SHALL encode temporal distances between segments using actual time values.

#### Scenario: Time-based sinusoidal encoding
- **WHEN** using sinusoidal positional encoding
- **THEN** frequencies are based on actual time distances in seconds

#### Scenario: Learnable time embeddings
- **WHEN** using learnable encodings
- **THEN** time distances are discretized and mapped to learned embeddings

#### Scenario: Relative vs absolute time
- **WHEN** encoding temporal positions
- **THEN** configuration supports both absolute timestamps and relative offsets

### Requirement: Sequence-Aware Loss Functions
The system SHALL provide loss functions suitable for temporal sequence modeling.

#### Scenario: Sequence classification loss
- **WHEN** predicting labels for all segments in a sequence
- **THEN** loss is averaged or summed over the sequence

#### Scenario: Transition prediction loss
- **WHEN** predicting transitions between segments
- **THEN** separate loss terms for intensity delta, type transition, and abruptness

#### Scenario: Temporal consistency regularization
- **WHEN** encouraging smooth temporal evolution
- **THEN** regularization term penalizes abrupt unexplained changes

#### Scenario: Contrastive sequence loss
- **WHEN** learning sequence representations
- **THEN** similar temporal patterns are pulled together in embedding space

### Requirement: Next-Segment Prediction
The system SHALL predict properties of the next segment given current and past segments.

#### Scenario: Next segment type prediction
- **WHEN** given segments [N-2, N-1, N]
- **THEN** predict the rebracketing type of segment N+1

#### Scenario: Next segment intensity prediction
- **WHEN** given segment history
- **THEN** predict the intensity value of the next segment

#### Scenario: Prediction confidence
- **WHEN** making next-segment predictions
- **THEN** confidence scores reflect model uncertainty

### Requirement: Album-Level Sequence Processing
The system SHALL respect album boundaries and process sequences within album context.

#### Scenario: Album metadata integration
- **WHEN** loading sequences
- **THEN** album ID and position within album are available

#### Scenario: Cross-album boundary prevention
- **WHEN** a context window would span multiple albums
- **THEN** the window is truncated or padded at album boundaries

#### Scenario: Album-wise evaluation
- **WHEN** evaluating sequence models
- **THEN** metrics can be computed per-album to assess album-specific patterns

### Requirement: Sequence Evaluation Metrics
The system SHALL compute metrics specific to temporal sequence prediction.

#### Scenario: Next-segment accuracy
- **WHEN** evaluating next-segment predictions
- **THEN** accuracy measures correct prediction of the next segment's properties

#### Scenario: Transition prediction MAE
- **WHEN** evaluating transition predictions
- **THEN** MAE measures accuracy of intensity delta predictions

#### Scenario: Sequence-level accuracy
- **WHEN** predicting entire sequences
- **THEN** accuracy is measured at the sequence level (all correct vs any incorrect)

#### Scenario: Temporal consistency score
- **WHEN** evaluating predicted sequences
- **THEN** smoothness of transitions is quantified

### Requirement: Temporal Pooling Strategies
The system SHALL aggregate sequence representations using configurable pooling strategies.

#### Scenario: Last-step pooling
- **WHEN** pooling strategy is "last"
- **THEN** the final timestep's hidden state represents the sequence

#### Scenario: Mean pooling
- **WHEN** pooling strategy is "mean"
- **THEN** hidden states are averaged across all timesteps

#### Scenario: Max pooling
- **WHEN** pooling strategy is "max"
- **THEN** maximum activation across timesteps is used

#### Scenario: Attention pooling
- **WHEN** pooling strategy is "attention"
- **THEN** learned attention weights aggregate timesteps

### Requirement: Temporal Sequence Configuration
The system SHALL extend configuration to support temporal modeling parameters.

#### Scenario: Context window size
- **WHEN** config.dataset.context_window is set to 5
- **THEN** 5 previous and 5 next segments are loaded

#### Scenario: Sequential model selection
- **WHEN** config.model.temporal.type is "lstm", "transformer", or "tcn"
- **THEN** the corresponding model architecture is instantiated

#### Scenario: Positional encoding type
- **WHEN** config.model.temporal.positional_encoding is "sinusoidal" or "learnable"
- **THEN** the specified encoding strategy is used

#### Scenario: Pooling strategy selection
- **WHEN** config.model.temporal.pooling is "last", "mean", "max", or "attention"
- **THEN** the specified pooling is applied to sequence outputs

### Requirement: Temporal Data Augmentation
The system SHALL support augmentation strategies for temporal sequences.

#### Scenario: Sequence dropout
- **WHEN** augmentation is enabled
- **THEN** random segments within the context window are dropped

#### Scenario: Temporal jittering
- **WHEN** augmentation is enabled
- **THEN** small random time shifts are applied to segment boundaries

#### Scenario: Sequence reversal
- **WHEN** augmentation is enabled
- **THEN** sequences are occasionally reversed to test bidirectional understanding
