# Chromatic Style Transfer

## ADDED Requirements

### Requirement: Chromatic Style Encoding
The system SHALL extract chromatic style representations that capture the ontological essence of each chromatic mode.

#### Scenario: Per-mode style embedding
- **WHEN** extracting style from a BLACK segment
- **THEN** a style vector capturing BLACK chromatic characteristics is returned

#### Scenario: Style vector dimensionality
- **WHEN** style encoder processes any segment
- **THEN** output is a fixed-dimensional style vector (e.g., 256-dim)

#### Scenario: Style normalization
- **WHEN** style vectors are extracted
- **THEN** they are normalized to unit length for stable interpolation

### Requirement: Content-Style Disentanglement
The system SHALL separate content (what is said) from style (how it's said in chromatic terms).

#### Scenario: Disentangled encoding
- **WHEN** a segment is encoded
- **THEN** independent content and style vectors are produced

#### Scenario: Orthogonality constraint
- **WHEN** training disentangled encoder
- **THEN** content and style vectors are encouraged to be orthogonal

#### Scenario: Reconstruction from content+style
- **WHEN** content and style vectors are fed to decoder
- **THEN** original segment is reconstructed

### Requirement: Style Transfer Generation
The system SHALL generate segments with content from one source and style from another chromatic mode.

#### Scenario: Cross-mode transfer
- **WHEN** BLACK content is combined with ORANGE style
- **THEN** a segment is generated that preserves BLACK semantics in ORANGE ontological mode

#### Scenario: Multi-modal generation
- **WHEN** generating transferred segments
- **THEN** text, audio, and MIDI outputs reflect the target chromatic style

#### Scenario: Style interpolation
- **WHEN** interpolating between RED and BLUE styles
- **THEN** intermediate chromatic characteristics are generated

### Requirement: Style Reconstruction Loss
The system SHALL train encoders and decoders to accurately reconstruct style from style vectors.

#### Scenario: Style reconstruction
- **WHEN** a segment's style is encoded and decoded
- **THEN** reconstruction loss measures fidelity to original style

#### Scenario: Per-modality style loss
- **WHEN** style includes text, audio, and MIDI characteristics
- **THEN** separate loss terms measure style preservation in each modality

### Requirement: Content Preservation Loss
The system SHALL ensure content semantics are preserved during style transfer.

#### Scenario: Semantic similarity
- **WHEN** content is transferred to a different style
- **THEN** semantic similarity (e.g., cosine similarity of embeddings) is maximized

#### Scenario: Key concept preservation
- **WHEN** style transfer is applied
- **THEN** core conceptual elements remain intact

### Requirement: Adversarial Training for Realism
The system SHALL use adversarial training to ensure transferred segments are indistinguishable from real segments of the target chromatic mode.

#### Scenario: Discriminator training
- **WHEN** discriminator is trained
- **THEN** it learns to distinguish real ORANGE segments from BLACK-to-ORANGE transfers

#### Scenario: Generator training
- **WHEN** generator is trained
- **THEN** it learns to fool the discriminator by producing realistic transfers

#### Scenario: Per-mode discriminators
- **WHEN** using separate discriminators per chromatic mode
- **THEN** each learns mode-specific realism criteria

### Requirement: Style Consistency Evaluation
The system SHALL measure whether transferred segments match the target chromatic style.

#### Scenario: Style classifier evaluation
- **WHEN** a segment is transferred to ORANGE
- **THEN** a style classifier verifies it is recognized as ORANGE

#### Scenario: Style confusion matrix
- **WHEN** evaluating transfers across all chromatic modes
- **THEN** confusion matrices show which transfers are successful

### Requirement: Content Preservation Evaluation
The system SHALL measure whether semantic content is preserved during style transfer.

#### Scenario: Semantic similarity score
- **WHEN** comparing source content and transferred content
- **THEN** embedding cosine similarity quantifies content preservation

#### Scenario: Key concept matching
- **WHEN** evaluating content preservation
- **THEN** presence of critical concepts is verified

### Requirement: Chromatic Mode Labels
The system SHALL integrate chromatic mode labels (BLACK, RED, ORANGE, YELLOW, GREEN, BLUE, INDIGO, VIOLET, WHITE) into the training pipeline.

#### Scenario: Mode label loading
- **WHEN** loading training data
- **THEN** chromatic mode is included as a categorical label

#### Scenario: Mode-conditioned generation
- **WHEN** generating a segment
- **THEN** target chromatic mode is provided as conditioning input

#### Scenario: Unknown mode handling
- **WHEN** a segment lacks chromatic mode annotation
- **THEN** it is either inferred, assigned to UNKNOWN, or excluded

### Requirement: Style Transfer Configuration
The system SHALL provide configuration for style transfer training and inference.

#### Scenario: Loss weight configuration
- **WHEN** config.training.style_transfer_loss_weights is specified
- **THEN** reconstruction, content, style, and adversarial losses are weighted accordingly

#### Scenario: Discriminator enable/disable
- **WHEN** config.training.adversarial.enabled is False
- **THEN** adversarial training is skipped

#### Scenario: Interpolation steps configuration
- **WHEN** performing style interpolation
- **THEN** number of intermediate steps is configurable

### Requirement: Gradient Balancing for Multi-Loss Training
The system SHALL balance gradients from multiple loss functions to ensure stable training.

#### Scenario: Per-loss gradient monitoring
- **WHEN** backward pass completes
- **THEN** gradient norms from each loss term are logged

#### Scenario: Adaptive loss weighting
- **WHEN** one loss dominates training
- **THEN** weights are adjusted to balance gradient magnitudes

### Requirement: Style Interpolation
The system SHALL enable smooth interpolation between chromatic styles.

#### Scenario: Linear style interpolation
- **WHEN** interpolating between RED and BLUE style vectors
- **THEN** intermediate vectors produce smooth chromatic transitions

#### Scenario: Spherical interpolation
- **WHEN** style vectors are normalized
- **THEN** spherical interpolation (slerp) maintains constant magnitude

#### Scenario: Interpolation visualization
- **WHEN** interpolating styles
- **THEN** generated segments at intermediate points are visualized
