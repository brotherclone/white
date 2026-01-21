# Data Augmentation

## ADDED Requirements

### Requirement: Audio Augmentation Pipeline
The system SHALL augment audio waveforms while preserving rebracketing labels.

#### Scenario: Time stretch
- **WHEN** time stretch augmentation is applied
- **THEN** audio duration is changed by ±10% without pitch shift

#### Scenario: Pitch shift
- **WHEN** pitch shift augmentation is applied
- **THEN** pitch is shifted by ±2 semitones without duration change

#### Scenario: Noise injection
- **WHEN** noise injection is applied
- **THEN** white or pink noise is added at low volume

#### Scenario: Reverb and EQ
- **WHEN** effect augmentation is applied
- **THEN** reverb, EQ, or compression effects are added

### Requirement: Text Augmentation Pipeline
The system SHALL augment text while preserving rebracketing markers and semantic content.

#### Scenario: Back-translation
- **WHEN** back-translation augmentation is applied
- **THEN** text is translated to another language and back to English

#### Scenario: Synonym replacement
- **WHEN** synonym replacement is applied
- **THEN** non-marker words are replaced with synonyms

#### Scenario: Rebracketing marker preservation
- **WHEN** any text augmentation is applied
- **THEN** rebracketing markers (brackets, special notation) are preserved

### Requirement: MIDI Augmentation Pipeline
The system SHALL augment MIDI events while preserving musical structure.

#### Scenario: Transposition
- **WHEN** transposition augmentation is applied
- **THEN** all pitches are shifted by ±3 semitones

#### Scenario: Velocity randomization
- **WHEN** velocity augmentation is applied
- **THEN** note velocities are randomly adjusted within a range

#### Scenario: Time quantization
- **WHEN** quantization augmentation is applied
- **THEN** note timings are quantized or humanized

### Requirement: Synthetic Data Generation
The system SHALL generate synthetic labeled segments using White Agent.

#### Scenario: Generate for underrepresented classes
- **WHEN** synthetic generation is enabled
- **THEN** segments for rare rebracketing types are generated

#### Scenario: Quality filtering
- **WHEN** synthetic data is generated
- **THEN** quality thresholds filter low-quality samples

#### Scenario: Integration with training
- **WHEN** synthetic data passes quality checks
- **THEN** it is added to the training dataset

### Requirement: Augmentation Configuration
The system SHALL provide comprehensive configuration for augmentation strategies.

#### Scenario: Per-modality augmentation enable
- **WHEN** config.augmentation.audio.enabled is True
- **THEN** audio augmentations are applied during training

#### Scenario: Augmentation probability
- **WHEN** config.augmentation.audio.probability is 0.5
- **THEN** each augmentation is applied with 50% probability

#### Scenario: Method selection
- **WHEN** config.augmentation.audio.methods includes ["time_stretch", "pitch_shift"]
- **THEN** only those methods are used

### Requirement: Label Preservation Validation
The system SHALL verify that augmentations do not corrupt labels.

#### Scenario: Pre-post augmentation validation
- **WHEN** augmentation is applied
- **THEN** rebracketing labels are verified unchanged

#### Scenario: Rebracketing marker validation
- **WHEN** text is augmented
- **THEN** presence and position of rebracketing markers are validated
