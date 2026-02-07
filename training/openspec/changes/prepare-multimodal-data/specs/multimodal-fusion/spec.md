# Multimodal Fusion

## ADDED Requirements

### Requirement: Multimodal Data Readiness
The system SHALL verify that training data meets minimum coverage and quality thresholds before multimodal model training begins.

#### Scenario: Text embedding availability
- **WHEN** the multimodal dataset is loaded for training
- **THEN** DeBERTa-v3 text embeddings (`[768]` half-float) SHALL be present for all segments with non-empty `lyric_text`

#### Scenario: Audio coverage threshold
- **WHEN** audio coverage is assessed
- **THEN** at least 80% of segments SHALL have `audio_waveform` binary data (current: 85.1%)

#### Scenario: MIDI coverage documentation
- **WHEN** MIDI coverage is assessed
- **THEN** per-album MIDI coverage percentages SHALL be documented and segments with `has_midi=False` SHALL be handled via modality masking (current overall: 43.3%)

#### Scenario: Label completeness audit
- **WHEN** the dataset is prepared for supervised training
- **THEN** UNLABELED segments (missing `rainbow_color`) SHALL be audited against manifest data and either labeled or explicitly excluded from color-supervised training

#### Scenario: Binary-flag consistency
- **WHEN** `has_midi` or `has_audio` flags are used for data loading decisions
- **THEN** they SHALL match actual binary column presence (non-null and non-empty)
