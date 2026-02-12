# Multimodal Fusion

## ADDED Requirements

### Requirement: Three-Pronged Lyric Encoding
The system SHALL encode lyrics at three levels — semantic, prosodic, and structural — and combine them into a unified lyric embedding. Prosodic encoding requires forced alignment; structural encoding requires lyric text and MIDI. Semantic encoding reuses the existing DeBERTa encoder from Phases 1-4.

#### Scenario: Semantic encoding
- **WHEN** lyric text is available for a segment
- **THEN** the existing DeBERTa-v3 encoder produces a `[batch, 768]` semantic embedding

#### Scenario: Prosodic encoding
- **WHEN** forced alignment data is available for a vocal segment
- **THEN** prosodic features (pitch contour, stress patterns, melisma) are extracted and encoded to `[batch, 256]`

#### Scenario: Structural encoding
- **WHEN** lyric text and MIDI are available for a segment
- **THEN** structural features (syllabic density, rhythmic alignment, phrase variance) are extracted and encoded to `[batch, 128]`

#### Scenario: Instrumental segment (no lyrics)
- **WHEN** a segment has no lyric text (`vocals=False` or `lyric_text` is empty)
- **THEN** all three lyric sub-encodings return zero tensors and the lyric modality mask is set to False

#### Scenario: Combined lyric embedding
- **WHEN** all available lyric sub-encodings are computed
- **THEN** they are concatenated to produce a lyric embedding up to `[batch, 1152]`
