# Change: Add Prosodic and Structural Lyric Encoding

## Why

The multimodal fusion model (Phase 3.1/3.2) captures audio, MIDI, and semantic text. But lyrics contain two additional dimensions that semantic embedding alone misses:

1. **Prosody**: How lyrics are *delivered* — pitch contour, stress patterns, melisma, legato vs staccato
2. **Structure**: Rhythmic form — syllabic density, phrase length, repetition patterns

These features encode chromatic distinctions that pure text meaning cannot:
- GREEN: Sustained notes on place names, legato phrasing, sparse syllables
- RED: Rapid syllabic delivery, staccato, syncopation
- VIOLET: Rubato, unexpected pitch leaps, metric modulation

## Prerequisites

- `add-multimodal-fusion` MUST be implemented first (audio + MIDI + semantic text fusion)
- Results from Phase 3.1/3.2 should demonstrate that audio+MIDI improves spatial accuracy before investing in prosodic complexity
- If Phase 3.1 alone achieves >85% spatial accuracy, this change may be deprioritized

## What Changes

### Prosodic Encoding (Phase 3.3)
- Set up forced alignment pipeline (Montreal Forced Aligner or Gentle)
- Extract prosodic features from alignment:
  - Pitch contour per syllable (mean, std, range)
  - Note duration per phoneme
  - Stress pattern matching (pitch vs lexical stress alignment)
  - Melisma detection (single note vs multiple per syllable)
  - Legato vs staccato (note overlap with next syllable)
- Add prosody MLP: prosodic features → `[batch, 256]`

### Structural Encoding (Phase 3.4)
- Extract structural features (no alignment needed — counting):
  - Notes per syllable, melisma ratio
  - Syllabic density (syllables per measure)
  - Rhythmic alignment (onset sync with beat grid)
  - Phrase length variance, repetition ratio
- Add structure MLP: structural features → `[batch, 128]`

### Combined Lyric Encoder
- `ThreeProngLyricEncoder` concatenates: semantic `[768]` + prosodic `[256]` + structural `[128]` = `[1152]`
- Replaces the semantic-only text input in the fusion model

## Impact

- Affected specs: multimodal-fusion (adds lyric encoding sub-requirements)
- Affected code: new `training/models/lyric_encoder.py`, fusion model forward pass dimensions change
- **External dependency**: Montreal Forced Aligner (Conda-based install)
- **BREAKING**: Fusion model input dimension changes from `768` (text only) to `1152` (three-pronged) — requires retraining

## Dependencies

**Python packages**:
```
montreal-forced-aligner  # Prosodic alignment (Conda)
```

**External tools**:
- Montreal Forced Aligner: https://github.com/MontrealCorpusTools/Montreal-Forced-Aligner
- Or Gentle: https://github.com/lowerquality/gentle (Python-based alternative)
