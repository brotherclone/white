# Implementation Tasks

## 1. Prosodic Feature Extraction (Phase 3.3)
- [ ] 1.1 Set up forced alignment pipeline (Montreal Forced Aligner or Gentle)
- [ ] 1.2 Extract prosodic features: pitch contour, note duration, stress alignment, melisma
- [ ] 1.3 Run alignment on all vocal segments and store features
- [ ] 1.4 Handle instrumental segments (no lyrics) — null prosodic features

## 2. Prosodic Encoder
- [ ] 2.1 Build prosody MLP: prosodic features → `[batch, 256]`
- [ ] 2.2 Test: prosodic encoder output for vocal vs instrumental segments

## 3. Structural Feature Extraction (Phase 3.4)
- [ ] 3.1 Extract structural features: syllabic density, rhythmic alignment, phrase variance, repetition ratio
- [ ] 3.2 Compute features from lyric text + MIDI (no alignment needed)
- [ ] 3.3 Handle instrumental segments — null structural features

## 4. Structural Encoder
- [ ] 4.1 Build structure MLP: structural features → `[batch, 128]`
- [ ] 4.2 Test: structural encoder output dimensions and value ranges

## 5. Combined Lyric Encoder
- [ ] 5.1 Implement `ThreeProngLyricEncoder` concatenating semantic `[768]` + prosodic `[256]` + structural `[128]`
- [ ] 5.2 Handle partial availability (prosodic or structural absent → zero-fill that sub-encoding)
- [ ] 5.3 Update fusion model to accept `[batch, 1152]` lyric input instead of `[batch, 768]`
- [ ] 5.4 Test: combined output is `[batch, 1152]` when all prongs available

## 6. Training & Evaluation
- [ ] 6.1 Retrain fusion model with three-pronged lyric input
- [ ] 6.2 Compare vs Phase 3.2 baseline (semantic-only lyrics)
- [ ] 6.3 Ablation: measure contribution of prosodic vs structural features independently

## 7. Documentation
- [ ] 7.1 Document forced alignment setup and requirements
- [ ] 7.2 Document prosodic and structural feature definitions
- [ ] 7.3 Document impact on model dimensions and training
