## REMOVED Requirements

### Requirement: Half-Bar Duration Grid
**Reason**: HR distribution is now applied during chord primitive candidate generation, not as a
separate pipeline phase. The `harmonic_rhythm/` production directory is no longer created.
**Migration**: Re-run chord generation to get candidates with HR baked in.

### Requirement: Drum Accent Extraction
**Reason**: Superseded by the chord primitive collapse. Drum alignment scoring for HR candidates
is no longer needed as a standalone scoring pass.
**Migration**: None required.

### Requirement: Drum Alignment Scoring
**Reason**: Superseded by the chord primitive collapse.
**Migration**: None required.

### Requirement: Chromatic Temporal Scoring
**Reason**: Superseded. Chromatic scoring now occurs at chord candidate generation time and
covers the full primitive (voicings + HR + strum).
**Migration**: None required.

### Requirement: Composite Scoring and Ranking
**Reason**: Superseded by the chord primitive collapse.
**Migration**: None required.

### Requirement: Harmonic Rhythm MIDI Output
**Reason**: The `harmonic_rhythm/candidates/` and `harmonic_rhythm/approved/` directories are
no longer generated. HR is baked into chord primitives in `chords/candidates/`.
**Migration**: None required.

### Requirement: Harmonic Rhythm CLI Interface
**Reason**: The `harmonic_rhythm_pipeline.py` module is removed. Users control HR via the
chord pipeline's `--seed` and `--strum-patterns` flags.
**Migration**: None required.
