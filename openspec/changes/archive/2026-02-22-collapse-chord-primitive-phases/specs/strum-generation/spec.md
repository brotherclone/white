## REMOVED Requirements

### Requirement: Strum Pattern Templates
**Reason**: Strum articulation is now applied during chord primitive candidate generation. The
strum template library (`strum_patterns.py`) is retained as an internal module used by the
chord pipeline, but it is no longer a public pipeline stage.
**Migration**: None required.

### Requirement: Strum Pipeline Input
**Reason**: The strum pipeline no longer reads from `chords/approved/` as a separate phase.
Strum is applied before promotion, at candidate generation time.
**Migration**: None required.

### Requirement: Strum Candidate Generation
**Reason**: Superseded by the chord primitive collapse. Strum patterns are now sampled and
applied within the chord pipeline candidate generation loop.
**Migration**: None required.

### Requirement: Strum MIDI Output
**Reason**: The `strums/candidates/` and `strums/approved/` directories are no longer generated.
Strummed MIDI lives in `chords/candidates/` as part of the chord primitive.
**Migration**: None required.

### Requirement: Strum CLI Interface
**Reason**: `strum_pipeline.py` is removed. Strum control is via the chord pipeline's
`--strum-patterns` flag.
**Migration**: None required.
