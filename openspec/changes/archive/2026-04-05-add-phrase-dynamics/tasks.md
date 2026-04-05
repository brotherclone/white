## 1. Curve Library
- [x] 1.1 Create `app/util/phrase_dynamics.py`
- [x] 1.2 Implement `DynamicCurve` enum: FLAT, LINEAR_CRESC, LINEAR_DIM, SWELL
- [x] 1.3 Implement `apply_dynamics_curve(notes, curve, min_vel, max_vel) → notes`
       — scales velocities along the curve across the note list while respecting clamps
- [x] 1.4 Implement `infer_curve(section_energy: str) → DynamicCurve`
       — intro→SWELL, chorus→LINEAR_CRESC, outro→LINEAR_DIM, else→FLAT
- [x] 1.5 Write `tests/util/test_phrase_dynamics.py` (all 4 curves, clamp enforcement)

## 2. Pipeline Integration
- [x] 2.1 `melody_pipeline.py`: call `apply_dynamics_curve` on note list before MIDI write;
       read curve from `song_proposal.yml` `dynamics` map or fall back to `infer_curve`
- [x] 2.2 `bass_pipeline.py`: same pattern
- [x] 2.3 `drum_pipeline.py`: same pattern (per-drum velocity scaling, not per-instrument)

## 3. Configuration
- [x] 3.1 Document `dynamics` map schema in `song_proposal.yml` docstring/comment:
       `dynamics: {verse_1: linear_cresc, chorus_1: flat}`
