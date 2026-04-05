# Change: Add phrase-level dynamic shaping to melody, bass, and drum MIDI

## Why
All generated MIDI currently uses fixed velocity tiers (accent/normal/ghost). Within a
section, there is no crescendo, diminuendo, or swell — every bar sounds equally loud.
Phrase-level dynamics applied as a velocity curve over the section's duration makes the
output feel performed rather than programmed.

## What Changes
- New utility: `app/util/phrase_dynamics.py` — applies a named dynamic curve
  (linear_cresc, linear_dim, swell, flat) to a sequence of MIDI note velocities
- Curve is specified per section in `song_proposal.yml` under a `dynamics` map, or
  inferred from section energy (intro→swell, chorus→linear_cresc, outro→linear_dim,
  verse/bridge→flat by default)
- Each pipeline (melody, bass, drum) applies `apply_dynamics_curve()` after generating
  note events, before writing MIDI bytes
- Velocity clamps respected: melody 60–127, bass 50–110, drums 45–127

## Impact
- Affected specs: melody-generation, bass-generation, drum-generation
- Affected code: new `app/util/phrase_dynamics.py`; `melody_pipeline.py`,
  `bass_pipeline.py`, `drum_pipeline.py` each get a one-line post-process call
- Not breaking — default curve for existing sections is `flat` (no change)
