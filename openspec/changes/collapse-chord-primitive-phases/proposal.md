# Change: Collapse HR + Strum into Chord Primitives

## Why

The production pipeline currently treats chord voicings, harmonic rhythm (HR), and strum articulation
as three separate sequential phases, each producing approved MIDI files that fan out into compound
section names (`hr_verse_003.mid`, `hr_chorus_002_strum.mid`). By the time melody generation runs,
the ancestry chain (`melody_hr_verse_003.mid`) makes the assembly table unreadable and breaks the
1:1 pairing between drums and chord variants.

## What Changes

- **Chord primitives are complete on promotion**: HR distribution and strum articulation are
  baked into each chord candidate before promotion. A promoted chord file (`verse.mid`) already
  contains the full rhythmic and articulatory treatment — no downstream HR or strum phase needed.
- **Scratch beats accompany candidates**: A lightweight scratch drum MIDI
  (`<candidate>_scratch.mid`) is auto-generated alongside each chord candidate for auditioning
  purposes. Scratch files are saved but never promoted and never referenced by downstream phases.
- **One approved chord per section label**: Promotion enforces a single file per section label.
  Users who want multiple loop variants for a section must use distinct labels from the start
  (e.g., `verse_a`, `verse_b`). The `verse_1.mid` / `verse_2.mid` numbering pattern is removed.
- **HR and Strum phases are retired**: `harmonic_rhythm/` and `strums/` directories are no longer
  generated. Their pipeline files (`harmonic_rhythm_pipeline.py`, `strum_pipeline.py`) and specs
  are removed. The `drum-generation` spec is updated to read section labels only from
  `chords/approved/`, removing any reference to HR variant names.

## Impact

- Affected specs: `chord-generation`, `drum-generation`, `production-review`,
  `harmonic-rhythm` (REMOVED), `strum-generation` (REMOVED)
- Affected code: `app/generators/midi/chord_pipeline.py`, `app/generators/midi/promote_chords.py`,
  `app/generators/midi/drum_pipeline.py`, `app/generators/midi/harmonic_rhythm_pipeline.py`
  (deleted), `app/generators/midi/strum_pipeline.py` (deleted), and all related tests
- **Conflict — `add-production-plan` (active, unimplemented)**: That change's production plan
  generation reads bar counts from `harmonic_rhythm/approved/` with a fallback to
  `chords/approved/`. Since HR no longer has its own directory, this fallback chain simplifies
  to `chords/approved/` only. The `add-production-plan` spec delta must be updated before
  implementation to remove the HR reference.
- **BREAKING**: Any songs with existing `harmonic_rhythm/` or `strums/` production directories
  will have those directories orphaned (not migrated). Manual re-run of chord generation is
  required for songs not yet past the chord approval stage.
