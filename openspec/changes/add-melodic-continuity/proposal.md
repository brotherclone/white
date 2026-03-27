# Change: Add cross-section melodic continuity constraint

## Why
Melody sections are generated independently; the last note of one section and the first
note of the next are chosen without reference to each other. This produces jarring leaps
at section boundaries that have to be fixed manually in Logic. A simple inter-section
interval constraint makes songs feel composed rather than assembled.

## What Changes
- After generating all melody candidates for a song, the pipeline applies a continuity
  pass: for each adjacent section pair in `production_plan.yml` order, candidates whose
  first note is more than N semitones from the last note of the preceding approved
  section are penalised in scoring (not excluded — human can still override)
- N defaults to 4 semitones (within a major 3rd); configurable per song via
  `song_proposal.yml` field `melodic_continuity_semitones`
- The penalty is a score multiplier: 0.85× per violated boundary
- Requires at least one approved melody section to compute the constraint

## Impact
- Affected specs: melody-generation
- Affected code: `app/generators/midi/pipelines/melody_pipeline.py`
- Not breaking — penalty only applies when a preceding approved section exists
