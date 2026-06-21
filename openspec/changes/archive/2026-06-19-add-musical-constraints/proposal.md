# Change: Add Musical Constraints to Song Proposals

## Why

Color agents write rich harmonic intent as prose — things like "I→IV→I, the self
departs briefly toward the other and returns" or "a single chord sustained for two
minutes." The chord pipeline reads only `key`, `bpm`, `time_sig`, and `color` from
the proposal; all harmonic philosophy is discarded. The pipeline then runs Markov
chains that have no awareness of the agent's stated intent.

Two concrete cases motivate this change:

1. **Explicit function sequences** — the Violet agent specifies `I→IV→I` as the
   governing harmonic motion. The current system may happen upon this by chance
   but cannot be directed to produce it.

2. **Minimal harmonic structures** — the Indigo agent may specify a single chord
   held for the duration of the song. The Markov generator always produces 4-chord
   progressions and cannot express this.

These are the most tractable part of the harmonic intent gap. Beat-level performance
directives ("resolves on the 'and' of beat 2") are arrangement instructions for Logic
and are explicitly out of scope here.

## What Changes

A new optional `musical_constraints` block is added to the song proposal YAML schema.
When present, its fields override Markov-based chord generation in `chord_pipeline`:

```yaml
musical_constraints:
  harmonic_sequence: "i iv i"   # space-separated Roman numerals → bypasses Markov
  performance_notes: |           # human-readable; surfaced in review.yml, no pipeline effect
    Single tonic sustained throughout. Phrase resolutions anticipate the lyric.
```

- `harmonic_sequence` — a space-separated string of Roman numeral function tokens
  (`I`, `IV`, `V`, `vi`, `i`, `iv`, etc.) in the order they should appear. Each
  token maps directly to the chord bank via the existing `get_chord_by_function`
  lookup. A single token (`"i"`) expresses a one-chord song.

- `performance_notes` — free prose. Passed through to `review.yml` as a top-level
  `performance_notes` field so the human reviewer sees it during the MIDI review
  stage. No pipeline logic reads it.

When `harmonic_sequence` is absent the chord pipeline runs exactly as today.

### What is NOT changed

- Beat-level timing directives are a Logic Assembly concern and remain manual.
- Lyric/harmony coordination ("pre-answer directive") requires knowing syllable
  positions at MIDI generation time — impossible in the current architecture.
- Color agent system prompts are not updated in this change. Agents can be updated
  to output `musical_constraints` as a follow-on. Until then, the field is written
  by hand when the agent prose makes the intent clear.
- Section-level overrides (`section_overrides` per verse/chorus/bridge) are deferred
  to a follow-on. In v1 the top-level `harmonic_sequence` applies to every section.

## Scope

- `packages/core` — new `MusicConstraints` Pydantic model
- `packages/composition` — `load_song_proposal_unified` parses `musical_constraints`
- `packages/generation` — `chord_pipeline` adds a constrained generation path

## Impact

- Affected specs: `composition-proposal`, `chord-generation`
- Affected code:
  - `packages/core/src/white_core/structures/` — new `MusicConstraints` model
  - `packages/composition/src/white_composition/production_plan.py` — `load_song_proposal_unified` reads `musical_constraints`
  - `packages/generation/src/white_generation/pipelines/chord_pipeline.py` — `build_constrained_candidates`, `run_chord_pipeline` branches on constraints
  - `packages/generation/tests/test_chord_pipeline.py` — new constrained-path tests
