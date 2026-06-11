## ADDED Requirements

### Requirement: Constrained Chord Generation Path

When the loaded song proposal contains a `musical_constraints.harmonic_sequence`, the
chord pipeline SHALL use a constrained generation path instead of Markov sampling.

A new function `build_constrained_candidates` SHALL be added to `chord_pipeline.py`.
It SHALL:

1. Parse `harmonic_sequence` into an ordered list of Roman numeral tokens
   (split on whitespace; e.g. `"i iv i"` → `["i", "iv", "i"]`).
2. For each token, look up chords from the chord bank via
   `gen.get_chord_by_function(key_root, mode, token, category="triad")`,
   falling back to any category if no triad exists.
3. If any token resolves to zero chords in the bank, skip that token with a warning
   rather than failing the whole pipeline.
4. Produce multiple candidate progressions by independently sampling one chord per
   token position from the available pool, up to `num_candidates` total. Each
   candidate is a different voicing combination of the fixed sequence.
5. Score each candidate through the same composite pipeline (theory + Refractor) used
   for Markov candidates.
6. Return the top `top_k` as ranked candidate dicts in the same format as
   `build_diatonic_candidates`, with `source: "constrained"` and
   `id: "constrained_NNN"`.

When `harmonic_sequence` is present, Markov generation SHALL still run alongside the
constrained path (same `num_candidates` budget), so the human reviewer always sees
both organic Markov results and the explicitly-directed sequence. Constrained
candidates appear at the top of `review.yml` ranked by composite score.

When `harmonic_sequence` is absent, behavior is unchanged from current.

#### Scenario: Constrained generation produces candidates for "i IV i"

- **GIVEN** a song proposal with `harmonic_sequence: "i IV i"` in C minor
- **WHEN** `run_chord_pipeline` is invoked
- **THEN** the pipeline SHALL call `build_constrained_candidates` with tokens
  `["i", "IV", "i"]`
- **AND** each constrained candidate SHALL be a three-chord progression where
  chord 1 and 3 use a minor tonic chord and chord 2 uses a major IV chord
- **AND** constrained candidates SHALL appear in `review.yml` with `source: "constrained"`
- **AND** Markov candidates SHALL also appear (source: "markov") in the same file

#### Scenario: Single-chord "one chord for two minutes" case

- **GIVEN** a proposal with `harmonic_sequence: "i"`
- **WHEN** the pipeline runs
- **THEN** `build_constrained_candidates` SHALL produce candidates each containing
  exactly one chord (the minor tonic in various voicings)
- **AND** Markov generation SHALL also run and produce its normal multi-chord results
- **AND** both appear in `review.yml` — the human can promote either

#### Scenario: Unknown function token — graceful skip

- **GIVEN** a `harmonic_sequence` containing a token with no matching chords in the
  bank (e.g. `"bVII"` in a key where that degree is absent)
- **WHEN** `build_constrained_candidates` processes it
- **THEN** a warning SHALL be printed but the pipeline SHALL NOT raise
- **AND** the token SHALL be skipped; remaining tokens proceed normally

#### Scenario: harmonic_sequence absent — unchanged behaviour

- **GIVEN** a proposal with no `musical_constraints` or no `harmonic_sequence`
- **WHEN** `run_chord_pipeline` is invoked
- **THEN** it SHALL behave exactly as before this change: Markov generation only,
  no constrained candidates added

### Requirement: Constrained Candidate Labelling in review.yml

Constrained candidates SHALL be visually distinguishable in `review.yml` from Markov
and diatonic candidates so the reviewer understands their origin.

Each constrained candidate entry SHALL include:
- `source: "constrained"`
- `harmonic_sequence` field echoing the token string from the proposal
  (e.g. `harmonic_sequence: "i iv i"`)

#### Scenario: review.yml distinguishes constrained from Markov

- **GIVEN** a pipeline run with `harmonic_sequence: "i iv i"`
- **WHEN** `review.yml` is written
- **THEN** constrained entries SHALL have `source: constrained` and
  `harmonic_sequence: "i iv i"`
- **AND** Markov entries SHALL have `source: markov` and no `harmonic_sequence` field
