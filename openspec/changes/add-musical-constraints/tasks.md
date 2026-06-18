# Tasks: add-musical-constraints

## Implementation Order

- [x] **1. `MusicConstraints` model** (`white_core`)
  - Add `packages/core/src/white_core/structures/music_constraints.py`
  - Pydantic `BaseModel` with `harmonic_sequence: str | None = None` and
    `performance_notes: str | None = None`
  - Export from `white_core.structures.__init__`
  - Unit test: valid construction, null defaults, extra fields ignored

- [x] **2. Proposal parser update** (`white_composition`)
  - In `load_song_proposal_unified` (`production_plan.py`), read
    `raw.get("musical_constraints")`, construct `MusicConstraints(**block)` when
    present, add `"musical_constraints": mc_or_none` to the returned dict
  - Guard: if `musical_constraints` in YAML is not a dict, log a warning and set `None`
  - Tests: proposal with full block, proposal without block, proposal with partial block

- [x] **3. `build_constrained_candidates`** (`white_generation`)
  - New function in `chord_pipeline.py` alongside `build_diatonic_candidates`
  - Signature: `build_constrained_candidates(harmonic_sequence, key_root, mode, bpm, time_sig, gen, rng, genre_families, num_candidates, scorer, concept_emb, target, theory_weight, chromatic_weight) → list[dict]`
  - Parse tokens, look up chords per token, produce `num_candidates` voicing
    combinations (sample one chord per position per candidate), score with
    `composite_score`, return top `num_candidates` in candidate dict format with
    `source: "constrained"` and `harmonic_sequence: <string>`
  - Unknown token → warn + skip; empty result after skips → return `[]`
  - Tests: three-chord sequence, single-chord sequence, unknown token graceful skip,
    no-op when `harmonic_sequence=None`

- [x] **4. Wire into `run_chord_pipeline`** (`white_generation`)
  - After loading `song_info`, extract `constraints = song_info.get("musical_constraints")`
  - If `constraints` and `constraints.harmonic_sequence`, call
    `build_constrained_candidates(...)` and prepend results to the scored pool before
    final ranking
  - If `constraints` and `constraints.performance_notes`, add `performance_notes` key
    to the `review` dict before writing `review.yml`
  - No change to Markov or diatonic paths

- [x] **5. `review.yml` labelling**
  - In `generate_review_yaml`, handle `source: "constrained"` alongside `"markov"` and
    `"diatonic"` — add `harmonic_sequence` field to constrained entries, friendly
    `notes` string: `"Constrained — sequence from proposal: <sequence>"`

- [x] **6. Tests**
  - `test_chord_pipeline.py`: integration test with a mock song proposal containing
    `harmonic_sequence: "i iv i"` — assert at least one constrained candidate in
    output with correct source and sequence fields
  - `test_chord_pipeline.py`: single-token `"i"` produces single-chord candidates
  - `test_chord_pipeline.py`: no `musical_constraints` → no constrained candidates,
    pipeline output unchanged

## Validation

- Run `openspec validate add-musical-constraints --strict` before implementation
- Run `pytest packages/generation/tests/test_chord_pipeline.py packages/composition/tests/ packages/core/tests/` after each task
- Spot-check: run chord pipeline on an existing proposal (no `musical_constraints`) and
  confirm `review.yml` is identical to pre-change output
- Spot-check: add `musical_constraints: {harmonic_sequence: "i iv i"}` to a test
  proposal and confirm constrained candidates appear at the top of `review.yml`

## Dependencies

- Task 1 must complete before Tasks 2 and 3
- Tasks 2 and 3 can run in parallel after Task 1
- Task 4 depends on Tasks 2 and 3
- Tasks 5 and 6 depend on Task 4
