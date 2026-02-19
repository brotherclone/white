# Tasks: Add Harmonic Rhythm Generation

## Implementation Order

### Core Pipeline

- [x] **Create `app/generators/midi/harmonic_rhythm.py`** — Duration distribution generator: enumerate valid half-bar distributions for N chords within min/max bounds, with seeded sampling when count > 200. Include the uniform baseline always.
- [x] **Create drum accent extractor** — Parse approved drum MIDI to extract accented beat positions (velocity >= 80), quantize to half-bar grid, produce accent mask per bar. Add fallback mask (bar starts only) when no drums exist. Lives in `harmonic_rhythm.py`.
- [x] **Implement drum alignment scoring** — Score each distribution by fraction of chord onsets landing on strong half-bar positions from the tiled drum accent mask.
- [x] **Implement MIDI generation for scoring** — Generate block-chord MIDI bytes from a distribution (each chord sustained for its assigned duration at song BPM). Reuse voicing data from approved chord MIDI files.
- [x] **Integrate ChromaticScorer** — Score each candidate's MIDI with ChromaticScorer, extract temporal match against color target. Compute concept embedding once per section.
- [x] **Implement composite scoring + ranking** — Weighted composite (0.3 drum alignment + 0.7 chromatic temporal), rank per section, top-k selection.
- [x] **Create `app/generators/midi/harmonic_rhythm_pipeline.py`** — Full orchestrator: read inputs (chord review, drum review), run generation + scoring, write candidates + review.yml. Include CLI with `--production-dir`, `--seed`, `--top-k`, `--onnx-path`.

### Tests

- [x] **Unit tests for distribution enumeration** — Verify: minimum 0.5 per chord, max total = N*2.0, uniform always included, sampling cap at 200, deterministic with seed.
- [x] **Unit tests for drum accent extraction** — Verify: accent threshold, quantization to half-bar grid, tolerance window, fallback mask.
- [x] **Unit tests for drum alignment scoring** — Verify: all-on-strong = 1.0, none-on-strong = 1/N (first chord), tiling across multiple bars.
- [x] **Unit tests for MIDI generation** — Verify: correct note durations match distribution, BPM is set, voicings preserved.
- [x] **Integration test** — End-to-end with mock ChromaticScorer: approved chords + approved drums → candidates + review.yml with expected structure.

### Strum Pipeline Modification

- [x] **Modify `strum_pipeline.py`** — When `harmonic_rhythm/approved/` exists, read approved duration map and pass variable durations to `strum_to_midi_bytes()` instead of uniform 1-bar. Pattern repeats for longer chords, truncates for shorter.
- [x] **Update `strum_to_midi_bytes()`** — Accept optional `durations: list[float]` parameter (bars per chord). When provided, each chord gets `duration * bar_ticks` instead of a uniform bar.
- [x] **Test strum backward compatibility** — Verify strum pipeline produces identical output when no harmonic rhythm is approved.

### Documentation

- [x] **Update `app/generators/midi/README.md`** — Add harmonic rhythm CLI docs, update pipeline order diagram.
- [x] **Update `training/openspec/TRAINING_ROADMAP.md`** — Add harmonic rhythm phase entry.

## Dependencies

- Distribution enumeration and drum accent extraction are independent (parallelizable)
- MIDI generation depends on distribution enumeration
- ChromaticScorer integration depends on MIDI generation
- Composite scoring depends on both drum alignment and chromatic scoring
- Strum modification depends on the core pipeline being complete
- Tests can be written alongside each implementation step
