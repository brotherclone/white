## 1. Chord Generation Pipeline

- [x] 1.1 Create `app/generators/midi/chord_pipeline.py` — main orchestrator that reads a song proposal, generates candidates, scores them, and writes output
- [x] 1.2 Add chromatic target mapping — function that maps rainbow color to target mode distributions (temporal/spatial/ontological soft targets)
- [x] 1.3 Integrate ChromaticScorer into chord pipeline — convert Markov-generated progressions to MIDI bytes, score batch with `ChromaticScorer.score_batch()`
- [x] 1.4 Implement composite scoring — weighted combination of existing theory scores and ChromaticScorer output, configurable weights
- [x] 1.5 Add MIDI export — write top-N candidates as .mid files with correct BPM/tempo from song proposal
- [x] 1.6 Add CLI entry point — `python -m app.generators.midi.chord_pipeline --thread <dir> --song <file>` with optional seed, num-candidates, top-k, weight params

## 2. Production Review Interface

- [x] 2.1 Add review YAML generation — write `review.yml` with candidate metadata, scores, and annotation placeholders after chord pipeline completes
- [x] 2.2 Add promotion command — `python -m app.generators.midi.promote_chords --review <path>` that copies approved candidates to `approved/` directory with label-based filenames
- [x] 2.3 Create output directory structure — `<thread>/production/<song_slug>/chords/{candidates,approved}/`

## 3. Testing

- [x] 3.1 Unit tests for chromatic target mapping (color → mode distributions)
- [x] 3.2 Unit tests for composite scoring (verify weighting, ranking)
- [x] 3.3 Integration test — run chord pipeline on a real song proposal, verify MIDI output and review.yml structure
- [x] 3.4 Test promotion command — verify approved candidates are copied with correct names

## 4. Documentation

- [x] 4.1 Update `app/generators/midi/prototype/README.md` to reference the new pipeline
- [x] 4.2 Update `training/openspec/TRAINING_ROADMAP.md` — add Step 8 entry linking to this pipeline
