## 1. Strum Pattern Templates

- [ ] 1.1 Define rhythm pattern data structure — `StumPattern(name, description, onsets_beats, durations_beats)` or equivalent
- [ ] 1.2 Implement 4/4 patterns — whole, half, quarter, eighth, syncopated, arp-up, arp-down
- [ ] 1.3 Implement 7/8 patterns — whole, grouped-322, grouped-223, eighth
- [ ] 1.4 Add fallback for unsupported time signatures — equal-subdivision whole and beat patterns

## 2. Strum Pipeline

- [ ] 2.1 Create `app/generators/midi/strum_pipeline.py` — main orchestrator that reads approved chords, applies patterns, writes output
- [ ] 2.2 Implement approved chord MIDI parsing — extract chord voicings (note sets per bar) from `.mid` files
- [ ] 2.3 Implement pattern application — take a chord voicing + rhythm pattern, produce new MIDI bytes with correct timing
- [ ] 2.4 Implement arpeggio handling — distribute chord tones across subdivisions with cycling for up/down patterns
- [ ] 2.5 Implement progression mode — concatenate all approved chords with the same pattern applied to each
- [ ] 2.6 Add review YAML generation — reuse chord review schema, add source chord and pattern metadata
- [ ] 2.7 Add CLI entry point — `python -m app.generators.midi.strum_pipeline --production-dir <path>` with optional `--mode`, `--patterns`

## 3. Testing

- [ ] 3.1 Unit tests for pattern templates — verify onset/duration sums match bar length for each time signature
- [ ] 3.2 Unit tests for MIDI parsing — extract voicings from a known MIDI file
- [ ] 3.3 Unit tests for pattern application — verify note count, timing, and duration in output MIDI
- [ ] 3.4 Unit tests for arpeggio handling — verify note ordering and cycling
- [ ] 3.5 Integration test — run strum pipeline on the approved Black song chords, verify output files and review.yml

## 4. Documentation

- [ ] 4.1 Update `training/openspec/TRAINING_ROADMAP.md` — add strum generation as a sub-step of Step 8
