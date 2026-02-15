## 1. Drum Pattern Template Library

- [ ] 1.1 Define template data structure — `DrumPattern(name, genre_family, energy, time_sig, description, voices)` with voice entries as `(beat_position, velocity_level)` tuples
- [ ] 1.2 Define GM percussion MIDI mapping constants and velocity level constants (accent=120, normal=90, ghost=45)
- [ ] 1.3 Implement 4/4 ambient templates — sparse kick, minimal hat, textural (low/medium/high energy)
- [ ] 1.4 Implement 4/4 electronic templates — pulse, four-on-floor, syncopated (low/medium/high)
- [ ] 1.5 Implement 4/4 krautrock templates — motorik (steady kick-every-beat, relentless hi-hat eighths, snare on 3), kosmische pulse variants (low/medium/high)
- [ ] 1.6 Implement 4/4 rock templates — basic backbeat, driving, half-time (low/medium/high)
- [ ] 1.7 Implement 4/4 experimental templates — irregular accents, rimshot-focused, sparse (low/medium/high)
- [ ] 1.8 Implement 7/8 templates — 3+2+2 and 2+2+3 groupings for ambient, electronic, krautrock, and rock families
- [ ] 1.9 Implement fallback pattern for unsupported time signatures — kick-on-1 minimal pattern
- [ ] 1.10 Unit tests for template validation — verify all templates have valid beat positions within bar length, valid velocity levels, and non-empty voices

## 2. Genre Family Mapping

- [ ] 2.1 Implement genre tag scanner — map genre strings to genre families via keyword matching
- [ ] 2.2 Unit tests for genre mapping — verify known genre tags map correctly, verify fallback to electronic

## 3. Drum Pipeline

- [ ] 3.1 Create `app/generators/midi/drum_pipeline.py` — main orchestrator
- [ ] 3.2 Implement section reader — parse chord `review.yml` to extract approved section labels and bar counts
- [ ] 3.3 Implement section-to-energy mapping with default levels and CLI override support
- [ ] 3.4 Implement template selection — filter templates by time signature, genre family, and energy (with adjacent energy inclusion)
- [ ] 3.5 Implement drum MIDI generation — convert template + BPM + bar count → MIDI bytes on channel 10
- [ ] 3.6 Implement ChromaticScorer integration — prepare concept once, score all candidates, compute composite (energy appropriateness + chromatic match)
- [ ] 3.7 Implement review YAML generation — reuse chord review schema, add section/genre/pattern/energy metadata per candidate
- [ ] 3.8 Add CLI entry point — `python -m app.generators.midi.drum_pipeline --production-dir <path>` with optional `--seed`, `--top-k`, `--energy-override`, `--genre-override`

## 4. Testing

- [ ] 4.1 Unit tests for section reader — parse a mock chord review.yml, extract sections and bar counts
- [ ] 4.2 Unit tests for template selection — verify correct filtering by time sig, genre, energy
- [ ] 4.3 Unit tests for drum MIDI generation — verify channel 10, correct note numbers, velocity values, bar repetition
- [ ] 4.4 Unit tests for energy appropriateness scoring — verify exact/adjacent/distant scores
- [ ] 4.5 Integration test — run drum pipeline on the approved Black song chords, verify output files and review.yml structure

## 5. Wiring

- [ ] 5.1 Update `training/openspec/TRAINING_ROADMAP.md` — add drum generation as a sub-step of Step 8
