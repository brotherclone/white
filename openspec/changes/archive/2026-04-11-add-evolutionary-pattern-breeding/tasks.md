## 1. Evolutionary Engine
- [x] 1.1 Create `app/generators/midi/patterns/pattern_evolution.py`
- [x] 1.2 Implement `breed_drum_patterns(concept_emb, seed_patterns, generations, population_size, top_n) → list[DrumPattern]`
- [x] 1.3 Implement `breed_bass_patterns(concept_emb, chord_progression, seed_patterns, generations, population_size, top_n) → list[BassPattern]`
- [x] 1.4 Implement `breed_melody_patterns(concept_emb, chord_progression, seed_patterns, generations, population_size, top_n) → list[MelodyPattern]`
- [x] 1.5 Crossover: voice-row swap for drums; bar-boundary splice for bass/melody
- [x] 1.6 Mutation: probability 0.15, pattern-type-specific perturbations
- [x] 1.7 Selection: tournament (k=3) + elitism (top 2)
- [x] 1.8 Fitness: Refractor score_batch for chromatic match

## 2. Pipeline Integration
- [x] 2.1 `drum_pipeline.py`: add `--evolve`, `--generations`, `--population` flags; merge evolved candidates
- [x] 2.2 `bass_pipeline.py`: same flags; merge evolved candidates
- [x] 2.3 `melody_pipeline.py`: same flags; merge evolved candidates

## 3. Tests
- [x] 3.1 `tests/generators/midi/test_pattern_evolution.py`
- [x] 3.2 Test crossover produces valid DrumPattern, BassPattern, MelodyPattern
- [x] 3.3 Test evolved patterns carry `evolved` tag
- [x] 3.4 Test `breed_*` returns exactly `top_n` patterns
- [x] 3.5 Test fitness ordering preserved
