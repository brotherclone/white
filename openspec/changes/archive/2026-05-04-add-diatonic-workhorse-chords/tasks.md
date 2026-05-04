# Tasks: add-diatonic-workhorse-chords

- [x] Add `DIATONIC_PATTERNS` constant to `chord_pipeline.py` — major and minor pattern sets
- [x] Implement `build_diatonic_candidates(key_root, mode, bpm, time_sig, gen, rng, genre_families)` — looks up chords by function, generates MIDI, returns list of candidate dicts with `source: diatonic`
- [x] Call `build_diatonic_candidates()` after Markov ranking, append results + write MIDI files before review.yml
- [x] Add `source` field to review.yml candidate schema; diatonic entries have `scores: null`, `rank: null`, `notes: "Diatonic workhorse — assign to verse sections"`
- [x] Update spec to reflect that sections aren't known at generation time (diatonic candidates go into the flat pool; reviewer assigns to verses)
- [x] Add unit tests: major patterns produced, minor patterns produced, missing degree skips pattern, review.yml diatonic fields, markov source field
