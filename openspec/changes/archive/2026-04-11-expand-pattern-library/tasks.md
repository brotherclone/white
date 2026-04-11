## 1. Tags on Pattern Dataclasses
- [x] 1.1 Add `tags: list[str] = field(default_factory=list)` to `DrumPattern` dataclass
- [x] 1.2 Add `tags: list[str] = field(default_factory=list)` to `BassPattern` dataclass
- [x] 1.3 Add `tags: list[str] = field(default_factory=list)` to `MelodyPattern` dataclass

## 2. New Drum Templates (sparse/atmospheric)
- [x] 2.1 Add `half_time_sparse` — kick beat 1, snare beat 3, open hat off-beat (tags: sparse)
- [x] 2.2 Add `ghost_verse` — ghost snare only, no kick, whisper hats (tags: sparse, ghost_only)
- [x] 2.3 Add `brushed_folk` — brush swells on 2 and 4, light kick (tags: sparse, brushed)
- [x] 2.4 Add `ambient_pulse_slow` — single kick every 2 bars, crash swell on bar 4 (tags: sparse, ambient)
- [x] 2.5 Add `kosmische_slow` — motorik feel at half tempo (tags: sparse, motorik)

## 3. New Bass Templates (drone/pedal)
- [x] 3.1 Add `root_drone` — whole-note root, no movement (tags: drone, minimal)
- [x] 3.2 Add `slow_pedal` — root beat 1, octave-down beat 3 (tags: pedal, minimal)
- [x] 3.3 Add `descending_sigh` — root→5th over bar (tags: minimal, drone)
- [x] 3.4 Add `sustained_fifth` — held 5th, velocity swell (tags: drone, pedal)
- [x] 3.5 Add `minimal_walk` — root + one passing tone (tags: minimal, walking)

## 4. New Melody Templates (lamentful/sparse)
- [x] 4.1 Add `slow_descent` — two long notes per bar with space (tags: lamentful, sparse, descent)
- [x] 4.2 Add `breath_phrase` — burst then long rest then resolve (tags: lamentful, sparse)
- [x] 4.3 Add `pentatonic_lament` — pentatonic, first note held (tags: lamentful, sparse)
- [x] 4.4 Add `floating_repeat` — same motif at shifting positions (tags: sparse)
- [x] 4.5 Add `single_line` — two long notes with breath gap (tags: lamentful, sparse)

## 5. Aesthetic Hints Detection
- [x] 5.1 In `init_production.py`: detect ambient/shoegaze cluster in `sounds_like`, write `aesthetic_hints` to `song_context.yml`
- [x] 5.2 Defined cluster: Grouper, Beach House, MBV, Low, Barwick, Boards of Canada, Stars of the Lid, etc. Threshold: 2+ matches

## 6. Tag-Weighted Pattern Selection
- [x] 6.1 In `drum_pipeline.py`: read `aesthetic_hints` from song context; apply +0.1 bonus / −0.05 penalty via `aesthetic_tag_adjustment`
- [x] 6.2 In `bass_pipeline.py`: same tag weighting logic
- [x] 6.3 In `melody_pipeline.py`: same tag weighting logic
- [x] 6.4 Shared helper: `app/generators/midi/patterns/aesthetic_hints.py`

## 7. Tests
- [x] 7.1 Test `tags` field exists on all three dataclasses (empty list default)
- [x] 7.2 Test filtering patterns by tag returns correct results
- [x] 7.3 Test aesthetic_hints detection in init_production
- [x] 7.4 Test tag weighting shifts composite scores in expected direction
