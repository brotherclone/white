# Tasks: expand-drum-templates

---

## Task 1 — Extend GM_PERCUSSION and GENRE_FAMILY_KEYWORDS

In `app/generators/midi/patterns/drum_patterns.py`:
- Add to `GM_PERCUSSION`: `"tambourine": 54`, `"conga_high": 63`, `"conga_low": 64`, `"cowbell": 56`
- Add to `GENRE_FAMILY_KEYWORDS`: `"breakbeat": ["breakbeat", "hip-hop", "boom bap", "break", "funk", "soul", "r&b", "groove"]`

**Validation:** run existing tests — they should all still pass after this step alone.

---

## Task 2 — Transcribe the 9 medium-energy classic break templates

Open the reference spreadsheet and transcribe each pattern into `DrumPattern`
objects. Map the 16-step grid to 0.25-increment float positions:

```
1=0.0  1e=0.25  1+=0.5  1a=0.75
2=1.0  2e=1.25  2+=1.5  2a=1.75
3=2.0  3e=2.25  3+=2.5  3a=2.75
4=3.0  4e=3.25  4+=3.5  4a=3.75
```

Patterns to transcribe (all `energy="medium"`, `genre_family="breakbeat"`, `time_sig=(4,4)`):
1. `billie_jean` — kick, snare, hh_closed
2. `funky_drummer` — kick, snare, hh_closed, hh_open (encode ghost notes faithfully)
3. `impeach_the_president` — kick, snare, hh_closed, hh_open
4. `when_the_levee_breaks` — kick, snare, hh_closed (the half-time thunder feel)
5. `walk_this_way` — kick, snare, hh_closed, hh_open
6. `its_a_new_day` — kick, snare, hh_closed
7. `papa_was_too` — kick, snare, hh_closed, tambourine
8. `the_big_beat` — kick, snare, clap
9. `ashleys_roachclip` — kick, snare, hh_closed, hh_open, conga_high, conga_low

**Validation:** `python -c "from app.generators.midi.patterns.drum_patterns import TEMPLATES_4_4_BREAKBEAT; print(len(TEMPLATES_4_4_BREAKBEAT))"` returns 9 or more.

---

## Task 3 — Add low and high energy variants

Add at minimum 3 low-energy and 3 high-energy templates to `TEMPLATES_4_4_BREAKBEAT`.
These can be stripped/full versions of the classic breaks or original patterns
inspired by the breakbeat idiom:

Low energy suggestions:
- `breakbeat_sparse` — kick on 1, snare on 3, ghost hat pulse (simple boom-bap skeleton)
- `funky_drummer_stripped` — kicks and snares only from Funky Drummer, no hats
- `when_the_levee_breaks_ghost` — Levee feel at half-volume, ghost everything

High energy suggestions:
- `billie_jean_full` — Billie Jean pattern with additional hat accents and crash on 1
- `funky_drummer_heavy` — Funky Drummer with added cowbell and accented offbeats
- `breakbeat_driving` — four-on-floor kick hybrid with breakbeat snare placement

**Validation:** `select_templates(ALL_TEMPLATES, (4,4), ["breakbeat"], "low")` returns at least 1 result;
same for `"high"`.

---

## Task 4 — Register in ALL_TEMPLATES

Add `*TEMPLATES_4_4_BREAKBEAT` to the `ALL_TEMPLATES` list in the registry section.

**Validation:** `len([t for t in ALL_TEMPLATES if t.genre_family == "breakbeat"]) >= 15`

---

## Task 5 — Tests

Add assertions to `tests/generators/midi/test_drum_pipeline.py`:

1. `test_breakbeat_family_in_genre_keywords` — assert `"breakbeat"` in `GENRE_FAMILY_KEYWORDS`
2. `test_hip_hop_maps_to_breakbeat` — `map_genres_to_families(["hip-hop"])` returns `["breakbeat"]`
3. `test_funk_maps_to_breakbeat` — `map_genres_to_families(["funk"])` contains `"breakbeat"`
4. `test_new_gm_voices_present` — assert `tambourine`, `conga_high`, `conga_low`, `cowbell` in `GM_PERCUSSION`
5. `test_breakbeat_has_all_energy_levels` — at least one low, medium, high per family enforcement
6. `test_nine_classic_breaks_present` — assert all 9 named templates exist
7. `test_funky_drummer_has_ghost_notes` — at least one ghost velocity in funky_drummer
8. `test_papa_was_too_has_tambourine` — `"tambourine"` in papa_was_too.voices
9. `test_ashleys_roachclip_has_congas` — conga_high or conga_low in ashleys_roachclip.voices
10. `test_all_breakbeat_positions_are_16th_grid` — all positions divisible by 0.25

All existing tests must still pass.

---

## Dependencies

- Task 1 is a prerequisite for all others (GM voices must exist before templates reference them)
- Tasks 2 and 3 can proceed in parallel once Task 1 is done
- Task 4 depends on Tasks 2 and 3
- Task 5 can be written alongside Tasks 2–4

## Reference

Spreadsheet: https://docs.google.com/spreadsheets/d/19_3BxUMy3uy1Gb0V8Wc-TcG7q16Amfn6e8QVw4-HuD0/
