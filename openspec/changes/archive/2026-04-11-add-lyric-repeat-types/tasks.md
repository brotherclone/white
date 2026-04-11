## 1. Production Plan Schema

- [x] 1.1 Add `lyric_repeat_type: str` field to `PlanSection` dataclass (default `"fresh"`)
- [x] 1.2 Add `_infer_repeat_type(label: str) -> str` helper:
         `chorus/refrain/hook` → `exact`, `verse/pre_chorus` → `variation`, else `fresh`
- [x] 1.3 Populate `lyric_repeat_type` in `generate_plan()` and `sync_plan_from_arrangement()`
         via `_infer_repeat_type`
- [x] 1.4 Round-trip `lyric_repeat_type` through YAML save/load
- [x] 1.5 Tests: infer helper covers all three outcomes; plan round-trip preserves the field;
         human override survives `--refresh`

## 2. Vocal Section Reading

- [x] 2.1 In `read_vocal_sections_from_arrangement`: add `lyric_repeat_type` to each
         section entry — load from `production_plan.yml` if present, else call
         `_infer_repeat_type(label)` as fallback
- [x] 2.2 For `exact` instances beyond the first (i.e., `label_seen_count[label] > 1`):
         set `lyric_repeat_type = LyricRepeatType.EXACT_REPEAT` so the prompt builder
         can skip them
- [x] 2.3 Tests: arrangement with chorus × 3 produces one `exact` entry + two
         `exact_repeat` entries; verse × 2 produces two `variation` entries

## 3. Prompt Builder

- [x] 3.1 In `_build_prompt` and `_build_white_cutup_prompt`:
         - Skip `exact_repeat` instances in the section list (they share the first block)
         - Under each `exact` section: add note
           `"# This section repeats verbatim — write it once, it will be reused"`
         - Under each `variation` instance (n > 1): add note
           `"# Variation {n} of {label}: same meter and rhyme scheme as {label}, but
             new images and lines"` and clarify which instance this is
         - `fresh` instances: no change (current behaviour)
- [x] 3.2 Update OUTPUT FORMAT instructions to describe the three modes
- [x] 3.3 Tests: prompt for a chorus-with-repeats contains the exact-repeat note and
         only one `[chorus]` block instruction; prompt for verse × 2 contains two
         variation block instructions

## 4. Section Parsing and Fitting

- [x] 4.1 In `_compute_fitting`: for `exact_repeat` instances, copy the fitting result
         from the first instance (same MIDI, same lyrics)
- [x] 4.2 Tests: fitting dict has entries for all instance keys;
         `exact_repeat` entries match the base entry

## 5. Output File Format

- [x] 5.1 Confirm output file for `exact` sections has one `[label]` block (no `_2` suffix).
         The arrangement drives repetition — the lyrics file only needs one copy.
- [x] 5.2 For `variation` instances the output retains the `[label_2]` suffix convention.
- [x] 5.3 Tests: promote + ACE Studio import path unaffected (no change to file format
         for non-exact sections)

## 6. Documentation

- [x] 6.1 Update `SONG_GENERATION_PROCESS.md` step 8 (Lyric Generation) with a note
         explaining the three repeat modes and how to override in `production_plan.yml`
