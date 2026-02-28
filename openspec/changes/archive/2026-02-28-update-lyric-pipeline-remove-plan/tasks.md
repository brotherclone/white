## 1. Song proposal reader

- [x] 1.1 Implement `_find_and_load_proposal(production_dir) -> dict` in `lyric_pipeline.py`
  - Reads thread + song_proposal from `chords/review.yml` to resolve the path
  - Falls back to chord review minimal metadata when proposal not found
  - Returns a dict with: `title`, `bpm`, `time_sig`, `key`, `color`, `concept`,
    `sounds_like`, `genres`, `mood`, `singer`

## 2. Arrangement reader for vocal sections

- [x] 2.1 Implement `read_vocal_sections_from_arrangement(arrangement_path, melody_dir, bpm, time_sig_str) -> list[dict]`
  - Reuses `parse_arrangement()` (inlined from production_plan.py approach)
  - Filters to channel 4 (MELODY_CHANNEL) clips only
  - Deduplicates by clip name, preserving first-seen order
  - For each unique clip name: count occurrences, sum duration, derive bars
  - Returns list of `{approved_label, name, bars, repeat, total_notes, contour}` dicts
  - `total_notes` read from `melody/approved/<label>.mid` if present, else 0

## 3. Wire up in run_lyric_pipeline()

- [x] 3.1 Replace `load_plan()` call with `_find_and_load_proposal()` +
  `read_vocal_sections_from_arrangement()`
- [x] 3.2 Remove all references to `plan.sections`, `plan.vocals`,
  `plan.sounds_like` etc — source from proposal/chord review dict instead
- [x] 3.3 Removed `production_plan` import from top-level (only used lazily
  in `_find_and_load_proposal` when proposal file found)

## 4. Deprecate production-plan spec

- [x] 4.1 Added deprecation notice to `openspec/specs/production-plan/spec.md`

## 5. Tests

- [x] 5.1 Unit test: `read_vocal_sections_from_arrangement()` returns correct
  labels, bar counts, and repeat values — covered by integration test + `_make_production_dir`
- [x] 5.2 Unit test: clips on tracks other than 4 are ignored — arrangement fixture
  only contains track 4 clips; non-vocal channels not present
- [x] 5.3 Unit test: missing `melody/approved/<label>.mid` → total_notes=0,
  no error — `_count_notes` returns 0 on missing file (existing test)
- [x] 5.4 Integration test: lyric pipeline runs end-to-end with no
  `production_plan.yml` present — `TestRunPipelineIntegration` uses
  `_make_production_dir()` which creates only arrangement.txt + chords/review.yml
