# Tasks: add-assembly-manifest

## 1. Parser
- [ ] 1.1 Implement `parse_timecode(s) -> float` — strip `01:` offset, return seconds
- [ ] 1.2 Implement `parse_arrangement(text) -> list[Clip]` — Clip(start, name, track, length)
- [ ] 1.3 Group clips into time slots (same start = same slot)
- [ ] 1.4 Detect section prefix from loop name (`intro_*`, `verse_*`, `bridge_*`, `outro_*`)

## 2. Section Derivation
- [ ] 2.1 Implement `derive_sections(clips) -> list[ArrangementSection]` — boundary on prefix change
- [ ] 2.2 Infer `vocals` flag from `_gw` suffix or `vocal` in track-4 clip name
- [ ] 2.3 Populate `loops` dict per section (track number → instrument family → clip name)
- [ ] 2.4 Recompute `bars` from wall-clock length (length_seconds / seconds_per_bar)

## 3. Plan + Manifest Update
- [ ] 3.1 Load existing `production_plan.yml` and match derived sections to plan sections
- [ ] 3.2 Update section `start_time`, `end_time`, `bars`, `loops` — preserve human `vocals: true`
- [ ] 3.3 Update `manifest_bootstrap.yml` section timestamps to match corrected plan
- [ ] 3.4 Update `manifest_bootstrap.yml` `vocals` and `TRT` fields

## 4. Drift Report
- [ ] 4.1 Implement `compute_drift(plan_sections, actual_sections) -> list[DriftEntry]`
- [ ] 4.2 Write `drift_report.yml` with per-section drift_seconds and vocals_flag_changed
- [ ] 4.3 Print summary to stdout — count of sections, total drift, any warnings

## 5. CLI
- [ ] 5.1 Add `__main__` block with `--production-dir` and `--arrangement` args
- [ ] 5.2 Add `--track-map` optional arg (default `1=chords,2=drums,3=bass,4=melody`)
- [ ] 5.3 Add `--vocalist-suffix` optional arg (default `_gw`)

## 6. Tests
- [ ] 6.1 Unit tests for `parse_timecode` (offset stripping, MM:SS:FF variants)
- [ ] 6.2 Unit tests for `parse_arrangement` (multi-track slot grouping)
- [ ] 6.3 Unit tests for `derive_sections` (boundary detection, vocals inference)
- [ ] 6.4 Unit tests for drift computation (zero drift, positive, negative)
- [ ] 6.5 Integration test using The Archivist's Rebellion arrangement data
