## 1. Arrangement Segmentation Helper
- [ ] 1.1 Implement `segment_ace_export_by_arrangement(word_events, arrangement_path, bpm) -> dict[str, list]`
        — maps word events to arrangement sections using the timecode ranges in `arrangement.txt`
        (track 4 clips only); returns `{section_label: [word_event, ...]}`

## 2. Drift Report
- [ ] 2.1 Implement `compare_section(approved_midi_path, ace_word_events) -> dict` — for one section:
        - `pitch_match_pct`: fraction of ACE note pitches within 2 semitones of the approved loop pitch sequence
        - `rhythm_drift_beats`: mean absolute onset difference (beat units) between matched note pairs
        - `lyric_edit_distance`: Levenshtein distance between approved lyrics and ACE words for that section
        - `note_count_delta`: ACE note count minus approved loop note count
- [ ] 2.2 Implement `generate_drift_report(production_dir) -> dict` — calls `load_ace_export`,
        segments by arrangement, compares each section, aggregates `overall_pitch_match`,
        `overall_rhythm_drift`, `total_lyric_edits`
- [ ] 2.3 Write `drift_report.yml` to the production directory with per-section and overall metrics
- [ ] 2.4 CLI: `python -m app.generators.midi.drift_report --production-dir <dir>`

## 3. Song Evaluator Extension
- [ ] 3.1 Add `--ace-import` flag to `song_evaluator.py`
- [ ] 3.2 When flag is set and `load_ace_export` returns data:
        - Compute `actual_vocal_coverage` as (total ACE note duration in beats) / (total arrangement beats)
        - Compute `actual_syllable_density` as (total word count from ACE export) / (vocal bars from arrangement)
        - Re-score chromatic alignment using actual ACE word list via Refractor text-only mode
- [ ] 3.3 Append `actual_vocal_coverage`, `actual_syllable_density`, `ace_chromatic_alignment` to
        `song_evaluation.yml` under a new `ace_actuals` key

## 4. Tests
- [ ] 4.1 Unit: `segment_ace_export_by_arrangement` — words before first section omitted, words in gap
        between sections omitted, words straddling a boundary assigned to the section they start in
- [ ] 4.2 Unit: `compare_section` — identical section returns 100% pitch match, 0 rhythm drift,
        edit_distance=0; transposed section returns reduced pitch match
- [ ] 4.3 Unit: `generate_drift_report` with mocked `load_ace_export` and arrangement
- [ ] 4.4 Integration: run drift report against `blue__rust_signal_memorial_v1` and assert
        `drift_report.yml` is written with expected top-level keys
