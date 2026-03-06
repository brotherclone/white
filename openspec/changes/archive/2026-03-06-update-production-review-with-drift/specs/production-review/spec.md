## ADDED Requirements

### Requirement: ACE Studio Drift Report

After vocal synthesis is complete, the pipeline SHALL compare the AI-generated melody
loops against the actual performed vocal in the ACE Studio export MIDI and write a
`drift_report.yml` to the production directory.

Drift is computed per arrangement section (using `arrangement.txt` track-4 clip ranges
to segment the ACE export) by comparing each section's approved MIDI loop against the
corresponding ACE word events. The report covers:

- **Pitch match** — fraction of ACE note pitches within ±2 semitones of the approved
  loop's pitch sequence at corresponding positions
- **Rhythm drift** — mean absolute onset difference in beats between matched note pairs
- **Lyric edit distance** — Levenshtein distance between the approved lyrics for that
  section and the words extracted from the ACE export
- **Note count delta** — ACE note count minus approved loop note count (positive = human
  added notes; negative = human removed notes)

Aggregate `overall_pitch_match`, `overall_rhythm_drift`, and `total_lyric_edits` are
computed across all sections and written at the top level of the report.

#### Scenario: Drift report generated

- **WHEN** `python -m app.generators.midi.drift_report --production-dir <dir>` is run
- **AND** a `VocalSynthvN/VocalSynthvN_*.mid` export exists
- **AND** `arrangement.txt` and `melody/approved/` are present
- **THEN** `drift_report.yml` is written to the production directory
- **AND** it contains one entry per vocal section with `pitch_match_pct`, `rhythm_drift_beats`,
  `lyric_edit_distance`, and `note_count_delta`
- **AND** aggregate metrics appear at the top level

#### Scenario: No ACE export — report skipped

- **WHEN** no ACE export MIDI is found in the production directory
- **THEN** the command logs a warning and exits without writing `drift_report.yml`

#### Scenario: Section absent from arrangement

- **WHEN** an approved loop label has no corresponding clip in `arrangement.txt` track 4
- **THEN** that section is omitted from the drift report with a debug log
- **AND** the remaining sections are still reported

---

### Requirement: Song Evaluator Actual Vocal Metrics

When an ACE Studio export MIDI is present, the song evaluator SHALL supplement its
estimated metrics with actuals derived from the real vocal data.

The `--ace-import` flag activates this behaviour. When active, the evaluator calls
`load_ace_export(production_dir)` and, if data is returned, computes:

- `actual_vocal_coverage` — total ACE note duration in beats divided by total
  arrangement duration in beats (replaces the bar-count estimate)
- `actual_syllable_density` — total word count from the ACE export divided by vocal
  bars from `arrangement.txt` (replaces the `lyrics.txt` heuristic)
- `ace_chromatic_alignment` — Refractor text-only score using the reconstructed ACE
  word list as the lyric input (supplements the `lyrics.txt`-based alignment score)

These fields are written under an `ace_actuals` key in `song_evaluation.yml` and do
not replace the existing estimated fields, preserving backwards compatibility.

#### Scenario: ACE actuals computed

- **WHEN** `song_evaluator.py --ace-import` is run
- **AND** `load_ace_export` returns word events
- **THEN** `song_evaluation.yml` contains an `ace_actuals` key with
  `actual_vocal_coverage`, `actual_syllable_density`, and `ace_chromatic_alignment`

#### Scenario: ACE import flag absent — behaviour unchanged

- **WHEN** `song_evaluator.py` is run without `--ace-import`
- **THEN** evaluation proceeds exactly as before
- **AND** no `ace_actuals` key is written

#### Scenario: ACE export missing with flag set

- **WHEN** `--ace-import` is passed but no export MIDI exists
- **THEN** a warning is logged
- **AND** evaluation completes with estimated metrics only (no `ace_actuals` key)
