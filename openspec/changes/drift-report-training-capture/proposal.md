# Change: Production decisions capture for training

## Why

Every completed White song is a labelled training example: the agent proposed
something, the pipeline generated candidates, a human promoted some and rejected
others, arranged them in Logic, and the final mix scored against the chromatic target.
That entire decision chain is future training data — but right now it lives scattered
across `review.yml` files, `arrangement.txt`, and `mix_score.yml` with no single
structured summary.

The goal (automated pipeline) requires a dataset of: concept + color → what structure
emerged → how far it drifted → final quality signal. `production_decisions.yml` is
that record, written once at session close.

## What Changes

A new `generate_decisions_file()` function (in `drift_report.py` or a new
`production_decisions.py`) reads the completed production directory and writes
`production_decisions.yml` containing:

- **Identity**: thread, color, title, key, BPM, time_sig, singer
- **Phase decisions** (for each completed phase — chords, drums, bass, melody):
  - `candidates_generated` — count from `review.yml`
  - `approved_count` — count of approved candidates
  - `approved_labels` — list of approved section labels
  - `mean_chromatic_score`, `mean_theory_score` — averages of approved candidates
- **Arrangement summary** (from `arrangement.txt`):
  - sections (name, bars, play_count, vocals)
  - total bars, total plays, section count
- **Mix scores** (from `melody/mix_score.yml` if present):
  - temporal, spatial, ontological, confidence, chromatic_match
- **Vocal drift** (from `drift_report.yml` if present):
  - overall_pitch_match, overall_rhythm_drift, total_lyric_edits

`production_decisions.yml` is written to the production directory root. The pipeline
runner's `status` command is updated to show whether it exists.

## Why this isn't just the drift report

`drift_report.yml` covers ACE Studio vocal drift against approved melody loops —
one narrow signal. `production_decisions.yml` covers the full production arc and
is intentionally structured for ML ingestion. Keeping them separate preserves the
single-purpose nature of `drift_report.yml`.

## Impact

- Affected code: `app/generators/midi/production/drift_report.py` (or new
  `production_decisions.py`), `app/generators/midi/production/pipeline_runner.py`
- Affected spec: `pipeline-orchestration`
- Reads only — does not modify any existing files
- New CLI flag: `python -m app.generators.midi.production.drift_report --production-dir <path> --decisions`
  (or standalone `production_decisions.py --production-dir <path>`)
