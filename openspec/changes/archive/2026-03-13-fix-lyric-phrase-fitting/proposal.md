# Change: Fix Lyric Phrase Fitting and Chromatic Scoring

## Why
Two bugs in `lyric_pipeline.py` identified from `violet__cultural_detector_living_v2`:

1. Syllable fitting was computed at section granularity (total syllables / total notes),
   which passed sections written as continuous prose. ACE Studio requires one lyric line
   per MIDI phrase group (notes separated by rests). A section where 3/4 phrases fit but
   1/4 is "splits needed" was incorrectly reported as paste-ready.

2. Refractor text-mode confidence for lyric text was extremely low (0.034–0.039),
   producing near-identical chromatic scores across all candidates regardless of content.
   The keyword hybrid fallback now blends keyword-based scoring when confidence < 0.2.

Also fixes a stale scenario in the lyric-generation spec where vocal sections were
described as coming from `production_plan.yml` — the pipeline has read from
`arrangement.txt` track 4 since `update-lyric-pipeline-remove-plan`.

## What Changes
- `extract_phrases(midi_path)` — groups note-on events into rest-separated phrases
- `_compute_fitting()` — now per-phrase; overall verdict driven by worst phrase
- Generation prompt — includes phrase counts and per-phrase syllable targets when MIDI
  available; instructs Claude to write one line per phrase
- `_keyword_score()` + `_blend_scores()` — keyword hybrid fallback for confidence < 0.2

## Impact
- Affected specs: `lyric-generation` (MODIFIED fitting requirement, ADDED keyword hybrid,
  MODIFIED stale vocal-sections scenario)
- Affected code: `app/generators/midi/pipelines/lyric_pipeline.py` (already implemented)
