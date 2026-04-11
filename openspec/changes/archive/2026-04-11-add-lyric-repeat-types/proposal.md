# Change: Lyric Repeat Type Annotation

## Why

When the same melody loop appears multiple times in an arrangement, the lyric
pipeline currently treats every instance as fully independent — Claude writes
fresh content for each occurrence. But arrangement repetition carries musical
intent: a chorus should repeat verbatim (identity is the point), verse lines
should vary in content but hold the same meter/rhyme scheme, and bridges or
climax sections may be genuinely unique. Without this signal, Claude has no
basis for making that distinction, so it defaults to fresh content everywhere —
leading to choruses that don't repeat and verses that share no connective tissue.

## What Changes

- **New `lyric_repeat_type` field** per section instance in the lyric prompt and
  in `production_plan.yml`. Three values:
  - `exact` — lyrics are written once and repeated verbatim (chorus, refrain, hook)
  - `variation` — each instance gets its own lines but shares rhyme scheme, meter,
    and core imagery with the other instances of the same loop (verse 2 vs. verse 1)
  - `fresh` — each instance is treated as independent (current behaviour; bridge,
    outro, unique climax)

- **Auto-detection from loop label** as the default, with manual override in
  `production_plan.yml`:
  - Label contains `chorus`, `refrain`, or `hook` → `exact`
  - Label contains `verse`, `pre_chorus`, or `pre-chorus` → `variation`
  - Everything else → `fresh`

- **Prompt instruction changes**: sections sharing a loop label are grouped in the
  prompt. `exact` groups are written once with a note that they repeat. `variation`
  groups are written as numbered instances with instructions to vary content but
  preserve structure. `fresh` instances continue to be written independently.

- **Output format**: `exact` sections share the same lyric block in the output file
  (one `[label]` header, one body); the lyrics file is still one block per unique
  melody loop — the arrangement is the authority for repetition, not the lyrics
  file. `variation` instances each get their own block with suffixed headers
  (`[melody_verse]`, `[melody_verse_2]`, …).

- **`production_plan.yml` schema addition**: optional `lyric_repeat_type` per
  section (`exact | variation | fresh`), populated by auto-detection on plan
  generation and editable by humans before lyric generation runs.

## Impact

- Affected specs: `lyric-generation`, `lyric-pipeline`
- Affected code:
  - `app/generators/midi/pipelines/lyric_pipeline.py` — prompt builder,
    `read_vocal_sections_from_arrangement`, `_parse_sections`, `_compute_fitting`
  - `app/generators/midi/production/production_plan.py` — `PlanSection` dataclass,
    `generate_plan`, `sync_plan_from_arrangement`
  - `app/generators/midi/production/assembly_manifest.py` — no change expected
- No breaking changes to the review YAML schema; `lyric_repeat_type` is additive
- Tests: `tests/generators/midi/test_lyric_pipeline.py` (new + updated)
