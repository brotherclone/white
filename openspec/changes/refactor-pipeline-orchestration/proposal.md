# Change: Refactor — Pipeline Orchestration Context

## Why
The pipeline orchestration spike (`2026-03-15-spike-pipeline-orchestration-design`)
identified four concrete correctness and maintainability problems:

1. **Concept text silently lost** — drum, bass, and melody phases fall back to
   `f"{color_name} chromatic concept"` because `chord_review.yml` does not carry the
   concept field. Refractor scores for these three phases are computed on a degraded
   placeholder rather than the actual song concept.

2. **Three divergent `load_song_proposal` implementations** — `chord_pipeline.py:162`,
   `production_plan.py:228`, `lyric_pipeline._find_and_load_proposal()`, and
   `composition_proposal.load_song_proposal_data()` return inconsistent field sets
   (`color` vs `color_name`, `time_sig` as tuple vs string, `singer` present or absent,
   `sounds_like` patched after load or missing entirely).

3. **No shared song context** — each phase independently navigates from its own working
   directory back to the original song proposal YAML, making the pipeline fragile if any
   pointer (thread path, song_proposal filename in chord_review) is wrong or missing.

4. **`sounds_like` blind in three phases** — drum, bass, and melody have no access to
   `sounds_like` at generation time; `initial_proposal.yml` is only checked by chord and
   lyric phases.

This refactor introduces `song_context.yml` as the canonical static metadata store for
a production directory, written once by `init_production.py` before any phase runs.

## What Changes
- `init_production.py`: writes `song_context.yml` alongside (then superseding)
  `initial_proposal.yml`
- New `load_song_context(production_dir) → dict` function in `init_production.py`
- New migration script `app/generators/midi/production/migrate_production_dir.py`
- `drum_pipeline`, `bass_pipeline`, `melody_pipeline`: read `concept` from
  `song_context.yml` instead of falling back to color string
- `production_plan.py`: add `load_song_proposal_unified()` — single canonical loader
  replacing the four divergent implementations
- All four loaders migrated to call `load_song_proposal_unified()`
- `load_initial_proposal()` updated to fall back transparently to `song_context.yml`

## Impact
- Affected specs: `pipeline-orchestration` (MODIFIED), `sounds-like-bootstrap` (MODIFIED)
- Affected code: `init_production.py`, `production_plan.py`, `drum_pipeline.py`,
  `bass_pipeline.py`, `melody_pipeline.py`, `lyric_pipeline.py`,
  `composition_proposal.py`, `chord_pipeline.py`
- Existing production dirs (violet, blue): require migration script before pipeline
  phases can read from `song_context.yml`; all phases retain fallback behavior for dirs
  that predate this change
