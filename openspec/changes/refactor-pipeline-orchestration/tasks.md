## Phase 1 — Bootstrap song_context.yml (no breaking changes)
- [ ] 1.1 Extend `init_production.py`: `write_initial_proposal()` writes `song_context.yml`
      in addition to `initial_proposal.yml`; include all fields:
      `title, song_slug, song_proposal, thread, color, concept, key, bpm, time_sig,
      singer, sounds_like, genres, mood, schema_version, generated, proposed_by, phases`
- [ ] 1.2 Add `load_song_context(production_dir: Path) -> dict` to `init_production.py`;
      returns `{}` if `song_context.yml` missing (same graceful contract as
      `load_initial_proposal`)
- [ ] 1.3 Update `load_initial_proposal()` to fall back to `song_context.yml` if
      `initial_proposal.yml` is absent (transparent backward compat for migrated dirs)
- [ ] 1.4 Write unit tests: round-trip write/load, missing file returns `{}`, fallback
      from `initial_proposal.yml` → `song_context.yml`

## Phase 2 — Migration script
- [ ] 2.1 Write `app/generators/midi/production/migrate_production_dir.py`:
      reads `chords/review.yml` → song proposal path → concept/genres/mood;
      reads `initial_proposal.yml` if present for `sounds_like`;
      detects phase completion by checking for approved candidates in each review.yml;
      writes `song_context.yml` — non-destructive (does not modify any existing file)
- [ ] 2.2 Add CLI: `--production-dir` (required), `--dry-run` flag that prints what
      would be written without writing
- [ ] 2.3 Write integration test: given a fixture production dir with `chords/review.yml`
      and a song proposal YAML, migration produces a valid `song_context.yml`

## Phase 3 — Propagate concept to drum/bass/melody
- [ ] 3.1 `drum_pipeline._load_song_info()`: call `load_song_context(production_dir)`;
      use `song_context["concept"]` for concept embedding; retain fallback to color string
      for production dirs without `song_context.yml`
- [ ] 3.2 Same change for `bass_pipeline._load_song_info()`
- [ ] 3.3 Same change for `melody_pipeline._load_song_info()`
- [ ] 3.4 Write tests: when `song_context.yml` present, concept text is used;
      when absent, fallback is unchanged

## Phase 4 — Unified load_song_proposal
- [ ] 4.1 Add `load_song_proposal_unified(proposal_path: Path, thread_dir: Path | None = None) -> dict`
      to `production_plan.py`; returns: `title, bpm, time_sig` (string), `key`, `color`,
      `concept`, `genres`, `mood`, `singer` (None if not in proposal), `sounds_like`,
      `key_root`, `mode` (parsed components for chord_pipeline)
- [ ] 4.2 Migrate `chord_pipeline.load_song_proposal()` to call `load_song_proposal_unified()`;
      keep manifest.yml fallback for concept; parse `key` into components inside
      chord_pipeline (not in the unified loader)
- [ ] 4.3 Migrate `lyric_pipeline._find_and_load_proposal()` to call
      `load_song_proposal_unified()` + `load_song_context()` for `sounds_like` and `singer`
- [ ] 4.4 Migrate `composition_proposal.load_song_proposal_data()` same way
- [ ] 4.5 Write tests: all four callers receive correct field sets; `time_sig` is always
      a string; `color` is always present; `color_name` returned for chord_pipeline callers

## Phase 5 — Cleanup (low priority, can be deferred)
- [ ] 5.1 Deprecate `initial_proposal.yml`: update docstring of `write_initial_proposal()`
      noting it will be removed in a future release; continue writing it for now
- [ ] 5.2 Remove `song_proposal` and `thread` fields from `chords/review.yml` output in
      `chord_pipeline.py` (these are now in `song_context.yml`); keep reading them for
      backward compat with existing dirs
- [ ] 5.3 Update `production_plan.py`: drop `sounds_like`, `bpm`, `color`, `time_sig`,
      `key` from `ProductionPlan` YAML output (these are in `song_context.yml`)
