# Tasks: Add Production Plan

## 1. Core Data Model and Generator

- [x] **Create `app/generators/midi/production_plan.py`** — data model (`ProductionPlanSection`, `ProductionPlan` as Pydantic models) + YAML serialization/deserialization
- [x] **Implement bar count derivation** — read `hr_distribution` from chord `review.yml` if present, else approved chord MIDI length, else chord count from candidate; report source used
- [x] **Implement plan generation** — read chord `review.yml`, extract approved section labels in order, derive bar counts, write `production_plan.yml` with defaults
- [x] **Implement `--refresh` mode** — reload bar counts from current approved loops, preserve human-edited fields, warn on orphaned sections
- [x] **Implement `--bootstrap-manifest` mode** — read completed plan, compute section durations (bars × bar_duration_seconds), emit `manifest_bootstrap.yml` with derivable manifest fields pre-filled and render-time fields as `null`
- [x] **Add CLI entry point** — `python -m app.generators.midi.production_plan --production-dir <path>` with `--refresh` and `--bootstrap-manifest` flags; print summary of sections and bar counts on completion

## 2. Drum Pipeline Integration

- [x] **Read production plan in drum pipeline** — in `drum_pipeline.py`, after loading song info, check for `production_plan.yml` in the production directory and load section order if present
- [x] **Add `next_section` annotation** — when writing `drums/review.yml`, add `next_section` field to each candidate based on the section's position in the production plan (null for last section)

## 3. Tests

- [x] **Unit tests for bar count derivation** — test all three sources: hr_distribution field, chord MIDI length, chord count fallback; test priority order
- [x] **Unit tests for plan generation** — verify schema, section order, default values, correct section count from mock chord review
- [x] **Unit tests for `--refresh`** — verify human-edited fields are preserved, orphaned sections produce warnings
- [x] **Unit tests for `--bootstrap-manifest`** — verify manifest fields are present, durations correct, null fields present with correct keys
- [x] **Unit test for drum `next_section`** — verify annotation is present when plan exists, absent when plan missing, null for last section

## 4. Documentation

- [x] **Update `app/generators/midi/README.md`** — add production plan to pipeline order diagram and CLI docs
- [x] **Update `training/openspec/TRAINING_ROADMAP.md`** — add production plan as the structural backbone step between melody and assembly

## Dependencies

- Bar count derivation must be complete before plan generation
- Plan generation must be complete before drum integration
- Tests can be written alongside each implementation step
- Manifest bootstrap is independent of drum integration
