# Change: Spike — Pipeline Orchestration Design Review

## Why
The music production pipeline grew iteratively (chords → drums → bass → melody → lyrics →
composition proposal) and works, but carries the design debt of that growth. Several
structural problems have surfaced:

1. **No production initialization step** — pipeline phases implicitly create their own
   directories; there is no canonical "start a song" entry point
2. **Two `load_song_proposal()` functions** — `chord_pipeline.py:161` and
   `production_plan.py:228` are separate implementations reading the same YAML format
   with overlapping but inconsistent fields
3. **No shared song context object** — each pipeline phase re-reads the song proposal,
   `chords/review.yml`, and (sometimes) `composition_proposal.yml` independently; metadata
   like `sounds_like`, `singer`, `concept` is re-derived per phase and sometimes lost
4. **`sounds_like` arrives too late** — populated in `composition_proposal.yml` which runs
   after all MIDI generation; no earlier source exists (addressed partially by
   `add-initial-sounds-like-proposal` but that is a patch, not a redesign)
5. **Composition proposal is both a creative deliverable and a data source** — it serves
   the human (rationale, section notes) and the pipeline (loop inventory, sounds_like),
   but its dual role is implicit and the machine-readable fields are not typed or validated
6. **Phase review files are per-phase** — `chords/review.yml`, `melody/review.yml`,
   `melody/lyrics_review.yml` each have different schemas; no unified pipeline status

This spike does not prescribe implementation. It produces a design document that:
- Maps the current data flow (what reads what, in what order)
- Identifies which problems are worth solving vs. acceptable as-is
- Proposes a target architecture for a "production context" that survives the full pipeline
- Estimates the scope of a follow-on refactor

## What Changes
- Spike output under `training/spikes/pipeline-orchestration/`
- No production code changes
- Findings inform follow-on `refactor-pipeline-orchestration` proposal (or a decision
  that the current architecture is good enough)

## Impact
- Affected specs: `pipeline-orchestration` (new, spike-only)
- Affected code: none (read-only analysis)
