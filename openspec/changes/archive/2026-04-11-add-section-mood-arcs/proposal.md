# Change: Section Mood Arcs

## Why

Right now each section gets its patterns independently. Chord candidates are
generated section by section, drum candidates are generated section by section,
and so on. The sections share a key and a chromatic concept, but they don't
know anything about each other during generation. The result is an album of
loops that share a mode rather than a song that goes somewhere.

Listening back to the v0 output, this is the most audible gap. Individual
sections are good. The transitions between them — the way energy should build
through a verse into a chorus, or drop away in a bridge before a final return —
are not composed. They happen by accident or not at all. The production plan
says "verse × 2, chorus, verse, bridge, chorus × 2, outro" but none of the
generation phases know that the second chorus should feel bigger than the first,
or that the bridge should drop to near-silence before the final chorus lifts.

The fix is to generate the arc first and constrain pattern selection to it.
A tension map — a numeric value per section describing its intended emotional
weight — turns the production plan from a sequence of loops into a compositional
directive. Each pattern selection can then ask: "does this candidate's energy
level match what this section's position in the arc requires?"

This is the difference between a pipeline that assembles music and one that
composes it.

## What Changes

### Tension map — `arc` field in `production_plan.yml`

Each section in the production plan gains an `arc` value: a float 0.0–1.0
representing intended emotional intensity. 0.0 is near-silence (intro exhale,
bridge held breath), 1.0 is the peak (final chorus, climax).

```yaml
sections:
  - label: intro
    bars: 4
    arc: 0.1
  - label: verse_1
    bars: 8
    arc: 0.3
  - label: pre_chorus
    bars: 4
    arc: 0.55
  - label: chorus_1
    bars: 8
    arc: 0.75
  - label: verse_2
    bars: 8
    arc: 0.4    # verse 2 slightly higher than verse 1 — building
  - label: bridge
    bars: 4
    arc: 0.15   # collapse before the final push
  - label: chorus_2
    bars: 8
    arc: 0.9    # bigger than chorus_1
  - label: outro
    bars: 8
    arc: 0.2
```

**Auto-generation via Claude:**

`composition_proposal.py` (already exists) is extended to generate the arc as
part of its compositional proposal. Claude receives the section sequence, the
concept, the color, and the approved loop inventory, and proposes arc values
with a rationale:

```
This is an elegy. The tension arc should reflect the shape of grief — not a
triumphant build, but a slow accumulation that never fully resolves. I've placed
the peak at chorus_2 (0.85) rather than 1.0, and the outro descends without
returning to the intro's quiet — it settles lower but with more weight.
```

The human can edit arc values in `production_plan.yml` before generation runs.

**Manual seeding:**

If `composition_proposal` is not run, `production_plan.py` seeds arc values
from section label heuristics (same logic used for `section_energy`):
- `intro`, `outro` → 0.15
- `verse` → 0.35
- `pre_chorus` → 0.55
- `chorus` → 0.75
- `bridge` → 0.2
- `climax`, `peak` → 0.9

These are reasonable defaults but always editable.

### Arc-aware candidate scoring

Each pipeline phase (drum, bass, melody) reads the `arc` value for the current
section and uses it as a scaling factor on the energy-related scoring dimension:

**Drum pipeline:**

The existing `energy_appropriateness` score becomes arc-weighted:
- Target energy is derived from `arc` instead of (or blended with) the section
  label heuristic
- A dense pattern scoring well on energy when `arc=0.8` would score poorly
  when `arc=0.15` — the arc makes the target explicit

**Bass pipeline:**

`arc` influences the target note density and movement rate:
- Low arc (≤0.25): strongly prefer pedal/drone templates
- Mid arc (0.25–0.65): balanced selection
- High arc (≥0.65): prefer more active movement; penalise root_drone

**Melody pipeline:**

`arc` maps to a target phrase density and interval character:
- Low arc: sparse phrases, long rests, stepwise descent preferred
- High arc: shorter rests, occasional leaps, more frequent phrase repetition

The arc influence is expressed as a score adjustment (not a hard filter), so
the Refractor's chromatic score still dominates. A very chromatically appropriate
pattern can still win even if it's slightly mis-matched to the arc — but the
mis-match is visible in the review.yml score breakdown.

### Arc visualisation in `pipeline status`

The `pipeline_runner` `status` command shows the arc as a small ASCII tension
graph alongside section statuses:

```
Song: Sequential Dissolution (black)

Section arc:
  intro         ▁  0.10  ✅ promoted
  verse_1       ███  0.35  ✅ promoted
  pre_chorus    █████  0.55  ✅ promoted
  chorus_1      ████████  0.75  🔄 generated
  verse_2       ████  0.40  ⏳ pending
  bridge        ▁  0.15  ⏳ pending
  chorus_2      █████████  0.90  ⏳ pending
  outro         ██  0.20  ⏳ pending
```

This makes the compositional shape visible at a glance during the review loop.

### Arc consistency check at promotion

`promote_part.py` gains an arc consistency check: if the promoted candidate's
measured energy deviates from the section's arc value by more than a threshold
(default 0.3), it prints a warning:

```
⚠ chorus_1: promoted pattern energy=0.42 vs arc=0.75 — significant mismatch.
  The chorus may feel flat relative to the intended arc.
  Promote anyway? [y/N]
```

This is a prompt, not a hard failure — the human decides. But it makes the
tension between what was intended and what was chosen visible at the moment of
commitment.

### `arc_delta` in `song_evaluation.yml`

The song evaluator gains an `arc_delta` metric: mean absolute deviation between
each section's arc value and the measured energy of its promoted patterns. Low
arc_delta means the production followed the compositional arc. High arc_delta
means it drifted — potentially intentionally (human override) or as a gap to
address in the next run.

`arc_delta` contributes to the `structural_integrity` composite score
(replacing or supplementing the existing drift formula for arrangement-first
songs that do have an arc).

## Impact

- Affected specs: `production-plan`, `chord-pipeline`, `drum-pipeline`,
  `bass-pipeline`, `melody-pipeline`, `song-evaluator`, `composition-proposal`
- Modified files:
  - `app/generators/midi/production/production_plan.py` — `arc` field on
    `PlanSection`; auto-seed from heuristics; `generate_plan()` writes arc values
  - `app/generators/midi/production/composition_proposal.py` — arc generation
    as part of compositional proposal
  - `app/generators/midi/pipelines/drum_pipeline.py` — arc-weighted energy scoring
  - `app/generators/midi/pipelines/bass_pipeline.py` — arc-weighted template preference
  - `app/generators/midi/pipelines/melody_pipeline.py` — arc-weighted phrase density
  - `app/generators/midi/production/promote_part.py` — arc consistency check
  - `app/generators/midi/production/pipeline_runner.py` — arc visualisation in status
  - `app/generators/midi/production/song_evaluator.py` — `arc_delta` metric
- Tests:
  - `tests/generators/midi/production/test_production_plan.py` — arc seeding,
    arc field round-trip
  - `tests/generators/midi/production/test_song_evaluator.py` — arc_delta calculation
  - `tests/generators/midi/production/test_promote_part.py` — arc consistency warning
  - Pipeline tests: arc value shifts scored candidate rankings in expected direction
