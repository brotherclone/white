# Change: Add Production Plan

## Why

The production pipeline generates loops (chords, drums, harmonic rhythm, strums, bass, melody) but has no concept of song structure — which sections exist, in what order, how many times each repeats, and how long each is. Without this, downstream phases can't be structure-aware. Specifically:

- Drum fills require knowing *when* a section transition is coming so they can signal the change
- Assembly requires a document that defines the full song arrangement
- The production pipeline has no bridge toward a final song manifest

A `production_plan.yml` living alongside the loop folders becomes the shared structural context for all subsequent phases.

## What Changes

- New `app/generators/midi/production_plan.py` — generates a `production_plan.yml` from approved chord sections, with sensible defaults for bar counts and section order
- New schema: `production_plan.yml` defines sections, bar counts, repeat counts, vocals flag, and (optionally) sounds-like references
- The drum pipeline (and future fill phase) reads the production plan to understand section transitions
- The production plan accumulates enough information to bootstrap a final song manifest at the end of production

## Impact

- Affected specs: `production-plan` (NEW), `drum-generation` (MODIFIED — reads plan for context)
- Affected code: `app/generators/midi/production_plan.py` (new), `app/generators/midi/drum_pipeline.py` (minor — plan-aware generation), `app/structures/manifests/` (no changes yet — plan feeds manifest eventually)
- The production plan is a human-edited document; the generator provides a starting point only

## Key Design Decisions

**What the plan contains (per section):**
- `name` — section label (verse, chorus, bridge, intro, outro, or custom)
- `bars` — number of bars (derived from the approved chord loop length × repeat of harmonic rhythm)
- `repeat` — how many times this section plays (human-set, default 1)
- `vocals` — whether this section has a vocal line (human-set, default false)
- `notes` — freeform human annotation

**What the plan does NOT contain:**
- Loop file references — these are derived dynamically from `approved/` directories at assembly time, keyed by section label
- Timestamps — those come at render time
- Audio file paths — those come at render time

**Section order:**
The generator outputs sections in the order they appear in the approved chord review (the human's labeling order). The human reorders by editing the YAML.

**Bar count derivation:**
Derived from approved chord MIDI length for that section. If harmonic rhythm is approved, that length is used. If not, defaults to N chords × 1 bar.

**Drum pipeline change:**
The drum pipeline gains an optional `--production-plan` flag. When provided, each section's approved drums carry a `next_section` annotation derived from the plan, enabling the future fill phase to know what comes next. For now this is read-only context — no functional change to drum output.
