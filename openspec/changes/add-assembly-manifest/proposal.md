# Change: Add Assembly Manifest Import

## Why

The production pipeline generates `production_plan.yml` and `manifest_bootstrap.yml` with
section timestamps computed from bar counts × BPM. When a human assembles loops in Logic Pro,
the actual placement diverges from the computed plan — sections start at different times,
vocal flags drift, and loop assignments are unknown to the pipeline. There is currently no
way to reconcile the plan against the real arrangement.

## What Changes

- New module `app/generators/midi/assembly_manifest.py` that parses a Logic Pro arrangement
  export (timecode format: `HH:MM:SS:FF.sub  loop_name  track  HH:MM:SS:FF.sub`) into a
  structured section map
- Derives actual section boundaries from loop name prefixes (`intro_*`, `verse_*`, `bridge_*`)
  and their wall-clock positions
- Updates `production_plan.yml` sections with corrected timestamps and a `loops` dict per
  section (chords / drums / bass / melody track assignments)
- Updates `manifest_bootstrap.yml` with real section start/end times and correct `vocals` flags
- Emits a `drift_report.yml` comparing computed vs actual timestamps, flagging any section
  boundary shifts or vocal flag mismatches
- CLI: `python -m app.generators.midi.assembly_manifest --production-dir <dir> --arrangement <file>`

## Impact

- Affected specs: `assembly-manifest` (new), `production-review` (vocals flag source of truth)
- Affected code: `production_plan.py` (section schema gains `loops` field), `manifest_bootstrap.yml` schema
- No breaking changes to upstream phases (chord, drum, bass, melody pipelines unaffected)
