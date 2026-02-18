# Design: Assembly Manifest Import

## Context

Logic Pro exports an arrangement region list in two formats:
- **Timecode**: `01:MM:SS:FF.sub  loop_name  track  00:MM:SS:FF.sub` (preferred — wall-clock, no BPM needed)
- **Bar/beat**: `bar beat sub tick  loop_name  track  bars 0 0 0` (requires BPM + time sig conversion)

The timecode format is the canonical input. The `01:` hour prefix is Logic's start offset and
is stripped during parsing. Frames and sub-frames are ignored (we only need MM:SS precision).

## Goals / Non-Goals

- **Goals**: Parse arrangement → derive sections → update plan + manifest → report drift
- **Non-Goals**: Modify approved MIDI files, push changes back to Logic, handle multi-track exports
  from other DAWs

## Decisions

### Section Detection

Loop name prefixes map to section labels:
- `intro_*` → Intro
- `verse_*` → Verse
- `bridge_*` → Bridge
- `outro_*` → Outro
- Unrecognised prefix → `unknown` (preserved, not discarded)

A section boundary is detected when the prefix changes between consecutive time slots.
Simultaneous multi-track clips at the same start time form one time slot.

### Track → Instrument Mapping

Track numbers from Logic export map to instrument families:
- Track 1 → `chords`
- Track 2 → `drums`
- Track 3 → `bass`
- Track 4 → `melody`

The mapping is configurable via a `--track-map` argument (e.g. `1=chords,2=drums`).
Default covers the standard 4-track layout used in White production.

### Vocals Flag Inference

A section has `vocals: true` if track 4 (melody) has any clip whose name ends in `_gw`
(Gabriel Walsh — the vocalist's initials used in loop naming) or contains `vocal`.
Human override in `production_plan.yml` is preserved if the flag is explicitly set to `true`.

### `loops` Field Schema

Added to each section in `production_plan.yml`:
```yaml
loops:
  chords: intro_arp_up
  drums: Drums Primitive
  bass: bass_intro_simple
  melody: melody_intro_04
```
Where multiple clips appear in one section for the same track, the last one wins
(represents the most developed layer). Future: list all.

### Drift Report

`drift_report.yml` is written alongside `production_plan.yml`:
```yaml
generated: <iso timestamp>
source_arrangement: <path>
sections:
  - name: Intro
    computed_start: '[00:00.000]'
    actual_start: '[00:00.000]'
    drift_seconds: 0
  - name: Bridge
    computed_start: '[01:00.000]'
    actual_start: '[00:50.000]'
    drift_seconds: -10
    vocals_flag_changed: false
```

## Risks / Trade-offs

- Loop name prefix convention must be maintained — if a loop is named without a recognisable
  prefix, it falls into `unknown`. Mitigation: warn in CLI output.
- The `_gw` vocals heuristic is song-specific. Mitigation: `--vocalist-suffix` flag.

## Open Questions

- Should `refresh_plan()` be extended to re-import arrangement on each run, or remain
  a manual one-shot import? → Manual (human decides when to re-import).
