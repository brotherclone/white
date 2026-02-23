# Capability: Assembly Manifest

## ADDED Requirements

### Requirement: Parse Logic Arrangement Export
The system SHALL parse a Logic Pro arrangement export in timecode format
(`HH:MM:SS:FF.sub  loop_name  track  HH:MM:SS:FF.sub`) into a list of timed clip placements,
stripping the `01:` hour offset and ignoring frames/sub-frames.

#### Scenario: Timecode parsing
- **WHEN** a line `01:00:50:00.00  bridge_eighth_hypnotic  1  00:00:20:00.00` is parsed
- **THEN** it yields clip `{start: 50.0, name: "bridge_eighth_hypnotic", track: 1, length: 20.0}`

#### Scenario: Multi-track simultaneity
- **WHEN** multiple lines share the same start timecode
- **THEN** they are grouped into one time slot with one clip per track

#### Scenario: Unrecognised prefix
- **WHEN** a loop name has no recognisable section prefix
- **THEN** it is labelled `unknown` and a warning is emitted to stderr

---

### Requirement: Derive Section Map from Arrangement
The system SHALL group clip time slots into named sections based on loop name prefixes
(`intro_*`, `verse_*`, `bridge_*`, `outro_*`), producing a list of sections with actual
wall-clock start and end times.

#### Scenario: Section boundary detection
- **WHEN** clip prefix changes from `intro_` to `bridge_` at 00:50
- **THEN** the Intro section ends at 00:50 and a Bridge section begins at 00:50

#### Scenario: Section with no melody
- **WHEN** a time slot has no track-4 clip
- **THEN** the section is derived from other tracks and `vocals` is set to `false`

#### Scenario: Vocals inference from loop name
- **WHEN** a track-4 clip name ends in `_gw` or contains `vocal`
- **THEN** `vocals` is set to `true` for that section

---

### Requirement: Update Production Plan with Actual Section Data
The system SHALL update `production_plan.yml` sections with corrected timestamps, actual
bar counts recomputed from wall-clock lengths, and a `loops` dict mapping instrument
family to the loop clip name placed in that section.

#### Scenario: Timestamp correction
- **WHEN** the arrangement import runs and the Bridge section actually starts at 00:50
  but the plan computed 01:00
- **THEN** the plan section `start_time` is updated to `[00:50.000]`

#### Scenario: Loops field population
- **WHEN** track 1 has `bridge_eighth_hypnotic` and track 4 has `bridge_2_arp_gw` in a Bridge section
- **THEN** `loops.chords: bridge_eighth_hypnotic` and `loops.melody: bridge_2_arp_gw` are written

#### Scenario: Human vocals override preserved
- **WHEN** a section has `vocals: true` set manually in the plan
- **THEN** the import does not override it to `false` even if no `_gw` clip is detected

---

### Requirement: Emit Drift Report
The system SHALL write `drift_report.yml` to the production directory, comparing each
section's computed timestamp against its actual timestamp and reporting the delta in seconds.

#### Scenario: No drift
- **WHEN** computed and actual timestamps match for all sections
- **THEN** all `drift_seconds` values are `0`

#### Scenario: Drift detected
- **WHEN** a section's actual start differs from its computed start by 10 seconds
- **THEN** `drift_seconds: -10` (or `+10`) is recorded and a warning is printed to stdout

---

### Requirement: CLI Entry Point
The system SHALL expose a CLI:
`python -m app.generators.midi.assembly_manifest --production-dir <dir> --arrangement <file>`
that reads the arrangement export, updates the plan and manifest, and prints a summary
of sections found and any drift detected.

#### Scenario: Successful import
- **WHEN** a valid arrangement file and production dir are provided
- **THEN** `production_plan.yml`, `manifest_bootstrap.yml`, and `drift_report.yml` are updated
  and a section summary is printed

#### Scenario: Missing arrangement file
- **WHEN** the `--arrangement` path does not exist
- **THEN** an error is raised and no files are modified
