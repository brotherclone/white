# bass-generation Specification

## Purpose
TBD - created by archiving change add-bass-line-generation. Update Purpose after archive.
## Requirements
### Requirement: Bass Pattern Templates
Each `BassPattern` in the template library SHALL carry an optional `tags: list[str]`
field drawn from a controlled vocabulary: `drone`, `pedal`, `walking`, `arpeggiated`,
`sustained`, `minimal`. Existing patterns without tags behave identically.

The library SHALL include the following additional drone/pedal templates:
- `root_drone` — single root note, whole-note duration, no movement
- `slow_pedal` — root on beat 1, octave below on beat 3
- `descending_sigh` — root → major 7th → 5th over 4 bars, stepwise descent
- `sustained_fifth` — held 5th drone across the bar, slight velocity swell
- `minimal_walk` — root + one passing tone approaching the next chord

All new templates SHALL carry `drone`, `pedal`, or `minimal` tags as appropriate.

#### Scenario: Tag field present on all patterns
- **WHEN** the bass pattern library is loaded
- **THEN** every `BassPattern` has a `tags` attribute (empty list if none assigned)

#### Scenario: Drone/pedal templates available
- **WHEN** the library is filtered for patterns tagged `drone` or `pedal`
- **THEN** at least 4 patterns are returned

### Requirement: Tone Resolution

The bass pipeline SHALL resolve template tone selections to actual MIDI note numbers using chord voicing data from the approved chords.

#### Scenario: Root resolution

- **WHEN** a template specifies tone_selection "root"
- **THEN** the pipeline SHALL extract the root note from the chord voicing and place it in the bass register (MIDI notes 24-60)
- **AND** if the chord root is above note 60, it SHALL be transposed down by octaves until it falls within range

#### Scenario: Interval-based resolution

- **WHEN** a template specifies "5th" or "3rd"
- **THEN** the pipeline SHALL compute the interval relative to the chord root
- **AND** the resulting note SHALL be clamped to the bass register (24-60)

#### Scenario: Approach and passing tones

- **WHEN** a template specifies "chromatic_approach" or "passing_tone"
- **THEN** the pipeline SHALL use the next chord's root to determine the target
- **AND** chromatic_approach SHALL produce the note one semitone below the next root
- **AND** passing_tone SHALL produce a scale-wise step between the current and next root
- **AND** for the last chord in a section, these SHALL fall back to "root"

### Requirement: Bass Pipeline Input

The bass pipeline SHALL read approved chords, harmonic rhythm, and drum patterns from the song's production directory.

#### Scenario: Read approved chord voicings

- **WHEN** the bass pipeline is invoked with a song production directory
- **THEN** it SHALL read approved chord MIDI files from `chords/approved/`
- **AND** extract chord roots and available tones from each voicing
- **AND** read the chord review YAML for section labels, BPM, time signature, and color

#### Scenario: Read approved harmonic rhythm

- **WHEN** the song has an approved harmonic rhythm
- **THEN** the pipeline SHALL read the harmonic rhythm review YAML for per-chord durations
- **AND** apply variable bar durations to the bass pattern (repeating or truncating as needed)
- **AND** fall back to 1 bar per chord if no approved harmonic rhythm exists

#### Scenario: Read approved drum patterns

- **WHEN** the song has approved drum patterns
- **THEN** the pipeline SHALL extract kick drum onset positions from the approved drum MIDI
- **AND** use these for kick alignment scoring
- **AND** proceed without kick alignment if no approved drums exist (theory score uses only root adherence and voice leading)

### Requirement: Bass MIDI Output

The bass pipeline SHALL write candidate MIDI files to the song's production bass directory.

#### Scenario: MIDI generation

- **WHEN** a bass candidate is generated for a section
- **THEN** the pipeline SHALL write a `.mid` file with bass notes on MIDI channel 0
- **AND** all notes SHALL be within MIDI notes 24-60 (bass register)
- **AND** the MIDI file SHALL use the song proposal's BPM for tempo
- **AND** the pattern SHALL repeat for the approved chord duration per the harmonic rhythm

#### Scenario: Output directory structure

- **WHEN** bass candidates are generated
- **THEN** they SHALL be placed in `<song>/bass/candidates/`
- **AND** an `approved/` directory SHALL be created alongside `candidates/`
- **AND** the directories SHALL be created if they do not exist

### Requirement: Composite Scoring
The bass pipeline composite scoring SHALL incorporate style reference profile
adjustments when `style_reference_profile` is present in `song_context.yml`.

- High `mean_duration_beats` (> 1.5 beats) → boost pedal/drone bass templates
- High `rest_ratio` (> 0.5) → boost minimal/drone templates; penalise walking
- Low `harmonic_rhythm` (< 0.5 changes/bar) → boost pedal/drone templates

#### Scenario: Long note reference boosts pedal bass
- **WHEN** `style_reference_profile.mean_duration_beats` is 2.3
- **AND** a pedal and a walking bass are candidates
- **THEN** the pedal pattern receives a higher score adjustment

#### Scenario: Missing profile — no adjustment
- **WHEN** no `style_reference_profile` is present
- **THEN** bass scoring proceeds unchanged

### Requirement: Review File Generation

The bass pipeline SHALL generate a YAML review file alongside the MIDI candidates, listing each candidate with its scores and placeholders for human annotation.

#### Scenario: Review file creation

- **WHEN** the bass pipeline completes scoring and MIDI output
- **THEN** it SHALL write a `review.yml` file in the song's bass directory
- **AND** each candidate SHALL include: id, midi file path, rank, section, style, pattern name, energy level, composite score, and score breakdowns (theory components + chromatic)

#### Scenario: Annotation placeholders

- **WHEN** the review file is generated
- **THEN** each candidate SHALL have `label: null`, `status: pending`, and `notes: ""` fields for the human to fill in

### Requirement: Bass CLI Interface

The bass pipeline SHALL be invocable from the command line.

#### Scenario: Basic invocation

- **WHEN** the user runs the bass pipeline CLI
- **THEN** it SHALL accept `--production-dir` (path to song production directory) and optional `--seed`, `--top-k` (per section), `--theory-weight` (default 0.3), `--chromatic-weight` (default 0.7), `--onnx-path`

#### Scenario: Progress output

- **WHEN** the bass pipeline runs
- **THEN** it SHALL print: sections found, chord roots extracted, templates selected per section, scoring progress, and top candidates per section with score breakdowns

### Requirement: Phrase-Level Velocity Shaping (Bass)
The bass pipeline SHALL apply the same dynamic curve mechanism as the melody pipeline,
with the bass velocity clamp (50–110) enforced after curve application.

#### Scenario: Diminuendo on outro
- **WHEN** no dynamics map is present and the section is labelled `outro`
- **THEN** the LINEAR_DIM curve is applied and all velocities remain ≥ 50

#### Scenario: Accent notes respect clamp ceiling
- **WHEN** a dynamic curve would push an accent note above 110
- **THEN** the velocity is clamped to 110

### Requirement: Bass Pipeline Evolve Flag
The bass pipeline CLI SHALL accept `--evolve`, `--generations` (int, default 8), and
`--population` (int, default 30) flags. When `--evolve` is passed, evolved bass
candidates SHALL be merged into the standard candidate pool before scoring. Evolved
candidates SHALL have their `id` field begin with `evolved_`.

#### Scenario: --evolve flag merges bass candidates
- **GIVEN** the bass pipeline is run with `--evolve`
- **WHEN** candidate generation completes
- **THEN** the candidate pool contains both hand-coded and evolved patterns

