# bass-generation Specification

## Purpose
Bass line generation for the music production pipeline. Generates bass patterns from approved chord progressions, harmonic rhythm, and drum patterns, scored with theory + chromatic composite, gated through human review.

## ADDED Requirements

### Requirement: Bass Pattern Templates

The bass generator SHALL define pattern templates as structured data with tone-selection rules and rhythmic positions. Each template SHALL specify which chord tone to play at each beat position, allowing the pipeline to resolve templates to actual MIDI notes from any chord voicing.

#### Scenario: Template structure

- **WHEN** a bass pattern template is defined
- **THEN** it SHALL include: name, style (root/walking/pedal/arpeggiated/octave/syncopated), energy level, time signature, description, and a notes list of (beat_position, tone_selection, velocity_level) tuples
- **AND** beat positions SHALL be floats relative to bar start (0 = beat 1)
- **AND** tone selections SHALL be one of: root, 5th, 3rd, octave_up, octave_down, chromatic_approach, passing_tone
- **AND** velocity levels SHALL be one of: accent (100), normal (80), ghost (50)

#### Scenario: 4/4 template availability

- **WHEN** the song proposal has a 4/4 time signature
- **THEN** the generator SHALL have templates across at least 4 styles (root, walking, arpeggiated, syncopated)
- **AND** each style SHALL have at least one template at low, medium, or high energy
- **AND** the total template count SHALL be at least 12

#### Scenario: 7/8 template availability

- **WHEN** the song proposal has a 7/8 time signature
- **THEN** the generator SHALL have at least 3 templates using group-aligned beat positions
- **AND** onset positions SHALL align with the 7 eighth-note subdivisions of the bar

#### Scenario: Custom time signature fallback

- **WHEN** the song proposal has a time signature without specific templates
- **THEN** the generator SHALL fall back to a minimal root-on-beat-1 pattern
- **AND** log a warning about the unsupported time signature

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

The bass pipeline SHALL score each candidate using theory metrics and ChromaticScorer, producing a single composite ranking per section.

#### Scenario: Theory scoring

- **WHEN** a bass candidate is scored
- **THEN** the pipeline SHALL compute a theory score as the mean of:
  - root_adherence: fraction of strong-beat notes that are the chord root (0.0-1.0)
  - kick_alignment: fraction of bass onsets coinciding with kick drum hits (0.0-1.0)
  - voice_leading: smoothness of bass movement between chords (0.0-1.0, inverse of interval size)
- **AND** if no drum data is available, kick_alignment SHALL be omitted from the mean

#### Scenario: Chromatic scoring

- **WHEN** a bass candidate is scored
- **THEN** the pipeline SHALL convert the candidate to MIDI bytes and score with `ChromaticScorer.score()`
- **AND** the concept embedding SHALL be computed once and reused across all candidates

#### Scenario: Composite ranking

- **WHEN** all candidates for a section are scored
- **THEN** the pipeline SHALL compute a weighted composite score (default: 30% theory, 70% chromatic)
- **AND** rank candidates by composite score descending per section
- **AND** present top-k candidates per section in the review file

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
