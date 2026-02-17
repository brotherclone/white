# melody-generation Specification

## Purpose

Melody generation for the music production pipeline. Generates singable melody MIDI from approved chord progressions within singer vocal range constraints. Vocal synthesis is handled externally in ACE Studio.

## ADDED Requirements

### Requirement: Melody Contour Templates

The melody generator SHALL define contour pattern templates as structured data with relative interval sequences and rhythmic positions. Each template SHALL specify signed semitone deltas from the previous note, allowing the pipeline to resolve templates to actual MIDI pitches within any singer's vocal range.

#### Scenario: Template structure

- **WHEN** a melody contour template is defined
- **THEN** it SHALL include: name, contour type (stepwise/arpeggiated/repeated/leap_step/pentatonic/scalar_run), energy level, time signature, description, intervals (signed semitone deltas), rhythm (onset positions in beats), and optional durations
- **AND** the first interval SHALL always be 0 (starting note, resolved from chord)
- **AND** rhythm and intervals lists SHALL have equal length

#### Scenario: 4/4 template availability

- **WHEN** the song proposal has a 4/4 time signature
- **THEN** the generator SHALL have at least 12 templates across multiple contour types
- **AND** contour types SHALL include at least: stepwise, arpeggiated, repeated, and leap_step

#### Scenario: 7/8 template availability

- **WHEN** the song proposal has a 7/8 time signature
- **THEN** the generator SHALL have at least 6 templates using group-aligned onset positions
- **AND** onset positions SHALL respect asymmetric groupings (e.g., 3+2+2 or 2+2+3)

#### Scenario: Custom time signature fallback

- **WHEN** the song proposal has a time signature without specific templates
- **THEN** the generator SHALL fall back to a minimal repeated-root pattern on beat 1
- **AND** log a warning about the unsupported time signature

### Requirement: Singer Vocal Ranges

The melody generator SHALL enforce singer-specific vocal range constraints on all generated melodies.

#### Scenario: Singer registry

- **WHEN** the pipeline is initialized
- **THEN** it SHALL define vocal ranges for: Busyayo (A2–E4, baritone), Gabriel (C3–G4, tenor), Robbie (C3–G4, tenor), Shirley (F3–C5, low alto), Katherine (A3–E5, high alto)

#### Scenario: Range clamping

- **WHEN** a resolved melody note falls outside the assigned singer's range
- **THEN** the pipeline SHALL mirror the interval direction (e.g., +3 becomes -3) to keep the note in range
- **AND** if mirroring also exceeds range, clamp to the nearest range boundary

#### Scenario: Singer assignment

- **WHEN** the song proposal specifies a singer name
- **THEN** the pipeline SHALL use that singer's vocal range
- **WHEN** no singer is specified
- **THEN** the pipeline SHALL infer a singer whose mid-range best covers the song's tonic pitch
- **AND** the `--singer` CLI flag SHALL override both proposal and inference

### Requirement: Melody Resolution

The melody pipeline SHALL resolve contour templates to actual MIDI note sequences using chord voicing data and singer range constraints.

#### Scenario: Starting pitch resolution

- **WHEN** a contour template is applied to a chord
- **THEN** the starting pitch SHALL be the chord root transposed into the singer's comfortable mid-range
- **AND** the mid-range SHALL be defined as the midpoint of the singer's range plus or minus 5 semitones

#### Scenario: Interval walking

- **WHEN** intervals are applied sequentially from the starting pitch
- **THEN** each note SHALL equal the previous note plus the signed interval delta
- **AND** notes exceeding the singer's range SHALL have their interval mirrored

#### Scenario: Strong-beat chord-tone snap

- **WHEN** a note falls on a strong beat (beat 1, beat 3 in 4/4)
- **THEN** if the note is not a chord tone (root, 3rd, 5th) it SHALL snap to the nearest chord tone within 2 semitones
- **AND** if no chord tone is within 2 semitones, the note SHALL remain unchanged

#### Scenario: Phrase ending resolution

- **WHEN** the last note of a section's melody is generated
- **THEN** it SHALL resolve to the root or 5th of the current chord within the singer's range

### Requirement: Melody Pipeline Input

The melody pipeline SHALL read approved chords, harmonic rhythm, and song proposal metadata from the production directory.

#### Scenario: Read approved chord voicings

- **WHEN** the melody pipeline is invoked with a song production directory
- **THEN** it SHALL read approved chord MIDI files from `chords/approved/`
- **AND** extract chord tones (root, 3rd, 5th) from each voicing
- **AND** read the chord review YAML for section labels, BPM, time signature, and color

#### Scenario: Read approved harmonic rhythm

- **WHEN** the song has an approved harmonic rhythm
- **THEN** the pipeline SHALL read per-chord durations from the harmonic rhythm review YAML
- **AND** fall back to 1 bar per chord if no approved harmonic rhythm exists

#### Scenario: Read chromatic synthesis reference

- **WHEN** a Chromatic Synthesis document exists for the production thread
- **THEN** the pipeline SHALL extract per-section lyrical themes
- **AND** include them as reference text in the review YAML (for human context, not scoring)

### Requirement: Melody MIDI Output

The melody pipeline SHALL write candidate MIDI files to the song's production melody directory.

#### Scenario: MIDI generation

- **WHEN** a melody candidate is generated for a section
- **THEN** the pipeline SHALL write a `.mid` file with melody notes on MIDI channel 0
- **AND** all notes SHALL be within the assigned singer's MIDI range
- **AND** the MIDI file SHALL use the song proposal's BPM for tempo

#### Scenario: Output directory structure

- **WHEN** melody candidates are generated
- **THEN** they SHALL be placed in `<song>/melody/candidates/`
- **AND** an `approved/` directory SHALL be created alongside `candidates/`
- **AND** the directories SHALL be created if they do not exist

### Requirement: Composite Scoring

The melody pipeline SHALL score each candidate using theory metrics and ChromaticScorer, producing a single composite ranking per section.

#### Scenario: Theory scoring — singability

- **WHEN** a melody candidate is scored for singability
- **THEN** the pipeline SHALL penalize intervals larger than an octave
- **AND** reward stepwise motion (1–2 semitones)
- **AND** penalize melodies using less than 50% of the singer's available range
- **AND** require at least one rest per 4 bars of melody

#### Scenario: Theory scoring — chord-tone alignment

- **WHEN** a melody candidate is scored for chord-tone alignment
- **THEN** the pipeline SHALL compute the fraction of strong-beat notes that are chord tones (root, 3rd, 5th)
- **AND** passing tones on weak beats SHALL not penalize the score

#### Scenario: Theory scoring — contour quality

- **WHEN** a melody candidate is scored for contour quality
- **THEN** the pipeline SHALL reward arch-shaped contours with a climax roughly 2/3 through the section
- **AND** penalize more than 4 consecutive identical pitches
- **AND** reward resolution to a stable chord tone on the final note

#### Scenario: Composite ranking

- **WHEN** all candidates for a section are scored
- **THEN** the pipeline SHALL compute: `composite = 0.30 * theory + 0.70 * chromatic`
- **AND** rank candidates by composite score descending per section
- **AND** present top-k candidates per section in the review file

### Requirement: Review File Generation

The melody pipeline SHALL generate a YAML review file alongside the MIDI candidates.

#### Scenario: Review file creation

- **WHEN** the melody pipeline completes scoring and MIDI output
- **THEN** it SHALL write a `review.yml` file in the song's melody directory
- **AND** each candidate SHALL include: id, midi file path, rank, section, contour type, pattern name, energy level, singer, composite score, and score breakdowns

#### Scenario: Chromatic synthesis reference

- **WHEN** chromatic synthesis thematic text is available for a section
- **THEN** the review file SHALL include a `thematic_reference` field per section containing the relevant excerpt
- **AND** this field is for human context only and does not affect scoring

#### Scenario: Annotation placeholders

- **WHEN** the review file is generated
- **THEN** each candidate SHALL have `label: null`, `status: pending`, and `notes: ""` fields for human annotation

### Requirement: Melody CLI Interface

The melody pipeline SHALL be invocable from the command line.

#### Scenario: Basic invocation

- **WHEN** the user runs the melody pipeline CLI
- **THEN** it SHALL accept `--production-dir`, `--singer` (name), `--seed`, `--top-k`, `--theory-weight` (default 0.3), `--chromatic-weight` (default 0.7), `--onnx-path`

#### Scenario: Progress output

- **WHEN** the melody pipeline runs
- **THEN** it SHALL print: singer assigned, sections found, templates selected per section, scoring progress, and top candidates per section with score breakdowns
