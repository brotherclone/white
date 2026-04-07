# melody-generation Specification

## Purpose
TBD - created by archiving change add-melody-lyrics-generation. Update Purpose after archive.
## Requirements
### Requirement: Melody Contour Templates

The melody generator SHALL define contour pattern templates as structured data with relative interval sequences and rhythmic positions. Each template SHALL specify signed semitone deltas from the previous note, allowing the pipeline to resolve templates to actual MIDI pitches within any singer's vocal range.

#### Scenario: Template structure

- **WHEN** a melody contour template is defined
- **THEN** it SHALL include: name, contour type, energy level, time signature, use_case ("vocal" or "lead"), description, intervals (signed semitone deltas), rhythm (onset positions in beats), and optional durations
- **AND** the first interval SHALL always be 0 (starting note, resolved from chord)
- **AND** rhythm and intervals lists SHALL have equal length

#### Scenario: use_case — vocal constraints

- **WHEN** a template has use_case "vocal"
- **THEN** it SHALL have no more than 6 onsets per bar in 4/4 time
- **AND** at least one explicit rest (gap ≥ 0.5 beats) within each bar
- **AND** at least one note with duration ≥ 1.5 beats per bar

#### Scenario: use_case — lead

- **WHEN** a template has use_case "lead"
- **THEN** it MAY have any note density and is intended for instrument tracks, not singer parts
- **AND** it SHALL be excluded from vocal candidate generation

#### Scenario: Template selection filters by use_case

- **WHEN** the pipeline generates a melody for a singer
- **THEN** it SHALL only select templates with use_case "vocal"
- **AND** lead templates SHALL not appear in vocal candidate output

#### Scenario: 4/4 vocal template availability

- **WHEN** the song proposal has a 4/4 time signature
- **THEN** the generator SHALL have at least 30 vocal templates
- **AND** vocal templates SHALL cover at least 6 named archetypes: declarative, call_and_rest, haiku, incantatory, drone_and_step, conversational

#### Scenario: 4/4 lead template availability

- **WHEN** a lead instrument part is being generated in 4/4
- **THEN** the generator SHALL have at least 12 lead templates

#### Scenario: Other time signature vocal template availability

- **WHEN** the song proposal has a 3/4, 6/8, or 7/8 time signature
- **THEN** the generator SHALL have at least 4 vocal templates for that time signature
- **AND** onset positions SHALL respect the meter's natural stress patterns

#### Scenario: Custom time signature fallback

- **WHEN** the song proposal has a time signature without specific templates
- **THEN** the generator SHALL fall back to a minimal repeated-root pattern on beat 1
- **AND** log a warning about the unsupported time signature

### Requirement: Singer Vocal Ranges

The melody generator SHALL enforce singer-specific vocal range constraints on all generated melodies.

#### Scenario: Singer registry

- **WHEN** the pipeline is initialized
- **THEN** it SHALL define vocal ranges for: Busayo (baritone), Gabriel (tenor), Robbie (tenor), Shirley (alto), Katherine (alto), Marvin (bass-baritone), Aloysius (tenor), Remez (tenor)

#### Scenario: Range clamping

- **WHEN** a resolved melody note falls outside the assigned singer's range
- **THEN** the pipeline SHALL mirror the interval direction to keep the note in range
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

The melody pipeline SHALL score each candidate using theory metrics and Refractor, producing a single composite ranking per section.

#### Scenario: Theory scoring — singability

- **WHEN** a melody candidate is scored for singability
- **THEN** the pipeline SHALL penalize intervals larger than an octave
- **AND** reward stepwise motion (1–2 semitones)
- **AND** penalize melodies using less than 50% of the singer's available range
- **AND** penalize more than 6 note onsets per bar in 4/4 (show-tune density)
- **AND** reward held notes with duration ≥ 1.5 beats
- **AND** require at least one rest per bar of vocal melody (not per 4 bars)

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

### Requirement: Review File Generation

The melody pipeline SHALL generate a YAML review file alongside the MIDI candidates.

#### Scenario: Review file creation

- **WHEN** the melody pipeline completes scoring and MIDI output
- **THEN** it SHALL write a `review.yml` file in the song's melody directory
- **AND** each candidate SHALL include: id, midi file path, rank, section, contour type, pattern name, energy level, use_case, singer, composite score, and score breakdowns

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

### Requirement: Melody Use-Case Annotation
The melody pipeline SHALL write a `use_case` field (`"vocal"` or `"instrumental"`) into
every candidate entry in `review.yml`, derived from the `MelodyPattern.use_case` attribute
of the winning template.

#### Scenario: Vocal template candidate
- **WHEN** a candidate is generated from a `MelodyPattern` with `use_case="vocal"`
- **THEN** its `review.yml` entry contains `use_case: vocal`

#### Scenario: Instrumental template candidate
- **WHEN** a candidate is generated from a `MelodyPattern` with `use_case="instrumental"`
- **THEN** its `review.yml` entry contains `use_case: instrumental`

#### Scenario: Promoted entry preserves use-case
- **WHEN** a candidate is approved and written to the promoted loop list
- **THEN** the promoted entry retains the `use_case` field

### Requirement: Cross-Section Melodic Continuity
The melody pipeline SHALL apply a continuity penalty to candidates whose opening note
creates a large interval leap from the closing note of the preceding approved melody
section, as ordered by `production_plan.yml`.

The penalty SHALL be a score multiplier of 0.85× applied when the interval exceeds
`melodic_continuity_semitones` (default: 4, configurable in `song_proposal.yml`).

#### Scenario: Smooth transition preferred
- **WHEN** two candidate templates for a section start within 4 semitones of the
  preceding section's last note
- **THEN** neither receives the continuity penalty and they are ranked by other factors

#### Scenario: Leap penalised
- **WHEN** a candidate's first note is more than `melodic_continuity_semitones` away
  from the preceding approved section's last note
- **THEN** the candidate's composite score is multiplied by 0.85

#### Scenario: No preceding section — no penalty
- **WHEN** the section being generated is the first approved melody section in the plan
- **THEN** no continuity penalty is applied to any candidate

#### Scenario: Custom threshold from proposal
- **WHEN** `song_proposal.yml` contains `melodic_continuity_semitones: 6`
- **THEN** the 0.85× penalty applies only when the interval exceeds 6 semitones

### Requirement: Phrase-Level Velocity Shaping (Melody)
The melody pipeline SHALL apply a dynamic curve to the velocity of all notes within a
generated section before writing the MIDI candidate file.

The curve SHALL be determined by (in priority order):
1. The `dynamics` map in `song_proposal.yml` for the current section label
2. `infer_curve(section_energy)` heuristic
3. Default: FLAT (no change)

#### Scenario: Linear crescendo applied
- **WHEN** a section is configured with `linear_cresc`
- **THEN** note velocities increase monotonically from the first note to the last,
  bounded by the melody velocity clamp (60–127)

#### Scenario: Swell on intro by default
- **WHEN** no dynamics map is present and the section is labelled `intro`
- **THEN** the SWELL curve is applied (velocities rise then fall across the section)

#### Scenario: Flat preserves existing velocities
- **WHEN** the effective curve is FLAT
- **THEN** note velocities are identical to the pre-curve values (no-op)

