# melody-generation Specification Delta

## MODIFIED Requirements

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
