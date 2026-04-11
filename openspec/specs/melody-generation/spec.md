# melody-generation Specification

## Purpose
TBD - created by archiving change add-melody-lyrics-generation. Update Purpose after archive.
## Requirements
### Requirement: Melody Contour Templates
Each `MelodyPattern` in the template library SHALL carry an optional `tags: list[str]`
field drawn from a controlled vocabulary: `stepwise`, `arpeggiated`, `descent`,
`wide_interval`, `sparse`, `dense`, `lamentful`. Existing patterns without tags
behave identically.

The library SHALL include the following additional lamentful/sparse templates:
- `slow_descent` — stepwise downward motion, quarter notes, phrase every 2 bars
- `breath_phrase` — 3-note phrase, long rest, 3-note phrase
- `pentatonic_lament` — minor pentatonic, descending, held notes
- `floating_repeat` — same 2-3 note motif repeated at slightly different rhythmic positions
- `single_line` — one note per bar, whole-note or dotted half

All new templates SHALL carry `lamentful`, `sparse`, or `stepwise` tags as appropriate.

#### Scenario: Tag field present on all patterns
- **WHEN** the melody pattern library is loaded
- **THEN** every `MelodyPattern` has a `tags` attribute (empty list if none assigned)

#### Scenario: Lamentful templates available
- **WHEN** the library is filtered for patterns tagged `lamentful`
- **THEN** at least 4 patterns are returned

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
The melody pipeline composite scoring SHALL incorporate style reference profile
adjustments when `style_reference_profile` is present in `song_context.yml`.

- High `rest_ratio` (> 0.5) → boost sparse/stepwise melody templates
- Low `note_density` (< 2.0) → boost sparse templates; penalise dense
- High `mean_duration_beats` (> 1.5) → boost descent/stepwise templates

#### Scenario: Sparse reference boosts sparse melody
- **WHEN** `style_reference_profile.rest_ratio` is 0.61
- **AND** a sparse and a dense melody template are candidates
- **THEN** the sparse template receives a higher score adjustment

#### Scenario: Missing profile — no adjustment
- **WHEN** no `style_reference_profile` is present
- **THEN** melody scoring proceeds unchanged

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

### Requirement: Melody Pipeline Evolve Flag
The melody pipeline CLI SHALL accept `--evolve`, `--generations` (int, default 8), and
`--population` (int, default 30) flags. When `--evolve` is passed, evolved melody
candidates SHALL be merged into the standard candidate pool before scoring. Evolved
candidates SHALL have their `id` field begin with `evolved_`.

#### Scenario: --evolve flag merges melody candidates
- **GIVEN** the melody pipeline is run with `--evolve`
- **WHEN** candidate generation completes
- **THEN** the candidate pool contains both hand-coded and evolved patterns

