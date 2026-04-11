# quartet-generation Specification

## Purpose
TBD - created by archiving change add-quartet-generation. Update Purpose after archive.
## Requirements
### Requirement: Four-Voice Counterpoint Generation
The quartet pipeline SHALL generate three counterpoint voices (violin_ii, viola, cello)
from an approved violin_i (melody) MIDI for a given section, modelling a string quartet.

Voice mapping:
- Channel 0: violin_i (read-only copy of approved melody)
- Channel 1: violin_ii (MelodyPattern)
- Channel 2: viola (MelodyPattern)
- Channel 3: cello (BassPattern)

#### Scenario: Voices within range
- **WHEN** the pipeline generates violin_ii, viola, and cello lines
- **THEN** all generated MIDI notes fall within the defined range for that voice type
  (violin_ii 55–84, viola 48–77, cello 36–60)

#### Scenario: No consecutive parallel 5ths or octaves
- **WHEN** two adjacent beats in any voice pair produce the same harmonic interval
  as the preceding two beats, and that interval is a perfect 5th (7 st) or octave (12 st)
- **THEN** the pipeline re-rolls the offending voice note by ±1 semitone

#### Scenario: Voice crossing resolution
- **WHEN** a lower voice note rises above the note in the voice immediately above it
- **THEN** the pipeline adjusts the offending note to restore correct voice order

### Requirement: Multi-Channel MIDI Output
The quartet pipeline SHALL write a single MIDI file with four tracks
(0=violin_i, 1=violin_ii, 2=viola, 3=cello) for each approved section.

#### Scenario: Four channels present
- **WHEN** a quartet candidate is generated for a section
- **THEN** the output MIDI file contains exactly four tracks, one per voice

#### Scenario: Violin I channel is read-only copy
- **WHEN** the quartet MIDI is written
- **THEN** channel 0 is byte-for-byte identical to the approved melody MIDI for that section

### Requirement: Quartet Review Candidates
The quartet pipeline SHALL write candidate entries to `review.yml` including a
`counterpoint_score` (0–1) based on parallel-interval violations and voice-crossing count.

#### Scenario: Counterpoint score in review.yml
- **WHEN** a quartet candidate is generated
- **THEN** its `review.yml` entry contains a numeric `counterpoint_score` between 0 and 1

