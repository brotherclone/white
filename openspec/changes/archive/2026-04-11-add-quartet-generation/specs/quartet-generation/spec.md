## ADDED Requirements

### Requirement: Four-Voice Counterpoint Generation
The quartet pipeline SHALL generate three counterpoint voices (alto, tenor, bass-voice)
from an approved soprano (melody) MIDI for a given section.

#### Scenario: Voices within range
- **WHEN** the pipeline generates alto, tenor, and bass-voice lines
- **THEN** all generated MIDI notes fall within the defined range for that voice type
  (alto 48–67, tenor 43–62, bass-voice 36–55)

#### Scenario: No consecutive parallel 5ths or octaves
- **WHEN** two adjacent beats in any voice pair produce the same harmonic interval
  as the preceding two beats, and that interval is a perfect 5th (7 st) or octave (12 st)
- **THEN** the pipeline re-rolls the offending voice note by ±1 semitone

#### Scenario: Voice crossing resolution
- **WHEN** alto note falls below tenor note at any beat
- **THEN** the pipeline swaps the two pitches to restore correct voice order

### Requirement: Multi-Channel MIDI Output
The quartet pipeline SHALL write a single MIDI file with four channels (0=soprano,
1=alto, 2=tenor, 3=bass-voice) for each approved section.

#### Scenario: Four channels present
- **WHEN** a quartet candidate is generated for a section
- **THEN** the output MIDI file contains exactly four tracks, one per voice

#### Scenario: Soprano channel is read-only copy
- **WHEN** the quartet MIDI is written
- **THEN** channel 0 is byte-for-byte identical to the approved melody MIDI for that section

### Requirement: Quartet Review Candidates
The quartet pipeline SHALL write candidate entries to `review.yml` including a
`counterpoint_score` (0–1) based on parallel-interval violations and voice-crossing count.

#### Scenario: Counterpoint score in review.yml
- **WHEN** a quartet candidate is generated
- **THEN** its `review.yml` entry contains a numeric `counterpoint_score` between 0 and 1
