# strum-generation Specification

## Purpose
TBD - created by archiving change add-strum-rhythm-generation. Update Purpose after archive.
## Requirements
### Requirement: Strum Pattern Templates

The strum generator SHALL define rhythm pattern templates as structured data, with templates for common time signatures (4/4, 7/8). Each template SHALL specify note onset positions and durations relative to the bar.

#### Scenario: 4/4 pattern availability

- **WHEN** the song proposal has a 4/4 time signature
- **THEN** the generator SHALL offer at least: whole, half, quarter, eighth, syncopated, arp-up, and arp-down patterns
- **AND** each pattern SHALL specify onset beats and note durations that sum to the full bar length

#### Scenario: 7/8 pattern availability

- **WHEN** the song proposal has a 7/8 time signature
- **THEN** the generator SHALL offer at least: whole, grouped-322, grouped-223, and eighth patterns
- **AND** onset positions SHALL be expressed in eighth-note units (0-6)

#### Scenario: Custom time signature fallback

- **WHEN** the song proposal has a time signature without specific templates
- **THEN** the generator SHALL fall back to subdividing the bar into equal beats
- **AND** offer at minimum: whole and equal-subdivision patterns

### Requirement: Strum Pipeline Input

The strum generator SHALL accept a song production directory containing approved chord MIDI files and the original song proposal metadata (BPM, time signature).

#### Scenario: Load approved chords

- **WHEN** the strum pipeline is invoked with a song production directory
- **THEN** it SHALL read all `.mid` files from `chords/approved/`
- **AND** parse the chord review YAML to retrieve BPM and time signature
- **AND** reject if no approved chords exist

#### Scenario: Parse chord voicings from MIDI

- **WHEN** an approved chord MIDI file is loaded
- **THEN** the pipeline SHALL extract the chord voicings (which notes are played per bar)
- **AND** preserve the original note pitches and count

### Requirement: Strum Candidate Generation

The strum generator SHALL apply each applicable rhythm pattern to each approved chord, producing one MIDI file per chord-pattern combination.

#### Scenario: Per-chord candidates

- **WHEN** strum generation runs in per-chord mode (default)
- **THEN** it SHALL produce one MIDI file per (approved chord, rhythm pattern) pair
- **AND** each file SHALL contain the chord's notes played with the pattern's onset and duration timing
- **AND** the MIDI file SHALL use the song proposal's BPM for tempo

#### Scenario: Progression mode candidates

- **WHEN** strum generation runs in progression mode
- **THEN** it SHALL produce one MIDI file per rhythm pattern
- **AND** each file SHALL contain all approved chords played in sequence with the same pattern applied to each bar
- **AND** chord ordering SHALL follow the order from the chord review file (by rank of approved candidates)

#### Scenario: Arpeggio patterns

- **WHEN** an arpeggio pattern (arp-up or arp-down) is applied to a chord
- **THEN** the generator SHALL distribute individual chord tones across the pattern's subdivisions
- **AND** arp-up SHALL order notes from lowest to highest MIDI pitch
- **AND** arp-down SHALL order notes from highest to lowest MIDI pitch
- **AND** if there are more subdivisions than chord tones, the arpeggio SHALL cycle

### Requirement: Strum MIDI Output

The strum generator SHALL write candidate MIDI files to the song's production strums directory.

#### Scenario: Output directory structure

- **WHEN** strum candidates are generated
- **THEN** they SHALL be placed in `<song>/strums/candidates/`
- **AND** files SHALL be named `<chord_label>_<pattern>.mid` (e.g., `verse_quarter.mid`)
- **AND** progression mode files SHALL be named `progression_<pattern>.mid`
- **AND** the directory SHALL be created if it does not exist

#### Scenario: Review file generation

- **WHEN** strum candidates are written
- **THEN** a `review.yml` SHALL be generated in `<song>/strums/`
- **AND** it SHALL follow the same schema as the chord review YAML (id, midi_file, rank, label, status, notes)
- **AND** each candidate SHALL include metadata: source chord label, pattern name, pattern description

### Requirement: Strum CLI Interface

The strum pipeline SHALL be invocable from the command line.

#### Scenario: Basic invocation

- **WHEN** the user runs the strum pipeline CLI
- **THEN** it SHALL accept `--production-dir` (path to song production directory) and optional `--mode` (per-chord or progression), `--patterns` (comma-separated pattern names to include)

#### Scenario: Progress output

- **WHEN** the strum pipeline runs
- **THEN** it SHALL print the number of approved chords found, patterns being applied, and files written

