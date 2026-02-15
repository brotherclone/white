## ADDED Requirements

### Requirement: Drum Pattern Templates

The drum generator SHALL define pattern templates as structured data with multi-voice support (kick, snare, hi-hat, toms, cymbals). Each template SHALL specify per-voice onset positions and velocity levels (accent, normal, ghost) relative to the bar.

#### Scenario: Template structure

- **WHEN** a drum pattern template is defined
- **THEN** it SHALL include: name, genre family, energy level, time signature, description, and a voices dict mapping voice names to lists of (beat_position, velocity_level) tuples
- **AND** beat positions SHALL be floats relative to the bar (0 = beat 1)
- **AND** velocity levels SHALL be one of: accent (120), normal (90), ghost (45)

#### Scenario: GM percussion MIDI mapping

- **WHEN** drum MIDI is generated from a template
- **THEN** voice names SHALL map to General MIDI channel 10 percussion note numbers (kick=36, snare=38, hh_closed=42, hh_open=46, etc.)
- **AND** all drum events SHALL be written to MIDI channel 10 (channel index 9)

#### Scenario: 4/4 template availability

- **WHEN** the song proposal has a 4/4 time signature
- **THEN** the generator SHALL have templates across at least 3 genre families
- **AND** each genre family SHALL have at least low, medium, and high energy templates

#### Scenario: 7/8 template availability

- **WHEN** the song proposal has a 7/8 time signature
- **THEN** the generator SHALL have templates using asymmetric groupings (e.g., 3+2+2, 2+2+3)
- **AND** onset positions SHALL align with the 7 eighth-note subdivisions of the bar

#### Scenario: Custom time signature fallback

- **WHEN** the song proposal has a time signature without specific templates
- **THEN** the generator SHALL fall back to a minimal kick-on-1 pattern subdivided to match the bar length
- **AND** log a warning about the unsupported time signature

### Requirement: Genre Family Mapping

The drum generator SHALL map song proposal genre tags to genre families that determine which templates are applicable.

#### Scenario: Genre tag scanning

- **WHEN** a song proposal has genre tags
- **THEN** the generator SHALL scan each tag for keywords that match genre families (ambient, electronic, krautrock, rock, classical, experimental, folk, jazz)
- **AND** multiple families MAY match for a single song

#### Scenario: No genre match fallback

- **WHEN** no genre tags match any genre family
- **THEN** the generator SHALL fall back to the `electronic` family
- **AND** log a warning about the fallback

### Requirement: Section-Aware Generation

The drum generator SHALL read approved chord labels to determine song sections and generate section-appropriate drum candidates.

#### Scenario: Read approved chord sections

- **WHEN** the drum pipeline is invoked with a song production directory
- **THEN** it SHALL read the chord `review.yml` to find approved candidates and their labels (verse, chorus, bridge, intro, outro)
- **AND** reject if no approved chords exist

#### Scenario: Section energy mapping

- **WHEN** drum candidates are generated for a section
- **THEN** the generator SHALL apply a default energy mapping (intro=low, verse=medium, chorus=high, bridge=low, outro=medium)
- **AND** the user MAY override the energy for any section via CLI

#### Scenario: Energy-adjacent inclusion

- **WHEN** templates are selected for a section
- **THEN** the generator SHALL include templates matching the target energy level AND templates one energy level away
- **AND** exact-match templates SHALL rank higher in the default ordering

### Requirement: Drum MIDI Output

The drum generator SHALL write candidate MIDI files to the song's production drums directory.

#### Scenario: MIDI file generation

- **WHEN** a drum candidate is generated for a section
- **THEN** the pipeline SHALL write a `.mid` file with drum events on MIDI channel 10
- **AND** the MIDI file SHALL use the song proposal's BPM for tempo
- **AND** the pattern SHALL repeat for the same number of bars as the approved chord for that section

#### Scenario: Output directory structure

- **WHEN** drum candidates are generated
- **THEN** they SHALL be placed in `<song>/drums/candidates/`
- **AND** files SHALL be named `<section>_<genre_family>_<pattern_name>.mid`
- **AND** the directory SHALL be created if it does not exist

### Requirement: Composite Scoring

The drum pipeline SHALL score each candidate using energy appropriateness and ChromaticScorer, producing a single composite ranking per section.

#### Scenario: Energy appropriateness scoring

- **WHEN** a drum candidate is scored
- **THEN** the pipeline SHALL compute an energy appropriateness score: 1.0 for exact energy match, 0.5 for one level away, 0.0 for two levels away

#### Scenario: Chromatic scoring

- **WHEN** a drum candidate is scored
- **THEN** the pipeline SHALL convert the candidate to MIDI bytes and score with `ChromaticScorer.score()`
- **AND** the concept embedding SHALL be computed once and reused across all candidates

#### Scenario: Composite ranking

- **WHEN** all candidates for a section are scored
- **THEN** the pipeline SHALL compute a weighted composite score (default: 30% energy appropriateness, 70% chromatic)
- **AND** rank candidates by composite score descending per section
- **AND** present top-k candidates per section in the review file

### Requirement: Review File Generation

The drum pipeline SHALL generate a YAML review file alongside the MIDI candidates, listing each candidate with its scores and placeholders for human annotation.

#### Scenario: Review file creation

- **WHEN** the drum pipeline completes scoring and MIDI output
- **THEN** it SHALL write a `review.yml` file in the song's drums directory
- **AND** each candidate SHALL include: id, midi file path, rank, section, genre family, pattern name, energy level, composite score, and score breakdowns

#### Scenario: Annotation placeholders

- **WHEN** the review file is generated
- **THEN** each candidate SHALL have `label: null`, `status: pending`, and `notes: ""` fields for the human to fill in

### Requirement: Drum CLI Interface

The drum pipeline SHALL be invocable from the command line.

#### Scenario: Basic invocation

- **WHEN** the user runs the drum pipeline CLI
- **THEN** it SHALL accept `--production-dir` (path to song production directory) and optional `--seed`, `--top-k` (per section), `--energy-override` (section=level pairs), `--genre-override` (force specific genre families)

#### Scenario: Progress output

- **WHEN** the drum pipeline runs
- **THEN** it SHALL print: sections found, genre families matched, templates selected per section, scoring progress, and top candidates per section with score breakdowns
