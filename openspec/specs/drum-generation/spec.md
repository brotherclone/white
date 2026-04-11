# drum-generation Specification

## Purpose
TBD - created by archiving change add-drum-pattern-generation. Update Purpose after archive.
## Requirements
### Requirement: Drum Pattern Templates
Each `DrumPattern` in the template library SHALL carry an optional `tags: list[str]`
field drawn from a controlled vocabulary: `sparse`, `dense`, `half_time`, `brushed`,
`motorik`, `ambient`, `ghost_only`, `electronic`. Existing patterns without tags
behave identically to current behaviour.

The library SHALL include the following additional sparse/atmospheric templates:
- `half_time_sparse` — kick on beat 1, snare on beat 3, open hat on the off-beat
- `ghost_verse` — ghost snare only, no kick, whisper hats
- `brushed_folk` — brush swells on 2 and 4, light kick, no hi-hat grid
- `ambient_pulse` — single low kick every 2 bars, crash swell on bar 4
- `kosmische_slow` — motorik feel at half tempo

All new templates SHALL carry `sparse` and/or `ambient` tags.

#### Scenario: Tag field present on all patterns
- **WHEN** the pattern library is loaded
- **THEN** every `DrumPattern` has a `tags` attribute (empty list if none assigned)

#### Scenario: Sparse templates available
- **WHEN** the library is filtered for patterns tagged `sparse`
- **THEN** at least 5 patterns are returned

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

The drum generator SHALL read approved chord labels to determine song sections and generate
section-appropriate drum candidates. Drum generation targets section labels only — it does not
distinguish between chord variants or HR-derived filenames.

#### Scenario: Read approved chord sections

- **WHEN** the drum pipeline is invoked with a song production directory
- **THEN** it SHALL read all `.mid` files in `chords/approved/` and derive section names from
  their filenames (e.g., `verse.mid` → section `verse`)
- **AND** reject if no approved chords exist
- **AND** ignore any `_scratch.mid` files present in `candidates/`

#### Scenario: Section energy mapping

- **WHEN** drum candidates are generated for a section
- **THEN** the generator SHALL apply a default energy mapping (intro=low, verse=medium,
  chorus=high, bridge=low, outro=medium)
- **AND** the user MAY override the energy for any section via CLI

#### Scenario: Energy-adjacent inclusion

- **WHEN** templates are selected for a section
- **THEN** the generator SHALL include templates matching the target energy level AND templates
  one energy level away
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
The drum pipeline composite scoring SHALL incorporate style reference profile
adjustments when `style_reference_profile` is present in `song_context.yml`.

- Low `note_density` (< 2.0 notes/bar) → boost sparse/ambient drum patterns
- High `velocity_variance` (> 20) → boost patterns with ghost notes
- Low `note_density` (< 1.5 notes/bar) → penalise dense/busy patterns

These SHALL be applied as score adjustments after arc and aesthetic hint adjustments.

#### Scenario: Low density reference boosts sparse drum patterns
- **WHEN** `style_reference_profile.note_density` is 1.8
- **AND** a sparse and a dense pattern are candidates
- **THEN** the sparse pattern receives a higher score adjustment than the dense pattern

#### Scenario: Missing profile — no adjustment
- **WHEN** no `style_reference_profile` is present in song_context
- **THEN** drum scoring proceeds unchanged (existing behaviour)

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

### Requirement: Phrase-Level Velocity Shaping (Drums)
The drum pipeline SHALL apply the dynamic curve to all drum note velocities within a
section (scaling all voices uniformly), with the drum velocity clamp (45–127) enforced.

#### Scenario: Crash accent preserved during crescendo
- **WHEN** a LINEAR_CRESC curve is applied to a section containing a crash accent note
- **THEN** the crash velocity is scaled but clamped to 127

#### Scenario: Ghost notes stay soft
- **WHEN** any dynamic curve is applied
- **THEN** ghost notes (originally at velocity 45) are scaled proportionally but
  never rise above 65 (one-third of the dynamic range)

### Requirement: Aesthetic Tag-Weighted Selection
Pipeline phases SHALL read an optional `aesthetic_hints` dict from `song_context.yml`.
When present, phases SHALL apply a score bonus of +0.1 to candidates whose pattern
tags match the hints (keys: `density` — `sparse` | `moderate` | `dense`; `texture` —
`hazy` | `clean` | `rhythmic`). A mis-matched density tag SHALL apply a penalty of
−0.05. The Refractor chromatic score remains dominant; tag weighting is a soft prior.

#### Scenario: Sparse density hint boosts sparse-tagged patterns
- **WHEN** `aesthetic_hints.density == "sparse"` and a candidate uses a `sparse`-tagged pattern
- **THEN** the candidate's composite score is increased by 0.1 relative to an untagged candidate

#### Scenario: No aesthetic_hints — behaviour unchanged
- **WHEN** `aesthetic_hints` is absent from song context
- **THEN** selection behaviour is identical to the pre-tags baseline

### Requirement: Drum Pipeline Evolve Flag
The drum pipeline CLI SHALL accept `--evolve`, `--generations` (int, default 8), and
`--population` (int, default 30) flags. When `--evolve` is passed, evolved drum
candidates SHALL be merged into the standard candidate pool before scoring. Evolved
candidates SHALL be written to `candidates/` with an `evolved_` filename prefix and
their `id` field SHALL begin with `evolved_`.

#### Scenario: --evolve flag merges candidates
- **GIVEN** the drum pipeline is run with `--evolve`
- **WHEN** candidate generation completes
- **THEN** the candidate pool contains both hand-coded and evolved patterns

#### Scenario: Evolved candidates use evolved_ prefix
- **GIVEN** `--evolve` is passed
- **WHEN** MIDI files are written
- **THEN** at least one filename begins with `evolved_`

