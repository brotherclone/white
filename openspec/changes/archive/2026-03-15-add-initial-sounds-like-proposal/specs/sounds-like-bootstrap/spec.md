## ADDED Requirements

### Requirement: Sounds-Like Bootstrap Generation
The system SHALL generate an initial `sounds_like` list from a song proposal before any
MIDI pipeline phase runs, by prompting Claude with the song's concept, color, genres, and
mood and requesting bare artist name strings (no descriptions, no annotations).

#### Scenario: Successful generation
- **WHEN** a song proposal with concept, color, and at least one of genres or mood is
  provided
- **THEN** Claude returns 4–7 artist name strings and `initial_proposal.yml` is written
  to the production directory with those names under the `sounds_like` key

#### Scenario: Claude returns annotated names
- **WHEN** Claude returns names with parenthetical context (e.g. "Sufjan Stevens
  (Illinois-era)")
- **THEN** the parser strips annotations and stores only the bare name

#### Scenario: Idempotent on re-run
- **WHEN** `initial_proposal.yml` already exists and `--force` is not set
- **THEN** the file is not regenerated; a note is printed and the existing file is used

---

### Requirement: Initial Proposal File
The system SHALL write `initial_proposal.yml` to the production directory containing
`sounds_like`, `color`, `concept`, `singer`, `key`, `bpm`, `time_sig`, a generated
timestamp, and `proposed_by: claude`.

#### Scenario: File structure valid
- **WHEN** `write_initial_proposal()` completes
- **THEN** `initial_proposal.yml` is valid YAML with all required fields; `sounds_like`
  is a list of strings (not dicts)

#### Scenario: Missing file returns empty dict
- **WHEN** `load_initial_proposal()` is called and no `initial_proposal.yml` exists
- **THEN** an empty dict is returned with no exception

---

### Requirement: Pipeline sounds_like Propagation
All MIDI pipeline phases (chord, drum, bass, melody, lyric) SHALL read `sounds_like` from
`initial_proposal.yml` when present, using it as the source of artist reference context
for `load_artist_context()`.

#### Scenario: Chord pipeline uses bootstrap sounds_like
- **WHEN** `initial_proposal.yml` exists with a non-empty `sounds_like` list
- **THEN** the chord generation prompt includes the STYLE REFERENCES block from
  `load_artist_context(sounds_like)`

#### Scenario: Lyric pipeline uses bootstrap sounds_like
- **WHEN** `initial_proposal.yml` exists with a non-empty `sounds_like` list
- **THEN** the lyric generation prompt includes the STYLE REFERENCES block; the pipeline
  does NOT zero out `sounds_like`

#### Scenario: Graceful fallback when bootstrap missing
- **WHEN** no `initial_proposal.yml` exists (production dir created before this feature)
- **THEN** pipelines fall back to the song proposal's own `sounds_like` field (or empty);
  no error is raised

---

### Requirement: Composition Proposal Seeding
The composition proposal generator SHALL read `sounds_like` from `initial_proposal.yml`
as a starting point and SHALL allow Claude to extend, replace, or confirm the list in the
composition proposal output.

#### Scenario: Bootstrap seeds composition proposal
- **WHEN** `initial_proposal.yml` exists before composition proposal generation
- **THEN** the composition proposal prompt includes the bootstrap sounds_like as "Existing
  sounds_like" context; Claude's output may differ from the bootstrap list
