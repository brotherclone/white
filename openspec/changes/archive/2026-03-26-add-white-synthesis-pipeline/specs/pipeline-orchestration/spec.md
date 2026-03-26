## ADDED Requirements

### Requirement: White Song Proposal — Sub-Proposals Field
The song proposal schema SHALL support an optional `sub_proposals` field: an ordered list
of production directory paths whose approved chord and lyric material is used as source
material for White synthesis.

`load_song_proposal_unified()` SHALL read `sub_proposals` from the proposal YAML and
include it in the returned dict as a list of strings (empty list if absent).
`song_context.yml` SHALL record `sub_proposals` when written by `init_production.py`.

#### Scenario: sub_proposals read from proposal YAML

- **WHEN** a song proposal YAML contains a `sub_proposals` list of path strings
- **THEN** `load_song_proposal_unified()` returns `sub_proposals` as a list of strings
- **AND** `song_context.yml` written by `init_production.py` includes the `sub_proposals` list

#### Scenario: sub_proposals absent defaults to empty list

- **WHEN** a song proposal YAML has no `sub_proposals` field
- **THEN** `load_song_proposal_unified()` returns `sub_proposals: []`
- **AND** non-White pipelines behave identically to before

#### Scenario: White pipeline resolves sub_proposals at runtime

- **WHEN** the chord or lyric pipeline is invoked on a White production directory
- **THEN** `sub_proposals` paths are resolved relative to the project root
- **AND** a clear error is raised for any path that does not exist or has no
  `chords/review.yml`
