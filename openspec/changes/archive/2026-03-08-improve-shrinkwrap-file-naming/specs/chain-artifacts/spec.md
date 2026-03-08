## MODIFIED Requirements

### Requirement: Chain Artifact Shrink-Wrap
The system SHALL provide a utility to clean up completed chain artifact threads by removing debug files, renaming directories to human-readable names, cleaning individual file names, and generating structured metadata summaries.

#### Scenario: Thread discovery
- **WHEN** the shrink-wrap utility scans `chain_artifacts/`
- **THEN** it identifies all UUID-named thread directories containing `all_song_proposals_*.yml`

#### Scenario: Directory renaming
- **WHEN** a thread is shrink-wrapped
- **THEN** the output directory is named `{color}-{slugified-title}` (e.g., `black-the-phantom-limb-protocol`) with no agent-name prefix

#### Scenario: File name cleaning — UUID and color-char prefix
- **WHEN** a file's name matches `<uuid>_<single-char>_<semantic-name>.<ext>`
- **THEN** the output file is written as `<semantic-name>.<ext>`

#### Scenario: File name cleaning — white_agent prefix
- **WHEN** a file's name matches `white_agent_<thread-uuid>_<TYPE>.<ext>`
- **THEN** the output file is written as `<type_lowercase>.<ext>` (e.g., `agent_voices.md`, `chromatic_synthesis.md`)

#### Scenario: File name cleaning — all_song_proposals thread suffix
- **WHEN** a file's name matches `all_song_proposals_<thread-uuid>.<ext>`
- **THEN** the output file is written as `all_song_proposals.<ext>`

#### Scenario: File name cleaning — song_proposal color prefix
- **WHEN** a file's name matches `song_proposal_<Color...>_<name>.<ext>` or `song_proposal_<char>_<name>.<ext>`
- **THEN** the output file is written as `<name>.<ext>` (e.g., `neural_network_incarnation_v2.yml`)

#### Scenario: File name collision handling
- **WHEN** two files in the same output subdirectory would produce the same clean name
- **THEN** the second file is written as `<name>_2.<ext>`, the third as `<name>_3.<ext>`, and so on

#### Scenario: In-file file_name field update
- **WHEN** a copied file (YAML or Markdown front-matter) contains a `file_name:` field
- **THEN** that field's value is rewritten to the clean output filename

#### Scenario: Debug artifact removal
- **WHEN** a thread is shrink-wrapped with default settings
- **THEN** intermediate files (rebracketing analyses, transformation traces, facet evolution) are excluded from the output

#### Scenario: Debug artifact archival
- **WHEN** shrink-wrap is run with `--archive` flag
- **THEN** debug artifacts are copied to a `.debug/` subdirectory instead of being excluded

#### Scenario: Dry run
- **WHEN** shrink-wrap is run with `--dry-run` flag
- **THEN** it reports what would change without modifying any files

#### Scenario: Summary manifest generation
- **WHEN** a thread is shrink-wrapped
- **THEN** a `manifest.yml` is generated containing: title, bpm, key, tempo, concept, rainbow_color, mood, genres, agent_name, original thread_id, and timestamp
