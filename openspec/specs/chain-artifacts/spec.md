# chain-artifacts Specification

## Purpose
TBD - created by archiving change add-chain-result-feedback. Update Purpose after archive.
## Requirements
### Requirement: Negative Constraint Generation
The system SHALL generate a negative constraints file from prior chain results to prevent the pipeline from converging on repeated outputs.

#### Scenario: Constraint file generation
- **WHEN** the constraint generator is run after shrink-wrapping
- **THEN** `shrink_wrapped/negative_constraints.yml` is created from `shrink_wrapped/index.yml`

#### Scenario: Key/BPM clustering detection
- **WHEN** more than 30% of prior proposals share the same key and similar BPM (within +/- 5)
- **THEN** that key/BPM combination is flagged as `avoid` in the constraints file

#### Scenario: Concept similarity detection
- **WHEN** multiple prior proposals contain similar concept text (shared keywords or phrases)
- **THEN** the repeated phrases are listed as concept keywords to avoid

#### Scenario: Title deduplication
- **WHEN** a title has already been used in a prior proposal
- **THEN** it is listed as an excluded title in the constraints file

#### Scenario: Manual override support
- **WHEN** the user adds or removes entries in the constraints file manually
- **THEN** the constraint generator preserves manual entries on subsequent runs

### Requirement: Constraint-Aware Proposal Generation
The system SHALL incorporate negative constraints when generating new song proposals to increase output diversity.

#### Scenario: Constraint loading at workflow start
- **WHEN** a new chain workflow starts and `shrink_wrapped/index.yml` exists
- **THEN** the constraints are loaded and made available to the White agent

#### Scenario: Soft avoidance
- **WHEN** a constraint has severity `avoid`
- **THEN** the White agent is prompted to deprioritize (not hard-block) that combination

#### Scenario: Hard exclusion
- **WHEN** a constraint has severity `exclude`
- **THEN** the White agent MUST NOT produce a proposal matching that constraint

#### Scenario: Constraint influence logging
- **WHEN** a new proposal is generated
- **THEN** the system logs which constraints influenced the output

### Requirement: Diversity Metrics
The system SHALL track diversity across all generated proposals and flag convergence.

#### Scenario: Key distribution
- **WHEN** diversity metrics are calculated
- **THEN** the entropy of the key distribution across all proposals is reported

#### Scenario: BPM spread
- **WHEN** diversity metrics are calculated
- **THEN** the standard deviation of BPM values across all proposals is reported

#### Scenario: Convergence warning
- **WHEN** key entropy drops below 2.0 bits or BPM standard deviation drops below 10
- **THEN** a warning is emitted recommending stronger constraints

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

### Requirement: Chain Artifact Index
The system SHALL maintain a top-level index of all shrink-wrapped chain artifacts for programmatic access.

#### Scenario: Index generation
- **WHEN** shrink-wrap processes threads
- **THEN** `shrink_wrapped/index.yml` is updated with an entry per shrink-wrapped thread

#### Scenario: Index entry structure
- **WHEN** an entry is added to the index
- **THEN** it contains: directory name, title, bpm, key, concept (truncated), rainbow_color, and timestamp

#### Scenario: Incremental updates
- **WHEN** new threads are shrink-wrapped
- **THEN** the index is appended to, not rebuilt from scratch (existing entries preserved)

