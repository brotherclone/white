## ADDED Requirements

### Requirement: Chain Artifact Shrink-Wrap
The system SHALL provide a utility to clean up completed chain artifact threads by removing debug files, renaming directories to human-readable names, and generating structured metadata summaries.

#### Scenario: Thread discovery
- **WHEN** the shrink-wrap utility scans `chain_artifacts/`
- **THEN** it identifies all UUID-named thread directories containing `all_song_proposals_*.yml`

#### Scenario: Directory renaming
- **WHEN** a thread is shrink-wrapped
- **THEN** the directory is renamed from UUID to `{color}-{slugified-title}` (e.g., `black-the-phantom-limb-protocol`)

#### Scenario: Debug artifact removal
- **WHEN** a thread is shrink-wrapped with default settings
- **THEN** intermediate files (rebracketing analyses, transformation traces, facet evolution) are deleted

#### Scenario: Debug artifact archival
- **WHEN** shrink-wrap is run with `--archive` flag
- **THEN** debug artifacts are moved to a `.debug/` subdirectory instead of deleted

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
- **THEN** `shrinkwrapped/index.yml` is updated with an entry per shrink-wrapped thread

#### Scenario: Index entry structure
- **WHEN** an entry is added to the index
- **THEN** it contains: directory name, title, bpm, key, concept (truncated), rainbow_color, and timestamp

#### Scenario: Incremental updates
- **WHEN** new threads are shrink-wrapped
- **THEN** the index is appended to, not rebuilt from scratch (existing entries preserved)
