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
The system SHALL incorporate negative constraints when generating new song proposals to
increase output diversity. Each chromatic agent SHALL produce exactly one final
`SongProposalIteration` marked `is_final=True`, which is the proposal surfaced for
human review and production. Internal intermediate iterations (EVP updates, reaction
book revisions, game run counter-proposals) are retained in state and in the
`all_song_proposals` bundle but are NOT written as standalone files.

#### Scenario: Constraint loading at workflow start
- **WHEN** a new chain workflow starts and `shrink_wrapped/index.yml` exists
- **THEN** the constraints are loaded and made available to the White agent

#### Scenario: Constraint influence on proposals
- **WHEN** a new proposal is generated
- **THEN** the White agent MUST NOT produce a proposal matching that constraint

#### Scenario: Constraint influence logging
- **WHEN** a new proposal is generated
- **THEN** the system logs which constraints influenced the output

---

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

### Requirement: Chain Artifact YAML Serialization
Chain artifact `save_file()` implementations that emit YAML SHALL produce output that is
readable by `yaml.safe_load()` without Python-specific tags. Enum fields MUST be serialised
as their string values (e.g. `"yml"`, `"newspaper_article"`), not as
`!!python/object/apply:app.structures.enums.*` decorated objects.

This is achieved by using `model_dump(mode="json")` instead of `model_dump(mode="python")`
when constructing the dict passed to `yaml.dump()`.

#### Scenario: Enum field serialization — clean value
- **WHEN** any YML-emitting artifact calls `save_file()`
- **THEN** enum fields in the output file contain only the enum's string value (e.g. `chain_artifact_type: newspaper_article`)
- **AND** the file contains no `!!python/object` or `!!python/object/apply` tags

#### Scenario: Round-trip safety
- **WHEN** a chain artifact YAML file is read back with `yaml.safe_load()`
- **THEN** it loads successfully without a `yaml.constructor.ConstructorError`

#### Scenario: Value unchanged
- **WHEN** a chain artifact YAML file is written with the fixed serializer
- **THEN** the enum's human-readable value (e.g. `"yml"`, `"symbolic_object"`, `"circular_time"`) is preserved unchanged

### Requirement: Final Proposal Flag
Each `SongProposalIteration` SHALL carry an `is_final` boolean field. Exactly one
iteration per chromatic agent run SHALL be marked `is_final=True` — the last resolved
proposal after all internal creative steps complete. Agents that produce a single
iteration treat it as implicitly final.

#### Scenario: Multi-iteration agent marks final
- **WHEN** Black, Red, or Yellow completes its internal workflow
- **THEN** exactly one `SongProposalIteration` in the run has `is_final=True`

#### Scenario: Single-iteration agent is implicitly final
- **WHEN** an agent produces exactly one proposal and none are marked `is_final`
- **THEN** `save_all_proposals` treats that single iteration as final

### Requirement: Selective Proposal File Output
`save_all_proposals` SHALL write standalone `song_proposal_<Color>_<id>.yml` files
only for iterations where `is_final=True`. All iterations SHALL continue to appear in
the `all_song_proposals_<thread>.yml` bundle for traceability.

#### Scenario: Only final proposals produce standalone files
- **WHEN** a thread has multiple iterations for a color (e.g. Black with EVP update)
- **THEN** only the `is_final=True` iteration is written as
  `song_proposal_Black_<id>.yml`; intermediate iterations appear only in
  `all_song_proposals_<thread>.yml`

#### Scenario: Full traceability preserved
- **WHEN** `all_song_proposals_<thread>.yml` is read
- **THEN** all iterations including non-final ones are present with their
  `is_final` flags

### Requirement: Opt-In HTML Artifact Generation
HTML artifact generation (character sheets, timeline pages, and other fiction rendering) SHALL be opt-in, controlled by a `--with-html` flag on `run_white_agent start`. When the flag is absent, agents that produce HTML SHALL skip that generation step entirely. HTML generation SHALL NOT run by default as it adds LLM and image generation cost with no current consumer. The capability is preserved for future UI integration (v2: candidate browser renders song-related fiction alongside MIDI review).

#### Scenario: HTML skipped by default
- **WHEN** `run_white_agent start` is invoked without `--with-html`
- **THEN** no HTML files are written to `chain_artifacts` and no related LLM or
  image generation calls are made

#### Scenario: HTML generated on request
- **WHEN** `run_white_agent start --with-html` is invoked
- **THEN** HTML artifacts (character sheets, timeline pages, etc.) are generated
  and written to `chain_artifacts/<thread>/html/` as before

#### Scenario: Shrinkwrap handles missing html directory
- **WHEN** a thread has no `html/` directory (because `--with-html` was not used)
- **THEN** shrinkwrap completes successfully and omits the `html/` directory from
  the output without error

