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

`WhiteAgent.start_workflow()` SHALL call `shrinkwrap()` twice per run:
1. **Pre-run** (existing): at the start, with no `thread_filter`, to pick up any threads from previous runs before loading negative constraints.
2. **Post-run** (new): after `workflow.invoke()` returns, with `thread_filter=<new thread_id>` and `scaffold=True`, so the newly created thread is immediately cleaned, manifested, and its production directories scaffolded into `shrink_wrapped/`. Any exception SHALL be caught and logged as a warning — it MUST NOT propagate or abort the return of `start_workflow()`.

After scaffolding all song productions, `shrinkwrap()` SHALL collect the union of `sounds_like` values across all newly written `manifest_bootstrap.yml` files and call `artist_catalog.generate_missing()` with that list. This call SHALL be wrapped in a try/except — a catalog update failure MUST NOT propagate or abort the shrinkwrap run.

All manifest files written by the shrinkwrap utility (`manifest.yml`, `manifest_bootstrap.yml`)
SHALL include `schema_version: "2.0.0"` as their first field. The `SHRINKWRAP_SCHEMA_VERSION`
constant in `shrinkwrap_chain_artifacts.py` provides this value to all writers.

#### Scenario: Thread discovery
- **WHEN** the shrinkwrap utility is pointed at a `chain_artifacts/` directory
- **THEN** it SHALL discover all UUID-named subdirectories
- **AND** process each as a separate thread

#### Scenario: Post-run shrinkwrap scaffolds new thread
- **WHEN** `start_workflow()` completes successfully
- **THEN** `shrinkwrap()` is called with `thread_filter=<new thread_id>` and `scaffold=True`
- **AND** the new thread's output directory is created under `shrink_wrapped/`
- **AND** `manifest_bootstrap.yml` is written for each song proposal found in the thread's `yml/` directory
- **AND** if shrinkwrap raises, `start_workflow()` logs a warning and returns normally

#### Scenario: Post-run shrinkwrap failure is non-fatal
- **WHEN** the post-run `shrinkwrap()` call raises any exception
- **THEN** `start_workflow()` logs a warning and returns the final agent state unchanged
- **AND** no exception is propagated to the caller

#### Scenario: manifest_bootstrap.yml includes sounds_like
- **WHEN** `scaffold_song_productions()` writes a `manifest_bootstrap.yml` for a song proposal
- **THEN** the file includes a `sounds_like` list extracted from the proposal YML
- **AND** if the proposal YML has no `sounds_like` field, the manifest contains `sounds_like: []`

#### Scenario: Artist catalog updated after scaffolding
- **WHEN** shrinkwrap finishes scaffolding one or more productions
- **THEN** `artist_catalog.generate_missing()` is called with the union of all `sounds_like` values from the newly scaffolded `manifest_bootstrap.yml` files
- **AND** any artists not yet in the catalog receive generated descriptions with `status: draft`

#### Scenario: Catalog update failure is non-fatal
- **WHEN** `artist_catalog.generate_missing()` raises any exception during a shrinkwrap run
- **THEN** shrinkwrap logs a warning and completes normally
- **AND** the scaffolded files are unaffected

#### Scenario: schema_version written to new manifests
- **WHEN** `write_manifest()` writes a `manifest.yml` or `scaffold_song_productions()` writes a `manifest_bootstrap.yml`
- **THEN** the file's first field is `schema_version: "2.0.0"`

---

### Requirement: Chain Artifact Index
The system SHALL maintain a top-level index of all shrink-wrapped chain artifacts for programmatic access.

The `index.yml` file SHALL include `schema_version: "2.0.0"` as its first field.

#### Scenario: Index generation
- **WHEN** shrink-wrap processes threads
- **THEN** `shrink_wrapped/index.yml` is updated with an entry per shrink-wrapped thread

#### Scenario: Index entry structure
- **WHEN** an entry is added to the index
- **THEN** it contains: directory name, title, bpm, key, concept (truncated), rainbow_color, and timestamp

#### Scenario: Incremental updates
- **WHEN** new threads are shrink-wrapped
- **THEN** the index is appended to, not rebuilt from scratch (existing entries preserved)

#### Scenario: schema_version written to index
- **WHEN** `write_index()` writes `index.yml`
- **THEN** the file's first field is `schema_version: "2.0.0"`

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

### Requirement: Shrinkwrap Manifest Migration
The shrinkwrap utility SHALL provide a `--migrate` CLI flag that backfills
`schema_version: "2.0.0"` onto all existing manifest files in a `shrink_wrapped/`
output directory and re-scaffolds any thread production directories that are missing
`manifest_bootstrap.yml` files.

The migration SHALL be idempotent: files that already have `schema_version: "2.0.0"`
are not rewritten. `song_context.yml` files with `schema_version: "1"` SHALL be
updated to `"2.0.0"`. A `--dry-run` flag SHALL print all planned changes without
writing any files.

#### Scenario: Migrate adds schema_version to legacy manifest.yml
- **WHEN** `--migrate` runs and finds a `manifest.yml` with no `schema_version` field
- **THEN** `schema_version: "2.0.0"` is prepended as the first field
- **AND** all other fields are preserved unchanged

#### Scenario: Migrate adds schema_version to legacy manifest_bootstrap.yml
- **WHEN** `--migrate` runs and finds a `manifest_bootstrap.yml` with no `schema_version` field
- **THEN** `schema_version: "2.0.0"` is prepended as the first field
- **AND** all other fields are preserved unchanged

#### Scenario: Migrate updates song_context.yml from v1 to v2
- **WHEN** `--migrate` runs and finds a `song_context.yml` with `schema_version: "1"`
- **THEN** `schema_version` is updated to `"2.0.0"`

#### Scenario: Migration is idempotent
- **WHEN** `--migrate` runs on a directory where all manifests already have `schema_version: "2.0.0"`
- **THEN** no files are rewritten

#### Scenario: Dry-run mode
- **WHEN** `--migrate --dry-run` is run
- **THEN** a list of all files that would be modified is printed
- **AND** no files are changed on disk

#### Scenario: Stub thread manifest for pre-scaffold thread
- **WHEN** `--migrate` finds a thread directory that has a `production/` subdirectory but no `manifest.yml` at its root
- **THEN** a stub `manifest.yml` is written with `schema_version: "2.0.0"`, `stub: true`, `title` derived by humanizing the directory name, and `null` for `bpm`, `key`, `concept`, `mood`, `genres`, `agent_name`, and `thread_id`

#### Scenario: Scaffold bootstraps from source YML when available
- **WHEN** `--migrate` finds a production sub-directory missing `manifest_bootstrap.yml`
- **AND** the thread has a `yml/` directory with matching proposal files
- **THEN** `scaffold_song_productions()` is called to create the bootstrap with full metadata and `schema_version: "2.0.0"`

#### Scenario: Stub bootstrap from production slug when no source YML exists
- **WHEN** `--migrate` finds a production sub-directory missing `manifest_bootstrap.yml`
- **AND** no `yml/` directory exists for the thread
- **THEN** a stub `manifest_bootstrap.yml` is written with `schema_version: "2.0.0"`, `stub: true`, `rainbow_color` and `title` parsed from the production slug's double-underscore color-prefix convention (`{color}__{title_slug}_v{n}`), and `null` for `bpm`, `key`, and `singer`

#### Scenario: Stub bootstrap slug fallback
- **WHEN** the production slug does not contain a double-underscore color prefix
- **THEN** the entire slug (with `_vN` suffix stripped) is humanized as the title and `rainbow_color` is `null`

### Requirement: Manifest Bootstrap Schema
The `manifest_bootstrap.yml` file SHALL contain the following fields:
`schema_version`, `stub`, `title`, `rainbow_color`, `bpm`, `key`, `singer`.
Optional fields SHALL include `suite`, `suite_part`, `suite_logic_path`, `sounds_like`,
`time_sig`, and the new lifecycle fields described below.

The `manifest_bootstrap.yml` file MAY contain a `lifecycle_status` field whose value is
one of `"merged"`, `"abandoned"`, or `"scrapped"`. When absent or `null`, the song is
considered active.

When `lifecycle_status` is `"merged"`, the file MAY also contain a `merged_with` field
holding a list of song IDs (strings in `{thread_slug}__{production_slug}` format) that
this song was merged into or from.

The `manifest_bootstrap.yml` file MAY contain a `uses_parts_from` field holding a list
of song IDs (same format) referring to scrapped songs whose material was reused. This
field is independent of `lifecycle_status` and may appear on any active song.

Writing or updating `manifest_bootstrap.yml` via `_synthesize_bootstrap_stub()` SHALL NOT
write `lifecycle_status`, `merged_with`, or `uses_parts_from` (they default absent and
are set only by lifecycle API actions).

#### Scenario: Active song manifest has no lifecycle field
- **WHEN** `manifest_bootstrap.yml` is written by `_synthesize_bootstrap_stub()`
- **THEN** no `lifecycle_status`, `merged_with`, or `uses_parts_from` keys are present

#### Scenario: Merged song manifest
- **WHEN** a merge action completes for songs A and B
- **THEN** Song A's `manifest_bootstrap.yml` contains `lifecycle_status: merged` and
  `merged_with: [<song_B_id>]`
- **AND** Song B's `manifest_bootstrap.yml` contains `lifecycle_status: merged` and
  `merged_with: [<song_A_id>]`

#### Scenario: Abandoned song manifest
- **WHEN** an abandon action completes for song A
- **THEN** Song A's `manifest_bootstrap.yml` contains `lifecycle_status: abandoned`

#### Scenario: Scrapped song manifest
- **WHEN** a scrap action completes for song A
- **THEN** Song A's `manifest_bootstrap.yml` contains `lifecycle_status: scrapped`

#### Scenario: Uses parts from
- **WHEN** a song's "Uses parts from" list is saved with scrapped song IDs
- **THEN** the active song's `manifest_bootstrap.yml` contains
  `uses_parts_from: [<scrapped_song_id>, ...]`
- **AND** the scrapped song's `manifest_bootstrap.yml` is NOT modified

