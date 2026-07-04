## MODIFIED Requirements

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

## ADDED Requirements

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
