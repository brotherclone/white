## MODIFIED Requirements

### Requirement: Chain Artifact Shrink-Wrap
The system SHALL provide a utility to clean up completed chain artifact threads by removing debug files, renaming directories to human-readable names, cleaning individual file names, and generating structured metadata summaries.

`WhiteAgent.start_workflow()` SHALL call `shrinkwrap()` twice per run:
1. **Pre-run** (existing): at the start, with no `thread_filter`, to pick up any threads from previous runs before loading negative constraints.
2. **Post-run** (new): after `workflow.invoke()` returns, with `thread_filter=<new thread_id>` and `scaffold=True`, so the newly created thread is immediately cleaned, manifested, and its production directories scaffolded into `shrink_wrapped/`. Any exception SHALL be caught and logged as a warning — it MUST NOT propagate or abort the return of `start_workflow()`.

After scaffolding all song productions, `shrinkwrap()` SHALL collect the union of `sounds_like` values across all newly written `manifest_bootstrap.yml` files and call `artist_catalog.generate_missing()` with that list. This call SHALL be wrapped in a try/except — a catalog update failure MUST NOT propagate or abort the shrinkwrap run.

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
