## MODIFIED Requirements

### Requirement: Chain Artifact Shrink-Wrap
The system SHALL provide a utility to clean up completed chain artifact threads by removing debug files, renaming directories to human-readable names, cleaning individual file names, and generating structured metadata summaries.

`WhiteAgent.start_workflow()` SHALL call `shrinkwrap()` twice per run:
1. **Pre-run** (existing): at the start, with no `thread_filter`, to pick up any threads from previous runs before loading negative constraints.
2. **Post-run** (new): after `workflow.invoke()` returns, with `thread_filter=<new thread_id>` and `scaffold=True`, so the newly created thread is immediately cleaned, manifested, and its production directories scaffolded into `shrink_wrapped/`. Any exception SHALL be caught and logged as a warning — it MUST NOT propagate or abort the return of `start_workflow()`.

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
