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

