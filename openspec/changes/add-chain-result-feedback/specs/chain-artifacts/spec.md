## ADDED Requirements

### Requirement: Negative Constraint Generation
The system SHALL generate a negative constraints file from prior chain results to prevent the pipeline from converging on repeated outputs.

#### Scenario: Constraint file generation
- **WHEN** the constraint generator is run after shrink-wrapping
- **THEN** `chain_artifacts/negative_constraints.yml` is created from `chain_artifacts/index.yml`

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
- **WHEN** a new chain workflow starts and `negative_constraints.yml` exists
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
