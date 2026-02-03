## ADDED Requirements

### Requirement: Song Proposal Input Mode
The validator SHALL support loading song proposals from chain_artifacts as an alternative input source.

#### Scenario: Load individual song proposal files
- **WHEN** user provides `--proposals-dir /path/to/chain_artifacts/<thread-id>/yml`
- **THEN** the validator SHALL find all `song_proposal_*.yml` files
- **AND** extract the `concept` field from each file for validation

#### Scenario: Load aggregated song proposals
- **WHEN** user provides `--thread-proposals <thread-id>`
- **THEN** the validator SHALL locate `all_song_proposals_<thread-id>.yml`
- **AND** extract each iteration's `concept` field from the `iterations` list

### Requirement: Ground Truth Label Extraction
The validator SHALL extract rainbow color mode labels from song proposals when available.

#### Scenario: Extract labels from rainbow_color field
- **WHEN** a song proposal contains a `rainbow_color` object
- **THEN** the validator SHALL extract `temporal_mode`, `ontological_mode`, and `objectional_mode` values
- **AND** use these as ground truth for comparison with model predictions

#### Scenario: Handle missing labels gracefully
- **WHEN** a song proposal has null or missing mode values in `rainbow_color`
- **THEN** the validator SHALL skip ground truth comparison for that dimension
- **AND** still produce validation output for the concept

### Requirement: Ground Truth Agreement Reporting
The validator SHALL report agreement statistics between predictions and ground truth labels.

#### Scenario: Compute per-dimension accuracy
- **WHEN** validation completes on proposals with ground truth labels
- **THEN** the validator SHALL compute agreement percentage for each dimension (temporal, spatial, ontological)
- **AND** display per-dimension accuracy in the summary

#### Scenario: Include agreement in batch output
- **WHEN** ground truth is available for a concept
- **THEN** the validation result SHALL indicate whether each dimension prediction matches ground truth
- **AND** the JSON output SHALL include a `ground_truth_comparison` field
