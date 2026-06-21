## REMOVED Requirements

### Requirement: Production Decisions File
**Reason**: `production_decisions.yml` was generated for exactly one song (a violet
fallback test thread) and never used for any real production. The file was intended
as a structured ML training record, but the training pipeline is complete and the
feature was never integrated into any downstream system. The `"decisions"` phase in
`pipeline_runner.py`, the `has_decisions` field in `scan_songs()`, the green checkmark
on song cards, the "Handoff to Logic" button on song cards (the old pre-board handoff
path, now superseded by `/board`), and the "Run decisions" button on `/candidates`
are all vestigial and SHALL be removed.

**Migration**: The one existing `production_decisions.yml` file in
`shrink_wrapped/violet-fallback-defensive-violet-response/` is left in place as an
inert data file; no reader depends on it after this removal.

#### Scenario: production_decisions.yml generated
- **WHEN** `production_decisions.py` is run on a directory with all phases completed
- **THEN** `production_decisions.yml` is written containing all sections

#### Scenario: Partial data available
- **WHEN** some artifacts (mix_score.yml, drift_report.yml) are missing
- **THEN** `production_decisions.yml` is still written with available data

#### Scenario: Null section for unavailable data
- **WHEN** a section's source artifact does not exist
- **THEN** the corresponding sections are emitted as `null` in `production_decisions.yml`

#### Scenario: Pipeline status reflects decisions
- **WHEN** `pipeline_runner status` is run
- **THEN** the output indicates whether `production_decisions.yml` exists
