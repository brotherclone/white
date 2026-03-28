## MODIFIED Requirements

### Requirement: Production Plan Schema
The `ProductionPlan` dataclass SHALL include the following fields:
- `sections: list[PlanSection]` — ordered section entries
- `rationale: str` — top-level compositional reasoning (empty string for mechanical plans)
- `proposed_by: str` — `"claude"` or `"mechanical"`

The `PlanSection` dataclass SHALL include:
- `label: str`, `bars: int`, `play_count: int`, `energy: str`
- `reason: str` — one-sentence note on placement (empty string for mechanical plans)

All fields SHALL survive a YAML save/load round-trip with no data loss.

#### Scenario: Round-trip preserves rationale and reasons
- **WHEN** a Claude-authored plan is saved to YAML and reloaded
- **THEN** `rationale`, `proposed_by`, and all per-section `reason` fields are identical
  to the original

#### Scenario: Refresh preserves human edits
- **WHEN** `refresh_plan()` is called on a plan where the user has manually edited
  `play_count`, `reason`, or section order
- **THEN** those edits are preserved and only `bars` is updated from the loop inventory
