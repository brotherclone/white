## ADDED Requirements

### Requirement: Compare Plan Against Arrangement

The system SHALL compare a `production_plan.yml` against an `arrangement.txt` export and
produce a `DriftReport` capturing section-level differences between Claude's proposed
arrangement and the human's actual Logic arrangement.

The report SHALL include:
- `proposed_sections` — ordered list of section names as Claude proposed them (expanded
  by `play_count`, so a section with `play_count: 2` appears twice)
- `actual_sections` — ordered list of section names as they appear in `arrangement.txt`
  (one entry per clip instance on track 1)
- `drift.removed` — labels present in the proposed set but absent from the actual
- `drift.added` — labels present in the actual but absent from the proposed set
- `drift.reordered` — `true` if the first-occurrence order of shared labels differs
  between proposed and actual
- `bar_deltas` — per unique label: `{proposed, actual, delta}` total bar counts summed
  across all instances (only for labels present in both proposed and actual)
- `energy_arc_correlation` — Pearson r between proposed and actual arc sequences,
  normalised to 100 sample points; `null` when either sequence has fewer than 2 entries
- `summary` — Claude-generated one-paragraph prose summary of the drift

#### Scenario: Section removed

- **WHEN** `production_plan.yml` proposes an `intro` section but no intro clips appear in `arrangement.txt`
- **THEN** `drift.removed` contains `"intro"`
- **AND** `bar_deltas` has no entry for `"intro"`

#### Scenario: Section added

- **WHEN** `arrangement.txt` contains an `interlude` clip not proposed in `production_plan.yml`
- **THEN** `drift.added` contains `"interlude"`
- **AND** `bar_deltas` has no entry for `"interlude"` (only shared labels get deltas)

#### Scenario: Section reordered

- **WHEN** Claude proposed `[verse, chorus, bridge]` but the actual arrangement is `[verse, bridge, chorus]`
- **THEN** `drift.reordered` is `true`

#### Scenario: Same order preserved

- **WHEN** the first-occurrence order of shared section labels is identical in proposed and actual
- **THEN** `drift.reordered` is `false`

#### Scenario: Bar count delta

- **WHEN** Claude proposed 16 total bars of `chorus` (8 bars × play_count 2) but the
  actual arrangement has 8 bars of `chorus` (one 8-bar instance)
- **THEN** `bar_deltas.chorus = {proposed: 16, actual: 8, delta: -8}`

#### Scenario: Zero bar delta

- **WHEN** proposed and actual bar totals match for a label
- **THEN** `bar_deltas[label].delta` is `0`

---

### Requirement: Energy Arc Correlation

The system SHALL compute a Pearson correlation coefficient between the proposed and actual
energy arc trajectories.

The proposed arc trajectory is built by expanding plan sections in proposed order (by
`play_count`) and reading their `arc` field. The actual arc trajectory is built from the
actual section sequence, using each section's `arc` value from the plan if present, else
inferring via `_infer_arc_from_label`. Both sequences are normalised to 100 sample points
via linear interpolation before computing Pearson r.

#### Scenario: Arc correlation computed

- **WHEN** both proposed and actual arc sequences have at least 2 entries
- **THEN** `energy_arc_correlation` is a float in `[-1.0, 1.0]`

#### Scenario: Constant arc — correlation undefined

- **WHEN** all arc values in either sequence are identical (zero variance)
- **THEN** `energy_arc_correlation` is `null`

#### Scenario: Insufficient data

- **WHEN** either the proposed or actual sequence has fewer than 2 section instances
- **THEN** `energy_arc_correlation` is `null`

---

### Requirement: Claude Drift Summary

The system SHALL call Claude to generate a one-paragraph prose summary of the drift data,
interpreting what the human changed and what that implies about Claude's compositional
judgement.

#### Scenario: Summary generated

- **WHEN** the Anthropic API is reachable
- **THEN** `summary` is a non-empty string of prose

#### Scenario: API unavailable

- **WHEN** the Anthropic API raises an exception
- **THEN** `summary` is set to an empty string
- **AND** `plan_drift_report.yml` is still written with all other fields populated

---

### Requirement: Plan Drift Report CLI

The system SHALL expose a CLI:

```
python -m white_composition.drift_report --production-dir <dir>
```

that reads `production_plan.yml` (and optionally `--arrangement <file>`, defaulting to
`<production-dir>/arrangement.txt`), computes the drift report, writes
`plan_drift_report.yml` to the production directory, and prints a human-readable summary.

#### Scenario: Report written

- **WHEN** both `production_plan.yml` and `arrangement.txt` exist in the production directory
- **THEN** `plan_drift_report.yml` is written to the production directory
- **AND** the CLI prints a summary showing proposed vs actual section counts, removed/added
  sections, and the energy arc correlation score

#### Scenario: Missing production plan

- **WHEN** `production_plan.yml` does not exist in the production directory
- **THEN** an error is printed to stdout and the command exits with code 1
- **AND** no `plan_drift_report.yml` is written

#### Scenario: Missing arrangement file

- **WHEN** `arrangement.txt` does not exist (and `--arrangement` is not provided)
- **THEN** an error is printed to stdout and the command exits with code 1
- **AND** no `plan_drift_report.yml` is written

#### Scenario: Custom arrangement path

- **WHEN** `--arrangement /path/to/custom.txt` is provided
- **THEN** that file is used instead of `<production-dir>/arrangement.txt`

#### Scenario: Skip summary flag

- **WHEN** `--no-claude` is passed
- **THEN** the Claude API is not called and `summary` is set to an empty string
- **AND** `plan_drift_report.yml` is still written with all other fields

---

### Requirement: Plan Drift Report I/O

The system SHALL provide `write_report()` and `load_report()` functions for persisting and
loading `DriftReport` objects as YAML.

#### Scenario: Round-trip through YAML

- **WHEN** a `DriftReport` is written with `write_report()` and reloaded with `load_report()`
- **THEN** all fields are preserved with correct types

#### Scenario: No report present

- **WHEN** `load_report()` is called and `plan_drift_report.yml` does not exist
- **THEN** `None` is returned
