## MODIFIED Requirements

### Requirement: Claude Composition Proposal

The system SHALL call the Claude API with the song proposal and loop inventory to
generate a creative arrangement proposal with explicit compositional rationale.

The prompt SHALL include:
- The White chromatic synthesis framework (color → temporal/spatial/ontological axes)
- Full song metadata: title, concept, mood, color, key, BPM, time_sig, sounds_like, genres
- The loop inventory as a readable table: loop label, instrument, bars, energy score
- An explicit instruction to propose a section-by-section arrangement arc — including
  which loops appear in which order, how many times each repeats, energy shape across
  the song, and transition notes between sections
- An instruction to suggest `sounds_like` reference artists that fit the proposed
  arrangement — Claude MAY affirm the existing sounds_like from the proposal or
  suggest alternatives/additions based on its arrangement choices
- An instruction to return a YAML block for machine parsing and a separate `rationale`
  field in plain prose explaining the compositional reasoning

The model SHALL be `claude-sonnet-4-6`.

`generate_plan()` SHALL accept a `use_claude: bool = True` parameter. When `True`
it calls `propose_arrangement()` after building the mechanical inventory. When the
API is unavailable or `use_claude=False`, it falls back silently to the mechanical
plan with no error.

The old mechanical-only path SHALL be available as `generate_plan_mechanical()`.

The `ProductionPlan` dataclass SHALL carry:
- `rationale: str` — Claude's top-level compositional reasoning
- `proposed_by: str` — `"claude"` when AI-authored, empty string otherwise

The `PlanSection` dataclass SHALL carry:
- `reason: str` — Claude's one-sentence note on each section placement

#### Scenario: Proposal generated successfully

- **WHEN** `python -m app.generators.midi.composition_proposal --production-dir <dir>` is run
- **AND** the loop inventory is non-empty
- **AND** the Claude API is reachable
- **THEN** `composition_proposal.yml` is written to the production directory
- **AND** it contains `proposed_by: claude`, `generated` (ISO timestamp),
  `color_target`, `loop_inventory`, `rationale`, and `proposed_sections`
- **AND** each entry in `proposed_sections` has: `name`, `repeat` (int ≥ 1),
  `energy_note` (string), `transition_note` (string), `loops` (dict mapping
  instrument → loop label)
- **AND** the top-level output contains a `sounds_like` list

#### Scenario: Claude-authored plan via generate_plan

- **WHEN** `generate_plan()` is called with `use_claude=True` and the API is reachable
- **THEN** the returned `ProductionPlan` has a non-empty `rationale` string and
  `proposed_by == "claude"`
- **AND** every `PlanSection` has a non-empty `reason` string

#### Scenario: API unavailable fallback

- **WHEN** `generate_plan()` is called with `use_claude=True` but the API is unreachable
- **THEN** the plan falls back to mechanical generation with no exception raised
- **AND** `proposed_by` is empty

#### Scenario: Mechanical generation still available

- **WHEN** `generate_plan()` is called with `use_claude=False`
- **THEN** a plan is produced without any API call and `proposed_by` is empty

#### Scenario: Claude returns malformed YAML

- **WHEN** Claude's response does not contain a parseable YAML block
- **THEN** the raw response is stored under `rationale` in the output file
- **AND** `proposed_sections` is set to an empty list
- **AND** the command exits with code 0

#### Scenario: API unreachable (CLI)

- **WHEN** the Claude API call raises a connection error during CLI invocation
- **THEN** the command exits with code 1 and a clear error message
- **AND** no `composition_proposal.yml` is written
