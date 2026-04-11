# composition-proposal Specification

## Purpose
TBD - created by archiving change add-claude-composition-proposal. Update Purpose after archive.
## Requirements
### Requirement: Loop Inventory Collection

Before generating a composition proposal, the system SHALL collect the approved loop
inventory across all pipeline phases for the song.

The inventory SHALL be derived from the approved review files (`chords/review.yml`,
`drums/review.yml`, `bass/review.yml`, `melody/review.yml`) and contain, per approved
loop: label, instrument, bar count, composite score, and any human notes from the
review file. Loops not yet approved (status ≠ `approved`) SHALL be excluded.

#### Scenario: Full inventory — all four instruments approved

- **WHEN** `build_loop_inventory(production_dir)` is called
- **AND** approved loops exist for chords, drums, bass, and melody
- **THEN** a dict is returned with keys `chords`, `drums`, `bass`, `melody`
- **AND** each value is a list of loop dicts with `label`, `bars`, `score`, `notes`

#### Scenario: Partial inventory — some instruments not yet approved

- **WHEN** approved loops exist for only some instruments
- **THEN** the inventory dict contains only those instruments
- **AND** a warning is logged listing missing instruments
- **AND** the proposal proceeds with whatever is available

#### Scenario: No approved loops at all

- **WHEN** no approved loops exist across any instrument
- **THEN** the command exits with an error: `"No approved loops found — run pipeline phases before generating a composition proposal"`

---

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
API is unavailable or `use_claude=False`, it SHALL fall back to the mechanical
plan without failing; it MAY print a warning to stdout but MUST NOT raise.

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
- **AND** each entry in `proposed_sections` has: `name`, `play_count` (int ≥ 1;
  `repeat` accepted as a legacy alias), `energy_note` (string), `transition_note`
  (string), `loops` (dict mapping instrument → loop label)
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

### Requirement: Composition Proposal Drift

When a `composition_proposal.yml` exists, the assembly manifest importer SHALL
compute a proposal drift block showing how the human's Logic arrangement diverged
from Claude's proposal.

Proposal drift is computed at the section level by comparing `proposed_sections`
in the proposal against the sections derived from `arrangement.txt`. The drift block
SHALL include:
- `sections_added` — section names present in the arrangement but absent from the proposal
- `sections_removed` — section names in the proposal but absent from the arrangement
- `repeat_deltas` — list of `{name, proposed, actual}` dicts where repeat count changed
- `order_changed` — bool, true if the relative ordering of shared sections differs

#### Scenario: Proposal drift computed on import

- **WHEN** `python -m app.generators.midi.assembly_manifest --import-arrangement ...` is run
- **AND** `composition_proposal.yml` exists in the production directory
- **THEN** the drift report includes a `proposal_drift` block with the four fields above

#### Scenario: No proposal — drift skipped

- **WHEN** `composition_proposal.yml` does not exist
- **THEN** the drift report is written without a `proposal_drift` block
- **AND** no warning is emitted

### Requirement: Composition Narrative Generation
`composition_proposal.py` SHALL expose a `generate_narrative(production_dir)` function
that calls the Claude API and writes `composition_narrative.yml` to the production
directory. The narrative describes each section along four controlled-vocabulary
dimensions: register, texture, harmonic_complexity, and rhythm_character. It also
designates `the_moment` (the emotional peak section) and provides free-text narrative
for each section.

#### Scenario: Narrative written with controlled vocabulary values
- **WHEN** `generate_narrative()` is called for a production directory with a
  production plan and song context
- **THEN** `composition_narrative.yml` is written with section entries containing
  valid values for register, texture, harmonic_complexity, and rhythm_character
- **AND** each section entry contains a `narrative` free-text field

#### Scenario: ACE Studio unreachable does not block generation
- **WHEN** the Claude API is unavailable
- **THEN** a warning is logged and None is returned (the narrative file is not written)

---

### Requirement: Narrative Constraints Module
`app/structures/music/narrative_constraints.py` SHALL provide a `NarrativeSection`
Pydantic model describing per-section generation constraints using controlled
vocabularies, and a `CompositionNarrative` top-level model holding the full document.

#### Scenario: Section constraints load from YAML
- **WHEN** a section entry is parsed from `composition_narrative.yml`
- **THEN** invalid vocabulary values raise a Pydantic validation error
- **AND** missing optional fields default to None

#### Scenario: Constraint extraction returns generation hints
- **WHEN** `extract_constraints(section_name, narrative)` is called
- **THEN** a dict of generation hints is returned (e.g. rhythm_character → target tag list)

