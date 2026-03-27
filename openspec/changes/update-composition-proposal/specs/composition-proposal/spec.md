## MODIFIED Requirements

### Requirement: Arrangement Arc Generation
The production plan generator SHALL call the Claude API with the full song proposal
(concept, mood, color, singer, key) and the approved loop inventory (sections, bar
lengths, energy scores) to produce a genuine arrangement arc.

The response SHALL be parsed into a `ProductionPlan` containing:
- An ordered list of `PlanSection` entries reflecting Claude's proposed structure
  (including repeats, dynamic arc, and section ordering)
- A `rationale` field with Claude's top-level compositional reasoning
- A `proposed_by: "claude"` marker
- A per-section `reason` field with a one-sentence note on each placement decision

#### Scenario: Claude-authored plan contains rationale
- **WHEN** `generate_plan()` is called with `use_claude=True` and the API is reachable
- **THEN** the returned `ProductionPlan` has a non-empty `rationale` string and
  `proposed_by == "claude"`

#### Scenario: Per-section reasons present
- **WHEN** a Claude-authored plan is generated
- **THEN** every `PlanSection` has a non-empty `reason` string

#### Scenario: API unavailable fallback
- **WHEN** `generate_plan()` is called with `use_claude=True` but the API is unreachable
- **THEN** the plan falls back to mechanical generation and `proposed_by == "mechanical"`

#### Scenario: Mechanical generation still available
- **WHEN** `generate_plan()` is called with `use_claude=False`
- **THEN** a plan is produced without any API call and `proposed_by == "mechanical"`
