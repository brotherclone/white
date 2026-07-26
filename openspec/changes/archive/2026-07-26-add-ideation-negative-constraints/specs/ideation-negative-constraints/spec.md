## ADDED Requirements

### Requirement: Negative Constraints Reach Every Color Agent
The system SHALL make the run's `negative_constraints` text (already loaded once
by `white_agent.py` from `negative_constraints.yml` via `generate_negative_constraints.format_for_prompt`)
available to every color agent's own state, not only `MainAgentState`.

#### Scenario: White constructs a color agent's sub-state
- **WHEN** `WhiteAgent.__call__` (or the equivalent per-agent invocation path)
  constructs a color agent's own state (e.g. `RedAgentState`, `OrangeAgentState`)
- **THEN** the constructed state includes `negative_constraints` copied from
  `MainAgentState.negative_constraints`

#### Scenario: No constraints loaded yet
- **WHEN** `MainAgentState.negative_constraints` is empty (no prior threads, or
  `negative_constraints.yml` absent)
- **THEN** each color agent's state receives an empty `negative_constraints`
  value and proceeds without an avoidance block, matching current behavior

### Requirement: Color Agent Counter-Proposals Honor Negative Constraints
Each color agent's `generate_alternate_song_spec` SHALL append its state's
`negative_constraints` text to the counter-proposal prompt when non-empty,
using the same append-after-prompt convention `white_agent.py` already uses
(`prompt + "\n\n" + state.negative_constraints`).

#### Scenario: Constraints present
- **WHEN** a color agent's `generate_alternate_song_spec` runs and
  `state.negative_constraints` is a non-empty string
- **THEN** the LLM prompt built for that agent's `SongProposalIteration` call
  includes the constraints text appended after the main prompt body

#### Scenario: Constraints absent
- **WHEN** `state.negative_constraints` is empty
- **THEN** the prompt is built exactly as it is today, with no appended block
  and no error or warning
