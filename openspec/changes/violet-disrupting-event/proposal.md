# Change: Disrupting event injection in Sultan of Solipsism interview

## Why

Every Sultan of Solipsism interview follows the same clean arc: select persona →
generate questions → simulate responses → synthesize. The results are good but
predictable. Real Lynchian weirdness comes from the intrusion of something that
doesn't belong — a stranger, a static burst, a memory from the wrong timeline.
The interview should feel like it could go sideways at any moment.

Adding a random disrupting event (probability 40% per run) injects a single surreal
exchange that breaks the interview's frame without derailing the synthesis. Over
multiple songs the effect is cumulative: sometimes the interview is clean, sometimes
something strange happened and the artifact records it.

## What Changes

- A `DisruptingEventType` enum is added with six Lynchian disruption types:
  `STRANGER_ENTERS`, `EQUIPMENT_FAILURE`, `MEMORY_INTRUSION`,
  `TEMPORAL_BLEED`, `TRANSMISSION_INTERFERENCE`, `IDENTITY_COLLAPSE`
- A `disrupting_event` optional field is added to `VioletAgentState`
- A new `inject_disrupting_event` node is added to the LangGraph workflow,
  running conditionally after `simulated_interview` (before `synthesize_interview`)
- With probability 0.4, the node calls the LLM to generate a short disruptive exchange
  (interviewer line + Gabe response) in the selected disruption style, and appends it
  to `state.interview_responses` along with a sentinel question
- `synthesize_interview` passes the disruption through to the artifact;
  `CircleJerkInterviewArtifact` records `disrupting_event_type` if present
- Probability is overridable via env var `VIOLET_DISRUPTION_PROBABILITY`

## Impact

- Affected code: `app/agents/violet_agent.py`, `app/agents/states/violet_agent_state.py`,
  `app/structures/artifacts/circle_jerk_interview_artifact.py`,
  `app/structures/enums/` (new `disrupting_event_type.py`)
- Affected spec: `violet-agent`
- No impact on downstream agents — the interview artifact is read-only by Prism
- Mock mode: disruption is skipped when `MOCK_MODE=true`
