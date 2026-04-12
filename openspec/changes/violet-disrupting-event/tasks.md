## 1. Enum

- [ ] 1.1 Create `app/structures/enums/disrupting_event_type.py` with
      `DisruptingEventType(str, Enum)` containing:
      `STRANGER_ENTERS`, `EQUIPMENT_FAILURE`, `MEMORY_INTRUSION`,
      `TEMPORAL_BLEED`, `TRANSMISSION_INTERFERENCE`, `IDENTITY_COLLAPSE`

## 2. State

- [ ] 2.1 Add `disrupting_event: Optional[DisruptingEventType] = None` to
      `VioletAgentState` in `app/agents/states/violet_agent_state.py`

## 3. Agent graph

- [ ] 3.1 Add `inject_disrupting_event` node to the LangGraph workflow in
      `app/agents/violet_agent.py` — placed after `simulated_interview`,
      before `synthesize_interview`, on a conditional edge:
      - Probability 0.4 (read from `VIOLET_DISRUPTION_PROBABILITY` env var, float)
      - If triggered: go to `inject_disrupting_event` → `synthesize_interview`
      - If not: go directly to `synthesize_interview`
- [ ] 3.2 Implement `inject_disrupting_event` method:
      - Randomly select a `DisruptingEventType`
      - Set `state.disrupting_event = event_type`
      - Call LLM to generate a short disruptive exchange (one interviewer line +
        one Gabe response) in the style of the selected type
      - Append a sentinel `VanityInterviewQuestion` (number = 99, marked as disruption)
        and corresponding `VanityInterviewResponse` to state
      - Skip entirely when `MOCK_MODE=true`

## 4. Artifact

- [ ] 4.1 Add `disrupting_event_type: Optional[str] = None` field to
      `CircleJerkInterviewArtifact`
- [ ] 4.2 In `synthesize_interview`, pass `state.disrupting_event` to the artifact
      so it appears in the saved YML and the rendered output

## 5. Tests

- [ ] 5.1 Unit test: `inject_disrupting_event` with probability forced to 1.0 —
      verify a disruption exchange is appended to responses and event type set in state
- [ ] 5.2 Unit test: probability forced to 0.0 — verify responses unchanged
- [ ] 5.3 Confirm existing Violet agent tests pass with the new conditional node
