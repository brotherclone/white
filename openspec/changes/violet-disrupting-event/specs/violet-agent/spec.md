## ADDED Requirements

### Requirement: Disrupting Event Injection
The Sultan of Solipsism (Violet Agent) SHALL inject a single Lynchian disrupting event into the interview with configurable probability, breaking the interview's frame without preventing synthesis. The probability SHALL default to 0.4 and SHALL be overridable via the `VIOLET_DISRUPTION_PROBABILITY` environment variable. Six disruption types SHALL be available: `STRANGER_ENTERS`, `EQUIPMENT_FAILURE`, `MEMORY_INTRUSION`, `TEMPORAL_BLEED`, `TRANSMISSION_INTERFERENCE`, `IDENTITY_COLLAPSE`.

#### Scenario: Disruption fires at configured probability
- **WHEN** `simulated_interview` completes
- **AND** a random draw is below `VIOLET_DISRUPTION_PROBABILITY` (default 0.4)
- **THEN** `inject_disrupting_event` runs and appends one additional Q&A exchange to `interview_responses`
- **AND** `state.disrupting_event` is set to the selected `DisruptingEventType`

#### Scenario: Disruption does not fire
- **WHEN** the random draw is at or above the probability threshold
- **THEN** `inject_disrupting_event` is skipped and `interview_responses` is unchanged
- **AND** `state.disrupting_event` remains `None`

#### Scenario: Disruption skipped in mock mode
- **WHEN** `MOCK_MODE=true`
- **THEN** `inject_disrupting_event` returns state unchanged without calling the LLM

#### Scenario: Disruption type recorded in artifact
- **WHEN** `synthesize_interview` runs after a disruption fired
- **THEN** `CircleJerkInterviewArtifact.disrupting_event_type` is set to the disruption type string
- **AND** the disruption exchange appears in the interview transcript
