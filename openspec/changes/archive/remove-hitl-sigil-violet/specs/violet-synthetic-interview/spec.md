# Violet Synthetic Interview

## Overview

The Violet agent interview process becomes fully synthetic. All interviews use the LLM + RAG corpus simulation path. The human interview option and HitL probability roll are removed.

## REMOVED Requirements

### Requirement: HitL Probability Roll
The `roll_for_hitl` node and 9% probability check are removed.

#### Scenario: No probability roll for human interview
- Given the Violet agent reaches the interview stage
- When questions have been generated
- Then no random roll determines interview type
- And the workflow proceeds directly to simulated interview

### Requirement: Human Interview Node
The `human_interview` node with CLI prompts is removed.

#### Scenario: No CLI prompts for interview
- Given the Violet agent is conducting an interview
- When the interview begins
- Then no Rich console prompts appear
- And no `Prompt.ask()` calls are made
- And the user is never asked to provide responses

### Requirement: Conditional Interview Routing
The `route_after_roll` conditional edges are removed.

#### Scenario: No conditional routing
- Given the workflow graph is created
- When edges are defined
- Then no conditional edges exist for human vs simulated interview
- And `generate_questions` connects directly to `simulated_interview`

### Requirement: Human Interview State Flag
The `needs_human_interview` field is removed from VioletAgentState.

#### Scenario: No HitL flag in state
- Given a VioletAgentState is created
- When the state is inspected
- Then no `needs_human_interview` field exists
- And no `hitl_probability` attribute exists on VioletAgent

## MODIFIED Requirements

### Requirement: Interview Workflow
The Violet agent SHALL always use the simulated interview path.

#### Scenario: Direct simulated interview
- Given the Violet agent has generated interview questions
- When the interview phase begins
- Then `simulated_interview` node is always invoked
- And the LLM generates responses using RAG corpus
- And the workflow continues to synthesis

### Requirement: Interview Artifact Metadata
Interview artifacts SHALL always indicate synthetic source.

#### Scenario: Artifact indicates synthetic interview
- Given an interview is completed
- When the CircleJerkInterviewArtifact is created
- Then `was_human_interview` is always False
- And the artifact is saved with synthetic responses

## ADDED Requirements

None.

## Related Capabilities

- Violet persona selection (unchanged)
- Violet question generation (unchanged)
- Violet interview synthesis (unchanged)
- Violet counter-proposal generation (unchanged)
