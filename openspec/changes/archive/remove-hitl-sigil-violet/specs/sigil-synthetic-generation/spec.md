# Sigil Synthetic Generation

## Overview

Sigil generation in the Black agent becomes fully synthetic with no Human-in-the-Loop pausing. Sigils are generated, saved as artifacts, and the workflow continues without interruption.

## REMOVED Requirements

### Requirement: Workflow Pause for Sigil Charging
The workflow no longer pauses at `await_human_action` node when a sigil is generated.

#### Scenario: Sigil generated without workflow pause
- Given the Black agent generates a sigil artifact
- When the sigil is saved to disk
- Then the workflow continues to END without pausing
- And no checkpoint state is persisted for resume

### Requirement: Todoist Task Creation
No Todoist tasks are created for sigil charging rituals.

#### Scenario: No Todoist integration
- Given a sigil artifact is generated
- When the sigil generation completes
- Then no HTTP request is made to Todoist API
- And no task_id or task_url is stored in state

### Requirement: Human Instructions for Charging
No human instructions are generated for sigil charging.

#### Scenario: No charging instructions in state
- Given a sigil is generated
- When the workflow completes
- Then `human_instructions` field does not exist in BlackAgentState
- And `pending_human_tasks` field does not exist in BlackAgentState

### Requirement: Proposal Update After Charging
The `update_alternate_song_spec_with_sigil` node is removed.

#### Scenario: No post-charging proposal revision
- Given a sigil artifact is created
- When the Black agent workflow completes
- Then the counter_proposal is not modified based on sigil
- And the sigil exists only as a standalone artifact

## MODIFIED Requirements

### Requirement: Sigil Generation Flow
Sigil generation SHALL be fire-and-forget with no workflow interruption.

#### Scenario: Simplified sigil generation
- Given the Black agent reaches the sigil generation step
- And the random chance allows sigil creation (~25%)
- When the sigil is generated
- Then the sigil artifact is saved to disk
- And the workflow proceeds directly to END
- And no pause or interrupt occurs

### Requirement: Sigil Artifact Persistence
Sigil artifacts SHALL be saved to disk without tracking for human action.

#### Scenario: Sigil saved as artifact only
- Given a sigil is generated with wish, intent, and glyph
- When `sigil_artifact.save_file()` is called
- Then the YAML file is written to `chain_artifacts/<thread_id>/yml/`
- And the artifact is appended to `state.artifacts`
- And no state fields track pending human tasks

## ADDED Requirements

None.

## Related Capabilities

- Black Agent EVP generation (unchanged)
- Black Agent counter-proposal generation (unchanged)
