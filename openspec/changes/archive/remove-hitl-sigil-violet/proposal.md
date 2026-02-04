# Remove HitL from Sigil and Violet Agents

## Summary

Remove Human-in-the-Loop (HitL) functionality from both the Sigil artifact generation (Black agent) and the Violet agent interview process. Both should operate fully synthetically to enable unattended batch generation without overnight workflow pauses.

## Motivation

The current implementation includes HitL pausing mechanisms that interrupt batch generation runs:

1. **Sigil (Black Agent)**: When a sigil is generated (25% chance), the workflow creates a Todoist task for manual "charging" and pauses via LangGraph's `interrupt_before` mechanism, requiring human intervention before resuming.

2. **Violet Agent**: Has a 9% chance (`hitl_probability = 0.09`) of rolling for human interview mode, which pauses for real Gabe to answer interview questions via CLI prompts.

These interruptions prevent overnight/unattended batch artifact generation. Since Indigo doesn't stop to solve puzzles and we need to generate many artifacts without supervision, both should become fully synthetic.

## Scope

### In Scope
- Remove `interrupt_before=["await_human_action"]` from Black agent workflow compilation
- Remove `await_human_action` node from Black agent graph
- Remove Todoist task creation for sigil charging
- Remove `awaiting_human_action`, `pending_human_tasks`, and `human_instructions` fields from BlackAgentState
- Keep random sigil generation chance (currently ~25% via `@skip_chance(1.0)` which allows 25% through)
- Remove `roll_for_hitl` node and `human_interview` node from Violet agent
- Remove `needs_human_interview` field from VioletAgentState
- Remove conditional routing between human/simulated interview
- Always use `simulated_interview` path in Violet
- Clean up CLI resume functionality related to these pause states
- Remove pickle-based pause state persistence for sigil workflow

### Out of Scope
- EVP generation (remains unchanged)
- Black agent's counter-proposal generation logic
- Violet's simulated interview quality or persona system
- Any other agent workflows

## Key Changes

### Black Agent (Sigil)
- Sigil generation becomes fire-and-forget: generate artifact, save to disk, continue
- No Todoist integration for sigil charging tasks
- Remove workflow checkpointing for sigil-specific pauses
- Simplify graph: `generate_sigil` → END (no `await_human_action` → `update_alternate_song_spec_with_sigil` path)

### Violet Agent
- Remove the HitL probability roll entirely
- Remove `human_interview` node
- Workflow becomes: `select_persona` → `generate_questions` → `simulated_interview` → `synthesize_interview` → `generate_alternate_song_spec`
- Remove CLI prompt code for human interview

### CLI/Runner
- Remove `resume` subcommand logic related to sigil pauses
- Remove pickle state file persistence for paused workflows
- Simplify workflow completion handling

## Risks & Mitigations

| Risk | Mitigation |
|------|------------|
| Loss of human creative input in sigils | Sigils were always primarily synthetic; charging was ceremonial |
| Loss of Gabe's authentic voice in interviews | RAG corpus provides authentic patterns; 9% was too rare to matter for training |
| Breaking existing paused workflows | Document upgrade path; existing pkl files become invalid |

## Success Criteria

1. Full workflow runs from start to finish without any CLI prompts or pauses
2. No pickle files created during workflow execution
3. Sigil artifacts still generated with same probability
4. Violet interviews always use simulated path
5. All existing tests pass (after updating test mocks)
