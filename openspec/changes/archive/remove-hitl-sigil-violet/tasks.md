# Tasks: Remove HitL from Sigil and Violet

## Phase 1: Violet Agent Simplification

- [x] **1.1** Remove `roll_for_hitl` node from Violet agent graph
- [x] **1.2** Remove `human_interview` node from Violet agent graph
- [x] **1.3** Remove conditional edges routing to human/simulated interview
- [x] **1.4** Update graph to: `generate_questions` → `simulated_interview` directly
- [x] **1.5** Remove `needs_human_interview` field from `VioletAgentState`
- [x] **1.6** Remove `hitl_probability` attribute from VioletAgent
- [x] **1.7** Clean up Rich console/prompt imports if no longer needed
- [x] **1.8** Update VioletAgentState docstring to remove HitL workflow reference

## Phase 2: Black Agent Sigil Simplification

- [x] **2.1** Remove `interrupt_before=["await_human_action"]` from workflow compilation
- [x] **2.2** Remove `await_human_action` node from Black agent graph
- [x] **2.3** Remove `update_alternate_song_spec_with_sigil` node (no longer needed without pause)
- [x] **2.4** Simplify graph routing: `generate_sigil` → END directly
- [x] **2.5** Remove Todoist task creation from `generate_sigil` method
- [x] **2.6** Remove `awaiting_human_action`, `pending_human_tasks`, `human_instructions` fields from `BlackAgentState`
- [x] **2.7** Remove `should_update_proposal_with_sigil` field from `BlackAgentState`
- [x] **2.8** Remove `route_after_sigil_chance` routing function (sigil just saves and returns)
- [x] **2.9** Simplify `generate_sigil` to just: generate → save → return (no HitL setup)
- [x] **2.10** Remove or archive `resume_black_workflow.py` (no longer needed)
- [x] **2.11** Remove Todoist import and usage from black_agent.py

## Phase 3: CLI and Runner Cleanup

- [x] **3.1** Remove `resume` subcommand from `run_white_agent.py`
- [x] **3.2** Remove pickle import and pause state persistence
- [x] **3.3** Remove `workflow_paused` checking and pkl file creation
- [x] **3.4** Simplify `start_workflow` to not handle pause states
- [x] **3.5** Remove pause-related logging messages

## Phase 4: State Cleanup

- [x] **4.1** Remove `workflow_paused` field from `MainAgentState`
- [x] **4.2** Remove `pause_reason` field from `MainAgentState`
- [x] **4.3** Remove `pending_human_action` field from `MainAgentState`
- [x] **4.4** Remove `dedupe_human_tasks` function from `black_agent_state.py`

## Phase 5: Test Updates

- [x] **5.1** Update `test_violet_agent.py` to remove HitL-related test cases
- [x] **5.2** Update `test_black_agent.py` to remove sigil pause test cases
- [x] **5.3** Update `test_white_agent.py` to remove workflow pause assertions
- [x] **5.4** Remove or update any mock files related to HitL flow
- [x] **5.5** Add test confirming Violet always uses simulated interview
- [x] **5.6** Add test confirming sigil generation completes without pause

## Phase 6: Documentation

- [x] **6.1** Update any agent documentation describing the HitL flow
- [x] **6.2** Remove references to Todoist integration for sigils

## Dependencies

- Phase 2 and Phase 1 can run in parallel
- Phase 3 depends on Phase 1 + Phase 2 completion
- Phase 4 depends on Phase 2 + Phase 3 completion
- Phase 5 can start after Phase 1 + Phase 2, runs parallel with Phase 3/4
- Phase 6 runs last

## Validation

After each phase:
```bash
pytest tests/ -v
python run_white_agent.py start --mode single_agent --agent violet
python run_white_agent.py start --mode single_agent --agent black
```

Full validation:
```bash
python run_white_agent.py start --mode full_spectrum
# Should complete without any pauses or prompts
```
