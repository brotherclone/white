# Tasks: Remove HitL from Sigil and Violet

## Phase 1: Violet Agent Simplification

- [ ] **1.1** Remove `roll_for_hitl` node from Violet agent graph
- [ ] **1.2** Remove `human_interview` node from Violet agent graph
- [ ] **1.3** Remove conditional edges routing to human/simulated interview
- [ ] **1.4** Update graph to: `generate_questions` → `simulated_interview` directly
- [ ] **1.5** Remove `needs_human_interview` field from `VioletAgentState`
- [ ] **1.6** Remove `hitl_probability` attribute from VioletAgent
- [ ] **1.7** Clean up Rich console/prompt imports if no longer needed
- [ ] **1.8** Update VioletAgentState docstring to remove HitL workflow reference

## Phase 2: Black Agent Sigil Simplification

- [ ] **2.1** Remove `interrupt_before=["await_human_action"]` from workflow compilation
- [ ] **2.2** Remove `await_human_action` node from Black agent graph
- [ ] **2.3** Remove `update_alternate_song_spec_with_sigil` node (no longer needed without pause)
- [ ] **2.4** Simplify graph routing: `generate_sigil` → END directly
- [ ] **2.5** Remove Todoist task creation from `generate_sigil` method
- [ ] **2.6** Remove `awaiting_human_action`, `pending_human_tasks`, `human_instructions` fields from `BlackAgentState`
- [ ] **2.7** Remove `should_update_proposal_with_sigil` field from `BlackAgentState`
- [ ] **2.8** Remove `route_after_sigil_chance` routing function (sigil just saves and returns)
- [ ] **2.9** Simplify `generate_sigil` to just: generate → save → return (no HitL setup)
- [ ] **2.10** Remove or archive `resume_black_workflow.py` (no longer needed)
- [ ] **2.11** Remove Todoist import and usage from black_agent.py

## Phase 3: CLI and Runner Cleanup

- [ ] **3.1** Remove `resume` subcommand from `run_white_agent.py`
- [ ] **3.2** Remove pickle import and pause state persistence
- [ ] **3.3** Remove `workflow_paused` checking and pkl file creation
- [ ] **3.4** Simplify `start_workflow` to not handle pause states
- [ ] **3.5** Remove pause-related logging messages

## Phase 4: State Cleanup

- [ ] **4.1** Remove `workflow_paused` field from `MainAgentState`
- [ ] **4.2** Remove `pause_reason` field from `MainAgentState`
- [ ] **4.3** Remove `pending_human_action` field from `MainAgentState`
- [ ] **4.4** Remove `dedupe_human_tasks` function from `black_agent_state.py`

## Phase 5: Test Updates

- [ ] **5.1** Update `test_violet_agent.py` to remove HitL-related test cases
- [ ] **5.2** Update `test_black_agent.py` to remove sigil pause test cases
- [ ] **5.3** Update `test_white_agent.py` to remove workflow pause assertions
- [ ] **5.4** Remove or update any mock files related to HitL flow
- [ ] **5.5** Add test confirming Violet always uses simulated interview
- [ ] **5.6** Add test confirming sigil generation completes without pause

## Phase 6: Documentation

- [ ] **6.1** Update any agent documentation describing the HitL flow
- [ ] **6.2** Remove references to Todoist integration for sigils

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
