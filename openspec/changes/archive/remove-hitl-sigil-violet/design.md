# Design: Remove HitL from Sigil and Violet

## Current Architecture

### Violet Agent Flow (Before)
```
START → select_persona → generate_questions → roll_for_hitl
                                                    ↓
                                    ┌───────────────┴───────────────┐
                                    ↓                               ↓
                            human_interview              simulated_interview
                            (CLI prompts)                (LLM + RAG corpus)
                                    ↓                               ↓
                                    └───────────────┬───────────────┘
                                                    ↓
                                        synthesize_interview → generate_alternate_song_spec → END
```

**Key components:**
- `hitl_probability = 0.09` - 9% chance of human interview
- `roll_for_hitl` node sets `needs_human_interview` flag
- `human_interview` node uses Rich console + Prompt for CLI input
- `route_after_roll` conditional edges

### Black Agent Flow (Before)
```
START → generate_alternate_song_spec → generate_evp → evaluate_evp
                                                          ↓
                                        ┌─────────────────┴─────────────────┐
                                        ↓                                   ↓
                            update_alternate_song_spec_with_evp      generate_sigil
                                        ↓                                   ↓
                                        └─────────────────┬─────────────────┘
                                                          ↓
                                              ┌───────────┴───────────┐
                                              ↓                       ↓
                                    await_human_action              END
                                    (interrupt_before)
                                              ↓
                            update_alternate_song_spec_with_sigil → END
```

**Key components:**
- `interrupt_before=["await_human_action"]` in workflow compilation
- SQLite checkpointer for state persistence during pause
- Todoist task creation for sigil charging
- `should_update_proposal_with_sigil` routing flag
- `pending_human_tasks`, `awaiting_human_action`, `human_instructions` state fields

---

## Proposed Architecture

### Violet Agent Flow (After)
```
START → select_persona → generate_questions → simulated_interview → synthesize_interview → generate_alternate_song_spec → END
```

**Simplifications:**
- Remove `roll_for_hitl` node entirely
- Remove `human_interview` node entirely
- Direct edge from `generate_questions` to `simulated_interview`
- No conditional routing needed

### Black Agent Flow (After)
```
START → generate_alternate_song_spec → generate_evp → evaluate_evp
                                                          ↓
                                        ┌─────────────────┴─────────────────┐
                                        ↓                                   ↓
                            update_alternate_song_spec_with_evp      generate_sigil → END
                                        ↓
                                  generate_sigil → END
```

**Simplifications:**
- Remove `await_human_action` node
- Remove `update_alternate_song_spec_with_sigil` node
- `generate_sigil` becomes terminal: generates artifact, saves to disk, returns
- No checkpointer needed (disable or simplify)
- No Todoist integration

---

## State Changes

### VioletAgentState
```python
# REMOVE:
needs_human_interview: bool = Field(default=False, ...)

# No changes to other fields
```

### BlackAgentState
```python
# REMOVE:
human_instructions: Annotated[Optional[str], ...] = ""
pending_human_tasks: Annotated[List[Dict[str, Any]], dedupe_human_tasks] = Field(...)
awaiting_human_action: Annotated[bool, ...] = False
should_update_proposal_with_sigil: Annotated[bool, ...] = False

# KEEP:
should_update_proposal_with_evp  # Still used for EVP routing
```

### MainAgentState
```python
# REMOVE:
workflow_paused: Annotated[bool, ...] = False
pause_reason: Annotated[Optional[str], ...] = None
pending_human_action: Annotated[Optional[Dict[str, Any]], ...] = None
```

---

## File Changes Summary

| File | Change Type | Description |
|------|-------------|-------------|
| `app/agents/violet_agent.py` | Modify | Remove HitL nodes, routing, probability |
| `app/agents/states/violet_agent_state.py` | Modify | Remove `needs_human_interview` |
| `app/agents/black_agent.py` | Modify | Remove pause nodes, Todoist, simplify sigil |
| `app/agents/states/black_agent_state.py` | Modify | Remove pause-related fields |
| `app/agents/states/white_agent_state.py` | Modify | Remove workflow_paused fields |
| `app/agents/workflow/resume_black_workflow.py` | Delete | No longer needed |
| `run_white_agent.py` | Modify | Remove resume command, pickle handling |

---

## Behavioral Changes

### Sigil Generation
| Aspect | Before | After |
|--------|--------|-------|
| Probability | ~25% (via skip_chance) | Same ~25% |
| Human charging | Required via Todoist | None (synthetic only) |
| Workflow pause | Yes, with checkpoint | No |
| Proposal update | After human completes | None (sigil is artifact-only) |

### Violet Interview
| Aspect | Before | After |
|--------|--------|-------|
| HitL probability | 9% | 0% |
| Interview source | Human or LLM+RAG | Always LLM+RAG |
| CLI prompts | Sometimes | Never |
| Workflow pause | Sometimes | Never |

---

## Migration Notes

1. **Existing paused workflows**: Any workflows paused at `await_human_action` will become invalid. Users should complete or discard them before upgrading.

2. **Pickle files**: Existing `paused_state.pkl` files will be orphaned. They can be safely deleted.

3. **Checkpoints**: The SQLite checkpoint database (`checkpoints/black_agent.db`) can be deleted after migration.

4. **Todoist tasks**: Any pending sigil charging tasks in Todoist will remain but won't be tracked by the system.
