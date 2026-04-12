## 1. Blue agent graph

- [x] 1.1 Remove `generate_tape_label` node from the LangGraph workflow in
      `app/agents/blue_agent.py` (remove `add_node`, `add_edge` calls)
- [x] 1.2 Remove the `generate_tape_label` method from `CassetteBearerAgent`
- [x] 1.3 Remove all `QuantumTapeLabelArtifact` and related quantum tape imports
      from `blue_agent.py`

## 2. State model

- [x] 2.1 Remove `tape_label` field from `BlueAgentState` in
      `app/structures/states/blue_agent_state.py`
- [x] 2.2 Remove any quantum tape imports from `blue_agent_state.py`

## 3. Tests

- [x] 3.1 Update Blue agent tests that assert `state.tape_label` or check
      for tape label in `state.artifacts` — remove those assertions
- [x] 3.2 Confirm remaining Blue agent tests still pass

## 4. Spec updates (no code)

- [x] 4.1 Validate with `openspec validate remove-blue-quantum-tape --strict`
