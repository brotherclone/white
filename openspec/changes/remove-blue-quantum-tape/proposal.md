# Change: Remove quantum tape label artifact from Blue agent

## Why

The `QuantumTapeLabelArtifact` (a rendered cassette label HTML) was a fun fiction
prop for the Cassette Bearer (Blue) agent, but it has no downstream consumer and
adds an LLM call, image generation, and a file that confuses the shrinkwrap output.
The alternate timeline narrative and biographical period are the meaningful Blue
artifacts — the physical tape label is decorative.

Removing it simplifies the Blue agent graph, reduces cost per run, and shrinks the
output bundle to what matters.

## What Changes

- `generate_tape_label` node is removed from the Blue agent LangGraph workflow
- The `tape_label` field is removed from `BlueAgentState`
- All `QuantumTapeLabelArtifact` imports and construction in `blue_agent.py` are removed
- The `QuantumTapeLabelArtifact` class and its structures/enums are **retained** but
  not invoked — they remain available for future use
- The `quantum-tape-artifact` spec requirements covering Blue agent tape label
  population and flatten completeness are REMOVED (they described the removed node)
- The `chain-artifacts` spec is updated: Blue's artifact list no longer includes
  the quantum tape label

## Impact

- Affected code: `app/agents/blue_agent.py`, `app/structures/states/blue_agent_state.py`
- Spec changes: `quantum-tape-artifact` (REMOVED 2 requirements), `chain-artifacts` (MODIFIED)
- The `QuantumTapeLabelArtifact` class files are untouched — no deletion
- Tests for `QuantumTapeLabelArtifact` are untouched (the class still works)
- Existing Blue agent tests that assert tape_label in state need updating
