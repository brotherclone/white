## 1. State plumbing
- [x] 1.1 Add `negative_constraints: Annotated[str, lambda x, y: y or x] = ""` to
      `BaseRainbowAgentState` (or each individual color agent state, if a shared
      base isn't practical) in `packages/ideation/src/white_ideation/agents/states/`
- [x] 1.2 In `white_agent.py`, pass `negative_constraints=state.negative_constraints`
      when constructing each color agent's sub-state in that agent's
      `__call__` method (mirrors the existing `white_proposal=current_proposal` pattern)

## 2. Prompt injection per agent
- [x] 2.1 Black Agent (`black_agent.py`, `generate_alternate_song_spec` and
      `update_alternate_song_spec_with_evp`): append constraints block
- [x] 2.2 Red Agent (`red_agent.py`, `generate_alternate_song_spec`) — also fixed
      an unrelated pre-existing bug found while testing this: the method never
      handled `_invoke_structured` returning a real `SongProposalIteration`
      (only a `dict`), so Red's counter-proposal was silently never appended
      to `song_proposals.iterations` on any successful run
- [x] 2.3 Orange Agent (`orange_agent.py`, `generate_alternate_song_spec`)
- [x] 2.4 Yellow Agent (`yellow_agent.py`, `generate_alternate_song_spec`)
- [x] 2.5 Green Agent (`green_agent.py`, `generate_alternate_song_spec`)
- [x] 2.6 Blue Agent (`blue_agent.py`, `generate_alternate_song_spec`)
- [x] 2.7 Indigo Agent (`indigo_agent.py`, the counter-proposal generation path
      in `generate_alternate_song_spec`)
- [x] 2.8 Violet Agent (`violet_agent.py`, the counter-proposal generation path)

## 3. Tests
- [x] 3.1 Unit test per agent: `negative_constraints` set on state → prompt
      passed to `_invoke_structured`/LLM call includes the text
- [x] 3.2 Unit test (representative, on Black Agent): `negative_constraints`
      empty → prompt unchanged from current behavior (regression guard)
- [x] 3.3 Integration check: covered by the per-agent prompt-content tests
      themselves, since each constructs state the same way white_agent.py does

## 4. Verification
- [x] 4.1 Run full test suite (`packages/ideation/tests`, `packages/core/tests`)
- [x] 4.2 Live smoke test: run the full proposal chain once and confirm (via
      chain_artifacts debug snapshots) that a color agent's counter-proposal
      prompt actually contains the constraints text
