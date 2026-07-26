# Change: Propagate negative constraints into each color agent's counter-proposal

## Why
`negative_constraints.yml` (derived from prior song titles/keys/BPM/concepts via
`generate_negative_constraints.py`) is currently injected only into White Agent's
initial proposal prompt and its final rewrite synthesis (`white_agent.py`). None
of the 8 individual color agents (Black, Red, Orange, Yellow, Green, Blue,
Indigo, Violet) reference it when generating their own counter-proposal in
`generate_alternate_song_spec` — confirmed by grep, only `white_agent.py`
imports/uses `negative_constraints` anywhere in `packages/ideation/src/white_ideation/agents/`.
So a color agent's own creative pass can freely drift back into territory
White explicitly tried to steer the run away from, and the constraint only
gets re-applied once, at the very end, when White does its final rewrite —
too late to have shaped what each agent actually wrote.

## What Changes
- Thread `negative_constraints` (already loaded once per run in `white_agent.py`)
  into each color agent's own state so it's available at counter-proposal time,
  not just in White's initial/final prompts.
- Each color agent's `generate_alternate_song_spec` prompt appends the same
  avoidance block format already used for White's initial proposal.
- No change to `generate_negative_constraints.py` itself or to
  `negative_constraints.yml`'s format — this is purely about propagating an
  already-computed value to more consumers.

## Impact
- Affected specs: `ideation-negative-constraints` (new capability)
- Affected code:
  - `packages/ideation/src/white_ideation/agents/states/*.py` — add a
    `negative_constraints: str` field (or similar) to each color agent's state
  - `packages/ideation/src/white_ideation/agents/white_agent.py` — pass
    `state.negative_constraints` down when constructing each color agent's
    sub-state (mirrors how `white_proposal` is already threaded through)
  - `packages/ideation/src/white_ideation/agents/{black,red,orange,yellow,green,blue,indigo,violet}_agent.py` —
    append the avoidance block in `generate_alternate_song_spec`'s prompt
