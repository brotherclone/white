# Change: Single final song proposal per chromatic agent

## Why

Black (ThreadKeepr), Red (Light Reader), and Yellow (Lord Pulsimore) each produce
multiple `SongProposalIteration` objects during their internal workflows — EVP updates,
reaction book revisions, and per-room game run counter-proposals. All of them are saved
as individual `song_proposal_<Color>_<id>.yml` files, leaving the human with no clear
signal about which one to use for production. The question "which Black is the one I
should make?" shouldn't need to be asked.

The internal creative iterations are valuable process and should be preserved for
traceability — but only the final resolved proposal from each agent should surface as a
standalone file.

## What Changes

- `SongProposalIteration` gains an `is_final: bool` field (default `False`)
- Each of the three offending agents marks exactly one iteration as `is_final=True`
  before handing back to Prism:
  - **Black**: the EVP-updated proposal (or the initial if EVP is skipped)
  - **Red**: the reaction-book-revised proposal
  - **Yellow**: the proposal produced after the last game run room
- `save_all_proposals` in `white_agent.py`:
  - Writes individual `song_proposal_<Color>_<id>.yml` **only** for `is_final=True`
    iterations
  - Continues to write `all_song_proposals_<thread>.yml` with all iterations for
    full traceability
- All other agents (Orange, Green, Blue, Indigo, Violet) already produce one iteration;
  their single proposal is implicitly marked `is_final=True` at save time

## Impact

- Affected specs: `chain-artifacts`
- Affected code: `app/structures/manifests/song_proposal.py`,
  `app/agents/white_agent.py` (`save_all_proposals`),
  `app/agents/black_agent.py`, `app/agents/red_agent.py`,
  `app/agents/yellow_agent.py`
- Non-breaking for shrinkwrap: fewer files per thread, `all_song_proposals` bundle
  unchanged
- The `song_proposals.iterations` list in state is not changed — all iterations
  remain available to Prism for rebracketing and synthesis
