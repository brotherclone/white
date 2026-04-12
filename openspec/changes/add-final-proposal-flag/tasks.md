## 1. Data model

- [ ] 1.1 Add `is_final: bool = False` field to `SongProposalIteration` in
      `app/structures/manifests/song_proposal.py`

## 2. Agent fixes — mark final proposal

- [ ] 2.1 **Black**: in `generate_sigil` (the last node before END), set
      `state.counter_proposal.is_final = True` on the active proposal
- [ ] 2.2 **Red**: identify the last node that touches `counter_proposal` before END;
      set `is_final = True` there
- [ ] 2.3 **Yellow**: in `render_game_run`, set `is_final = True` on the last
      appended counter_proposal before saving

## 3. White agent — save_all_proposals

- [ ] 3.1 In `save_all_proposals`, write individual `song_proposal_*.yml` only when
      `iteration.is_final is True`
- [ ] 3.2 For agents that produce exactly one iteration (Orange, Green, Blue, Indigo,
      Violet), treat a single-iteration list as implicitly final — mark it `is_final=True`
      before writing if none are already marked

## 4. HTML opt-in flag

- [ ] 4.1 Add `--with-html` flag to `run_white_agent start` in `run_white_agent.py`
- [ ] 4.2 Pass `with_html: bool` through to `MainAgentState` (or via env var
      `WHITE_WITH_HTML`) so agents can read it without deep parameter threading
- [ ] 4.3 In each agent that generates HTML (at minimum Lord Pulsimore character
      sheets; audit others), gate the HTML generation block behind `with_html`
- [ ] 4.4 Confirm shrinkwrap handles missing `html/` directory without error
      (it already does — verify with a dry run)

## 5. Tests

- [ ] 5.1 Unit test `save_all_proposals`: two iterations, only the `is_final` one
      produces a standalone file; both appear in `all_song_proposals`
- [ ] 5.2 Unit test single-iteration fallback: unmarked single iteration is saved
- [ ] 5.3 Confirm existing black/red/yellow agent tests still pass
- [ ] 5.4 Unit test `--with-html` flag: HTML generation called when set, skipped
      when absent
