## 1. Data model

- [x] 1.1 Add `is_final: bool = False` field to `SongProposalIteration` in
      `app/structures/manifests/song_proposal.py`

## 2. Agent fixes — mark final proposal

- [x] 2.1 **Black**: in `generate_sigil` (the last node before END), set
      `state.counter_proposal.is_final = True` on the active proposal
- [x] 2.2 **Red**: identify the last node that touches `counter_proposal` before END;
      set `is_final = True` there (in `evaluate_books_versus_proposals` when routing done)
- [x] 2.3 **Yellow**: in `render_game_run`, set `is_final = True` on the last
      appended counter_proposal before saving

## 3. White agent — save_all_proposals

- [x] 3.1 In `save_all_proposals`, write individual `song_proposal_*.yml` only when
      `iteration.is_final is True`
- [x] 3.2 For agents that produce exactly one iteration (Orange, Green, Blue, Indigo,
      Violet), treat a single-iteration list as implicitly final — mark it `is_final=True`
      before writing if none are already marked

## 4. HTML opt-in flag

- [x] 4.1 Add `--with-html` flag to `run_white_agent start` in `run_white_agent.py`
- [x] 4.2 Pass `with_html: bool` via env var `WHITE_WITH_HTML` set at startup
- [x] 4.3 In each agent that generates HTML (Lord Pulsimore character sheets), gate
      the HTML generation block behind `WHITE_WITH_HTML`
- [x] 4.4 Confirm shrinkwrap handles missing `html/` directory without error
      (verified: `copy_thread_files` iterates existing subdirs only)

## 5. Tests

- [x] 5.1 Unit test `save_all_proposals`: two iterations, only the `is_final` one
      produces a standalone file; both appear in `all_song_proposals`
- [x] 5.2 Unit test single-iteration fallback: unmarked single iteration is saved
- [x] 5.3 Confirm existing black/red/yellow agent tests still pass (3069 passed)
- [x] 5.4 Unit test `--with-html` flag: HTML generation called when set, skipped
      when absent
