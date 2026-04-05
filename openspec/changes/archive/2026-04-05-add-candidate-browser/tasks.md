## 1. Data Layer
- [ ] 1.1 Write `load_all_candidates(production_dir) → list[CandidateEntry]` — walks all
       phase `review.yml` files and collects pending candidates
- [ ] 1.2 Write `approve_candidate(candidate: CandidateEntry)` — writes `label: approved`
       to the correct `review.yml` entry in-place
- [ ] 1.3 Write `reject_candidate(candidate: CandidateEntry)` — writes `label: rejected`

## 2. Terminal UI
- [ ] 2.1 Build main table view with `rich.table`: columns = phase, section, template,
       composite_score, label
- [ ] 2.2 Implement keyboard navigation: ↑/↓ to select, `a` to approve, `r` to reject,
       `p` to play (macOS `open`), `q` to quit
- [ ] 2.3 Show score breakdown panel (theory/chromatic/diversity factor) for selected row
- [ ] 2.4 Auto-refresh table after approve/reject

## 3. CLI Entry Point
- [ ] 3.1 `python -m app.tools.candidate_browser --production-dir <path>`
- [ ] 3.2 `--phase <chords|drums|bass|melody|quartet>` filter flag (default: all)
- [ ] 3.3 `--section <label>` filter flag

## 4. Tests
- [ ] 4.1 Unit tests for `load_all_candidates` and `approve_candidate` (no UI tests needed)
