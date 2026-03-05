## 1. Loop Inventory Builder
- [ ] 1.1 Implement `build_loop_inventory(production_dir) -> dict` — reads approved
        review.yml files for chords, drums, bass, melody; extracts per-loop label,
        bar count, energy score, and section role. Returns a structured dict keyed
        by instrument → list of available loops with metadata.
- [ ] 1.2 Implement `load_song_proposal(production_dir) -> dict` — locates the song
        proposal YAML (via `chords/review.yml` source_proposal field or glob fallback),
        returns concept, mood, color, title, key, bpm, time_sig, sounds_like, genres.

## 2. Claude Prompt + API Call
- [ ] 2.1 Implement `build_prompt(proposal, inventory) -> str` — constructs the
        system + user prompt. System context: White project chromatic synthesis
        framework, what each color means, what a good arrangement arc looks like.
        User prompt: song metadata, loop inventory table (label, bars, energy, instrument),
        explicit ask for a section-by-section arrangement with repeat counts and rationale.
- [ ] 2.2 Implement `call_claude(prompt) -> str` — calls `claude-sonnet-4-6` via
        Anthropic SDK, returns the raw text response. Raises on API error.
- [ ] 2.3 Implement `parse_response(raw) -> dict` — extracts structured proposal from
        Claude's response. Claude is instructed to return a fenced YAML block; parser
        extracts it. Falls back to storing raw text under `rationale` if parsing fails.

## 3. Output
- [ ] 3.1 Implement `write_proposal(production_dir, proposal_dict)` — writes
        `composition_proposal.yml` with: `proposed_by: claude`, `generated` (ISO
        timestamp), `color_target`, `loop_inventory` (what was available), `rationale`
        (prose), `sounds_like` (Claude's suggested reference artists for the proposed
        arrangement), `proposed_sections` list (each with `name`, `repeat`, `energy_note`,
        `transition_note`, `loops` dict mapping instrument → loop label).
- [ ] 3.2 CLI: `python -m app.generators.midi.composition_proposal --production-dir <dir>`
        Exits with a clear message if no approved loops found. Warns but proceeds if
        only some instruments have approved loops.

## 4. Drift Integration
- [ ] 4.1 Extend `compute_drift` in `assembly_manifest.py` to accept an optional
        `proposal_path` argument. When provided, computes section-level diff between
        the proposed section sequence and the actual `arrangement.txt` sections:
        - `sections_added` — sections in arrangement not in proposal
        - `sections_removed` — sections in proposal not in arrangement
        - `repeat_deltas` — sections where repeat count changed (proposed vs actual)
        - `order_changed` — bool, true if section ordering differs
- [ ] 4.2 Write `proposal_drift` block into the existing drift report output when
        a proposal is present.

## 5. Tests
- [ ] 5.1 Unit: `build_loop_inventory` with mocked review files — correct labels,
        bar counts, energy scores extracted
- [ ] 5.2 Unit: `parse_response` — valid fenced YAML extracted; graceful fallback
        on malformed response
- [ ] 5.3 Unit: `compute_drift` with proposal_path — sections_added, sections_removed,
        repeat_deltas computed correctly
- [ ] 5.4 Integration: run against `blue__rust_signal_memorial_v1` with mocked Claude
        response and assert `composition_proposal.yml` written with expected keys
