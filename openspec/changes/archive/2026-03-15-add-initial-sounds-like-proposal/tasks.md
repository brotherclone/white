## 1. Claude Prompt for sounds_like Generation
- [x] 1.1 Write `_build_sounds_like_prompt(meta)` — takes concept, color, genres, mood
      from song proposal; asks Claude for 4–7 reference artists as bare name strings;
      instructs: "return only a YAML list of artist names, no descriptions, no commentary"
- [x] 1.2 Add `_parse_sounds_like_response(text) -> list[str]` — parses the bare name
      list from Claude's response; strips any parenthetical annotations if Claude adds
      them anyway

## 2. initial_proposal.yml Writer
- [x] 2.1 Add `write_initial_proposal(production_dir, meta, sounds_like)` — writes
      `initial_proposal.yml` with: sounds_like (list of str), color, concept, singer,
      key, bpm, time_sig, generated timestamp, proposed_by: claude
- [x] 2.2 Add `load_initial_proposal(production_dir) -> dict` — reads
      `initial_proposal.yml` if present; returns {} if missing (graceful fallback)
- [x] 2.3 Idempotent: if `initial_proposal.yml` already exists, skip generation and
      print a note (allows `--force` flag to regenerate)

## 3. CLI / Entry Point
- [x] 3.1 New `app/generators/midi/production/init_production.py` with `--production-dir`
      and `--song-proposal` flags; reads song proposal, calls Claude, writes
      `initial_proposal.yml`, prints the generated sounds_like list
- [x] 3.2 Add `--force` flag to regenerate even if `initial_proposal.yml` exists
- [x] 3.3 Document in chord_pipeline.py CLI help: "run init_production.py first"

## 4. Pipeline Wiring
- [x] 4.1 `chord_pipeline.py`: after loading song proposal, call `load_initial_proposal()`
      and use its `sounds_like` if present; fall back to proposal's own sounds_like field
- [x] 4.2 `lyric_pipeline.py`: remove `meta["sounds_like"] = []`; read from
      `initial_proposal.yml` via `load_initial_proposal()`
- [x] 4.3 `composition_proposal.py`: seed sounds_like from `initial_proposal.yml` when
      building the prompt; Claude can extend or replace

## 5. Tests
- [x] 5.1 Unit: `_parse_sounds_like_response()` handles bare names, annotated names,
      YAML list format, and numbered list format
- [x] 5.2 Unit: `write_initial_proposal()` + `load_initial_proposal()` round-trip
- [x] 5.3 Unit: `load_initial_proposal()` returns {} gracefully when file missing
- [x] 5.4 Integration: stub Claude API; verify full init flow writes expected YAML
