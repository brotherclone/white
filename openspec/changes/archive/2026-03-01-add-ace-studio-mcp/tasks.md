## Phase 1 — Discovery (always runs)

- [ ] 1.1 Create `app/reference/mcp/ace_studio/` package (`__init__.py`)
- [ ] 1.2 Implement `app/reference/mcp/ace_studio/probe.py`
       - Connect to `http://localhost:21572/mcp` via `tools/list` JSON-RPC
       - Write `tool_manifest.json` alongside the script
       - Check for required capability keywords (project, track, midi, lyric, singer)
       - If any missing: write `FEASIBILITY.md` and exit with code 1
       - If all present: print summary and exit 0
- [ ] 1.3 Add `tool_manifest.json` to `.gitignore`
- [ ] 1.4 Run probe against live ACE Studio instance; commit result
       - If feasibility gate **fails**: commit `FEASIBILITY.md`, stop here
       - If feasibility gate **passes**: proceed to Phase 2

## Phase 2 — Client wrapper (conditional on Phase 1 passing)

- [ ] 2.1 Implement `app/reference/mcp/ace_studio/client.py`
       - `AceStudioClient.__init__(base_url)` — loads `tool_manifest.json`
       - `_call(tool_name, **kwargs) → dict` — POST JSON-RPC, raise on error
       - `_find_tool(*keywords) → str` — fuzzy match tool name from manifest
       - `get_project_info() → dict`
       - `create_project(title, bpm, key, time_sig) → dict`
       - `add_vocal_track(singer) → dict`
       - `import_midi(track_id, midi_path) → dict`
       - `set_lyrics(track_id, lyrics_text) → dict`
- [ ] 2.2 Tests: `tests/reference/mcp/test_ace_studio_client.py`
       - Mock HTTP server (`responses` or `httpretty`)
       - `test_get_project_info_returns_dict`
       - `test_create_project_passes_correct_params`
       - `test_find_tool_fuzzy_match`
       - `test_server_unreachable_raises_connection_error`

## Phase 3 — Pipeline integration (conditional on Phase 2)

- [ ] 3.1 Implement `app/generators/midi/ace_studio_export.py`
       - `export_to_ace_studio(production_dir) → dict | None`
       - Reads `assembled/assembled_melody.mid`, `melody/lyrics.txt`,
         `production_plan.yml`
       - Returns `None` (with warning) if ACE Studio unreachable
       - Returns result dict with `project_id`, `track_id` on success
- [ ] 3.2 CLI: `python -m app.generators.midi.ace_studio_export --production-dir ...`
- [ ] 3.3 Tests: `tests/generators/midi/test_ace_studio_export.py`
       - `test_export_happy_path` — mock client, verify call sequence
       - `test_export_returns_none_when_unreachable`
       - `test_export_skips_when_no_assembled_midi`
       - `test_export_skips_when_no_lyrics`
