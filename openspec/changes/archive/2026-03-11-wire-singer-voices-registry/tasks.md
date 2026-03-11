# Tasks: wire-singer-voices-registry

- [x] Add `load_singer_registry(path=None)` helper to `ace_studio_export.py` — loads
      `singer_voices.yml`, returns dict keyed by lowercase name, returns `{}` with
      warning if file missing
- [x] Add `resolve_ace_voice_name(singer_name, registry)` helper — implements the
      4-step resolution order; returns the name to pass to `find_singer()`
- [x] Update `export_to_ace_studio` to call registry load + resolve before
      `find_singer()`; update the result dict to include `ace_studio_voice` field
- [x] Confirm `singer_voices.yml` spelling: "Busyayo" lowercases to "busyayo" matching
      `melody_patterns.py` — no fix needed
- [x] Write unit tests for `load_singer_registry` (happy path, missing file)
- [x] Write unit tests for `resolve_ace_voice_name` (mapped+confirmed, mapped+null,
      not-in-registry, empty singer, case-insensitive)
- [x] Write integration test for `export_to_ace_studio` with mocked `AceStudioClient`
      confirming correct voice name is passed to `find_singer()` for Shirley
- [x] Update `SONG_GENERATION_PROCESS.md`: updated singer mapping note + removed
      `singer_voices.yml` not wired entry from known bugs table
