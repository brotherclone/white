## 1. FastAPI backend

- [ ] 1.1 Add `POST /promote` endpoint to `app/tools/candidate_server.py`:
      - Request body: `{production_dir: str, phase: str}`
      - Validate `phase` is one of `chords`, `drums`, `bass`, `melody`, `quartet`
      - Call `pipeline_runner` promote logic (import directly or subprocess)
      - Return `{"ok": true, "promoted_count": N}` on success
      - Return 400 for invalid phase, 500 with error detail on failure
- [ ] 1.2 Add `promote` to the CORS-allowed methods if not already covered

## 2. Next.js frontend

- [ ] 2.1 Add a Promote button to the phase filter toolbar (alongside the
      phase dropdown and status filter)
- [ ] 2.2 Button disabled state: when `selectedPhase` is `null`, `""`, or `"all"`;
      add tooltip text "Select a phase to enable promote"
- [ ] 2.3 Button enabled state: when a single phase is selected
- [ ] 2.4 On click:
      - POST `{production_dir: currentProductionDir, phase: selectedPhase}` to `/promote`
      - Show success toast: "Promoted N files for {phase}"
      - Show error toast on failure with the server's error detail
      - Refetch `/candidates` to update status badges in the table
- [ ] 2.5 Disable button and show spinner while promotion is in-flight

## 3. Evolve endpoint + button

- [ ] 3.1 Add `POST /evolve` endpoint to `app/tools/candidate_server.py`:
      - Request body: `{production_dir: str, phase: str}`
      - Validate `phase` is one of `drums`, `bass`, `melody`; return 400 for others
      - Call the appropriate `breed_*_patterns` pipeline (import from `pattern_evolution`)
      - Return `{"ok": true, "evolved_count": N}` on success
- [ ] 3.2 Add Evolve button to the phase toolbar in `web/`:
      - Visible only when selectedPhase is `drums`, `bass`, or `melody`
      - POST to `/evolve`, show spinner during generation (can take 10–30s)
      - On success: toast "Added N evolved candidates", refetch candidate list
      - Evolved candidates show an "evolved" badge in the table

## 4. ACE Studio endpoints + buttons

- [ ] 4.1 Add `POST /ace/export` endpoint:
      - Calls `export_to_ace_studio(production_dir)` from `ace_studio_export.py`
      - Returns `{"ok": true, "singer": ..., "sections": [...]}` on success
      - Returns 503 with "ACE Studio not running" if `ConnectionError` raised
- [ ] 4.2 Add `POST /ace/import` endpoint:
      - Calls `locate_and_ingest_render(production_dir)` from `ace_studio_import.py`
      - Returns `{"ok": true, "render_path": ...}` on success
      - Returns 404 if no WAV render found
- [ ] 4.3 Add ACE Studio buttons to the melody phase toolbar in `web/`:
      - "Export to ACE Studio" button: visible when phase=melody and melody is promoted;
        POST to `/ace/export`, show spinner, toast with singer + sections on success
      - "Import Render" button: visible after export; POST to `/ace/import`,
        toast render path on success

## 5. Tests

- [ ] 5.1 API test: `POST /promote` with valid phase → 200 and `promoted_count` present
- [ ] 5.2 API test: `POST /promote` with invalid phase → 400
- [ ] 5.3 API test: `POST /evolve` with `drums` → 200 and `evolved_count` present
- [ ] 5.4 API test: `POST /evolve` with `chords` → 400
- [ ] 5.5 API test: `POST /ace/export` with ACE Studio unreachable → 503
- [ ] 5.6 Frontend: Evolve button absent when phase is "chords", present when "drums"
- [ ] 5.7 Frontend: Promote button disabled when phase is "all", enabled when "chords"
