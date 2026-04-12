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

## 3. Tests

- [ ] 3.1 API test: `POST /promote` with valid phase → 200 and `promoted_count` present
- [ ] 3.2 API test: `POST /promote` with invalid phase → 400
- [ ] 3.3 Frontend: confirm Promote button renders as disabled when phase is "all"
      and enabled when phase is "chords"
