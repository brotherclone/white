## 1. Backend — Logic Handoff Module

- [x] 1.1 Create `packages/composition/src/white_composition/logic_handoff.py`
  - `MixStage` str enum with the nine stages in order
  - `handoff(production_dir: Path) -> Path` — scaffold Logic folder, copy seed, copy MIDI, move text files, write `composition.yml`
  - `read_composition(logic_song_dir: Path) -> dict | None`
  - `write_stage(logic_song_dir: Path, stage: str) -> None`
  - `add_version(logic_song_dir: Path) -> int` — appends new version, returns new version number
- [x] 1.2 Add `LOGIC_OUTPUT_DIR` to `.env` (document, do not commit value)
- [x] 1.3 Write unit tests in `tests/composition/test_logic_handoff.py`
  - Scaffold creates correct folder/file structure
  - Re-handoff skips seed copy, preserves composition.yml
  - MIDI copy populates phase subfolders
  - Missing LOGIC_OUTPUT_DIR raises EnvironmentError
  - Stage advance writes correct value

## 2. Backend — API Endpoints

- [x] 2.1 Add `_handoff_job` global state to `candidate_server.py` (same shape as `_run_job`)
- [x] 2.2 Add `POST /handoff` — runs `logic_handoff.handoff()` in background thread
- [x] 2.3 Add `GET /handoff/status` — returns handoff job state
- [x] 2.4 Add `GET /composition` — reads `composition.yml` via `read_composition()`; returns `{"status": "not_initialized"}` if absent
- [x] 2.5 Add `PATCH /composition/stage` — calls `write_stage()`; validates stage is a valid `MixStage`
- [x] 2.6 Add `POST /composition/version` — calls `add_version()`

## 3. Frontend — Types and API Client

- [x] 3.1 Add `MixStage` type and `CompositionEntry` interface to `lib/types.ts`
- [x] 3.2 Add to `lib/api.ts`: `startHandoff()`, `getHandoffStatus()`, `fetchComposition()`, `advanceStage()`, `addVersion()`

## 4. Frontend — Navigation Restructure

- [x] 4.1 Replace `app/page.tsx` with a minimal landing page — two links: "Generation" (→ `/songs`) and "Composition Board" (→ `/board`)
- [x] 4.2 Move current `app/page.tsx` songs index logic to `app/songs/page.tsx`
- [x] 4.3 Update `← Songs` breadcrumb in `app/candidates/page.tsx` to link to `/songs`
- [x] 4.4 Update `activateSong` redirect in songs page to navigate to `/candidates` (unchanged)

## 5. Frontend — Candidate Browser Trim

- [x] 5.1 Update `PHASES` constant in `app/candidates/page.tsx` to `["chords", "drums", "bass", "melody"]`
- [x] 5.2 Update pipeline status strip filter to the same four phases

## 6. Frontend — Songs Index Handoff Button

- [x] 6.1 Add "Handoff" button to each song card on `app/songs/page.tsx`
  - Calls `POST /handoff`, shows spinner, shows success/error toast

## 7. Frontend — Composition Board

- [x] 7.1 Create `app/board/page.tsx` — swimlane layout
  - Fetch all songs via `GET /songs`, then `GET /composition` per song
  - Render one column per `MixStage`; songs without `composition.yml` hidden
  - Each card: title, thread slug, color dot, version badge
  - Advance arrow calls `PATCH /composition/stage`, re-fetches

## 8. Validation

- [x] 8.1 `openspec validate add-logic-handoff --strict` passes
- [x] 8.2 All new tests pass (`pytest tests/composition/test_logic_handoff.py`)
- [x] 8.3 TypeScript check clean (`npx tsc --noEmit` in `packages/client`)
- [ ] 8.4 Manual smoke test: handoff a song, verify Logic folder created, board shows card
- [ ] 8.5 Verify `/` landing renders, `/songs` shows song list, breadcrumb on `/candidates` links to `/songs`
