## 1. Backend — stage field
- [ ] 1.1 Add `stage` computation to `scan_songs` in `candidate_server.py`:
       `ideation` when `song_context.yml` absent; `composition` when `LOGIC_OUTPUT_DIR`
       is set and `composition.yml` exists in the Logic song dir (imported from
       `white_composition.logic_handoff`); `generation` otherwise — wrap the Logic check
       in a try/except so a missing env var or import error falls back to `generation`
- [ ] 1.2 Update `test_song_entry_shape` in `test_candidate_server.py` to assert
       `"stage"` in the shape; add `test_stage_ideation`, `test_stage_generation`, and
       `test_stage_composition` (mocking LOGIC_OUTPUT_DIR for composition check)

## 2. TypeScript types
- [ ] 2.1 Add `stage: "ideation" | "generation" | "composition"` to `SongEntry` in
       `packages/client/lib/types.ts`

## 3. Frontend — route restructure
- [ ] 3.1 Move song browser logic from `app/songs/page.tsx` to `app/page.tsx` (replace
       the current two-link landing page); keep the same component name or rename to
       `SongBrowserPage`
- [ ] 3.2 Replace `app/songs/page.tsx` with a client-side redirect to `/` using
       Next.js `redirect()` or `useEffect` + `router.replace("/")`

## 4. Frontend — song card stage badge
- [ ] 4.1 Replace the "not initialized" span on song cards with a stage badge that
       reads "Ideation", "Generation", or "Composition" derived from `song.stage`
- [ ] 4.2 Update `handleSelect` routing logic:
       - `stage === "ideation"` → activate + init + navigate to `/candidates` (existing path)
       - `stage === "generation"` → activate + navigate to `/candidates`
       - `stage === "composition"` → activate + navigate to `/board`

## 5. Frontend — Run Agent navigation
- [ ] 5.1 Add a "Run Agent" link (or button styled as a link) to the song browser page
       header pointing to `/agent`

## 6. Frontend — agent run screen
- [ ] 6.1 Create `app/agent/page.tsx` containing: a `← Songs` link back to `/`, a
       "Generate New Song" button that calls `POST /generate`, a spinner during polling,
       and success/error toasts — move the generate state and polling logic from the
       song browser into this new component
- [ ] 6.2 Remove the generate state, polling logic, and "Generate New Song" button from
       the song browser page (now at `app/page.tsx`)

## 7. Dev launch script
- [ ] 7.1 Create `dev.sh` at the repo root: load `.env` if present, validate
       `SHRINK_WRAPPED_DIR` is set (exit 1 with message if not), start
       `uvicorn white_api.candidate_server:app --reload --port 8000
       -- --shrink-wrapped-dir $SHRINK_WRAPPED_DIR` in background, then start
       `cd packages/client && npm run dev`; trap SIGINT/SIGTERM to kill both PIDs
- [ ] 7.2 `chmod +x dev.sh`
