# Change: Add sample browser and restore quartet button

## Why
Two features are missing from the candidate browser:
1. **Chromatic samples** — the Refractor can score audio segments from `staged_raw_material`
   against a song's color target, giving the producer a ranked palette of reference/loop
   material to use in Logic. The `white_composition.retrieve_samples` module was specced
   and archived (2026-03-15) but the file was never written; `grain_synthesizer.py` already
   imports from it and will fail at runtime.
2. **Quartet generation** — the "Run Strings" button is coded in the UI but has a silent
   gap: if `song_context.yml` carries `quartet: pending` (a valid intermediate state),
   `canRunQuartet` evaluates to `false` and neither the status indicator nor the button
   renders. The spec also does not clearly place the button alongside "Handoff to Logic".

## What Changes

### Sample browser
- Implement `white_composition.retrieve_samples` (`load_clap_index`, `retrieve_by_color`)
  using the precomputed CLAP parquet at `training_data_clap_embeddings.parquet`
- Add `GET /samples` endpoint — returns top-N segments scored for the active song's color
- Add `GET /audio/{segment_id}` endpoint — streams the pre-extracted WAV from
  `staged_raw_material` for in-browser auditioning
- Add `POST /samples/{segment_id}/export` endpoint — copies the segment WAV to
  `$LOGIC_OUTPUT_DIR/<thread>/<title>/Samples/` immediately (no need to wait for handoff)
- Add a **Samples panel** in the candidate browser, visible regardless of active phase
  filter, showing ranked segment rows with inline audio player and per-row Export button

### Logic handoff samples sweep
- During `handoff()`, copy any WAV files already in `<production_dir>/samples/` to the
  Logic `Samples/` folder so that a full re-handoff stays in sync

### Quartet button
- Fix `canRunQuartet`: treat `"pending"` as equivalent to absent — button should show when
  `quartetStatus` is `undefined`, `null`, or `"pending"`
- Spec explicitly positions the quartet button alongside "Handoff to Logic" when melody
  is promoted

## Notes
- The CLAP parquet uses precomputed embeddings; the Refractor's lazy CLAP encoder is
  NOT invoked for retrieval — this is purely a parquet lookup + cosine-rank.
- `staged_raw_material` WAVs are already segment-level files — no time-slicing needed.
- The existing `chromatic-sample-retrieval` spec covers the library layer; new requirements
  here cover the API and UI layer only.

## Impact
- Affected specs: `candidate-browser-web`, `logic-handoff`
- Affected code:
  - `packages/composition/src/white_composition/retrieve_samples.py` — new file
  - `packages/api/src/white_api/candidate_server.py` — three new endpoints
  - `packages/api/tests/test_candidate_server.py` — shape tests for new endpoints
  - `packages/composition/src/white_composition/logic_handoff.py` — samples sweep in `handoff()`
  - `packages/client/lib/api.ts` — new fetch helpers
  - `packages/client/lib/types.ts` — new `SampleEntry` type
  - `packages/client/app/candidates/page.tsx` — Samples panel + quartet fix
