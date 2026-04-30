# Change: Add Logic Handoff and Composition Board

## Why

The generation pipeline (init → chords → drums → bass → melody) produces approved MIDI
files, but handing them off to Logic Pro for arrangement and mixing is currently a
fully manual process: creating the project folder, copying files, and tracking mix
progress are all done by hand. The `lyrics` and `decisions` phases clutter the
candidate-browser UI even though they are not part of the active review workflow.

## What Changes

- **Trim UI pipeline to melody** — remove `lyrics`, `decisions`, and `quartet` from the
  candidate-browser phase filter and pipeline status strip. The backend code is
  preserved; only the web UI drops these phases.
- **`logic_handoff.py`** (new module in `white_composition`) — given an active
  production directory, it:
  1. Creates `$LOGIC_OUTPUT_DIR/<thread_slug>/<song_title>/` on the fast drive.
  2. Copies the seed Logic project (`packages/composition/logic/seed/seed.logicx`)
     into that folder, renamed to `<song_title>.logicx`.
  3. Creates `MIDI/chords/`, `MIDI/bass/`, `MIDI/drums/`, `MIDI/melody/` subfolders
     and copies each phase's approved `.mid` files into the appropriate subfolder.
  4. Moves any `arrangement.txt` and lyrics files found in the production dir into
     the Logic song folder as siblings of the `.logicx` file.
  5. Creates `composition.yml` in the Logic song folder recording version 1 at
     stage `structure`.
- **`composition.yml` schema** — tracks one or more versions, each with a mix stage
  drawn from an ordered set: `structure → lyrics → recording → vocal_placeholders →
  augmentation → cleaning → rough_mix → mix_candidate → final_mix`.
- **API endpoints** — `POST /handoff`, `GET /composition`, `PATCH /composition/stage`,
  `POST /composition/version` added to `candidate_server.py`.
- **New root index** — `/` becomes a minimal landing page with two navigation links:
  "Generation" (→ `/songs`) and "Composition Board" (→ `/board`). The current songs
  index moves from `/` to `/songs`.
- **Composition Board UI** — new Next.js page at `/board` displaying all songs as
  cards in a horizontal swimlane (one column per mix stage). Cards advance between
  columns to update the stage.

## Impact

- Affected specs: `candidate-browser-web`, new `logic-handoff`
- Affected code:
  - `packages/client/app/page.tsx` (new landing — two links)
  - `packages/client/app/songs/page.tsx` (current songs index, moved from `/`)
  - `packages/client/app/candidates/page.tsx` (remove lyrics/decisions/quartet from PHASES; update breadcrumb base path)
  - `packages/client/app/board/page.tsx` (new)
  - `packages/client/lib/api.ts` (new endpoints)
  - `packages/client/lib/types.ts` (new types)
  - `packages/api/src/white_api/candidate_server.py` (new endpoints)
  - `packages/composition/src/white_composition/logic_handoff.py` (new)
- New env var: `LOGIC_OUTPUT_DIR` (required for handoff)
- Seed Logic project: `packages/composition/logic/seed/seed.logicx` (already present)
- No breaking changes to existing generation pipeline or CLI tools
