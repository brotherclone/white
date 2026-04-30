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

## Future Work

Ideas worth speccing as follow-on changes once this is running in production:

### Multi-song Composition Board
The board currently fetches composition for the active song only. A true kanban would
show one card per song per stage across all handed-off songs — visible without switching
active songs. Each card would need to call `GET /composition` with a song identifier
rather than relying on the server's active-song context. Requires either a new
`GET /composition/<song_id>` endpoint or embedding `logic_project_path` in the songs
index so the board can resolve each song's composition independently.

### thread_slug empty-string bug in composition.yml
`composition.yml` is written with `ctx.get("thread", "")` which leaves `thread_slug`
blank when the YAML field exists but is empty. Should use `ctx.get("thread") or
production_dir.parent.parent.name` (same fix already applied to the path resolution).

### Re-handoff as explicit MIDI sync
Today re-running handoff skips the seed copy (correct) but also silently re-copies MIDI.
A dedicated "Sync MIDI" action (separate from full handoff) would make it safe to call
after evolving or re-promoting without the risk of accidentally re-scaffolding a folder.
Could be a `POST /handoff/sync` that only runs step 2 (MIDI copy) with no folder/seed
side effects.

### Drift Report
`composition.yml` knows when stages advance and which MIDI was handed off. A drift
report comparing Claude's production plan (bar counts, section order, energy arc) to
the actual Logic arrangement after mixing would close the creative feedback loop. The
report could flag sections that were cut, reordered, or significantly lengthened, and
feed that signal back into future generation weighting. Likely a CLI tool that reads
`composition.yml` + `production_plan.yml` and writes `drift_report.yml` alongside them.

### Version Notes
The `notes` field on each composition version is always `""`. A small text input on the
board card (blur-to-save, same pattern as the label field in the candidate browser)
would let the mixer annotate each version checkpoint ("rough strings added", "vocal comp
done") and make the version history actually useful as a production log.

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
