## Context

The White music production pipeline ends at melody promotion. From there, files need to
land in a specific Logic Pro project layout on a fast local drive for arrangement and
mixing work. Progress through mixing stages is currently untracked. `production_decisions.py`
captures training data but is ML-focused, not workflow-focused.

## Goals / Non-Goals

- Goals:
  - Zero-click Logic project setup from the candidate browser
  - Track per-song mix stage in a durable, human-readable file (`composition.yml`)
  - Kanban board so mix status is visible across all songs at a glance
  - Keep backend lyrics/decisions code intact for future use
- Non-Goals:
  - Modifying Logic Pro projects programmatically
  - Audio file management (only MIDI is copied)
  - Replacing `production_decisions.yml` (it remains for ML training use)

## Decisions

### `composition.yml` lives in the Logic song folder, not in shrink_wrapped

Rationale: `composition.yml` describes the Logic-side mix workflow, not the generation
pipeline. Keeping it co-located with the `.logicx` file makes it easy to find from
the Finder and avoids polluting the `shrink_wrapped` production dir with Logic-specific
state. The API reads it from `$LOGIC_OUTPUT_DIR/<thread>/<title>/composition.yml`.

### Mix stage is a linear enum, not free-form tags

The nine stages are ordered. A song always moves forward (or is pinned at a stage).
Using an enum rather than freeform text enables the swimlane UI to have fixed columns
and lets the API validate transitions.

Mix stages in order:
```
structure → lyrics → recording → vocal_placeholders → augmentation →
cleaning → rough_mix → mix_candidate → final_mix
```

### Versions are append-only entries inside `composition.yml`

Each time the user bumps a version (e.g. after a major mix revision), a new entry is
appended. The `current_version` key points to the latest. This gives a lightweight
audit trail without a database.

### Seed is copied, not symlinked

Logic Pro resolves relative audio paths from the project bundle location. A symlink
would share the same bundle path as the seed, breaking audio file resolution.
A full directory copy (`shutil.copytree`) is the correct approach.

### `arrangement.txt` and lyrics detection

The handoff script looks for `arrangement.txt` and any file matching `lyrics*.txt`
or `*.lrc` inside the production dir. These are moved (not copied) into the Logic
song folder. If none are found, the script continues without error.

### API reads composition.yml via LOGIC_OUTPUT_DIR

The candidate server resolves `$LOGIC_OUTPUT_DIR/<thread_slug>/<title>/composition.yml`
for the active song. If the file does not exist (handoff not yet run), endpoints return
`{"status": "not_initialized"}` rather than 404, to give the UI a clean state to
display a "Run Handoff" prompt.

## Risks / Trade-offs

- **Large seed copy** — `seed.logicx` is a Logic folder bundle. `shutil.copytree` is
  synchronous; if the bundle is large (>500 MB), the `POST /handoff` call will block.
  Mitigation: run handoff in a background thread (same pattern as `/pipeline/run`).
- **LOGIC_OUTPUT_DIR not set** — handoff endpoint returns 422 with a clear message.

## Open Questions

- Should the board page live at `/board` or be a tab on the existing songs index (`/`)?
  Current proposal: separate `/board` route, linked from the songs index header.
