# Change: Phase-gated promote button in candidate review UI

## Why

The web UI already has approve/reject/play working per-candidate. What it doesn't
have is a way to promote — the step that actually copies approved MIDI files to
`approved/` and marks the phase as promoted in `song_context.yml`. Currently
promotion is a separate manual CLI step.

The promote button needs to know which phase it's promoting (chords, drums, bass,
melody, quartet) to run `pipeline_runner promote --phase <phase>`. Allowing promote
when no phase filter is set would mean guessing the phase from the candidate list,
which is error-prone. The safe design: the Promote button is disabled until the
phase filter is set to a single phase. Once set, it's unambiguous.

## What Changes

**FastAPI backend (candidate_server.py)**
- New endpoint: `POST /promote` with body `{production_dir, phase}`
- Calls `pipeline_runner.promote_phase(production_dir, phase)` (or shells out to
  `python -m app.generators.midi.production.pipeline_runner promote
  --production-dir <path> --phase <phase>`)
- Returns `{ok: true, promoted_count: N}` on success
- Returns 400 if phase is not one of the valid values, 500 if promotion fails

**Next.js frontend (web/)**
- Promote button added to the phase filter toolbar
- Button is **disabled** (greyed out, `cursor-not-allowed`) when the phase dropdown
  is set to "all" or unset
- Button is **enabled** when a single phase is selected
- On click: POST to `/promote`, show a toast with the result (N files promoted),
  refetch candidate list to reflect updated statuses
- Tooltip on disabled state: "Select a phase to enable promote"

## Impact

- Affected code: `app/tools/candidate_server.py` (new endpoint),
  `web/app/page.tsx` or toolbar component (promote button state)
- Affected spec: `candidate-browser-web` (MODIFIED)
- No change to `candidate_browser.py` (terminal browser) or `promote_part.py`
- The promote endpoint is idempotent — re-running is safe
