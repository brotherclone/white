# Change: Add web-based candidate browser

## Why
The terminal candidate browser (`app/tools/candidate_browser.py`) cannot show all
candidates simultaneously, has no reliable scrolling, and cannot play MIDI in-context.
A browser-based UI solves all three: native scrolling, in-browser MIDI playback via
the `html-midi-player` web component, and full keyboard + mouse interaction without
terminal size constraints.

## What Changes
- New `app/tools/candidate_server.py` — FastAPI app that serves candidates as JSON,
  handles approve/reject writes, and streams MIDI file bytes; runs on port 8000
- New `web/` — Next.js 15 app (App Router, TypeScript, Tailwind CSS); runs on port 3000
  in dev, or builds to static output for production
- Launch (dev): `python -m app.tools.candidate_server --production-dir <path>` in one
  terminal; `cd web && npm run dev` in another; opens `http://localhost:3000`
- New Python dependencies: `fastapi` and `uvicorn`
- New JS toolchain: Node.js / npm (Next.js, TypeScript, Tailwind) — lives entirely
  under `web/` and does not affect the Python project

## Architecture
```
┌─────────────────────────────┐      ┌──────────────────────────────┐
│  Next.js (localhost:3000)   │ ───▶ │  FastAPI (localhost:8000)    │
│  app/page.tsx               │      │  GET  /candidates            │
│  components/CandidateTable  │      │  POST /candidates/{id}/approve│
│  components/MidiPlayer      │      │  POST /candidates/{id}/reject │
│  components/ScorePanel      │      │  GET  /midi/{id}             │
└─────────────────────────────┘      └──────────────────────────────┘
```

## Impact
- Affected specs: candidate-browser-web (new); candidate-browser (existing terminal
  tool is unchanged — both coexist)
- Affected code: new `app/tools/candidate_server.py`, new `web/` directory
- Not breaking — `review.yml` format unchanged; terminal tool still works
