# Change: Add web-based candidate browser

## Why
The terminal candidate browser (`app/tools/candidate_browser.py`) cannot show all
candidates simultaneously, has no reliable scrolling, and cannot play MIDI in-context.
A browser-based UI solves all three: native scrolling, in-browser MIDI playback via
the `<midi-player>` web component, and full keyboard + mouse interaction without
terminal size constraints.

## What Changes
- New `app/tools/candidate_server.py` — FastAPI app that serves candidates as JSON,
  handles approve/reject writes, and streams MIDI file bytes
- New `app/tools/static/browser.html` — self-contained vanilla HTML/JS page; no build
  step, no npm, no framework
- Launch: `python -m app.tools.candidate_server --production-dir <path>` opens
  `http://localhost:8000` automatically in the default browser
- New dependencies: `fastapi` and `uvicorn` (small, well-maintained, fits existing
  httpx stack)

## Impact
- Affected specs: candidate-browser-web (new); candidate-browser (existing terminal
  tool is unchanged — both coexist)
- Affected code: new `app/tools/candidate_server.py`, new `app/tools/static/browser.html`
- Not breaking — `review.yml` format unchanged; terminal tool still works
