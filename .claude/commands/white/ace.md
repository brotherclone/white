---
name: White: ACE Studio
description: Push melody MIDI and lyrics to ACE Studio, check export status, or ingest a completed WAV render. ACE Studio 2.0 must be running locally.
category: White
tags: [white, ace-studio, vocals]
---

Manage the ACE Studio vocal synthesis step for a White production.

**The ACE Studio workflow**

1. **export** — After melody is promoted, push MIDI + lyrics into ACE Studio
2. Open ACE Studio, edit syllables, adjust pitch, render to WAV
3. **import** — Pull the rendered WAV back into the production dir
4. Run `/white session-close` to score the mix and commit

---

**Steps**

1. Identify the production directory from `$ARGUMENTS`, recent context, or ask.

2. Based on `$ARGUMENTS`:

   **"export"** (or no argument — export is the default action):
   - Run `wpipe ace export --production-dir <path>`
   - This pushes `assembled/assembled_melody.mid` + `melody/lyrics.txt` into ACE Studio
   - Each arrangement section becomes a separate clip on the vocal track
   - Singer is resolved via `app/reference/mcp/ace_studio/singer_voices.yml`
   - Result is written to `song_context.yml` under `ace_studio:`
   - After export: remind the user to open ACE Studio, edit, and render

   **"status"**:
   - Run `wpipe ace status --production-dir <path>`
   - Shows project name, track, singer, sections exported, and render path

   **"import"** (after rendering in ACE Studio):
   - Run `wpipe ace import --production-dir <path>`
   - Searches for `VocalSynthv*/VocalSynthv*_*.wav` in the production dir
   - Copies to `melody/ace_render.wav` and updates `song_context.yml`
   - After import: ready to score the mix with `/white session-close`

**Prerequisites**
- ACE Studio 2.0 must be running (MCP server at `http://localhost:21572/mcp`)
- For export: melody phase must be promoted (`melody/approved/` exists)
- For import: WAV render must be in `VocalSynthv*/` subfolder or production dir root

**Troubleshooting**
- If export fails with connection error: launch ACE Studio first
- If singer isn't found: check `app/reference/mcp/ace_studio/singer_voices.yml`
- To refresh the tool manifest after an ACE Studio update:
  `python -m app.reference.mcp.ace_studio.probe`
