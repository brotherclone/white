---
name: White: Session Close
description: Finish a production session — score the mix, shrinkwrap the thread, and commit. Run this when a song is done or at end of a work session.
category: White
tags: [white, session, shrinkwrap, score]
---

Close out a White production session cleanly.

**Steps**

1. **Identify context**
   - Find the production dir (from `$ARGUMENTS`, recent git status, or recently modified files under `shrink_wrapped/`)
   - Find the thread UUID from the production dir path (the UUID segment in `shrink_wrapped/<uuid>/...`)

2. **Score the mix** (if a bounced audio file exists)
   - Check `shrink_wrapped/<thread>/production/<slug>/song_context.yml` for a `mix_scored_at` timestamp — if already scored and no new bounce is indicated, skip.
   - Ask the user for the path to the audio bounce if not obvious (or if `$ARGUMENTS` contains a path, use it).
   - Run: `wscore --mix-file <audio-path> --production-dir <production-dir>`

3. **Capture production decisions**
   - Run: `python -m app.generators.midi.production.production_decisions --production-dir <production-dir>`
   - This writes `production_decisions.yml` to the production directory root.
   - Skip gracefully if the song is incomplete (partially produced songs still get a partial record).

4. **Shrinkwrap** the thread
   - Run: `wshrink --thread <uuid>`
   - If the thread already exists in `shrink_wrapped/` and there are new files, delete the stale copy first (`rm -rf shrink_wrapped/<thread-name>/`) then re-run.

4. **Stage and commit**
   - Stage: the updated `shrink_wrapped/<thread>/` tree and any modified `shrink_wrapped/index.yml` or `shrink_wrapped/negative_constraints.yml`
   - Commit with a message that names the song, e.g.: `song: finish <song-title> (<color>) production session`

5. **Report** what was done: phases completed, mix score (temporal/spatial/ontological), files committed.

**Notes**
- Don't push unless the user asks.
- If score_mix fails (no audio, ONNX error, etc.), note it and continue with shrinkwrap and commit.
- The audio bounce lives outside the repo — it doesn't get committed.
