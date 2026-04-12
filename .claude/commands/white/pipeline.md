---
name: White: Pipeline
description: Show pipeline status, run the next phase, or run a specific phase for a production directory.
category: White
tags: [white, pipeline, production]
---

You are helping manage the White music production pipeline for a specific song.

**Steps**

1. Identify the production directory:
   - If `$ARGUMENTS` contains a path, use that.
   - Otherwise, look for an obvious current production dir: check recent git status, recently modified files under `shrink_wrapped/`, or ask the user which song they're working on.

2. Run `wpipe status --production-dir <path>` (or `python -m app.generators.midi.production.pipeline_runner status --production-dir <path>` if aliases not loaded) to show the current phase status.

3. Based on `$ARGUMENTS` determine the action:
   - No action or "status" → just show status and stop.
   - "next" → run `wpipe next --production-dir <path>` and print the suggested command, then ask if you should run it.
   - "run" → run `wpipe run --production-dir <path>` to execute the next pending phase.
   - "promote" → run `wpipe promote --production-dir <path>` to promote approved candidates.
   - A specific phase name (e.g. "chords", "drums", "bass", "melody", "production_plan", "score_mix") → run that phase directly.

4. After running a phase, show the updated status and indicate the next step.

**Notes**
- The production dir is always inside `shrink_wrapped/<thread>/production/<song_slug>/`
- If the user says a song name (e.g. "violet song" or "network dreams"), resolve it to the right dir by searching `shrink_wrapped/`
- Never run `promote` without first confirming the user has reviewed and labelled candidates
