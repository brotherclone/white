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
   - "evolve drums/bass/melody" → run that phase with `--evolve` (see Evolutionary Mode below).
   - A specific phase name (e.g. "chords", "drums", "bass", "melody", "production_plan", "score_mix") → run that phase directly.

4. After running a phase, show the updated status and indicate the next step.

**Evolutionary Mode (`--evolve`)**

Any of the drums, bass, or melody phases can be run with evolutionary breeding instead
of pure template selection. This uses tournament selection + crossover on the template
library, scored by Refractor, to generate novel pattern hybrids. Good when standard
templates feel too familiar.

```
# Drums with evolution (default: 8 generations, population 30)
python -m app.generators.midi.pipelines.drum_pipeline --production-dir <path> --evolve

# Bass with more generations for more exotic results
python -m app.generators.midi.pipelines.bass_pipeline --production-dir <path> --evolve --generations 16

# Melody evolution
python -m app.generators.midi.pipelines.melody_pipeline --production-dir <path> --evolve
```

Evolved candidates appear alongside normal candidates in the review — labelled
`is_evolved: true` in `review.yml`. You review and promote them exactly the same way.

**Notes**
- The production dir is always inside `shrink_wrapped/<thread>/production/<song_slug>/`
- If the user says a song name (e.g. "violet song" or "network dreams"), resolve it to the right dir by searching `shrink_wrapped/`
- Never run `promote` without first confirming the user has reviewed and labelled candidates
