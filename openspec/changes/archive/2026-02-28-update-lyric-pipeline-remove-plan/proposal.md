# Change: Remove production_plan dependency from lyric pipeline

## Why

`production_plan.yml` was designed as a human-maintained contract between the
generation phases and downstream tools. In practice the arrangement (exported
from Logic as `arrangement.txt`) is the real contract — it captures exactly
which MIDI clips are placed on which tracks at which timecodes after human
editing. Driving the lyric pipeline directly from `arrangement.txt` + the song
proposal YAML eliminates a manual sync step and makes the pipeline reflect what
the human actually built.

Key insight from the production process:
- Track 4 in `arrangement.txt` = melody clips = implicitly vocal
- No `vocals: true/false` flag needed — presence on track 4 is the signal
- Bar counts come from clip durations × BPM/time_sig, not from a plan

## What Changes

- `lyric_pipeline.py`: replace `load_plan()` with `parse_arrangement()` +
  song proposal YAML reader
- One lyric block generated per unique melody label found on track 4 in
  the arrangement (already aligned with `update-lyric-loop-headers`)
- Bar counts derived from clip duration in arrangement
- `vocals_planned` / `vocals` fields no longer consulted
- `production_plan.yml` is not read, not required, not created by this pipeline
- Drum pipeline `next_section` annotation silently skipped if no plan present
  (already graceful — no code change needed)

## Impact

- Affected specs: `lyric-pipeline` (MODIFIED), `production-plan` (DEPRECATED)
- Affected code: `app/generators/midi/lyric_pipeline.py`
- No changes needed to chord, drum, bass, or melody pipelines
- `production_plan.py` left in place but no longer part of the standard workflow
