# Change: Pipeline Run Orchestrator

## Why

The production pipeline has five generation phases (chords, drums, bass, melody,
lyrics) each with a promotion step, plus init and assembly on either end. Every
phase is a working CLI. But running a song from scratch still requires the operator
to remember the exact command for each phase, the right flags, the correct paths,
and the right order — then repeat this for every promote step.

After two full runs producing over an hour of music, the friction is clear: it
isn't any single command that's hard, it's holding the state of where you are
in the sequence and what to do next. The pipeline lives in Gabe's head between
sessions. That's a bottleneck and a recall error waiting to happen.

The orchestrator replaces that mental load with a state-aware CLI: it knows
where each song is in its lifecycle, what the next step is, and emits the exact
command (or runs it) with the correct paths pre-filled.

## What Changes

### New module: `app/generators/midi/production/pipeline_runner.py`

A state-aware orchestrator that reads `song_context.yml` from a production
directory and drives the pipeline forward.

**Core concept:** each phase has a `status` in `song_context.yml`
(`pending | in_progress | generated | promoted`). The orchestrator reads these
statuses and knows what's runnable, what's waiting on human review, and what's
done. It never skips a promotion step — it pauses and prompts the human.

**Primary commands (all via `python -m app.generators.midi.production.pipeline_runner`):**

```
# Show where a song is in its lifecycle
pipeline status --production-dir <path>

# Start or resume — runs next auto-runnable phase, then pauses for review
pipeline run --production-dir <path>

# After human has labelled candidates in review.yml, promote and advance
pipeline promote --production-dir <path> --phase chords

# Run all phases through to lyrics, pausing at each promotion gate
pipeline run --production-dir <path> --through lyrics

# Show the next command without running it
pipeline next --production-dir <path>
```

**`pipeline status` output (example):**

```
Song: The Last Pollinators Elegy (green)
Dir:  shrink_wrapped/the-breathing-machine/production/green__last_pollinators_elegy_v1

  init_production   ✅  complete
  chords            ✅  promoted (8 sections)
  drums             ✅  promoted (8 sections)
  bass              ✅  promoted (8 sections)
  melody            🔄  generated — awaiting promotion
  lyrics            ⏳  pending melody promotion

Next: review melody/review.yml, then run: pipeline promote --phase melody
```

**`pipeline run` behaviour:**

1. Reads `song_context.yml` phase statuses
2. Finds the first `pending` phase whose upstream dependency is `promoted`
3. Runs that phase's pipeline with the correct arguments (production-dir, song-proposal,
   any phase-specific flags inferred from song context)
4. Updates phase status to `generated`
5. Prints the review.yml path and stops — never auto-promotes
6. Tells the operator exactly what to do next

**Phase dependency graph (hardcoded):**

```
init_production → chords → drums → bass → melody → lyrics
```

Each phase depends on the previous being `promoted`. Lyrics additionally requires
melody `promoted`.

### `song_context.yml` phase tracking

`pipeline_runner` reads and writes the `phases` dict in `song_context.yml`.
Phases start `pending`; the runner sets `in_progress` when starting a phase and
`generated` when it completes. `promote` sets `promoted`.

`promote_part.py` is updated to write `promoted` status back to `song_context.yml`
after a successful promotion run (so status stays in sync even when promote is
called directly).

### `pipeline promote` — guided promotion

Wraps `promote_part.py` with a confirmation step:

1. Reads `review.yml` for the phase
2. Shows a summary: how many candidates, how many labelled, any conflicts
3. Asks for confirmation before calling `promote_part`
4. On success, updates phase status and prints the next command

This makes promotion a two-step human gesture (label in the file, confirm in the
terminal) rather than a single CLI invocation that could silently overwrite work.

### Multi-song run: `pipeline batch`

For running a whole thread's worth of songs in sequence:

```
pipeline batch --thread shrink_wrapped/the-breathing-machine-learns-to-sing \
               --phase chords
```

Finds all production dirs in the thread, runs the named phase for any song where
that phase is `pending`, and stops before promoting anything. Useful for generating
all chord candidates across a thread in one sitting before sitting down to review.

## Impact

- Affected specs: `init-production`, `promote-part`, `chord-pipeline`,
  `drum-pipeline`, `bass-pipeline`, `melody-pipeline`, `lyric-pipeline`
- New files:
  - `app/generators/midi/production/pipeline_runner.py`
- Modified files:
  - `app/generators/midi/production/promote_part.py` — write phase status to song_context.yml
  - `app/generators/midi/production/init_production.py` — ensure phases dict written on init
- No breaking changes to any existing pipeline CLIs; orchestrator is additive
- Tests: `tests/generators/midi/production/test_pipeline_runner.py`
  - Status reading with various phase combinations
  - Next-phase resolution logic
  - Promote gate confirmation flow (mocked user input)
  - Batch phase enumeration
