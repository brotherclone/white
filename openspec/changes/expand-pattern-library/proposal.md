# Change: Expand Pattern Library — Hazy / Sparse / Lamentful Templates

## Why

Two full pipeline runs produced over an hour of music. Listening back, the most
listenable results cluster around a specific aesthetic: big hazy synths, slow
melodic movement, soft lamentful vocals, lots of space. Reference artists —
Grouper, Beach House, My Bloody Valentine, Boards of Canada, Julianna Barwick —
appeared repeatedly in `sounds_like` outputs, confirming the Refractor already
wants to go here.

The current pattern library was written generically to prove the pipeline worked.
It skews busy: active bass walking, dense hi-hat grids, fast melodic phrases.
That's the opposite of the target aesthetic. More templates won't help if they're
all the same character — what's needed is templates that are sparser, hazier, and
more patient, so the scoring system has good options to choose from when the
concept points toward that aesthetic cluster.

Additionally, there is currently no way to tag patterns by aesthetic character,
which means the selection logic can't distinguish "motorik groove" from
"brushed sparse" from "ambient drone" even when the concept strongly implies one.

## What Changes

### Aesthetic tags on existing pattern dataclasses

Each `DrumPattern`, `BassPattern`, and `MelodyPattern` gains an optional
`tags: list[str]` field. Tags are free-form but drawn from a controlled
vocabulary per instrument family. Examples:

- Drums: `sparse`, `dense`, `half_time`, `brushed`, `motorik`, `ambient`,
  `ghost_only`, `electronic`
- Bass: `drone`, `pedal`, `walking`, `arpeggiated`, `sustained`, `minimal`
- Melody: `stepwise`, `arpeggiated`, `descent`, `wide_interval`, `sparse`,
  `dense`, `lamentful`

Scoring functions may use tags as a soft prior when the concept/color implies
an aesthetic cluster (e.g. if `sounds_like` contains Grouper, prefer
`sparse` + `lamentful` over `dense`).

### New drum templates — sparse / atmospheric family

Add at minimum:

- **`half_time_sparse`** — kick on 1, snare on 3, open hat on the off-beat,
  nothing else. One bar of space per two bars of pattern.
- **`ghost_verse`** — ghost snare only, no kick, whisper-quiet hats. For
  sections that should feel like held breath.
- **`brushed_folk`** — brush swells on 2 and 4, light kick, no hi-hat grid.
- **`ambient_pulse`** — single low kick every 2 bars, crash swell on bar 4.
  Suitable for intro and outro sections.
- **`kosmische_slow`** — like motorik but half-tempo, fewer hat subdivisions.

All new templates tagged `sparse` and/or `ambient`.

### New bass templates — drone / pedal family

Add at minimum:

- **`root_drone`** — single root note, whole-note duration, no movement.
  Sustains across chord changes (chord changes heard in upper voices only).
- **`slow_pedal`** — root on beat 1, octave below on beat 3. Two notes per bar.
- **`descending_sigh`** — root → major 7th → 5th over 4 bars, stepwise descent.
  Resolves to root. The most lamentful bass contour.
- **`sustained_fifth`** — held 5th drone across the bar, slight swell in velocity.
- **`minimal_walk`** — root + one passing tone approaching the next chord.
  Contrast with existing dense walking patterns.

All new templates tagged `drone`, `pedal`, or `minimal` as appropriate.

### New melody templates — lamentful / sparse family

Add at minimum:

- **`slow_descent`** — stepwise downward motion, quarter notes, phrase every 2
  bars. Lots of rest. Maximum lament per MIDI byte.
- **`breath_phrase`** — 3-note phrase, long rest, 3-note phrase. Mimics a breath
  pattern. Uncanny because the rests feel too long.
- **`pentatonic_lament`** — minor pentatonic, descending, with occasional held
  notes and vibrato-implied velocity swell.
- **`floating_repeat`** — same 2-3 note motif repeated at slightly different
  rhythmic positions each bar. Creates a drifting, disoriented quality.
- **`single_line`** — one note per bar, whole-note or dotted half. Barely a
  melody. For sections where the texture is the point.

All new templates tagged `lamentful`, `sparse`, or `stepwise` as appropriate.

### `sounds_like`-aware pattern selection hint

`init_production.py` prompt updated to include the aesthetic cluster when
relevant artist names are detected. If the `sounds_like` list contains artists
in the ambient/shoegaze/drone cluster (Grouper, Beach House, MBV, Low, Barwick,
BoC, Stars of the Lid, etc.), the `song_context.yml` gains an `aesthetic_hints`
field:

```yaml
aesthetic_hints:
  density: sparse     # sparse | moderate | dense
  texture: hazy       # hazy | clean | rhythmic
  vocal_register: lamentful  # lamentful | expressive | robotic
```

Pipeline phases read `aesthetic_hints` and use tags to weight the candidate pool
before scoring (sparse-tagged patterns get a score bonus when density=sparse).
This is a soft prior — the Refractor still has final say.

## Impact

- Affected specs: `drum-pattern-generation`, `bass-line-generation`,
  `melody-lyrics-generation`, `init-production`
- Affected code:
  - `app/generators/midi/patterns/drum_patterns.py` — new templates + `tags` field
  - `app/generators/midi/patterns/bass_patterns.py` — new templates + `tags` field
  - `app/generators/midi/patterns/melody_patterns.py` — new templates + `tags` field
  - `app/generators/midi/pipelines/drum_pipeline.py` — read aesthetic_hints, apply tag weighting
  - `app/generators/midi/pipelines/bass_pipeline.py` — same
  - `app/generators/midi/pipelines/melody_pipeline.py` — same
  - `app/generators/midi/production/init_production.py` — aesthetic_hints detection + write
- No breaking changes: `tags` is additive; existing patterns without tags behave identically
- Tests: new pattern unit tests; pipeline tests for tag weighting; init_production tests
  for aesthetic_hints detection
