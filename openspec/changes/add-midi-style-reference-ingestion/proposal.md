# Change: MIDI Style Reference Ingestion

## Why

The `sounds_like` field in `song_context.yml` is currently used in two ways:
it feeds into the DeBERTa concept embedding (text describing the artists shifts
the Refractor's scoring space) and it informs lyric prompting. Neither of these
touches the actual note-level generation. "Sounds like Grouper" changes how the
model scores patterns, but it doesn't change how those patterns are generated in
the first place.

The gap: generation is driven by Markov chains seeded from the hand-coded
template library. The chains don't know what Grouper actually does — the
characteristic very long sustains, phrases that start and stop without warning,
the way the bass barely moves while the melody drifts. That musical DNA isn't
in the templates or the embedding. It has to be extracted from the source
material.

MIDI transcriptions of reference artists are widely available. If we can pull
a handful of MIDI files for each `sounds_like` artist and extract statistical
features from them — note density per bar, velocity distribution, interval
histogram, average note duration, harmonic rhythm — we can use those features
to bias the Markov chain generation toward that artist's musical behaviour.
The result is generation that actually sounds more like the reference, not just
one that scores higher against a text description of the reference.

## What Changes

### New module: `app/generators/midi/style_reference.py`

Handles fetching, caching, and feature extraction for MIDI style references.

**Fetching**

Given an artist name, search one or more MIDI databases (Hooktheory, freemidi.org,
or a local cache populated by the user) for matching files. We do not auto-download
from scraped sources — the fetcher checks a local `style_references/` directory
first, then optionally queries an approved public API.

Local directory structure:
```
style_references/
  grouper/
    dragging_a_dead_deer.mid
    heavy_water_I.mid
  beach_house/
    space_song.mid
  ...
```

If no local files exist for an artist and no API is configured, the feature
extraction is skipped gracefully and that artist's contribution is zeroed out
(existing behaviour preserved).

**Feature extraction**

For each MIDI file, extract a `StyleProfile` (Pydantic model in
`app/structures/music/style_profile.py`):

```python
class StyleProfile(BaseModel):
    artist: str
    note_density: float          # mean notes per bar
    note_density_variance: float # how much density varies bar-to-bar
    mean_duration_beats: float   # mean note duration in beats
    duration_variance: float
    velocity_mean: float
    velocity_variance: float
    interval_histogram: dict[int, float]  # semitone → relative frequency
    harmonic_rhythm: float       # mean chord changes per bar
    rest_ratio: float            # proportion of bars with >50% silence
    phrase_length_mean: float    # mean notes per phrase (silence-delimited)
```

Profiles for all `sounds_like` artists are averaged (weighted equally) into a
single `AggregateStyleProfile` that represents the target sonic character.

**Caching**

Extracted profiles are cached as YAML in `style_references/<artist>/profile.yml`
so they are only recomputed when the source MIDI files change.

### Integration with `init_production.py`

After generating `sounds_like` and writing `song_context.yml`, `init_production`
runs style reference extraction for each listed artist (if local MIDI files
exist) and writes the aggregate profile to `song_context.yml`:

```yaml
style_reference_profile:
  note_density: 1.8          # Grouper: very sparse
  mean_duration_beats: 2.3   # long sustained notes
  rest_ratio: 0.61           # lots of silence
  interval_histogram:
    0: 0.42                  # lots of unisons / repeated notes
    2: 0.18                  # stepwise
    -2: 0.21
    7: 0.08                  # occasional 5ths
  harmonic_rhythm: 0.4       # chords change slowly
  velocity_mean: 68
  velocity_variance: 12      # dynamics are consistent, not dramatic
```

If no MIDI files are available for any artist, this block is omitted and the
pipeline behaves as before.

### Markov chain biasing in pipeline phases

Each pipeline (chord, drum, bass, melody) reads `style_reference_profile` from
`song_context.yml` and uses it to bias generation:

**Chord pipeline:**
- `harmonic_rhythm` → sets the HR distribution prior (slow harmonic rhythm = favour
  longer durations in the strum pattern)
- `interval_histogram` → weights the Markov transition probabilities (intervals
  that appear frequently in the reference get higher transition weights)

**Drum pipeline:**
- `note_density` → scales the candidate filter threshold (low density → prefer
  sparser templates; if evolving, seed population from `sparse`-tagged patterns)
- `velocity_variance` → sets the ghost note probability in template generation
  (low variance → quieter, more consistent velocities)

**Bass pipeline:**
- `mean_duration_beats` → biases toward longer-duration bass templates when high
- `rest_ratio` → scales probability of rest events in template generation
- `harmonic_rhythm` → if slow, favour pedal/drone templates over walking

**Melody pipeline:**
- `interval_histogram` → adjusts interval weights in the melodic Markov chain
  (so a Grouper-like reference increases unison and stepwise probability)
- `phrase_length_mean` → target phrase length for melodic phrase generation
- `note_density` → sets rest duration scaling

The biasing is a weighted blend: `style_weight` parameter (default 0.4) controls
how strongly the reference profile pulls the generator. At 0.0, existing behaviour
is preserved. At 1.0, the generator attempts to match the reference profile exactly.
The Refractor still does final scoring — the style reference biases the generation
distribution, not the selection.

### `style_weight` in `song_context.yml`

```yaml
style_reference_profile:
  ...
  style_weight: 0.4   # editable before generation runs
```

Human can tune this per song. A concept that needs to deviate from the reference
cluster can set this lower. A song that should be deep in Grouper territory can
push it to 0.6–0.7.

### Local MIDI population utility

A helper to make populating the local cache easy:

```
python -m app.generators.midi.style_reference populate \
    --artist "Grouper" \
    --files ~/Downloads/grouper_midi/*.mid
```

Copies the files into `style_references/grouper/`, extracts the profile, and
writes `profile.yml`. Reports the key features so the user can verify the
extraction looks right before it affects generation.

## Impact

- New files:
  - `app/generators/midi/style_reference.py`
  - `app/structures/music/style_profile.py` (Pydantic model)
- Modified files:
  - `app/generators/midi/production/init_production.py` — profile extraction + write
  - `app/generators/midi/pipelines/chord_pipeline.py` — profile-biased HR + transitions
  - `app/generators/midi/pipelines/drum_pipeline.py` — density + velocity biasing
  - `app/generators/midi/pipelines/bass_pipeline.py` — duration + rest biasing
  - `app/generators/midi/pipelines/melody_pipeline.py` — interval + phrase biasing
- No breaking changes; profile is optional and pipelines degrade gracefully when absent
- Tests:
  - `tests/generators/midi/test_style_reference.py`
    - Feature extraction produces valid `StyleProfile` from a known MIDI file
    - Aggregate across multiple profiles weighted correctly
    - Profile caching round-trips via YAML
    - Graceful degradation when no local files exist
  - Pipeline tests: biasing moves generation statistics in the expected direction
    (lower note density when style profile has low density, etc.)
