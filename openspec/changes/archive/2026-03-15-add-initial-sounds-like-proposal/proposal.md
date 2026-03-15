# Change: Add Initial Sounds-Like Bootstrap at Production Dir Creation

## Why
`sounds_like` is currently only populated in `composition_proposal.yml`, which is generated
*after* all MIDI phases (chords → drums → bass → melody) have already run. By that point,
artist style context cannot influence any generation step. The chord, drum, bass, and melody
pipelines all receive no artist reference context, and the lyric pipeline explicitly zeros
out `sounds_like` because there is no reliable source for it at that point.

The root cause: there is no production-dir initialization step that derives `sounds_like`
from the song proposal before generation begins. The song proposal YAMLs (e.g.
`cultural_detector_living_v2.yml`) carry rich concept, color, genre, and mood data that
Claude can use to suggest reference artists — but nothing reads them for this purpose
at the right moment in the chain.

## What Changes
- New CLI command `init_production.py` (or `--init` flag on chord_pipeline) that runs
  before any MIDI phase and writes `initial_proposal.yml` to the production directory
- `initial_proposal.yml` contains: `sounds_like` (Claude-generated list of artist name
  strings), `color`, `concept`, `singer`, `key`, `bpm`, `time_sig` — no loop inventory,
  no sections
- All pipeline phases (chord, drum, bass, melody, lyric) read `sounds_like` from
  `initial_proposal.yml` when present, falling back to the song proposal YAML
- `composition_proposal.py` reads `initial_proposal.yml` as its starting point and may
  extend or replace `sounds_like` in its output
- Artist names in `initial_proposal.yml` are bare (e.g. "Sufjan Stevens") not annotated
  strings, so they are directly usable by `load_artist_context()`

## Impact
- Affected specs: `sounds-like-bootstrap` (new), `chord-generation` (MODIFIED — reads
  sounds_like from initial_proposal.yml), `lyric-generation` (MODIFIED — no longer zeros
  out sounds_like)
- Affected code:
  - New: `app/generators/midi/production/init_production.py`
  - Modified: `app/generators/midi/pipelines/chord_pipeline.py` — sounds_like loading
  - Modified: `app/generators/midi/pipelines/lyric_pipeline.py` — remove sounds_like=[]
  - Modified: `app/generators/midi/production/composition_proposal.py` — seed from
    initial_proposal.yml
  - New: `tests/generators/midi/test_init_production.py`
