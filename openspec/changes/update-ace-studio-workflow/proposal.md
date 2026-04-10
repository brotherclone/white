# Change: ACE Studio Workflow — Reduce Per-Song Friction

## Why

ACE Studio is the most time-consuming manual step in the pipeline. The MCP client
exists (`app/reference/mcp/ace_studio/client.py`) and the export script can push
assembled MIDI + lyrics to an open ACE Studio project. But several friction points
remain that require manual intervention on every song:

1. **Singer lookup is manual.** Finding the right singer ID requires probing the
   ACE Studio API, searching by name, reading back a list of results, and copying
   the ID into the export call. This happens every run, for every song.

2. **Track state is opaque.** Before export, the operator has to manually verify
   which ACE Studio track is empty and ready, and pass that track index. If the
   project has been modified, this breaks silently.

3. **Export is all-or-nothing.** The current export pushes one big clip spanning
   the entire assembled melody. ACE Studio shows this as a single undifferentiated
   block. Section boundaries (verse, chorus, bridge) are invisible in the editor,
   making lyric editing and per-section adjustments awkward.

4. **No import path from ACE Studio back to the pipeline.** After rendering in
   ACE Studio, the rendered audio has to be manually located, renamed, and dropped
   into the production directory. There's no canonical path for the render — it ends
   up wherever ACE Studio puts it.

5. **No session persistence.** Every export starts from a blank ACE Studio project
   because there's no way to resume or identify the project associated with a
   production directory. If a session is interrupted, the operator starts from
   scratch.

## What Changes

### Singer registry — local cache of singer IDs

`app/reference/mcp/ace_studio/singer_voices.yml` already exists as a registry.
Extend it to be the canonical source of truth for singer → ACE Studio ID mappings,
with lazy refresh:

```yaml
singers:
  Busyayo:
    ace_id: 42
    group: "ACE Virtual"
    last_verified: "2026-04-10"
  Gabriel:
    ace_id: 17
    group: "ACE Virtual"
    last_verified: "2026-04-10"
```

`ace_studio_export.py` looks up singer by name in the registry first. If not found
or `--refresh-singers` flag passed, it queries ACE Studio live and updates the
registry. Singer lookup becomes zero-friction for known singers.

### Auto track selection

`AceStudioClient.find_available_track()` — scans the track list and returns the
first track with no clips (or the first track if all have clips, with a warning).
Export script uses this automatically. Operator no longer needs to know or pass
a track index.

Optionally: `--track-name <name>` to target a specific named track (useful when
the project has a designated vocal track).

### Section-aware export

Instead of one clip spanning the full assembly, export N clips — one per vocal
section in the arrangement. Each clip is named after its section label
(`verse_1`, `chorus_1`, etc.) and placed at the correct tick position.

This requires knowing section boundaries: they are read from `production_plan.yml`
(already has bar counts per section) and converted to tick positions using the
song BPM and time signature.

Result: ACE Studio shows named clips for each section, making editing and
lyric review match the arrangement structure the operator already knows.

**New `AceStudioClient` method:**

```python
def add_section_clips(
    self,
    sections: list[dict],  # [{name, start_tick, dur_ticks, notes, lyrics}, ...]
    track_index: int,
    language: str = "ENG",
) -> list[dict]:
    """Add one clip per section, each with notes and lyrics pre-loaded."""
```

### ACE Studio project association

`song_context.yml` gains an `ace_studio` block written by the export script:

```yaml
ace_studio:
  project_name: "VocalSynthv0"
  exported_at: "2026-04-10T14:22:00Z"
  track_index: 0
  singer: "Busyayo"
  sections_exported: [verse_1, chorus_1, verse_2, chorus_2, outro]
  render_path: null  # filled in by ace_studio_import
```

On re-export, the script detects the existing association and warns before
overwriting (or use `--force` to overwrite).

### Import path — render location + ingest

`ace_studio_import.py` gains a `--locate-render` mode:

```
python -m app.generators.midi.production.ace_studio_import \
    --production-dir <path> \
    --locate-render
```

This queries ACE Studio for the most recent export path for the associated project,
copies the render to `melody/ace_render.wav` in the production directory, and writes
the path to `song_context.yml` under `ace_studio.render_path`.

The `pipeline_runner` can call this automatically after a "render complete"
confirmation prompt, closing the loop between ACE Studio and the pipeline.

### `ace` subcommand on the pipeline runner

Convenience wrapper so ACE Studio operations fit the orchestrator flow:

```
pipeline ace export --production-dir <path>
pipeline ace status --production-dir <path>   # show export association
pipeline ace import --production-dir <path>   # locate and ingest render
```

## Impact

- Affected specs: `ace-studio-mcp`, `init-production`
- Modified files:
  - `app/reference/mcp/ace_studio/client.py` — `find_available_track()`,
    `add_section_clips()`
  - `app/reference/mcp/ace_studio/singer_voices.yml` — extend schema, add refresh logic
  - `app/generators/midi/production/ace_studio_export.py` — registry lookup,
    auto track, section-aware clip placement, song_context write
  - `app/generators/midi/production/ace_studio_import.py` — `--locate-render` mode
  - `app/generators/midi/production/pipeline_runner.py` — `ace` subcommand
    (depends on `add-pipeline-orchestrator`)
  - `app/generators/midi/production/init_production.py` — song_context ace block init
- Tests:
  - `tests/reference/mcp/test_ace_studio_client.py` — track selection, section clips
  - `tests/generators/midi/production/test_ace_studio_export.py` — registry lookup,
    section export, song_context write
  - `tests/generators/midi/production/test_ace_studio_import.py` — locate-render
