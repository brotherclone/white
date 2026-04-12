# ACE Studio MCP — Status

**Status**: OPERATIONAL — 71 tools discovered, client in active use

`tool_manifest.json` contains 71 tools covering playback, tracks, MIDI, lyrics,
singers, and project management. The client is wired into the production pipeline
via `wpipe ace export/status/import`.

## How the integration works

**export** — pushes assembled melody MIDI + approved lyrics into ACE Studio:
- Opens a project, finds an empty track, resolves the singer via `singer_voices.yml`
- Loads MIDI section by section (one clip per arrangement section)
- Writes `ace_studio` block to `song_context.yml` with project/track/singer metadata

**status** — reads `song_context.yml` and prints the current ACE Studio association
(project name, track index, singer, exported sections, render path)

**import** — ingests the ACE Studio WAV render back into the production dir:
- Searches for `VocalSynthv*/VocalSynthv*_*.wav` or `*.wav` matching VocalSynth pattern
- Copies to `melody/ace_render.wav`
- Updates `song_context.yml` with the render path

## Usage

```
wpipe ace export --production-dir <path>   # push MIDI + lyrics to ACE Studio
wpipe ace status --production-dir <path>   # check association
wpipe ace import --production-dir <path>   # ingest WAV render
```

ACE Studio 2.0 must be running locally. The MCP server listens at
`http://localhost:21572/mcp`. Re-run `python -m app.reference.mcp.ace_studio.probe`
after any ACE Studio update to refresh `tool_manifest.json`.
