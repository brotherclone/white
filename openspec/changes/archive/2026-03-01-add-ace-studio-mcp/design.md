# Design: ACE Studio MCP Integration

## Transport

ACE Studio MCP uses **Streamable HTTP** at `http://localhost:21572/mcp`. This is MCP
protocol 2025-03-26 (not the legacy 2024-11-05 SSE variant). The Python `mcp` library
≥ 1.6 supports Streamable HTTP via `streamablehttp_client`. Alternatively a raw
`httpx` client against the JSON-RPC endpoint works for simple tool calls.

```
Client (our pipeline) ──POST──► http://localhost:21572/mcp
                     ◄──JSON────
```

No authentication. Localhost trust only.

## Probe strategy

`probe.py` sends `tools/list` JSON-RPC to enumerate available tools and writes the
result to `tool_manifest.json`. The feasibility check is a simple set intersection:
required capability names (flexible — we check by capability keyword, not exact name)
vs. what's in the manifest.

```python
# Pseudocode
tools = mcp_call("tools/list")
tool_manifest = {t["name"]: t for t in tools}
# write tool_manifest.json

required_keywords = ["project", "track", "midi", "lyric", "singer"]
found = {kw for kw in required_keywords
         if any(kw in name for name in tool_manifest)}
if found != set(required_keywords):
    write_feasibility_md(missing=required_keywords - found)
    sys.exit(1)
```

## Client wrapper design

`client.py` wraps individual tool calls into methods. Since tool names are unknown until
probe, the wrapper is built from `tool_manifest.json` at runtime rather than
hardcoded names:

```python
class AceStudioClient:
    def __init__(self, base_url="http://localhost:21572/mcp"):
        self._url = base_url
        self._manifest = self._load_manifest()

    def _call(self, tool_name: str, **kwargs) -> dict: ...
    def _find_tool(self, *keywords: str) -> str: ...  # fuzzy match from manifest

    def get_project_info(self) -> dict:
        return self._call(self._find_tool("project", "info"))

    def create_project(self, title, bpm, key, time_sig) -> dict: ...
    def add_vocal_track(self, singer: str) -> dict: ...
    def import_midi(self, track_id: str, midi_path: str) -> dict: ...
    def set_lyrics(self, track_id: str, lyrics: str) -> dict: ...
```

This keeps Phase 2 decoupled from the exact tool names discovered in Phase 1.

## Pipeline integration design

`ace_studio_export.py` reads:
- `production_dir/assembled/assembled_melody.mid`
- `production_dir/melody/lyrics.txt`
- `production_dir/production_plan.yml` (singer, BPM, key, song_title)

And calls, in order:
1. `client.create_project(title, bpm, key, time_sig)` → `project_id`
2. `client.add_vocal_track(singer)` → `track_id`
3. `client.import_midi(track_id, assembled_melody_path)`
4. `client.set_lyrics(track_id, lyrics_text)`

If the ACE Studio MCP server is unreachable (connection refused), the function logs a
warning and returns early — the pipeline continues without blocking on vocal export.

## Bail-out conditions

| Condition | Action |
|-----------|--------|
| Server unreachable (connection refused) | Log warning, return early — not an error |
| `tools/list` returns empty or errors | Write `FEASIBILITY.md`, exit probe with code 1 |
| Required capability keyword absent from manifest | Write `FEASIBILITY.md`, exit probe with code 1 |
| Required tool found but schema incompatible | Note in `FEASIBILITY.md`, mark as partial |
| ACE Studio version < 2.0 | Out of scope — assume user has correct version |

## File layout

```
app/reference/mcp/ace_studio/
  __init__.py
  probe.py              # Phase 1: discovery + feasibility gate
  client.py             # Phase 2: thin HTTP wrapper (conditional)
  tool_manifest.json    # Written by probe.py, gitignored (machine-specific)
  FEASIBILITY.md        # Written by probe.py if gate fails (committed if present)

app/generators/midi/
  ace_studio_export.py  # Phase 3: pipeline integration (conditional)

tests/reference/mcp/
  test_ace_studio_probe.py   # Tests probe against mock HTTP server
  test_ace_studio_client.py  # Tests client wrapper against mock HTTP server
  test_ace_studio_export.py  # Tests export against mock client
```

`tool_manifest.json` is gitignored because it reflects a running local instance.
`FEASIBILITY.md` is committed if it exists — documents why we stopped.
