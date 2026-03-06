# Change: Add ACE Studio MCP Integration

## Why

The melody pipeline outputs MIDI + approved lyrics, which are currently imported into
ACE Studio by hand: open ACE Studio, create a project, import the MIDI, assign the
singer, paste lyrics section by section. That handoff is the last remaining manual step
between the pipeline and a rendered vocal take.

ACE Studio 2.0 ships an experimental MCP server at `http://localhost:21572/mcp`. If it
exposes the right primitives — track creation, MIDI import, lyric assignment, singer
selection — we can automate the handoff entirely from the pipeline.

This is an **exploratory / feasibility-gated change**. The ACE Studio MCP is documented
only at the prompt level; actual tool names and parameters are not published. The first
task is discovery. If the required tools are absent or insufficient, we stop and document
the gap rather than build a brittle workaround.

## What Changes

### Phase 1 — Discovery (always runs, gated)
- **`app/reference/mcp/ace_studio/probe.py`** — connect to the running ACE Studio MCP,
  list available tools with their schemas, write `tool_manifest.json` to the same
  directory. Bail-out if server unreachable or required tools absent (see Feasibility
  Gate below).

### Phase 2 — Client wrapper (conditional on Phase 1 passing)
- **`app/reference/mcp/ace_studio/client.py`** — thin Python client over the ACE Studio
  MCP Streamable HTTP transport. Wraps the minimum required tools into named, typed
  methods used by the pipeline.

### Phase 3 — Pipeline integration (conditional on Phase 2)
- **`app/generators/midi/ace_studio_export.py`** — reads the assembled melody MIDI
  (`assembled/assembled_melody.mid`), the approved lyrics (`melody/lyrics.txt`), and the
  production plan (singer, BPM, key, song title) and calls the ACE Studio MCP to
  create/update a project, import MIDI, and assign lyrics.

## Feasibility Gate

After Phase 1 probe, the integration **proceeds only if all of the following tools are
confirmed present** with usable schemas:

| Required capability | Expected tool (name TBC from probe) |
|---------------------|--------------------------------------|
| List/get project info | `get_project` or equivalent |
| Create or open a project | `create_project` or equivalent |
| Add a vocal track with singer assignment | `add_track` / `set_singer` or equivalent |
| Import or set MIDI on a track | `import_midi` / `set_midi` or equivalent |
| Assign lyrics to a section | `set_lyrics` or equivalent |

If any required capability is missing: **stop, write `FEASIBILITY.md` documenting the
gap, and do not proceed to Phase 2 or 3.**

Optional but desirable (integration proceeds without these):
- Trigger render / bounce
- Set BPM / time signature
- Set key

## Impact

- **New directory**: `app/reference/mcp/ace_studio/`
- **New file** (conditional): `app/generators/midi/ace_studio_export.py`
- **New spec**: `ace-studio-mcp`
- **No changes to existing pipeline files** — integration is additive, not replacing
  the manual workflow
- Requires ACE Studio 2.0 running locally; pipeline step is a no-op if server
  unreachable

## Risks

- ACE Studio MCP is explicitly marked experimental; tool shapes may change between
  ACE Studio releases
- Streamable HTTP transport (not stdio) — requires `httpx` or `requests`, not the
  `mcp` FastMCP client library used by our servers
- No authentication mechanism documented; assumes local trust (localhost only)
- Lyric assignment granularity unknown — may not support per-section assignment
  matching our `[section_name]` format
