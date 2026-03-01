#!/usr/bin/env python3
"""
ACE Studio MCP Probe — Phase 1 feasibility gate.

Connects to the ACE Studio MCP server at http://localhost:21572/mcp,
lists available tools, checks for required capabilities, and writes either:

  tool_manifest.json  — on success (all required capabilities found)
  FEASIBILITY.md      — on failure (missing capabilities or unreachable)

Exit codes:
  0  — all required capabilities present; safe to proceed to Phase 2
  1  — server unreachable, or one or more required capabilities missing

Usage:
    python -m app.reference.mcp.ace_studio.probe
    python -m app.reference.mcp.ace_studio.probe --url http://localhost:21572/mcp
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

import httpx

HERE = Path(__file__).parent

ACE_STUDIO_URL = "http://localhost:21572/mcp"
TOOL_MANIFEST_PATH = HERE / "tool_manifest.json"
FEASIBILITY_PATH = HERE / "FEASIBILITY.md"
CONNECT_TIMEOUT = 5.0
REQUEST_TIMEOUT = 15.0

# Capability keywords that must each match at least one tool name.
# We check by substring (case-insensitive) to be robust against exact naming.
REQUIRED_KEYWORDS: list[str] = [
    "project",
    "track",
    "midi",
    "lyric",
    "singer",
]

# Optional but desirable — not a gate condition
OPTIONAL_KEYWORDS: list[str] = [
    "render",
    "bpm",
    "key",
]


# ---------------------------------------------------------------------------
# MCP JSON-RPC helpers
# ---------------------------------------------------------------------------


def _parse_response(resp: httpx.Response) -> dict:
    """Parse JSON or first SSE data line from a response."""
    content_type = resp.headers.get("content-type", "")
    if "text/event-stream" in content_type:
        for line in resp.text.splitlines():
            line = line.strip()
            if line.startswith("data:"):
                return json.loads(line[5:].strip())
        raise ValueError("SSE stream contained no data: line")
    return resp.json()


def _post(
    client: httpx.Client,
    url: str,
    payload: dict,
    session_id: Optional[str] = None,
) -> dict:
    """POST a JSON-RPC message; parse JSON or first SSE data line."""
    headers: dict[str, str] = {
        "Content-Type": "application/json",
        "Accept": "application/json, text/event-stream",
    }
    if session_id:
        headers["Mcp-Session-Id"] = session_id
    resp = client.post(url, json=payload, headers=headers)
    resp.raise_for_status()
    return _parse_response(resp)


def _initialize(client: httpx.Client, url: str) -> tuple[dict, Optional[str]]:
    """Send MCP initialize handshake; return (server_capabilities, session_id)."""
    resp = client.post(
        url,
        json={
            "jsonrpc": "2.0",
            "method": "initialize",
            "id": 1,
            "params": {
                "protocolVersion": "2025-03-26",
                "capabilities": {},
                "clientInfo": {"name": "white-probe", "version": "1.0"},
            },
        },
        headers={
            "Content-Type": "application/json",
            "Accept": "application/json, text/event-stream",
        },
    )
    resp.raise_for_status()
    session_id: Optional[str] = resp.headers.get("Mcp-Session-Id")
    result = _parse_response(resp)
    # Send initialized notification (fire-and-forget; ignore response)
    try:
        _post(
            client,
            url,
            {"jsonrpc": "2.0", "method": "notifications/initialized"},
            session_id=session_id,
        )
    except Exception:
        pass
    return result.get("result", {}), session_id


def _list_tools(
    client: httpx.Client, url: str, session_id: Optional[str] = None
) -> list[dict]:
    """Call tools/list and return the tools array."""
    result = _post(
        client,
        url,
        {
            "jsonrpc": "2.0",
            "method": "tools/list",
            "id": 2,
        },
        session_id=session_id,
    )
    if "error" in result:
        raise RuntimeError(f"tools/list error: {result['error']}")
    return result.get("result", {}).get("tools", [])


# ---------------------------------------------------------------------------
# Feasibility check
# ---------------------------------------------------------------------------


def check_capabilities(tools: list[dict]) -> tuple[set[str], set[str], set[str]]:
    """Return (found_required, missing_required, found_optional).

    Checks both tool names and descriptions so that capabilities described in
    natural language (e.g. "sets the singer for a track") are also matched.
    """
    searchable = " ".join(
        (t.get("name", "") + " " + t.get("description", "")).lower() for t in tools
    )
    found_required = {kw for kw in REQUIRED_KEYWORDS if kw in searchable}
    missing_required = set(REQUIRED_KEYWORDS) - found_required
    found_optional = {kw for kw in OPTIONAL_KEYWORDS if kw in searchable}
    return found_required, missing_required, found_optional


# ---------------------------------------------------------------------------
# Output writers
# ---------------------------------------------------------------------------


def write_manifest(tools: list[dict], server_info: dict) -> None:
    manifest = {
        "server": server_info,
        "tool_count": len(tools),
        "tools": {t["name"]: t for t in tools},
    }
    TOOL_MANIFEST_PATH.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"  Written: {TOOL_MANIFEST_PATH}")


def write_feasibility(
    reason: str,
    missing: Optional[set] = None,
    tools: Optional[list] = None,
) -> None:
    lines = [
        "# ACE Studio MCP — Feasibility Report",
        "",
        f"**Status**: BLOCKED — {reason}",
        "",
    ]
    if missing:
        lines += [
            "## Missing Required Capabilities",
            "",
            "The following capability keywords were not matched by any tool name:",
            "",
        ]
        for kw in sorted(missing):
            lines.append(f"- `{kw}`")
        lines.append("")

    if tools:
        lines += [
            "## Discovered Tools",
            "",
            "These tools *were* found on the server:",
            "",
        ]
        for t in tools:
            desc = t.get("description", "")
            lines.append(
                f"- **{t['name']}** — {desc}" if desc else f"- **{t['name']}**"
            )
        lines.append("")

    lines += [
        "## Next Steps",
        "",
        "- Check ACE Studio release notes for updated tool names.",
        "- Re-run `python -m app.reference.mcp.ace_studio.probe` after an ACE Studio update.",
        "- If capabilities are partially present, consider Phase 2 with reduced scope.",
        "",
        "_Generated by `app/reference/mcp/ace_studio/probe.py`_",
    ]

    FEASIBILITY_PATH.write_text("\n".join(lines), encoding="utf-8")
    print(f"  Written: {FEASIBILITY_PATH}")


# ---------------------------------------------------------------------------
# Main probe
# ---------------------------------------------------------------------------


def run_probe(url: str = ACE_STUDIO_URL) -> int:
    """Run the full probe. Returns exit code (0 = pass, 1 = fail)."""
    print("\nACE Studio MCP Probe")
    print(f"  Target: {url}")
    print()

    # --- Connect ---
    try:
        with httpx.Client(
            timeout=httpx.Timeout(REQUEST_TIMEOUT, connect=CONNECT_TIMEOUT)
        ) as client:
            try:
                server_info, session_id = _initialize(client, url)
            except httpx.ConnectError:
                print("ERROR: Could not connect to ACE Studio MCP server.")
                print(f"       Is ACE Studio 2.0 running? Expected server at {url}")
                write_feasibility("ACE Studio MCP server not reachable at " + url)
                return 1
            except httpx.TimeoutException:
                print("ERROR: Connection timed out.")
                write_feasibility("Connection timed out connecting to " + url)
                return 1

            if session_id:
                print(f"  Session ID: {session_id}")
            print(f"  Connected. Server info: {json.dumps(server_info, indent=4)}")

            # --- List tools ---
            try:
                tools = _list_tools(client, url, session_id=session_id)
            except Exception as e:
                print(f"ERROR: tools/list failed: {e}")
                write_feasibility(f"tools/list call failed: {e}")
                return 1

    except httpx.HTTPStatusError as e:
        print(f"ERROR: HTTP {e.response.status_code} from server: {e}")
        write_feasibility(f"HTTP error {e.response.status_code} from server")
        return 1

    if not tools:
        print("ERROR: Server returned zero tools.")
        write_feasibility("Server returned an empty tools list")
        return 1

    print(f"\n  Discovered {len(tools)} tool(s):")
    for t in tools:
        print(f"    {t['name']}")
        if t.get("description"):
            print(f"      {t['description'][:80]}")

    # --- Capability check ---
    found_required, missing_required, found_optional = check_capabilities(tools)

    print(f"\n  Required capabilities ({len(REQUIRED_KEYWORDS)}):")
    for kw in REQUIRED_KEYWORDS:
        status = "PASS" if kw in found_required else "FAIL"
        print(f"    [{status}] {kw}")

    print(f"\n  Optional capabilities ({len(OPTIONAL_KEYWORDS)}):")
    for kw in OPTIONAL_KEYWORDS:
        status = "yes" if kw in found_optional else " no"
        print(f"    [{status}]  {kw}")

    if missing_required:
        print(
            f"\n  FEASIBILITY GATE FAILED — missing: {', '.join(sorted(missing_required))}"
        )
        write_feasibility(
            f"Required capabilities not found: {', '.join(sorted(missing_required))}",
            missing=missing_required,
            tools=tools,
        )
        print(
            "\n  Do not proceed to Phase 2 until ACE Studio exposes all required tools."
        )
        return 1

    # --- All good ---
    print("\n  FEASIBILITY GATE PASSED — all required capabilities present.")
    write_manifest(tools, server_info)
    print("\n  Proceed to Phase 2 (client wrapper).")
    return 0


def main():
    parser = argparse.ArgumentParser(
        description="Probe ACE Studio MCP server for required capabilities"
    )
    parser.add_argument(
        "--url",
        default=ACE_STUDIO_URL,
        help=f"MCP server URL (default: {ACE_STUDIO_URL})",
    )
    args = parser.parse_args()
    sys.exit(run_probe(args.url))


if __name__ == "__main__":
    main()
