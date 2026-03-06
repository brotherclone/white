"""
Tests for app.reference.mcp.ace_studio.probe

Uses unittest.mock to patch httpx.Client so no live ACE Studio instance is needed.
"""

import json
from unittest.mock import MagicMock, patch

import httpx

from app.reference.mcp.ace_studio.probe import (
    REQUIRED_KEYWORDS,
    check_capabilities,
    run_probe,
    write_feasibility,
    write_manifest,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

TOOL_LIST_ALL = [
    {"name": "get_project_info", "description": "Get current project metadata"},
    {"name": "create_project", "description": "Create a new project"},
    {"name": "add_track", "description": "Add a track with singer assignment"},
    {"name": "import_midi", "description": "Import MIDI data to a track"},
    {"name": "set_lyrics", "description": "Assign lyrics to a vocal track"},
]

TOOL_LIST_MISSING_LYRIC = [
    {"name": "get_project_info", "description": ""},
    {"name": "create_project", "description": ""},
    {"name": "add_track", "description": ""},
    {"name": "import_midi", "description": ""},
    # no lyric or singer tool
]


def _mock_client(tools: list[dict], connect_error: bool = False):
    """Return a mock httpx.Client context manager."""
    mock_client = MagicMock()

    if connect_error:
        mock_client.__enter__.return_value.post.side_effect = httpx.ConnectError(
            "refused"
        )
        return mock_client

    def fake_post(url, json=None, headers=None):
        resp = MagicMock()
        resp.raise_for_status = MagicMock()
        resp.headers = {"content-type": "application/json"}
        method = (json or {}).get("method", "")
        if method == "initialize":
            resp.json.return_value = {
                "jsonrpc": "2.0",
                "id": 1,
                "result": {"serverInfo": {"name": "ACE Studio MCP", "version": "0.1"}},
            }
        elif method == "notifications/initialized":
            resp.json.return_value = {"jsonrpc": "2.0", "result": None}
        elif method == "tools/list":
            resp.json.return_value = {
                "jsonrpc": "2.0",
                "id": 2,
                "result": {"tools": tools},
            }
        else:
            resp.json.return_value = {"jsonrpc": "2.0", "result": {}}
        return resp

    mock_client.__enter__.return_value.post.side_effect = fake_post
    return mock_client


# ---------------------------------------------------------------------------
# check_capabilities
# ---------------------------------------------------------------------------


class TestCheckCapabilities:
    def test_all_required_present(self):
        found, missing, _ = check_capabilities(TOOL_LIST_ALL)
        assert missing == set()
        assert found == set(REQUIRED_KEYWORDS)

    def test_missing_lyric_and_singer(self):
        found, missing, _ = check_capabilities(TOOL_LIST_MISSING_LYRIC)
        assert "lyric" in missing
        assert "singer" in missing

    def test_empty_tools_all_missing(self):
        _, missing, _ = check_capabilities([])
        assert missing == set(REQUIRED_KEYWORDS)

    def test_case_insensitive_match(self):
        tools = [{"name": "SET_LYRICS_FOR_TRACK"}, {"name": "ASSIGN_SINGER"}]
        found, missing, _ = check_capabilities(tools)
        assert "lyric" in found
        assert "singer" in found

    def test_optional_keywords_detected(self):
        tools = TOOL_LIST_ALL + [{"name": "set_bpm"}, {"name": "trigger_render"}]
        _, _, found_opt = check_capabilities(tools)
        assert "bpm" in found_opt
        assert "render" in found_opt

    def test_optional_not_required(self):
        _, missing, _ = check_capabilities(TOOL_LIST_ALL)
        # optional keywords missing should not cause gate failure
        assert missing == set()


# ---------------------------------------------------------------------------
# write_manifest / write_feasibility
# ---------------------------------------------------------------------------


class TestWriteManifest:
    def test_writes_json_file(self, tmp_path):
        with patch(
            "app.reference.mcp.ace_studio.probe.TOOL_MANIFEST_PATH",
            tmp_path / "tool_manifest.json",
        ):
            write_manifest(TOOL_LIST_ALL, {"serverInfo": {"name": "ACE Studio"}})
            manifest = json.loads((tmp_path / "tool_manifest.json").read_text())
        assert manifest["tool_count"] == len(TOOL_LIST_ALL)
        assert "get_project_info" in manifest["tools"]

    def test_manifest_keyed_by_name(self, tmp_path):
        with patch(
            "app.reference.mcp.ace_studio.probe.TOOL_MANIFEST_PATH",
            tmp_path / "tool_manifest.json",
        ):
            write_manifest(TOOL_LIST_ALL, {})
            manifest = json.loads((tmp_path / "tool_manifest.json").read_text())
        for t in TOOL_LIST_ALL:
            assert t["name"] in manifest["tools"]


class TestWriteFeasibility:
    def test_writes_markdown_file(self, tmp_path):
        with patch(
            "app.reference.mcp.ace_studio.probe.FEASIBILITY_PATH",
            tmp_path / "FEASIBILITY.md",
        ):
            write_feasibility("server unreachable")
            content = (tmp_path / "FEASIBILITY.md").read_text()
        assert "BLOCKED" in content
        assert "server unreachable" in content

    def test_lists_missing_capabilities(self, tmp_path):
        with patch(
            "app.reference.mcp.ace_studio.probe.FEASIBILITY_PATH",
            tmp_path / "FEASIBILITY.md",
        ):
            write_feasibility("missing caps", missing={"lyric", "singer"})
            content = (tmp_path / "FEASIBILITY.md").read_text()
        assert "lyric" in content
        assert "singer" in content

    def test_lists_discovered_tools(self, tmp_path):
        with patch(
            "app.reference.mcp.ace_studio.probe.FEASIBILITY_PATH",
            tmp_path / "FEASIBILITY.md",
        ):
            write_feasibility(
                "missing caps", missing={"lyric"}, tools=TOOL_LIST_MISSING_LYRIC
            )
            content = (tmp_path / "FEASIBILITY.md").read_text()
        assert "import_midi" in content


# ---------------------------------------------------------------------------
# run_probe
# ---------------------------------------------------------------------------


class TestRunProbe:
    def test_returns_0_when_all_capabilities_present(self, tmp_path):
        with (
            patch(
                "app.reference.mcp.ace_studio.probe.TOOL_MANIFEST_PATH",
                tmp_path / "tool_manifest.json",
            ),
            patch(
                "app.reference.mcp.ace_studio.probe.FEASIBILITY_PATH",
                tmp_path / "FEASIBILITY.md",
            ),
            patch("httpx.Client", return_value=_mock_client(TOOL_LIST_ALL)),
        ):
            code = run_probe()
        assert code == 0
        assert (tmp_path / "tool_manifest.json").exists()
        assert not (tmp_path / "FEASIBILITY.md").exists()

    def test_returns_1_when_capability_missing(self, tmp_path):
        with (
            patch(
                "app.reference.mcp.ace_studio.probe.TOOL_MANIFEST_PATH",
                tmp_path / "tool_manifest.json",
            ),
            patch(
                "app.reference.mcp.ace_studio.probe.FEASIBILITY_PATH",
                tmp_path / "FEASIBILITY.md",
            ),
            patch("httpx.Client", return_value=_mock_client(TOOL_LIST_MISSING_LYRIC)),
        ):
            code = run_probe()
        assert code == 1
        assert (tmp_path / "FEASIBILITY.md").exists()
        assert not (tmp_path / "tool_manifest.json").exists()

    def test_returns_1_when_server_unreachable(self, tmp_path):
        with (
            patch(
                "app.reference.mcp.ace_studio.probe.TOOL_MANIFEST_PATH",
                tmp_path / "tool_manifest.json",
            ),
            patch(
                "app.reference.mcp.ace_studio.probe.FEASIBILITY_PATH",
                tmp_path / "FEASIBILITY.md",
            ),
            patch("httpx.Client", return_value=_mock_client([], connect_error=True)),
        ):
            code = run_probe()
        assert code == 1
        assert (tmp_path / "FEASIBILITY.md").exists()

    def test_returns_1_when_empty_tool_list(self, tmp_path):
        with (
            patch(
                "app.reference.mcp.ace_studio.probe.TOOL_MANIFEST_PATH",
                tmp_path / "tool_manifest.json",
            ),
            patch(
                "app.reference.mcp.ace_studio.probe.FEASIBILITY_PATH",
                tmp_path / "FEASIBILITY.md",
            ),
            patch("httpx.Client", return_value=_mock_client([])),
        ):
            code = run_probe()
        assert code == 1

    def test_manifest_not_written_on_failure(self, tmp_path):
        with (
            patch(
                "app.reference.mcp.ace_studio.probe.TOOL_MANIFEST_PATH",
                tmp_path / "tool_manifest.json",
            ),
            patch(
                "app.reference.mcp.ace_studio.probe.FEASIBILITY_PATH",
                tmp_path / "FEASIBILITY.md",
            ),
            patch("httpx.Client", return_value=_mock_client(TOOL_LIST_MISSING_LYRIC)),
        ):
            run_probe()
        assert not (tmp_path / "tool_manifest.json").exists()

    def test_sse_response_parsed(self, tmp_path):
        """Probe handles SSE text/event-stream responses from the server."""
        import json as _json

        def fake_post_sse(url, json=None, headers=None):
            resp = MagicMock()
            resp.raise_for_status = MagicMock()
            resp.headers = {"content-type": "text/event-stream"}
            method = (json or {}).get("method", "")
            if method == "initialize":
                data = {"jsonrpc": "2.0", "id": 1, "result": {"serverInfo": {}}}
            elif method == "tools/list":
                data = {"jsonrpc": "2.0", "id": 2, "result": {"tools": TOOL_LIST_ALL}}
            else:
                data = {"jsonrpc": "2.0", "result": None}
            resp.text = f"data: {_json.dumps(data)}\n\n"
            return resp

        mock_client = MagicMock()
        mock_client.__enter__.return_value.post.side_effect = fake_post_sse

        with (
            patch(
                "app.reference.mcp.ace_studio.probe.TOOL_MANIFEST_PATH",
                tmp_path / "tool_manifest.json",
            ),
            patch(
                "app.reference.mcp.ace_studio.probe.FEASIBILITY_PATH",
                tmp_path / "FEASIBILITY.md",
            ),
            patch("httpx.Client", return_value=mock_client),
        ):
            code = run_probe()
        assert code == 0
