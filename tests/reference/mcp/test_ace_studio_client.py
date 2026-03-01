"""
Tests for app.reference.mcp.ace_studio.client

Uses unittest.mock to patch httpx so no live ACE Studio instance is needed.
"""

import json
from unittest.mock import MagicMock, patch

import httpx
import pytest

from app.reference.mcp.ace_studio.client import AceStudioClient

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

MINIMAL_TOOLS = {
    "get_project_status_info": {
        "name": "get_project_status_info",
        "description": "Returns basic information about the current project",
    },
    "set_tempo_automation": {
        "name": "set_tempo_automation",
        "description": "Sets the tempo automation of the current project",
    },
    "set_timesignature_list": {
        "name": "set_timesignature_list",
        "description": "Sets the time signatures of the current project",
    },
    "get_content_track_basic_info_list": {
        "name": "get_content_track_basic_info_list",
        "description": "Returns a list of basic information for all content tracks",
    },
    "get_available_sound_source_list": {
        "name": "get_available_sound_source_list",
        "description": "Get available sound sources including singers",
    },
    "load_new_sound_source_on_track": {
        "name": "load_new_sound_source_on_track",
        "description": "Load a singer onto a track",
    },
    "add_new_clip": {
        "name": "add_new_clip",
        "description": "Places a new empty note clip on a track",
    },
    "ask_editor_to_open": {
        "name": "ask_editor_to_open",
        "description": "Requests the pattern editor window to open",
    },
    "add_notes_in_editor": {
        "name": "add_notes_in_editor",
        "description": "Adds notes to the current pattern editor",
    },
    "get_note_clip_lyrics": {
        "name": "get_note_clip_lyrics",
        "description": "Returns sentence-level lyrics for a Sing clip",
    },
}


def _tool_resp(data: dict) -> MagicMock:
    """Build a mock httpx.Response for a successful tools/call result."""
    resp = MagicMock()
    resp.raise_for_status = MagicMock()
    resp.headers = {"content-type": "application/json"}
    resp.json.return_value = {
        "jsonrpc": "2.0",
        "id": 1,
        "result": {
            "content": [{"type": "text", "text": json.dumps(data)}],
            "isError": False,
        },
    }
    return resp


def _make_client() -> AceStudioClient:
    """Return a pre-connected client with minimal manifest and mock HTTP layer."""
    client = AceStudioClient.__new__(AceStudioClient)
    client._url = "http://localhost:21572/mcp"
    client._session_id = "test-session-id"
    client._call_id = 0
    client._tools = dict(MINIMAL_TOOLS)
    client._connected = True
    client._timeout = httpx.Timeout(30.0)

    # Inject a mock via the _make_http_client seam so _raw_post uses it.
    mock_http = MagicMock()
    mock_ctx = MagicMock()
    mock_ctx.__enter__ = MagicMock(return_value=mock_http)
    mock_ctx.__exit__ = MagicMock(return_value=False)
    client._make_http_client = MagicMock(return_value=mock_ctx)
    # Expose mock_http as _http so tests can set return values the same way.
    client._http = mock_http
    return client


# ---------------------------------------------------------------------------
# _find_tool
# ---------------------------------------------------------------------------


class TestFindTool:
    def test_single_keyword_match(self):
        client = _make_client()
        assert client._find_tool("project") == "get_project_status_info"

    def test_multi_keyword_narrows_match(self):
        client = _make_client()
        assert client._find_tool("new", "clip") == "add_new_clip"

    def test_case_insensitive(self):
        client = _make_client()
        assert client._find_tool("PROJECT") == "get_project_status_info"

    def test_not_found_raises_key_error(self):
        client = _make_client()
        with pytest.raises(KeyError, match="nonexistent"):
            client._find_tool("nonexistent")

    def test_returns_first_match(self):
        client = _make_client()
        # Both "get_content_track_basic_info_list" and others contain "track"
        result = client._find_tool("track", "basic")
        assert "track" in result and "basic" in result


# ---------------------------------------------------------------------------
# _call dispatch
# ---------------------------------------------------------------------------


class TestCall:
    def test_posts_tools_call_method(self):
        client = _make_client()
        client._http.post.return_value = _tool_resp({"name": "test"})
        client._call("get_project_status_info")
        payload = client._http.post.call_args.kwargs["json"]
        assert payload["method"] == "tools/call"

    def test_passes_tool_name_in_params(self):
        client = _make_client()
        client._http.post.return_value = _tool_resp({})
        client._call("get_project_status_info")
        payload = client._http.post.call_args.kwargs["json"]
        assert payload["params"]["name"] == "get_project_status_info"

    def test_passes_kwargs_as_arguments(self):
        client = _make_client()
        client._http.post.return_value = _tool_resp({})
        client._call("set_tempo_automation", points=[{"pos": 0, "value": 120}])
        payload = client._http.post.call_args.kwargs["json"]
        assert payload["params"]["arguments"]["points"][0]["value"] == 120

    def test_includes_session_id_header(self):
        client = _make_client()
        client._http.post.return_value = _tool_resp({})
        client._call("get_project_status_info")
        headers = client._http.post.call_args.kwargs["headers"]
        assert headers["Mcp-Session-Id"] == "test-session-id"

    def test_raises_on_mcp_error(self):
        client = _make_client()
        resp = MagicMock()
        resp.raise_for_status = MagicMock()
        resp.headers = {"content-type": "application/json"}
        resp.json.return_value = {
            "jsonrpc": "2.0",
            "id": 1,
            "error": {"code": -32601, "message": "Method not found"},
        }
        client._http.post.return_value = resp
        with pytest.raises(RuntimeError, match="MCP error"):
            client._call("get_project_status_info")

    def test_raises_on_is_error_true(self):
        client = _make_client()
        resp = MagicMock()
        resp.raise_for_status = MagicMock()
        resp.headers = {"content-type": "application/json"}
        resp.json.return_value = {
            "jsonrpc": "2.0",
            "id": 1,
            "result": {
                "content": [{"type": "text", "text": "tool failed"}],
                "isError": True,
            },
        }
        client._http.post.return_value = resp
        with pytest.raises(RuntimeError, match="returned error"):
            client._call("get_project_status_info")

    def test_unwraps_json_text_content(self):
        client = _make_client()
        client._http.post.return_value = _tool_resp({"projectName": "My Song"})
        result = client._call("get_project_status_info")
        assert result["projectName"] == "My Song"


# ---------------------------------------------------------------------------
# High-level methods
# ---------------------------------------------------------------------------


class TestGetProjectInfo:
    def test_returns_dict(self):
        client = _make_client()
        client._http.post.return_value = _tool_resp(
            {"projectName": "My Song", "duration": 240}
        )
        result = client.get_project_info()
        assert isinstance(result, dict)
        assert result["projectName"] == "My Song"

    def test_calls_correct_tool(self):
        client = _make_client()
        client._http.post.return_value = _tool_resp({})
        client.get_project_info()
        payload = client._http.post.call_args.kwargs["json"]
        assert payload["params"]["name"] == "get_project_status_info"


class TestSetTempo:
    def test_passes_bpm_in_points(self):
        client = _make_client()
        client._http.post.return_value = _tool_resp({})
        client.set_tempo(140.0)
        args = client._http.post.call_args.kwargs["json"]["params"]["arguments"]
        assert args["points"][0]["value"] == 140.0
        assert args["points"][0]["pos"] == 0

    def test_calls_correct_tool(self):
        client = _make_client()
        client._http.post.return_value = _tool_resp({})
        client.set_tempo(120.0)
        payload = client._http.post.call_args.kwargs["json"]
        assert payload["params"]["name"] == "set_tempo_automation"


class TestSetTimeSignature:
    def test_passes_numerator_denominator(self):
        client = _make_client()
        client._http.post.return_value = _tool_resp({})
        client.set_time_signature(3, 4)
        args = client._http.post.call_args.kwargs["json"]["params"]["arguments"]
        sig = args["signatures"][0]
        assert sig["numerator"] == 3
        assert sig["denominator"] == 4
        assert sig["barPos"] == 0


class TestListTracks:
    def test_returns_list(self):
        client = _make_client()
        client._http.post.return_value = _tool_resp(
            {"tracks": [{"trackIndex": 0, "name": "Vocal"}]}
        )
        tracks = client.list_tracks()
        assert isinstance(tracks, list)
        assert tracks[0]["name"] == "Vocal"

    def test_handles_bare_list_response(self):
        client = _make_client()
        # If ACE Studio returns a bare list instead of {"tracks": [...]}
        resp = MagicMock()
        resp.raise_for_status = MagicMock()
        resp.headers = {"content-type": "application/json"}
        resp.json.return_value = {
            "jsonrpc": "2.0",
            "id": 1,
            "result": {
                "content": [{"type": "text", "text": json.dumps([{"trackIndex": 0}])}],
                "isError": False,
            },
        }
        client._http.post.return_value = resp
        tracks = client.list_tracks()
        assert isinstance(tracks, list)


class TestFindSinger:
    def test_passes_keyword_and_type(self):
        client = _make_client()
        client._http.post.return_value = _tool_resp({"list": []})
        client.find_singer("Shirley")
        args = client._http.post.call_args.kwargs["json"]["params"]["arguments"]
        assert args["keyword"] == "Shirley"
        assert args["type"] == "sing"

    def test_returns_list_from_response(self):
        client = _make_client()
        client._http.post.return_value = _tool_resp(
            {"list": [{"id": 42, "name": "Shirley"}]}
        )
        result = client.find_singer("Shirley")
        assert result[0]["id"] == 42


class TestLoadSinger:
    def test_passes_required_params(self):
        client = _make_client()
        client._http.post.return_value = _tool_resp({})
        client.load_singer(track_index=0, singer_id=42)
        args = client._http.post.call_args.kwargs["json"]["params"]["arguments"]
        assert args["trackIndex"] == 0
        assert args["id"] == 42
        assert args["soundSourceType"] == "sing"

    def test_passes_router_id_when_given(self):
        client = _make_client()
        client._http.post.return_value = _tool_resp({})
        client.load_singer(track_index=1, singer_id=5, router_id=3)
        args = client._http.post.call_args.kwargs["json"]["params"]["arguments"]
        assert args["routerId"] == 3


class TestAddClip:
    def test_passes_required_fields(self):
        client = _make_client()
        client._http.post.return_value = _tool_resp({})
        client.add_clip(track_index=0, pos=0, dur=3840)
        args = client._http.post.call_args.kwargs["json"]["params"]["arguments"]
        assert args["trackIndex"] == 0
        assert args["pos"] == 0
        assert args["dur"] == 3840
        assert args["type"] == "sing"

    def test_passes_optional_name(self):
        client = _make_client()
        client._http.post.return_value = _tool_resp({})
        client.add_clip(track_index=0, pos=0, dur=3840, name="Verse 1")
        args = client._http.post.call_args.kwargs["json"]["params"]["arguments"]
        assert args["name"] == "Verse 1"

    def test_omits_name_when_not_given(self):
        client = _make_client()
        client._http.post.return_value = _tool_resp({})
        client.add_clip(track_index=0, pos=0, dur=3840)
        args = client._http.post.call_args.kwargs["json"]["params"]["arguments"]
        assert "name" not in args


class TestAddNotesWithLyrics:
    def test_passes_notes_and_lyric_sentence(self):
        client = _make_client()
        client._http.post.return_value = _tool_resp({})
        notes = [{"pitch": 60, "dur": 480}, {"pitch": 62, "dur": 480}]
        client.add_notes_with_lyrics(notes, "Hello world")
        args = client._http.post.call_args.kwargs["json"]["params"]["arguments"]
        assert args["notes"] == notes
        assert args["lyric_sentence"] == "Hello world"
        assert args["language"] == "ENG"

    def test_passes_offset_when_given(self):
        client = _make_client()
        client._http.post.return_value = _tool_resp({})
        client.add_notes_with_lyrics([{"pitch": 60, "dur": 480}], "Hi", offset=960)
        args = client._http.post.call_args.kwargs["json"]["params"]["arguments"]
        assert args["offset"] == 960

    def test_omits_offset_when_not_given(self):
        client = _make_client()
        client._http.post.return_value = _tool_resp({})
        client.add_notes_with_lyrics([{"pitch": 60, "dur": 480}], "Hi")
        args = client._http.post.call_args.kwargs["json"]["params"]["arguments"]
        assert "offset" not in args


# ---------------------------------------------------------------------------
# Connection error
# ---------------------------------------------------------------------------


class TestConnectionError:
    def test_connect_raises_connection_error_when_unreachable(self, tmp_path):
        with patch(
            "app.reference.mcp.ace_studio.client.TOOL_MANIFEST_PATH",
            tmp_path / "tool_manifest.json",
        ):
            client = AceStudioClient()

        # Inject a _make_http_client seam that raises ConnectError on post
        mock_http = MagicMock()
        mock_http.post.side_effect = httpx.ConnectError("refused")
        mock_ctx = MagicMock()
        mock_ctx.__enter__ = MagicMock(return_value=mock_http)
        mock_ctx.__exit__ = MagicMock(return_value=False)
        client._make_http_client = MagicMock(return_value=mock_ctx)

        with pytest.raises(ConnectionError, match="Cannot reach ACE Studio"):
            client.connect()

    def test_not_connected_raises_runtime_error(self):
        client = AceStudioClient.__new__(AceStudioClient)
        client._url = "http://localhost:21572/mcp"
        client._session_id = None
        client._call_id = 0
        client._tools = {}
        client._connected = False
        client._timeout = httpx.Timeout(30.0)
        client._make_http_client = MagicMock()
        with pytest.raises(RuntimeError, match="not connected"):
            client._raw_post({"jsonrpc": "2.0", "method": "tools/call", "id": 1})


# ---------------------------------------------------------------------------
# Manifest loading
# ---------------------------------------------------------------------------


class TestManifestLoading:
    def test_loads_tools_from_manifest(self, tmp_path):
        manifest = {
            "server": {},
            "tool_count": 1,
            "tools": {"my_tool": {"name": "my_tool", "description": ""}},
        }
        mf = tmp_path / "tool_manifest.json"
        mf.write_text(json.dumps(manifest))
        with patch("app.reference.mcp.ace_studio.client.TOOL_MANIFEST_PATH", mf):
            client = AceStudioClient()
        assert "my_tool" in client._tools

    def test_empty_tools_when_no_manifest(self, tmp_path):
        with patch(
            "app.reference.mcp.ace_studio.client.TOOL_MANIFEST_PATH",
            tmp_path / "missing.json",
        ):
            client = AceStudioClient()
        assert client._tools == {}
