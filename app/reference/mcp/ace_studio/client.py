"""
ACE Studio MCP Client — Phase 2

Thin stateful wrapper over the ACE Studio Streamable HTTP MCP transport.
Loads tool names from tool_manifest.json at runtime so no strings are hardcoded.

Usage (context manager):
    with AceStudioClient() as ace:
        info = ace.get_project_info()
        tracks = ace.list_tracks()

Usage (manual):
    ace = AceStudioClient()
    ace.connect()
    try:
        info = ace.get_project_info()
    finally:
        ace.close()
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import httpx

HERE = Path(__file__).parent
TOOL_MANIFEST_PATH = HERE / "tool_manifest.json"
ACE_STUDIO_URL = "http://localhost:21572/mcp"
CONNECT_TIMEOUT = 5.0
REQUEST_TIMEOUT = 30.0


# ---------------------------------------------------------------------------
# Transport helpers
# ---------------------------------------------------------------------------


def _parse_mcp_response(resp: httpx.Response) -> dict:
    """Parse JSON or first SSE data line from a response."""
    content_type = resp.headers.get("content-type", "")
    if "text/event-stream" in content_type:
        for line in resp.text.splitlines():
            line = line.strip()
            if line.startswith("data:"):
                return json.loads(line[5:].strip())
        raise ValueError("SSE stream contained no data: line")
    return resp.json()


# ---------------------------------------------------------------------------
# Client
# ---------------------------------------------------------------------------


class AceStudioClient:
    """Stateful MCP client for ACE Studio 2.0.

    Call connect() (or use as a context manager) before making tool calls.
    The manifest is loaded from tool_manifest.json written by the probe.
    """

    def __init__(self, base_url: str = ACE_STUDIO_URL) -> None:
        self._url = base_url
        self._session_id: Optional[str] = None
        self._call_id = 0
        self._tools: dict = self._load_manifest()
        self._connected: bool = False
        self._timeout = httpx.Timeout(REQUEST_TIMEOUT, connect=CONNECT_TIMEOUT)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def connect(self) -> "AceStudioClient":
        """Perform the MCP initialize handshake and mark the client as connected."""
        # ACE Studio's HTTP/1.1 keep-alive implementation returns stale responses
        # when connections are reused, so we use a fresh httpx.Client per request
        # (see _raw_post).  We use a sentinel to track connected state.
        self._connected = True
        try:
            self._handshake()
        except httpx.ConnectError as exc:
            self._connected = False
            raise ConnectionError(f"Cannot reach ACE Studio at {self._url}") from exc
        return self

    def close(self) -> None:
        """Mark the client as disconnected."""
        self._connected = False

    def __enter__(self) -> "AceStudioClient":
        return self.connect()

    def __exit__(self, *_) -> None:
        self.close()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _load_manifest() -> dict:
        if TOOL_MANIFEST_PATH.exists():
            return json.loads(TOOL_MANIFEST_PATH.read_text())["tools"]
        return {}

    def _make_http_client(self) -> httpx.Client:
        """Return a fresh httpx.Client for a single request. Override in tests."""
        return httpx.Client(timeout=self._timeout)

    def _next_id(self) -> int:
        self._call_id += 1
        return self._call_id

    def _raw_post(self, payload: dict) -> httpx.Response:
        if not self._connected:
            raise RuntimeError("Client not connected — call connect() first")
        headers: dict[str, str] = {
            "Content-Type": "application/json",
            "Accept": "application/json, text/event-stream",
        }
        if self._session_id:
            headers["Mcp-Session-Id"] = self._session_id
        # ACE Studio's HTTP/1.1 keep-alive returns stale responses when connections
        # are reused. Using a fresh httpx.Client per request is the only reliable fix.
        with self._make_http_client() as client:
            resp = client.post(self._url, json=payload, headers=headers)
        resp.raise_for_status()
        return resp

    def _post(self, payload: dict) -> dict:
        return _parse_mcp_response(self._raw_post(payload))

    def _handshake(self) -> None:
        resp = self._raw_post(
            {
                "jsonrpc": "2.0",
                "method": "initialize",
                "id": self._next_id(),
                "params": {
                    "protocolVersion": "2025-03-26",
                    "capabilities": {},
                    "clientInfo": {"name": "white-ace-client", "version": "1.0"},
                },
            }
        )
        self._session_id = resp.headers.get("Mcp-Session-Id")
        # Consume the response body so the connection is released back to the pool.
        # Without this, httpx holds the TCP connection open with an unread body,
        # and subsequent requests on the same connection read stale data.
        _parse_mcp_response(resp)
        try:
            self._post({"jsonrpc": "2.0", "method": "notifications/initialized"})
        except Exception:
            pass

    def _find_tool(self, *keywords: str) -> str:
        """Return the first manifest tool whose name contains all keywords (case-insensitive)."""
        for name in self._tools:
            name_lower = name.lower()
            if all(kw.lower() in name_lower for kw in keywords):
                return name
        raise KeyError(f"No tool matching {keywords!r} found in manifest")

    def _call(self, tool_name: str, **params) -> dict:
        """Invoke a named tool via tools/call; return the parsed result dict.

        Raises RuntimeError on MCP-level error or isError response.
        """
        rpc = self._post(
            {
                "jsonrpc": "2.0",
                "method": "tools/call",
                "id": self._next_id(),
                "params": {"name": tool_name, "arguments": params},
            }
        )
        if "error" in rpc:
            raise RuntimeError(f"MCP error calling {tool_name!r}: {rpc['error']}")
        result = rpc.get("result", {})
        if result.get("isError"):
            content = result.get("content", [])
            msg = (
                content[0].get("text", "unknown error") if content else "unknown error"
            )
            raise RuntimeError(f"Tool {tool_name!r} returned error: {msg}")
        # Prefer structuredContent (ACE Studio 2025-03-26 always includes it)
        if "structuredContent" in result:
            return result["structuredContent"]
        # Scan all content blocks for the first valid JSON one
        for block in result.get("content", []):
            if block.get("type") == "text":
                try:
                    return json.loads(block["text"])
                except (json.JSONDecodeError, KeyError):
                    continue
        # Fall back to the raw result
        return result

    # ------------------------------------------------------------------
    # Project
    # ------------------------------------------------------------------

    def get_project_info(self) -> dict:
        """Return current project name, duration, and status."""
        return self._call("get_project_status_info")

    def set_tempo(self, bpm: float) -> dict:
        """Set a single constant tempo (BPM) from tick 0."""
        return self._call("set_tempo_automation", points=[{"pos": 0, "value": bpm}])

    def set_time_signature(self, numerator: int, denominator: int) -> dict:
        """Set a single time signature from bar 0."""
        return self._call(
            "set_timesignature_list",
            signatures=[
                {"barPos": 0, "numerator": numerator, "denominator": denominator}
            ],
        )

    # ------------------------------------------------------------------
    # Tracks
    # ------------------------------------------------------------------

    def list_tracks(self) -> list[dict]:
        """Return basic info for all content tracks in the project."""
        result = self._call("get_content_track_basic_info_list")
        # ACE Studio may return {"tracks": [...]} or a bare list
        if isinstance(result, dict):
            return result.get("tracks", [])
        return result

    def find_available_track(self) -> int:
        """Return the index of the first track that has no clips.

        Falls back to 0 with a logged warning when all tracks are occupied.
        """
        import logging

        tracks = self.list_tracks()
        if not tracks:
            return 0
        for idx, track in enumerate(tracks):
            clips = (
                track.get("clips") or track.get("clipCount") or track.get("clip_count")
            )
            if not clips:
                return track.get("trackIndex", track.get("index", idx))
        logging.getLogger(__name__).warning(
            "All %d ACE Studio track(s) have clips — defaulting to track 0", len(tracks)
        )
        return 0

    def find_singer(self, keyword: str, language: str = "English") -> list[dict]:
        """Search available sound sources by name keyword; returns list of singer dicts."""
        result = self._call(
            "get_available_sound_source_list",
            type="voice",
            keyword=keyword,
            language=language,
        )
        if isinstance(result, dict):
            return result.get("list", [])
        return result

    def load_singer(
        self,
        track_index: int,
        singer_id: int,
        group: str = "",
        router_id: Optional[int] = None,
    ) -> dict:
        """Load a singer onto a track by sound-source ID (from find_singer results)."""
        kwargs: dict = {
            "trackIndex": track_index,
            "soundSourceType": "voice",
            "id": singer_id,
            "group": group,
        }
        if router_id is not None:
            kwargs["routerId"] = router_id
        return self._call("load_new_sound_source_on_track", **kwargs)

    # ------------------------------------------------------------------
    # Clips & notes
    # ------------------------------------------------------------------

    def add_clip(
        self,
        track_index: int,
        pos: int,
        dur: int,
        name: Optional[str] = None,
    ) -> dict:
        """Place a new sing clip on a track. pos and dur are in ticks."""
        kwargs: dict = {
            "trackIndex": track_index,
            "pos": pos,
            "dur": dur,
            "type": "sing",
        }
        if name is not None:
            kwargs["name"] = name
        return self._call("add_new_clip", **kwargs)

    def open_editor(self) -> dict:
        """Request ACE Studio to open the pattern editor for the active clip."""
        return self._call("ask_editor_to_open")

    def add_notes_with_lyrics(
        self,
        notes: list[dict],
        lyric_sentence: str,
        language: str = "ENG",
        offset: Optional[int] = None,
    ) -> dict:
        """Add notes to the open pattern editor with auto-distributed lyrics.

        Each note dict: {"pitch": int, "dur": int} — pitch is MIDI note number,
        dur is duration in ticks. lyric_sentence is distributed across notes
        automatically by ACE Studio.
        """
        kwargs: dict = {
            "notes": notes,
            "lyric_sentence": lyric_sentence,
            "language": language,
        }
        if offset is not None:
            kwargs["offset"] = offset
        return self._call("add_notes_in_editor", **kwargs)

    def add_section_clips(
        self,
        sections: list[dict],
        track_index: int,
        language: str = "ENG",
    ) -> list[dict]:
        """Add one clip per section, each with notes and lyrics pre-loaded.

        Each section dict:
            name        — section label used as clip name
            start_tick  — absolute start tick in ACE_TPB space
            dur_ticks   — clip duration in ticks
            notes       — list of {"pos": int, "pitch": int, "dur": int}
                          (pos relative to clip start)
            lyrics      — lyric sentence string for this section
        Returns list of result dicts from add_clip / add_notes_with_lyrics.
        """
        results = []
        for sec in sections:
            name = sec.get("name", "")
            start = sec["start_tick"]
            dur = sec["dur_ticks"]
            notes = sec.get("notes", [])
            lyrics = sec.get("lyrics", "")

            clip_result = self.add_clip(
                track_index=track_index,
                pos=start,
                dur=dur,
                name=name or None,
            )
            notes_result = {}
            if notes:
                self.open_editor()
                notes_result = self.add_notes_with_lyrics(
                    notes, lyrics or "", language=language
                )
            results.append({"clip": clip_result, "notes": notes_result})
        return results

    def get_clip_lyrics(self, track_index: int, clip_index: int) -> dict:
        """Return sentence-level lyrics for a Sing clip."""
        return self._call(
            "get_note_clip_lyrics",
            trackIndex=track_index,
            clipIndex=clip_index,
        )
