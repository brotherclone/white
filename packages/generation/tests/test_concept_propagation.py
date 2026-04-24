"""Tests for concept text propagation via song_context.yml.

Verifies that load_song_context returns the real concept text, and that
the three-way fallback logic (song_info → song_context → color string)
is exercised correctly at the unit level.
"""

from __future__ import annotations

from pathlib import Path

from app.generators.midi.production.init_production import (
    load_song_context,
    write_initial_proposal,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write_song_context(prod_dir: Path, concept: str, color: str = "Red") -> None:
    meta = {
        "title": "Test Song",
        "color": color,
        "concept": concept,
        "singer": "Gabriel",
        "key": "C minor",
        "bpm": 120,
        "time_sig": "4/4",
        "genres": ["ambient"],
        "mood": ["melancholic"],
    }
    write_initial_proposal(prod_dir, meta, [])


def _concept_from_context_or_fallback(
    song_info: dict, prod_path: Path
) -> tuple[str, bool]:
    """Replicate the three-way fallback logic used in drum/bass/melody pipelines.

    Returns (concept_text, used_fallback).
    """
    concept_text = song_info.get("concept", "")
    if not concept_text:
        _ctx = load_song_context(prod_path)
        concept_text = _ctx.get("concept", "")
    used_fallback = False
    if not concept_text:
        concept_text = f"{song_info.get('color_name', 'Unknown')} chromatic concept"
        used_fallback = True
    return concept_text, used_fallback


# ---------------------------------------------------------------------------
# Tests: load_song_context
# ---------------------------------------------------------------------------


class TestLoadSongContextForPipelines:

    def test_returns_concept_when_written(self, tmp_path):
        _write_song_context(tmp_path, concept="A song about entropy and memory")
        ctx = load_song_context(tmp_path)
        assert ctx["concept"] == "A song about entropy and memory"

    def test_returns_empty_dict_when_absent(self, tmp_path):
        ctx = load_song_context(tmp_path)
        assert ctx == {}

    def test_returns_color_from_context(self, tmp_path):
        _write_song_context(tmp_path, concept="Test", color="Violet")
        ctx = load_song_context(tmp_path)
        assert ctx["color"] == "Violet"


# ---------------------------------------------------------------------------
# Tests: three-way concept fallback logic
# ---------------------------------------------------------------------------


class TestConceptFallbackLogic:
    """Unit tests for the exact fallback chain used in drum/bass/melody pipelines:
    song_info["concept"]  →  song_context.yml  →  "{color} chromatic concept"
    """

    def test_uses_song_info_concept_when_present(self, tmp_path):
        _write_song_context(tmp_path, concept="Context concept")
        song_info = {"concept": "Song info concept", "color_name": "Red"}
        result, used_fallback = _concept_from_context_or_fallback(song_info, tmp_path)
        assert result == "Song info concept"
        assert not used_fallback

    def test_falls_back_to_song_context_when_song_info_empty(self, tmp_path):
        _write_song_context(tmp_path, concept="Real concept from context file")
        song_info = {"concept": "", "color_name": "Red"}
        result, used_fallback = _concept_from_context_or_fallback(song_info, tmp_path)
        assert result == "Real concept from context file"
        assert not used_fallback

    def test_falls_back_to_color_string_when_no_context(self, tmp_path):
        # No song_context.yml written
        song_info = {"concept": "", "color_name": "Indigo"}
        result, used_fallback = _concept_from_context_or_fallback(song_info, tmp_path)
        assert result == "Indigo chromatic concept"
        assert used_fallback

    def test_song_context_used_when_song_info_has_no_concept_key(self, tmp_path):
        _write_song_context(tmp_path, concept="Concept without key in song_info")
        song_info = {"color_name": "Blue"}  # no concept key at all
        result, used_fallback = _concept_from_context_or_fallback(song_info, tmp_path)
        assert result == "Concept without key in song_info"
        assert not used_fallback

    def test_song_context_concept_survives_different_colors(self, tmp_path):
        _write_song_context(tmp_path, concept="Yellow concept text", color="Yellow")
        song_info = {"concept": "", "color_name": "Yellow"}
        result, used_fallback = _concept_from_context_or_fallback(song_info, tmp_path)
        assert "yellow concept" in result.lower()
        assert not used_fallback

    def test_all_three_phases_can_read_same_context(self, tmp_path):
        _write_song_context(tmp_path, concept="Shared concept for all phases")
        for color_name in ["Red", "Red", "Red"]:  # simulate drum, bass, melody
            song_info = {"concept": "", "color_name": color_name}
            result, _ = _concept_from_context_or_fallback(song_info, tmp_path)
            assert result == "Shared concept for all phases"
