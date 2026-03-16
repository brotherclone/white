"""Tests for init_production.py."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import yaml

from app.generators.midi.production.init_production import (
    _parse_sounds_like_response,
    load_initial_proposal,
    load_song_context,
    write_initial_proposal,
    init_production,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_proposal_yml(tmp_path: Path, extra: dict | None = None) -> Path:
    """Write a minimal song proposal YAML and return its path."""
    data = {
        "title": "Test Song",
        "bpm": 120,
        "tempo": {"numerator": 4, "denominator": 4},
        "key": "C major",
        "rainbow_color": {"color_name": "Red"},
        "concept": "A test concept about memory and loss",
        "genres": ["folk", "ambient"],
        "mood": ["melancholic", "sparse"],
        "singer": "Gabriel",
    }
    if extra:
        data.update(extra)
    path = tmp_path / "proposal.yml"
    path.write_text(yaml.dump(data))
    return path


# ---------------------------------------------------------------------------
# 5.1 Unit: _parse_sounds_like_response
# ---------------------------------------------------------------------------


class TestParseSoundsLikeResponse:

    def test_yaml_list_format(self):
        text = "- Sufjan Stevens\n- Bon Iver\n- The National\n"
        result = _parse_sounds_like_response(text)
        assert result == ["Sufjan Stevens", "Bon Iver", "The National"]

    def test_numbered_dot_format(self):
        text = "1. Sufjan Stevens\n2. Bon Iver\n3. The National\n"
        result = _parse_sounds_like_response(text)
        assert result == ["Sufjan Stevens", "Bon Iver", "The National"]

    def test_numbered_paren_format(self):
        text = "1) Sufjan Stevens\n2) Bon Iver\n"
        result = _parse_sounds_like_response(text)
        assert result == ["Sufjan Stevens", "Bon Iver"]

    def test_bare_names_one_per_line(self):
        text = "Sufjan Stevens\nBon Iver\nThe National\n"
        result = _parse_sounds_like_response(text)
        assert result == ["Sufjan Stevens", "Bon Iver", "The National"]

    def test_strips_parenthetical_annotations(self):
        text = "- Sufjan Stevens (Illinois-era)\n- Bon Iver (debut album)\n"
        result = _parse_sounds_like_response(text)
        assert result == ["Sufjan Stevens", "Bon Iver"]

    def test_strips_quotes(self):
        text = "- \"Sufjan Stevens\"\n- 'Bon Iver'\n"
        result = _parse_sounds_like_response(text)
        assert result == ["Sufjan Stevens", "Bon Iver"]

    def test_empty_lines_skipped(self):
        text = "- Sufjan Stevens\n\n- Bon Iver\n\n"
        result = _parse_sounds_like_response(text)
        assert result == ["Sufjan Stevens", "Bon Iver"]

    def test_empty_string_returns_empty_list(self):
        result = _parse_sounds_like_response("")
        assert result == []

    def test_mixed_formats_handled(self):
        text = "- Sufjan Stevens\n1. Bon Iver\nThe National\n"
        result = _parse_sounds_like_response(text)
        assert len(result) == 3
        assert "Sufjan Stevens" in result
        assert "Bon Iver" in result
        assert "The National" in result


# ---------------------------------------------------------------------------
# 5.2 Unit: write_initial_proposal + load_initial_proposal round-trip
# ---------------------------------------------------------------------------


class TestWriteAndLoadInitialProposal:

    def _make_meta(self) -> dict:
        return {
            "color": "Red",
            "concept": "Memory and loss",
            "singer": "Gabriel",
            "key": "C major",
            "bpm": 120,
            "time_sig": "4/4",
        }

    def test_file_is_written(self, tmp_path):
        meta = self._make_meta()
        out = write_initial_proposal(tmp_path, meta, ["Sufjan Stevens", "Bon Iver"])
        assert out.exists()
        assert out.name == "initial_proposal.yml"

    def test_round_trip_sounds_like(self, tmp_path):
        meta = self._make_meta()
        sounds_like = ["Sufjan Stevens", "Bon Iver", "The National"]
        write_initial_proposal(tmp_path, meta, sounds_like)
        loaded = load_initial_proposal(tmp_path)
        assert loaded["sounds_like"] == sounds_like

    def test_round_trip_metadata_fields(self, tmp_path):
        meta = self._make_meta()
        write_initial_proposal(tmp_path, meta, [])
        loaded = load_initial_proposal(tmp_path)
        assert loaded["color"] == "Red"
        assert loaded["concept"] == "Memory and loss"
        assert loaded["singer"] == "Gabriel"
        assert loaded["key"] == "C major"
        assert loaded["bpm"] == 120
        assert loaded["time_sig"] == "4/4"

    def test_proposed_by_is_claude(self, tmp_path):
        write_initial_proposal(tmp_path, self._make_meta(), [])
        loaded = load_initial_proposal(tmp_path)
        assert loaded["proposed_by"] == "claude"

    def test_generated_timestamp_present(self, tmp_path):
        write_initial_proposal(tmp_path, self._make_meta(), [])
        loaded = load_initial_proposal(tmp_path)
        assert "generated" in loaded
        assert loaded["generated"]

    def test_production_dir_created_if_missing(self, tmp_path):
        nested = tmp_path / "deep" / "nested"
        write_initial_proposal(nested, self._make_meta(), [])
        assert (nested / "initial_proposal.yml").exists()

    def test_sounds_like_is_list_of_strings(self, tmp_path):
        write_initial_proposal(tmp_path, self._make_meta(), ["Artist A", "Artist B"])
        loaded = load_initial_proposal(tmp_path)
        assert isinstance(loaded["sounds_like"], list)
        for name in loaded["sounds_like"]:
            assert isinstance(name, str)


# ---------------------------------------------------------------------------
# 5.3 Unit: load_initial_proposal returns {} when file missing
# ---------------------------------------------------------------------------


class TestLoadInitialProposalMissing:

    def test_returns_empty_dict_when_missing(self, tmp_path):
        result = load_initial_proposal(tmp_path)
        assert result == {}

    def test_no_exception_when_dir_is_empty(self, tmp_path):
        result = load_initial_proposal(tmp_path / "nonexistent")
        assert result == {}


# ---------------------------------------------------------------------------
# 5.4 Integration: stub Claude API — verify full init flow
# ---------------------------------------------------------------------------


class TestInitProductionIntegration:

    def _make_claude_response(self) -> str:
        return "- Sufjan Stevens\n- Bon Iver\n- The National\n- Low\n"

    def test_writes_initial_proposal_yml(self, tmp_path):
        proposal_path = _make_proposal_yml(tmp_path)
        prod_dir = tmp_path / "production" / "test_song"

        with patch(
            "app.generators.midi.production.init_production._call_claude",
            return_value=self._make_claude_response(),
        ):
            out = init_production(
                production_dir=prod_dir,
                song_proposal_path=proposal_path,
            )

        assert out.exists()
        loaded = yaml.safe_load(out.read_text())
        assert "sounds_like" in loaded
        assert isinstance(loaded["sounds_like"], list)
        assert len(loaded["sounds_like"]) == 4

    def test_parsed_names_are_bare_strings(self, tmp_path):
        proposal_path = _make_proposal_yml(tmp_path)
        prod_dir = tmp_path / "production" / "test_song"

        with patch(
            "app.generators.midi.production.init_production._call_claude",
            return_value="- Sufjan Stevens (Illinois-era)\n- Bon Iver\n",
        ):
            out = init_production(
                production_dir=prod_dir, song_proposal_path=proposal_path
            )

        loaded = yaml.safe_load(out.read_text())
        assert "Sufjan Stevens" in loaded["sounds_like"]
        for name in loaded["sounds_like"]:
            assert "(" not in name

    def test_idempotent_skips_if_exists(self, tmp_path):
        proposal_path = _make_proposal_yml(tmp_path)
        prod_dir = tmp_path / "production" / "test_song"
        prod_dir.mkdir(parents=True)
        existing = prod_dir / "initial_proposal.yml"
        existing.write_text(yaml.dump({"sounds_like": ["Pre-existing Artist"]}))

        call_count = 0

        def fake_claude(prompt, model):
            nonlocal call_count
            call_count += 1
            return "- New Artist\n"

        with patch(
            "app.generators.midi.production.init_production._call_claude",
            side_effect=fake_claude,
        ):
            out = init_production(
                production_dir=prod_dir,
                song_proposal_path=proposal_path,
                force=False,
            )

        assert call_count == 0, "Claude should not be called when file exists"
        loaded = yaml.safe_load(out.read_text())
        assert loaded["sounds_like"] == ["Pre-existing Artist"]

    def test_force_regenerates(self, tmp_path):
        proposal_path = _make_proposal_yml(tmp_path)
        prod_dir = tmp_path / "production" / "test_song"
        prod_dir.mkdir(parents=True)
        (prod_dir / "initial_proposal.yml").write_text(
            yaml.dump({"sounds_like": ["Old Artist"]})
        )

        with patch(
            "app.generators.midi.production.init_production._call_claude",
            return_value="- New Artist\n",
        ):
            out = init_production(
                production_dir=prod_dir,
                song_proposal_path=proposal_path,
                force=True,
            )

        loaded = yaml.safe_load(out.read_text())
        assert "New Artist" in loaded["sounds_like"]
        assert "Old Artist" not in loaded["sounds_like"]


# ---------------------------------------------------------------------------
# 5.5 Unit: song_context.yml written alongside initial_proposal.yml
# ---------------------------------------------------------------------------


class TestSongContextYml:

    def _make_meta(self) -> dict:
        return {
            "title": "My Song",
            "song_slug": "my-song",
            "song_proposal": "my_song.yml",
            "thread": "/path/to/thread",
            "color": "Red",
            "concept": "Memory and loss",
            "singer": "Gabriel",
            "key": "C major",
            "bpm": 120,
            "time_sig": "4/4",
            "genres": ["folk", "ambient"],
            "mood": ["melancholic"],
        }

    def test_song_context_written_alongside_initial_proposal(self, tmp_path):
        write_initial_proposal(tmp_path, self._make_meta(), ["Sufjan Stevens"])
        assert (tmp_path / "initial_proposal.yml").exists()
        assert (tmp_path / "song_context.yml").exists()

    def test_song_context_round_trip(self, tmp_path):
        meta = self._make_meta()
        write_initial_proposal(tmp_path, meta, ["Sufjan Stevens", "Bon Iver"])
        ctx = load_song_context(tmp_path)
        assert ctx["color"] == "Red"
        assert ctx["concept"] == "Memory and loss"
        assert ctx["title"] == "My Song"
        assert ctx["key"] == "C major"
        assert ctx["bpm"] == 120
        assert ctx["time_sig"] == "4/4"
        assert ctx["singer"] == "Gabriel"
        assert ctx["sounds_like"] == ["Sufjan Stevens", "Bon Iver"]
        assert ctx["genres"] == ["folk", "ambient"]
        assert ctx["schema_version"] == "1"
        assert ctx["proposed_by"] == "claude"

    def test_song_context_has_phases_block(self, tmp_path):
        write_initial_proposal(tmp_path, self._make_meta(), [])
        ctx = load_song_context(tmp_path)
        assert "phases" in ctx
        assert ctx["phases"]["chords"] == "pending"
        assert ctx["phases"]["drums"] == "pending"
        assert ctx["phases"]["lyrics"] == "pending"

    def test_load_song_context_returns_empty_when_missing(self, tmp_path):
        result = load_song_context(tmp_path)
        assert result == {}

    def test_load_song_context_returns_empty_for_nonexistent_dir(self, tmp_path):
        result = load_song_context(tmp_path / "nonexistent")
        assert result == {}


# ---------------------------------------------------------------------------
# 5.6 Unit: load_initial_proposal falls back to song_context.yml
# ---------------------------------------------------------------------------


class TestLoadInitialProposalFallback:

    def test_falls_back_to_song_context_when_initial_proposal_absent(self, tmp_path):
        # Write only song_context.yml (simulates migrated dir)
        import yaml as _yaml

        ctx = {
            "schema_version": "1",
            "generated": "2026-01-01T00:00:00+00:00",
            "proposed_by": "claude",
            "sounds_like": ["Grouper", "The Caretaker"],
            "color": "Violet",
            "concept": "A dream of forgetting",
            "singer": "Katherine",
            "key": "D minor",
            "bpm": 80,
            "time_sig": "4/4",
            "genres": ["ambient"],
            "mood": ["dreamy"],
            "phases": {},
        }
        (tmp_path / "song_context.yml").write_text(_yaml.dump(ctx))
        # No initial_proposal.yml present
        result = load_initial_proposal(tmp_path)
        assert result["sounds_like"] == ["Grouper", "The Caretaker"]
        assert result["color"] == "Violet"

    def test_prefers_initial_proposal_when_both_exist(self, tmp_path):
        import yaml as _yaml

        (tmp_path / "initial_proposal.yml").write_text(
            _yaml.dump({"sounds_like": ["From initial"], "color": "Red"})
        )
        (tmp_path / "song_context.yml").write_text(
            _yaml.dump({"sounds_like": ["From context"], "color": "Blue"})
        )
        result = load_initial_proposal(tmp_path)
        assert result["sounds_like"] == ["From initial"]

    def test_returns_empty_when_neither_exists(self, tmp_path):
        result = load_initial_proposal(tmp_path)
        assert result == {}
