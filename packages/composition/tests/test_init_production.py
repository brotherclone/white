"""Tests for init_production.py."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest
import yaml
from white_composition.init_production import (
    _parse_sounds_like_response,
    init_production,
    load_initial_proposal,
    load_song_context,
    write_initial_proposal,
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
        assert out.name == "song_context.yml"

    def test_round_trip_sounds_like(self, tmp_path):
        meta = self._make_meta()
        sounds_like = ["Sufjan Stevens", "Bon Iver", "The National"]
        write_initial_proposal(tmp_path, meta, sounds_like)
        loaded = load_song_context(tmp_path)
        assert loaded["sounds_like"] == sounds_like

    def test_round_trip_metadata_fields(self, tmp_path):
        meta = self._make_meta()
        write_initial_proposal(tmp_path, meta, [])
        loaded = load_song_context(tmp_path)
        assert loaded["color"] == "Red"
        assert loaded["concept"] == "Memory and loss"
        assert loaded["singer"] == "Gabriel"
        assert loaded["key"] == "C major"
        assert loaded["bpm"] == 120
        assert loaded["time_sig"] == "4/4"

    def test_proposed_by_is_claude(self, tmp_path):
        write_initial_proposal(tmp_path, self._make_meta(), [])
        loaded = load_song_context(tmp_path)
        assert loaded["proposed_by"] == "claude"

    def test_generated_timestamp_present(self, tmp_path):
        write_initial_proposal(tmp_path, self._make_meta(), [])
        loaded = load_song_context(tmp_path)
        assert "generated" in loaded
        assert loaded["generated"]

    def test_production_dir_created_if_missing(self, tmp_path):
        nested = tmp_path / "deep" / "nested"
        write_initial_proposal(nested, self._make_meta(), [])
        assert (nested / "song_context.yml").exists()

    def test_sounds_like_is_list_of_strings(self, tmp_path):
        write_initial_proposal(tmp_path, self._make_meta(), ["Artist A", "Artist B"])
        loaded = load_song_context(tmp_path)
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

    def test_writes_song_context_yml(self, tmp_path):
        proposal_path = _make_proposal_yml(tmp_path)
        prod_dir = tmp_path / "production" / "test_song"

        with patch(
            "white_composition.init_production._call_claude",
            return_value=self._make_claude_response(),
        ):
            out = init_production(
                production_dir=prod_dir,
                song_proposal_path=proposal_path,
            )

        assert out.name == "song_context.yml"
        assert out.exists()
        loaded = yaml.safe_load(out.read_text())
        assert "sounds_like" in loaded
        assert isinstance(loaded["sounds_like"], list)
        assert len(loaded["sounds_like"]) == 4

    def test_parsed_names_are_bare_strings(self, tmp_path):
        proposal_path = _make_proposal_yml(tmp_path)
        prod_dir = tmp_path / "production" / "test_song"

        with patch(
            "white_composition.init_production._call_claude",
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
        existing = prod_dir / "song_context.yml"
        existing.write_text(yaml.dump({"sounds_like": ["Pre-existing Artist"]}))

        call_count = 0

        def fake_claude(prompt, model):
            nonlocal call_count
            call_count += 1
            return "- New Artist\n"

        with patch(
            "white_composition.init_production._call_claude",
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
        (prod_dir / "song_context.yml").write_text(
            yaml.dump({"sounds_like": ["Old Artist"]})
        )

        with patch(
            "white_composition.init_production._call_claude",
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

    def test_song_context_written(self, tmp_path):
        write_initial_proposal(tmp_path, self._make_meta(), ["Sufjan Stevens"])
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

    def test_prefers_song_context_when_both_exist(self, tmp_path):
        """song_context.yml takes precedence over initial_proposal.yml."""
        import yaml as _yaml

        (tmp_path / "initial_proposal.yml").write_text(
            _yaml.dump({"sounds_like": ["From initial"], "color": "Red"})
        )
        (tmp_path / "song_context.yml").write_text(
            _yaml.dump({"sounds_like": ["From context"], "color": "Blue"})
        )
        result = load_initial_proposal(tmp_path)
        assert result["sounds_like"] == ["From context"]
        assert result["color"] == "Blue"

    def test_returns_empty_when_neither_exists(self, tmp_path):
        result = load_initial_proposal(tmp_path)
        assert result == {}


# ---------------------------------------------------------------------------
# Aesthetic hints detection
# ---------------------------------------------------------------------------


class TestAestheticHints:
    def test_ambient_cluster_detected(self, tmp_path):
        from white_composition.init_production import (
            write_initial_proposal,
        )

        meta = {
            "title": "Hazy Song",
            "bpm": 80,
            "time_sig": "4/4",
            "key": "D minor",
            "color": "Violet",
            "concept": "A slow dissolution",
            "singer": "katherine",
            "song_slug": "hazy_song_v1",
        }
        write_initial_proposal(tmp_path, meta, ["Grouper", "Beach House", "Low"])
        ctx = load_song_context(tmp_path)
        assert "aesthetic_hints" in ctx
        assert ctx["aesthetic_hints"]["density"] == "sparse"
        assert ctx["aesthetic_hints"]["texture"] == "hazy"

    def test_no_cluster_no_hints(self, tmp_path):
        from white_composition.init_production import (
            write_initial_proposal,
        )

        meta = {
            "title": "Rock Song",
            "bpm": 140,
            "time_sig": "4/4",
            "key": "A major",
            "color": "Red",
            "concept": "Driving force",
            "singer": "gabriel",
            "song_slug": "rock_song_v1",
        }
        write_initial_proposal(tmp_path, meta, ["AC/DC", "Led Zeppelin"])
        ctx = load_song_context(tmp_path)
        assert "aesthetic_hints" not in ctx

    def test_single_ambient_artist_no_hints(self, tmp_path):
        from white_composition.init_production import (
            write_initial_proposal,
        )

        meta = {
            "title": "Mild Song",
            "bpm": 90,
            "time_sig": "4/4",
            "key": "E minor",
            "color": "Blue",
            "concept": "Mild drift",
            "singer": "gabriel",
            "song_slug": "mild_song_v1",
        }
        # Only one ambient cluster artist — threshold is 2
        write_initial_proposal(tmp_path, meta, ["Grouper", "Radiohead"])
        ctx = load_song_context(tmp_path)
        assert "aesthetic_hints" not in ctx


# ---------------------------------------------------------------------------
# Pattern tag fields
# ---------------------------------------------------------------------------


class TestPatternTags:
    def test_drum_patterns_have_tags_field(self):
        from white_generation.patterns.drum_patterns import ALL_TEMPLATES

        for p in ALL_TEMPLATES:
            assert hasattr(p, "tags"), f"{p.name} missing tags"
            assert isinstance(p.tags, list), f"{p.name}.tags not a list"

    def test_bass_patterns_have_tags_field(self):
        from white_generation.patterns.bass_patterns import ALL_TEMPLATES

        for p in ALL_TEMPLATES:
            assert hasattr(p, "tags"), f"{p.name} missing tags"
            assert isinstance(p.tags, list), f"{p.name}.tags not a list"

    def test_melody_patterns_have_tags_field(self):
        from white_generation.patterns.melody_patterns import ALL_TEMPLATES

        for p in ALL_TEMPLATES:
            assert hasattr(p, "tags"), f"{p.name} missing tags"
            assert isinstance(p.tags, list), f"{p.name}.tags not a list"

    def test_sparse_drum_templates_filterable(self):
        from white_generation.patterns.drum_patterns import ALL_TEMPLATES

        sparse = [p for p in ALL_TEMPLATES if "sparse" in p.tags]
        assert len(sparse) >= 5, f"Expected ≥5 sparse drum templates, got {len(sparse)}"

    def test_drone_bass_templates_filterable(self):
        from white_generation.patterns.bass_patterns import ALL_TEMPLATES

        drone_pedal = [p for p in ALL_TEMPLATES if {"drone", "pedal"} & set(p.tags)]
        assert (
            len(drone_pedal) >= 4
        ), f"Expected ≥4 drone/pedal bass templates, got {len(drone_pedal)}"

    def test_lamentful_melody_templates_filterable(self):
        from white_generation.patterns.melody_patterns import ALL_TEMPLATES

        lamentful = [p for p in ALL_TEMPLATES if "lamentful" in p.tags]
        assert (
            len(lamentful) >= 4
        ), f"Expected ≥4 lamentful melody templates, got {len(lamentful)}"


# ---------------------------------------------------------------------------
# Aesthetic tag adjustment
# ---------------------------------------------------------------------------


class TestAestheticTagAdjustment:
    def test_sparse_hint_boosts_sparse_pattern(self):
        from white_generation.patterns.aesthetic_hints import (
            aesthetic_tag_adjustment,
        )

        adj = aesthetic_tag_adjustment(["sparse", "ambient"], {"density": "sparse"})
        assert adj == 0.10

    def test_dense_hint_penalises_sparse_pattern(self):
        from white_generation.patterns.aesthetic_hints import (
            aesthetic_tag_adjustment,
        )

        adj = aesthetic_tag_adjustment(["sparse"], {"density": "dense"})
        assert adj == -0.05

    def test_no_hints_returns_zero(self):
        from white_generation.patterns.aesthetic_hints import (
            aesthetic_tag_adjustment,
        )

        assert aesthetic_tag_adjustment(["sparse"], None) == 0.0
        assert aesthetic_tag_adjustment([], {"density": "sparse"}) == 0.0

    def test_no_matching_tags_returns_zero(self):
        from white_generation.patterns.aesthetic_hints import (
            aesthetic_tag_adjustment,
        )

        adj = aesthetic_tag_adjustment(["motorik"], {"density": "sparse"})
        assert adj == 0.0


class TestArcToEnergy:
    def test_low_arc(self):
        from white_generation.patterns.aesthetic_hints import arc_to_energy

        assert arc_to_energy(0.10) == "low"
        assert arc_to_energy(0.0) == "low"
        assert arc_to_energy(0.29) == "low"

    def test_medium_arc(self):
        from white_generation.patterns.aesthetic_hints import arc_to_energy

        assert arc_to_energy(0.30) == "medium"
        assert arc_to_energy(0.50) == "medium"
        assert arc_to_energy(0.65) == "medium"

    def test_high_arc(self):
        from white_generation.patterns.aesthetic_hints import arc_to_energy

        assert arc_to_energy(0.66) == "high"
        assert arc_to_energy(0.85) == "high"
        assert arc_to_energy(1.0) == "high"


class TestArcTagAdjustment:
    def test_low_arc_boosts_drone_pedal(self):
        from white_generation.patterns.aesthetic_hints import arc_tag_adjustment

        assert arc_tag_adjustment(0.10, ["drone"]) == pytest.approx(0.10)
        assert arc_tag_adjustment(0.10, ["pedal"]) == pytest.approx(0.10)
        assert arc_tag_adjustment(0.10, ["lamentful"]) == pytest.approx(0.10)

    def test_high_arc_penalises_root_drone(self):
        from white_generation.patterns.aesthetic_hints import arc_tag_adjustment

        assert arc_tag_adjustment(0.80, ["root_drone"]) == pytest.approx(-0.05)

    def test_high_arc_no_penalty_other_tags(self):
        from white_generation.patterns.aesthetic_hints import arc_tag_adjustment

        assert arc_tag_adjustment(0.80, ["walking"]) == pytest.approx(0.0)

    def test_mid_arc_returns_zero(self):
        from white_generation.patterns.aesthetic_hints import arc_tag_adjustment

        assert arc_tag_adjustment(0.50, ["drone"]) == pytest.approx(0.0)
        assert arc_tag_adjustment(0.50, ["root_drone"]) == pytest.approx(0.0)

    def test_empty_tags_returns_zero(self):
        from white_generation.patterns.aesthetic_hints import arc_tag_adjustment

        assert arc_tag_adjustment(0.10, []) == pytest.approx(0.0)
