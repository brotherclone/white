"""Tests for the unified song proposal loader and its four callers.

Verifies task 4.5 acceptance criteria:
- All four callers receive correct field sets
- time_sig is always a "N/N" string
- color is always present
- color_name returned for chord_pipeline callers (remapped from color)
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from app.generators.midi.production.init_production import write_initial_proposal
from app.generators.midi.production.production_plan import load_song_proposal_unified

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _write_proposal(path: Path, **overrides) -> Path:
    """Write a minimal song proposal YAML and return its path."""
    proposal = {
        "title": "Test Song",
        "bpm": 100,
        "tempo": {"numerator": 7, "denominator": 8},
        "key": "F# minor",
        "rainbow_color": {"color_name": "Violet"},
        "concept": "A song about fractals",
        "genres": ["ambient", "electronic"],
        "mood": ["melancholic"],
        "singer": "Gabriel",
        "sounds_like": ["Aphex Twin", "Boards of Canada"],
    }
    proposal.update(overrides)
    path.write_text(yaml.dump(proposal, allow_unicode=True))
    return path


def _write_chord_review(
    chords_dir: Path, thread: str, proposal_file: str, **extra
) -> None:
    chords_dir.mkdir(parents=True, exist_ok=True)
    review = {
        "thread": thread,
        "song_proposal": proposal_file,
        "color": "Violet",
        "singer": "Shirley",
        **extra,
    }
    (chords_dir / "review.yml").write_text(yaml.dump(review, allow_unicode=True))


# ---------------------------------------------------------------------------
# Tests: load_song_proposal_unified
# ---------------------------------------------------------------------------


class TestLoadSongProposalUnified:

    def test_basic_fields_returned(self, tmp_path):
        p = _write_proposal(tmp_path / "song.yml")
        result = load_song_proposal_unified(p)
        assert result["title"] == "Test Song"
        assert result["bpm"] == 100
        assert result["key"] == "F# minor"
        assert result["color"] == "Violet"
        assert result["concept"] == "A song about fractals"
        assert result["genres"] == ["ambient", "electronic"]
        assert result["mood"] == ["melancholic"]
        assert result["singer"] == "Gabriel"
        assert result["sounds_like"] == ["Aphex Twin", "Boards of Canada"]

    def test_time_sig_always_string(self, tmp_path):
        """tempo dict must be normalised to 'N/N' string."""
        p = _write_proposal(tmp_path / "song.yml")
        result = load_song_proposal_unified(p)
        assert result["time_sig"] == "7/8"
        assert isinstance(result["time_sig"], str)
        assert "/" in result["time_sig"]

    def test_time_sig_string_passthrough(self, tmp_path):
        proposal = {
            "title": "T",
            "bpm": 120,
            "key": "C major",
            "rainbow_color": {"color_name": "Red"},
            "concept": "c",
            "genres": [],
            "mood": [],
            "singer": "",
            "time_sig": "4/4",
        }
        p = tmp_path / "song.yml"
        p.write_text(yaml.dump(proposal))
        result = load_song_proposal_unified(p)
        assert result["time_sig"] == "4/4"

    def test_color_always_present_from_dict(self, tmp_path):
        p = _write_proposal(tmp_path / "song.yml")
        result = load_song_proposal_unified(p)
        assert result["color"] == "Violet"

    def test_color_from_flat_string(self, tmp_path):
        proposal = {
            "title": "T",
            "bpm": 120,
            "key": "C major",
            "rainbow_color": "Red",
            "concept": "c",
            "genres": [],
            "mood": [],
            "singer": "",
        }
        p = tmp_path / "song.yml"
        p.write_text(yaml.dump(proposal))
        result = load_song_proposal_unified(p)
        assert result["color"] == "Red"

    def test_key_components_parsed(self, tmp_path):
        p = _write_proposal(tmp_path / "song.yml")
        result = load_song_proposal_unified(p)
        assert result["key_root"] == "F#"
        assert result["mode"] == "Minor"

    def test_key_components_major(self, tmp_path):
        proposal = {
            "title": "T",
            "bpm": 120,
            "key": "D major",
            "rainbow_color": "Blue",
            "concept": "c",
            "genres": [],
            "mood": [],
            "singer": "",
        }
        p = tmp_path / "song.yml"
        p.write_text(yaml.dump(proposal))
        result = load_song_proposal_unified(p)
        assert result["key_root"] == "D"
        assert result["mode"] == "Major"

    def test_concept_fallback_from_manifest(self, tmp_path):
        thread_dir = tmp_path / "thread"
        thread_dir.mkdir()
        manifest = {"concept": "Concept from manifest"}
        (thread_dir / "manifest.yml").write_text(yaml.dump(manifest))

        proposal = {
            "title": "T",
            "bpm": 120,
            "key": "C major",
            "rainbow_color": "Red",
            "concept": "",
            "genres": [],
            "mood": [],
            "singer": "",
        }
        p = tmp_path / "song.yml"
        p.write_text(yaml.dump(proposal))

        result = load_song_proposal_unified(p, thread_dir=thread_dir)
        assert result["concept"] == "Concept from manifest"

    def test_proposal_concept_wins_over_manifest(self, tmp_path):
        thread_dir = tmp_path / "thread"
        thread_dir.mkdir()
        (thread_dir / "manifest.yml").write_text(
            yaml.dump({"concept": "Manifest concept"})
        )

        p = _write_proposal(tmp_path / "song.yml", concept="Proposal concept")
        result = load_song_proposal_unified(p, thread_dir=thread_dir)
        assert result["concept"] == "Proposal concept"

    def test_song_filename_returned(self, tmp_path):
        p = _write_proposal(tmp_path / "my_song.yml")
        result = load_song_proposal_unified(p)
        assert result["song_filename"] == "my_song.yml"

    def test_sounds_like_defaults_to_empty_list(self, tmp_path):
        proposal = {
            "title": "T",
            "bpm": 120,
            "key": "C major",
            "rainbow_color": "Red",
            "concept": "c",
            "genres": [],
            "mood": [],
            "singer": "",
        }
        p = tmp_path / "song.yml"
        p.write_text(yaml.dump(proposal))
        result = load_song_proposal_unified(p)
        assert result["sounds_like"] == []


# ---------------------------------------------------------------------------
# Tests: chord_pipeline.load_song_proposal (caller 1)
# ---------------------------------------------------------------------------


class TestChordPipelineCaller:

    def _make_proposal(self, thread_dir: Path, filename: str) -> Path:
        yml_dir = thread_dir / "yml"
        yml_dir.mkdir(parents=True, exist_ok=True)
        p = yml_dir / filename
        _write_proposal(p)
        return p

    def test_color_name_field_present(self, tmp_path):
        """chord_pipeline remaps 'color' → 'color_name'."""
        thread_dir = tmp_path / "thread"
        self._make_proposal(thread_dir, "song.yml")

        from app.generators.midi.pipelines.chord_pipeline import load_song_proposal

        result = load_song_proposal(thread_dir, "song.yml")
        assert "color_name" in result
        assert result["color_name"] == "Violet"

    def test_time_sig_is_tuple(self, tmp_path):
        """chord_pipeline needs (num, den) tuple, not string."""
        thread_dir = tmp_path / "thread"
        self._make_proposal(thread_dir, "song.yml")

        from app.generators.midi.pipelines.chord_pipeline import load_song_proposal

        result = load_song_proposal(thread_dir, "song.yml")
        assert isinstance(result["time_sig"], tuple)
        assert result["time_sig"] == (7, 8)

    def test_key_components_present(self, tmp_path):
        thread_dir = tmp_path / "thread"
        self._make_proposal(thread_dir, "song.yml")

        from app.generators.midi.pipelines.chord_pipeline import load_song_proposal

        result = load_song_proposal(thread_dir, "song.yml")
        assert result["key_root"] == "Gb"  # F# normalised to flat spelling for chord DB
        assert result["mode"] == "Minor"

    def test_bpm_is_int(self, tmp_path):
        thread_dir = tmp_path / "thread"
        self._make_proposal(thread_dir, "song.yml")

        from app.generators.midi.pipelines.chord_pipeline import load_song_proposal

        result = load_song_proposal(thread_dir, "song.yml")
        assert isinstance(result["bpm"], int)

    def test_file_not_found_raises(self, tmp_path):
        from app.generators.midi.pipelines.chord_pipeline import load_song_proposal

        with pytest.raises(FileNotFoundError):
            load_song_proposal(tmp_path, "missing.yml")


# ---------------------------------------------------------------------------
# Tests: lyric_pipeline._find_and_load_proposal (caller 3)
# ---------------------------------------------------------------------------


class TestLyricPipelineCaller:

    def _setup(self, tmp_path: Path) -> tuple[Path, Path]:
        """Create thread dir with proposal + production dir with chord review."""
        thread_dir = tmp_path / "thread"
        yml_dir = thread_dir / "yml"
        yml_dir.mkdir(parents=True, exist_ok=True)
        _write_proposal(yml_dir / "song.yml")

        prod_dir = tmp_path / "prod"
        prod_dir.mkdir()
        _write_chord_review(prod_dir / "chords", str(thread_dir), "song.yml")
        return thread_dir, prod_dir

    def test_returns_required_fields(self, tmp_path):
        _, prod_dir = self._setup(tmp_path)

        from app.generators.midi.pipelines.lyric_pipeline import _find_and_load_proposal

        result = _find_and_load_proposal(prod_dir)
        for field in (
            "title",
            "bpm",
            "time_sig",
            "key",
            "color",
            "concept",
            "genres",
            "mood",
            "singer",
            "sounds_like",
        ):
            assert field in result, f"missing field: {field}"

    def test_time_sig_is_string(self, tmp_path):
        _, prod_dir = self._setup(tmp_path)

        from app.generators.midi.pipelines.lyric_pipeline import _find_and_load_proposal

        result = _find_and_load_proposal(prod_dir)
        assert isinstance(result["time_sig"], str)
        assert "/" in result["time_sig"]

    def test_color_present(self, tmp_path):
        _, prod_dir = self._setup(tmp_path)

        from app.generators.midi.pipelines.lyric_pipeline import _find_and_load_proposal

        result = _find_and_load_proposal(prod_dir)
        assert result["color"] == "Violet"

    def test_sounds_like_from_song_context(self, tmp_path):
        """sounds_like from song_context.yml takes precedence over proposal."""
        thread_dir, prod_dir = self._setup(tmp_path)

        # Write song_context with different sounds_like
        write_initial_proposal(
            prod_dir,
            {
                "title": "T",
                "color": "Violet",
                "concept": "c",
                "singer": "Gabriel",
                "key": "F# minor",
                "bpm": 100,
                "time_sig": "4/4",
                "genres": [],
                "mood": [],
            },
            ["Brian Eno"],
        )

        from app.generators.midi.pipelines.lyric_pipeline import _find_and_load_proposal

        result = _find_and_load_proposal(prod_dir)
        assert result["sounds_like"] == ["Brian Eno"]

    def test_returns_empty_when_no_chord_review(self, tmp_path):
        from app.generators.midi.pipelines.lyric_pipeline import _find_and_load_proposal

        result = _find_and_load_proposal(tmp_path)
        assert result == {}

    def test_returns_empty_when_proposal_missing(self, tmp_path):
        prod_dir = tmp_path / "prod"
        prod_dir.mkdir()
        # chord review points to non-existent thread/file
        _write_chord_review(prod_dir / "chords", "/no/such/thread", "missing.yml")

        from app.generators.midi.pipelines.lyric_pipeline import _find_and_load_proposal

        result = _find_and_load_proposal(prod_dir)
        assert result == {}


# ---------------------------------------------------------------------------
# Tests: composition_proposal.load_song_proposal_data (caller 4)
# ---------------------------------------------------------------------------


class TestCompositionProposalCaller:

    def _setup(self, tmp_path: Path) -> tuple[Path, Path]:
        thread_dir = tmp_path / "thread"
        yml_dir = thread_dir / "yml"
        yml_dir.mkdir(parents=True, exist_ok=True)
        _write_proposal(yml_dir / "song.yml")

        prod_dir = tmp_path / "prod"
        prod_dir.mkdir()
        _write_chord_review(prod_dir / "chords", str(thread_dir), "song.yml")
        return thread_dir, prod_dir

    def test_returns_required_fields(self, tmp_path):
        _, prod_dir = self._setup(tmp_path)

        from app.generators.midi.production.composition_proposal import (
            load_song_proposal_data,
        )

        result = load_song_proposal_data(prod_dir)
        for field in (
            "title",
            "bpm",
            "time_sig",
            "key",
            "color",
            "concept",
            "genres",
            "mood",
            "singer",
            "sounds_like",
        ):
            assert field in result, f"missing field: {field}"

    def test_time_sig_is_string(self, tmp_path):
        _, prod_dir = self._setup(tmp_path)

        from app.generators.midi.production.composition_proposal import (
            load_song_proposal_data,
        )

        result = load_song_proposal_data(prod_dir)
        assert isinstance(result["time_sig"], str)
        assert "/" in result["time_sig"]

    def test_color_present(self, tmp_path):
        _, prod_dir = self._setup(tmp_path)

        from app.generators.midi.production.composition_proposal import (
            load_song_proposal_data,
        )

        result = load_song_proposal_data(prod_dir)
        assert result["color"] == "Violet"

    def test_sounds_like_from_song_context(self, tmp_path):
        """sounds_like from song_context.yml takes precedence over proposal."""
        thread_dir, prod_dir = self._setup(tmp_path)

        write_initial_proposal(
            prod_dir,
            {
                "title": "T",
                "color": "Violet",
                "concept": "c",
                "singer": "Gabriel",
                "key": "F# minor",
                "bpm": 100,
                "time_sig": "4/4",
                "genres": [],
                "mood": [],
            },
            ["Klaus Schulze"],
        )

        from app.generators.midi.production.composition_proposal import (
            load_song_proposal_data,
        )

        result = load_song_proposal_data(prod_dir)
        assert result["sounds_like"] == ["Klaus Schulze"]

    def test_returns_empty_when_no_chord_review(self, tmp_path):
        from app.generators.midi.production.composition_proposal import (
            load_song_proposal_data,
        )

        result = load_song_proposal_data(tmp_path)
        assert result == {}

    def test_color_fallback_to_chord_review(self, tmp_path):
        """If proposal has no color, falls back to chord_review color."""
        thread_dir = tmp_path / "thread"
        yml_dir = thread_dir / "yml"
        yml_dir.mkdir(parents=True, exist_ok=True)

        # Proposal with empty rainbow_color
        proposal = {
            "title": "T",
            "bpm": 120,
            "key": "C major",
            "rainbow_color": "",
            "concept": "c",
            "genres": [],
            "mood": [],
            "singer": "",
        }
        (yml_dir / "song.yml").write_text(yaml.dump(proposal))

        prod_dir = tmp_path / "prod"
        prod_dir.mkdir()
        _write_chord_review(
            prod_dir / "chords", str(thread_dir), "song.yml", color="Green"
        )

        from app.generators.midi.production.composition_proposal import (
            load_song_proposal_data,
        )

        result = load_song_proposal_data(prod_dir)
        assert result["color"] == "Green"
