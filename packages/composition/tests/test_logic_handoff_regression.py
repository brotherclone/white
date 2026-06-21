"""Tests for regression_info() in logic_handoff."""

from pathlib import Path

import pytest

from white_composition.logic_handoff import regression_info


def _make_logic_dir(tmp_path: Path, files: list[str]) -> Path:
    """Create a fake Logic song directory with specified relative file paths."""
    d = tmp_path / "logic_song"
    d.mkdir()
    for rel in files:
        p = d / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("placeholder")
    return d


# ---------------------------------------------------------------------------
# Task 8.8: regression_info — validation, destructive detection, multi-stage
# ---------------------------------------------------------------------------


class TestRegressionInfoValidation:
    def test_forward_movement_raises(self, tmp_path):
        d = _make_logic_dir(tmp_path, [])
        with pytest.raises(ValueError, match="Cannot regress forward"):
            regression_info(d, "lyrics", "recording")

    def test_same_stage_raises(self, tmp_path):
        d = _make_logic_dir(tmp_path, [])
        with pytest.raises(ValueError, match="Cannot regress forward"):
            regression_info(d, "lyrics", "lyrics")

    def test_unknown_current_stage_raises(self, tmp_path):
        d = _make_logic_dir(tmp_path, [])
        with pytest.raises(ValueError, match="Unknown current stage"):
            regression_info(d, "nonexistent", "lyrics")

    def test_unknown_target_stage_raises(self, tmp_path):
        d = _make_logic_dir(tmp_path, [])
        with pytest.raises(ValueError, match="Unknown target stage"):
            regression_info(d, "recording", "nonexistent")


class TestRegressionInfoDestructive:
    def test_vocal_placeholders_to_lyrics_is_destructive(self, tmp_path):
        d = _make_logic_dir(tmp_path, ["MIDI/melody/vocal_placeholder_intro.mid"])
        result = regression_info(d, "vocal_placeholders", "lyrics")
        assert result["destructive"] is True
        assert any(
            "vocal_placeholder_intro.mid" in f for f in result["files_to_delete"]
        )

    def test_mix_candidate_to_rough_mix_non_destructive(self, tmp_path):
        # No REGRESSION_FILE_MAP entries for mix_candidate or rough_mix
        d = _make_logic_dir(tmp_path, [])
        result = regression_info(d, "mix_candidate", "rough_mix")
        assert result["destructive"] is False
        assert result["files_to_delete"] == []

    def test_no_matching_files_non_destructive(self, tmp_path):
        # Stage has patterns but no files exist
        d = _make_logic_dir(tmp_path, [])
        result = regression_info(d, "recording", "lyrics")
        assert result["destructive"] is False
        assert result["files_to_delete"] == []


class TestRegressionInfoMultiStage:
    def test_skips_over_multiple_stages_collects_all(self, tmp_path):
        # Regressing from "recording" to "structure" passes through lyrics,
        # vocal_placeholders, and recording — all three should be collected.
        d = _make_logic_dir(
            tmp_path,
            [
                "lyrics_verse.txt",
                "MIDI/melody/vocal_placeholder_verse.mid",
                "Recordings/guitar_take_1.wav",
            ],
        )
        result = regression_info(d, "recording", "structure")
        files = result["files_to_delete"]
        assert any("lyrics_verse.txt" in f for f in files)
        assert any("vocal_placeholder_verse.mid" in f for f in files)
        assert any("guitar_take_1.wav" in f for f in files)
        assert result["destructive"] is True

    def test_target_stage_files_not_deleted(self, tmp_path):
        # Regressing to "lyrics": lyrics files should NOT be in the delete list
        # because the target stage itself is preserved
        d = _make_logic_dir(
            tmp_path,
            [
                "lyrics_verse.txt",
                "MIDI/melody/vocal_placeholder_verse.mid",
            ],
        )
        result = regression_info(d, "vocal_placeholders", "lyrics")
        # vocal_placeholder files in the vocal_placeholders stage → deleted
        assert any(
            "vocal_placeholder_verse.mid" in f for f in result["files_to_delete"]
        )
        # lyrics_verse.txt belongs to "lyrics" stage — the target, not deleted
        assert not any("lyrics_verse.txt" in f for f in result["files_to_delete"])

    def test_no_files_returns_empty_list(self, tmp_path):
        d = _make_logic_dir(tmp_path, [])
        result = regression_info(d, "augmentation", "lyrics")
        assert result == {"destructive": False, "files_to_delete": []}
