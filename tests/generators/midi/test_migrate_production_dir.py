"""Tests for migrate_production_dir.py."""

from __future__ import annotations

from pathlib import Path

import yaml

from app.generators.midi.production.migrate_production_dir import (
    migrate_production_dir,
    _detect_phase_status,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_chord_review(
    tmp_path: Path, thread: str, song_proposal: str, **overrides
) -> Path:
    """Write a minimal chords/review.yml and return its path."""
    chords_dir = tmp_path / "chords"
    chords_dir.mkdir(parents=True, exist_ok=True)
    data = {
        "pipeline": "chord_pipeline",
        "song_proposal": song_proposal,
        "thread": thread,
        "key": "F# minor",
        "bpm": 96,
        "color": "Red",
        "time_sig": "4/4",
        "singer": "Gabriel",
        "scoring_weights": {"theory": 0.3, "chromatic": 0.7},
        "candidates": [
            {
                "label": "verse",
                "status": "approved",
                "midi_file": "candidates/verse.mid",
                "score": 0.82,
            },
            {
                "label": "chorus",
                "status": "pending",
                "midi_file": "candidates/chorus.mid",
                "score": 0.71,
            },
        ],
        **overrides,
    }
    path = chords_dir / "review.yml"
    path.write_text(yaml.dump(data))
    return path


def _make_song_proposal(tmp_path: Path, name: str = "my_song.yml") -> Path:
    """Write a minimal song proposal YAML under tmp_path/yml/ and return its path."""
    yml_dir = tmp_path / "yml"
    yml_dir.mkdir(parents=True, exist_ok=True)
    data = {
        "title": "My Test Song",
        "bpm": 96,
        "tempo": {"numerator": 4, "denominator": 4},
        "key": "F# minor",
        "rainbow_color": {"color_name": "Red"},
        "concept": "A test concept about memory and loss in vivid detail.",
        "genres": ["ambient", "folk"],
        "mood": ["melancholic", "sparse"],
        "singer": "Gabriel",
    }
    path = yml_dir / name
    path.write_text(yaml.dump(data))
    return path


# ---------------------------------------------------------------------------
# Tests: _detect_phase_status
# ---------------------------------------------------------------------------


class TestDetectPhaseStatus:

    def test_pending_when_no_review_files(self, tmp_path):
        statuses = _detect_phase_status(tmp_path)
        for phase in ["chords", "drums", "bass", "melody", "lyrics"]:
            assert statuses[phase] == "pending"

    def test_complete_when_approved_candidate(self, tmp_path):
        chords_dir = tmp_path / "chords"
        chords_dir.mkdir()
        (chords_dir / "review.yml").write_text(
            yaml.dump({"candidates": [{"label": "verse", "status": "approved"}]})
        )
        statuses = _detect_phase_status(tmp_path)
        assert statuses["chords"] == "complete"

    def test_pending_when_no_approved_candidate(self, tmp_path):
        chords_dir = tmp_path / "chords"
        chords_dir.mkdir()
        (chords_dir / "review.yml").write_text(
            yaml.dump({"candidates": [{"label": "verse", "status": "pending"}]})
        )
        statuses = _detect_phase_status(tmp_path)
        assert statuses["chords"] == "pending"

    def test_composition_proposal_complete_when_file_exists(self, tmp_path):
        (tmp_path / "composition_proposal.yml").write_text(
            yaml.dump({"proposed_by": "claude"})
        )
        statuses = _detect_phase_status(tmp_path)
        assert statuses["composition_proposal"] == "complete"


# ---------------------------------------------------------------------------
# Tests: migrate_production_dir
# ---------------------------------------------------------------------------


class TestMigrateProductionDir:

    def test_writes_song_context_yml(self, tmp_path):
        thread_dir = tmp_path / "thread"
        _make_song_proposal(thread_dir)
        prod_dir = tmp_path / "production" / "red__my_song"
        prod_dir.mkdir(parents=True)
        _make_chord_review(
            prod_dir, thread=str(thread_dir), song_proposal="my_song.yml"
        )

        result = migrate_production_dir(prod_dir)

        assert result is not None
        assert (prod_dir / "song_context.yml").exists()

    def test_context_has_required_fields(self, tmp_path):
        thread_dir = tmp_path / "thread"
        _make_song_proposal(thread_dir)
        prod_dir = tmp_path / "production" / "red__my_song"
        prod_dir.mkdir(parents=True)
        _make_chord_review(
            prod_dir, thread=str(thread_dir), song_proposal="my_song.yml"
        )

        migrate_production_dir(prod_dir)
        ctx = yaml.safe_load((prod_dir / "song_context.yml").read_text())

        assert ctx["color"] == "Red"
        assert ctx["key"] == "F# minor"
        assert ctx["bpm"] == 96
        assert ctx["time_sig"] == "4/4"
        assert ctx["singer"] == "Gabriel"
        assert ctx["schema_version"] == "1"
        assert "phases" in ctx

    def test_concept_loaded_from_proposal(self, tmp_path):
        thread_dir = tmp_path / "thread"
        _make_song_proposal(thread_dir)
        prod_dir = tmp_path / "production" / "red__my_song"
        prod_dir.mkdir(parents=True)
        _make_chord_review(
            prod_dir, thread=str(thread_dir), song_proposal="my_song.yml"
        )

        migrate_production_dir(prod_dir)
        ctx = yaml.safe_load((prod_dir / "song_context.yml").read_text())

        assert "memory and loss" in ctx["concept"].lower()

    def test_sounds_like_from_initial_proposal(self, tmp_path):
        thread_dir = tmp_path / "thread"
        _make_song_proposal(thread_dir)
        prod_dir = tmp_path / "production" / "red__my_song"
        prod_dir.mkdir(parents=True)
        _make_chord_review(
            prod_dir, thread=str(thread_dir), song_proposal="my_song.yml"
        )
        (prod_dir / "initial_proposal.yml").write_text(
            yaml.dump({"sounds_like": ["Grouper", "Stars of the Lid"]})
        )

        migrate_production_dir(prod_dir)
        ctx = yaml.safe_load((prod_dir / "song_context.yml").read_text())

        assert ctx["sounds_like"] == ["Grouper", "Stars of the Lid"]

    def test_phase_status_detected(self, tmp_path):
        thread_dir = tmp_path / "thread"
        _make_song_proposal(thread_dir)
        prod_dir = tmp_path / "production" / "red__my_song"
        prod_dir.mkdir(parents=True)
        _make_chord_review(
            prod_dir, thread=str(thread_dir), song_proposal="my_song.yml"
        )
        # Chord phase has approved candidate
        drums_dir = prod_dir / "drums"
        drums_dir.mkdir()
        (drums_dir / "review.yml").write_text(
            yaml.dump({"candidates": [{"label": "verse", "status": "approved"}]})
        )

        migrate_production_dir(prod_dir)
        ctx = yaml.safe_load((prod_dir / "song_context.yml").read_text())

        assert (
            ctx["phases"]["chords"] == "complete"
        )  # from _make_chord_review (has approved)
        assert ctx["phases"]["drums"] == "complete"  # from drums/review.yml
        assert ctx["phases"]["bass"] == "pending"  # no review

    def test_non_destructive_existing_files_unchanged(self, tmp_path):
        thread_dir = tmp_path / "thread"
        _make_song_proposal(thread_dir)
        prod_dir = tmp_path / "production" / "red__my_song"
        prod_dir.mkdir(parents=True)
        review_path = _make_chord_review(
            prod_dir, thread=str(thread_dir), song_proposal="my_song.yml"
        )
        original_review = review_path.read_text()

        migrate_production_dir(prod_dir)

        assert review_path.read_text() == original_review

    def test_dry_run_does_not_write(self, tmp_path):
        thread_dir = tmp_path / "thread"
        _make_song_proposal(thread_dir)
        prod_dir = tmp_path / "production" / "red__my_song"
        prod_dir.mkdir(parents=True)
        _make_chord_review(
            prod_dir, thread=str(thread_dir), song_proposal="my_song.yml"
        )

        migrate_production_dir(prod_dir, dry_run=True)

        assert not (prod_dir / "song_context.yml").exists()

    def test_dry_run_returns_context_dict(self, tmp_path):
        thread_dir = tmp_path / "thread"
        _make_song_proposal(thread_dir)
        prod_dir = tmp_path / "production" / "red__my_song"
        prod_dir.mkdir(parents=True)
        _make_chord_review(
            prod_dir, thread=str(thread_dir), song_proposal="my_song.yml"
        )

        result = migrate_production_dir(prod_dir, dry_run=True)

        assert result is not None
        assert result["color"] == "Red"

    def test_returns_none_when_no_chord_review(self, tmp_path):
        prod_dir = tmp_path / "production" / "empty"
        prod_dir.mkdir(parents=True)

        result = migrate_production_dir(prod_dir)

        assert result is None
        assert not (prod_dir / "song_context.yml").exists()

    def test_skips_if_song_context_already_exists(self, tmp_path):
        prod_dir = tmp_path / "production" / "already_migrated"
        prod_dir.mkdir(parents=True)
        existing = {"schema_version": "1", "color": "Blue"}
        (prod_dir / "song_context.yml").write_text(yaml.dump(existing))

        result = migrate_production_dir(prod_dir)

        assert result is None
        # File unchanged
        ctx = yaml.safe_load((prod_dir / "song_context.yml").read_text())
        assert ctx["color"] == "Blue"

    def test_fallback_when_proposal_path_missing(self, tmp_path):
        prod_dir = tmp_path / "production" / "red__my_song"
        prod_dir.mkdir(parents=True)
        # Thread/proposal don't exist on disk — should fall back to chord review values
        _make_chord_review(
            prod_dir,
            thread="/nonexistent/thread",
            song_proposal="nonexistent.yml",
        )

        result = migrate_production_dir(prod_dir)

        assert result is not None
        ctx = yaml.safe_load((prod_dir / "song_context.yml").read_text())
        assert ctx["color"] == "Red"  # from chord review
        assert ctx["bpm"] == 96
