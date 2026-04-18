"""Tests for pipeline_runner.py."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import yaml


def _write_song_context(production_dir: Path, phases: dict | None = None) -> None:
    ctx = {
        "schema_version": "1",
        "title": "Test Song",
        "color": "Red",
        "concept": "test",
        "singer": "gabriel",
        "song_proposal": "",
        "key": "C major",
        "bpm": 120,
        "time_sig": "4/4",
        "sounds_like": [],
        "genres": [],
        "mood": [],
        "phases": phases
        or {
            "init_production": "pending",
            "chords": "pending",
            "drums": "pending",
            "bass": "pending",
            "melody": "pending",
            "lyrics": "pending",
        },
    }
    (production_dir / "song_context.yml").write_text(yaml.dump(ctx))


# ---------------------------------------------------------------------------
# 1. read_phase_statuses
# ---------------------------------------------------------------------------


class TestReadPhaseStatuses:
    def test_reads_all_phases(self, tmp_path):
        from app.generators.midi.production.pipeline_runner import read_phase_statuses

        _write_song_context(
            tmp_path,
            {
                "init_production": "promoted",
                "chords": "promoted",
                "drums": "generated",
                "bass": "pending",
                "melody": "pending",
                "lyrics": "pending",
            },
        )
        statuses = read_phase_statuses(tmp_path)
        assert statuses["init_production"] == "promoted"
        assert statuses["chords"] == "promoted"
        assert statuses["drums"] == "generated"
        assert statuses["bass"] == "pending"

    def test_defaults_missing_phases_to_pending(self, tmp_path):
        from app.generators.midi.production.pipeline_runner import read_phase_statuses

        _write_song_context(tmp_path, {})
        statuses = read_phase_statuses(tmp_path)
        for phase in ["init_production", "chords", "drums", "bass", "melody", "lyrics"]:
            assert statuses[phase] == "pending"

    def test_missing_song_context_defaults_all_pending(self, tmp_path):
        from app.generators.midi.production.pipeline_runner import read_phase_statuses

        statuses = read_phase_statuses(tmp_path)
        for phase in ["init_production", "chords"]:
            assert statuses[phase] == "pending"


# ---------------------------------------------------------------------------
# 2. get_next_runnable_phase
# ---------------------------------------------------------------------------


class TestGetNextRunnablePhase:
    def test_all_pending_returns_init(self):
        from app.generators.midi.production.pipeline_runner import (
            get_next_runnable_phase,
        )

        statuses = {
            p: "pending"
            for p in ["init_production", "chords", "drums", "bass", "melody", "lyrics"]
        }
        assert get_next_runnable_phase(statuses) == "init_production"

    def test_init_promoted_returns_chords(self):
        from app.generators.midi.production.pipeline_runner import (
            get_next_runnable_phase,
        )

        statuses = {
            "init_production": "promoted",
            "chords": "pending",
            "drums": "pending",
            "bass": "pending",
            "melody": "pending",
            "lyrics": "pending",
        }
        assert get_next_runnable_phase(statuses) == "chords"

    def test_chords_generated_not_yet_promoted_blocks_drums(self):
        from app.generators.midi.production.pipeline_runner import (
            get_next_runnable_phase,
        )

        statuses = {
            "init_production": "promoted",
            "chords": "generated",  # not yet promoted
            "drums": "pending",
            "bass": "pending",
            "melody": "pending",
            "lyrics": "pending",
        }
        # chords not promoted → drums blocked → only chords could run but it's already generated
        result = get_next_runnable_phase(statuses)
        assert result is None

    def test_all_promoted_returns_none(self):
        from app.generators.midi.production.pipeline_runner import (
            get_next_runnable_phase,
        )

        statuses = {
            p: "promoted"
            for p in [
                "init_production",
                "chords",
                "drums",
                "bass",
                "melody",
                "lyrics",
                "decisions",
            ]
        }
        assert get_next_runnable_phase(statuses) is None

    def test_melody_promoted_returns_lyrics(self):
        from app.generators.midi.production.pipeline_runner import (
            get_next_runnable_phase,
        )

        statuses = {
            "init_production": "promoted",
            "chords": "promoted",
            "drums": "promoted",
            "bass": "promoted",
            "melody": "promoted",
            "lyrics": "pending",
        }
        assert get_next_runnable_phase(statuses) == "lyrics"


# ---------------------------------------------------------------------------
# 3. write_phase_status
# ---------------------------------------------------------------------------


class TestWritePhaseStatus:
    def test_updates_single_phase(self, tmp_path):
        from app.generators.midi.production.pipeline_runner import (
            read_phase_statuses,
            write_phase_status,
        )

        _write_song_context(tmp_path)
        write_phase_status(tmp_path, "chords", "generated")
        statuses = read_phase_statuses(tmp_path)
        assert statuses["chords"] == "generated"
        assert statuses["drums"] == "pending"  # unchanged

    def test_noop_when_no_song_context(self, tmp_path):
        from app.generators.midi.production.pipeline_runner import write_phase_status

        # Should not raise
        write_phase_status(tmp_path, "chords", "generated")


# ---------------------------------------------------------------------------
# 4. cmd_status
# ---------------------------------------------------------------------------


class TestCmdStatus:
    def test_prints_phase_names(self, tmp_path, capsys):
        from app.generators.midi.production.pipeline_runner import cmd_status

        _write_song_context(
            tmp_path,
            {
                "init_production": "promoted",
                "chords": "promoted",
                "drums": "pending",
                "bass": "pending",
                "melody": "pending",
                "lyrics": "pending",
            },
        )
        cmd_status(tmp_path)
        out = capsys.readouterr().out
        assert "init_production" in out
        assert "chords" in out
        assert "drums" in out
        assert "✅" in out  # promoted icon

    def test_prints_next_command(self, tmp_path, capsys):
        from app.generators.midi.production.pipeline_runner import cmd_status

        _write_song_context(
            tmp_path,
            {
                "init_production": "promoted",
                "chords": "pending",
                "drums": "pending",
                "bass": "pending",
                "melody": "pending",
                "lyrics": "pending",
            },
        )
        cmd_status(tmp_path)
        out = capsys.readouterr().out
        assert "chords" in out


# ---------------------------------------------------------------------------
# 5. cmd_run (mocked subprocess)
# ---------------------------------------------------------------------------


class TestCmdRun:
    def test_runs_next_phase_and_updates_status(self, tmp_path):
        from app.generators.midi.production.pipeline_runner import (
            cmd_run,
            read_phase_statuses,
        )

        _write_song_context(
            tmp_path,
            {
                "init_production": "promoted",
                "chords": "pending",
                "drums": "pending",
                "bass": "pending",
                "melody": "pending",
                "lyrics": "pending",
            },
        )
        with patch(
            "app.generators.midi.production.pipeline_runner.subprocess.run"
        ) as mock_run:
            mock_run.return_value = MagicMock(returncode=0)
            result = cmd_run(tmp_path)

        assert result == 0
        statuses = read_phase_statuses(tmp_path)
        assert statuses["chords"] == "generated"
        assert statuses["drums"] == "pending"  # not advanced

    def test_resets_status_on_failure(self, tmp_path):
        from app.generators.midi.production.pipeline_runner import (
            cmd_run,
            read_phase_statuses,
        )

        _write_song_context(
            tmp_path,
            {
                "init_production": "promoted",
                "chords": "pending",
                "drums": "pending",
                "bass": "pending",
                "melody": "pending",
                "lyrics": "pending",
            },
        )
        with patch(
            "app.generators.midi.production.pipeline_runner.subprocess.run"
        ) as mock_run:
            mock_run.return_value = MagicMock(returncode=1)
            result = cmd_run(tmp_path)

        assert result == 1
        statuses = read_phase_statuses(tmp_path)
        assert statuses["chords"] == "pending"  # reset

    def test_returns_0_when_nothing_to_run(self, tmp_path):
        from app.generators.midi.production.pipeline_runner import cmd_run

        _write_song_context(
            tmp_path,
            {
                p: "promoted"
                for p in [
                    "init_production",
                    "chords",
                    "drums",
                    "bass",
                    "melody",
                    "lyrics",
                ]
            },
        )
        result = cmd_run(tmp_path)
        assert result == 0


# ---------------------------------------------------------------------------
# 6. promote_part status sync
# ---------------------------------------------------------------------------


class TestPromotePartStatusSync:
    def test_promote_writes_promoted_status(self, tmp_path):
        from app.generators.midi.production.promote_part import _sync_promote_status

        _write_song_context(
            tmp_path,
            {
                "init_production": "promoted",
                "chords": "generated",
                "drums": "pending",
                "bass": "pending",
                "melody": "pending",
                "lyrics": "pending",
            },
        )
        # Simulate chords/review.yml path
        chords_dir = tmp_path / "chords"
        chords_dir.mkdir()
        review_path = chords_dir / "review.yml"
        review_path.write_text("candidates: []")

        _sync_promote_status(review_path)

        with open(tmp_path / "song_context.yml") as f:
            ctx = yaml.safe_load(f)
        assert ctx["phases"]["chords"] == "promoted"
        assert ctx["phases"]["drums"] == "pending"  # unchanged

    def test_lyrics_review_maps_to_lyrics_phase(self, tmp_path):
        from app.generators.midi.production.promote_part import _sync_promote_status

        _write_song_context(tmp_path, {"lyrics": "generated"})
        melody_dir = tmp_path / "melody"
        melody_dir.mkdir()
        review_path = melody_dir / "lyrics_review.yml"
        review_path.write_text("candidates: []")

        _sync_promote_status(review_path)

        with open(tmp_path / "song_context.yml") as f:
            ctx = yaml.safe_load(f)
        assert ctx["phases"]["lyrics"] == "promoted"

    def test_noop_when_no_song_context(self, tmp_path):
        from app.generators.midi.production.promote_part import _sync_promote_status

        drums_dir = tmp_path / "drums"
        drums_dir.mkdir()
        review_path = drums_dir / "review.yml"
        review_path.write_text("candidates: []")
        # No song_context.yml — should not raise
        _sync_promote_status(review_path)


# ---------------------------------------------------------------------------
# 7. cmd_batch
# ---------------------------------------------------------------------------


class TestCmdBatch:
    def test_batch_finds_pending_dirs(self, tmp_path):
        from app.generators.midi.production.pipeline_runner import cmd_batch

        # Create two production dirs
        for name in ["song_a", "song_b"]:
            d = tmp_path / "production" / name
            d.mkdir(parents=True)
            _write_song_context(
                d,
                {
                    "init_production": "promoted",
                    "chords": "pending",
                    "drums": "pending",
                    "bass": "pending",
                    "melody": "pending",
                    "lyrics": "pending",
                },
            )

        with patch(
            "app.generators.midi.production.pipeline_runner.subprocess.run"
        ) as mock_run:
            mock_run.return_value = MagicMock(returncode=0)
            result = cmd_batch(tmp_path, "chords")

        assert result == 0
        assert mock_run.call_count == 2

    def test_batch_skips_promoted_dirs(self, tmp_path):
        from app.generators.midi.production.pipeline_runner import cmd_batch

        # song_a already has chords promoted
        d_a = tmp_path / "production" / "song_a"
        d_a.mkdir(parents=True)
        _write_song_context(
            d_a,
            {
                "init_production": "promoted",
                "chords": "promoted",
                "drums": "pending",
                "bass": "pending",
                "melody": "pending",
                "lyrics": "pending",
            },
        )

        # song_b has chords pending
        d_b = tmp_path / "production" / "song_b"
        d_b.mkdir(parents=True)
        _write_song_context(
            d_b,
            {
                "init_production": "promoted",
                "chords": "pending",
                "drums": "pending",
                "bass": "pending",
                "melody": "pending",
                "lyrics": "pending",
            },
        )

        with patch(
            "app.generators.midi.production.pipeline_runner.subprocess.run"
        ) as mock_run:
            mock_run.return_value = MagicMock(returncode=0)
            result = cmd_batch(tmp_path, "chords")

        assert result == 0
        assert mock_run.call_count == 1  # only song_b
