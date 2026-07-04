"""Tests for lp_sides."""

import numpy as np
import soundfile as sf

from white_composition.lp_sides import (
    SidesDocument,
    assign_song,
    load_sides,
    mix_duration_seconds,
    move_song,
    remove_song,
    save_sides,
    side_totals,
)


def _write_wav(path, seconds: float = 2.0, samplerate: int = 8000):
    samples = np.zeros(int(seconds * samplerate), dtype="float32")
    sf.write(str(path), samples, samplerate)


class TestMixDurationSeconds:
    def test_valid_wav(self, tmp_path):
        wav_path = tmp_path / "mix.wav"
        _write_wav(wav_path, seconds=3.0)
        duration = mix_duration_seconds(wav_path)
        assert duration == 3.0

    def test_missing_file(self, tmp_path):
        assert mix_duration_seconds(tmp_path / "nope.wav") is None

    def test_unreadable_file(self, tmp_path):
        bad_path = tmp_path / "bad.wav"
        bad_path.write_text("not audio")
        assert mix_duration_seconds(bad_path) is None


class TestLoadSaveSides:
    def test_load_absent_returns_empty(self, tmp_path):
        doc = load_sides(tmp_path)
        assert set(doc.sides.keys()) == {"A", "B", "C", "D"}
        assert all(side.songs == [] for side in doc.sides.values())
        assert doc.side_limit_seconds == 1200.0

    def test_round_trip(self, tmp_path):
        doc = SidesDocument.empty()
        assign_song(doc, "thread__song_a", "A", 0, 187.4)
        save_sides(tmp_path, doc)

        loaded = load_sides(tmp_path)
        assert loaded.sides["A"].songs[0].song_id == "thread__song_a"
        assert loaded.sides["A"].songs[0].duration_seconds == 187.4


class TestAssignMoveRemove:
    def test_assign_inserts_at_position(self):
        doc = SidesDocument.empty()
        assign_song(doc, "song_1", "A", 0, 100.0)
        assign_song(doc, "song_2", "A", 0, 200.0)
        ids = [s.song_id for s in doc.sides["A"].songs]
        assert ids == ["song_2", "song_1"]

    def test_assign_removes_from_previous_side(self):
        doc = SidesDocument.empty()
        assign_song(doc, "song_1", "A", 0, 100.0)
        assign_song(doc, "song_1", "B", 0, 100.0)
        assert doc.sides["A"].songs == []
        assert doc.sides["B"].songs[0].song_id == "song_1"

    def test_assign_unknown_side_raises(self):
        doc = SidesDocument.empty()
        try:
            assign_song(doc, "song_1", "Z", 0, 100.0)
            assert False, "expected ValueError"
        except ValueError:
            pass

    def test_move_between_sides_preserves_duration(self):
        doc = SidesDocument.empty()
        assign_song(doc, "song_1", "A", 0, 150.0)
        move_song(doc, "song_1", "C", 0)
        assert doc.sides["A"].songs == []
        assert doc.sides["C"].songs[0].duration_seconds == 150.0

    def test_move_unassigned_song_raises(self):
        doc = SidesDocument.empty()
        try:
            move_song(doc, "ghost", "A", 0)
            assert False, "expected ValueError"
        except ValueError:
            pass

    def test_remove_song(self):
        doc = SidesDocument.empty()
        assign_song(doc, "song_1", "A", 0, 100.0)
        remove_song(doc, "song_1")
        assert doc.sides["A"].songs == []

    def test_remove_absent_song_is_noop(self):
        doc = SidesDocument.empty()
        remove_song(doc, "ghost")
        assert all(side.songs == [] for side in doc.sides.values())


class TestSideTotals:
    def test_totals_under_limit(self):
        doc = SidesDocument.empty()
        assign_song(doc, "song_1", "A", 0, 300.0)
        assign_song(doc, "song_2", "A", 1, 400.0)
        totals = side_totals(doc)
        assert totals["A"]["total_seconds"] == 700.0
        assert totals["A"]["over_limit"] is False

    def test_totals_over_limit(self):
        doc = SidesDocument.empty()
        assign_song(doc, "song_1", "A", 0, 700.0)
        assign_song(doc, "song_2", "A", 1, 700.0)
        totals = side_totals(doc)
        assert totals["A"]["total_seconds"] == 1400.0
        assert totals["A"]["over_limit"] is True

    def test_empty_side_zero_total(self):
        doc = SidesDocument.empty()
        totals = side_totals(doc)
        assert totals["B"]["total_seconds"] == 0.0
        assert totals["B"]["over_limit"] is False
