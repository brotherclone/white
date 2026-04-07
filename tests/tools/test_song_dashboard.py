"""Tests for app/tools/song_dashboard.py"""

from pathlib import Path

import yaml

from app.tools.song_dashboard import (
    STATUS_APPROVED,
    STATUS_NO_CANDIDATES,
    STATUS_NOT_STARTED,
    STATUS_PENDING,
    _color_rank,
    build_table,
    phase_status,
    scan_album,
    scan_production_dir,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _write_review(path: Path, candidates: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.dump({"candidates": candidates}))


def _write_plan(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.dump(data))


# ---------------------------------------------------------------------------
# phase_status
# ---------------------------------------------------------------------------


class TestPhaseStatus:
    def test_not_started_when_dir_missing(self, tmp_path):
        result = phase_status(tmp_path / "song", "chords")
        assert result == STATUS_NOT_STARTED

    def test_no_candidates_when_review_missing(self, tmp_path):
        phase_dir = tmp_path / "song" / "chords"
        phase_dir.mkdir(parents=True)
        result = phase_status(tmp_path / "song", "chords")
        assert result == STATUS_NO_CANDIDATES

    def test_no_candidates_when_empty_review(self, tmp_path):
        prod = tmp_path / "song"
        _write_review(prod / "chords" / "review.yml", [])
        result = phase_status(prod, "chords")
        assert result == STATUS_NO_CANDIDATES

    def test_approved_when_any_candidate_approved(self, tmp_path):
        prod = tmp_path / "song"
        _write_review(
            prod / "chords" / "review.yml",
            [{"id": "a", "status": "approved"}, {"id": "b", "status": "pending"}],
        )
        result = phase_status(prod, "chords")
        assert result == STATUS_APPROVED

    def test_approved_case_insensitive(self, tmp_path):
        prod = tmp_path / "song"
        _write_review(
            prod / "chords" / "review.yml",
            [{"id": "a", "status": "Approved"}],
        )
        result = phase_status(prod, "chords")
        assert result == STATUS_APPROVED

    def test_pending_when_candidates_but_none_approved(self, tmp_path):
        prod = tmp_path / "song"
        _write_review(
            prod / "chords" / "review.yml",
            [{"id": "a", "status": "pending"}, {"id": "b", "status": "rejected"}],
        )
        result = phase_status(prod, "chords")
        assert result == STATUS_PENDING

    def test_no_candidates_on_corrupt_yaml(self, tmp_path):
        prod = tmp_path / "song"
        phase_dir = prod / "chords"
        phase_dir.mkdir(parents=True)
        (phase_dir / "review.yml").write_text("{{{{not yaml")
        result = phase_status(prod, "chords")
        assert result == STATUS_NO_CANDIDATES


# ---------------------------------------------------------------------------
# scan_production_dir
# ---------------------------------------------------------------------------


class TestScanProductionDir:
    def _make_prod(self, tmp_path, color="red", key="C major", bpm=120):
        prod = tmp_path / "red__test_v1"
        prod.mkdir(parents=True)
        _write_plan(
            prod / "production_plan.yml",
            {
                "color": color,
                "key": key,
                "bpm": bpm,
                "singer": "Gabriel",
                "sections": [
                    {"label": "intro", "bars": 4, "play_count": 1},
                    {"label": "verse", "bars": 8, "play_count": 2},
                ],
            },
        )
        return prod

    def test_basic_fields(self, tmp_path):
        prod = self._make_prod(tmp_path, color="red", key="C major", bpm=120)
        status = scan_production_dir(prod, "album1")
        assert status.color == "red"
        assert status.key == "C major"
        assert status.bpm == "120"
        assert status.singer == "Gabriel"
        assert status.plan_present is True
        assert status.slug == "red__test_v1"
        assert status.album_slug == "album1"

    def test_bar_count_from_plan(self, tmp_path):
        prod = self._make_prod(tmp_path)
        # intro: 4*1=4, verse: 8*2=16 → total 20
        status = scan_production_dir(prod, "album1")
        assert status.total_approved_bars == 20

    def test_phase_statuses_present(self, tmp_path):
        prod = self._make_prod(tmp_path)
        status = scan_production_dir(prod, "album1")
        assert set(status.phase_statuses.keys()) == {
            "chords",
            "drums",
            "bass",
            "melody",
            "quartet",
        }

    def test_lyrics_present_flag(self, tmp_path):
        prod = self._make_prod(tmp_path)
        status = scan_production_dir(prod, "album1")
        assert status.lyrics_present is False
        # Create the lyrics file
        (prod / "melody").mkdir(parents=True)
        (prod / "melody" / "lyrics.txt").write_text("some lyrics")
        status2 = scan_production_dir(prod, "album1")
        assert status2.lyrics_present is True

    def test_color_falls_back_to_chord_review(self, tmp_path):
        prod = tmp_path / "test_song"
        prod.mkdir(parents=True)
        # No production_plan.yml
        _write_review(
            prod / "chords" / "review.yml",
            [{"id": "a", "status": "approved", "color": "blue"}],
        )
        # Write review with color at top level
        (prod / "chords" / "review.yml").write_text(
            yaml.dump(
                {"color": "blue", "candidates": [{"id": "a", "status": "approved"}]}
            )
        )
        status = scan_production_dir(prod, "album1")
        assert status.color == "blue"


# ---------------------------------------------------------------------------
# scan_album
# ---------------------------------------------------------------------------


class TestScanAlbum:
    def _make_album(self, tmp_path):
        album = tmp_path / "test-album"
        prod_root = album / "production"

        # Song 1: red
        red = prod_root / "red__song_a_v1"
        red.mkdir(parents=True)
        _write_plan(
            red / "production_plan.yml",
            {
                "color": "red",
                "key": "C major",
                "bpm": 100,
                "singer": "Shirley",
                "sections": [],
            },
        )

        # Song 2: blue
        blue = prod_root / "blue__song_b_v1"
        blue.mkdir(parents=True)
        _write_plan(
            blue / "production_plan.yml",
            {
                "color": "blue",
                "key": "G minor",
                "bpm": 130,
                "singer": "Gabriel",
                "sections": [],
            },
        )

        # Empty dir — should be skipped
        (prod_root / "empty_dir").mkdir()

        return album

    def test_returns_all_non_empty_dirs(self, tmp_path):
        album = self._make_album(tmp_path)
        results = scan_album(album)
        assert len(results) == 2

    def test_missing_production_dir_returns_empty(self, tmp_path):
        album = tmp_path / "empty-album"
        album.mkdir()
        assert scan_album(album) == []

    def test_colors_read_correctly(self, tmp_path):
        album = self._make_album(tmp_path)
        results = scan_album(album)
        colors = {s.color for s in results}
        assert colors == {"red", "blue"}


# ---------------------------------------------------------------------------
# _color_rank
# ---------------------------------------------------------------------------


class TestColorRank:
    def test_red_before_blue(self):
        assert _color_rank("red") < _color_rank("blue")

    def test_unknown_color_is_last(self):
        assert _color_rank("pink") == 99

    def test_black_near_end(self):
        assert _color_rank("black") < 99


# ---------------------------------------------------------------------------
# build_table
# ---------------------------------------------------------------------------


class TestBuildTable:
    def _make_status(self, color="red", slug="red__test_v1"):
        from app.tools.song_dashboard import (
            STATUS_APPROVED,
            STATUS_NOT_STARTED,
            SongStatus,
        )

        return SongStatus(
            slug=slug,
            album_slug="test-album",
            color=color,
            singer="Gabriel",
            key="C major",
            bpm="120",
            phase_statuses={
                "chords": STATUS_APPROVED,
                "drums": STATUS_NOT_STARTED,
                "bass": STATUS_NOT_STARTED,
                "melody": STATUS_NOT_STARTED,
                "quartet": STATUS_NOT_STARTED,
            },
            total_approved_bars=16,
            plan_present=True,
            lyrics_present=False,
        )

    def test_table_builds_without_error(self):
        statuses = [
            self._make_status("red"),
            self._make_status("blue", "blue__test_v1"),
        ]
        table = build_table(statuses)
        assert table is not None

    def test_table_has_correct_column_count(self):
        table = build_table([self._make_status()])
        assert (
            len(table.columns) == 13
        )  # Production Run + Color + Singer + Key + BPM + 5 phases + Bars + Plan + Lyr
