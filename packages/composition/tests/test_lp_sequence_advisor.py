"""Tests for lp_sequence_advisor."""

from pathlib import Path

import yaml

from white_composition.lp_sequence_advisor import (
    analyze_sides,
    format_report_text,
    write_report,
)
from white_composition.lp_sides import SidesDocument, assign_song, save_sides


def _make_song(
    album_dir: Path,
    thread_slug: str,
    production_slug: str,
    *,
    rainbow_color: str | None = None,
    bpm: int | None = None,
) -> str:
    prod_dir = album_dir / thread_slug / "production" / production_slug
    prod_dir.mkdir(parents=True, exist_ok=True)
    manifest = {"title": production_slug}
    if rainbow_color is not None:
        manifest["rainbow_color"] = rainbow_color
    if bpm is not None:
        manifest["bpm"] = bpm
    with open(prod_dir / "manifest_bootstrap.yml", "w") as f:
        yaml.dump(manifest, f)
    return f"{thread_slug}__{production_slug}"


class TestAnalyzeSides:
    def test_empty_album_no_songs_placed(self, tmp_path):
        report = analyze_sides(tmp_path)
        assert all(s["song_count"] == 0 for s in report["sides"].values())
        assert report["suggestions"] == []

    def test_single_song_no_suggestion(self, tmp_path):
        song_id = _make_song(tmp_path, "t1", "song_a", rainbow_color="Blue", bpm=90)
        doc = SidesDocument.empty()
        assign_song(doc, song_id, "A", 0, 200.0)
        save_sides(tmp_path, doc)

        report = analyze_sides(tmp_path)
        assert report["sides"]["A"]["song_count"] == 1
        assert report["sides"]["A"]["color_distribution"] == {"Blue": 1}
        assert report["sides"]["A"]["bpm_range"] == {"min": 90, "max": 90}
        assert report["suggestions"] == []

    def test_color_cluster_flagged(self, tmp_path):
        id1 = _make_song(tmp_path, "t1", "song_a", rainbow_color="Blue", bpm=90)
        id2 = _make_song(tmp_path, "t1", "song_b", rainbow_color="Blue", bpm=95)
        id3 = _make_song(tmp_path, "t1", "song_c", rainbow_color="Orange", bpm=140)
        doc = SidesDocument.empty()
        assign_song(doc, id1, "A", 0, 200.0)
        assign_song(doc, id2, "A", 1, 210.0)
        assign_song(doc, id3, "B", 0, 220.0)
        save_sides(tmp_path, doc)

        report = analyze_sides(tmp_path)
        assert len(report["suggestions"]) == 1
        assert "Side A" in report["suggestions"][0]
        assert "Blue" in report["suggestions"][0]
        assert (
            "song_c" in report["suggestions"][0] or "Orange" in report["suggestions"][0]
        )

    def test_over_limit_reflected(self, tmp_path):
        id1 = _make_song(tmp_path, "t1", "song_a", rainbow_color="Red")
        doc = SidesDocument.empty()
        assign_song(doc, id1, "A", 0, 1300.0)
        save_sides(tmp_path, doc)

        report = analyze_sides(tmp_path)
        assert report["sides"]["A"]["over_limit"] is True


class TestFormatReportText:
    def test_no_songs_placed_note(self):
        report = {"sides": {n: {"song_count": 0} for n in "ABCD"}, "suggestions": []}
        text = format_report_text(report)
        assert "nothing to analyze" in text.lower()

    def test_includes_suggestions(self):
        report = {
            "sides": {
                "A": {
                    "song_count": 2,
                    "total_seconds": 400.0,
                    "over_limit": False,
                    "color_distribution": {"Blue": 2},
                    "bpm_range": {"min": 90, "max": 95},
                },
                "B": {"song_count": 0},
                "C": {"song_count": 0},
                "D": {"song_count": 0},
            },
            "suggestions": ["Side A is 100% Blue (2/2 songs) — consider more variety."],
        }
        text = format_report_text(report)
        assert "Suggestions:" in text
        assert "100% Blue" in text


class TestWriteReport:
    def test_writes_yaml(self, tmp_path):
        report = {"sides": {}, "suggestions": []}
        out_path = tmp_path / "report.yml"
        write_report(out_path, report)
        with open(out_path) as f:
            data = yaml.safe_load(f)
        assert data["suggestions"] == []
