"""Tests for lyric_feedback_export.py"""

import json

import yaml


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

LYRICS_DRAFT = """\
[verse]
machines hum the first refrain
cycles old as iron rain

[chorus]
breathe breathe breathe again
"""

LYRICS_EDITED = """\
[verse]
machines learn to breathe alone
silicon dreams in monotone

[chorus]
breathe again breathe again
"""

LYRICS_REVIEW = {
    "pipeline": "lyric-generation",
    "bpm": 120,
    "time_sig": "4/4",
    "color": "Red",
    "singer": "Gabriel",
    "vocal_sections": [
        {
            "name": "verse",
            "bars": 4,
            "repeat": 1,
            "total_notes": 20,
            "contour": "stepwise",
        },
        {
            "name": "chorus",
            "bars": 2,
            "repeat": 2,
            "total_notes": 10,
            "contour": "arpeggiated",
        },
    ],
    "candidates": [
        {
            "id": "lyrics_02",
            "file": "candidates/lyrics_02.txt",
            "status": "approved",
            "chromatic": {
                "temporal": {"past": 0.7, "present": 0.2, "future": 0.1},
                "spatial": {"thing": 0.6, "place": 0.3, "person": 0.1},
                "ontological": {"imagined": 0.1, "forgotten": 0.1, "known": 0.8},
                "confidence": 0.04,
                "match": 0.65,
            },
            "fitting": {
                "verse": {
                    "syllables": 12,
                    "notes": 20,
                    "ratio": 0.6,
                    "verdict": "spacious",
                }
            },
        }
    ],
}


def _make_prod_dir(
    tmp_path,
    *,
    lyrics=LYRICS_EDITED,
    draft=LYRICS_DRAFT,
    review=None,
    eval_data=None,
    slug="test_song",
):
    """Build a minimal production directory."""
    prod_dir = tmp_path / "production" / slug
    melody_dir = prod_dir / "melody"
    (melody_dir / "candidates").mkdir(parents=True)
    (melody_dir / "approved").mkdir(parents=True)
    (prod_dir / "chords").mkdir(parents=True)

    # arrangement.txt: verse (4 bars = 8s) once, chorus (2 bars = 4s) twice — all track 4
    # At 120 BPM 4/4: secs_per_bar = 2.0
    arrangement = (
        "01:00:00:00.00\tverse\t4\t00:00:08:00.00\n"
        "01:00:08:00.00\tchorus\t4\t00:00:04:00.00\n"
        "01:00:12:00.00\tchorus\t4\t00:00:04:00.00\n"
    )
    (prod_dir / "arrangement.txt").write_text(arrangement)

    # chords/review.yml: minimal metadata (no thread → triggers fallback path)
    chord_review = {"bpm": 120, "time_sig": "4/4", "key": "C major", "color": "Red"}
    with open(prod_dir / "chords" / "review.yml", "w") as f:
        yaml.dump(chord_review, f)

    if lyrics is not None:
        (melody_dir / "lyrics.txt").write_text(lyrics)
    if draft is not None:
        (melody_dir / "lyrics_draft.txt").write_text(draft)

    if review is None:
        review = LYRICS_REVIEW
    with open(melody_dir / "lyrics_review.yml", "w") as f:
        yaml.dump(review, f, default_flow_style=False)

    if eval_data is not None:
        with open(prod_dir / "song_evaluation.yml", "w") as f:
            yaml.dump(eval_data, f, default_flow_style=False)

    return prod_dir


# ---------------------------------------------------------------------------
# collect_song_record
# ---------------------------------------------------------------------------


class TestCollectSongRecord:
    def test_happy_path(self, tmp_path):
        """Returns a full record with both texts and fitting metrics."""
        from app.generators.midi.lyric_feedback_export import collect_song_record

        prod_dir = _make_prod_dir(tmp_path)
        record = collect_song_record(prod_dir)

        assert record is not None
        assert record["song_slug"] == "test_song"
        assert record["color"] == "Red"
        assert record["singer"] == "Gabriel"
        assert record["draft_text"] == LYRICS_DRAFT
        assert record["edited_text"] == LYRICS_EDITED
        assert record["edited"] is True
        assert record["draft_chromatic_match"] == 0.65
        assert record["draft_fitting"] is not None
        assert record["edited_fitting"] is not None
        assert "overall" in record["draft_fitting"]
        assert "overall" in record["edited_fitting"]

    def test_edited_chromatic_match_from_eval(self, tmp_path):
        """lyrics_edited_chromatic_match pulled from song_evaluation.yml."""
        from app.generators.midi.lyric_feedback_export import collect_song_record

        prod_dir = _make_prod_dir(
            tmp_path,
            eval_data={"lyrics_edited_chromatic_match": 0.71, "composite_score": 0.65},
        )
        record = collect_song_record(prod_dir)
        assert record["edited_chromatic_match"] == 0.71

    def test_edited_chromatic_match_null_when_no_eval(self, tmp_path):
        """edited_chromatic_match is null when song_evaluation.yml absent."""
        from app.generators.midi.lyric_feedback_export import collect_song_record

        prod_dir = _make_prod_dir(tmp_path)
        record = collect_song_record(prod_dir)
        assert record["edited_chromatic_match"] is None

    def test_no_lyrics_returns_none(self, tmp_path):
        """Returns None when lyrics.txt doesn't exist."""
        from app.generators.midi.lyric_feedback_export import collect_song_record

        prod_dir = _make_prod_dir(tmp_path, lyrics=None, draft=None)
        (prod_dir / "melody" / "lyrics.txt").unlink(missing_ok=True)
        record = collect_song_record(prod_dir)
        assert record is None

    def test_no_draft_returns_partial_record(self, tmp_path):
        """Missing draft → record with draft_text=null and edited=null."""
        from app.generators.midi.lyric_feedback_export import collect_song_record

        prod_dir = _make_prod_dir(tmp_path, draft=None)
        record = collect_song_record(prod_dir)

        assert record is not None
        assert record["draft_text"] is None
        assert record["edited"] is None
        assert record["draft_fitting"] is None
        assert record["edited_text"] == LYRICS_EDITED

    def test_no_edits_detected(self, tmp_path):
        """Identical texts → edited=False."""
        from app.generators.midi.lyric_feedback_export import collect_song_record

        prod_dir = _make_prod_dir(tmp_path, lyrics=LYRICS_DRAFT, draft=LYRICS_DRAFT)
        record = collect_song_record(prod_dir)
        assert record["edited"] is False

    def test_vocal_sections_excludes_approved_label(self, tmp_path):
        """approved_label is stripped from the vocal_sections output."""
        from app.generators.midi.lyric_feedback_export import collect_song_record

        prod_dir = _make_prod_dir(tmp_path)
        record = collect_song_record(prod_dir)
        for section in record["vocal_sections"]:
            assert "approved_label" not in section


# ---------------------------------------------------------------------------
# export_feedback
# ---------------------------------------------------------------------------


class TestExportFeedback:
    def test_writes_jsonl(self, tmp_path):
        """Each song produces one JSON line."""
        from app.generators.midi.lyric_feedback_export import export_feedback

        dirs = [
            _make_prod_dir(tmp_path, slug="song_a"),
            _make_prod_dir(tmp_path, slug="song_b"),
        ]
        output = tmp_path / "out.jsonl"
        export_feedback(dirs, output)

        lines = output.read_text().strip().splitlines()
        assert len(lines) == 2
        for line in lines:
            obj = json.loads(line)
            assert "song_slug" in obj
            assert "draft_text" in obj
            assert "edited_text" in obj

    def test_skips_dirs_without_lyrics(self, tmp_path):
        """Directories with no lyrics.txt are silently skipped."""
        from app.generators.midi.lyric_feedback_export import export_feedback

        dirs = [
            _make_prod_dir(tmp_path, slug="with_lyrics"),
            _make_prod_dir(tmp_path, slug="no_lyrics", lyrics=None, draft=None),
        ]
        (tmp_path / "production" / "no_lyrics" / "melody" / "lyrics.txt").unlink(
            missing_ok=True
        )
        output = tmp_path / "out.jsonl"
        summary = export_feedback(dirs, output)
        assert summary["total"] == 1

    def test_summary_counts(self, tmp_path):
        """Summary counts confirmed_edits, no_edits, null_drafts correctly."""
        from app.generators.midi.lyric_feedback_export import export_feedback

        dirs = [
            _make_prod_dir(
                tmp_path, slug="edited", lyrics=LYRICS_EDITED, draft=LYRICS_DRAFT
            ),
            _make_prod_dir(
                tmp_path, slug="unchanged", lyrics=LYRICS_DRAFT, draft=LYRICS_DRAFT
            ),
            _make_prod_dir(tmp_path, slug="no_draft", draft=None),
        ]
        output = tmp_path / "out.jsonl"
        summary = export_feedback(dirs, output)

        assert summary["total"] == 3
        assert summary["confirmed_edits"] == 1
        assert summary["no_edits"] == 1
        assert summary["null_drafts"] == 1

    def test_size_advisory_printed(self, tmp_path, capsys):
        """Advisory message printed when confirmed edits < 20."""
        from app.generators.midi.lyric_feedback_export import export_feedback

        dirs = [_make_prod_dir(tmp_path, slug="only_one")]
        output = tmp_path / "out.jsonl"
        export_feedback(dirs, output)

        captured = capsys.readouterr()
        assert "suggest 20+" in captured.out

    def test_no_advisory_when_enough_data(self, tmp_path, capsys):
        """No advisory when >= 20 confirmed-edit songs."""
        from app.generators.midi.lyric_feedback_export import export_feedback

        dirs = [
            _make_prod_dir(
                tmp_path,
                slug=f"song_{i:02d}",
                lyrics=LYRICS_EDITED,
                draft=LYRICS_DRAFT,
            )
            for i in range(20)
        ]
        output = tmp_path / "out.jsonl"
        export_feedback(dirs, output)

        captured = capsys.readouterr()
        assert "suggest 20+" not in captured.out
