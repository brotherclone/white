"""Tests for the lyric generation pipeline."""

import io
from unittest.mock import MagicMock, patch

import mido
import pytest
import yaml


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_midi_bytes(note_count: int = 4, ticks_per_beat: int = 480) -> bytes:
    """Create a minimal MIDI file with the given number of note_on events."""
    mid = mido.MidiFile(ticks_per_beat=ticks_per_beat)
    track = mido.MidiTrack()
    mid.tracks.append(track)
    track.append(mido.MetaMessage("set_tempo", tempo=mido.bpm2tempo(120), time=0))
    for i in range(note_count):
        track.append(mido.Message("note_on", note=60 + i, velocity=80, time=0))
        track.append(
            mido.Message("note_off", note=60 + i, velocity=0, time=ticks_per_beat)
        )
    track.append(mido.MetaMessage("end_of_track", time=0))
    buf = io.BytesIO()
    mid.save(file=buf)
    return buf.getvalue()


def make_meta(
    color="Red",
    bpm=120,
    time_sig="4/4",
    key="C major",
    concept="a test concept",
    title="Test Song",
    sounds_like=None,
) -> dict:
    """Create a minimal song metadata dict (replaces ProductionPlan in tests)."""
    return {
        "color": color,
        "bpm": bpm,
        "time_sig": time_sig,
        "key": key,
        "concept": concept,
        "title": title,
        "sounds_like": sounds_like or [],
        "genres": [],
        "mood": [],
        "singer": "",
    }


# ---------------------------------------------------------------------------
# 1. _count_notes
# ---------------------------------------------------------------------------


class TestCountNotes:
    def test_count_notes(self, tmp_path):
        from app.generators.midi.lyric_pipeline import _count_notes

        midi_path = tmp_path / "test.mid"
        midi_path.write_bytes(make_midi_bytes(note_count=8))
        assert _count_notes(midi_path) == 8

    def test_count_notes_zero_velocity_excluded(self, tmp_path):
        """note_off encoded as note_on velocity=0 should not be counted."""
        from app.generators.midi.lyric_pipeline import _count_notes

        mid = mido.MidiFile(ticks_per_beat=480)
        track = mido.MidiTrack()
        mid.tracks.append(track)
        track.append(mido.Message("note_on", note=60, velocity=80, time=0))
        track.append(mido.Message("note_on", note=60, velocity=0, time=480))
        track.append(mido.MetaMessage("end_of_track", time=0))
        path = tmp_path / "t.mid"
        buf = io.BytesIO()
        mid.save(file=buf)
        path.write_bytes(buf.getvalue())
        assert _count_notes(path) == 1

    def test_count_notes_missing_file(self, tmp_path):
        from app.generators.midi.lyric_pipeline import _count_notes

        assert _count_notes(tmp_path / "nonexistent.mid") == 0


# ---------------------------------------------------------------------------
# 2. _fitting_verdict
# ---------------------------------------------------------------------------


class TestFittingVerdict:
    def test_spacious(self):
        from app.generators.midi.lyric_pipeline import _fitting_verdict

        assert _fitting_verdict(0.50) == "spacious"
        assert _fitting_verdict(0.74) == "spacious"

    def test_paste_ready(self):
        from app.generators.midi.lyric_pipeline import _fitting_verdict

        assert _fitting_verdict(0.75) == "paste-ready"
        assert _fitting_verdict(1.00) == "paste-ready"
        assert _fitting_verdict(1.10) == "paste-ready"

    def test_tight_but_workable(self):
        from app.generators.midi.lyric_pipeline import _fitting_verdict

        assert _fitting_verdict(1.11) == "tight but workable"
        assert _fitting_verdict(1.30) == "tight but workable"

    def test_splits_needed(self):
        from app.generators.midi.lyric_pipeline import _fitting_verdict

        assert _fitting_verdict(1.31) == "splits needed"
        assert _fitting_verdict(2.00) == "splits needed"


# ---------------------------------------------------------------------------
# 3. _compute_fitting
# ---------------------------------------------------------------------------


class TestComputeFitting:
    def test_ratio_calculation(self):
        """Given known syllables/notes, verify ratio and verdict."""
        from app.generators.midi.lyric_pipeline import _compute_fitting

        # "Hello world" = 3 syllables, "beautiful morning" = 5 syllables → 8 total
        text = "[verse]\nHello world\nbeautiful morning\n"
        vocal_sections = [{"name": "verse", "total_notes": 10}]
        result = _compute_fitting(text, vocal_sections)
        assert result["verse"]["syllables"] == 8
        assert result["verse"]["notes"] == 10
        assert result["verse"]["ratio"] == pytest.approx(0.8, abs=0.001)
        assert result["verse"]["verdict"] == "paste-ready"

    def test_overall_verdict_worst_wins(self):
        """If one section is 'tight but workable', overall should reflect that."""
        from app.generators.midi.lyric_pipeline import _compute_fitting

        # verse: 8 syllables / 10 notes = 0.8 → paste-ready
        # chorus: lots of syllables for few notes → splits needed
        text = "[verse]\nHello world\nbeautiful morning\n[chorus]\nA B C D E F G H I J K L M N O P\n"
        vocal_sections = [
            {"name": "verse", "total_notes": 10},
            {"name": "chorus", "total_notes": 5},
        ]
        result = _compute_fitting(text, vocal_sections)
        # chorus syllables >> 5 notes → splits needed
        assert result["chorus"]["verdict"] == "splits needed"
        assert result["overall"] == "splits needed"

    def test_overall_spacious_treated_as_paste_ready(self):
        """A spacious verdict alone should yield 'paste-ready' overall (not 'spacious')."""
        from app.generators.midi.lyric_pipeline import _compute_fitting

        text = "[verse]\nHi\n"  # 1 syllable
        vocal_sections = [{"name": "verse", "total_notes": 10}]  # ratio = 0.1
        result = _compute_fitting(text, vocal_sections)
        assert result["verse"]["verdict"] == "spacious"
        assert result["overall"] == "paste-ready"

    def test_strips_comment_lines(self):
        """Lines starting with # should not contribute syllables."""
        from app.generators.midi.lyric_pipeline import _compute_fitting

        text = "[verse]\n# stage direction\nHello world\n"
        vocal_sections = [{"name": "verse", "total_notes": 3}]
        result = _compute_fitting(text, vocal_sections)
        # Only "Hello world" = 3 syllables
        assert result["verse"]["syllables"] == 3

    def test_zero_notes_ratio_defaults_to_one(self):
        """Section with 0 notes should default to ratio=1.0."""
        from app.generators.midi.lyric_pipeline import _compute_fitting

        text = "[verse]\nHello\n"
        vocal_sections = [{"name": "verse", "total_notes": 0}]
        result = _compute_fitting(text, vocal_sections)
        assert result["verse"]["ratio"] == 1.0


# ---------------------------------------------------------------------------
# 4. _parse_sections
# ---------------------------------------------------------------------------


class TestParseSections:
    def test_basic_parsing(self):
        from app.generators.midi.lyric_pipeline import _parse_sections

        text = "[verse]\nline one\nline two\n[chorus]\nchorus line\n"
        result = _parse_sections(text)
        assert "verse" in result
        assert "line one" in result["verse"]
        assert "chorus" in result

    def test_strips_comment_lines(self):
        from app.generators.midi.lyric_pipeline import _parse_sections

        text = "[verse]\n# a comment\nreal line\n"
        result = _parse_sections(text)
        assert "real line" in result["verse"]
        assert "# a comment" not in result["verse"]

    def test_empty_if_no_headers(self):
        from app.generators.midi.lyric_pipeline import _parse_sections

        result = _parse_sections("just some text without headers")
        assert result == {}

    def test_section_name_lowercased(self):
        from app.generators.midi.lyric_pipeline import _parse_sections

        text = "[Verse]\nsome line\n"
        result = _parse_sections(text)
        assert "verse" in result


# ---------------------------------------------------------------------------
# 5. Syllable targets
# ---------------------------------------------------------------------------


class TestSyllableTargets:
    def test_targets_from_note_count(self):
        """notes=52 → floor(52*0.75)=39, floor(52*1.05)=54"""
        import math

        notes = 52
        lo = math.floor(notes * 0.75)
        hi = math.floor(notes * 1.05)
        assert lo == 39
        assert hi == 54


# ---------------------------------------------------------------------------
# 6. _next_candidate_id
# ---------------------------------------------------------------------------


class TestNextCandidateId:
    def test_empty_review(self):
        from app.generators.midi.lyric_pipeline import _next_candidate_id

        review = {"candidates": []}
        assert _next_candidate_id(review) == "lyrics_01"

    def test_with_existing_candidates(self):
        from app.generators.midi.lyric_pipeline import _next_candidate_id

        review = {
            "candidates": [
                {"id": "lyrics_01"},
                {"id": "lyrics_02"},
                {"id": "lyrics_03"},
            ]
        }
        assert _next_candidate_id(review) == "lyrics_04"

    def test_gaps_still_increment_from_max(self):
        """Gaps in numbering should not affect the next ID (uses max)."""
        from app.generators.midi.lyric_pipeline import _next_candidate_id

        review = {"candidates": [{"id": "lyrics_01"}, {"id": "lyrics_05"}]}
        assert _next_candidate_id(review) == "lyrics_06"


# ---------------------------------------------------------------------------
# 7. _load_or_init_review
# ---------------------------------------------------------------------------


class TestLoadOrInitReview:
    def test_creates_fresh_when_missing(self, tmp_path):
        from app.generators.midi.lyric_pipeline import _load_or_init_review

        meta = make_meta()
        review = _load_or_init_review(tmp_path, meta, model="claude-sonnet-4-6", seed=7)
        assert review["pipeline"] == "lyric-generation"
        assert review["color"] == "Red"
        assert review["seed"] == 7
        assert review["candidates"] == []

    def test_loads_existing_file(self, tmp_path):
        from app.generators.midi.lyric_pipeline import (
            LYRICS_REVIEW_FILENAME,
            _load_or_init_review,
        )

        existing = {
            "pipeline": "lyric-generation",
            "candidates": [{"id": "lyrics_01", "status": "approved"}],
        }
        review_path = tmp_path / LYRICS_REVIEW_FILENAME
        with open(review_path, "w") as f:
            yaml.dump(existing, f)

        meta = make_meta()
        result = _load_or_init_review(tmp_path, meta, model="m", seed=1)
        # Existing entry must be preserved unchanged
        assert result["candidates"][0]["id"] == "lyrics_01"
        assert result["candidates"][0]["status"] == "approved"

    def test_review_append_preserves_existing(self, tmp_path):
        """Existing approved entry must not be touched after a second run."""
        from app.generators.midi.lyric_pipeline import (
            LYRICS_REVIEW_FILENAME,
            _load_or_init_review,
            _next_candidate_id,
        )

        review_path = tmp_path / LYRICS_REVIEW_FILENAME
        initial = {
            "pipeline": "lyric-generation",
            "candidates": [{"id": "lyrics_01", "status": "approved", "notes": "keep"}],
        }
        with open(review_path, "w") as f:
            yaml.dump(initial, f)

        meta = make_meta()
        review = _load_or_init_review(tmp_path, meta, model="m", seed=1)
        new_id = _next_candidate_id(review)
        assert new_id == "lyrics_02"
        # Original entry untouched
        assert review["candidates"][0]["status"] == "approved"
        assert review["candidates"][0]["notes"] == "keep"


# ---------------------------------------------------------------------------
# 8. _build_prompt
# ---------------------------------------------------------------------------


class TestBuildPrompt:
    def test_has_concept(self):
        from app.generators.midi.lyric_pipeline import _build_prompt

        meta = make_meta(concept="machines learning to breathe")
        vocal_sections = [
            {
                "name": "verse",
                "bars": 4,
                "repeat": 2,
                "total_notes": 20,
                "contour": "stepwise",
            }
        ]
        syllable_targets = {"verse": (15, 21)}
        prompt = _build_prompt(meta, vocal_sections, syllable_targets)
        assert "machines learning to breathe" in prompt

    def test_has_syllable_targets(self):
        from app.generators.midi.lyric_pipeline import _build_prompt

        meta = make_meta()
        vocal_sections = [
            {
                "name": "verse",
                "bars": 4,
                "repeat": 1,
                "total_notes": 52,
                "contour": "arpeggiated",
            }
        ]
        syllable_targets = {"verse": (39, 54)}
        prompt = _build_prompt(meta, vocal_sections, syllable_targets)
        assert "39" in prompt
        assert "54" in prompt

    def test_has_section_headers(self):
        from app.generators.midi.lyric_pipeline import _build_prompt

        meta = make_meta()
        vocal_sections = [
            {
                "name": "verse",
                "bars": 4,
                "repeat": 2,
                "total_notes": 20,
                "contour": "stepwise",
            },
            {
                "name": "chorus",
                "bars": 2,
                "repeat": 4,
                "total_notes": 10,
                "contour": "arpeggiated",
            },
        ]
        syllable_targets = {"verse": (15, 21), "chorus": (7, 10)}
        prompt = _build_prompt(meta, vocal_sections, syllable_targets)
        assert "[verse]" in prompt
        assert "[chorus]" in prompt


# ---------------------------------------------------------------------------
# 9. sync_lyric_candidates
# ---------------------------------------------------------------------------


class TestSyncLyricCandidates:
    def test_adds_untracked_stub(self, tmp_path):
        from app.generators.midi.lyric_pipeline import (
            LYRICS_REVIEW_FILENAME,
            sync_lyric_candidates,
        )

        melody_dir = tmp_path / "melody"
        candidates_dir = melody_dir / "candidates"
        candidates_dir.mkdir(parents=True)
        (candidates_dir / "lyrics_04.txt").write_text("some lyrics")

        review_path = melody_dir / LYRICS_REVIEW_FILENAME
        with open(review_path, "w") as f:
            yaml.dump({"pipeline": "lyric-generation", "candidates": []}, f)

        added = sync_lyric_candidates(melody_dir)
        assert added == 1

        with open(review_path) as f:
            updated = yaml.safe_load(f)
        ids = [c["id"] for c in updated["candidates"]]
        assert "lyrics_04" in ids

    def test_skips_already_tracked(self, tmp_path):
        from app.generators.midi.lyric_pipeline import (
            LYRICS_REVIEW_FILENAME,
            sync_lyric_candidates,
        )

        melody_dir = tmp_path / "melody"
        candidates_dir = melody_dir / "candidates"
        candidates_dir.mkdir(parents=True)
        (candidates_dir / "lyrics_01.txt").write_text("lyrics")

        review_path = melody_dir / LYRICS_REVIEW_FILENAME
        existing = {
            "pipeline": "lyric-generation",
            "candidates": [
                {
                    "id": "lyrics_01",
                    "file": "candidates/lyrics_01.txt",
                    "status": "pending",
                }
            ],
        }
        with open(review_path, "w") as f:
            yaml.dump(existing, f)

        added = sync_lyric_candidates(melody_dir)
        assert added == 0

    def test_no_review_returns_zero(self, tmp_path):
        from app.generators.midi.lyric_pipeline import sync_lyric_candidates

        melody_dir = tmp_path / "melody"
        melody_dir.mkdir()
        result = sync_lyric_candidates(melody_dir)
        assert result == 0


# ---------------------------------------------------------------------------
# 10. promote_part — .txt support
# ---------------------------------------------------------------------------


class TestPromotePartTxt:
    def _make_review(self, tmp_path, candidates):
        part_dir = tmp_path / "melody"
        cand_dir = part_dir / "candidates"
        cand_dir.mkdir(parents=True)
        approved_dir = part_dir / "approved"
        approved_dir.mkdir(parents=True)

        review = {"candidates": candidates}
        review_path = part_dir / "lyrics_review.yml"
        with open(review_path, "w") as f:
            yaml.dump(review, f)
        return part_dir, review_path

    def test_promote_txt_happy_path(self, tmp_path):
        """Approved .txt candidate gets copied to melody/lyrics.txt."""
        from app.generators.midi.promote_part import promote_part

        part_dir, review_path = self._make_review(
            tmp_path,
            [
                {
                    "id": "lyrics_01",
                    "file": "candidates/lyrics_01.txt",
                    "status": "approved",
                }
            ],
        )
        (part_dir / "candidates" / "lyrics_01.txt").write_text("hello world lyrics")

        promote_part(str(review_path))
        assert (part_dir / "lyrics.txt").exists()
        assert (part_dir / "lyrics.txt").read_text() == "hello world lyrics"

    def test_promote_txt_conflict_fails(self, tmp_path):
        """Two approved .txt candidates → error, no copy performed."""
        from app.generators.midi.promote_part import promote_part

        part_dir, review_path = self._make_review(
            tmp_path,
            [
                {
                    "id": "lyrics_01",
                    "file": "candidates/lyrics_01.txt",
                    "status": "approved",
                },
                {
                    "id": "lyrics_02",
                    "file": "candidates/lyrics_02.txt",
                    "status": "approved",
                },
            ],
        )
        (part_dir / "candidates" / "lyrics_01.txt").write_text("first")
        (part_dir / "candidates" / "lyrics_02.txt").write_text("second")

        promote_part(str(review_path))
        # Neither should have been copied
        assert not (part_dir / "lyrics.txt").exists()

    def test_promote_midi_unchanged(self, tmp_path):
        """Existing MIDI promotion logic still works alongside .txt support."""
        from app.generators.midi.promote_part import promote_part

        part_dir = tmp_path / "melody"
        cand_dir = part_dir / "candidates"
        cand_dir.mkdir(parents=True)
        (part_dir / "approved").mkdir(parents=True)

        midi_bytes = make_midi_bytes(4)
        (cand_dir / "melody_verse_01.mid").write_bytes(midi_bytes)

        review = {
            "candidates": [
                {
                    "id": "melody_verse_01",
                    "midi_file": "candidates/melody_verse_01.mid",
                    "label": "verse",
                    "rank": 1,
                    "status": "approved",
                }
            ]
        }
        review_path = part_dir / "review.yml"
        with open(review_path, "w") as f:
            yaml.dump(review, f)

        promote_part(str(review_path))
        assert (part_dir / "approved" / "verse.mid").exists()

    def test_promote_clean_removes_lyrics_txt(self, tmp_path):
        """--clean should unlink lyrics.txt if it exists."""
        from app.generators.midi.promote_part import promote_part

        part_dir = tmp_path / "melody"
        (part_dir / "approved").mkdir(parents=True)
        (part_dir / "candidates").mkdir(parents=True)
        lyrics_txt = part_dir / "lyrics.txt"
        lyrics_txt.write_text("old lyrics")

        review = {"candidates": []}
        review_path = part_dir / "review.yml"
        with open(review_path, "w") as f:
            yaml.dump(review, f)

        promote_part(str(review_path), clean=True)
        assert not lyrics_txt.exists()

    def test_promote_txt_writes_draft(self, tmp_path):
        """Approved .txt promotion also writes lyrics_draft.txt alongside lyrics.txt."""
        from app.generators.midi.promote_part import promote_part

        part_dir, review_path = self._make_review(
            tmp_path,
            [
                {
                    "id": "lyrics_01",
                    "file": "candidates/lyrics_01.txt",
                    "status": "approved",
                }
            ],
        )
        (part_dir / "candidates" / "lyrics_01.txt").write_text("draft content")

        promote_part(str(review_path))
        assert (part_dir / "lyrics.txt").read_text() == "draft content"
        assert (part_dir / "lyrics_draft.txt").read_text() == "draft content"

    def test_promote_clean_removes_draft(self, tmp_path):
        """--clean unlinks lyrics_draft.txt if it exists."""
        from app.generators.midi.promote_part import promote_part

        part_dir = tmp_path / "melody"
        (part_dir / "approved").mkdir(parents=True)
        (part_dir / "candidates").mkdir(parents=True)
        (part_dir / "lyrics_draft.txt").write_text("old draft")

        review = {"candidates": []}
        review_path = part_dir / "review.yml"
        with open(review_path, "w") as f:
            yaml.dump(review, f)

        promote_part(str(review_path), clean=True)
        assert not (part_dir / "lyrics_draft.txt").exists()


# ---------------------------------------------------------------------------
# 11. Integration tests (mock API + scorer)
# ---------------------------------------------------------------------------


def _make_mock_scorer(match_score=0.7):
    """Build a mock ChromaticScorer that returns deterministic results."""
    mock = MagicMock()
    import numpy as np

    mock.prepare_concept.return_value = np.zeros(768, dtype=np.float32)
    mock.score_batch.return_value = [
        {
            "temporal": {"past": 0.6, "present": 0.3, "future": 0.1},
            "spatial": {"thing": 0.7, "place": 0.2, "person": 0.1},
            "ontological": {"imagined": 0.1, "forgotten": 0.1, "known": 0.8},
            "confidence": 0.042,
            "rank": 0,
            "candidate": {"lyric_text": "PLACEHOLDER"},
        }
    ]
    return mock


def _make_production_dir(tmp_path):
    """Set up a minimal production directory with arrangement.txt + melody approved MIDI.

    arrangement.txt has one 'verse' clip on track 4 (8 s = 4 bars @ 120 BPM 4/4).
    chords/review.yml supplies minimal metadata used as fallback when no proposal exists.
    """
    prod_dir = tmp_path / "production" / "test_song"
    melody_dir = prod_dir / "melody"
    (melody_dir / "approved").mkdir(parents=True)
    (melody_dir / "candidates").mkdir(parents=True)
    (prod_dir / "chords").mkdir(parents=True)

    # Write a melody MIDI (10 notes)
    midi_bytes = make_midi_bytes(note_count=10)
    (melody_dir / "approved" / "verse.mid").write_bytes(midi_bytes)

    # arrangement.txt: one verse clip on track 4, 8 seconds (4 bars @ 120 BPM 4/4)
    arrangement = "01:00:00:00.00\tverse\t4\t00:00:08:00.00\n"
    (prod_dir / "arrangement.txt").write_text(arrangement)

    # chords/review.yml: minimal metadata (no thread/song_proposal → triggers fallback)
    chord_review = {
        "bpm": 120,
        "time_sig": "4/4",
        "key": "C major",
        "color": "Red",
        "song_slug": "test_song",
    }
    with open(prod_dir / "chords" / "review.yml", "w") as f:
        yaml.dump(chord_review, f)

    return prod_dir


class TestRunPipelineIntegration:
    def test_run_pipeline_writes_candidate_files(self, tmp_path):
        """Pipeline should write N .txt files in melody/candidates/."""
        import numpy as np
        from app.generators.midi.lyric_pipeline import run_lyric_pipeline

        prod_dir = _make_production_dir(tmp_path)

        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.content = [MagicMock(text="[verse]\nHello world\n")]
        mock_client.messages.create.return_value = mock_response

        mock_scorer = MagicMock()
        mock_scorer.prepare_concept.return_value = np.zeros(768, dtype=np.float32)
        mock_scorer.score_batch.return_value = [
            {
                "temporal": {"past": 0.6, "present": 0.3, "future": 0.1},
                "spatial": {"thing": 0.7, "place": 0.2, "person": 0.1},
                "ontological": {"imagined": 0.1, "forgotten": 0.1, "known": 0.8},
                "confidence": 0.042,
                "rank": 0,
                "candidate": {"lyric_text": "[verse]\nHello world\n"},
            }
        ]

        with (
            patch("anthropic.Anthropic", return_value=mock_client),
            patch(
                "training.chromatic_scorer.ChromaticScorer", return_value=mock_scorer
            ),
        ):
            run_lyric_pipeline(
                production_dir=str(prod_dir),
                num_candidates=1,
                model="claude-sonnet-4-6",
            )

        candidates_dir = prod_dir / "melody" / "candidates"
        txt_files = list(candidates_dir.glob("*.txt"))
        assert len(txt_files) == 1
        assert txt_files[0].name == "lyrics_01.txt"

    def test_run_pipeline_writes_review_yml(self, tmp_path):
        """Pipeline should write lyrics_review.yml with one candidate entry."""
        import numpy as np
        from app.generators.midi.lyric_pipeline import (
            LYRICS_REVIEW_FILENAME,
            run_lyric_pipeline,
        )

        prod_dir = _make_production_dir(tmp_path)

        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.content = [MagicMock(text="[verse]\nTest lyrics\n")]
        mock_client.messages.create.return_value = mock_response

        mock_scorer = MagicMock()
        mock_scorer.prepare_concept.return_value = np.zeros(768, dtype=np.float32)
        mock_scorer.score_batch.return_value = [
            {
                "temporal": {"past": 0.6, "present": 0.3, "future": 0.1},
                "spatial": {"thing": 0.7, "place": 0.2, "person": 0.1},
                "ontological": {"imagined": 0.1, "forgotten": 0.1, "known": 0.8},
                "confidence": 0.042,
                "rank": 0,
                "candidate": {"lyric_text": "[verse]\nTest lyrics\n"},
            }
        ]

        with (
            patch("anthropic.Anthropic", return_value=mock_client),
            patch(
                "training.chromatic_scorer.ChromaticScorer", return_value=mock_scorer
            ),
        ):
            run_lyric_pipeline(
                production_dir=str(prod_dir),
                num_candidates=1,
            )

        review_path = prod_dir / "melody" / LYRICS_REVIEW_FILENAME
        assert review_path.exists()
        with open(review_path) as f:
            review = yaml.safe_load(f)
        assert len(review["candidates"]) == 1
        cand = review["candidates"][0]
        assert cand["id"] == "lyrics_01"
        assert cand["status"] == "pending"
        assert "chromatic" in cand
        assert "fitting" in cand

    def test_run_pipeline_appends_not_overwrites(self, tmp_path):
        """Second run should append new entries; first entry must be preserved."""
        from app.generators.midi.lyric_pipeline import (
            LYRICS_REVIEW_FILENAME,
            run_lyric_pipeline,
        )

        prod_dir = _make_production_dir(tmp_path)

        def run_once():
            import numpy as np

            mock_client = MagicMock()
            mock_response = MagicMock()
            mock_response.content = [MagicMock(text="[verse]\nSome lyrics\n")]
            mock_client.messages.create.return_value = mock_response

            mock_scorer = MagicMock()
            mock_scorer.prepare_concept.return_value = np.zeros(768, dtype=np.float32)
            mock_scorer.score_batch.return_value = [
                {
                    "temporal": {"past": 0.6, "present": 0.3, "future": 0.1},
                    "spatial": {"thing": 0.7, "place": 0.2, "person": 0.1},
                    "ontological": {"imagined": 0.1, "forgotten": 0.1, "known": 0.8},
                    "confidence": 0.042,
                    "rank": 0,
                    "candidate": {"lyric_text": "[verse]\nSome lyrics\n"},
                }
            ]

            with (
                patch("anthropic.Anthropic", return_value=mock_client),
                patch(
                    "training.chromatic_scorer.ChromaticScorer",
                    return_value=mock_scorer,
                ),
            ):
                run_lyric_pipeline(
                    production_dir=str(prod_dir),
                    num_candidates=1,
                )

        run_once()

        # Mark first candidate as approved
        review_path = prod_dir / "melody" / LYRICS_REVIEW_FILENAME
        with open(review_path) as f:
            review = yaml.safe_load(f)
        review["candidates"][0]["status"] = "approved"
        with open(review_path, "w") as f:
            yaml.dump(review, f, default_flow_style=False, sort_keys=False)

        run_once()

        with open(review_path) as f:
            final = yaml.safe_load(f)

        assert len(final["candidates"]) == 2
        # First entry still approved
        assert final["candidates"][0]["status"] == "approved"
        # Second entry has incremented ID
        assert final["candidates"][1]["id"] == "lyrics_02"
