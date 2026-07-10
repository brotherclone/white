"""Tests for the lyric generation pipeline."""

import io
from pathlib import Path
from unittest.mock import MagicMock, patch

import mido
import pytest
import yaml

_NO_MIDI_DIR = Path("/nonexistent")


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
        from white_generation.pipelines.lyric_pipeline import _count_notes

        midi_path = tmp_path / "test.mid"
        midi_path.write_bytes(make_midi_bytes(note_count=8))
        assert _count_notes(midi_path) == 8

    def test_count_notes_zero_velocity_excluded(self, tmp_path):
        """note_off encoded as note_on velocity=0 should not be counted."""
        from white_generation.pipelines.lyric_pipeline import _count_notes

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
        from white_generation.pipelines.lyric_pipeline import _count_notes

        assert _count_notes(tmp_path / "nonexistent.mid") == 0


# ---------------------------------------------------------------------------
# 2. _fitting_verdict
# ---------------------------------------------------------------------------


class TestFittingVerdict:
    def test_spacious(self):
        from white_generation.pipelines.lyric_pipeline import _fitting_verdict

        assert _fitting_verdict(0.50) == "spacious"
        assert _fitting_verdict(0.74) == "spacious"

    def test_paste_ready(self):
        from white_generation.pipelines.lyric_pipeline import _fitting_verdict

        assert _fitting_verdict(0.75) == "paste-ready"
        assert _fitting_verdict(1.00) == "paste-ready"
        assert _fitting_verdict(1.10) == "paste-ready"

    def test_tight_but_workable(self):
        from white_generation.pipelines.lyric_pipeline import _fitting_verdict

        assert _fitting_verdict(1.11) == "tight but workable"
        assert _fitting_verdict(1.30) == "tight but workable"

    def test_splits_needed(self):
        from white_generation.pipelines.lyric_pipeline import _fitting_verdict

        assert _fitting_verdict(1.31) == "splits needed"
        assert _fitting_verdict(2.00) == "splits needed"


# ---------------------------------------------------------------------------
# 2b. _phrase_syllable_range
# ---------------------------------------------------------------------------


class TestPhraseSyllableRange:
    def test_short_phrase_widened_beyond_strict_multiplier(self):
        from white_generation.pipelines.lyric_pipeline import _phrase_syllable_range

        # Strict 0.8x-1.15x multiplier would give (1, 3) for a 2-note phrase.
        lo, hi = _phrase_syllable_range(2)
        assert (lo, hi) != (1, 3)
        assert lo <= 1
        assert hi >= 4

    def test_one_note_phrase_allows_more_than_two_syllables(self):
        from white_generation.pipelines.lyric_pipeline import _phrase_syllable_range

        lo, hi = _phrase_syllable_range(1)
        assert lo == 0
        assert hi >= 3

    def test_short_phrase_lower_bound_never_negative(self):
        from white_generation.pipelines.lyric_pipeline import _phrase_syllable_range

        lo, _hi = _phrase_syllable_range(1)
        assert lo >= 0

    def test_longer_phrase_unaffected(self):
        import math

        from white_generation.pipelines.lyric_pipeline import _phrase_syllable_range

        # Phrases above SHORT_PHRASE_NOTE_THRESHOLD keep the original multiplier.
        for notes in (4, 8, 16):
            lo, hi = _phrase_syllable_range(notes)
            assert lo == math.floor(notes * 0.8)
            assert hi == math.ceil(notes * 1.15)


# ---------------------------------------------------------------------------
# 3. _compute_fitting
# ---------------------------------------------------------------------------


class TestComputeFitting:
    def test_ratio_calculation(self):
        """Given known syllables/notes, verify ratio and verdict."""
        from white_generation.pipelines.lyric_pipeline import _compute_fitting

        # "Hello world" = 3 syllables, "beautiful morning" = 5 syllables → 8 total
        text = "[verse]\nHello world\nbeautiful morning\n"
        vocal_sections = [{"name": "verse", "total_notes": 10}]
        result = _compute_fitting(text, vocal_sections, _NO_MIDI_DIR)
        assert result["verse"]["syllables"] == 8
        assert result["verse"]["notes"] == 10
        assert result["verse"]["ratio"] == pytest.approx(0.8, abs=0.001)
        assert result["verse"]["verdict"] == "paste-ready"

    def test_overall_verdict_worst_wins(self):
        """If one section is 'tight but workable', overall should reflect that."""
        from white_generation.pipelines.lyric_pipeline import _compute_fitting

        # verse: 8 syllables / 10 notes = 0.8 → paste-ready
        # chorus: lots of syllables for few notes → splits needed
        text = "[verse]\nHello world\nbeautiful morning\n[chorus]\nA B C D E F G H I J K L M N O P\n"
        vocal_sections = [
            {"name": "verse", "total_notes": 10},
            {"name": "chorus", "total_notes": 5},
        ]
        result = _compute_fitting(text, vocal_sections, _NO_MIDI_DIR)
        # chorus syllables >> 5 notes → splits needed
        assert result["chorus"]["verdict"] == "splits needed"
        assert result["overall"] == "splits needed"

    def test_overall_spacious_treated_as_paste_ready(self):
        """A spacious verdict alone should yield 'paste-ready' overall (not 'spacious')."""
        from white_generation.pipelines.lyric_pipeline import _compute_fitting

        text = "[verse]\nHi\n"  # 1 syllable
        vocal_sections = [{"name": "verse", "total_notes": 10}]  # ratio = 0.1
        result = _compute_fitting(text, vocal_sections, _NO_MIDI_DIR)
        assert result["verse"]["verdict"] == "spacious"
        assert result["overall"] == "paste-ready"

    def test_strips_comment_lines(self):
        """Lines starting with # should not contribute syllables."""
        from white_generation.pipelines.lyric_pipeline import _compute_fitting

        text = "[verse]\n# stage direction\nHello world\n"
        vocal_sections = [{"name": "verse", "total_notes": 3}]
        result = _compute_fitting(text, vocal_sections, _NO_MIDI_DIR)
        # Only "Hello world" = 3 syllables
        assert result["verse"]["syllables"] == 3

    def test_zero_notes_ratio_defaults_to_one(self):
        """Section with 0 notes should default to ratio=1.0."""
        from white_generation.pipelines.lyric_pipeline import _compute_fitting

        text = "[verse]\nHello\n"
        vocal_sections = [{"name": "verse", "total_notes": 0}]
        result = _compute_fitting(text, vocal_sections, _NO_MIDI_DIR)
        assert result["verse"]["ratio"] == 1.0


# ---------------------------------------------------------------------------
# 4. _parse_sections
# ---------------------------------------------------------------------------


class TestParseSections:
    def test_basic_parsing(self):
        from white_generation.pipelines.lyric_pipeline import _parse_sections

        text = "[verse]\nline one\nline two\n[chorus]\nchorus line\n"
        result = _parse_sections(text)
        assert "verse" in result
        assert "line one" in result["verse"]
        assert "chorus" in result

    def test_strips_comment_lines(self):
        from white_generation.pipelines.lyric_pipeline import _parse_sections

        text = "[verse]\n# a comment\nreal line\n"
        result = _parse_sections(text)
        assert "real line" in result["verse"]
        assert "# a comment" not in result["verse"]

    def test_empty_if_no_headers(self):
        from white_generation.pipelines.lyric_pipeline import _parse_sections

        result = _parse_sections("just some text without headers")
        assert result == {}

    def test_section_name_lowercased(self):
        from white_generation.pipelines.lyric_pipeline import _parse_sections

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
        from white_generation.pipelines.lyric_pipeline import _next_candidate_id

        review = {"candidates": []}
        assert _next_candidate_id(review) == "lyrics_01"

    def test_with_existing_candidates(self):
        from white_generation.pipelines.lyric_pipeline import _next_candidate_id

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
        from white_generation.pipelines.lyric_pipeline import _next_candidate_id

        review = {"candidates": [{"id": "lyrics_01"}, {"id": "lyrics_05"}]}
        assert _next_candidate_id(review) == "lyrics_06"


# ---------------------------------------------------------------------------
# 7. _load_or_init_review
# ---------------------------------------------------------------------------


class TestLoadOrInitReview:
    def test_creates_fresh_when_missing(self, tmp_path):
        from white_generation.pipelines.lyric_pipeline import _load_or_init_review

        meta = make_meta()
        review = _load_or_init_review(tmp_path, meta, model="claude-sonnet-4-6", seed=7)
        assert review["pipeline"] == "lyric-generation"
        assert review["color"] == "Red"
        assert review["seed.logicx"] == 7
        assert review["candidates"] == []

    def test_loads_existing_file(self, tmp_path):
        from white_generation.pipelines.lyric_pipeline import (
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
        from white_generation.pipelines.lyric_pipeline import (
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
        from white_generation.pipelines.lyric_pipeline import _build_prompt

        meta = make_meta(concept="machines learning to breathe")
        vocal_sections = [
            {
                "name": "verse",
                "bars": 4,
                "play_count": 2,
                "total_notes": 20,
                "contour": "stepwise",
            }
        ]
        syllable_targets = {"verse": (15, 21)}
        prompt = _build_prompt(meta, vocal_sections, syllable_targets)
        assert "machines learning to breathe" in prompt

    def test_has_syllable_targets(self):
        from white_generation.pipelines.lyric_pipeline import _build_prompt

        meta = make_meta()
        vocal_sections = [
            {
                "name": "verse",
                "bars": 4,
                "play_count": 1,
                "total_notes": 52,
                "contour": "arpeggiated",
            }
        ]
        syllable_targets = {"verse": (39, 54)}
        prompt = _build_prompt(meta, vocal_sections, syllable_targets)
        assert "39" in prompt
        assert "54" in prompt

    def test_has_section_headers(self):
        from white_generation.pipelines.lyric_pipeline import _build_prompt

        meta = make_meta()
        vocal_sections = [
            {
                "name": "verse",
                "bars": 4,
                "play_count": 2,
                "total_notes": 20,
                "contour": "stepwise",
            },
            {
                "name": "chorus",
                "bars": 2,
                "play_count": 4,
                "total_notes": 10,
                "contour": "arpeggiated",
            },
        ]
        syllable_targets = {"verse": (15, 21), "chorus": (7, 10)}
        prompt = _build_prompt(meta, vocal_sections, syllable_targets)
        assert "[verse]" in prompt
        assert "[chorus]" in prompt

    def test_short_phrases_get_melisma_note(self):
        from white_generation.pipelines.lyric_pipeline import Phrase, _build_prompt

        meta = make_meta()
        vocal_sections = [
            {
                "name": "bridge",
                "bars": 2,
                "play_count": 1,
                "total_notes": 4,
                "contour": "stepwise",
                "phrases": [
                    Phrase(start_tick=0, end_tick=100, note_count=1),
                    Phrase(start_tick=200, end_tick=300, note_count=2),
                ],
            }
        ]
        syllable_targets = {"bridge": (2, 4)}
        prompt = _build_prompt(meta, vocal_sections, syllable_targets)
        assert "melisma" in prompt

    def test_long_phrases_no_melisma_note(self):
        from white_generation.pipelines.lyric_pipeline import Phrase, _build_prompt

        meta = make_meta()
        vocal_sections = [
            {
                "name": "verse",
                "bars": 4,
                "play_count": 1,
                "total_notes": 16,
                "contour": "scalar_run",
                "phrases": [
                    Phrase(start_tick=0, end_tick=100, note_count=8),
                    Phrase(start_tick=200, end_tick=300, note_count=8),
                ],
            }
        ]
        syllable_targets = {"verse": (12, 17)}
        prompt = _build_prompt(meta, vocal_sections, syllable_targets)
        assert "melisma" not in prompt


# ---------------------------------------------------------------------------
# 9. sync_lyric_candidates
# ---------------------------------------------------------------------------


class TestSyncLyricCandidates:
    def test_adds_untracked_stub(self, tmp_path):
        from white_generation.pipelines.lyric_pipeline import (
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
        from white_generation.pipelines.lyric_pipeline import (
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
        from white_generation.pipelines.lyric_pipeline import sync_lyric_candidates

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
        from white_composition.promote_part import promote_part

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
        from white_composition.promote_part import promote_part

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
        from white_composition.promote_part import promote_part

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
        from white_composition.promote_part import promote_part

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
        from white_composition.promote_part import promote_part

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
        from white_composition.promote_part import promote_part

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
    """Build a mock Refractor that returns deterministic results."""
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

        from white_generation.pipelines.lyric_pipeline import run_lyric_pipeline

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
            patch("white_analysis.refractor.Refractor", return_value=mock_scorer),
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

    def test_negative_constraints_injected_into_prompt(self, tmp_path):
        """When lyrics_negative_constraints.yml exists at the album root, its
        avoidance block should reach the Claude prompt."""
        import numpy as np
        import yaml as _yaml

        from white_generation.pipelines.lyric_pipeline import run_lyric_pipeline

        album_dir = tmp_path
        prod_dir = _make_production_dir(album_dir / "thread_one")
        with open(album_dir / "lyrics_negative_constraints.yml", "w") as f:
            _yaml.dump(
                {
                    "overused_words": [
                        {
                            "word": "blue",
                            "reason": "'blue' appears in 3/4 songs' lyrics (75%)",
                        }
                    ]
                },
                f,
            )

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
            patch("white_analysis.refractor.Refractor", return_value=mock_scorer),
        ):
            run_lyric_pipeline(
                production_dir=str(prod_dir),
                num_candidates=1,
                model="claude-sonnet-4-6",
            )

        sent_prompt = mock_client.messages.create.call_args.kwargs["messages"][0][
            "content"
        ]
        assert "blue" in sent_prompt
        assert "75%" in sent_prompt

    def test_run_pipeline_writes_review_yml(self, tmp_path):
        """Pipeline should write lyrics_review.yml with one candidate entry."""
        import numpy as np

        from white_generation.pipelines.lyric_pipeline import (
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
            patch("white_analysis.refractor.Refractor", return_value=mock_scorer),
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
        from white_generation.pipelines.lyric_pipeline import (
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
                    "white_analysis.refractor.Refractor",
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


# ---------------------------------------------------------------------------
# White lyric cut-up functions
# ---------------------------------------------------------------------------


class TestCollectSubLyrics:
    def test_finds_approved_from_review(self, tmp_path):
        from white_generation.pipelines.lyric_pipeline import collect_sub_lyrics

        prod = tmp_path / "prod_red"
        cands = prod / "melody" / "candidates"
        cands.mkdir(parents=True)
        (cands / "lyrics_01.txt").write_text("Red verse one\nRed chorus")
        (cands / "lyrics_02.txt").write_text("Red draft two")
        review = {
            "candidates": [
                {"id": "lyrics_01", "file": "lyrics_01.txt", "status": "approved"},
                {"id": "lyrics_02", "file": "lyrics_02.txt", "status": "pending"},
            ]
        }
        with open(cands / "lyrics_review.yml", "w") as f:
            import yaml

            yaml.dump(review, f)
        # chord review for color
        (prod / "chords").mkdir()
        with open(prod / "chords" / "review.yml", "w") as f:
            yaml.dump({"color": "Red", "bpm": 80, "key": "C minor"}, f)

        results = collect_sub_lyrics([prod])
        assert len(results) == 1
        assert results[0]["color"] == "Red"
        assert "Red verse one" in results[0]["lyrics_text"]

    def test_fallback_to_all_txt_when_no_review(self, tmp_path):
        from white_generation.pipelines.lyric_pipeline import collect_sub_lyrics

        prod = tmp_path / "prod_blue"
        cands = prod / "melody" / "candidates"
        cands.mkdir(parents=True)
        (cands / "lyrics_01.txt").write_text("Blue text one")
        (cands / "lyrics_02.txt").write_text("Blue text two")

        results = collect_sub_lyrics([prod])
        assert len(results) == 2

    def test_empty_when_no_melody_candidates(self, tmp_path):
        from white_generation.pipelines.lyric_pipeline import collect_sub_lyrics

        prod = tmp_path / "prod_empty"
        prod.mkdir()
        results = collect_sub_lyrics([prod])
        assert results == []

    def test_skips_empty_txt(self, tmp_path):
        from white_generation.pipelines.lyric_pipeline import collect_sub_lyrics

        prod = tmp_path / "prod_empty_txt"
        cands = prod / "melody" / "candidates"
        cands.mkdir(parents=True)
        (cands / "lyrics_01.txt").write_text("   ")  # whitespace only

        results = collect_sub_lyrics([prod])
        assert results == []


class TestBuildWhiteCutupPrompt:
    def _dummy_meta(self):
        return {
            "title": "Test White Song",
            "color": "White",
            "bpm": 108,
            "time_sig": "4/4",
            "key": "Bb major",
            "concept": "synthesis concept",
        }

    def _dummy_sections(self):
        return [
            {
                "name": "verse",
                "bars": 4,
                "play_count": 1,
                "total_notes": 16,
                "contour": "stepwise",
                "phrases": [],
            }
        ]

    def _dummy_targets(self):
        return {"verse": (12, 17)}

    def test_source_lyrics_section_present(self):
        from white_generation.pipelines.lyric_pipeline import (
            _build_white_cutup_prompt,
        )

        sub_lyrics = [
            {
                "source_dir": "/tmp/red",
                "color": "Red",
                "lyrics_text": "Red line one\nRed line two",
            },
        ]
        prompt = _build_white_cutup_prompt(
            self._dummy_meta(),
            self._dummy_sections(),
            self._dummy_targets(),
            sub_lyrics,
        )
        assert "## Red" in prompt
        assert "Red line one" in prompt
        assert "[verse]" in prompt

    def test_fallback_prompt_when_no_sub_lyrics(self):
        from white_generation.pipelines.lyric_pipeline import (
            _build_white_cutup_prompt,
        )

        prompt = _build_white_cutup_prompt(
            self._dummy_meta(), self._dummy_sections(), self._dummy_targets(), []
        )
        assert "White synthesis" in prompt
        assert "[verse]" in prompt
        assert "## " not in prompt  # no source sections

    def test_non_white_pipeline_unchanged(self, tmp_path):
        """Non-White color songs use the standard _build_prompt, not the cut-up variant."""
        from white_generation.pipelines.lyric_pipeline import _build_prompt

        meta = {
            "title": "Blue Song",
            "color": "Blue",
            "bpm": 60,
            "time_sig": "4/4",
            "key": "G minor",
            "concept": "blue concept",
        }
        sections = [
            {
                "name": "verse",
                "bars": 4,
                "play_count": 1,
                "total_notes": 16,
                "contour": "stepwise",
                "phrases": [],
            }
        ]
        prompt = _build_prompt(meta, sections, {"verse": (12, 17)})
        assert "SOURCE LYRICS" not in prompt
        assert "cut-up" not in prompt

    def test_short_phrases_get_melisma_note(self):
        from white_generation.pipelines.lyric_pipeline import (
            Phrase,
            _build_white_cutup_prompt,
        )

        sections = [
            {
                "name": "bridge",
                "bars": 2,
                "play_count": 1,
                "total_notes": 3,
                "contour": "stepwise",
                "phrases": [
                    Phrase(start_tick=0, end_tick=100, note_count=1),
                    Phrase(start_tick=200, end_tick=300, note_count=2),
                ],
            }
        ]
        prompt = _build_white_cutup_prompt(
            self._dummy_meta(), sections, {"bridge": (2, 3)}, []
        )
        assert "melisma" in prompt


# ---------------------------------------------------------------------------
# Lyric repeat type — read_vocal_sections_from_arrangement
# ---------------------------------------------------------------------------


def _make_arrangement_txt(sections: list[dict], bpm: int = 120) -> str:
    """Build a minimal arrangement.txt with track-4 melody clips.

    sections: list of {name, bars} dicts.
    """
    lines = []
    secs_per_bar = (4.0 / 4) * (60.0 / bpm)  # 4/4 assumed
    cursor = 0.0
    for s in sections:
        dur = s["bars"] * secs_per_bar

        def _tc(t):
            h = int(t // 3600)
            t %= 3600
            m = int(t // 60)
            t %= 60
            frames = (t - int(t)) * 30
            return f"{h:02d}:{m:02d}:{int(t):02d}:{frames:05.2f}"

        lines.append(f"{_tc(cursor)}\t{s['name']}\t4\t{_tc(dur)}")
        cursor += dur
    return "\n".join(lines) + "\n"


class TestReadVocalSectionsRepeatType:
    def _make_melody_dir(self, tmp_path):
        melody_dir = tmp_path / "melody"
        melody_dir.mkdir()
        return melody_dir

    def test_chorus_x3_produces_exact_then_exact_repeat(self, tmp_path):
        from white_generation.pipelines.lyric_pipeline import (
            read_vocal_sections_from_arrangement,
        )

        arr = tmp_path / "arrangement.txt"
        arr.write_text(
            _make_arrangement_txt(
                [
                    {"name": "chorus", "bars": 4},
                    {"name": "chorus", "bars": 4},
                    {"name": "chorus", "bars": 4},
                ]
            )
        )
        melody_dir = self._make_melody_dir(tmp_path)
        sections = read_vocal_sections_from_arrangement(arr, melody_dir, 120, "4/4")
        assert len(sections) == 3
        assert sections[0]["lyric_repeat_type"] == "exact"
        assert sections[1]["lyric_repeat_type"] == "exact_repeat"
        assert sections[2]["lyric_repeat_type"] == "exact_repeat"

    def test_verse_x2_produces_two_variation_entries(self, tmp_path):
        from white_generation.pipelines.lyric_pipeline import (
            read_vocal_sections_from_arrangement,
        )

        arr = tmp_path / "arrangement.txt"
        arr.write_text(
            _make_arrangement_txt(
                [
                    {"name": "verse", "bars": 4},
                    {"name": "verse", "bars": 4},
                ]
            )
        )
        melody_dir = self._make_melody_dir(tmp_path)
        sections = read_vocal_sections_from_arrangement(arr, melody_dir, 120, "4/4")
        assert len(sections) == 2
        assert sections[0]["lyric_repeat_type"] == "variation"
        assert sections[1]["lyric_repeat_type"] == "variation"

    def test_plan_override_takes_priority(self, tmp_path):
        """lyric_repeat_type in production_plan.yml overrides label inference."""
        import yaml

        from white_generation.pipelines.lyric_pipeline import (
            read_vocal_sections_from_arrangement,
        )

        arr = tmp_path / "arrangement.txt"
        arr.write_text(_make_arrangement_txt([{"name": "bridge", "bars": 4}]))
        (tmp_path / "production_plan.yml").write_text(
            yaml.dump(
                {
                    "sections": [
                        {"name": "bridge", "lyric_repeat_type": "exact", "bars": 4}
                    ]
                }
            )
        )
        melody_dir = self._make_melody_dir(tmp_path)
        sections = read_vocal_sections_from_arrangement(
            arr, melody_dir, 120, "4/4", production_dir=tmp_path
        )
        assert sections[0]["lyric_repeat_type"] == "exact"

    def test_plan_override_typo_normalised_to_fresh(self, tmp_path):
        """A typo like 'Exact' in production_plan.yml is normalised to 'fresh', not 'exact'."""
        import yaml

        from white_generation.pipelines.lyric_pipeline import (
            read_vocal_sections_from_arrangement,
        )

        arr = tmp_path / "arrangement.txt"
        arr.write_text(_make_arrangement_txt([{"name": "bridge", "bars": 4}]))
        # Typo: capitalised 'Exact' — should normalise to 'exact' (valid)
        (tmp_path / "production_plan.yml").write_text(
            yaml.dump(
                {
                    "sections": [
                        {"name": "bridge", "lyric_repeat_type": "Exact", "bars": 4}
                    ]
                }
            )
        )
        melody_dir = self._make_melody_dir(tmp_path)
        sections = read_vocal_sections_from_arrangement(
            arr, melody_dir, 120, "4/4", production_dir=tmp_path
        )
        # 'Exact' normalises to 'exact' which is valid — override should apply
        assert sections[0]["lyric_repeat_type"] == "exact"

    def test_plan_override_invalid_falls_back_to_inferred(self, tmp_path):
        """An unrecognised repeat type in production_plan.yml falls back to label inference."""
        import yaml

        from white_generation.pipelines.lyric_pipeline import (
            read_vocal_sections_from_arrangement,
        )

        arr = tmp_path / "arrangement.txt"
        arr.write_text(_make_arrangement_txt([{"name": "chorus", "bars": 4}]))
        # Invalid value — should normalise to 'fresh', so label inference kicks in → 'exact'
        (tmp_path / "production_plan.yml").write_text(
            yaml.dump(
                {
                    "sections": [
                        {"name": "chorus", "lyric_repeat_type": "nonsense", "bars": 4}
                    ]
                }
            )
        )
        melody_dir = self._make_melody_dir(tmp_path)
        sections = read_vocal_sections_from_arrangement(
            arr, melody_dir, 120, "4/4", production_dir=tmp_path
        )
        # 'nonsense' → 'fresh' (normalised), no override stored → inferred from 'chorus' → 'exact'
        assert sections[0]["lyric_repeat_type"] == "exact"


# ---------------------------------------------------------------------------
# Lyric repeat type — prompt builder
# ---------------------------------------------------------------------------


class TestBuildPromptRepeatTypes:
    def _meta(self):
        return {
            "title": "T",
            "color": "Red",
            "bpm": 120,
            "time_sig": "4/4",
            "key": "C major",
            "concept": "test",
        }

    def _sec(self, name, approved_label=None, repeat_type="fresh"):
        return {
            "name": name,
            "approved_label": approved_label or name,
            "bars": 4,
            "play_count": 1,
            "total_notes": 16,
            "contour": "stepwise",
            "phrases": [],
            "lyric_repeat_type": repeat_type,
        }

    def test_exact_repeat_section_skipped(self):
        from white_core.enums.lyric_repeat_type import LyricRepeatType
        from white_generation.pipelines.lyric_pipeline import _build_prompt

        sections = [
            self._sec("chorus", repeat_type="exact"),
            self._sec(
                "chorus_2",
                approved_label="chorus",
                repeat_type=LyricRepeatType.EXACT_REPEAT,
            ),
            self._sec(
                "chorus_3",
                approved_label="chorus",
                repeat_type=LyricRepeatType.EXACT_REPEAT,
            ),
        ]
        prompt = _build_prompt(self._meta(), sections, {})
        assert "[chorus]" in prompt
        assert "[chorus_2]" not in prompt
        assert "[chorus_3]" not in prompt

    def test_exact_annotation_present(self):
        from white_generation.pipelines.lyric_pipeline import _build_prompt

        sections = [self._sec("chorus", repeat_type="exact")]
        prompt = _build_prompt(self._meta(), sections, {})
        assert "repeats verbatim" in prompt

    def test_variation_annotation_on_second_instance(self):
        from white_generation.pipelines.lyric_pipeline import _build_prompt

        sections = [
            self._sec("verse", repeat_type="variation"),
            self._sec("verse_2", approved_label="verse", repeat_type="variation"),
        ]
        prompt = _build_prompt(self._meta(), sections, {})
        assert "[verse]" in prompt
        assert "[verse_2]" in prompt
        assert "Variation 2 of verse" in prompt

    def test_fresh_sections_unchanged(self):
        from white_generation.pipelines.lyric_pipeline import _build_prompt

        sections = [self._sec("bridge", repeat_type="fresh")]
        prompt = _build_prompt(self._meta(), sections, {})
        assert "[bridge]" in prompt
        assert "verbatim" not in prompt
        # No per-section variation annotation (boilerplate mentions "Variation" in format docs)
        assert "Variation 1 of" not in prompt
        assert "Variation 2 of" not in prompt


# ---------------------------------------------------------------------------
# Lyric repeat type — _compute_fitting exact_repeat copy
# ---------------------------------------------------------------------------


class TestComputeFittingExactRepeat:
    def test_exact_repeat_copies_from_source(self, tmp_path):
        from white_core.enums.lyric_repeat_type import LyricRepeatType
        from white_generation.pipelines.lyric_pipeline import _compute_fitting

        # First instance is 'chorus', second is 'chorus_2' (exact_repeat)
        # Use the enum directly — the pipeline stores LyricRepeatType.EXACT_REPEAT,
        # never the raw string (which _normalize_repeat_type rejects as external input)
        vocal_sections = [
            {
                "name": "chorus",
                "total_notes": 8,
                "play_count": 1,
                "lyric_repeat_type": LyricRepeatType.EXACT,
                "exact_source": "chorus",
            },
            {
                "name": "chorus_2",
                "total_notes": 8,
                "play_count": 1,
                "lyric_repeat_type": LyricRepeatType.EXACT_REPEAT,
                "exact_source": "chorus",
            },
        ]
        # Only write lyrics for the base section (no [chorus_2] block)
        text = "[chorus]\nHello world\nBeautiful day\n"
        melody_dir = tmp_path / "melody"
        melody_dir.mkdir()
        result = _compute_fitting(text, vocal_sections, melody_dir)
        # chorus_2 should have the same fitting result as chorus
        assert "chorus" in result
        assert "chorus_2" in result
        assert result["chorus_2"] == result["chorus"]


# ---------------------------------------------------------------------------
# Rhyme scheme assignment
# ---------------------------------------------------------------------------


def make_vocal_section(
    name,
    approved_label=None,
    phrase_count=4,
    lyric_repeat_type=None,
    play_count=1,
):
    from white_generation.pipelines.lyric_pipeline import Phrase

    return {
        "name": name,
        "approved_label": approved_label or name,
        "bars": 4,
        "play_count": play_count,
        "total_notes": phrase_count * 4,
        "contour": "stepwise",
        "lyric_repeat_type": lyric_repeat_type,
        "phrases": [
            Phrase(start_tick=i * 100, end_tick=i * 100 + 50, note_count=4)
            for i in range(phrase_count)
        ],
    }


class TestDefaultRhymeScheme:
    def test_two_lines_is_aa(self):
        from white_generation.pipelines.lyric_pipeline import _default_rhyme_scheme

        assert _default_rhyme_scheme(2) == "AA"

    def test_four_lines_is_xaxa(self):
        from white_generation.pipelines.lyric_pipeline import _default_rhyme_scheme

        assert _default_rhyme_scheme(4) == "XAXA"

    def test_other_counts_unrhymed(self):
        from white_generation.pipelines.lyric_pipeline import _default_rhyme_scheme

        assert _default_rhyme_scheme(3) == "none"
        assert _default_rhyme_scheme(5) == "none"
        assert _default_rhyme_scheme(0) == "none"


class TestAssignRhymeSchemes:
    def test_default_xaxa_for_four_lines(self):
        from white_generation.pipelines.lyric_pipeline import assign_rhyme_schemes

        sections = [make_vocal_section("verse", phrase_count=4)]
        schemes = assign_rhyme_schemes(sections)
        assert schemes["verse"] == "XAXA"

    def test_proposal_override(self):
        from white_generation.pipelines.lyric_pipeline import assign_rhyme_schemes

        sections = [make_vocal_section("chorus", phrase_count=4)]
        schemes = assign_rhyme_schemes(sections, {"chorus": "AABB"})
        assert schemes["chorus"] == "AABB"

    def test_partial_x_override(self):
        from white_generation.pipelines.lyric_pipeline import assign_rhyme_schemes

        sections = [make_vocal_section("verse_1", phrase_count=4)]
        schemes = assign_rhyme_schemes(sections, {"verse_1": "AXAX"})
        assert schemes["verse_1"] == "AXAX"

    def test_explicit_none_disables_rhyme(self):
        from white_generation.pipelines.lyric_pipeline import assign_rhyme_schemes

        sections = [make_vocal_section("bridge", phrase_count=2)]
        schemes = assign_rhyme_schemes(sections, {"bridge": "none"})
        assert schemes["bridge"] == "none"

    def test_repeated_section_reuses_scheme(self):
        from white_generation.pipelines.lyric_pipeline import assign_rhyme_schemes

        sections = [
            make_vocal_section("verse_1", phrase_count=4),
            make_vocal_section("verse_2", phrase_count=2),  # would default to AA
        ]
        schemes = assign_rhyme_schemes(sections, {"verse_1": "ABAB"})
        assert schemes["verse_1"] == "ABAB"
        assert schemes["verse_2"] == "ABAB"

    def test_exact_repeat_sections_are_skipped(self):
        from white_core.enums.lyric_repeat_type import LyricRepeatType
        from white_generation.pipelines.lyric_pipeline import assign_rhyme_schemes

        sections = [
            make_vocal_section(
                "chorus", phrase_count=2, lyric_repeat_type=LyricRepeatType.EXACT
            ),
            make_vocal_section(
                "chorus_2",
                approved_label="chorus",
                phrase_count=2,
                lyric_repeat_type=LyricRepeatType.EXACT_REPEAT,
            ),
        ]
        schemes = assign_rhyme_schemes(sections)
        assert "chorus" in schemes
        assert "chorus_2" not in schemes


class TestRhymeSchemeLinePairs:
    def test_xaxa_pairs_lines_2_and_4(self):
        from white_generation.pipelines.lyric_pipeline import rhyme_scheme_line_pairs

        assert rhyme_scheme_line_pairs("XAXA") == [(2, 4, "A")]

    def test_abab_pairs_both_groups(self):
        from white_generation.pipelines.lyric_pipeline import rhyme_scheme_line_pairs

        pairs = rhyme_scheme_line_pairs("ABAB")
        assert (1, 3, "A") in pairs
        assert (2, 4, "B") in pairs

    def test_none_scheme_has_no_pairs(self):
        from white_generation.pipelines.lyric_pipeline import rhyme_scheme_line_pairs

        assert rhyme_scheme_line_pairs("none") == []
        assert rhyme_scheme_line_pairs("") == []


class TestRhymeSchemePromptText:
    def test_xaxa_describes_free_and_rhyming_lines(self):
        from white_generation.pipelines.lyric_pipeline import (
            _rhyme_scheme_prompt_text,
        )

        text = _rhyme_scheme_prompt_text("XAXA")
        assert "2 and 4" in text
        assert "free" in text

    def test_none_scheme_returns_none(self):
        from white_generation.pipelines.lyric_pipeline import (
            _rhyme_scheme_prompt_text,
        )

        assert _rhyme_scheme_prompt_text("none") is None
        assert _rhyme_scheme_prompt_text("") is None


# ---------------------------------------------------------------------------
# Rhyme verification (CMUdict + suffix heuristic)
# ---------------------------------------------------------------------------


class TestCheckRhymePair:
    def test_dictionary_hit_match(self):
        from white_generation.pipelines.lyric_pipeline import check_rhyme_pair

        result = check_rhyme_pair("cat", "hat")
        assert result["status"] == "match"
        assert result["method"] == "cmudict"

    def test_dictionary_hit_real_mismatch(self):
        from white_generation.pipelines.lyric_pipeline import check_rhyme_pair

        result = check_rhyme_pair("cat", "dog")
        assert result["status"] == "fail"
        assert result["method"] == "cmudict"

    def test_out_of_dictionary_falls_back_to_maybe(self):
        from white_generation.pipelines.lyric_pipeline import check_rhyme_pair

        # Invented/portmanteau words absent from CMUdict
        result = check_rhyme_pair("rebracketing", "infranym")
        assert result["status"] == "maybe"

    def test_one_word_out_of_dictionary_is_maybe_not_fail(self):
        from white_generation.pipelines.lyric_pipeline import check_rhyme_pair

        result = check_rhyme_pair("cat", "zzznonwordzzz")
        assert result["status"] == "maybe"


# ---------------------------------------------------------------------------
# Verify / revise loop
# ---------------------------------------------------------------------------


class TestCheckCandidate:
    def test_clean_draft_has_no_issues(self):
        from white_generation.pipelines.lyric_pipeline import (
            Phrase,
            _check_candidate,
        )

        sections = [
            {
                "name": "verse",
                "play_count": 1,
                "lyric_repeat_type": None,
                "phrases": [
                    Phrase(0, 50, 2),
                    Phrase(100, 150, 2),
                ],
            }
        ]
        # 2-note phrases -> _phrase_syllable_range widens to include 1-2 syllables
        text = "[verse]\nCat\nHat\n"
        issues = _check_candidate(text, sections, {"verse": "AA"})
        assert issues["rhyme_issues"] == []

    def test_rhyme_mismatch_is_flagged(self):
        from white_generation.pipelines.lyric_pipeline import (
            Phrase,
            _check_candidate,
        )

        sections = [
            {
                "name": "verse",
                "play_count": 1,
                "lyric_repeat_type": None,
                "phrases": [Phrase(0, 50, 2), Phrase(100, 150, 2)],
            }
        ]
        text = "[verse]\nThe cat\nThe dog\n"
        issues = _check_candidate(text, sections, {"verse": "AA"})
        assert len(issues["rhyme_issues"]) == 1
        assert issues["rhyme_issues"][0]["letter"] == "A"

    def test_syllable_miss_is_flagged(self):
        from white_generation.pipelines.lyric_pipeline import (
            Phrase,
            _check_candidate,
        )

        sections = [
            {
                "name": "verse",
                "play_count": 1,
                "lyric_repeat_type": None,
                "phrases": [Phrase(0, 500, 12)],  # needs ~9-14 syllables
            }
        ]
        text = "[verse]\nHi\n"  # wildly under target
        issues = _check_candidate(text, sections, {"verse": "none"})
        assert len(issues["syllable_issues"]) == 1

    def test_missing_line_is_flagged_not_silently_skipped(self):
        """A draft with fewer lines than phrases must be flagged, not skipped."""
        from white_generation.pipelines.lyric_pipeline import (
            Phrase,
            _check_candidate,
        )

        sections = [
            {
                "name": "verse",
                "play_count": 1,
                "lyric_repeat_type": None,
                "phrases": [Phrase(0, 50, 2), Phrase(100, 150, 2)],
            }
        ]
        # Only one line provided for a two-phrase section — the second is missing.
        text = "[verse]\nCat\n"
        issues = _check_candidate(text, sections, {"verse": "AA"})
        assert len(issues["syllable_issues"]) == 1
        missing = issues["syllable_issues"][0]
        assert missing["line_idx"] == 2
        assert missing["text"] == ""
        assert missing["syllables"] == 0


class TestGenerateLyricCandidate:
    def test_clean_draft_skips_revision(self):
        """A draft that passes every check on the first try makes exactly one API call."""
        from white_generation.pipelines.lyric_pipeline import (
            Phrase,
            generate_lyric_candidate,
        )

        sections = [
            {
                "name": "verse",
                "play_count": 1,
                "lyric_repeat_type": None,
                "phrases": [Phrase(0, 50, 2), Phrase(100, 150, 2)],
            }
        ]
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.content = [MagicMock(text="[verse]\nCat\nHat\n")]
        mock_client.messages.create.return_value = mock_response

        text, outcome = generate_lyric_candidate(
            mock_client, "prompt text", "claude-sonnet-4-6", sections, {"verse": "AA"}
        )
        assert mock_client.messages.create.call_count == 1
        assert outcome["turns_used"] == 0

    def test_failing_draft_triggers_bounded_revision(self):
        """A draft that keeps failing is revised exactly MAX_REVISION_TURNS times."""
        from white_generation.pipelines.lyric_pipeline import (
            MAX_REVISION_TURNS,
            Phrase,
            generate_lyric_candidate,
        )

        sections = [
            {
                "name": "verse",
                "play_count": 1,
                "lyric_repeat_type": None,
                "phrases": [Phrase(0, 50, 2), Phrase(100, 150, 2)],
            }
        ]
        mock_client = MagicMock()
        mock_response = MagicMock()
        # Mismatched rhyme every time — never resolves
        mock_response.content = [MagicMock(text="[verse]\nThe cat\nThe dog\n")]
        mock_client.messages.create.return_value = mock_response

        text, outcome = generate_lyric_candidate(
            mock_client, "prompt text", "claude-sonnet-4-6", sections, {"verse": "AA"}
        )
        assert outcome["turns_used"] == MAX_REVISION_TURNS
        assert mock_client.messages.create.call_count == MAX_REVISION_TURNS + 1
        assert outcome["final_rhyme_issues"] > 0

    def test_revision_message_names_only_failing_lines(self):
        from white_generation.pipelines.lyric_pipeline import (
            Phrase,
            generate_lyric_candidate,
        )

        sections = [
            {
                "name": "verse",
                "play_count": 1,
                "lyric_repeat_type": None,
                "phrases": [Phrase(0, 50, 2), Phrase(100, 150, 2)],
            }
        ]
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.content = [MagicMock(text="[verse]\nThe cat\nThe dog\n")]
        mock_client.messages.create.return_value = mock_response

        generate_lyric_candidate(
            mock_client, "prompt text", "claude-sonnet-4-6", sections, {"verse": "AA"}
        )
        # `messages` is mutated in place across calls, so every recorded call
        # shares the same final list object — index 2 (the first revision
        # turn) is set once and never touched by later appends.
        final_messages = mock_client.messages.create.call_args_list[-1].kwargs[
            "messages"
        ]
        revision_text = final_messages[2]["content"]
        assert "cat" in revision_text
        assert "dog" in revision_text
        assert "do not rhyme" in revision_text


# ---------------------------------------------------------------------------
# Chorus/hook content guidance
# ---------------------------------------------------------------------------


class TestHookGuidance:
    def test_chorus_label_gets_hook_guidance(self):
        from white_core.enums.lyric_repeat_type import LyricRepeatType
        from white_generation.pipelines.lyric_pipeline import _build_prompt

        meta = make_meta()
        sections = [
            make_vocal_section(
                "chorus_1", phrase_count=4, lyric_repeat_type=LyricRepeatType.EXACT
            )
        ]
        syllable_targets = {"chorus_1": (10, 16)}
        prompt = _build_prompt(meta, sections, syllable_targets)
        assert "HOOK GUIDANCE" in prompt

    def test_refrain_and_hook_labels_also_qualify(self):
        from white_core.enums.lyric_repeat_type import LyricRepeatType
        from white_generation.pipelines.lyric_pipeline import _build_prompt

        meta = make_meta()
        for label in ("refrain", "hook"):
            sections = [
                make_vocal_section(
                    label, phrase_count=4, lyric_repeat_type=LyricRepeatType.EXACT
                )
            ]
            prompt = _build_prompt(meta, sections, {label: (10, 16)})
            assert "HOOK GUIDANCE" in prompt

    def test_verse_does_not_get_hook_guidance(self):
        from white_core.enums.lyric_repeat_type import LyricRepeatType
        from white_generation.pipelines.lyric_pipeline import _build_prompt

        meta = make_meta()
        sections = [
            make_vocal_section(
                "verse_1", phrase_count=4, lyric_repeat_type=LyricRepeatType.VARIATION
            )
        ]
        prompt = _build_prompt(meta, sections, {"verse_1": (10, 16)})
        assert "HOOK GUIDANCE" not in prompt

    def test_exact_repeat_instance_not_reprompted(self):
        from white_core.enums.lyric_repeat_type import LyricRepeatType
        from white_generation.pipelines.lyric_pipeline import _build_prompt

        meta = make_meta()
        sections = [
            make_vocal_section(
                "chorus", phrase_count=4, lyric_repeat_type=LyricRepeatType.EXACT
            ),
            make_vocal_section(
                "chorus_2",
                approved_label="chorus",
                phrase_count=4,
                lyric_repeat_type=LyricRepeatType.EXACT_REPEAT,
            ),
        ]
        prompt = _build_prompt(
            meta, sections, {"chorus": (10, 16), "chorus_2": (10, 16)}
        )
        assert "[chorus_2]" not in prompt
        # Only one hook guidance block should appear (for the EXACT instance)
        assert prompt.count("HOOK GUIDANCE") == 1
