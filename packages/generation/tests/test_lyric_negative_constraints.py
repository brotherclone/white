"""Tests for lyric_negative_constraints."""

from pathlib import Path

import yaml

from white_generation.lyric_negative_constraints import (
    analyze_word_frequency,
    collect_lyric_texts,
    format_for_prompt,
    generate_constraints,
    load_constraints,
    write_constraints,
)


def _write_song_lyrics(
    album_dir: Path, thread: str, production: str, text: str
) -> None:
    melody_dir = album_dir / thread / "production" / production / "melody"
    melody_dir.mkdir(parents=True, exist_ok=True)
    (melody_dir / "lyrics.txt").write_text(text)


class TestCollectLyricTexts:
    def test_finds_promoted_lyrics(self, tmp_path):
        _write_song_lyrics(tmp_path, "red-thread", "song_one", "[verse]\nBlue thing")
        results = collect_lyric_texts(tmp_path)
        assert len(results) == 1
        assert results[0]["song_id"] == "red-thread__song_one"
        assert "Blue thing" in results[0]["text"]

    def test_skips_empty_lyrics(self, tmp_path):
        _write_song_lyrics(tmp_path, "red-thread", "song_one", "   ")
        results = collect_lyric_texts(tmp_path)
        assert results == []

    def test_no_songs(self, tmp_path):
        assert collect_lyric_texts(tmp_path) == []


class TestAnalyzeWordFrequency:
    def test_overused_short_word_flagged(self):
        song_texts = [
            {"song_id": "a", "text": "[verse]\nblue thing is here"},
            {"song_id": "b", "text": "[verse]\nnow it's blue and gone"},
            {"song_id": "c", "text": "[verse]\nblue skies again"},
            {"song_id": "d", "text": "[verse]\nsomething else entirely different"},
        ]
        result = analyze_word_frequency(song_texts, threshold=0.3)
        words = {e["word"] for e in result["overused_words"]}
        assert "blue" in words

    def test_below_threshold_not_flagged(self):
        song_texts = [
            {"song_id": "a", "text": "[verse]\nblue thing"},
            {"song_id": "b", "text": "[verse]\nsomething else"},
            {"song_id": "c", "text": "[verse]\nanother line"},
            {"song_id": "d", "text": "[verse]\nfully unrelated"},
        ]
        result = analyze_word_frequency(song_texts, threshold=0.3)
        words = {e["word"] for e in result["overused_words"]}
        assert "blue" not in words

    def test_stopwords_excluded(self):
        song_texts = [
            {"song_id": "a", "text": "the you and it"},
            {"song_id": "b", "text": "the you and it"},
            {"song_id": "c", "text": "the you and it"},
        ]
        result = analyze_word_frequency(song_texts, threshold=0.3)
        assert result["overused_words"] == []

    def test_multisyllable_words_excluded(self):
        song_texts = [
            {"song_id": "a", "text": "consciousness examining consciousness"},
            {"song_id": "b", "text": "consciousness again"},
            {"song_id": "c", "text": "consciousness returns"},
        ]
        result = analyze_word_frequency(song_texts, threshold=0.3)
        words = {e["word"] for e in result["overused_words"]}
        assert "consciousness" not in words

    def test_empty_input(self):
        result = analyze_word_frequency([])
        assert result["overused_words"] == []


class TestGenerateConstraints:
    def test_note_for_too_few_songs(self, tmp_path):
        _write_song_lyrics(tmp_path, "red-thread", "song_one", "[verse]\nblue thing")
        constraints = generate_constraints(tmp_path)
        assert constraints["song_count"] == 1
        assert "note" in constraints

    def test_no_note_with_enough_songs(self, tmp_path):
        for i in range(3):
            _write_song_lyrics(
                tmp_path, "red-thread", f"song_{i}", f"[verse]\nunique text {i}"
            )
        constraints = generate_constraints(tmp_path)
        assert constraints["song_count"] == 3
        assert "note" not in constraints


class TestFormatForPrompt:
    def test_empty_when_no_overused_words(self):
        assert format_for_prompt({"overused_words": []}) == ""

    def test_includes_words_and_reasons(self):
        constraints = {
            "overused_words": [
                {"word": "blue", "reason": "'blue' appears in 3/4 songs' lyrics (75%)"}
            ]
        }
        block = format_for_prompt(constraints)
        assert "blue" in block
        assert "75%" in block


class TestWriteAndLoadConstraints:
    def test_round_trip(self, tmp_path):
        constraints = {
            "generated_from": str(tmp_path),
            "song_count": 2,
            "overused_words": [{"word": "blue", "count": 2, "fraction": 1.0}],
        }
        out_path = tmp_path / "lyrics_negative_constraints.yml"
        write_constraints(out_path, constraints)
        loaded = load_constraints(tmp_path)
        assert loaded["song_count"] == 2
        assert loaded["overused_words"][0]["word"] == "blue"

    def test_load_absent_returns_none(self, tmp_path):
        assert load_constraints(tmp_path) is None

    def test_writes_valid_yaml(self, tmp_path):
        out_path = tmp_path / "lyrics_negative_constraints.yml"
        write_constraints(out_path, {"song_count": 0, "overused_words": []})
        with open(out_path) as f:
            data = yaml.safe_load(f)
        assert data["song_count"] == 0
