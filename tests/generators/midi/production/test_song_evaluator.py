"""Tests for song_evaluator pure-logic helpers."""

from pathlib import Path

import pytest

from app.generators.midi.production.song_evaluator import (
    _chromatic_consistency,
    _compute_theory_score,
    _count_syllables,
    _parse_arrangement_txt_metrics,
    _safe_mean,
)

# ---------------------------------------------------------------------------
# _safe_mean
# ---------------------------------------------------------------------------


class TestSafeMean:
    def test_empty_returns_zero(self):
        assert _safe_mean([]) == 0.0

    def test_single_value(self):
        assert _safe_mean([0.8]) == pytest.approx(0.8)

    def test_multiple_values(self):
        assert _safe_mean([0.5, 1.0]) == pytest.approx(0.75)

    def test_all_zeros(self):
        assert _safe_mean([0.0, 0.0, 0.0]) == pytest.approx(0.0)

    def test_all_ones(self):
        assert _safe_mean([1.0, 1.0]) == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# _chromatic_consistency
# ---------------------------------------------------------------------------


class TestChromaticConsistency:
    def test_empty_returns_one(self):
        assert _chromatic_consistency([]) == 1.0

    def test_single_score_returns_one(self):
        assert _chromatic_consistency([0.5]) == 1.0

    def test_identical_scores_return_one(self):
        assert _chromatic_consistency([0.8, 0.8, 0.8]) == pytest.approx(1.0)

    def test_max_spread_low_consistency(self):
        result = _chromatic_consistency([0.0, 1.0])
        assert result < 0.5  # stdev of [0,1] is ~0.707 → 1 - 0.707 ≈ 0.29

    def test_result_clamped_at_zero(self):
        # Very high stdev → result would be negative → clamped to 0
        result = _chromatic_consistency([0.0, 0.0, 1.0, 1.0])
        assert result >= 0.0

    def test_two_similar_scores(self):
        result = _chromatic_consistency([0.9, 0.91])
        assert result > 0.9


# ---------------------------------------------------------------------------
# _compute_theory_score
# ---------------------------------------------------------------------------


class TestComputeTheoryScore:
    def test_drums_uses_energy_appropriateness(self):
        candidate = {"scores": {"energy_appropriateness": 0.75}}
        assert _compute_theory_score(candidate, "drums") == pytest.approx(0.75)

    def test_drums_missing_score_returns_zero(self):
        candidate = {"scores": {}}
        assert _compute_theory_score(candidate, "drums") == pytest.approx(0.0)

    def test_chords_averages_theory_fields(self):
        candidate = {
            "scores": {"theory": {"melody": 0.8, "voice_leading": 0.6, "variety": 1.0}}
        }
        result = _compute_theory_score(candidate, "chords")
        assert result == pytest.approx((0.8 + 0.6 + 1.0) / 3)

    def test_bass_averages_correct_fields(self):
        candidate = {
            "scores": {
                "theory": {
                    "root_adherence": 1.0,
                    "voice_leading": 0.5,
                    "kick_alignment": 0.75,
                }
            }
        }
        result = _compute_theory_score(candidate, "bass")
        assert result == pytest.approx((1.0 + 0.5 + 0.75) / 3)

    def test_melody_averages_correct_fields(self):
        candidate = {
            "scores": {
                "theory": {
                    "singability": 0.9,
                    "chord_tone_alignment": 0.7,
                    "contour_quality": 0.8,
                }
            }
        }
        result = _compute_theory_score(candidate, "melody")
        assert result == pytest.approx((0.9 + 0.7 + 0.8) / 3)

    def test_missing_theory_subfields_default_zero(self):
        candidate = {"scores": {"theory": {}}}
        result = _compute_theory_score(candidate, "chords")
        assert result == pytest.approx(0.0)

    def test_missing_scores_key(self):
        candidate = {}
        assert _compute_theory_score(candidate, "chords") == pytest.approx(0.0)

    def test_unknown_phase_returns_zero(self):
        candidate = {"scores": {"theory": {"x": 1.0}}}
        assert _compute_theory_score(candidate, "unknown_phase") == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# _count_syllables
# ---------------------------------------------------------------------------


class TestCountSyllables:
    def test_simple_one_syllable(self):
        assert _count_syllables("red") == 1

    def test_two_syllable_word(self):
        assert _count_syllables("music") == 2

    def test_three_syllable_word(self):
        assert _count_syllables("temporal") == 3

    def test_silent_e_reduced(self):
        # "make" has a vowel group "a" + silent "e" → 1 syllable
        assert _count_syllables("make") == 1

    def test_le_ending_not_reduced(self):
        # "little" ends in "-le" → should not reduce; has 2 syllables
        assert _count_syllables("little") == 2

    def test_empty_string(self):
        assert _count_syllables("") == 0

    def test_numbers_and_punctuation_ignored(self):
        # "123" has no alpha → 0
        assert _count_syllables("123") == 0

    def test_hyphenated_word(self):
        # "self-made" → "self" (1) + "made" (1 due to silent-e) = 2
        assert _count_syllables("self-made") == 2

    def test_multi_word_line(self):
        count = _count_syllables("the red and blue")
        assert count >= 4  # at least one syllable per word

    def test_minimum_one_per_word(self):
        # Even a word with no counted vowels gets at least 1
        assert _count_syllables("brrr") >= 1


# ---------------------------------------------------------------------------
# _parse_arrangement_txt_metrics
# ---------------------------------------------------------------------------


class TestParseArrangementTxtMetrics:
    def _write_arr(self, tmp_path: Path, lines: list[str]) -> Path:
        arr = tmp_path / "arrangement.txt"
        arr.write_text("\n".join(lines))
        return tmp_path

    def test_absent_file_returns_none(self, tmp_path):
        result = _parse_arrangement_txt_metrics(tmp_path)
        assert result is None

    def test_empty_file_returns_none(self, tmp_path):
        (tmp_path / "arrangement.txt").write_text("")
        result = _parse_arrangement_txt_metrics(tmp_path)
        assert result is None

    def test_basic_single_clip(self, tmp_path):
        # start_bar beat sub tick  label  track  clip_bars 0 0 0
        lines = ["1 1 1 1 verse 1 4 0 0 0"]
        self._write_arr(tmp_path, lines)
        result = _parse_arrangement_txt_metrics(tmp_path)
        assert result is not None
        # total_bars = max(1+4) - 1 = 4
        assert result["total_bars"] == 4

    def test_section_variety_all_different(self, tmp_path):
        lines = [
            "1 1 1 1 verse 1 4 0 0 0",
            "5 1 1 1 chorus 1 4 0 0 0",
            "9 1 1 1 bridge 1 4 0 0 0",
        ]
        self._write_arr(tmp_path, lines)
        result = _parse_arrangement_txt_metrics(tmp_path)
        assert result["section_variety"] == pytest.approx(1.0)
        assert result["unique_sections"] == 3

    def test_section_variety_all_same(self, tmp_path):
        lines = [
            "1 1 1 1 verse 1 4 0 0 0",
            "5 1 1 1 verse 1 4 0 0 0",
        ]
        self._write_arr(tmp_path, lines)
        result = _parse_arrangement_txt_metrics(tmp_path)
        assert result["section_variety"] == pytest.approx(0.5)
        assert result["unique_sections"] == 1

    def test_vocal_coverage(self, tmp_path):
        # chord track (1) 8 bars total; melody track (4) covers 4 bars
        lines = [
            "1 1 1 1 verse 1 8 0 0 0",
            "1 1 1 1 verse 4 4 0 0 0",
        ]
        self._write_arr(tmp_path, lines)
        result = _parse_arrangement_txt_metrics(tmp_path)
        # total_bars = max(1+8, 1+4) - 1 = 8
        assert result["total_bars"] == 8
        assert result["vocal_coverage"] == pytest.approx(4 / 8)

    def test_no_vocal_clips(self, tmp_path):
        lines = ["1 1 1 1 verse 1 4 0 0 0"]
        self._write_arr(tmp_path, lines)
        result = _parse_arrangement_txt_metrics(tmp_path)
        assert result["vocal_coverage"] == pytest.approx(0.0)

    def test_malformed_line_skipped(self, tmp_path):
        lines = [
            "bad line",
            "1 1 1 1 verse 1 4 0 0 0",
        ]
        self._write_arr(tmp_path, lines)
        result = _parse_arrangement_txt_metrics(tmp_path)
        assert result is not None
        assert result["total_bars"] == 4

    def test_all_malformed_returns_none(self, tmp_path):
        lines = ["bad", "also bad"]
        self._write_arr(tmp_path, lines)
        result = _parse_arrangement_txt_metrics(tmp_path)
        assert result is None
