"""
Tests for word boundary matching utilities.
"""

import pytest
from white_extraction.extractors.manifest_extractor.word_matching import (
    calculate_coverage,
    calculate_density,
    count_pattern_matches,
    count_word_matches,
    find_unique_word_matches,
)


class TestCountWordMatches:
    def test_empty_word_list_returns_zero(self):
        assert count_word_matches([], "some text here") == 0

    def test_empty_text_returns_zero(self):
        assert count_word_matches(["when", "where"], "") == 0

    def test_both_empty_returns_zero(self):
        assert count_word_matches([], "") == 0

    def test_single_match(self):
        assert count_word_matches(["when"], "when I was young") == 1

    def test_multiple_matches_same_word(self):
        assert count_word_matches(["when"], "when I go when I stay") == 2

    def test_word_boundary_no_partial_match(self):
        # "when" should NOT match inside "whenever"
        assert count_word_matches(["when"], "whenever I think about it") == 0

    def test_word_boundary_no_prefix_match(self):
        # "real" should NOT match "really"
        assert count_word_matches(["real"], "it was really a dream") == 0

    def test_word_boundary_no_suffix_match(self):
        # "act" should NOT match "actually"
        assert count_word_matches(["act"], "he did actually act strangely") == 1

    def test_case_insensitive(self):
        assert count_word_matches(["when"], "When I was young") == 1
        assert count_word_matches(["WHEN"], "when I was young") == 1

    def test_multiple_words_in_list(self):
        count = count_word_matches(["red", "blue"], "the red car and blue sky")
        assert count == 2

    def test_multi_word_phrase(self):
        assert count_word_matches(["in fact"], "in fact it happened") == 1

    def test_multi_word_phrase_no_partial(self):
        # "in fact" should not match "in factory"
        assert count_word_matches(["in fact"], "working in factory") == 0

    def test_word_at_start_of_text(self):
        assert count_word_matches(["hello"], "hello world") == 1

    def test_word_at_end_of_text(self):
        assert count_word_matches(["world"], "hello world") == 1

    def test_counts_all_occurrences(self):
        # "like" appears 3 times
        assert count_word_matches(["like"], "like this and like that and like so") == 3


class TestFindUniqueWordMatches:
    def test_empty_word_list_returns_empty_set(self):
        assert find_unique_word_matches([], "some text") == set()

    def test_empty_text_returns_empty_set(self):
        assert find_unique_word_matches(["when"], "") == set()

    def test_returns_set_not_list(self):
        result = find_unique_word_matches(["when"], "when I was young")
        assert isinstance(result, set)

    def test_deduplicates_repeated_word(self):
        # "when" appears twice but should only appear once in set
        result = find_unique_word_matches(["when"], "when I go when I stay")
        assert result == {"when"}

    def test_finds_multiple_distinct_words(self):
        result = find_unique_word_matches(["red", "blue"], "the red car and blue sky")
        assert result == {"red", "blue"}

    def test_only_matched_words_returned(self):
        result = find_unique_word_matches(
            ["red", "green", "blue"], "a red and blue thing"
        )
        assert result == {"red", "blue"}
        assert "green" not in result

    def test_normalizes_to_lowercase(self):
        result = find_unique_word_matches(["When"], "When I go WHEN I stay")
        assert result == {"when"}

    def test_word_boundary_respected(self):
        result = find_unique_word_matches(["when"], "whenever I think")
        assert result == set()


class TestCountPatternMatches:
    def test_empty_patterns_returns_zero(self):
        assert count_pattern_matches([], "some text") == 0

    def test_empty_text_returns_zero(self):
        assert count_pattern_matches([r"\bwhen\b"], "") == 0

    def test_single_pattern_match(self):
        assert count_pattern_matches([r"\bthe year was\b"], "the year was 1993") == 1

    def test_single_pattern_no_match(self):
        assert (
            count_pattern_matches([r"\bthe year was\b"], "it happened in spring") == 0
        )

    def test_multiple_patterns_summed(self):
        patterns = [r"\bback in\b", r"\bused to\b"]
        text = "back in those days I used to walk"
        assert count_pattern_matches(patterns, text) == 2

    def test_year_pattern(self):
        count = count_pattern_matches([r"\bin \d{4}\b"], "in 1993 everything changed")
        assert count == 1

    def test_case_insensitive(self):
        count = count_pattern_matches([r"\bback in\b"], "Back In those days")
        assert count == 1

    def test_multiple_matches_of_same_pattern(self):
        count = count_pattern_matches([r"\bwhen\b"], "when I go when I stay")
        assert count == 2


class TestCalculateDensity:
    def test_empty_text_returns_zero(self):
        assert calculate_density(5, "") == 0.0

    def test_zero_matches_returns_zero(self):
        assert calculate_density(0, "one two three") == 0.0

    def test_one_match_in_ten_words(self):
        text = "one two three four five six seven eight nine ten"
        result = calculate_density(1, text)
        assert result == pytest.approx(10.0)

    def test_one_match_in_one_word(self):
        result = calculate_density(1, "word")
        assert result == pytest.approx(100.0)

    def test_five_matches_in_hundred_words(self):
        text = " ".join(["word"] * 100)
        result = calculate_density(5, text)
        assert result == pytest.approx(5.0)

    def test_density_can_exceed_100(self):
        # More matches than words is theoretically possible (multi-word phrases counted)
        text = "a b"
        result = calculate_density(3, text)
        assert result == pytest.approx(150.0)


class TestCalculateCoverage:
    def test_empty_word_list_returns_zero(self):
        assert calculate_coverage({"word"}, []) == 0.0

    def test_no_matches_returns_zero(self):
        assert calculate_coverage(set(), ["word1", "word2"]) == 0.0

    def test_full_coverage(self):
        result = calculate_coverage({"word1", "word2"}, ["word1", "word2"])
        assert result == pytest.approx(1.0)

    def test_partial_coverage(self):
        result = calculate_coverage({"word1"}, ["word1", "word2", "word3", "word4"])
        assert result == pytest.approx(0.25)

    def test_half_coverage(self):
        result = calculate_coverage({"a", "b"}, ["a", "b", "c", "d"])
        assert result == pytest.approx(0.5)

    def test_result_bounded_at_one(self):
        # unique_matches can't exceed word_list size in practice, but coverage
        # formula is just len(unique) / len(list), so verify normal case
        result = calculate_coverage({"a", "b", "c"}, ["a", "b", "c"])
        assert result <= 1.0
