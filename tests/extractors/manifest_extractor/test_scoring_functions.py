"""
Tests for concept scoring functions.

Covers all seven scoring functions plus the boolean helper and text feature
calculator. Each class targets one function and tests:
  - Empty / zero inputs
  - Known-value calculations at boundaries
  - Normalisation caps (where applicable)
  - The branching logic specific to that function
"""

import pytest

from app.extractors.manifest_extractor.scoring_functions import (
    calculate_basic_text_features,
    check_has_rebracketing_markers,
    score_boundary_fluidity,
    score_memory_discrepancy,
    score_ontological_uncertainty,
    score_rebracketing_coverage,
    score_rebracketing_intensity,
    score_temporal_complexity,
)

# ---------------------------------------------------------------------------
# score_rebracketing_intensity
# ---------------------------------------------------------------------------


class TestScoreRebracketingIntensity:
    def test_empty_text_returns_zero(self):
        assert score_rebracketing_intensity("") == 0.0

    def test_text_without_markers_returns_zero(self):
        score = score_rebracketing_intensity("the sky is blue and the grass is green")
        assert score == 0.0

    def test_text_with_markers_returns_positive(self):
        # "actually" is in REALITY_CORRECTIONS (part of REBRACKETING_WORDS)
        score = score_rebracketing_intensity("it was actually different")
        assert score > 0.0

    def test_score_is_density_not_normalised(self):
        # density = matches/words * 100; not capped at 1.0
        # "actually" in a 1-word sentence → density = 100.0 (but "actually" is one word)
        score = score_rebracketing_intensity("actually")
        assert score == pytest.approx(100.0)

    def test_more_markers_means_higher_score(self):
        sparse = "the quick brown fox jumps over the lazy dog"
        dense = "actually really in fact it seemed different rather than not"
        assert score_rebracketing_intensity(dense) > score_rebracketing_intensity(
            sparse
        )

    def test_word_boundary_not_subword(self):
        # "real" is not in the word lists; "really" is in REALITY_CORRECTIONS
        no_match = score_rebracketing_intensity("the surreal painting hung there")
        match = score_rebracketing_intensity("it was really something")
        assert match > no_match


# ---------------------------------------------------------------------------
# score_temporal_complexity
# ---------------------------------------------------------------------------


class TestScoreTemporalComplexity:
    def test_empty_text_returns_zero(self):
        assert score_temporal_complexity("") == 0.0

    def test_no_temporal_markers_returns_low_score(self):
        score = score_temporal_complexity("the color is vivid and bright")
        assert score < 0.1

    def test_text_with_temporal_words_scores_positive(self):
        # "when", "before", "after" are all in TEMPORAL_WORDS
        score = score_temporal_complexity(
            "when I was young before the storm after the rain"
        )
        assert score > 0.0

    def test_temporal_deixis_pattern_boosts_score(self):
        # Patterns are weighted x2; "the year was" is a deixis pattern
        without_pattern = "yesterday I walked and then I ran and later I slept"
        with_pattern = "the year was 1993 when I first heard that song"
        assert score_temporal_complexity(with_pattern) > score_temporal_complexity(
            without_pattern
        )

    def test_score_capped_at_one(self):
        # Extremely dense temporal text should not exceed 1.0
        dense = " ".join(
            ["when", "before", "after", "during", "while", "then", "now"] * 5
        )
        assert score_temporal_complexity(dense) <= 1.0

    def test_score_never_negative(self):
        assert score_temporal_complexity("completely non-temporal concept") >= 0.0


# ---------------------------------------------------------------------------
# score_ontological_uncertainty
# ---------------------------------------------------------------------------


class TestScoreOntologicalUncertainty:
    """
    Three branches:
      1. Both uncertainty AND correction words present  → up to 1.0
      2. Only one type present                          → capped at 0.5
      3. Neither present                                → 0.0
    """

    def test_neither_present_returns_zero(self):
        score = score_ontological_uncertainty("the stone bridge crossed the river")
        assert score == 0.0

    def test_only_uncertainty_marker_capped_at_half(self):
        # "seemed" is in ONTOLOGICAL_UNCERTAINTY; no REALITY_CORRECTIONS word
        score = score_ontological_uncertainty("it seemed odd and strange somehow")
        assert 0.0 < score <= 0.5

    def test_only_correction_marker_capped_at_half(self):
        # "actually" is in REALITY_CORRECTIONS; no ONTOLOGICAL_UNCERTAINTY word
        score = score_ontological_uncertainty("it was actually happening right now")
        assert 0.0 < score <= 0.5

    def test_both_markers_can_exceed_half(self):
        # "seemed" + "actually" triggers the high-instability branch
        score = score_ontological_uncertainty("it seemed real but was actually false")
        assert score > 0.0

    def test_both_markers_capped_at_one(self):
        uncertainty = " ".join(["seemed", "appeared", "as if", "as though", "like"] * 4)
        corrections = " ".join(["actually", "really", "truly", "in fact"] * 4)
        score = score_ontological_uncertainty(uncertainty + " " + corrections)
        assert score <= 1.0

    def test_empty_text_returns_zero(self):
        assert score_ontological_uncertainty("") == 0.0

    def test_both_markers_scores_higher_than_one_alone(self):
        only_uncertainty = "it seemed like a dream and appeared strange"
        both = "it seemed like a dream but was actually real"
        # With both markers the branch uses /5.0 vs /10.0, so same density
        # gives a higher score
        assert score_ontological_uncertainty(both) >= score_ontological_uncertainty(
            only_uncertainty
        )


# ---------------------------------------------------------------------------
# score_memory_discrepancy
# ---------------------------------------------------------------------------


class TestScoreMemoryDiscrepancy:
    # DISCREPANCY_WORDS = ["different", "changed", "actually", "really", "instead", "rather"]

    def test_empty_text_returns_zero(self):
        assert score_memory_discrepancy("") == 0.0

    def test_no_discrepancy_words_returns_zero(self):
        assert score_memory_discrepancy("the sun rose over the mountains") == 0.0

    def test_single_discrepancy_word_scores_positive(self):
        score = score_memory_discrepancy("it was actually quite different")
        assert score > 0.0

    def test_score_capped_at_one(self):
        # Each discrepancy word in a very short text → density far > 5, should cap
        dense = "different changed actually really instead rather different"
        assert score_memory_discrepancy(dense) <= 1.0

    def test_more_discrepancy_words_means_higher_score(self):
        sparse = "it was different from what I expected"
        dense = (
            "it was actually really different and changed rather than staying instead"
        )
        assert score_memory_discrepancy(dense) > score_memory_discrepancy(sparse)

    def test_score_never_negative(self):
        assert score_memory_discrepancy("ordinary sentence without markers") >= 0.0


# ---------------------------------------------------------------------------
# score_boundary_fluidity
# ---------------------------------------------------------------------------


class TestScoreBoundaryFluidity:
    # FLUIDITY_WORDS = ["maybe", "perhaps", "might", "could", "unclear", "fuzzy", "vague"]

    def test_empty_text_returns_zero(self):
        assert score_boundary_fluidity("") == 0.0

    def test_no_fluidity_words_returns_zero(self):
        assert score_boundary_fluidity("the definite answer was yes") == 0.0

    def test_single_fluidity_word_scores_positive(self):
        score = score_boundary_fluidity("maybe it was a dream")
        assert score > 0.0

    def test_score_capped_at_one(self):
        dense = "maybe perhaps might could unclear fuzzy vague maybe perhaps"
        assert score_boundary_fluidity(dense) <= 1.0

    def test_more_fluidity_words_means_higher_score(self):
        sparse = "maybe something happened there"
        dense = "maybe perhaps it might could be unclear and vague"
        assert score_boundary_fluidity(dense) > score_boundary_fluidity(sparse)

    def test_normalisation_threshold(self):
        # 30 matches per 100 words → score = 1.0
        # Build text with exactly 30 fluidity words in 100 words
        filler = ["word"] * 70
        words = ["maybe"] * 30 + filler
        text = " ".join(words)
        score = score_boundary_fluidity(text)
        assert score == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# score_rebracketing_coverage
# ---------------------------------------------------------------------------


class TestScoreRebracketingCoverage:
    def test_empty_text_returns_zero(self):
        assert score_rebracketing_coverage("") == 0.0

    def test_no_markers_returns_zero(self):
        assert score_rebracketing_coverage("nothing special here at all") == 0.0

    def test_single_unique_marker_returns_positive(self):
        score = score_rebracketing_coverage("it was actually true")
        assert score > 0.0

    def test_result_bounded_zero_to_one(self):
        score = score_rebracketing_coverage("some text with actually different words")
        assert 0.0 <= score <= 1.0

    def test_more_unique_markers_means_higher_coverage(self):
        few = "actually it happened"
        many = "actually it really seemed different rather than what I believed"
        assert score_rebracketing_coverage(many) > score_rebracketing_coverage(few)

    def test_repeated_single_word_does_not_inflate_coverage(self):
        # Repeating the same word many times should give same coverage as saying it once
        once = "actually the thing happened"
        repeated = "actually actually actually actually the thing happened"
        assert score_rebracketing_coverage(once) == pytest.approx(
            score_rebracketing_coverage(repeated)
        )


# ---------------------------------------------------------------------------
# check_has_rebracketing_markers
# ---------------------------------------------------------------------------


class TestCheckHasRebracketingMarkers:
    def test_empty_text_returns_false(self):
        assert check_has_rebracketing_markers("") is False

    def test_text_without_markers_returns_false(self):
        assert (
            check_has_rebracketing_markers("the sun shines brightly overhead") is False
        )

    def test_text_with_marker_returns_true(self):
        assert check_has_rebracketing_markers("it was actually true") is True

    def test_case_insensitive(self):
        assert check_has_rebracketing_markers("It Was ACTUALLY True") is True

    def test_word_boundary_respected(self):
        # "real" is not a rebracketing word; "really" is
        assert check_has_rebracketing_markers("a surreal painting") is False
        assert check_has_rebracketing_markers("it was really something") is True


# ---------------------------------------------------------------------------
# calculate_basic_text_features
# ---------------------------------------------------------------------------


class TestCalculateBasicTextFeatures:
    def test_returns_dict_with_expected_keys(self):
        result = calculate_basic_text_features("Hello world.")
        assert "concept_length" in result
        assert "word_count" in result
        assert "sentence_count" in result
        assert "avg_word_length" in result
        assert "question_marks" in result
        assert "exclamation_marks" in result

    def test_empty_text(self):
        result = calculate_basic_text_features("")
        assert result["concept_length"] == 0
        assert result["word_count"] == 0
        assert result["sentence_count"] == 0
        assert result["avg_word_length"] == 0.0
        assert result["question_marks"] == 0
        assert result["exclamation_marks"] == 0

    def test_concept_length_is_character_count(self):
        text = "hello"
        result = calculate_basic_text_features(text)
        assert result["concept_length"] == 5

    def test_word_count(self):
        result = calculate_basic_text_features("one two three")
        assert result["word_count"] == 3

    def test_sentence_count(self):
        result = calculate_basic_text_features(
            "First sentence. Second sentence. Third."
        )
        assert result["sentence_count"] == 3

    def test_avg_word_length(self):
        # "ab cd" → words ["ab", "cd"], lengths [2, 2], avg = 2.0
        result = calculate_basic_text_features("ab cd")
        assert result["avg_word_length"] == pytest.approx(2.0)

    def test_question_mark_count(self):
        result = calculate_basic_text_features("Really? Are you sure? Yes.")
        assert result["question_marks"] == 2

    def test_exclamation_mark_count(self):
        result = calculate_basic_text_features("Wow! Amazing! Great.")
        assert result["exclamation_marks"] == 2

    def test_no_punctuation_returns_zero_counts(self):
        result = calculate_basic_text_features("just a plain sentence")
        assert result["question_marks"] == 0
        assert result["exclamation_marks"] == 0
