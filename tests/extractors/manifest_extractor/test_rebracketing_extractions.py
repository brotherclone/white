"""
Test rebracketing concept extraction functionality.

Tests word boundary matching, temporal scoring, ontological uncertainty,
and the full concept extraction pipeline.
"""

from app.extractors.manifest_extractor.word_matching import count_word_matches
from app.extractors.manifest_extractor.scoring_functions import (
    score_temporal_complexity,
    score_ontological_uncertainty,
)
from app.extractors.manifest_extractor.concept_extractor import ConceptExtractor


def test_word_boundary_matching():
    """Test that word boundary matching works correctly"""
    # Should NOT match "when" in "whenever"
    text = "whenever I think about it"
    count = count_word_matches(["when"], text)
    assert count == 0, f"Expected 0, got {count} - 'when' shouldn't match 'whenever'"

    # Should match "when" as standalone word
    text2 = "when I was young"
    count2 = count_word_matches(["when"], text2)
    assert count2 == 1, f"Expected 1, got {count2} - should match standalone 'when'"


def test_temporal_scoring():
    """Test that temporal scoring catches memory anchors"""
    # High temporal score - memory anchor
    text1 = "the year was '93 in newton where i fell"
    score1 = score_temporal_complexity(text1)
    assert score1 > 0.3, f"Should be high (>0.3), got {score1}"

    # Low temporal score - no temporal markers
    text2 = "the color was different"
    score2 = score_temporal_complexity(text2)
    assert score2 < 0.3, f"Should be low (<0.3), got {score2}"


def test_ontological_uncertainty():
    """Test that ontological uncertainty scoring works"""
    # High uncertainty - both "seemed" and "actually"
    text1 = "it seemed red but was actually orange"
    score1 = score_ontological_uncertainty(text1)
    assert score1 > 0.3, f"Should be high (>0.3), got {score1}"

    # Low uncertainty - no markers
    text2 = "the injury was different"
    score2 = score_ontological_uncertainty(text2)
    assert score2 < 0.2, f"Should be low (<0.2), got {score2}"


def test_full_extraction():
    """Test full concept extraction pipeline"""
    extractor = ConceptExtractor(
        track_id="1_5",
        concept_text="The injury was actually different than I remembered",
        lyric_text="The year was '93 in Newton",
        track_duration=247.3,
        rainbow_color_mnemonic="V",
    )

    # Test methodology features
    features = extractor.get_methodology_features()
    assert features.word_count > 0
    assert features.has_rebracketing_markers

    # Test rebracketing analysis
    analysis = extractor.get_rebracketing_analysis()
    assert analysis.rebracketing_type in [
        "causal",
        "spatial",
        "temporal",
        "perceptual",
        "experiential",
    ]
    assert 0.0 <= analysis.rebracketing_intensity <= 100.0
    assert 0.0 <= analysis.temporal_complexity_score <= 1.0
