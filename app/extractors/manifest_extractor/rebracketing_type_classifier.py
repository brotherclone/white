"""
Rebracketing Type Classifier

Determines which domain the rebracketing operates in:
- CAUSAL: Cause and effect, injury narratives
- SPATIAL: Place and location, where things are
- PERCEPTUAL: How things appear, look, sound
- EXPERIENTIAL: Touch, taste, smell, bodily sensations
- TEMPORAL: Time, memory, when things happened (default)

Simple scoring: count words from each domain, pick highest.
If tie or no matches, defaults to TEMPORAL.
"""

import random
from app.extractors.manifest_extractor.word_matching import count_word_matches
from app.reference.rebracketing_words.causal_words import CAUSAL_WORDS
from app.reference.rebracketing_words.spatial_words import SPATIAL_WORDS
from app.reference.rebracketing_words.perceptual_words import PERCEPTUAL_WORDS
from app.reference.rebracketing_words.experiential_words import EXPERIENTIAL_WORDS
from app.structures.enums.rebracketing_analysis_type import RebracketingAnalysisType


def classify_by_domain(text: str) -> RebracketingAnalysisType:
    """
    Classify which domain this rebracketing operates in.

    Args:
        text: Normalized concept text (lowercase)

    Returns:
        One of: 'CAUSAL', 'SPATIAL', 'PERCEPTUAL', 'EXPERIENTIAL', 'TEMPORAL'
    """
    scores = {
        RebracketingAnalysisType.CAUSAL: count_word_matches(CAUSAL_WORDS, text),
        RebracketingAnalysisType.SPATIAL: count_word_matches(SPATIAL_WORDS, text),
        RebracketingAnalysisType.PERCEPTUAL: count_word_matches(PERCEPTUAL_WORDS, text),
        RebracketingAnalysisType.EXPERIENTIAL: count_word_matches(
            EXPERIENTIAL_WORDS, text
        ),
    }

    max_score = max(scores.values())

    # No matches in any domain = default to TEMPORAL
    if max_score == 0:
        return RebracketingAnalysisType.TEMPORAL

    # Find all domains with max score
    top_domains = [domain for domain, score in scores.items() if score == max_score]

    # If tie, pick randomly (or could use hierarchy)
    return random.choice(top_domains)


def classify_by_domain_with_confidence(text: str) -> tuple[str, float]:
    """
    Classify domain and return confidence score.

    Args:
        text: Normalized concept text

    Returns:
        (domain_name, confidence_score)
        Confidence = (max_score - second_highest) / max_score
    """
    scores = {
        "CAUSAL": count_word_matches(CAUSAL_WORDS, text),
        "SPATIAL": count_word_matches(SPATIAL_WORDS, text),
        "PERCEPTUAL": count_word_matches(PERCEPTUAL_WORDS, text),
        "EXPERIENTIAL": count_word_matches(EXPERIENTIAL_WORDS, text),
    }

    sorted_scores = sorted(scores.values(), reverse=True)
    max_score = sorted_scores[0]

    if max_score == 0:
        return "TEMPORAL", 0.0

    # Calculate confidence
    second_highest = sorted_scores[1] if len(sorted_scores) > 1 else 0
    confidence = (max_score - second_highest) / max_score if max_score > 0 else 0.0

    # Get domain with max score
    domain = [d for d, s in scores.items() if s == max_score][0]

    return domain, confidence
