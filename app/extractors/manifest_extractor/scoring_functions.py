"""
Scoring functions for concept analysis.

Each function does ONE thing: takes text, returns a score.
All use proper word boundary matching from word_matching.py
"""

from app.extractors.manifest_extractor.word_matching import (
    count_word_matches,
    find_unique_word_matches,
    count_pattern_matches,
    calculate_density,
    calculate_coverage,
)
from app.reference.rebracketing_words.temporal_words import (
    TEMPORAL_WORDS,
    TEMPORAL_DEIXIS_PATTERNS,
)
from app.reference.rebracketing_words.rebracketing_words import (
    REBRACKETING_WORDS,
    ONTOLOGICAL_UNCERTAINTY,
    REALITY_CORRECTIONS,
)
from app.reference.rebracketing_words.discrepancy_words import DISCREPANCY_WORDS
from app.reference.rebracketing_words.fluidity_words import FLUIDITY_WORDS


def score_rebracketing_intensity(text: str) -> float:
    """
    Core methodology marker density.

    High score = lots of ontological boundary crossing signals.
    Returns: matches per 100 words
    """
    match_count = count_word_matches(REBRACKETING_WORDS, text)
    return calculate_density(match_count, text)


def score_temporal_complexity(text: str) -> float:
    """
    How much temporal boundary work is happening.

    High score = complex temporal framing.
    Returns: 0.0 to 1.0 (normalized)
    """
    # Word matches
    word_matches = count_word_matches(TEMPORAL_WORDS, text)

    # Pattern matches (temporal deixis)
    pattern_matches = count_pattern_matches(TEMPORAL_DEIXIS_PATTERNS, text)

    total_matches = word_matches + (pattern_matches * 2)  # Weight patterns higher
    density = calculate_density(total_matches, text)

    # Normalize to 0-1 (10+ matches per 100 words = 1.0)
    return min(density / 10.0, 1.0)


def score_ontological_uncertainty(text: str) -> float:
    """
    IMAGINED bleeding into REAL.

    High score = unstable boundary between appearance and reality.
    Looks for: uncertainty words PLUS correction words (the tension)

    Returns: 0.0 to 1.0
    """
    uncertainty_count = count_word_matches(ONTOLOGICAL_UNCERTAINTY, text)
    correction_count = count_word_matches(REALITY_CORRECTIONS, text)

    # Both present = high instability
    if uncertainty_count > 0 and correction_count > 0:
        combined_density = calculate_density(uncertainty_count + correction_count, text)
        return min(combined_density / 5.0, 1.0)

    # Only one present = moderate
    elif uncertainty_count > 0 or correction_count > 0:
        density = calculate_density(uncertainty_count + correction_count, text)
        return min(density / 10.0, 0.5)

    # Neither present = no uncertainty
    return 0.0


def score_memory_discrepancy(text: str) -> float:
    """
    Memory revision and correction severity.

    High score = strong signals of memory being corrected/challenged.
    Returns: 0.0 to 1.0
    """
    match_count = count_word_matches(DISCREPANCY_WORDS, text)
    density = calculate_density(match_count, text)

    # Normalize (5+ matches per 100 words = 1.0)
    return min(density / 5.0, 1.0)


def score_boundary_fluidity(text: str) -> float:
    """
    How fluid/fuzzy are the categorical boundaries.

    High score = lots of "maybe", "unclear", "vague" type language.
    Returns: 0.0 to 1.0
    """
    match_count = count_word_matches(FLUIDITY_WORDS, text)
    density = calculate_density(match_count, text)

    # Normalize (3+ matches per 100 words = 1.0)
    return min(density / 3.0, 1.0)


def score_rebracketing_coverage(text: str) -> float:
    """
    What percentage of rebracketing markers appear at least once.

    Different from intensity - this measures breadth not depth.
    High score = many different types of rebracketing present.

    Returns: 0.0 to 1.0
    """
    unique_matches = find_unique_word_matches(REBRACKETING_WORDS, text)
    return calculate_coverage(unique_matches, REBRACKETING_WORDS)


def check_has_rebracketing_markers(text: str) -> bool:
    """
    Simple boolean: does this concept have ANY rebracketing markers?

    Returns: True if any core markers present
    """
    count = count_word_matches(REBRACKETING_WORDS, text)
    return count > 0


def calculate_basic_text_features(text: str) -> dict:
    """
    Simple text statistics that don't need word lists.

    Returns: dict with concept_length, word_count, sentence_count, etc.
    """
    sentences = [s.strip() for s in text.split(".") if s.strip()]
    words = text.split()

    avg_word_length = 0.0
    if len(words) > 0:
        avg_word_length = sum(len(word) for word in words) / len(words)

    return {
        "concept_length": len(text),
        "word_count": len(words),
        "sentence_count": len(sentences),
        "avg_word_length": avg_word_length,
        "question_marks": text.count("?"),
        "exclamation_marks": text.count("!"),
    }
