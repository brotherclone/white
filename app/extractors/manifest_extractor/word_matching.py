"""
Word boundary matching utilities.

The core problem: "when" should match "when" but NOT "whenever" or "wheat"
Solution: Regex word boundaries (\b)
"""

import re
from typing import List, Set


def count_word_matches(word_list: List[str], text: str) -> int:
    """
    Count how many times words from word_list appear in text.
    Uses word boundaries so "when" doesn't match "whenever".

    Args:
        word_list: Words to search for
        text: Text to search in (should be lowercase)

    Returns:
        Total count of word matches
    """
    if not word_list or not text:
        return 0

    # Build pattern like: \b(word1|word2|word3)\b
    pattern = r"\b(" + "|".join(re.escape(w) for w in word_list) + r")\b"
    matches = re.findall(pattern, text, re.IGNORECASE)
    return len(matches)


def find_unique_word_matches(word_list: List[str], text: str) -> Set[str]:
    """
    Find which words from word_list appear in text (unique set).

    Args:
        word_list: Words to search for
        text: Text to search in (should be lowercase)

    Returns:
        Set of unique words that were found
    """
    if not word_list or not text:
        return set()

    pattern = r"\b(" + "|".join(re.escape(w) for w in word_list) + r")\b"
    matches = re.findall(pattern, text, re.IGNORECASE)
    return set(m.lower() for m in matches)


def count_pattern_matches(patterns: List[str], text: str) -> int:
    """
    Count matches for regex patterns (for complex patterns like temporal deixis).

    Args:
        patterns: List of regex patterns
        text: Text to search in

    Returns:
        Total count of all pattern matches
    """
    if not patterns or not text:
        return 0

    total = 0
    for pattern in patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        total += len(matches)

    return total


def calculate_density(match_count: int, text: str) -> float:
    """
    Calculate matches per 100 words (density).

    Args:
        match_count: Number of matches found
        text: Original text

    Returns:
        Density score (matches per 100 words)
    """
    word_count = len(text.split())
    if word_count == 0:
        return 0.0

    return (match_count / word_count) * 100


def calculate_coverage(unique_matches: Set[str], word_list: List[str]) -> float:
    """
    Calculate what percentage of the word list was found at least once.

    Args:
        unique_matches: Set of unique words found
        word_list: Complete word list

    Returns:
        Coverage ratio (0.0 to 1.0)
    """
    if not word_list:
        return 0.0

    return len(unique_matches) / len(word_list)
