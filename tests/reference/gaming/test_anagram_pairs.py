"""Tests for anagram validation and pre-made pairs."""

import re

import pytest

from app.agents.tools.encodings.anagram_encodings import AnagramEncoding
from app.reference.gaming.anagram_pairs import ANAGRAM_PAIRS


def is_valid_anagram(a: str, b: str) -> bool:
    """Check if two strings are anagrams (ignoring non-alpha chars)."""
    a_clean = re.sub(r"[^A-Z]", "", a.upper())
    b_clean = re.sub(r"[^A-Z]", "", b.upper())
    return sorted(a_clean) == sorted(b_clean)


class TestAnagramValidation:
    """Test the AnagramEncoding validator."""

    def test_simple_anagram_passes(self):
        """Basic anagram without punctuation should pass."""
        encoding = AnagramEncoding(
            secret_word="listen",
            surface_phrase="silent",
            surface_text="The silent night",
            letter_bank="eilnst",
        )
        assert encoding.surface_phrase == "silent"

    def test_anagram_with_spaces_passes(self):
        """Anagram with spaces should pass."""
        encoding = AnagramEncoding(
            secret_word="dormitory",
            surface_phrase="dirty room",
            surface_text="A dirty room awaits",
            letter_bank="dimorrtoy",
        )
        assert encoding.surface_phrase == "dirty room"

    def test_anagram_with_apostrophe_passes(self):
        """Anagram with apostrophe should pass (the original bug)."""
        encoding = AnagramEncoding(
            secret_word="A Decimal Point",
            surface_phrase="I'm A Dot In Place",
            surface_text="I'm a dot in place, marking time",
            letter_bank="aacdeiiilmnopt",
        )
        assert encoding.surface_phrase == "I'm A Dot In Place"

    def test_anagram_with_punctuation_passes(self):
        """Anagram with various punctuation should pass."""
        encoding = AnagramEncoding(
            secret_word="oh, what a clever boy",
            surface_phrase="yeah, blowtorch ave.",
            surface_text="Yeah, blowtorch ave. is where we meet",
            letter_bank="aabceehhloorttvwwy",
        )
        assert encoding.surface_phrase == "yeah, blowtorch ave."

    def test_invalid_anagram_raises(self):
        """Non-anagram should raise ValueError."""
        with pytest.raises(ValueError, match="is not an anagram"):
            AnagramEncoding(
                secret_word="hello",
                surface_phrase="world",
                surface_text="The world is round",
                letter_bank="dehlorw",
            )

    def test_case_insensitive(self):
        """Validation should be case insensitive."""
        encoding = AnagramEncoding(
            secret_word="SILENT",
            surface_phrase="Listen",
            surface_text="Listen to the sound",
            letter_bank="eilnst",
        )
        assert encoding.surface_phrase == "Listen"


class TestAnagramPairs:
    """Test the pre-made anagram pairs."""

    def test_pairs_list_not_empty(self):
        """Should have pre-made pairs available."""
        assert len(ANAGRAM_PAIRS) > 0

    def test_minimum_pair_count(self):
        """Should have at least 25 pairs for variety."""
        assert len(ANAGRAM_PAIRS) >= 25

    @pytest.mark.parametrize("secret,surface", ANAGRAM_PAIRS)
    def test_all_pairs_are_valid_anagrams(self, secret: str, surface: str):
        """Every pre-made pair should be a valid anagram."""
        assert is_valid_anagram(
            secret, surface
        ), f"'{surface}' is not a valid anagram of '{secret}'"

    @pytest.mark.parametrize("secret,surface", ANAGRAM_PAIRS)
    def test_all_pairs_pass_validator(self, secret: str, surface: str):
        """Every pre-made pair should pass the AnagramEncoding validator."""
        letter_bank = "".join(sorted(re.sub(r"[^a-z]", "", secret.lower())))
        encoding = AnagramEncoding(
            secret_word=secret,
            surface_phrase=surface,
            surface_text=f"The phrase '{surface}' in context",
            letter_bank=letter_bank,
        )
        assert encoding.surface_phrase == surface

    def test_pairs_have_variety(self):
        """Pairs should have varied lengths."""
        lengths = [len(s) for s, _ in ANAGRAM_PAIRS]
        # Should have short (< 10), medium (10-20), and long (> 20) pairs
        assert any(
            anagram_length < 10 for anagram_length in lengths
        ), "Need some short anagrams"
        assert any(
            10 <= anagram_length <= 20 for anagram_length in lengths
        ), "Need some medium anagrams"
        assert any(
            anagram_length > 20 for anagram_length in lengths
        ), "Need some long anagrams"
