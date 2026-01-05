from typing import List

from app.agents.tools.encodings.acrostic_encoding import AcrosticEncoding
from app.agents.tools.encodings.anagram_encodings import AnagramEncoding
from app.agents.tools.encodings.riddle_encoding import RiddleEncoding


def create_acrostic_encoding(
    secret_word: str, generated_lines: List[str]
) -> AcrosticEncoding:
    """
    Create acrostic encoding from pre-generated lines.

    Args:
        secret_word: Word to encode
        generated_lines: LLM-generated lines (from agent node)

    Returns:
        AcrosticEncoding with validated structure
    """
    # Join lines for surface text
    surface_text = "\n".join(generated_lines)

    return AcrosticEncoding(
        secret_word=secret_word,
        surface_text=surface_text,
        lines=generated_lines,
        reveal_pattern="first_letter",
    )


def create_riddle_encoding(
    secret_word: str, generated_riddle: str, difficulty: str = "medium"
) -> RiddleEncoding:
    """
    Create riddle encoding from pre-generated riddle text.

    Args:
        secret_word: Word that's the answer
        generated_riddle: LLM-generated riddle (from agent node)
        difficulty: easy/medium/hard

    Returns:
        RiddleEncoding with validated structure
    """
    # Extract individual lines as clues
    clue_lines = [line.strip() for line in generated_riddle.split("\n") if line.strip()]

    return RiddleEncoding(
        secret_word=secret_word,
        surface_text=generated_riddle,
        riddle_text=generated_riddle,
        clue_lines=clue_lines,
        difficulty=difficulty,
    )


def create_anagram_encoding(secret_word: str, surface_phrase: str) -> AnagramEncoding:
    """
    Create anagram encoding from pre-validated phrases.

    Args:
        secret_word: The hidden phrase
        surface_phrase: The visible anagram

    Returns:
        AnagramEncoding

    Raises:
        ValueError: If phrases are not valid anagrams
    """
    # Validate anagram
    secret_clean = secret_word.upper().replace(" ", "")
    surface_clean = surface_phrase.upper().replace(" ", "")

    if sorted(secret_clean) != sorted(surface_clean):
        raise ValueError(
            f"'{surface_phrase}' is not a valid anagram of '{secret_word}'"
        )

    letter_bank = "".join(sorted(secret_clean))

    return AnagramEncoding(
        secret_word=secret_word,
        surface_text=f"[Repeat as refrain: '{surface_phrase}']",
        surface_phrase=surface_phrase,
        letter_bank=letter_bank,
        usage_instruction="Emphasize through repetition, isolation, or vocal effect",
    )
