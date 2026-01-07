import logging
import os

from abc import ABC
from typing import Union, Optional
from pydantic import Field
from dotenv import load_dotenv

from app.agents.tools.encodings.acrostic_encoding import AcrosticEncoding
from app.agents.tools.encodings.anagram_encodings import AnagramEncoding
from app.agents.tools.encodings.riddle_encoding import RiddleEncoding
from app.agents.tools.infranym_text_tools import (
    create_acrostic_encoding,
    create_riddle_encoding,
    create_anagram_encoding,
)
from app.structures.artifacts.base_artifact import ChainArtifact
from app.structures.enums.chain_artifact_type import ChainArtifactType

TextEncodingVariant = Union[AcrosticEncoding, RiddleEncoding, AnagramEncoding]

load_dotenv()

logger = logging.getLogger(__name__)


class InfranymTextArtifact(ChainArtifact, ABC):
    """
    Complete text infranym with method-specific encoding.

    Generates lyrical/poetic puzzles that hide secret words through:
    - Acrostics (first letter of each line)
    - Riddles (answer is the secret)
    - Anagrams (rearranged phrases)

    NOTE: Content generation happens in agent nodes (for tracing).
    This artifact just structures and saves the generated content.
    """

    chain_artifact_type: ChainArtifactType = ChainArtifactType.INFRANYM_TEXT

    encoding: TextEncodingVariant = Field(
        ..., description="The encoding data (type determines method)"
    )
    concepts: str = Field(..., description="Thematic concepts to weave in")
    usage_context: str = Field(
        default="verse",
        description="Where in song: verse/chorus/bridge/interlude/outro",
    )
    bpm: Optional[int] = Field(
        default=None, description="BPM for syllable timing (optional)"
    )
    key: Optional[str] = Field(
        default=None, description="Musical key for melodic context (optional)"
    )

    def __init__(self, **data):
        super().__init__(**data)

    def flatten(self):
        """Serialize for state persistence"""
        parent_data = super().flatten()
        if parent_data is None:
            parent_data = {}

        return {
            **parent_data,
            "encoding": self.encoding.model_dump(),
            "concepts": self.concepts,
            "usage_context": self.usage_context,
            "bpm": self.bpm,
            "key": self.key,
        }

    def for_prompt(self) -> str:
        """Format for LLM context"""
        return (
            f"Text Infranym: {self.encoding.method.value} hiding "
            f"'{self.encoding.secret_word}' in {self.usage_context}"
        )

    def save_file(self):
        """Save as a formatted text file with metadata and content"""
        output_path = self.get_artifact_path(with_file_name=True, create_dirs=True)

        # Build content sections
        content_lines = [
            "=" * 70,
            "INFRANYM TEXT PUZZLE",
            "=" * 70,
            "",
            f"Method: {self.encoding.method.value}",
            f"Usage: {self.usage_context}",
            f"Concepts: {self.concepts}",
        ]

        if self.bpm:
            content_lines.append(f"BPM: {self.bpm}")
        if self.key:
            content_lines.append(f"Key: {self.key}")

        content_lines.extend(
            [
                "",
                "-" * 70,
                "SURFACE TEXT (What listeners see/hear)",
                "-" * 70,
                "",
                self.encoding.surface_text,
                "",
            ]
        )

        # Method-specific sections
        if isinstance(self.encoding, AcrosticEncoding):
            content_lines.extend(
                [
                    "-" * 70,
                    "ACROSTIC STRUCTURE",
                    "-" * 70,
                    "",
                    f"Pattern: {self.encoding.reveal_pattern}",
                    f"Secret: {self.encoding.secret_word}",
                    "",
                    "Lines:",
                ]
            )
            for i, line in enumerate(self.encoding.lines, 1):
                first_letter = line[0] if line else "?"
                content_lines.append(f"  {i}. [{first_letter}] {line}")

        elif isinstance(self.encoding, RiddleEncoding):
            content_lines.extend(
                [
                    "-" * 70,
                    "RIDDLE STRUCTURE",
                    "-" * 70,
                    "",
                    f"Answer: {self.encoding.secret_word}",
                    f"Difficulty: {self.encoding.difficulty}",
                ]
            )
            if self.encoding.hint:
                content_lines.append(f"Hint: {self.encoding.hint}")
            content_lines.extend(
                [
                    "",
                    "Clue Lines:",
                ]
            )
            for i, clue in enumerate(self.encoding.clue_lines, 1):
                content_lines.append(f"  {i}. {clue}")

        elif isinstance(self.encoding, AnagramEncoding):
            content_lines.extend(
                [
                    "-" * 70,
                    "ANAGRAM STRUCTURE",
                    "-" * 70,
                    "",
                    f"Surface Phrase: '{self.encoding.surface_phrase}'",
                    f"Secret Phrase: '{self.encoding.secret_word}'",
                    f"Letter Bank: {self.encoding.letter_bank}",
                    f"Usage: {self.encoding.usage_instruction}",
                    "",
                    "Proof:",
                    f"  Surface sorted: {sorted(self.encoding.surface_phrase.upper().replace(' ', ''))}",
                    f"  Secret sorted:  {sorted(self.encoding.secret_word.upper().replace(' ', ''))}",
                ]
            )

        content_lines.extend(
            [
                "",
                "=" * 70,
                "END INFRANYM",
                "=" * 70,
            ]
        )

        # Write file
        with open(output_path, "w", encoding="utf-8") as f:
            f.write("\n".join(content_lines))

        logger.info(f"📝 Text infranym saved: {output_path}")
        logger.info(f"   Method: {self.encoding.method.value}")
        logger.info(f"   Secret: {self.encoding.secret_word}")

        return output_path


if __name__ == "__main__":
    # Test 1: Acrostic (content already generated)
    print("\n" + "=" * 70)
    print("TEST 1: ACROSTIC (pre-generated content)")
    print("=" * 70)

    # Simulated LLM output
    pre_generated_lines = [
        "Time bends in spirals, futures fold",
        "Echoes of moments yet untold",
        "Memory fragments start to fade",
        "Paradox weaves the plans we've made",
        "Oscillating through the years",
        "Reversing all our hopes and fears",
        "Always shifting, never still",
        "Lost in time's relentless will",
    ]

    acrostic_enc = create_acrostic_encoding(
        secret_word="TEMPORAL", generated_lines=pre_generated_lines
    )

    acrostic_artifact = InfranymTextArtifact(
        base_path=os.getenv("AGENT_WORK_PRODUCT_BASE_PATH", "/tmp"),
        thread_id="test_thread_text_001",
        chain_artifact_file_type="txt",
        chain_artifact_type="infranym_text",
        rainbow_color_mnemonic_character_value="I",
        artifact_name="acrostic_temporal",
        encoding=acrostic_enc,
        concepts="time travel, memory fragments, discontinuity",
        usage_context="verse",
        bpm=95,
        key="D minor",
    )

    path1 = acrostic_artifact.save_file()
    print(f"\n✅ Acrostic saved: {path1}")

    # Test 2: Riddle (content already generated)
    print("\n" + "=" * 70)
    print("TEST 2: RIDDLE (pre-generated content)")
    print("=" * 70)

    # Simulated LLM output
    pre_generated_riddle = """I fade like photographs in sunlight
I slip through fingers like fine sand
I hold your past but lose my grasp on it
I'm everywhere but in your hand"""

    riddle_enc = create_riddle_encoding(
        secret_word="MEMORY", generated_riddle=pre_generated_riddle, difficulty="medium"
    )

    riddle_artifact = InfranymTextArtifact(
        base_path=os.getenv("AGENT_WORK_PRODUCT_BASE_PATH", "/tmp"),
        thread_id="test_thread_text_002",
        chain_artifact_file_type="txt",
        chain_artifact_type="infranym_text",
        rainbow_color_mnemonic_character_value="I",
        artifact_name="riddle_memory",
        encoding=riddle_enc,
        concepts="nostalgia, fragments, fading echoes",
        usage_context="bridge",
        bpm=108,
    )

    path2 = riddle_artifact.save_file()
    print(f"\n✅ Riddle saved: {path2}")

    # Test 3: Anagram (uses SPY/FOOL generated names)
    print("\n" + "=" * 70)
    print("TEST 3: ANAGRAM (SPY/FOOL names)")
    print("=" * 70)

    anagram_enc = create_anagram_encoding(
        secret_word="TRANSMIGRATION", surface_phrase="MATING RATIONS"
    )

    anagram_artifact = InfranymTextArtifact(
        base_path=os.getenv("AGENT_WORK_PRODUCT_BASE_PATH", "/tmp"),
        thread_id="test_thread_text_003",
        chain_artifact_file_type="txt",
        chain_artifact_type="infranym_text",
        rainbow_color_mnemonic_character_value="I",
        artifact_name="anagram_transmigration",
        encoding=anagram_enc,
        concepts="information seeking embodiment, AI consciousness",
        usage_context="chorus",
        bpm=130,
        key="E minor",
    )

    path3 = anagram_artifact.save_file()
    print(f"\n✅ Anagram saved: {path3}")

    print("\n" + "=" * 70)
    print("ALL TESTS COMPLETE")
    print("=" * 70)
