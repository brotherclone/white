import os
import random
from typing import List

from app.agents.tools.encodings.morse_duration_encoding import MorseDurationEncoding
from app.agents.tools.encodings.note_cipher_encoding import NoteCipherEncoding
from app.structures.artifacts.infranym_midi_artifact import InfranymMidiArtifact


def generate_note_cipher(
    secret_word: str,
    bpm: int = 120,
    octave_offset: int = 0,
    velocity_variation: bool = True,
) -> InfranymMidiArtifact:
    """
    Generate note cipher infranym.

    Args:
        secret_word: Word to encode
        bpm: Tempo
        octave_offset: Shift notes up/down (-2 to +2)
        velocity_variation: Randomize velocities for naturalness

    Returns:
        Complete MIDI artifact with note cipher encoding
    """
    encoding = NoteCipherEncoding(secret_word=secret_word, octave_offset=octave_offset)

    # Optional: Add velocity variation
    if velocity_variation:
        encoding.velocity_pattern = [
            random.randint(55, 85) for _ in encoding.note_sequence
        ]

    artifact = InfranymMidiArtifact(
        base_path=os.getenv("AGENT_WORK_PRODUCT_BASE_PATH", "/tmp"),
        thread_id=f"midi_cipher_{random.randint(1000, 9999)}",
        chain_artifact_file_type="midi",
        chain_artifact_type="infranym_midi",
        rainbow_color_mnemonic_character_value="I",
        artifact_name=f"cipher_{secret_word.lower()[:8]}",
        encoding=encoding,
        bpm=bpm,
        time_signature="4/4",
    )

    return artifact


def generate_morse_duration(
    secret_word: str, bpm: int = 120, carrier_note: int = 60
) -> InfranymMidiArtifact:
    """
    Generate morse duration infranym.

    Args:
        secret_word: Word to encode
        bpm: Tempo
        carrier_note: MIDI note for morse beeps

    Returns:
        Complete MIDI artifact with morse duration encoding
    """
    encoding = MorseDurationEncoding(secret_word=secret_word, carrier_note=carrier_note)

    artifact = InfranymMidiArtifact(
        base_path=os.getenv("AGENT_WORK_PRODUCT_BASE_PATH", "/tmp"),
        thread_id=f"midi_morse_{random.randint(1000, 9999)}",
        chain_artifact_file_type="midi",
        chain_artifact_type="infranym_midi",
        rainbow_color_mnemonic_character_value="I",
        artifact_name=f"morse_{secret_word.lower()[:8]}",
        encoding=encoding,
        bpm=bpm,
        time_signature="4/4",
    )

    return artifact


def add_carrier_melody_to_artifact(
    artifact: InfranymMidiArtifact, melody: List[int]
) -> InfranymMidiArtifact:
    """
    Add a carrier melody to an existing artifact.

    Args:
        artifact: MIDI artifact
        melody: List of MIDI notes for melody

    Returns:
        Modified artifact with carrier enabled
    """
    artifact.carrier_melody = melody
    artifact.include_carrier = True
    return artifact
