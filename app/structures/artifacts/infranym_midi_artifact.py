import logging

from abc import ABC
from typing import Union, Optional, List
from mido import MidiFile, MidiTrack, MetaMessage, Message
from pydantic import Field
from dotenv import load_dotenv

from app.agents.tools.encodings.morse_duration_encoding import MorseDurationEncoding
from app.agents.tools.encodings.note_cipher_encoding import NoteCipherEncoding
from app.structures.artifacts.midi_artifact_file import MidiArtifactFile
from app.structures.enums.chain_artifact_type import ChainArtifactType

MidiEncodingVariant = Union[NoteCipherEncoding, MorseDurationEncoding]

load_dotenv()


class InfranymMidiArtifact(MidiArtifactFile, ABC):
    """
    Complete MIDI infranym with method-specific encoding.

    Generates MIDI files that hide secret words through:
    - Note Cipher (pitch sequence spells word)
    - Morse Duration (rhythm encodes morse code)
    """

    chain_artifact_type: ChainArtifactType = ChainArtifactType.INFRANYM_MIDI

    encoding: MidiEncodingVariant = Field(
        ..., description="The encoding data (type determines method)"
    )
    bpm: int = Field(default=120, description="Base tempo")
    key: Optional[str] = Field(default=None, description="Key signature (for context)")
    time_signature: str = Field(default="4/4", description="Time signature")
    carrier_melody: Optional[List[int]] = Field(
        default=None, description="Main melody notes to camouflage the encoded message"
    )
    include_carrier: bool = Field(
        default=False,
        description="Whether to include carrier melody alongside encoded message",
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
            "bpm": self.bpm,
            "key": self.key,
            "time_signature": self.time_signature,
            "include_carrier": self.include_carrier,
        }

    def for_prompt(self) -> str:
        """Format for LLM context"""
        return (
            f"MIDI Infranym: {self.encoding.method.value} hiding "
            f"'{self.encoding.secret_word}' at {self.bpm} BPM"
        )

    def save_file(self):
        """Generate and save MIDI file with encoded message"""
        output_path = self.get_artifact_path(with_file_name=True, create_dirs=True)

        # Create MIDI file
        mid = MidiFile(type=1)  # Type 1 = multi-track

        # Track 0: Tempo and metadata
        meta_track = MidiTrack()
        mid.tracks.append(meta_track)

        # Set tempo
        tempo = int(60_000_000 / self.bpm)  # Microseconds per beat
        meta_track.append(MetaMessage("set_tempo", tempo=tempo, time=0))

        # Set time signature
        numerator, denominator = map(int, self.time_signature.split("/"))
        meta_track.append(
            MetaMessage(
                "time_signature", numerator=numerator, denominator=denominator, time=0
            )
        )

        # Track name (contains hint!)
        meta_track.append(
            MetaMessage(
                "track_name", name=f"Infranym: {self.encoding.method.value}", time=0
            )
        )

        # Add text metadata with decoding hint
        meta_track.append(
            MetaMessage(
                "text",
                text=f"Cipher: {self.encoding.method.value} | Solution length: {len(self.encoding.secret_word)}",
                time=0,
            )
        )

        # Track 1: Encoded message
        message_track = MidiTrack()
        mid.tracks.append(message_track)
        message_track.append(MetaMessage("track_name", name="Hidden Message", time=0))

        # Encode based on method
        if isinstance(self.encoding, NoteCipherEncoding):
            self._encode_note_cipher(message_track, self.encoding)
        elif isinstance(self.encoding, MorseDurationEncoding):
            self._encode_morse_duration(message_track, self.encoding)

        # Optional: Track 2: Carrier melody (camouflage)
        if self.include_carrier and self.carrier_melody:
            carrier_track = MidiTrack()
            mid.tracks.append(carrier_track)
            carrier_track.append(MetaMessage("track_name", name="Melody", time=0))
            self._add_carrier_melody(carrier_track, self.carrier_melody)

        # Save file
        mid.save(output_path)

        logging.info(f"🎹 MIDI infranym saved: {output_path}")
        logging.info(f"   Method: {self.encoding.method.value}")
        logging.info(f"   Secret: {self.encoding.secret_word}")
        logging.info(f"   Tracks: {len(mid.tracks)}")

        return output_path

    def _encode_note_cipher(self, track: MidiTrack, enc: NoteCipherEncoding):
        """Write secret as note sequence (pitch = letter)"""
        velocities = enc.velocity_pattern or [64] * len(enc.note_sequence)

        # Ensure velocity list matches note count
        while len(velocities) < len(enc.note_sequence):
            velocities.append(64)

        time_delta = 0  # Time since last event
        note_duration = 480  # Quarter note

        for note, velocity in zip(enc.note_sequence, velocities):
            # Note on
            track.append(
                Message("note_on", note=note, velocity=velocity, time=time_delta)
            )

            # Note off
            track.append(Message("note_off", note=note, velocity=0, time=note_duration))

            time_delta = 0  # Delta for next note_on

    def _encode_morse_duration(self, track: MidiTrack, enc: MorseDurationEncoding):
        """Write secret as morse code rhythm"""
        time_delta = 0

        for char in enc.morse_pattern:
            if char == ".":
                # Dot (short note)
                track.append(
                    Message(
                        "note_on", note=enc.carrier_note, velocity=80, time=time_delta
                    )
                )
                track.append(
                    Message(
                        "note_off",
                        note=enc.carrier_note,
                        velocity=0,
                        time=enc.dot_duration,
                    )
                )
                time_delta = enc.letter_gap // 3  # Small gap between symbols

            elif char == "-":
                # Dash (long note)
                track.append(
                    Message(
                        "note_on", note=enc.carrier_note, velocity=80, time=time_delta
                    )
                )
                track.append(
                    Message(
                        "note_off",
                        note=enc.carrier_note,
                        velocity=0,
                        time=enc.dash_duration,
                    )
                )
                time_delta = enc.letter_gap // 3

            elif char == " ":
                # Letter gap (silence)
                time_delta = enc.letter_gap

            elif char == "/":
                # Word gap (longer silence)
                time_delta = enc.word_gap

    def _add_carrier_melody(self, track: MidiTrack, melody_notes: List[int]):
        """Add a carrier melody to camouflage the encoded message"""
        time_delta = 0
        note_duration = 480  # Quarter note

        for note in melody_notes:
            track.append(Message("note_on", note=note, velocity=70, time=time_delta))
            track.append(Message("note_off", note=note, velocity=0, time=note_duration))
            time_delta = 0


if __name__ == "__main__":
    # Import functions for testing (avoids circular import at module level)
    from app.agents.tools.infranym_midi_tools import (
        generate_note_cipher,
        generate_morse_duration,
        add_carrier_melody_to_artifact,
    )

    # Test 1: Note Cipher
    print("\n" + "=" * 70)
    print("TEST 1: NOTE CIPHER")
    print("=" * 70)

    cipher_artifact = generate_note_cipher(
        secret_word="TEMPORAL",
        bpm=95,
        octave_offset=1,  # Shift up one octave
        velocity_variation=True,
    )

    path1 = cipher_artifact.save_file()
    print(f"\n✅ Note cipher saved: {path1}")
    print(f"📄 Preview:\n{cipher_artifact.for_prompt()}")
    print(f"🎵 Note sequence: {cipher_artifact.encoding.note_sequence}")

    # Test 2: Morse Duration
    print("\n" + "=" * 70)
    print("TEST 2: MORSE DURATION")
    print("=" * 70)

    morse_artifact = generate_morse_duration(
        secret_word="SOS", bpm=120, carrier_note=72  # C5
    )

    path2 = morse_artifact.save_file()
    print(f"\n✅ Morse duration saved: {path2}")
    print(f"📄 Preview:\n{morse_artifact.for_prompt()}")
    print(f"📡 Morse pattern: {morse_artifact.encoding.morse_pattern}")

    # Test 3: With Carrier Melody
    print("\n" + "=" * 70)
    print("TEST 3: NOTE CIPHER + CARRIER MELODY")
    print("=" * 70)

    # Simple C major scale as carrier
    carrier = [60, 62, 64, 65, 67, 69, 71, 72]  # C D E F G A B C

    cipher_with_carrier = generate_note_cipher(
        secret_word="MEMORY", bpm=108, octave_offset=0
    )

    cipher_with_carrier = add_carrier_melody_to_artifact(cipher_with_carrier, carrier)

    path3 = cipher_with_carrier.save_file()
    print(f"\n✅ Cipher with carrier saved: {path3}")
    print(f"📄 Preview:\n{cipher_with_carrier.for_prompt()}")
    print(f"🎵 Hidden: {cipher_with_carrier.encoding.note_sequence}")
    print(f"🎵 Carrier: {cipher_with_carrier.carrier_melody}")

    print("\n" + "=" * 70)
    print("ALL TESTS COMPLETE")
    print("=" * 70)
    print("\nImport these MIDI files into Logic Pro to hear the encoded messages!")
