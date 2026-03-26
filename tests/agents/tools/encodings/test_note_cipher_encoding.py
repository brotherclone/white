"""Tests for NoteCipherEncoding — letter-to-MIDI-pitch mapping."""

from app.agents.tools.encodings.note_cipher_encoding import NoteCipherEncoding
from app.reference.encodings.letter_to_midi import LETTER_TO_MIDI_MAP


def _make(secret, octave_offset=0, **kwargs):
    return NoteCipherEncoding(secret_word=secret, octave_offset=octave_offset, **kwargs)


class TestBasicEncoding:
    def test_single_letter(self):
        enc = _make("A")
        assert enc.note_sequence == [60]

    def test_known_word(self):
        # T=79 E=64 M=72 P=75 O=74 R=77 A=60 L=71
        enc = _make("TEMPORAL")
        assert enc.note_sequence == [79, 64, 72, 75, 74, 77, 60, 71]

    def test_lowercase_normalised(self):
        enc_upper = _make("SOS")
        enc_lower = _make("sos")
        assert enc_upper.note_sequence == enc_lower.note_sequence

    def test_space_maps_to_59(self):
        enc = _make("A B")
        assert 59 in enc.note_sequence

    def test_length_matches_mappable_chars(self):
        secret = "HELLO"
        enc = _make(secret)
        expected_len = sum(1 for c in secret.upper() if c in LETTER_TO_MIDI_MAP)
        assert len(enc.note_sequence) == expected_len

    def test_unmapped_chars_skipped(self):
        # Digits and punctuation are not in the map
        enc = _make("A1B!")
        assert enc.note_sequence == [60, 61]  # A=60, B=61; 1 and ! skipped


class TestOctaveOffset:
    def test_positive_offset_shifts_up(self):
        enc = _make("A", octave_offset=1)
        assert enc.note_sequence == [72]  # 60 + 12

    def test_negative_offset_shifts_down(self):
        enc = _make("A", octave_offset=-1)
        assert enc.note_sequence == [48]  # 60 - 12

    def test_clamp_upper_bound(self):
        # Z=85, offset +4 → 85+48=133 → clamped to 127
        enc = _make("Z", octave_offset=4)
        assert enc.note_sequence == [127]

    def test_clamp_lower_bound(self):
        # A=60, offset -6 → 60-72=-12 → clamped to 0
        enc = _make("A", octave_offset=-6)
        assert enc.note_sequence == [0]

    def test_zero_offset_unchanged(self):
        enc_no_offset = _make("HELLO")
        enc_zero = _make("HELLO", octave_offset=0)
        assert enc_no_offset.note_sequence == enc_zero.note_sequence


class TestEdgeCases:
    def test_empty_secret_yields_empty_sequence(self):
        enc = _make("")
        assert enc.note_sequence == []

    def test_encoding_map_populated_automatically(self):
        enc = _make("A")
        assert enc.encoding_map == LETTER_TO_MIDI_MAP

    def test_method_field_set(self):
        enc = _make("A")
        from app.structures.enums.infranym_method import InfranymMethod

        assert enc.method == InfranymMethod.NOTE_CIPHER

    def test_precomputed_note_sequence_not_recomputed(self):
        # Passing note_sequence explicitly should use that value
        enc = NoteCipherEncoding(
            secret_word="A",
            note_sequence=[99],
        )
        assert enc.note_sequence == [99]
