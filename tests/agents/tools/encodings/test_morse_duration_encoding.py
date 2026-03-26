"""Tests for MorseDurationEncoding — letter-to-Morse-pattern encoding."""

from app.agents.tools.encodings.morse_duration_encoding import MorseDurationEncoding
from app.reference.encodings.morse_code import MORSE_CODE


def _make(secret, **kwargs):
    return MorseDurationEncoding(secret_word=secret, **kwargs)


class TestMorsePattern:
    def test_single_letter_s(self):
        enc = _make("S")
        assert enc.morse_pattern == "..."

    def test_sos(self):
        enc = _make("SOS")
        assert enc.morse_pattern == "... --- ..."

    def test_lowercase_normalised(self):
        enc_upper = _make("SOS")
        enc_lower = _make("sos")
        assert enc_upper.morse_pattern == enc_lower.morse_pattern

    def test_space_maps_to_slash(self):
        # Space is "/" in MORSE_CODE
        enc = _make("S S")
        assert "/" in enc.morse_pattern

    def test_word_joined_by_spaces(self):
        enc = _make("AB")
        a_code = MORSE_CODE["A"]
        b_code = MORSE_CODE["B"]
        assert enc.morse_pattern == f"{a_code} {b_code}"

    def test_all_letters_mappable(self):
        enc = _make("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
        parts = enc.morse_pattern.split(" ")
        assert len(parts) == 26
        assert all(p for p in parts)

    def test_empty_secret_yields_empty_pattern(self):
        enc = _make("")
        assert enc.morse_pattern == ""


class TestUnmappedChars:
    def test_digits_skipped(self, caplog):
        import logging

        with caplog.at_level(logging.WARNING):
            enc = _make("A1B")
        a_code = MORSE_CODE["A"]
        b_code = MORSE_CODE["B"]
        assert enc.morse_pattern == f"{a_code} {b_code}"

    def test_punctuation_skipped_with_warning(self, caplog):
        import logging

        with caplog.at_level(logging.WARNING):
            enc = _make("S!")
        assert enc.morse_pattern == "..."
        assert any("!" in r.message for r in caplog.records)

    def test_only_unmapped_yields_empty(self, caplog):
        import logging

        with caplog.at_level(logging.WARNING):
            enc = _make("123")
        assert enc.morse_pattern == ""


class TestFields:
    def test_default_carrier_note(self):
        enc = _make("A")
        assert enc.carrier_note == 60

    def test_default_dot_duration(self):
        enc = _make("A")
        assert enc.dot_duration == 240

    def test_default_dash_duration(self):
        enc = _make("A")
        assert enc.dash_duration == 720

    def test_custom_carrier_note(self):
        enc = _make("A", carrier_note=48)
        assert enc.carrier_note == 48

    def test_method_field_set(self):
        from app.structures.enums.infranym_method import InfranymMethod

        enc = _make("A")
        assert enc.method == InfranymMethod.MORSE_DURATION

    def test_precomputed_pattern_not_overwritten(self):
        enc = MorseDurationEncoding(secret_word="A", morse_pattern="custom")
        assert enc.morse_pattern == "custom"
