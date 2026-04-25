"""Tests for AcrosticEncoding — first-letter acrostic validation."""

import pytest
from white_ideation.agents.tools.encodings.acrostic_encoding import AcrosticEncoding


def _make(secret, lines, surface_text="placeholder surface text"):
    return AcrosticEncoding(
        secret_word=secret,
        lines=lines,
        surface_text=surface_text,
    )


class TestValidAcrostic:
    def test_matching_lengths_accepted(self):
        enc = _make("HI", ["Hello world", "In the sky"])
        assert enc.lines == ["Hello world", "In the sky"]

    def test_single_char_secret(self):
        enc = _make("X", ["Xenon glows"])
        assert len(enc.lines) == 1

    def test_long_secret(self):
        lines = [f"Line {i} starts here" for i in range(8)]
        enc = _make("TEMPORAL", lines)
        assert len(enc.lines) == 8

    def test_reveal_pattern_default(self):
        enc = _make("AB", ["Alpha", "Bravo"])
        assert enc.reveal_pattern == "first_letter"

    def test_method_field_set(self):
        from white_core.enums.infranym_method import InfranymMethod

        enc = _make("AB", ["Alpha", "Bravo"])
        assert enc.method == InfranymMethod.ACROSTIC_LYRICS


class TestValidationErrors:
    def test_too_few_lines_raises(self):
        with pytest.raises(ValueError, match="Acrostic requires"):
            _make("HELLO", ["Only one line"])

    def test_too_many_lines_raises(self):
        with pytest.raises(ValueError, match="Acrostic requires"):
            _make("HI", ["Line1", "Line2", "Line3"])

    def test_empty_secret_zero_lines_ok(self):
        enc = _make("", [])
        assert enc.lines == []

    def test_empty_secret_one_line_raises(self):
        with pytest.raises(ValueError):
            _make("", ["One line"])

    def test_error_message_contains_secret(self):
        with pytest.raises(ValueError, match="HELLO"):
            _make("HELLO", ["Only one line"])
