import pytest

from app.util.lrc_validator import LRCValidator


@pytest.fixture
def validator():
    return LRCValidator()


def test_valid_lrc(validator):
    lrc = """[ti:Test Title]
[ar:Test Artist]
[al:Test Album]
[00:01.000]First line
[00:02.000]Second line
[00:03.000]Third line"""
    is_valid, errors = validator.validate(lrc)
    assert is_valid
    assert errors == []


def test_missing_title(validator):
    lrc = """[ar:Test Artist]
[al:Test Album]
[00:01.000]Line"""
    is_valid, errors = validator.validate(lrc)
    assert not is_valid
    assert any("title" in e for e in errors)


def test_missing_artist(validator):
    lrc = """[ti:Test Title]
[al:Test Album]
[00:01.000]Line"""
    is_valid, errors = validator.validate(lrc)
    assert not is_valid
    assert any("artist" in e for e in errors)


def test_missing_album(validator):
    lrc = """[ti:Test Title]
[ar:Test Artist]
[00:01.000]Line"""
    is_valid, errors = validator.validate(lrc)
    assert not is_valid
    assert any("album" in e for e in errors)


def test_no_timestamps(validator):
    lrc = """[ti:Test Title]
[ar:Test Artist]
[al:Test Album]
No timestamps here"""
    is_valid, errors = validator.validate(lrc)
    assert not is_valid
    assert any("No valid timestamp" in e for e in errors)


def test_non_sequential_timestamps(validator):
    lrc = """[ti:Test Title]
[ar:Test Artist]
[al:Test Album]
[00:02.000]Second line
[00:01.000]First line"""
    is_valid, errors = validator.validate(lrc)
    assert not is_valid
    assert any("Non-sequential timestamp" in e for e in errors)


def test_empty_file(validator):
    lrc = ""
    is_valid, errors = validator.validate(lrc)
    assert not is_valid
    assert any("Missing title" in e for e in errors)
    assert any("Missing artist" in e for e in errors)
    assert any("Missing album" in e for e in errors)
    assert any("No valid timestamp" in e for e in errors)


def test_malformed_timestamp(validator):
    lrc = """[ti:Test Title]
[ar:Test Artist]
[al:Test Album]
[bad:timestamp]Bad line"""
    is_valid, errors = validator.validate(lrc)
    assert not is_valid
    assert any("No valid timestamp" in e for e in errors)


def test_smpte_timestamp_detected(validator):
    """SMPTE-like timestamps [MM:SS:FF.f] should be flagged."""
    lrc = """[ti:Test Title]
[ar:Test Artist]
[al:Test Album]
[00:01.000]First line
[03:33:19.3]SMPTE leak
[04:00.000]Next line"""
    is_valid, errors = validator.validate(lrc)
    assert not is_valid
    assert any("SMPTE" in e and "03:33:19.3" in e for e in errors)


def test_colon_before_milliseconds(validator):
    """[MM:SS:mmm] with colon instead of dot should be flagged."""
    lrc = """[ti:Test Title]
[ar:Test Artist]
[al:Test Album]
[00:01.000]First line
[01:04:110]Colon before ms
[02:00.000]Next line"""
    is_valid, errors = validator.validate(lrc)
    assert not is_valid
    assert any("01:04:110" in e for e in errors)


def test_dot_instead_of_colon(validator):
    """[MM.SS.mmm] with dot between min/sec should be flagged."""
    lrc = """[ti:Test Title]
[ar:Test Artist]
[al:Test Album]
[00:01.000]First line
[01.37.025]Dot separator
[02:00.000]Next line"""
    is_valid, errors = validator.validate(lrc)
    assert not is_valid
    assert any("01.37.025" in e for e in errors)


def test_four_digit_milliseconds(validator):
    """[MM:SS.mmmm] with 4-digit fractional should be flagged."""
    lrc = """[ti:Test Title]
[ar:Test Artist]
[al:Test Album]
[00:01.000]First line
[01:03.2047]Too many digits
[02:00.000]Next line"""
    is_valid, errors = validator.validate(lrc)
    assert not is_valid
    assert any("01:03.2047" in e for e in errors)


def test_large_gap_warning(validator):
    """Large gaps between timestamps should produce warnings."""
    lrc = """[ti:Test Title]
[ar:Test Artist]
[al:Test Album]
[00:01.000]First line
[01:00.000]Big gap here"""
    is_valid, errors = validator.validate(lrc)
    # Warnings are appended to errors list but large gap alone doesn't invalidate
    assert any("Large gap" in e for e in errors)


def test_slash_before_milliseconds(validator):
    """[MM:SS/mmm] with slash instead of dot should be flagged."""
    lrc = """[ti:Test Title]
[ar:Test Artist]
[al:Test Album]
[00:01.000]First line
[02:52/071]Slash separator
[03:00.000]Next line"""
    is_valid, errors = validator.validate(lrc)
    assert not is_valid
    assert any("02:52/071" in e for e in errors)


def test_metadata_tags_not_flagged(validator):
    """Known metadata tags should not be flagged as malformed timestamps."""
    lrc = """[ti:Test Title]
[ar:Test Artist]
[al:Test Album]
[by:Some Tool]
[offset:500]
[00:01.000]First line"""
    is_valid, errors = validator.validate(lrc)
    assert is_valid


def test_bom_character_handling(validator):
    """Test that BOM character is stripped properly."""
    lrc = "\ufeff[ti:Test Title]\n[ar:Test Artist]\n[al:Test Album]\n[00:01.000]Line"
    is_valid, errors = validator.validate(lrc)
    assert is_valid
    assert errors == []


def test_multiple_timestamps_same_line(validator):
    """Test handling of multiple timestamps on the same line."""
    lrc = """[ti:Test Title]
[ar:Test Artist]
[al:Test Album]
[00:01.000][00:01.500]First line
[00:02.000]Second line"""
    is_valid, errors = validator.validate(lrc)
    assert is_valid


def test_timestamps_with_different_formats(validator):
    """Test various valid timestamp formats."""
    lrc = """[ti:Test Title]
[ar:Test Artist]
[al:Test Album]
[0:01.0]Single digit minute
[00:02.00]Double digit subsecond
[00:03.123]Triple digit subsecond
[1:04]No subseconds"""
    is_valid, errors = validator.validate(lrc)
    assert is_valid


def test_case_insensitive_metadata(validator):
    """Test that metadata tags are case insensitive."""
    lrc = """[TI:Test Title]
[AR:Test Artist]
[AL:Test Album]
[00:01.000]Line"""
    is_valid, errors = validator.validate(lrc)
    assert is_valid


def test_metadata_with_spaces(validator):
    """Test metadata tags with extra spaces."""
    lrc = """[ti:  Test Title  ]
[ar:  Test Artist  ]
[al:  Test Album  ]
[00:01.000]Line"""
    is_valid, errors = validator.validate(lrc)
    assert is_valid


def test_same_timestamp_multiple_times(validator):
    """Test that duplicate timestamps are allowed (same time)."""
    lrc = """[ti:Test Title]
[ar:Test Artist]
[al:Test Album]
[00:01.000]First line
[00:01.000]Second line at same time
[00:02.000]Third line"""
    is_valid, errors = validator.validate(lrc)
    assert is_valid


def test_non_sequential_different_lines(validator):
    """Test non-sequential timestamps on different lines."""
    lrc = """[ti:Test Title]
[ar:Test Artist]
[al:Test Album]
[00:05.000]Line 1
[00:03.000]Line 2
[00:06.000]Line 3"""
    is_valid, errors = validator.validate(lrc)
    assert not is_valid
    assert any("Non-sequential timestamp" in e for e in errors)


def test_timestamp_at_zero(validator):
    """Test that timestamps starting at 00:00 are valid."""
    lrc = """[ti:Test Title]
[ar:Test Artist]
[al:Test Album]
[00:00.000]First line at zero
[00:01.000]Second line"""
    is_valid, errors = validator.validate(lrc)
    assert is_valid


def test_large_minute_values(validator):
    """Test handling of large minute values."""
    lrc = """[ti:Test Title]
[ar:Test Artist]
[al:Test Album]
[00:01.000]First line
[99:59.999]Last line with large minutes"""
    is_valid, errors = validator.validate(lrc)
    assert is_valid


def test_mixed_valid_invalid_lines(validator):
    """Test file with mix of valid and invalid content."""
    lrc = """[ti:Test Title]
[ar:Test Artist]
[al:Test Album]
Some random text without timestamp
[00:01.000]Valid line
Another line without timestamp
[00:02.000]Another valid line"""
    is_valid, errors = validator.validate(lrc)
    assert is_valid  # Should be valid as long as there are some timestamps


def test_all_metadata_errors(validator):
    """Test that all missing metadata errors are reported."""
    lrc = """Just some content
[00:01.000]A timestamp"""
    is_valid, errors = validator.validate(lrc)
    assert not is_valid
    assert len(errors) == 3  # Missing title, artist, and album


def test_whitespace_lines(validator):
    """Test handling of whitespace and empty lines."""
    lrc = """[ti:Test Title]
[ar:Test Artist]
[al:Test Album]

[00:01.000]First line

[00:02.000]Second line
    """
    is_valid, errors = validator.validate(lrc)
    assert is_valid
