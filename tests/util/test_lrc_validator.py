# moved from /test/util/
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

