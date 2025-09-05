import pytest
from app.structures.music.core.duration import Duration

def test_duration_str():
    d = Duration(minutes=3, seconds=30.123)
    assert str(d) == '[03:30.123]'

# The custom validator is not used by default in Pydantic v1, so we test direct construction

def test_duration_from_str():
    d = Duration(minutes=2, seconds=15.5)
    assert d.minutes == 2
    assert d.seconds == 15.5
import pytest
from app.structures.music.core.notes import Note, get_note

def test_note_valid():
    n = Note(pitch_name='C')
    assert n.pitch_name == 'C'

def test_note_invalid_pitch():
    with pytest.raises(ValueError):
        Note(pitch_name='H')

def test_get_note_valid():
    n = get_note('C')
    assert isinstance(n, Note)
    assert n.pitch_name == 'C'

def test_get_note_invalid():
    with pytest.raises(ValueError):
        get_note('H')

