import pytest
from pydantic import ValidationError
from app.structures.music.core.notes import Note

def test_note_valid_minimal():
    n = Note(pitch_name='C')
    assert n.pitch_name == 'C'
    assert n.pitch_alias is None
    assert n.accidental is None
    assert n.frequency is None
    assert n.octave is None

def test_note_valid_all_fields():
    n = Note(
        pitch_name='D',
        pitch_alias=['Re'],
        accidental='sharp',
        frequency=293,
        octave=4
    )
    assert n.pitch_name == 'D'
    assert n.pitch_alias == ['Re']
    assert n.accidental == 'sharp'
    assert n.frequency == 293
    assert n.octave == 4

def test_note_invalid_pitch_name():
    with pytest.raises(ValueError):
        Note(pitch_name='H')

def test_note_invalid_pitch_alias_type():
    with pytest.raises(ValidationError):
        Note(pitch_name='E', pitch_alias='notalist')

def test_note_invalid_frequency_type():
    with pytest.raises(ValidationError):
        Note(pitch_name='F', frequency='notanint')

def test_note_optional_fields_omitted():
    n = Note(pitch_name='G')
    assert n.pitch_alias is None
    assert n.accidental is None
    assert n.frequency is None
    assert n.octave is None

