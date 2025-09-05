import pytest
from app.structures.music.core.key_signature import Mode, ModeName, KeySignature, get_mode
from app.structures.music.core.notes import Note

def test_mode_valid():
    m = Mode(name=ModeName.MAJOR, intervals=[2,2,1,2,2,2,1])
    assert m.name == ModeName.MAJOR

def test_mode_invalid():
    with pytest.raises(ValueError):
        Mode(name='notamode')

def test_get_mode_valid():
    m = get_mode('major')
    assert isinstance(m, Mode)
    assert m.name == ModeName.MAJOR

def test_key_signature_valid():
    n = Note(pitch_name='C')
    m = Mode(name=ModeName.MINOR, intervals=[2,1,2,2,1,2,2])
    ks = KeySignature(note=n, mode=m)
    assert ks.note.pitch_name == 'C'
    assert ks.mode.name == ModeName.MINOR

