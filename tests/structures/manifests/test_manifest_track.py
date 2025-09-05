from app.structures.manifests.manifest_track import ManifestTrack


def test_manifest_track_required_fields():
    t = ManifestTrack(id=1, description='Guitar track')
    assert t.id == 1
    assert t.description == 'Guitar track'
    assert t.audio_file is None
    assert t.midi_file is None
    assert t.group is None
    assert t.midi_group_file is None

def test_manifest_track_all_fields():
    t = ManifestTrack(id=2, description='Drums', audio_file='drums.wav', midi_file='drums.mid', group='rhythm', midi_group_file='group.mid')
    assert t.audio_file == 'drums.wav'
    assert t.midi_file == 'drums.mid'
    assert t.group == 'rhythm'
    assert t.midi_group_file == 'group.mid'

def test_manifest_track_missing_required():
    with pytest.raises(ValidationError):
        ManifestTrack(description='No ID')
    with pytest.raises(ValidationError):
        ManifestTrack(id=3)
import pytest
from app.structures.manifests.manifest_song_structure import ManifestSongStructure

def test_manifest_song_structure_required_fields():
    s = ManifestSongStructure(section_name='Verse', start_time='0:00', end_time='0:30')
    assert s.section_name == 'Verse'
    assert s.start_time == '0:00'
    assert s.end_time == '0:30'
    assert s.description is None

def test_manifest_song_structure_with_description():
    s = ManifestSongStructure(section_name='Chorus', start_time='0:30', end_time='1:00', description='Main chorus')
    assert s.description == 'Main chorus'

import pytest
from pydantic import ValidationError

def test_manifest_song_structure_missing_required():
    with pytest.raises(ValidationError):
        ManifestSongStructure(start_time='0:00', end_time='0:30')

