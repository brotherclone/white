import pytest
from pydantic import ValidationError
from app.structures.manifests.manifest_song_structure import ManifestSongStructure
from app.structures.music.core.duration import Duration

def test_manifest_song_structure_valid_str_times():
    s = ManifestSongStructure(
        section_name="Verse",
        start_time="00:00.000",
        end_time="00:30.000",
        description="First verse"
    )
    assert s.section_name == "Verse"
    assert s.start_time == "00:00.000"
    assert s.end_time == "00:30.000"
    assert s.description == "First verse"

def test_manifest_song_structure_valid_duration_times():
    s = ManifestSongStructure(
        section_name="Chorus",
        start_time=Duration(minutes=4, seconds=0),
        end_time=Duration(minutes=8, seconds=0),
    )
    assert s.section_name == "Chorus"
    assert isinstance(s.start_time, Duration)
    assert isinstance(s.end_time, Duration)
    assert s.description is None

def test_manifest_song_structure_missing_required():
    with pytest.raises(ValidationError):
        ManifestSongStructure(start_time="00:00.000", end_time="00:30.000")
    with pytest.raises(ValidationError):
        ManifestSongStructure(section_name="Bridge", end_time="00:30.000")
    with pytest.raises(ValidationError):
        ManifestSongStructure(section_name="Bridge", start_time="00:00.000")

def test_manifest_song_structure_invalid_types():
    with pytest.raises(ValidationError):
        ManifestSongStructure(
            section_name=123,
            start_time="00:00.000",
            end_time="00:30.000"
        )
    with pytest.raises(ValidationError):
        ManifestSongStructure(
            section_name="Outro",
            start_time=None,
            end_time="00:30.000"
        )

