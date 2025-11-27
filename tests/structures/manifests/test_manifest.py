import pytest

from app.structures.manifests.manifest import Manifest
from app.structures.music.core.key_signature import KeySignature
from app.structures.music.core.time_signature import TimeSignature


@pytest.fixture
def minimal_manifest_data():
    return {
        "bpm": 120,
        "manifest_id": "01_01",
        "tempo": "4/4",
        "key": "C major",
        "rainbow_color": "red",
        "title": "Test Song",
        "release_date": "2025-01-01",
        "album_sequence": 1,
        "main_audio_file": "main.wav",
        "TRT": "3:30",
        "vocals": True,
        "lyrics": True,
        "sounds_like": [],
        "structure": [],
        "mood": ["happy"],
        "genres": ["rock"],
        "lrc_file": None,
        "concept": "Test concept",
        "audio_tracks": [],
    }


def test_manifest_parses_tempo_and_key(minimal_manifest_data):
    manifest = Manifest(**minimal_manifest_data)
    assert isinstance(manifest.tempo, TimeSignature)
    assert manifest.tempo.numerator == 4
    assert manifest.tempo.denominator == 4
    assert isinstance(manifest.key, KeySignature)
    assert manifest.key.note.pitch_name == "C"
    assert manifest.key.mode.name.value.lower() == "major"


def test_manifest_invalid_tempo(minimal_manifest_data):
    data = minimal_manifest_data.copy()
    data["tempo"] = "not_a_time_sig"
    manifest = Manifest(**data)
    assert isinstance(manifest.tempo, str)


def test_manifest_invalid_key(minimal_manifest_data):
    data = minimal_manifest_data.copy()
    data["key"] = "invalidkey"
    manifest = Manifest(**data)
    # Should fall back to string if invalid, not raise
    assert isinstance(manifest.key, str)


def test_manifest_missing_optional_fields(minimal_manifest_data):
    data = minimal_manifest_data.copy()
    data.pop("lrc_file", None)
    manifest = Manifest(**data)
    assert hasattr(manifest, "lrc_file")
    assert manifest.lrc_file is None
