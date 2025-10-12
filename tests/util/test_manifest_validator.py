import os
import yaml
import glob
import tempfile
import pytest
from types import SimpleNamespace
from app.util import manifest_validator as mv
from unittest.mock import patch, mock_open
from app.util.manifest_loader import load_manifest

def test_validate_discogs_ids_no_sounds_like():
    data = {'title': 'Test'}
    is_valid, errors = mv.validate_discogs_ids(data)
    assert is_valid
    assert errors == []

def test_validate_discogs_ids_not_list():
    data = {'sounds_like': 'notalist'}
    is_valid, errors = mv.validate_discogs_ids(data)
    assert not is_valid
    assert 'sounds_like section is not a list' in errors[0]

@patch('app.util.manifest_validator.discogs_client.Client')
@patch('app.util.manifest_validator.load_dotenv')
@patch('os.environ.get', return_value='dummy_token')
def test_validate_discogs_ids_valid(mock_env, mock_dotenv, mock_client):
    mock_instance = mock_client.return_value
    mock_artist = type('MockArtist', (), {'name': 'Artist'})()
    mock_instance.artist.return_value = mock_artist
    data = {'sounds_like': [{'name': 'Artist', 'discogs_id': 12345}]}
    is_valid, errors = mv.validate_discogs_ids(data)
    assert is_valid
    assert errors == []

@patch('app.util.manifest_validator.discogs_client.Client')
@patch('app.util.manifest_validator.load_dotenv')
@patch('os.environ.get', return_value='dummy_token')
def test_validate_discogs_ids_missing_fields(mock_env, mock_dotenv, mock_client):
    data = {'sounds_like': [{'name': 'Artist'}]}
    is_valid, errors = mv.validate_discogs_ids(data)
    assert not is_valid
    assert any("missing required 'name' or 'discogs_id'" in e for e in errors)

@patch('app.util.manifest_validator.discogs_client.Client')
@patch('app.util.manifest_validator.load_dotenv')
@patch('os.environ.get', return_value='dummy_token')
def test_validate_discogs_ids_not_dict(mock_env, mock_dotenv, mock_client):
    data = {'sounds_like': ['notadict']}
    is_valid, errors = mv.validate_discogs_ids(data)
    assert not is_valid
    assert any('is not a dictionary' in e for e in errors)

@patch('app.util.manifest_loader.Manifest')
@patch('builtins.open', new_callable=mock_open, read_data='title: Test Song\nartist: Test Artist')
def test_load_manifest_success(mock_file, mock_manifest):
    mock_manifest.return_value = 'manifest_obj'
    result = load_manifest('dummy_path.yml')
    mock_file.assert_called_once_with('dummy_path.yml', 'r')
    mock_manifest.assert_called_once_with(title='Test Song', artist='Test Artist')
    assert result == 'manifest_obj'

@patch('builtins.open', side_effect=FileNotFoundError)
def test_load_manifest_file_not_found(mock_file):
    with pytest.raises(FileNotFoundError):
        load_manifest('missing.yml')

def test_timestamp_to_ms():
    assert mv.timestamp_to_ms("[00:00.000]") == 0
    assert mv.timestamp_to_ms("[00:01.500]") == 1500
    assert mv.timestamp_to_ms("[02:03.250]") == 123250


def test_validate_timestamp_format_valid_and_invalid():
    data_good = {"structure": [{"start_time": "[00:00.000]", "end_time": "[00:05.000]"}]}
    ok, errs = mv.validate_timestamp_format(data_good)
    assert ok is True and errs == []

    data_bad = {"structure": [{"start_time": "00:00.000", "end_time": "[bad]"}, "not_a_section"]}
    ok2, errs2 = mv.validate_timestamp_format(data_bad)
    assert ok2 is False
    assert any("Invalid timestamp format" in e or "is not a dictionary" in e for e in errs2)


def test_validate_lyrics_has_lrc():
    ok, msg = mv.validate_lyrics_has_lrc({})
    assert ok is True
    ok2, msg2 = mv.validate_lyrics_has_lrc({"lyrics": True})
    assert ok2 is False and "lrc_file" in msg2
    ok3, msg3 = mv.validate_lyrics_has_lrc({"lyrics": False})
    assert ok3 is True


def test_validate_required_properties_and_structure_and_tracks():
    empty = {}
    ok, errs = mv.validate_required_properties(empty)
    assert ok is False
    assert any("Missing required property" in e for e in errs)

    # Structure items missing fields
    data = {
        "bpm": 120,
        "manifest_id": "m1",
        "tempo": "4/4",
        "key": "C major",
        "rainbow_color": "R",
        "title": "t",
        "release_date": "2020-01-01",
        "album_sequence": 1,
        "main_audio_file": "a.wav",
        "TRT": "[00:10.000]",
        "vocals": False,
        "lyrics": False,
        "structure": [{"section_name": "s1"}],
        "mood": [],
        "sounds_like": [],
        "genres": [],
        "concept": "",
        "audio_tracks": [{"id": 1}],
    }
    ok2, errs2 = mv.validate_required_properties(data)
    assert ok2 is False
    # should report missing section props and audio track description and file type
    assert any("Structure section 1 missing required property" in e for e in errs2)
    assert any("Audio track 1 missing required property: description" in e for e in errs2)
    assert any("Audio track 1 missing at least one file type" in e for e in errs2)


def test_validate_structure_timestamps_overlap_and_parse_error():
    # Overlapping sections
    data = {"structure": [
        {"section_name": "A", "start_time": "[00:00.000]", "end_time": "[00:05.000]"},
        {"section_name": "B", "start_time": "[00:04.000]", "end_time": "[00:06.000]"}
    ]}
    ok, errs = mv.validate_structure_timestamps(data)
    assert ok is False
    assert any("Overlapping sections" in e for e in errs)

    # Parse error: missing start_time/key
    bad = {"structure": [{"section_name": "X"}]}
    ok2, errs2 = mv.validate_structure_timestamps(bad)
    assert ok2 is False
    assert any("Error parsing structure timestamps" in e for e in errs2)


def test_check_no_tk_fields():
    data = {"a": "TK", "b": {"c": ["x", "TK"]}}
    errs = mv.check_no_tk_fields(data)
    assert len(errs) == 2
    assert any("a' is set to 'TK'" in e or "b.c[1]" in e or "[1]" in e for e in errs)


def test_validate_discogs_ids_mocked(monkeypatch):
    # prepare yaml_data with sounds_like entries
    yaml_data = {"sounds_like": [{"name": "The Band", "discogs_id": 1234}, {"name": "Other", "discogs_id": 2222}]}

    # create fake client and artist objects
    class FakeArtist:
        def __init__(self, name):
            self.name = name
    class FakeClient:
        def __init__(self, ua, user_token=None):
            pass
        def artist(self, discogs_id):
            if discogs_id == 1234:
                return FakeArtist("The Band")
            raise Exception("not found")

    monkeypatch.setattr(mv, 'discogs_client', SimpleNamespace(Client=FakeClient))
    ok, errs = mv.validate_discogs_ids(yaml_data)
    assert ok is False
    assert any("Error checking Discogs ID" in e or "corresponds to" in e or "not found" in e for e in errs)


def test_validate_yaml_file_and_directory(tmp_path, monkeypatch):
    # build a complete valid YAML manifest that should pass validators (mock discogs)
    manifest = {
        "bpm": 120,
        "manifest_id": "m1",
        "tempo": "4/4",
        "key": "C major",
        "rainbow_color": "R",
        "title": "T",
        "release_date": "2020-01-01",
        "album_sequence": 1,
        "main_audio_file": "a.wav",
        "TRT": "[00:00.000]",
        "vocals": False,
        "lyrics": False,
        "structure": [{"section_name": "s1", "start_time": "[00:00.000]", "end_time": "[00:05.000]", "description": "d"}],
        "mood": [],
        "sounds_like": [{"name": "Artist", "discogs_id": 1}],
        "genres": [],
        "concept": "",
        "audio_tracks": [{"id": 1, "description": "d1", "audio_file": "a.wav"}],
        "lrc_file": "song.lrc",
        "main_audio_file": "a.wav"
    }
    # create files referenced
    (tmp_path / "a.wav").write_text("x")
    (tmp_path / "song.lrc").write_text("x")

    # mock discogs client to return matching name
    class FakeArtist:
        def __init__(self, name):
            self.name = name
    class FakeClient:
        def __init__(self, ua, user_token=None):
            pass
        def artist(self, discogs_id):
            return FakeArtist("Artist")
    monkeypatch.setattr(mv, 'discogs_client', SimpleNamespace(Client=FakeClient))

    # write YAML file
    p = tmp_path / "m.yml"
    p.write_text(yaml.safe_dump(manifest))

    ok, errors = mv.validate_yaml_file(str(p))
    assert ok is True, errors

    # validate_directory should find this file and return valid
    ok2, all_errs = mv.validate_directory(str(tmp_path))
    assert ok2 is True
