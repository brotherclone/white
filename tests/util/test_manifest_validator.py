import pytest
from unittest.mock import patch
from app.util.manifest_validator import validate_discogs_ids

def test_validate_discogs_ids_no_sounds_like():
    data = {'title': 'Test'}
    is_valid, errors = validate_discogs_ids(data)
    assert is_valid
    assert errors == []

def test_validate_discogs_ids_not_list():
    data = {'sounds_like': 'notalist'}
    is_valid, errors = validate_discogs_ids(data)
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
    is_valid, errors = validate_discogs_ids(data)
    assert is_valid
    assert errors == []

@patch('app.util.manifest_validator.discogs_client.Client')
@patch('app.util.manifest_validator.load_dotenv')
@patch('os.environ.get', return_value='dummy_token')
def test_validate_discogs_ids_missing_fields(mock_env, mock_dotenv, mock_client):
    data = {'sounds_like': [{'name': 'Artist'}]}
    is_valid, errors = validate_discogs_ids(data)
    assert not is_valid
    assert any("missing required 'name' or 'discogs_id'" in e for e in errors)

@patch('app.util.manifest_validator.discogs_client.Client')
@patch('app.util.manifest_validator.load_dotenv')
@patch('os.environ.get', return_value='dummy_token')
def test_validate_discogs_ids_not_dict(mock_env, mock_dotenv, mock_client):
    data = {'sounds_like': ['notadict']}
    is_valid, errors = validate_discogs_ids(data)
    assert not is_valid
    assert any('is not a dictionary' in e for e in errors)
import pytest
from unittest.mock import patch, mock_open
from app.util.manifest_loader import load_manifest

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
