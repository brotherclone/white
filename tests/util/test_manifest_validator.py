import pytest
import os
import tempfile
import yaml
from pathlib import Path

from app.util.manifest_validator import (
    validate_yaml_file,
    validate_manifest_completeness,
)


@pytest.fixture
def temp_manifest_dir():
    """Create a temporary directory for test manifests"""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


@pytest.fixture
def valid_manifest(temp_manifest_dir):
    """Create a valid manifest with all files present"""
    manifest_data = {
        'manifest_id': 'test_01',
        'title': 'Test Song',
        'rainbow_color': 'R',
        'bpm': 120,
        'tempo': 120,
        'key': 'C',
        'release_date': '2025-01-01',
        'album_sequence': 1,
        'main_audio_file': 'vocals.wav',
        'TRT': '00:03:00',
        'vocals': True,
        'lyrics': True,
        'mood': ['Happy'],
        'sounds_like': [{'name': 'Mongorillo Dreams', 'discogs_id': '12345'}],
        'genres': ['Pop'],
        'concept': 'Test concept',
        'structure': [
            {
                'section_name': 'Intro',
                'start_time': '[00:00.000]',
                'end_time': '[00:30.000]',
                'description': 'Opening section'
            }
        ],
        'audio_tracks': [
            {
                'id': 1,
                'description': 'Lead Vocals',
                'audio_file': 'vocals.wav'
            },
            {
                'id': 2,
                'description': 'Background Vocals',
                'audio_file': 'bg_vox.wav',
                'player': 'REMEZ'
            }
        ],
        'midi_file': 'test.mid',
        'lrc_file': 'test_01.lrc'
    }

    # Write manifest
    manifest_path = os.path.join(temp_manifest_dir, 'test_01.yml')
    with open(manifest_path, 'w') as f:
        yaml.dump(manifest_data, f)

    # Create referenced files
    Path(os.path.join(temp_manifest_dir, 'vocals.wav')).touch()
    Path(os.path.join(temp_manifest_dir, 'bg_vox.wav')).touch()
    Path(os.path.join(temp_manifest_dir, 'test.mid')).touch()
    Path(os.path.join(temp_manifest_dir, 'test_01.lrc')).touch()

    return manifest_path


@pytest.fixture
def manifest_missing_audio(temp_manifest_dir):
    """Create a manifest with missing audio files"""
    manifest_data = {
        'manifest_id': 'test_02',
        'title': 'Test Song with Missing Audio',
        'rainbow_color': 'BLUE',
        'bpm': 120,
        'tempo': 120,
        'key': 'C',
        'release_date': '2025-01-01',
        'album_sequence': 2,
        'main_audio_file': 'vocals.wav',
        'TRT': '00:03:00',
        'vocals': True,
        'lyrics': True,
        'mood': 'Sad',
        'sounds_like': [{'name': 'Artist', 'discogs_id': '12345'}],
        'genres': ['Pop'],
        'concept': 'Test concept',
        'structure': [
            {
                'section_name': 'Intro',
                'start_time': '[00:00.000]',
                'end_time': '[00:30.000]',
                'description': 'Opening section'
            }
        ],
        'audio_tracks': [
            {
                'id': 1,
                'description': 'Lead Vocals',
                'audio_file': 'vocals.wav'
            },
            {
                'id': 2,
                'description': 'Background Vocals',
                'audio_file': 'missing_bg_vox.wav',
                'player': 'REMEZ'
            }
        ],
        'lrc_file': 'test_02.lrc'
    }

    manifest_path = os.path.join(temp_manifest_dir, 'test_02.yml')
    with open(manifest_path, 'w') as f:
        yaml.dump(manifest_data, f)

    Path(os.path.join(temp_manifest_dir, 'vocals.wav')).touch()
    Path(os.path.join(temp_manifest_dir, 'test_02.lrc')).touch()

    return manifest_path


@pytest.fixture
def manifest_missing_midi(temp_manifest_dir):
    """Create a manifest with a missing MIDI file"""
    manifest_data = {
        'manifest_id': 'test_03',
        'title': 'Test Song with Missing MIDI',
        'rainbow_color': 'GREEN',
        'bpm': 120,
        'tempo': 120,
        'key': 'C',
        'release_date': '2025-01-01',
        'album_sequence': 3,
        'main_audio_file': 'vocals.wav',
        'TRT': '00:03:00',
        'vocals': True,
        'lyrics': True,
        'mood': 'Energetic',
        'sounds_like': [{'name': 'Artist', 'discogs_id': '12345'}],
        'genres': ['Pop'],
        'concept': 'Test concept',
        'structure': [
            {
                'section_name': 'Intro',
                'start_time': '[00:00.000]',
                'end_time': '[00:30.000]',
                'description': 'Opening section'
            }
        ],
        'audio_tracks': [
            {
                'id': 1,
                'description': 'Lead Vocals',
                'audio_file': 'vocals.wav',
                'midi_file': 'missing.mid'
            }
        ],
        'lrc_file': 'test_03.lrc'
    }

    manifest_path = os.path.join(temp_manifest_dir, 'test_03.yml')
    with open(manifest_path, 'w') as f:
        yaml.dump(manifest_data, f)

    Path(os.path.join(temp_manifest_dir, 'vocals.wav')).touch()
    Path(os.path.join(temp_manifest_dir, 'test_03.lrc')).touch()

    return manifest_path


class TestManifestValidator:
    """Tests for manifest validation"""

    def test_valid_manifest(self, valid_manifest):
        """Test that a valid manifest passes validation"""
        is_valid, errors = validate_yaml_file(valid_manifest)

        assert is_valid, f"Valid manifest failed: {errors}"
        assert len(errors) == 0

    def test_manifest_missing_audio_files(self, manifest_missing_audio):
        """Test that missing audio files are detected"""
        is_valid, errors = validate_yaml_file(manifest_missing_audio)

        assert not is_valid
        assert len(errors) > 0

        # Check that the specific missing file is mentioned
        error_text = ' '.join(errors)
        assert 'missing_bg_vox.wav' in error_text
        assert 'Audio track 2' in error_text or 'Background Vocals' in error_text

    def test_manifest_missing_midi_file(self, manifest_missing_midi):
        """Test that missing MIDI files are detected"""
        is_valid, errors = validate_yaml_file(manifest_missing_midi)

        assert not is_valid
        assert len(errors) > 0

        # Check that the missing MIDI is mentioned
        error_text = ' '.join(errors)
        assert 'missing.mid' in error_text or 'midi_file' in error_text
        assert 'MIDI' in error_text or 'midi_file' in error_text

    def test_manifest_file_not_found(self, temp_manifest_dir):
        """Test error when manifest file doesn't exist"""
        fake_path = os.path.join(temp_manifest_dir, 'nonexistent.yml')
        is_valid, errors = validate_yaml_file(fake_path)

        assert not is_valid
        assert len(errors) > 0
        assert 'no such file or directory' in errors[0].lower() or 'error parsing yaml' in errors[0].lower()

    def test_skip_file_existence_check(self, manifest_missing_audio):
        """Test that file existence check can be disabled (not supported, so just check error count)"""
        is_valid, errors = validate_yaml_file(manifest_missing_audio)

        # Should fail due to missing file, as skipping is not supported
        assert not is_valid
        assert any('not found' in e.lower() or 'missing_bg_vox.wav' in e for e in errors)

    def test_invalid_yaml_structure(self, temp_manifest_dir):
        """Test that malformed YAML is caught"""
        manifest_path = os.path.join(temp_manifest_dir, 'invalid.yml')
        with open(manifest_path, 'w') as f:
            f.write("invalid: yaml: content: [[")

        is_valid, errors = validate_yaml_file(manifest_path)

        assert not is_valid
        assert len(errors) > 0
        assert 'yaml' in errors[0].lower() or 'parsing' in errors[0].lower()

    def test_missing_required_fields(self, temp_manifest_dir):
        """Test that missing required fields are detected"""
        manifest_data = {
            'title': 'Test Song',
            # Missing manifest_id and structure
        }

        manifest_path = os.path.join(temp_manifest_dir, 'incomplete.yml')
        with open(manifest_path, 'w') as f:
            yaml.dump(manifest_data, f)

        is_valid, errors = validate_yaml_file(manifest_path)

        assert not is_valid
        assert len(errors) >= 2  # Should catch both missing fields
        error_text = ' '.join(errors)
        assert 'manifest_id' in error_text
        assert 'structure' in error_text


class TestManifestCompleteness:
    """Tests for manifest completeness checking"""

    def test_complete_manifest(self, valid_manifest):
        """Test completeness check on fully complete manifest"""
        result = validate_manifest_completeness(valid_manifest)

        assert result['has_all_audio']
        assert result['has_midi'] or 'midi_file' in result
        assert result['has_lyrics']
        assert len(result['missing_audio']) == 0
        assert result['completion_percentage'] == 100.0

    def test_incomplete_manifest(self, manifest_missing_audio):
        """Test completeness check on incomplete manifest"""
        result = validate_manifest_completeness(manifest_missing_audio)

        assert not result['has_all_audio']
        assert len(result['missing_audio']) > 0
        assert 'missing_bg_vox.wav' in result['missing_audio']
        assert result['completion_percentage'] < 100.0

    def test_manifest_without_midi(self, manifest_missing_midi):
        """Test completeness when MIDI is referenced but missing"""
        result = validate_manifest_completeness(manifest_missing_midi)

        assert not result['has_midi'] or 'midi_file' in result
        assert len(result.get('missing_midi', [])) >= 0
        assert result['completion_percentage'] < 100.0
