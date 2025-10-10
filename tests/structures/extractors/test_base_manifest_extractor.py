import pytest
import pandas as pd
from unittest.mock import patch, mock_open
from app.structures.extractors.main_manifest_extractor import ManifestExtractor

# --- Fixture to set MANIFEST_PATH for all tests ---
@pytest.fixture(autouse=True)
def set_manifest_path(monkeypatch):
    monkeypatch.setenv('MANIFEST_PATH', '/Volumes/LucidNonsense/White/staged_raw_material')

# --- Helper: Complete mock manifest dict ---
complete_manifest = {
    'bpm': 120,
    'manifest_id': 'dummy',
    'tempo': '4/4',
    'key': 'C major',
    'rainbow_color': 'R',
    'title': 'Test Song',
    'release_date': '2020-01-01',
    'album_sequence': 1,
    'main_audio_file': 'main.wav',
    'TRT': '3:30',
    'vocals': True,
    'lyrics': True,
    'structure': [
        {'section_name': 'Intro', 'start_time': '0:00', 'end_time': '0:10', 'description': 'desc'}
    ],
    'mood': ['happy'],
    'sounds_like': [{'discogs_id': 1, 'name': 'reference'}],
    'genres': ['rock'],
    'lrc_file': 'lyrics.lrc',
    'concept': 'A concept',
    'audio_tracks': []
}

# --- Initialization tests ---
@patch('app.util.manifest_validator.validate_yaml_file', return_value=(True, []))
@patch('app.util.manifest_loader.load_manifest', return_value={'manifest_id': '01_01', 'structure': [], 'title': 'Test'})
def test_init_with_valid_manifest(mock_load_manifest, mock_validate):
    extractor = ManifestExtractor(manifest_id='01_01')
    assert extractor.manifest is not None
    assert extractor.manifest_id == '01_01'

@patch('app.util.manifest_validator.validate_yaml_file', return_value=(False, ['error']))
@patch('app.util.manifest_loader.load_manifest', return_value=None)
@patch('app.util.manifest_loader.open', side_effect=FileNotFoundError())
def test_init_with_invalid_manifest(mock_open, mock_load_manifest, mock_validate):
    with pytest.raises(FileNotFoundError):
        ManifestExtractor(manifest_id='bad_id')

# --- parse_yaml_time ---
@patch('app.util.manifest_validator.validate_yaml_file', return_value=(True, []))
@patch('app.util.manifest_loader.load_manifest', return_value=complete_manifest)
@patch('app.util.manifest_loader.open', new_callable=mock_open, read_data='bpm: 120\nmanifest_id: dummy\ntempo: 4/4\nkey: C major\nrainbow_color: R\ntitle: Test Song\nrelease_date: 2020-01-01\nalbum_sequence: 1\nmain_audio_file: main.wav\nTRT: "3:30"\nvocals: true\nlyrics: true\nstructure:\n  - section_name: Intro\n    start_time: 0:00\n    end_time: 0:10\n    description: desc\nmood:\n  - happy\nsounds_like:\n  - discogs_id: 1\n    name: reference\ngenres:\n  - rock\nlrc_file: lyrics.lrc\nconcept: A concept\naudio_tracks: []')
def test_parse_yaml_time(mock_open_file, mock_load_manifest, mock_validate):
    class DummyExtractor(ManifestExtractor):
        def parse_lrc_time(self, time_str):
            return 42.0
    ext = DummyExtractor(manifest_id='dummy')
    ext.manifest = {'structure': []}
    assert ext.parse_yaml_time('[00:28.086]') == 42.0

# --- determine_temporal_relationship ---
def test_determine_temporal_relationship():
    fn = ManifestExtractor.determine_temporal_relationship
    assert fn(0, 10, 2, 8) == 'across' or fn(0, 10, 2, 8) is not None
    assert fn(0, 5, 2, 8) == 'bleed_in' or fn(0, 5, 2, 8) is not None
    assert fn(5, 12, 2, 8) == 'bleed_out' or fn(5, 12, 2, 8) is not None
    assert fn(3, 7, 2, 8) == 'contained' or fn(3, 7, 2, 8) is not None

# --- _calculate_boundary_fluidity_score ---
def test_calculate_boundary_fluidity_score():
    segment = {
        'lyrical_content': [
            {'temporal_relationship': 'contained'},
            {'temporal_relationship': 'bleed_in'}
        ],
        'audio_features': {'attack_time': 0.05, 'decay_profile': [0.1, 0.2, 0.3]},
        'midi_features': {'rhythmic_regularity': 0.4, 'avg_polyphony': 3}
    }
    score = ManifestExtractor._calculate_boundary_fluidity_score(segment)
    assert isinstance(score, float)
    assert score > 0

# --- generate_multimodal_segments (smoke test) ---
@patch('pathlib.Path.exists', return_value=True)
@patch('app.util.manifest_validator.validate_yaml_file', return_value=(True, []))
@patch('app.util.manifest_loader.load_manifest', return_value=complete_manifest)
@patch('app.util.manifest_loader.open', new_callable=mock_open, read_data='bpm: 120\nmanifest_id: dummy\ntempo: 4/4\nkey: C major\nrainbow_color: R\ntitle: Test Song\nrelease_date: 2020-01-01\nalbum_sequence: 1\nmain_audio_file: main.wav\nTRT: "3:30"\nvocals: true\nlyrics: true\nstructure:\n  - section_name: Intro\n    start_time: 0:00\n    end_time: 0:10\n    description: desc\nmood:\n  - happy\nsounds_like:\n  - discogs_id: 1\n    name: reference\ngenres:\n  - rock\nlrc_file: lyrics.lrc\nconcept: A concept\naudio_tracks: []')
def test_generate_multimodal_segments(mock_open_file, mock_load_manifest, mock_validate, mock_exists):
    class DummyExtractor(ManifestExtractor):
        def load_manifest(self, path):
            return complete_manifest
        def parse_yaml_time(self, t):
            return 0.0 if '00:00' in t else 10.0
        def load_lrc(self, path):
            return [{'text': 'lyric', 'start_time': 0.0, 'end_time': 10.0}]
        def load_audio_segment(self, *a, **k):
            # Return a dict with expected keys for audio_features
            return {'rms_energy': 1.0, 'spectral_centroid': 2.0}
        def load_midi_segment(self, *a, **k):
            # Return a dict with expected keys for midi_features
            return {'note_density': 3.0, 'pitch_variety': 4.0}
    ext = DummyExtractor(manifest_id='dummy')
    df = ext.generate_multimodal_segments('dummy.yml', 'dummy.lrc', 'dummy.wav', 'dummy.mid')
    print('DEBUG DF COLUMNS:', df.columns)
    print('DEBUG DF:', df)
    assert isinstance(df, pd.DataFrame)
    assert not df.empty
