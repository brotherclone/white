import numpy as np
import pandas as pd
import pytest
from unittest.mock import patch
from app.structures.extractors.audio_extractor import AudioExtractor
from app.structures.music.rainbow_table.raindbow_audio_feature import RainbowAudioFeature

@pytest.fixture
def dummy_audio_feature():
    return RainbowAudioFeature(
        audio_file_path='dummy.wav',
        segment_start_time=0.0,
        segment_end_time=1.0,
        duration=1.0,
        duration_samples=44100,
        is_mostly_silence=False
    )

@patch('app.structures.extractors.base_manifest_extractor.load_manifest', return_value={})
def test_load_raw_audio_segment_valid(mock_manifest, tmp_path, dummy_audio_feature):
    # Create a dummy wav file
    dummy_path = tmp_path / 'dummy.wav'
    sr = 44100
    y = np.random.randn(sr)
    import soundfile as sf
    sf.write(dummy_path, y, sr)
    dummy_audio_feature.audio_file_path = str(dummy_path)
    segment_row = pd.Series({'audio_features': dummy_audio_feature})
    extractor = AudioExtractor(sample_rate=sr, manifest_id='dummy')
    with patch('librosa.load', return_value=(y, sr)) as mock_load:
        segment = extractor.load_raw_audio_segment(segment_row)
        mock_load.assert_called_once_with(str(dummy_path), sr=sr)
        assert isinstance(segment, np.ndarray)
        assert segment.shape[0] == sr

@patch('app.structures.extractors.base_manifest_extractor.load_manifest', return_value={})
def test_load_raw_audio_segment_missing_file(mock_manifest, dummy_audio_feature):
    dummy_audio_feature.audio_file_path = 'nonexistent.wav'
    segment_row = pd.Series({'audio_features': dummy_audio_feature})
    extractor = AudioExtractor(sample_rate=44100, manifest_id='dummy')
    segment = extractor.load_raw_audio_segment(segment_row)
    assert isinstance(segment, np.ndarray)
    assert segment.size == 0

@patch('app.structures.extractors.base_manifest_extractor.load_manifest', return_value={})
def test_load_raw_audio_segment_none_path(mock_manifest, dummy_audio_feature):
    dummy_audio_feature.audio_file_path = None
    segment_row = pd.Series({'audio_features': dummy_audio_feature})
    extractor = AudioExtractor(sample_rate=44100, manifest_id='dummy')
    segment = extractor.load_raw_audio_segment(segment_row)
    assert isinstance(segment, np.ndarray)
    assert segment.size == 0

@patch('app.structures.extractors.base_manifest_extractor.load_manifest', return_value={})
def test_load_raw_audio_segment_missing_fields(mock_manifest):
    # Missing audio_features
    segment_row = pd.Series({})
    extractor = AudioExtractor(sample_rate=44100, manifest_id='dummy')
    with pytest.raises(KeyError):
        extractor.load_raw_audio_segment(segment_row)
