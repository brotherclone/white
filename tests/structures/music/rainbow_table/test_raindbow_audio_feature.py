import numpy as np
import pytest
from pydantic import ValidationError

from app.structures.music.rainbow_table.raindbow_audio_feature import \
    RainbowAudioFeature


def test_rainbow_audio_feature():
    """Test RainbowAudioFeature creation with required fields"""
    feature = RainbowAudioFeature(
        audio_file_path="/path/to/audio.wav",
        segment_start_time=0.0,
        segment_end_time=5.0,
        duration=5.0,
    )
    assert feature.audio_file_path == "/path/to/audio.wav"
    assert feature.segment_start_time == 0.0
    assert feature.segment_end_time == 5.0
    assert feature.duration == 5.0


def test_rainbow_audio_feature_with_optional_fields():
    """Test RainbowAudioFeature with optional fields"""
    feature = RainbowAudioFeature(
        audio_file_path="/test.wav",
        segment_start_time=1.0,
        segment_end_time=3.0,
        duration=2.0,
        duration_samples=88200,
        peak_amplitude=0.8,
        rms_energy=0.5,
        spectral_centroid=1500.0,
        zero_crossing_rate=0.1,
        tempo=120.0,
    )
    assert feature.duration_samples == 88200
    assert feature.peak_amplitude == 0.8
    assert feature.rms_energy == 0.5
    assert feature.tempo == 120.0


def test_rainbow_audio_feature_with_numpy_arrays():
    """Test RainbowAudioFeature with numpy array fields"""
    mfcc_data = np.array([[1.0, 2.0, 3.0]])
    chroma_data = np.array([[0.5, 0.6, 0.7]])

    feature = RainbowAudioFeature(
        audio_file_path="/test.wav",
        segment_start_time=0.0,
        segment_end_time=1.0,
        duration=1.0,
        mfcc=mfcc_data,
        chroma=chroma_data,
    )
    assert feature.mfcc is not None
    assert feature.chroma is not None
    assert np.array_equal(feature.mfcc, mfcc_data)
    assert np.array_equal(feature.chroma, chroma_data)


def test_rainbow_audio_feature_silence_analysis():
    """Test RainbowAudioFeature with silence analysis fields"""
    feature = RainbowAudioFeature(
        audio_file_path="/test.wav",
        segment_start_time=0.0,
        segment_end_time=10.0,
        duration=10.0,
        is_mostly_silence=False,
        non_silence_ratio=0.85,
        silence_gaps=[1.0, 2.5],
        silence_confidence=0.95,
    )
    assert feature.is_mostly_silence is False
    assert feature.non_silence_ratio == 0.85
    assert len(feature.silence_gaps) == 2
    assert feature.silence_confidence == 0.95


def test_rainbow_audio_feature_missing_required():
    """Test that required fields are enforced"""
    with pytest.raises(ValidationError):
        RainbowAudioFeature(segment_start_time=0.0, segment_end_time=1.0, duration=1.0)


def test_rainbow_audio_feature_non_silence_regions():
    """Test non_silence_regions field"""
    feature = RainbowAudioFeature(
        audio_file_path="/test.wav",
        segment_start_time=0.0,
        segment_end_time=10.0,
        duration=10.0,
        non_silence_regions=[(0.0, 2.5), (3.0, 8.0)],
    )
    assert len(feature.non_silence_regions) == 2
    assert feature.non_silence_regions[0] == (0.0, 2.5)
