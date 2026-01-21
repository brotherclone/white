"""Tests for audio I/O utilities."""

import io
from unittest.mock import patch

import numpy as np
import pytest
import soundfile as sf

from app.util.audio_io import load_audio


@pytest.fixture
def sample_audio_mono(tmp_path):
    """Create a sample mono audio file."""
    file_path = tmp_path / "test_mono.wav"
    sample_rate = 44100
    duration = 1.0  # seconds
    samples = int(sample_rate * duration)

    # Generate a simple sine wave
    t = np.linspace(0, duration, samples, dtype=np.float32)
    data = np.sin(2 * np.pi * 440 * t).astype(np.float32)

    sf.write(str(file_path), data, sample_rate)
    return str(file_path), sample_rate, samples


@pytest.fixture
def sample_audio_stereo(tmp_path):
    """Create a sample stereo audio file."""
    file_path = tmp_path / "test_stereo.wav"
    sample_rate = 44100
    duration = 1.0
    samples = int(sample_rate * duration)

    # Generate stereo data (2 channels)
    t = np.linspace(0, duration, samples, dtype=np.float32)
    left = np.sin(2 * np.pi * 440 * t).astype(np.float32)
    right = np.sin(2 * np.pi * 880 * t).astype(np.float32)
    data = np.column_stack((left, right))

    sf.write(str(file_path), data, sample_rate)
    return str(file_path), sample_rate, samples


def test_load_audio_mono_file(sample_audio_mono):
    """Test loading a mono audio file."""
    file_path, expected_sr, expected_samples = sample_audio_mono

    data, sr = load_audio(file_path)

    assert sr == expected_sr
    assert data.shape == (expected_samples,)
    assert data.dtype == np.float32
    assert np.all(np.abs(data) <= 1.0)


def test_load_audio_stereo_to_mono(sample_audio_stereo):
    """Test loading stereo audio and converting to mono."""
    file_path, expected_sr, expected_samples = sample_audio_stereo

    data, sr = load_audio(file_path, mono=True)

    assert sr == expected_sr
    assert data.ndim == 1  # Should be 1D (mono)
    assert data.shape[0] == expected_samples
    assert data.dtype == np.float32


def test_load_audio_stereo_keep_channels(sample_audio_stereo):
    """Test loading stereo audio without converting to mono."""
    file_path, expected_sr, expected_samples = sample_audio_stereo

    data, sr = load_audio(file_path, mono=False)

    assert sr == expected_sr
    assert data.ndim == 2  # Should be 2D (stereo)
    assert data.shape == (expected_samples, 2)
    assert data.dtype == np.float32


def test_load_audio_resample_without_librosa(sample_audio_mono):
    """Test resampling audio without librosa (fallback mode)."""
    file_path, original_sr, original_samples = sample_audio_mono
    target_sr = 22050

    with patch("app.util.audio_io.librosa", None):
        data, sr = load_audio(file_path, sr=target_sr)

    assert sr == target_sr
    assert data.dtype == np.float32
    # Check approximate length (due to resampling)
    expected_samples = int(original_samples * target_sr / original_sr)
    assert abs(data.shape[0] - expected_samples) <= 1


def test_load_audio_same_sample_rate(sample_audio_mono):
    """Test loading audio with same sample rate (no resampling)."""
    file_path, original_sr, original_samples = sample_audio_mono

    data, sr = load_audio(file_path, sr=original_sr)

    assert sr == original_sr
    assert data.shape[0] == original_samples
    assert data.dtype == np.float32


def test_load_audio_no_sample_rate_specified(sample_audio_mono):
    """Test loading audio without specifying sample rate."""
    file_path, original_sr, original_samples = sample_audio_mono

    data, sr = load_audio(file_path, sr=None)

    assert sr == original_sr
    assert data.shape[0] == original_samples


def test_load_audio_resample_zero_samples(tmp_path):
    """Test resampling with extremely short audio that results in zero samples."""
    file_path = tmp_path / "tiny.wav"
    sample_rate = 44100

    # Create a very short audio file (1 sample)
    data = np.array([0.1], dtype=np.float32)
    sf.write(str(file_path), data, sample_rate)

    # Try to resample to very low sample rate without librosa
    with patch("app.util.audio_io.librosa", None):
        result, sr = load_audio(str(file_path), sr=1)

    assert sr == 1
    assert isinstance(result, np.ndarray)
    assert result.dtype == np.float32


def test_load_audio_file_like_object():
    """Test loading audio from a file-like object."""
    # Create in-memory audio data
    sample_rate = 44100
    duration = 0.1
    samples = int(sample_rate * duration)

    t = np.linspace(0, duration, samples, dtype=np.float32)
    audio_data = np.sin(2 * np.pi * 440 * t).astype(np.float32)

    # Write to BytesIO
    buffer = io.BytesIO()
    sf.write(buffer, audio_data, sample_rate, format="WAV")
    buffer.seek(0)

    # Load from buffer
    data, sr = load_audio(buffer)

    assert sr == sample_rate
    assert len(data) == samples
    assert data.dtype == np.float32


def test_load_audio_data_range(sample_audio_mono):
    """Test that loaded audio data is in valid range."""
    file_path, _, _ = sample_audio_mono

    data, _ = load_audio(file_path)

    # Audio should be normalized to -1.0 to 1.0 range
    assert np.all(data >= -1.0)
    assert np.all(data <= 1.0)


def test_load_audio_dtype_consistency(sample_audio_mono):
    """Test that output dtype is always float32."""
    file_path, expected_sr, _ = sample_audio_mono

    # Load without resampling
    data1, _ = load_audio(file_path)
    assert data1.dtype == np.float32

    # Load with resampling
    data2, _ = load_audio(file_path, sr=22050)
    assert data2.dtype == np.float32


def test_load_audio_resample_upsampling(sample_audio_mono):
    """Test upsampling audio to higher sample rate."""
    file_path, original_sr, original_samples = sample_audio_mono
    target_sr = original_sr * 2  # Double the sample rate

    with patch("app.util.audio_io.librosa", None):
        data, sr = load_audio(file_path, sr=target_sr)

    assert sr == target_sr
    # Should have approximately double the samples
    expected_samples = original_samples * 2
    assert abs(data.shape[0] - expected_samples) <= 1


def test_load_audio_stereo_averaging(tmp_path):
    """Test that stereo to mono conversion properly averages channels."""
    file_path = tmp_path / "test_stereo_distinct.wav"
    sample_rate = 44100
    samples = 100

    # Create stereo with distinct values per channel
    left = np.ones(samples, dtype=np.float32) * 0.5
    right = np.ones(samples, dtype=np.float32) * -0.5
    stereo_data = np.column_stack((left, right))

    sf.write(str(file_path), stereo_data, sample_rate)

    data, _ = load_audio(str(file_path), mono=True)

    # Average should be close to 0
    assert data.ndim == 1
    assert np.allclose(data, 0.0, atol=1e-6)
