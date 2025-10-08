import os
import tempfile
import numpy as np
import pytest
from unittest import mock
from app.agents.tools import audio_tools
from app.agents.enums.noise_type import NoiseType
from app.agents.enums.chain_artifact_file_type import ChainArtifactFileType
from app.agents.models.audio_chain_artifact_file import AudioChainArtifactFile
from app.structures.concepts.rainbow_table_color import the_rainbow_table_colors


def test_generate_speech_like_noise_length():
    duration = 1.0
    sr = 8000
    result = audio_tools.generate_speech_like_noise(duration, sr)
    assert isinstance(result, bytes)
    assert len(result) == int(duration * sr) * 2  # int16

def test_generate_noise_white_length():
    duration = 0.5
    sr = 16000
    result = audio_tools.generate_noise(duration, NoiseType.WHITE, sample_rate=sr)
    assert isinstance(result, bytes)
    assert len(result) == int(duration * sr) * 2

def test_generate_noise_invalid_type():
    duration = 0.1
    sr = 8000
    result = audio_tools.generate_noise(duration, None, sample_rate=sr)
    assert isinstance(result, bytes)
    assert len(result) == int(duration * sr) * 2

def test_pitch_shift_audio_bytes_identity():
    sr = 8000
    arr = (np.random.rand(sr) * 2 - 1).astype(np.float32)
    arr_bytes = (arr * 32767).astype(np.int16).tobytes()
    shifted = audio_tools.pitch_shift_audio_bytes(arr_bytes, cents=0, sample_rate=sr)
    assert isinstance(shifted, bytes)
    assert len(shifted) == len(arr_bytes)

def test_micro_stutter_audio_bytes():
    sr = 8000
    arr = (np.random.rand(sr) * 2 - 1).astype(np.float32)
    arr_bytes = (arr * 32767).astype(np.int16).tobytes()
    stuttered = audio_tools.micro_stutter_audio_bytes(arr_bytes, stutter_probability=1.0, stutter_length_ms=10, sample_rate=sr)
    assert isinstance(stuttered, bytes)
    assert len(stuttered) >= len(arr_bytes)

def test_gate_audio_bytes():
    sr = 8000
    arr = (np.random.rand(sr) * 2 - 1).astype(np.float32)
    arr_bytes = (arr * 32767).astype(np.int16).tobytes()
    gated = audio_tools.gate_audio_bytes(arr_bytes, gate_probability=1.0, gate_length_ms=10, sample_rate=sr)
    assert isinstance(gated, bytes)
    assert len(gated) == len(arr_bytes)
    assert gated.count(b'\x00') > 0

def test_bit_crush_audio_bytes():
    arr = (np.random.rand(1000) * 2 - 1).astype(np.float32)
    arr_bytes = (arr * 32767).astype(np.int16).tobytes()
    crushed = audio_tools.bit_crush_audio_bytes(arr_bytes, intensity=1.0)
    assert isinstance(crushed, bytes)
    assert len(crushed) == len(arr_bytes)

def test_apply_speech_hallucination_processing():
    sr = 8000
    arr = (np.random.rand(sr) * 2 - 1).astype(np.float32)
    arr_bytes = (arr * 32767).astype(np.int16).tobytes()
    hallucinated = audio_tools.apply_speech_hallucination_processing(arr_bytes, hallucination_intensity=0.5, sample_rate=sr)
    assert isinstance(hallucinated, bytes)
    assert len(hallucinated) > 0

def test_save_wav_from_bytes(tmp_path):
    sr = 8000
    arr = (np.random.rand(sr) * 2 - 1).astype(np.float32)
    arr_bytes = (arr * 32767).astype(np.int16).tobytes()
    out_path = tmp_path / "test.wav"
    audio_tools.save_wav_from_bytes(str(out_path), arr_bytes, sample_rate=sr)
    assert out_path.exists()
    assert out_path.stat().st_size > 0

def test_find_wav_files(tmp_path):
    f1 = tmp_path / "a.wav"
    f2 = tmp_path / "b.WAV"
    f3 = tmp_path / "c.mp3"
    f1.write_bytes(b"data")
    f2.write_bytes(b"data")
    f3.write_bytes(b"data")
    files = audio_tools.find_wav_files(str(tmp_path), None)
    assert str(f1) in files
    assert str(f2) in files
    assert str(f3) not in files

def test_extract_non_silent_segments():
    sr = 8000
    audio = np.zeros(sr)
    audio[100:200] = 1.0
    segments = audio_tools.extract_non_silent_segments(audio, sr, min_duration=0.005)
    assert isinstance(segments, list)
    assert any(len(seg) >= int(0.005 * sr) for seg in segments)

def test_create_audio_mosaic_chain_artifact(tmp_path, monkeypatch):
    # Mock AudioChainArtifactFile and file I/O
    class DummyArtifact:
        def __init__(self, path, sr):
            self._path = path
            self.sample_rate = sr
            self.rainbow_color = the_rainbow_table_colors['Z']
            self.chain_artifact_file_type = ChainArtifactFileType.AUDIO
        def get_artifact_path(self, with_file_name=True):
            return self._path
    # Create dummy wav files
    import soundfile as sf
    sr = 8000
    arr = np.random.rand(sr * 2).astype(np.float32)
    paths = []
    for i in range(3):
        p = tmp_path / f"test_{i}.wav"
        sf.write(str(p), arr, sr)
        paths.append(str(p))
    artifacts = [DummyArtifact(p, sr) for p in paths]
    monkeypatch.setenv("AGENT_WORK_PRODUCT_BASE_PATH", str(tmp_path))
    artifact = audio_tools.create_audio_mosaic_chain_artifact(artifacts, 50, 1.0, thread_id="tst")
    assert isinstance(artifact, AudioChainArtifactFile)
    assert os.path.exists(artifact.artifact_path)
    assert artifact.duration > 0

def test_blend_with_noise(tmp_path):
    import soundfile as sf
    sr = 8000
    arr = np.random.rand(sr).astype(np.float32)
    in_path = tmp_path / "in.wav"
    sf.write(str(in_path), arr, sr)
    out_dir = tmp_path / "out"
    out_path = audio_tools.blend_with_noise(str(in_path), 0.5, str(out_dir))
    assert os.path.exists(out_path)
    assert out_path.endswith("_blended.wav")

def test_create_blended_audio_chain_artifact(tmp_path, monkeypatch):
    # Prepare dummy mosaic artifact
    class DummyArtifact(AudioChainArtifactFile):
        def get_artifact_path(self, with_file_name=True):
            return str(tmp_path / "mosaic.wav")
    sr = 8000
    arr = np.random.rand(sr).astype(np.float32)
    import soundfile as sf
    sf.write(tmp_path / "mosaic.wav", arr, sr)
    monkeypatch.setenv("AGENT_WORK_PRODUCT_BASE_PATH", str(tmp_path))
    mosaic = DummyArtifact(
        base_path=str(tmp_path),
        rainbow_color=the_rainbow_table_colors['Z'],
        chain_artifact_file_type=ChainArtifactFileType.AUDIO,
        file_name="mosaic.wav",
        artifact_path=str(tmp_path / "mosaic.wav"),
        duration=1.0,
        sample_rate=sr,
        channels=1,
        bit_depth=16
    )
    artifact = audio_tools.create_blended_audio_chain_artifact(mosaic, 0.5, thread_id="tst")
    assert isinstance(artifact, AudioChainArtifactFile)
    # The file is not written by create_blended_audio_chain_artifact, but blend_with_noise is called

