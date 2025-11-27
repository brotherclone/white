import io
import os

import numpy as np
import soundfile as sf

from app.agents.tools import audio_tools
from app.agents.tools.audio_tools import find_wav_files_prioritized
from app.structures.enums.noise_type import NoiseType


def _sine_pcm_bytes(duration_s=0.1, freq=440.0, sr=22050, amplitude=0.5):
    t = np.linspace(0, duration_s, int(sr * duration_s), endpoint=False)
    wave = amplitude * np.sin(2 * np.pi * freq * t)
    pcm = (wave * 32767.0).astype(np.int16)
    return pcm.tobytes()


def test_generate_noise_returns_bytes_and_expected_length():
    sr = 8000
    dur = 0.05
    b = audio_tools.generate_noise(dur, NoiseType.WHITE, sample_rate=sr)
    assert isinstance(b, (bytes, bytearray))
    expected_samples = int(dur * sr)
    assert len(b) == expected_samples * 2  # int16 -> 2 bytes per sample


def test_generate_speech_like_noise_not_all_zero():
    sr = 8000
    dur = 0.05
    b = audio_tools.generate_speech_like_noise(dur, sample_rate=sr)
    arr = np.frombuffer(b, dtype=np.int16)
    assert arr.size == int(dur * sr)
    assert np.any(arr != 0)


def test_pitch_shift_preserves_length():
    sr = 22050
    pcm = _sine_pcm_bytes(duration_s=0.1, sr=sr)
    out = audio_tools.pitch_shift_audio_bytes(pcm, cents=100, sample_rate=sr)
    assert isinstance(out, (bytes, bytearray))
    assert len(out) == len(pcm)


def test_micro_stutter_increases_or_keeps_length_when_forced():
    sr = 22050
    pcm = _sine_pcm_bytes(duration_s=0.2, sr=sr)
    out = audio_tools.micro_stutter_audio_bytes(
        pcm, stutter_probability=1.0, stutter_length_ms=50, sample_rate=sr
    )
    assert isinstance(out, (bytes, bytearray))
    assert len(out) >= len(pcm)


def test_bit_crush_changes_samples_and_keeps_length():
    sr = 22050
    pcm = _sine_pcm_bytes(duration_s=0.1, sr=sr)
    out = audio_tools.bit_crush_audio_bytes(pcm, intensity=1.0)
    in_arr = np.frombuffer(pcm, dtype=np.int16)
    out_arr = np.frombuffer(out, dtype=np.int16)
    assert out_arr.shape == in_arr.shape
    assert not np.array_equal(out_arr, in_arr)


def test_gate_audio_bytes_on_wav_container_zeroes_range(tmp_path):
    sr = 8000
    dur = 0.1
    t = np.linspace(0, dur, int(sr * dur), endpoint=False)
    wave = 0.5 * np.sin(2 * np.pi * 440 * t).astype(np.float32)
    buf = io.BytesIO()
    sf.write(buf, wave, sr, format="WAV", subtype="PCM_16")
    wav_bytes = buf.getvalue()
    start_sec = 0.02
    end_sec = 0.04
    out_bytes = audio_tools.gate_audio_bytes(
        wav_bytes, start_sec=start_sec, end_sec=end_sec
    )
    out_buf = io.BytesIO(out_bytes)
    data, out_sr = sf.read(out_buf, dtype="float32", always_2d=False)
    assert out_sr == sr
    s_idx = int(round(start_sec * sr))
    e_idx = int(round(end_sec * sr))
    segment = data[s_idx:e_idx]
    assert np.allclose(segment, 0.0, atol=1e-4)


def _make_files(tmp_path, names):
    for n in names:
        p = tmp_path / n
        p.write_bytes(b"")
    return [str(tmp_path / n) for n in names]


def test_find_wav_files_prioritized_orders_priority_first(tmp_path):
    names = [
        "a.wav",
        "lead_vox.wav",
        "vocal_1.wav",
        "vox_lead.wav",
        "Vocal_solo.WAV",
        "notwav.txt",
    ]
    expected_wavs = sorted(
        (p for p in _make_files(tmp_path, names) if p.lower().endswith(".wav")),
        key=lambda p: os.path.basename(p).lower(),
    )
    found = find_wav_files_prioritized(str(tmp_path), prefix=None)
    priority_keywords = ["vocal", "vox"]
    priority = [
        p
        for p in expected_wavs
        if any(k in os.path.basename(p).lower() for k in priority_keywords)
    ]
    non_priority = [p for p in expected_wavs if p not in priority]
    expected_order = sorted(
        priority, key=lambda p: os.path.basename(p).lower()
    ) + sorted(non_priority, key=lambda p: os.path.basename(p).lower())

    assert found == expected_order


def test_extract_non_silent_segments_detects_tone():
    sr = 1000
    silence = np.zeros(500, dtype=np.float32)
    t = np.linspace(0, 0.5, 500, endpoint=False)
    tone = 0.5 * np.sin(2 * np.pi * 200 * t).astype(np.float32)
    audio = np.concatenate([silence, tone, silence])
    segments = audio_tools.extract_non_silent_segments(
        audio, sr, min_duration=0.1, top_db=20
    )
    assert any(len(seg) >= int(0.1 * sr) for seg in segments)
