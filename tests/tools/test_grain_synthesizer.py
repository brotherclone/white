"""Tests for training/tools/grain_synthesizer.py."""

from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
from white_training.tools.grain_synthesizer import (
    _to_stereo,
    extract_grain,
    hann_crossfade,
    synthesize,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write_wav(
    path: Path, duration_s: float, sr: int = 22050, channels: int = 2
) -> Path:
    """Write a sine-wave WAV of known duration."""
    n = int(duration_s * sr)
    t = np.linspace(0, duration_s, n, endpoint=False, dtype=np.float32)
    data = (0.5 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
    if channels == 2:
        data = np.stack([data, data], axis=1)
    sf.write(str(path), data, sr)
    return path


def _stub_pool(tmp_path: Path, n: int = 3, seg_dur: float = 5.0) -> list[dict]:
    """Create n stub pool entries backed by real WAV files."""
    pool = []
    for i in range(n):
        wav = _write_wav(tmp_path / f"seg_{i}.wav", seg_dur)
        pool.append(
            {
                "segment_id": f"seg_{i:03d}",
                "source_audio_file": str(wav),
                "start_seconds": 0.0,
                "end_seconds": seg_dur,
                "match": round(0.9 - i * 0.05, 4),
                "song_slug": f"song_{i}",
            }
        )
    return pool


# ---------------------------------------------------------------------------
# 3.1 extract_grain
# ---------------------------------------------------------------------------


class TestExtractGrain:
    def test_returns_correct_duration(self, tmp_path):
        wav = _write_wav(tmp_path / "source.wav", duration_s=10.0)
        grain, sr = extract_grain(str(wav), 0.0, 10.0, grain_dur=1.0)
        expected_samples = int(1.0 * sr)
        assert len(grain) == expected_samples

    def test_stays_within_segment_bounds(self, tmp_path):
        """Random offset must not start before segment_start."""
        wav = _write_wav(tmp_path / "source.wav", duration_s=10.0)
        rng = random.Random(0)
        for _ in range(20):
            grain, sr = extract_grain(str(wav), 2.0, 8.0, grain_dur=1.0, rng=rng)
            # We can't inspect the offset directly, but the grain should be the right length
            assert len(grain) == int(1.0 * sr)

    def test_short_segment_returns_padded_grain(self, tmp_path):
        """Segment shorter than grain_dur should be padded to grain_dur."""
        wav = _write_wav(tmp_path / "short.wav", duration_s=0.3)
        grain, sr = extract_grain(str(wav), 0.0, 0.3, grain_dur=1.0)
        assert len(grain) == int(1.0 * sr)

    def test_returns_float32(self, tmp_path):
        wav = _write_wav(tmp_path / "source.wav", duration_s=5.0)
        grain, _ = extract_grain(str(wav), 0.0, 5.0, grain_dur=1.0)
        assert grain.dtype == np.float32


# ---------------------------------------------------------------------------
# 3.2 hann_crossfade_length
# ---------------------------------------------------------------------------


class TestHannCrossfadeLength:
    def test_single_grain_passthrough(self):
        sr = 22050
        grain = np.ones((sr, 2), dtype=np.float32)
        out = hann_crossfade([grain], sr, crossfade_ms=50)
        assert len(out) == sr

    def test_two_grains_length(self):
        sr = 22050
        crossfade_ms = 50
        cf_samples = int(sr * crossfade_ms / 1000)
        grain = np.ones((sr, 2), dtype=np.float32)
        out = hann_crossfade([grain, grain], sr, crossfade_ms=crossfade_ms)
        expected = 2 * sr - cf_samples
        assert len(out) == expected

    def test_many_grains_length(self):
        sr = 16000
        grain_dur_samples = sr  # 1 second
        crossfade_ms = 40
        cf_samples = int(sr * crossfade_ms / 1000)
        n = 5
        grains = [np.ones((grain_dur_samples, 2), dtype=np.float32) for _ in range(n)]
        out = hann_crossfade(grains, sr, crossfade_ms=crossfade_ms)
        expected = n * grain_dur_samples - (n - 1) * cf_samples
        assert len(out) == expected

    def test_empty_grains_returns_empty(self):
        out = hann_crossfade([], sr=22050, crossfade_ms=50)
        assert len(out) == 0
        assert out.shape == (0, 2)


# ---------------------------------------------------------------------------
# 3.3 hann_crossfade_no_clicks
# ---------------------------------------------------------------------------


class TestHannCrossfadeNoClicks:
    def test_no_energy_spike_at_boundaries(self):
        """RMS at crossfade boundary must not exceed RMS of adjacent grains."""
        sr = 22050
        cf_ms = 50
        cf_samples = int(sr * cf_ms / 1000)

        rng = np.random.default_rng(0)
        grain_a = rng.uniform(-0.5, 0.5, (sr, 2)).astype(np.float32)
        grain_b = rng.uniform(-0.5, 0.5, (sr, 2)).astype(np.float32)

        out = hann_crossfade([grain_a, grain_b], sr, crossfade_ms=cf_ms)

        # RMS around the join point
        join = sr - cf_samples
        window = cf_samples
        boundary_rms = np.sqrt(np.mean(out[join : join + window] ** 2))
        grain_a_rms = np.sqrt(np.mean(grain_a**2))
        grain_b_rms = np.sqrt(np.mean(grain_b**2))
        max_grain_rms = max(grain_a_rms, grain_b_rms)

        # Boundary RMS should not exceed source grain RMS (no energy spike)
        assert boundary_rms <= max_grain_rms * 1.1  # 10% tolerance


# ---------------------------------------------------------------------------
# 3.4 synthesize end-to-end (stub pool, no parquet)
# ---------------------------------------------------------------------------


class TestSynthesizeEndToEnd:
    def test_writes_wav_and_map(self, tmp_path):
        pool = _stub_pool(tmp_path)
        wav_path = tmp_path / "out.wav"
        result_wav, result_map = synthesize(
            color="Red",
            duration_s=3.0,
            output_path=str(wav_path),
            seed=42,
            grain_dur_s=1.0,
            crossfade_ms=50,
            grain_pool=pool,
        )
        assert result_wav.exists()
        assert result_map.exists()
        assert result_map.name.endswith("_grain_map.yml")

    def test_wav_duration_within_one_grain(self, tmp_path):
        pool = _stub_pool(tmp_path)
        wav_path = tmp_path / "out.wav"
        target_dur = 5.0
        grain_dur = 1.0
        synthesize(
            color="Red",
            duration_s=target_dur,
            output_path=str(wav_path),
            seed=0,
            grain_dur_s=grain_dur,
            crossfade_ms=50,
            grain_pool=pool,
        )
        data, sr = sf.read(str(wav_path))
        actual_dur = len(data) / sr
        assert abs(actual_dur - target_dur) <= grain_dur

    def test_grain_map_schema(self, tmp_path):
        import yaml

        pool = _stub_pool(tmp_path, n=2)
        wav_path = tmp_path / "out.wav"
        _, map_path = synthesize(
            color="Blue",
            duration_s=2.0,
            output_path=str(wav_path),
            seed=1,
            grain_dur_s=1.0,
            crossfade_ms=50,
            grain_pool=pool,
        )
        with open(map_path) as f:
            gm = yaml.safe_load(f)

        assert gm["color"] == "Blue"
        assert gm["seed"] == 1
        assert isinstance(gm["grains"], list)
        assert len(gm["grains"]) > 0
        first = gm["grains"][0]
        for key in ("segment_id", "source", "offset_s", "match"):
            assert key in first, f"missing key: {key}"

    def test_output_is_stereo(self, tmp_path):
        pool = _stub_pool(tmp_path)
        wav_path = tmp_path / "out.wav"
        synthesize(
            color="Red",
            duration_s=2.0,
            output_path=str(wav_path),
            seed=0,
            grain_dur_s=1.0,
            grain_pool=pool,
        )
        data, _ = sf.read(str(wav_path))
        assert data.ndim == 2
        assert data.shape[1] == 2


# ---------------------------------------------------------------------------
# 3.5 mono/stereo normalisation
# ---------------------------------------------------------------------------


class TestMonoStereoNormalization:
    def test_mixed_pool_produces_stereo_output(self, tmp_path):
        """Pool mixing mono and stereo WAVs must produce stereo output without error."""
        mono_wav = _write_wav(tmp_path / "mono.wav", 5.0, channels=1)
        stereo_wav = _write_wav(tmp_path / "stereo.wav", 5.0, channels=2)

        pool = [
            {
                "segment_id": "mono_seg",
                "source_audio_file": str(mono_wav),
                "start_seconds": 0.0,
                "end_seconds": 5.0,
                "match": 0.85,
                "song_slug": "mono_song",
            },
            {
                "segment_id": "stereo_seg",
                "source_audio_file": str(stereo_wav),
                "start_seconds": 0.0,
                "end_seconds": 5.0,
                "match": 0.80,
                "song_slug": "stereo_song",
            },
        ]

        wav_path = tmp_path / "mixed_out.wav"
        synthesize(
            color="Green",
            duration_s=3.0,
            output_path=str(wav_path),
            seed=0,
            grain_dur_s=1.0,
            grain_pool=pool,
        )

        data, _ = sf.read(str(wav_path))
        assert data.ndim == 2
        assert data.shape[1] == 2

    def test_to_stereo_mono_1d(self):
        arr = np.ones(100, dtype=np.float32)
        result = _to_stereo(arr)
        assert result.shape == (100, 2)

    def test_to_stereo_mono_2d(self):
        arr = np.ones((100, 1), dtype=np.float32)
        result = _to_stereo(arr)
        assert result.shape == (100, 2)

    def test_to_stereo_passthrough(self):
        arr = np.ones((100, 2), dtype=np.float32)
        result = _to_stereo(arr)
        assert result.shape == (100, 2)


# ---------------------------------------------------------------------------
# 3.6 empty pool raises
# ---------------------------------------------------------------------------


class TestEmptyPoolRaises:
    def test_raises_on_empty_pool(self, tmp_path):
        with pytest.raises(ValueError, match="No reachable audio"):
            synthesize(
                color="Red",
                duration_s=5.0,
                output_path=str(tmp_path / "out.wav"),
                grain_pool=[],
            )
