"""Tests for score_mix.py."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest
import yaml

from app.generators.midi.production.score_mix import (
    aggregate_chunk_scores,
    chromatic_drift_report,
    chunk_audio,
    score_mix,
    write_mix_score,
)

# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------


def _make_score_result(
    temporal=None, spatial=None, ontological=None, confidence=0.85, chromatic_match=0.72
):
    return {
        "temporal": temporal or {"past": 0.75, "present": 0.15, "future": 0.10},
        "spatial": spatial or {"thing": 0.70, "place": 0.20, "person": 0.10},
        "ontological": ontological
        or {"imagined": 0.10, "forgotten": 0.10, "known": 0.80},
        "confidence": confidence,
        "chromatic_match": chromatic_match,
    }


# Red target: temporal Past, spatial Thing, ontological Known
RED_TARGET = {
    "temporal": [0.8, 0.1, 0.1],
    "spatial": [0.8, 0.1, 0.1],
    "ontological": [0.1, 0.1, 0.8],
}


# ---------------------------------------------------------------------------
# 4.1 Unit: chromatic_drift_report
# ---------------------------------------------------------------------------


class TestChromaticDriftReport:

    def test_deltas_computed_for_all_three_dimensions(self):
        result = _make_score_result()
        drift = chromatic_drift_report(result, RED_TARGET)
        assert "temporal_delta" in drift
        assert "spatial_delta" in drift
        assert "ontological_delta" in drift
        assert "overall_drift" in drift

    def test_on_target_mix_near_zero(self):
        # Predicted distributions perfectly match Red target
        result = _make_score_result(
            temporal={"past": 0.8, "present": 0.1, "future": 0.1},
            spatial={"thing": 0.8, "place": 0.1, "person": 0.1},
            ontological={"imagined": 0.1, "forgotten": 0.1, "known": 0.8},
        )
        drift = chromatic_drift_report(result, RED_TARGET)
        assert abs(drift["temporal_delta"]) < 0.01
        assert abs(drift["spatial_delta"]) < 0.01
        assert abs(drift["ontological_delta"]) < 0.01
        assert drift["overall_drift"] < 0.01

    def test_signed_delta_negative_when_under_predicting(self):
        # Mix predicts only 0.5 Past when target expects 0.8
        result = _make_score_result(
            temporal={"past": 0.5, "present": 0.3, "future": 0.2},
        )
        drift = chromatic_drift_report(result, RED_TARGET)
        assert (
            drift["temporal_delta"] < 0
        ), "Under-prediction should yield negative delta"

    def test_signed_delta_positive_when_over_predicting(self):
        # Mix predicts 0.95 Past when target expects 0.8
        result = _make_score_result(
            temporal={"past": 0.95, "present": 0.03, "future": 0.02},
        )
        drift = chromatic_drift_report(result, RED_TARGET)
        assert (
            drift["temporal_delta"] > 0
        ), "Over-prediction should yield positive delta"

    def test_overall_drift_is_mean_absolute_delta(self):
        result = _make_score_result(
            temporal={"past": 0.6, "present": 0.2, "future": 0.2},  # delta = -0.2
            spatial={"thing": 0.9, "place": 0.05, "person": 0.05},  # delta = +0.1
            ontological={
                "imagined": 0.1,
                "forgotten": 0.1,
                "known": 0.8,
            },  # delta = 0.0
        )
        drift = chromatic_drift_report(result, RED_TARGET)
        expected = round((0.2 + 0.1 + 0.0) / 3, 4)
        assert abs(drift["overall_drift"] - expected) < 0.001

    def test_overall_drift_non_negative(self):
        result = _make_score_result()
        drift = chromatic_drift_report(result, RED_TARGET)
        assert drift["overall_drift"] >= 0.0


# ---------------------------------------------------------------------------
# 4.2 Unit: write_mix_score
# ---------------------------------------------------------------------------


class TestWriteMixScore:

    def test_file_written_to_melody_dir(self, tmp_path):
        melody_dir = tmp_path / "melody"
        result = _make_score_result()
        drift = {
            "temporal_delta": -0.05,
            "spatial_delta": 0.02,
            "ontological_delta": 0.0,
            "overall_drift": 0.023,
        }

        out = write_mix_score(result, drift, melody_dir, audio_path="/some/bounce.wav")

        assert out == melody_dir / "mix_score.yml"
        assert out.exists()

    def test_yaml_round_trips_correctly(self, tmp_path):
        melody_dir = tmp_path / "melody"
        result = _make_score_result()
        drift = {
            "temporal_delta": -0.1,
            "spatial_delta": 0.05,
            "ontological_delta": 0.0,
            "overall_drift": 0.05,
        }

        write_mix_score(result, drift, melody_dir)

        with open(melody_dir / "mix_score.yml") as f:
            loaded = yaml.safe_load(f)

        assert "temporal" in loaded
        assert "spatial" in loaded
        assert "ontological" in loaded
        assert "confidence" in loaded
        assert "chromatic_match" in loaded
        assert "drift" in loaded
        assert "metadata" in loaded

    def test_drift_nested_in_output(self, tmp_path):
        melody_dir = tmp_path / "melody"
        drift = {
            "temporal_delta": -0.1,
            "spatial_delta": 0.05,
            "ontological_delta": 0.0,
            "overall_drift": 0.05,
        }
        write_mix_score(_make_score_result(), drift, melody_dir)

        with open(melody_dir / "mix_score.yml") as f:
            loaded = yaml.safe_load(f)

        assert loaded["drift"]["overall_drift"] == 0.05
        assert loaded["drift"]["temporal_delta"] == -0.1

    def test_metadata_includes_audio_path_and_timestamp(self, tmp_path):
        melody_dir = tmp_path / "melody"
        audio = Path("/music/my_bounce.wav")
        write_mix_score(_make_score_result(), {}, melody_dir, audio_path=audio)

        with open(melody_dir / "mix_score.yml") as f:
            loaded = yaml.safe_load(f)

        assert loaded["metadata"]["audio_file"] == str(audio)
        assert loaded["metadata"]["timestamp"] is not None

    def test_existing_file_overwritten(self, tmp_path):
        melody_dir = tmp_path / "melody"
        melody_dir.mkdir()
        stale = melody_dir / "mix_score.yml"
        stale.write_text("stale: true\n")

        write_mix_score(_make_score_result(), {}, melody_dir)

        with open(stale) as f:
            loaded = yaml.safe_load(f)
        assert "stale" not in loaded

    def test_melody_dir_created_if_missing(self, tmp_path):
        melody_dir = tmp_path / "nested" / "melody"
        assert not melody_dir.exists()
        write_mix_score(_make_score_result(), {}, melody_dir)
        assert melody_dir.exists()


# ---------------------------------------------------------------------------
# 4.3 Integration: score_mix with stubbed Refractor
# ---------------------------------------------------------------------------


class TestScoreMixIntegration:

    def _make_production_dir(self, tmp_path: Path) -> Path:
        """Scaffold a minimal production directory with chords/review.yml."""
        prod = tmp_path / "production" / "my_song"
        (prod / "chords").mkdir(parents=True)
        (prod / "melody").mkdir(parents=True)

        # chords/review.yml pointing to a song proposal
        thread_dir = tmp_path / "thread"
        yml_dir = thread_dir / "yml"
        yml_dir.mkdir(parents=True)

        proposal = {
            "title": "Test Song",
            "bpm": 120,
            "time_sig": "4/4",
            "key": "C major",
            "rainbow_color": {"color_name": "Red"},
            "concept": "Test concept",
            "genres": ["rock"],
            "mood": "melancholic",
            "singer": "Gabriel",
        }
        (yml_dir / "test_proposal.yml").write_text(yaml.dump(proposal))

        review = {
            "thread": str(thread_dir),
            "song_proposal": "test_proposal.yml",
            "bpm": 120,
            "time_sig": "4/4",
        }
        (prod / "chords" / "review.yml").write_text(yaml.dump(review))

        return prod

    def _make_audio_file(self, tmp_path: Path) -> Path:
        """Create a dummy WAV file."""
        import struct
        import wave

        wav_path = tmp_path / "bounce.wav"
        with wave.open(str(wav_path), "w") as f:
            f.setnchannels(1)
            f.setsampwidth(2)
            f.setframerate(44100)
            # 0.1 seconds of silence
            f.writeframes(struct.pack("<" + "h" * 4410, *([0] * 4410)))
        return wav_path

    def _make_mock_scorer(self):
        """Build a Refractor mock that returns a fixed score."""
        scorer = MagicMock()
        scorer.prepare_audio.return_value = np.zeros(512, dtype=np.float32)
        scorer.prepare_concept.return_value = np.ones(768, dtype=np.float32) * 0.1
        scorer.score.return_value = {
            "temporal": {"past": 0.75, "present": 0.15, "future": 0.10},
            "spatial": {"thing": 0.70, "place": 0.20, "person": 0.10},
            "ontological": {"imagined": 0.10, "forgotten": 0.10, "known": 0.80},
            "confidence": 0.85,
            "rank": 0,
            "candidate": {},
        }
        return scorer

    def test_score_mix_writes_expected_structure(self, tmp_path):
        prod = self._make_production_dir(tmp_path)
        audio = self._make_audio_file(tmp_path)
        scorer = self._make_mock_scorer()

        score_result, drift = score_mix(audio, prod, _scorer=scorer)

        # Write out
        out = write_mix_score(score_result, drift, prod / "melody", audio_path=audio)
        assert out.exists()

        with open(out) as f:
            loaded = yaml.safe_load(f)

        assert set(loaded.keys()) >= {
            "temporal",
            "spatial",
            "ontological",
            "confidence",
            "chromatic_match",
            "drift",
            "metadata",
        }
        assert 0.0 <= loaded["confidence"] <= 1.0
        assert 0.0 <= loaded["chromatic_match"] <= 1.0
        assert "overall_drift" in loaded["drift"]

    def test_score_mix_passes_audio_and_concept_emb_to_scorer(self, tmp_path):
        prod = self._make_production_dir(tmp_path)
        audio = self._make_audio_file(tmp_path)
        scorer = self._make_mock_scorer()

        score_mix(audio, prod, _scorer=scorer)

        # Refractor.score should have been called with audio_emb and a real concept_emb
        call_kwargs = scorer.score.call_args.kwargs
        assert "audio_emb" in call_kwargs
        assert call_kwargs["audio_emb"] is not None
        # prepare_concept was called (concept is always required — never zeroed out)
        scorer.prepare_concept.assert_called_once()
        concept = call_kwargs.get("concept_emb")
        assert concept is not None
        assert not np.allclose(concept, np.zeros(768, dtype=np.float32))

    def test_score_mix_chromatic_match_in_range(self, tmp_path):
        prod = self._make_production_dir(tmp_path)
        audio = self._make_audio_file(tmp_path)
        scorer = self._make_mock_scorer()

        score_result, _ = score_mix(audio, prod, _scorer=scorer)

        assert "chromatic_match" in score_result
        assert 0.0 <= score_result["chromatic_match"] <= 1.0

    def test_score_mix_drift_keys_present(self, tmp_path):
        prod = self._make_production_dir(tmp_path)
        audio = self._make_audio_file(tmp_path)
        scorer = self._make_mock_scorer()

        _, drift = score_mix(audio, prod, _scorer=scorer)

        assert "temporal_delta" in drift
        assert "spatial_delta" in drift
        assert "ontological_delta" in drift
        assert "overall_drift" in drift

    def test_score_mix_produces_chunk_count(self, tmp_path):
        prod = self._make_production_dir(tmp_path)
        audio = self._make_audio_file(tmp_path)
        scorer = self._make_mock_scorer()

        score_result, _ = score_mix(audio, prod, _scorer=scorer)

        assert "chunk_count" in score_result
        assert score_result["chunk_count"] >= 1

    def test_score_mix_calls_scorer_per_chunk(self, tmp_path):
        """scorer.score should be called once per chunk (at least once)."""
        prod = self._make_production_dir(tmp_path)
        audio = self._make_audio_file(tmp_path)
        scorer = self._make_mock_scorer()

        score_result, _ = score_mix(audio, prod, _scorer=scorer)

        assert scorer.score.call_count == score_result["chunk_count"]

    def test_write_mix_score_includes_chunk_count(self, tmp_path):
        prod = self._make_production_dir(tmp_path)
        audio = self._make_audio_file(tmp_path)
        scorer = self._make_mock_scorer()

        score_result, drift = score_mix(audio, prod, _scorer=scorer)
        out = write_mix_score(score_result, drift, prod / "melody", audio_path=audio)

        with open(out) as f:
            loaded = yaml.safe_load(f)

        assert "chunk_count" in loaded
        assert loaded["chunk_count"] >= 1
        assert "chunk_stride_s" in loaded

    def test_write_mix_score_includes_refractor_cdm_metadata(self, tmp_path):
        melody_dir = tmp_path / "melody"
        result = _make_score_result()
        result["chunk_count"] = 3
        drift = {
            "temporal_delta": 0.0,
            "spatial_delta": 0.0,
            "ontological_delta": 0.0,
            "overall_drift": 0.0,
        }

        write_mix_score(
            result, drift, melody_dir, cdm_onnx_path="/path/to/refractor_cdm.onnx"
        )

        with open(melody_dir / "mix_score.yml") as f:
            loaded = yaml.safe_load(f)

        assert loaded["metadata"]["refractor_cdm"] == "/path/to/refractor_cdm.onnx"

    def test_write_mix_score_cdm_none_when_disabled(self, tmp_path):
        melody_dir = tmp_path / "melody"
        result = _make_score_result()
        result["chunk_count"] = 1
        drift = {}

        write_mix_score(result, drift, melody_dir, cdm_onnx_path=None)

        with open(melody_dir / "mix_score.yml") as f:
            loaded = yaml.safe_load(f)

        assert loaded["metadata"]["refractor_cdm"] is None


# ---------------------------------------------------------------------------
# chunk_audio unit tests
# ---------------------------------------------------------------------------


class TestChunkAudio:

    def test_single_chunk_for_short_audio(self):
        # 10s at 48000 Hz — shorter than 30s window
        sr = 48000
        waveform = np.zeros(sr * 10, dtype=np.float32)
        chunks = chunk_audio(waveform, sr)
        assert len(chunks) == 1

    def test_short_chunk_padded_to_chunk_length(self):
        sr = 48000
        waveform = np.zeros(sr * 10, dtype=np.float32)
        chunks = chunk_audio(waveform, sr, chunk_size_s=30.0)
        assert len(chunks[0]) == sr * 30

    def test_correct_count_for_60s_audio(self):
        # 60s audio, 30s chunks, 5s stride → starts at 0, 5, 10, ..., 55 → 12 chunks
        sr = 48000
        waveform = np.zeros(sr * 60, dtype=np.float32)
        chunks = chunk_audio(waveform, sr, chunk_size_s=30.0, stride_s=5.0)
        assert len(chunks) == 12

    def test_all_chunks_same_length(self):
        sr = 48000
        waveform = np.random.randn(sr * 90).astype(np.float32)
        chunks = chunk_audio(waveform, sr, chunk_size_s=30.0, stride_s=10.0)
        lengths = {len(c) for c in chunks}
        assert len(lengths) == 1

    def test_resamples_from_44100(self):
        sr = 44100
        duration_s = 10
        waveform = np.zeros(sr * duration_s, dtype=np.float32)
        chunks = chunk_audio(waveform, sr, chunk_size_s=30.0)
        # After resample to 48000, still shorter than chunk_size → 1 chunk of 30*48000
        assert len(chunks) == 1
        assert len(chunks[0]) == 48000 * 30

    def test_empty_waveform_returns_one_padded_chunk(self):
        sr = 48000
        waveform = np.zeros(0, dtype=np.float32)
        chunks = chunk_audio(waveform, sr)
        assert len(chunks) == 1
        assert len(chunks[0]) == int(30.0 * 48000)

    def test_custom_params(self):
        sr = 48000
        waveform = np.zeros(sr * 20, dtype=np.float32)
        chunks = chunk_audio(waveform, sr, chunk_size_s=10.0, stride_s=5.0)
        # 20s audio, 10s chunks, 5s stride → starts at 0, 5, 10, 15 → 4 chunks
        assert len(chunks) == 4

    def test_returns_float32(self):
        sr = 48000
        waveform = np.random.randn(sr * 60).astype(np.float64)  # float64 input
        chunks = chunk_audio(waveform, sr)
        assert all(c.dtype == np.float32 for c in chunks)


# ---------------------------------------------------------------------------
# aggregate_chunk_scores unit tests
# ---------------------------------------------------------------------------


def _chunk_result(confidence: float, temporal_past: float = 0.33) -> dict:
    rem = (1.0 - temporal_past) / 2.0
    return {
        "temporal": {"past": temporal_past, "present": rem, "future": rem},
        "spatial": {"thing": 0.33, "place": 0.33, "person": 0.34},
        "ontological": {"imagined": 0.33, "forgotten": 0.33, "known": 0.34},
        "confidence": confidence,
    }


class TestAggregateChunkScores:

    def test_single_chunk_passthrough(self):
        r = _chunk_result(0.8, temporal_past=0.9)
        agg = aggregate_chunk_scores([r])
        assert abs(agg["temporal"]["past"] - 0.9) < 1e-6
        assert agg["confidence"] == pytest.approx(0.8)
        assert agg["chunk_count"] == 1

    def test_confidence_is_mean_of_chunks(self):
        results = [_chunk_result(0.4), _chunk_result(0.8), _chunk_result(0.6)]
        agg = aggregate_chunk_scores(results)
        assert agg["confidence"] == pytest.approx((0.4 + 0.8 + 0.6) / 3, rel=1e-5)

    def test_higher_confidence_dominates(self):
        low = _chunk_result(0.1, temporal_past=0.1)
        high = _chunk_result(0.9, temporal_past=0.9)
        agg = aggregate_chunk_scores([low, high])
        # High-confidence chunk (0.9) should dominate
        assert agg["temporal"]["past"] > 0.7

    def test_zero_confidence_fallback_uniform_mean(self):
        r1 = _chunk_result(0.0, temporal_past=0.9)
        r2 = _chunk_result(0.0, temporal_past=0.1)
        agg = aggregate_chunk_scores([r1, r2])
        assert abs(agg["temporal"]["past"] - 0.5) < 1e-5

    def test_all_keys_present(self):
        agg = aggregate_chunk_scores([_chunk_result(0.5)])
        assert "temporal" in agg
        assert "spatial" in agg
        assert "ontological" in agg
        assert "confidence" in agg
        assert "chunk_count" in agg

    def test_raises_on_empty(self):
        with pytest.raises(ValueError):
            aggregate_chunk_scores([])

    def test_exact_weighted_math(self):
        r1 = _chunk_result(0.2, temporal_past=0.8)
        r2 = _chunk_result(0.8, temporal_past=0.2)
        agg = aggregate_chunk_scores([r1, r2])
        # weights: 0.2/(0.2+0.8)=0.2, 0.8/(0.2+0.8)=0.8
        expected_past = 0.2 * 0.8 + 0.8 * 0.2
        assert abs(agg["temporal"]["past"] - expected_past) < 1e-5
