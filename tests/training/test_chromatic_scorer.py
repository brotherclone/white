"""Tests for ChromaticScorer — fitness function for evolutionary music composition."""

import io
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_midi_bytes():
    """Create minimal valid MIDI bytes for testing."""
    import mido

    mid = mido.MidiFile()
    track = mido.MidiTrack()
    mid.tracks.append(track)
    track.append(mido.Message("note_on", note=60, velocity=100, time=0))
    track.append(mido.Message("note_off", note=60, velocity=0, time=480))
    track.append(mido.Message("note_on", note=64, velocity=80, time=0))
    track.append(mido.Message("note_off", note=64, velocity=0, time=480))
    track.append(mido.MetaMessage("end_of_track", time=0))

    buf = io.BytesIO()
    mid.save(file=buf)
    return buf.getvalue()


def _make_mock_session():
    """Create a mock ONNX InferenceSession that returns valid probability distributions."""
    session = MagicMock()

    def mock_run(output_names, inputs):
        batch_size = inputs["piano_roll"].shape[0]
        # Return valid softmax distributions + confidence
        temporal = np.full((batch_size, 3), 1 / 3, dtype=np.float32)
        spatial = np.full((batch_size, 3), 1 / 3, dtype=np.float32)
        ontological = np.full((batch_size, 3), 1 / 3, dtype=np.float32)
        # Vary confidence by batch index for ranking tests
        confidence = np.array(
            [[0.5 + 0.01 * i] for i in range(batch_size)], dtype=np.float32
        )
        return [temporal, spatial, ontological, confidence]

    session.run = mock_run
    return session


# ---------------------------------------------------------------------------
# Unit Tests (mocked ONNX — no model file needed)
# ---------------------------------------------------------------------------


class TestScorerWithMockedONNX:
    """Tests using a mocked ONNX session — verifies input shapes and output structure."""

    @pytest.fixture
    def scorer(self):
        from training.chromatic_scorer import ChromaticScorer

        s = ChromaticScorer.__new__(ChromaticScorer)
        s._session = _make_mock_session()
        s._deberta_tokenizer = None
        s._deberta_model = None
        s._clap_processor = None
        s._clap_model = None
        return s

    def test_score_with_precomputed_embeddings(self, scorer):
        """score() with all precomputed embeddings — no encoder loading needed."""
        result = scorer.score(
            midi_bytes=_make_midi_bytes(),
            concept_emb=np.random.randn(768).astype(np.float32),
            lyric_emb=np.random.randn(768).astype(np.float32),
        )

        assert "temporal" in result
        assert "spatial" in result
        assert "ontological" in result
        assert "confidence" in result

        # Verify mode keys
        assert set(result["temporal"].keys()) == {"past", "present", "future"}
        assert set(result["spatial"].keys()) == {"thing", "place", "person"}
        assert set(result["ontological"].keys()) == {"imagined", "forgotten", "known"}

    def test_score_output_is_valid_distribution(self, scorer):
        """Each dimension's probabilities should sum to ~1.0."""
        result = scorer.score(
            midi_bytes=_make_midi_bytes(),
            concept_emb=np.random.randn(768).astype(np.float32),
        )

        for dim in ["temporal", "spatial", "ontological"]:
            total = sum(result[dim].values())
            assert abs(total - 1.0) < 0.01, f"{dim} sum={total}"

        assert 0.0 <= result["confidence"] <= 1.0

    def test_score_midi_only(self, scorer):
        """MIDI-only scoring (no audio, no lyrics) — the primary evolutionary use case."""
        result = scorer.score(
            midi_bytes=_make_midi_bytes(),
            concept_emb=np.random.randn(768).astype(np.float32),
        )
        assert result["confidence"] >= 0.0

    def test_score_no_midi(self, scorer):
        """Scoring with no MIDI (concept-only) should still work via null embeddings."""
        result = scorer.score(
            concept_emb=np.random.randn(768).astype(np.float32),
        )
        assert "temporal" in result

    def test_score_requires_concept(self, scorer):
        """Must provide concept_text or concept_emb."""
        with pytest.raises(ValueError, match="concept"):
            scorer.score(midi_bytes=_make_midi_bytes())

    def test_score_batch_returns_ranked_list(self, scorer):
        """score_batch() returns results sorted by confidence descending."""
        concept_emb = np.random.randn(768).astype(np.float32)
        candidates = [{"midi_bytes": _make_midi_bytes()} for _ in range(10)]

        results = scorer.score_batch(candidates, concept_emb=concept_emb)

        assert len(results) == 10
        # Verify descending confidence order
        confidences = [r["confidence"] for r in results]
        assert confidences == sorted(confidences, reverse=True)
        # Verify ranks
        assert [r["rank"] for r in results] == list(range(10))

    def test_score_batch_preserves_candidate_reference(self, scorer):
        """Each result should reference its original candidate."""
        concept_emb = np.random.randn(768).astype(np.float32)
        midi = _make_midi_bytes()
        candidates = [{"midi_bytes": midi, "id": i} for i in range(5)]

        results = scorer.score_batch(candidates, concept_emb=concept_emb)

        # All original candidates should be present
        returned_ids = sorted(r["candidate"]["id"] for r in results)
        assert returned_ids == [0, 1, 2, 3, 4]

    def test_score_batch_empty(self, scorer):
        """Empty candidate list returns empty results."""
        concept_emb = np.random.randn(768).astype(np.float32)
        results = scorer.score_batch([], concept_emb=concept_emb)
        assert results == []

    def test_score_batch_shared_concept(self, scorer):
        """All candidates in a batch use the same concept embedding."""
        concept_emb = np.random.randn(768).astype(np.float32)

        # Capture the inputs passed to ONNX
        captured = {}
        original_run = scorer._session.run

        def capture_run(output_names, inputs):
            captured.update(inputs)
            return original_run(output_names, inputs)

        scorer._session.run = capture_run

        candidates = [{"midi_bytes": _make_midi_bytes()} for _ in range(3)]
        scorer.score_batch(candidates, concept_emb=concept_emb)

        # All rows should have the same concept embedding
        assert captured["concept_emb"].shape == (3, 768)
        np.testing.assert_array_equal(
            captured["concept_emb"][0], captured["concept_emb"][1]
        )
        np.testing.assert_array_equal(
            captured["concept_emb"][1], captured["concept_emb"][2]
        )

    def test_score_batch_input_shapes(self, scorer):
        """Verify all ONNX input shapes are correct."""
        captured = {}
        original_run = scorer._session.run

        def capture_run(output_names, inputs):
            captured.update(inputs)
            return original_run(output_names, inputs)

        scorer._session.run = capture_run

        concept_emb = np.random.randn(768).astype(np.float32)
        candidates = [{"midi_bytes": _make_midi_bytes()} for _ in range(4)]
        scorer.score_batch(candidates, concept_emb=concept_emb)

        assert captured["piano_roll"].shape == (4, 1, 128, 256)
        assert captured["audio_emb"].shape == (4, 512)
        assert captured["concept_emb"].shape == (4, 768)
        assert captured["lyric_emb"].shape == (4, 768)
        assert captured["has_audio"].shape == (4,)
        assert captured["has_midi"].shape == (4,)
        assert captured["has_lyric"].shape == (4,)

    def test_score_batch_has_midi_flags(self, scorer):
        """has_midi should be True only for candidates with valid MIDI."""
        captured = {}
        original_run = scorer._session.run

        def capture_run(output_names, inputs):
            captured.update(inputs)
            return original_run(output_names, inputs)

        scorer._session.run = capture_run

        concept_emb = np.random.randn(768).astype(np.float32)
        candidates = [
            {"midi_bytes": _make_midi_bytes()},
            {"midi_bytes": None},
            {"midi_bytes": _make_midi_bytes()},
        ]
        scorer.score_batch(candidates, concept_emb=concept_emb)

        assert captured["has_midi"].tolist() == [True, False, True]

    def test_score_batch_50_candidates(self, scorer):
        """Batch of 50 candidates — the target evolutionary batch size."""
        concept_emb = np.random.randn(768).astype(np.float32)
        midi = _make_midi_bytes()
        candidates = [{"midi_bytes": midi} for _ in range(50)]

        results = scorer.score_batch(candidates, concept_emb=concept_emb)

        assert len(results) == 50
        # All should have valid structure
        for r in results:
            assert 0.0 <= r["confidence"] <= 1.0
            assert r["rank"] >= 0


# ---------------------------------------------------------------------------
# Integration Tests (real ONNX model — skip if model not present)
# ---------------------------------------------------------------------------

_onnx_path = (
    Path(__file__).resolve().parent.parent.parent
    / "training"
    / "data"
    / "fusion_model.onnx"
)
_has_onnx = _onnx_path.exists()

try:
    import onnxruntime  # noqa: F401

    _has_ort = True
except ImportError:
    _has_ort = False


@pytest.mark.skipif(
    not (_has_onnx and _has_ort), reason="ONNX model or onnxruntime not available"
)
class TestScorerIntegration:
    """Integration tests using the real fusion_model.onnx."""

    @pytest.fixture
    def scorer(self):
        from training.chromatic_scorer import ChromaticScorer

        return ChromaticScorer(onnx_path=str(_onnx_path))

    def test_score_single_midi(self, scorer):
        """Score a single MIDI candidate with precomputed concept embedding."""
        concept_emb = np.random.randn(768).astype(np.float32)
        result = scorer.score(
            midi_bytes=_make_midi_bytes(),
            concept_emb=concept_emb,
        )

        # Valid probability distributions
        for dim in ["temporal", "spatial", "ontological"]:
            total = sum(result[dim].values())
            assert abs(total - 1.0) < 0.01, f"{dim} sum={total}"

        assert 0.0 <= result["confidence"] <= 1.0

    def test_score_concept_only(self, scorer):
        """Score with only concept (no MIDI, no audio) — null embeddings handle it."""
        concept_emb = np.random.randn(768).astype(np.float32)
        result = scorer.score(concept_emb=concept_emb)

        assert "temporal" in result
        assert 0.0 <= result["confidence"] <= 1.0

    def test_score_batch_50(self, scorer):
        """Batch of 50 candidates through the real ONNX model."""
        concept_emb = np.random.randn(768).astype(np.float32)
        midi = _make_midi_bytes()
        candidates = [{"midi_bytes": midi} for _ in range(50)]

        results = scorer.score_batch(candidates, concept_emb=concept_emb)

        assert len(results) == 50
        confidences = [r["confidence"] for r in results]
        assert confidences == sorted(confidences, reverse=True)

        for r in results:
            for dim in ["temporal", "spatial", "ontological"]:
                total = sum(r[dim].values())
                assert abs(total - 1.0) < 0.01

    def test_score_batch_mixed_modalities(self, scorer):
        """Batch with some MIDI-present and some MIDI-absent candidates."""
        concept_emb = np.random.randn(768).astype(np.float32)
        midi = _make_midi_bytes()
        candidates = [
            {"midi_bytes": midi},
            {"midi_bytes": None},
            {"midi_bytes": midi},
            {},
            {"midi_bytes": midi},
        ]

        results = scorer.score_batch(candidates, concept_emb=concept_emb)
        assert len(results) == 5
        # All should produce valid outputs
        for r in results:
            assert 0.0 <= r["confidence"] <= 1.0

    def test_prepare_concept_caching(self, scorer):
        """prepare_concept returns consistent embeddings for the same text."""
        # This test only runs if DeBERTa can be loaded; skip otherwise
        try:
            emb1 = scorer.prepare_concept("RED temporal=Past spatial=Thing")
            emb2 = scorer.prepare_concept("RED temporal=Past spatial=Thing")
        except Exception:
            pytest.skip("DeBERTa not available")

        assert emb1.shape == (768,)
        np.testing.assert_array_almost_equal(emb1, emb2)
