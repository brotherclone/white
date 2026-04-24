"""Tests for RefractorCDMModel architecture and ONNX export."""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest

try:
    import torch

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    import onnx  # noqa: F401
    import onnxruntime  # noqa: F401

    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False

pytestmark = pytest.mark.skipif(not TORCH_AVAILABLE, reason="torch not installed")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _load_model(**kwargs):
    # Load directly to avoid training/models/__init__.py eager imports (circular import)
    import importlib.util
    from pathlib import Path

    spec = importlib.util.spec_from_file_location(
        "refractor_cdm_model",
        Path(__file__).parent.parent.parent
        / "packages"
        / "training"
        / "src"
        / "white_training"
        / "models"
        / "refractor_cdm_model.py",
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.RefractorCDMModel(**kwargs)


def _load_export_onnx():
    import importlib.util
    from pathlib import Path

    spec = importlib.util.spec_from_file_location(
        "refractor_cdm_model",
        Path(__file__).parent.parent.parent
        / "packages"
        / "training"
        / "src"
        / "white_training"
        / "models"
        / "refractor_cdm_model.py",
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.export_onnx


def _rand_batch(batch: int = 4, clap_dim: int = 512, concept_dim: int = 768):
    return (
        torch.randn(batch, clap_dim),
        torch.randn(batch, concept_dim),
    )


# ---------------------------------------------------------------------------
# Forward pass shape tests
# ---------------------------------------------------------------------------


class TestRefractorCDMModelForward:

    def test_output_shapes_with_concept(self):
        model = _load_model()
        model.eval()
        clap, concept = _rand_batch(batch=4)
        with torch.no_grad():
            t, s, o = model(clap, concept)
        assert t.shape == (4, 3)
        assert s.shape == (4, 3)
        assert o.shape == (4, 3)

    def test_output_shapes_without_concept(self):
        model = _load_model(use_concept=False)
        model.eval()
        clap, _ = _rand_batch(batch=4)
        with torch.no_grad():
            t, s, o = model(clap)
        assert t.shape == (4, 3)
        assert s.shape == (4, 3)
        assert o.shape == (4, 3)

    def test_outputs_are_probability_distributions(self):
        model = _load_model()
        model.eval()
        clap, concept = _rand_batch(batch=8)
        with torch.no_grad():
            t, s, o = model(clap, concept)
        for dist in (t, s, o):
            sums = dist.sum(dim=-1)
            assert torch.allclose(
                sums, torch.ones(8), atol=1e-5
            ), "outputs must sum to 1"
            assert (dist >= 0).all(), "outputs must be non-negative"

    def test_outputs_are_independent(self):
        """The three heads should produce distinct outputs for the same input."""
        model = _load_model()
        model.eval()
        clap, concept = _rand_batch(batch=2)
        with torch.no_grad():
            t, s, o = model(clap, concept)
        # All three identical would be suspicious — they should differ
        assert not torch.allclose(t, s), "temporal and spatial heads should differ"
        assert not torch.allclose(t, o), "temporal and ontological heads should differ"

    def test_single_item_batch(self):
        model = _load_model()
        model.eval()
        clap, concept = _rand_batch(batch=1)
        with torch.no_grad():
            t, s, o = model(clap, concept)
        assert t.shape == (1, 3)
        assert s.shape == (1, 3)
        assert o.shape == (1, 3)

    def test_custom_hidden_dims(self):
        model = _load_model(hidden_dims=[512, 256, 128])
        model.eval()
        clap, concept = _rand_batch(batch=2)
        with torch.no_grad():
            t, s, o = model(clap, concept)
        assert t.shape == (2, 3)

    def test_concept_none_when_use_concept_false(self):
        """Passing concept_emb=None with use_concept=False should not raise."""
        model = _load_model(use_concept=False)
        model.eval()
        clap, _ = _rand_batch(batch=2)
        with torch.no_grad():
            t, s, o = model(clap, concept_emb=None)
        assert t.shape == (2, 3)


# ---------------------------------------------------------------------------
# ONNX round-trip tests
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not ONNX_AVAILABLE, reason="onnx/onnxruntime not installed")
class TestRefractorCDMModelOnnxExport:

    @pytest.fixture()
    def trained_model(self):
        model = _load_model()
        model.eval()
        return model

    def _export_to_bytes(self, model, use_concept=True):
        export_onnx = _load_export_onnx()
        with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as f:
            path = f.name
        export_onnx(model, path, use_concept=use_concept)
        data = Path(path).read_bytes()
        Path(path).unlink()
        return data

    def test_onnx_export_produces_bytes(self, trained_model):
        data = self._export_to_bytes(trained_model)
        assert len(data) > 0

    def test_onnx_output_names(self, trained_model):
        import onnx

        data = self._export_to_bytes(trained_model)
        model_proto = onnx.load_from_string(data)
        output_names = [o.name for o in model_proto.graph.output]
        assert "temporal" in output_names
        assert "spatial" in output_names
        assert "ontological" in output_names

    def test_onnx_round_trip_matches_torch(self, trained_model):
        import onnxruntime as ort

        data = self._export_to_bytes(trained_model)
        session = ort.InferenceSession(data, providers=["CPUExecutionProvider"])

        clap = np.random.randn(2, 512).astype(np.float32)
        concept = np.random.randn(2, 768).astype(np.float32)

        # ONNX inference (concatenated input)
        x = np.concatenate([clap, concept], axis=-1)
        ort_out = session.run(None, {"input": x})
        t_ort, s_ort, o_ort = ort_out

        # PyTorch inference
        with torch.no_grad():
            t_pt, s_pt, o_pt = trained_model(
                torch.FloatTensor(clap.tolist()),
                torch.FloatTensor(concept.tolist()),
            )

        np.testing.assert_allclose(t_ort, t_pt.numpy(), atol=1e-5)
        np.testing.assert_allclose(s_ort, s_pt.numpy(), atol=1e-5)
        np.testing.assert_allclose(o_ort, o_pt.numpy(), atol=1e-5)

    def test_onnx_export_no_concept(self, trained_model):
        """ONNX export without concept embedding should accept clap_emb-only input."""
        import onnxruntime as ort

        model = _load_model(use_concept=False)
        model.eval()
        data = self._export_to_bytes(model, use_concept=False)
        session = ort.InferenceSession(data, providers=["CPUExecutionProvider"])

        clap = np.random.randn(3, 512).astype(np.float32)
        ort_out = session.run(None, {"input": clap})
        assert len(ort_out) == 3
        assert ort_out[0].shape == (3, 3)

    def test_onnx_dynamic_batch(self, trained_model):
        """ONNX model should handle variable batch sizes."""
        import onnxruntime as ort

        data = self._export_to_bytes(trained_model)
        session = ort.InferenceSession(data, providers=["CPUExecutionProvider"])

        for batch in (1, 4, 16):
            x = np.random.randn(batch, 512 + 768).astype(np.float32)
            ort_out = session.run(None, {"input": x})
            assert ort_out[0].shape == (batch, 3)
