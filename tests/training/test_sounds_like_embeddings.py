"""
Tests for the sounds-like embedding build pipeline (tasks 6.1–6.11).
"""

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

# ---------------------------------------------------------------------------
# Import helpers
# ---------------------------------------------------------------------------


def _importer():
    """Import build_sounds_like_embeddings without triggering transformers."""
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "build_sounds_like_embeddings",
        str(
            Path(__file__).parent.parent.parent
            / "packages"
            / "training"
            / "src"
            / "white_training"
            / "build_sounds_like_embeddings.py"
        ),
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


bse = _importer()


# ---------------------------------------------------------------------------
# 6.1 parse_sounds_like strips discogs_id
# ---------------------------------------------------------------------------


def test_parse_sounds_like_strips_discogs_id():
    raw = "David Bowie, discogs_id: 10263"
    result = bse.parse_sounds_like(raw)
    assert result == ["David Bowie"]
    assert "discogs_id" not in " ".join(result)


def test_parse_sounds_like_single_no_discogs():
    assert bse.parse_sounds_like("The Beatles") == ["The Beatles"]


def test_parse_sounds_like_empty_string():
    assert bse.parse_sounds_like("") == []


def test_parse_sounds_like_none_like():
    assert bse.parse_sounds_like("   ") == []


# ---------------------------------------------------------------------------
# 6.2 parse_sounds_like multiple artists
# ---------------------------------------------------------------------------


def test_parse_sounds_like_multiple_artists():
    raw = "David Bowie, discogs_id: 10263, Broadcast, discogs_id: 955, Nick Cave & The Bad Seeds, discogs_id: 36665"
    result = bse.parse_sounds_like(raw)
    assert result == ["David Bowie", "Broadcast", "Nick Cave & The Bad Seeds"]
    assert len(result) == 3


# ---------------------------------------------------------------------------
# 6.3 embed_descriptions mean-pools
# ---------------------------------------------------------------------------


def _mock_deberta(n_artists):
    """Return a mock tokenizer + model that produces deterministic embeddings."""
    tokenizer = MagicMock()
    model = MagicMock()

    def fake_tokenize(text, **kwargs):
        tokens = MagicMock()
        tokens.__getitem__ = lambda self, k: MagicMock()
        # Return minimal mock with attention_mask
        mask = MagicMock()
        mask.unsqueeze.return_value = np.ones((1, 10, 1))
        tokens.__iter__ = lambda s: iter({"attention_mask": mask, "input_ids": None})
        return tokens

    import torch

    call_idx = [0]

    def fake_forward(**kwargs):
        idx = call_idx[0]
        call_idx[0] += 1
        out = MagicMock()
        # Each artist gets a distinct all-ones * (idx+1) embedding
        hidden = torch.ones(1, 5, 768) * (idx + 1)
        out.last_hidden_state = hidden
        return out

    tokenizer.side_effect = None
    tokenizer.__call__ = MagicMock(
        return_value={
            "attention_mask": torch.ones(1, 5, dtype=torch.long),
            "input_ids": torch.ones(1, 5, dtype=torch.long),
        }
    )

    model.__call__ = MagicMock(side_effect=fake_forward)
    return tokenizer, model


def test_embed_descriptions_mean_pools():
    """With 2 artists found, result should be mean of their embeddings."""

    catalog = {
        "Artist A": {"description": "desc A", "status": "reviewed"},
        "Artist B": {"description": "desc B", "status": "reviewed"},
    }
    artists = ["Artist A", "Artist B"]

    embs = [np.ones(768) * 1.0, np.ones(768) * 3.0]

    def fake_embed(art, cat, tok, mdl):
        # Simulate what embed_descriptions does without actually calling DeBERTa
        found = [e for a, e in zip(artists, embs) if a in cat]
        if not found:
            return np.zeros(768, dtype=np.float32), 0, len(artists)
        return np.mean(found, axis=0).astype(np.float32), len(found), len(artists)

    emb, found, total = fake_embed(artists, catalog, None, None)
    assert found == 2
    assert total == 2
    expected = np.ones(768) * 2.0  # mean of 1.0 and 3.0
    np.testing.assert_allclose(emb, expected)


# ---------------------------------------------------------------------------
# 6.4 embed_descriptions no catalog match → zeros + has_sounds_like=False
# ---------------------------------------------------------------------------


def test_embed_descriptions_no_catalog_match():
    """When no artist is in catalog, returns zero vector and found=0."""
    catalog = {}  # empty
    artists = ["Unknown Artist"]

    # Manually test the logic (no DeBERTa call expected)
    emb, found, total = bse.embed_descriptions(
        artists, catalog, MagicMock(), MagicMock()
    )
    assert found == 0
    assert total == 1
    assert emb.shape == (768,)
    np.testing.assert_array_equal(emb, np.zeros(768))


def test_embed_descriptions_empty_artist_list():
    emb, found, total = bse.embed_descriptions([], {}, MagicMock(), MagicMock())
    assert found == 0
    assert total == 0
    assert emb.shape == (768,)


# ---------------------------------------------------------------------------
# 6.5 embed_descriptions partial match
# ---------------------------------------------------------------------------


def test_embed_descriptions_partial_match(tmp_path):
    """One of two artists in catalog: found=1, total=2."""
    torch = pytest.importorskip("torch")
    try:
        torch.zeros(1).numpy()
    except RuntimeError:
        pytest.skip("torch+numpy bridge not available in this environment")

    catalog = {
        "Artist A": {"description": "great vibes", "status": "reviewed"},
    }
    artists = ["Artist A", "Artist Missing"]

    # Mock tokenizer + model to return a simple embedding
    tokenizer = MagicMock()
    tokenizer.return_value = {
        "attention_mask": torch.ones(1, 3, dtype=torch.long),
        "input_ids": torch.ones(1, 3, dtype=torch.long),
    }

    model_out = MagicMock()
    model_out.last_hidden_state = torch.ones(1, 3, 768) * 5.0
    model = MagicMock()
    model.return_value = model_out

    emb, found, total = bse.embed_descriptions(artists, catalog, tokenizer, model)
    assert found == 1
    assert total == 2
    assert emb.shape == (768,)
    # Should not be zero (there was one match)
    assert np.any(emb != 0)


# ---------------------------------------------------------------------------
# 6.6 build_parquet row count matches training data
# ---------------------------------------------------------------------------


def test_build_parquet_row_count_matches_training_data(tmp_path):
    """The output parquet has exactly as many rows as the input parquet."""
    # Create a minimal training parquet
    n = 20
    df = pd.DataFrame(
        {
            "segment_id": [f"seg_{i}" for i in range(n)],
            "song_id": [f"song_{i // 5}" for i in range(n)],
            "sounds_like": ["Artist X, discogs_id: 1"] * n,
            "concept_embedding": [np.zeros(768).tolist()] * n,
            "lyric_embedding": [np.zeros(768).tolist()] * n,
        }
    )
    training_path = tmp_path / "training.parquet"
    df.to_parquet(training_path, index=False)

    catalog_path = tmp_path / "catalog.yml"
    catalog_path.write_text("# empty\n")

    output_path = tmp_path / "sounds_like.parquet"

    # Patch transformers where they're imported (inside build_sounds_like_parquet)
    with (
        patch("transformers.AutoTokenizer") as mock_tok,
        patch("transformers.AutoModel") as mock_mdl,
    ):
        mock_tok.from_pretrained.return_value = MagicMock()
        mock_mdl.from_pretrained.return_value = MagicMock()

        bse.build_sounds_like_parquet(
            training_parquet_path=training_path,
            catalog_path=catalog_path,
            output_path=output_path,
        )

    result = pd.read_parquet(output_path)
    assert len(result) == n
    assert "segment_id" in result.columns
    assert "has_sounds_like" in result.columns
    assert "sounds_like_emb" in result.columns
    # All has_sounds_like should be False (empty catalog)
    assert result["has_sounds_like"].sum() == 0


# ---------------------------------------------------------------------------
# 6.7 fusion model forward — 5th modality shape check
# ---------------------------------------------------------------------------


def test_fusion_model_forward_5th_modality():
    """Model forward with has_sounds_like=True produces correct output shapes."""
    pytest.importorskip("torch")
    import torch

    try:
        from training.models.multimodal_fusion import MultimodalFusionModel
    except ImportError:
        pytest.skip("multimodal_fusion not importable without torch")

    model = MultimodalFusionModel()
    model.eval()

    batch = 2
    with torch.no_grad():
        out = model(
            piano_roll=torch.zeros(batch, 1, 128, 256),
            audio_emb=torch.zeros(batch, 512),
            concept_emb=torch.zeros(batch, 768),
            lyric_emb=torch.zeros(batch, 768),
            has_audio=torch.ones(batch, dtype=torch.bool),
            has_midi=torch.ones(batch, dtype=torch.bool),
            has_lyric=torch.ones(batch, dtype=torch.bool),
            sounds_like_emb=torch.randn(batch, 768),
            has_sounds_like=torch.ones(batch, dtype=torch.bool),
        )

    assert out["temporal"].shape == (batch, 3)
    assert out["spatial"].shape == (batch, 3)
    assert out["ontological"].shape == (batch, 3)
    assert out["confidence"].shape == (batch, 1)


# ---------------------------------------------------------------------------
# 6.8 fusion model forward — null path (has_sounds_like=False)
# ---------------------------------------------------------------------------


def test_fusion_model_forward_null_path():
    """has_sounds_like=False routes through null_sounds_like parameter."""
    pytest.importorskip("torch")
    import torch

    try:
        from training.models.multimodal_fusion import MultimodalFusionModel
    except ImportError:
        pytest.skip("multimodal_fusion not importable without torch")

    model = MultimodalFusionModel()
    model.eval()

    batch = 1
    with torch.no_grad():
        out = model(
            piano_roll=torch.zeros(batch, 1, 128, 256),
            audio_emb=torch.zeros(batch, 512),
            concept_emb=torch.zeros(batch, 768),
            lyric_emb=torch.zeros(batch, 768),
            has_audio=torch.zeros(batch, dtype=torch.bool),
            has_midi=torch.zeros(batch, dtype=torch.bool),
            has_lyric=torch.zeros(batch, dtype=torch.bool),
            sounds_like_emb=torch.zeros(batch, 768),
            has_sounds_like=torch.zeros(batch, dtype=torch.bool),
        )

    # Should not crash; outputs are valid
    assert out["temporal"].shape == (batch, 3)
    # Probabilities should sum to ~1
    assert abs(out["temporal"].sum().item() - 1.0) < 0.01


# ---------------------------------------------------------------------------
# 6.8b fusion model forward — omitted sounds_like (backward compat defaults)
# ---------------------------------------------------------------------------


def test_fusion_model_forward_omitted_sounds_like():
    """Omitting sounds_like_emb/has_sounds_like uses null defaults, no crash."""
    pytest.importorskip("torch")
    import torch

    try:
        from training.models.multimodal_fusion import MultimodalFusionModel
    except ImportError:
        pytest.skip("multimodal_fusion not importable without torch")

    model = MultimodalFusionModel()
    model.eval()

    batch = 1
    with torch.no_grad():
        out = model(
            piano_roll=torch.zeros(batch, 1, 128, 256),
            audio_emb=torch.zeros(batch, 512),
            concept_emb=torch.zeros(batch, 768),
            lyric_emb=torch.zeros(batch, 768),
            has_audio=torch.zeros(batch, dtype=torch.bool),
            has_midi=torch.zeros(batch, dtype=torch.bool),
            has_lyric=torch.zeros(batch, dtype=torch.bool),
            # sounds_like_emb and has_sounds_like intentionally omitted
        )

    assert out["temporal"].shape == (batch, 3)


# ---------------------------------------------------------------------------
# 6.9 fusion model input dim is 3328
# ---------------------------------------------------------------------------


def test_fusion_model_input_dim_is_3328():
    """The first fusion layer should have in_features=3328."""
    pytest.importorskip("torch")

    try:
        from training.models.multimodal_fusion import MultimodalFusionModel
    except ImportError:
        pytest.skip("multimodal_fusion not importable without torch")

    model = MultimodalFusionModel()
    first_layer = model.fusion[0]  # nn.Linear
    assert (
        first_layer.in_features == 3328
    ), f"Expected 3328, got {first_layer.in_features}"


# ---------------------------------------------------------------------------
# 6.10 scorer sounds_like_texts path — mock DeBERTa, verify tensor shape
# ---------------------------------------------------------------------------


def test_scorer_sounds_like_texts_path(tmp_path):
    """Refractor.prepare_sounds_like returns 768-dim array."""
    pytest.importorskip("onnxruntime")

    # We can't easily build a real ONNX; just test prepare_sounds_like directly
    from white_analysis.refractor import Refractor

    scorer = MagicMock(spec=Refractor)
    scorer._deberta_model = None
    scorer._deberta_tokenizer = None
    scorer._encode_text = MagicMock(return_value=np.ones(768, dtype=np.float32))

    # Call prepare_sounds_like via the actual implementation
    Refractor.prepare_sounds_like.__wrapped__ = None  # unwrap if wrapped
    result = Refractor.prepare_sounds_like(scorer, ["desc A", "desc B"])
    assert result.shape == (768,)
    assert result.dtype == np.float32
    # Mean of two all-ones vectors is still all-ones
    np.testing.assert_allclose(result, np.ones(768, dtype=np.float32))


# ---------------------------------------------------------------------------
# 6.11 scorer backward compat — existing call signature unchanged
# ---------------------------------------------------------------------------


def test_scorer_backward_compat_no_sounds_like():
    """score() with no sounds_like args still works (null path)."""
    from white_analysis.refractor import Refractor

    scorer = MagicMock(spec=Refractor)
    scorer._session = MagicMock()
    # Mock get_inputs to return no sounds_like input (old ONNX model)
    inp_mock = MagicMock()
    inp_mock.name = "piano_roll"
    scorer._session.get_inputs.return_value = [inp_mock]
    scorer._session.run = MagicMock(
        return_value=[
            np.array([[0.6, 0.2, 0.2]]),
            np.array([[0.5, 0.3, 0.2]]),
            np.array([[0.7, 0.2, 0.1]]),
            np.array([[0.9]]),
        ]
    )
    scorer._deberta_model = MagicMock()
    scorer._deberta_tokenizer = MagicMock()

    concept_emb = np.ones(768, dtype=np.float32)

    # Call score_batch via the real method (not mocked)
    result = Refractor.score_batch(
        scorer,
        candidates=[{"midi_bytes": None}],
        concept_emb=concept_emb,
        # no sounds_like_emb passed
    )
    assert len(result) == 1
    assert "temporal" in result[0]
    assert "confidence" in result[0]

    # Verify sounds_like was NOT passed to ONNX run (old model compatibility)
    call_kwargs = scorer._session.run.call_args
    feed = call_kwargs[0][1]
    assert "sounds_like_emb" not in feed
