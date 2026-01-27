"""
Unit tests for regression components.

Tests RegressionHead, RainbowTableRegressionHead, MultiTaskModel,
and related utilities.
"""

import pytest
import torch


class TestRegressionHead:
    """Test RegressionHead module."""

    def setup_method(self):
        """Setup test fixtures."""
        self.input_dim = 768
        self.batch_size = 4
        self.hidden_dims = [256, 128]

    def test_initialization(self):
        """Test regression head initialization."""
        from models.regression_head import RegressionHead

        head = RegressionHead(
            input_dim=self.input_dim,
            num_targets=3,
            hidden_dims=self.hidden_dims,
            dropout=0.3,
            output_activation="sigmoid",
        )

        assert head.input_dim == self.input_dim
        assert head.num_targets == 3
        assert head.output_activation == "sigmoid"

    def test_forward_pass_bounded(self):
        """Test forward pass with sigmoid activation."""
        from models.regression_head import RegressionHead

        head = RegressionHead(
            input_dim=self.input_dim,
            num_targets=3,
            hidden_dims=self.hidden_dims,
            output_activation="sigmoid",
        )

        embeddings = torch.randn(self.batch_size, self.input_dim)
        predictions, variance = head(embeddings)

        assert predictions.shape == (self.batch_size, 3)
        assert variance is None  # No uncertainty by default
        assert torch.all(predictions >= 0) and torch.all(predictions <= 1)

    def test_forward_pass_unbounded(self):
        """Test forward pass without activation."""
        from models.regression_head import RegressionHead

        head = RegressionHead(
            input_dim=self.input_dim,
            num_targets=2,
            hidden_dims=self.hidden_dims,
            output_activation=None,
        )

        embeddings = torch.randn(self.batch_size, self.input_dim)
        predictions, variance = head(embeddings)

        assert predictions.shape == (self.batch_size, 2)
        # Unbounded can have any values
        assert predictions.dtype == torch.float32

    def test_uncertainty_estimation(self):
        """Test uncertainty prediction."""
        from models.regression_head import RegressionHead

        head = RegressionHead(
            input_dim=self.input_dim,
            num_targets=2,
            hidden_dims=self.hidden_dims,
            output_activation="sigmoid",
            predict_uncertainty=True,
        )

        embeddings = torch.randn(self.batch_size, self.input_dim)
        predictions, variance = head(embeddings)

        assert predictions.shape == (self.batch_size, 2)
        assert variance is not None
        assert variance.shape == (self.batch_size, 2)
        assert torch.all(variance > 0)  # Variance must be positive

    def test_predict_method(self):
        """Test predict convenience method."""
        from models.regression_head import RegressionHead

        head = RegressionHead(
            input_dim=self.input_dim,
            num_targets=1,
            hidden_dims=self.hidden_dims,
            output_activation="sigmoid",
            predict_uncertainty=True,
        )

        embeddings = torch.randn(self.batch_size, self.input_dim)

        # Without uncertainty
        preds = head.predict(embeddings, return_uncertainty=False)
        assert preds.shape == (self.batch_size, 1)

        # With uncertainty
        preds, var = head.predict(embeddings, return_uncertainty=True)
        assert preds.shape == (self.batch_size, 1)
        assert var.shape == (self.batch_size, 1)

    def test_gradient_flow(self):
        """Test gradient flow through regression head."""
        from models.regression_head import RegressionHead

        head = RegressionHead(
            input_dim=self.input_dim,
            num_targets=3,
            hidden_dims=self.hidden_dims,
        )

        embeddings = torch.randn(self.batch_size, self.input_dim, requires_grad=True)
        predictions, _ = head(embeddings)

        loss = predictions.sum()
        loss.backward()

        assert embeddings.grad is not None
        assert embeddings.grad.shape == embeddings.shape


class TestMultiTargetRegressionHead:
    """Test MultiTargetRegressionHead module."""

    def setup_method(self):
        """Setup test fixtures."""
        self.input_dim = 768
        self.batch_size = 4

    def test_multi_target_initialization(self):
        """Test multi-target head initialization."""
        from models.regression_head import MultiTargetRegressionHead

        head = MultiTargetRegressionHead(
            input_dim=self.input_dim,
            target_configs={
                "intensity": {"num_targets": 1, "activation": "sigmoid"},
                "fluidity": {"num_targets": 1, "activation": "sigmoid"},
                "complexity": {"num_targets": 1, "activation": None},
            },
            shared_hidden_dims=[256],
        )

        assert "intensity" in head.target_names
        assert "fluidity" in head.target_names
        assert "complexity" in head.target_names

    def test_multi_target_forward(self):
        """Test multi-target forward pass."""
        from models.regression_head import MultiTargetRegressionHead

        head = MultiTargetRegressionHead(
            input_dim=self.input_dim,
            target_configs={
                "intensity": {"num_targets": 1, "activation": "sigmoid"},
                "temporal": {"num_targets": 3, "activation": "softmax"},
            },
            shared_hidden_dims=[256],
        )

        embeddings = torch.randn(self.batch_size, self.input_dim)
        outputs = head(embeddings)

        assert "intensity" in outputs
        assert "temporal" in outputs
        assert outputs["intensity"].shape == (self.batch_size, 1)
        assert outputs["temporal"].shape == (self.batch_size, 3)

        # Check intensity is bounded
        assert torch.all(outputs["intensity"] >= 0)
        assert torch.all(outputs["intensity"] <= 1)

        # Check temporal sums to 1 (softmax)
        assert torch.allclose(
            outputs["temporal"].sum(dim=-1),
            torch.ones(self.batch_size),
            atol=1e-5,
        )


class TestRainbowTableRegressionHead:
    """Test RainbowTableRegressionHead module."""

    def setup_method(self):
        """Setup test fixtures."""
        self.input_dim = 768
        self.batch_size = 4
        self.hidden_dims = [256, 128]

    def test_initialization(self):
        """Test Rainbow Table head initialization."""
        from models.rainbow_table_regression_head import RainbowTableRegressionHead

        head = RainbowTableRegressionHead(
            input_dim=self.input_dim,
            hidden_dims=self.hidden_dims,
            dropout=0.3,
        )

        assert head.input_dim == self.input_dim

    def test_forward_pass_shapes(self):
        """Test output shapes."""
        from models.rainbow_table_regression_head import RainbowTableRegressionHead

        head = RainbowTableRegressionHead(
            input_dim=self.input_dim,
            hidden_dims=self.hidden_dims,
        )

        embeddings = torch.randn(self.batch_size, self.input_dim)
        scores = head(embeddings)

        assert scores.temporal_scores.shape == (self.batch_size, 3)
        assert scores.spatial_scores.shape == (self.batch_size, 3)
        assert scores.ontological_scores.shape == (self.batch_size, 3)
        assert scores.chromatic_confidence.shape == (self.batch_size, 1)

    def test_softmax_constraint(self):
        """Test that dimension scores sum to 1."""
        from models.rainbow_table_regression_head import RainbowTableRegressionHead

        head = RainbowTableRegressionHead(
            input_dim=self.input_dim,
            hidden_dims=self.hidden_dims,
        )

        embeddings = torch.randn(self.batch_size, self.input_dim)
        scores = head(embeddings)

        # Each dimension should sum to 1
        assert torch.allclose(
            scores.temporal_scores.sum(dim=-1),
            torch.ones(self.batch_size),
            atol=1e-5,
        )
        assert torch.allclose(
            scores.spatial_scores.sum(dim=-1),
            torch.ones(self.batch_size),
            atol=1e-5,
        )
        assert torch.allclose(
            scores.ontological_scores.sum(dim=-1),
            torch.ones(self.batch_size),
            atol=1e-5,
        )

    def test_confidence_bounded(self):
        """Test confidence is bounded [0, 1]."""
        from models.rainbow_table_regression_head import RainbowTableRegressionHead

        head = RainbowTableRegressionHead(
            input_dim=self.input_dim,
            hidden_dims=self.hidden_dims,
        )

        embeddings = torch.randn(self.batch_size, self.input_dim)
        scores = head(embeddings)

        assert torch.all(scores.chromatic_confidence >= 0)
        assert torch.all(scores.chromatic_confidence <= 1)

    def test_mode_prediction(self):
        """Test mode prediction from scores."""
        from models.rainbow_table_regression_head import RainbowTableRegressionHead

        head = RainbowTableRegressionHead(
            input_dim=self.input_dim,
            hidden_dims=self.hidden_dims,
        )

        embeddings = torch.randn(self.batch_size, self.input_dim)
        scores = head(embeddings)

        temporal, spatial, ontological = head.get_dominant_modes(scores)

        assert len(temporal) == self.batch_size
        assert len(spatial) == self.batch_size
        assert len(ontological) == self.batch_size

        # Check valid modes
        valid_temporal = ["past", "present", "future"]
        valid_spatial = ["thing", "place", "person"]
        valid_ontological = ["imagined", "forgotten", "known"]

        for t, s, o in zip(temporal, spatial, ontological):
            assert t in valid_temporal
            assert s in valid_spatial
            assert o in valid_ontological

    def test_album_prediction(self):
        """Test album prediction from scores."""
        from models.rainbow_table_regression_head import RainbowTableRegressionHead

        head = RainbowTableRegressionHead(
            input_dim=self.input_dim,
            hidden_dims=self.hidden_dims,
        )

        embeddings = torch.randn(self.batch_size, self.input_dim)
        scores = head(embeddings)

        albums = head.predict_album(scores)

        assert len(albums) == self.batch_size

        valid_albums = [
            "Orange",
            "Red",
            "Violet",
            "Yellow",
            "Green",
            "Indigo",
            "Blue",
            "Black",
        ]
        for album in albums:
            assert album in valid_albums

    def test_combined_mode_string(self):
        """Test combined mode string generation."""
        from models.rainbow_table_regression_head import RainbowTableRegressionHead

        head = RainbowTableRegressionHead(
            input_dim=self.input_dim,
            hidden_dims=self.hidden_dims,
        )

        embeddings = torch.randn(self.batch_size, self.input_dim)
        scores = head(embeddings)

        modes = head.predict_combined_mode(scores)

        assert len(modes) == self.batch_size
        for mode in modes:
            # Should be format "Temporal_Spatial_Ontological"
            parts = mode.split("_")
            assert len(parts) == 3

    def test_with_uncertainty(self):
        """Test uncertainty estimation."""
        from models.rainbow_table_regression_head import RainbowTableRegressionHead

        head = RainbowTableRegressionHead(
            input_dim=self.input_dim,
            hidden_dims=self.hidden_dims,
            predict_uncertainty=True,
        )

        embeddings = torch.randn(self.batch_size, self.input_dim)
        scores = head(embeddings)

        assert scores.temporal_uncertainty is not None
        assert scores.spatial_uncertainty is not None
        assert scores.ontological_uncertainty is not None

        assert scores.temporal_uncertainty.shape == (self.batch_size, 3)
        assert torch.all(scores.temporal_uncertainty > 0)

    def test_temperature_scaling(self):
        """Test temperature parameter for softmax."""
        from models.rainbow_table_regression_head import RainbowTableRegressionHead

        head = RainbowTableRegressionHead(
            input_dim=self.input_dim,
            hidden_dims=self.hidden_dims,
        )

        embeddings = torch.randn(self.batch_size, self.input_dim)

        # Low temperature = sharper distribution
        scores_sharp = head(embeddings, temperature=0.5)
        # High temperature = softer distribution
        scores_soft = head(embeddings, temperature=2.0)

        # Sharp should have higher max values
        sharp_max = scores_sharp.temporal_scores.max(dim=-1).values.mean()
        soft_max = scores_soft.temporal_scores.max(dim=-1).values.mean()

        assert sharp_max > soft_max


class TestHybridStateDetector:
    """Test HybridStateDetector."""

    def setup_method(self):
        """Setup test fixtures."""
        from models.rainbow_table_regression_head import HybridStateDetector

        self.detector = HybridStateDetector(
            dominant_threshold=0.6,
            hybrid_margin=0.15,
            diffuse_threshold=0.2,
        )

    def test_dominant_detection(self):
        """Test detection of dominant state."""
        from models.rainbow_table_regression_head import TEMPORAL_MODES

        # Clear dominant: past=0.9
        scores = torch.tensor([[0.9, 0.05, 0.05]])
        results = self.detector.detect_state(scores, TEMPORAL_MODES)

        assert results[0]["state"] == "dominant"
        assert results[0]["top_mode"] == "past"

    def test_hybrid_detection(self):
        """Test detection of hybrid state."""
        from models.rainbow_table_regression_head import TEMPORAL_MODES

        # Hybrid: past=0.45, present=0.45, future=0.1
        scores = torch.tensor([[0.45, 0.45, 0.1]])
        results = self.detector.detect_state(scores, TEMPORAL_MODES)

        assert results[0]["state"] == "hybrid"

    def test_diffuse_detection(self):
        """Test detection of diffuse state."""
        from models.rainbow_table_regression_head import TEMPORAL_MODES

        # Diffuse: roughly uniform
        scores = torch.tensor([[0.34, 0.33, 0.33]])
        results = self.detector.detect_state(scores, TEMPORAL_MODES)

        assert results[0]["state"] == "diffuse"

    def test_full_state_analysis(self):
        """Test full ontological state analysis."""
        from models.rainbow_table_regression_head import (
            OntologicalScores,
        )

        # Create mock scores
        scores = OntologicalScores(
            temporal_scores=torch.tensor([[0.9, 0.05, 0.05]]),
            spatial_scores=torch.tensor([[0.8, 0.15, 0.05]]),
            ontological_scores=torch.tensor([[0.7, 0.2, 0.1]]),
            chromatic_confidence=torch.tensor([[0.85]]),
        )

        analysis = self.detector.analyze_full_state(scores)

        assert len(analysis) == 1
        assert "overall_state" in analysis[0]
        assert "chromatic_confidence" in analysis[0]
        assert "hybrid_flags" in analysis[0]


class TestTransmigrationCalculator:
    """Test TransmigrationCalculator."""

    def test_dimension_distance(self):
        """Test distance computation between score vectors."""
        from models.rainbow_table_regression_head import TransmigrationCalculator

        source = torch.tensor([[1.0, 0.0, 0.0]])
        target = torch.tensor([[0.0, 1.0, 0.0]])

        dist = TransmigrationCalculator.dimension_distance(source, target)

        # L2 distance should be sqrt(2) ≈ 1.414
        assert torch.isclose(dist, torch.tensor([1.414]), atol=0.01)

    def test_total_distance(self):
        """Test total transmigration distance."""
        from models.rainbow_table_regression_head import (
            TransmigrationCalculator,
            OntologicalScores,
        )

        source = OntologicalScores(
            temporal_scores=torch.tensor([[1.0, 0.0, 0.0]]),
            spatial_scores=torch.tensor([[1.0, 0.0, 0.0]]),
            ontological_scores=torch.tensor([[1.0, 0.0, 0.0]]),
            chromatic_confidence=torch.tensor([[0.9]]),
        )

        target = OntologicalScores(
            temporal_scores=torch.tensor([[0.0, 1.0, 0.0]]),
            spatial_scores=torch.tensor([[0.0, 1.0, 0.0]]),
            ontological_scores=torch.tensor([[0.0, 1.0, 0.0]]),
            chromatic_confidence=torch.tensor([[0.9]]),
        )

        dist = TransmigrationCalculator.total_distance(source, target)

        # sqrt(3 * 2) ≈ 2.449
        assert dist.item() > 2.0

    def test_distances_to_all_albums(self):
        """Test computing distances to all albums."""
        from models.rainbow_table_regression_head import (
            TransmigrationCalculator,
            OntologicalScores,
        )

        # Orange album concept: Past_Thing_Imagined
        scores = OntologicalScores(
            temporal_scores=torch.tensor([[0.9, 0.05, 0.05]]),
            spatial_scores=torch.tensor([[0.9, 0.05, 0.05]]),
            ontological_scores=torch.tensor([[0.9, 0.05, 0.05]]),
            chromatic_confidence=torch.tensor([[0.9]]),
        )

        distances = TransmigrationCalculator.distances_to_all_albums(scores)

        assert "Orange" in distances
        assert "Blue" in distances
        assert "Black" in distances

        # Should be closest to Orange
        assert distances["Orange"].item() < distances["Blue"].item()


class TestOntologicalScoresDataclass:
    """Test OntologicalScores dataclass."""

    def test_to_dict(self):
        """Test conversion to dictionary."""
        from models.rainbow_table_regression_head import OntologicalScores

        scores = OntologicalScores(
            temporal_scores=torch.tensor([[0.5, 0.3, 0.2]]),
            spatial_scores=torch.tensor([[0.6, 0.2, 0.2]]),
            ontological_scores=torch.tensor([[0.7, 0.2, 0.1]]),
            chromatic_confidence=torch.tensor([[0.8]]),
        )

        d = scores.to_dict()

        assert "temporal_scores" in d
        assert "spatial_scores" in d
        assert "ontological_scores" in d
        assert "chromatic_confidence" in d

    def test_detach(self):
        """Test detach method."""
        from models.rainbow_table_regression_head import OntologicalScores

        scores = OntologicalScores(
            temporal_scores=torch.tensor([[0.5, 0.3, 0.2]], requires_grad=True),
            spatial_scores=torch.tensor([[0.6, 0.2, 0.2]], requires_grad=True),
            ontological_scores=torch.tensor([[0.7, 0.2, 0.1]], requires_grad=True),
            chromatic_confidence=torch.tensor([[0.8]], requires_grad=True),
        )

        detached = scores.detach()

        assert not detached.temporal_scores.requires_grad
        assert not detached.spatial_scores.requires_grad


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
