"""
Unit tests for MultiClassRebracketingClassifier.

Tests model architecture, forward pass, predictions, and class weight computation.
"""

import pytest
import torch
from models.multiclass_classifier import (
    MultiClassRebracketingClassifier,
    MultiClassRainbowModel,
)


class TestMultiClassRebracketingClassifier:
    """Test multi-class classifier."""

    def setup_method(self):
        """Setup test fixtures."""
        self.input_dim = 768
        self.num_classes = 8
        self.batch_size = 4
        self.hidden_dims = [256, 128]

    def test_initialization(self):
        """Test classifier initialization."""
        classifier = MultiClassRebracketingClassifier(
            input_dim=self.input_dim,
            num_classes=self.num_classes,
            hidden_dims=self.hidden_dims,
            dropout=0.3,
            activation="relu",
            multi_label=False,
        )

        assert classifier.input_dim == self.input_dim
        assert classifier.num_classes == self.num_classes
        assert not classifier.multi_label

    def test_forward_pass_single_label(self):
        """Test forward pass for single-label classification."""
        classifier = MultiClassRebracketingClassifier(
            input_dim=self.input_dim,
            num_classes=self.num_classes,
            hidden_dims=self.hidden_dims,
            multi_label=False,
        )

        embeddings = torch.randn(self.batch_size, self.input_dim)
        logits = classifier(embeddings)

        assert logits.shape == (self.batch_size, self.num_classes)
        assert logits.dtype == torch.float32

    def test_forward_pass_multi_label(self):
        """Test forward pass for multi-label classification."""
        classifier = MultiClassRebracketingClassifier(
            input_dim=self.input_dim,
            num_classes=self.num_classes,
            hidden_dims=self.hidden_dims,
            multi_label=True,
        )

        embeddings = torch.randn(self.batch_size, self.input_dim)
        logits = classifier(embeddings)

        # Output shape should be same for both modes
        assert logits.shape == (self.batch_size, self.num_classes)

    def test_predict_single_label(self):
        """Test prediction for single-label mode."""
        classifier = MultiClassRebracketingClassifier(
            input_dim=self.input_dim,
            num_classes=self.num_classes,
            hidden_dims=self.hidden_dims,
            multi_label=False,
        )

        embeddings = torch.randn(self.batch_size, self.input_dim)
        predictions = classifier.predict(embeddings)

        assert predictions.shape == (self.batch_size,)
        assert predictions.dtype == torch.int64
        assert torch.all(predictions >= 0)
        assert torch.all(predictions < self.num_classes)

    def test_predict_multi_label(self):
        """Test prediction for multi-label mode."""
        classifier = MultiClassRebracketingClassifier(
            input_dim=self.input_dim,
            num_classes=self.num_classes,
            hidden_dims=self.hidden_dims,
            multi_label=True,
        )

        embeddings = torch.randn(self.batch_size, self.input_dim)
        predictions = classifier.predict(embeddings, threshold=0.5)

        assert predictions.shape == (self.batch_size, self.num_classes)
        assert predictions.dtype == torch.int64
        assert torch.all((predictions == 0) | (predictions == 1))

    def test_different_activations(self):
        """Test different activation functions."""
        for activation in ["relu", "gelu", "tanh"]:
            classifier = MultiClassRebracketingClassifier(
                input_dim=self.input_dim,
                num_classes=self.num_classes,
                hidden_dims=self.hidden_dims,
                activation=activation,
            )

            embeddings = torch.randn(self.batch_size, self.input_dim)
            logits = classifier(embeddings)

            assert logits.shape == (self.batch_size, self.num_classes)

    def test_with_class_weights(self):
        """Test classifier with class weights."""
        class_weights = torch.tensor([1.0, 2.0, 1.5, 0.5, 1.0, 1.2, 0.8, 1.1])

        classifier = MultiClassRebracketingClassifier(
            input_dim=self.input_dim,
            num_classes=self.num_classes,
            hidden_dims=self.hidden_dims,
            class_weights=class_weights,
        )

        assert torch.allclose(classifier.class_weights, class_weights)

    def test_compute_class_weights_balanced(self):
        """Test balanced class weight computation."""
        class_counts = {0: 100, 1: 50, 2: 200, 3: 25, 4: 150, 5: 75, 6: 300, 7: 10}

        weights = MultiClassRebracketingClassifier.compute_class_weights(
            class_counts=class_counts,
            num_classes=self.num_classes,
            mode="balanced",
        )

        assert weights.shape == (self.num_classes,)
        # Rare classes should have higher weights
        assert weights[7] > weights[6]  # Class 7 (10 samples) > Class 6 (300 samples)

    def test_compute_class_weights_uniform(self):
        """Test uniform class weight computation."""
        class_counts = {0: 100, 1: 50, 2: 200}

        weights = MultiClassRebracketingClassifier.compute_class_weights(
            class_counts=class_counts,
            num_classes=self.num_classes,
            mode="uniform",
        )

        assert weights.shape == (self.num_classes,)
        assert torch.all(weights == 1.0)

    def test_gradient_flow(self):
        """Test that gradients flow correctly."""
        classifier = MultiClassRebracketingClassifier(
            input_dim=self.input_dim,
            num_classes=self.num_classes,
            hidden_dims=self.hidden_dims,
        )

        embeddings = torch.randn(self.batch_size, self.input_dim, requires_grad=True)
        logits = classifier(embeddings)

        # Backprop
        loss = logits.sum()
        loss.backward()

        # Check gradients exist
        assert embeddings.grad is not None
        assert embeddings.grad.shape == embeddings.shape


class TestMultiClassRainbowModel:
    """Test complete multi-class Rainbow model."""

    def setup_method(self):
        """Setup test fixtures."""
        self.num_classes = 8
        self.batch_size = 2
        self.seq_len = 128

    def test_model_initialization(self):
        """Test model initialization."""
        from models.text_encoder import TextEncoder

        text_encoder = TextEncoder(
            model_name="microsoft/deberta-v3-base",
            pooling="mean",
        )

        classifier = MultiClassRebracketingClassifier(
            input_dim=text_encoder.hidden_size,
            num_classes=self.num_classes,
        )

        model = MultiClassRainbowModel(
            text_encoder=text_encoder,
            classifier=classifier,
        )

        assert model.text_encoder is not None
        assert model.classifier is not None

    def test_forward_pass(self):
        """Test complete forward pass."""
        from models.text_encoder import TextEncoder

        text_encoder = TextEncoder(
            model_name="microsoft/deberta-v3-base",
            pooling="mean",
        )

        classifier = MultiClassRebracketingClassifier(
            input_dim=text_encoder.hidden_size,
            num_classes=self.num_classes,
        )

        model = MultiClassRainbowModel(
            text_encoder=text_encoder,
            classifier=classifier,
        )

        input_ids = torch.randint(0, 1000, (self.batch_size, self.seq_len))
        attention_mask = torch.ones(self.batch_size, self.seq_len)

        with torch.no_grad():
            logits = model(input_ids, attention_mask)

        assert logits.shape == (self.batch_size, self.num_classes)

    def test_predict(self):
        """Test prediction method."""
        from models.text_encoder import TextEncoder

        text_encoder = TextEncoder(
            model_name="microsoft/deberta-v3-base",
            pooling="mean",
        )

        classifier = MultiClassRebracketingClassifier(
            input_dim=text_encoder.hidden_size,
            num_classes=self.num_classes,
            multi_label=False,
        )

        model = MultiClassRainbowModel(
            text_encoder=text_encoder,
            classifier=classifier,
        )

        input_ids = torch.randint(0, 1000, (self.batch_size, self.seq_len))
        attention_mask = torch.ones(self.batch_size, self.seq_len)

        predictions = model.predict(input_ids, attention_mask)

        assert predictions.shape == (self.batch_size,)
        assert torch.all(predictions >= 0)
        assert torch.all(predictions < self.num_classes)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
