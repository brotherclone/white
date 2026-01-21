"""
Unit tests for LabelEncoder.

Tests label encoding/decoding, class distribution analysis,
and edge cases.
"""

import pytest
import numpy as np
import torch
from core.multiclass_pipeline import LabelEncoder


class TestLabelEncoder:
    """Test label encoder."""

    def setup_method(self):
        """Setup test fixtures."""
        self.class_mapping = {
            "spatial": 0,
            "temporal": 1,
            "causal": 2,
            "perceptual": 3,
            "memory": 4,
            "ontological": 5,
            "narrative": 6,
            "identity": 7,
        }
        self.encoder = LabelEncoder(self.class_mapping)

    def test_initialization(self):
        """Test encoder initialization."""
        assert self.encoder.num_classes == 8
        assert len(self.encoder.class_mapping) == 8
        assert len(self.encoder.inverse_mapping) == 8

    def test_encode_single_label(self):
        """Test encoding a single label."""
        idx = self.encoder.encode("spatial", multi_label=False)
        assert idx == 0

        idx = self.encoder.encode("temporal", multi_label=False)
        assert idx == 1

        idx = self.encoder.encode("identity", multi_label=False)
        assert idx == 7

    def test_decode_single_label(self):
        """Test decoding a single label."""
        label = self.encoder.decode(0, multi_label=False)
        assert label == "spatial"

        label = self.encoder.decode(7, multi_label=False)
        assert label == "identity"

    def test_encode_decode_roundtrip(self):
        """Test that encode-decode is reversible."""
        for label_name in self.class_mapping.keys():
            idx = self.encoder.encode(label_name, multi_label=False)
            decoded = self.encoder.decode(idx, multi_label=False)
            assert decoded == label_name

    def test_encode_multi_label(self):
        """Test encoding multiple labels."""
        labels = ["spatial", "temporal"]
        vector = self.encoder.encode(labels, multi_label=True)

        assert vector.shape == (8,)
        assert vector[0] == 1.0  # spatial
        assert vector[1] == 1.0  # temporal
        assert vector[2] == 0.0  # causal
        assert np.sum(vector) == 2.0

    def test_decode_multi_label(self):
        """Test decoding multiple labels."""
        vector = np.array([1, 0, 1, 0, 0, 1, 0, 0], dtype=np.float32)
        labels = self.encoder.decode(vector, multi_label=True)

        assert len(labels) == 3
        assert "spatial" in labels
        assert "causal" in labels
        assert "ontological" in labels

    def test_decode_torch_tensor(self):
        """Test decoding torch tensors."""
        idx_tensor = torch.tensor(3)
        label = self.encoder.decode(idx_tensor, multi_label=False)
        assert label == "perceptual"

        vector_tensor = torch.tensor([0, 1, 0, 0, 1, 0, 0, 0], dtype=torch.float32)
        labels = self.encoder.decode(vector_tensor, multi_label=True)
        assert "temporal" in labels
        assert "memory" in labels

    def test_get_class_distribution_single_label(self):
        """Test class distribution for single-label."""
        labels = ["spatial", "temporal", "spatial", "causal", "spatial"]
        dist = self.encoder.get_class_distribution(labels, multi_label=False)

        assert dist["spatial"] == 3
        assert dist["temporal"] == 1
        assert dist["causal"] == 1

    def test_get_class_distribution_multi_label(self):
        """Test class distribution for multi-label."""
        labels = [
            ["spatial", "temporal"],
            ["spatial"],
            ["causal", "perceptual", "memory"],
        ]
        dist = self.encoder.get_class_distribution(labels, multi_label=True)

        assert dist["spatial"] == 2
        assert dist["temporal"] == 1
        assert dist["causal"] == 1
        assert dist["perceptual"] == 1
        assert dist["memory"] == 1

    def test_unknown_label_raises_error(self):
        """Test that unknown labels raise errors."""
        with pytest.raises(ValueError, match="Unknown label"):
            self.encoder.encode("unknown_type", multi_label=False)

    def test_unknown_index_raises_error(self):
        """Test that unknown indices raise errors."""
        with pytest.raises(ValueError, match="Unknown class index"):
            self.encoder.decode(99, multi_label=False)

    def test_encode_list_as_single_label(self):
        """Test encoding a list in single-label mode (takes first element)."""
        labels_list = ["temporal", "spatial"]
        idx = self.encoder.encode(labels_list, multi_label=False)
        # Should take first element
        assert idx == 1  # temporal

    def test_empty_list_handling(self):
        """Test handling of empty lists."""
        # Empty list in multi-label should give zero vector
        vector = self.encoder.encode([], multi_label=True)
        assert np.sum(vector) == 0

    def test_mixed_case_sensitivity(self):
        """Test that encoder is case-sensitive."""
        # This should fail because "Spatial" != "spatial"
        with pytest.raises(ValueError):
            self.encoder.encode("Spatial", multi_label=False)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
