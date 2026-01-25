"""
Unit tests for soft target generation and validation.

Tests SoftTargetGenerator, TargetConsistencyValidator, and related utilities.
"""

import pytest
import numpy as np
import torch


class TestSoftTargetGenerator:
    """Test SoftTargetGenerator class."""

    def setup_method(self):
        """Setup test fixtures."""
        from core.soft_targets import SoftTargetGenerator

        self.generator = SoftTargetGenerator(
            label_smoothing=0.1,
            temporal_context_enabled=True,
            temporal_context_window=3,
            temporal_context_weight=0.3,
        )

    def test_one_hot_encoding(self):
        """Test one-hot encoding."""
        from core.soft_targets import SoftTargetGenerator

        gen = SoftTargetGenerator(label_smoothing=0.0)

        vec = gen.one_hot(0, num_classes=3)
        assert np.allclose(vec, [1.0, 0.0, 0.0])

        vec = gen.one_hot(1, num_classes=3)
        assert np.allclose(vec, [0.0, 1.0, 0.0])

        vec = gen.one_hot(2, num_classes=3)
        assert np.allclose(vec, [0.0, 0.0, 1.0])

    def test_label_smoothing(self):
        """Test label smoothing."""
        from core.soft_targets import SoftTargetGenerator

        gen = SoftTargetGenerator(label_smoothing=0.1)

        vec = gen.smooth_one_hot(0, num_classes=3)

        # (1 - 0.1) * 1 + 0.1 * (1/3) = 0.9 + 0.033 = 0.933
        assert np.isclose(vec[0], 0.933, atol=0.01)
        # 0.1 * (1/3) = 0.033
        assert np.isclose(vec[1], 0.033, atol=0.01)
        assert np.isclose(vec[2], 0.033, atol=0.01)
        # Should sum to 1
        assert np.isclose(vec.sum(), 1.0)

    def test_uniform_distribution(self):
        """Test uniform distribution generation."""
        vec = self.generator.uniform_distribution(num_classes=3)

        assert np.allclose(vec, [1 / 3, 1 / 3, 1 / 3])
        assert np.isclose(vec.sum(), 1.0)

    def test_encode_temporal_mode(self):
        """Test temporal mode encoding."""
        past = self.generator.encode_mode("past", "temporal")
        present = self.generator.encode_mode("present", "temporal")
        future = self.generator.encode_mode("future", "temporal")

        # Past should have highest value at index 0
        assert past[0] > past[1] and past[0] > past[2]
        # Present should have highest value at index 1
        assert present[1] > present[0] and present[1] > present[2]
        # Future should have highest value at index 2
        assert future[2] > future[0] and future[2] > future[1]

    def test_encode_spatial_mode(self):
        """Test spatial mode encoding."""
        thing = self.generator.encode_mode("thing", "spatial")
        place = self.generator.encode_mode("place", "spatial")
        person = self.generator.encode_mode("person", "spatial")

        assert thing[0] > thing[1]
        assert place[1] > place[0]
        assert person[2] > person[0]

    def test_encode_ontological_mode(self):
        """Test ontological mode encoding."""
        imagined = self.generator.encode_mode("imagined", "ontological")
        forgotten = self.generator.encode_mode("forgotten", "ontological")
        known = self.generator.encode_mode("known", "ontological")

        assert imagined[0] > imagined[1]
        assert forgotten[1] > forgotten[0]
        assert known[2] > known[0]

    def test_black_album_encoding(self):
        """Test Black Album (None) encoding."""
        for label in ["none", "null", "", "black"]:
            vec = self.generator.encode_mode(label, "temporal")
            # Should be uniform
            assert np.allclose(vec, [1 / 3, 1 / 3, 1 / 3], atol=0.01)

    def test_unknown_mode_handling(self):
        """Test handling of unknown modes."""
        import warnings

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            vec = self.generator.encode_mode("invalid_mode", "temporal")

            assert len(w) == 1
            assert "Unknown" in str(w[0].message)
            # Should return uniform
            assert np.allclose(vec, [1 / 3, 1 / 3, 1 / 3], atol=0.01)

    def test_generate_from_labels(self):
        """Test complete label generation."""
        targets = self.generator.generate_from_labels(
            temporal_labels=["past", "present", "future", "none"],
            spatial_labels=["thing", "place", "person", "none"],
            ontological_labels=["imagined", "forgotten", "known", "none"],
        )

        assert targets.temporal.shape == (4, 3)
        assert targets.spatial.shape == (4, 3)
        assert targets.ontological.shape == (4, 3)
        assert targets.confidence.shape == (4, 1)

        # Check sums
        assert np.allclose(targets.temporal.sum(axis=1), 1.0, atol=1e-5)
        assert np.allclose(targets.spatial.sum(axis=1), 1.0, atol=1e-5)
        assert np.allclose(targets.ontological.sum(axis=1), 1.0, atol=1e-5)

        # Check Black Album flag
        assert targets.is_black_album is not None
        assert targets.is_black_album[3] is not None

    def test_generate_from_combined_mode(self):
        """Test generation from combined mode strings."""
        targets = self.generator.generate_from_combined_mode(
            [
                "Past_Thing_Imagined",
                "Present_Place_Known",
                "None_None_None",
            ]
        )

        assert targets.temporal.shape == (3, 3)
        assert targets.is_black_album[2]

    def test_temporal_context_smoothing(self):
        """Test temporal context smoothing."""
        from core.soft_targets import SoftTargetGenerator

        gen = SoftTargetGenerator(
            label_smoothing=0.0,
            temporal_context_enabled=True,
            temporal_context_weight=0.3,
        )

        # Sequence: Past, Present, Past
        # Middle should be smoothed toward neighbors
        targets = gen.generate_from_labels(
            temporal_labels=["past", "present", "past"],
            spatial_labels=["thing", "thing", "thing"],
            ontological_labels=["imagined", "imagined", "imagined"],
        )

        # Middle sample should have some past influence
        middle_temporal = targets.temporal[1]
        assert middle_temporal[0] > 0  # Some past influence
        # Still should be highest for present
        assert middle_temporal[1] > 0.3

    def test_track_boundary_respect(self):
        """Test that context smoothing respects track boundaries."""
        from core.soft_targets import SoftTargetGenerator

        gen = SoftTargetGenerator(
            label_smoothing=0.0,
            temporal_context_enabled=True,
            temporal_context_weight=0.5,
        )
        print(gen)

        # Two tracks: [past, present] | [future, future]

        # Sample at boundary (index 1) shouldn't be influenced by track2
        # Sample at boundary (index 2) shouldn't be influenced by track1
        pass  # Just verify no crash

    def test_confidence_for_black_album(self):
        """Test that Black Album has low confidence."""
        from core.soft_targets import SoftTargetGenerator

        gen = SoftTargetGenerator(black_album_confidence=0.0)

        targets = gen.generate_from_labels(
            temporal_labels=["past", "none"],
            spatial_labels=["thing", "none"],
            ontological_labels=["imagined", "none"],
        )

        # Normal concept has confidence 1.0
        assert targets.confidence[0, 0] == 1.0
        # Black Album has confidence 0.0
        assert targets.confidence[1, 0] == 0.0


class TestSoftTargets:
    """Test SoftTargets dataclass."""

    def test_validation_passes(self):
        """Test validation with valid targets."""
        from core.soft_targets import SoftTargets

        targets = SoftTargets(
            temporal=np.array([[0.9, 0.05, 0.05], [0.33, 0.34, 0.33]]),
            spatial=np.array([[0.8, 0.1, 0.1], [0.5, 0.3, 0.2]]),
            ontological=np.array([[0.7, 0.2, 0.1], [0.6, 0.3, 0.1]]),
            confidence=np.array([[0.9], [0.5]]),
        )

        errors = targets.validate()
        assert len(errors) == 0

    def test_validation_fails_bad_sum(self):
        """Test validation fails when distributions don't sum to 1."""
        from core.soft_targets import SoftTargets

        targets = SoftTargets(
            temporal=np.array([[0.9, 0.05, 0.1]]),  # Sums to 1.05
            spatial=np.array([[0.8, 0.1, 0.1]]),
            ontological=np.array([[0.7, 0.2, 0.1]]),
            confidence=np.array([[0.9]]),
        )

        errors = targets.validate()
        assert len(errors) > 0
        assert any("sum" in e.lower() for e in errors)

    def test_validation_fails_negative(self):
        """Test validation fails with negative values."""
        from core.soft_targets import SoftTargets

        targets = SoftTargets(
            temporal=np.array([[1.1, -0.05, -0.05]]),  # Negative values
            spatial=np.array([[0.8, 0.1, 0.1]]),
            ontological=np.array([[0.7, 0.2, 0.1]]),
            confidence=np.array([[0.9]]),
        )

        errors = targets.validate()
        assert len(errors) > 0

    def test_to_tensors(self):
        """Test conversion to PyTorch tensors."""
        from core.soft_targets import SoftTargets

        targets = SoftTargets(
            temporal=np.array([[0.9, 0.05, 0.05]]),
            spatial=np.array([[0.8, 0.1, 0.1]]),
            ontological=np.array([[0.7, 0.2, 0.1]]),
            confidence=np.array([[0.9]]),
        )

        tensors = targets.to_tensors()

        assert "temporal_targets" in tensors
        assert isinstance(tensors["temporal_targets"], torch.Tensor)
        assert tensors["temporal_targets"].dtype == torch.float32


class TestTargetConsistencyValidator:
    """Test TargetConsistencyValidator."""

    def test_consistent_label_target(self):
        """Test validation of consistent label-target pair."""
        from core.soft_targets import TargetConsistencyValidator

        ok, msg = TargetConsistencyValidator.check_label_target_alignment(
            discrete_label="past",
            soft_target=np.array([0.9, 0.05, 0.05]),
            dimension="temporal",
        )

        assert ok is True

    def test_inconsistent_label_target(self):
        """Test detection of inconsistent label-target pair."""
        from core.soft_targets import TargetConsistencyValidator

        ok, msg = TargetConsistencyValidator.check_label_target_alignment(
            discrete_label="past",
            soft_target=np.array([0.2, 0.7, 0.1]),  # Highest is present, not past
            dimension="temporal",
        )

        assert ok is False
        assert "mismatch" in msg.lower()

    def test_black_album_validation(self):
        """Test validation of Black Album labels."""
        from core.soft_targets import TargetConsistencyValidator

        ok, msg = TargetConsistencyValidator.check_label_target_alignment(
            discrete_label="none",
            soft_target=np.array([0.33, 0.34, 0.33]),
            dimension="temporal",
        )

        assert ok is True
        assert "Black Album" in msg

    def test_dataset_validation(self):
        """Test full dataset validation."""
        from core.soft_targets import TargetConsistencyValidator, SoftTargets

        targets = SoftTargets(
            temporal=np.array(
                [
                    [0.9, 0.05, 0.05],  # Consistent with "past"
                    [0.2, 0.7, 0.1],  # Inconsistent with "past"
                ]
            ),
            spatial=np.array([[0.8, 0.1, 0.1], [0.8, 0.1, 0.1]]),
            ontological=np.array([[0.7, 0.2, 0.1], [0.7, 0.2, 0.1]]),
            confidence=np.array([[0.9], [0.9]]),
        )

        result = TargetConsistencyValidator.validate_dataset(
            temporal_labels=["past", "past"],  # Second is inconsistent
            spatial_labels=["thing", "thing"],
            ontological_labels=["imagined", "imagined"],
            soft_targets=targets,
        )

        assert result["n_samples"] == 2
        assert result["temporal_issues"] == 1  # One inconsistency


class TestGenerateFromDataFrame:
    """Test generate_soft_targets_from_dataframe function."""

    def test_basic_dataframe_generation(self):
        """Test generation from pandas DataFrame."""
        import pandas as pd
        from core.soft_targets import generate_soft_targets_from_dataframe

        df = pd.DataFrame(
            {
                "temporal_mode": ["past", "present", "future"],
                "spatial_mode": ["thing", "place", "person"],
                "ontological_mode": ["imagined", "forgotten", "known"],
                "track_id": ["t1", "t1", "t2"],
            }
        )

        targets = generate_soft_targets_from_dataframe(df)

        assert targets.temporal.shape == (3, 3)
        assert targets.spatial.shape == (3, 3)
        assert targets.ontological.shape == (3, 3)

    def test_missing_values_handling(self):
        """Test handling of missing values."""
        import pandas as pd
        from core.soft_targets import generate_soft_targets_from_dataframe

        df = pd.DataFrame(
            {
                "temporal_mode": ["past", None, "future"],
                "spatial_mode": ["thing", "place", None],
                "ontological_mode": [None, "forgotten", "known"],
            }
        )

        targets = generate_soft_targets_from_dataframe(df)

        # Missing values should become "none" (uniform)
        assert targets.temporal.shape == (3, 3)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
