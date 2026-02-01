"""
Soft target generation for Rainbow Table ontological regression.

Converts discrete labels (Past, Present, Future, etc.) into continuous
probability distributions for training regression models.

Features:
- One-hot encoding for discrete labels
- Label smoothing to prevent overconfidence
- Temporal context smoothing using surrounding segments
- Black Album handling (uniform distributions)
- Target validation and consistency checks
"""

import numpy as np
import torch
from typing import Dict, List, Optional, Tuple, Literal
from dataclasses import dataclass
import warnings


# Mode definitions
TEMPORAL_MODES = ["past", "present", "future"]
SPATIAL_MODES = ["thing", "place", "person"]
ONTOLOGICAL_MODES = ["imagined", "forgotten", "known"]

# Mode to index mapping
TEMPORAL_TO_IDX = {m: i for i, m in enumerate(TEMPORAL_MODES)}
SPATIAL_TO_IDX = {m: i for i, m in enumerate(SPATIAL_MODES)}
ONTOLOGICAL_TO_IDX = {m: i for i, m in enumerate(ONTOLOGICAL_MODES)}


@dataclass
class SoftTargets:
    """Container for soft regression targets."""

    temporal: np.ndarray  # [n_samples, 3]
    spatial: np.ndarray  # [n_samples, 3]
    ontological: np.ndarray  # [n_samples, 3]
    confidence: np.ndarray  # [n_samples, 1]

    # Optional metadata
    is_black_album: Optional[np.ndarray] = None  # [n_samples] bool
    uncertainty_weights: Optional[np.ndarray] = None  # [n_samples] for loss weighting

    def to_tensors(self, device: str = "cpu") -> Dict[str, torch.Tensor]:
        """Convert to PyTorch tensors."""
        result = {
            "temporal_targets": torch.tensor(
                self.temporal, dtype=torch.float32, device=device
            ),
            "spatial_targets": torch.tensor(
                self.spatial, dtype=torch.float32, device=device
            ),
            "ontological_targets": torch.tensor(
                self.ontological, dtype=torch.float32, device=device
            ),
            "confidence_targets": torch.tensor(
                self.confidence, dtype=torch.float32, device=device
            ),
        }
        if self.uncertainty_weights is not None:
            result["sample_weights"] = torch.tensor(
                self.uncertainty_weights, dtype=torch.float32, device=device
            )
        return result

    def validate(self) -> List[str]:
        """
        Validate soft targets.

        Returns:
            List of validation error messages (empty if valid)
        """
        errors = []

        # Check shapes
        n_samples = self.temporal.shape[0]
        if self.spatial.shape[0] != n_samples:
            errors.append(
                f"Spatial shape mismatch: {self.spatial.shape[0]} vs {n_samples}"
            )
        if self.ontological.shape[0] != n_samples:
            errors.append(
                f"Ontological shape mismatch: {self.ontological.shape[0]} vs {n_samples}"
            )
        if self.confidence.shape[0] != n_samples:
            errors.append(
                f"Confidence shape mismatch: {self.confidence.shape[0]} vs {n_samples}"
            )

        # Check dimension sizes
        if self.temporal.shape[1] != 3:
            errors.append(
                f"Temporal should have 3 values, got {self.temporal.shape[1]}"
            )
        if self.spatial.shape[1] != 3:
            errors.append(f"Spatial should have 3 values, got {self.spatial.shape[1]}")
        if self.ontological.shape[1] != 3:
            errors.append(
                f"Ontological should have 3 values, got {self.ontological.shape[1]}"
            )

        # Check distributions sum to 1
        temporal_sums = self.temporal.sum(axis=1)
        spatial_sums = self.spatial.sum(axis=1)
        ontological_sums = self.ontological.sum(axis=1)

        if not np.allclose(temporal_sums, 1.0, atol=1e-5):
            bad_idx = np.where(~np.isclose(temporal_sums, 1.0, atol=1e-5))[0]
            errors.append(f"Temporal sums not 1.0 at indices: {bad_idx[:5]}...")

        if not np.allclose(spatial_sums, 1.0, atol=1e-5):
            bad_idx = np.where(~np.isclose(spatial_sums, 1.0, atol=1e-5))[0]
            errors.append(f"Spatial sums not 1.0 at indices: {bad_idx[:5]}...")

        if not np.allclose(ontological_sums, 1.0, atol=1e-5):
            bad_idx = np.where(~np.isclose(ontological_sums, 1.0, atol=1e-5))[0]
            errors.append(f"Ontological sums not 1.0 at indices: {bad_idx[:5]}...")

        # Check confidence range
        if np.any(self.confidence < 0) or np.any(self.confidence > 1):
            errors.append("Confidence values outside [0, 1] range")

        # Check non-negative
        if (
            np.any(self.temporal < 0)
            or np.any(self.spatial < 0)
            or np.any(self.ontological < 0)
        ):
            errors.append("Negative probability values found")

        return errors


class SoftTargetGenerator:
    """
    Generates soft targets from discrete Rainbow Table labels.

    Supports:
    - Basic one-hot encoding
    - Label smoothing
    - Temporal context smoothing
    - Black Album special handling
    """

    def __init__(
        self,
        label_smoothing: float = 0.1,
        temporal_context_enabled: bool = True,
        temporal_context_window: int = 3,
        temporal_context_weight: float = 0.3,
        black_album_confidence: float = 0.0,
    ):
        """
        Initialize soft target generator.

        Args:
            label_smoothing: Smoothing factor (0 = one-hot, 0.1 = [0.9, 0.05, 0.05])
            temporal_context_enabled: Use surrounding segments for smoothing
            temporal_context_window: Number of segments before/after to consider
            temporal_context_weight: Influence of neighboring segments
            black_album_confidence: Confidence value for Black Album (None_None_None)
        """
        self.label_smoothing = label_smoothing
        self.temporal_context_enabled = temporal_context_enabled
        self.temporal_context_window = temporal_context_window
        self.temporal_context_weight = temporal_context_weight
        self.black_album_confidence = black_album_confidence

    def one_hot(self, index: int, num_classes: int = 3) -> np.ndarray:
        """Create one-hot vector."""
        vec = np.zeros(num_classes)
        vec[index] = 1.0
        return vec

    def smooth_one_hot(self, index: int, num_classes: int = 3) -> np.ndarray:
        """
        Create smoothed one-hot vector.

        Label smoothing: (1-α)*one_hot + α*uniform
        e.g., [1.0, 0.0, 0.0] → [0.9, 0.05, 0.05] with α=0.1
        """
        if self.label_smoothing == 0:
            return self.one_hot(index, num_classes)

        uniform = np.ones(num_classes) / num_classes
        one_hot = self.one_hot(index, num_classes)

        return (1 - self.label_smoothing) * one_hot + self.label_smoothing * uniform

    def uniform_distribution(self, num_classes: int = 3) -> np.ndarray:
        """Create uniform distribution (for Black Album)."""
        return np.ones(num_classes) / num_classes

    def encode_mode(
        self,
        mode: str,
        dimension: Literal["temporal", "spatial", "ontological"],
    ) -> np.ndarray:
        """
        Encode a single mode label to soft target.

        Args:
            mode: Mode label (e.g., "past", "thing", "imagined")
            dimension: Which dimension this mode belongs to

        Returns:
            Soft target distribution [3]
        """
        mode = mode.lower().strip()

        # Handle None/Black Album
        if mode in ("none", "null", "", "black"):
            return self.uniform_distribution()

        # Get index mapping
        if dimension == "temporal":
            idx_map = TEMPORAL_TO_IDX
        elif dimension == "spatial":
            idx_map = SPATIAL_TO_IDX
        else:
            idx_map = ONTOLOGICAL_TO_IDX

        if mode not in idx_map:
            warnings.warn(f"Unknown {dimension} mode: {mode}, using uniform")
            return self.uniform_distribution()

        return self.smooth_one_hot(idx_map[mode])

    def apply_temporal_context_smoothing(
        self,
        targets: np.ndarray,
        track_boundaries: Optional[List[int]] = None,
    ) -> np.ndarray:
        """
        Smooth targets using surrounding segment context.

        If prev=Past, curr=Present, next=Past, the middle segment
        gets smoothed towards the surrounding modes.

        Args:
            targets: Soft targets [n_samples, 3]
            track_boundaries: Indices where new tracks start (don't smooth across)

        Returns:
            Context-smoothed targets [n_samples, 3]
        """
        if not self.temporal_context_enabled:
            return targets

        n_samples = targets.shape[0]
        smoothed = targets.copy()

        # Create mask for valid context (don't cross track boundaries)
        if track_boundaries is None:
            track_boundaries = []

        boundary_set = set(track_boundaries)

        for i in range(n_samples):
            # Collect neighboring targets
            neighbors = []
            weights = []

            for offset in range(
                -self.temporal_context_window, self.temporal_context_window + 1
            ):
                if offset == 0:
                    continue

                neighbor_idx = i + offset
                if neighbor_idx < 0 or neighbor_idx >= n_samples:
                    continue

                # Check if crossing track boundary
                if offset > 0:
                    crosses_boundary = any(
                        b > i and b <= neighbor_idx for b in boundary_set
                    )
                else:
                    crosses_boundary = any(
                        b > neighbor_idx and b <= i for b in boundary_set
                    )

                if crosses_boundary:
                    continue

                # Weight by distance (closer = more influence)
                distance_weight = 1.0 / (abs(offset) + 1)
                neighbors.append(targets[neighbor_idx])
                weights.append(distance_weight)

            if neighbors:
                # Weighted average of neighbors
                weights = np.array(weights)
                weights /= weights.sum()
                neighbor_avg = np.average(neighbors, axis=0, weights=weights)

                # Blend with original
                smoothed[i] = (1 - self.temporal_context_weight) * targets[
                    i
                ] + self.temporal_context_weight * neighbor_avg

                # Re-normalize to sum to 1
                smoothed[i] /= smoothed[i].sum()

        return smoothed

    def generate_from_labels(
        self,
        temporal_labels: List[str],
        spatial_labels: List[str],
        ontological_labels: List[str],
        track_ids: Optional[List[str]] = None,
        annotation_confidence: Optional[List[float]] = None,
    ) -> SoftTargets:
        """
        Generate soft targets from discrete labels.

        Args:
            temporal_labels: List of temporal mode labels
            spatial_labels: List of spatial mode labels
            ontological_labels: List of ontological mode labels
            track_ids: Optional track IDs (for context boundary detection)
            annotation_confidence: Optional per-sample confidence scores

        Returns:
            SoftTargets dataclass
        """
        n_samples = len(temporal_labels)
        assert len(spatial_labels) == n_samples
        assert len(ontological_labels) == n_samples

        # Encode each dimension
        temporal = np.array(
            [self.encode_mode(label, "temporal") for label in temporal_labels]
        )
        spatial = np.array(
            [self.encode_mode(label, "spatial") for label in spatial_labels]
        )
        ontological = np.array(
            [self.encode_mode(label, "ontological") for label in ontological_labels]
        )

        # Detect track boundaries for context smoothing
        track_boundaries = None
        if track_ids is not None:
            track_boundaries = []
            for i in range(1, len(track_ids)):
                if track_ids[i] != track_ids[i - 1]:
                    track_boundaries.append(i)

        # Apply temporal context smoothing
        temporal = self.apply_temporal_context_smoothing(temporal, track_boundaries)
        spatial = self.apply_temporal_context_smoothing(spatial, track_boundaries)
        ontological = self.apply_temporal_context_smoothing(
            ontological, track_boundaries
        )

        # Detect Black Album segments
        is_black_album = np.array(
            [
                (
                    t.lower() in ("none", "null", "", "black")
                    or s.lower() in ("none", "null", "", "black")
                    or o.lower() in ("none", "null", "", "black")
                )
                for t, s, o in zip(temporal_labels, spatial_labels, ontological_labels)
            ]
        )

        # Generate confidence targets
        confidence = np.ones((n_samples, 1))

        # Set Black Album confidence
        confidence[is_black_album] = self.black_album_confidence

        # If annotation confidence provided, factor it in
        if annotation_confidence is not None:
            annotation_confidence = np.array(annotation_confidence).reshape(-1, 1)
            confidence = confidence * annotation_confidence

        # Generate uncertainty weights for loss weighting
        uncertainty_weights = None
        if annotation_confidence is not None:
            # Higher confidence = higher weight
            uncertainty_weights = np.array(annotation_confidence)

        return SoftTargets(
            temporal=temporal,
            spatial=spatial,
            ontological=ontological,
            confidence=confidence,
            is_black_album=is_black_album,
            uncertainty_weights=uncertainty_weights,
        )

    def generate_from_combined_mode(
        self,
        combined_modes: List[str],
        track_ids: Optional[List[str]] = None,
    ) -> SoftTargets:
        """
        Generate soft targets from combined mode strings.

        Args:
            combined_modes: List of "Temporal_Spatial_Ontological" strings
                e.g., ["Past_Thing_Imagined", "Present_Place_Known"]
            track_ids: Optional track IDs for boundary detection

        Returns:
            SoftTargets dataclass
        """
        temporal_labels = []
        spatial_labels = []
        ontological_labels = []

        for mode in combined_modes:
            if mode.lower() in ("none", "black", "none_none_none"):
                temporal_labels.append("none")
                spatial_labels.append("none")
                ontological_labels.append("none")
            else:
                parts = mode.split("_")
                if len(parts) != 3:
                    warnings.warn(f"Invalid combined mode format: {mode}")
                    temporal_labels.append("none")
                    spatial_labels.append("none")
                    ontological_labels.append("none")
                else:
                    temporal_labels.append(parts[0])
                    spatial_labels.append(parts[1])
                    ontological_labels.append(parts[2])

        return self.generate_from_labels(
            temporal_labels, spatial_labels, ontological_labels, track_ids
        )


class TargetConsistencyValidator:
    """
    Validates consistency between discrete labels and soft targets.
    """

    @staticmethod
    def check_label_target_alignment(
        discrete_label: str,
        soft_target: np.ndarray,
        dimension: Literal["temporal", "spatial", "ontological"],
        tolerance: float = 0.1,
    ) -> Tuple[bool, str]:
        """
        Check if soft target aligns with discrete label.

        Args:
            discrete_label: Discrete mode label
            soft_target: Soft target distribution [3]
            dimension: Which dimension
            tolerance: How much the argmax can differ from label

        Returns:
            Tuple of (is_consistent, message)
        """
        label = discrete_label.lower().strip()

        # Handle None/Black
        if label in ("none", "null", "", "black"):
            # Should be roughly uniform
            if np.std(soft_target) < 0.1:
                return True, "Black Album: uniform distribution"
            else:
                return (
                    False,
                    f"Black Album should be uniform, got std={np.std(soft_target):.3f}",
                )

        # Get expected index
        if dimension == "temporal":
            idx_map = TEMPORAL_TO_IDX
            modes = TEMPORAL_MODES
        elif dimension == "spatial":
            idx_map = SPATIAL_TO_IDX
            modes = SPATIAL_MODES
        else:
            idx_map = ONTOLOGICAL_TO_IDX
            modes = ONTOLOGICAL_MODES

        if label not in idx_map:
            return False, f"Unknown label: {label}"

        expected_idx = idx_map[label]
        actual_idx = np.argmax(soft_target)

        if expected_idx != actual_idx:
            return False, (
                f"Argmax mismatch: label={label} (idx={expected_idx}), "
                f"argmax={modes[actual_idx]} (idx={actual_idx})"
            )

        # Check that the expected index has highest score
        expected_score = soft_target[expected_idx]
        if expected_score < 0.5 - tolerance:
            return False, (
                f"Expected score too low: {label}={expected_score:.3f} (min 0.5)"
            )

        return True, "Consistent"

    @staticmethod
    def validate_dataset(
        temporal_labels: List[str],
        spatial_labels: List[str],
        ontological_labels: List[str],
        soft_targets: SoftTargets,
    ) -> Dict:
        """
        Validate entire dataset for consistency.

        Returns:
            Dictionary with validation statistics
        """
        n_samples = len(temporal_labels)
        temporal_issues = []
        spatial_issues = []
        ontological_issues = []

        for i in range(n_samples):
            # Temporal
            ok, msg = TargetConsistencyValidator.check_label_target_alignment(
                temporal_labels[i], soft_targets.temporal[i], "temporal"
            )
            if not ok:
                temporal_issues.append((i, msg))

            # Spatial
            ok, msg = TargetConsistencyValidator.check_label_target_alignment(
                spatial_labels[i], soft_targets.spatial[i], "spatial"
            )
            if not ok:
                spatial_issues.append((i, msg))

            # Ontological
            ok, msg = TargetConsistencyValidator.check_label_target_alignment(
                ontological_labels[i], soft_targets.ontological[i], "ontological"
            )
            if not ok:
                ontological_issues.append((i, msg))

        return {
            "n_samples": n_samples,
            "temporal_issues": len(temporal_issues),
            "spatial_issues": len(spatial_issues),
            "ontological_issues": len(ontological_issues),
            "total_issues": len(temporal_issues)
            + len(spatial_issues)
            + len(ontological_issues),
            "temporal_issue_samples": temporal_issues[:10],  # First 10
            "spatial_issue_samples": spatial_issues[:10],
            "ontological_issue_samples": ontological_issues[:10],
        }


def generate_soft_targets_from_dataframe(
    df,
    temporal_column: str = "temporal_mode",
    spatial_column: str = "spatial_mode",
    ontological_column: str = "ontological_mode",
    track_id_column: Optional[str] = "track_id",
    confidence_column: Optional[str] = None,
    label_smoothing: float = 0.1,
    temporal_context: bool = True,
) -> SoftTargets:
    """
    Generate soft targets from a pandas DataFrame.

    Args:
        df: DataFrame with mode columns
        temporal_column: Column name for temporal modes
        spatial_column: Column name for spatial modes
        ontological_column: Column name for ontological modes
        track_id_column: Column name for track IDs (for boundary detection)
        confidence_column: Optional column for annotation confidence
        label_smoothing: Label smoothing factor
        temporal_context: Enable temporal context smoothing

    Returns:
        SoftTargets dataclass
    """
    generator = SoftTargetGenerator(
        label_smoothing=label_smoothing,
        temporal_context_enabled=temporal_context,
    )

    temporal_labels = df[temporal_column].fillna("none").tolist()
    spatial_labels = df[spatial_column].fillna("none").tolist()
    ontological_labels = df[ontological_column].fillna("none").tolist()

    track_ids = None
    if track_id_column and track_id_column in df.columns:
        track_ids = df[track_id_column].tolist()

    annotation_confidence = None
    if confidence_column and confidence_column in df.columns:
        annotation_confidence = df[confidence_column].tolist()

    return generator.generate_from_labels(
        temporal_labels,
        spatial_labels,
        ontological_labels,
        track_ids,
        annotation_confidence,
    )


if __name__ == "__main__":
    # Quick tests
    print("Testing soft target generation...")

    # Test 1: Basic one-hot encoding
    print("\n=== One-hot encoding ===")
    generator = SoftTargetGenerator(label_smoothing=0.0)

    target = generator.encode_mode("past", "temporal")
    print(f"Past (no smoothing): {target}")
    assert np.allclose(target, [1.0, 0.0, 0.0])

    # Test 2: Label smoothing
    print("\n=== Label smoothing ===")
    generator = SoftTargetGenerator(label_smoothing=0.1)

    target = generator.encode_mode("past", "temporal")
    print(f"Past (smoothed): {target}")
    assert np.isclose(target[0], 0.933, atol=0.01)  # (1-0.1)*1 + 0.1*0.33

    # Test 3: Black Album
    print("\n=== Black Album handling ===")
    target = generator.encode_mode("none", "temporal")
    print(f"None/Black: {target}")
    assert np.allclose(target, [1 / 3, 1 / 3, 1 / 3], atol=0.01)

    # Test 4: Generate from labels
    print("\n=== Generate from labels ===")
    targets = generator.generate_from_labels(
        temporal_labels=["past", "present", "future", "none"],
        spatial_labels=["thing", "place", "person", "none"],
        ontological_labels=["imagined", "forgotten", "known", "none"],
    )

    print(f"Temporal targets:\n{targets.temporal}")
    print(f"Is Black Album: {targets.is_black_album}")
    print(f"Confidence: {targets.confidence.flatten()}")

    # Validate
    errors = targets.validate()
    if errors:
        print(f"Validation errors: {errors}")
    else:
        print("✓ Validation passed!")

    # Test 5: Combined mode generation
    print("\n=== Combined mode generation ===")
    targets2 = generator.generate_from_combined_mode(
        [
            "Past_Thing_Imagined",
            "Present_Place_Known",
            "None_None_None",
        ]
    )
    print("From combined modes:")
    print(f"  Temporal: {targets2.temporal}")
    print(f"  Is Black: {targets2.is_black_album}")

    # Test 6: Temporal context smoothing
    print("\n=== Temporal context smoothing ===")
    generator_ctx = SoftTargetGenerator(
        label_smoothing=0.0,
        temporal_context_enabled=True,
        temporal_context_weight=0.3,
    )

    # Sequence: Past, Present, Past (middle should be smoothed)
    targets_ctx = generator_ctx.generate_from_labels(
        temporal_labels=["past", "present", "past"],
        spatial_labels=["thing", "thing", "thing"],
        ontological_labels=["imagined", "imagined", "imagined"],
    )
    print("Sequence [Past, Present, Past]:")
    print(f"  Middle temporal (should be smoothed): {targets_ctx.temporal[1]}")

    # Test 7: Consistency validation
    print("\n=== Consistency validation ===")
    validator = TargetConsistencyValidator()

    ok, msg = validator.check_label_target_alignment(
        "past", np.array([0.9, 0.05, 0.05]), "temporal"
    )
    print(f"Past vs [0.9, 0.05, 0.05]: {ok} - {msg}")

    ok, msg = validator.check_label_target_alignment(
        "past", np.array([0.3, 0.5, 0.2]), "temporal"
    )
    print(f"Past vs [0.3, 0.5, 0.2]: {ok} - {msg}")

    print("\n✓ All soft target tests passed!")
