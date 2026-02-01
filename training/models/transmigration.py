"""
Advanced transmigration analysis for Rainbow Pipeline.

Extends the basic TransmigrationCalculator with:
- Intermediate state generation
- Dimension priority ranking
- Minimal edit suggestions
- Path visualization utilities
"""

from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import numpy as np

import torch

from models.rainbow_table_regression_head import (
    OntologicalScores,
    TEMPORAL_MODES,
    SPATIAL_MODES,
    ONTOLOGICAL_MODES,
    ALBUM_MAPPING,
)


@dataclass
class TransmigrationStep:
    """A single step in a transmigration path."""

    # State at this step
    temporal: np.ndarray  # [3]
    spatial: np.ndarray  # [3]
    ontological: np.ndarray  # [3]

    # Step metadata
    step_index: int
    step_type: str  # "start", "intermediate", "end"
    dimension_changed: Optional[str] = None
    change_description: Optional[str] = None

    # Derived properties
    @property
    def album(self) -> str:
        """Predicted album at this step."""
        t_idx = int(np.argmax(self.temporal))
        s_idx = int(np.argmax(self.spatial))
        o_idx = int(np.argmax(self.ontological))

        mode_tuple = (
            TEMPORAL_MODES[t_idx],
            SPATIAL_MODES[s_idx],
            ONTOLOGICAL_MODES[o_idx],
        )
        return ALBUM_MAPPING.get(mode_tuple, "Black")

    @property
    def combined_mode(self) -> str:
        """Combined mode string."""
        t_idx = int(np.argmax(self.temporal))
        s_idx = int(np.argmax(self.spatial))
        o_idx = int(np.argmax(self.ontological))

        return f"{TEMPORAL_MODES[t_idx].capitalize()}_{SPATIAL_MODES[s_idx].capitalize()}_{ONTOLOGICAL_MODES[o_idx].capitalize()}"

    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            "step_index": self.step_index,
            "step_type": self.step_type,
            "temporal": self.temporal.tolist(),
            "spatial": self.spatial.tolist(),
            "ontological": self.ontological.tolist(),
            "album": self.album,
            "combined_mode": self.combined_mode,
            "dimension_changed": self.dimension_changed,
            "change_description": self.change_description,
        }


@dataclass
class TransmigrationPath:
    """Complete transmigration path from source to target."""

    steps: List[TransmigrationStep]
    total_distance: float
    dimension_priority: List[str]
    is_feasible: bool
    feasibility_score: float

    @property
    def n_steps(self) -> int:
        """Number of steps in path."""
        return len(self.steps)

    @property
    def source_album(self) -> str:
        """Starting album."""
        return self.steps[0].album

    @property
    def target_album(self) -> str:
        """Target album."""
        return self.steps[-1].album

    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            "steps": [s.to_dict() for s in self.steps],
            "total_distance": self.total_distance,
            "dimension_priority": self.dimension_priority,
            "n_steps": self.n_steps,
            "source_album": self.source_album,
            "target_album": self.target_album,
            "is_feasible": self.is_feasible,
            "feasibility_score": self.feasibility_score,
        }


@dataclass
class EditSuggestion:
    """A minimal edit suggestion for improving ontological alignment."""

    dimension: str  # temporal, spatial, ontological
    current_mode: str
    current_score: float
    target_mode: str
    target_score: float
    priority: int
    effort: str  # "low", "medium", "high"
    description: str

    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            "dimension": self.dimension,
            "current_mode": self.current_mode,
            "current_score": self.current_score,
            "target_mode": self.target_mode,
            "target_score": self.target_score,
            "priority": self.priority,
            "effort": self.effort,
            "description": self.description,
        }


class AdvancedTransmigrationCalculator:
    """
    Advanced transmigration analysis with path generation and optimization.
    """

    def __init__(
        self,
        n_intermediate_steps: int = 5,
        feasibility_threshold: float = 1.5,
        min_score_threshold: float = 0.6,
    ):
        """
        Initialize calculator.

        Args:
            n_intermediate_steps: Default number of intermediate states
            feasibility_threshold: Max distance for feasible transmigration
            min_score_threshold: Minimum score for "dominant" mode
        """
        self.n_intermediate_steps = n_intermediate_steps
        self.feasibility_threshold = feasibility_threshold
        self.min_score_threshold = min_score_threshold

    def compute_dimension_distance(
        self,
        source: np.ndarray,
        target: np.ndarray,
    ) -> float:
        """Compute L2 distance between score vectors."""
        return float(np.linalg.norm(source - target))

    def compute_total_distance(
        self,
        source_temporal: np.ndarray,
        source_spatial: np.ndarray,
        source_ontological: np.ndarray,
        target_temporal: np.ndarray,
        target_spatial: np.ndarray,
        target_ontological: np.ndarray,
    ) -> float:
        """Compute total Euclidean distance across all dimensions."""
        t_dist = self.compute_dimension_distance(source_temporal, target_temporal)
        s_dist = self.compute_dimension_distance(source_spatial, target_spatial)
        o_dist = self.compute_dimension_distance(source_ontological, target_ontological)
        return float(np.sqrt(t_dist**2 + s_dist**2 + o_dist**2))

    def rank_dimension_priority(
        self,
        source_temporal: np.ndarray,
        source_spatial: np.ndarray,
        source_ontological: np.ndarray,
        target_temporal: np.ndarray,
        target_spatial: np.ndarray,
        target_ontological: np.ndarray,
    ) -> List[Tuple[str, float]]:
        """
        Rank dimensions by transmigration effort required.

        Returns dimensions sorted by distance (largest first = highest priority).

        Args:
            source_*: Current state arrays
            target_*: Target state arrays

        Returns:
            List of (dimension_name, distance) tuples sorted by distance descending
        """
        distances = [
            (
                "temporal",
                self.compute_dimension_distance(source_temporal, target_temporal),
            ),
            (
                "spatial",
                self.compute_dimension_distance(source_spatial, target_spatial),
            ),
            (
                "ontological",
                self.compute_dimension_distance(source_ontological, target_ontological),
            ),
        ]

        # Sort by distance descending (largest change first)
        distances.sort(key=lambda x: x[1], reverse=True)
        return distances

    def generate_intermediate_states(
        self,
        source_temporal: np.ndarray,
        source_spatial: np.ndarray,
        source_ontological: np.ndarray,
        target_temporal: np.ndarray,
        target_spatial: np.ndarray,
        target_ontological: np.ndarray,
        n_steps: Optional[int] = None,
        interpolation: str = "linear",
    ) -> List[TransmigrationStep]:
        """
        Generate intermediate states for a transmigration path.

        Args:
            source_*: Starting state arrays
            target_*: Target state arrays
            n_steps: Number of intermediate steps (default: self.n_intermediate_steps)
            interpolation: Interpolation method ("linear", "ease_in_out", "sequential")

        Returns:
            List of TransmigrationStep objects including start and end
        """
        n_steps = n_steps or self.n_intermediate_steps

        steps = []

        # Add starting state
        steps.append(
            TransmigrationStep(
                temporal=source_temporal.copy(),
                spatial=source_spatial.copy(),
                ontological=source_ontological.copy(),
                step_index=0,
                step_type="start",
            )
        )

        if interpolation == "linear":
            # Simple linear interpolation
            for i in range(1, n_steps + 1):
                t = i / (n_steps + 1)

                temporal = self._interpolate_and_normalize(
                    source_temporal, target_temporal, t
                )
                spatial = self._interpolate_and_normalize(
                    source_spatial, target_spatial, t
                )
                ontological = self._interpolate_and_normalize(
                    source_ontological, target_ontological, t
                )

                steps.append(
                    TransmigrationStep(
                        temporal=temporal,
                        spatial=spatial,
                        ontological=ontological,
                        step_index=i,
                        step_type="intermediate",
                        change_description=f"Linear interpolation at t={t:.2f}",
                    )
                )

        elif interpolation == "ease_in_out":
            # Smooth easing function
            for i in range(1, n_steps + 1):
                t = i / (n_steps + 1)
                # Smooth step function: 3t^2 - 2t^3
                t_smooth = 3 * t**2 - 2 * t**3

                temporal = self._interpolate_and_normalize(
                    source_temporal, target_temporal, t_smooth
                )
                spatial = self._interpolate_and_normalize(
                    source_spatial, target_spatial, t_smooth
                )
                ontological = self._interpolate_and_normalize(
                    source_ontological, target_ontological, t_smooth
                )

                steps.append(
                    TransmigrationStep(
                        temporal=temporal,
                        spatial=spatial,
                        ontological=ontological,
                        step_index=i,
                        step_type="intermediate",
                        change_description=f"Ease-in-out interpolation at t={t:.2f}",
                    )
                )

        elif interpolation == "sequential":
            # Change one dimension at a time (most interpretable)
            priority = self.rank_dimension_priority(
                source_temporal,
                source_spatial,
                source_ontological,
                target_temporal,
                target_spatial,
                target_ontological,
            )

            current_temporal = source_temporal.copy()
            current_spatial = source_spatial.copy()
            current_ontological = source_ontological.copy()

            step_idx = 1
            for dim_name, _ in priority:
                if dim_name == "temporal":
                    current_temporal = target_temporal.copy()
                    dim_changed = "temporal"
                elif dim_name == "spatial":
                    current_spatial = target_spatial.copy()
                    dim_changed = "spatial"
                else:
                    current_ontological = target_ontological.copy()
                    dim_changed = "ontological"

                steps.append(
                    TransmigrationStep(
                        temporal=current_temporal.copy(),
                        spatial=current_spatial.copy(),
                        ontological=current_ontological.copy(),
                        step_index=step_idx,
                        step_type="intermediate",
                        dimension_changed=dim_changed,
                        change_description=f"Shift {dim_changed} dimension to target",
                    )
                )
                step_idx += 1

        # Add ending state
        steps.append(
            TransmigrationStep(
                temporal=target_temporal.copy(),
                spatial=target_spatial.copy(),
                ontological=target_ontological.copy(),
                step_index=len(steps),
                step_type="end",
            )
        )

        return steps

    def _interpolate_and_normalize(
        self,
        source: np.ndarray,
        target: np.ndarray,
        t: float,
    ) -> np.ndarray:
        """
        Interpolate between source and target, then normalize to sum to 1.

        Args:
            source: Source array
            target: Target array
            t: Interpolation parameter (0 = source, 1 = target)

        Returns:
            Interpolated and normalized array
        """
        interpolated = (1 - t) * source + t * target
        # Normalize to ensure sum is 1
        return interpolated / interpolated.sum()

    def compute_feasibility(
        self,
        total_distance: float,
        confidence: float = 1.0,
    ) -> Tuple[bool, float]:
        """
        Assess feasibility of a transmigration.

        Args:
            total_distance: Total ontological distance
            confidence: Current chromatic confidence

        Returns:
            Tuple of (is_feasible, feasibility_score)
        """
        # Feasibility score: inverse of distance scaled by confidence
        # Higher score = more feasible
        if total_distance == 0:
            return True, 1.0

        raw_score = 1.0 / (1.0 + total_distance)
        adjusted_score = raw_score * confidence

        is_feasible = total_distance < self.feasibility_threshold

        return is_feasible, float(adjusted_score)

    def generate_transmigration_path(
        self,
        source: OntologicalScores,
        target_album: str,
        sample_idx: int = 0,
        n_steps: Optional[int] = None,
        interpolation: str = "sequential",
    ) -> TransmigrationPath:
        """
        Generate full transmigration path from current state to target album.

        Args:
            source: Current OntologicalScores
            target_album: Target album name
            sample_idx: Index of sample in batch
            n_steps: Number of intermediate steps
            interpolation: Interpolation method

        Returns:
            TransmigrationPath object
        """
        # Get source arrays
        source_temporal = source.temporal_scores[sample_idx].detach().cpu().numpy()
        source_spatial = source.spatial_scores[sample_idx].detach().cpu().numpy()
        source_ontological = (
            source.ontological_scores[sample_idx].detach().cpu().numpy()
        )
        confidence = source.chromatic_confidence[sample_idx].detach().cpu().item()

        # Get target arrays (canonical album position)
        target_temporal, target_spatial, target_ontological = self._get_album_target(
            target_album
        )

        # Compute priority ranking
        priority = self.rank_dimension_priority(
            source_temporal,
            source_spatial,
            source_ontological,
            target_temporal,
            target_spatial,
            target_ontological,
        )

        # Compute total distance
        total_dist = self.compute_total_distance(
            source_temporal,
            source_spatial,
            source_ontological,
            target_temporal,
            target_spatial,
            target_ontological,
        )

        # Assess feasibility
        is_feasible, feasibility_score = self.compute_feasibility(
            total_dist, confidence
        )

        # Generate intermediate steps
        steps = self.generate_intermediate_states(
            source_temporal,
            source_spatial,
            source_ontological,
            target_temporal,
            target_spatial,
            target_ontological,
            n_steps=n_steps,
            interpolation=interpolation,
        )

        return TransmigrationPath(
            steps=steps,
            total_distance=total_dist,
            dimension_priority=[p[0] for p in priority],
            is_feasible=is_feasible,
            feasibility_score=feasibility_score,
        )

    def _get_album_target(
        self,
        album: str,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Get canonical target arrays for an album."""
        # Canonical album positions
        album_targets = {
            "Orange": ("past", "thing", "imagined"),
            "Red": ("past", "thing", "forgotten"),
            "Violet": ("past", "person", "known"),
            "Yellow": ("present", "place", "imagined"),
            "Green": ("present", "person", "known"),
            "Indigo": ("present", "person", "forgotten"),
            "Blue": ("future", "place", "known"),
            "White": ("present", "person", "known"),  # Same as Green
            "Black": None,  # Uniform
        }

        modes = album_targets.get(album)

        if modes is None:
            # Black album: uniform distribution
            return (
                np.array([1 / 3, 1 / 3, 1 / 3]),
                np.array([1 / 3, 1 / 3, 1 / 3]),
                np.array([1 / 3, 1 / 3, 1 / 3]),
            )

        # Create one-hot encoded targets
        t_idx = TEMPORAL_MODES.index(modes[0])
        s_idx = SPATIAL_MODES.index(modes[1])
        o_idx = ONTOLOGICAL_MODES.index(modes[2])

        temporal = np.zeros(3)
        temporal[t_idx] = 1.0

        spatial = np.zeros(3)
        spatial[s_idx] = 1.0

        ontological = np.zeros(3)
        ontological[o_idx] = 1.0

        return temporal, spatial, ontological

    def generate_minimal_edit_suggestions(
        self,
        source: OntologicalScores,
        target_album: str,
        sample_idx: int = 0,
        max_suggestions: int = 3,
    ) -> List[EditSuggestion]:
        """
        Generate minimal edit suggestions for reaching target album.

        Args:
            source: Current OntologicalScores
            target_album: Target album name
            sample_idx: Index of sample in batch
            max_suggestions: Maximum number of suggestions

        Returns:
            List of EditSuggestion objects sorted by priority
        """
        # Get arrays
        source_temporal = source.temporal_scores[sample_idx].detach().cpu().numpy()
        source_spatial = source.spatial_scores[sample_idx].detach().cpu().numpy()
        source_ontological = (
            source.ontological_scores[sample_idx].detach().cpu().numpy()
        )

        target_temporal, target_spatial, target_ontological = self._get_album_target(
            target_album
        )

        # Get dimension priorities
        priority = self.rank_dimension_priority(
            source_temporal,
            source_spatial,
            source_ontological,
            target_temporal,
            target_spatial,
            target_ontological,
        )

        suggestions = []

        for priority_idx, (dim_name, distance) in enumerate(priority):
            if len(suggestions) >= max_suggestions:
                break

            if distance < 0.1:  # Skip if already close
                continue

            if dim_name == "temporal":
                source_arr = source_temporal
                target_arr = target_temporal
                mode_names = TEMPORAL_MODES
            elif dim_name == "spatial":
                source_arr = source_spatial
                target_arr = target_spatial
                mode_names = SPATIAL_MODES
            else:
                source_arr = source_ontological
                target_arr = target_ontological
                mode_names = ONTOLOGICAL_MODES

            current_idx = int(np.argmax(source_arr))
            target_idx = int(np.argmax(target_arr))

            current_mode = mode_names[current_idx]
            target_mode = mode_names[target_idx]

            current_score = float(source_arr[current_idx])
            target_score = float(target_arr[target_idx])

            # Determine effort based on distance
            if distance < 0.5:
                effort = "low"
            elif distance < 1.0:
                effort = "medium"
            else:
                effort = "high"

            # Generate description
            if current_mode == target_mode:
                description = f"Strengthen {dim_name} focus on {target_mode} (currently {current_score:.0%})"
            else:
                description = (
                    f"Shift {dim_name} orientation from {current_mode} to {target_mode}"
                )

            suggestions.append(
                EditSuggestion(
                    dimension=dim_name,
                    current_mode=current_mode,
                    current_score=current_score,
                    target_mode=target_mode,
                    target_score=target_score,
                    priority=priority_idx + 1,
                    effort=effort,
                    description=description,
                )
            )

        return suggestions


class TransmigrationVisualizer:
    """
    Utilities for visualizing transmigration paths.

    Generates data structures suitable for plotting with matplotlib or web visualization.
    """

    @staticmethod
    def path_to_coordinates(
        path: TransmigrationPath,
    ) -> Dict[str, List[List[float]]]:
        """
        Convert path to coordinate lists for plotting.

        Args:
            path: TransmigrationPath object

        Returns:
            Dict with keys 'temporal', 'spatial', 'ontological', each containing
            a list of [x, y, z] coordinates for each step
        """
        coords = {
            "temporal": [],
            "spatial": [],
            "ontological": [],
        }

        for step in path.steps:
            coords["temporal"].append(step.temporal.tolist())
            coords["spatial"].append(step.spatial.tolist())
            coords["ontological"].append(step.ontological.tolist())

        return coords

    @staticmethod
    def path_to_2d_projection(
        path: TransmigrationPath,
        projection: str = "temporal_ontological",
    ) -> List[Tuple[float, float]]:
        """
        Project path to 2D for simple visualization.

        Args:
            path: TransmigrationPath object
            projection: Projection type ('temporal_ontological', 'temporal_spatial', etc.)

        Returns:
            List of (x, y) tuples for each step
        """
        points = []

        for step in path.steps:
            if projection == "temporal_ontological":
                # X: future vs past (step.temporal[2] - step.temporal[0])
                # Y: known vs imagined (step.ontological[2] - step.ontological[0])
                x = float(step.temporal[2] - step.temporal[0])
                y = float(step.ontological[2] - step.ontological[0])
            elif projection == "temporal_spatial":
                x = float(step.temporal[2] - step.temporal[0])
                y = float(step.spatial[2] - step.spatial[0])
            elif projection == "spatial_ontological":
                x = float(step.spatial[2] - step.spatial[0])
                y = float(step.ontological[2] - step.ontological[0])
            else:
                raise ValueError(f"Unknown projection: {projection}")

            points.append((x, y))

        return points

    @staticmethod
    def generate_plotly_data(
        path: TransmigrationPath,
    ) -> Dict:
        """
        Generate Plotly-compatible trace data for 3D visualization.

        Args:
            path: TransmigrationPath object

        Returns:
            Dict suitable for Plotly trace
        """
        # Use dominant mode indices as coordinates
        x = []  # temporal
        y = []  # spatial
        z = []  # ontological

        colors = []
        texts = []

        for step in path.steps:
            # Convert distributions to single coordinates
            # Use weighted centroid in [0, 1, 2] space
            t_coord = float(np.dot(step.temporal, [0, 1, 2]))
            s_coord = float(np.dot(step.spatial, [0, 1, 2]))
            o_coord = float(np.dot(step.ontological, [0, 1, 2]))

            x.append(t_coord)
            y.append(s_coord)
            z.append(o_coord)

            # Color by step type
            if step.step_type == "start":
                colors.append("green")
            elif step.step_type == "end":
                colors.append("red")
            else:
                colors.append("blue")

            texts.append(
                f"Step {step.step_index}: {step.combined_mode}<br>Album: {step.album}"
            )

        return {
            "type": "scatter3d",
            "mode": "lines+markers",
            "x": x,
            "y": y,
            "z": z,
            "marker": {
                "color": colors,
                "size": 8,
            },
            "line": {
                "color": "rgba(100, 100, 100, 0.5)",
                "width": 3,
            },
            "text": texts,
            "hoverinfo": "text",
        }


if __name__ == "__main__":
    print("Testing advanced transmigration module...")

    # Create test scores
    batch_size = 2
    temporal = torch.tensor(
        [
            [0.7, 0.2, 0.1],  # Past-dominant
            [0.3, 0.4, 0.3],  # Present-ish
        ]
    )
    spatial = torch.tensor(
        [
            [0.6, 0.3, 0.1],  # Thing-dominant
            [0.2, 0.5, 0.3],  # Place-ish
        ]
    )
    ontological = torch.tensor(
        [
            [0.8, 0.1, 0.1],  # Imagined-dominant
            [0.3, 0.3, 0.4],  # Known-ish
        ]
    )
    confidence = torch.tensor([[0.85], [0.6]])

    scores = OntologicalScores(
        temporal_scores=temporal,
        spatial_scores=spatial,
        ontological_scores=ontological,
        chromatic_confidence=confidence,
    )

    calc = AdvancedTransmigrationCalculator()

    # Test 1: Dimension priority
    print("\n=== Dimension Priority ===")
    priority = calc.rank_dimension_priority(
        temporal[0].numpy(),
        spatial[0].numpy(),
        ontological[0].numpy(),
        np.array([0, 0, 1]),  # Future
        np.array([0, 1, 0]),  # Place
        np.array([0, 0, 1]),  # Known
    )
    print(f"Priority: {priority}")

    # Test 2: Generate path to Blue
    print("\n=== Transmigration Path to Blue ===")
    path = calc.generate_transmigration_path(scores, "Blue", sample_idx=0)
    print(f"Steps: {path.n_steps}")
    print(f"Distance: {path.total_distance:.3f}")
    print(f"Feasible: {path.is_feasible}")
    print(f"Priority: {path.dimension_priority}")

    for step in path.steps:
        print(
            f"  Step {step.step_index} ({step.step_type}): {step.combined_mode} -> {step.album}"
        )

    # Test 3: Minimal edit suggestions
    print("\n=== Minimal Edit Suggestions ===")
    suggestions = calc.generate_minimal_edit_suggestions(scores, "Green", sample_idx=0)
    for s in suggestions:
        print(f"  [{s.priority}] {s.description} (effort: {s.effort})")

    # Test 4: Visualization data
    print("\n=== Visualization Data ===")
    coords = TransmigrationVisualizer.path_to_coordinates(path)
    print(f"Temporal coords: {len(coords['temporal'])} points")

    projection = TransmigrationVisualizer.path_to_2d_projection(path)
    print(f"2D projection: {projection}")

    plotly_data = TransmigrationVisualizer.generate_plotly_data(path)
    print(f"Plotly trace type: {plotly_data['type']}")

    print("\n✓ All transmigration tests passed!")
