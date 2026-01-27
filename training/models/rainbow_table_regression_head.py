"""
Rainbow Table Ontological Regression Head.

Predicts continuous scores for the three ontological dimensions
(temporal, spatial, ontological) plus chromatic confidence.

Total outputs: 10
- Temporal: [past, present, future] with softmax (sum to 1.0)
- Spatial: [thing, place, person] with softmax (sum to 1.0)
- Ontological: [imagined, forgotten, known] with softmax (sum to 1.0)
- Confidence: chromatic_confidence with sigmoid [0, 1]
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass


@dataclass
class OntologicalScores:
    """Container for ontological regression outputs."""

    # Dimension distributions (each sums to 1.0)
    temporal_scores: torch.Tensor  # [batch, 3] -> [past, present, future]
    spatial_scores: torch.Tensor  # [batch, 3] -> [thing, place, person]
    ontological_scores: torch.Tensor  # [batch, 3] -> [imagined, forgotten, known]

    # Overall confidence
    chromatic_confidence: torch.Tensor  # [batch, 1] -> [0, 1]

    # Optional uncertainty estimates
    temporal_uncertainty: Optional[torch.Tensor] = None
    spatial_uncertainty: Optional[torch.Tensor] = None
    ontological_uncertainty: Optional[torch.Tensor] = None

    def to_dict(self) -> Dict[str, torch.Tensor]:
        """Convert to dictionary format."""
        return {
            "temporal_scores": self.temporal_scores,
            "spatial_scores": self.spatial_scores,
            "ontological_scores": self.ontological_scores,
            "chromatic_confidence": self.chromatic_confidence,
        }

    def detach(self) -> "OntologicalScores":
        """Return detached copy."""
        return OntologicalScores(
            temporal_scores=self.temporal_scores.detach(),
            spatial_scores=self.spatial_scores.detach(),
            ontological_scores=self.ontological_scores.detach(),
            chromatic_confidence=self.chromatic_confidence.detach(),
            temporal_uncertainty=(
                self.temporal_uncertainty.detach()
                if self.temporal_uncertainty is not None
                else None
            ),
            spatial_uncertainty=(
                self.spatial_uncertainty.detach()
                if self.spatial_uncertainty is not None
                else None
            ),
            ontological_uncertainty=(
                self.ontological_uncertainty.detach()
                if self.ontological_uncertainty is not None
                else None
            ),
        )


# Constants for mode indices
TEMPORAL_MODES = ["past", "present", "future"]
SPATIAL_MODES = ["thing", "place", "person"]
ONTOLOGICAL_MODES = ["imagined", "forgotten", "known"]

# Album mapping from combined modes
ALBUM_MAPPING = {
    ("past", "thing", "imagined"): "Orange",
    ("past", "thing", "forgotten"): "Red",
    ("past", "thing", "known"): "Violet",
    ("past", "place", "imagined"): "Orange",
    ("past", "place", "forgotten"): "Red",
    ("past", "place", "known"): "Violet",
    ("past", "person", "imagined"): "Orange",
    ("past", "person", "forgotten"): "Red",
    ("past", "person", "known"): "Violet",
    ("present", "thing", "imagined"): "Yellow",
    ("present", "thing", "forgotten"): "Indigo",
    ("present", "thing", "known"): "Green",
    ("present", "place", "imagined"): "Yellow",
    ("present", "place", "forgotten"): "Indigo",
    ("present", "place", "known"): "Green",
    ("present", "person", "imagined"): "Yellow",
    ("present", "person", "forgotten"): "Indigo",
    ("present", "person", "known"): "Green",
    ("future", "thing", "imagined"): "Blue",
    ("future", "thing", "forgotten"): "Blue",
    ("future", "thing", "known"): "Blue",
    ("future", "place", "imagined"): "Blue",
    ("future", "place", "forgotten"): "Blue",
    ("future", "place", "known"): "Blue",
    ("future", "person", "imagined"): "Blue",
    ("future", "person", "forgotten"): "Blue",
    ("future", "person", "known"): "Blue",
}


class RainbowTableRegressionHead(nn.Module):
    """
    Ontological regression head for Rainbow Table mode prediction.

    Outputs 10 continuous values:
    - Temporal [3]: past, present, future (softmax)
    - Spatial [3]: thing, place, person (softmax)
    - Ontological [3]: imagined, forgotten, known (softmax)
    - Confidence [1]: chromatic_confidence (sigmoid)
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int] = [256, 128],
        dropout: float = 0.3,
        activation: str = "relu",
        predict_uncertainty: bool = False,
    ):
        """
        Initialize Rainbow Table regression head.

        Args:
            input_dim: Size of input embeddings
            hidden_dims: List of hidden layer sizes for shared representation
            dropout: Dropout probability
            activation: Hidden layer activation function
            predict_uncertainty: If True, also predict variance for each dimension
        """
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.predict_uncertainty = predict_uncertainty

        # Build shared MLP layers
        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))

            if activation == "relu":
                layers.append(nn.ReLU())
            elif activation == "gelu":
                layers.append(nn.GELU())
            elif activation == "tanh":
                layers.append(nn.Tanh())
            else:
                raise ValueError(f"Unknown activation: {activation}")

            if dropout > 0:
                layers.append(nn.Dropout(dropout))

            prev_dim = hidden_dim

        self.shared_layers = nn.Sequential(*layers)
        self.shared_dim = prev_dim

        # Per-dimension output heads (logits before softmax)
        self.temporal_head = nn.Linear(prev_dim, 3)
        self.spatial_head = nn.Linear(prev_dim, 3)
        self.ontological_head = nn.Linear(prev_dim, 3)

        # Confidence head (logit before sigmoid)
        self.confidence_head = nn.Linear(prev_dim, 1)

        # Optional uncertainty heads
        if predict_uncertainty:
            self.temporal_var_head = nn.Linear(prev_dim, 3)
            self.spatial_var_head = nn.Linear(prev_dim, 3)
            self.ontological_var_head = nn.Linear(prev_dim, 3)

    def forward(
        self,
        embeddings: torch.Tensor,
        temperature: float = 1.0,
    ) -> OntologicalScores:
        """
        Predict ontological scores from embeddings.

        Args:
            embeddings: Input embeddings [batch, input_dim]
            temperature: Softmax temperature (>1 = softer, <1 = sharper)

        Returns:
            OntologicalScores dataclass with all predictions
        """
        # Shared representation
        hidden = self.shared_layers(embeddings)

        # Per-dimension logits
        temporal_logits = self.temporal_head(hidden)
        spatial_logits = self.spatial_head(hidden)
        ontological_logits = self.ontological_head(hidden)
        confidence_logit = self.confidence_head(hidden)

        # Apply softmax to dimension distributions
        temporal_scores = F.softmax(temporal_logits / temperature, dim=-1)
        spatial_scores = F.softmax(spatial_logits / temperature, dim=-1)
        ontological_scores = F.softmax(ontological_logits / temperature, dim=-1)

        # Apply sigmoid to confidence
        chromatic_confidence = torch.sigmoid(confidence_logit)

        # Uncertainty estimation
        temporal_uncertainty = None
        spatial_uncertainty = None
        ontological_uncertainty = None

        if self.predict_uncertainty:
            temporal_uncertainty = F.softplus(self.temporal_var_head(hidden))
            spatial_uncertainty = F.softplus(self.spatial_var_head(hidden))
            ontological_uncertainty = F.softplus(self.ontological_var_head(hidden))

        return OntologicalScores(
            temporal_scores=temporal_scores,
            spatial_scores=spatial_scores,
            ontological_scores=ontological_scores,
            chromatic_confidence=chromatic_confidence,
            temporal_uncertainty=temporal_uncertainty,
            spatial_uncertainty=spatial_uncertainty,
            ontological_uncertainty=ontological_uncertainty,
        )

    def forward_logits(
        self,
        embeddings: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass returning raw logits (for loss computation).

        Args:
            embeddings: Input embeddings [batch, input_dim]

        Returns:
            Tuple of (temporal_logits, spatial_logits, ontological_logits, confidence_logit)
        """
        hidden = self.shared_layers(embeddings)

        temporal_logits = self.temporal_head(hidden)
        spatial_logits = self.spatial_head(hidden)
        ontological_logits = self.ontological_head(hidden)
        confidence_logit = self.confidence_head(hidden)

        return temporal_logits, spatial_logits, ontological_logits, confidence_logit

    def get_dominant_modes(
        self,
        scores: OntologicalScores,
    ) -> Tuple[List[str], List[str], List[str]]:
        """
        Get dominant mode for each dimension.

        Args:
            scores: OntologicalScores from forward pass

        Returns:
            Tuple of (temporal_modes, spatial_modes, ontological_modes) lists
        """
        temporal_idx = scores.temporal_scores.argmax(dim=-1).tolist()
        spatial_idx = scores.spatial_scores.argmax(dim=-1).tolist()
        ontological_idx = scores.ontological_scores.argmax(dim=-1).tolist()

        if not isinstance(temporal_idx, list):
            temporal_idx = [temporal_idx]
            spatial_idx = [spatial_idx]
            ontological_idx = [ontological_idx]

        temporal_modes = [TEMPORAL_MODES[i] for i in temporal_idx]
        spatial_modes = [SPATIAL_MODES[i] for i in spatial_idx]
        ontological_modes = [ONTOLOGICAL_MODES[i] for i in ontological_idx]

        return temporal_modes, spatial_modes, ontological_modes

    def predict_album(
        self,
        scores: OntologicalScores,
    ) -> List[str]:
        """
        Predict album assignment from ontological scores.

        Args:
            scores: OntologicalScores from forward pass

        Returns:
            List of album names (one per sample in batch)
        """
        temporal, spatial, ontological = self.get_dominant_modes(scores)

        albums = []
        for t, s, o in zip(temporal, spatial, ontological):
            album = ALBUM_MAPPING.get((t, s, o), "Black")
            albums.append(album)

        return albums

    def predict_combined_mode(
        self,
        scores: OntologicalScores,
    ) -> List[str]:
        """
        Predict combined mode string (e.g., "Past_Thing_Imagined").

        Args:
            scores: OntologicalScores from forward pass

        Returns:
            List of combined mode strings
        """
        temporal, spatial, ontological = self.get_dominant_modes(scores)

        modes = []
        for t, s, o in zip(temporal, spatial, ontological):
            mode = f"{t.capitalize()}_{s.capitalize()}_{o.capitalize()}"
            modes.append(mode)

        return modes


class HybridStateDetector:
    """
    Detects hybrid/liminal states in ontological predictions.

    States:
    - dominant: Top score > threshold (clear assignment)
    - hybrid: Top two scores within margin (liminal state)
    - diffuse: All scores near uniform (unclear)
    """

    def __init__(
        self,
        dominant_threshold: float = 0.6,
        hybrid_margin: float = 0.15,
        diffuse_threshold: float = 0.2,
    ):
        """
        Initialize hybrid state detector.

        Args:
            dominant_threshold: Minimum score for dominant classification
            hybrid_margin: Maximum difference between top 2 for hybrid
            diffuse_threshold: Maximum deviation from uniform for diffuse
        """
        self.dominant_threshold = dominant_threshold
        self.hybrid_margin = hybrid_margin
        self.diffuse_threshold = diffuse_threshold

    def detect_state(
        self,
        scores: torch.Tensor,
        mode_names: List[str],
    ) -> List[Dict]:
        """
        Detect state for a single dimension.

        Args:
            scores: Score tensor [batch, 3]
            mode_names: List of mode names for this dimension

        Returns:
            List of state dictionaries with keys:
                - state: "dominant", "hybrid", or "diffuse"
                - top_mode: Primary mode name
                - top_score: Primary mode score
                - secondary_mode: Secondary mode (for hybrid)
                - secondary_score: Secondary score (for hybrid)
        """
        batch_size = scores.shape[0]
        results = []

        for i in range(batch_size):
            sample_scores = scores[i]

            # Sort scores descending
            sorted_scores, sorted_indices = torch.sort(sample_scores, descending=True)

            top_score = sorted_scores[0].item()
            second_score = sorted_scores[1].item()
            third_score = sorted_scores[2].item()

            top_idx = sorted_indices[0].item()
            second_idx = sorted_indices[1].item()

            result = {
                "top_mode": mode_names[top_idx],
                "top_score": top_score,
                "secondary_mode": mode_names[second_idx],
                "secondary_score": second_score,
            }

            # Check for diffuse state (all near uniform 0.33)
            uniform = 1.0 / 3.0
            max_dev = max(
                abs(top_score - uniform),
                abs(second_score - uniform),
                abs(third_score - uniform),
            )

            if max_dev <= self.diffuse_threshold:
                result["state"] = "diffuse"
            # Check for dominant state
            elif top_score >= self.dominant_threshold:
                result["state"] = "dominant"
            # Check for hybrid state
            elif (top_score - second_score) <= self.hybrid_margin:
                result["state"] = "hybrid"
            else:
                # Between dominant threshold and hybrid margin
                result["state"] = "partial"

            results.append(result)

        return results

    def analyze_full_state(
        self,
        scores: OntologicalScores,
    ) -> List[Dict]:
        """
        Analyze full ontological state including all dimensions.

        Args:
            scores: OntologicalScores from regression head

        Returns:
            List of analysis dictionaries per sample
        """
        temporal_states = self.detect_state(scores.temporal_scores, TEMPORAL_MODES)
        spatial_states = self.detect_state(scores.spatial_scores, SPATIAL_MODES)
        ontological_states = self.detect_state(
            scores.ontological_scores, ONTOLOGICAL_MODES
        )

        batch_size = scores.temporal_scores.shape[0]
        results = []

        for i in range(batch_size):
            confidence = scores.chromatic_confidence[i].item()

            t_state = temporal_states[i]
            s_state = spatial_states[i]
            o_state = ontological_states[i]

            # Count diffuse dimensions
            diffuse_count = sum(
                1 for s in [t_state, s_state, o_state] if s["state"] == "diffuse"
            )

            # Count hybrid dimensions
            hybrid_count = sum(
                1 for s in [t_state, s_state, o_state] if s["state"] == "hybrid"
            )

            # Determine overall classification
            if diffuse_count == 3:
                overall_state = "black_album_candidate"
            elif diffuse_count >= 2:
                overall_state = "highly_diffuse"
            elif hybrid_count >= 2 and diffuse_count == 0:
                overall_state = "multi_hybrid"
            elif hybrid_count >= 1:
                overall_state = "partial_hybrid"
            elif all(s["state"] == "dominant" for s in [t_state, s_state, o_state]):
                overall_state = "dominant"
            else:
                overall_state = "mixed"

            # Build hybrid flags
            hybrid_flags = []
            if t_state["state"] == "hybrid":
                hybrid_flags.append(
                    f"temporal_hybrid_{t_state['top_mode']}_{t_state['secondary_mode']}"
                )
            if s_state["state"] == "hybrid":
                hybrid_flags.append(
                    f"spatial_hybrid_{s_state['top_mode']}_{s_state['secondary_mode']}"
                )
            if o_state["state"] == "hybrid":
                hybrid_flags.append(
                    f"ontological_hybrid_{o_state['top_mode']}_{o_state['secondary_mode']}"
                )

            if t_state["state"] == "diffuse":
                hybrid_flags.append("temporal_diffuse")
            if s_state["state"] == "diffuse":
                hybrid_flags.append("spatial_diffuse")
            if o_state["state"] == "diffuse":
                hybrid_flags.append("ontological_diffuse")

            results.append(
                {
                    "overall_state": overall_state,
                    "chromatic_confidence": confidence,
                    "temporal": t_state,
                    "spatial": s_state,
                    "ontological": o_state,
                    "diffuse_count": diffuse_count,
                    "hybrid_count": hybrid_count,
                    "hybrid_flags": hybrid_flags,
                }
            )

        return results


class TransmigrationCalculator:
    """
    Computes transmigration distances between ontological states.
    """

    @staticmethod
    def dimension_distance(
        source: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute L2 distance between score vectors.

        Args:
            source: Source scores [batch, 3] or [3]
            target: Target scores [batch, 3] or [3]

        Returns:
            Distance tensor [batch] or scalar
        """
        return torch.norm(source - target, p=2, dim=-1)

    @staticmethod
    def total_distance(
        source_scores: OntologicalScores,
        target_scores: OntologicalScores,
    ) -> torch.Tensor:
        """
        Compute total transmigration distance across all dimensions.

        distance = sqrt(temporal² + spatial² + ontological²)

        Args:
            source_scores: Source ontological scores
            target_scores: Target ontological scores

        Returns:
            Total distance [batch]
        """
        t_dist = TransmigrationCalculator.dimension_distance(
            source_scores.temporal_scores,
            target_scores.temporal_scores,
        )
        s_dist = TransmigrationCalculator.dimension_distance(
            source_scores.spatial_scores,
            target_scores.spatial_scores,
        )
        o_dist = TransmigrationCalculator.dimension_distance(
            source_scores.ontological_scores,
            target_scores.ontological_scores,
        )

        return torch.sqrt(t_dist**2 + s_dist**2 + o_dist**2)

    @staticmethod
    def distance_to_mode(
        scores: OntologicalScores,
        target_temporal: str,
        target_spatial: str,
        target_ontological: str,
    ) -> torch.Tensor:
        """
        Compute distance from current scores to a specific mode.

        Args:
            scores: Current ontological scores
            target_temporal: Target temporal mode (past, present, future)
            target_spatial: Target spatial mode (thing, place, person)
            target_ontological: Target ontological mode (imagined, forgotten, known)

        Returns:
            Distance tensor [batch]
        """
        device = scores.temporal_scores.device

        # Create one-hot targets
        t_idx = TEMPORAL_MODES.index(target_temporal)
        s_idx = SPATIAL_MODES.index(target_spatial)
        o_idx = ONTOLOGICAL_MODES.index(target_ontological)

        t_target = F.one_hot(torch.tensor(t_idx, device=device), num_classes=3).float()
        s_target = F.one_hot(torch.tensor(s_idx, device=device), num_classes=3).float()
        o_target = F.one_hot(torch.tensor(o_idx, device=device), num_classes=3).float()

        # Expand for batch
        batch_size = scores.temporal_scores.shape[0]
        t_target = t_target.unsqueeze(0).expand(batch_size, -1)
        s_target = s_target.unsqueeze(0).expand(batch_size, -1)
        o_target = o_target.unsqueeze(0).expand(batch_size, -1)

        t_dist = TransmigrationCalculator.dimension_distance(
            scores.temporal_scores, t_target
        )
        s_dist = TransmigrationCalculator.dimension_distance(
            scores.spatial_scores, s_target
        )
        o_dist = TransmigrationCalculator.dimension_distance(
            scores.ontological_scores, o_target
        )

        return torch.sqrt(t_dist**2 + s_dist**2 + o_dist**2)

    @staticmethod
    def distances_to_all_albums(
        scores: OntologicalScores,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute distances to canonical positions of all albums.

        Args:
            scores: Current ontological scores

        Returns:
            Dictionary mapping album names to distance tensors
        """
        # Define canonical album modes (most representative)
        album_modes = {
            "Orange": ("past", "thing", "imagined"),
            "Red": ("past", "thing", "forgotten"),
            "Violet": ("past", "person", "known"),
            "Yellow": ("present", "place", "imagined"),
            "Green": ("present", "person", "known"),
            "Indigo": ("present", "person", "forgotten"),
            "Blue": ("future", "place", "known"),
            "Black": None,  # Special case: uniform distribution
        }

        distances = {}
        for album, modes in album_modes.items():
            if modes is None:
                # Black album: distance to uniform
                uniform = torch.ones_like(scores.temporal_scores) / 3.0
                t_dist = TransmigrationCalculator.dimension_distance(
                    scores.temporal_scores, uniform
                )
                s_dist = TransmigrationCalculator.dimension_distance(
                    scores.spatial_scores, uniform
                )
                o_dist = TransmigrationCalculator.dimension_distance(
                    scores.ontological_scores, uniform
                )
                distances[album] = torch.sqrt(t_dist**2 + s_dist**2 + o_dist**2)
            else:
                distances[album] = TransmigrationCalculator.distance_to_mode(
                    scores, modes[0], modes[1], modes[2]
                )

        return distances


if __name__ == "__main__":
    # Quick tests
    print("Testing RainbowTableRegressionHead...")

    batch_size = 4
    input_dim = 768
    embeddings = torch.randn(batch_size, input_dim)

    # Test 1: Basic forward pass
    print("\n=== Basic forward pass ===")
    head = RainbowTableRegressionHead(
        input_dim=input_dim,
        hidden_dims=[256, 128],
        dropout=0.3,
    )

    scores = head(embeddings)
    print(f"Temporal scores shape: {scores.temporal_scores.shape}")
    print(f"Spatial scores shape: {scores.spatial_scores.shape}")
    print(f"Ontological scores shape: {scores.ontological_scores.shape}")
    print(f"Confidence shape: {scores.chromatic_confidence.shape}")

    # Verify softmax sums to 1
    t_sum = scores.temporal_scores.sum(dim=-1)
    s_sum = scores.spatial_scores.sum(dim=-1)
    o_sum = scores.ontological_scores.sum(dim=-1)
    print(f"Temporal sum: {t_sum} (should be ~1.0)")
    print(f"Spatial sum: {s_sum} (should be ~1.0)")
    print(f"Ontological sum: {o_sum} (should be ~1.0)")

    print(
        "✓ Test passed!"
        if all(
            [
                scores.temporal_scores.shape == (batch_size, 3),
                scores.spatial_scores.shape == (batch_size, 3),
                scores.ontological_scores.shape == (batch_size, 3),
                scores.chromatic_confidence.shape == (batch_size, 1),
                torch.allclose(t_sum, torch.ones(batch_size), atol=1e-5),
            ]
        )
        else "✗ Test failed!"
    )

    # Test 2: Mode prediction
    print("\n=== Mode prediction ===")
    temporal, spatial, ontological = head.get_dominant_modes(scores)
    print(f"Temporal modes: {temporal}")
    print(f"Spatial modes: {spatial}")
    print(f"Ontological modes: {ontological}")

    combined = head.predict_combined_mode(scores)
    print(f"Combined modes: {combined}")

    albums = head.predict_album(scores)
    print(f"Albums: {albums}")
    print("✓ Mode prediction works!")

    # Test 3: Hybrid state detection
    print("\n=== Hybrid state detection ===")
    detector = HybridStateDetector()
    analysis = detector.analyze_full_state(scores)
    for i, a in enumerate(analysis):
        print(
            f"Sample {i}: {a['overall_state']}, confidence={a['chromatic_confidence']:.3f}"
        )
        if a["hybrid_flags"]:
            print(f"  Flags: {a['hybrid_flags']}")
    print("✓ Hybrid detection works!")

    # Test 4: Transmigration distance
    print("\n=== Transmigration distance ===")
    calc = TransmigrationCalculator()
    distances = calc.distances_to_all_albums(scores)
    print("Distances to albums:")
    for album, dist in distances.items():
        print(f"  {album}: {dist.mean().item():.3f}")
    print("✓ Transmigration calculation works!")

    # Test 5: With uncertainty
    print("\n=== With uncertainty estimation ===")
    head_unc = RainbowTableRegressionHead(
        input_dim=input_dim,
        hidden_dims=[256],
        predict_uncertainty=True,
    )
    scores_unc = head_unc(embeddings)
    print(f"Temporal uncertainty shape: {scores_unc.temporal_uncertainty.shape}")
    print(f"Uncertainty values: {scores_unc.temporal_uncertainty[0]}")
    print("✓ Uncertainty estimation works!")
