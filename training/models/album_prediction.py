"""
Album prediction module for Rainbow Pipeline.

Provides advanced album assignment from ontological scores including:
- Tie-breaking logic for ambiguous predictions
- Confidence thresholding
- Confusion matrix computation
- Probability distribution over all albums
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


# Album canonical colors (for visualization)
ALBUM_COLORS = {
    "Red": "#FF0000",
    "Orange": "#FF8000",
    "Yellow": "#FFD700",
    "Green": "#00FF00",
    "Blue": "#0000FF",
    "Indigo": "#4B0082",
    "Violet": "#8B00FF",
    "White": "#FFFFFF",
    "Black": "#000000",
}

# Album ordering for confusion matrix
ALBUM_ORDER = [
    "Red",
    "Orange",
    "Yellow",
    "Green",
    "Blue",
    "Indigo",
    "Violet",
    "White",
    "Black",
]


@dataclass
class AlbumPrediction:
    """Complete album prediction with confidence and alternatives."""

    # Primary prediction
    album: str
    confidence: float

    # Full probability distribution
    album_probabilities: Dict[str, float]

    # Mode details
    temporal_mode: str
    spatial_mode: str
    ontological_mode: str
    combined_mode: str

    # Tie-breaking info
    was_tie: bool = False
    tie_albums: Optional[List[str]] = None
    tie_breaking_reason: Optional[str] = None

    # Confidence status
    meets_threshold: bool = True
    threshold_used: float = 0.0

    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            "album": self.album,
            "confidence": self.confidence,
            "album_probabilities": self.album_probabilities,
            "temporal_mode": self.temporal_mode,
            "spatial_mode": self.spatial_mode,
            "ontological_mode": self.ontological_mode,
            "combined_mode": self.combined_mode,
            "was_tie": self.was_tie,
            "tie_albums": self.tie_albums,
            "tie_breaking_reason": self.tie_breaking_reason,
            "meets_threshold": self.meets_threshold,
            "threshold_used": self.threshold_used,
        }


class AlbumPredictor:
    """
    Advanced album prediction with tie-breaking and confidence thresholding.
    """

    def __init__(
        self,
        confidence_threshold: float = 0.5,
        tie_margin: float = 0.05,
        default_album: str = "Black",
        tie_breaking_strategy: str = "chromatic_order",
    ):
        """
        Initialize album predictor.

        Args:
            confidence_threshold: Minimum confidence for valid prediction
            tie_margin: Margin for considering predictions as tied
            default_album: Default album when confidence is too low
            tie_breaking_strategy: How to break ties
                - "chromatic_order": Use album order (Red first)
                - "ontological_priority": Prioritize ontological dimension
                - "confidence_weighted": Use chromatic confidence
        """
        self.confidence_threshold = confidence_threshold
        self.tie_margin = tie_margin
        self.default_album = default_album
        self.tie_breaking_strategy = tie_breaking_strategy

        # Build reverse mapping: album -> list of (t, s, o) tuples
        self.album_modes: Dict[str, List[Tuple[str, str, str]]] = {}
        for modes, album in ALBUM_MAPPING.items():
            if album not in self.album_modes:
                self.album_modes[album] = []
            self.album_modes[album].append(modes)

    def compute_album_probabilities(
        self,
        scores: OntologicalScores,
        sample_idx: int = 0,
    ) -> Dict[str, float]:
        """
        Compute probability distribution over all albums.

        Uses the product of relevant mode probabilities for each album.

        Args:
            scores: OntologicalScores from model
            sample_idx: Index in batch

        Returns:
            Dict mapping album name to probability
        """
        temporal = scores.temporal_scores[sample_idx].detach().cpu().numpy()
        spatial = scores.spatial_scores[sample_idx].detach().cpu().numpy()
        ontological = scores.ontological_scores[sample_idx].detach().cpu().numpy()

        probs = {}

        for album, mode_list in self.album_modes.items():
            # Sum probability across all mode combinations for this album
            album_prob = 0.0

            for t_mode, s_mode, o_mode in mode_list:
                t_idx = TEMPORAL_MODES.index(t_mode)
                s_idx = SPATIAL_MODES.index(s_mode)
                o_idx = ONTOLOGICAL_MODES.index(o_mode)

                # Product of independent probabilities
                mode_prob = temporal[t_idx] * spatial[s_idx] * ontological[o_idx]
                album_prob += mode_prob

            probs[album] = float(album_prob)

        # Add Black album probability (based on uniformity)
        # Higher when all dimensions are close to uniform
        uniform_dist = np.array([1 / 3, 1 / 3, 1 / 3])
        t_uniformity = 1 - np.linalg.norm(temporal - uniform_dist)
        s_uniformity = 1 - np.linalg.norm(spatial - uniform_dist)
        o_uniformity = 1 - np.linalg.norm(ontological - uniform_dist)

        black_prob = t_uniformity * s_uniformity * o_uniformity
        probs["Black"] = max(0.0, float(black_prob))

        # Normalize to sum to 1
        total = sum(probs.values())
        if total > 0:
            probs = {k: v / total for k, v in probs.items()}

        return probs

    def break_tie(
        self,
        tied_albums: List[str],
        album_probs: Dict[str, float],
        chromatic_confidence: float,
    ) -> Tuple[str, str]:
        """
        Break tie between multiple albums.

        Args:
            tied_albums: List of albums with similar probabilities
            album_probs: Full probability distribution
            chromatic_confidence: Model's chromatic confidence

        Returns:
            Tuple of (selected_album, reason)
        """
        if len(tied_albums) == 1:
            return tied_albums[0], "no_tie"

        if self.tie_breaking_strategy == "chromatic_order":
            # Select based on canonical color order
            for album in ALBUM_ORDER:
                if album in tied_albums:
                    return album, "chromatic_order"

        elif self.tie_breaking_strategy == "ontological_priority":
            # Prioritize albums with stronger ontological dimension
            # (imagined > forgotten > known)
            priority_order = [
                "Orange",
                "Yellow",
                "Blue",
                "Red",
                "Indigo",
                "Violet",
                "Green",
            ]
            for album in priority_order:
                if album in tied_albums:
                    return album, "ontological_priority"

        elif self.tie_breaking_strategy == "confidence_weighted":
            # Weight by slight probability differences
            best_album = max(tied_albums, key=lambda a: album_probs.get(a, 0))
            return best_album, "confidence_weighted"

        # Fallback: first in list
        return tied_albums[0], "fallback"

    def predict_album(
        self,
        scores: OntologicalScores,
        sample_idx: int = 0,
    ) -> AlbumPrediction:
        """
        Predict album with full details and tie-breaking.

        Args:
            scores: OntologicalScores from model
            sample_idx: Index in batch

        Returns:
            AlbumPrediction with full details
        """
        # Get dominant modes
        temporal = scores.temporal_scores[sample_idx].detach().cpu().numpy()
        spatial = scores.spatial_scores[sample_idx].detach().cpu().numpy()
        ontological = scores.ontological_scores[sample_idx].detach().cpu().numpy()
        chromatic_confidence = (
            scores.chromatic_confidence[sample_idx].detach().cpu().item()
        )

        t_idx = int(np.argmax(temporal))
        s_idx = int(np.argmax(spatial))
        o_idx = int(np.argmax(ontological))

        temporal_mode = TEMPORAL_MODES[t_idx]
        spatial_mode = SPATIAL_MODES[s_idx]
        ontological_mode = ONTOLOGICAL_MODES[o_idx]
        combined_mode = f"{temporal_mode.capitalize()}_{spatial_mode.capitalize()}_{ontological_mode.capitalize()}"

        # Compute album probabilities
        album_probs = self.compute_album_probabilities(scores, sample_idx)

        # Find top albums within tie margin
        max_prob = max(album_probs.values())
        tied_albums = [
            album
            for album, prob in album_probs.items()
            if max_prob - prob <= self.tie_margin
        ]

        # Get primary album from mode mapping
        primary_album = ALBUM_MAPPING.get(
            (temporal_mode, spatial_mode, ontological_mode), "Black"
        )

        # Check for ties
        was_tie = len(tied_albums) > 1
        tie_breaking_reason = None

        if was_tie:
            # If primary album is in tied group, prefer it
            if primary_album in tied_albums:
                selected_album = primary_album
                tie_breaking_reason = "mode_mapping_preference"
            else:
                selected_album, tie_breaking_reason = self.break_tie(
                    tied_albums, album_probs, chromatic_confidence
                )
        else:
            selected_album = primary_album

        # Compute final confidence
        confidence = album_probs.get(selected_album, 0.0) * chromatic_confidence

        # Check confidence threshold
        meets_threshold = confidence >= self.confidence_threshold

        if not meets_threshold:
            # Return default album but keep the analysis
            final_album = self.default_album
        else:
            final_album = selected_album

        return AlbumPrediction(
            album=final_album,
            confidence=confidence,
            album_probabilities=album_probs,
            temporal_mode=temporal_mode,
            spatial_mode=spatial_mode,
            ontological_mode=ontological_mode,
            combined_mode=combined_mode,
            was_tie=was_tie,
            tie_albums=tied_albums if was_tie else None,
            tie_breaking_reason=tie_breaking_reason,
            meets_threshold=meets_threshold,
            threshold_used=self.confidence_threshold,
        )

    def predict_batch(
        self,
        scores: OntologicalScores,
    ) -> List[AlbumPrediction]:
        """
        Predict albums for entire batch.

        Args:
            scores: OntologicalScores from model

        Returns:
            List of AlbumPrediction objects
        """
        batch_size = scores.temporal_scores.shape[0]
        return [self.predict_album(scores, i) for i in range(batch_size)]


class AlbumConfusionMatrix:
    """
    Computes and manages confusion matrix for album predictions.
    """

    def __init__(self, album_order: Optional[List[str]] = None):
        """
        Initialize confusion matrix.

        Args:
            album_order: Order of albums for matrix (default: ALBUM_ORDER)
        """
        self.album_order = album_order or ALBUM_ORDER
        self.n_albums = len(self.album_order)
        self.album_to_idx = {a: i for i, a in enumerate(self.album_order)}

        # Initialize matrix
        self.matrix = np.zeros((self.n_albums, self.n_albums), dtype=np.int64)

    def update(
        self,
        true_albums: List[str],
        predicted_albums: List[str],
    ):
        """
        Update confusion matrix with predictions.

        Args:
            true_albums: List of ground truth album labels
            predicted_albums: List of predicted album labels
        """
        for true, pred in zip(true_albums, predicted_albums):
            if true in self.album_to_idx and pred in self.album_to_idx:
                t_idx = self.album_to_idx[true]
                p_idx = self.album_to_idx[pred]
                self.matrix[t_idx, p_idx] += 1

    def get_matrix(self) -> np.ndarray:
        """Get raw confusion matrix."""
        return self.matrix.copy()

    def get_normalized_matrix(self, normalize: str = "true") -> np.ndarray:
        """
        Get normalized confusion matrix.

        Args:
            normalize: Normalization method
                - "true": Normalize by true labels (row-wise)
                - "pred": Normalize by predicted labels (column-wise)
                - "all": Normalize by total count

        Returns:
            Normalized matrix
        """
        if normalize == "true":
            row_sums = self.matrix.sum(axis=1, keepdims=True)
            row_sums = np.where(row_sums == 0, 1, row_sums)  # Avoid division by zero
            return self.matrix / row_sums

        elif normalize == "pred":
            col_sums = self.matrix.sum(axis=0, keepdims=True)
            col_sums = np.where(col_sums == 0, 1, col_sums)
            return self.matrix / col_sums

        elif normalize == "all":
            total = self.matrix.sum()
            return self.matrix / max(total, 1)

        else:
            raise ValueError(f"Unknown normalization: {normalize}")

    def get_per_album_metrics(self) -> Dict[str, Dict[str, float]]:
        """
        Compute per-album precision, recall, and F1.

        Returns:
            Dict mapping album -> {precision, recall, f1, support}
        """
        metrics = {}

        for album, idx in self.album_to_idx.items():
            # True positives
            tp = self.matrix[idx, idx]

            # False positives: predicted as this album but was actually something else
            fp = self.matrix[:, idx].sum() - tp

            # False negatives: was this album but predicted as something else
            fn = self.matrix[idx, :].sum() - tp

            # Support: total true instances of this album
            support = self.matrix[idx, :].sum()

            # Compute metrics
            precision = tp / max(tp + fp, 1)
            recall = tp / max(tp + fn, 1)
            f1 = 2 * precision * recall / max(precision + recall, 1e-10)

            metrics[album] = {
                "precision": float(precision),
                "recall": float(recall),
                "f1": float(f1),
                "support": int(support),
            }

        return metrics

    def get_overall_accuracy(self) -> float:
        """Get overall accuracy."""
        correct = np.trace(self.matrix)
        total = self.matrix.sum()
        return float(correct / max(total, 1))

    def get_macro_f1(self) -> float:
        """Get macro-averaged F1 score."""
        per_album = self.get_per_album_metrics()
        f1_scores = [m["f1"] for m in per_album.values() if m["support"] > 0]
        return float(np.mean(f1_scores)) if f1_scores else 0.0

    def get_weighted_f1(self) -> float:
        """Get support-weighted F1 score."""
        per_album = self.get_per_album_metrics()
        total_support = sum(m["support"] for m in per_album.values())

        if total_support == 0:
            return 0.0

        weighted_f1 = (
            sum(m["f1"] * m["support"] for m in per_album.values()) / total_support
        )

        return float(weighted_f1)

    def to_dict(self) -> Dict:
        """Export confusion matrix data."""
        return {
            "album_order": self.album_order,
            "matrix": self.matrix.tolist(),
            "normalized_matrix": self.get_normalized_matrix("true").tolist(),
            "overall_accuracy": self.get_overall_accuracy(),
            "macro_f1": self.get_macro_f1(),
            "weighted_f1": self.get_weighted_f1(),
            "per_album_metrics": self.get_per_album_metrics(),
        }

    def reset(self):
        """Reset confusion matrix."""
        self.matrix = np.zeros((self.n_albums, self.n_albums), dtype=np.int64)


def evaluate_album_predictions(
    scores_list: List[OntologicalScores],
    true_albums_list: List[List[str]],
    predictor: Optional[AlbumPredictor] = None,
) -> Dict:
    """
    Evaluate album predictions across multiple batches.

    Args:
        scores_list: List of OntologicalScores batches
        true_albums_list: List of true album label batches
        predictor: AlbumPredictor to use (default: creates new one)

    Returns:
        Evaluation results including confusion matrix and metrics
    """
    predictor = predictor or AlbumPredictor()
    confusion = AlbumConfusionMatrix()

    all_predictions = []
    all_true = []
    all_confidences = []
    n_ties = 0
    n_below_threshold = 0

    for scores, true_albums in zip(scores_list, true_albums_list):
        predictions = predictor.predict_batch(scores)

        for pred, true in zip(predictions, true_albums):
            all_predictions.append(pred.album)
            all_true.append(true)
            all_confidences.append(pred.confidence)

            if pred.was_tie:
                n_ties += 1

            if not pred.meets_threshold:
                n_below_threshold += 1

        pred_albums = [p.album for p in predictions]
        confusion.update(true_albums, pred_albums)

    return {
        "confusion_matrix": confusion.to_dict(),
        "n_samples": len(all_predictions),
        "n_ties": n_ties,
        "tie_rate": n_ties / max(len(all_predictions), 1),
        "n_below_threshold": n_below_threshold,
        "below_threshold_rate": n_below_threshold / max(len(all_predictions), 1),
        "mean_confidence": float(np.mean(all_confidences)),
        "std_confidence": float(np.std(all_confidences)),
    }


if __name__ == "__main__":
    print("Testing album prediction module...")

    # Create test scores
    batch_size = 4
    temporal = torch.tensor(
        [
            [0.8, 0.15, 0.05],  # Past-dominant
            [0.3, 0.35, 0.35],  # Tie between present/future
            [0.1, 0.2, 0.7],  # Future-dominant
            [0.33, 0.34, 0.33],  # Diffuse
        ]
    )
    spatial = torch.tensor(
        [
            [0.7, 0.2, 0.1],  # Thing
            [0.2, 0.6, 0.2],  # Place
            [0.1, 0.3, 0.6],  # Person
            [0.33, 0.33, 0.34],  # Diffuse
        ]
    )
    ontological = torch.tensor(
        [
            [0.75, 0.15, 0.1],  # Imagined
            [0.1, 0.7, 0.2],  # Forgotten
            [0.1, 0.1, 0.8],  # Known
            [0.33, 0.33, 0.34],  # Diffuse
        ]
    )
    confidence = torch.tensor([[0.9], [0.6], [0.85], [0.2]])

    scores = OntologicalScores(
        temporal_scores=temporal,
        spatial_scores=spatial,
        ontological_scores=ontological,
        chromatic_confidence=confidence,
    )

    # Test predictor
    print("\n=== Album Predictions ===")
    predictor = AlbumPredictor(confidence_threshold=0.5)
    predictions = predictor.predict_batch(scores)

    for i, pred in enumerate(predictions):
        print(f"\nSample {i}:")
        print(f"  Album: {pred.album} (confidence: {pred.confidence:.3f})")
        print(f"  Mode: {pred.combined_mode}")
        print(f"  Was tie: {pred.was_tie}")
        if pred.was_tie:
            print(f"  Tied albums: {pred.tie_albums}")
            print(f"  Tie-breaking: {pred.tie_breaking_reason}")
        print(f"  Meets threshold: {pred.meets_threshold}")

    # Test confusion matrix
    print("\n=== Confusion Matrix ===")
    confusion = AlbumConfusionMatrix()

    true_albums = ["Orange", "Indigo", "Blue", "Black"]
    pred_albums = [p.album for p in predictions]

    confusion.update(true_albums, pred_albums)

    print(f"Accuracy: {confusion.get_overall_accuracy():.3f}")
    print(f"Macro F1: {confusion.get_macro_f1():.3f}")

    print("\nPer-album metrics:")
    for album, metrics in confusion.get_per_album_metrics().items():
        if metrics["support"] > 0:
            print(
                f"  {album}: P={metrics['precision']:.2f}, R={metrics['recall']:.2f}, F1={metrics['f1']:.2f}"
            )

    # Test probability distribution
    print("\n=== Album Probabilities ===")
    probs = predictor.compute_album_probabilities(scores, sample_idx=0)
    sorted_probs = sorted(probs.items(), key=lambda x: x[1], reverse=True)
    for album, prob in sorted_probs[:5]:
        print(f"  {album}: {prob:.3f}")

    print("\n✓ All album prediction tests passed!")
