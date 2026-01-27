#!/usr/bin/env python3
"""
Concept Validation Chain - SEPARATE from main workflow

Run this in /training directory to validate concepts that were already generated.
Requires torch (GPU optional but faster).

Usage:
    # Validate concepts from a thread
    python validate_concepts.py --thread-id f410b5f7-abc-def

    # Validate all recent concepts
    python validate_concepts.py --recent 10

    # Validate a single concept file
    python validate_concepts.py --concept-file /path/to/concept.yml

    # Batch validate from directory
    python validate_concepts.py --concepts-dir /chain_artifacts/
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
import yaml
import json
import argparse
from typing import Dict, List, Optional
from dataclasses import dataclass, asdict
from enum import Enum
import sys

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))


# ============================================================================
# VALIDATION RESULT TYPES (Same as before, but standalone)
# ============================================================================


class ValidationStatus(str, Enum):
    """Validation decision status"""

    ACCEPT = "ACCEPT"
    ACCEPT_HYBRID = "ACCEPT_HYBRID"
    ACCEPT_BLACK = "ACCEPT_BLACK"
    REJECT = "REJECT"


@dataclass
class ValidationResult:
    """Structured result from concept validation"""

    concept_text: str
    temporal_scores: Dict[str, float]
    spatial_scores: Dict[str, float]
    ontological_scores: Dict[str, float]
    chromatic_confidence: float
    predicted_album: str
    predicted_mode: str
    validation_status: ValidationStatus
    hybrid_flags: List[str]
    transmigration_distances: Dict[str, float]
    suggestions: List[str]
    rejection_reason: Optional[str] = None


# ============================================================================
# MINIMAL MODEL ARCHITECTURE (matches training)
# ============================================================================


class RegressionHead(nn.Module):
    """Rainbow Table regression head - matches training architecture"""

    def __init__(self, input_dim=768, hidden_dim=256, dropout=0.3):
        super().__init__()

        self.shared = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        self.temporal_head = nn.Linear(hidden_dim, 3)
        self.spatial_head = nn.Linear(hidden_dim, 3)
        self.ontological_head = nn.Linear(hidden_dim, 3)
        self.confidence_head = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        x = self.shared(x)
        return {
            "temporal": F.softmax(self.temporal_head(x), dim=-1),
            "spatial": F.softmax(self.spatial_head(x), dim=-1),
            "ontological": F.softmax(self.ontological_head(x), dim=-1),
            "confidence": torch.sigmoid(self.confidence_head(x)),
        }


# ============================================================================
# CONCEPT VALIDATOR (Standalone, no LangGraph)
# ============================================================================


class ConceptValidator:
    """
    Validates concepts using Phase 4 regression model.
    Runs separately from main workflow - no dependencies on LangGraph.
    """

    ALBUM_MAP = {
        ("Past", "Thing", "Imagined"): "Orange",
        ("Past", "Thing", "Known"): "Red",
        ("Future", "Place", "Imagined"): "Yellow",
        ("Future", "Place", "Forgotten"): "Green",
        ("Present", "Person", "Forgotten"): "Blue",
        ("None", "None", "None"): "Black",
    }

    MODE_NAMES = {
        "temporal": ["Past", "Present", "Future"],
        "spatial": ["Thing", "Place", "Person"],
        "ontological": ["Imagined", "Forgotten", "Known"],
    }

    def __init__(
        self,
        model_path: str,
        device: str = "cpu",
        confidence_threshold: float = 0.7,
        dominant_threshold: float = 0.6,
        hybrid_threshold: float = 0.15,
        diffuse_threshold: float = 0.2,
    ):
        self.device = device
        self.confidence_threshold = confidence_threshold
        self.dominant_threshold = dominant_threshold
        self.hybrid_threshold = hybrid_threshold
        self.diffuse_threshold = diffuse_threshold

        # Load model
        self.model = RegressionHead(
            input_dim=768, hidden_dim=256, dropout=0.0  # No dropout at inference
        )

        if Path(model_path).exists():
            state_dict = torch.load(model_path, map_location=device)
            self.model.load_state_dict(state_dict)
            print(f"✅ Loaded model from {model_path}")
        else:
            print(f"⚠️  Model not found: {model_path}")
            print("   Using random weights (for testing only!)")

        self.model.to(device)
        self.model.eval()

    def validate_concept(self, concept_text: str) -> ValidationResult:
        """Validate a concept and return structured result"""

        # TODO: Extract real features
        # For now using random embedding as placeholder
        embedding = torch.randn(1, 768).to(self.device)

        # Get predictions
        with torch.no_grad():
            predictions = self.model(embedding)

        # Convert to numpy
        temporal = predictions["temporal"][0].cpu().numpy()
        spatial = predictions["spatial"][0].cpu().numpy()
        ontological = predictions["ontological"][0].cpu().numpy()
        confidence = predictions["confidence"][0].item()

        # Create score dictionaries
        temporal_scores = dict(zip(self.MODE_NAMES["temporal"], temporal))
        spatial_scores = dict(zip(self.MODE_NAMES["spatial"], spatial))
        ontological_scores = dict(zip(self.MODE_NAMES["ontological"], ontological))

        # Predict album
        predicted_mode, predicted_album = self._predict_album(
            temporal, spatial, ontological
        )

        # Detect hybrid states
        hybrid_flags = self._detect_hybrid_states(
            temporal, spatial, ontological, confidence
        )

        # Make validation decision
        validation_status, rejection_reason = self._make_decision(
            temporal, spatial, ontological, confidence, hybrid_flags
        )

        # Compute transmigration distances
        transmigration_distances = self._compute_transmigration_distances(
            temporal, spatial, ontological
        )

        # Generate suggestions
        suggestions = self._generate_suggestions(
            validation_status,
            temporal_scores,
            spatial_scores,
            ontological_scores,
            confidence,
            hybrid_flags,
            predicted_album,
        )

        return ValidationResult(
            concept_text=concept_text,
            temporal_scores=temporal_scores,
            spatial_scores=spatial_scores,
            ontological_scores=ontological_scores,
            chromatic_confidence=confidence,
            predicted_album=predicted_album,
            predicted_mode=predicted_mode,
            validation_status=validation_status,
            hybrid_flags=hybrid_flags,
            transmigration_distances=transmigration_distances,
            suggestions=suggestions,
            rejection_reason=rejection_reason,
        )

    def _predict_album(self, temporal, spatial, ontological):
        """Predict album from scores"""
        temporal_mode = self.MODE_NAMES["temporal"][temporal.argmax()]
        spatial_mode = self.MODE_NAMES["spatial"][spatial.argmax()]
        ontological_mode = self.MODE_NAMES["ontological"][ontological.argmax()]

        mode_str = f"{temporal_mode}_{spatial_mode}_{ontological_mode}"
        key = (temporal_mode, spatial_mode, ontological_mode)
        album = self.ALBUM_MAP.get(key, "Unknown")

        return mode_str, album

    def _detect_hybrid_states(self, temporal, spatial, ontological, confidence):
        """Detect hybrid/diffuse states"""
        import numpy as np

        flags = []

        for name, scores in [
            ("temporal", temporal),
            ("spatial", spatial),
            ("ontological", ontological),
        ]:
            sorted_scores = np.sort(scores)[::-1]
            top1, top2 = sorted_scores[0], sorted_scores[1]

            if top1 > self.dominant_threshold:
                mode_name = self.MODE_NAMES[name][scores.argmax()]
                flags.append(f"{name}_dominant_{mode_name.lower()}")
            elif abs(top1 - top2) < self.hybrid_threshold:
                flags.append(f"{name}_hybrid")
            elif (scores.max() - scores.min()) < self.diffuse_threshold:
                flags.append(f"{name}_diffuse")

        diffuse_count = sum(1 for f in flags if "diffuse" in f)
        if diffuse_count == 3:
            flags.append("black_album_candidate")

        return flags

    def _make_decision(self, temporal, spatial, ontological, confidence, hybrid_flags):
        """Make accept/reject decision"""

        if "black_album_candidate" in hybrid_flags:
            return ValidationStatus.ACCEPT_BLACK, None

        diffuse_count = sum(1 for f in hybrid_flags if "diffuse" in f)
        if diffuse_count >= 2:
            return ValidationStatus.REJECT, "diffuse_ontology"

        if confidence < 0.4:
            return ValidationStatus.REJECT, "low_confidence"

        hybrid_count = sum(1 for f in hybrid_flags if "hybrid" in f)
        if hybrid_count > 0 and hybrid_count <= 2 and confidence > 0.5:
            return ValidationStatus.ACCEPT_HYBRID, None

        if confidence > self.confidence_threshold:
            dominant_count = sum(1 for f in hybrid_flags if "dominant" in f)
            if dominant_count >= 2:
                return ValidationStatus.ACCEPT, None

        return ValidationStatus.ACCEPT_HYBRID, None

    def _compute_transmigration_distances(self, temporal, spatial, ontological):
        """Compute distances to each album"""
        import numpy as np

        distances = {}

        current_state = np.concatenate([temporal, spatial, ontological])

        album_targets = {
            "Orange": np.array([1, 0, 0, 1, 0, 0, 1, 0, 0]),
            "Red": np.array([1, 0, 0, 1, 0, 0, 0, 0, 1]),
            "Yellow": np.array([0, 0, 1, 0, 1, 0, 1, 0, 0]),
            "Green": np.array([0, 0, 1, 0, 1, 0, 0, 1, 0]),
            "Blue": np.array([0, 1, 0, 0, 0, 1, 0, 1, 0]),
            "Black": np.array(
                [1 / 3, 1 / 3, 1 / 3, 1 / 3, 1 / 3, 1 / 3, 1 / 3, 1 / 3, 1 / 3]
            ),
        }

        for album, target in album_targets.items():
            distance = np.linalg.norm(current_state - target)
            distances[album] = float(distance)

        return distances

    def _generate_suggestions(
        self,
        status,
        temporal_scores,
        spatial_scores,
        ontological_scores,
        confidence,
        hybrid_flags,
        predicted_album,
    ):
        """Generate actionable suggestions"""
        suggestions = []

        if status == ValidationStatus.ACCEPT:
            suggestions.append(f"Concept strongly fits {predicted_album} Album")
        elif status == ValidationStatus.ACCEPT_HYBRID:
            suggestions.append(f"Hybrid concept - suggest {predicted_album} Album")
        elif status == ValidationStatus.ACCEPT_BLACK:
            suggestions.append("Concept fits Black Album (None_None_None mode)")
        elif status == ValidationStatus.REJECT:
            all_scores = {
                "temporal": temporal_scores,
                "spatial": spatial_scores,
                "ontological": ontological_scores,
            }
            for dim_name, scores in all_scores.items():
                max_score = max(scores.values())
                if max_score < 0.5:
                    top_mode = max(scores, key=scores.get)
                    suggestions.append(
                        f"Strengthen {dim_name}_{top_mode} "
                        f"(currently {max_score:.2f}, target >0.6)"
                    )

        return suggestions


# ============================================================================
# CONCEPT LOADER (Read concepts from files)
# ============================================================================


def load_concept_from_file(filepath: Path) -> Optional[str]:
    """Load concept text from YML or JSON file"""
    try:
        with open(filepath) as f:
            if filepath.suffix == ".yml" or filepath.suffix == ".yaml":
                data = yaml.safe_load(f)
            else:
                data = json.load(f)

        # Try different possible field names
        for field in ["concept", "content", "text", "white_concept"]:
            if field in data:
                return data[field]

        # If data is just a string
        if isinstance(data, str):
            return data

        print(f"⚠️  Could not find concept text in {filepath}")
        return None

    except Exception as e:
        print(f"❌ Error loading {filepath}: {e}")
        return None


def find_concepts_in_directory(
    directory: Path, pattern: str = "**/*.yml"
) -> List[Path]:
    """Find concept files in directory"""
    return list(directory.glob(pattern))


# ============================================================================
# BATCH VALIDATION
# ============================================================================


def validate_batch(
    validator: ConceptValidator,
    concept_files: List[Path],
    output_dir: Optional[Path] = None,
) -> List[ValidationResult]:
    """Validate multiple concepts and optionally save results"""

    results = []

    print(f"\n🔍 Validating {len(concept_files)} concepts...\n")

    for i, filepath in enumerate(concept_files, 1):
        concept_text = load_concept_from_file(filepath)
        if concept_text is None:
            continue

        print(f"[{i}/{len(concept_files)}] {filepath.name}")

        result = validator.validate_concept(concept_text)
        results.append(result)

        # Print summary
        status_emoji = {
            ValidationStatus.ACCEPT: "✅",
            ValidationStatus.ACCEPT_HYBRID: "⚠️ ",
            ValidationStatus.ACCEPT_BLACK: "🖤",
            ValidationStatus.REJECT: "❌",
        }

        print(
            f"  {status_emoji[result.validation_status]} {result.validation_status.value}"
        )
        print(f"  Album: {result.predicted_album}")
        print(f"  Confidence: {result.chromatic_confidence:.2f}")
        if result.hybrid_flags:
            print(f"  Flags: {', '.join(result.hybrid_flags[:3])}")
        print()

    # Save results if output directory specified
    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
        output_file = output_dir / "validation_results.json"

        with open(output_file, "w") as f:
            json.dump([asdict(r) for r in results], f, indent=2, default=str)

        print(f"💾 Saved results to {output_file}")

    # Print summary statistics
    print("\n" + "=" * 60)
    print("VALIDATION SUMMARY")
    print("=" * 60)

    status_counts = {}
    for result in results:
        status = result.validation_status.value
        status_counts[status] = status_counts.get(status, 0) + 1

    for status, count in status_counts.items():
        pct = (count / len(results)) * 100
        print(f"  {status}: {count} ({pct:.1f}%)")

    print(f"\nTotal: {len(results)} concepts validated")
    print("=" * 60 + "\n")

    return results


# ============================================================================
# MAIN CLI
# ============================================================================


def main():
    parser = argparse.ArgumentParser(
        description="Validate White Agent concepts using Phase 4 model"
    )

    # Input sources (mutually exclusive)
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "--concept-file", type=Path, help="Single concept file to validate"
    )
    input_group.add_argument(
        "--concepts-dir", type=Path, help="Directory containing concept files"
    )
    input_group.add_argument(
        "--thread-id", help="Validate concepts from specific thread ID"
    )
    input_group.add_argument(
        "--recent", type=int, help="Validate N most recent concepts"
    )

    # Model configuration
    parser.add_argument(
        "--model",
        type=Path,
        default=Path("output/phase4_best.pt"),
        help="Path to trained model (default: output/phase4_best.pt)",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        choices=["cpu", "cuda", "mps"],
        help="Device to run on",
    )

    # Output
    parser.add_argument(
        "--output-dir", type=Path, help="Save validation results to directory"
    )

    # Thresholds
    parser.add_argument("--confidence-threshold", type=float, default=0.7)
    parser.add_argument("--dominant-threshold", type=float, default=0.6)
    parser.add_argument("--hybrid-threshold", type=float, default=0.15)
    parser.add_argument("--diffuse-threshold", type=float, default=0.2)

    args = parser.parse_args()

    # Initialize validator
    print("\n🔧 Initializing validator...")
    validator = ConceptValidator(
        model_path=str(args.model),
        device=args.device,
        confidence_threshold=args.confidence_threshold,
        dominant_threshold=args.dominant_threshold,
        hybrid_threshold=args.hybrid_threshold,
        diffuse_threshold=args.diffuse_threshold,
    )

    # Find concepts to validate
    concept_files = []

    if args.concept_file:
        concept_files = [args.concept_file]

    elif args.concepts_dir:
        concept_files = find_concepts_in_directory(args.concepts_dir)

    elif args.thread_id:
        # Look in chain_artifacts for this thread
        base_path = Path("/chain_artifacts") / args.thread_id
        if base_path.exists():
            concept_files = find_concepts_in_directory(base_path)
        else:
            print(f"❌ Thread directory not found: {base_path}")
            return

    elif args.recent:
        # Find N most recent concept files
        artifacts_path = Path("/chain_artifacts")
        if artifacts_path.exists():
            all_concepts = find_concepts_in_directory(artifacts_path)
            # Sort by modification time
            all_concepts.sort(key=lambda p: p.stat().st_mtime, reverse=True)
            concept_files = all_concepts[: args.recent]
        else:
            print(f"❌ Artifacts directory not found: {artifacts_path}")
            return

    if not concept_files:
        print("❌ No concept files found")
        return

    # Run validation
    results = validate_batch(validator, concept_files, output_dir=args.output_dir)
    print("\nValidation complete.")
    print(results)


if __name__ == "__main__":
    main()
