import os
import re
from dataclasses import dataclass
from typing import Any, Dict, Optional

from dotenv import load_dotenv

from app.util.manifest_loader import load_manifest


@dataclass
class RebrackettingAnalysis:
    """Structured analysis of rebracketing methodology from concept field"""

    original_memory: Optional[str] = None
    corrected_memory: Optional[str] = None
    rebracketing_type: Optional[str] = None
    temporal_context: Optional[str] = None
    creative_transformation: Optional[str] = None
    memory_discrepancy_severity: float = 0.0  # 0-1 scale
    ontological_category: Optional[str] = None  # PAST + THING + IMAGINED analysis


class ConceptExtractor:
    """Enhanced extractor for concept field rebracketing analysis"""

    def __init__(self, manifest_id: str):
        load_dotenv()
        self.manifest_id = manifest_id
        self.manifest_path = os.path.join(
            os.environ["MANIFEST_PATH"], manifest_id, f"{manifest_id}.yml"
        )

        if not os.path.exists(self.manifest_path):
            raise ValueError(f"Manifest file not found: {self.manifest_path}")

        # Load the manifest
        self.manifest = load_manifest(self.manifest_path)
        if self.manifest is None:
            raise ValueError("Manifest could not be loaded.")

        self.concept_analysis = None
        if self.manifest and hasattr(self.manifest, "concept"):
            self.concept_analysis = self.analyze_concept_field(self.manifest.concept)

    def analyze_concept_field(self, concept_text: str) -> RebrackettingAnalysis:
        """Parse concept field for rebracketing methodology patterns"""
        if not concept_text:
            return RebrackettingAnalysis()

        analysis = RebrackettingAnalysis()

        # Extract memory discrepancy patterns
        memory_patterns = [
            r"misremembered as (.+?) rather than (.+?)[.!]",
            r"incorrectly remembered as (.+?) instead of (.+?)[.!]",
            r"was rebracketed from \"(.+?)\" to \"(.+?)\"",
            r"thing (?:that was )?misremember(?:ed)? was (.+?)[.!]",
            r"The (.+?) was misremembered as (.+?)[.!]",
        ]

        for pattern in memory_patterns:
            match = re.search(pattern, concept_text, re.IGNORECASE)
            if match:
                if len(match.groups()) == 2:
                    analysis.original_memory = match.group(1).strip()
                    analysis.corrected_memory = match.group(2).strip()
                elif len(match.groups()) == 1:
                    analysis.original_memory = match.group(1).strip()
                break

        # Classify rebracketing type based on concept content
        analysis.rebracketing_type = self._classify_rebracketing_type(concept_text)

        # Extract temporal/spatial context
        analysis.temporal_context = self._extract_temporal_context(concept_text)

        # Analyze creative transformation method
        analysis.creative_transformation = self._analyze_creative_transformation(
            concept_text
        )

        # Calculate severity of memory discrepancy
        analysis.memory_discrepancy_severity = self._calculate_discrepancy_severity(
            analysis
        )

        # Determine ontological category (PAST + THING + IMAGINED analysis)
        analysis.ontological_category = self._determine_ontological_category(
            concept_text
        )

        return analysis

    def _classify_rebracketing_type(self, concept_text: str) -> str:
        """Classify the type of rebracketing based on content patterns"""
        concept_lower = concept_text.lower()

        if any(
            word in concept_lower for word in ["color", "blue", "purple", "appearance"]
        ):
            return "perceptual_rebracketing"
        elif any(
            word in concept_lower
            for word in ["location", "above", "below", "near", "place"]
        ):
            return "spatial_rebracketing"
        elif any(
            word in concept_lower
            for word in ["accident", "injury", "happened", "occurred"]
        ):
            return "causal_rebracketing"
        elif any(
            word in concept_lower for word in ["cookie", "wafer", "object", "thing"]
        ):
            return "object_rebracketing"
        elif any(
            word in concept_lower
            for word in ["meditation", "memory", "imagined", "dreamed"]
        ):
            return "experiential_rebracketing"
        else:
            return "temporal_rebracketing"

    def _extract_temporal_context(self, concept_text: str) -> Optional[str]:
        """Extract temporal context clues from concept"""
        temporal_patterns = [
            r"(\d{4})",  # Years
            r"(high school|college|childhood|teen|youth)",
            r"(1\d{2}0s|1\d{3}0s)",  # Decades
            r"(age \d+)",
            r"(when I was \d+)",
            r"(back in|during the|in the) ([\w\s]+)",
        ]

        for pattern in temporal_patterns:
            match = re.search(pattern, concept_text, re.IGNORECASE)
            if match:
                return match.group(0)
        return None

    def _analyze_creative_transformation(self, concept_text: str) -> Optional[str]:
        """Identify how the memory discrepancy becomes creative material"""
        transformation_patterns = [
            r"The song is about (.+?)[.!]",
            r"This (?:song|track) (?:explores|examines|is about) (.+?)[.!]",
            r"(?:represents|embodies|captures) (.+?)[.!]",
            r"The (?:concept|idea|theme) of (.+?)[.!]",
        ]

        for pattern in transformation_patterns:
            match = re.search(pattern, concept_text, re.IGNORECASE)
            if match:
                return match.group(1).strip()
        return None

    def _calculate_discrepancy_severity(self, analysis: RebrackettingAnalysis) -> float:
        """Calculate how significant the memory discrepancy is (0-1 scale)"""
        if not analysis.original_memory or not analysis.corrected_memory:
            return 0.0

        # Simple heuristic based on semantic distance
        original_words = set(analysis.original_memory.lower().split())
        corrected_words = set(analysis.corrected_memory.lower().split())

        if not original_words or not corrected_words:
            return 0.5

        overlap = len(original_words.intersection(corrected_words))
        total_unique = len(original_words.union(corrected_words))

        # Higher discrepancy = less overlap
        return 1.0 - (overlap / total_unique) if total_unique > 0 else 0.5

    def _determine_ontological_category(self, concept_text: str) -> str:
        """Determine if this fits PAST + THING + IMAGINED pattern"""
        concept_lower = concept_text.lower()

        # Look for temporal markers (PAST)
        has_past = any(
            marker in concept_lower
            for marker in [
                "memory",
                "remembered",
                "recall",
                "high school",
                "childhood",
                "teen",
                "was",
                "used to",
                "back",
                "ago",
                "when i",
                "years",
            ]
        )

        # Look for concrete object markers (THING)
        has_thing = any(
            marker in concept_lower
            for marker in [
                "tooth",
                "shirt",
                "sign",
                "cookie",
                "wafer",
                "bottle",
                "accident",
                "building",
                "place",
                "space",
                "location",
                "object",
            ]
        )

        # Look for imaginative transformation markers (IMAGINED)
        has_imagined = any(
            marker in concept_lower
            for marker in [
                "misremember",
                "incorrect",
                "rebracketed",
                "imagined",
                "thought",
                "felt like",
                "seemed",
                "appeared",
                "meditation",
                "dream",
            ]
        )

        components = []
        if has_past:
            components.append("PAST")
        if has_thing:
            components.append("THING")
        if has_imagined:
            components.append("IMAGINED")

        return " + ".join(components) if components else "UNKNOWN"

    def generate_rebracketing_training_features(self) -> Dict[str, Any]:
        """Generate training features specific to rebracketing methodology"""
        if not self.concept_analysis:
            return {}

        return {
            "has_memory_discrepancy": bool(self.concept_analysis.original_memory),
            "rebracketing_type": self.concept_analysis.rebracketing_type,
            "discrepancy_severity": self.concept_analysis.memory_discrepancy_severity,
            "ontological_pattern": self.concept_analysis.ontological_category,
            "temporal_context": self.concept_analysis.temporal_context,
            "creative_transformation_method": self.concept_analysis.creative_transformation,
            # Additional computed features
            "follows_orange_pattern": self.concept_analysis.ontological_category
            == "PAST + THING + IMAGINED",
            "memory_boundary_crossing": self.concept_analysis.memory_discrepancy_severity
            > 0.5,
            "temporal_rebracketing_complexity": self._calculate_temporal_complexity(),
        }

    def _calculate_temporal_complexity(self) -> float:
        """Calculate complexity of temporal boundary crossing"""
        if not self.concept_analysis:
            return 0.0

        complexity = 0.0

        # Base complexity from discrepancy severity
        complexity += self.concept_analysis.memory_discrepancy_severity * 0.4

        # Additional complexity from multiple temporal references
        if self.concept_analysis.temporal_context:
            complexity += 0.3

        # Complexity from creative transformation sophistication
        if self.concept_analysis.creative_transformation:
            transformation_indicators = ["surreal", "multiple", "complex", "layers"]
            text = self.concept_analysis.creative_transformation.lower()
            complexity += sum(
                0.1 for indicator in transformation_indicators if indicator in text
            )

        return min(complexity, 1.0)

    def export_rebracketing_dataset(self) -> Dict[str, Any]:
        """Export complete rebracketing analysis for training dataset"""
        base_data = {
            "manifest_id": self.manifest_id,
            "title": self.manifest.title if self.manifest else None,
            "rainbow_color": self.manifest.rainbow_color if self.manifest else None,
            "raw_concept": self.manifest.concept if self.manifest else None,
        }

        if self.concept_analysis:
            base_data.update(
                {
                    "original_memory": self.concept_analysis.original_memory,
                    "corrected_memory": self.concept_analysis.corrected_memory,
                    "rebracketing_analysis": self.concept_analysis.__dict__,
                    "training_features": self.generate_rebracketing_training_features(),
                }
            )

        return base_data


if __name__ == "__main__":
    # Test with one of your Orange album concept fields
    test_concept = """A nostalgic reflection of my high school band days is re-bracketed in a surreal manner with abrupt genre shifts and eclectic instrumentation. The shoegaze guitars are anachronistic - filtered by LFO, Let It Bleed style boys choir buts up against 70s garage punk and 50s rock and roll. The song is a chaotic yet harmonious blend of these disparate elements, creating a unique soundscape that is both intense and eclectic. The past thing imagined is the broken front tooth which was incorrectly remembered as happening in an accident rather than on a beer bottle."""

    # You'd need to set up proper manifest loading, but this shows the concept analysis
    extractor = ConceptExtractor(
        manifest_id="03_02"
    )  # This would load your actual manifest
    analysis = extractor.analyze_concept_field(test_concept)

    print("=== REBRACKETING ANALYSIS ===")
    print(f"Original Memory: {analysis.original_memory}")
    print(f"Corrected Memory: {analysis.corrected_memory}")
    print(f"Rebracketing Type: {analysis.rebracketing_type}")
    print(f"Ontological Category: {analysis.ontological_category}")
    print(f"Discrepancy Severity: {analysis.memory_discrepancy_severity:.3f}")
    print(f"Creative Transformation: {analysis.creative_transformation}")
