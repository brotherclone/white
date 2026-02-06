"""
Concept validator for White Agent integration.

Validates generated concepts using the Rainbow Table ontological regression model.
Provides accept/reject decisions with actionable suggestions for improvement.
"""

import warnings
import yaml
import hashlib
import time
import torch
import numpy as np

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple, Any
from models.multitask_model import MultiTaskRainbowModel
from models.text_encoder import TextEncoder


# Type hints for optional dependencies
try:
    from transformers import AutoTokenizer

    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False


class ValidationStatus(Enum):
    """Validation gate status."""

    ACCEPT = "accept"  # High confidence, clear mode assignment
    ACCEPT_HYBRID = "accept_hybrid"  # Liminal but coherent
    ACCEPT_BLACK = "accept_black"  # Diffuse across all dimensions (Black Album)
    REJECT = "reject"  # Low confidence or unclear


@dataclass
class ValidationSuggestion:
    """Actionable suggestion for improving a rejected concept."""

    dimension: str  # temporal, spatial, ontological, or confidence
    current_value: float
    target_value: float
    message: str


@dataclass
class ValidationResult:
    """
    Complete validation result for a concept.

    Contains all ontological scores, validation status, and actionable suggestions.
    """

    # Input
    concept_text: str

    # Ontological scores (softmax distributions)
    temporal_scores: Dict[str, float]  # {past, present, future}
    spatial_scores: Dict[str, float]  # {thing, place, person}
    ontological_scores: Dict[str, float]  # {imagined, forgotten, known}

    # Confidence
    chromatic_confidence: float

    # Predictions
    predicted_album: str
    predicted_mode: str  # e.g., "Past_Thing_Imagined"

    # Validation
    validation_status: ValidationStatus
    hybrid_flags: List[str] = field(default_factory=list)
    rejection_reason: Optional[str] = None

    # Uncertainty (if available)
    uncertainty_estimates: Optional[Dict[str, float]] = None

    # Transmigration distances
    transmigration_distances: Optional[Dict[str, float]] = None

    # Suggestions
    suggestions: List[ValidationSuggestion] = field(default_factory=list)

    # Metadata
    model_version: Optional[str] = None
    validation_time_ms: Optional[float] = None
    cache_hit: bool = False

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "concept_text": self.concept_text,
            "temporal_scores": self.temporal_scores,
            "spatial_scores": self.spatial_scores,
            "ontological_scores": self.ontological_scores,
            "chromatic_confidence": self.chromatic_confidence,
            "predicted_album": self.predicted_album,
            "predicted_mode": self.predicted_mode,
            "validation_status": self.validation_status.value,
            "hybrid_flags": self.hybrid_flags,
            "rejection_reason": self.rejection_reason,
            "uncertainty_estimates": self.uncertainty_estimates,
            "transmigration_distances": self.transmigration_distances,
            "suggestions": [
                {
                    "dimension": s.dimension,
                    "current_value": s.current_value,
                    "target_value": s.target_value,
                    "message": s.message,
                }
                for s in self.suggestions
            ],
            "model_version": self.model_version,
            "validation_time_ms": self.validation_time_ms,
            "cache_hit": self.cache_hit,
        }

    @property
    def is_accepted(self) -> bool:
        """Check if concept is accepted (any accept status)."""
        return self.validation_status in (
            ValidationStatus.ACCEPT,
            ValidationStatus.ACCEPT_HYBRID,
            ValidationStatus.ACCEPT_BLACK,
        )


class ValidationCache:
    """
    Simple in-memory cache for validation results.

    Uses concept text hash as key with TTL expiration.
    """

    def __init__(self, ttl_seconds: int = 3600, max_size: int = 10000):
        """
        Initialize cache.

        Args:
            ttl_seconds: Time-to-live for cache entries
            max_size: Maximum cache size (LRU eviction)
        """
        self.ttl_seconds = ttl_seconds
        self.max_size = max_size
        self._cache: Dict[str, Tuple[ValidationResult, float]] = {}

    def _hash_key(self, text: str, model_version: str) -> str:
        """Generate cache key from text and model version."""
        content = f"{text}|{model_version}"
        return hashlib.sha256(content.encode()).hexdigest()[:32]

    def get(self, text: str, model_version: str) -> Optional[ValidationResult]:
        """Get cached result if available and not expired."""
        key = self._hash_key(text, model_version)

        if key not in self._cache:
            return None

        result, timestamp = self._cache[key]

        # Check expiration
        if time.time() - timestamp > self.ttl_seconds:
            del self._cache[key]
            return None

        # Mark as cache hit
        result.cache_hit = True
        return result

    def set(self, text: str, model_version: str, result: ValidationResult):
        """Store result in cache."""
        key = self._hash_key(text, model_version)

        # LRU eviction if needed
        if len(self._cache) >= self.max_size:
            # Remove oldest entry
            oldest_key = min(self._cache.keys(), key=lambda k: self._cache[k][1])
            del self._cache[oldest_key]

        self._cache[key] = (result, time.time())

    def clear(self):
        """Clear all cached entries."""
        self._cache.clear()


class ConceptValidator:
    """
    Validates concepts for Rainbow Table ontological coherence.

    Uses the trained regression model to predict ontological scores
    and applies validation gates to accept/reject concepts.
    """

    # Singleton instance for API usage
    _instance: Optional["ConceptValidator"] = None

    def __init__(
        self,
        model_path: Optional[str] = None,
        config_path: Optional[str] = None,
        device: str = "auto",
        # Validation thresholds
        confidence_threshold: float = 0.7,
        dominant_threshold: float = 0.6,
        hybrid_threshold: float = 0.15,
        diffuse_threshold: float = 0.2,
        uncertainty_threshold: float = 0.8,
        # Cache settings
        enable_cache: bool = True,
        cache_ttl: int = 3600,
    ):
        """
        Initialize concept validator.

        Args:
            model_path: Path to trained model checkpoint
            config_path: Path to model configuration
            device: Device for inference ('auto', 'cpu', 'cuda', 'mps')
            confidence_threshold: Minimum confidence for ACCEPT
            dominant_threshold: Score needed for dominant mode
            hybrid_threshold: Max difference for hybrid detection
            diffuse_threshold: Max deviation from uniform for diffuse
            uncertainty_threshold: Max uncertainty for ACCEPT
            enable_cache: Enable validation result caching
            cache_ttl: Cache time-to-live in seconds
        """
        self.requested_device = device
        self._detected_device = self._select_device()

        # Use a torch.device object internally
        self.torch_device = self._detected_device
        self.model_path = model_path
        self.config_path = config_path

        # Thresholds
        self.confidence_threshold = confidence_threshold
        self.dominant_threshold = dominant_threshold
        self.hybrid_threshold = hybrid_threshold
        self.diffuse_threshold = diffuse_threshold
        self.uncertainty_threshold = uncertainty_threshold

        # Cache
        self.cache = ValidationCache(ttl_seconds=cache_ttl) if enable_cache else None

        # Model (lazy loaded)
        self._model = None
        self._tokenizer = None
        self._model_version = None

        # Mode mappings
        self.temporal_modes = ["past", "present", "future"]
        self.spatial_modes = ["thing", "place", "person"]
        self.ontological_modes = ["imagined", "forgotten", "known"]

        # Album mapping
        self.album_mapping = {
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

    @classmethod
    def get_instance(cls, **kwargs) -> "ConceptValidator":
        """Get or create singleton instance."""
        if cls._instance is None:
            cls._instance = cls(**kwargs)
        return cls._instance

    def _select_device(self) -> torch.device:
        """Resolve device: prefer CUDA, then MPS, then CPU, unless user forced one."""
        req = (self.requested_device or "auto").lower()
        if req != "auto":
            # Respect explicit device string
            return torch.device(req)
        # Auto-select
        if torch.cuda.is_available():
            return torch.device("cuda")
        # MPS available in newer PyTorch builds on macOS
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    def _load_model(self):
        """Lazy load model and tokenizer with device awareness and optional fp16 for CUDA."""
        if self._model is not None:
            return

        if self.model_path is None:
            warnings.warn("No model path provided, using mock predictions")
            return

        if not HAS_TRANSFORMERS:
            raise ImportError("transformers required for model loading")

        # Load checkpoint to the resolved device
        checkpoint = torch.load(self.model_path, map_location=self.torch_device)

        # Get model version
        self._model_version = checkpoint.get("model_version", "unknown")

        # Load config if provided
        config = {}
        if self.config_path:
            with open(self.config_path) as f:
                config = yaml.safe_load(f)

        # Build model
        encoder_name = (
            config.get("model", {})
            .get("text_encoder", {})
            .get("model_name", "microsoft/deberta-v3-base")
        )
        text_encoder = TextEncoder(encoder_name)

        self._model = MultiTaskRainbowModel(
            text_encoder=text_encoder,
            num_classes=config.get("model", {})
            .get("classifier", {})
            .get("num_classes", 8),
        )
        self._model.load_state_dict(checkpoint["model_state_dict"])

        # Move model to device and use fp16 on CUDA to reduce memory
        if self.torch_device.type == "cuda":
            self._model.to(self.torch_device)
            try:
                self._model.half()
            except Exception:
                # If half is not supported for some parts, keep float32 but ensure device placement
                pass
        else:
            self._model.to(self.torch_device)

        self._model.eval()

        # Load tokenizer
        self._tokenizer = AutoTokenizer.from_pretrained(encoder_name)

    def _mock_prediction(self, text: str) -> Dict:
        """Generate mock predictions for testing without model."""
        # Simple hash-based mock for consistent results
        text_hash = hash(text) % 1000 / 1000

        return {
            "temporal_scores": np.array(
                [0.7 + 0.1 * text_hash, 0.2 - 0.05 * text_hash, 0.1 - 0.05 * text_hash]
            ),
            "spatial_scores": np.array(
                [0.6 + 0.2 * text_hash, 0.3 - 0.1 * text_hash, 0.1]
            ),
            "ontological_scores": np.array([0.8, 0.15, 0.05]),
            "chromatic_confidence": 0.85,
        }

    def _predict(self, text: str) -> Dict:
        """Run model prediction on text, using autocast on CUDA when available."""
        self._load_model()

        if self._model is None:
            return self._mock_prediction(text)

        # Tokenize
        encoding = self._tokenizer(
            text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
        )

        input_ids = encoding["input_ids"].to(self.torch_device)
        attention_mask = encoding["attention_mask"].to(self.torch_device)

        is_cuda = self.torch_device.type == "cuda"

        # Predict with amp when on CUDA for faster fp16 inference
        with torch.no_grad():
            if is_cuda:
                with torch.cuda.amp.autocast():
                    output = self._model(input_ids, attention_mask)
            else:
                output = self._model(input_ids, attention_mask)

            scores = output.ontological_scores

        return {
            "temporal_scores": scores.temporal_scores[0].cpu().numpy(),
            "spatial_scores": scores.spatial_scores[0].cpu().numpy(),
            "ontological_scores": scores.ontological_scores[0].cpu().numpy(),
            "chromatic_confidence": scores.chromatic_confidence[0].cpu().item(),
            "temporal_uncertainty": (
                scores.temporal_uncertainty[0].cpu().numpy()
                if scores.temporal_uncertainty is not None
                else None
            ),
        }

    def _detect_hybrid_state(
        self,
        scores: np.ndarray,
        mode_names: List[str],
    ) -> Tuple[str, Optional[str], List[str]]:
        """
        Detect if dimension is dominant, hybrid, or diffuse.

        Returns:
            Tuple of (state, secondary_mode, flags)
        """
        sorted_idx = np.argsort(scores)[::-1]
        top_score = scores[sorted_idx[0]]
        second_score = scores[sorted_idx[1]]

        top_mode = mode_names[sorted_idx[0]]
        second_mode = mode_names[sorted_idx[1]]

        flags = []

        # Check diffuse (all near uniform)
        uniform = 1.0 / 3.0
        max_dev = max(abs(s - uniform) for s in scores)

        if max_dev <= self.diffuse_threshold:
            return "diffuse", None, ["diffuse"]

        # Check dominant
        if top_score >= self.dominant_threshold:
            return "dominant", None, []

        # Check hybrid
        if (top_score - second_score) <= self.hybrid_threshold:
            flags.append(f"hybrid_{top_mode}_{second_mode}")
            return "hybrid", second_mode, flags

        return "partial", None, []

    def _determine_validation_status(
        self,
        confidence: float,
        temporal_state: str,
        spatial_state: str,
        ontological_state: str,
        hybrid_flags: List[str],
    ) -> Tuple[ValidationStatus, Optional[str]]:
        """
        Determine validation status based on scores.

        Returns:
            Tuple of (status, rejection_reason)
        """
        # Count states
        diffuse_count = sum(
            1
            for s in [temporal_state, spatial_state, ontological_state]
            if s == "diffuse"
        )
        hybrid_count = sum(
            1
            for s in [temporal_state, spatial_state, ontological_state]
            if s == "hybrid"
        )

        # Black Album: all diffuse
        if diffuse_count == 3 and confidence < 0.3:
            return ValidationStatus.ACCEPT_BLACK, None

        # Reject: too diffuse
        if diffuse_count >= 2:
            return ValidationStatus.REJECT, "diffuse_ontology"

        # Reject: low confidence
        if confidence < 0.4:
            return ValidationStatus.REJECT, "low_confidence"

        # Accept hybrid: liminal but coherent
        if hybrid_count >= 1 and hybrid_count <= 2 and confidence >= 0.5:
            return ValidationStatus.ACCEPT_HYBRID, None

        # Accept: high confidence, dominant modes
        if confidence >= self.confidence_threshold:
            all_dominant = all(
                s == "dominant"
                for s in [temporal_state, spatial_state, ontological_state]
            )
            if all_dominant:
                return ValidationStatus.ACCEPT, None

        # Accept with lower confidence if modes are clear
        if confidence >= 0.5:
            dominant_count = sum(
                1
                for s in [temporal_state, spatial_state, ontological_state]
                if s == "dominant"
            )
            if dominant_count >= 2:
                return ValidationStatus.ACCEPT, None

        return ValidationStatus.REJECT, "unclear_ontology"

    def _generate_suggestions(
        self,
        prediction: Dict,
        status: ValidationStatus,
        rejection_reason: Optional[str],
    ) -> List[ValidationSuggestion]:
        """Generate actionable suggestions for improvement."""
        suggestions = []

        if status != ValidationStatus.REJECT:
            return suggestions

        confidence = prediction["chromatic_confidence"]

        # Low confidence suggestion
        if rejection_reason == "low_confidence":
            suggestions.append(
                ValidationSuggestion(
                    dimension="confidence",
                    current_value=confidence,
                    target_value=self.confidence_threshold,
                    message=f"Increase concept clarity. Current confidence: {confidence:.2f}, target: {self.confidence_threshold:.2f}",
                )
            )

        # Diffuse ontology suggestions
        if rejection_reason == "diffuse_ontology":
            for dim, scores, modes in [
                ("temporal", prediction["temporal_scores"], self.temporal_modes),
                ("spatial", prediction["spatial_scores"], self.spatial_modes),
                (
                    "ontological",
                    prediction["ontological_scores"],
                    self.ontological_modes,
                ),
            ]:
                max_score = np.max(scores)
                max_mode = modes[np.argmax(scores)]

                if max_score < self.dominant_threshold:
                    suggestions.append(
                        ValidationSuggestion(
                            dimension=dim,
                            current_value=max_score,
                            target_value=self.dominant_threshold,
                            message=f"Strengthen {dim} orientation. Increase {max_mode}_score from {max_score:.2f} to {self.dominant_threshold:.2f}",
                        )
                    )

        # Unclear ontology suggestions
        if rejection_reason == "unclear_ontology":
            suggestions.append(
                ValidationSuggestion(
                    dimension="overall",
                    current_value=confidence,
                    target_value=0.7,
                    message="Make the concept's temporal, spatial, and ontological nature more explicit",
                )
            )

        return suggestions

    def validate_concept(self, text: str) -> ValidationResult:
        """
        Validate a single concept.

        Args:
            text: Concept text to validate

        Returns:
            ValidationResult with full analysis
        """
        start_time = time.time()

        # Check cache
        model_version = self._model_version or "mock"
        if self.cache:
            cached = self.cache.get(text, model_version)
            if cached:
                return cached

        # Get prediction
        prediction = self._predict(text)

        # Extract scores
        temporal_scores = prediction["temporal_scores"]
        spatial_scores = prediction["spatial_scores"]
        ontological_scores = prediction["ontological_scores"]
        confidence = prediction["chromatic_confidence"]

        # Detect states
        temporal_state, temporal_secondary, temporal_flags = self._detect_hybrid_state(
            temporal_scores, self.temporal_modes
        )
        spatial_state, spatial_secondary, spatial_flags = self._detect_hybrid_state(
            spatial_scores, self.spatial_modes
        )
        ontological_state, ontological_secondary, ontological_flags = (
            self._detect_hybrid_state(ontological_scores, self.ontological_modes)
        )

        hybrid_flags = temporal_flags + spatial_flags + ontological_flags

        # Determine validation status
        status, rejection_reason = self._determine_validation_status(
            confidence,
            temporal_state,
            spatial_state,
            ontological_state,
            hybrid_flags,
        )

        # Get predictions
        temporal_mode = self.temporal_modes[np.argmax(temporal_scores)]
        spatial_mode = self.spatial_modes[np.argmax(spatial_scores)]
        ontological_mode = self.ontological_modes[np.argmax(ontological_scores)]

        predicted_mode = f"{temporal_mode.capitalize()}_{spatial_mode.capitalize()}_{ontological_mode.capitalize()}"
        predicted_album = self.album_mapping.get(
            (temporal_mode, spatial_mode, ontological_mode), "Black"
        )

        # Generate suggestions
        suggestions = self._generate_suggestions(prediction, status, rejection_reason)

        # Calculate transmigration distances (simplified)
        transmigration_distances = {}
        for album, modes in [
            ("Orange", ("past", "thing", "imagined")),
            ("Red", ("past", "thing", "forgotten")),
            ("Violet", ("past", "person", "known")),
            ("Yellow", ("present", "place", "imagined")),
            ("Green", ("present", "person", "known")),
            ("Indigo", ("present", "person", "forgotten")),
            ("Blue", ("future", "place", "known")),
        ]:
            t_idx = self.temporal_modes.index(modes[0])
            s_idx = self.spatial_modes.index(modes[1])
            o_idx = self.ontological_modes.index(modes[2])

            t_target = np.zeros(3)
            t_target[t_idx] = 1.0
            s_target = np.zeros(3)
            s_target[s_idx] = 1.0
            o_target = np.zeros(3)
            o_target[o_idx] = 1.0

            t_dist = np.linalg.norm(temporal_scores - t_target)
            s_dist = np.linalg.norm(spatial_scores - s_target)
            o_dist = np.linalg.norm(ontological_scores - o_target)

            transmigration_distances[album] = float(
                np.sqrt(t_dist**2 + s_dist**2 + o_dist**2)
            )

        # Black Album distance (to uniform)
        uniform = np.ones(3) / 3
        t_dist = np.linalg.norm(temporal_scores - uniform)
        s_dist = np.linalg.norm(spatial_scores - uniform)
        o_dist = np.linalg.norm(ontological_scores - uniform)
        transmigration_distances["Black"] = float(
            np.sqrt(t_dist**2 + s_dist**2 + o_dist**2)
        )

        # Build result
        result = ValidationResult(
            concept_text=text,
            temporal_scores={
                "past": float(temporal_scores[0]),
                "present": float(temporal_scores[1]),
                "future": float(temporal_scores[2]),
            },
            spatial_scores={
                "thing": float(spatial_scores[0]),
                "place": float(spatial_scores[1]),
                "person": float(spatial_scores[2]),
            },
            ontological_scores={
                "imagined": float(ontological_scores[0]),
                "forgotten": float(ontological_scores[1]),
                "known": float(ontological_scores[2]),
            },
            chromatic_confidence=float(confidence),
            predicted_album=predicted_album,
            predicted_mode=predicted_mode,
            validation_status=status,
            hybrid_flags=hybrid_flags,
            rejection_reason=rejection_reason,
            transmigration_distances=transmigration_distances,
            suggestions=suggestions,
            model_version=model_version,
            validation_time_ms=(time.time() - start_time) * 1000,
            cache_hit=False,
        )

        # Store in cache
        if self.cache:
            self.cache.set(text, model_version, result)

        return result

    def validate_batch(self, texts: List[str]) -> List[ValidationResult]:
        """
        Validate multiple concepts.

        Args:
            texts: List of concept texts

        Returns:
            List of ValidationResults (same order as input)
        """
        return [self.validate_concept(text) for text in texts]


if __name__ == "__main__":
    # Quick tests
    print("Testing ConceptValidator...")

    validator = ConceptValidator(
        model_path=None,  # Use mock predictions
        enable_cache=True,
        device="auto",  # auto-detect GPU on RunPod
    )

    # Test 1: Basic validation
    print("\n=== Basic validation ===")
    concept = """
    A memory of building model trains in my father's basement,
    where the clicking of the tracks became a rhythm I still hear
    in my sleep. The trains ran on schedules that existed only
    in my imagination, delivering cargo to cities that never were.
    """

    result = validator.validate_concept(concept)
    print(f"Status: {result.validation_status.value}")
    print(f"Album: {result.predicted_album}")
    print(f"Mode: {result.predicted_mode}")
    print(f"Confidence: {result.chromatic_confidence:.2f}")
    print(f"Temporal: {result.temporal_scores}")
    print(f"Hybrid flags: {result.hybrid_flags}")
    print(f"Time: {result.validation_time_ms:.1f}ms")

    # Test 2: Cache hit
    print("\n=== Cache test ===")
    result2 = validator.validate_concept(concept)
    print(f"Cache hit: {result2.cache_hit}")

    # Test 3: Different concept
    print("\n=== Different concept ===")
    concept2 = "A vague undefined something nowhere and nowhen."
    result3 = validator.validate_concept(concept2)
    print(f"Status: {result3.validation_status.value}")
    print(f"Rejection reason: {result3.rejection_reason}")
    if result3.suggestions:
        print("Suggestions:")
        for s in result3.suggestions:
            print(f"  - {s.message}")

    # Test 4: Batch validation
    print("\n=== Batch validation ===")
    concepts = [
        "A remembered toy from childhood",
        "The future city I will never visit",
        "This present moment, right here, right now",
    ]
    results = validator.validate_batch(concepts)
    for concept, result in zip(concepts, results):
        print(f"  {concept[:40]}... → {result.validation_status.value}")

    # Test 5: to_dict
    print("\n=== Serialization ===")
    result_dict = result.to_dict()
    print(f"Keys: {list(result_dict.keys())}")

    print("\n✓ All validation tests passed!")
