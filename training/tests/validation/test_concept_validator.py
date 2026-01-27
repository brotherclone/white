"""
Unit tests for ConceptValidator and related classes.

Tests validation logic, caching, and result generation.
"""

import pytest
import numpy as np
import time


class TestValidationStatus:
    """Test ValidationStatus enum."""

    def test_status_values(self):
        """Test that all expected status values exist."""
        from validation.concept_validator import ValidationStatus

        assert ValidationStatus.ACCEPT.value == "accept"
        assert ValidationStatus.ACCEPT_HYBRID.value == "accept_hybrid"
        assert ValidationStatus.ACCEPT_BLACK.value == "accept_black"
        assert ValidationStatus.REJECT.value == "reject"


class TestValidationResult:
    """Test ValidationResult dataclass."""

    def test_is_accepted_true_for_accept(self):
        """Test is_accepted returns True for ACCEPT."""
        from validation.concept_validator import ValidationResult, ValidationStatus

        result = ValidationResult(
            concept_text="test",
            temporal_scores={"past": 0.8, "present": 0.1, "future": 0.1},
            spatial_scores={"thing": 0.9, "place": 0.05, "person": 0.05},
            ontological_scores={"imagined": 0.7, "forgotten": 0.2, "known": 0.1},
            chromatic_confidence=0.85,
            predicted_album="Orange",
            predicted_mode="Past_Thing_Imagined",
            validation_status=ValidationStatus.ACCEPT,
        )

        assert result.is_accepted is True

    def test_is_accepted_true_for_hybrid(self):
        """Test is_accepted returns True for ACCEPT_HYBRID."""
        from validation.concept_validator import ValidationResult, ValidationStatus

        result = ValidationResult(
            concept_text="test",
            temporal_scores={"past": 0.5, "present": 0.4, "future": 0.1},
            spatial_scores={"thing": 0.9, "place": 0.05, "person": 0.05},
            ontological_scores={"imagined": 0.7, "forgotten": 0.2, "known": 0.1},
            chromatic_confidence=0.6,
            predicted_album="Orange",
            predicted_mode="Past_Thing_Imagined",
            validation_status=ValidationStatus.ACCEPT_HYBRID,
        )

        assert result.is_accepted is True

    def test_is_accepted_true_for_black(self):
        """Test is_accepted returns True for ACCEPT_BLACK."""
        from validation.concept_validator import ValidationResult, ValidationStatus

        result = ValidationResult(
            concept_text="test",
            temporal_scores={"past": 0.33, "present": 0.34, "future": 0.33},
            spatial_scores={"thing": 0.33, "place": 0.34, "person": 0.33},
            ontological_scores={"imagined": 0.33, "forgotten": 0.34, "known": 0.33},
            chromatic_confidence=0.1,
            predicted_album="Black",
            predicted_mode="Present_Place_Forgotten",
            validation_status=ValidationStatus.ACCEPT_BLACK,
        )

        assert result.is_accepted is True

    def test_is_accepted_false_for_reject(self):
        """Test is_accepted returns False for REJECT."""
        from validation.concept_validator import ValidationResult, ValidationStatus

        result = ValidationResult(
            concept_text="test",
            temporal_scores={"past": 0.4, "present": 0.3, "future": 0.3},
            spatial_scores={"thing": 0.4, "place": 0.3, "person": 0.3},
            ontological_scores={"imagined": 0.4, "forgotten": 0.3, "known": 0.3},
            chromatic_confidence=0.3,
            predicted_album="Orange",
            predicted_mode="Past_Thing_Imagined",
            validation_status=ValidationStatus.REJECT,
            rejection_reason="diffuse_ontology",
        )

        assert result.is_accepted is False

    def test_to_dict(self):
        """Test serialization to dictionary."""
        from validation.concept_validator import (
            ValidationResult,
            ValidationStatus,
            ValidationSuggestion,
        )

        result = ValidationResult(
            concept_text="test concept",
            temporal_scores={"past": 0.8, "present": 0.1, "future": 0.1},
            spatial_scores={"thing": 0.9, "place": 0.05, "person": 0.05},
            ontological_scores={"imagined": 0.7, "forgotten": 0.2, "known": 0.1},
            chromatic_confidence=0.85,
            predicted_album="Orange",
            predicted_mode="Past_Thing_Imagined",
            validation_status=ValidationStatus.ACCEPT,
            hybrid_flags=["hybrid_past_present"],
            suggestions=[
                ValidationSuggestion(
                    dimension="confidence",
                    current_value=0.5,
                    target_value=0.7,
                    message="Increase clarity",
                )
            ],
            model_version="v1.0",
            validation_time_ms=10.5,
        )

        d = result.to_dict()

        assert d["concept_text"] == "test concept"
        assert d["predicted_album"] == "Orange"
        assert d["validation_status"] == "accept"
        assert d["chromatic_confidence"] == 0.85
        assert len(d["suggestions"]) == 1
        assert d["suggestions"][0]["dimension"] == "confidence"


class TestValidationCache:
    """Test ValidationCache class."""

    def test_cache_get_miss(self):
        """Test cache miss returns None."""
        from validation.concept_validator import ValidationCache

        cache = ValidationCache(ttl_seconds=60)
        result = cache.get("test text", "v1.0")
        assert result is None

    def test_cache_set_and_get(self):
        """Test cache set and get."""
        from validation.concept_validator import (
            ValidationCache,
            ValidationResult,
            ValidationStatus,
        )

        cache = ValidationCache(ttl_seconds=60)

        result = ValidationResult(
            concept_text="test",
            temporal_scores={"past": 0.8, "present": 0.1, "future": 0.1},
            spatial_scores={"thing": 0.9, "place": 0.05, "person": 0.05},
            ontological_scores={"imagined": 0.7, "forgotten": 0.2, "known": 0.1},
            chromatic_confidence=0.85,
            predicted_album="Orange",
            predicted_mode="Past_Thing_Imagined",
            validation_status=ValidationStatus.ACCEPT,
        )

        cache.set("test text", "v1.0", result)
        cached = cache.get("test text", "v1.0")

        assert cached is not None
        assert cached.cache_hit is True
        assert cached.predicted_album == "Orange"

    def test_cache_expiration(self):
        """Test cache entry expiration."""
        from validation.concept_validator import (
            ValidationCache,
            ValidationResult,
            ValidationStatus,
        )

        cache = ValidationCache(ttl_seconds=1)  # 1 second TTL

        result = ValidationResult(
            concept_text="test",
            temporal_scores={"past": 0.8, "present": 0.1, "future": 0.1},
            spatial_scores={"thing": 0.9, "place": 0.05, "person": 0.05},
            ontological_scores={"imagined": 0.7, "forgotten": 0.2, "known": 0.1},
            chromatic_confidence=0.85,
            predicted_album="Orange",
            predicted_mode="Past_Thing_Imagined",
            validation_status=ValidationStatus.ACCEPT,
        )

        cache.set("test text", "v1.0", result)

        # Should exist
        assert cache.get("test text", "v1.0") is not None

        # Wait for expiration
        time.sleep(1.5)

        # Should be expired
        assert cache.get("test text", "v1.0") is None

    def test_cache_lru_eviction(self):
        """Test LRU eviction when cache is full."""
        from validation.concept_validator import (
            ValidationCache,
            ValidationResult,
            ValidationStatus,
        )

        cache = ValidationCache(ttl_seconds=60, max_size=3)

        def make_result(text):
            return ValidationResult(
                concept_text=text,
                temporal_scores={"past": 0.8, "present": 0.1, "future": 0.1},
                spatial_scores={"thing": 0.9, "place": 0.05, "person": 0.05},
                ontological_scores={"imagined": 0.7, "forgotten": 0.2, "known": 0.1},
                chromatic_confidence=0.85,
                predicted_album="Orange",
                predicted_mode="Past_Thing_Imagined",
                validation_status=ValidationStatus.ACCEPT,
            )

        # Add 3 items
        cache.set("text1", "v1.0", make_result("text1"))
        cache.set("text2", "v1.0", make_result("text2"))
        cache.set("text3", "v1.0", make_result("text3"))

        # All should exist
        assert cache.get("text1", "v1.0") is not None
        assert cache.get("text2", "v1.0") is not None
        assert cache.get("text3", "v1.0") is not None

        # Add 4th item - should evict oldest
        cache.set("text4", "v1.0", make_result("text4"))

        assert cache.get("text4", "v1.0") is not None
        # One of the previous should be evicted
        cached_count = sum(
            1 for t in ["text1", "text2", "text3"] if cache.get(t, "v1.0") is not None
        )
        assert cached_count <= 2

    def test_cache_clear(self):
        """Test cache clear."""
        from validation.concept_validator import (
            ValidationCache,
            ValidationResult,
            ValidationStatus,
        )

        cache = ValidationCache(ttl_seconds=60)

        result = ValidationResult(
            concept_text="test",
            temporal_scores={"past": 0.8, "present": 0.1, "future": 0.1},
            spatial_scores={"thing": 0.9, "place": 0.05, "person": 0.05},
            ontological_scores={"imagined": 0.7, "forgotten": 0.2, "known": 0.1},
            chromatic_confidence=0.85,
            predicted_album="Orange",
            predicted_mode="Past_Thing_Imagined",
            validation_status=ValidationStatus.ACCEPT,
        )

        cache.set("test", "v1.0", result)
        assert cache.get("test", "v1.0") is not None

        cache.clear()
        assert cache.get("test", "v1.0") is None


class TestConceptValidator:
    """Test ConceptValidator class."""

    def setup_method(self):
        """Setup test validator."""
        from validation.concept_validator import ConceptValidator

        self.validator = ConceptValidator(
            model_path=None,  # Use mock predictions
            enable_cache=False,  # Disable cache for testing
        )

    def test_validate_concept_returns_result(self):
        """Test validation returns a ValidationResult."""
        from validation.concept_validator import ValidationResult

        result = self.validator.validate_concept("A memory from childhood")

        assert isinstance(result, ValidationResult)
        assert result.concept_text == "A memory from childhood"
        assert result.predicted_album is not None
        assert result.predicted_mode is not None

    def test_validate_concept_scores_sum_to_one(self):
        """Test that dimension scores sum to approximately 1."""
        result = self.validator.validate_concept("Test concept text")

        temporal_sum = sum(result.temporal_scores.values())
        spatial_sum = sum(result.spatial_scores.values())
        ontological_sum = sum(result.ontological_scores.values())

        assert np.isclose(temporal_sum, 1.0, atol=0.01)
        assert np.isclose(spatial_sum, 1.0, atol=0.01)
        assert np.isclose(ontological_sum, 1.0, atol=0.01)

    def test_validate_concept_confidence_in_range(self):
        """Test confidence is between 0 and 1."""
        result = self.validator.validate_concept("Test concept")

        assert 0 <= result.chromatic_confidence <= 1

    def test_validate_concept_includes_timing(self):
        """Test that validation time is recorded."""
        result = self.validator.validate_concept("Test concept")

        assert result.validation_time_ms is not None
        assert result.validation_time_ms > 0

    def test_validate_batch(self):
        """Test batch validation."""
        concepts = [
            "A remembered toy",
            "A forgotten place",
            "An imagined future",
        ]

        results = self.validator.validate_batch(concepts)

        assert len(results) == 3
        for concept, result in zip(concepts, results):
            assert result.concept_text == concept

    def test_transmigration_distances(self):
        """Test transmigration distances are computed."""
        result = self.validator.validate_concept("Test concept")

        assert result.transmigration_distances is not None
        assert "Orange" in result.transmigration_distances
        assert "Red" in result.transmigration_distances
        assert "Blue" in result.transmigration_distances
        assert "Black" in result.transmigration_distances

        # Distances should be non-negative
        for album, distance in result.transmigration_distances.items():
            assert distance >= 0

    def test_album_prediction(self):
        """Test album prediction is valid."""
        result = self.validator.validate_concept("Test concept")

        valid_albums = [
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
        assert result.predicted_album in valid_albums

    def test_mode_prediction_format(self):
        """Test mode prediction has correct format."""
        result = self.validator.validate_concept("Test concept")

        # Should be Temporal_Spatial_Ontological
        parts = result.predicted_mode.split("_")
        assert len(parts) == 3

        temporal_modes = ["Past", "Present", "Future"]
        spatial_modes = ["Thing", "Place", "Person"]
        ontological_modes = ["Imagined", "Forgotten", "Known"]

        assert parts[0] in temporal_modes
        assert parts[1] in spatial_modes
        assert parts[2] in ontological_modes

    def test_caching_enabled(self):
        """Test caching when enabled."""
        from validation.concept_validator import ConceptValidator

        cached_validator = ConceptValidator(
            model_path=None,
            enable_cache=True,
            cache_ttl=60,
        )

        concept = "A unique test concept for caching"

        # First call - cache miss
        result1 = cached_validator.validate_concept(concept)
        assert result1.cache_hit is False

        # Second call - cache hit
        result2 = cached_validator.validate_concept(concept)
        assert result2.cache_hit is True

    def test_singleton_instance(self):
        """Test singleton pattern."""
        from validation.concept_validator import ConceptValidator

        # Reset singleton
        ConceptValidator._instance = None

        instance1 = ConceptValidator.get_instance(model_path=None)
        instance2 = ConceptValidator.get_instance(model_path=None)

        assert instance1 is instance2

        # Clean up
        ConceptValidator._instance = None


class TestHybridStateDetection:
    """Test hybrid state detection logic."""

    def setup_method(self):
        """Setup test validator."""
        from validation.concept_validator import ConceptValidator

        self.validator = ConceptValidator(
            model_path=None,
            enable_cache=False,
            dominant_threshold=0.6,
            hybrid_threshold=0.15,
            diffuse_threshold=0.2,
        )

    def test_dominant_state_detection(self):
        """Test detection of dominant state."""
        scores = np.array([0.8, 0.15, 0.05])
        mode_names = ["past", "present", "future"]

        state, secondary, flags = self.validator._detect_hybrid_state(
            scores, mode_names
        )

        assert state == "dominant"
        assert secondary is None
        assert len(flags) == 0

    def test_hybrid_state_detection(self):
        """Test detection of hybrid state."""
        scores = np.array([0.45, 0.40, 0.15])  # Top two close
        mode_names = ["past", "present", "future"]

        state, secondary, flags = self.validator._detect_hybrid_state(
            scores, mode_names
        )

        assert state == "hybrid"
        assert secondary == "present"
        assert len(flags) == 1
        assert "hybrid_past_present" in flags[0]

    def test_diffuse_state_detection(self):
        """Test detection of diffuse state."""
        scores = np.array([0.34, 0.33, 0.33])  # Near uniform
        mode_names = ["past", "present", "future"]

        state, secondary, flags = self.validator._detect_hybrid_state(
            scores, mode_names
        )

        assert state == "diffuse"
        assert "diffuse" in flags


class TestValidationStatusDetermination:
    """Test validation status determination logic."""

    def setup_method(self):
        """Setup test validator."""
        from validation.concept_validator import ConceptValidator

        self.validator = ConceptValidator(
            model_path=None,
            enable_cache=False,
            confidence_threshold=0.7,
        )

    def test_accept_high_confidence_dominant(self):
        """Test ACCEPT for high confidence with dominant modes."""
        from validation.concept_validator import ValidationStatus

        status, reason = self.validator._determine_validation_status(
            confidence=0.85,
            temporal_state="dominant",
            spatial_state="dominant",
            ontological_state="dominant",
            hybrid_flags=[],
        )

        assert status == ValidationStatus.ACCEPT
        assert reason is None

    def test_accept_hybrid_for_liminal(self):
        """Test ACCEPT_HYBRID for liminal but coherent."""
        from validation.concept_validator import ValidationStatus

        status, reason = self.validator._determine_validation_status(
            confidence=0.6,
            temporal_state="hybrid",
            spatial_state="dominant",
            ontological_state="dominant",
            hybrid_flags=["hybrid_past_present"],
        )

        assert status == ValidationStatus.ACCEPT_HYBRID
        assert reason is None

    def test_accept_black_for_all_diffuse(self):
        """Test ACCEPT_BLACK for all diffuse with low confidence."""
        from validation.concept_validator import ValidationStatus

        status, reason = self.validator._determine_validation_status(
            confidence=0.2,
            temporal_state="diffuse",
            spatial_state="diffuse",
            ontological_state="diffuse",
            hybrid_flags=["diffuse", "diffuse", "diffuse"],
        )

        assert status == ValidationStatus.ACCEPT_BLACK
        assert reason is None

    def test_reject_too_diffuse(self):
        """Test REJECT for too diffuse."""
        from validation.concept_validator import ValidationStatus

        status, reason = self.validator._determine_validation_status(
            confidence=0.6,
            temporal_state="diffuse",
            spatial_state="diffuse",
            ontological_state="dominant",
            hybrid_flags=[],
        )

        assert status == ValidationStatus.REJECT
        assert reason == "diffuse_ontology"

    def test_reject_low_confidence(self):
        """Test REJECT for low confidence."""
        from validation.concept_validator import ValidationStatus

        status, reason = self.validator._determine_validation_status(
            confidence=0.3,
            temporal_state="dominant",
            spatial_state="dominant",
            ontological_state="dominant",
            hybrid_flags=[],
        )

        assert status == ValidationStatus.REJECT
        assert reason == "low_confidence"


class TestSuggestionGeneration:
    """Test suggestion generation for rejected concepts."""

    def setup_method(self):
        """Setup test validator."""
        from validation.concept_validator import ConceptValidator

        self.validator = ConceptValidator(
            model_path=None,
            enable_cache=False,
            confidence_threshold=0.7,
            dominant_threshold=0.6,
        )

    def test_low_confidence_suggestion(self):
        """Test suggestion for low confidence rejection."""
        from validation.concept_validator import ValidationStatus

        prediction = {
            "temporal_scores": np.array([0.8, 0.1, 0.1]),
            "spatial_scores": np.array([0.8, 0.1, 0.1]),
            "ontological_scores": np.array([0.8, 0.1, 0.1]),
            "chromatic_confidence": 0.3,
        }

        suggestions = self.validator._generate_suggestions(
            prediction,
            ValidationStatus.REJECT,
            "low_confidence",
        )

        assert len(suggestions) >= 1
        confidence_suggestions = [s for s in suggestions if s.dimension == "confidence"]
        assert len(confidence_suggestions) == 1
        assert confidence_suggestions[0].current_value == 0.3

    def test_diffuse_ontology_suggestion(self):
        """Test suggestion for diffuse ontology rejection."""
        from validation.concept_validator import ValidationStatus

        prediction = {
            "temporal_scores": np.array([0.4, 0.3, 0.3]),
            "spatial_scores": np.array([0.4, 0.3, 0.3]),
            "ontological_scores": np.array([0.4, 0.3, 0.3]),
            "chromatic_confidence": 0.6,
        }

        suggestions = self.validator._generate_suggestions(
            prediction,
            ValidationStatus.REJECT,
            "diffuse_ontology",
        )

        assert len(suggestions) >= 1
        # Should suggest increasing dimension scores
        dimension_suggestions = [
            s
            for s in suggestions
            if s.dimension in ["temporal", "spatial", "ontological"]
        ]
        assert len(dimension_suggestions) >= 1

    def test_no_suggestions_for_accept(self):
        """Test no suggestions generated for accepted concepts."""
        from validation.concept_validator import ValidationStatus

        prediction = {
            "temporal_scores": np.array([0.8, 0.1, 0.1]),
            "spatial_scores": np.array([0.8, 0.1, 0.1]),
            "ontological_scores": np.array([0.8, 0.1, 0.1]),
            "chromatic_confidence": 0.9,
        }

        suggestions = self.validator._generate_suggestions(
            prediction,
            ValidationStatus.ACCEPT,
            None,
        )

        assert len(suggestions) == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
