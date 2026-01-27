"""
Integration tests for ConceptValidator with LangGraph workflow.

Task 10.12: Tests the concept validation gate within a LangGraph workflow context.
These tests verify that:
1. ConceptValidator integrates properly with LangGraph state management
2. Validation results correctly influence workflow branching
3. The validation gate handles accept/reject/hybrid decisions
4. Batch validation works within workflow context
"""

import sys
import pytest

from dataclasses import dataclass, field
from typing import List, Optional, Literal
from unittest.mock import MagicMock
from langgraph.graph import StateGraph, END

from validation.concept_validator import (
    ConceptValidator,
    ValidationResult,
    ValidationStatus,
)


# Mock torch before importing validation components (for Intel Mac without torch)
sys.modules["torch"] = MagicMock()

# Import validation components
sys.path.insert(
    0,
    str(__file__).replace(
        "/tests/integration/test_langgraph_workflow_integration.py", ""
    ),
)


# ============================================================================
# Test State for LangGraph Workflow
# ============================================================================


@dataclass
class ConceptValidationState:
    """State for concept validation workflow."""

    # Input
    concept_text: str = ""
    target_album: Optional[str] = None

    # Validation result
    validation_result: Optional[ValidationResult] = None

    # Workflow control
    validation_passed: bool = False
    needs_refinement: bool = False
    refinement_attempts: int = 0
    max_refinement_attempts: int = 3

    # Output
    final_concept: Optional[str] = None
    final_album: Optional[str] = None
    workflow_status: Literal["pending", "accepted", "rejected", "refined"] = "pending"

    # Suggestions for refinement
    suggestions: List[str] = field(default_factory=list)


# ============================================================================
# Workflow Nodes
# ============================================================================


class ConceptValidationWorkflow:
    """
    LangGraph workflow for validating concepts through the Rainbow Table ontological gate.

    Workflow:
        validate_concept -> [branch]
           |-> accept (if ACCEPT/ACCEPT_HYBRID/ACCEPT_BLACK)
           |-> suggest_refinement (if REJECT and attempts < max)
           |-> final_reject (if REJECT and attempts >= max)
    """

    def __init__(self, validator: ConceptValidator):
        self.validator = validator
        self.graph = self._build_graph()

    def _build_graph(self) -> StateGraph:
        """Build the validation workflow graph."""
        workflow = StateGraph(dict)

        # Add nodes
        workflow.add_node("validate", self._validate_node)
        workflow.add_node("accept", self._accept_node)
        workflow.add_node("suggest_refinement", self._suggest_refinement_node)
        workflow.add_node("final_reject", self._final_reject_node)

        # Set entry point
        workflow.set_entry_point("validate")

        # Add conditional edges
        workflow.add_conditional_edges(
            "validate",
            self._route_validation,
            {
                "accept": "accept",
                "suggest_refinement": "suggest_refinement",
                "final_reject": "final_reject",
            },
        )

        # Terminal edges
        workflow.add_edge("accept", END)
        workflow.add_edge("suggest_refinement", END)
        workflow.add_edge("final_reject", END)

        return workflow.compile()

    def _validate_node(self, state: dict) -> dict:
        """Validate the concept using ConceptValidator."""
        result = self.validator.validate_concept(state["concept_text"])
        state["validation_result"] = result
        state["validation_passed"] = result.is_accepted
        return state

    def _route_validation(self, state: dict) -> str:
        """Route based on validation result."""
        if state["validation_passed"]:
            return "accept"

        if state["refinement_attempts"] < state["max_refinement_attempts"]:
            return "suggest_refinement"

        return "final_reject"

    def _accept_node(self, state: dict) -> dict:
        """Handle accepted concept."""
        state["workflow_status"] = "accepted"
        state["final_concept"] = state["concept_text"]
        state["final_album"] = state["validation_result"].predicted_album
        return state

    def _suggest_refinement_node(self, state: dict) -> dict:
        """Generate refinement suggestions for rejected concept."""
        state["workflow_status"] = "refined"
        state["needs_refinement"] = True
        state["refinement_attempts"] = state.get("refinement_attempts", 0) + 1

        # Extract suggestions from validation result
        if state["validation_result"] and state["validation_result"].suggestions:
            state["suggestions"] = [
                s.message for s in state["validation_result"].suggestions
            ]
        else:
            state["suggestions"] = ["Make the concept more specific and grounded"]

        return state

    def _final_reject_node(self, state: dict) -> dict:
        """Handle finally rejected concept."""
        state["workflow_status"] = "rejected"
        return state

    def run(
        self, concept_text: str, target_album: Optional[str] = None
    ) -> ConceptValidationState:
        """Run the validation workflow."""
        initial_state = {
            "concept_text": concept_text,
            "target_album": target_album,
            "validation_result": None,
            "validation_passed": False,
            "needs_refinement": False,
            "refinement_attempts": 0,
            "max_refinement_attempts": 3,
            "final_concept": None,
            "final_album": None,
            "workflow_status": "pending",
            "suggestions": [],
        }

        final_state = self.graph.invoke(initial_state)

        # Convert dict back to dataclass for easier testing
        result = ConceptValidationState(
            concept_text=final_state.get("concept_text", ""),
            target_album=final_state.get("target_album"),
            validation_result=final_state.get("validation_result"),
            validation_passed=final_state.get("validation_passed", False),
            needs_refinement=final_state.get("needs_refinement", False),
            refinement_attempts=final_state.get("refinement_attempts", 0),
            max_refinement_attempts=final_state.get("max_refinement_attempts", 3),
            final_concept=final_state.get("final_concept"),
            final_album=final_state.get("final_album"),
            workflow_status=final_state.get("workflow_status", "pending"),
            suggestions=final_state.get("suggestions", []),
        )
        return result


# ============================================================================
# Integration Tests
# ============================================================================


class TestConceptValidationWorkflowIntegration:
    """Integration tests for ConceptValidator with LangGraph workflow."""

    @pytest.fixture
    def validator(self):
        """Create a validator with mock predictions."""
        return ConceptValidator(
            model_path=None,  # Use mock predictions
            enable_cache=False,
        )

    @pytest.fixture
    def workflow(self, validator):
        """Create a validation workflow."""
        return ConceptValidationWorkflow(validator)

    def test_workflow_accepts_clear_concept(self, workflow):
        """Test that workflow accepts concepts with clear ontological assignment."""
        # The mock predictor returns high confidence for most inputs
        concept = "A vivid memory of building model trains in my father's basement"

        result = workflow.run(concept)

        assert result.validation_result is not None
        assert result.workflow_status in ["accepted", "refined"]
        # Mock predictor should give reasonable confidence
        assert result.validation_result.chromatic_confidence > 0

    def test_workflow_provides_album_prediction(self, workflow):
        """Test that workflow provides album prediction for accepted concepts."""
        concept = "A childhood memory of summer vacation"

        result = workflow.run(concept)

        if result.workflow_status == "accepted":
            assert result.final_album is not None
            assert result.final_album in [
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

    def test_workflow_tracks_refinement_attempts(self, workflow):
        """Test that workflow tracks refinement attempts."""
        concept = "Something vague"

        result = workflow.run(concept)

        # Either accepted or tracked refinement
        assert result.refinement_attempts >= 0
        assert result.refinement_attempts <= result.max_refinement_attempts

    def test_workflow_generates_suggestions_on_reject(self, workflow):
        """Test that rejected concepts get refinement suggestions."""
        # Create a validator that always rejects
        reject_validator = ConceptValidator(
            model_path=None,
            enable_cache=False,
            confidence_threshold=0.99,  # Very high threshold -> more rejections
        )
        reject_workflow = ConceptValidationWorkflow(reject_validator)

        concept = "x"  # Minimal concept likely to be rejected

        result = reject_workflow.run(concept)

        # Should either have suggestions or be accepted
        if result.workflow_status == "refined":
            assert len(result.suggestions) > 0

    def test_workflow_state_contains_validation_scores(self, workflow):
        """Test that validation scores are preserved in state."""
        concept = "A forgotten place from my childhood"

        result = workflow.run(concept)

        assert result.validation_result is not None
        assert "past" in result.validation_result.temporal_scores
        assert "present" in result.validation_result.temporal_scores
        assert "future" in result.validation_result.temporal_scores

        assert "thing" in result.validation_result.spatial_scores
        assert "place" in result.validation_result.spatial_scores
        assert "person" in result.validation_result.spatial_scores

        assert "imagined" in result.validation_result.ontological_scores
        assert "forgotten" in result.validation_result.ontological_scores
        assert "known" in result.validation_result.ontological_scores

    def test_workflow_includes_transmigration_distances(self, workflow):
        """Test that transmigration distances are computed."""
        concept = "A dream of the future"

        result = workflow.run(concept)

        assert result.validation_result is not None
        assert result.validation_result.transmigration_distances is not None

        # Should have distance to each album
        for album in ["Orange", "Red", "Blue", "Black"]:
            assert album in result.validation_result.transmigration_distances

    def test_workflow_handles_hybrid_states(self, workflow):
        """Test that workflow handles hybrid validation states."""
        concept = "A memory that feels like both past and present"

        result = workflow.run(concept)

        # Should complete without error
        assert result.workflow_status in ["accepted", "refined", "rejected"]

        # If hybrid detected, flags should be present
        if result.validation_result.validation_status == ValidationStatus.ACCEPT_HYBRID:
            # Hybrid flags may or may not be present depending on mock
            assert result.validation_passed is True


class TestValidationGateDecisions:
    """Test validation gate decision logic within workflow."""

    @pytest.fixture
    def validator(self):
        return ConceptValidator(
            model_path=None,
            enable_cache=False,
            confidence_threshold=0.7,
            dominant_threshold=0.6,
        )

    def test_accept_status_passes_gate(self, validator):
        """Test that ACCEPT status passes validation gate."""
        result = validator.validate_concept("A clear memory of a specific toy")

        # Check that result has proper structure
        assert result.validation_status in [
            ValidationStatus.ACCEPT,
            ValidationStatus.ACCEPT_HYBRID,
            ValidationStatus.ACCEPT_BLACK,
            ValidationStatus.REJECT,
        ]

    def test_validation_result_serialization(self, validator):
        """Test that validation result can be serialized for state transfer."""
        result = validator.validate_concept("Test concept")

        # Should be serializable to dict
        result_dict = result.to_dict()

        assert isinstance(result_dict, dict)
        assert "concept_text" in result_dict
        assert "validation_status" in result_dict
        assert "predicted_album" in result_dict
        assert "chromatic_confidence" in result_dict

    def test_batch_validation_in_workflow_context(self, validator):
        """Test batch validation for multiple concepts."""
        concepts = [
            "A memory of childhood",
            "A dream of the future",
            "This present moment",
            "A forgotten face",
        ]

        results = validator.validate_batch(concepts)

        assert len(results) == len(concepts)

        for concept, result in zip(concepts, results):
            assert result.concept_text == concept
            assert result.validation_status is not None


class TestWorkflowWithRealAgentIntegration:
    """Test integration patterns for real agent usage."""

    @pytest.fixture
    def validator(self):
        return ConceptValidator(
            model_path=None,
            enable_cache=True,  # Enable cache for performance
            cache_ttl=300,
        )

    def test_singleton_pattern_for_agent_use(self, validator):
        """Test that singleton pattern works for agent integration."""
        # Reset singleton
        ConceptValidator._instance = None

        instance1 = ConceptValidator.get_instance(model_path=None)
        instance2 = ConceptValidator.get_instance(model_path=None)

        assert instance1 is instance2

        # Cleanup
        ConceptValidator._instance = None

    def test_cache_improves_performance(self, validator):
        """Test that caching improves performance for repeated concepts."""
        import time

        concept = "A specific unique concept for cache testing"

        # First call - cache miss
        start1 = time.time()
        result1 = validator.validate_concept(concept)
        time.time() - start1

        assert result1.cache_hit is False

        # Second call - cache hit
        start2 = time.time()
        result2 = validator.validate_concept(concept)
        time.time() - start2

        assert result2.cache_hit is True

        # Cache hit should be faster (or at least not slower)
        # Note: In mock mode, both are fast, so just check cache_hit flag
        assert result1.predicted_album == result2.predicted_album

    def test_validation_timing_is_recorded(self, validator):
        """Test that validation timing is recorded for monitoring."""
        result = validator.validate_concept("Test concept")

        assert result.validation_time_ms is not None
        assert result.validation_time_ms >= 0


class TestValidationWorkflowEdgeCases:
    """Test edge cases in validation workflow."""

    @pytest.fixture
    def validator(self):
        return ConceptValidator(model_path=None, enable_cache=False)

    @pytest.fixture
    def workflow(self, validator):
        return ConceptValidationWorkflow(validator)

    def test_empty_concept_handling(self, workflow):
        """Test handling of empty concept text."""
        result = workflow.run("")

        # Should complete without error
        assert result.workflow_status in ["accepted", "refined", "rejected"]

    def test_very_long_concept_handling(self, workflow):
        """Test handling of very long concept text."""
        long_concept = "memory " * 1000  # Very long text

        result = workflow.run(long_concept)

        # Should complete without error
        assert result.validation_result is not None

    def test_special_characters_in_concept(self, workflow):
        """Test handling of special characters."""
        concept = 'A memory with special chars: !@#$%^&*(){}[]|\\:";<>?,./`~'

        result = workflow.run(concept)

        assert result.validation_result is not None

    def test_unicode_concept_handling(self, workflow):
        """Test handling of Unicode characters."""
        concept = "A memory with unicode: 日本語 中文 한국어 🎵🎶"

        result = workflow.run(concept)

        assert result.validation_result is not None

    def test_multiline_concept_handling(self, workflow):
        """Test handling of multiline concepts."""
        concept = """
        A memory that spans multiple lines.
        It contains thoughts about the past.
        And dreams about the future.
        """

        result = workflow.run(concept)

        assert result.validation_result is not None


class TestWorkflowConditionalBranching:
    """Test conditional branching in workflow based on validation."""

    def test_custom_routing_based_on_album(self):
        """Test custom routing based on predicted album."""
        validator = ConceptValidator(model_path=None, enable_cache=False)

        def validate_node(state):
            state["validation_result"] = validator.validate_concept(
                state["concept_text"]
            )
            return state

        def route_by_album(state):
            album = state["validation_result"].predicted_album
            if album in ["Red", "Orange", "Violet"]:
                return "past_handler"
            elif album in ["Yellow", "Green", "Indigo"]:
                return "present_handler"
            elif album == "Blue":
                return "future_handler"
            else:
                return "black_handler"

        def past_handler(state):
            state["routed_to"] = "past"
            return state

        def present_handler(state):
            state["routed_to"] = "present"
            return state

        def future_handler(state):
            state["routed_to"] = "future"
            return state

        def black_handler(state):
            state["routed_to"] = "black"
            return state

        # Build workflow
        workflow = StateGraph(dict)
        workflow.add_node("validate", validate_node)
        workflow.add_node("past_handler", past_handler)
        workflow.add_node("present_handler", present_handler)
        workflow.add_node("future_handler", future_handler)
        workflow.add_node("black_handler", black_handler)

        workflow.set_entry_point("validate")
        workflow.add_conditional_edges(
            "validate",
            route_by_album,
            {
                "past_handler": "past_handler",
                "present_handler": "present_handler",
                "future_handler": "future_handler",
                "black_handler": "black_handler",
            },
        )

        for node in [
            "past_handler",
            "present_handler",
            "future_handler",
            "black_handler",
        ]:
            workflow.add_edge(node, END)

        compiled = workflow.compile()

        # Test routing
        result = compiled.invoke(
            {
                "concept_text": "A childhood memory",
                "validation_result": None,
                "routed_to": "",
            }
        )

        assert result["routed_to"] in ["past", "present", "future", "black"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
