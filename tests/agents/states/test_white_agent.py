import pytest
from app.agents.states.white_agent_state import (
    MainAgentState,
)
from app.structures.manifests.song_proposal import SongProposal, SongProposalIteration
from app.structures.concepts.transformation_trace import TransformationTrace
from app.structures.concepts.facet_evolution import FacetEvolution
from app.structures.enums.white_facet import WhiteFacet
from app.structures.artifacts.artifact_relationship import ArtifactRelationship


def test_main_agent_state_initialization():
    """Test MainAgentState initializes with correct defaults."""
    state = MainAgentState(thread_id="test_thread")

    assert state.thread_id == "test_thread"
    assert isinstance(state.song_proposals, SongProposal)
    assert len(state.song_proposals.iterations) == 0
    assert state.artifacts == []
    assert state.workflow_paused is False
    assert state.pause_reason is None
    assert state.pending_human_action is None
    assert state.rebracketing_analysis is None
    assert state.document_synthesis is None
    assert state.meta_rebracketing is None
    assert state.chromatic_synthesis is None
    assert state.white_facet is None
    assert state.white_facet_metadata is None
    assert state.facet_evolution is None
    assert state.transformation_traces == []
    assert state.artifact_relationships == []
    assert state.ready_for_red is False
    assert state.ready_for_orange is False
    assert state.ready_for_yellow is False
    assert state.ready_for_green is False
    assert state.ready_for_blue is False
    assert state.ready_for_indigo is False
    assert state.ready_for_violet is False
    assert state.ready_for_white is False
    assert state.run_finished is False
    assert state.enabled_agents == [
        "black",
        "red",
        "orange",
        "yellow",
        "green",
        "blue",
        "indigo",
        "violet",
    ]
    assert state.stop_after_agent is None


def test_main_agent_state_with_song_proposals():
    """Test MainAgentState with song proposal iterations."""
    iteration = SongProposalIteration(
        iteration_id="test_iteration",
        bpm=120,
        tempo="4/4",
        key="C Major",
        rainbow_color="black",
        title="Test Song",
        mood=["dark"],
        genres=["experimental"],
        concept="A deep philosophical exploration examining the boundaries between consciousness and unconsciousness, manifesting through ritual practice and symbolic representation.",
    )

    state = MainAgentState(
        thread_id="test_thread",
        song_proposals=SongProposal(iterations=[iteration]),
    )

    assert len(state.song_proposals.iterations) == 1
    assert state.song_proposals.iterations[0].iteration_id == "test_iteration"
    assert state.song_proposals.iterations[0].bpm == 120


def test_main_agent_state_with_transformation_traces():
    """Test MainAgentState with transformation traces."""
    trace = TransformationTrace(
        agent_name="black",
        iteration_id="test_iteration",
        boundaries_shifted=["CHAOS → ORDER"],
        patterns_revealed=["Pattern A"],
        semantic_resonances={"resonates_with": ["red"]},
    )

    state = MainAgentState(
        thread_id="test_thread",
        transformation_traces=[trace],
    )

    assert len(state.transformation_traces) == 1
    assert state.transformation_traces[0].agent_name == "black"
    assert "CHAOS → ORDER" in state.transformation_traces[0].boundaries_shifted


def test_main_agent_state_workflow_control():
    """Test workflow control fields."""
    state = MainAgentState(
        thread_id="test_thread",
        workflow_paused=True,
        pause_reason="Awaiting human ritual completion",
        pending_human_action={"agent": "black", "action": "ritual"},
    )

    assert state.workflow_paused is True
    assert state.pause_reason == "Awaiting human ritual completion"
    assert state.pending_human_action["agent"] == "black"


def test_main_agent_state_rebracketing_fields():
    """Test rebracketing analysis fields."""
    state = MainAgentState(
        thread_id="test_thread",
        rebracketing_analysis="Black agent analysis",
        document_synthesis="Synthesized document",
        meta_rebracketing="Meta-rebracketing across all lenses",
        chromatic_synthesis="Final chromatic synthesis",
    )

    assert state.rebracketing_analysis == "Black agent analysis"
    assert state.document_synthesis == "Synthesized document"
    assert state.meta_rebracketing == "Meta-rebracketing across all lenses"
    assert state.chromatic_synthesis == "Final chromatic synthesis"


def test_main_agent_state_facet_evolution():
    """Test white facet evolution tracking."""
    # Get first available WhiteFacet value dynamically
    facet_values = list(WhiteFacet)
    if not facet_values:
        pytest.skip("No WhiteFacet values available")

    first_facet = facet_values[0]

    facet_evolution = FacetEvolution(
        initial_facet=first_facet,
        initial_metadata={"description": "Initial cognitive lens"},
    )

    state = MainAgentState(
        thread_id="test_thread",
        white_facet=first_facet,
        white_facet_metadata={"description": "Current lens"},
        facet_evolution=facet_evolution,
    )

    assert state.white_facet == first_facet
    assert state.white_facet_metadata["description"] == "Current lens"
    assert state.facet_evolution.initial_facet == first_facet


def test_main_agent_state_readiness_flags():
    """Test agent readiness flags can be set."""
    state = MainAgentState(
        thread_id="test_thread",
        ready_for_red=True,
        ready_for_orange=True,
    )

    assert state.ready_for_red is True
    assert state.ready_for_orange is True
    assert state.ready_for_yellow is False
    assert state.ready_for_green is False


def test_main_agent_state_enabled_agents():
    """Test enabled_agents can be customized."""
    state = MainAgentState(
        thread_id="test_thread",
        enabled_agents=["black", "red", "orange"],
    )

    assert state.enabled_agents == ["black", "red", "orange"]
    assert "yellow" not in state.enabled_agents


def test_main_agent_state_stop_after_agent():
    """Test stop_after_agent field."""
    state = MainAgentState(
        thread_id="test_thread",
        stop_after_agent="orange",
    )

    assert state.stop_after_agent == "orange"


def test_main_agent_state_run_finished():
    """Test run_finished flag."""
    state = MainAgentState(
        thread_id="test_thread",
        run_finished=True,
    )

    assert state.run_finished is True


def test_main_agent_state_artifact_relationships():
    """Test artifact relationships tracking."""
    relationship = ArtifactRelationship(
        artifact_id="artifact_1",
        resonant_agents=["red", "orange"],
        entangled_with=["artifact_2", "artifact_3"],
        temporal_depth={"black": "chaos boundary", "red": "archival depth"},
        semantic_tags=["rebracketing", "transformation"],
    )

    state = MainAgentState(
        thread_id="test_thread",
        artifact_relationships=[relationship],
    )

    assert len(state.artifact_relationships) == 1
    assert state.artifact_relationships[0].artifact_id == "artifact_1"
    assert "red" in state.artifact_relationships[0].resonant_agents
    assert "artifact_2" in state.artifact_relationships[0].entangled_with


def test_main_agent_state_complex_workflow():
    """Test a complex workflow scenario with multiple state fields."""
    facet_values = list(WhiteFacet)
    if not facet_values:
        pytest.skip("No WhiteFacet values available")

    iteration = SongProposalIteration(
        iteration_id="complex_test",
        bpm=140,
        tempo="4/4",
        key="D Minor",
        rainbow_color="red",
        title="Complex Test Song",
        mood=["literary", "dark"],
        genres=["art-rock"],
        concept="An exploration of temporal boundaries and archival consciousness through literary archaeology and textual memory reanimation practices.",
    )

    trace = TransformationTrace(
        agent_name="red",
        iteration_id="complex_test",
        boundaries_shifted=["PAST → PRESENT", "ARCHIVE → LIVING TEXT"],
        patterns_revealed=["Literary archaeology", "Temporal collapse"],
        semantic_resonances={"resonates_with": ["orange", "blue"]},
    )

    facet_evolution = FacetEvolution(
        initial_facet=facet_values[0],
        initial_metadata={"description": "Initial"},
    )

    state = MainAgentState(
        thread_id="complex_workflow",
        song_proposals=SongProposal(iterations=[iteration]),
        transformation_traces=[trace],
        workflow_paused=False,
        rebracketing_analysis="Complex analysis",
        document_synthesis="Complex synthesis",
        ready_for_red=True,
        ready_for_orange=True,
        facet_evolution=facet_evolution,
        enabled_agents=["black", "red", "orange", "yellow"],
        stop_after_agent="yellow",
    )

    # Verify all fields are properly set
    assert state.thread_id == "complex_workflow"
    assert len(state.song_proposals.iterations) == 1
    assert len(state.transformation_traces) == 1
    assert state.ready_for_red is True
    assert state.ready_for_orange is True
    assert state.stop_after_agent == "yellow"
    assert "yellow" in state.enabled_agents


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
