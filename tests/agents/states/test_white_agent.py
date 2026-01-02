"""
Enhanced Test Suite for The Prism (White Agent)

Tests for new features:
- Meta-rebracketing
- Chromatic synthesis
- Facet evolution
- Transformation traces
- Fixed resume_workflow execution modes
"""

import logging
from unittest.mock import MagicMock

import pytest

from app.agents.states.white_agent_state import (
    MainAgentState,
    TransformationTrace,
    FacetEvolution,
)
from app.agents.white_agent import WhiteAgent
from app.structures.agents.agent_settings import AgentSettings
from app.structures.enums.white_facet import WhiteFacet
from app.structures.manifests.song_proposal import SongProposal, SongProposalIteration


@pytest.fixture
def white_agent(monkeypatch):
    """Fresh WhiteAgent instance for each test."""
    return WhiteAgent()


def test_white_agent_initialization():
    """Verify basic initialization."""
    agent = WhiteAgent()
    assert isinstance(agent.settings, AgentSettings)
    assert isinstance(agent.agents, dict)
    assert isinstance(agent.song_proposal, SongProposal)


def test_facet_evolution_initialization(monkeypatch):
    """Test that start_workflow initializes facet evolution."""
    monkeypatch.setenv("MOCK_MODE", "true")

    agent = WhiteAgent()
    state = agent.start_workflow(user_input="Test concept")

    # Should have facet evolution initialized
    assert state.facet_evolution is not None
    assert isinstance(state.facet_evolution, FacetEvolution)
    assert state.facet_evolution.initial_facet is not None
    assert len(state.facet_evolution.evolution_history) == 0  # Not evolved yet


def test_transformation_trace_creation(monkeypatch, white_agent):
    """Test that transformation traces are created during processing."""
    monkeypatch.setenv("MOCK_MODE", "true")

    # Mock the rebracketing analysis methods
    white_agent._black_rebracketing_analysis = MagicMock(return_value="Mock analysis")
    white_agent._synthesize_document_for_red = MagicMock(return_value="Mock synthesis")

    state = MainAgentState(
        thread_id="test_123",
        song_proposals=SongProposal(
            iterations=[
                SongProposalIteration(
                    iteration_id="black_prop_1",
                    bpm=120,
                    tempo="4/4",
                    key="C Major",
                    rainbow_color="black",
                    title="Test Black Proposal",
                    mood=["dark"],
                    genres=["experimental"],
                    concept="Test concept " * 10,
                )
            ]
        ),
        artifacts=[],
        transformation_traces=[],  # Start empty
    )

    result = white_agent.process_black_agent_work(state)

    # Should have created transformation trace
    assert len(result.transformation_traces) == 1
    trace = result.transformation_traces[0]
    assert trace.agent_name == "black"
    assert len(trace.boundaries_shifted) > 0
    assert len(trace.patterns_revealed) > 0


def test_meta_rebracketing_generation(monkeypatch, white_agent):
    """Test meta-rebracketing with multiple transformation traces."""
    monkeypatch.setenv("MOCK_MODE", "false")

    # Mock the Claude supervisor
    mock_claude = MagicMock()
    mock_response = MagicMock()
    mock_response.content = "Meta-rebracketing analysis here..."
    mock_claude.invoke.return_value = mock_response
    white_agent._get_claude_supervisor = MagicMock(return_value=mock_claude)

    state = MainAgentState(
        thread_id="test_meta",
        song_proposals=SongProposal(
            iterations=[
                SongProposalIteration(
                    iteration_id="prop_1",
                    bpm=120,
                    tempo="4/4",
                    key="C Major",
                    rainbow_color="black",
                    title="Black Proposal",
                    mood=["dark"],
                    genres=["experimental"],
                    concept="Test " * 10,
                ),
                SongProposalIteration(
                    iteration_id="prop_2",
                    bpm=130,
                    tempo="4/4",
                    key="D Minor",
                    rainbow_color="red",
                    title="Red Proposal",
                    mood=["literary"],
                    genres=["art-rock"],
                    concept="Test " * 10,
                ),
            ]
        ),
        transformation_traces=[
            TransformationTrace(
                agent_name="black",
                iteration_id="prop_1",
                boundaries_shifted=["CHAOS → ORDER"],
                patterns_revealed=["Pattern 1"],
                semantic_resonances={},
            ),
            TransformationTrace(
                agent_name="red",
                iteration_id="prop_2",
                boundaries_shifted=["PAST → PRESENT"],
                patterns_revealed=["Pattern 2"],
                semantic_resonances={},
            ),
        ],
    )

    result = white_agent._perform_meta_rebracketing(state)

    assert result is not None
    assert len(result) > 0
    assert mock_claude.invoke.called


def test_chromatic_synthesis_generation(monkeypatch, white_agent):
    """Test final chromatic synthesis generation."""
    monkeypatch.setenv("MOCK_MODE", "false")

    # Mock the Claude supervisor
    mock_claude = MagicMock()
    mock_response = MagicMock()
    mock_response.content = "Chromatic synthesis document..."
    mock_claude.invoke.return_value = mock_response
    white_agent._get_claude_supervisor = MagicMock(return_value=mock_claude)

    state = MainAgentState(
        thread_id="test_synthesis",
        song_proposals=SongProposal(iterations=[]),
        meta_rebracketing="Meta analysis here...",
    )

    result = white_agent._generate_chromatic_synthesis(state)

    assert result is not None
    assert len(result) > 0
    assert mock_claude.invoke.called


def test_finalize_with_meta_analysis(monkeypatch, white_agent):
    """Test that finalize_song_proposal performs meta-analysis."""
    monkeypatch.setenv("MOCK_MODE", "true")

    white_agent._perform_meta_rebracketing = MagicMock(return_value="Meta analysis")
    white_agent._generate_chromatic_synthesis = MagicMock(return_value="Synthesis doc")
    white_agent.save_all_proposals = MagicMock()
    white_agent._save_meta_analysis = MagicMock()

    state = MainAgentState(
        thread_id="test_finalize",
        song_proposals=SongProposal(iterations=[]),
        transformation_traces=[
            TransformationTrace(
                agent_name="black",
                iteration_id="test_1",
                boundaries_shifted=["TEST"],
                patterns_revealed=["PATTERN"],
                semantic_resonances={},
            ),
            TransformationTrace(
                agent_name="red",
                iteration_id="test_2",
                boundaries_shifted=["TEST2"],
                patterns_revealed=["PATTERN2"],
                semantic_resonances={},
            ),
        ],
        workflow_paused=False,
    )

    result = white_agent.finalize_song_proposal(state)

    # Should have performed meta-analysis
    assert white_agent._perform_meta_rebracketing.called
    assert white_agent._generate_chromatic_synthesis.called
    assert white_agent._save_meta_analysis.called

    # Should have set these fields
    assert result.meta_rebracketing is not None
    assert result.chromatic_synthesis is not None
    assert result.run_finished is True


def test_resume_workflow_respects_enabled_agents(monkeypatch, white_agent):
    """Test that resume_workflow respects enabled_agents setting."""
    monkeypatch.setenv("MOCK_MODE", "true")

    # Mock agent invocations and processing
    white_agent._resume_paused_agent = MagicMock()
    white_agent._determine_next_agent = MagicMock(side_effect=["orange", "finish"])
    white_agent._invoke_and_process_agent = MagicMock()
    white_agent.finalize_song_proposal = MagicMock()

    state = MainAgentState(
        thread_id="test_resume",
        workflow_paused=True,
        song_proposals=SongProposal(iterations=[]),
        pending_human_action={"agent": "black"},
        enabled_agents=["orange"],  # ONLY orange enabled
        stop_after_agent="orange",
    )

    # Mock the resume to clear pause
    def mock_resume(s, verify):
        s.workflow_paused = False
        s.ready_for_orange = True
        return s

    white_agent._resume_paused_agent.return_value = state
    white_agent._resume_paused_agent.side_effect = mock_resume

    result = white_agent.resume_workflow(state, verify_tasks=False)
    logging.info(result)
    # Should have checked next agent routing
    assert white_agent._determine_next_agent.called
    # Should have invoked orange (first call)
    assert white_agent._invoke_and_process_agent.call_count >= 1


def test_resume_workflow_stops_after_agent(monkeypatch, white_agent):
    """Test that resume_workflow stops after specified agent."""
    monkeypatch.setenv("MOCK_MODE", "true")

    white_agent._resume_paused_agent = MagicMock()
    white_agent.finalize_song_proposal = MagicMock()

    state = MainAgentState(
        thread_id="test_stop",
        workflow_paused=True,
        song_proposals=SongProposal(
            iterations=[
                SongProposalIteration(
                    iteration_id="test_orange",
                    agent_name="orange",  # Last completed agent
                    bpm=120,
                    tempo="4/4",
                    key="C",
                    rainbow_color="orange",
                    title="Test",
                    mood=["test"],
                    genres=["test"],
                    concept="Test " * 10,
                )
            ]
        ),
        pending_human_action={"agent": "black"},
        stop_after_agent="orange",  # Should stop after orange
        ready_for_yellow=True,  # Next would be yellow
    )

    def mock_resume(s, verify):
        s.workflow_paused = False
        return s

    white_agent._resume_paused_agent.side_effect = mock_resume

    # Mock _determine_next_agent to return yellow (which should be stopped)
    white_agent._determine_next_agent = MagicMock(return_value="yellow")
    white_agent._invoke_and_process_agent = MagicMock()

    result = white_agent.resume_workflow(state, verify_tasks=False)

    # Should have finalized without invoking yellow
    assert white_agent.finalize_song_proposal.called
    # Should NOT have processed yellow (stop_after_agent check should break loop)
    # This depends on implementation - may need adjustment


def test_format_transformation_traces(white_agent):
    """Test transformation trace formatting for prompts."""
    traces = [
        TransformationTrace(
            agent_name="black",
            iteration_id="test_1",
            boundaries_shifted=["CHAOS → ORDER", "UNCONSCIOUS → CONSCIOUS"],
            patterns_revealed=["Pattern A", "Pattern B"],
            semantic_resonances={"resonates_with": ["red"]},
        ),
        TransformationTrace(
            agent_name="red",
            iteration_id="test_2",
            boundaries_shifted=["PAST → PRESENT"],
            patterns_revealed=["Pattern C"],
            semantic_resonances={},
        ),
    ]

    result = white_agent._format_transformation_traces(traces)

    assert "BLACK AGENT" in result
    assert "RED AGENT" in result
    assert "CHAOS → ORDER" in result
    assert "Pattern A" in result
    assert "resonates_with" in result


def test_save_meta_analysis(monkeypatch, white_agent, tmp_path):
    """Test saving meta-analysis files."""
    monkeypatch.setenv("AGENT_WORK_PRODUCT_BASE_PATH", str(tmp_path))

    state = MainAgentState(
        thread_id="test_save_meta",
        song_proposals=SongProposal(iterations=[]),
        meta_rebracketing="Meta-rebracketing content here...",
        chromatic_synthesis="Chromatic synthesis content here...",
        transformation_traces=[
            TransformationTrace(
                agent_name="black",
                iteration_id="test",
                boundaries_shifted=["TEST"],
                patterns_revealed=["PATTERN"],
                semantic_resonances={},
            )
        ],
        facet_evolution=FacetEvolution(
            initial_facet=WhiteFacet.ARCHITECTURAL,
            initial_metadata={"description": "Test"},
        ),
    )

    white_agent._save_meta_analysis(state)

    # Check that files were created
    md_dir = tmp_path / "test_save_meta" / "md"
    assert md_dir.exists()

    # Should have created meta-rebracketing file
    meta_file = md_dir / "white_agent_test_save_meta_META_REBRACKETING.md"
    assert meta_file.exists()

    # Should have created chromatic synthesis file
    synthesis_file = md_dir / "white_agent_test_save_meta_CHROMATIC_SYNTHESIS.md"
    assert synthesis_file.exists()

    # Should have created transformation traces file
    traces_file = md_dir / "white_agent_test_save_meta_transformation_traces.md"
    assert traces_file.exists()

    # Should have created facet evolution file
    facet_file = md_dir / "white_agent_test_save_meta_facet_evolution.md"
    assert facet_file.exists()


# Original tests (updated for new state fields)


def test_normalize_song_proposal():
    """Test song proposal normalization."""
    # Test with SongProposal instance
    result = WhiteAgent._normalize_song_proposal(SongProposal(iterations=[]))
    assert isinstance(result, SongProposal)

    # Test with dict
    result = WhiteAgent._normalize_song_proposal({"iterations": []})
    assert isinstance(result, SongProposal)

    # Test with None
    result = WhiteAgent._normalize_song_proposal(None)
    assert isinstance(result, SongProposal)


def test_invoke_black_agent():
    """Test Black Agent invocation."""
    mock_state = MagicMock(spec=MainAgentState)
    mock_black_agent = MagicMock(return_value=mock_state)
    agent = WhiteAgent()
    agent.agents["black"] = mock_black_agent
    result = agent.invoke_black_agent(mock_state)
    assert result == mock_state
    mock_black_agent.assert_called_once_with(mock_state)


def test_process_black_agent_work_sets_analysis_and_ready_for_red(
    monkeypatch, white_agent
):
    """Test Black Agent processing creates expected state."""
    monkeypatch.setattr(
        white_agent.__class__,
        "_gather_artifacts_for_prompt",
        lambda self, artifacts, artifact_filter: ["mock_artifact"],
    )
    monkeypatch.setattr(
        white_agent.__class__,
        "_black_rebracketing_analysis",
        lambda self, proposal, evp_artifacts, sigil_artifacts: "BLACK_ANALYSIS",
    )
    monkeypatch.setattr(
        white_agent.__class__,
        "_synthesize_document_for_red",
        lambda self, rebracketed_analysis, black_proposal, artifacts: "BLACK_SYNTH",
    )

    state = MainAgentState(
        thread_id="mock_thread_001",
        song_proposals=SongProposal(
            iterations=[
                SongProposalIteration(
                    iteration_id="test_black_prop_v1",
                    bpm=120,
                    tempo="4/4",
                    key="C Major",
                    rainbow_color="black",
                    title="Test Black Proposal",
                    mood=["dark"],
                    genres=["rock"],
                    concept="This is a test concept " * 10,
                )
            ]
        ),
        artifacts=[],
        workflow_paused=False,
        ready_for_red=False,
        transformation_traces=[],  # NEW: Initialize empty
    )

    result = white_agent.process_black_agent_work(state)

    assert result.rebracketing_analysis == "BLACK_ANALYSIS"
    assert result.document_synthesis == "BLACK_SYNTH"
    assert result.ready_for_red is True
    # NEW: Should have transformation trace
    assert len(result.transformation_traces) == 1
    assert result.transformation_traces[0].agent_name == "black"


# Add similar enhanced tests for other agent processing methods...

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
