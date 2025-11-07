from unittest.mock import MagicMock, patch

import pytest

from app.agents.states.white_agent_state import MainAgentState
from app.agents.white_agent import WhiteAgent
from app.structures.agents.agent_settings import AgentSettings
from app.structures.manifests.song_proposal import SongProposal, SongProposalIteration


def test_white_agent_initialization():
    agent = WhiteAgent()
    assert isinstance(agent.settings, AgentSettings)
    assert isinstance(agent.agents, dict)
    assert isinstance(agent.processors, dict)
    assert isinstance(agent.song_proposal, SongProposal)


@pytest.mark.parametrize(
    "input_val,expected_type",
    [
        (SongProposal(iterations=[]), SongProposal),
        ({"iterations": []}, SongProposal),
        (None, SongProposal),
    ],
)
def test_normalize_song_proposal(input_val, expected_type):
    result = WhiteAgent._normalize_song_proposal(input_val)
    assert isinstance(result, expected_type)


def test_invoke_black_agent():
    mock_state = MagicMock(spec=MainAgentState)
    mock_black_agent = MagicMock(return_value=mock_state)
    agent = WhiteAgent()
    agent.agents["black"] = mock_black_agent  # Inject mock
    result = agent.invoke_black_agent(mock_state)
    assert result == mock_state
    mock_black_agent.assert_called_once_with(mock_state)


def test_invoke_red_agent():
    mock_state = MagicMock(spec=MainAgentState)
    mock_red_agent = MagicMock(return_value=mock_state)
    agent = WhiteAgent()
    agent.agents["red"] = mock_red_agent  # Inject mock
    result = agent.invoke_red_agent(mock_state)
    assert result == mock_state
    mock_red_agent.assert_called_once_with(mock_state)


def test_resume_after_black_agent_ritual(monkeypatch):
    """Test resuming workflow after black agent ritual completion"""

    monkeypatch.setenv("MOCK_MODE", "true")
    with patch(
        "app.agents.white_agent.resume_black_agent_workflow_with_agent"
    ) as mock_resume:
        mock_resume.return_value = {
            "counter_proposal": SongProposalIteration(
                iteration_id="mock_123",
                bpm=120,
                tempo="4/4",
                key="C Major",
                rainbow_color="black",
                title="Black Counter Proposal",
                mood=["dark"],
                genres=["experimental"],
                concept="Mock counter proposal concept. " * 10,
            ),
            "artifacts": [],
        }

        agent = WhiteAgent()
        agent._black_rebracketing_analysis = MagicMock(return_value="Mock rebracketing")
        agent._synthesize_document_for_red = MagicMock(return_value="Mock synthesis")
        state = MainAgentState(
            thread_id="test-thread-123",
            workflow_paused=True,
            song_proposals=SongProposal(iterations=[]),
            artifacts=[],
            pending_human_action={
                "agent": "black",
                "black_config": {"configurable": {"thread_id": "test-thread-123"}},
            },
        )
        updated_state = agent.resume_after_black_agent_ritual(state, verify_tasks=False)
        assert updated_state.workflow_paused is False
        assert (
            updated_state.song_proposals.iterations[-1].title
            == "Black Counter Proposal"
        )
