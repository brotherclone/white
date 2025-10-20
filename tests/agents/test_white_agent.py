import pytest
from unittest.mock import MagicMock, patch
from app.agents.white_agent import WhiteAgent
from app.agents.models.agent_settings import AgentSettings
from app.structures.concepts.rainbow_table_color import the_rainbow_table_colors
from app.structures.manifests.song_proposal import SongProposal, SongProposalIteration
from app.agents.states.main_agent_state import MainAgentState

def test_white_agent_initialization():
    agent = WhiteAgent()
    assert isinstance(agent.settings, AgentSettings)
    assert isinstance(agent.agents, dict)
    assert isinstance(agent.processors, dict)
    assert isinstance(agent.song_proposal, SongProposal)

@pytest.mark.parametrize("input_val,expected_type", [
    (SongProposal(iterations=[]), SongProposal),
    ({"iterations": []}, SongProposal),
    (None, SongProposal),
])
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

@patch("app.agents.white_agent.resume_black_agent_workflow")
def test_resume_after_black_agent_ritual(mock_resume):
    state = MagicMock(spec=MainAgentState)
    state.pending_human_action = {
        "agent": "black",
        "black_config": {"foo": "bar"}
    }
    state.song_proposals = SongProposal(iterations=[])
    mock_resume.return_value = {"counter_proposal": SongProposalIteration(
        iteration_id="123",
        bpm=120,
        tempo="4/4",
        key="C Major",
        rainbow_color=the_rainbow_table_colors['Z'],
        title="Test",
        mood=[],
        genres=[],
        concept="Test concept"
    )}
    updated_state = WhiteAgent.resume_after_black_agent_ritual(state)
    assert updated_state.song_proposals.iterations[-1].title == "Test"
    assert updated_state.pending_human_action is None
    assert updated_state.workflow_paused is False
