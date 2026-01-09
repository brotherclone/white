from app.agents.states.yellow_agent_state import YellowAgentState
from app.agents.yellow_agent import YellowAgent
from app.structures.manifests.song_proposal import SongProposalIteration


def test_generate_alternate_song_spec_mock(monkeypatch):
    """Test that generate_alternate_song_spec loads from mock in mock mode"""
    monkeypatch.setenv("MOCK_MODE", "true")
    agent = YellowAgent()
    state = YellowAgentState()
    result_state = agent.generate_alternate_song_spec(state)
    assert result_state.counter_proposal is not None
    assert isinstance(result_state.counter_proposal, SongProposalIteration)
    assert getattr(result_state.counter_proposal, "title", None)


def test_yellow_agent_initialization():
    """Test that YellowAgent can be initialized"""
    agent = YellowAgent()
    assert agent is not None
    assert agent.room_generator is not None
    assert agent.action_generator is not None
    assert agent.music_extractor is not None
    assert agent.max_rooms == 4
