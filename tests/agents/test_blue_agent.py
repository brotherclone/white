from app.agents.blue_agent import BlueAgent
from app.agents.states.blue_agent_state import BlueAgentState
from app.structures.manifests.song_proposal import SongProposalIteration


def test_generate_alternate_song_spec_mock():
    agent = BlueAgent()
    state = BlueAgentState()
    result_state = agent.generate_alternate_song_spec(state)
    assert result_state.counter_proposal is not None
    assert isinstance(result_state.counter_proposal, SongProposalIteration)
    assert getattr(result_state.counter_proposal, "title", None)
