from app.agents.orange_agent import OrangeAgent
from app.agents.states.orange_rainbow_state import OrangeAgentState
from app.structures.manifests.song_proposal import SongProposalIteration


def test_generate_alternate_song_spec_mock():
    agent = OrangeAgent()
    state = OrangeAgentState()
    result_state = agent.generate_alternate_song_spec(state)
    assert result_state.counter_proposal is not None
    assert isinstance(result_state.counter_proposal, SongProposalIteration)
    assert getattr(result_state.counter_proposal, "title", None)
