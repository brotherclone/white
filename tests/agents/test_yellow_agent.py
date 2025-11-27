from app.agents.states.yellow_agent_state import YellowAgentState
from app.agents.yellow_agent import YellowAgent
from app.structures.manifests.song_proposal import SongProposalIteration


def test_generate_alternate_song_spec_mock():
    agent = YellowAgent()
    state = YellowAgentState()
    result_state = agent.generate_alternate_song_spec(state)
    assert result_state.counter_proposal is not None
    assert isinstance(result_state.counter_proposal, SongProposalIteration)
    assert getattr(result_state.counter_proposal, "title", None)
