from app.agents.indigo_agent import IndigoAgent
from app.agents.states.indigo_agent_state import IndigoAgentState
from app.structures.manifests.song_proposal import SongProposalIteration


def test_generate_alternate_song_spec_mock():
    agent = IndigoAgent()
    state = IndigoAgentState()
    result_state = agent.generate_alternate_song_spec(state)
    assert result_state.counter_proposal is not None
    assert isinstance(result_state.counter_proposal, SongProposalIteration)
    assert getattr(result_state.counter_proposal, "title", None)
