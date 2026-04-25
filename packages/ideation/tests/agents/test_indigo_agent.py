from white_core.manifests.song_proposal import SongProposalIteration
from white_ideation.agents.indigo_agent import IndigoAgent
from white_ideation.agents.states.indigo_agent_state import IndigoAgentState


def test_generate_alternate_song_spec_mock():
    agent = IndigoAgent()
    state = IndigoAgentState()
    result_state = agent.generate_alternate_song_spec(state)
    assert result_state.counter_proposal is not None
    assert isinstance(result_state.counter_proposal, SongProposalIteration)
    assert getattr(result_state.counter_proposal, "title", None)
