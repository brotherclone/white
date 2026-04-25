from white_core.manifests.song_proposal import SongProposalIteration
from white_ideation.agents.blue_agent import BlueAgent
from white_ideation.agents.states.blue_agent_state import BlueAgentState


def test_generate_alternate_song_spec_mock():
    agent = BlueAgent()
    state = BlueAgentState()
    state.thread_id = "test_thread"
    result_state = agent.generate_alternate_song_spec(state)
    assert result_state.counter_proposal is not None
    assert isinstance(result_state.counter_proposal, SongProposalIteration)
    assert getattr(result_state.counter_proposal, "title", None)
