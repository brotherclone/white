from app.agents.states.violet_agent_state import VioletAgentState
from app.agents.violet_agent import VioletAgent
from app.structures.manifests.song_proposal import SongProposalIteration
from app.structures.concepts.vanity_persona import VanityPersona


def test_generate_alternate_song_spec_mock():
    agent = VioletAgent()

    # Create mock VanityPersona objects
    interviewer = VanityPersona(first_name="Jane", last_name="Interviewer")
    interviewee = VanityPersona(first_name="John", last_name="Interviewee")

    state = VioletAgentState(
        interviewer_persona=interviewer, interviewee_persona=interviewee
    )
    result_state = agent.generate_alternate_song_spec(state)
    assert result_state.counter_proposal is not None
    assert isinstance(result_state.counter_proposal, SongProposalIteration)
    assert getattr(result_state.counter_proposal, "title", None)
