from app.agents.states.violet_agent_state import VioletAgentState
from app.agents.violet_agent import VioletAgent
from app.structures.manifests.song_proposal import SongProposal, SongProposalIteration
from app.structures.concepts.vanity_persona import VanityPersona

# ToDo: Add more tests for VioletAgent methods


def test_generate_alternate_song_spec_mock(monkeypatch):
    """Test that generate_alternate_song_spec loads from mock in mock mode"""
    monkeypatch.setenv("MOCK_MODE", "true")
    agent = VioletAgent()

    # Create a minimal white_proposal that the method expects
    white_proposal = SongProposalIteration(
        iteration_id="test_white_001",
        bpm=120,
        tempo="4/4",
        key="C Major",
        rainbow_color="white",
        title="Test White Proposal",
        mood=["contemplative"],
        genres=["experimental"],
        concept="A test proposal for the Violet agent to respond to through dialectical pressure-testing and adversarial interview techniques.",
    )

    # Create interviewer persona
    interviewer = VanityPersona(first_name="Jane", last_name="Interviewer")

    state = VioletAgentState(
        thread_id="test_thread_001",
        white_proposal=white_proposal,
        interviewer_persona=interviewer,
        song_proposals=SongProposal(iterations=[white_proposal]),
    )

    result_state = agent.generate_alternate_song_spec(state)
    assert result_state.counter_proposal is not None
    assert isinstance(result_state.counter_proposal, SongProposalIteration)
    assert getattr(result_state.counter_proposal, "title", None)


def test_violet_agent_initialization():
    """Test that VioletAgent can be initialized"""
    agent = VioletAgent()
    assert agent is not None
    assert agent.llm is not None
    assert agent.gabe_corpus is not None
    assert agent.hitl_probability == 0.09
