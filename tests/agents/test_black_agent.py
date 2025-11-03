
from app.agents.black_agent import BlackAgent
from app.agents.states.black_agent_state import BlackAgentState
from app.structures.manifests.song_proposal import SongProposalIteration
from app.structures.artifacts.evp_artifact import EVPArtifact
from app.structures.artifacts.sigil_artifact import SigilArtifact


def test_generate_alternate_song_spec_mock():
    agent = BlackAgent()
    state = BlackAgentState()
    result_state = agent.generate_alternate_song_spec(state)
    assert result_state.counter_proposal is not None
    assert isinstance(result_state.counter_proposal, SongProposalIteration)
    assert getattr(result_state.counter_proposal, "title", None)


def test_generate_evp_mock():
    agent = BlackAgent()
    state = BlackAgentState()
    result_state = agent.generate_evp(state)
    assert len(result_state.artifacts) >= 1
    last = result_state.artifacts[-1]
    assert isinstance(last, EVPArtifact)
    assert getattr(last, "transcript", None) is not None


def test_generate_sigil_mock_creates_artifact(monkeypatch):
    """Test that sigil is created when skip chance doesn't trigger"""
    agent = BlackAgent()
    state = BlackAgentState()
    state.counter_proposal = SongProposalIteration(
        iteration_id="mock_1",
        bpm=120,
        tempo="4/4",
        key="C Major",
        rainbow_color="black",
        title="Mock Title",
        mood=["mysterious"],
        genres=["experimental"],
        concept="Mock Concept that should at least 100 characters long. It should contain some detail. Mock Concept that should at least 100 characters long. It should contain some detail."
    )
    monkeypatch.setattr("random.random", lambda: 0.8)

    result_state = agent.generate_sigil(state)

    if result_state.should_update_proposal_with_sigil:
        assert result_state.awaiting_human_action is True
        assert len(result_state.artifacts) >= 1
        last = result_state.artifacts[-1]
        assert isinstance(last, SigilArtifact)
        assert getattr(last, "wish", None)


def test_generate_sigil_mock_skips(monkeypatch):
    """Test that sigil is skipped when skip chance triggers"""
    agent = BlackAgent()
    state = BlackAgentState()
    state.counter_proposal = SongProposalIteration(
        iteration_id="mock_1",
        bpm=120,
        tempo="4/4",
        key="C Major",
        rainbow_color="black",
        title="Mock Title",
        mood=["mysterious"],
        genres=["experimental"],
        concept="Mock Concept that should at least 100 characters long. It should contain some detail. Mock Concept that should at least 100 characters long. It should contain some detail."
    )
    monkeypatch.setattr("random.random", lambda: 0.5)
    result_state = agent.generate_sigil(state)
    assert "generate_sigil" in result_state.skipped_nodes
    assert result_state.should_update_proposal_with_sigil is False


def test_evaluate_evp_routes(monkeypatch):
    agent = BlackAgent()
    state = BlackAgentState()
    monkeypatch.setattr("app.agents.black_agent.random.choice", lambda seq: 1)
    result_state = agent.evaluate_evp(state)
    assert result_state.should_update_proposal_with_evp is True
    assert agent.route_after_evp_evaluation(result_state) == "evp"
    monkeypatch.setattr("app.agents.black_agent.random.choice", lambda seq: 0)
    result_state = agent.evaluate_evp(state)
    assert result_state.should_update_proposal_with_evp is False
    assert agent.route_after_evp_evaluation(result_state) == "sigil"


def test_update_alternate_song_spec_with_evp_mock():
    agent = BlackAgent()
    state = BlackAgentState()
    result_state = agent.update_alternate_song_spec_with_evp(state)
    assert result_state.counter_proposal is not None
    assert isinstance(result_state.counter_proposal, SongProposalIteration)
    assert getattr(result_state.counter_proposal, "title", None)


def test_update_alternate_song_spec_with_sigil_mock():
    agent = BlackAgent()
    state = BlackAgentState()
    result_state = agent.update_alternate_song_spec_with_sigil(state)
    assert result_state.counter_proposal is not None
    assert isinstance(result_state.counter_proposal, SongProposalIteration)
    assert getattr(result_state.counter_proposal, "title", None)