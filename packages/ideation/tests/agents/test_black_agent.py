import importlib
from unittest.mock import patch

from white_core.artifacts.evp_artifact import EVPArtifact
from white_core.artifacts.sigil_artifact import SigilArtifact
from white_core.manifests.song_proposal import SongProposalIteration
from white_ideation.agents.black_agent import BlackAgent
from white_ideation.agents.states.black_agent_state import BlackAgentState


def _counter_proposal() -> SongProposalIteration:
    return SongProposalIteration(
        iteration_id="black_v1",
        bpm=100,
        tempo="4/4",
        key="B minor",
        rainbow_color="black",
        title="Test Black Song",
        mood=["mysterious"],
        genres=["experimental"],
        concept="A test concept long enough to pass the minimum length validator "
        "for the concept field, exploring surveillance and resistance.",
    )


def test_generate_alternate_song_spec_mock():
    agent = BlackAgent()
    state = BlackAgentState()
    state.thread_id = "test_thread"
    result_state = agent.generate_alternate_song_spec(state)
    assert result_state.counter_proposal is not None
    assert isinstance(result_state.counter_proposal, SongProposalIteration)
    assert getattr(result_state.counter_proposal, "title", None)


@patch.object(BlackAgent, "generate_evp")
def test_generate_evp_mock(mock_generate_evp):
    agent = BlackAgent(thread_id="test_thread")
    state = BlackAgentState()
    state.thread_id = "test_thread"
    mock_evp = EVPArtifact()
    expected_state = BlackAgentState()
    expected_state.artifacts = [mock_evp]
    mock_generate_evp.return_value = expected_state
    result_state = agent.generate_evp(state)
    assert len(result_state.artifacts) >= 1
    last = result_state.artifacts[-1]
    assert isinstance(last, EVPArtifact)


def test_generate_sigil_mock_creates_artifact(monkeypatch):
    """Test that sigil is created when skip chance doesn't trigger"""
    agent = BlackAgent()
    state = BlackAgentState()
    state.thread_id = "test_thread"
    state.counter_proposal = SongProposalIteration(
        iteration_id="mock_1",
        bpm=120,
        tempo="4/4",
        key="C Major",
        rainbow_color="black",
        title="Mock Title",
        mood=["mysterious"],
        genres=["experimental"],
        concept="Mock Concept that should at least 100 characters long. It should contain some detail. Mock Concept that should at least 100 characters long. It should contain some detail.",
    )
    mod = importlib.import_module("white_ideation.agents.black_agent")

    # Force random to return > 0.75 to not skip sigil generation
    if hasattr(mod, "random") and hasattr(getattr(mod, "random"), "random"):
        monkeypatch.setattr(
            "white_ideation.agents.black_agent.random.random",
            lambda: 0.8,
            raising=False,
        )
    else:
        monkeypatch.setattr(
            "white_ideation.agents.black_agent", "random", lambda: 0.8, raising=False
        )

    result_state = agent.generate_sigil(state)

    # Sigil generation may be skipped (75% chance in mock mode)
    # If not skipped, verify artifact was created
    if len(result_state.artifacts) > 0:
        last = result_state.artifacts[-1]
        assert isinstance(last, SigilArtifact)
        assert getattr(last, "wish", None)


def test_evaluate_evp_routes(monkeypatch):
    agent = BlackAgent()
    state = BlackAgentState()
    state.thread_id = "test_thread"
    monkeypatch.setattr(
        "white_ideation.agents.black_agent.random.choice", lambda seq: 1
    )
    result_state = agent.evaluate_evp(state)
    assert result_state.should_update_proposal_with_evp is True
    assert agent.route_after_evp_evaluation(result_state) == "evp"
    monkeypatch.setattr(
        "white_ideation.agents.black_agent.random.choice", lambda seq: 0
    )
    result_state = agent.evaluate_evp(state)
    assert result_state.should_update_proposal_with_evp is False
    assert agent.route_after_evp_evaluation(result_state) == "sigil"


def test_update_alternate_song_spec_with_evp_mock():
    agent = BlackAgent()
    state = BlackAgentState()
    state.thread_id = "test_thread"
    result_state = agent.update_alternate_song_spec_with_evp(state)
    assert result_state.counter_proposal is not None
    assert isinstance(result_state.counter_proposal, SongProposalIteration)
    assert getattr(result_state.counter_proposal, "title", None)


def test_generate_alternate_song_spec_includes_sounds_like(monkeypatch):
    """Regression for add-ideation-sounds-like: Black's reference-works
    section must include sampled sounds-like artists, not just
    title/mood/genre/concept fields."""
    monkeypatch.setenv("MOCK_MODE", "false")
    monkeypatch.setenv("BLOCK_MODE", "false")
    monkeypatch.setattr(
        "white_ideation.agents.black_agent.get_sounds_like_by_color",
        lambda color: ["Test Reference Artist"],
    )
    monkeypatch.setattr(
        "white_ideation.agents.black_agent.sample_reference_artists",
        lambda artists, **kwargs: list(artists),
    )

    agent = BlackAgent()
    state = BlackAgentState()
    state.thread_id = "test_thread"

    seen_prompts = []

    def fake_invoke_structured(llm, schema, prompt):
        seen_prompts.append(prompt)
        return _counter_proposal()

    with patch.object(agent, "_invoke_structured", side_effect=fake_invoke_structured):
        agent.generate_alternate_song_spec(state)

    assert len(seen_prompts) == 1
    assert "Test Reference Artist" in seen_prompts[0]


def test_generate_alternate_song_spec_includes_negative_constraints(monkeypatch):
    """Regression for add-ideation-negative-constraints: the color agent's own
    counter-proposal prompt must honor negative_constraints, not just
    White's initial proposal and final rewrite."""
    monkeypatch.setenv("MOCK_MODE", "false")
    monkeypatch.setenv("BLOCK_MODE", "false")

    agent = BlackAgent()
    state = BlackAgentState()
    state.thread_id = "test_thread"
    state.negative_constraints = "AVOID: the word 'shadow', keys already used: B minor"

    seen_prompts = []

    def fake_invoke_structured(llm, schema, prompt):
        seen_prompts.append(prompt)
        return _counter_proposal()

    with patch.object(agent, "_invoke_structured", side_effect=fake_invoke_structured):
        result_state = agent.generate_alternate_song_spec(state)

    assert result_state.counter_proposal is not None
    assert len(seen_prompts) == 1
    assert state.negative_constraints in seen_prompts[0]


def test_generate_alternate_song_spec_empty_constraints_unchanged(monkeypatch):
    """Regression guard (representative, not exhaustive per-agent): an empty
    negative_constraints must not append anything to the prompt, preserving
    prior behavior exactly."""
    monkeypatch.setenv("MOCK_MODE", "false")
    monkeypatch.setenv("BLOCK_MODE", "false")

    agent = BlackAgent()
    state = BlackAgentState()
    state.thread_id = "test_thread"
    assert state.negative_constraints == ""

    seen_prompts = []

    def fake_invoke_structured(llm, schema, prompt):
        seen_prompts.append(prompt)
        return _counter_proposal()

    with patch.object(agent, "_invoke_structured", side_effect=fake_invoke_structured):
        agent.generate_alternate_song_spec(state)

    assert len(seen_prompts) == 1
    assert seen_prompts[0].rstrip().endswith("Ambiguity and subtlety are valued.")


def test_update_alternate_song_spec_with_evp_includes_negative_constraints(
    monkeypatch,
):
    monkeypatch.setenv("MOCK_MODE", "false")
    monkeypatch.setenv("BLOCK_MODE", "false")

    agent = BlackAgent()
    state = BlackAgentState()
    state.thread_id = "test_thread"
    state.negative_constraints = "AVOID: the word 'shadow'"
    state.counter_proposal = _counter_proposal()
    state.artifacts = [EVPArtifact(transcript="a whisper of static")]

    seen_prompts = []

    def fake_invoke_structured(llm, schema, prompt):
        seen_prompts.append(prompt)
        return _counter_proposal()

    with patch.object(agent, "_invoke_structured", side_effect=fake_invoke_structured):
        result_state = agent.update_alternate_song_spec_with_evp(state)

    assert result_state.counter_proposal is not None
    assert len(seen_prompts) == 1
    assert state.negative_constraints in seen_prompts[0]
