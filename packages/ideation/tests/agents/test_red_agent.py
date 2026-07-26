import importlib
from unittest.mock import patch

import pytest
import yaml

from white_core.manifests.song_proposal import SongProposal, SongProposalIteration
from white_ideation.agents.red_agent import RedAgent
from white_ideation.agents.states.red_agent_state import RedAgentState

MODULE_PATH = "white_ideation.agents.red_agent"


class DummyReactionBookArtifact:
    def __init__(self, **data):
        self.__dict__.update(data)


class DummyTextChainArtifactFile:
    def __init__(self, **data):
        self.__dict__.update(data)

    def get_artifact_path(self):
        return getattr(self, "artifact_path", "/tmp/mock.md")


class DummyChatAnthropic:
    def __init__(self, *args, **kwargs):
        pass


@pytest.fixture(autouse=True)
def reload_module(monkeypatch):
    # Ensure fresh import for each test so our monkeypatches apply cleanly
    if MODULE_PATH in importlib.sys.modules:
        importlib.reload(importlib.import_module(MODULE_PATH))
    yield
    if MODULE_PATH in importlib.sys.modules:
        importlib.reload(importlib.import_module(MODULE_PATH))


def write_yaml(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(data))


def test_generate_alternate_song_spec_mock():
    agent = RedAgent()
    state = RedAgentState()
    state.thread_id = "mock_thread_001"
    result_state = agent.generate_alternate_song_spec(state)
    assert result_state.counter_proposal is not None
    assert isinstance(result_state.counter_proposal, SongProposalIteration)
    assert getattr(result_state.counter_proposal, "title", None)


def test_write_reaction_book_pages_no_reaction_book_does_not_crash(monkeypatch):
    """Regression: generate_reaction_book sets current_reaction_book to None
    on failure. write_reaction_book_pages used to build its prompt via
    BookMaker.format_card_catalog(state.current_reaction_book), which
    dereferences the object with no guard — an uncaught AttributeError
    before the method's own try/except even started."""
    monkeypatch.setenv("MOCK_MODE", "false")
    monkeypatch.setenv("BLOCK_MODE", "false")

    agent = RedAgent()
    state = RedAgentState()
    state.thread_id = "test-thread"
    state.current_reaction_book = None

    result_state = agent.write_reaction_book_pages(state)

    assert result_state is not None
    assert result_state.current_reaction_book is None


def test_generate_alternate_song_spec_sets_counter_proposal_from_pydantic_result(
    monkeypatch,
):
    """Regression: _invoke_structured's normal return shape is a real
    SongProposalIteration instance (confirmed via PydanticToolsParser), not
    a dict. The code only handled the dict case, so on every real,
    successful LLM call state.counter_proposal was silently never set and
    nothing was appended to song_proposals.iterations — Red Agent's
    contribution vanished from every run without ever raising an error."""
    monkeypatch.setenv("MOCK_MODE", "false")
    monkeypatch.setenv("BLOCK_MODE", "false")

    agent = RedAgent()
    state = RedAgentState()
    state.thread_id = "test-thread"
    state.song_proposals = SongProposal(iterations=[])

    counter_proposal = SongProposalIteration(
        iteration_id="red_v1",
        bpm=120,
        tempo="4/4",
        key="D minor",
        rainbow_color="red",
        title="Test Red Song",
        mood=["literary"],
        genres=["chamber pop"],
        concept="A test concept long enough to pass the minimum length validator "
        "for the concept field, exploring literary archaeology.",
    )
    with patch.object(agent, "_invoke_structured", return_value=counter_proposal):
        result_state = agent.generate_alternate_song_spec(state)

    assert result_state.counter_proposal is counter_proposal
    assert counter_proposal in result_state.song_proposals.iterations


def test_generate_alternate_song_spec_includes_sounds_like(monkeypatch):
    """Regression for add-ideation-sounds-like: Red's reference-works
    section must include sampled sounds-like artists."""
    monkeypatch.setenv("MOCK_MODE", "false")
    monkeypatch.setenv("BLOCK_MODE", "false")
    monkeypatch.setattr(
        "white_ideation.agents.red_agent.get_sounds_like_by_color",
        lambda color: ["Test Reference Artist"],
    )
    monkeypatch.setattr(
        "white_ideation.agents.red_agent.sample_reference_artists",
        lambda artists, **kwargs: list(artists),
    )

    agent = RedAgent()
    state = RedAgentState()
    state.thread_id = "test-thread"
    state.song_proposals = SongProposal(iterations=[])

    counter_proposal = SongProposalIteration(
        iteration_id="red_v1",
        bpm=120,
        tempo="4/4",
        key="D minor",
        rainbow_color="red",
        title="Test Red Song",
        mood=["literary"],
        genres=["chamber pop"],
        concept="A test concept long enough to pass the minimum length validator "
        "for the concept field, exploring literary archaeology.",
    )
    seen_prompts = []

    def fake_invoke_structured(llm, schema, prompt):
        seen_prompts.append(prompt)
        return counter_proposal

    with patch.object(agent, "_invoke_structured", side_effect=fake_invoke_structured):
        agent.generate_alternate_song_spec(state)

    assert len(seen_prompts) == 1
    assert "Test Reference Artist" in seen_prompts[0]


def test_generate_alternate_song_spec_includes_negative_constraints(monkeypatch):
    """Regression for add-ideation-negative-constraints: Red's own
    counter-proposal prompt must honor negative_constraints, not just
    White's initial proposal and final rewrite."""
    monkeypatch.setenv("MOCK_MODE", "false")
    monkeypatch.setenv("BLOCK_MODE", "false")

    agent = RedAgent()
    state = RedAgentState()
    state.thread_id = "test-thread"
    state.song_proposals = SongProposal(iterations=[])
    state.negative_constraints = "AVOID: the word 'book', keys already used: C Major"

    counter_proposal = SongProposalIteration(
        iteration_id="red_v1",
        bpm=120,
        tempo="4/4",
        key="D minor",
        rainbow_color="red",
        title="Test Red Song",
        mood=["literary"],
        genres=["chamber pop"],
        concept="A test concept long enough to pass the minimum length validator "
        "for the concept field, exploring literary archaeology.",
    )
    seen_prompts = []

    def fake_invoke_structured(llm, schema, prompt):
        seen_prompts.append(prompt)
        return counter_proposal

    with patch.object(agent, "_invoke_structured", side_effect=fake_invoke_structured):
        result_state = agent.generate_alternate_song_spec(state)

    assert result_state.counter_proposal is not None
    assert len(seen_prompts) == 1
    assert state.negative_constraints in seen_prompts[0]
