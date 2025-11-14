import pytest
import yaml
from types import SimpleNamespace
import importlib

from app.agents.red_agent import RedAgent
from app.agents.states.red_agent_state import RedAgentState
from app.structures.manifests.song_proposal import SongProposalIteration

MODULE_PATH = "app.agents.red_agent"


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
    result_state = agent.generate_alternate_song_spec(state)
    assert result_state.counter_proposal is not None
    assert isinstance(result_state.counter_proposal, SongProposalIteration)
    assert getattr(result_state.counter_proposal, "title", None)


def test_generate_book_mock_mode(tmp_path, monkeypatch):
    ra = importlib.import_module(MODULE_PATH)
    monkeypatch.setattr(ra, "ChatAnthropic", DummyChatAnthropic)
    monkeypatch.setattr(ra, "ReactionBookArtifact", DummyReactionBookArtifact)
    monkeypatch.setenv("MOCK_MODE", "true")
    monkeypatch.setenv("AGENT_MOCK_DATA_PATH", str(tmp_path))
    mock_file = tmp_path / "red_book_artifact_mock.yml"
    write_yaml(mock_file, {"title": "Mock Book", "author": "Mock Author"})
    agent = ra.RedAgent()
    state = SimpleNamespace(artifacts=[], should_create_book=True)
    result = agent.generate_book(state)
    assert result is state
    assert result.should_create_book is False
    assert len(result.artifacts) == 1
    assert isinstance(result.artifacts[0], DummyReactionBookArtifact)
    assert result.artifacts[0].title == "Mock Book"


def test_generate_reaction_book_mock_mode(tmp_path, monkeypatch):
    ra = importlib.import_module(MODULE_PATH)
    monkeypatch.setattr(ra, "ChatAnthropic", DummyChatAnthropic)
    monkeypatch.setattr(ra, "ReactionBookArtifact", DummyReactionBookArtifact)
    monkeypatch.setenv("MOCK_MODE", "true")
    monkeypatch.setenv("AGENT_MOCK_DATA_PATH", str(tmp_path))
    mock_file = tmp_path / "red_reaction_book_data_mock.yml"
    write_yaml(mock_file, {"title": "Reaction Mock", "author": "React Author"})
    agent = ra.RedAgent()
    state = SimpleNamespace(
        artifacts=[], reaction_level=0, should_respond_with_reaction_book=True
    )
    result = agent.generate_reaction_book(state)
    assert result is state
    assert result.reaction_level == 1
    assert result.should_respond_with_reaction_book is False
    assert len(result.artifacts) == 1
    assert isinstance(result.artifacts[0], DummyReactionBookArtifact)
    assert result.artifacts[0].author == "React Author"


def test_write_reaction_book_pages_mock_mode(tmp_path, monkeypatch):
    ra = importlib.import_module(MODULE_PATH)
    monkeypatch.setattr(ra, "ChatAnthropic", DummyChatAnthropic)
    monkeypatch.setattr(ra, "TextChainArtifactFile", DummyTextChainArtifactFile)
    monkeypatch.setenv("MOCK_MODE", "true")
    monkeypatch.setenv("AGENT_MOCK_DATA_PATH", str(tmp_path))
    page1_file = tmp_path / "red_reaction_book_page_1_mock.yml"
    page2_file = tmp_path / "red_reaction_book_page_2_mock.yml"
    write_yaml(
        page1_file, {"artifact_name": "excerpt_1", "text_content": "Page 1 text"}
    )
    write_yaml(
        page2_file, {"artifact_name": "excerpt_2", "text_content": "Page 2 text"}
    )
    agent = ra.RedAgent()
    state = SimpleNamespace(
        artifacts=[], current_reaction_book={"author": "X"}, thread_id="tid-1"
    )
    result = agent.write_reaction_book_pages(state)
    assert result is state
    assert any(
        getattr(a, "artifact_name", None) == "excerpt_1" for a in state.artifacts
    )
    assert any(
        getattr(a, "artifact_name", None) == "excerpt_2" for a in state.artifacts
    )
    assert state.current_reaction_book is None
