import pytest
import yaml
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
    state.thread_id = "mock_thread_001"
    result_state = agent.generate_alternate_song_spec(state)
    assert result_state.counter_proposal is not None
    assert isinstance(result_state.counter_proposal, SongProposalIteration)
    assert getattr(result_state.counter_proposal, "title", None)
