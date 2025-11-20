import yaml

from types import SimpleNamespace
from collections.abc import Mapping

from app.agents.orange_agent import OrangeAgent
from app.agents.states.orange_agent_state import OrangeAgentState
from app.structures.manifests.song_proposal import SongProposalIteration

# Dupe the agent
import app.agents.orange_agent as orange_mod


def test_generate_alternate_song_spec_mock():
    agent = OrangeAgent()
    state = OrangeAgentState()
    result_state = agent.generate_alternate_song_spec(state)
    assert result_state.counter_proposal is not None
    assert isinstance(result_state.counter_proposal, SongProposalIteration)
    assert getattr(result_state.counter_proposal, "title", None)


def test_synthesize_base_story_mock(tmp_path, monkeypatch):
    mock_data = {
        "thread_id": "test-thread",
        "headline": "Mock Headline",
        "date": "1985-07-14",
        "source": "Sussex County Independent",
        "location": "Newton, NJ",
        "text": "This is a mock article for testing.",
        "tags": ["mock", "test"],
    }
    mock_file = tmp_path / "orange_base_story_mock.yml"
    mock_file.write_text(yaml.safe_dump(mock_data))
    monkeypatch.setenv("MOCK_MODE", "true")
    monkeypatch.setenv("AGENT_MOCK_DATA_PATH", str(tmp_path))
    dummy_self = SimpleNamespace()
    state = SimpleNamespace(artifacts=[])
    OrangeAgent.synthesize_base_story(dummy_self, state)
    assert hasattr(state, "synthesized_story")
    assert state.synthesized_story.headline == mock_data["headline"]
    assert state.synthesized_story.date == mock_data["date"]
    assert len(state.artifacts) == 1
    assert state.artifacts[0].text == mock_data["text"]


def test_add_to_corpus_success(monkeypatch):
    monkeypatch.setenv("MOCK_MODE", "false")

    captured = {}

    def fake_add_story(**kwargs):
        captured.update(kwargs)
        return ("story123", 0.85)

    fake_corpus = SimpleNamespace(add_story=fake_add_story)
    dummy_self = SimpleNamespace(corpus=fake_corpus)

    from app.structures.artifacts.newspaper_artifact import NewspaperArtifact

    state = SimpleNamespace(
        synthesized_story=NewspaperArtifact(
            thread_id="test-thread",
            headline="Local Band Sparks Midnight Mystery",
            date="1990-06-15",
            source="Sussex County Independent",
            text="A curious hum was reported after the late show.",
            location="Newton, NJ",
            tags=["music", "weird"],
        ),
        selected_story_id=None,
    )

    result = OrangeAgent.add_to_corpus(dummy_self, state)

    assert result.selected_story_id == "story123"
    assert captured["headline"] == state.synthesized_story.headline
    assert captured["text"] == state.synthesized_story.text


def test_add_to_corpus_failure_sets_fallback(monkeypatch):
    monkeypatch.setenv("MOCK_MODE", "false")

    def failing_add_story(**kwargs):
        raise RuntimeError("corpus unavailable")

    fake_corpus = SimpleNamespace(add_story=failing_add_story)
    dummy_self = SimpleNamespace(corpus=fake_corpus)

    from app.structures.artifacts.newspaper_artifact import NewspaperArtifact

    state = SimpleNamespace(
        synthesized_story=NewspaperArtifact(
            thread_id="test-thread",
            headline="Strange Signals Near High School",
            date="1988-09-01",
            source="New Jersey Herald",
            text="Residents reported unexplained electronic sounds.",
            location="Sparta Township, NJ",
            tags=["unexplained"],
        ),
        selected_story_id=None,
    )

    result = OrangeAgent.add_to_corpus(dummy_self, state)

    assert isinstance(result.selected_story_id, str)
    assert result.selected_story_id.startswith("fallback_")


def test_select_symbolic_object_non_mock(monkeypatch):
    monkeypatch.setenv("MOCK_MODE", "false")
    updated_text = "This is the updated story text with the Strange Compass inserted."

    class FakeContent:
        def __init__(self, text):
            self.text = text

    class FakeResponse:
        def __init__(self, text):
            self.content = [FakeContent(text)]

    captured = {}

    def fake_get_story(story_id):
        return {
            "headline": "Local Band Mystery",
            "date": "1989-05-12",
            "location": "Newton, NJ",
            "text": "Original story text.",
            "symbolic_object_desc": "an old compass",
        }

    def fake_insert(story_id, category, description, updated_text):
        captured.update(
            {
                "story_id": story_id,
                "category": category,
                "description": description,
                "updated_text": updated_text,
            }
        )

    fake_corpus = SimpleNamespace(
        get_story=fake_get_story, insert_symbolic_object=fake_insert
    )

    fake_messages = SimpleNamespace(create=lambda **kwargs: FakeResponse(updated_text))
    fake_anthropic_client = SimpleNamespace(messages=fake_messages)

    dummy_self = SimpleNamespace(
        corpus=fake_corpus, anthropic_client=fake_anthropic_client
    )

    from app.structures.artifacts.newspaper_artifact import NewspaperArtifact

    state = SimpleNamespace(
        selected_story_id="story-xyz",
        synthesized_story=NewspaperArtifact(
            thread_id="test-thread", text="Original story text."
        ),
        symbolic_object=SimpleNamespace(
            name="Strange Compass", symbolic_object_category="instrument"
        ),
    )

    result = OrangeAgent.select_symbolic_object(dummy_self, state)

    assert result.synthesized_story.text == updated_text
    assert captured["story_id"] == state.selected_story_id
    assert captured["category"] == state.symbolic_object.symbolic_object_category
    assert captured["description"] == state.symbolic_object.name
    assert captured["updated_text"] == updated_text


def test_insert_symbolic_object_node_mock(monkeypatch, tmp_path):
    monkeypatch.setenv("MOCK_MODE", "true")
    monkeypatch.setenv("AGENT_MOCK_DATA_PATH", str(tmp_path))

    # create a minimal mock yaml (method just prints and returns)
    mock_data = {
        "name": "Mock Compass",
        "symbolic_object_category": "instrument",
        "description": "A mock object used by tests.",
    }
    (tmp_path / "orange_mock_object_selection.yml").write_text(
        yaml.safe_dump(mock_data)
    )

    state = SimpleNamespace(
        symbolic_object=SimpleNamespace(
            name="Mock Compass", symbolic_object_category="instrument"
        )
    )
    state.state = state  # code prints via state.state
    state.selected_story_id = "story-1"
    state.artifacts = []

    # call staticmethod without an instance
    result = OrangeAgent.insert_symbolic_object_node(state)

    assert result is state
    assert hasattr(state, "symbolic_object")
    assert state.symbolic_object.name == "Mock Compass"


def test_insert_symbolic_object_node_non_mock_calls_mcp(monkeypatch):
    monkeypatch.setenv("MOCK_MODE", "false")

    captured = {}

    def fake_insert_symbolic_object(story_id, object_category, custom_object):
        captured.update(
            {
                "story_id": story_id,
                "object_category": object_category,
                "custom_object": custom_object,
            }
        )

    # monkeypatch the imported mcp function used by the module
    monkeypatch.setattr(
        orange_mod, "insert_symbolic_object", fake_insert_symbolic_object
    )

    # build a state with nested reference used in prints
    obj = SimpleNamespace(name="Strange Compass", symbolic_object_category="instrument")
    state = SimpleNamespace(selected_story_id="story-abc", symbolic_object=obj)
    state.state = state
    state.artifacts = []

    result = OrangeAgent.insert_symbolic_object_node(state)

    assert result is state
    assert captured["story_id"] == "story-abc"
    assert captured["object_category"] == "instrument"
    assert captured["custom_object"] == "Strange Compass"


def test_gonzo_rewrite_node_mock(monkeypatch, tmp_path):
    monkeypatch.setenv("MOCK_MODE", "true")
    monkeypatch.setenv("AGENT_MOCK_DATA_PATH", str(tmp_path))

    # write a mock gonzo YAML
    mock_article = {
        "thread_id": "t1",
        "headline": "Mock Gonzo Headline",
        "date": "1988-04-01",
        "source": "Mock Source",
        "location": "Newton, NJ",
        "text": "This is the mock gonzo article.",
        "tags": ["mock"],
    }
    (tmp_path / "orange_gonzo_rewrite.yml").write_text(yaml.safe_dump(mock_article))

    state = SimpleNamespace(thread_id="t1", artifacts=[])
    state.gonzo_perspective = "first-person"
    state.gonzo_intensity = 2

    dummy_self = SimpleNamespace()
    result = OrangeAgent.gonzo_rewrite_node(dummy_self, state)

    assert result is state
    assert hasattr(state, "mythologized_story")
    # mythologized_story should have headline from the YAML
    assert getattr(state.mythologized_story, "headline", None) == "Mock Gonzo Headline"


def test_gonzo_rewrite_node_non_mock(monkeypatch):
    monkeypatch.setenv("MOCK_MODE", "false")

    # fake anthropic response
    gonzo_text = "Gonzo rewritten content with vivid paranoia."

    class FakeContent:
        def __init__(self, text):
            self.text = text

    class FakeResponse:
        def __init__(self, text):
            self.content = [FakeContent(text)]

    fake_messages = SimpleNamespace(create=lambda **kwargs: FakeResponse(gonzo_text))
    fake_anthropic_client = SimpleNamespace(messages=fake_messages)

    captured = {}

    def fake_add_gonzo_rewrite(story_id, gonzo_text, perspective, intensity):
        captured.update(
            {
                "story_id": story_id,
                "gonzo_text": gonzo_text,
                "perspective": perspective,
                "intensity": intensity,
            }
        )

    # Create a mapping that returns 'text' via __getitem__ but does not include 'text' in its keys()
    class FakeStory(Mapping):
        def __init__(self, data):
            self._data = data

        def __getitem__(self, key):
            return self._data[key]

        def __iter__(self):
            # exclude 'text' so that `**story` won't pass it as a kwarg
            return (k for k in self._data.keys() if k != "text")

        def __len__(self):
            return len([k for k in self._data.keys() if k != "text"])

    story_data = {
        "thread_id": "t-100",
        "headline": "Local Oddities",
        "date": "1990-06-01",
        "location": "Newton, NJ",
        "text": "Original story text that will be rewritten.",
        "source": "Sussex Courier",
        "tags": ["weird"],
    }
    fake_story = FakeStory(story_data)

    fake_corpus = SimpleNamespace(
        get_story=lambda sid: fake_story, add_gonzo_rewrite=fake_add_gonzo_rewrite
    )

    class FakeNewspaper:
        def __init__(self, **kwargs):
            # the final text will be provided as 'text' kw
            self._kwargs = kwargs
            self.text = kwargs.get("text", "")

        def get_text_content(self):
            return self.text

    monkeypatch.setattr(orange_mod, "NewspaperArtifact", FakeNewspaper)

    dummy_self = SimpleNamespace(
        corpus=fake_corpus, anthropic_client=fake_anthropic_client
    )

    state = SimpleNamespace(
        thread_id="t-100",
        selected_story_id="story-xyz",
        gonzo_perspective="first-person",
        gonzo_intensity=3,
        synthesized_story={
            "headline": "Local Oddities",
            "text": "Original story text that will be rewritten.",
        },
        artifacts=[],
    )

    result = OrangeAgent.gonzo_rewrite_node(dummy_self, state)

    assert result is state
    assert captured["story_id"] == state.selected_story_id
    assert captured["gonzo_text"] == gonzo_text
    assert captured["perspective"] == state.gonzo_perspective
    assert captured["intensity"] == state.gonzo_intensity

    assert hasattr(state, "mythologized_story")
    assert getattr(state.mythologized_story, "text", None) == gonzo_text

    assert len(state.artifacts) >= 1
    assert state.artifacts[0] == state.synthesized_story
