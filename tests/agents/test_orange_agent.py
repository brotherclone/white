"""
Tests for OrangeAgent (Rows Bud - Sussex Mythologizer)

These tests use proper mocking of the OrangeAgent instance.
"""

from unittest.mock import patch, MagicMock
import yaml

from app.agents.orange_agent import OrangeAgent
from app.agents.states.orange_agent_state import OrangeAgentState
from app.structures.manifests.song_proposal import SongProposalIteration
from app.structures.artifacts.newspaper_artifact import NewspaperArtifact
from app.structures.artifacts.symbolic_object_artifact import SymbolicObjectArtifact


def create_mock_agent():
    """Create an OrangeAgent with mocked dependencies."""
    with patch(
        "app.reference.mcp.rows_bud.orange_corpus.get_corpus"
    ) as mock_get_corpus:
        mock_get_corpus.return_value = MagicMock()
        with patch.dict(
            "os.environ",
            {
                "ANTHROPIC_API_KEY": "test-key",
                "ORANGE_CORPUS_DIR": "/tmp/test",
            },
        ):
            agent = OrangeAgent()
    return agent


# =============================================================================
# Test: generate_alternate_song_spec (mock mode)
# =============================================================================


@patch.dict("os.environ", {"MOCK_MODE": "true"})
def test_generate_alternate_song_spec_mock(tmp_path, monkeypatch):
    """Test that generate_alternate_song_spec loads mock data correctly."""
    monkeypatch.setenv("AGENT_MOCK_DATA_PATH", str(tmp_path))
    monkeypatch.setenv("AGENT_WORK_PRODUCT_BASE_PATH", str(tmp_path))

    # Create mock counter proposal YAML with valid data
    mock_proposal = {
        "title": "Mock Orange Song",
        "rainbow_color": "O",
        "bpm": 182,
        "key": "E minor",
        "mood": ["mysterious", "dark"],  # Must be a list
        "genres": ["rock", "alternative"],
        "concept": "A test concept that is at least twenty-five characters long for validation",
        "iteration_id": "mock_orange_iteration_1",  # Must match pattern
    }
    (tmp_path / "orange_counter_proposal_mock.yml").write_text(
        yaml.safe_dump(mock_proposal)
    )

    agent = create_mock_agent()
    state = OrangeAgentState(thread_id="test-thread")
    result_state = agent.generate_alternate_song_spec(state)

    assert result_state.counter_proposal is not None
    assert isinstance(result_state.counter_proposal, SongProposalIteration)
    assert result_state.counter_proposal.title == "Mock Orange Song"


# =============================================================================
# Test: synthesize_base_story (mock mode)
# =============================================================================


@patch.dict("os.environ", {"MOCK_MODE": "true"})
def test_synthesize_base_story_mock(tmp_path, monkeypatch):
    """Test that synthesize_base_story creates a NewspaperArtifact in mock mode."""
    monkeypatch.setenv("AGENT_MOCK_DATA_PATH", str(tmp_path))
    monkeypatch.setenv("AGENT_WORK_PRODUCT_BASE_PATH", str(tmp_path))

    # Create mock story YAML - correct filename is orange_base_story_mock.yml
    mock_story = {
        "thread_id": "test-thread",
        "headline": "Mock Headline for Test",
        "date": "1988-06-15",
        "source": "Sussex County Herald",
        "text": "This is the mock story text.",
        "location": "Newton, NJ",
        "tags": ["test", "mock"],
    }
    (tmp_path / "orange_base_story_mock.yml").write_text(yaml.safe_dump(mock_story))

    agent = create_mock_agent()
    state = OrangeAgentState(thread_id="test-thread")
    result_state = agent.synthesize_base_story(state)

    assert result_state.synthesized_story is not None
    assert result_state.synthesized_story.headline == "Mock Headline for Test"


# =============================================================================
# Test: add_to_corpus - success path
# =============================================================================


@patch.dict("os.environ", {"MOCK_MODE": "false"})
def test_add_to_corpus_success(tmp_path, monkeypatch):
    """Test add_to_corpus successfully adds story to corpus."""
    monkeypatch.setenv("AGENT_WORK_PRODUCT_BASE_PATH", str(tmp_path))

    agent = create_mock_agent()
    agent.corpus.add_story = MagicMock(return_value=("story123", 0.85))

    state = OrangeAgentState(
        thread_id="test-thread",
        synthesized_story=NewspaperArtifact(
            thread_id="test-thread",
            headline="Local Band Sparks Midnight Mystery",
            date="1990-06-15",
            source="Sussex County Independent",
            text="A curious hum was reported after the late show.",
            location="Newton, NJ",
            tags=["music", "weird"],
        ),
    )

    result = agent.add_to_corpus(state)

    assert result.selected_story_id == "story123"
    agent.corpus.add_story.assert_called_once()
    call_kwargs = agent.corpus.add_story.call_args.kwargs
    assert call_kwargs["headline"] == "Local Band Sparks Midnight Mystery"


# =============================================================================
# Test: add_to_corpus - failure sets fallback ID
# =============================================================================


@patch.dict("os.environ", {"MOCK_MODE": "false"})
def test_add_to_corpus_failure_sets_fallback(tmp_path, monkeypatch):
    """Test that corpus failure results in fallback story ID."""
    monkeypatch.setenv("AGENT_WORK_PRODUCT_BASE_PATH", str(tmp_path))

    agent = create_mock_agent()
    agent.corpus.add_story = MagicMock(side_effect=RuntimeError("corpus unavailable"))

    state = OrangeAgentState(
        thread_id="test-thread",
        synthesized_story=NewspaperArtifact(
            thread_id="test-thread",
            headline="Strange Signals Near High School",
            date="1988-09-01",
            source="New Jersey Herald",
            text="Residents reported unexplained electronic sounds.",
            location="Sparta Township, NJ",
            tags=["unexplained"],
        ),
    )

    result = agent.add_to_corpus(state)

    assert isinstance(result.selected_story_id, str)
    assert result.selected_story_id.startswith("fallback_")


# =============================================================================
# Test: select_symbolic_object (mock mode)
# =============================================================================


@patch.dict("os.environ", {"MOCK_MODE": "true"})
def test_select_symbolic_object_mock(tmp_path, monkeypatch):
    """Test select_symbolic_object loads mock object selection."""
    monkeypatch.setenv("AGENT_MOCK_DATA_PATH", str(tmp_path))
    monkeypatch.setenv("AGENT_WORK_PRODUCT_BASE_PATH", str(tmp_path))

    # Use valid symbolic_object_category enum value
    mock_object = {
        "name": "Strange Compass",
        "symbolic_object_category": "liminal_objects",  # Valid enum value
        "description": "A compass that points toward mystery.",
    }
    (tmp_path / "orange_mock_object_selection.yml").write_text(
        yaml.safe_dump(mock_object)
    )

    agent = create_mock_agent()
    state = OrangeAgentState(
        thread_id="test-thread",
        selected_story_id="story-xyz",
        synthesized_story=NewspaperArtifact(
            thread_id="test-thread",
            text="Original story text.",
        ),
    )

    result = agent.select_symbolic_object(state)

    assert result.symbolic_object is not None
    assert result.symbolic_object.name == "Strange Compass"


# =============================================================================
# Test: insert_symbolic_object_node (mock mode)
# =============================================================================


@patch.dict("os.environ", {"MOCK_MODE": "true"})
def test_insert_symbolic_object_node_mock(tmp_path, monkeypatch):
    """Test insert_symbolic_object_node in mock mode."""
    monkeypatch.setenv("AGENT_MOCK_DATA_PATH", str(tmp_path))
    monkeypatch.setenv("AGENT_WORK_PRODUCT_BASE_PATH", str(tmp_path))

    # Create mock object YAML with valid enum
    mock_object = {
        "name": "Mock Compass",
        "symbolic_object_category": "liminal_objects",
        "description": "A mock object used by tests.",
    }
    (tmp_path / "orange_mock_object_selection.yml").write_text(
        yaml.safe_dump(mock_object)
    )

    agent = create_mock_agent()
    state = OrangeAgentState(
        thread_id="test-thread",
        selected_story_id="story-1",
        symbolic_object=SymbolicObjectArtifact(
            thread_id="test-thread",
            name="Mock Compass",
            symbolic_object_category="liminal_objects",  # Valid enum value
        ),
    )

    result = agent.insert_symbolic_object_node(state)

    assert result is not None
    assert result.symbolic_object.name == "Mock Compass"


# =============================================================================
# Test: gonzo_rewrite_node (mock mode)
# =============================================================================


@patch.dict("os.environ", {"MOCK_MODE": "true"})
def test_gonzo_rewrite_node_mock(tmp_path, monkeypatch):
    """Test gonzo_rewrite_node loads mock gonzo article."""
    monkeypatch.setenv("AGENT_MOCK_DATA_PATH", str(tmp_path))
    monkeypatch.setenv("AGENT_WORK_PRODUCT_BASE_PATH", str(tmp_path))

    mock_article = {
        "thread_id": "test-thread",
        "headline": "Mock Gonzo Headline",
        "date": "1988-04-01",
        "source": "Mock Source",
        "location": "Newton, NJ",
        "text": "This is the mock gonzo article.",
        "tags": ["mock"],
    }
    (tmp_path / "orange_gonzo_rewrite.yml").write_text(yaml.safe_dump(mock_article))

    agent = create_mock_agent()
    state = OrangeAgentState(
        thread_id="test-thread",
        gonzo_perspective="first-person",
        gonzo_intensity=2,
        synthesized_story=NewspaperArtifact(
            thread_id="test-thread",
            headline="Original",
            text="Original story",
        ),
    )

    result = agent.gonzo_rewrite_node(state)

    assert result.mythologized_story is not None
    assert result.mythologized_story.headline == "Mock Gonzo Headline"


# =============================================================================
# Test: gonzo_rewrite_node (non-mock mode with mocked LLM)
# =============================================================================


@patch.dict("os.environ", {"MOCK_MODE": "false"})
def test_gonzo_rewrite_node_non_mock(tmp_path, monkeypatch):
    """Test gonzo_rewrite_node with mocked LLM response."""
    monkeypatch.setenv("AGENT_WORK_PRODUCT_BASE_PATH", str(tmp_path))

    gonzo_text = "Gonzo rewritten content with vivid paranoia."

    # Mock LLM response
    mock_llm_response = MagicMock()
    mock_llm_response.content = gonzo_text

    mock_llm = MagicMock()
    mock_llm.invoke = MagicMock(return_value=mock_llm_response)

    # Mock corpus - return dict that matches expected format
    story_data = {
        "headline": "Local Oddities",
        "date": "1990-06-01",
        "location": "Newton, NJ",
        "text": "Original story text that will be rewritten.",
        "source": "Sussex Courier",
        "tags": ["weird"],
        "symbolic_object_desc": "a mysterious compass",
    }

    agent = create_mock_agent()
    agent.llm = mock_llm
    agent.corpus.get_story = MagicMock(return_value=story_data)
    agent.corpus.add_gonzo_rewrite = MagicMock()

    state = OrangeAgentState(
        thread_id="test-thread",
        selected_story_id="story-xyz",
        gonzo_perspective="first-person",
        gonzo_intensity=3,
        synthesized_story=NewspaperArtifact(
            thread_id="test-thread",
            headline="Local Oddities",
            text="Original story text that will be rewritten.",
        ),
    )

    result = agent.gonzo_rewrite_node(state)

    assert result.mythologized_story is not None
    # The LLM response gets used to create the new artifact
    assert result.mythologized_story.text == gonzo_text

    # Verify corpus was updated
    agent.corpus.add_gonzo_rewrite.assert_called_once()


# =============================================================================
# Test: Full workflow integration (mock mode)
# =============================================================================


@patch.dict("os.environ", {"MOCK_MODE": "true"})
def test_orange_agent_full_workflow_mock(tmp_path, monkeypatch):
    """Test the full Orange Agent workflow in mock mode."""
    monkeypatch.setenv("AGENT_MOCK_DATA_PATH", str(tmp_path))
    monkeypatch.setenv("AGENT_WORK_PRODUCT_BASE_PATH", str(tmp_path))
    monkeypatch.setenv("ORANGE_CORPUS_DIR", str(tmp_path))

    # Create all required mock files with valid data
    mock_story = {
        "thread_id": "test-thread",
        "headline": "Mock Story Headline",
        "date": "1988-06-15",
        "source": "Mock Source",
        "text": "Mock story text.",
        "location": "Newton, NJ",
        "tags": ["test"],
    }
    (tmp_path / "orange_base_story_mock.yml").write_text(yaml.safe_dump(mock_story))

    mock_object = {
        "name": "Mock Object",
        "symbolic_object_category": "liminal_objects",
        "description": "A mock symbolic object.",
    }
    (tmp_path / "orange_mock_object_selection.yml").write_text(
        yaml.safe_dump(mock_object)
    )

    mock_gonzo = {
        "thread_id": "test-thread",
        "headline": "Gonzo Mock Headline",
        "date": "1988-06-15",
        "source": "Gonzo Source",
        "text": "Gonzo mock text.",
        "location": "Newton, NJ",
        "tags": ["gonzo"],
    }
    (tmp_path / "orange_gonzo_rewrite.yml").write_text(yaml.safe_dump(mock_gonzo))

    mock_proposal = {
        "title": "Final Mock Song",
        "rainbow_color": "O",
        "bpm": 182,
        "key": "E minor",
        "mood": ["mysterious", "tense"],
        "genres": ["rock"],
        "concept": "A mock concept that is definitely more than twenty-five characters long",
        "iteration_id": "final_mock_song_1",
    }
    (tmp_path / "orange_counter_proposal_mock.yml").write_text(
        yaml.safe_dump(mock_proposal)
    )

    # Verify agent creation works
    agent = create_mock_agent()
    assert agent is not None
    assert agent.corpus is not None
