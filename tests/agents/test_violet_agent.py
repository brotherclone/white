import pytest

from app.agents.states.violet_agent_state import VioletAgentState
from app.agents.violet_agent import VioletAgent
from app.agents.states.white_agent_state import MainAgentState
from app.structures.manifests.song_proposal import SongProposal, SongProposalIteration
from app.structures.concepts.vanity_persona import VanityPersona
from app.structures.concepts.vanity_interview_question import VanityInterviewQuestion
from app.structures.concepts.vanity_interview_response import VanityInterviewResponse
from app.structures.artifacts.circle_jerk_interview_artifact import (
    CircleJerkInterviewArtifact,
)


@pytest.fixture
def mock_env_vars(monkeypatch):
    """Set up mock environment variables for testing."""
    monkeypatch.setenv("MOCK_MODE", "true")
    monkeypatch.setenv("BLOCK_MODE", "false")
    monkeypatch.setenv("AGENT_MOCK_DATA_PATH", "tests/mocks")
    monkeypatch.setenv("AGENT_WORK_PRODUCT_BASE_PATH", "/tmp/test_artifacts")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")


@pytest.fixture
def violet_agent():
    """Create a VioletAgent instance for testing."""
    return VioletAgent()


@pytest.fixture
def violet_agent_state():
    """Create a VioletAgentState instance for testing."""
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

    return VioletAgentState(
        thread_id="mock_thread_001",
        white_proposal=white_proposal,
        song_proposals=SongProposal(iterations=[white_proposal]),
    )


class TestVioletAgentInitialization:
    """Tests for VioletAgent initialization."""

    def test_violet_agent_instantiation(self):
        """Test that VioletAgent can be instantiated."""
        agent = VioletAgent()
        assert agent is not None
        assert isinstance(agent, VioletAgent)

    def test_violet_agent_has_llm(self):
        """Test that VioletAgent has an LLM configured."""
        agent = VioletAgent()
        assert hasattr(agent, "llm")
        assert agent.llm is not None

    def test_violet_agent_has_settings(self):
        """Test that VioletAgent has settings configured."""
        agent = VioletAgent()
        assert hasattr(agent, "settings")
        assert agent.settings is not None

    def test_violet_agent_has_gabe_corpus(self):
        """Test that VioletAgent has gabe_corpus loaded."""
        agent = VioletAgent()
        assert hasattr(agent, "gabe_corpus")
        assert agent.gabe_corpus is not None
        assert isinstance(agent.gabe_corpus, str)


class TestVioletAgentGraphCreation:
    """Tests for VioletAgent graph creation."""

    def test_create_graph(self, violet_agent):
        """Test that create_graph returns a valid StateGraph."""
        graph = violet_agent.create_graph()
        assert graph is not None

    def test_graph_has_all_nodes(self, violet_agent):
        """Test that the graph has all required nodes."""
        graph = violet_agent.create_graph()
        compiled = graph.compile()

        expected_nodes = [
            "select_persona",
            "generate_questions",
            "simulated_interview",
            "synthesize_interview",
            "generate_alternate_song_spec",
        ]

        for node in expected_nodes:
            assert node in compiled.nodes

    def test_graph_workflow_structure(self, violet_agent):
        """Test that the graph has the correct workflow structure."""
        graph = violet_agent.create_graph()
        compiled = graph.compile()

        assert compiled is not None
        assert len(compiled.nodes) > 0


class TestSelectPersona:
    """Tests for select_persona node."""

    def test_select_persona_creates_new_persona(self, violet_agent_state):
        """Test that select_persona creates a new persona when none exists."""
        assert violet_agent_state.interviewer_persona is None

        result = VioletAgent.select_persona(violet_agent_state)

        assert result is not None
        assert isinstance(result, VioletAgentState)
        assert result.interviewer_persona is not None
        assert isinstance(result.interviewer_persona, VanityPersona)

    def test_select_persona_reuses_existing_persona(self, violet_agent_state):
        """Test that select_persona reuses existing persona."""
        existing_persona = VanityPersona(first_name="Jane", last_name="Interviewer")
        violet_agent_state.interviewer_persona = existing_persona

        result = VioletAgent.select_persona(violet_agent_state)

        assert result.interviewer_persona == existing_persona
        assert result.interviewer_persona.first_name == "Jane"


class TestGenerateQuestions:
    """Tests for generate_questions node."""

    def test_generate_questions_mock_mode(
        self, violet_agent, violet_agent_state, mock_env_vars
    ):
        """Test generate_questions in mock mode."""
        violet_agent_state.interviewer_persona = VanityPersona()

        result = violet_agent.generate_questions(violet_agent_state)

        assert result is not None
        assert isinstance(result, VioletAgentState)
        assert result.interview_questions is not None
        assert isinstance(result.interview_questions, list)
        assert len(result.interview_questions) > 0
        assert all(
            isinstance(q, VanityInterviewQuestion) for q in result.interview_questions
        )

    def test_generate_questions_returns_state(
        self, violet_agent, violet_agent_state, mock_env_vars
    ):
        """Test that generate_questions returns the modified state."""
        violet_agent_state.interviewer_persona = VanityPersona()
        original_thread_id = violet_agent_state.thread_id

        result = violet_agent.generate_questions(violet_agent_state)

        assert result.thread_id == original_thread_id
        assert isinstance(result, VioletAgentState)


class TestSimulatedInterview:
    """Tests for simulated_interview node."""

    def test_simulated_interview_mock_mode(
        self, violet_agent, violet_agent_state, mock_env_vars
    ):
        """Test simulated_interview in mock mode."""
        violet_agent_state.interviewer_persona = VanityPersona()
        violet_agent_state.interview_questions = [
            VanityInterviewQuestion(number=1, question="Test question 1?"),
            VanityInterviewQuestion(number=2, question="Test question 2?"),
            VanityInterviewQuestion(number=3, question="Test question 3?"),
        ]

        result = violet_agent.simulated_interview(violet_agent_state)

        assert result is not None
        assert isinstance(result, VioletAgentState)
        assert result.interview_responses is not None
        assert isinstance(result.interview_responses, list)
        assert all(
            isinstance(r, VanityInterviewResponse) for r in result.interview_responses
        )

    def test_simulated_interview_returns_state(
        self, violet_agent, violet_agent_state, mock_env_vars
    ):
        """Test that simulated_interview returns the modified state."""
        violet_agent_state.interviewer_persona = VanityPersona()
        violet_agent_state.interview_questions = [
            VanityInterviewQuestion(number=1, question="Test?"),
        ]
        original_thread_id = violet_agent_state.thread_id

        result = violet_agent.simulated_interview(violet_agent_state)

        assert result.thread_id == original_thread_id
        assert isinstance(result, VioletAgentState)


class TestSynthesizeInterview:
    """Tests for synthesize_interview node."""

    def test_synthesize_interview_creates_artifact(
        self, violet_agent_state, mock_env_vars
    ):
        """Test that synthesize_interview creates an interview artifact."""
        violet_agent_state.interviewer_persona = VanityPersona()
        violet_agent_state.interview_questions = [
            VanityInterviewQuestion(number=1, question="Test question?"),
        ]
        violet_agent_state.interview_responses = [
            VanityInterviewResponse(question_number=1, response="Test response."),
        ]

        result = VioletAgent.synthesize_interview(violet_agent_state)

        assert result is not None
        assert isinstance(result, VioletAgentState)
        assert result.circle_jerk_interview is not None
        assert isinstance(result.circle_jerk_interview, CircleJerkInterviewArtifact)

    def test_synthesize_interview_adds_to_artifacts(
        self, violet_agent_state, mock_env_vars
    ):
        """Test that synthesize_interview adds artifact to artifacts list."""
        violet_agent_state.interviewer_persona = VanityPersona()
        violet_agent_state.interview_questions = [
            VanityInterviewQuestion(number=1, question="Test question?"),
        ]
        violet_agent_state.interview_responses = [
            VanityInterviewResponse(question_number=1, response="Test response."),
        ]

        result = VioletAgent.synthesize_interview(violet_agent_state)

        assert len(result.artifacts) > 0
        assert result.circle_jerk_interview in result.artifacts


class TestGenerateAlternateSongSpec:
    """Tests for generate_alternate_song_spec node."""

    def test_generate_alternate_song_spec_mock(
        self, violet_agent, violet_agent_state, mock_env_vars
    ):
        """Test generate_alternate_song_spec in mock mode."""
        violet_agent_state.interviewer_persona = VanityPersona()

        result = violet_agent.generate_alternate_song_spec(violet_agent_state)

        assert result is not None
        assert isinstance(result, VioletAgentState)
        assert result.counter_proposal is not None
        assert isinstance(result.counter_proposal, SongProposalIteration)

    def test_generate_alternate_song_spec_has_title(
        self, violet_agent, violet_agent_state, mock_env_vars
    ):
        """Test that generated proposal has a title."""
        violet_agent_state.interviewer_persona = VanityPersona()

        result = violet_agent.generate_alternate_song_spec(violet_agent_state)

        assert result.counter_proposal.title is not None
        assert isinstance(result.counter_proposal.title, str)
        assert len(result.counter_proposal.title) > 0

    def test_generate_alternate_song_spec_has_required_fields(
        self, violet_agent, violet_agent_state, mock_env_vars
    ):
        """Test that generated proposal has all required fields."""
        violet_agent_state.interviewer_persona = VanityPersona()

        result = violet_agent.generate_alternate_song_spec(violet_agent_state)

        proposal = result.counter_proposal
        assert hasattr(proposal, "bpm")
        assert hasattr(proposal, "tempo")
        assert hasattr(proposal, "key")
        assert hasattr(proposal, "rainbow_color")
        assert hasattr(proposal, "mood")
        assert hasattr(proposal, "genres")
        assert hasattr(proposal, "concept")


class TestLoadCorpus:
    """Tests for _load_corpus helper method."""

    def test_load_corpus_with_directory(self, tmp_path):
        """Test loading corpus from a directory of markdown files."""
        corpus_dir = tmp_path / "corpus"
        corpus_dir.mkdir()

        (corpus_dir / "file1.md").write_text("Content from file 1")
        (corpus_dir / "file2.md").write_text("Content from file 2")

        result = VioletAgent._load_corpus(corpus_dir)

        assert isinstance(result, str)
        assert "Content from file 1" in result
        assert "Content from file 2" in result

    def test_load_corpus_with_single_file(self, tmp_path):
        """Test loading corpus from a single file."""
        corpus_file = tmp_path / "corpus.md"
        corpus_file.write_text("Single file content")

        result = VioletAgent._load_corpus(corpus_file)

        assert isinstance(result, str)
        assert "Single file content" in result

    def test_load_corpus_nonexistent_path(self, tmp_path):
        """Test loading corpus from nonexistent path returns empty string."""
        nonexistent = tmp_path / "nonexistent"

        result = VioletAgent._load_corpus(nonexistent)

        assert isinstance(result, str)
        assert result == ""


class TestVioletAgentFullWorkflow:
    """Tests for the full VioletAgent workflow."""

    def test_full_workflow_mock_mode(self, violet_agent, mock_env_vars):
        """Test the full violet agent workflow in mock mode."""
        proposals = SongProposal(
            iterations=[
                SongProposalIteration(
                    iteration_id="white_001",
                    bpm=120,
                    tempo="4/4",
                    key="C Major",
                    rainbow_color="white",
                    title="Initial Song",
                    mood=["contemplative"],
                    genres=["experimental"],
                    concept="A test concept exploring dialectical tension through adversarial interview techniques and philosophical inquiry, examining the intersections of critique, defense, and synthesis in artistic discourse through a methodology of provocative questioning.",
                )
            ]
        )

        main_state = MainAgentState(
            thread_id="mock_thread_001", song_proposals=proposals
        )

        result = violet_agent(main_state)

        assert result is not None
        assert isinstance(result, MainAgentState)
        assert len(result.song_proposals.iterations) == 2  # Original + counter proposal

    def test_workflow_adds_counter_proposal(self, violet_agent, mock_env_vars):
        """Test that the workflow adds a counter proposal to the song proposals."""
        proposals = SongProposal(
            iterations=[
                SongProposalIteration(
                    iteration_id="white_001",
                    bpm=120,
                    tempo="4/4",
                    key="C Major",
                    rainbow_color="white",
                    title="Initial",
                    mood=["test"],
                    genres=["test"],
                    concept="Testing the workflow counter proposal mechanism through philosophical dialectical synthesis and adversarial questioning, exploring how critique transforms through discourse into refined artistic vision.",
                )
            ]
        )

        main_state = MainAgentState(
            thread_id="mock_thread_001", song_proposals=proposals
        )

        original_count = len(main_state.song_proposals.iterations)
        result = violet_agent(main_state)

        assert len(result.song_proposals.iterations) == original_count + 1

    def test_workflow_adds_artifacts(self, violet_agent, mock_env_vars):
        """Test that the workflow adds artifacts to the state."""
        proposals = SongProposal(
            iterations=[
                SongProposalIteration(
                    iteration_id="white_001",
                    bpm=120,
                    tempo="4/4",
                    key="C Major",
                    rainbow_color="white",
                    title="Initial",
                    mood=["test"],
                    genres=["test"],
                    concept="Testing artifact generation through the violet agent workflow and dialectical interview process, documenting the iterative refinement of artistic ideas through structured critical engagement and response.",
                )
            ]
        )

        main_state = MainAgentState(
            thread_id="mock_thread_001", song_proposals=proposals
        )

        result = violet_agent(main_state)

        assert hasattr(result, "artifacts")
        # Artifacts should be added during the workflow
        assert isinstance(result.artifacts, list)


class TestVioletAgentStateManagement:
    """Tests for VioletAgent state management."""

    def test_state_initialization(self):
        """Test VioletAgentState initialization."""
        state = VioletAgentState(thread_id="test-thread")

        assert state.thread_id == "test-thread"
        assert state.interviewer_persona is None
        assert state.interview_questions is None
        assert state.interview_responses is None
        assert state.circle_jerk_interview is None
        assert state.counter_proposal is None

    def test_state_can_hold_persona(self):
        """Test that VioletAgentState can hold a persona."""
        state = VioletAgentState(thread_id="test-thread")
        persona = VanityPersona(first_name="Jane", last_name="Doe")

        state.interviewer_persona = persona

        assert state.interviewer_persona == persona
        assert state.interviewer_persona.first_name == "Jane"

    def test_state_can_hold_questions_and_responses(self):
        """Test that VioletAgentState can hold questions and responses."""
        state = VioletAgentState(thread_id="test-thread")

        questions = [
            VanityInterviewQuestion(number=1, question="Question 1?"),
            VanityInterviewQuestion(number=2, question="Question 2?"),
        ]
        responses = [
            VanityInterviewResponse(question_number=1, response="Response 1"),
            VanityInterviewResponse(question_number=2, response="Response 2"),
        ]

        state.interview_questions = questions
        state.interview_responses = responses

        assert state.interview_questions == questions
        assert state.interview_responses == responses
        assert len(state.interview_questions) == 2
        assert len(state.interview_responses) == 2


class TestVioletAgentErrorHandling:
    """Tests for VioletAgent error handling."""

    def test_generate_questions_handles_missing_mock(
        self, violet_agent, violet_agent_state, monkeypatch
    ):
        """Test that generate_questions handles missing mock file gracefully."""
        monkeypatch.setenv("MOCK_MODE", "true")
        monkeypatch.setenv("BLOCK_MODE", "false")
        monkeypatch.setenv("AGENT_MOCK_DATA_PATH", "/nonexistent/path")
        violet_agent_state.interviewer_persona = VanityPersona()

        # Should not raise an exception in non-blocking mode
        result = violet_agent.generate_questions(violet_agent_state)
        assert isinstance(result, VioletAgentState)
        # Should have fallback questions
        assert result.interview_questions is not None

    def test_generate_questions_raises_in_block_mode(
        self, violet_agent, violet_agent_state, monkeypatch
    ):
        """Test that generate_questions raises exception in block mode when mock fails."""
        monkeypatch.setenv("MOCK_MODE", "true")
        monkeypatch.setenv("BLOCK_MODE", "true")
        monkeypatch.setenv("AGENT_MOCK_DATA_PATH", "/nonexistent/path")
        violet_agent_state.interviewer_persona = VanityPersona()

        with pytest.raises(Exception):
            violet_agent.generate_questions(violet_agent_state)

    def test_generate_alternate_song_spec_raises_in_block_mode(
        self, violet_agent, violet_agent_state, monkeypatch
    ):
        """Test that generate_alternate_song_spec raises exception in block mode when mock fails."""
        monkeypatch.setenv("MOCK_MODE", "true")
        monkeypatch.setenv("BLOCK_MODE", "true")
        monkeypatch.setenv("AGENT_MOCK_DATA_PATH", "/nonexistent/path")
        violet_agent_state.interviewer_persona = VanityPersona()

        with pytest.raises(Exception):
            violet_agent.generate_alternate_song_spec(violet_agent_state)
