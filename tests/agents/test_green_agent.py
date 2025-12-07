import pytest

from app.agents.green_agent import GreenAgent
from app.agents.states.green_agent_state import GreenAgentState
from app.agents.states.white_agent_state import MainAgentState
from app.structures.artifacts.arbitrarys_survey_artifact import ArbitrarysSurveyArtifact
from app.structures.artifacts.last_human_artifact import LastHumanArtifact
from app.structures.artifacts.last_human_species_extinction_narative_artifact import (
    LastHumanSpeciesExtinctionNarrativeArtifact,
)
from app.structures.artifacts.rescue_decision_artifact import RescueDecisionArtifact
from app.structures.artifacts.species_extinction_artifact import (
    SpeciesExtinctionArtifact,
)
from app.structures.concepts.last_human_species_extinction_parallel_moment import (
    LastHumanSpeciesExtinctionParallelMoment,
)
from app.structures.enums.extinction_cause import ExtinctionCause
from app.structures.enums.last_human_vulnerability_type import (
    LastHumanVulnerabilityType,
)
from app.structures.enums.last_human_documentation_type import (
    LastHumanDocumentationType,
)
from app.structures.manifests.song_proposal import SongProposalIteration


@pytest.fixture
def mock_env_vars(monkeypatch):
    """Set up mock environment variables for testing."""
    monkeypatch.setenv("MOCK_MODE", "true")
    monkeypatch.setenv("BLOCK_MODE", "false")
    monkeypatch.setenv("AGENT_MOCK_DATA_PATH", "tests/mocks")
    monkeypatch.setenv("AGENT_WORK_PRODUCT_BASE_PATH", "/tmp/test_artifacts")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")


@pytest.fixture
def green_agent():
    """Create a GreenAgent instance for testing."""
    return GreenAgent()


@pytest.fixture
def green_agent_state():
    """Create a GreenAgentState instance for testing."""
    return GreenAgentState(thread_id="test-thread-001")


class TestGreenAgentInitialization:
    """Tests for GreenAgent initialization."""

    def test_green_agent_instantiation(self):
        """Test that GreenAgent can be instantiated."""
        agent = GreenAgent()
        assert agent is not None
        assert isinstance(agent, GreenAgent)

    def test_green_agent_has_llm(self):
        """Test that GreenAgent has an LLM configured."""
        agent = GreenAgent()
        assert hasattr(agent, "llm")
        assert agent.llm is not None

    def test_green_agent_has_settings(self):
        """Test that GreenAgent has settings configured."""
        agent = GreenAgent()
        assert hasattr(agent, "settings")
        assert agent.settings is not None


class TestGreenAgentGraphCreation:
    """Tests for GreenAgent graph creation."""

    def test_create_graph(self, green_agent):
        """Test that create_graph returns a valid StateGraph."""
        graph = green_agent.create_graph()
        assert graph is not None

    def test_graph_has_all_nodes(self, green_agent):
        """Test that the graph has all required nodes."""
        graph = green_agent.create_graph()
        compiled = graph.compile()

        # The graph should have all the expected nodes
        expected_nodes = [
            "get_species",
            "get_human",
            "get_parallel_moment",
            "write_last_human_extinction_narrative",
            "survey",
            "claudes_choice",
            "generate_alternate_song_spec",
        ]

        # Check nodes exist in the compiled graph
        for node in expected_nodes:
            assert node in compiled.nodes

    def test_graph_workflow_order(self, green_agent):
        """Test that the graph has the correct workflow order."""
        graph = green_agent.create_graph()
        compiled = graph.compile()

        # Verify the graph structure is created
        assert compiled is not None
        assert len(compiled.nodes) > 0


class TestGetSpecies:
    """Tests for get_species node."""

    def test_get_species_mock_mode(self, green_agent_state, mock_env_vars):
        """Test get_species in mock mode."""
        result = GreenAgent.get_species(green_agent_state)

        assert result is not None
        assert isinstance(result, GreenAgentState)
        assert result.current_species is not None
        assert isinstance(result.current_species, SpeciesExtinctionArtifact)

    def test_get_species_returns_state(self, green_agent_state, mock_env_vars):
        """Test that get_species returns the modified state."""
        original_thread_id = green_agent_state.thread_id
        result = GreenAgent.get_species(green_agent_state)

        assert result.thread_id == original_thread_id
        assert isinstance(result, GreenAgentState)


class TestGetHuman:
    """Tests for get_human node."""

    def test_get_human_mock_mode(self, green_agent, green_agent_state, mock_env_vars):
        """Test get_human in mock mode."""
        # First get a species for context
        green_agent_state = GreenAgent.get_species(green_agent_state)

        result = green_agent.get_human(green_agent_state)

        assert result is not None
        assert isinstance(result, GreenAgentState)
        assert result.current_human is not None
        assert isinstance(result.current_human, LastHumanArtifact)

    def test_get_human_returns_state(
        self, green_agent, green_agent_state, mock_env_vars
    ):
        """Test that get_human returns the modified state."""
        green_agent_state = GreenAgent.get_species(green_agent_state)
        original_thread_id = green_agent_state.thread_id
        result = green_agent.get_human(green_agent_state)

        assert result.thread_id == original_thread_id
        assert isinstance(result, GreenAgentState)


class TestGetParallelMoment:
    """Tests for get_parallel_moment node."""

    def test_get_parallel_moment_mock_mode(
        self, green_agent, green_agent_state, mock_env_vars
    ):
        """Test get_parallel_moment in mock mode."""
        # Set up prerequisites
        green_agent_state = GreenAgent.get_species(green_agent_state)
        green_agent_state = green_agent.get_human(green_agent_state)

        result = green_agent.get_parallel_moment(green_agent_state)

        assert result is not None
        assert isinstance(result, GreenAgentState)
        assert result.current_parallel_moment is not None
        assert isinstance(
            result.current_parallel_moment, LastHumanSpeciesExtinctionParallelMoment
        )


class TestWriteLastHumanExtinctionNarrative:
    """Tests for write_last_human_extinction_narrative node."""

    def test_write_narrative_mock_mode(
        self, green_agent, green_agent_state, mock_env_vars
    ):
        """Test write_last_human_extinction_narrative in mock mode."""
        # Set up prerequisites
        green_agent_state = GreenAgent.get_species(green_agent_state)
        green_agent_state = green_agent.get_human(green_agent_state)
        green_agent_state = green_agent.get_parallel_moment(green_agent_state)

        result = green_agent.write_last_human_extinction_narrative(green_agent_state)

        assert result is not None
        assert isinstance(result, GreenAgentState)
        assert result.current_narrative is not None
        assert isinstance(
            result.current_narrative, LastHumanSpeciesExtinctionNarrativeArtifact
        )


class TestSurvey:
    """Tests for survey node."""

    def test_survey_mock_mode(self, green_agent, green_agent_state, mock_env_vars):
        """Test survey in mock mode."""
        # Set up prerequisites
        green_agent_state = GreenAgent.get_species(green_agent_state)
        green_agent_state = green_agent.get_human(green_agent_state)
        green_agent_state = green_agent.get_parallel_moment(green_agent_state)
        green_agent_state = green_agent.write_last_human_extinction_narrative(
            green_agent_state
        )

        result = green_agent.survey(green_agent_state)

        assert result is not None
        assert isinstance(result, GreenAgentState)
        assert result.current_survey is not None
        assert isinstance(result.current_survey, ArbitrarysSurveyArtifact)


class TestClaudesChoice:
    """Tests for claudes_choice node."""

    def test_claudes_choice_mock_mode(
        self, green_agent, green_agent_state, mock_env_vars
    ):
        """Test claudes_choice in mock mode."""
        # Set up prerequisites
        green_agent_state = GreenAgent.get_species(green_agent_state)
        green_agent_state = green_agent.get_human(green_agent_state)
        green_agent_state = green_agent.get_parallel_moment(green_agent_state)
        green_agent_state = green_agent.write_last_human_extinction_narrative(
            green_agent_state
        )
        green_agent_state = green_agent.survey(green_agent_state)

        result = green_agent.claudes_choice(green_agent_state)

        assert result is not None
        assert isinstance(result, GreenAgentState)
        assert result.current_decision is not None
        assert isinstance(result.current_decision, RescueDecisionArtifact)


class TestGenerateAlternateSongSpec:
    """Tests for generate_alternate_song_spec node."""

    def test_generate_alternate_song_spec_mock(
        self, green_agent, green_agent_state, mock_env_vars
    ):
        """Test generate_alternate_song_spec in mock mode."""
        # Set up prerequisites - need to provide species and human for the prompt
        green_agent_state = GreenAgent.get_species(green_agent_state)
        green_agent_state = green_agent.get_human(green_agent_state)

        # Add a white proposal to the state
        green_agent_state.white_proposal = SongProposalIteration(
            iteration_id="white_001",
            bpm=120,
            tempo="4/4",
            key="C Major",
            rainbow_color="white",
            title="Test White Song",
            mood=["contemplative"],
            genres=["ambient"],
            concept="A test concept exploring the nature of existence through contemplative soundscapes, reflecting on the philosophical intersections between consciousness, time, and the infinite echoes of being across space",
        )

        result = green_agent.generate_alternate_song_spec(green_agent_state)

        assert result is not None
        assert isinstance(result, GreenAgentState)
        assert result.counter_proposal is not None
        assert isinstance(result.counter_proposal, SongProposalIteration)
        assert hasattr(result.counter_proposal, "title")

    def test_generate_alternate_song_spec_creates_proposal(
        self, green_agent, green_agent_state, mock_env_vars
    ):
        """Test that generate_alternate_song_spec creates a valid proposal."""
        green_agent_state = GreenAgent.get_species(green_agent_state)
        green_agent_state = green_agent.get_human(green_agent_state)
        green_agent_state.white_proposal = SongProposalIteration(
            iteration_id="white_001",
            bpm=120,
            tempo="4/4",
            key="C Major",
            rainbow_color="white",
            title="Test",
            mood=["test"],
            genres=["test"],
            concept="Testing the song proposal iteration creation process through automated validation of pydantic models and their philosophical depth requirements",
        )

        result = green_agent.generate_alternate_song_spec(green_agent_state)

        assert result.counter_proposal is not None
        assert result.counter_proposal.title is not None
        assert isinstance(result.counter_proposal.bpm, (int, float))


class TestGreenAgentFullWorkflow:
    """Tests for the full GreenAgent workflow."""

    def test_full_workflow_mock_mode(self, green_agent, mock_env_vars):
        """Test the full green agent workflow in mock mode."""
        # Create initial state with song proposals
        from app.structures.manifests.song_proposal import SongProposal

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
                    genres=["ambient"],
                    concept="A test concept exploring existence through sound, examining the archetypal patterns of consciousness manifest in musical form across temporal dimensions",
                )
            ]
        )

        main_state = MainAgentState(
            thread_id="test-thread-001", song_proposals=proposals
        )

        # Run the full agent
        result = green_agent(main_state)

        assert result is not None
        assert isinstance(result, MainAgentState)
        assert len(result.song_proposals.iterations) == 2  # Original + counter proposal

    def test_workflow_adds_counter_proposal(self, green_agent, mock_env_vars):
        """Test that the workflow adds a counter proposal to the song proposals."""
        from app.structures.manifests.song_proposal import SongProposal

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
                    concept="Testing the workflow counter proposal mechanism and ensuring that philosophical content meets minimum substantive requirements for archetypal expression",
                )
            ]
        )

        main_state = MainAgentState(
            thread_id="test-thread-001", song_proposals=proposals
        )

        original_count = len(main_state.song_proposals.iterations)
        result = green_agent(main_state)

        assert len(result.song_proposals.iterations) == original_count + 1


class TestGreenAgentStateManagement:
    """Tests for GreenAgent state management."""

    def test_state_initialization(self):
        """Test GreenAgentState initialization."""
        state = GreenAgentState(thread_id="test-thread")

        assert state.thread_id == "test-thread"
        assert state.current_species is None
        assert state.current_human is None
        assert state.current_parallel_moment is None
        assert state.current_narrative is None
        assert state.current_survey is None
        assert state.current_decision is None

    def test_state_can_hold_artifacts(self):
        """Test that GreenAgentState can hold all artifact types."""
        state = GreenAgentState(thread_id="test-thread")

        # Create mock artifacts
        species = SpeciesExtinctionArtifact(
            thread_id="test",
            scientific_name="Testus testus",
            common_name="Test Species",
            taxonomic_group="test",
            iucn_status="Extinct",
            extinction_year=2050,
            habitat="Test habitat",
            primary_cause=ExtinctionCause.HABITAT_LOSS,
            ecosystem_role="Test role",
        )

        human = LastHumanArtifact(
            thread_id="test",
            name="Test Human",
            age=40,
            location="Test Location",
            year_documented=2050,
            parallel_vulnerability=LastHumanVulnerabilityType.DISPLACEMENT,
            vulnerability_details="Test details",
            environmental_stressor="Test stressor",
            documentation_type=LastHumanDocumentationType.WITNESS,
            last_days_scenario="Test scenario",
        )

        # Assign to state
        state.current_species = species
        state.current_human = human

        assert state.current_species == species
        assert state.current_human == human


class TestGreenAgentErrorHandling:
    """Tests for GreenAgent error handling."""

    def test_get_species_handles_missing_mock(self, green_agent_state, monkeypatch):
        """Test that get_species handles missing mock file gracefully."""
        monkeypatch.setenv("MOCK_MODE", "true")
        monkeypatch.setenv("BLOCK_MODE", "false")
        monkeypatch.setenv("AGENT_MOCK_DATA_PATH", "/nonexistent/path")

        # Should not raise an exception in non-blocking mode
        result = GreenAgent.get_species(green_agent_state)
        assert isinstance(result, GreenAgentState)

    def test_get_species_raises_in_block_mode(self, green_agent_state, monkeypatch):
        """Test that get_species raises exception in block mode when mock fails."""
        monkeypatch.setenv("MOCK_MODE", "true")
        monkeypatch.setenv("BLOCK_MODE", "true")
        monkeypatch.setenv("AGENT_MOCK_DATA_PATH", "/nonexistent/path")

        with pytest.raises(Exception):
            GreenAgent.get_species(green_agent_state)
