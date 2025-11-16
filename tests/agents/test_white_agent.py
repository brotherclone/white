from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from app.agents.states.white_agent_state import MainAgentState
from app.agents.white_agent import WhiteAgent
from app.structures.agents.agent_settings import AgentSettings
from app.structures.manifests.song_proposal import SongProposal, SongProposalIteration


@pytest.fixture
def white_agent(monkeypatch):
    # Prevent any heavy external calls by patching LLM helpers at class-level if needed.
    # Return a fresh WhiteAgent instance for each test.
    wa = WhiteAgent()
    return wa


def test_white_agent_initialization():
    agent = WhiteAgent()
    assert isinstance(agent.settings, AgentSettings)
    assert isinstance(agent.agents, dict)
    assert isinstance(agent.processors, dict)
    assert isinstance(agent.song_proposal, SongProposal)


@pytest.mark.parametrize(
    "input_val,expected_type",
    [
        (SongProposal(iterations=[]), SongProposal),
        ({"iterations": []}, SongProposal),
        (None, SongProposal),
    ],
)
def test_normalize_song_proposal(input_val, expected_type):
    result = WhiteAgent._normalize_song_proposal(input_val)
    assert isinstance(result, expected_type)


def test_invoke_black_agent():
    mock_state = MagicMock(spec=MainAgentState)
    mock_black_agent = MagicMock(return_value=mock_state)
    agent = WhiteAgent()
    agent.agents["black"] = mock_black_agent  # Inject mock
    result = agent.invoke_black_agent(mock_state)
    assert result == mock_state
    mock_black_agent.assert_called_once_with(mock_state)


def test_invoke_red_agent():
    mock_state = MagicMock(spec=MainAgentState)
    mock_red_agent = MagicMock(return_value=mock_state)
    agent = WhiteAgent()
    agent.agents["red"] = mock_red_agent  # Inject mock
    result = agent.invoke_red_agent(mock_state)
    assert result == mock_state
    mock_red_agent.assert_called_once_with(mock_state)


def test_invoke_orange_agent():
    mock_state = MagicMock(spec=MainAgentState)
    mock_orange_agent = MagicMock(return_value=mock_state)
    agent = WhiteAgent()
    agent.agents["orange"] = mock_orange_agent  # Inject mock
    result = agent.invoke_orange_agent(mock_state)
    assert result == mock_state
    mock_orange_agent.assert_called_once_with(mock_state)


def test_resume_after_black_agent_ritual(monkeypatch):
    """Test resuming workflow after black agent ritual completion"""

    monkeypatch.setenv("MOCK_MODE", "true")
    with patch(
        "app.agents.white_agent.resume_black_agent_workflow_with_agent"
    ) as mock_resume:
        mock_resume.return_value = {
            "counter_proposal": SongProposalIteration(
                iteration_id="mock_123",
                bpm=120,
                tempo="4/4",
                key="C Major",
                rainbow_color="black",
                title="Black Counter Proposal",
                mood=["dark"],
                genres=["experimental"],
                concept="Mock counter proposal concept. " * 10,
            ),
            "artifacts": [],
        }

        agent = WhiteAgent()
        agent._black_rebracketing_analysis = MagicMock(return_value="Mock rebracketing")
        agent._synthesize_document_for_red = MagicMock(return_value="Mock synthesis")
        state = MainAgentState(
            thread_id="test-thread-123",
            workflow_paused=True,
            song_proposals=SongProposal(iterations=[]),
            artifacts=[],
            pending_human_action={
                "agent": "black",
                "black_config": {"configurable": {"thread_id": "test-thread-123"}},
            },
        )
        updated_state = agent.resume_after_black_agent_ritual(state, verify_tasks=False)
        assert updated_state.workflow_paused is False
        assert (
            updated_state.song_proposals.iterations[-1].title
            == "Black Counter Proposal"
        )


def test_process_black_agent_work_sets_analysis_and_ready_for_red(
    monkeypatch, white_agent
):
    # Arrange
    # Patch normalization to return a predictable proposal list
    monkeypatch.setattr(
        white_agent.__class__,
        "_normalize_song_proposal",
        lambda self, proposal: SimpleNamespace(
            iterations=[{"iteration_id": "black-prop"}]
        ),
    )

    # Patch the analysis and synthesis helpers to return deterministic values
    monkeypatch.setattr(
        white_agent.__class__,
        "_black_rebracketing_analysis",
        lambda self, state, proposal, evp_artifacts, sigil_artifacts: "BLACK_ANALYSIS",
    )
    monkeypatch.setattr(
        white_agent.__class__,
        "_synthesize_document_for_red",
        lambda self, state, rebracketed_analysis, black_proposal, artifacts: "BLACK_SYNTH",
    )

    # Create a minimal state object
    state = SimpleNamespace(
        song_proposals=SongProposal(
            iterations=[
                SongProposalIteration(
                    iteration_id="test_black_prop_v1",
                    bpm=120,
                    tempo="4/4",
                    key="C Major",
                    rainbow_color="black",
                    title="Test Black Proposal",
                    mood=["dark"],
                    genres=["rock"],
                    concept="This is a test concept that explores the archetypal journey through darkness and rebirth, examining how the shadow self must be confronted and integrated before transcendence can occur in the alchemical process.",
                )
            ]
        ),
        artifacts=[
            SimpleNamespace(chain_artifact_type="evp"),
            SimpleNamespace(chain_artifact_type="sigil"),
        ],
        workflow_paused=False,
        pending_human_action=None,
        ready_for_red=False,
    )

    # Act
    result = white_agent.process_black_agent_work(state)

    # Assert
    assert getattr(result, "rebracketing_analysis") == "BLACK_ANALYSIS"
    assert getattr(result, "document_synthesis") == "BLACK_SYNTH"
    assert result.ready_for_red is True


def test_process_red_agent_work_sets_analysis_and_ready_for_orange(
    monkeypatch, white_agent
):
    # Arrange
    monkeypatch.setattr(
        white_agent.__class__,
        "_normalize_song_proposal",
        lambda self, proposal: SimpleNamespace(
            iterations=[{"iteration_id": "red-prop"}]
        ),
    )

    monkeypatch.setattr(
        white_agent.__class__,
        "_red_rebracketing_analysis",
        lambda self, state, proposal, book_artifacts: "RED_ANALYSIS",
    )
    monkeypatch.setattr(
        white_agent.__class__,
        "_synthesize_document_for_orange",
        lambda self, state, rebracketed_analysis, red_proposal, artifacts: "RED_SYNTH",
    )

    state = SimpleNamespace(
        song_proposals={"iterations": [{"iteration_id": "red-prop"}]},
        artifacts=[SimpleNamespace(chain_artifact_type="book")],
        ready_for_orange=False,
        ready_for_red=True,
    )

    # Act
    result = white_agent.process_red_agent_work(state)

    # Assert
    assert getattr(result, "rebracketing_analysis") == "RED_ANALYSIS"
    assert getattr(result, "document_synthesis") == "RED_SYNTH"
    assert result.ready_for_orange is True
    assert result.ready_for_red is False


def test_process_orange_agent_work_sets_analysis_and_ready_for_yellow(
    monkeypatch, white_agent
):
    # Arrange
    monkeypatch.setattr(
        white_agent.__class__,
        "_normalize_song_proposal",
        lambda self, proposal: SimpleNamespace(
            iterations=[{"iteration_id": "orange-prop"}]
        ),
    )

    monkeypatch.setattr(
        white_agent.__class__,
        "_orange_rebracketing_analysis",
        lambda self, state, proposal, newspaper_artifacts: "ORANGE_ANALYSIS",
    )
    monkeypatch.setattr(
        white_agent.__class__,
        "_synthesize_document_for_yellow",
        lambda self, state, rebracketed_analysis, orange_proposal, artifacts: "ORANGE_SYNTH",
    )

    state = SimpleNamespace(
        song_proposals={"iterations": [{"iteration_id": "orange-prop"}]},
        artifacts=[SimpleNamespace(chain_artifact_type="newspaper_article")],
        ready_for_orange=True,
        ready_for_yellow=False,
    )

    # Act
    result = white_agent.process_orange_agent_work(state)

    # Assert
    assert getattr(result, "rebracketing_analysis") == "ORANGE_ANALYSIS"
    assert getattr(result, "document_synthesis") == "ORANGE_SYNTH"
    assert result.ready_for_yellow is True
    assert result.ready_for_orange is False
