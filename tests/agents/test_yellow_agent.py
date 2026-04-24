from unittest.mock import MagicMock, patch

from white_core.manifests.song_proposal import SongProposalIteration

from app.agents.states.yellow_agent_state import YellowAgentState
from app.agents.yellow_agent import YellowAgent


def test_generate_alternate_song_spec_mock(monkeypatch):
    """Test that generate_alternate_song_spec loads from mock in mock mode"""
    monkeypatch.setenv("MOCK_MODE", "true")
    agent = YellowAgent()
    state = YellowAgentState()
    result_state = agent.generate_alternate_song_spec(state)
    assert result_state.counter_proposal is not None
    assert isinstance(result_state.counter_proposal, SongProposalIteration)
    assert getattr(result_state.counter_proposal, "title", None)


def test_yellow_agent_initialization():
    """Test that YellowAgent can be initialized"""
    agent = YellowAgent()
    assert agent is not None
    assert agent.room_generator is not None
    assert agent.action_generator is not None
    assert agent.music_extractor is not None
    assert agent.max_rooms == 4


def test_generate_characters_skips_html_by_default(monkeypatch):
    """create_character_sheet is NOT called when WHITE_WITH_HTML is absent."""
    monkeypatch.setenv("MOCK_MODE", "false")
    monkeypatch.delenv("WHITE_WITH_HTML", raising=False)

    mock_char = MagicMock()

    state = YellowAgentState(thread_id="test-thread-no-html")

    with patch("app.agents.yellow_agent.roll_dice", return_value=[1]):
        with patch(
            "app.agents.yellow_agent.PulsarPalaceCharacter.create_random",
            return_value=mock_char,
        ):
            YellowAgent.generate_characters(state)

    mock_char.create_character_sheet.assert_not_called()
    mock_char.create_portrait.assert_called_once()


def test_generate_characters_calls_html_when_flag_set(monkeypatch):
    """create_character_sheet IS called when WHITE_WITH_HTML=true."""
    monkeypatch.setenv("MOCK_MODE", "false")
    monkeypatch.setenv("WHITE_WITH_HTML", "true")

    mock_char = MagicMock()

    state = YellowAgentState(thread_id="test-thread-html")

    with patch("app.agents.yellow_agent.roll_dice", return_value=[1]):
        with patch(
            "app.agents.yellow_agent.PulsarPalaceCharacter.create_random",
            return_value=mock_char,
        ):
            YellowAgent.generate_characters(state)

    mock_char.create_character_sheet.assert_called_once()
