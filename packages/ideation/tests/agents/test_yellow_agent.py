from unittest.mock import MagicMock, patch

from white_core.manifests.song_proposal import SongProposalIteration
from white_ideation.agents.states.yellow_agent_state import YellowAgentState
from white_ideation.agents.yellow_agent import YellowAgent


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


def test_generate_characters_always_calls_character_sheet(monkeypatch):
    """create_character_sheet is always called (character sheets are now Markdown)."""
    monkeypatch.setenv("MOCK_MODE", "false")

    mock_char = MagicMock()

    state = YellowAgentState(thread_id="test-thread")

    with patch("white_ideation.agents.yellow_agent.roll_dice", return_value=[1]):
        with patch(
            "white_ideation.agents.yellow_agent.PulsarPalaceCharacter.create_random",
            return_value=mock_char,
        ):
            YellowAgent.generate_characters(state)

    mock_char.create_character_sheet.assert_called_once()
    mock_char.create_portrait.assert_called_once()


def test_generate_alternate_song_spec_includes_sounds_like(monkeypatch):
    """Regression for add-ideation-sounds-like: Yellow's reference-works
    section must include sampled sounds-like artists."""
    monkeypatch.setenv("MOCK_MODE", "false")
    monkeypatch.setenv("BLOCK_MODE", "false")
    monkeypatch.setattr(
        "white_ideation.agents.yellow_agent.get_sounds_like_by_color",
        lambda color: ["Test Reference Artist"],
    )
    monkeypatch.setattr(
        "white_ideation.agents.yellow_agent.sample_reference_artists",
        lambda artists, **kwargs: list(artists),
    )

    agent = YellowAgent()
    narrative = MagicMock()
    narrative.story = ["A strange room hums with static."]
    state = YellowAgentState.model_construct(
        thread_id="test-thread",
        rooms=[MagicMock()],
        encounter_narrative_artifact=narrative,
        negative_constraints="",
        white_proposal=None,
        artifacts=[],
    )

    extracted_proposal = MagicMock()
    extracted_proposal.bpm = 100
    extracted_proposal.key = "A minor"
    extracted_proposal.mood = "eerie"
    extracted_proposal.genres = "synthwave"

    counter_proposal = SongProposalIteration(
        iteration_id="yellow_v1",
        bpm=100,
        tempo="4/4",
        key="A minor",
        rainbow_color="Y",
        title="Test Yellow Song",
        mood=["eerie"],
        genres=["synthwave"],
        concept="A test concept long enough to pass the minimum length validator "
        "for the concept field, exploring the Pulsar Palace game session.",
    )
    seen_prompts = []

    def fake_invoke_structured(llm, schema, prompt):
        seen_prompts.append(prompt)
        return counter_proposal

    with (
        patch.object(
            agent.music_extractor,
            "extract_song_proposal",
            return_value=extracted_proposal,
        ),
        patch.object(agent, "_invoke_structured", side_effect=fake_invoke_structured),
    ):
        agent.generate_alternate_song_spec(state)

    assert len(seen_prompts) == 1
    assert "Test Reference Artist" in seen_prompts[0]


def test_generate_alternate_song_spec_includes_negative_constraints(monkeypatch):
    """Regression for add-ideation-negative-constraints: Yellow's own
    counter-proposal prompt must honor negative_constraints, not just
    White's initial proposal and final rewrite."""
    monkeypatch.setenv("MOCK_MODE", "false")
    monkeypatch.setenv("BLOCK_MODE", "false")

    agent = YellowAgent()
    narrative = MagicMock()
    narrative.story = ["A strange room hums with static."]
    # model_construct bypasses validation: rooms/encounter_narrative_artifact
    # are normally strict Pydantic models, but extract_song_proposal (which
    # is what actually reads them) is mocked below, so real nested objects
    # aren't needed for this test.
    state = YellowAgentState.model_construct(
        thread_id="test-thread",
        rooms=[MagicMock()],
        encounter_narrative_artifact=narrative,
        negative_constraints="AVOID: the word 'void', keys already used: C Major",
        white_proposal=None,
        artifacts=[],
    )

    extracted_proposal = MagicMock()
    extracted_proposal.bpm = 100
    extracted_proposal.key = "A minor"
    extracted_proposal.mood = "eerie"
    extracted_proposal.genres = "synthwave"

    counter_proposal = SongProposalIteration(
        iteration_id="yellow_v1",
        bpm=100,
        tempo="4/4",
        key="A minor",
        rainbow_color="Y",
        title="Test Yellow Song",
        mood=["eerie"],
        genres=["synthwave"],
        concept="A test concept long enough to pass the minimum length validator "
        "for the concept field, exploring the Pulsar Palace game session.",
    )
    seen_prompts = []

    def fake_invoke_structured(llm, schema, prompt):
        seen_prompts.append(prompt)
        return counter_proposal

    with (
        patch.object(
            agent.music_extractor,
            "extract_song_proposal",
            return_value=extracted_proposal,
        ),
        patch.object(agent, "_invoke_structured", side_effect=fake_invoke_structured),
    ):
        result_state = agent.generate_alternate_song_spec(state)

    assert result_state.counter_proposal is not None
    assert len(seen_prompts) == 1
    assert state.negative_constraints in seen_prompts[0]
