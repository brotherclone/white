import re
from unittest.mock import MagicMock, patch

from white_core.manifests.song_proposal import SongProposal, SongProposalIteration
from white_ideation.agents.indigo_agent import IndigoAgent, _parse_proposal_response
from white_ideation.agents.states.indigo_agent_state import IndigoAgentState


def test_generate_alternate_song_spec_mock():
    agent = IndigoAgent()
    state = IndigoAgentState()
    result_state = agent.generate_alternate_song_spec(state)
    assert result_state.counter_proposal is not None
    assert isinstance(result_state.counter_proposal, SongProposalIteration)
    assert getattr(result_state.counter_proposal, "title", None)


def _proposal(**overrides) -> SongProposalIteration:
    base = dict(
        iteration_id="prior_v3",
        bpm=120,
        key="C major",
        rainbow_color="Blue",
        title="Prior Proposal",
        mood=["melancholic"],
        genres=["ambient"],
        concept="A test concept that is long enough to pass the minimum length "
        "validator for the concept field.",
    )
    base.update(overrides)
    return SongProposalIteration(**base)


def test_generate_alternate_song_spec_mock_continues_predecessor_chain():
    """Regression: iteration_number must continue from the proposal this
    counters, not snapshot the overall list length — a long, unrelated
    song_proposals list must not inflate Indigo's own chain position."""
    agent = IndigoAgent()
    predecessor = _proposal(iteration_id="prior_v3", iteration_number=3)
    unrelated = [
        _proposal(iteration_id=f"unrelated_v{i}", iteration_number=None)
        for i in range(10)
    ]
    state = IndigoAgentState(
        white_proposal=predecessor,
        song_proposals=SongProposal(iterations=[*unrelated, predecessor]),
    )
    result_state = agent.generate_alternate_song_spec(state)
    assert result_state.counter_proposal.iteration_number == 4


def test_parse_proposal_response_iteration_id_is_slug():
    response = """Title: Echoes of a Borrowed Life
Key: D minor
BPM: 90
Tempo: Slow
Mood: melancholy, reflective
Genres: indie, chamber pop
Concept: A life lived in someone else's shadow
"""
    result = _parse_proposal_response(response)
    assert re.fullmatch(
        r"indigo_[a-z0-9_]+_v1", result.iteration_id
    ), f"iteration_id {result.iteration_id!r} is not a slug"
    assert not any(
        c.isdigit()
        and len(result.iteration_id) > 15
        and result.iteration_id.count("_") == 0
        for c in result.iteration_id
    ), "iteration_id looks like a timestamp"
    assert result.iteration_id.startswith("indigo_")
    assert result.iteration_id.endswith("_v1")


def test_is_valid_anagram_true_positive():
    agent = IndigoAgent()
    assert agent._is_valid_anagram("Eleven plus two", "Twelve plus one")


def test_is_valid_anagram_false_for_mismatched_letters():
    agent = IndigoAgent()
    assert not agent._is_valid_anagram("The Silent Answer", "Completely Different")


def test_generate_alternate_song_spec_includes_negative_constraints(monkeypatch):
    """Regression for add-ideation-negative-constraints: Indigo's own
    counter-proposal prompt must honor negative_constraints, not just
    White's initial proposal and final rewrite."""
    monkeypatch.setenv("MOCK_MODE", "false")
    monkeypatch.setenv("BLOCK_MODE", "false")

    agent = IndigoAgent()
    predecessor = _proposal(iteration_id="prior_v3", iteration_number=3)
    state = IndigoAgentState(
        white_proposal=predecessor,
        song_proposals=SongProposal(iterations=[predecessor]),
        secret_name="The Silent Answer",
        surface_name="White Sea Lanterns",
        negative_constraints="AVOID: the word 'puzzle', keys already used: C major",
    )

    response_text = """Title: Test Indigo Song
Key: E minor
BPM: 100
Tempo: Moderate
Mood: mysterious, layered
Genres: art pop
Concept: A test concept about encoded meaning and revelation through layered puzzles."""
    mock_response = MagicMock()
    mock_response.content = response_text
    mock_llm = MagicMock()
    mock_llm.invoke = MagicMock(return_value=mock_response)

    with patch.object(agent, "llm", mock_llm):
        result_state = agent.generate_alternate_song_spec(state)

    assert result_state.counter_proposal is not None
    mock_llm.invoke.assert_called_once()
    sent_prompt = mock_llm.invoke.call_args[0][0]
    assert state.negative_constraints in sent_prompt


def test_is_valid_anagram_none_surface_returns_false_not_crash():
    """Regression: a failed SPY generation (empty structured output) used to
    leave surface_name as None, which crashed validate_anagram with
    TypeError: 'NoneType' object is not iterable and killed the whole
    workflow instead of falling through to the existing retry/fallback."""
    agent = IndigoAgent()
    assert agent._is_valid_anagram("The Silent Answer", None) is False
    assert agent._is_valid_anagram(None, "The Silent Answer") is False
    assert agent._is_valid_anagram(None, None) is False


def test_spy_choose_letter_bank_falls_back_when_response_has_no_letters(monkeypatch):
    """Regression: if the SPY's LLM response has zero alphabetic characters
    after filtering, `state.letter_bank = ""` used to be set — an empty
    string is falsy, so IndigoAgentState's `lambda x, y: y or x` reducer
    silently discards it and keeps the default None. fool_arrange_secret's
    ' '.join(state.letter_bank) then crashes the whole workflow on the very
    next node with no error logged in between. Must fall back to a real
    letter bank instead, matching the existing exception-path behavior."""
    monkeypatch.setenv("MOCK_MODE", "false")
    monkeypatch.setenv("BLOCK_MODE", "false")

    agent = IndigoAgent()
    state = IndigoAgentState(thread_id="test-thread", concepts="memory, static")

    mock_response = MagicMock()
    mock_response.content = "12345 !!! ???"  # no alphabetic characters at all
    mock_llm = MagicMock()
    mock_llm.invoke = MagicMock(return_value=mock_response)

    with patch.object(agent, "llm", mock_llm):
        result_state = agent.spy_choose_letter_bank(state)

    assert result_state.letter_bank
    assert result_state.letter_bank.isalpha()


def test_fool_arrange_secret_guards_against_missing_letter_bank(monkeypatch):
    """Defense in depth: even if letter_bank somehow stays unset (e.g. the
    LangGraph reducer edge case above, or a future caller that skips
    spy_choose_letter_bank), fool_arrange_secret must not crash."""
    monkeypatch.setenv("MOCK_MODE", "false")
    monkeypatch.setenv("BLOCK_MODE", "false")

    agent = IndigoAgent()
    state = IndigoAgentState(
        thread_id="test-thread", concepts="memory, static", letter_bank=None
    )

    mock_response = MagicMock()
    mock_response.content = "Test Secret Name"
    mock_llm = MagicMock()
    mock_llm.invoke = MagicMock(return_value=mock_response)

    with patch.object(agent, "llm", mock_llm):
        result_state = agent.fool_arrange_secret(state)

    assert result_state.letter_bank
    assert result_state.secret_name == "Test Secret Name"
