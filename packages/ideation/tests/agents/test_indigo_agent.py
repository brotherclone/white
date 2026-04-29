import re

from white_core.manifests.song_proposal import SongProposalIteration
from white_ideation.agents.indigo_agent import IndigoAgent, _parse_proposal_response
from white_ideation.agents.states.indigo_agent_state import IndigoAgentState


def test_generate_alternate_song_spec_mock():
    agent = IndigoAgent()
    state = IndigoAgentState()
    result_state = agent.generate_alternate_song_spec(state)
    assert result_state.counter_proposal is not None
    assert isinstance(result_state.counter_proposal, SongProposalIteration)
    assert getattr(result_state.counter_proposal, "title", None)


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
