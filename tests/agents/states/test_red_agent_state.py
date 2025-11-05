from app.agents.states.red_agent_state import RedAgentState
from app.structures.agents.base_rainbow_agent_state import \
    BaseRainbowAgentState


def test_red_agent_defaults():
    state = RedAgentState()
    assert isinstance(state, BaseRainbowAgentState)


def test_red_agent_state_custom_fields():
    state = RedAgentState()
    assert state.should_create_book is True
    assert state.should_respond_with_reaction_book is False
