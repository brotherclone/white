from app.agents.states.orange_agent_state import OrangeAgentState
from app.structures.agents.base_rainbow_agent_state import BaseRainbowAgentState


def test_orange_agent_state_defaults():
    state = OrangeAgentState()
    assert isinstance(state, BaseRainbowAgentState)


def test_orange_agent_state_custom_fields():
    state = OrangeAgentState()
    assert state.gonzo_intensity == 3
