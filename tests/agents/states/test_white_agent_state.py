from app.agents.states.base_rainbow_agent_state import BaseRainbowAgentState
from app.agents.states.white_agent_state import MainAgentState


def test_white_agent_defaults():
    state = MainAgentState(
        thread_id="test_thread_id",
    )
    assert state.thread_id == "test_thread_id"
    assert not issubclass(MainAgentState, BaseRainbowAgentState)
