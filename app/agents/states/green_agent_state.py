from app.structures.agents.base_rainbow_agent_state import BaseRainbowAgentState

class GreenAgentState(BaseRainbowAgentState):
    def __init__(self, **data):
        super().__init__(**data)