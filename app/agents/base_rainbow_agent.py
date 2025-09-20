from langgraph.graph import StateGraph
from pydantic import BaseModel

from app.agents.models.agent_settings import AgentSettings


class BaseRainbowAgent(BaseModel):

    """Base class for all Rainbow Agents"""

    settings: AgentSettings | None = None
    graph: StateGraph = None

    def __init__(self, **data):
        super().__init__(**data)
        self.graph: StateGraph = self.create_graph()