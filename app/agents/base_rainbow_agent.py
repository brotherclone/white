from langgraph.graph import StateGraph
from pydantic import BaseModel, ConfigDict
from typing import Optional
from abc import ABC, abstractmethod
from app.agents.models.agent_settings import AgentSettings


class BaseRainbowAgent(BaseModel, ABC):

    """Base class for all Rainbow Agents"""

    model_config = ConfigDict(arbitrary_types_allowed=True, extra='allow')

    settings: AgentSettings | None = None
    graph: Optional[StateGraph] = None

    def __init__(self, **data):
        super().__init__(**data)
        # Initialize the graph after Pydantic initialization
        self.graph = self.create_graph()

    @abstractmethod
    def create_graph(self) -> StateGraph:
        """Override this method in subclasses to define the agent's workflow"""
        raise NotImplementedError("Subclasses must implement create_graph method")

    @abstractmethod
    def generate_document(self):
        raise NotImplementedError("Subclasses must implement generate_document method")

    @abstractmethod
    def generate_alternate_song_spec(self):
        raise NotImplementedError("Subclasses must implement generate_alternate_song_spec method")

    @abstractmethod
    def contribute(self):
        raise NotImplementedError("Subclasses must implement contribute method")