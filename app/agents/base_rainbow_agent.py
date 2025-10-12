import random

from langchain_anthropic import ChatAnthropic
from langgraph.graph import StateGraph
from pydantic import BaseModel, ConfigDict
from typing import Optional, List,Callable, Union
from abc import ABC, abstractmethod
from dotenv import load_dotenv
from functools import wraps


from app.agents.models.agent_settings import AgentSettings
from app.agents.models.base_chain_artifact import ChainArtifact
from app.agents.states.base_rainbow_agent_state import BaseRainbowAgentState

load_dotenv()

Chance = Union[float, Callable[[object], float]]

class BaseRainbowAgent(BaseModel, ABC):
    """Base class for all Rainbow Agents"""

    model_config = ConfigDict(arbitrary_types_allowed=True, extra='allow')

    settings: AgentSettings | None = None
    graph: Optional[StateGraph] = None
    chain_artifacts: List[ChainArtifact] = []

    def __init__(self, **data):
        super().__init__(**data)
        self.graph = self.create_graph()

    @abstractmethod
    def create_graph(self) -> StateGraph:
        raise NotImplementedError("Subclasses must implement create_graph method")

    @abstractmethod
    def generate_alternate_song_spec(self, agent_state: BaseRainbowAgentState) -> StateGraph:
        raise NotImplementedError("Subclasses must implement generate_alternate_song_spec method")

    def _get_claude(self) -> ChatAnthropic:
        return ChatAnthropic(
            model_name=self.settings.anthropic_sub_model_name,
            api_key=self.settings.anthropic_api_key,
            temperature=self.settings.temperature,
            max_retries=self.settings.max_retries,
            timeout=self.settings.timeout,
            stop=self.settings.stop
        )
