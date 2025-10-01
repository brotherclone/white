import os

from langchain_anthropic import ChatAnthropic
from langgraph.graph import StateGraph
from pydantic import BaseModel, ConfigDict
from typing import Optional
from abc import ABC, abstractmethod
from dotenv import load_dotenv

from app.agents.models.agent_settings import AgentSettings
from app.structures.concepts.rainbow_table_color import RainbowTableColor
from app.structures.manifests.song_proposal import SongProposal
from app.util.manifest_loader import load_manifest

load_dotenv()


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




def _get_claude(self) -> ChatAnthropic:
    return ChatAnthropic(
        model_name=self.settings.anthropic_sub_model_name,
        api_key=self.settings.anthropic_api_key,
        temperature=self.settings.temperature,
        max_retries=self.settings.max_retries,
        timeout=self.settings.timeout,
        stop=self.settings.stop
    )

