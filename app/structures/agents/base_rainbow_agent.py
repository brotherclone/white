from abc import ABC, abstractmethod
from typing import Callable, List, Optional, Union

from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langgraph.graph import StateGraph
from pydantic import BaseModel, ConfigDict

from app.structures.agents.agent_settings import AgentSettings
from app.structures.agents.base_rainbow_agent_state import BaseRainbowAgentState
from app.structures.artifacts.base_artifact import ChainArtifact

load_dotenv()

Chance = Union[float, Callable[[object], float]]


class BaseRainbowAgent(BaseModel, ABC):
    """Base class for all Rainbow Agents"""

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="allow")

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
    def generate_alternate_song_spec(
        self, agent_state: BaseRainbowAgentState
    ) -> BaseRainbowAgentState:
        raise NotImplementedError(
            "Subclasses must implement generate_alternate_song_spec method"
        )

    def _get_claude(self) -> ChatAnthropic:
        return ChatAnthropic(
            model_name=self.settings.anthropic_sub_model_name,
            api_key=self.settings.anthropic_api_key,
            temperature=self.settings.temperature,
            max_retries=self.settings.max_retries,
            timeout=self.settings.timeout,
            stop=self.settings.stop,
            max_tokens=self.settings.max_tokens,
        )


def skip_chance(chance, rng=None):
    rng = rng or __import__("random").random

    def decorator(fn):
        from functools import wraps

        @wraps(fn)
        def wrapper(self, state, *args, **kwargs):
            p = chance(self) if callable(chance) else chance
            if rng() < p:
                skipped = getattr(state, "skipped_nodes", [])
                skipped.append(fn.__name__)
                setattr(state, "skipped_nodes", skipped)
                return state
            return fn(self, state, *args, **kwargs)

        return wrapper

    return decorator
