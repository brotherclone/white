from abc import ABC, abstractmethod
from typing import Callable, List, Optional, TypeVar, Union

from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain_core.exceptions import OutputParserException
from langchain_core.messages import HumanMessage
from langchain_core.output_parsers.openai_tools import PydanticToolsParser
from langgraph.graph import StateGraph
from pydantic import BaseModel, ConfigDict

from white_core.agents.agent_settings import AgentSettings
from white_core.agents.base_rainbow_agent_state import BaseRainbowAgentState
from white_core.artifacts.base_artifact import ChainArtifact

StructuredSchema = TypeVar("StructuredSchema", bound=BaseModel)

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
            max_retries=self.settings.max_retries,
            timeout=self.settings.timeout,
            stop=self.settings.stop,
            max_tokens=self.settings.max_tokens,
        )

    @staticmethod
    def _invoke_structured(
        llm: ChatAnthropic,
        schema: type[StructuredSchema],
        prompt,
        max_attempts: int = 3,
    ) -> StructuredSchema:
        """Invoke `llm` for `schema`-shaped output without forcing tool_choice.

        `ChatAnthropic.with_structured_output()` forces tool_choice to the
        target tool. Models with server-side adaptive thinking (e.g.
        claude-fable-5) can't be forced into a tool call while thinking, and
        silently return a tool call with truncated or empty arguments instead
        of erroring — the resulting object then fails Pydantic validation
        with every field reported missing. Binding the tool with
        tool_choice="auto" instead lets the model think before deciding to
        call it, which avoids the truncation.

        The tradeoff of "auto" is that prompts written as prose ("write two
        pages...") rather than an explicit tool-call instruction sometimes
        get a prose answer back with no tool call at all, since most of
        these prompts predate "auto" and were tuned for the old forced
        tool_choice. This is worse on schemas with many fields and
        narrative-heavy prompts — measured as high as a 1-in-3 miss rate on
        LastHumanArtifact with no framing, dropping to 0-in-6 once the
        tool-call requirement was stated up front rather than left implicit.
        So state it up front on every attempt, escalating on retry.
        """
        framing = (
            f"You must respond by calling the {schema.__name__} tool with your "
            "answer as its arguments. Do not write the answer as plain text.\n\n"
        )
        reinforcement = (
            f"You did not call the {schema.__name__} tool last time. You must "
            "respond by calling it now, with your answer as its arguments — "
            "not as plain text."
        )
        tool_llm = llm.bind_tools([schema], tool_choice="auto")
        attempt_input = (
            f"{framing}{prompt}"
            if isinstance(prompt, str)
            else [HumanMessage(content=framing), *prompt]
        )
        for attempt in range(max_attempts):
            response = tool_llm.invoke(attempt_input)
            if response.tool_calls:
                return PydanticToolsParser(tools=[schema], first_tool_only=True).invoke(
                    response
                )
            attempt_input = (
                f"{prompt}\n\n{reinforcement}"
                if isinstance(prompt, str)
                else [*prompt, HumanMessage(content=reinforcement)]
            )
        raise OutputParserException(
            f"Model did not call the {schema.__name__} tool after "
            f"{max_attempts} attempts"
        )

    @staticmethod
    def _extract_text(content) -> str:
        if isinstance(content, list):
            return "".join(
                (
                    block.get("text", "")
                    if isinstance(block, dict)
                    else getattr(block, "text", str(block))
                )
                for block in content
            )
        return str(content) if content is not None else ""


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
