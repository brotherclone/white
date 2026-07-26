import pytest
from langchain_core.exceptions import OutputParserException
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableLambda
from pydantic import BaseModel

from white_core.agents.base_rainbow_agent import BaseRainbowAgent


class DummySchema(BaseModel):
    text: str


class FakeLLM:
    """Stands in for ChatAnthropic: records the tool_choice it was bound with
    and the prompts it was invoked with, and returns responses from a queue
    (one per call) so retry behavior can be exercised without a live API call."""

    def __init__(self, responses: list[AIMessage]):
        self._responses = list(responses)
        self.bind_tools_kwargs = None
        self.prompts_seen: list = []

    def bind_tools(self, tools, **kwargs):
        self.bind_tools_kwargs = kwargs
        return RunnableLambda(self._invoke)

    def _invoke(self, prompt):
        self.prompts_seen.append(prompt)
        return self._responses.pop(0)


def _tool_call_response(text: str) -> AIMessage:
    return AIMessage(
        content="",
        tool_calls=[{"name": "DummySchema", "args": {"text": text}, "id": "1"}],
    )


def _no_tool_call_response() -> AIMessage:
    return AIMessage(content="just talked instead of calling the tool", tool_calls=[])


def test_invoke_structured_does_not_force_tool_choice():
    llm = FakeLLM([_tool_call_response("hi")])

    result = BaseRainbowAgent._invoke_structured(llm, DummySchema, "prompt")

    assert isinstance(result, DummySchema)
    assert result.text == "hi"
    # This is the actual bug being fixed: with_structured_output() forces
    # tool_choice to a specific tool name, which conflicts with server-side
    # adaptive thinking and truncates the tool call. "auto" avoids that.
    assert llm.bind_tools_kwargs == {"tool_choice": "auto"}


def test_invoke_structured_raises_when_model_always_skips_the_tool():
    llm = FakeLLM([_no_tool_call_response()] * 3)

    with pytest.raises(OutputParserException):
        BaseRainbowAgent._invoke_structured(llm, DummySchema, "prompt", max_attempts=3)

    assert len(llm.prompts_seen) == 3


def test_invoke_structured_states_the_tool_requirement_up_front():
    """Measured directly against the LastHumanArtifact schema: an unframed
    prompt missed the tool call 1-in-3 times; stating the requirement up
    front (not just as a retry nudge) brought that to 0-in-6. So attempt 1
    must already carry the framing, not just later retries."""
    llm = FakeLLM([_tool_call_response("hi")])

    BaseRainbowAgent._invoke_structured(llm, DummySchema, "write a book page")

    assert len(llm.prompts_seen) == 1
    assert "must respond by calling the DummySchema tool" in llm.prompts_seen[0]
    assert llm.prompts_seen[0].endswith("write a book page")


def test_invoke_structured_retries_with_reinforcement_and_succeeds():
    """The model skipping the tool on the first pass (common with tool_choice
    "auto" on prose-style prompts) should be recovered by a reinforced retry
    rather than immediately falling back to generic stub content."""
    llm = FakeLLM([_no_tool_call_response(), _tool_call_response("recovered")])

    result = BaseRainbowAgent._invoke_structured(llm, DummySchema, "write a book page")

    assert result.text == "recovered"
    assert len(llm.prompts_seen) == 2
    assert "must respond by calling the DummySchema tool" in llm.prompts_seen[0]
    assert "did not call the DummySchema tool" in llm.prompts_seen[1]
    assert llm.prompts_seen[1].startswith("write a book page")
