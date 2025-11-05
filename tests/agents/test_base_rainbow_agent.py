from types import SimpleNamespace

import pytest

from app.structures.agents.base_rainbow_agent import (BaseRainbowAgent,
                                                      skip_chance)
from app.structures.artifacts.base_chain_artifact import ChainArtifact

# A concrete agent to instantiate and test abstract behavior
GRAPH_SENTINEL = object()


class ConcreteAgent(BaseRainbowAgent):
    def create_graph(self):
        return GRAPH_SENTINEL

    def generate_alternate_song_spec(self, agent_state):
        # simple pass-through for testing
        return self.graph

    def export_chain_artifacts(self, agent_state):
        return self.graph


def test_base_is_abstract():
    # Attempting to instantiate the abstract BaseRainbowAgent should raise TypeError
    with pytest.raises(TypeError):
        BaseRainbowAgent()  # abstract methods prevent instantiation


def test_create_graph_called_and_assigned():
    agent = ConcreteAgent()
    assert agent.graph is GRAPH_SENTINEL


def test__get_claude_uses_settings(monkeypatch):
    # Prepare a dummy settings object with required attributes
    settings = SimpleNamespace(
        anthropic_sub_model_name="claude-1",
        anthropic_api_key="test-key",
        temperature=0.7,
        max_retries=2,
        timeout=30,
        stop=["\n"],
    )

    # Fake ChatAnthropic that captures init kwargs
    captured = {}

    class FakeChatAnthropic:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setattr(
        "app.structures.agents.base_rainbow_agent.ChatAnthropic", FakeChatAnthropic
    )

    agent = ConcreteAgent.model_construct(settings=settings)
    result = agent._get_claude()

    assert isinstance(result, FakeChatAnthropic)
    assert captured["model_name"] == settings.anthropic_sub_model_name
    assert captured["api_key"] == settings.anthropic_api_key
    assert captured["temperature"] == settings.temperature
    assert captured["max_retries"] == settings.max_retries
    assert captured["timeout"] == settings.timeout
    assert captured["stop"] == settings.stop


def test_chain_artifacts_instance_is_independent():
    a1 = ConcreteAgent()
    a2 = ConcreteAgent()

    # modify one instance's chain_artifacts and ensure the other is unaffected
    artifact = ChainArtifact(chain_artifact_type="test-artifact")
    a1.chain_artifacts.append(artifact)
    assert len(a1.chain_artifacts) == 1
    assert a1.chain_artifacts[0].chain_artifact_type == "test-artifact"
    assert a2.chain_artifacts == []


class _TestAgent:
    @skip_chance(1.0, rng=lambda: 0.0)  # force skip
    def node_to_skip(self, state):
        # would mutate state if not skipped
        state.was_run = True
        return state


def test_skip_method_records_and_skips():
    state = SimpleNamespace()
    agent = _TestAgent()
    result = agent.node_to_skip(state)
    assert result is state
    assert not hasattr(state, "was_run")
    assert getattr(state, "skipped_nodes") == ["node_to_skip"]
