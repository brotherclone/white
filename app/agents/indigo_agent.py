import os
from abc import ABC

from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langgraph.graph.state import CompiledStateGraph, StateGraph

from app.agents.base_rainbow_agent import BaseRainbowAgent
from app.agents.states.indigo_agent_state import IndigoAgentState
from app.agents.states.white_agent_state import MainAgentState

load_dotenv()

class IndigoAgent(BaseRainbowAgent, ABC):

    """Anagram/Hidden Pattern Decoder - Finds hidden information"""

    def __init__(self, **data):
        # Ensure settings are initialized if not provided
        if 'settings' not in data or data['settings'] is None:
            from app.agents.models.agent_settings import AgentSettings
            data['settings'] = AgentSettings()

        super().__init__(**data)

        # Verify settings are properly initialized
        if self.settings is None:
            from app.agents.models.agent_settings import AgentSettings
            self.settings = AgentSettings()

        self.llm = ChatAnthropic(
            temperature=self.settings.temperature,
            api_key=self.settings.anthropic_api_key,
            model_name=self.settings.anthropic_model_name,
            max_retries=self.settings.max_retries,
            timeout=self.settings.timeout,
            stop=self.settings.stop
        )
        self.state_graph = IndigoAgentState()

    def __call__(self, state: MainAgentState) -> MainAgentState:
        print("💜 INDIGO AGENT: Decoding Hidden Patterns...")
        return state

    def create_graph(self) -> StateGraph:
        """Create the IndigoAgent's internal workflow graph"""
        from langgraph.graph import END

        graph = StateGraph(IndigoAgentState)

        return graph

    def generate_document(self):
        raise NotImplementedError("Subclasses must implement generate_document method")

    def generate_alternate_song_spec(self):
        raise NotImplementedError("Subclasses must implement generate_alternate_song_spec method")

    def contribute(self):
        raise NotImplementedError("Subclasses must implement contribute method")

    def export_chain_artifacts(self, state: IndigoAgentState) -> IndigoAgentState:
        for artifact in state.artifacts:
            print(artifact)
        return state
