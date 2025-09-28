from abc import ABC
from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langgraph.graph.state import StateGraph

from app.agents.base_rainbow_agent import BaseRainbowAgent
from app.agents.states.main_agent_state import MainAgentState
from app.agents.states.violet_agent_state import VioletAgentState
from app.agents.states.yellow_agent_state import YellowAgentState

load_dotenv()

class YellowAgent(BaseRainbowAgent, ABC):

    """Pulsar Palace RPG Runner - Automated RPG sessions"""

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
        self.state_graph = YellowAgentState()

    def __call__(self, state: MainAgentState) -> MainAgentState:
        print("💛 YELLOW AGENT: Running RPG Session...")

        return state

    def create_graph(self) -> StateGraph:
        graph = StateGraph(VioletAgentState)
        return graph
    def generate_document(self):
        raise NotImplementedError("Subclasses must implement generate_document method")

    def generate_alternate_song_spec(self):
        raise NotImplementedError("Subclasses must implement generate_alternate_song_spec method")

    def contribute(self):
        raise NotImplementedError("Subclasses must implement contribute method")
