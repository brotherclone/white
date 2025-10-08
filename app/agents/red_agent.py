from abc import ABC
from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langgraph.graph.state import StateGraph


from app.agents.base_rainbow_agent import BaseRainbowAgent
from app.agents.states.main_agent_state import MainAgentState
from app.agents.states.red_agent_state import RedAgentState

load_dotenv()

class RedAgent(BaseRainbowAgent, ABC):

    """Convoluted/Trashy Literature Generator - Baroque academic prose / Pulpy Scifi/Sexpliotations"""

    def __init__(self, **data):
        # Ensure settings are initialized if not provided
        if 'settings' not in data or data['settings'] is None:
            from app.agents.models.agent_settings import AgentSettings
            data['settings'] = AgentSettings()
        super().__init__(**data)
        self.llm = ChatAnthropic(
            temperature=self.settings.temperature,
            api_key=self.settings.anthropic_api_key,
            model_name=self.settings.anthropic_model_name,
            max_retries=self.settings.max_retries,
            timeout=self.settings.timeout,
            stop=self.settings.stop
        )
        self.state_graph = RedAgentState()

    def __call__(self, state: MainAgentState) -> MainAgentState:
        print("❤️ RED AGENT: Generating Convoluted Literature...")

        return state

    def create_graph(self) -> StateGraph:
        """Create the RedAgent's internal workflow graph"""


        graph = StateGraph(RedAgentState)

        return graph

    def generate_document(self):
        raise NotImplementedError("Subclasses must implement generate_document method")

    def generate_alternate_song_spec(self):
        raise NotImplementedError("Subclasses must implement generate_alternate_song_spec method")

    def contribute(self):
        raise NotImplementedError("Subclasses must implement contribute method")
