from abc import ABC

from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langgraph.constants import START, END
from langgraph.graph import StateGraph


from app.agents.base_rainbow_agent import BaseRainbowAgent
from app.agents.states.black_agent_state import BlackAgentState
from app.agents.states.main_agent_state import MainAgentState
from app.agents.tools.magick_tools import SigilTools

load_dotenv()

class BlackAgent(BaseRainbowAgent, ABC):

    """EVP/Sigil Generator - Audio analysis that hallucinates messages"""

    def __init__(self, **data):
        if 'settings' not in data or data['settings'] is None:
            from app.agents.models.agent_settings import AgentSettings
            data['settings'] = AgentSettings()

        super().__init__(**data)

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
        self.current_session_sigils = []
        self.sigil_tools = SigilTools()
        self.state_graph = BlackAgentState()


    def __call__(self, state: MainAgentState) -> MainAgentState:


        return state

    def create_graph(self) -> StateGraph:
        """Create the BlackAgent's internal workflow graph"""

        black_workflow = StateGraph(BlackAgentState)
        black_workflow.add_node("critique", self.critique())
        black_workflow.add_node("generate_document", self.generate_document())
        black_workflow.add_node("generate_alternate_song_spec", self.generate_alternate_song_spec())
        black_workflow.add_node("contribute", self.contribute())
        black_workflow.add_edge(START,"critique")
        black_workflow.add_edge("critique","generate_document")
        black_workflow.add_edge("generate_document","critique")
        black_workflow.add_edge("generate_document","generate_alternate_song_spec")
        black_workflow.add_edge("generate_alternate_song_spec","critique")
        black_workflow.add_edge("critique",END)

        return black_workflow

    def end(self):
        pass

    def critique(self):
       #If only white proposal

       #compare my proposal to white's proposal
       pass

    def generate_document(self):
        raise NotImplementedError("Subclasses must implement generate_document method")

    def generate_alternate_song_spec(self):
        raise NotImplementedError("Subclasses must implement generate_alternate_song_spec method")

    def contribute(self, **kwargs):
        raise NotImplementedError("Subclasses must implement contribute method")

