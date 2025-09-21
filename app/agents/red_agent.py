from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langgraph.graph.state import CompiledStateGraph, StateGraph


from app.agents.base_rainbow_agent import BaseRainbowAgent
from app.agents.states.main_agent_state import MainAgentState
from app.agents.states.red_agent_state import RedAgentState

load_dotenv()

class RedAgent(BaseRainbowAgent):

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
        from langgraph.graph import END

        graph = StateGraph(RedAgentState)

        # Add nodes for the RedAgent's workflow
        graph.add_node("generate_baroque_title", self._generate_baroque_title_node)
        graph.add_node("write_academic_content", self._write_academic_content_node)
        graph.add_node("create_citations", self._create_citations_node)

        # Define the workflow: generate title → write content → create citations
        graph.set_entry_point("generate_baroque_title")
        graph.add_edge("generate_baroque_title", "write_academic_content")
        graph.add_edge("write_academic_content", "create_citations")
        graph.add_edge("create_citations", END)

        return graph

    def _generate_baroque_title_node(self, state: RedAgentState) -> RedAgentState:
        """Node for generating baroque academic titles"""
        if not hasattr(state, 'baroque_title'):
            state.baroque_title = "The Phenomenological Apparatus of Spectral Significances: A Derridean Framework"
        return state

    def _write_academic_content_node(self, state: RedAgentState) -> RedAgentState:
        """Node for writing convoluted academic content"""
        if not hasattr(state, 'academic_pages'):
            state.academic_pages = [
                "The recursive temporalities embedded within spectral analysis necessitate a fundamental reconsideration...",
                "Through poststructural hermeneutics, we observe the persistent emergence of EVP significances..."
            ]
        return state

    def _create_citations_node(self, state: RedAgentState) -> RedAgentState:
        """Node for creating academic citations"""
        if not hasattr(state, 'citations'):
            state.citations = ["Derrida, J. (1967). Of Grammatology", "Foucault, M. (1969). The Archaeology of Knowledge"]
        return state
