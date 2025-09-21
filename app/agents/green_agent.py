from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langgraph.constants import END
from langgraph.graph.state import StateGraph

from app.agents.base_rainbow_agent import BaseRainbowAgent
from app.agents.states.green_agent_state import GreenAgentState
from app.agents.states.main_agent_state import MainAgentState

load_dotenv()

class GreenAgent(BaseRainbowAgent):

    """Environmental Data Poeticizer - Converts data to poetic descriptions"""

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
        self.state_graph = GreenAgentState()


    def __call__(self, state: MainAgentState) -> MainAgentState:
        print("💚 GREEN AGENT: Poeticizing Environmental Data...")

        return state


    def create_graph(self) -> StateGraph:
        graph = StateGraph(GreenAgentState)
        graph.add_node("research_data_set", self._research_data_set)
        graph.add_node("chart_destructive_patterns", self.chart_destructive_patterns)
        graph.add_node("project_environmental_impact", self.project_environmental_impact)
        graph.add_node("list_species_extinction", self.list_species_extinction)
        graph.add_node("project_human_impact", self.project_human_impact)
        graph.add_edge("research_data_set", "chart_destructive_patterns")
        graph.add_edge("chart_destructive_patterns", "project_environmental_impact")
        graph.add_edge("project_environmental_impact", "list_species_extinction")
        graph.add_edge("list_species_extinction", "project_human_impact")
        graph.add_edge("project_human_impact", END)
        return graph

    @staticmethod
    def _research_data_set(state: GreenAgentState) -> GreenAgentState:
        """Research and gather environmental data from various sources."""
        # Placeholder for data research logic
        if not hasattr(state, 'researched_data'):
            state.researched_data = "Sample environmental data"
        return state

    @staticmethod
    def chart_destructive_patterns(state: GreenAgentState) -> GreenAgentState:
        """Analyze data to identify destructive environmental patterns."""
        # Placeholder for pattern analysis logic
        if not hasattr(state, 'destructive_patterns'):
            state.destructive_patterns = "Identified destructive patterns"
        return state

    @staticmethod
    def project_environmental_impact(state: GreenAgentState) -> GreenAgentState:
        """Project future environmental impacts based on current data."""
        # Placeholder for impact projection logic
        if not hasattr(state, 'environmental_impact'):
            state.environmental_impact = "Projected environmental impact"
        return state

    @staticmethod
    def list_species_extinction(state: GreenAgentState) -> GreenAgentState:
        """List species at risk of extinction based on environmental data."""
        # Placeholder for species extinction logic
        if not hasattr(state, 'species_extinction'):
            state.species_extinction = "List of species at risk of extinction"
        return state

    @staticmethod
    def project_human_impact(state: GreenAgentState) -> GreenAgentState:
        """Project human impact on the environment based on current trends."""
        # Placeholder for human impact projection logic
        if not hasattr(state, 'human_impact'):
            state.human_impact = "Projected human impact on the environment"
        return state