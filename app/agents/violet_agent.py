from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langgraph.graph import StateGraph
from langgraph.graph.state import CompiledStateGraph

from app.agents.base_rainbow_agent import BaseRainbowAgent
from app.agents.states.main_agent_state import MainAgentState
from app.agents.states.violet_agent_state import VioletAgentState

load_dotenv()
class VioletAgent(BaseRainbowAgent):

    """Mirror/Conversation Imitator - Mimics user style"""

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
        self.state_graph = VioletAgentState()

    def __call__(self, state: MainAgentState) -> MainAgentState:
        print("💜 VIOLET AGENT: Mirroring Conversation Style...")

        # Mock output for now
        state.violet_content = {
            "analyzed_style": "Creative technologist with mystical interests",
            "mirrored_response": "You know what? I've been thinking about EVP phenomena too - there's something about the way digital artifacts create their own mythology...",
            "style_elements": ["casual enthusiasm", "technical mysticism", "creative confidence"],
            "meta_commentary": "This response attempts to mirror your blend of technical expertise and mystical curiosity"
        }

        return state

    def create_graph(self) -> StateGraph:
        """Create the VioletAgent's internal workflow graph"""
        from langgraph.graph import END

        graph = StateGraph(VioletAgentState)

        # Add nodes for the VioletAgent's workflow
        graph.add_node("analyze_style", self._analyze_style_node)
        graph.add_node("mirror_response", self._mirror_response_node)
        graph.add_node("update_profile", self._update_profile_node)

        # Define the workflow: analyze style → mirror response → update profile
        graph.set_entry_point("analyze_style")
        graph.add_edge("analyze_style", "mirror_response")
        graph.add_edge("mirror_response", "update_profile")
        graph.add_edge("update_profile", END)

        return graph

    def _analyze_style_node(self, state: VioletAgentState) -> VioletAgentState:
        """Node for analyzing user conversation style"""
        if not hasattr(state, 'style_analysis'):
            state.style_analysis = "Detected: technical, creative, mystical interests"
        return state

    def _mirror_response_node(self, state: VioletAgentState) -> VioletAgentState:
        """Node for generating mirrored responses"""
        if not hasattr(state, 'mirrored_content'):
            state.mirrored_content = "Mirrored response generated based on style analysis"
        return state

    def _update_profile_node(self, state: VioletAgentState) -> VioletAgentState:
        """Node for updating psychological profile"""
        if not hasattr(state, 'psychological_profile'):
            state.psychological_profile = "Profile updated with new interaction data"
        return state
