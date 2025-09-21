
from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langgraph.constants import END
from langgraph.graph.state import CompiledStateGraph, StateGraph

from app.agents.base_rainbow_agent import BaseRainbowAgent
from app.agents.states.main_agent_state import MainAgentState
from app.agents.states.orange_rainbow_state import OrangeAgentState

load_dotenv()

class OrangeAgent(BaseRainbowAgent):

    """Sussex Mythologizer - 1990s local lore creator"""

    def __init__(self, **data):
        # Ensure settings are initialized if not provided
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
        self.state_graph = OrangeAgentState()


    def __call__(self, state: MainAgentState) -> MainAgentState:
        print("🧡 ORANGE AGENT: Mythologizing Sussex...")

        return state

    def create_graph(self)->StateGraph:
        graph = StateGraph(OrangeAgentState)
        graph.add_node("new_jersey_archival_research", OrangeAgent._new_jersey_archival_research)
        graph.add_node("create_fake_bands", OrangeAgent._create_fake_bands)
        graph.add_node("create_composite_friends", OrangeAgent._create_composite_friends)
        graph.add_node("find_object_fixation", OrangeAgent._find_object_fixation)
        graph.add_node("misremember_events", OrangeAgent._misremember_events)
        graph.add_node("astral_project_with_past_self_sleeve", OrangeAgent._astral_project_with_past_self_sleeve)
        graph.add_node("generate_memory_drive", OrangeAgent._generate_memory_drive)
        graph.add_edge("new_jersey_archival_research", "create_fake_bands")
        graph.add_edge("create_fake_bands", "create_composite_friends")
        graph.add_edge("create_composite_friends", "find_object_fixation")
        graph.add_edge("find_object_fixation", "misremember_events")
        graph.add_edge("misremember_events", "astral_project_with_past_self_sleeve")
        graph.add_edge("astral_project_with_past_self_sleeve", "generate_memory_drive")
        graph.add_edge("generate_memory_drive", END)
        return graph

    @staticmethod
    def _new_jersey_archival_research(state: OrangeAgentState) -> OrangeAgentState:
        """Node for research functionality"""
        # Placeholder for research logic
        if not hasattr(state, 'research_notes'):
            state.sigil_data = "Star Ledger: 1994 article on local band 'The Electric Sheep'"
        return state


    @staticmethod
    def _create_fake_bands(state: OrangeAgentState) -> OrangeAgentState:
        """Node for fake band creation functionality"""
        # Placeholder for fake band creation logic
        if not hasattr(state, 'fake_bands'):
            state.fake_bands = ["The Electric Sheep", "Neon Mirage", "The Velvet Echoes"]
        return state

    @staticmethod
    def _create_composite_friends(state: OrangeAgentState) -> OrangeAgentState:
        """Node for composite friend creation functionality"""
        # Placeholder for composite friend creation logic
        if not hasattr(state, 'composite_friends'):
            state.composite_friends = ["Alex - guitarist, loves sci-fi", "Jamie - drummer, into vintage synths"]
        return state

    @staticmethod
    def _find_object_fixation(state: OrangeAgentState) -> OrangeAgentState:
        """Node for object fixation finding functionality"""
        # Placeholder for object fixation finding logic
        if not hasattr(state, 'object_fixations'):
            state.object_fixations = ["Old cassette tapes", "Vintage band posters"]
        return state

    @staticmethod
    def _misremember_events(state: OrangeAgentState) -> OrangeAgentState:
        """Node for event misremembering functionality"""
        # Placeholder for event misremembering logic
        if not hasattr(state, 'misremembered_events'):
            state.misremembered_events = ["Concerts that never happened", "Band breakups that were just rumors"]
        return state

    @staticmethod
    def _astral_project_with_past_self_sleeve(state: OrangeAgentState) -> OrangeAgentState:
        """Node for astral projection functionality"""
        # Placeholder for astral projection logic
        if not hasattr(state, 'astral_projections'):
            state.astral_projections = ["Visions of 90s New Jersey music scene", "Encounters with past selves at gigs"]
        return state

    @staticmethod
    def _generate_memory_drive(state: OrangeAgentState) -> OrangeAgentState:
        """Node for memory drive generation functionality"""
        # Placeholder for memory drive generation logic
        if not hasattr(state, 'memory_drive'):
            state.memory_drive = {
                    "start":{
                        "latitude":40.0583,
                        "longitude":-74.4057,
                        "description":"Starting point in New Jersey"
                    },
                    "end":{
                        "latitude":40.7128,
                        "longitude":-74.0060,
                        "description":"Ending point in New York City"
                    },
            }
        return state