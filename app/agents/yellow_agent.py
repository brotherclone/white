from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langgraph.constants import END
from langgraph.graph.state import StateGraph

from app.agents.base_rainbow_agent import BaseRainbowAgent
from app.agents.states.main_agent_state import MainAgentState
from app.agents.states.violet_agent_state import VioletAgentState
from app.agents.states.yellow_agent_state import YellowAgentState

load_dotenv()

class YellowAgent(BaseRainbowAgent):

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
        graph.add_node("run_rpg_session", self._run_rpg_session)
        graph.add_node("create_room", self._create_room)
        graph.add_node("create_encounter", self._create_encounter)
        graph.add_edge("run_rpg_session", "create_room")
        graph.add_edge("create_room", "create_encounter")
        graph.add_edge("create_encounter", END)
        return graph

    @staticmethod
    def _run_rpg_session(state: YellowAgentState)-> YellowAgentState:
        """Node for RPG run"""
        # Placeholder RPG run logic
        if not hasattr(state, 'rpg_session'):
            state.rpg_session = {
                'players': [],
                'current_room': None,
                'encounters': []
            }
        return state

    @staticmethod
    def _create_room(state: YellowAgentState)-> YellowAgentState:
        """Node for creating a new room in the RPG session"""
        # Placeholder room creation logic
        if not hasattr(state, 'room'):
            state.room = {
                'description': 'A dark, eerie chamber filled with ancient artifacts.',
                'monsters': [],
                'treasures': []
            }
        return state

    @staticmethod
    def _create_encounter(state: YellowAgentState)-> YellowAgentState:
        """Node for creating an encounter in the RPG session"""
        # Placeholder encounter creation logic
        if not hasattr(state, 'encounter'):
            state.encounter = {
                'type': 'combat',
                'monsters': ['Goblin', 'Orc'],
                'treasures': ['Gold Coin', 'Magic Sword']
            }
        return state

