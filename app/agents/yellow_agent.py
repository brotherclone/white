import logging
import os
import time
import uuid
from abc import ABC

import yaml
from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langgraph.graph.state import StateGraph

from app.structures.agents.base_rainbow_agent import BaseRainbowAgent
from app.agents.states.white_agent_state import MainAgentState
from app.agents.states.violet_agent_state import VioletAgentState
from app.agents.states.yellow_agent_state import YellowAgentState
from app.structures.agents.base_rainbow_agent_state import BaseRainbowAgentState
from app.structures.concepts.rainbow_table_color import the_rainbow_table_colors
from app.structures.manifests.song_proposal import SongProposalIteration
from app.util.manifest_loader import get_my_reference_proposals

load_dotenv()

class YellowAgent(BaseRainbowAgent, ABC):

    """Pulsar Palace RPG Runner - Automated RPG sessions"""

    def __init__(self, **data):
        # Ensure settings are initialized if not provided
        if 'settings' not in data or data['settings'] is None:
            from app.structures.agents.agent_settings import AgentSettings
            data['settings'] = AgentSettings()
        super().__init__(**data)
        # Verify settings are properly initialized
        if self.settings is None:
            from app.structures.agents.agent_settings import AgentSettings
            self.settings = AgentSettings()
        self.llm = ChatAnthropic(
            temperature=self.settings.temperature,
            api_key=self.settings.anthropic_api_key,
            model_name=self.settings.anthropic_model_name,
            max_retries=self.settings.max_retries,
            timeout=self.settings.timeout,
            stop=self.settings.stop
        )

    def __call__(self, state: MainAgentState) -> MainAgentState:
        print("💛 YELLOW AGENT: Running RPG Session...")

        return state

    def create_graph(self) -> StateGraph:
        graph = StateGraph(VioletAgentState)
        return graph

    def generate_alternate_song_spec(self, state: YellowAgentState) -> YellowAgentState:
        mock_mode = os.getenv("MOCK_MODE", "false").lower() == "true"
        if mock_mode:
            with open("/tests/mocks/green_counter_proposal_mock.yml", "r") as f:
                data = yaml.safe_load(f)
            counter_proposal = SongProposalIteration(**data)
            state.counter_proposal = counter_proposal
            return state
        else:

            prompt = f"""
                          Current song proposal:
                          {state.white_proposal}

                          Reference works in this artist's style paying close attention to 'concept' property:
                          {get_my_reference_proposals('Y')}

                          In your counter proposal your 'rainbow_color' property should always be:
                          {the_rainbow_table_colors['Y']}
               """

            claude = self._get_claude()
            proposer = claude.with_structured_output(SongProposalIteration)

            try:
                result = proposer.invoke(prompt)
                if isinstance(result, dict):
                    counter_proposal = SongProposalIteration(**result)
                else:
                    counter_proposal = result
            except Exception as e:
                logging.error(f"Anthropic model call failed: {e!s}")
                timestamp = int(time.time() * 1000)
                counter_proposal = SongProposalIteration(
                    iteration_id=f"fallback_error_{timestamp}",
                    bpm=110.33,
                    tempo="4/4",
                    key="E Major",
                    rainbow_color="yellow",
                    title="Fallback: Yellow Song",
                    mood=["mind-melting"],
                    genres=["electronic"],
                    concept="Fallback stub because Anthropic model unavailable"
                )

            state.counter_proposal = counter_proposal
            return state