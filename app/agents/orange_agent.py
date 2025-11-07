import logging
import os
import time
from abc import ABC

import yaml
from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langgraph.graph.state import StateGraph

from app.agents.states.orange_rainbow_state import OrangeAgentState
from app.agents.states.white_agent_state import MainAgentState
from app.structures.agents.base_rainbow_agent import BaseRainbowAgent
from app.structures.concepts.rainbow_table_color import the_rainbow_table_colors
from app.structures.manifests.song_proposal import SongProposalIteration
from app.util.manifest_loader import get_my_reference_proposals

load_dotenv()


class OrangeAgent(BaseRainbowAgent, ABC):
    """Sussex Mythologizer - 1990s local lore creator"""

    def __init__(self, **data):
        # Ensure settings are initialized if not provided
        if "settings" not in data or data["settings"] is None:
            from app.structures.agents.agent_settings import AgentSettings

            data["settings"] = AgentSettings()
        super().__init__(**data)
        if self.settings is None:
            from app.structures.agents.agent_settings import AgentSettings

            self.settings = AgentSettings()
        self.llm = ChatAnthropic(
            temperature=self.settings.temperature,
            api_key=self.settings.anthropic_api_key,
            model_name=self.settings.anthropic_model_name,
            max_retries=self.settings.max_retries,
            timeout=self.settings.timeout,
            stop=self.settings.stop,
        )

    def __call__(self, state: MainAgentState) -> MainAgentState:
        print("🧡 ORANGE AGENT: Mythologizing Sussex...")

        return state

    def create_graph(self) -> StateGraph:
        graph = StateGraph(OrangeAgentState)

        return graph

    def generate_alternate_song_spec(self, state: OrangeAgentState) -> OrangeAgentState:
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
                          {get_my_reference_proposals('O')}

                          In your counter proposal your 'rainbow_color' property should always be:
                          {the_rainbow_table_colors['O']}
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
                    bpm=110,
                    tempo="4/4",
                    key="D Major",
                    rainbow_color="orange",
                    title="Fallback: Orange Song",
                    mood=["nostalgic"],
                    genres=["alternative"],
                    concept="Fallback stub because Anthropic model unavailable",
                )

            state.counter_proposal = counter_proposal
            return state
