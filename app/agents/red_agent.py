import os
import uuid
import yaml
import logging

from abc import ABC
from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langgraph.checkpoint.memory import MemorySaver
from langgraph.constants import START, END
from langgraph.graph import StateGraph

from app.agents.base_rainbow_agent import BaseRainbowAgent
from app.agents.models.agent_settings import AgentSettings
from app.agents.states.base_rainbow_agent_state import BaseRainbowAgentState
from app.agents.states.main_agent_state import MainAgentState
from app.agents.states.red_agent_state import RedAgentState
from app.structures.concepts.rainbow_table_color import the_rainbow_table_colors
from app.structures.manifests.song_proposal import SongProposalIteration, SongProposal
from app.util.manifest_loader import get_my_reference_proposals

load_dotenv()
logging.basicConfig(level=logging.INFO)

class RedAgent(BaseRainbowAgent, ABC):

    """The Light Reader."""

    def __init__(self, **data):
        if 'settings' not in data or data['settings'] is None:
            data['settings'] = AgentSettings()

        super().__init__(**data)

        if self.settings is None:
            self.settings = AgentSettings()

        self.llm = ChatAnthropic(
            temperature=self.settings.temperature,
            api_key=self.settings.anthropic_api_key,
            model_name=self.settings.anthropic_model_name,
            max_retries=self.settings.max_retries,
            timeout=self.settings.timeout,
            stop=self.settings.stop
        )
        self.state_graph = RedAgentState(
            thread_id=f"red_thread_{uuid.uuid4()}",
            song_proposals=None,
            black_to_white_proposal=None,
            counter_proposal=None,
            artifacts=[],
            pending_human_tasks=[],
            awaiting_human_action=False
        )

    def __call__(self, state: MainAgentState) -> MainAgentState:

        """Entry point when White Agent invokes Red Agent"""
        current_proposal = state.song_proposals.iterations[-1]

        red_agent_state = RedAgentState(
            black_to_white_proposal=current_proposal,
            song_proposals=state.song_proposals,
            thread_id=state.thread_id,
            artifacts=[],
        )
        if not hasattr(self, '_compiled_workflow'):
            self._compiled_workflow = self.create_graph().compile(
                checkpointer=MemorySaver(),
                interrupt_before=["await_human_action"]
            )
        red_config = {"configurable": {"thread_id": f"{state.thread_id}"}}
        snapshot = self._compiled_workflow.get_state(red_config)
        if snapshot.next:
            pass
        else:
            pass
        return state

    def create_graph(self) -> StateGraph:


        graph = StateGraph(RedAgentState)

        return graph


    def generate_alternate_song_spec(self, agent_state: RedAgentState) -> RedAgentState:
        raise NotImplementedError("Subclasses must implement generate_alternate_song_spec method")