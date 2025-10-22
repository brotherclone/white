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
from app.agents.states.white_agent_state import MainAgentState
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
            should_respond_with_reaction_book=False,
            reaction_level=0
        )

    def __call__(self, state: MainAgentState) -> MainAgentState:

        """Entry point when White Agent invokes Red Agent"""

        return state

    def create_graph(self) -> StateGraph:
        """
        1. Generate book
        2. Generate a counterproposal based on the current book
        3. Evaluate book and proposal
        4. Route - write a reaction book then new proposal | new proposal | finish
        :return:
        """

        red_workflow = StateGraph(RedAgentState)
        red_workflow.add_node("generate_book", self.generate_book)
        red_workflow.add_node("generate_alternate_song_spec", self.generate_alternate_song_spec)
        red_workflow.add_node("evaluate_books_versus_proposals", self.evaluate_books_versus_proposals)
        red_workflow.add_node("generate_reaction_book", self.generate_book)

        red_workflow.add_edge(START, "generate_book")
        red_workflow.add_edge("generate_book", "generate_alternate_song_spec")
        red_workflow.add_edge("generate_alternate_song_spec", "evaluate_books_versus_proposals")
        red_workflow.add_conditional_edges(
            "evaluate_books_versus_proposals",
            self.route_after_evaluate_books_versus_proposals,
            {
                "new_book": "generate_book",
                "reaction_book": "generate_reaction_book",
                "done": END
            }
        )
        return red_workflow


    def generate_alternate_song_spec(self, state: RedAgentState) -> RedAgentState:

        mock_mode = os.getenv("MOCK_MODE", "false").lower() == "true"

        if mock_mode:
            with open("/Volumes/LucidNonsense/White/app/agents/mocks/red_counter_proposal_mock.yml", "r") as f:
                data = yaml.safe_load(f)
                counter_proposal = SongProposalIteration(**data)
                state.counter_proposal = counter_proposal
                return state
        raise NotImplementedError("Subclasses must implement generate_alternate_song_spec method")

    def generate_book(self, state: RedAgentState) -> RedAgentState:
        mock_mode = os.getenv("MOCK_MODE", "false").lower() == "true"

        if mock_mode:
            pass
        return state

    def _generate_book_outline(self):
        pass

    def _generate_book_page(self):
        pass

    def generate_reaction_book(self, state: RedAgentState) -> RedAgentState:
        mock_mode = os.getenv("MOCK_MODE", "false").lower() == "true"

        if mock_mode:
            pass
        return state

    def evaluate_books_versus_proposals(self, state: RedAgentState) -> RedAgentState:
        mock_mode = os.getenv("MOCK_MODE", "false").lower() == "true"

        if mock_mode:
            pass
        return state

    def route_after_evaluate_books_versus_proposals(self, state: RedAgentState) -> str:
        mock_mode = os.getenv("MOCK_MODE", "false").lower() == "true"

        if mock_mode:
            pass
        return "done"