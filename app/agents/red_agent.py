import os
import random
import uuid
import yaml
import logging

from abc import ABC
from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain_core.runnables import RunnableConfig
from langgraph.constants import START, END
from langgraph.graph import StateGraph

from app.agents.base_rainbow_agent import BaseRainbowAgent
from app.agents.enums.chain_artifact_file_type import ChainArtifactFileType
from app.agents.models.agent_settings import AgentSettings
from app.agents.models.book_artifact import ReactionBookArtifact, BookArtifact
from app.agents.models.book_data import BookDataPageCollection
from app.agents.models.text_chain_artifact_file import TextChainArtifactFile
from app.agents.states.white_agent_state import MainAgentState
from app.agents.states.red_agent_state import RedAgentState
from app.structures.concepts.rainbow_table_color import the_rainbow_table_colors
from app.structures.manifests.song_proposal import SongProposalIteration, SongProposal
from app.agents.tools.book_tool import BookMaker
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
            main_generated_book=None,
            artifacts=[],
            should_respond_with_reaction_book=False,
            reaction_level=0
        )

    def __call__(self, state: MainAgentState) -> MainAgentState:

        """Entry point when White Agent invokes Red Agent"""

        current_proposal = state.song_proposals.iterations[-1]
        red_state = RedAgentState(
            song_proposals=state.song_proposals,
            black_to_white_proposal=current_proposal,
            counter_proposal=None,
            artifacts=[],
            should_respond_with_reaction_book=False,
            reaction_level=0
        )
        red_config: RunnableConfig = {"configurable": {"thread_id": f"{state.thread_id}"}}
        result = self._compiled_workflow.invoke(red_state.model_dump(), config=red_config)
        snapshot = self._compiled_workflow.get_state(red_config)
        if snapshot.next:
            pass
        else:
            state.song_proposals = result.get("song_proposals") or state.song_proposals
            if result.get("counter_proposal"):
                state.song_proposals.iterations.append(result["counter_proposal"])
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
        else:
            # Prompt to generate a counter-proposal using the book
            raise NotImplementedError("Subclasses must implement generate_alternate_song_spec method")

    def generate_book(self, state: RedAgentState) -> RedAgentState:
        mock_mode = os.getenv("MOCK_MODE", "false").lower() == "true"
        if mock_mode:
            with open("/Volumes/LucidNonsense/White/app/agents/mocks/red_book_artifact_mock.yml", "r") as f:
                data = yaml.safe_load(f)
                book = ReactionBookArtifact(**data)
                state.artifacts.append(book)
                return state
        else:
            book_data = BookMaker.generate_random_book()
            prompt = f"""
            You are the Light Reader, a hermitic keeper of books rare and unusual. For you, these books are your
            only means of communicating with the outside world. Like a man trapped in the proverbial Chinese Box
            you don't know what the texts mean, nor what effect they illicit in their readers. But for you each
            book magically sneak into the world is a way of validating your existence. Do your best to speak in the
            voice of the provided author in their style and convey their message and domain expertise. You don't know
            where the titles come from just that if you write they will your book will somehow find its way into the 
            world. You feel a door has opened, the stars align and now here is the book you should write:
            
            {BookMaker.format_card_catalog(book_data)}
            
            Start by outlining the basic structure of the book. Then choose two sections and, in the suggested
            style write the fourth and fifth pages in those sections.
            
            Format your pages in the following way:
            
            page_1.text_content: "This is the first page of the book."
            page_2.text_content: "This is the second page of the book."
            
            """
            claude = self._get_claude()
            proposer = claude.with_structured_output(BookDataPageCollection)
            try:
                result = proposer.invoke(prompt)
                if isinstance(result, dict):
                    state.main_generated_book = BookArtifact(
                        book_data=book_data,
                        excerpts=[result["page_1"], result["page_2"]],
                        thread_id=state.thread_id
                    )
                else:
                    state.main_generated_book = BookArtifact(
                        book_data=book_data,
                        excerpts=[],
                        thread_id=state.thread_id
                    )
                    # ToDo: Dump yml and make MD
            except Exception as e:
                logging.error(f"Anthropic model call failed: {e!s}")
            return state


    def generate_reaction_book(self, state: RedAgentState) -> RedAgentState:
        mock_mode = os.getenv("MOCK_MODE", "false").lower() == "true"
        if mock_mode:
            with open("/Volumes/LucidNonsense/White/app/agents/mocks/red_book_reaction_artifact_mock.yml", "r") as f:
                data = yaml.safe_load(f)
                reaction_book = ReactionBookArtifact(**data)
                state.artifacts.append(reaction_book)
                state.reaction_level += 1
                return state
        else:
            # Grab the book artifact
            # Make a new reaction book artifact which is usually literary criticism, but sometimes parody, or reimagining
            # Prompt to write pages
            # Parse pages and book to text chain artifacts
            # Add book artifact to state
            # Dump yml and make MD
            # Increment reaction level
            return state

    def evaluate_books_versus_proposals(self, state: RedAgentState) -> RedAgentState:
        mock_mode = os.getenv("MOCK_MODE", "false").lower() == "true"
        if mock_mode:
            pass
        # Look at the current proposal and synthesize the books and reaction books
        # Decide if the books are enough to address the proposal
        # Update state with evaluation results
        return state

    def route_after_evaluate_books_versus_proposals(self) -> str:
        mock_mode = os.getenv("MOCK_MODE", "false").lower() == "true"
        if mock_mode:
            choices = ["new_book", "reaction_book", "done"]
            return random.choice(choices)
        else:
            # Bail if the reaction level is too high
            raise NotImplementedError("Subclasses must implement route_after_evaluate_books_versus_proposals method")

    def export_chain_artifacts(self, state: RedAgentState) -> RedAgentState:
        for artifact in state.artifacts:
            print(artifact)
        return state