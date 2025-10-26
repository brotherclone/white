import os
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
from app.agents.models.book_data import BookDataPageCollection, BookData
from app.agents.models.text_chain_artifact_file import TextChainArtifactFile
from app.agents.states.white_agent_state import MainAgentState
from app.agents.states.red_agent_state import RedAgentState
from app.agents.tools.text_tools import save_artifact_to_md
from app.structures.concepts.rainbow_table_color import the_rainbow_table_colors
from app.structures.concepts.yes_or_no import YesOrNo
from app.structures.manifests.song_proposal import SongProposalIteration, SongProposal
from app.agents.tools.book_tool import BookMaker
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
            should_create_book=True,
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

        red_workflow = StateGraph(RedAgentState)
        red_workflow.add_node("generate_book", self.generate_book)
        red_workflow.add_node("generate_alternate_song_spec", self.generate_alternate_song_spec)
        red_workflow.add_node("evaluate_books_versus_proposals", self.evaluate_books_versus_proposals)
        red_workflow.add_node("generate_reaction_book", self.generate_book)
        red_workflow.add_node("write_reaction_book_pages", self.write_reaction_book_pages)

        red_workflow.add_edge(START, "generate_book")
        red_workflow.add_edge("generate_book", "generate_alternate_song_spec")
        red_workflow.add_edge("generate_reaction_book", "write_reaction_book_pages")
        red_workflow.add_edge("write_reaction_book_pages", "evaluate_books_versus_proposals")
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
        else:
            summary = self._format_books_for_prompt(state)

            prompt = f"""
            
            Your Light Reading book plus reactions to your work in from the literature:
            {summary}
    
            Current song proposal:
            {state.white_proposal}
    
            Reference works in this artist's style paying close attention to 'concept' property:
            {get_my_reference_proposals('R')}
            
            In your counter proposal your 'rainbow_color' property should always be:
            {the_rainbow_table_colors['R']}
    
           
            """

            claude = self._get_claude()
            proposer = claude.with_structured_output(SongProposalIteration)
            try:
                result = proposer.invoke(prompt)
                if isinstance(result, dict):
                    counter_proposal = SongProposalIteration(**result)
                    state.song_proposals.iterations.append(self.counter_proposal)
                    state.counter_proposal = counter_proposal
                else:
                    state.counter_proposal = None
            except Exception as e:
                logging.error(f"Anthropic model call failed: {e!s}")
        return state

    def generate_book(self, state: RedAgentState) -> RedAgentState:
        mock_mode = os.getenv("MOCK_MODE", "false").lower() == "true"
        if mock_mode:
            with open("/Volumes/LucidNonsense/White/app/agents/mocks/red_book_artifact_mock.yml", "r") as f:
                data = yaml.safe_load(f)
                book = ReactionBookArtifact(**data)
                state.artifacts.append(book)
                state.should_create_book = False
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
                    page_1 = TextChainArtifactFile(
                        text_content=result["page_1"],
                        thread_id=state.thread_id,
                        rainbow_color=the_rainbow_table_colors['R'],
                        base_path=os.getenv("AGENT_WORK_PRODUCT_BASE_PATH"),
                        chain_artifact_file_type=ChainArtifactFileType.MARKDOWN,
                        artifact_name=f"{book_data.author}_excerpt_1",
                    )
                    save_artifact_to_md(page_1)
                    page_2 = TextChainArtifactFile(
                        text_content=result["page_2"],
                        thread_id=state.thread_id,
                        rainbow_color=the_rainbow_table_colors['R'],
                        base_path=os.getenv("AGENT_WORK_PRODUCT_BASE_PATH"),
                        chain_artifact_file_type=ChainArtifactFileType.MARKDOWN,
                        artifact_name=f"{book_data.author}_excerpt_2",
                    )
                    save_artifact_to_md(page_2)
                    state.main_generated_book = BookArtifact(
                        book_data=book_data,
                        excerpts=[page_1, page_2],
                        thread_id=state.thread_id
                    )
                    state.should_create_book = False
                else:
                    state.main_generated_book = None
                    state.should_create_book = True
            except Exception as e:
                logging.error(f"Anthropic model call failed: {e!s}")
            return state


    def generate_reaction_book(self, state: RedAgentState) -> RedAgentState:
        mock_mode = os.getenv("MOCK_MODE", "false").lower() == "true"
        if mock_mode:
            with open("/app/agents/mocks/red_reaction_book_data_mock.yml", "r") as f:
                data = yaml.safe_load(f)
                reaction_book = ReactionBookArtifact(**data)
                state.artifacts.append(reaction_book)
                state.reaction_level += 1
                state.should_respond_with_reaction_book = False
                return state
        else:
            prompt = f"""
            Imagine yourself a small-time academic, novelist, or new-age writer. You come across this book:
            
            {BookMaker.format_card_catalog(state.main_generated_book.book_data)}
            
            You are moved to write your own book in response. It might be literary criticism, a parody, a reimagining, or
            a continuation of the original work. Write this new book in the style of your chosen author and make sure to include
            the following details:
            
            Respond in the following format:
            title="Title goes here",
            subtitle="Optional subtitle goes here",
            author="Author goes here",
            author_credentials="Optional author credentials goes here",
            year="year goes here but it should be after the original book",
            publisher="Publisher goes here",
            publisher_type= can be PublisherType.UNIVERSITY or PublisherType.VANITY,
            edition=random.choice(["1st", "2nd", "3rd", "Revised", "Expanded", "Mass Market"]),
            pages="int page count",
            isbn="fake isbn",
            catalog_number="int catalog number",
            condition="condition of the book - or just N/A",
            abstract=cls.generate_abstract(topic, the_genre),
            notable_quote=cls.generate_quote(topic, author, the_genre),
            """
            claude = self._get_claude()
            proposer = claude.with_structured_output(BookData)
            try:
                result = proposer.invoke(prompt)
                if isinstance(result, dict):
                    state.current_reaction_book = BookData(**result)
                    state.reaction_level +=1
                else:
                    state.current_reaction_book = None
                state.should_respond_with_reaction_book = False
            except Exception as e:
                logging.error(f"Anthropic model call failed: {e!s}")
            return state

    def write_reaction_book_pages(self, state: RedAgentState) -> RedAgentState:
        mock_mode = os.getenv("MOCK_MODE", "false").lower() == "true"
        if mock_mode:
            pass
        else:
            prompt = f"""
                Imagine yourself a small-time academic, novelist, or new-age writer. You have become obsessed with this book:
                
                {BookMaker.format_card_catalog(state.main_generated_book.book_data)}
                
                You are moved to write your own book in response:
                
                {BookMaker.format_card_catalog(state.current_reaction_book)}
                
                Let's create some sample pages. Start by outlining the basic structure of the book. 
                Then choose two sections and, in the suggested style write the fourth and fifth pages in those sections.
                
                Format your pages in the following way:
                
                page_1.text_content: "This is the first page of the book."
                page_2.text_content: "This is the second page of the book."
            """
            claude = self._get_claude()
            proposer = claude.with_structured_output(BookDataPageCollection)
            try:
                result = proposer.invoke(prompt)
                if isinstance(result, dict):
                    page_1 = TextChainArtifactFile(
                        text_content=result["page_1"],
                        thread_id=state.thread_id,
                        rainbow_color=the_rainbow_table_colors['R'],
                        base_path=os.getenv("AGENT_WORK_PRODUCT_BASE_PATH"),
                        chain_artifact_file_type=ChainArtifactFileType.MARKDOWN,
                        artifact_name=f"{state.current_reaction_book.author}_excerpt_1",
                    )
                    save_artifact_to_md(page_1)
                    state.artifacts.append(page_1)
                    page_2 = TextChainArtifactFile(
                        text_content=result["page_2"],
                        thread_id=state.thread_id,
                        rainbow_color=the_rainbow_table_colors['R'],
                        base_path=os.getenv("AGENT_WORK_PRODUCT_BASE_PATH"),
                        chain_artifact_file_type=ChainArtifactFileType.MARKDOWN,
                        artifact_name=f"{state.current_reaction_book.author}_excerpt_2",
                    )
                    save_artifact_to_md(page_2)
                    state.artifacts.append(page_2)
                    state.current_reaction_book = None
                else:
                   pass
            except Exception as e:
                logging.error(f"Anthropic model call failed: {e!s}")
        return state

    @staticmethod
    def _format_books_for_prompt(state: RedAgentState) -> str:
        parts = []
        main = getattr(state, "main_generated_book", None)
        if main and getattr(main, "book_data", None):
            parts.append("Main book:\n" + BookMaker.format_card_catalog(main.book_data))
        reaction_books = []
        for a in getattr(state, "artifacts", []) or []:
            if hasattr(a, "book_data"):
                reaction_books.append(a.book_data)
            elif isinstance(a, BookData):
                reaction_books.append(a)
            if len(reaction_books) >= 3:
                break
        for i, rb in enumerate(reaction_books, start=1):
            parts.append(f"Reaction book {i}:\n" + BookMaker.format_card_catalog(rb))
        return "\n\n".join(parts) if parts else "No books available."

    def evaluate_books_versus_proposals(self, state: RedAgentState) -> RedAgentState:
        mock_mode = os.getenv("MOCK_MODE", "false").lower() == "true"
        if mock_mode:
            if state.reaction_level==1:
                state.should_create_book = True
            elif state.reaction_level==2:
                state.should_create_book = False
                state.should_respond_with_reaction_book = True
            elif state.reaction_level >= 3:
                state.should_create_book = False
                state.should_respond_with_reaction_book = False
        else:
            answer_format = {
                "new_book": YesOrNo,
                "reaction_book": YesOrNo,
                "done": YesOrNo,
            }
            summary = self._format_books_for_prompt(state)
            prompt = f"""
            {summary}
            {state.counter_proposal}
            """
            claude = self._get_claude()
            proposer = claude.with_structured_output(answer_format)
            try:
                result = proposer.invoke(prompt)
                if isinstance(result, dict):
                    if result.get("new_book") == YesOrNo.YES:
                        state.should_create_book = True
                    else:
                        state.should_create_book = False
                    if result.get("reaction_book") == YesOrNo.YES:
                        state.should_respond_with_reaction_book = True
                    else:
                        state.should_respond_with_reaction_book = False
                else:
                    state.should_create_book = False
                    state.should_respond_with_reaction_book = False
            except Exception as e:
                logging.error(f"Anthropic model call failed: {e!s}")
        return state

    def route_after_evaluate_books_versus_proposals(self) -> str:
        if self.state.should_create_book:
            return "new_book"
        elif self.state.should_respond_with_reaction_book:
            return "reaction_book"
        else:
            return "done"

