import logging
import os
import time
import yaml

from abc import ABC
from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langgraph.constants import END, START
from langgraph.graph import StateGraph

from app.agents.states.red_agent_state import RedAgentState
from app.agents.states.white_agent_state import MainAgentState
from app.agents.tools.book_tool import BookMaker
from app.agents.workflow.agent_error_handler import agent_error_handler
from app.structures.agents.agent_settings import AgentSettings
from app.structures.agents.base_rainbow_agent import BaseRainbowAgent
from app.util.agent_state_utils import get_state_snapshot
from app.structures.artifacts.book_artifact import BookArtifact, BookPageCollection
from app.structures.concepts.book_evaluation import BookEvaluationDecision
from app.structures.concepts.rainbow_table_color import the_rainbow_table_colors
from app.structures.manifests.song_proposal import SongProposalIteration
from app.util.manifest_loader import get_my_reference_proposals

load_dotenv()
logging.basicConfig(level=logging.INFO)

logger = logging.getLogger(__name__)


class RedAgent(BaseRainbowAgent, ABC):
    """The Light Reader."""

    def __init__(self, **data):
        if "settings" not in data or data["settings"] is None:
            data["settings"] = AgentSettings()
        super().__init__(**data)
        if self.settings is None:
            self.settings = AgentSettings()
        self.llm = ChatAnthropic(
            temperature=self.settings.temperature,
            api_key=self.settings.anthropic_api_key,
            model_name=self.settings.anthropic_model_name,
            max_retries=self.settings.max_retries,
            timeout=self.settings.timeout,
            stop=self.settings.stop,
            max_tokens=self.settings.max_tokens,
        )

    def __call__(self, state: MainAgentState) -> MainAgentState:
        """Entry point when White Agent invokes Red Agent"""

        current_proposal = state.song_proposals.iterations[-1]
        red_state = RedAgentState(
            thread_id=state.thread_id,
            song_proposals=state.song_proposals,
            white_proposal=current_proposal,
            counter_proposal=None,
            artifacts=[],
            should_respond_with_reaction_book=False,
            should_create_book=True,
            reaction_level=0,
        )
        red_graph = self.create_graph()
        compiled_graph = red_graph.compile()
        result = compiled_graph.invoke(red_state.model_dump())
        if isinstance(result, RedAgentState):
            final_state = result
        elif isinstance(result, dict):
            final_state = RedAgentState(**result)
        else:
            raise TypeError(f"Unexpected result type: {type(result)}")
        if final_state.counter_proposal:
            state.song_proposals.iterations.append(final_state.counter_proposal)
        if final_state.artifacts:
            state.artifacts = final_state.artifacts
        return state

    @staticmethod
    def _ensure_dict(obj):
        """Return a plain dict for pydantic/dataclass or dict-like objects."""
        if isinstance(obj, dict):
            return obj.copy()
        if hasattr(obj, "model_dump"):
            return obj.model_dump()
        if hasattr(obj, "__dict__"):
            return dict(obj.__dict__)
        raise TypeError(f"Cannot convert object of type {type(obj)} to dict")

    def create_graph(self) -> StateGraph:
        """
        Create the workflow for the Red Agent.
        1. Generate a book.
        2. Use the book to make a counter-proposal.
        3. Decide whether to respond with a reaction book, a new book or if you're.
        4. If so, generate a reaction book.
        5. Then generate the pages of the reaction book.
        6. Repeat steps 1-5 until the proposal is accepted.
        :return:
        """
        red_workflow = StateGraph(RedAgentState)
        red_workflow.add_node("generate_book", self.generate_book)
        red_workflow.add_node(
            "generate_alternate_song_spec", self.generate_alternate_song_spec
        )
        red_workflow.add_node(
            "evaluate_books_versus_proposals", self.evaluate_books_versus_proposals
        )
        red_workflow.add_node("generate_reaction_book", self.generate_reaction_book)
        red_workflow.add_node(
            "write_reaction_book_pages", self.write_reaction_book_pages
        )
        red_workflow.add_edge(START, "generate_book")
        red_workflow.add_edge("generate_book", "generate_alternate_song_spec")
        red_workflow.add_edge("generate_reaction_book", "write_reaction_book_pages")
        red_workflow.add_edge(
            "write_reaction_book_pages", "evaluate_books_versus_proposals"
        )
        red_workflow.add_edge(
            "generate_alternate_song_spec", "evaluate_books_versus_proposals"
        )
        red_workflow.add_conditional_edges(
            "evaluate_books_versus_proposals",
            self.route_after_evaluate_books_versus_proposals,
            {
                "new_book": "generate_book",
                "reaction_book": "generate_reaction_book",
                "done": END,
            },
        )
        return red_workflow

    @agent_error_handler("The Light Reader")
    def generate_alternate_song_spec(self, state: RedAgentState) -> RedAgentState:
        get_state_snapshot(
            state,
            "generate_alternate_song_spec_enter",
            state.thread_id,
            "The Light Reader",
        )
        mock_mode = os.getenv("MOCK_MODE", "false").lower() == "true"
        block_mode = os.getenv("BLOCK_MODE", "false").lower() == "true"
        if mock_mode:
            try:
                with open(
                    f"{os.getenv('AGENT_MOCK_DATA_PATH')}/red_counter_proposal_mock.yml",
                    "r",
                ) as f:
                    data = yaml.safe_load(f)
                    counter_proposal = SongProposalIteration(**data)
                    state.counter_proposal = counter_proposal
            except Exception as e:
                error_msg = f"Failed to read mock counter proposal: {e!s}"
                logger.error(error_msg)
                if block_mode:
                    raise Exception(error_msg)
        else:
            summary = self._format_books_for_prompt(state)
            prompt = f"""
            You are the Light Reader, a hermitic keeper of books rare and unusual. For you, these books are your
            only means of communicating to the outside world. You've been given a unique job today and that is
            to create a proposal for a song. This proposal should take into consider the current proposal from
            the Black Agent as well as the book you last authored. Create a counter proposal for a song that shares aspects
            of the current proposal but ultimately is about your last book and works that reference it. Use the bank
            of real 'red' song manifests to help you craft your counter proposal.
            
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
                    return state
                if not isinstance(result, SongProposalIteration):
                    error_msg = f"Expected SongProposalIteration, got {type(result)}"
                    if block_mode:
                        raise TypeError(error_msg)
                    logger.warning(error_msg)
            except Exception as e:
                print(
                    f"Anthropic model call failed: {e!s}; returning stub SongProposalIteration for red's counter proposal after authoring a book."
                )
                if block_mode:
                    raise Exception("Anthropic model call failed")
                else:
                    timestamp = int(time.time() * 1000)
                    counter_proposal = SongProposalIteration(
                        iteration_id=f"fallback_error_{timestamp}",
                        bpm=120,
                        tempo="4/4",
                        key="C Major",
                        rainbow_color="red",
                        title="Fallback: Red Song",
                        mood=["obscure"],
                        genres=["rock", "electronic"],
                        concept="Fallback stub because Anthropic model unavailable. Fallback stub because Anthropic model unavailable. Fallback stub because Anthropic model unavailable.",
                    )
                    state.counter_proposal = counter_proposal
        get_state_snapshot(
            state,
            "generate_alternate_song_spec_exit",
            state.thread_id,
            "The Light Reader",
        )
        return state

    @agent_error_handler("The Light Reader")
    def generate_book(self, state: RedAgentState) -> RedAgentState:
        get_state_snapshot(
            state, "generate_book_enter", state.thread_id, "The Light Reader"
        )
        mock_mode = os.getenv("MOCK_MODE", "false").lower() == "true"
        block_mode = os.getenv("BLOCK_MODE", "false").lower() == "true"
        if mock_mode:
            try:
                with open(
                    f"{os.getenv('AGENT_MOCK_DATA_PATH')}/red_book_artifact_mock.yml",
                    "r",
                ) as f:
                    data = yaml.safe_load(f)
                    data["base_path"] = os.getenv(
                        "AGENT_WORK_PRODUCT_BASE_PATH", "chain_artifacts"
                    )
                    data["thread_id"] = state.thread_id
                    book = BookArtifact(**data)
                    book.save_file()
                    state.artifacts.append(book)
                    state.should_create_book = False
                    state.reaction_level += 1
                    get_state_snapshot(
                        state, "generate_book_exit", state.thread_id, "The Light Reader"
                    )
            except Exception as e:
                error_msg = f"Failed to read or save mock book artifact: {e!s}"
                logger.error(error_msg)
                if block_mode:
                    raise Exception(error_msg)
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
            
            page_1_text: "This is the first page of the book."
            page_2_text: "This is the second page of the book."
            
            """
            claude = self._get_claude()
            proposer = claude.with_structured_output(BookPageCollection)
            try:
                result = proposer.invoke(prompt)
                if isinstance(result, BookPageCollection):
                    page_1_text = result.page_1_text
                    page_2_text = result.page_2_text
                elif isinstance(result, dict):
                    page_1_text = result.get("page_1_text", "")
                    page_2_text = result.get("page_2_text", "")
                else:
                    logger.error(f"Unexpected result type: {type(result)}")
                    if block_mode:
                        raise TypeError(f"Unexpected result type: {type(result)}")
                    else:
                        logger.warning(f"Unexpected result type: {type(result)}")
                        page_1_text = ""
                        page_2_text = ""
                state.reaction_level += 1
                book_dict = book_data.model_dump()
                # Remove fields that should use class defaults, not source-generated values
                book_dict.pop("chain_artifact_file_type", None)
                book_dict.pop("chain_artifact_type", None)
                book_dict.pop("file_name", None)
                book_dict.pop("file_path", None)
                book_dict.pop("artifact_id", None)  # Let class generate UUID

                book_dict["excerpts"] = [page_1_text, page_2_text]
                book_dict["thread_id"] = state.thread_id
                book_dict["base_path"] = os.getenv(
                    "AGENT_WORK_PRODUCT_BASE_PATH", "artifacts"
                )
                state.main_generated_book = BookArtifact(**book_dict)
                state.artifacts.append(state.main_generated_book)
                try:
                    state.main_generated_book.save_file()
                except Exception as e:
                    error_msg = f"Failed to save book artifact file: {e!s}"
                    logger.error(error_msg)
                    if block_mode:
                        raise Exception(error_msg)
                state.should_create_book = False
                return state
            except Exception as e:
                logger.error(f"Book generation failed: {e!s}")
                state.should_create_book = False
                if block_mode:
                    raise Exception("Anthropic model call failed")
            return state

    @agent_error_handler("The Light Reader")
    def generate_reaction_book(self, state: RedAgentState) -> RedAgentState:
        get_state_snapshot(
            state, "generate_reaction_book_enter", state.thread_id, "The Light Reader"
        )
        mock_mode = os.getenv("MOCK_MODE", "false").lower() == "true"
        block_mode = os.getenv("BLOCK_MODE", "false").lower() == "true"
        if mock_mode:
            try:
                with open(
                    f"{os.getenv('AGENT_MOCK_DATA_PATH')}/red_reaction_book_data_mock.yml",
                    "r",
                ) as f:
                    book_data = yaml.safe_load(f)
                    if isinstance(book_data, dict):
                        book_dict = book_data.copy()
                    elif hasattr(book_data, "model_dump"):
                        book_dict = book_data.model_dump()
                    elif hasattr(book_data, "__dict__"):
                        book_dict = dict(book_data.__dict__)
                    else:
                        error_msg = f"Cannot convert book_data of type {type(book_data)} to dict"
                        logger.error(error_msg)
                        if block_mode:
                            raise TypeError(error_msg)
                        state.should_respond_with_reaction_book = False
                        return state
                    book_dict["thread_id"] = state.thread_id
                    book_dict["base_path"] = os.getenv(
                        "AGENT_WORK_PRODUCT_BASE_PATH", "artifacts"
                    )
                    state.current_reaction_book = BookArtifact(**book_dict)
                    state.artifacts.append(state.current_reaction_book)
                    state.reaction_level += 1
                    state.should_respond_with_reaction_book = False
            except Exception as e:
                error_msg = f"Failed to read mock reaction book data: {e!s}"
                logger.error(error_msg)
                if block_mode:
                    raise Exception(error_msg)
                state.should_respond_with_reaction_book = False
            return state
        else:
            prompt = f"""
            Imagine yourself a small-time academic, novelist, or new-age writer. You come across this book:
            
            {BookMaker.format_card_catalog(state.main_generated_book)}
            
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
            proposer = claude.with_structured_output(BookArtifact)
            try:
                result = proposer.invoke(prompt)
                if isinstance(result, BookArtifact):
                    result_dict = result.model_dump()
                elif isinstance(result, dict):
                    result_dict = result
                else:
                    error_msg = f"Expected BookArtifact, got {type(result)}"
                    logger.error(error_msg)
                    state.current_reaction_book = None
                    if block_mode:
                        raise TypeError(error_msg)
                    state.should_respond_with_reaction_book = False
                    return state

                # Remove fields that should use class defaults, not LLM-generated values
                result_dict.pop("chain_artifact_file_type", None)
                result_dict.pop("chain_artifact_type", None)
                result_dict.pop("file_name", None)
                result_dict.pop("file_path", None)
                result_dict.pop("artifact_id", None)  # Let class generate UUID

                result_dict["thread_id"] = state.thread_id
                result_dict["base_path"] = os.getenv(
                    "AGENT_ARTIFACTS_PATH", "artifacts"
                )
                state.current_reaction_book = BookArtifact(**result_dict)
                state.artifacts.append(state.current_reaction_book)
                state.reaction_level += 1
                state.should_respond_with_reaction_book = False
                return state
            except Exception as e:
                logger.error(f"Anthropic model call failed: {e!s}")
                state.current_reaction_book = None
                print(
                    f"Anthropic model call failed: {e!s}; returning stub SongProposalIteration for black's first counter proposal."
                )
                if block_mode:
                    raise Exception("Anthropic model call failed")
            return state

    @agent_error_handler("The Light Reader")
    def write_reaction_book_pages(self, state: RedAgentState) -> RedAgentState | None:
        get_state_snapshot(
            state,
            "write_reaction_book_pages_enter",
            state.thread_id,
            "The Light Reader",
        )
        mock_mode = os.getenv("MOCK_MODE", "false").lower() == "true"
        block_mode = os.getenv("BLOCK_MODE", "false").lower() == "true"
        if mock_mode:
            try:
                with open(
                    f"{os.getenv('AGENT_MOCK_DATA_PATH')}/red_reaction_book_page_1_mock.yml",
                    "r",
                ) as f:
                    page_1 = yaml.safe_load(f)
                with open(
                    f"{os.getenv('AGENT_MOCK_DATA_PATH')}/red_reaction_book_page_2_mock.yml",
                    "r",
                ) as f:
                    page_2 = yaml.safe_load(f)
                state.current_reaction_book.excerpts = [page_1, page_2]
                state.artifacts.append(state.current_reaction_book)
            except Exception as e:
                error_msg = f"Failed to read mock reaction book pages: {e!s}"
                logger.error(error_msg)
                if block_mode:
                    raise Exception(error_msg)
            return state
        else:
            prompt = f"""
                Imagine yourself a small-time academic, novelist, or new-age writer. You have become obsessed with this book:
                
                {BookMaker.format_card_catalog(state.main_generated_book)}
                
                You are moved to write your own book in response:
                
                {BookMaker.format_card_catalog(state.current_reaction_book)}
                
                Let's create some sample pages. Start by outlining the basic structure of the book. 
                Then choose two sections and, in the suggested style write the fourth and fifth pages in those sections.
                
                Format your pages in the following way:
                
                page_1.text_content: "This is the first page of the book."
                page_2.text_content: "This is the second page of the book."
            """
            claude = self._get_claude()
            proposer = claude.with_structured_output(BookPageCollection)
            try:
                result = proposer.invoke(prompt)
                if isinstance(result, dict):
                    reaction_book_pages = BookPageCollection(**result)
                    state.current_reaction_book.excerpts = [
                        reaction_book_pages.page_1_text,
                        reaction_book_pages.page_2_text,
                    ]
                    state.artifacts.append(state.current_reaction_book)
                    return state
                if not isinstance(result, BookPageCollection):
                    error_msg = f"Expected BookPageCollection, got {type(result)}"
                    if block_mode:
                        raise TypeError(error_msg)
                    logger.warning(error_msg)
            except Exception as e:
                print(
                    f"Anthropic model call failed: {e!s}; returning stub SongProposalIteration for writing reaction book pages after authoring a book."
                )
                if block_mode:
                    raise Exception("Anthropic model call failed")
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
            elif isinstance(a, BookArtifact):
                reaction_books.append(a)
            if len(reaction_books) >= 3:
                break
        for i, rb in enumerate(reaction_books, start=1):
            parts.append(f"Reaction book {i}:\n" + BookMaker.format_card_catalog(rb))
        return "\n\n".join(parts) if parts else "No books available."

    @agent_error_handler("The Light Reader")
    def evaluate_books_versus_proposals(self, state: RedAgentState) -> RedAgentState:
        get_state_snapshot(
            state,
            "evaluate_books_versus_proposals_enter",
            state.thread_id,
            "The Light Reader",
        )
        if state.reaction_level >= 3:
            logger.info(
                f"🛑 Reaction limit reached ({state.reaction_level}), ending book generation"
            )
            state.should_create_book = False
            state.should_respond_with_reaction_book = False
            return state
        mock_mode = os.getenv("MOCK_MODE", "false").lower() == "true"
        if mock_mode:
            if state.reaction_level == 1:
                state.should_create_book = True
            elif state.reaction_level == 2:
                state.should_create_book = False
                state.should_respond_with_reaction_book = True
            elif state.reaction_level >= 3:
                state.should_create_book = False
                state.should_respond_with_reaction_book = False
        else:
            summary = self._format_books_for_prompt(state)
            prompt = f"""
            You are evaluating whether to continue generating books or if you're satisfied with the current collection.

            Current books and reactions:
            {summary}

            Current counter proposal:
            {state.counter_proposal}

            Decide:
            - new_book: true if you want to generate an entirely new book
            - reaction_book: true if you want to generate a reaction/response to an existing book
            - done: true if you're satisfied and want to stop generating books

            Usually you should set done=true after generating 2-3 books total.
            """

            claude = self._get_claude()
            proposer = claude.with_structured_output(BookEvaluationDecision)
            try:
                result = proposer.invoke(prompt)
                if isinstance(result, dict):
                    state.should_create_book = result.get("new_book", False)
                    state.should_respond_with_reaction_book = result.get(
                        "reaction_book", False
                    )
                elif isinstance(result, BookEvaluationDecision):
                    state.should_create_book = result.new_book
                    state.should_respond_with_reaction_book = result.reaction_book
                else:
                    logger.warning(f"Unexpected result type: {type(result)}")
                    state.should_create_book = False
                    state.should_respond_with_reaction_book = False

            except Exception as e:
                logger.error(f"Book evaluation failed: {e!s}")
                state.should_create_book = False
                state.should_respond_with_reaction_book = False
        get_state_snapshot(
            state,
            "evaluate_books_versus_proposals_exit",
            state.thread_id,
            "The Light Reader",
        )
        return state

    @staticmethod
    @agent_error_handler("The Light Reader")
    def route_after_evaluate_books_versus_proposals(state: RedAgentState) -> str:
        get_state_snapshot(
            state,
            "route_after_evaluate_books_versus_proposals_enter",
            state.thread_id,
            "The Light Reader",
        )
        if state.should_create_book:
            get_state_snapshot(
                state,
                "route_after_evaluate_books_versus_proposals_exit",
                state.thread_id,
                "The Light Reader",
            )
            return "new_book"
        elif state.should_respond_with_reaction_book:
            get_state_snapshot(
                state,
                "route_after_evaluate_books_versus_proposals_exit",
                state.thread_id,
                "The Light Reader",
            )
            return "reaction_book"
        else:
            get_state_snapshot(
                state,
                "route_after_evaluate_books_versus_proposals_exit",
                state.thread_id,
                "The Light Reader",
            )
            return "done"
