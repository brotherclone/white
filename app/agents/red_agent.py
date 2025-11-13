import logging
import os
from abc import ABC

import yaml
from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langgraph.constants import END, START
from langgraph.graph import StateGraph

from app.agents.states.red_agent_state import RedAgentState
from app.agents.states.white_agent_state import MainAgentState
from app.agents.tools.book_tool import BookMaker
from app.agents.tools.text_tools import save_artifact_file_to_md
from app.structures.agents.agent_settings import AgentSettings
from app.structures.agents.base_rainbow_agent import BaseRainbowAgent
from app.structures.artifacts.book_artifact import BookArtifact, ReactionBookArtifact
from app.structures.artifacts.book_data import BookData, BookDataPageCollection
from app.structures.artifacts.text_artifact_file import TextChainArtifactFile
from app.structures.concepts.book_evaluation import BookEvaluationDecision
from app.structures.concepts.rainbow_table_color import the_rainbow_table_colors
from app.structures.enums.chain_artifact_file_type import ChainArtifactFileType
from app.structures.manifests.song_proposal import SongProposalIteration
from app.util.manifest_loader import get_my_reference_proposals

load_dotenv()
logging.basicConfig(level=logging.INFO)


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
        )

    def __call__(self, state: MainAgentState) -> MainAgentState:
        """Entry point when White Agent invokes Red Agent"""

        current_proposal = state.song_proposals.iterations[-1]
        red_state = RedAgentState(
            thread_id=state.thread_id,
            song_proposals=state.song_proposals,
            black_to_white_proposal=current_proposal,
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

        return state

    def create_graph(self) -> StateGraph:

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

    def generate_alternate_song_spec(self, state: RedAgentState) -> RedAgentState:
        mock_mode = os.getenv("MOCK_MODE", "false").lower() == "true"
        if mock_mode:
            with open(
                f"{os.getenv('AGENT_MOCK_DATA_PATH')}/red_counter_proposal_mock.yml",
                "r",
            ) as f:
                data = yaml.safe_load(f)
                counter_proposal = SongProposalIteration(**data)
                state.counter_proposal = counter_proposal
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
                else:
                    state.counter_proposal = None
            except Exception as e:
                logging.error(f"Anthropic model call failed: {e!s}")
        return state

    def generate_book(self, state: RedAgentState) -> RedAgentState:
        mock_mode = os.getenv("MOCK_MODE", "false").lower() == "true"
        if mock_mode:
            with open(
                f"{os.getenv('AGENT_MOCK_DATA_PATH')}/red_book_artifact_mock.yml", "r"
            ) as f:
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
                if isinstance(result, BookDataPageCollection):
                    page_1_text = result.page_1
                    page_2_text = result.page_2
                elif isinstance(result, dict):
                    page_1_text = result.get("page_1", "")
                    page_2_text = result.get("page_2", "")
                else:
                    logging.error(f"Unexpected result type: {type(result)}")
                    state.should_create_book = False
                    return state
                page_1 = TextChainArtifactFile(
                    text_content=page_1_text,
                    thread_id=state.thread_id,
                    rainbow_color=the_rainbow_table_colors["R"],
                    base_path=os.getenv("AGENT_WORK_PRODUCT_BASE_PATH"),
                    chain_artifact_file_type=ChainArtifactFileType.MARKDOWN,
                    artifact_name=f"{book_data.author}_main_book_page_1",
                )
                try:
                    save_artifact_file_to_md(page_1)
                    logging.info(
                        f"✅ Saved main book page 1: {page_1.get_artifact_path()}"
                    )
                except Exception as e:
                    logging.error(f"Failed to save page 1: {e}")
                page_2 = TextChainArtifactFile(
                    text_content=page_2_text,
                    thread_id=state.thread_id,
                    rainbow_color=the_rainbow_table_colors["R"],
                    base_path=os.getenv("AGENT_WORK_PRODUCT_BASE_PATH"),
                    chain_artifact_file_type=ChainArtifactFileType.MARKDOWN,
                    artifact_name=f"{book_data.author}_main_book_page_2",
                )
                try:
                    save_artifact_file_to_md(page_2)
                    logging.info(
                        f"✅ Saved main book page 2: {page_2.get_artifact_path()}"
                    )
                except Exception as e:
                    logging.error(f"Failed to save page 2: {e}")
                state.reaction_level += 1
                state.main_generated_book = BookArtifact(
                    book_data=book_data,
                    excerpts=[page_1, page_2],
                    thread_id=state.thread_id,
                )
                state.should_create_book = False  # ✅ Success - don't loop

            except Exception as e:
                logging.error(f"Book generation failed: {e!s}")
                state.should_create_book = False  # ✅ Don't loop on error!

            return state

    def generate_reaction_book(self, state: RedAgentState) -> RedAgentState:
        mock_mode = os.getenv("MOCK_MODE", "false").lower() == "true"
        if mock_mode:
            with open(
                f"{os.getenv('AGENT_MOCK_DATA_PATH')}/red_reaction_book_data_mock.yml",
                "r",
            ) as f:
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
                if isinstance(result, BookData):
                    state.current_reaction_book = result
                    state.reaction_level += 1
                elif isinstance(result, dict):
                    state.current_reaction_book = BookData(**result)
                    state.reaction_level += 1
                else:
                    logging.error(f"Unexpected result type: {type(result)}")
                    state.current_reaction_book = None

                state.should_respond_with_reaction_book = False

            except Exception as e:
                logging.error(f"Anthropic model call failed: {e!s}")
                state.current_reaction_book = None  # ✅ Set to None on error

            return state

    def write_reaction_book_pages(self, state: RedAgentState) -> RedAgentState | None:
        mock_mode = os.getenv("MOCK_MODE", "false").lower() == "true"
        if mock_mode:
            with open(
                f"{os.getenv('AGENT_MOCK_DATA_PATH')}/red_reaction_book_page_1_mock.yml",
                "r",
            ) as f:
                data = yaml.safe_load(f)
                page_1 = TextChainArtifactFile(**data)
            with open(
                f"{os.getenv('AGENT_MOCK_DATA_PATH')}/red_reaction_book_page_2_mock.yml",
                "r",
            ) as f:
                data = yaml.safe_load(f)
                page_2 = TextChainArtifactFile(**data)
            state.artifacts.append(page_1)
            state.artifacts.append(page_2)
            state.current_reaction_book = None
            return state
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
                if isinstance(result, BookDataPageCollection):
                    page_1_text = result.page_1
                    page_2_text = result.page_2
                elif isinstance(result, dict):
                    page_1_text = result.get("page_1", "")
                    page_2_text = result.get("page_2", "")
                else:
                    logging.error(f"Unexpected result type: {type(result)}")
                    state.current_reaction_book = None
                    return state
                # Create page 1 artifact
                page_1 = TextChainArtifactFile(
                    text_content=page_1_text,
                    thread_id=state.thread_id,
                    rainbow_color=the_rainbow_table_colors["R"],
                    base_path=os.getenv("AGENT_WORK_PRODUCT_BASE_PATH"),
                    chain_artifact_file_type=ChainArtifactFileType.MARKDOWN,
                    artifact_name=f"{state.current_reaction_book.author}_excerpt_1",
                )
                try:
                    save_artifact_file_to_md(page_1)
                    logging.info(
                        f"✅ Saved page 1: {page_1.artifact_name} to {page_1.get_artifact_path()}"
                    )
                except ValueError as ve:
                    logging.error(f"Failed to save page 1 {page_1.artifact_name}: {ve}")
                except Exception as se:
                    logging.error(f"Unexpected error saving page 1: {se}")
                state.artifacts.append(page_1)

                # Create page 2 artifact
                page_2 = TextChainArtifactFile(
                    text_content=page_2_text,
                    thread_id=state.thread_id,
                    rainbow_color=the_rainbow_table_colors["R"],
                    base_path=os.getenv("AGENT_WORK_PRODUCT_BASE_PATH"),
                    chain_artifact_file_type=ChainArtifactFileType.MARKDOWN,
                    artifact_name=f"{state.current_reaction_book.author}_excerpt_2",
                )
                try:
                    save_artifact_file_to_md(page_2)
                    logging.info(
                        f"✅ Saved page 2: {page_2.artifact_name} to {page_2.get_artifact_path()}"
                    )
                    return state
                except ValueError as ve:
                    logging.error(f"Failed to save page 2 {page_2.artifact_name}: {ve}")
                except Exception as se:
                    logging.error(f"Unexpected error saving page 2: {se}")
                state.artifacts.append(page_2)
                state.current_reaction_book = None
                return state
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
        if state.reaction_level >= 3:
            logging.info(
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
            # ✅ FIXED: Use the proper Pydantic model
            proposer = claude.with_structured_output(BookEvaluationDecision)

            try:
                result = proposer.invoke(prompt)

                # ✅ Handle both dict and Pydantic model responses
                if isinstance(result, dict):
                    state.should_create_book = result.get("new_book", False)
                    state.should_respond_with_reaction_book = result.get(
                        "reaction_book", False
                    )
                elif isinstance(result, BookEvaluationDecision):
                    state.should_create_book = result.new_book
                    state.should_respond_with_reaction_book = result.reaction_book
                else:
                    logging.warning(f"Unexpected result type: {type(result)}")
                    state.should_create_book = False
                    state.should_respond_with_reaction_book = False

            except Exception as e:
                logging.error(f"Book evaluation failed: {e!s}")
                # Default to done if evaluation fails
                state.should_create_book = False
                state.should_respond_with_reaction_book = False

        return state

    @staticmethod
    def route_after_evaluate_books_versus_proposals(state: RedAgentState) -> str:
        if state.should_create_book:
            return "new_book"
        elif state.should_respond_with_reaction_book:
            return "reaction_book"
        else:
            return "done"
