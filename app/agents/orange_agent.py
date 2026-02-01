import logging
import os
import random
import time
import yaml

from abc import ABC
from datetime import datetime, timedelta
from typing import Optional
from anthropic import Anthropic
from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langgraph.constants import START
from langgraph.graph.state import StateGraph
from langchain_core.messages import HumanMessage
from pydantic import Field

from app.agents.states.orange_agent_state import OrangeAgentState
from app.agents.states.white_agent_state import MainAgentState
from app.reference.mcp.rows_bud.orange_corpus import OrangeMythosCorpus, get_corpus
from app.agents.workflow.agent_error_handler import agent_error_handler
from app.structures.agents.base_rainbow_agent import BaseRainbowAgent
from app.structures.enums.chain_artifact_file_type import ChainArtifactFileType
from app.structures.enums.symbolic_object_category import SymbolicObjectCategory
from app.util.agent_state_utils import get_state_snapshot
from app.structures.artifacts.newspaper_artifact import NewspaperArtifact
from app.structures.artifacts.symbolic_object_artifact import SymbolicObjectArtifact
from app.structures.concepts.rainbow_table_color import the_rainbow_table_colors
from app.structures.manifests.song_proposal import SongProposalIteration
from app.util.manifest_loader import get_my_reference_proposals
from app.structures.agents.agent_settings import AgentSettings

load_dotenv()

logger = logging.getLogger(__name__)


class OrangeAgent(BaseRainbowAgent, ABC):
    """Sussex Mythologizer - Rows Bud, the 182 BPM transmission keeper"""

    corpus: Optional[OrangeMythosCorpus] = Field(default=None)
    anthropic_client: Optional[Anthropic] = Field(default=None)

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
        corpus_dir: str = os.getenv("ORANGE_CORPUS_DIR")
        self.corpus = get_corpus(corpus_dir)

    def __call__(self, state: MainAgentState) -> MainAgentState:
        """Entry point when called from White Agent"""
        current_proposal = state.song_proposals.iterations[-1]
        orange_state = OrangeAgentState(
            white_proposal=current_proposal,
            thread_id=state.thread_id,
            artifacts=[],
            synthesized_story=None,
            search_results=None,
            corpus_stories=None,
            selected_story_id=None,
            symbolic_object=None,
            gonzo_perspective=None,
            gonzo_intensity=3,
            mythologized_story=None,
            counter_proposal=None,
        )
        orange_graph = self.create_graph()
        compiled_graph = orange_graph.compile()
        result = compiled_graph.invoke(orange_state.model_dump())
        if isinstance(result, OrangeAgentState):
            final_state = result
        elif isinstance(result, dict):
            final_state = OrangeAgentState(**result)
        else:
            raise TypeError(f"Unexpected result type: {type(result)}")
        if final_state.counter_proposal:
            state.song_proposals.iterations.append(final_state.counter_proposal)
        if final_state.artifacts:
            state.artifacts = final_state.artifacts
        return state

    def create_graph(self) -> StateGraph:
        """Build Orange's mythology synthesis workflow"""
        work_flow = StateGraph(OrangeAgentState)

        # Nodes
        work_flow.add_node("synthesize_base_story", self.synthesize_base_story)
        work_flow.add_node("add_to_corpus", self.add_to_corpus)
        work_flow.add_node("select_symbolic_object", self.select_symbolic_object)
        work_flow.add_node(
            "insert_symbolic_object_node", self.insert_symbolic_object_node
        )
        work_flow.add_node("gonzo_rewrite_node", self.gonzo_rewrite_node)
        work_flow.add_node(
            "generate_alternate_song_spec", self.generate_alternate_song_spec
        )

        # Flow
        work_flow.add_edge(START, "synthesize_base_story")
        work_flow.add_edge("synthesize_base_story", "add_to_corpus")
        work_flow.add_edge("add_to_corpus", "select_symbolic_object")
        work_flow.add_edge("select_symbolic_object", "insert_symbolic_object_node")
        work_flow.add_edge("insert_symbolic_object_node", "gonzo_rewrite_node")
        work_flow.add_edge("gonzo_rewrite_node", "generate_alternate_song_spec")

        return work_flow

    @agent_error_handler("Rows Bud")
    def generate_alternate_song_spec(self, state: OrangeAgentState) -> OrangeAgentState:
        get_state_snapshot(
            state, "generate_alternate_song_spec_entry", state.thread_id, "Rows Bud"
        )
        mock_mode = os.getenv("MOCK_MODE", "false").lower() == "true"
        block_mode = os.getenv("BLOCK_MODE", "false").lower() == "true"
        if mock_mode:
            try:
                with open(
                    os.path.join(
                        os.getenv("AGENT_MOCK_DATA_PATH"),
                        "orange_counter_proposal_mock.yml",
                    ),
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
            return state
        else:
            prompt = f"""Generate Orange's counter-proposal based on the mythologized story.

            WHITE'S PROPOSAL:
            {state.white_proposal.model_dump_json(indent=2)}

            MYTHOLOGIZED STORY:
            Headline: {state.mythologized_story.headline}
            Date: {state.mythologized_story.date}
            Location: {state.mythologized_story.location}
            Symbolic Object: {state.symbolic_object.name if state.symbolic_object else 'N/A'}

            Story Text:
            {state.mythologized_story.text}

            ORANGE'S METHODOLOGY:
            Orange mythologizes - transforms factual into legendary through:
            - Temporal/spatial grounding (Sussex County, 1975-1995)
            - Symbolic object insertion (makes abstract concrete)
            - Gonzo narrative voice (paranoia, embedded journalism, perception shifts)
            - Local legend creation (information seeking physical manifestation)

            REFERENCE WORKS (Orange style):
            {get_my_reference_proposals('O')}

            Generate a counter-proposal that:
            1. Maintains White's core concept but adds Orange's mythology layer
            2. Uses story's date/location as temporal coordinates
            3. Centers on the symbolic object as conceptual anchor
            4. Adopts gonzo perspective for lyrical voice
            5. BPM should reference 182 (the transmission speed) or its subdivisions
            6. Always sets rainbow_color to: {the_rainbow_table_colors['O']}
            """
            claude = self._get_claude()
            proposer = claude.with_structured_output(SongProposalIteration)

            try:
                result = proposer.invoke(prompt)
                if isinstance(result, dict):
                    counter_proposal = SongProposalIteration(**result)
                elif isinstance(result, SongProposalIteration):
                    counter_proposal = result
                else:
                    error_msg = f"Expected SongProposalIteration, got {type(result)}"
                    logger.error(error_msg)
                    if block_mode:
                        raise TypeError(error_msg)
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
            except Exception as e:
                error_msg = f"Orange counter proposal LLM call failed: {e!s}"
                logger.error(error_msg)
                if block_mode:
                    raise Exception(error_msg)
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
            get_state_snapshot(
                state, "generate_alternate_song_spec_exit", state.thread_id, "Rows Bud"
            )
            return state

    @agent_error_handler("Rows Bud")
    def synthesize_base_story(self, state: OrangeAgentState) -> OrangeAgentState:
        """Synthesize a plausible Sussex County newspaper story (1975-1995)"""
        get_state_snapshot(
            state, "synthesize_base_story_entry", state.thread_id, "Rows Bud"
        )
        mock_mode = os.getenv("MOCK_MODE", "false").lower() == "true"
        block_mode = os.getenv("BLOCK_MODE", "false").lower() == "true"
        if mock_mode:
            try:
                with open(
                    os.path.join(
                        os.getenv("AGENT_MOCK_DATA_PATH"), "orange_base_story_mock.yml"
                    ),
                    "r",
                ) as f:
                    data = yaml.safe_load(f)
                    data["thread_id"] = state.thread_id
                    data["base_path"] = os.getenv(
                        "AGENT_WORK_PRODUCT_BASE_PATH", "chain_artifacts"
                    )
                    story = NewspaperArtifact(**data)
                    combined = story.get_text_content()
                    story.text = combined
                    story.save_file()
                    state.synthesized_story = story
                    state.artifacts.append(story)
            except Exception as e:
                error_msg = f"Failed to read or save mock base story: {e!s}"
                logger.error(error_msg)
                if block_mode:
                    raise Exception(error_msg)
            return state
        else:
            start_date = datetime(1975, 1, 1)
            end_date = datetime(1995, 12, 31)
            random_days = random.randint(0, (end_date - start_date).days)
            story_date = start_date + timedelta(days=random_days)
            locations = [
                "Vernon Township",
                "Sussex Borough",
                "Newton",
                "Sparta Township",
                "Andover Township",
                "Hamburg Borough",
                "Franklin Borough",
                "Hopatcong",
                "Montague Township",
                "Stanhope",
                "Wantage Township",
                "Byram Township",
            ]
            sources = [
                "Sussex County Independent",
                "New Jersey Herald",
                "The Advertiser-News",
                "Sussex County Courier",
                "Vernon Township Times",
                "Newton Register",
            ]
            prompt = f"""
            You are synthesizing a NEW JERSEY newspaper article that NEVER EXISTED but could have.
            WHITE ALBUM CONTEXT:
            The White Album explores information seeking physical manifestation. The EMORY transmission 
            has propagated for 70 years.
    
            WHITE'S CURRENT PROPOSAL:
            
            {state.white_proposal.model_dump_json(indent=2)}
    
            REQUIREMENTS:
            - Takes place in Sussex County, New Jersey
            - Date: {story_date.strftime('%Y-%m-%d')} (between 1975-1995)
            - Location: {random.choice(locations)}, NJ
            - Source: {random.choice(sources)}
            - Style: Straight journalism (pre-gonzo, factual tone)
            - Must include 2+ of these themes:
              * Rock bands / local music scene
              * Teenage/youth crime or delinquency
              * Unexplained phenomena / weird occurrences
              * Mental health incidents / psychiatric themes
              * Psychedelic drugs / consciousness alteration
    
            MYTHOLOGIZABLE ANGLE:
            The story must have something that COULD become a local legend - an unexplained element, 
            a missing detail, a conspiracy-worthy gap. Think: band that disappeared, frequencies that 
            caused effects, places where time behaved strangely, objects with inexplicable properties.
    
            The story should resonate thematically with White's proposal ({state.white_proposal.concept}).
            """
            claude = self._get_claude()
            proposer = claude.with_structured_output(NewspaperArtifact)
            try:
                result = proposer.invoke(prompt)
                if isinstance(result, dict):
                    # Remove fields that should use class defaults, not LLM-generated values
                    result.pop("chain_artifact_file_type", None)
                    result.pop("chain_artifact_type", None)
                    result.pop("file_name", None)
                    result.pop("file_path", None)
                    result.pop("thread_id", None)  # Will be set below
                    result.pop("artifact_id", None)  # Let class generate UUID

                    result["thread_id"] = state.thread_id
                    result["base_path"] = os.getenv(
                        "AGENT_WORK_PRODUCT_BASE_PATH", "chain_artifacts"
                    )
                    state.synthesized_story = NewspaperArtifact(**result)
                    combined = state.synthesized_story.get_text_content()
                    state.synthesized_story.text = combined
                    try:
                        state.synthesized_story.save_file()
                    except Exception as e:
                        error_msg = f"Failed to save synthesized story file: {e!s}"
                        logger.error(error_msg)
                        if block_mode:
                            raise Exception(error_msg)
                    state.artifacts.append(state.synthesized_story)
                    return state
                elif isinstance(result, NewspaperArtifact):
                    # Override any LLM-generated values with correct ones
                    result.thread_id = state.thread_id
                    result.base_path = os.getenv(
                        "AGENT_WORK_PRODUCT_BASE_PATH", "chain_artifacts"
                    )
                    # Force correct file type (class default is YML)
                    result.chain_artifact_file_type = ChainArtifactFileType.YML
                    # Regenerate file_name with correct values
                    result.get_file_name()
                    state.synthesized_story = result
                    # Ensure text is populated from article content
                    if not state.synthesized_story.text:
                        combined = state.synthesized_story.get_text_content()
                        state.synthesized_story.text = combined
                    try:
                        state.synthesized_story.save_file()
                    except Exception as e:
                        error_msg = f"Failed to save synthesized story file: {e!s}"
                        logger.error(error_msg)
                        if block_mode:
                            raise Exception(error_msg)
                    state.artifacts.append(state.synthesized_story)
                    return state
                else:
                    error_msg = (
                        f"Expected NewspaperArtifact or dict, got {type(result)}"
                    )
                    logger.error(error_msg)
                    if block_mode:
                        raise TypeError(error_msg)
                    fallback_story = NewspaperArtifact(
                        thread_id=state.thread_id,
                        headline="Strange Frequencies Reported Near High School Band Room",
                        base_path=os.getenv(
                            "AGENT_WORK_PRODUCT_BASE_PATH", "chain_artifacts"
                        ),
                        date=story_date.strftime("%Y-%m-%d"),
                        source=random.choice(sources),
                        location=random.choice(locations) + ", NJ",
                        text="Local residents reported unusual electronic sounds emanating from the high school after hours.",
                        tags=["rock_bands", "unexplained"],
                    )
                    state.synthesized_story = fallback_story
                    return state
            except Exception as e:
                error_msg = f"Base story synthesis LLM call failed: {e!s}"
                logger.error(error_msg)
                if block_mode:
                    raise Exception(error_msg)
        get_state_snapshot(
            state, "synthesize_base_story_exit", state.thread_id, "Rows Bud"
        )
        return state

    @agent_error_handler("Rows Bud")
    def add_to_corpus(self, state: OrangeAgentState) -> OrangeAgentState:
        get_state_snapshot(state, "add_to_corpus_entry", state.thread_id, "Rows Bud")
        mock_mode = os.getenv("MOCK_MODE", "false").lower() == "true"
        if mock_mode:
            print("[MOCK] Would call orange-mythos:add_story_to_corpus")
            get_state_snapshot(state, "add_to_corpus_exit", state.thread_id, "Rows Bud")
            return state
        else:
            try:
                story = state.synthesized_story
                if not story.text:
                    logger.warning("Story text is None, using headline as fallback")
                    story.text = story.headline
                story_id, score = self.corpus.add_story(
                    headline=story.headline,
                    date=story.date,
                    source=story.source,
                    text=story.text,
                    location=story.location,
                    tags=story.tags,
                )
                state.selected_story_id = story_id
                print(f"   Added: {story_id} (score: {score:.2f})")
            except Exception as e:
                logger.error(f"Corpus addition failed: {e}")
                state.selected_story_id = f"fallback_{int(time.time() * 1000)}"
            get_state_snapshot(state, "add_to_corpus_exit", state.thread_id, "Rows Bud")
            return state

    @agent_error_handler("Rows Bud")
    def select_symbolic_object(self, state: OrangeAgentState) -> OrangeAgentState:
        """Analyze the story and select a symbolic object category"""
        get_state_snapshot(
            state, "select_symbolic_object_entry", state.thread_id, "Rows Bud"
        )
        mock_mode = os.getenv("MOCK_MODE", "false").lower() == "true"
        block_mode = os.getenv("BLOCK_MODE", "false").lower() == "true"

        if mock_mode:
            try:
                with open(
                    os.path.join(
                        os.getenv("AGENT_MOCK_DATA_PATH"),
                        "orange_mock_object_selection.yml",
                    ),
                    "r",
                ) as f:
                    data = yaml.safe_load(f)
                    data["thread_id"] = state.thread_id
                    data["base_path"] = os.getenv(
                        "AGENT_WORK_PRODUCT_BASE_PATH", "chain_artifacts"
                    )
                    obj = SymbolicObjectArtifact(**data)
                    state.symbolic_object = obj
            except Exception as e:
                error_msg = f"Failed to read mock symbolic object: {e!s}"
                logger.error(error_msg)
                if block_mode:
                    raise Exception(error_msg)
            get_state_snapshot(
                state, "select_symbolic_object_exit", state.thread_id, "Rows Bud"
            )
            return state

        # Non-mock: Actually generate the symbolic object
        else:
            if not state.synthesized_story:
                logger.warning("No synthesized story - creating fallback object")
                state.symbolic_object = SymbolicObjectArtifact(
                    thread_id=state.thread_id,
                    name="mysterious radio equipment",
                    symbolic_object_category=SymbolicObjectCategory.INFORMATION_ARTIFACTS,
                    description="Unexplained electronic broadcasting device",
                    base_path=os.getenv(
                        "AGENT_WORK_PRODUCT_BASE_PATH", "chain_artifacts"
                    ),
                )
                get_state_snapshot(
                    state, "select_symbolic_object_exit", state.thread_id, "Rows Bud"
                )
                return state

            prompt = f"""Analyze this Sussex County newspaper story and select a symbolic object.

    STORY:
    Headline: {state.synthesized_story.headline}
    Date: {state.synthesized_story.date}
    Location: {state.synthesized_story.location}

    Full Text:
    {state.synthesized_story.text}

    WHITE'S CONCEPT (for thematic resonance):
    {state.white_proposal.concept}

    TASK: Select ONE symbolic object that:
    1. Could plausibly exist in 1975-1995 Sussex County, New Jersey
    2. Makes abstract concepts concrete (information → physical object)
    3. Has mythological weight (could become legendary)
    4. Connects to the story's themes and White's concepts
    5. Fits one of these categories:
       - CIRCULAR_TIME: Clocks, calendars, loops, temporal markers
       - INFORMATION_ARTIFACTS: Newspapers, broadcasts, transmissions, recordings
       - LIMINAL_OBJECTS: Doorways, thresholds, portals, boundaries
       - PSYCHOGEOGRAPHIC: Maps, coordinates, dimensional markers

    EXAMPLES:
    - "Nash's 182 BPM clock" (CIRCULAR_TIME) - A metronome that marks impossible rhythm
    - "The EMORY transmission" (INFORMATION_ARTIFACTS) - A persistent broadcast signal
    - "The Loomis Avenue threshold" (LIMINAL_OBJECTS) - A doorway between states
    - "Route 23 temporal marker" (PSYCHOGEOGRAPHIC) - A highway mile marker where time shifts

    Return a SymbolicObjectArtifact with:
    - name: Specific, evocative (e.g., "The Sussex Frequency Analyzer")
    - symbolic_object_category: One of the four categories above
    - description: 1-2 sentences about what it is and why it matters
    """

            claude = self._get_claude()
            proposer = claude.with_structured_output(SymbolicObjectArtifact)

            try:
                result = proposer.invoke(prompt)
                if isinstance(result, dict):
                    # Remove fields that should use class defaults, not LLM-generated values
                    result.pop("chain_artifact_file_type", None)
                    result.pop("chain_artifact_type", None)
                    result.pop("file_name", None)
                    result.pop("file_path", None)
                    result.pop("thread_id", None)  # Will be set below
                    result.pop("artifact_id", None)  # Let class generate UUID

                    result["thread_id"] = state.thread_id
                    result["base_path"] = os.getenv(
                        "AGENT_WORK_PRODUCT_BASE_PATH", "chain_artifacts"
                    )
                    state.symbolic_object = SymbolicObjectArtifact(**result)
                elif isinstance(result, SymbolicObjectArtifact):
                    # Override any LLM-generated values with correct ones
                    result.thread_id = state.thread_id
                    result.base_path = os.getenv(
                        "AGENT_WORK_PRODUCT_BASE_PATH", "chain_artifacts"
                    )
                    # Force correct file type (class default is YML)
                    result.chain_artifact_file_type = ChainArtifactFileType.YML
                    # Regenerate file_name with correct values
                    result.get_file_name()
                    state.symbolic_object = result
                else:
                    error_msg = (
                        f"Expected SymbolicObjectArtifact or dict, got {type(result)}"
                    )
                    logger.warning(error_msg)
                    if block_mode:
                        raise TypeError(error_msg)
                    state.symbolic_object = SymbolicObjectArtifact(
                        thread_id=state.thread_id,
                        name="mysterious radio equipment",
                        symbolic_object_category=SymbolicObjectCategory.INFORMATION_ARTIFACTS,
                        description="Unexplained electronic device found at the scene",
                        base_path=os.getenv(
                            "AGENT_WORK_PRODUCT_BASE_PATH", "chain_artifacts"
                        ),
                    )

                logger.info(f"Selected symbolic object: {state.symbolic_object.name}")

            except Exception as e:
                error_msg = f"Symbolic object selection LLM call failed: {e!s}"
                logger.error(error_msg)
                if block_mode:
                    raise Exception(error_msg)
                state.symbolic_object = SymbolicObjectArtifact(
                    thread_id=state.thread_id,
                    name="unidentified transmission device",
                    symbolic_object_category=SymbolicObjectCategory.INFORMATION_ARTIFACTS,
                    description="Electronic equipment of unknown origin",
                    base_path=os.getenv(
                        "AGENT_WORK_PRODUCT_BASE_PATH", "chain_artifacts"
                    ),
                )

            get_state_snapshot(
                state, "select_symbolic_object_exit", state.thread_id, "Rows Bud"
            )
            return state

    @agent_error_handler("Rows Bud")
    def insert_symbolic_object_node(self, state: OrangeAgentState) -> OrangeAgentState:
        """Insert a symbolic object into the story via MCP tool"""
        get_state_snapshot(
            state, "insert_symbolic_object_node_entry", state.thread_id, "Rows Bud"
        )
        if not state.symbolic_object:
            logger.warning("No symbolic object to insert - skipping")
            get_state_snapshot(
                state, "insert_symbolic_object_node_exit", state.thread_id, "Rows Bud"
            )
            return state
        if not state.selected_story_id:
            logger.warning("No story ID for object insertion - skipping")
            get_state_snapshot(
                state, "insert_symbolic_object_node_exit", state.thread_id, "Rows Bud"
            )
            return state
        mock_mode = os.getenv("MOCK_MODE", "false").lower() == "true"
        if mock_mode:
            print(f"[MOCK] Would insert: {state.symbolic_object.name} into story")
            get_state_snapshot(
                state, "insert_symbolic_object_node_exit", state.thread_id, "Rows Bud"
            )
            return state
        try:
            object_desc = state.symbolic_object.description
            object_category = state.symbolic_object.symbolic_object_category.value

            # Get the story first
            story = self.corpus.get_story(state.selected_story_id)
            if not story:
                logger.warning(f"Story {state.selected_story_id} not found in corpus")
                get_state_snapshot(
                    state,
                    "insert_symbolic_object_node_exit",
                    state.thread_id,
                    "Rows Bud",
                )
                return state

            # Build the prompt (same as MCP tool)
            prompt = f"""Insert this symbolic object into the story naturally and seamlessly.

                    ORIGINAL STORY:
                    {story.get('text', '')}

                    SYMBOLIC OBJECT: {object_desc}
                    CATEGORY: {object_category}

                    CRITICAL RULES:
                    - The object did NOT exist in the original story
                    - Insert it as if it was always there and was discovered/noticed
                    - Make it feel central to the narrative, not forced
                    - Keep the journalistic tone (this is pre-gonzo rewriting)
                    - The object should raise questions, create mystery
                    - It should feel like a detail the original journalist might have overlooked or downplayed

                    LOCATION CONTEXT: {story.get('location', 'Sussex County')} in {story.get('date', '1980s')}

                    Return ONLY the updated story text with the object naturally integrated. 
                    Do not add any preamble or explanation."""

            # Make Anthropic API call (same as MCP tool)
            response = self.llm.invoke([HumanMessage(content=prompt)], max_tokens=2000)
            updated_text = response.content.strip()

            self.corpus.insert_symbolic_object(
                story_id=state.selected_story_id,
                category=object_category,
                description=object_desc,
                updated_text=updated_text,
            )

            # Save the symbolic object artifact
            if state.symbolic_object.base_path in (None, "/"):
                state.symbolic_object.base_path = os.getenv(
                    "AGENT_WORK_PRODUCT_BASE_PATH", "chain_artifacts"
                )
            if state.symbolic_object.thread_id in (None, "UNKNOWN_THREAD_ID"):
                state.symbolic_object.thread_id = state.thread_id
            try:
                state.symbolic_object.save_file()
                state.artifacts.append(state.symbolic_object)
            except Exception as e:
                logger.warning(f"Could not save symbolic object artifact: {e!s}")

            logger.info(f"Object inserted: {state.symbolic_object.name}")

        except Exception as e:
            logger.error(f"Object insertion failed: {e!s}")

        get_state_snapshot(
            state, "insert_symbolic_object_node_exit", state.thread_id, "Rows Bud"
        )
        return state

    @agent_error_handler("Rows Bud")
    def gonzo_rewrite_node(self, state: OrangeAgentState) -> OrangeAgentState:
        """Rewrite the story in gonzo journalism style via MCP tool"""
        get_state_snapshot(
            state, "gonzo_rewrite_node_entry", state.thread_id, "Rows Bud"
        )
        mock_mode = os.getenv("MOCK_MODE", "false").lower() == "true"
        block_mode = os.getenv("BLOCK_MODE", "false").lower() == "true"
        if not state.synthesized_story:
            logger.warning("No synthesized story for gonzo rewrite - using fallback")
            get_state_snapshot(
                state, "gonzo_rewrite_node_exit", state.thread_id, "Rows Bud"
            )
            return state
        if mock_mode:
            try:
                with open(
                    os.path.join(
                        os.getenv("AGENT_MOCK_DATA_PATH"), "orange_gonzo_rewrite.yml"
                    ),
                    "r",
                ) as f:
                    data = yaml.safe_load(f)
                    data["thread_id"] = state.thread_id
                    data["base_path"] = os.getenv(
                        "AGENT_WORK_PRODUCT_BASE_PATH", "chain_artifacts"
                    )
                    new_article = NewspaperArtifact(**data)
                    state.mythologized_story = new_article
                    # Match non-mock processing
                    combined = state.mythologized_story.get_text_content()
                    state.mythologized_story.text = combined
                    try:
                        state.mythologized_story.save_file()
                    except Exception as e:
                        logger.warning(
                            f"Mock mode: Could not save mythologized story file: {e!s}"
                        )
                    state.artifacts.append(state.mythologized_story)
            except Exception as e:
                error_msg = f"Failed to read mock gonzo rewrite: {e!s}"
                logger.error(error_msg)
                if block_mode:
                    raise Exception(error_msg)
            get_state_snapshot(
                state, "gonzo_rewrite_node_exit", state.thread_id, "Rows Bud"
            )
            return state
        try:
            story = self.corpus.get_story(state.selected_story_id)
            if not story:
                logger.warning("Could not retrieve story, using synthesized story")
                story = {
                    "headline": state.synthesized_story.headline,
                    "date": state.synthesized_story.date,
                    "location": state.synthesized_story.location,
                    "text": state.synthesized_story.text,
                    "source": state.synthesized_story.source,
                    "tags": state.synthesized_story.tags,
                }
            intensity_styles = {
                1: "Subtle first-person observer",
                2: "Embedded journalist, growing suspicion",
                3: "Active participant, perception shifts begin",
                4: "Deep paranoia, conspiracy emerging",
                5: "Full Hunter S. Thompson - reality unhinged",
            }
            obj_desc = story.get("symbolic_object_desc", "a mysterious object")
            prompt = f"""Rewrite this Sussex County story in gonzo journalism style.

            ORIGINAL STORY:
            Headline: {story['headline']}
            Date: {story['date']}
            Location: {story['location']}
    
            {story['text']}
    
            SYMBOLIC OBJECT (central to rewrite): {obj_desc}
    
            GONZO PARAMETERS:
            - Perspective: {state.gonzo_perspective} (first-person, embedded)
            - Intensity: {state.gonzo_intensity}/5 - {intensity_styles[state.gonzo_intensity]}
    
            CHARACTERISTICS:
            1. First-person embedded journalism
            2. Paranoia & conspiracy - official story doesn't add up
            3. Perception shifts - reality feels unstable
            4. Authority distrust - police/officials hiding truth
            5. Vivid sensory details - Pine smell, electronic hum
            6. Symbolic object is THE KEY - proof, evidence, impossible but real
            7. Investigator becomes part of story
            8. Sussex County mythology - New Jersey gothic
    
            Return ONLY the gonzo rewritten story."""

            response = self.llm.invoke(
                [HumanMessage(content=prompt)],
                temperature=0.8 + (state.gonzo_intensity * 0.05),
                max_tokens=3000,
            )
            gonzo_text = response.content.strip()

            self.corpus.add_gonzo_rewrite(
                story_id=state.selected_story_id,
                gonzo_text=gonzo_text,
                perspective=state.gonzo_perspective,
                intensity=state.gonzo_intensity,
            )
            story_without_text = {k: v for k, v in story.items() if k != "text"}
            state.mythologized_story = NewspaperArtifact(
                **story_without_text,
                text=gonzo_text,
                thread_id=state.thread_id,
                base_path=os.getenv("AGENT_WORK_PRODUCT_BASE_PATH", "chain_artifacts"),
            )

            combined = state.mythologized_story.get_text_content()
            state.mythologized_story.text = combined
            try:
                state.mythologized_story.save_file()
            except Exception as e:
                error_msg = f"Failed to save mythologized story file: {e!s}"
                logger.error(error_msg)
                if block_mode:
                    raise Exception(error_msg)
            state.artifacts.append(state.mythologized_story)
            get_state_snapshot(
                state, "gonzo_rewrite_node_exit", state.thread_id, "Rows Bud"
            )
            return state

        except Exception as e:
            error_msg = f"Gonzo rewrite LLM call failed: {e!s}"
            logger.error(error_msg)
            if block_mode:
                raise Exception(error_msg)
            state.mythologized_story = state.synthesized_story

        get_state_snapshot(
            state, "gonzo_rewrite_node_exit", state.thread_id, "Rows Bud"
        )
        return state
