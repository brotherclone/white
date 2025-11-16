import logging
import os
import random
import time
import yaml

from abc import ABC
from datetime import datetime, timedelta
from typing import cast, Iterable, Any, Optional
from anthropic import Anthropic
from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langgraph.constants import END, START
from langgraph.graph.state import StateGraph
from pydantic import Field

from app.agents.states.orange_agent_state import OrangeAgentState
from app.agents.states.white_agent_state import MainAgentState
from app.agents.tools.text_tools import save_artifact_to_md
from app.reference.mcp.rows_bud.orange_corpus import OrangeMythosCorpus, get_corpus
from app.reference.mcp.rows_bud.orange_mythos_server import insert_symbolic_object
from app.structures.agents.base_rainbow_agent import BaseRainbowAgent
from app.structures.artifacts.newspaper_artifact import NewspaperArtifact
from app.structures.artifacts.symbolic_object_artifact import SymbolicObjectArtifact
from app.structures.artifacts.text_artifact_file import TextChainArtifactFile
from app.structures.concepts.rainbow_table_color import the_rainbow_table_colors
from app.structures.enums.chain_artifact_file_type import ChainArtifactFileType
from app.structures.manifests.song_proposal import SongProposalIteration
from app.util.manifest_loader import get_my_reference_proposals

load_dotenv()


class OrangeAgent(BaseRainbowAgent, ABC):
    """Sussex Mythologizer - Rows Bud, the 182 BPM transmission keeper"""

    corpus: Optional[OrangeMythosCorpus] = Field(default=None)
    anthropic_client: Optional[Anthropic] = Field(default=None)

    def __init__(self, **data):
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
        work_flow.add_edge("generate_alternate_song_spec", END)

        return work_flow

    def generate_alternate_song_spec(self, state: OrangeAgentState) -> OrangeAgentState:
        mock_mode = os.getenv("MOCK_MODE", "false").lower() == "true"
        if mock_mode:
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

    def synthesize_base_story(self, state: OrangeAgentState) -> OrangeAgentState:
        """Synthesize a plausible Sussex County newspaper story (1975-1995)"""
        mock_mode = os.getenv("MOCK_MODE", "false").lower() == "true"
        if mock_mode:
            with open(
                os.path.join(
                    os.getenv("AGENT_MOCK_DATA_PATH"), "orange_base_story_mock.yml"
                ),
                "r",
            ) as f:
                data = yaml.safe_load(f)
                story = NewspaperArtifact(**data)
                state.synthesized_story = story
                state.artifacts.append(story)
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
                    if result["thread_id"] is None:
                        result["thread_id"] = state.thread_id
                    state.synthesized_story = NewspaperArtifact(**result)
                    combined = state.synthesized_story.get_text_content()
                    text_artifact = TextChainArtifactFile(
                        thread_id=state.thread_id,
                        text_content=combined,
                        rainbow_color=the_rainbow_table_colors["O"],
                        base_path=os.getenv("AGENT_WORK_PRODUCT_BASE_PATH"),
                        chain_artifact_file_type=ChainArtifactFileType.MARKDOWN,
                    )
                    save_artifact_to_md(text_artifact)
                    state.artifacts.append(text_artifact)
                    state.artifacts.append(state.synthesized_story)
                    return state
                else:
                    fallback_story = NewspaperArtifact(
                        thread_id=state.thread_id,
                        headline="Strange Frequencies Reported Near High School Band Room",
                        date=story_date.strftime("%Y-%m-%d"),
                        source=random.choice(sources),
                        location=random.choice(locations) + ", NJ",
                        text="Local residents reported unusual electronic sounds emanating from the high school after hours.",
                        tags=["rock_bands", "unexplained"],
                    )
                    state.synthesized_story = fallback_story
                    return state
            except Exception as e:
                logging.error(f"Anthropic model call failed: {e!s}")
        return state

    def add_to_corpus(self, state: OrangeAgentState) -> OrangeAgentState:
        """Add a synthesized story to mythology corpus (optional, for training data)"""
        mock_mode = os.getenv("MOCK_MODE", "false").lower() == "true"
        if mock_mode:
            print("[MOCK] Would call orange-mythos:add_story_to_corpus")
            return state
        try:
            story = state.synthesized_story
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
            logging.error(f"Corpus addition failed: {e}")
            state.selected_story_id = f"fallback_{int(time.time() * 1000)}"
        return state

    def select_symbolic_object(self, state: OrangeAgentState) -> OrangeAgentState:
        """Analyze the story and select a symbolic object category"""
        mock_mode = os.getenv("MOCK_MODE", "false").lower() == "true"
        if mock_mode:
            with open(
                os.path.join(
                    os.getenv("AGENT_MOCK_DATA_PATH"),
                    "orange_mock_object_selection.yml",
                ),
                "r",
            ) as f:
                data = yaml.safe_load(f)
                data["thread_id"] = state.thread_id
                obj = SymbolicObjectArtifact(**data)
                state.symbolic_object = obj
            return state
        else:
            try:
                story = self.corpus.get_story(state.selected_story_id)
                prompt = f"""Insert this symbolic object into the story naturally.
    
                ORIGINAL STORY:
                {story['text']}
    
                SYMBOLIC OBJECT: {state.symbolic_object}
                CATEGORY: {state.symbolic_object.symbolic_object_category}
    
                RULES:
                - Object did NOT exist in original story
                - Insert as if it was always there
                - Keep journalistic tone (pre-gonzo)
                - Make it central to the narrative
    
                Return ONLY the updated story text."""
                response = self.anthropic_client.messages.create(
                    model="claude-sonnet-4-20250514",
                    max_tokens=2000,
                    messages=cast(Iterable[Any], [{"role": "user", "content": prompt}]),
                )
                updated_text = response.content[0].text.strip()
                self.corpus.insert_symbolic_object(
                    story_id=state.selected_story_id,
                    category=state.symbolic_object.symbolic_object_category,
                    description=state.symbolic_object.name,
                    updated_text=updated_text,
                )
                state.synthesized_story.text = updated_text
                print(f"   Object inserted: {state.symbolic_object.name}")

            except Exception as e:
                logging.error(f"Object insertion failed: {e}")

        return state

    @staticmethod
    def insert_symbolic_object_node(state: OrangeAgentState) -> OrangeAgentState:
        """Insert a symbolic object into the story via MCP tool"""
        mock_mode = os.getenv("MOCK_MODE", "false").lower() == "true"
        if mock_mode:
            print(f"[MOCK] Would insert: {state.symbolic_object.name} into story")
            return state
        try:
            insert_symbolic_object(
                story_id=state.selected_story_id,
                object_category=state.symbolic_object.symbolic_object_category,
                custom_object=state.symbolic_object.name,
            )
            print(f"Object inserted: {state.symbolic_object.name}")
        except Exception as e:
            logging.error(f"Object insertion failed: {e!s}")

        return state

    def gonzo_rewrite_node(self, state: OrangeAgentState) -> OrangeAgentState:
        """Rewrite the story in gonzo journalism style via MCP tool"""
        mock_mode = os.getenv("MOCK_MODE", "false").lower() == "true"
        if mock_mode:
            with open(
                os.path.join(
                    os.getenv("AGENT_MOCK_DATA_PATH"), "orange_gonzo_rewrite.yml"
                ),
                "r",
            ) as f:
                data = yaml.safe_load(f)
                new_article = NewspaperArtifact(**data)
                state.mythologized_story = new_article
            return state
        try:
            story = self.corpus.get_story(state.selected_story_id)
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

            response = self.anthropic_client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=3000,
                temperature=0.8 + (state.gonzo_intensity * 0.05),
                messages=cast(Iterable[Any], [{"role": "user", "content": prompt}]),
            )

            gonzo_text = response.content[0].text.strip()

            self.corpus.add_gonzo_rewrite(
                story_id=state.selected_story_id,
                gonzo_text=gonzo_text,
                perspective=state.gonzo_perspective,
                intensity=state.gonzo_intensity,
            )
            state.mythologized_story = NewspaperArtifact(**story, text=gonzo_text)
            combined = state.mythologized_story.get_text_content()
            text_artifact = TextChainArtifactFile(
                thread_id=state.thread_id,
                text_content=combined,
                rainbow_color=the_rainbow_table_colors["O"],
                base_path=os.getenv("AGENT_WORK_PRODUCT_BASE_PATH"),
                chain_artifact_file_type=ChainArtifactFileType.MARKDOWN,
            )
            save_artifact_to_md(text_artifact)
            state.artifacts.append(text_artifact)
            state.artifacts.append(state.synthesized_story)
            return state

        except Exception as e:
            logging.error(f"Gonzo rewrite failed: {e}")
            state.mythologized_story = state.synthesized_story

        return state
