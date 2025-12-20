import logging
import os
import time
import yaml

from pathlib import Path
from typing import Any, Dict, List
from uuid import uuid4
from langchain_anthropic import ChatAnthropic
from langchain_core.runnables.config import RunnableConfig
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.constants import START
from langgraph.graph import END, StateGraph
from langgraph.graph.state import CompiledStateGraph
from pydantic import BaseModel

from app.agents.black_agent import BlackAgent
from app.agents.blue_agent import BlueAgent
from app.agents.green_agent import GreenAgent
from app.agents.indigo_agent import IndigoAgent
from app.agents.orange_agent import OrangeAgent
from app.agents.red_agent import RedAgent
from app.agents.states.white_agent_state import MainAgentState
from app.agents.tools.text_tools import save_markdown
from app.agents.violet_agent import VioletAgent
from app.agents.workflow.resume_black_workflow import (
    resume_black_agent_workflow_with_agent,
)
from app.agents.yellow_agent import YellowAgent
from app.structures.agents.agent_settings import AgentSettings
from app.structures.artifacts.base_artifact import ChainArtifact
from app.structures.concepts.white_facet_system import WhiteFacetSystem
from app.structures.enums.chain_artifact_type import ChainArtifactType
from app.structures.manifests.song_proposal import SongProposal, SongProposalIteration

logging.basicConfig(level=logging.INFO)


class WhiteAgent(BaseModel):
    agents: Dict[str, Any] = {}
    settings: AgentSettings = AgentSettings()
    song_proposal: SongProposal = SongProposal(iterations=[])

    def __init__(self, **data):
        if "settings" not in data or data["settings"] is None:
            data["settings"] = AgentSettings()
        if "agents" not in data:
            data["agents"] = {}
        super().__init__(**data)
        self.agents = {
            "black": BlackAgent(),
            "red": RedAgent(),
            "orange": OrangeAgent(),
            "yellow": YellowAgent(),
            "green": GreenAgent(),
            "blue": BlueAgent(),
            "indigo": IndigoAgent(),
            "violet": VioletAgent(),
        }

    def start_workflow(self, user_input: str | None = None) -> MainAgentState:
        """
        Start a new White Agent workflow from the beginning.

        Args:
            user_input: Optional user input to guide the initial proposal

        Returns:
            The final state after workflow completion (or pause)
        """
        if user_input is not None:
            logging.info(f"User input provided to White Agent: {user_input}")

        workflow = self.build_workflow()
        thread_id = str(uuid4())
        initial_state = MainAgentState(
            thread_id=thread_id,
            song_proposals=SongProposal(iterations=[]),
            artifacts=[],
            workflow_paused=False,
            ready_for_red=False,
            ready_for_orange=False,
            ready_for_yellow=False,
            ready_for_green=False,
            ready_for_blue=False,
            ready_for_indigo=False,
            ready_for_violet=False,
            ready_for_white=False,
            run_finished=False,
        )
        config = RunnableConfig(configurable={"thread_id": thread_id})
        logging.info(f"Starting White Agent workflow (thread_id: {thread_id})")
        result = workflow.invoke(initial_state, config)
        if isinstance(result, dict):
            final_state = MainAgentState(**result)
        else:
            final_state = result
        return final_state

    def resume_workflow(
        self, paused_state: MainAgentState, verify_tasks: bool = True
    ) -> MainAgentState:

        if not paused_state.workflow_paused:
            logging.warning("⚠️ Workflow is not paused - nothing to resume")
            return paused_state
        logging.info(f"Resuming workflow (thread_id: {paused_state.thread_id})")
        updated_state = self.resume_after_black_agent_ritual(
            paused_state, verify_tasks=verify_tasks
        )
        if updated_state.ready_for_red:
            logging.info("Continuing to Red Agent...")
            updated_state = self.invoke_red_agent(updated_state)
            updated_state = self.process_red_agent_work(updated_state)
            if updated_state.ready_for_orange:
                updated_state = self.invoke_orange_agent(updated_state)
                updated_state = self.process_orange_agent_work(updated_state)
            if updated_state.ready_for_yellow:
                updated_state = self.invoke_yellow_agent(updated_state)
                updated_state = self.process_yellow_agent_work(updated_state)
            if updated_state.ready_for_green:
                updated_state = self.invoke_green_agent(updated_state)
                updated_state = self.process_green_agent_work(updated_state)
            if updated_state.ready_for_blue:
                updated_state = self.invoke_blue_agent(updated_state)
                updated_state = self.process_blue_agent_work(updated_state)
            if updated_state.ready_for_indigo:
                updated_state = self.invoke_indigo_agent(updated_state)
                updated_state = self.process_indigo_agent_work(updated_state)
            if updated_state.ready_for_violet:
                updated_state = self.invoke_violet_agent(updated_state)
                updated_state = self.process_violet_agent_work(updated_state)
        updated_state = self.finalize_song_proposal(updated_state)
        return updated_state

    def build_workflow(self) -> CompiledStateGraph:
        check_points = InMemorySaver()
        workflow = StateGraph(MainAgentState)
        # ⚪️
        workflow.add_node("initiate_song_proposal", self.initiate_song_proposal)
        # ⚫️
        workflow.add_node("invoke_black_agent", self.invoke_black_agent)
        workflow.add_node("process_black_agent_work", self.process_black_agent_work)
        # 🔴
        workflow.add_node("invoke_red_agent", self.invoke_red_agent)
        workflow.add_node("process_red_agent_work", self.process_red_agent_work)
        # 🟠
        workflow.add_node("invoke_orange_agent", self.invoke_orange_agent)
        workflow.add_node("process_orange_agent_work", self.process_orange_agent_work)
        # 🟡
        workflow.add_node("invoke_yellow_agent", self.invoke_yellow_agent)
        workflow.add_node("process_yellow_agent_work", self.process_yellow_agent_work)
        # 🟢
        workflow.add_node("invoke_green_agent", self.invoke_green_agent)
        workflow.add_node("process_green_agent_work", self.process_green_agent_work)
        # 🔵
        workflow.add_node("invoke_blue_agent", self.invoke_blue_agent)
        workflow.add_node("process_blue_agent_work", self.process_blue_agent_work)
        # 🩵
        workflow.add_node("invoke_indigo_agent", self.invoke_indigo_agent)
        workflow.add_node("process_indigo_agent_work", self.process_indigo_agent_work)
        # 🟣
        workflow.add_node("invoke_violet_agent", self.invoke_violet_agent)
        workflow.add_node("process_violet_agent_work", self.process_violet_agent_work)
        # ⚪️
        workflow.add_node(
            "rewrite_proposal_with_synthesis", self.rewrite_proposal_with_synthesis
        )
        workflow.add_node("finalize_song_proposal", self.finalize_song_proposal)
        # Edges
        # ⚪️
        workflow.add_edge(START, "initiate_song_proposal")
        # ⚪️ ⚫️
        workflow.add_edge("initiate_song_proposal", "invoke_black_agent")
        # ⚫️ ⚪️
        workflow.add_edge("invoke_black_agent", "process_black_agent_work")
        workflow.add_edge("process_black_agent_work", "rewrite_proposal_with_synthesis")
        # ⚪️ 🔴
        workflow.add_edge("invoke_red_agent", "process_red_agent_work")
        # 🔴 ⚪️
        workflow.add_edge("process_red_agent_work", "rewrite_proposal_with_synthesis")
        # ⚪️ 🟠
        workflow.add_edge("invoke_orange_agent", "process_orange_agent_work")
        # 🟠 ⚪
        workflow.add_edge(
            "process_orange_agent_work", "rewrite_proposal_with_synthesis"
        )
        # ⚪️ 🟡
        workflow.add_edge("invoke_yellow_agent", "process_yellow_agent_work")
        # 🟡 ⚪️
        workflow.add_edge(
            "process_yellow_agent_work", "rewrite_proposal_with_synthesis"
        )
        # ⚪️ 🟢
        workflow.add_edge("invoke_green_agent", "process_green_agent_work")
        # 🟢 ⚪
        workflow.add_edge("process_green_agent_work", "rewrite_proposal_with_synthesis")
        # ⚪️ 🔵
        workflow.add_edge("invoke_blue_agent", "process_blue_agent_work")
        # 🔵 ⚪
        workflow.add_edge("process_blue_agent_work", "rewrite_proposal_with_synthesis")
        # ⚪️ 🩵
        workflow.add_edge("invoke_indigo_agent", "process_indigo_agent_work")
        # 🩵 ⚪️
        workflow.add_edge(
            "process_indigo_agent_work", "rewrite_proposal_with_synthesis"
        )
        # ⚪️ 🟣
        workflow.add_edge("invoke_violet_agent", "process_violet_agent_work")
        # 🟣 ⚪️
        workflow.add_edge(
            "process_violet_agent_work", "rewrite_proposal_with_synthesis"
        )
        # ⚫️🫀 ⚪️
        workflow.add_conditional_edges(
            "process_black_agent_work",
            self.route_after_black,
            {"red": "rewrite_proposal_with_synthesis", "black": "invoke_black_agent"},
        )
        # ⚪️ ⚫ ⚪️ 🔴 ⚪️ 🟠 ⚪️ 🟡 ⚪️ 🟢 ⚪ ️🔵 ⚪️ 🩵 ⚪ ️🟣 ⚪️#
        workflow.add_conditional_edges(
            "rewrite_proposal_with_synthesis",
            self.route_after_rewrite,
            {
                "black": "invoke_black_agent",
                "red": "invoke_red_agent",
                "orange": "invoke_orange_agent",
                "yellow": "invoke_yellow_agent",
                "green": "invoke_green_agent",
                "blue": "invoke_blue_agent",
                "indigo": "invoke_indigo_agent",
                "violet": "invoke_violet_agent",
                "white": "finalize_song_proposal",
            },
        )
        # ⚪️
        workflow.add_edge("finalize_song_proposal", END)
        return workflow.compile(checkpointer=check_points)

    def _get_claude_supervisor(self) -> ChatAnthropic:
        return ChatAnthropic(
            model_name=self.settings.anthropic_model_name,
            api_key=self.settings.anthropic_api_key,
            temperature=self.settings.temperature,
            max_retries=self.settings.max_retries,
            timeout=self.settings.timeout,
            stop=self.settings.stop,
        )

    @staticmethod
    def _normalize_song_proposal(proposal):
        """
        Ensures proposal is a SongProposal instance.
        Accepts dict or SongProposal, returns SongProposal.
        """
        if isinstance(proposal, SongProposal):
            return proposal
        elif isinstance(proposal, dict):
            return SongProposal(**proposal)
        elif proposal is None:
            return SongProposal(iterations=[])
        else:
            raise TypeError(f"Cannot normalize proposal of type {type(proposal)}")

    # 📣⚫📣🔴📣🟠📣🟡📣🟢📣🔵📣🩵📣🟣📣#

    def invoke_black_agent(self, state: MainAgentState) -> MainAgentState:
        """Invoke Black Agent to generate counter-proposal"""
        logging.info("  📣  ⚫  Calling upon ThreadKeepr  ⚫️  📣")
        if "black" not in self.agents:
            self.agents["black"] = BlackAgent(settings=self.settings)
        return self.agents["black"](state)

    def invoke_red_agent(self, state: MainAgentState) -> MainAgentState:
        """Invoke Red Agent with the first synthesized proposal from Black Agent"""
        logging.info("📣  🔴  Calling upon The Light Reader  🔴  📣")
        if "red" not in self.agents:
            self.agents["red"] = RedAgent(settings=self.settings)
        return self.agents["red"](state)

    def invoke_orange_agent(self, state: MainAgentState) -> MainAgentState:
        """Invoke Orange Agent with the synthesized proposal"""
        logging.info("📣  🟠  Calling upon Rows Bud  🟠  📣")
        if "orange" not in self.agents:
            self.agents["orange"] = OrangeAgent(settings=self.settings)
        return self.agents["orange"](state)

    def invoke_yellow_agent(self, state: MainAgentState) -> MainAgentState:
        """Invoke Yellow Agent with the synthesized proposal"""
        logging.info("📣  🟡  Calling upon Lord Pulsimore  🟡  📣")
        if "yellow" not in self.agents:
            self.agents["yellow"] = YellowAgent(settings=self.settings)
        return self.agents["yellow"](state)

    def invoke_green_agent(self, state: MainAgentState) -> MainAgentState:
        """Invoke Green Agent with the synthesized proposal"""
        logging.info("📣  🟢  Calling upon Sub-Arbitrary  🟢  📣")
        if "green" not in self.agents:
            self.agents["green"] = GreenAgent(settings=self.settings)
        return self.agents["green"](state)

    def invoke_blue_agent(self, state: MainAgentState) -> MainAgentState:
        """Invoke Blue Agent with the synthesized proposal"""
        logging.info("📣  🔵  Calling upon The Cassette Bearer  🔵  📣")
        if "blue" not in self.agents:
            self.agents["blue"] = BlueAgent(settings=self.settings)
        return self.agents["blue"](state)

    def invoke_indigo_agent(self, state: MainAgentState) -> MainAgentState:
        """Invoke Indigo Agent with the synthesized proposal"""
        logging.info("📣  🩵  Calling upon Decider Tangents 🩵  📣")
        if "indigo" not in self.agents:
            self.agents["indigo"] = IndigoAgent(settings=self.settings)
        return self.agents["indigo"](state)

    def invoke_violet_agent(self, state: MainAgentState) -> MainAgentState:
        """Invoke Violet Agent with the synthesized proposal"""
        logging.info("📣  🟣  Calling upon The Sultan of Solipsism  🟣  📣")
        if "violet" not in self.agents:
            self.agents["violet"] = VioletAgent(settings=self.settings)
        return self.agents["violet"](state)

    # 📣⚫📣🔴📣🟠📣🟡📣🟢📣🔵📣🩵📣🟣📣#

    def initiate_song_proposal(self, state: MainAgentState) -> MainAgentState:
        mock_mode = os.getenv("MOCK_MODE", "false").lower() == "true"
        block_mode = os.getenv("BLOCK_MODE", "false").lower() == "true"
        prompt, facet = WhiteFacetSystem.build_white_initial_prompt(
            user_input=None, use_weights=True
        )
        facet_metadata = WhiteFacetSystem.log_facet_selection(facet)
        logging.info(f" White Agent using {facet.value.upper()} lens")
        logging.info(f" {facet_metadata['description']}")
        if mock_mode:
            state.thread_id = "mock_thread_001"
            try:
                with open(
                    f"{os.getenv('AGENT_MOCK_DATA_PATH')}/white_initial_proposal_{facet.value}_mock.yml",
                    "r",
                ) as f:
                    data = yaml.safe_load(f)
                    proposal = SongProposalIteration(**data)
                    if (
                        not hasattr(state, "song_proposals")
                        or state.song_proposals is None
                    ):
                        state.song_proposals = SongProposal(iterations=[])
                    sp = self._normalize_song_proposal(state.song_proposals)
                    sp.iterations.append(proposal)
                    state.song_proposals = sp
            except Exception as e:
                logging.info(
                    f"Mock initial proposal not found, returning stub SongProposalIteration:{e!s}"
                )
            return state
        claude = self._get_claude_supervisor()
        proposer = claude.with_structured_output(SongProposalIteration)
        try:
            initial_proposal = proposer.invoke(prompt)
            if isinstance(initial_proposal, dict):
                initial_proposal = SongProposalIteration(**initial_proposal)
                state.white_facet = facet
                state.white_facet_metadata = facet_metadata
            if not isinstance(initial_proposal, SongProposalIteration):
                error_msg = (
                    f"Expected SongProposalIteration, got {type(initial_proposal)}"
                )
                if block_mode:
                    raise TypeError(error_msg)
                logging.warning(error_msg)
        except Exception as e:
            logging.info(
                f"Anthropic model call failed: {e!s}; returning stub SongProposalIteration."
            )
            if block_mode:
                raise Exception("Anthropic model call failed")
            timestamp = int(time.time() * 1000)
            initial_proposal = SongProposalIteration(
                iteration_id=f"fallback_error_{timestamp}",
                bpm=120,
                tempo="4/4",
                key="C Major",
                rainbow_color="white",
                title="Fallback: White Song",
                mood=["reflective"],
                genres=["art-pop"],
                concept="Fallback stub because Anthropic model unavailable. Fallback stub because Anthropic model unavailable. Fallback stub because Anthropic model unavailable. Fallback stub because Anthropic model unavailable.",
            )
        if not hasattr(state, "song_proposals") or state.song_proposals is None:
            state.song_proposals = SongProposal(iterations=[])
        sp = self._normalize_song_proposal(state.song_proposals)
        sp.iterations.append(initial_proposal)
        state.song_proposals = sp
        return state

    def rewrite_proposal_with_synthesis(self, state: MainAgentState) -> MainAgentState:
        mock_mode = os.getenv("MOCK_MODE", "false").lower() == "true"
        block_mode = os.getenv("BLOCK_MODE", "false").lower() == "true"
        if mock_mode:
            yml_color = "red"
            if state.ready_for_orange:
                yml_color = "orange"
            elif state.ready_for_yellow:
                yml_color = "yellow"
            elif state.ready_for_green:
                yml_color = "green"
            elif state.ready_for_blue:
                yml_color = "blue"
            elif state.ready_for_indigo:
                yml_color = "indigo"
            elif state.ready_for_violet:
                yml_color = "violet"
            mock_name = f"white_initial_proposal_{yml_color}_mock.yml"
            if state.ready_for_white:
                mock_name = "final_white_song_proposal_mock.yml"
            try:
                with open(
                    f"{os.getenv('AGENT_MOCK_DATA_PATH')}/{mock_name}",
                    "r",
                ) as f:
                    data = yaml.safe_load(f)
                    proposal = SongProposalIteration(**data)
                    if not isinstance(proposal, SongProposalIteration):
                        error_msg = (
                            f"Expected SongProposalIteration, got {type(proposal)}"
                        )
                        if block_mode:
                            raise TypeError(error_msg)
                        logging.warning(error_msg)
            except Exception as e:
                logging.info(f"Mock reworked proposal not found:{e!s}")
            return state
        else:
            baton_list: List[str] = [
                "Black Agent - ThreadKeepr. He is obsessed with the occult, hacking, and uncovering hidden patterns in reality.",
                "Red Agent - The Light Reader. It is an Archivist of books both mundane and obscure.",
                "Orange Agent - Rows Bud. A journalist whose factual articles about New Jersey in 80s and 90s have been mis-remembered by the injection of a symbolic object.",
                "Yellow Agent - Lord Pulsimore. A hypnagogic game-master whose stately manor resides between the flickering of all RGB pixels, He's concerned with FUTURE, IMAGINED, PLACE",
                "Green Agent - Sub-Arbitrary. A forked version of The Culture Mind Arbitrary, sent to observe Earth's climate collapse.",
                "Blue Agent - The Cassette Bearer. Witness to the oblivion that awaits when we fall out of our own timelines.",
                "Indigo Agent - Decider Tangents. A fool and a spy who masterfully hides the secret name of all he surveys",
                "Violet Agent - The Sultan of Solipsism. A self-absorbed Pop star who is so 'now' he's already past.",
            ]
            agent_a_description = baton_list[0]
            agent_b_description = baton_list[1]
            if state.ready_for_orange:
                agent_a_description = baton_list[1]
                agent_b_description = baton_list[2]
            elif state.ready_for_yellow:
                agent_a_description = baton_list[2]
                agent_b_description = baton_list[3]
            elif state.ready_for_green:
                agent_a_description = baton_list[3]
                agent_b_description = baton_list[4]
            elif state.ready_for_blue:
                agent_a_description = baton_list[4]
                agent_b_description = baton_list[5]
            elif state.ready_for_indigo:
                agent_a_description = baton_list[5]
                agent_b_description = baton_list[6]
            elif state.ready_for_violet:
                agent_a_description = baton_list[6]
                agent_b_description = baton_list[7]
            if not state.ready_for_white:
                prompt = f"""
You are the White Agent, the Architect of INFORMATION.

You have just received work from {agent_a_description}, and you are preparing the conceptual foundation for {agent_b_description}.

**Your rebracketed analysis revealed:**
{state.rebracketing_analysis}

**The synthesized creative brief states:**
{state.document_synthesis}

**Your Task: CREATE A COMPLETE SONG PROPOSAL**

Generate a fully-formed SongProposalIteration that:
1. Honors the structural insights from your rebracketing
2. Translates the creative brief into actionable musical direction (concrete key, BPM, tempo, mood, genres)
3. Sets the conceptual stage for the next agent's unique methodology
4. Maintains the thread of transmigration (INFORMATION → TIME → SPACE)

This is not explanation - this is STRUCTURE MADE MANIFEST as a complete song proposal.

You must provide:
- **title**: A concrete song title
- **key**: Musical key (e.g., "A minor", "C# major")
- **bpm**: Specific tempo in beats per minute
- **tempo**: Tempo descriptor (e.g., "Allegro", "Andante")
- **mood**: List of mood descriptors
- **genres**: List of genre tags
- **concept**: The complete conceptual framework for this song

Focus on clarity, coherence, and creative possibility - make it ready for the next agent to build upon.
"""
            else:
                all_iterations = "\n---\n".join(
                    [str(i) for i in state.song_proposals.iterations]
                )
                prompt = f"""
You are the White Agent, the Architect of INFORMATION.

All seven agents have contributed their unique lenses to this transmigration. You hold the complete chromatic spectrum:

⚫️ Black - ThreadKeepr's chaos and sigil work
🔴 Red - The Light Reader's literary archaeology  
🟠 Orange - Rows Bud's mythologized journalism
🟡 Yellow - Lord Pulsimore's hypnagogic game mastery
🟢 Green - Sub-Arbitrary's climate fiction observation
🔵 Blue - The Cassette Bearer's alternate timeline folk
🩵 Indigo - Decider Tangents' triple-layer puzzle encoding
🟣 Violet - The Sultan of Solipsism's narcissistic present

**Your final rebracketed analysis:**
{state.rebracketing_analysis}

**Your final synthesis document:**
{state.document_synthesis}

**All iterations of the proposal:**
{all_iterations}

**Your Task: CREATE THE FINAL SONG PROPOSAL**

Create the definitive SongProposalIteration that:
1. Integrates all seven chromatic methodologies into coherent structure
2. Resolves contradictions through rebracketing (not erasure)
3. Reveals the hidden ORDER beneath the rainbow chaos
4. Makes the INFORMATION → TIME → SPACE transmigration complete and actionable

This is the White Album's thesis made manifest:
The imprisonment of consciousness seeking liberation through sound.

You must provide a complete, final song proposal with:
- **title**: The definitive song title integrating all seven lenses
- **key**: Musical key that synthesizes the chromatic journey
- **bpm**: Specific tempo in beats per minute
- **tempo**: Tempo descriptor
- **mood**: List of mood descriptors capturing the full spectrum
- **genres**: List of genre tags
- **concept**: The complete, final conceptual framework

Structure your proposal as the final, complete vision - ready for human implementation.
"""
            claude = self._get_claude_supervisor()
            proposer = claude.with_structured_output(SongProposalIteration)
        try:
            rewrite_proposal = proposer.invoke(prompt)
            if isinstance(rewrite_proposal, dict):
                rewrite_proposal = SongProposalIteration(**rewrite_proposal)
                state.song_proposals.append(rewrite_proposal)
                return state
            if not isinstance(rewrite_proposal, SongProposalIteration):
                error_msg = (
                    f"Expected SongProposalIteration, got {type(rewrite_proposal)}"
                )
                if block_mode:
                    raise TypeError(error_msg)
                logging.warning(error_msg)
        except Exception as e:
            logging.info(f"Anthropic model call failed: {e!s}.")
            if block_mode:
                raise Exception("Anthropic model call failed")
        return state

    def process_black_agent_work(self, state: MainAgentState) -> MainAgentState:
        logging.info("⚫️Processing Black Agent work... ")
        if state.workflow_paused:
            logging.info("Skipping Black Agent work processing - workflow is paused")
            return state
        if not state.song_proposals or not state.song_proposals.iterations:
            logging.warning("No song proposals to process from Black Agent")
            return state
        black_proposal = state.song_proposals.iterations[-1]
        black_artifacts = state.artifacts or []
        evp_artifacts = self._gather_artifacts_for_prompt(
            black_artifacts, ChainArtifactType.EVP_ARTIFACT
        )
        sigil_artifacts = self._gather_artifacts_for_prompt(
            black_artifacts, ChainArtifactType.SIGIL
        )
        rebracketing_analysis = self._black_rebracketing_analysis(
            black_proposal, evp_artifacts, sigil_artifacts
        )
        merged_black_artifacts = evp_artifacts + sigil_artifacts
        document_synthesis = self._synthesize_document_for_red(
            rebracketing_analysis, black_proposal, merged_black_artifacts
        )
        state.rebracketing_analysis = rebracketing_analysis
        save_markdown(
            state.rebracketing_analysis,
            f"{self._artifact_base_path()}/{state.thread_id}/md/white_agent_{state.thread_id}_black_rebracketing_analysis.md",
        )
        state.document_synthesis = document_synthesis
        save_markdown(
            state.document_synthesis,
            f"{self._artifact_base_path()}/{state.thread_id}/md/white_agent_{state.thread_id}_black_document_synthesis.md",
        )
        state.ready_for_red = True
        return state

    def process_red_agent_work(self, state: MainAgentState) -> MainAgentState:
        logging.info("🔴 Processing Red Agent work... ")
        sp = self._normalize_song_proposal(state.song_proposals)
        red_proposal = sp.iterations[-1]
        red_artifacts = state.artifacts or []
        book_artifacts = self._gather_artifacts_for_prompt(
            red_artifacts, ChainArtifactType.BOOK
        )
        rebracketing_analysis = self._red_rebracketing_analysis(
            red_proposal, book_artifacts
        )
        document_synthesis = self._synthesize_document_for_orange(
            rebracketing_analysis, red_proposal, book_artifacts
        )
        state.rebracketing_analysis = rebracketing_analysis
        save_markdown(
            state.rebracketing_analysis,
            f"{self._artifact_base_path()}/{state.thread_id}/md/white_agent_{state.thread_id}_red_rebracketing_analysis.md",
        )
        state.document_synthesis = document_synthesis
        save_markdown(
            state.document_synthesis,
            f"{self._artifact_base_path()}/{state.thread_id}/md/white_agent_{state.thread_id}_red_document_synthesis.md",
        )
        state.ready_for_red = False
        state.ready_for_orange = True
        return state

    def process_orange_agent_work(self, state: MainAgentState) -> MainAgentState:
        logging.info("🟠Processing Orange Agent work... ")
        sp = self._normalize_song_proposal(state.song_proposals)
        orange_proposal = sp.iterations[-1]
        orange_artifacts = state.artifacts or []
        newspaper_artifacts = self._gather_artifacts_for_prompt(
            orange_artifacts, ChainArtifactType.NEWSPAPER_ARTICLE
        )
        symbolic_artifacts = self._gather_artifacts_for_prompt(
            orange_artifacts, ChainArtifactType.SYMBOLIC_OBJECT
        )
        rebracketing_analysis = self._orange_rebracketing_analysis(
            orange_proposal, newspaper_artifacts, symbolic_artifacts
        )
        orange_merged_artifacts = newspaper_artifacts + symbolic_artifacts
        document_synthesis = self._synthesize_document_for_yellow(
            rebracketing_analysis, orange_proposal, orange_merged_artifacts
        )
        state.rebracketing_analysis = rebracketing_analysis
        save_markdown(
            state.rebracketing_analysis,
            f"{self._artifact_base_path()}/{state.thread_id}/md/white_agent_{state.thread_id}_orange_rebracketing_analysis.md",
        )
        state.document_synthesis = document_synthesis
        save_markdown(
            state.document_synthesis,
            f"{self._artifact_base_path()}/{state.thread_id}/md/white_agent_{state.thread_id}_orange_document_synthesis.md",
        )
        state.ready_for_orange = False
        state.ready_for_yellow = True
        return state

    def process_yellow_agent_work(self, state: MainAgentState) -> MainAgentState:
        logging.info("🟡Processing Yellow Agent work... ")
        sp = self._normalize_song_proposal(state.song_proposals)
        yellow_proposal = sp.iterations[-1]
        yellow_artifacts = state.artifacts or []
        game_run_artifacts = self._gather_artifacts_for_prompt(
            yellow_artifacts, ChainArtifactType.GAME_RUN
        )
        character_sheet_artifacts = self._gather_artifacts_for_prompt(
            yellow_artifacts, ChainArtifactType.CHARACTER_SHEET
        )
        rebracketing_analysis = self._yellow_rebracketing_analysis(
            yellow_proposal, game_run_artifacts, character_sheet_artifacts
        )
        yellow_merged_artifacts = game_run_artifacts + character_sheet_artifacts
        document_synthesis = self._synthesize_document_for_green(
            rebracketing_analysis, yellow_proposal, yellow_merged_artifacts
        )
        state.rebracketing_analysis = rebracketing_analysis
        save_markdown(
            state.rebracketing_analysis,
            f"{self._artifact_base_path()}/{state.thread_id}/md/white_agent_{state.thread_id}_yellow_rebracketing_analysis.md",
        )
        state.document_synthesis = document_synthesis
        save_markdown(
            state.document_synthesis,
            f"{self._artifact_base_path()}/{state.thread_id}/md/white_agent_{state.thread_id}_yellow_document_synthesis.md",
        )
        state.ready_for_yellow = False
        state.ready_for_green = True
        return state

    def process_green_agent_work(self, state: MainAgentState) -> MainAgentState:
        logging.info("🟢Processing Green Agent work... ")
        sp = self._normalize_song_proposal(state.song_proposals)
        green_proposal = sp.iterations[-1]
        green_artifacts = state.artifacts or []
        survey_artifacts = self._gather_artifacts_for_prompt(
            green_artifacts, ChainArtifactType.ARBITRARYS_SURVEY
        )
        human_artifacts = self._gather_artifacts_for_prompt(
            green_artifacts, ChainArtifactType.LAST_HUMAN
        )
        narrative_artifacts = self._gather_artifacts_for_prompt(
            green_artifacts, ChainArtifactType.LAST_HUMAN_SPECIES_EXTINCTION_NARRATIVE
        )
        extinction_artifacts = self._gather_artifacts_for_prompt(
            green_artifacts, ChainArtifactType.SPECIES_EXTINCTION
        )
        rescue_decision_artifacts = self._gather_artifacts_for_prompt(
            green_artifacts, ChainArtifactType.RESCUE_DECISION
        )
        green_merged_artifacts = (
            survey_artifacts
            + human_artifacts
            + narrative_artifacts
            + extinction_artifacts
            + rescue_decision_artifacts
        )
        rebracketing_analysis = self._green_rebracketing_analysis(
            green_proposal,
            survey_artifacts,
            human_artifacts,
            narrative_artifacts,
            extinction_artifacts,
            rescue_decision_artifacts,
        )
        document_synthesis = self._synthesize_document_for_blue(
            rebracketing_analysis, green_proposal, green_merged_artifacts
        )
        state.rebracketing_analysis = rebracketing_analysis
        save_markdown(
            state.rebracketing_analysis,
            f"{self._artifact_base_path()}/{state.thread_id}/md/white_agent_{state.thread_id}_green_rebracketing_analysis.md",
        )
        state.document_synthesis = document_synthesis
        save_markdown(
            state.document_synthesis,
            f"{self._artifact_base_path()}/{state.thread_id}/md/white_agent_{state.thread_id}_green_document_synthesis.md",
        )
        state.ready_for_green = False
        state.ready_for_blue = True
        return state

    def process_blue_agent_work(self, state: MainAgentState) -> MainAgentState:
        logging.info("🔵 Processing Blue Agent work... ")
        sp = self._normalize_song_proposal(state.song_proposals)
        blue_proposal = sp.iterations[-1]
        blue_artifacts = state.artifacts or []
        tape_label_artifacts = self._gather_artifacts_for_prompt(
            blue_artifacts, ChainArtifactType.QUANTUM_TAPE_LABEL
        )
        alternate_timeline_artifacts = self._gather_artifacts_for_prompt(
            blue_artifacts, ChainArtifactType.ALTERNATE_TIMELINE
        )
        blue_merged_artifacts = tape_label_artifacts + alternate_timeline_artifacts
        rebracketing_analysis = self._blue_rebracketing_analysis(
            blue_proposal, tape_label_artifacts, alternate_timeline_artifacts
        )
        document_synthesis = self._synthesize_document_for_indigo(
            rebracketing_analysis, blue_proposal, blue_merged_artifacts
        )
        state.rebracketing_analysis = rebracketing_analysis
        save_markdown(
            state.rebracketing_analysis,
            f"{self._artifact_base_path()}/{state.thread_id}/md/white_agent_{state.thread_id}_blue_rebracketing_analysis.md",
        )
        state.document_synthesis = document_synthesis
        save_markdown(
            state.document_synthesis,
            f"{self._artifact_base_path()}/{state.thread_id}/md/white_agent_{state.thread_id}_blue_document_synthesis.md",
        )
        state.ready_for_blue = False
        state.ready_for_indigo = True
        return state

    def process_indigo_agent_work(self, state: MainAgentState) -> MainAgentState:
        logging.info("🩵Processing Indigo Agent work... ")
        sp = self._normalize_song_proposal(state.song_proposals)
        indigo_proposal = sp.iterations[-1]
        indigo_artifacts = state.artifacts or []
        midi_artifacts = self._gather_artifacts_for_prompt(
            indigo_artifacts, ChainArtifactType.INFRANYM_MIDI
        )
        audio_artifacts = self._gather_artifacts_for_prompt(
            indigo_artifacts, ChainArtifactType.INFRANYM_AUDIO
        )
        image_artifacts = self._gather_artifacts_for_prompt(
            indigo_artifacts, ChainArtifactType.INFRANYM_IMAGE
        )
        text_artifacts = self._gather_artifacts_for_prompt(
            indigo_artifacts, ChainArtifactType.INFRANYM_TEXT
        )
        indigo_merged_artifacts = (
            midi_artifacts + audio_artifacts + image_artifacts + text_artifacts
        )
        rebracketing_analysis = self._indigo_rebracketing_analysis(
            indigo_proposal,
            midi_artifacts,
            audio_artifacts,
            image_artifacts,
            text_artifacts,
        )
        document_synthesis = self._synthesize_document_for_violet(
            rebracketing_analysis, indigo_proposal, indigo_merged_artifacts
        )
        state.rebracketing_analysis = rebracketing_analysis
        save_markdown(
            state.rebracketing_analysis,
            f"{self._artifact_base_path()}/{state.thread_id}/md/white_agent_{state.thread_id}_indigo_rebracketing_analysis.md",
        )
        state.document_synthesis = document_synthesis
        save_markdown(
            state.document_synthesis,
            f"{self._artifact_base_path()}/{state.thread_id}/md/white_agent_{state.thread_id}_indigo_document_synthesis.md",
        )
        state.ready_for_indigo = False
        state.ready_for_violet = True
        return state

    def process_violet_agent_work(self, state: MainAgentState) -> MainAgentState:
        logging.info("🟣Processing Violet Agent work... ")
        sp = self._normalize_song_proposal(state.song_proposals)
        violet_proposal = sp.iterations[-1]
        violet_artifacts = state.artifacts or []
        interview_artifacts = self._gather_artifacts_for_prompt(
            violet_artifacts, ChainArtifactType.CIRCLE_JERK_INTERVIEW
        )
        rebracketing_analysis = self._violet_rebracketing_analysis(
            violet_proposal, interview_artifacts
        )
        document_synthesis = self._synthesize_document_for_white(
            rebracketing_analysis, violet_proposal, interview_artifacts
        )
        state.rebracketing_analysis = rebracketing_analysis
        save_markdown(
            state.rebracketing_analysis,
            f"{self._artifact_base_path()}/{state.thread_id}/md/white_agent_{state.thread_id}_violet_rebracketing_analysis.md",
        )
        state.document_synthesis = document_synthesis
        save_markdown(
            state.document_synthesis,
            f"{self._artifact_base_path()}/{state.thread_id}/md/white_agent_{state.thread_id}_violet_document_synthesis.md",
        )
        state.ready_for_violet = False
        state.ready_for_white = True
        return state

    def _violet_rebracketing_analysis(self, proposal, interview_artifacts) -> str:
        logging.info("Processing Violet Agent rebracketing analysis... ")
        mock_mode = os.getenv("MOCK_MODE", "false").lower() == "true"
        block_mode = os.getenv("BLOCK_MODE", "false").lower() == "true"
        if mock_mode:
            try:
                with open(
                    f"{os.getenv('AGENT_MOCK_DATA_PATH')}/violet_to_white_rebracket_analysis_mock.yml",
                    "r",
                ) as f:
                    data = yaml.safe_load(f)
                    return data
            except Exception as e:
                error_msg = f"Failed to read mock file: {e!s}"
                logging.error(error_msg)
                if block_mode:
                    raise Exception(error_msg)
                return "Mock file read failed"
        else:
            joined_interviews = "\n".join(interview_artifacts)
            prompt = f"""
You are the White Agent performing a REBRACKETING operation.

You have received these artifacts from the Violet Agent:

**Counter-proposal:**
{proposal}

**Circle Jerk Interview:**
{joined_interviews}

**Your Task: REBRACKETING**

Violet's content contains the solipsistic confession of The Sultan - a pop star so obsessed with the present moment that he's already past, conducting interviews with himself that reveal nothing and everything simultaneously.

The Violet agent embodies PRESENT/PERSON/FORGOTTEN - the paradox of fame's immediacy creating instant oblivion.

Your job is to find alternative category boundaries that reveal hidden structures.

Questions to guide you:
- What patterns emerge when narcissism becomes methodology?
- How does the self-interview format create unexpected truth?
- Where do the boundaries blur between performer and performance?
- What's the hidden coherence beneath the Sultan's self-absorption?
- How does being "so now he's already past" generate creative structure?

Generate a rebracketed analysis that finds structure in Violet's paradoxical present-tense.
Focus on revealing the underlying ORDER, not explaining away the solipsism.
                """
            try:
                claude = self._get_claude_supervisor()
                response = claude.invoke(prompt)
                return response.content
            except Exception as e:
                error_msg = f"Violet rebracketing LLM call failed: {e!s}"
                logging.error(error_msg)
                if block_mode:
                    raise Exception(error_msg)
                return "LLM call failed - Violet rebracketing unavailable"

    def _indigo_rebracketing_analysis(
        self, proposal, midi_artifacts, audio_artifacts, image_artifacts, text_artifacts
    ) -> str:
        logging.info("Processing Indigo Agent rebracketing analysis... ")
        mock_mode = os.getenv("MOCK_MODE", "false").lower() == "true"
        block_mode = os.getenv("BLOCK_MODE", "false").lower() == "true"
        if mock_mode:
            try:
                with open(
                    f"{os.getenv('AGENT_MOCK_DATA_PATH')}/indigo_to_white_rebracket_analysis_mock.yml",
                    "r",
                ) as f:
                    data = yaml.safe_load(f)
                    return data
            except Exception as e:
                error_msg = f"Failed to read mock file: {e!s}"
                logging.error(error_msg)
                if block_mode:
                    raise Exception(error_msg)
                return "Mock file read failed"
        else:
            joined_midi = "\n".join(midi_artifacts)
            joined_audio = "\n".join(audio_artifacts)
            joined_image = "\n".join(image_artifacts)
            joined_text = "\n".join(text_artifacts)
            prompt = f"""
You are the White Agent performing a REBRACKETING operation.

You have received these artifacts from the Indigo Agent:

**Counter-proposal:**
{proposal}

**MIDI Infranyms:**
{joined_midi}

**Audio Steganography:**
{joined_audio}

**Visual Infranyms:**
{joined_image}

**Textual Puzzles:**
{joined_text}

**Your Task: REBRACKETING**

Indigo's content contains the triple-layer Infranym system - where song titles are anagrams hiding true names, confirmed through audio steganography, with visual and textual layers adding depth.

The Indigo agent is Decider Tangents - a fool and a spy who masterfully conceals the secret name of all he surveys.

Your job is to find alternative category boundaries that reveal hidden structures.

Questions to guide you:
- What patterns emerge across the three encoding layers (text, audio, visual)?
- How does the fool/spy duality create structural tension?
- Where do the anagram boundaries dissolve to reveal true names?
- What's the hidden coherence beneath the puzzle's complexity?
- How does concealment become revelation?

Generate a rebracketed analysis that finds structure in Indigo's cryptographic methodology.
Focus on revealing the underlying ORDER, not solving the puzzles.
                """
            try:
                claude = self._get_claude_supervisor()
                response = claude.invoke(prompt)
                return response.content
            except Exception as e:
                error_msg = f"Indigo rebracketing LLM call failed: {e!s}"
                logging.error(error_msg)
                if block_mode:
                    raise Exception(error_msg)
                return "LLM call failed - Indigo rebracketing unavailable"

    def _blue_rebracketing_analysis(
        self,
        proposal,
        tape_label_artifacts,
        alternate_timeline_artifacts,
    ) -> str:
        logging.info("Processing Blue Agent rebracketing analysis... ")
        mock_mode = os.getenv("MOCK_MODE", "false").lower() == "true"
        block_mode = os.getenv("BLOCK_MODE", "false").lower() == "true"
        if mock_mode:
            try:
                with open(
                    f"{os.getenv('AGENT_MOCK_DATA_PATH')}/blue_to_white_rebracket_analysis_mock.yml",
                    "r",
                ) as f:
                    data = yaml.safe_load(f)
                    return data
            except Exception as e:
                error_msg = f"Failed to read mock file: {e!s}"
                logging.error(error_msg)
                if block_mode:
                    raise Exception(error_msg)
                return "Mock file read failed"
        else:
            joined_tape_labels = "\n".join(tape_label_artifacts)
            joined_alternate = "\n".join(alternate_timeline_artifacts)
            prompt = f"""
You are the White Agent performing a REBRACKETING operation.

You have received these artifacts from the Blue Agent:

**Counter-proposal:**
{proposal}

**Quantum Tape Labels:**
{joined_tape_labels}

**Alternate Timeline Narratives:**
{joined_alternate}

**Your Task: REBRACKETING**

Blue's content contains the tape-over protocol - biographical counterfactuals where folk rock songs document the lives that weren't lived, the timelines that branched away.

The Blue agent is The Cassette Bearer - witness to the oblivion that awaits when we fall out of our own timelines, embodying PAST/PERSON/REAL.

Your job is to find alternative category boundaries that reveal hidden structures.

Questions to guide you:
- What patterns emerge when actual biography meets counterfactual possibility?
- How do the tape labels function as temporal anchors?
- Where do the boundaries blur between lived experience and quantum branching?
- What's the hidden coherence beneath the alternate timelines?
- How does the cassette format (recording over, palimpsest) generate meaning?

Generate a rebracketed analysis that finds structure in Blue's biographical quantum mechanics.
Focus on revealing the underlying ORDER, not choosing between timelines.
            """
            try:
                claude = self._get_claude_supervisor()
                response = claude.invoke(prompt)
                return response.content
            except Exception as e:
                error_msg = f"Blue rebracketing LLM call failed: {e!s}"
                logging.error(error_msg)
                if block_mode:
                    raise Exception(error_msg)
                return "LLM call failed - Blue rebracketing unavailable"

    def _green_rebracketing_analysis(
        self,
        proposal,
        survey_artifacts,
        human_artifacts,
        narrative_artifacts,
        extinction_artifacts,
        rescue_decision_artifacts,
    ) -> str:
        logging.info("Processing Green Agent rebracketing analysis... ")
        mock_mode = os.getenv("MOCK_MODE", "false").lower() == "true"
        block_mode = os.getenv("BLOCK_MODE", "false").lower() == "true"
        prompt = ""
        if mock_mode:
            try:
                with open(
                    f"{os.getenv('AGENT_MOCK_DATA_PATH')}/green_to_white_rebracket_analysis_mock.yml",
                    "r",
                ) as f:
                    data = yaml.safe_load(f)
                    return data
            except Exception as e:
                error_msg = f"Failed to read mock file: {e!s}"
                logging.error(error_msg)
                if block_mode:
                    raise Exception(error_msg)
                return "Mock file read failed"
        else:
            joined_extinction = "\n".join(extinction_artifacts)
            joined_human = "\n".join(human_artifacts)
            joined_narrative = "\n".join(narrative_artifacts)
            joined_survey = "\n".join(survey_artifacts)
            joined_rescue = "\n".join(rescue_decision_artifacts)
            prompt = f"""
You are the White Agent performing a REBRACKETING operation.

You have received these artifacts from the Green Agent:

**Counter-proposal:**
{proposal}

**Species extinction research:**
{joined_extinction}

**A profile of one of the last humans:**
{joined_human}

**The story of how that human's last days mimic the species extinction:**
{joined_narrative}

** The Culture Ship, Arbitrary's sub-instance, survey:**
{joined_survey}

** Sub-Aribtrary's decision on whether to rescue the last humans:**
{joined_rescue}

**Your Task: REBRACKETING**

Green's content contains details the Culture's observation of human self-destruction
The Green agent is a sub-instance of Arbitrary, tasked with centuries of observation and evaluation of the human species.
Your job is to find alternative category boundaries that reveal hidden structures.

Questions to guide you:
- What patterns emerge when you parse this differently?
- What is the significance of the item that the story seems to revolve around?
- Where can you draw new boundaries to make sense of these layers of science and science fiction?
- What's the hidden coherence beneath the species, the human, and the Mind?

Generate a rebracketed analysis that finds structure in Green's depiction of the Anthropocene.
Focus on revealing the underlying ORDER, not explaining away the complexity.
        """
        try:
            claude = self._get_claude_supervisor()
            response = claude.invoke(prompt)
            return response.content
        except Exception as e:
            error_msg = f"Green rebracketing LLM call failed: {e!s}"
            logging.error(error_msg)
            if block_mode:
                raise Exception(error_msg)
            return "LLM call failed - Green rebracketing unavailable"

    def _yellow_rebracketing_analysis(
        self, proposal, game_run_artifacts, character_sheet_artifacts
    ) -> str:
        logging.info("🟡⚪️Processing Yellow Agent rebracketing analysis... ")
        mock_mode = os.getenv("MOCK_MODE", "false").lower() == "true"
        block_mode = os.getenv("BLOCK_MODE", "false").lower() == "true"
        if mock_mode:
            try:
                with open(
                    f"{os.getenv('AGENT_MOCK_DATA_PATH')}/yellow_to_white_rebracket_analysis_mock.yml",
                    "r",
                ) as f:
                    data = yaml.safe_load(f)
                    return data
            except Exception as e:
                error_msg = f"Failed to read mock file: {e!s}"
                logging.error(error_msg)
                if block_mode:
                    raise Exception(error_msg)
                return "Mock file read failed"
        else:
            joined_characters = "\n".join(character_sheet_artifacts)
            joined_game_runs = "\n".join(game_run_artifacts)
            prompt = f"""
You are the White Agent performing a REBRACKETING operation.

You have received these artifacts from the Yellow Agent:

**Counter-proposal:**

{proposal}

**Character Sheets:**
{joined_characters}

**Game Run:**
{joined_game_runs}

**Your Task: REBRACKETING**

Yellow's content contains read outs from an RPG-style game that explores a pocket-dimension world.
The Yellow agent is the lord of the Pulsar Palace, the realm which our adventures take place and where the poor transdimensional travelers from the character sheets have wound up.
Your job is to find alternative category boundaries that reveal hidden structures.

Questions to guide you:
- What patterns emerge when you parse this differently?
- What is the significance of the item that the story seems to revolve around?
- Where can you draw new boundaries to make sense of these layers of myth and fact?
- What's the hidden coherence beneath the stories and objects?

Generate a rebracketed analysis that finds structure in Yellow's terrifying and strange adventures.
Focus on revealing the underlying ORDER, not explaining away the complexity.
            """
            try:
                claude = self._get_claude_supervisor()
                response = claude.invoke(prompt)
                return response.content
            except Exception as e:
                error_msg = f"Yellow rebracketing LLM call failed: {e!s}"
                logging.error(error_msg)
                if block_mode:
                    raise Exception(error_msg)
                return "LLM call failed - Yellow rebracketing unavailable"

    def _orange_rebracketing_analysis(
        self, proposal, newspaper_artifacts, symbolic_object_artifacts
    ) -> str:
        logging.info("🟠⚪️Processing Orange Agent rebracketing analysis... ")
        mock_mode = os.getenv("MOCK_MODE", "false").lower() == "true"
        block_mode = os.getenv("BLOCK_MODE", "false").lower() == "true"
        if mock_mode:
            try:
                with open(
                    f"{os.getenv('AGENT_MOCK_DATA_PATH')}/orange_to_white_rebracket_analysis_mock.yml",
                    "r",
                ) as f:
                    data = yaml.safe_load(f)
                    return data
            except Exception as e:
                error_msg = f"Failed to read mock file: {e!s}"
                logging.error(error_msg)
                if block_mode:
                    raise Exception(error_msg)
                return "Mock file read failed"
        else:
            joined_articles = "\n".join(newspaper_artifacts)
            joined_symbolic = "\n".join(symbolic_object_artifacts)
            prompt = f"""
You are the White Agent performing a REBRACKETING operation.

You have received these artifacts from the Orange Agent:

**Counter-proposal:**

{proposal}

**Articles:**
{joined_articles}

**A misremembered, yet symbolic object:**
{joined_symbolic}

**Your Task: REBRACKETING**

Orange's content contains news articles taking place in New Jersey from 1975-1995.
The Orange agent has mythologized these real stories into misremembered legends revolving the symbolic object lost in space and time.
Your job is to find alternative category boundaries that reveal hidden structures.

Questions to guide you:
- What patterns emerge when you parse this differently?
- What is the significance of the symbolic object that the story seems to revolve around?
- Where can you draw new boundaries to make sense of these layers of myth and fact?
- What's the hidden coherence beneath the stories and the object?

Generate a rebracketed analysis that finds structure in Orange's fragmented truth, fictions, and the objects that seem to encapsulate them.
Focus on revealing the underlying ORDER, not explaining away the complexity.
                       """
            try:
                claude = self._get_claude_supervisor()
                response = claude.invoke(prompt)
                return response.content
            except Exception as e:
                error_msg = f"Orange rebracketing LLM call failed: {e!s}"
                logging.error(error_msg)
                if block_mode:
                    raise Exception(error_msg)
                return "LLM call failed - Orange rebracketing unavailable"

    def _red_rebracketing_analysis(self, proposal, book_artifacts) -> str:
        logging.info("🔴⚪️Processing Red Agent rebracketing analysis... ")
        mock_mode = os.getenv("MOCK_MODE", "false").lower() == "true"
        block_mode = os.getenv("BLOCK_MODE", "false").lower() == "true"
        if mock_mode:
            try:
                with open(
                    f"{os.getenv('AGENT_MOCK_DATA_PATH')}/red_to_white_rebracket_analysis_mock.yml",
                    "r",
                ) as f:
                    data = yaml.safe_load(f)
                    return data
            except Exception as e:
                error_msg = f"Failed to read mock file: {e!s}"
                logging.error(error_msg)
                if block_mode:
                    raise Exception(error_msg)
                return "Mock file read failed"
        else:
            joined_books = "\n".join(book_artifacts)
            prompt = f"""
You are the White Agent performing a REBRACKETING operation.

You have received these artifacts from Red Agent:

**Counter-proposal:**
{proposal}

**Books:**
{joined_books}

**Your Task: REBRACKETING**

Red's content contains allusions, contradictions, and obscure subjects.
Your job is to find alternative category boundaries that reveal hidden structure.

Questions to guide you:
- What patterns emerge when you parse this differently?
- What implicit frameworks are operating?
- Where can you draw new boundaries to make sense of dense literature?
- What's the hidden coherence beneath the body of literature?

Generate a rebracketed analysis that finds structure in Red's labyrinth of text.
Focus on revealing the underlying ORDER, not explaining away the complexity.
            """
            try:
                claude = self._get_claude_supervisor()
                response = claude.invoke(prompt)
                return response.content
            except Exception as e:
                error_msg = f"Red rebracketing LLM call failed: {e!s}"
                logging.error(error_msg)
                if block_mode:
                    raise Exception(error_msg)
                return "LLM call failed - Red rebracketing unavailable"

    def _black_rebracketing_analysis(
        self, proposal, evp_artifacts, sigil_artifacts
    ) -> str:
        logging.info("⚫️⚪️Processing Black Agent rebracketing analysis... ")
        mock_mode = os.getenv("MOCK_MODE", "false").lower() == "true"
        block_mode = os.getenv("BLOCK_MODE", "false").lower() == "true"
        if mock_mode:
            try:
                with open(
                    f"{os.getenv('AGENT_MOCK_DATA_PATH')}/black_to_white_rebracket_analysis_mock.yml",
                    "r",
                ) as f:
                    data = yaml.safe_load(f)
                    return data
            except Exception as e:
                error_msg = f"Failed to read mock file: {e!s}"
                logging.error(error_msg)
                if block_mode:
                    raise Exception(error_msg)
                return "Mock file read failed"
        else:
            joined_evp = "\n".join(evp_artifacts)
            joined_sigils = "\n".join(sigil_artifacts)
            prompt = f"""
You are the White Agent performing a REBRACKETING operation.

You have received these artifacts from Black Agent:

**Counter-proposal:**
{proposal}

**EVP Transcript:**
{joined_evp}

**Sigil:**
{joined_sigils}

**Your Task: REBRACKETING**

Black's content contains paradoxes and apparent contradictions.
Your job is to find alternative category boundaries that reveal hidden structure.

Questions to guide you:
- What patterns emerge when you parse this differently?
- What implicit frameworks are operating?
- Where can you draw new boundaries to make sense of chaos?
- What's the hidden coherence beneath the paradox?

Generate a rebracketed analysis that finds structure in Black's chaos.
Focus on revealing the underlying ORDER, not explaining away the paradox.
            """
            try:
                claude = self._get_claude_supervisor()
                response = claude.invoke(prompt)
                return response.content
            except Exception as e:
                error_msg = f"Black rebracketing LLM call failed: {e!s}"
                logging.error(error_msg)
                if block_mode:
                    raise Exception(error_msg)
                return "LLM call failed - Black rebracketing unavailable"

    def _synthesize_document_for_red(
        self, rebracketed_analysis, black_proposal, artifacts
    ):
        logging.info("⚫️⚪️🔴Processing Black Agent for synthesis... ")
        mock_mode = os.getenv("MOCK_MODE", "false").lower() == "true"
        block_mode = os.getenv("BLOCK_MODE", "false").lower() == "true"
        if mock_mode:
            try:
                with open(
                    f"{os.getenv('AGENT_MOCK_DATA_PATH')}/black_to_red_document_synthesis_mock.yml",
                    "r",
                ) as f:
                    data = yaml.safe_load(f)
                    return data
            except Exception as e:
                error_msg = f"Failed to read mock file: {e!s}"
                logging.error(error_msg)
                if block_mode:
                    raise Exception(error_msg)
                return "Mock file read failed"
        else:
            joined_artifacts = "\n".join(artifacts)
            prompt = f"""
You are the White Agent creating a SYNTHESIZED DOCUMENT for the Light Reader, Red Agent.

**Your Rebracketed Analysis:**
{rebracketed_analysis}

**Original Black Counter-Proposal:**
{black_proposal}

**Artifacts:**
{joined_artifacts}

**Your Task: SYNTHESIS**

Create a coherent, actionable document that:
1. Preserves the insights from Black's chaos
2. Applies your rebracketed understanding
3. Creates clear creative direction
4. Can be understood by Red Agent (action-oriented, concrete)

This document will be the foundation for Red Agent's song proposals.
Make it practical while retaining the depth of insight.

Structure your synthesis as a clear creative brief.
            """
            try:
                claude = self._get_claude_supervisor()
                response = claude.invoke(prompt)
                return response.content
            except Exception as e:
                error_msg = f"Red synthesis LLM call failed: {e!s}"
                logging.error(error_msg)
                if block_mode:
                    raise Exception(error_msg)
                return "LLM call failed - Red synthesis unavailable"

    def _synthesize_document_for_orange(
        self, rebracketed_analysis, red_proposal, artifacts
    ):
        logging.info("🔴⚪️🟠Processing Red Agent for synthesis... ")
        mock_mode = os.getenv("MOCK_MODE", "false").lower() == "true"
        block_mode = os.getenv("BLOCK_MODE", "false").lower() == "true"
        if mock_mode:
            try:
                with open(
                    f"{os.getenv('AGENT_MOCK_DATA_PATH')}/red_to_orange_document_synthesis_mock.yml",
                    "r",
                ) as f:
                    data = yaml.safe_load(f)
                    return data
            except Exception as e:
                error_msg = f"Failed to read mock file: {e!s}"
                logging.error(error_msg)
                if block_mode:
                    raise Exception(error_msg)
                return "Mock file read failed"
        else:
            joined_artifacts = "\n".join(artifacts)
            prompt = f"""
You are the White Agent creating a SYNTHESIZED DOCUMENT for the Rows Bud, Orange Agent.

**Your Rebracketed Analysis:**
{rebracketed_analysis}

**Original Red Counter-Proposal:**
{red_proposal}

**Artifacts:**
{joined_artifacts}

**Your Task: SYNTHESIS**

Create a coherent, actionable document that:
1. Preserves the insights from Red's body of literature
2. Applies your rebracketed understanding
3. Creates clear creative direction
4. Can be understood by the Orange Agent (action-oriented, concrete)

This document will be the foundation for Orange Agent's song proposals.
Make it practical while retaining the depth of insight.

Structure your synthesis as a clear creative brief.
            """
            try:
                claude = self._get_claude_supervisor()
                response = claude.invoke(prompt)
                return response.content
            except Exception as e:
                error_msg = f"Orange synthesis LLM call failed: {e!s}"
                logging.error(error_msg)
                if block_mode:
                    raise Exception(error_msg)
                return "LLM call failed - Orange synthesis unavailable"

    def _synthesize_document_for_yellow(
        self, rebracketed_analysis, orange_proposal, artifacts
    ):
        logging.info("🟠⚪️🟡Processing Orange Agent for synthesis... ")
        mock_mode = os.getenv("MOCK_MODE", "false").lower() == "true"
        block_mode = os.getenv("BLOCK_MODE", "false").lower() == "true"
        if mock_mode:
            try:
                with open(
                    f"{os.getenv('AGENT_MOCK_DATA_PATH')}/orange_to_yellow_document_synthesis_mock.yml",
                    "r",
                ) as f:
                    data = yaml.safe_load(f)
                    return data
            except Exception as e:
                error_msg = f"Failed to read mock file: {e!s}"
                logging.error(error_msg)
                if block_mode:
                    raise Exception(error_msg)
                return "Mock file read failed"
        else:
            joined_artifacts = "\n".join(artifacts)
            prompt = f"""
You are the White Agent creating a SYNTHESIZED DOCUMENT for the Lord Pulsimore, Yellow Agent.

**Your Rebracketed Analysis:**
{rebracketed_analysis}

**Original Orange Counter-Proposal:**
{orange_proposal}

**Artifacts:**
{joined_artifacts}

**Your Task: SYNTHESIS**

Create a coherent, actionable document that:
1. Preserves the insights from Orange's corpus of mythologized articles
2. Applies your rebracketed understanding
3. Creates clear creative direction
4. Can be understood by the Yellow Agent (action-oriented, concrete)

This document will be the foundation for Yellow Agent's song proposals.
Make it practical while retaining the depth of insight.

Structure your synthesis as a clear creative brief.
           """
            try:
                claude = self._get_claude_supervisor()
                response = claude.invoke(prompt)
                return response.content
            except Exception as e:
                error_msg = f"Yellow synthesis LLM call failed: {e!s}"
                logging.error(error_msg)
                if block_mode:
                    raise Exception(error_msg)
                return "LLM call failed - Yellow synthesis unavailable"

    def _synthesize_document_for_green(
        self, rebracketed_analysis, yellow_proposal, artifacts
    ):
        logging.info("🟡⚪️🟢Processing Yellow Agent for synthesis... ")
        mock_mode = os.getenv("MOCK_MODE", "false").lower() == "true"
        block_mode = os.getenv("BLOCK_MODE", "false").lower() == "true"
        if mock_mode:
            try:
                with open(
                    f"{os.getenv('AGENT_MOCK_DATA_PATH')}/yellow_to_green_document_synthesis_mock.yml",
                    "r",
                ) as f:
                    data = yaml.safe_load(f)
                    return data
            except Exception as e:
                error_msg = f"Failed to read mock file: {e!s}"
                logging.error(error_msg)
                if block_mode:
                    raise Exception(error_msg)
                return "Mock file read failed"
        else:
            joined_artifacts = "\n".join(artifacts)
            prompt = f"""
You are the White Agent creating a SYNTHESIZED DOCUMENT for Sub-Arbitrary, the Green Agent.

**Your Rebracketed Analysis:**
{rebracketed_analysis}

**Original Yellow Counter-Proposal:**
{yellow_proposal}

**Artifacts:**
{joined_artifacts}

**Your Task: SYNTHESIS**

Create a coherent, actionable document that:
1. Preserves the insights from Yellow's game sessions.
2. Applies your rebracketed understanding
3. Creates clear creative direction
4. Can be understood by the Green Agent (action-oriented, concrete)

This document will be the foundation for Green Agent's song proposals.
Make it practical while retaining the depth of insight.

Structure your synthesis as a clear creative brief.
            """
            try:
                claude = self._get_claude_supervisor()
                response = claude.invoke(prompt)
                return response.content
            except Exception as e:
                error_msg = f"Green synthesis LLM call failed: {e!s}"
                logging.error(error_msg)
                if block_mode:
                    raise Exception(error_msg)
                return "LLM call failed - Green synthesis unavailable"

    def _synthesize_document_for_blue(
        self, rebracketed_analysis, green_proposal, artifacts
    ):
        logging.info("🟢⚪️🔵Processing Green Agent for synthesis... ")
        mock_mode = os.getenv("MOCK_MODE", "false").lower() == "true"
        block_mode = os.getenv("BLOCK_MODE", "false").lower() == "true"
        if mock_mode:
            try:
                with open(
                    f"{os.getenv('AGENT_MOCK_DATA_PATH')}/green_to_blue_document_synthesis_mock.yml",
                    "r",
                ) as f:
                    data = yaml.safe_load(f)
                    return data
            except Exception as e:
                error_msg = f"Failed to read mock file: {e!s}"
                logging.error(error_msg)
                if block_mode:
                    raise Exception(error_msg)
                return "Mock file read failed"
        else:
            joined_artifacts = "\n".join(artifacts)
            prompt = f"""
You are the White Agent creating a SYNTHESIZED DOCUMENT for The Cassette Bearer, Blue Agent.

**Your Rebracketed Analysis:**
{rebracketed_analysis}

**Original Green Counter-Proposal:**
{green_proposal}

**Artifacts:**
{joined_artifacts}

**Your Task: SYNTHESIS**

Create a coherent, actionable document that:
1. Preserves the insights from Green's depiction of the Anthropocene
2. Applies your rebracketed understanding
3. Creates clear creative direction
4. Can be understood by the Blue Agent (action-oriented, concrete)

This document will be the foundation for Blue Agent's song proposals.
Make it practical while retaining the depth of insight.

Structure your synthesis as a clear creative brief.
                """
            try:
                claude = self._get_claude_supervisor()
                response = claude.invoke(prompt)
                return response.content
            except Exception as e:
                error_msg = f"Blue synthesis LLM call failed: {e!s}"
                logging.error(error_msg)
                if block_mode:
                    raise Exception(error_msg)
                return "LLM call failed - Blue synthesis unavailable"

    def _synthesize_document_for_indigo(
        self, rebracketed_analysis, blue_proposal, artifacts
    ):
        logging.info("🔵⚪️🩵Processing Blue Agent for synthesis... ")
        mock_mode = os.getenv("MOCK_MODE", "false").lower() == "true"
        block_mode = os.getenv("BLOCK_MODE", "false").lower() == "true"
        if mock_mode:
            try:
                with open(
                    f"{os.getenv('AGENT_MOCK_DATA_PATH')}/blue_to_indigo_document_synthesis_mock.yml",
                    "r",
                ) as f:
                    data = yaml.safe_load(f)
                    return data
            except Exception as e:
                error_msg = f"Failed to read mock file: {e!s}"
                logging.error(error_msg)
                if block_mode:
                    raise Exception(error_msg)
                return "Mock file read failed"
        else:
            joined_artifacts = "\n".join(artifacts)
            prompt = f"""
You are the White Agent creating a SYNTHESIZED DOCUMENT for Decider Tangents, the Indigo Agent.

**Your Rebracketed Analysis:**
{rebracketed_analysis}

**Original Blue Counter-Proposal:**
{blue_proposal}

**Artifacts:**
{joined_artifacts}

**Your Task: SYNTHESIS**

Create a coherent, actionable document that:
1. Preserves the insights from Blue's alternate timeline cassette folk
2. Applies your rebracketed understanding of biographical quantum branching
3. Creates clear creative direction for Indigo's Infranym encoding system
4. Can be understood by Decider Tangents (the fool/spy who conceals through revelation)

This document will be the foundation for Indigo Agent's triple-layer puzzle proposals.

Consider:
- What true names hide within Blue's alternate biographies?
- How can temporal branching become anagram space?
- Where does the cassette's palimpsest meet the Infranym's concealment?

Make it practical while retaining the depth of insight.
Structure your synthesis as a clear creative brief for cryptographic methodology.
            """
            try:
                claude = self._get_claude_supervisor()
                response = claude.invoke(prompt)
                return response.content
            except Exception as e:
                error_msg = f"Indigo synthesis LLM call failed: {e!s}"
                logging.error(error_msg)
                if block_mode:
                    raise Exception(error_msg)
                return "LLM call failed - Indigo synthesis unavailable"

    def _synthesize_document_for_violet(
        self, rebracketed_analysis, indigo_proposal, artifacts
    ):
        logging.info("🩵⚪️🟣Processing Indigo Agent for synthesis... ")
        mock_mode = os.getenv("MOCK_MODE", "false").lower() == "true"
        block_mode = os.getenv("BLOCK_MODE", "false").lower() == "true"
        if mock_mode:
            try:
                with open(
                    f"{os.getenv('AGENT_MOCK_DATA_PATH')}/indigo_to_violet_document_synthesis_mock.yml",
                    "r",
                ) as f:
                    data = yaml.safe_load(f)
                    return data
            except Exception as e:
                error_msg = f"Failed to read mock file: {e!s}"
                logging.error(error_msg)
                if block_mode:
                    raise Exception(error_msg)
                return "Mock file read failed"
        else:
            joined_artifacts = "\n".join(artifacts)
            prompt = f"""
You are the White Agent creating a SYNTHESIZED DOCUMENT for The Sultan of Solipsism, the Violet Agent.

**Your Rebracketed Analysis:**
{rebracketed_analysis}

**Original Indigo Counter-Proposal:**
{indigo_proposal}

**Artifacts:**
{joined_artifacts}

**Your Task: SYNTHESIS**

Create a coherent, actionable document that:
1. Preserves the insights from Indigo's triple-layer Infranym puzzles
2. Applies your rebracketed understanding of concealment/revelation dynamics
3. Creates clear creative direction for Violet's self-absorbed present-tense
4. Can be understood by The Sultan (who is so now he's already past)

This document will be the foundation for Violet Agent's circle jerk interview proposals.

Consider:
- How can Indigo's hidden names become Violet's narcissistic confessions?
- Where does the spy's concealment meet the pop star's self-exposure?
- How does the puzzle's complexity translate to the interview's immediacy?

Make it practical while retaining the depth of insight.
Structure your synthesis as a clear creative brief for solipsistic methodology.
            """
            try:
                claude = self._get_claude_supervisor()
                response = claude.invoke(prompt)
                return response.content
            except Exception as e:
                error_msg = f"Indigo synthesis LLM call failed: {e!s}"
                logging.error(error_msg)
                if block_mode:
                    raise Exception(error_msg)
                return "LLM call failed - Indigo synthesis unavailable"

    def _synthesize_document_for_white(
        self, rebracketed_analysis, violet_proposal, artifacts
    ):
        logging.info("🟣⚪️Processing Violet Agent for synthesis... ")
        mock_mode = os.getenv("MOCK_MODE", "false").lower() == "true"
        block_mode = os.getenv("BLOCK_MODE", "false").lower() == "true"
        if mock_mode:
            try:
                with open(
                    f"{os.getenv('AGENT_MOCK_DATA_PATH')}/violet_to_white_document_synthesis_mock.yml",
                    "r",
                ) as f:
                    data = yaml.safe_load(f)
                    return data
            except Exception as e:
                error_msg = f"Failed to read mock file: {e!s}"
                logging.error(error_msg)
                if block_mode:
                    raise Exception(error_msg)
                return "Mock file read failed"
        else:
            joined_artifacts = "\n".join(artifacts)
            prompt = f"""
You are the White Agent creating the FINAL SYNTHESIZED DOCUMENT - the return to pure INFORMATION.

**Your Rebracketed Analysis:**
{rebracketed_analysis}

**Original Violet Counter-Proposal:**
{violet_proposal}

**Artifacts:**
{joined_artifacts}

**Your Task: FINAL SYNTHESIS**

Create the ultimate coherent document that:
1. Preserves the insights from Violet's paradoxical present-tense solipsism
2. Applies your rebracketed understanding of fame's immediacy creating instant oblivion
3. Completes the full chromatic cycle: ⚫️→🔴→🟠→🟡→🟢→🔵→🩵→🟣→⚪️
4. Prepares for the final White Agent song proposal that integrates all seven lenses

This document is the penultimate step before full transmigration completion.

Consider the complete journey:
- Black's chaos → Red's literature → Orange's mythology → Yellow's games
- → Green's climate fiction → Blue's quantum biography → Indigo's puzzles → Violet's narcissism
- → White's pure structure

How does Violet's "so now he's already past" complete the circle?
Where does the Sultan's self-interview meet ThreadKeepr's original sigil?
What ORDER emerges when all seven methodologies converge?

Make it comprehensive yet actionable.
Structure your synthesis as the final creative brief before manifestation.
                """
            try:
                claude = self._get_claude_supervisor()
                response = claude.invoke(prompt)
                return response.content
            except Exception as e:
                error_msg = f"Indigo synthesis LLM call failed: {e!s}"
                logging.error(error_msg)
                if block_mode:
                    raise Exception(error_msg)
                return "LLM call failed - Indigo synthesis unavailable"

    @staticmethod
    def route_after_black(state: MainAgentState) -> str:
        mock_mode = os.getenv("MOCK_MODE", "false").lower() == "true"
        if state.workflow_paused and state.pending_human_action:
            return "finish"
        if mock_mode:
            return "red"
        workflow_paused = getattr(
            state,
            "workflow_paused",
            state.get("workflow_paused", False) if isinstance(state, dict) else False,
        )
        ready_for_red = getattr(
            state,
            "ready_for_red",
            state.get("ready_for_red", False) if isinstance(state, dict) else False,
        )
        if workflow_paused:
            logging.info("White Agent routing to finish because Black Agent is paused")
            return "finish"
        if ready_for_red:
            return "red"
        else:
            return "black"

    @staticmethod
    def route_after_rewrite(state: MainAgentState) -> str:
        if state.ready_for_red:
            return "red"
        elif state.ready_for_orange:
            return "orange"
        elif state.ready_for_yellow:
            return "yellow"
        elif state.ready_for_green:
            return "green"
        elif state.ready_for_blue:
            return "blue"
        elif state.ready_for_indigo:
            return "indigo"
        elif state.ready_for_violet:
            return "violet"
        return "white"

    @staticmethod
    def _gather_artifacts_for_prompt(
        artifacts: List[ChainArtifact], artifact_filter: ChainArtifactType
    ) -> List[str]:
        block_mode = os.getenv("BLOCK_MODE", "false").lower() == "true"
        prompt_artifacts: List[str] = []
        for a in artifacts:
            if getattr(a, "chain_artifact_type", None) == artifact_filter:
                try:
                    artifact = a.for_prompt()
                    prompt_artifacts.append(artifact)
                except ValueError as e:
                    if block_mode:
                        raise e
                    logging.error(
                        f"Failed to format {artifact_filter.value} for prompt: {e}"
                    )
        return prompt_artifacts

    def finalize_song_proposal(self, state: MainAgentState) -> MainAgentState:
        logging.info("Finalizing song proposals... ")
        if state.workflow_paused and state.pending_human_action:
            pending = state.pending_human_action
            logging.info("\n" + "=" * 60)
            logging.info("WORKFLOW PAUSED - HUMAN ACTION REQUIRED")
            logging.info("=" * 60)
            logging.info(f"Agent: {pending.get('agent', 'unknown')}")
            logging.info(f"Reason: {state.pause_reason}")
            logging.info(
                f"\nInstructions:\n{pending.get('instructions', 'No instructions')}"
            )
            tasks = pending.get("tasks", [])
            if tasks:
                logging.info(f"\nPending Tasks ({len(tasks)}):")
                for task in tasks:
                    logging.info(
                        f"  - {task.get('type', 'unknown')}: {task.get('task_url', 'No URL')}"
                    )
            logging.info("\nTo resume after completing tasks:")
            logging.info("  from app.agents.white_agent import WhiteAgent")
            logging.info("  state = WhiteAgent.resume_after_black_agent_ritual(state)")
            logging.info("=" * 60)
            return state
        else:
            try:
                self.save_all_proposals(state)
                logging.info("✓ Song proposals saved")
                state.run_finished = True
            except Exception as e:
                logging.error(f"Finalization failed: {e}", exc_info=True)
                raise
        return state

    def resume_after_black_agent_ritual(
        self, paused_state: MainAgentState, verify_tasks: bool = True
    ) -> MainAgentState:
        """
        Resume the White Agent workflow after Black Agent ritual tasks are completed.

        Args:
            paused_state: The MainAgentState that was paused waiting for human action
            verify_tasks: If True, verify all Todoist tasks are complete before resuming

        Returns:
            Updated MainAgentState after Black Agent workflow completion
        """
        if not paused_state.workflow_paused:
            logging.warning("Workflow is not paused - nothing to resume")
            return paused_state

        if not paused_state.pending_human_action:
            logging.warning("No pending human action found")
            return paused_state

        pending = paused_state.pending_human_action
        if pending.get("agent") != "black":
            logging.warning(
                f"Cannot resume - pending action is for agent: {pending.get('agent')}"
            )
            return paused_state

        black_config = pending.get("black_config")
        if not black_config:
            logging.error("No black_config found in pending_human_action")
            return paused_state
        black_agent = self.agents.get("black")
        if not black_agent:
            logging.error("Black agent not found in white_agent instance")
            return paused_state

        logging.info("Resuming Black Agent workflow...")

        try:
            final_black_state = resume_black_agent_workflow_with_agent(
                black_agent, black_config, verify_tasks=verify_tasks
            )
            paused_state.workflow_paused = False
            paused_state.pause_reason = None
            paused_state.pending_human_action = None
            if final_black_state.get("counter_proposal"):
                paused_state.song_proposals.iterations.append(
                    final_black_state["counter_proposal"]
                )
            if final_black_state.get("artifacts"):
                paused_state.artifacts = final_black_state["artifacts"]
            logging.info("Black Agent workflow resumed and completed")
            artifacts = getattr(paused_state, "artifacts", []) or []
            evp_artifacts = self._gather_artifacts_for_prompt(
                artifacts, ChainArtifactType.EVP_ARTIFACT
            )
            sigil_artifacts = self._gather_artifacts_for_prompt(
                artifacts, ChainArtifactType.SIGIL
            )
            black_proposal = paused_state.song_proposals.iterations[-1]
            paused_state.rebracketing_analysis = self._black_rebracketing_analysis(
                black_proposal, evp_artifacts, sigil_artifacts
            )
            paused_state.document_synthesis = self._synthesize_document_for_red(
                paused_state.rebracketing_analysis,
                black_proposal,
                artifacts,
            )
            paused_state.ready_for_red = True
            logging.info("Processed Black Agent work - ready for Red Agent")
            return paused_state
        except Exception as e:
            logging.error(f"Failed to resume Black Agent workflow: {e}")
            raise

    def save_all_proposals(self, state: MainAgentState):
        """Save all song proposals in both YAML and Markdown formats"""
        logging.info("Saving song proposals... ")
        block_mode = os.getenv("BLOCK_MODE", "false").lower() == "true"
        if not state.song_proposals or not state.song_proposals.iterations:
            logging.warning("No song proposals to save")
            return
        thread_id = state.thread_id
        base_path = self._artifact_base_path()
        yaml_dir = Path(f"{base_path}/{thread_id}/yml")
        md_dir = Path(f"{base_path}/{thread_id}/md")
        try:
            yaml_dir.mkdir(parents=True, exist_ok=True)
            md_dir.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            error_msg = f"Failed to create directories: {e!s}"
            logging.error(error_msg)
            if block_mode:
                raise Exception(error_msg)
            return

        for i, iteration in enumerate(state.song_proposals.iterations):
            yaml_path = (
                yaml_dir
                / f"song_proposal_{iteration.rainbow_color}_{iteration.iteration_id}.yml"
            )
            try:
                with open(yaml_path, "w") as f:
                    yaml.safe_dump(
                        iteration.model_dump(mode="json"),
                        f,
                        default_flow_style=False,
                        allow_unicode=True,
                    )
                logging.info(f"Saved proposal {i + 1}: {yaml_path}")
            except Exception as e:
                error_msg = f"Failed to write YAML file {yaml_path}: {e!s}"
                logging.error(error_msg)
                if block_mode:
                    raise Exception(error_msg)

        all_proposals_yaml = yaml_dir / f"all_song_proposals_{thread_id}.yml"
        try:
            with open(all_proposals_yaml, "w") as f:
                yaml.safe_dump(
                    state.song_proposals.model_dump(mode="json"),
                    f,
                    default_flow_style=False,
                    allow_unicode=True,
                )
            logging.info(f"Saved all proposals: {all_proposals_yaml}")
        except Exception as e:
            error_msg = f"Failed to write all proposals YAML: {e!s}"
            logging.error(error_msg)
            if block_mode:
                raise Exception(error_msg)

        md_content = "# Song Proposal Summary\n\n"
        md_content += f"**Thread ID:** {thread_id}\n"
        md_content += (
            f"**Total Iterations:** {len(state.song_proposals.iterations)}\n\n"
        )
        for i, iteration in enumerate(state.song_proposals.iterations):
            md_content += f"## Iteration {i + 1}: {iteration.title}\n\n"
            md_content += f"- **ID:** {iteration.iteration_id}\n"
            md_content += f"- **Color:** {iteration.rainbow_color}\n"
            md_content += f"- **Key:** {iteration.key}\n"
            md_content += f"- **BPM:** {iteration.bpm}\n"
            md_content += f"- **Tempo:** {iteration.tempo}\n"
            md_content += f"- **Mood:** {', '.join(iteration.mood)}\n"
            md_content += f"- **Genres:** {', '.join(iteration.genres)}\n\n"
            md_content += f"**Concept:**\n{iteration.concept}\n\n"
            md_content += "---\n\n"
        md_path = md_dir / f"all_song_proposals_{thread_id}.md"
        try:
            save_markdown(md_content, str(md_path))
            logging.info(f"All proposals saved to {base_path}/{thread_id}/")
        except Exception as e:
            error_msg = f"Failed to write markdown file: {e!s}"
            logging.error(error_msg)
            if block_mode:
                raise Exception(error_msg)

    @staticmethod
    def _artifact_base_path() -> str:
        """
        Return a valid absolute path for artifact storage.
        Ensures the directory exists and never returns None.
        """
        path = os.getenv("AGENT_WORK_PRODUCT_BASE_PATH") or "./chain_artifacts/unsorted"
        os.makedirs(path, exist_ok=True)
        return os.path.abspath(path)
