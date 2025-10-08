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

from app.agents.enums.sigil_state import SigilState
from app.agents.enums.sigil_type import SigilType
from app.agents.models.agent_settings import AgentSettings
from app.agents.base_rainbow_agent import BaseRainbowAgent
from app.agents.models.evp_artifact import EVPArtifact
from app.agents.models.sigil_artifact import SigilArtifact
from app.agents.states.black_agent_state import BlackAgentState
from app.agents.states.main_agent_state import MainAgentState
from app.agents.tools.audio_tools import get_audio_segments_as_chain_artifacts, \
    create_audio_mosaic_chain_artifact, create_blended_audio_chain_artifact
from app.agents.tools.magick_tools import SigilTools
from app.agents.tools.speech_tools import chain_artifact_file_from_speech_to_text
from app.structures.concepts.rainbow_table_color import the_rainbow_table_colors
from app.structures.manifests.song_proposal import SongProposalIteration, SongProposal
from app.util.manifest_loader import get_my_reference_proposals
from app.reference.mcp.todoist.main import (
    create_sigil_charging_task
)

load_dotenv()
logging.basicConfig(level=logging.INFO)


class BlackAgent(BaseRainbowAgent, ABC):
    """EVP/Sigil Generator - Audio analysis that hallucinates messages"""

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
        self.current_session_sigils = []
        self.state_graph = BlackAgentState(
            thread_id=f"black_{uuid.uuid4()}",
            song_proposal=None,
            white_proposal=None,
            counter_proposal=None,
            artifacts=[],
            awaiting_human_action=False,
            human_instructions="",
            pending_human_tasks=[]
        )

    def __call__(self, state: MainAgentState) -> MainAgentState:
        """Entry point when White Agent invokes Black Agent"""

        current_proposal = state.song_proposals.iterations[-1]
        black_state = BlackAgentState(
            white_proposal=current_proposal,
            thread_id=state.thread_id
        )

        if not hasattr(self, '_compiled_workflow'):
            self._compiled_workflow = self.create_graph().compile(
                checkpointer=MemorySaver(),
                interrupt_before=["await_human_action"]  # Interrupt before human step
            )

        black_config = {"configurable": {"thread_id": f"black_{state.thread_id}"}}
        result = self._compiled_workflow.invoke(black_state, config=black_config)
        snapshot = self._compiled_workflow.get_state(black_config)
        if snapshot.next:
            final_black_state = snapshot.values
            state.pending_human_action = {
                "agent": "black",
                "action": "sigil_charging",
                "instructions": final_black_state.get("human_instructions", "Black Agent needs human input"),
                "pending_tasks": final_black_state.get("pending_human_tasks", []),
                "black_config": black_config,
                "resume_instructions": """
                After completing the ritual tasks:
                1. Mark all Todoist tasks as complete
                2. Call resume_black_agent_workflow(black_config) to continue
                """
            }
        else:
            final_black_state = snapshot.values
            if final_black_state.get("counter_proposal"):
                state.song_proposals.iterations.append(final_black_state["counter_proposal"])

        return state

    def route_after_spec(self, state: BlackAgentState) -> str:
        """
        Routing function (NOT a node!) - returns next node name as string.
        Used in add_conditional_edges.

        Logic:
        1. If awaiting human action -> pause workflow
        2. If no sigil yet -> generate sigil
        3. If no EVP yet -> generate EVP
        4. If have both artifacts -> finalize
        5. Otherwise -> done
        """
        has_sigil = any(getattr(a, 'type', None) == "sigil" for a in state.artifacts)
        has_evp = any(getattr(a, 'type', None) == "evp" for a in state.artifacts)

        # Priority 1: If waiting for human, go to await node
        if state.awaiting_human_action:
            return "await_human"

        # Priority 2: Generate missing artifacts
        if not has_sigil:
            return "need_sigil"
        if not has_evp:
            return "need_evp"

        # Priority 3: Have all artifacts, finalize
        if has_sigil and has_evp and not state.counter_proposal:
            return "ready_for_proposal"

        # Priority 4: Everything done
        return "done"

    def create_graph(self) -> StateGraph:
        """Create the BlackAgent's internal workflow graph"""

        black_workflow = StateGraph(BlackAgentState)

        # Add nodes (these MUST return updated state)
        black_workflow.add_node("generate_alternate_song_spec", self.generate_alternate_song_spec)
        black_workflow.add_node("generate_sigil", self.generate_sigil)
        black_workflow.add_node("generate_evp", self.generate_evp)
        black_workflow.add_node("await_human_action", self.await_human_action)
        black_workflow.add_node("finalize_counter_proposal", self.finalize_counter_proposal)

        # Start with spec generation
        black_workflow.add_edge(START, "generate_alternate_song_spec")

        # After initial spec, check what's needed
        black_workflow.add_conditional_edges(
            "generate_alternate_song_spec",
            self.route_after_spec,
            {
                "need_sigil": "generate_sigil",
                "need_evp": "generate_evp",
                "await_human": "await_human_action",
                "ready_for_proposal": "finalize_counter_proposal",
                "done": END
            }
        )

        # After generating sigil, check if we need EVP or should wait for human
        black_workflow.add_conditional_edges(
            "generate_sigil",
            self.route_after_spec,
            {
                "need_sigil": END,  # Should never happen - prevent loop
                "need_evp": "generate_evp",
                "await_human": "await_human_action",
                "ready_for_proposal": "finalize_counter_proposal",
                "done": END
            }
        )

        # After generating EVP, check if we need sigil or should finalize
        black_workflow.add_conditional_edges(
            "generate_evp",
            self.route_after_spec,
            {
                "need_sigil": "generate_sigil",
                "need_evp": END,  # Should never happen - prevent loop
                "await_human": "await_human_action",
                "ready_for_proposal": "finalize_counter_proposal",
                "done": END
            }
        )

        # Human action interrupts here, then finalizes when resumed
        black_workflow.add_edge("await_human_action", "finalize_counter_proposal")

        # Finalize and end
        black_workflow.add_edge("finalize_counter_proposal", END)

        return black_workflow

    def generate_alternate_song_spec(self, state: BlackAgentState) -> BlackAgentState:
        """Generate initial counter-proposal or iterate on existing one"""

        mock_mode = os.getenv("MOCK_MODE", "false").lower() == "true"

        if mock_mode:
            with open("/Volumes/LucidNonsense/White/app/agents/mocks/black_counter_proposal_mock.yml", "r") as f:
                data = yaml.safe_load(f)
            counter_proposal = SongProposalIteration(**data)
            if not state.song_proposal:
                state.song_proposal = SongProposal(iterations=[])
            state.song_proposal.iterations.append(counter_proposal)
            return state

        prompt = f"""
        General writing style: While this character is a bit over the top - the writing shouldn't be. 
        References to occultism, hacking, spycraft, and surrealist art should be subtle and woven into 
        the narrative. The tone should be darkly humorous, with a touch of absurdity. Our black agent 
        isn't exactly thrilled to be in his position or to have discovered what he has.

        Context: You are the black agent, keeper of the conjurer's thread. You live on the edge of 
        reality, pushed to the brink of madness by the Demiurge that rules the world. A rare voice of 
        light and hope, your hero David Bowie, has just died. You have lost your hero and the worst 
        person in the world, the fascist conman Donald Trump, a man who embodies the Demiurge, has won 
        control of the world. Your life of hacking, occultism, spycraft, and surrealist art has placed 
        you on every list there is. 

        Look at the current song proposal. Is this the kind of song that will resist the Demiurge and 
        his minions? If not, how can it be improved? What is missing? What is wrong? What is the hidden 
        meaning? What is the secret message? How can it be made more subtly subversive? How can it be 
        made more powerful to those in the know? How can it be made more magical?

        Current song proposal:
        {state.white_proposal}

        Here are The Rainbow Table song manifests for black, The Conjurer's Thread as a basis for your 
        counter proposal:
        {get_my_reference_proposals('Z')}

        Create a counter-proposal that enhances the subversive magical potential of this song.
        """

        claude = self._get_claude()
        proposer = claude.with_structured_output(SongProposalIteration)

        try:
            result = proposer.invoke(prompt)
            if isinstance(result, dict):
                result = self.normalize_song_proposal_data(result)
                counter_proposal = SongProposalIteration(**result)
            else:
                counter_proposal = result
        except Exception as e:
            logging.error(f"Anthropic model call failed: {e!s}")
            counter_proposal = SongProposalIteration(
                iteration_id=str(uuid.uuid4()),
                bpm=120,
                tempo="4/4",
                key="C Major",
                rainbow_color="black",
                title="Fallback: Black Song",
                mood=["dark"],
                genres=["experimental"],
                concept="Fallback stub because Anthropic model unavailable"
            )

        if not state.song_proposal:
            state.song_proposal = SongProposal(iterations=[])
        state.song_proposal.iterations.append(counter_proposal)

        return state

    def generate_sigil(self, state: BlackAgentState) -> BlackAgentState:

        """Generate a sigil artifact and create Todoist task for charging"""

        mock_mode = os.getenv("MOCK_MODE", "false").lower() == "true"

        if mock_mode:
            mock_path = "/Volumes/LucidNonsense/White/app/agents/mocks/black_sigil_artifact_mock.yml"
            try:
                with open(mock_path, "r") as f:
                    data = yaml.safe_load(f)
                sigil_artifact = SigilArtifact(**data)
                state.artifacts.append(sigil_artifact)
                # In mock mode, also set awaiting_human_action
                state.awaiting_human_action = True
                state.human_instructions = "MOCK: Sigil charging task would be created"
                return state
            except FileNotFoundError:
                logging.warning(f"Mock file not found at {mock_path}, using real generation")

        sigil_maker = SigilTools()
        current_proposal = state.song_proposal.iterations[-1]

        prompt = f"""
        Distill this counter-proposal into a short, actionable wish statement that captures how 
        the song could embody higher occult meaning and resistance against the Demiurge.

        Counter-proposal:
        Title: {current_proposal.title}
        Concept: {current_proposal.concept}
        Mood: {', '.join(current_proposal.mood)}

        Format: A single sentence starting with "I will..." or "This song will..."
        Example: "I will weave hidden frequencies that awaken dormant resistance."
        """
        claude = self._get_claude()
        wish_response = claude.invoke(prompt)
        wish_text = wish_response.content
        statement_of_intent = sigil_maker.create_statement_of_intent(wish_text, True)
        description, components = sigil_maker.generate_word_method_sigil(statement_of_intent)
        charging_instructions = sigil_maker.charge_sigil()
        sigil_artifact = SigilArtifact(
            wish=wish_text,
            statement_of_intent=statement_of_intent,
            glyph_description=description,
            glyph_components=components,
            sigil_type=SigilType.WORD_METHOD,
            activation_state=SigilState.CREATED,  # Not charged yet!
            charging_instructions=charging_instructions,
            thread_id=state.thread_id,
            type="sigil"  # For routing
        )

        state.artifacts.append(sigil_artifact)

        try:
            task_result = create_sigil_charging_task(
                sigil_description=description,
                charging_instructions=charging_instructions,
                song_title=current_proposal.title,
                section_name="Black Agent - Sigil Work"
            )

            state.pending_human_tasks.append({
                "type": "sigil_charging",
                "task_id": task_result["id"],
                "task_url": task_result["url"],
                "artifact_index": len(state.artifacts) - 1,
                "sigil_wish": wish_text
            })

            state.awaiting_human_action = True
            state.human_instructions = f"""
                                        🜏 SIGIL CHARGING REQUIRED
                                        
                                        A sigil has been generated for '{current_proposal.title}'.
                                        
                                        **Todoist Task:** {task_result['url']}
                                        
                                        **Wish:** {wish_text}
                                        
                                        **Glyph:** {description}
                                        
                                        **Instructions:**
                                        {charging_instructions}
                                        
                                        Mark the Todoist task complete after charging, then resume the workflow.
                                        """

            logging.info(f"✓ Created Todoist task for sigil charging: {task_result['id']}")

        except Exception as e:
            logging.error(f"✗ Failed to create Todoist task: {e}")
            state.awaiting_human_action = True
            state.human_instructions = f"""
                                        ⚠️ SIGIL CHARGING REQUIRED (Todoist task creation failed)
                                        
                                        Manually charge the sigil for '{current_proposal.title}':
                                        
                                        **Wish:** {wish_text}
                                        **Glyph:** {description}
                                        
                                        {charging_instructions}
                                        """
        return state

    @staticmethod
    def generate_evp(state: BlackAgentState) -> BlackAgentState:
        """Generate EVP artifact and optionally create analysis task"""

        mock_mode = os.getenv("MOCK_MODE", "false").lower() == "true"

        if mock_mode:
            with open("/Volumes/LucidNonsense/White/app/agents/mocks/black_evp_artifact_mock.yml", "r") as f:
                data = yaml.safe_load(f)
            evp_artifact = EVPArtifact(**data)
            evp_artifact.type = "evp"  # For routing
            state.artifacts.append(evp_artifact)
            return state
        current_proposal = state.song_proposal.iterations[-1]
        segments = get_audio_segments_as_chain_artifacts(
            2.0, 9,
            the_rainbow_table_colors['Z'],
            state.thread_id
        )
        mosaic = create_audio_mosaic_chain_artifact(
            segments, 50,
            getattr(current_proposal, 'target_length', 180),  # Default 3 min
            state.thread_id
        )
        blended = create_blended_audio_chain_artifact(
            mosaic, 0.33,
            state.thread_id
        )
        transcript = chain_artifact_file_from_speech_to_text(
            blended,
            state.thread_id
        )
        evp_artifact = EVPArtifact(
            audio_segments=segments,
            transcript=transcript,
            audio_mosaic=mosaic,
            noise_blended_audio=blended,
            thread_id=state.thread_id,
            type="evp"  # For routing
        )
        state.artifacts.append(evp_artifact)
        return state

    @staticmethod
    def await_human_action(state: BlackAgentState) -> BlackAgentState:
        """
        Node that workflow interrupts at. When workflow resumes, this passes through.
        This is where human completes ritual tasks before workflow continues.
        """
        logging.info("⏸️  Workflow interrupted - awaiting human action on sigil charging")

        # This node just passes through - the interrupt happens BEFORE entering it
        # When workflow is resumed, it will execute and move to next node

        return state

    def finalize_counter_proposal(self, state: BlackAgentState) -> BlackAgentState:
        """
        Create final counter-proposal incorporating sigil and EVP insights.
        This runs AFTER human has charged the sigil.
        """

        current_proposal = state.song_proposal.iterations[-1]
        sigil = next((a for a in state.artifacts if getattr(a, 'type', None) == 'sigil'), None)
        evp = next((a for a in state.artifacts if getattr(a, 'type', None) == 'evp'), None)
        artifact_context = ""
        if sigil:
            artifact_context += f"\nCharged Sigil Wish: {sigil.wish}"
            artifact_context += f"\nGlyph: {sigil.glyph_description}"
        if evp:
            transcript_text = evp.transcript.content if hasattr(evp.transcript, 'content') else str(evp.transcript)
            artifact_context += f"\nEVP Transcript: {transcript_text}"
        prompt = f"""
        You are the Black Agent. The sigil has been charged and the spirits have spoken through EVP.
        Create a final counter-proposal that incorporates these magical elements.

        Current Proposal:
        {current_proposal}

        Magical Artifacts:
        {artifact_context}

        The counter-proposal should subtly reference the sigil's intent and EVP messages without 
        being obvious. Weave the occult resistance into the song's concept and lyrics.
        """

        claude = self._get_claude()
        proposer = claude.with_structured_output(SongProposalIteration)

        try:
            result = proposer.invoke(prompt)
            if isinstance(result, dict):
                result = self.normalize_song_proposal_data(result)
                counter_proposal = SongProposalIteration(**result)
            else:
                counter_proposal = result
        except Exception as e:
            logging.error(f"Failed to create final counter-proposal: {e}")
            # Use the current proposal as fallback
            counter_proposal = current_proposal

        state.counter_proposal = counter_proposal
        state.awaiting_human_action = False  # Done with human tasks

        logging.info(f"✓ Finalized counter-proposal: {counter_proposal.title}")

        return state