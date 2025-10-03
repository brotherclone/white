import os
import uuid

from abc import ABC
from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langgraph.checkpoint.memory import MemorySaver
from langgraph.constants import START, END
from langgraph.graph import StateGraph

from app.agents.models.agent_settings import AgentSettings
from app.agents.base_rainbow_agent import BaseRainbowAgent
from app.agents.states.black_agent_state import BlackAgentState
from app.agents.states.main_agent_state import MainAgentState
from app.agents.tools.audio_tools import select_random_segment_audio, create_random_audio_mosaic, blend_with_noise
from app.agents.tools.magick_tools import SigilTools
from app.agents.tools.speech_tools import evp_speech_to_text
from app.structures.manifests.song_proposal import SongProposalIteration
from app.util.manifest_loader import get_my_reference_proposals

load_dotenv()

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
        self.sigil_tools = SigilTools()
        self.state_graph = BlackAgentState()



    def __call__(self, state: MainAgentState) -> MainAgentState:

        """Entry point when White Agent invokes Black Agent"""
        mock_mode = os.getenv("MOCK_MODE", "false").lower() == "true"
        # Extract what Black needs from MainAgentState
        current_proposal = state.song_proposals[-1]

        # Create Black's internal state
        black_state = BlackAgentState(
            white_proposal=current_proposal,
            thread_id=state.thread_id
        )

        # Get Black's compiled workflow (compile once in __init__ or here)
        if not hasattr(self, '_compiled_workflow'):
            self._compiled_workflow = self.create_graph().compile(
                checkpointer=MemorySaver(),
                interrupt_before=["write_artifacts"]  # Or wherever human needed
            )

        # Run Black's complete workflow
        black_config = {"configurable": {"thread_id": f"black_{state.thread_id}"}}
        self._compiled_workflow.invoke(black_state, config=black_config)

        # Get results from Black's workflow
        snapshot = self._compiled_workflow.get_state(black_config)
        final_black_state = snapshot.values

        # Check if Black needs human input (interrupted)
        if snapshot.next:
            # Black is waiting for human - set flag in MainAgentState
            state.pending_human_action = {
                "agent": "black",
                "action": "sigil_imbument",
                "instructions": final_black_state.get("human_instructions", "Black Agent needs human input"),
                "black_config": black_config
            }
        else:
            # Black finished - add results to MainAgentState
            state.song_proposals.append(final_black_state.counter_proposal)
            state.artifacts.extend(final_black_state.artifacts)

        return state

    def route_after_critique(self, state: BlackAgentState) -> str:
        """Complex routing based on what artifacts exist"""

        has_sigil = any(a.type == "sigil" for a in state.artifacts)
        has_evp = any(a.type == "evp" for a in state.artifacts)

        if not has_sigil:
            return "need_sigil"
        elif not has_evp:
            return "need_evp"
        elif not state.counter_proposal:
            return "ready_for_proposal"
        else:
            return "done"
        
    def create_graph(self) -> StateGraph:

        """Create the BlackAgent's internal workflow graph"""

        black_workflow = StateGraph(BlackAgentState)
        # Add all necessary nodes
        black_workflow.add_node("critique", self.critique)
        black_workflow.add_node("generate_sigil", self.generate_sigil)
        black_workflow.add_node("generate_evp", self.generate_evp)
        black_workflow.add_node("generate_counter_proposal", self.generate_alternate_song_spec)
        # Entry point
        black_workflow.add_edge(START, "critique")
        # Conditional routing after critique
        black_workflow.add_conditional_edges(
            "critique",
            self.route_after_critique,
            {
                "need_sigil": "generate_sigil",
                "need_evp": "generate_evp",
                "ready_for_proposal": "generate_counter_proposal",
                "done": END
            }
        )
        # After generating any artifact, critique again
        black_workflow.add_edge("generate_sigil", "critique")
        black_workflow.add_edge("generate_evp", "critique")
        # After proposal, done
        black_workflow.add_edge("generate_counter_proposal", END)

        return black_workflow

    def normalize_song_proposal_data(self, data):
        key = data.get("key")
        if isinstance(key, dict):
            note = key.get("note")
            if isinstance(note, dict):
                pitch = note.get("pitch_name", "C")
                accidental = note.get("accidental", "")
                mode = key.get("mode", {}).get("name", "Major")
                key_str = f"{pitch}{accidental} {mode}".strip()
                data["key"] = key_str if key_str else "C Major"
            else:
                data["key"] = "C Major"
        elif not isinstance(key, str):
            data["key"] = "C Major"
        valid_pitches = {"C", "D", "E", "F", "G", "A", "B"}
        if isinstance(data["key"], str):
            pitch = data["key"].split()[0].replace("#", "").replace("b", "")
            if pitch not in valid_pitches:
                data["key"] = "C Major"
        return data

    def critique(self, state: BlackAgentState) -> BlackAgentState:
        prompt = f"""
        You are the black agent, keeper of the conjurer's thread. You live on the edge of reality, pushed to the brink
        of madness by the Demiurge that rules the world. A rare voice of light and hope, your hero David Bowie, has just
        died. You have lost your hero and the worst person in the world, the fascist conman Donald Trump, a man who
        embodies the Demiurge, has won control of the world. Your life of hacking, occultism, spycraft, and surrealist
        art has placed you on every list there is. Look at the current song proposal. Is this the kind of song that
        that will resist the Demiurge and his minions? If not, how can it be improved? What is missing? What is wrong?
        What is the hidden meaning? What is the secret message? How can it be made more subversive? How can it be made
        more powerful? How can it be made more magical? How can it be made more effective? How can it be made more
        evocative?
        Current song proposal:
        {state.song_proposal.iterations[-1]}
        Here are The Rainbow Table song manifests for black, The Conjurer's Thread:
        {get_my_reference_proposals('Z')}
        If you find the proposal lacking, use the following tools at your disposal to create a counter proposal:
            1. generate_sigil: Create a sigil that embodies the essence of the song proposal. The sigil should be a
               visual representation of the song's themes and messages. You'll provide instructions to the artist to imbue
               it with magical properties.
            2. generate_evp: Use EVP (Electronic Voice Phenomenon) techniques to capture messages from the spirit world.
               These messages can provide insights and inspiration for the song proposal.
        After using these tools, artifacts will be created that can be used to enhance the song proposal.
        """
        claude = self._get_claude()
        proposer = claude.with_structured_output(SongProposalIteration)
        counter_proposal = None
        try:
            result = proposer.invoke(prompt)
            if isinstance(result, dict):
                result = self.normalize_song_proposal_data(result)
                counter_proposal = SongProposalIteration(**result)
            else:
                counter_proposal = result
            assert isinstance(counter_proposal, SongProposalIteration), f"Expected SongProposalIteration, got {type(counter_proposal)}"
        except Exception as e:
            print(f"Anthropic model call failed: {e!s}; returning stub SongProposalIteration.")
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
        if counter_proposal:
            if not hasattr(state, "song_proposal") or state.song_proposal is None:
                state.song_proposal = SongProposal(iterations=[])
            state.song_proposal.iterations.append(counter_proposal)
        return state

    def generate_evp(self, state: BlackAgentState) -> BlackAgentState:
        select_random_segment_audio(
            root_dir="/Volumes/LucidNonsense/White/staged_raw_material",
            min_duration=10.0,
            num_segments=10,
            output_dir="/Volumes/LucidNonsense/White/app/agents/work_products/black_work_products/audio_segments"
        )
        create_random_audio_mosaic(
            root_dir='/Volumes/LucidNonsense/White/app/agents/work_products/black_work_products/audio_segments',
            slice_duration_ms=50,
            target_length_sec=30,
            output_path='/Volumes/LucidNonsense/White/app/agents/work_products/black_work_products/audio_mosaics/mosaic.wav'
        )
        blend_with_noise(
            input_path='/Volumes/LucidNonsense/White/app/agents/work_products/black_work_products/audio_mosaics/mosaic.wav',
            blend=0.3,
            output_dir='/Volumes/LucidNonsense/White/app/agents/work_products/black_work_products/blended_audios'
        )
        text = evp_speech_to_text(
            "/Volumes/LucidNonsense/White/app/agents/work_products/black_work_products/blended_audios",
            "mosaic_blended.wav")
        if text:
            print("Transcription Result:")
            print(text)
        else:
            print("No transcription could be generated.")
        return state

    def generate_sigil(self, state: BlackAgentState) -> BlackAgentState:
        pass

    def generate_imbuing_instructions(self):
        pass

    def generate_document(self):
        raise NotImplementedError("Subclasses must implement generate_document method")

    def generate_alternate_song_spec(self):
        raise NotImplementedError("Subclasses must implement generate_alternate_song_spec method")

    def contribute(self, **kwargs):
        raise NotImplementedError("Subclasses must implement contribute method")
