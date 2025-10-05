import os
import uuid
import yaml

from abc import ABC
from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langgraph.checkpoint.memory import MemorySaver
from langgraph.constants import START, END
from langgraph.graph import StateGraph

from app.agents.models.agent_settings import AgentSettings
from app.agents.base_rainbow_agent import BaseRainbowAgent
from app.agents.models.evp_artifact import EVPArtifact
from app.agents.states.base_rainbow_agent_state import BaseRainbowAgentState
from app.agents.states.black_agent_state import BlackAgentState
from app.agents.states.main_agent_state import MainAgentState
from app.agents.tools.audio_tools import select_random_segment_audio
from app.agents.tools.speech_tools import evp_speech_to_text
from app.structures.manifests.song_proposal import SongProposalIteration, SongProposal
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
        self.state_graph = BlackAgentState()

    def __call__(self, state: MainAgentState) -> MainAgentState:

        """Entry point when White Agent invokes Black Agent"""

        current_proposal = state.song_proposals[-1]
        black_state = BlackAgentState(
            white_proposal=current_proposal,
            thread_id=state.thread_id
        )
        if not hasattr(self, '_compiled_workflow'):
            self._compiled_workflow = self.create_graph().compile(
                checkpointer=MemorySaver(),
                interrupt_before=["write_artifacts"]  # Or wherever human needed
            )
        black_config = {"configurable": {"thread_id": f"black_{state.thread_id}"}}
        self._compiled_workflow.invoke(black_state, config=black_config)
        snapshot = self._compiled_workflow.get_state(black_config)
        final_black_state = snapshot.values
        if snapshot.next:
            state.pending_human_action = {
                "agent": "black",
                "action": "sigil_imbument",
                "instructions": final_black_state.get("human_instructions", "Black Agent needs human input"),
                "black_config": black_config
            }
        else:
            state.song_proposals.iterations.append(final_black_state.counter_proposal)
        return state


    def route_after_spec(self, state: BlackAgentState) -> str:

        """Complex routing based on what artifacts exist"""

        has_sigil = any(a.type == "sigil" for a in state.artifacts)  # change to content
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
        black_workflow.add_node("generate_alternate_song_spec", self.generate_alternate_song_spec)
        black_workflow.add_node("generate_sigil", self.generate_sigil)
        black_workflow.add_node("generate_evp", self.generate_evp)
        black_workflow.add_node("route_after_spec", self.route_after_spec)
        black_workflow.add_edge(START, "generate_alternate_song_spec")
        black_workflow.add_conditional_edges(
            "route_after_spec",
            self.route_after_spec,
            {
                "need_sigil": "generate_sigil",
                "need_evp": "generate_evp",
                "ready_for_proposal": "generate_alternate_song_spec",
                "done": END
            }
        )
        black_workflow.add_edge("generate_sigil", "generate_alternate_song_spec")
        black_workflow.add_edge("generate_evp", "generate_alternate_song_spec")
        black_workflow.add_edge("generate_alternate_song_spec", END)

        return black_workflow


    def generate_alternate_song_spec(self, state: BlackAgentState) -> BlackAgentState:
        mock_mode = os.getenv("MOCK_MODE", "false").lower() == "true"
        if mock_mode:
            with open("/Volumes/LucidNonsense/White/app/agents/mocks/black_counter_proposal_mock.yml", "r") as f:
                data = yaml.safe_load(f)
            counter_proposal = SongProposalIteration(**data)
            if counter_proposal:
                if not hasattr(state, "song_proposal") or state.song_proposal is None:
                    state.song_proposal = SongProposal(iterations=[])
                state.song_proposal.iterations.append(counter_proposal)
            return state
        prompt = f"""
            General writing style: While this character is a bit over the top - the writing shouldn't be. References to
            occultism, hacking, spycraft, and surrealist art should be subtle and woven into the narrative. The tone should be
            darkly humorous, with a touch of absurdity. Our black agent isn't exactly thrilled to be in his position or to
            have discovered what he has.
            Context: You are the black agent, keeper of the conjurer's thread. You live on the edge of reality, pushed to the
            brink of madness by the Demiurge that rules the world. A rare voice of light and hope, your hero David Bowie, 
            has just died. You have lost your hero and the worst person in the world, the fascist conman Donald Trump, a man 
            who embodies the Demiurge, has won control of the world. ("The Man Who Sold the World"?) Your life of hacking, 
            occultism, spycraft, and surrealist art has placed you on every list there is. Look at the current song proposal. 
            Is this the kind of song that that will resist the Demiurge and his minions? If not, how can it be improved? What
            is missing? What is wrong? What is the hidden meaning? What is the secret message? How can it be made more 
            subtly subversive? How can it be made more powerful to those in the know ? How can it be made more magical? 
            Current song proposal:
            {state.song_proposal.iterations[-1]}
            Here are The Rainbow Table song manifests for black, The Conjurer's Thread as a basis for your counter proposal:
            {get_my_reference_proposals('Z')}
            If you find your idea lacking, use the following tools at your disposal to create a counter proposal:
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
            assert isinstance(counter_proposal,
                              SongProposalIteration), f"Expected SongProposalIteration, got {type(counter_proposal)}"
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

    def contribute(self,agent_state: BaseRainbowAgentState) -> StateGraph:
        pass

    def generate_document(self, agent_state: BaseRainbowAgentState) -> StateGraph:
        pass

    @staticmethod
    def generate_evp(state: BlackAgentState) -> BlackAgentState:
        mock_mode = os.getenv("MOCK_MODE", "false").lower() == "true"
        if mock_mode:
            with open("/Volumes/LucidNonsense/White/app/agents/mocks/black_evp_artifact_mock.yml", "r") as f:
                data = yaml.safe_load(f)
            state.evp_artifact = EVPArtifact(**data)
            return state
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

    def generate_sigil(self):
        pass