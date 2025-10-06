import os
import uuid
import yaml

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
from app.agents.models.text_chain_artifact_file import TextChainArtifactFile
from app.agents.states.black_agent_state import BlackAgentState
from app.agents.states.main_agent_state import MainAgentState
from app.agents.tools.audio_tools import get_audio_segments_as_chain_artifacts, \
    create_audio_mosaic_chain_artifact, create_blended_audio_chain_artifact
from app.agents.tools.magick_tools import SigilTools
from app.agents.tools.speech_tools import chain_artifact_file_from_speech_to_text
from app.structures.concepts.rainbow_table_color import the_rainbow_table_colors
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
        black_workflow.add_edge("generate_alternate_song_spec", "route_after_spec")
        black_workflow.add_conditional_edges(
            "route_after_spec",
            self.route_after_spec,
            {
                "need_sigil": "generate_sigil",
                "need_evp": "generate_evp",
                "ready_for_proposal": "generate_alternate_song_spec",  # Loop back
                "done": END
            }
        )
        black_workflow.add_edge("generate_sigil", "route_after_spec")
        black_workflow.add_edge("generate_evp", "route_after_spec")

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

    @staticmethod
    def generate_evp(state: BlackAgentState) -> BlackAgentState:
        mock_mode = os.getenv("MOCK_MODE", "false").lower() == "true"
        if mock_mode:
            with open("/Volumes/LucidNonsense/White/app/agents/mocks/black_evp_artifact_mock.yml", "r") as f:
                data = yaml.safe_load(f)
            state.evp_artifact = EVPArtifact(**data)
            return state
        segments = get_audio_segments_as_chain_artifacts(2.0, 9, the_rainbow_table_colors['Z'], state.thread_id)
        mosiac = create_audio_mosaic_chain_artifact(segments, 50,  state.song_proposal.target_length, state.thread_id)
        blended = create_blended_audio_chain_artifact(mosiac, 0.33, state.thread_id)
        transcript = chain_artifact_file_from_speech_to_text(blended, state.thread_id)
        evp_artifact = EVPArtifact(
            audio_segments=segments,
            transcript=transcript,
            audio_mosiac=mosiac,
            noise_blended_audio=blended,
            thread_id=state.thread_id
        )
        state.artifacts.append(evp_artifact)
        return state

    def generate_sigil(self, state: BlackAgentState) -> BlackAgentState:
        mock_mode = os.getenv("MOCK_MODE", "false").lower() == "true"
        if mock_mode:
            # Load mock sigil artifact
            mock_path = "/Volumes/LucidNonsense/White/app/agents/mocks/black_sigil_artifact_mock.yml"
            try:
                with open(mock_path, "r") as f:
                    data = yaml.safe_load(f)
                sigil_artifact = SigilArtifact(**data)
                state.artifacts.append(sigil_artifact)
                return state
            except FileNotFoundError:
                print(f"Warning: Mock file not found at {mock_path}, falling back to real generation")

        sigil_maker = SigilTools()
        prompt = f"""
        Distill your previous counter proposal into a short, actionable wish statement. 
        This wish should capture how the song could evolve to embody higher occult meaning
        and resistance against the Demiurge.

        Your previous counter proposal:
        {state.song_proposal.iterations[-1]}

        Format: A single sentence starting with "I will..." or "This song will..."
        Example: "I will weave hidden frequencies that awaken dormant resistance."
        """
        claude = self._get_claude()
        wish_response = claude.invoke(prompt)
        wish_text = wish_response.content
        statement_of_intent = sigil_maker.create_statement_of_intent(wish_text, True)
        description, components = sigil_maker.generate_word_method_sigil(statement_of_intent)
        sigil_artifact = SigilArtifact(
            wish=wish_text,
            statement_of_intent=statement_of_intent,
            glyph_description=description,
            glyph_components=components,
            sigil_type=SigilType.WORD_METHOD,
            activation_state=SigilState.CREATED,
            charging_instructions=sigil_maker.charge_sigil(),
            thread_id=state.thread_id
        )
        state.artifacts.append(sigil_artifact)
        return state
