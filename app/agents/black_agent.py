from abc import ABC

from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langgraph.constants import START, END
from langgraph.graph import StateGraph


from app.agents.base_rainbow_agent import BaseRainbowAgent
from app.agents.states.black_agent_state import BlackAgentState
from app.agents.states.main_agent_state import MainAgentState
from app.agents.tools.magick_tools import SigilTools
from app.structures.concepts.rainbow_table_color import RainbowTableColor
from app.util.manifest_loader import get_my_reference_proposals

load_dotenv()

class BlackAgent(BaseRainbowAgent, ABC):

    """EVP/Sigil Generator - Audio analysis that hallucinates messages"""

    def __init__(self, **data):
        if 'settings' not in data or data['settings'] is None:
            from app.agents.models.agent_settings import AgentSettings
            data['settings'] = AgentSettings()

        super().__init__(**data)

        if self.settings is None:
            from app.agents.models.agent_settings import AgentSettings
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

        return black_workflow

    def critique_proposal(self):

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

        initial_prompt = f"""
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
        {self.state_graph.song_proposal[-1]}
        Here are The Rainbow Table song manifests for black, The Conjurer's Thread:
        {get_my_reference_proposals('Z')}
        If you find the proposal lacking, use the following tools at your disposal to create a counter proposal:
            1. generate_sigil: Create a sigil that embodies the essence of the song proposal. The sigil should be a
               visual representation of the song's themes and messages. You'll provide instructions to the artist to imbue
               it with magical properties.
            2. guide_by_voices: Use EVP (Electronic Voice Phenomenon) techniques to capture messages from the spirit world.
               These messages can provide insights and inspiration for the song proposal.
        After using these tools, artifacts will be created that can be used to enhance the song proposal.
        """

    def generate_sigil(self):
        pass

    def generate_imbuing_instructions(self):
        pass

    def guide_by_voices(self):
        pass

    def generate_document(self):
        raise NotImplementedError("Subclasses must implement generate_document method")

    def generate_alternate_song_spec(self):
        raise NotImplementedError("Subclasses must implement generate_alternate_song_spec method")

    def contribute(self, **kwargs):
        raise NotImplementedError("Subclasses must implement contribute method")

