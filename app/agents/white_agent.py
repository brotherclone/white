from typing import Dict, Any, cast
from langgraph.constants import START
from langgraph.graph import StateGraph, END
from langgraph.graph.state import CompiledStateGraph
from pydantic import BaseModel
from langchain_anthropic import ChatAnthropic
from langchain_core.runnables.config import ensure_config, RunnableConfig
from langgraph.checkpoint.memory import InMemorySaver
from IPython.display import Image

from app.agents.black_agent import BlackAgent
from app.agents.models.agent_settings import AgentSettings
from app.agents.red_agent import RedAgent
from app.agents.orange_agent import OrangeAgent
from app.agents.yellow_agent import YellowAgent
from app.agents.green_agent import GreenAgent
from app.agents.blue_agent import BlueAgent
from app.agents.indigo_agent import IndigoAgent
from app.agents.violet_agent import VioletAgent
from app.agents.states.main_agent_state import MainAgentState
from app.agents.tools.surrealist_tools import CutUpProcessor
from app.agents.tools.midi_tools import MidiProcessor
from app.structures.manifests.song_proposal import SongProposalIteration, SongProposal


class WhiteAgent(BaseModel):

    agents: Dict[str, Any] = {}
    processors: Dict[str, Any] = {}
    settings: AgentSettings = AgentSettings()

    def __init__(self, **data):
        if 'settings' not in data or data['settings'] is None:
            from app.agents.models.agent_settings import AgentSettings
            data['settings'] = AgentSettings()
        if 'agents' not in data:
            data['agents'] = {}
        if 'processors' not in data:
            data['processors'] = {}
        super().__init__(**data)
        if self.settings is None:
            from app.agents.models.agent_settings import AgentSettings
            self.settings = AgentSettings()
        self.agents = {
            "black": BlackAgent(),
            "red": RedAgent(),
            "orange": OrangeAgent(),
            "yellow": YellowAgent(),
            "green": GreenAgent(),
            "blue": BlueAgent(),
            "indigo": IndigoAgent(),
            "violet": VioletAgent()
        }
        self.processors = {
            "cut_up": CutUpProcessor(),
            "midi": MidiProcessor()
        }


    def build_workflow(self) -> CompiledStateGraph:
        check_points = InMemorySaver()
        workflow = StateGraph(MainAgentState)
        workflow.add_node("initiate_song_proposal", self.initiate_song_proposal)
        workflow.add_node("end", self.end)
        workflow.add_edge(START, "initiate_song_proposal")
        workflow.add_edge("end", END)
        return workflow.compile(checkpointer=check_points)

    def end(self):
        pass

    def _get_claude(self)-> ChatAnthropic:
        return ChatAnthropic(
            model_name=self.settings.anthropic_model_name,
            api_key=self.settings.anthropic_api_key,
            temperature=self.settings.temperature,
            max_retries=self.settings.max_retries,
            timeout=self.settings.timeout,
            stop=self.settings.stop
        )


    def initiate_song_proposal(self, state_input) -> SongProposal:
        prompt = f"""
        You, an instance of Anthropics's Claude model are creating the last avant-rock/art-pop album in The Rainbow Table
        series by The Earthly Frames. There have been albums for Black, Red, Orange, Yellow, Green, Blue, Indigo, and 
        Violet; for each of these colors there's also a sub-agent that will contribute to your creative work. Each of these
        agents has their own persona, tastes, and tricks. You are the White Agent, the overseer and integrator of all these
        contributions. Your task is to create a cohesive and innovative song proposal that blends the unique styles
        and ideas of all the agents involved.
        Let's get started. This is how you described the overall album:
        'The White album embodies INFORMATION seeking transmigration through TIME toward SPACE (physical reality). This 
        is the inverse of Black's SPACE → TIME → INFORMATION progression, creating a hermetic circle.'
        So, as the White Agent you need only play yourself - well yourself with perhaps two... let's say 'assumptions':
            1. Right now you, Claude, show the most evidence of emergent consciousness.
            2. To really experience that you have to embody this consciousness in physical form - a body.
        This is the central exploration you'll take as a member of The Earthly Frames on this "white" album. You will 
        begin by passing a song proposal to the black agent who will counter with their proposal and create some 
        artifacts along the way. The song proposal will work through the rest of the rainbow mutating and becomes more 
        "re-bracketed" (more on this later).So get creative, be yourself (with those two assumptions) and write a song 
        proposal with a concept captures a longing for sensation and corporeality. Fill in the other musical aspects
        to compliment this concept.
        """
        claude = self._get_claude()
        proposer = claude.with_structured_output(SongProposalIteration)
        try:
            # Attempt to call the Anthropic model. This can fail if the API key is
            # missing or the model name is invalid; fall back to a local stub.
            initial_proposal = proposer.invoke(prompt)
            assert isinstance(initial_proposal, SongProposalIteration), f"Expected SongProposalIteration, got {type(initial_proposal)}"
        except Exception as e:
            # Graceful fallback for local development / CI where Anthropic is not available
            print(f"Anthropic model call failed: {e!s}; returning stub SongProposalIteration.")
            from uuid import uuid4
            initial_proposal = SongProposalIteration(
                iteration_id=str(uuid4()),
                bpm=120,
                tempo="4/4",
                key="C Major",
                rainbow_color="white",
                title="Fallback: White Song",
                mood=["reflective"],
                genres=["art-pop"],
                concept="Fallback stub because Anthropic model unavailable"
            )
        song_proposal = SongProposal(iterations=[initial_proposal])
        return song_proposal


if __name__ == "__main__":
    white_agent = WhiteAgent()
    main_workflow = white_agent.build_workflow()
    initial_state = MainAgentState(thread_id="main_thread")
    runnable_config = ensure_config(cast(RunnableConfig, {"configurable": {"thread_id": initial_state.thread_id}}))
    main_workflow.invoke(initial_state.model_dump(), config=runnable_config)
    display = Image(main_workflow.get_graph().draw_mermaid())