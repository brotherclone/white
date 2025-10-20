import logging
import yaml
import os

from typing import Dict, Any, cast
from langgraph.constants import START
from langgraph.graph import StateGraph, END
from langgraph.graph.state import CompiledStateGraph
from pydantic import BaseModel
from langchain_anthropic import ChatAnthropic
from langchain_core.runnables.config import ensure_config, RunnableConfig
from langgraph.checkpoint.memory import InMemorySaver
from uuid import uuid4

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
from app.structures.manifests.song_proposal import SongProposalIteration, SongProposal
from app.agents.workflow.resume_black_workflow import resume_black_agent_workflow

logging.basicConfig(level=logging.INFO)

class WhiteAgent(BaseModel):

    agents: Dict[str, Any] = {}
    processors: Dict[str, Any] = {}
    settings: AgentSettings = AgentSettings()
    song_proposal: SongProposal = SongProposal(iterations=[])

    def __init__(self, **data):
        if 'settings' not in data or data['settings'] is None:
            data['settings'] = AgentSettings()
        if 'agents' not in data:
            data['agents'] = {}
        if 'processors' not in data:
            data['processors'] = {}
        super().__init__(**data)
        if self.settings is None:
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


    def build_workflow(self) -> CompiledStateGraph:
        check_points = InMemorySaver()
        workflow = StateGraph(MainAgentState)
        workflow.add_node("initiate_song_proposal", self.initiate_song_proposal)
        workflow.add_node("invoke_black_agent", self.invoke_black_agent)
        workflow.add_node("process_black_agent_work", self.process_black_agent_work)
        workflow.add_node("invoke_red_agent", self.invoke_red_agent)
        workflow.add_node("process_red_agent_work", self.process_red_agent_work)
        # workflow.add_node("invoke_orange_agent", self.invoke_orange_agent)
        # workflow.add_node("invoke_yellow_agent", self.invoke_yellow_agent)
        # workflow.add_node("invoke_green_agent", self.invoke_green_agent)
        # workflow.add_node("invoke_blue_agent", self.invoke_blue_agent)
        # workflow.add_node("invoke_indigo_agent", self.invoke_indigo_agent)
        # workflow.add_node("invoke_violet_agent", self.invoke_violet_agent)



        workflow.add_edge(START, "initiate_song_proposal")
        workflow.add_edge("initiate_song_proposal", "invoke_black_agent")
        workflow.add_edge("invoke_black_agent", "process_black_agent_work")
        workflow.add_conditional_edges(
            "process_black_agent_work",
            self.route_after_black,
            {
                "red": "invoke_red_agent",
                "black": "invoke_black_agent",
                "finish": END
            }
        )
        workflow.add_edge("invoke_red_agent", "process_red_agent_work")
        # Tmp
        workflow.add_edge("process_red_agent_work", "finalize_song_proposal")

        workflow.add_edge("finalize_song_proposal", END)

        return workflow.compile(checkpointer=check_points)

    def _get_claude_supervisor(self)-> ChatAnthropic:
        return ChatAnthropic(
            model_name=self.settings.anthropic_model_name,
            api_key=self.settings.anthropic_api_key,
            temperature=self.settings.temperature,
            max_retries=self.settings.max_retries,
            timeout=self.settings.timeout,
            stop=self.settings.stop
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

    def invoke_black_agent(self, state: MainAgentState) -> MainAgentState:
        """Invoke Black Agent to generate counter-proposal"""
        if "black" not in self.agents:
            self.agents["black"] = BlackAgent(settings=self.settings)
        return self.agents["black"](state)

    def invoke_red_agent(self, state: MainAgentState) -> MainAgentState:
        """Invoke Red Agent with the first synthesized proposal from Black Agent"""
        if "red" not in self.agents:
            self.agents["red"] = BlackAgent(settings=self.settings)
        return self.agents["red"](state)

    @staticmethod
    def resume_after_black_agent_ritual(
            state: MainAgentState
    ) -> MainAgentState:
        """
        Resume the White Agent workflow after Black Agent's ritual tasks are complete.
        This should be called after human marks Todoist tasks complete.

        Returns:
            Updated state with Black Agent's counter-proposal integrated
        """

        if not state.pending_human_action:
            logging.warning("No pending human action to resume from")
            return state

        pending = state.pending_human_action

        if pending.get('agent') != 'black':
            logging.warning(f"Pending action is for {pending.get('agent')}, not black agent")
            return state
        black_config = pending.get('black_config')
        if not black_config:
            raise ValueError("No black_config found in pending_human_action")
        final_black_state = resume_black_agent_workflow(black_config, verify_tasks=True)
        counter_proposal = final_black_state.get('counter_proposal')
        if counter_proposal:
            if not state.song_proposals:
                state.song_proposals = SongProposal(iterations=[])
            state.song_proposals.iterations.append(counter_proposal)
            logging.info(f"✓ Integrated Black Agent counter-proposal: {counter_proposal.title}")
        else:
            logging.warning("Black Agent did not produce counter-proposal")
        state.pending_human_action = None
        state.workflow_paused = False
        state.pause_reason = None
        return state

    # ToDo: Add 'lenses' for variation
    def initiate_song_proposal(self, state: MainAgentState) -> MainAgentState:
        mock_mode = os.getenv("MOCK_MODE", "false").lower() == "true"
        if mock_mode:
            with open("/Volumes/LucidNonsense/White/app/agents/mocks/white_initial_proposal.mock.yml", "r") as f:
                data = yaml.safe_load(f)
                proposal = SongProposalIteration(**data)
                if not hasattr(state, "song_proposals") or state.song_proposals is None:
                    state.song_proposals = SongProposal(iterations=[]).model_dump()
                sp = self._normalize_song_proposal(state.song_proposals)
                sp.iterations.append(proposal)
                state.song_proposals = sp.model_dump()
            return state
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
        claude = self._get_claude_supervisor()
        proposer = claude.with_structured_output(SongProposalIteration)
        try:
            initial_proposal = proposer.invoke(prompt)
            if isinstance(initial_proposal, dict):
                initial_proposal = SongProposalIteration(**initial_proposal)
            assert isinstance(initial_proposal, SongProposalIteration), f"Expected SongProposalIteration, got {type(initial_proposal)}"
        except Exception as e:
            print(f"Anthropic model call failed: {e!s}; returning stub SongProposalIteration.")
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
        if not hasattr(state, "song_proposals") or state.song_proposals is None:
            state.song_proposals = SongProposal(iterations=[]).model_dump()
        sp = self._normalize_song_proposal(state.song_proposals)
        sp.iterations.append(initial_proposal)
        state.song_proposals = sp.model_dump()
        return state

    def process_black_agent_work(self, state: MainAgentState) -> MainAgentState:


        mock_mode = os.getenv("MOCK_MODE", "false").lower() == "true"
        if mock_mode:
            # ToDo: Add mock data
            pass

        black_proposal = state.song_proposals.iterations[-1]
        black_artifacts = state.artifacts or []
        evp_artifacts = [a for a in black_artifacts if a.chain_artifact_type == "evp"]
        sigil_artifacts = [a for a in black_artifacts if a.chain_artifact_type == "sigil"]

        rebracketing_analysis = self._black_rebracketing_analysis(
            black_proposal, evp_artifacts, sigil_artifacts
        )
        document_synthesis = self._synthesize_document_for_red(
            rebracketing_analysis, black_proposal, black_artifacts
        )
        # ToDo: Modify state for these
        state.rebracketing_analysis = rebracketing_analysis
        state.document_synthesis = document_synthesis
        state.ready_for_red = True

        return state

    def process_red_agent_work(self, state: MainAgentState) -> MainAgentState:
        # ToDo: Add Red Agent work
        return state

    def _black_rebracketing_analysis(self, proposal, evp_artifacts, sigil_artifacts)-> str:
        mock_mode = os.getenv("MOCK_MODE", "false").lower() == "true"
        if mock_mode:
            # ToDo: Add mock data
            pass

        prompt = f"""
            You are the White Agent performing a REBRACKETING operation.

            You have received these artifacts from Black Agent:

            **Counter-proposal:**
            {proposal}

            **EVP Transcript:** 
            {evp_artifacts[0].transcript if evp_artifacts else "None"}

            **Sigil Status:**
            {sigil_artifacts[0].activation_state if sigil_artifacts else "None"}

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

        claude = self._get_claude()
        response = claude.invoke(prompt)

        return response.content

    def _synthesize_document_for_red(self, rebracketed_analysis, black_proposal, artifacts):
        mock_mode = os.getenv("MOCK_MODE", "false").lower() == "true"
        if mock_mode:
            # ToDo: Add mock data
            pass

        prompt = f"""
            You are the White Agent creating a SYNTHESIZED DOCUMENT for Red Agent.

            **Your Rebracketed Analysis:**
            {rebracketed_analysis}

            **Original Black Counter-Proposal:**
            {black_proposal}

            **Artifacts Present:**
            {len(artifacts)} artifacts (EVP, sigil, etc.)

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

        claude = self._get_claude()
        response = claude.invoke(prompt)

        return response.content

    @staticmethod
    def route_after_black(state: MainAgentState) -> str:
        mock_mode = os.getenv("MOCK_MODE", "false").lower() == "true"
        if mock_mode:
            # ToDo: Add mock data
            pass
        # ToDo: Add checks for red, black, finish
        if state.ready_for_red:
            return "red"
            # return "black"
        return "finish"




if __name__ == "__main__":
    white_agent = WhiteAgent(settings=AgentSettings())
    main_workflow = white_agent.build_workflow()
    initial_state = MainAgentState(thread_id="main_thread")
    runnable_config = ensure_config(cast(RunnableConfig, {"configurable": {"thread_id": initial_state.thread_id}}))
    main_workflow.invoke(initial_state.model_dump(), config=runnable_config)
