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
from app.structures.agents.agent_settings import AgentSettings
from app.agents.red_agent import RedAgent
from app.agents.orange_agent import OrangeAgent
from app.agents.yellow_agent import YellowAgent
from app.agents.green_agent import GreenAgent
from app.agents.blue_agent import BlueAgent
from app.agents.indigo_agent import IndigoAgent
from app.agents.violet_agent import VioletAgent
from app.agents.states.white_agent_state import MainAgentState
from app.structures.concepts.white_facet_system import WhiteFacetSystem
from app.structures.manifests.song_proposal import SongProposalIteration, SongProposal
from app.agents.workflow.resume_black_workflow import resume_black_agent_workflow_with_agent

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


    def start_workflow(self, user_input: str | None = None) -> MainAgentState:
        """
        Start a new White Agent workflow from the beginning.

        Args:
            user_input: Optional user input to guide the initial proposal

        Returns:
            The final state after workflow completion (or pause)

        Example:
            >>> white = WhiteAgent()
            >>> final_state = white.start_workflow()
            >>> if final_state.workflow_paused:
            >>>     # Complete the ritual tasks, then:
            >>>     resumed_state = white.resume_workflow(final_state)
        """
        workflow = self.build_workflow()
        thread_id = str(uuid4())

        initial_state = MainAgentState(
            thread_id=thread_id,
            song_proposals=SongProposal(iterations=[]),
            artifacts=[],
            workflow_paused=False,
            ready_for_red=False
        )

        config = RunnableConfig(
            configurable={"thread_id": thread_id}
        )

        logging.info(f"🎵 Starting White Agent workflow (thread_id: {thread_id})")

        result = workflow.invoke(initial_state, config)

        # Convert dict result to MainAgentState if needed
        if isinstance(result, dict):
            final_state = MainAgentState(**result)
        else:
            final_state = result

        return final_state

    def resume_workflow(self, paused_state: MainAgentState, verify_tasks: bool = True) -> MainAgentState:
        """
        Resume a paused workflow after human action is complete.

        Args:
            paused_state: The state that was returned when workflow paused
            verify_tasks: If True, verify Todoist tasks are complete before resuming

        Returns:
            The final state after workflow completion

        Example:
            >>> # After completing ritual tasks:
            >>> white = WhiteAgent()
            >>> resumed_state = white.resume_workflow(paused_state)
        """
        if not paused_state.workflow_paused:
            logging.warning("⚠️  Workflow is not paused - nothing to resume")
            return paused_state

        logging.info(f"🔄 Resuming workflow (thread_id: {paused_state.thread_id})")

        # Resume Black Agent ritual completion
        updated_state = self.resume_after_black_agent_ritual(
            paused_state,
            verify_tasks=verify_tasks
        )

        # Continue the workflow by invoking Red Agent if ready
        if updated_state.ready_for_red:
            logging.info("▶️  Continuing to Red Agent...")
            updated_state = self.invoke_red_agent(updated_state)
            updated_state = self.process_red_agent_work(updated_state)

        # Finalize
        updated_state = self.finalize_song_proposal(updated_state)

        return updated_state

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
        workflow.add_node("finalize_song_proposal", self.finalize_song_proposal)
        workflow.add_edge(START, "initiate_song_proposal")
        workflow.add_edge("initiate_song_proposal", "invoke_black_agent")
        workflow.add_edge("invoke_black_agent", "process_black_agent_work")
        workflow.add_edge("invoke_red_agent", "process_red_agent_work")
        workflow.add_conditional_edges(
            "process_black_agent_work",
            self.route_after_black,
            {
                "red": "invoke_red_agent",
                "black": "invoke_black_agent",
                "finish": "finalize_song_proposal"
            }
        )
        workflow.add_conditional_edges(
            "process_red_agent_work",
            self.route_after_red,
            {
                "finish": "finalize_song_proposal"
            }
        )
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
            self.agents["red"] = RedAgent(settings=self.settings)
        return self.agents["red"](state)


    def initiate_song_proposal(self, state: MainAgentState) -> MainAgentState:
        mock_mode = os.getenv("MOCK_MODE", "false").lower() == "true"
        prompt, facet = WhiteFacetSystem.build_white_initial_prompt(
            user_input=None,
            use_weights=True
        )
        facet_metadata = WhiteFacetSystem.log_facet_selection(facet)
        print(f"🔍 White Agent using {facet.value.upper()} lens")
        print(f"   {facet_metadata['description']}")
        if mock_mode:
            try:
                with open(f"{os.getenv('AGENT_MOCK_DATA_PATH')}/white_initial_proposal_{facet.value}_mock.yml", "r") as f:
                    data = yaml.safe_load(f)
                    proposal = SongProposalIteration(**data)
                    if not hasattr(state, "song_proposals") or state.song_proposals is None:
                        state.song_proposals = SongProposal(iterations=[])
                    sp = self._normalize_song_proposal(state.song_proposals)
                    sp.iterations.append(proposal)
                    state.song_proposals = sp
            except Exception as e:
                print(f"Mock initial proposal not found, returning stub SongProposalIteration:{e!s}")
            return state
        claude = self._get_claude_supervisor()
        proposer = claude.with_structured_output(SongProposalIteration)
        try:
            initial_proposal = proposer.invoke(prompt)
            if isinstance(initial_proposal, dict):
                initial_proposal = SongProposalIteration(**initial_proposal)
                state.white_facet = facet
                state.white_facet_metadata = facet_metadata
                print(f"{facet}{facet_metadata} {initial_proposal}")
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
                concept="Fallback stub because Anthropic model unavailable. Fallback stub because Anthropic model unavailable. Fallback stub because Anthropic model unavailable. Fallback stub because Anthropic model unavailable."
            )
        if not hasattr(state, "song_proposals") or state.song_proposals is None:
            state.song_proposals = SongProposal(iterations=[])
        sp = self._normalize_song_proposal(state.song_proposals)
        sp.iterations.append(initial_proposal)
        state.song_proposals = sp
        return state

    def process_black_agent_work(self, state: MainAgentState) -> MainAgentState:
        # Check if workflow is paused waiting for human action
        if state.workflow_paused and state.pending_human_action:
            logging.info("⏸️  Workflow paused - waiting for human to complete ritual tasks")
            return state

        # Normalize song_proposals to object before accessing
        sp = self._normalize_song_proposal(state.song_proposals)
        black_proposal = sp.iterations[-1]
        black_artifacts = state.artifacts or []
        evp_artifacts = [a for a in black_artifacts if a.chain_artifact_type == "evp"]
        sigil_artifacts = [a for a in black_artifacts if a.chain_artifact_type == "sigil"]

        rebracketing_analysis = self._black_rebracketing_analysis(
            black_proposal, evp_artifacts, sigil_artifacts
        )
        document_synthesis = self._synthesize_document_for_red(
            rebracketing_analysis, black_proposal, black_artifacts
        )
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
            with open(f"{os.getenv('AGENT_MOCK_DATA_PATH')}/black_to_white_rebracket_analysis_mock.yml", "r") as f:
                data = yaml.safe_load(f)
                return data
        else:
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

            claude = self._get_claude_supervisor()
            response = claude.invoke(prompt)

            return response.content

    def _synthesize_document_for_red(self, rebracketed_analysis, black_proposal, artifacts):
        mock_mode = os.getenv("MOCK_MODE", "false").lower() == "true"
        if mock_mode:
            with open(f"{os.getenv('AGENT_MOCK_DATA_PATH')}/black_to_white_document_synthesis_mock.yml", "r") as f:
                data = yaml.safe_load(f)
                return data
        else:
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

            claude = self._get_claude_supervisor()
            response = claude.invoke(prompt)

            return response.content

    @staticmethod
    def route_after_black(state: MainAgentState) -> str:
        # If workflow is paused for human action, end here
        if state.workflow_paused and state.pending_human_action:
            return "finish"

        mock_mode = os.getenv("MOCK_MODE", "false").lower() == "true"
        if mock_mode:
            return "red"
        if state.ready_for_red:
            return "red"
        return "finish"

    @staticmethod
    def route_after_red(state: MainAgentState) -> str:
        return "finish"

    @staticmethod
    def finalize_song_proposal(state: MainAgentState) -> MainAgentState:
        # If workflow is paused, provide instructions and don't save
        if state.workflow_paused and state.pending_human_action:
            pending = state.pending_human_action
            logging.info("\n" + "="*60)
            logging.info("⏸️  WORKFLOW PAUSED - HUMAN ACTION REQUIRED")
            logging.info("="*60)
            logging.info(f"Agent: {pending.get('agent', 'unknown')}")
            logging.info(f"Reason: {state.pause_reason}")
            logging.info(f"\nInstructions:\n{pending.get('instructions', 'No instructions')}")

            tasks = pending.get('tasks', [])
            if tasks:
                logging.info(f"\nPending Tasks ({len(tasks)}):")
                for task in tasks:
                    logging.info(f"  - {task.get('type', 'unknown')}: {task.get('task_url', 'No URL')}")

            logging.info("\nTo resume after completing tasks:")
            logging.info("  from app.agents.white_agent import WhiteAgent")
            logging.info(f"  state = WhiteAgent.resume_after_black_agent_ritual(state)")
            logging.info("="*60)
            return state

        # Normal completion - save proposals
        state.song_proposals.save_all_proposals()
        logging.info("✓ Song proposals saved")
        return state

    def resume_after_black_agent_ritual(self, paused_state: MainAgentState, verify_tasks: bool = True) -> MainAgentState:
        """
        Resume the White Agent workflow after Black Agent ritual tasks are completed.

        Args:
            paused_state: The MainAgentState that was paused waiting for human action
            verify_tasks: If True, verify all Todoist tasks are complete before resuming

        Returns:
            Updated MainAgentState after Black Agent workflow completion
        """
        # Use the module-level resume_black_agent_workflow (imported at top of file)
        # This allows tests to patch `resume_black_agent_workflow` in this module.

        if not paused_state.workflow_paused:
            logging.warning("Workflow is not paused - nothing to resume")
            return paused_state

        if not paused_state.pending_human_action:
            logging.warning("No pending human action found")
            return paused_state

        pending = paused_state.pending_human_action
        if pending.get('agent') != 'black':
            logging.warning(f"Cannot resume - pending action is for agent: {pending.get('agent')}")
            return paused_state

        black_config = pending.get('black_config')
        if not black_config:
            logging.error("No black_config found in pending_human_action")
            return paused_state

        # Use self to access the BlackAgent instance
        black_agent = self.agents.get('black')
        if not black_agent:
            logging.error("Black agent not found in white_agent instance")
            return paused_state

        logging.info("🔄 Resuming Black Agent workflow...")

        try:
            # Resume the Black Agent workflow using the existing black_agent instance
            # This ensures the checkpointer has the saved state
            final_black_state = resume_black_agent_workflow_with_agent(
                black_agent,
                black_config,
                verify_tasks=verify_tasks
            )

            # Update the main state with Black Agent results
            paused_state.workflow_paused = False
            paused_state.pause_reason = None
            paused_state.pending_human_action = None

            if final_black_state.get('counter_proposal'):
                paused_state.song_proposals.iterations.append(final_black_state['counter_proposal'])

            if final_black_state.get('artifacts'):
                paused_state.artifacts = final_black_state['artifacts']

            logging.info("✓ Black Agent workflow resumed and completed")

            # Now continue with the rest of the White Agent workflow
            # Process the Black Agent's work
            artifacts = getattr(paused_state, 'artifacts', []) or []
            evp_artifacts = [a for a in artifacts if getattr(a, 'chain_artifact_type', None) == "evp"]
            sigil_artifacts = [a for a in artifacts if getattr(a, 'chain_artifact_type', None) == "sigil"]

            black_proposal = paused_state.song_proposals.iterations[-1]

            paused_state.rebracketing_analysis = self._black_rebracketing_analysis(
                black_proposal, evp_artifacts, sigil_artifacts
            )
            paused_state.document_synthesis = self._synthesize_document_for_red(
                paused_state.rebracketing_analysis, black_proposal, artifacts
            )
            paused_state.ready_for_red = True

            logging.info("✓ Processed Black Agent work - ready for Red Agent")

            return paused_state

        except Exception as e:
            logging.error(f"Failed to resume Black Agent workflow: {e}")
            raise
