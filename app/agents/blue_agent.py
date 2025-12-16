import logging
import os
import time
import yaml

from abc import ABC
from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langgraph.graph import StateGraph

from app.agents.states.blue_agent_state import BlueAgentState
from app.agents.states.white_agent_state import MainAgentState
from app.structures.agents.base_rainbow_agent import BaseRainbowAgent
from app.structures.concepts.rainbow_table_color import the_rainbow_table_colors
from app.structures.manifests.song_proposal import SongProposalIteration
from app.util.manifest_loader import get_my_reference_proposals

load_dotenv()


class BlueAgent(BaseRainbowAgent, ABC):
    """Alternate Life Branching - Biographical alternate histories"""

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

    def __call__(self, state: MainAgentState) -> MainAgentState:
        current_proposal = state.song_proposals.iterations[-1]
        blue_state = BlueAgentState(
            thread_id=state.thread_id,
            song_proposals=state.song_proposals,
            white_proposal=current_proposal,
            counter_proposal=None,
            artifacts=[],
            biographical_timeline=None,
            forgotten_periods=[],
            selected_period=None,
            alternate_history=None,
            tape_label=None,
        )
        blue_graph = self.create_graph()
        compiled_graph = blue_graph.compile()
        result = compiled_graph.invoke(blue_state.model_dump())
        if isinstance(result, BlueAgentState):
            final_state = result
        elif isinstance(result, dict):
            final_state = BlueAgentState(**result)
        else:
            raise TypeError(f"Unexpected result type: {type(result)}")
        if final_state.counter_proposal:
            state.song_proposals.iterations.append(final_state.counter_proposal)
        if final_state.artifacts:
            state.artifacts = final_state.artifacts
        return state

    def create_graph(self) -> StateGraph:
        work_flow = StateGraph(BlueAgentState)
        # Nodes
        work_flow.add_node(
            "generate_alternate_song_spec", self.generate_alternate_song_spec
        )
        # Edges

        return work_flow

    def load_biographical_data(self, state: BlueAgentState) -> BlueAgentState:
        pass

    def select_year(self, state: BlueAgentState) -> BlueAgentState:
        pass

    def evaluate_timeline_frailty(self, state: BlueAgentState) -> BlueAgentState:
        pass

    def route_after_evaluate_timeline_frailty(self, state: BlueAgentState) -> str:
        pass

    def generate_alternate_history(self, state: BlueAgentState) -> BlueAgentState:
        pass

    def extract_musical_parameters(self, state: BlueAgentState) -> BlueAgentState:
        pass

    def generate_tape_label(self, state: BlueAgentState) -> BlueAgentState:
        pass

    def generate_alternate_song_spec(self, state: BlueAgentState) -> BlueAgentState:
        mock_mode = os.getenv("MOCK_MODE", "false").lower() == "true"
        block_mode = os.getenv("BLOCK_MODE", "false").lower() == "true"
        if mock_mode:
            try:
                with open(
                    f"{os.getenv('AGENT_MOCK_DATA_PATH')}/blue_counter_proposal_mock.yml",
                    "r",
                ) as f:
                    data = yaml.safe_load(f)
                counter_proposal = SongProposalIteration(**data)
                state.counter_proposal = counter_proposal
                return state
            except Exception as e:
                error_msg = f"Failed to read mock counter proposal: {e!s}"
                logging.error(error_msg)
                if block_mode:
                    raise Exception(error_msg)
            return state
        else:

            prompt = f"""
           
Current song proposal:
{state.white_proposal}

Reference works in this artist's style paying close attention to 'concept' property:
{get_my_reference_proposals('B')}

In your counter proposal your 'rainbow_color' property should always be:
{the_rainbow_table_colors['B']}


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
                timestamp = int(time.time() * 1000)
                logging.error(f"Anthropic model call failed: {e!s}")
                counter_proposal = SongProposalIteration(
                    iteration_id=f"fallback_error_{timestamp}",
                    bpm=110,
                    tempo="3/4",
                    key="G Major",
                    rainbow_color="blue",
                    title="Fallback: Blue Song",
                    mood=["melancholic"],
                    genres=["folk rock"],
                    concept="Fallback stub because Anthropic model unavailable",
                )

            state.counter_proposal = counter_proposal
            return state
