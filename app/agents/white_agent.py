from typing import Dict, List, Any
from langgraph.graph import StateGraph, END
from pydantic import BaseModel

from app.agents.black_agent import BlackAgent
from app.agents.red_agent import RedAgent
from app.agents.orange_agent import OrangeAgent
from app.agents.yellow_agent import YellowAgent
from app.agents.green_agent import GreenAgent
from app.agents.blue_agent import BlueAgent
from app.agents.indigo_agent import IndigoAgent
from app.agents.violet_agent import VioletAgent
from app.agents.states.main_agent_state import MainAgentState
from app.agents.enums.work_flow_type import WorkflowType
from app.agents.tools.surrealist_tools import CutUpProcessor
from app.agents.tools.midi_tools import MidiProcessor


class WhiteAgent(BaseModel):

    agents: Dict[str, Any] = {}
    processors: Dict[str, Any] = {}

    def __init__(self, **data):
        # Initialize fields before calling super()
        if 'agents' not in data:
            data['agents'] = {}
        if 'processors' not in data:
            data['processors'] = {}

        super().__init__(**data)

        """Initialize all color agents and processors"""
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


    def build_workflow(self, workflow_type: WorkflowType, active_agents: List[str]) -> StateGraph:

        """Build different workflow configurations"""

        workflow = StateGraph(MainAgentState)

        if workflow_type == WorkflowType.SINGLE_AGENT:
            # Just one agent → cut-up → MIDI
            agent_name = active_agents[0]
            workflow.add_node(agent_name, self.agents[agent_name])
            workflow.add_node("cut_up", self.processors["cut_up"])
            workflow.add_node("midi", self.processors["midi"])

            workflow.set_entry_point(agent_name)
            workflow.add_edge(agent_name, "cut_up")
            workflow.add_edge("cut_up", "midi")
            workflow.add_edge("midi", END)

        elif workflow_type == WorkflowType.CHAIN:
            previous_node = None
            for agent_name in active_agents:
                workflow.add_node(agent_name, self.agents[agent_name])
                if previous_node:
                    workflow.add_edge(previous_node, agent_name)
                else:
                    workflow.set_entry_point(agent_name)
                previous_node = agent_name

            workflow.add_node("cut_up", self.processors["cut_up"])
            workflow.add_node("midi", self.processors["midi"])
            workflow.add_edge(previous_node, "cut_up")
            workflow.add_edge("cut_up", "midi")
            workflow.add_edge("midi", END)

        elif workflow_type == WorkflowType.FULL_SPECTRUM:
            # Sequential execution to avoid concurrent state updates
            workflow.add_node("black", self.agents["black"])
            workflow.set_entry_point("black")

            # Sequential chain instead of parallel to avoid concurrent updates
            workflow.add_node("red", self.agents["red"])
            workflow.add_node("orange", self.agents["orange"])
            workflow.add_node("yellow", self.agents["yellow"])
            workflow.add_node("green", self.agents["green"])
            workflow.add_node("blue", self.agents["blue"])
            workflow.add_node("indigo", self.agents["indigo"])
            workflow.add_node("violet", self.agents["violet"])

            # Create a sequential chain: black → red → orange → yellow → green → blue → indigo → violet
            workflow.add_edge("black", "red")
            workflow.add_edge("red", "orange")
            workflow.add_edge("orange", "yellow")
            workflow.add_edge("yellow", "green")
            workflow.add_edge("green", "blue")
            workflow.add_edge("blue", "indigo")
            workflow.add_edge("indigo", "violet")

            # Final processing
            workflow.add_node("cut_up", self.processors["cut_up"])
            workflow.add_node("midi", self.processors["midi"])
            workflow.add_edge("violet", "cut_up")
            workflow.add_edge("cut_up", "midi")
            workflow.add_edge("midi", END)

        return workflow.compile()
