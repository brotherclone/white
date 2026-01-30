import os
from typing import List, Optional, TYPE_CHECKING

from dotenv import load_dotenv
from langsmith import traceable

if TYPE_CHECKING:
    from app.agents.states.white_agent_state import MainAgentState

load_dotenv()

os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "white-album-debug"


@traceable(run_type="chain", name="White Agent Full Concept Workflow")
def run_white_agent_workflow(
    concept: Optional[str] = None,
    enabled_agents: Optional[List[str]] = None,
    stop_after_agent: Optional[str] = None,
) -> "MainAgentState":
    """
    Run the White Agent workflow with full LangSmith tracing.

    Args:
        concept: Optional initial song concept
        enabled_agents: Optional list of agents to enable
        stop_after_agent: Optional agent to stop after

    Returns:
        Final MainAgentState after workflow completion
    """
    # Lazy import to avoid circular dependency
    from app.agents.white_agent import WhiteAgent

    white = WhiteAgent()
    return white.start_workflow(
        user_input=concept,
        enabled_agents=enabled_agents,
        stop_after_agent=stop_after_agent,
    )
