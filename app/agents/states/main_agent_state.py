from pydantic import Field
from typing import Any, Dict, List, Optional
from app.agents.states.base_rainbow_agent_state import BaseRainbowAgentState
from app.structures.manifests.song_proposal import SongProposal


class MainAgentState(BaseRainbowAgentState):

    """Central state that flows through all color agents"""

    thread_id: str = "main_thread"
    song_proposal: SongProposal | None = None
    active_agents: List[str] = Field(default_factory=list)

    def __init__(self, **data):
        super().__init__(**data)
