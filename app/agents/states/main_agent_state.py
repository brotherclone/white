from pydantic import Field
from typing import Any, Dict, List, Optional
from app.agents.states.base_rainbow_agent_state import BaseRainbowAgentState
from app.structures.manifests.song_proposal import SongProposal


class MainAgentState(BaseRainbowAgentState):

    """Central state that flows through all color agents"""

    thread_id: str = "main_thread"
    song_proposal: SongProposal | None = None
    black_content: Dict[str, Any] = Field(default_factory=dict)
    black_proposal: SongProposal | None = None
    red_content: Dict[str, Any] = Field(default_factory=dict)
    red_proposal: SongProposal | None = None
    orange_content: Dict[str, Any] = Field(default_factory=dict)
    orange_proposal: SongProposal | None = None
    yellow_content: Dict[str, Any] = Field(default_factory=dict)
    yellow_proposal: SongProposal | None = None
    green_content: Dict[str, Any] = Field(default_factory=dict)
    green_proposal: SongProposal | None = None
    blue_content: Dict[str, Any] = Field(default_factory=dict)
    blue_proposal: SongProposal | None = None
    indigo_content: Dict[str, Any] = Field(default_factory=dict)
    indigo_proposal: SongProposal | None = None
    violet_content: Dict[str, Any] = Field(default_factory=dict)
    violet_proposal: SongProposal | None = None

    cut_up_fragments: List[str] = Field(default_factory=list)
    midi_data: Optional[Dict] = None

    # Workflow control
    active_agents: List[str] = Field(default_factory=list)

    def __init__(self, **data):
        super().__init__(**data)
