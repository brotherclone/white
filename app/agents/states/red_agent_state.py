import uuid
from typing import Optional, List, Any, Dict

from pydantic import Field

from app.agents.states.base_rainbow_agent_state import BaseRainbowAgentState
from app.structures.manifests.song_proposal import SongProposalIteration, SongProposal


class RedAgentState(BaseRainbowAgentState):

    thread_id: str = f"red_thread_{uuid.uuid4()}"
    white_proposal: Optional[SongProposalIteration] = None
    song_proposals: Optional[SongProposal] = None
    counter_proposal: Optional[SongProposalIteration] = None
    artifacts: List[Any] = Field(default_factory=list)
    human_instructions: Optional[str] = ""
    pending_human_tasks: List[Dict[str, Any]] = Field(default_factory=list)
    awaiting_human_action: bool = False

    def __init__(self, **data):
        super().__init__(**data)