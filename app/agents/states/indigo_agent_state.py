import uuid
from typing import Optional, Any, List

from pydantic import Field

from app.structures.agents.base_rainbow_agent_state import BaseRainbowAgentState
from app.structures.manifests.song_proposal import SongProposalIteration, SongProposal


class IndigoAgentState(BaseRainbowAgentState):
    thread_id: str = f"black_thread_{uuid.uuid4()}"
    white_proposal: Optional[SongProposalIteration] = None
    song_proposals: Optional[SongProposal] = None
    counter_proposal: Optional[SongProposalIteration] = None
    artifacts: List[Any] = Field(default_factory=list)