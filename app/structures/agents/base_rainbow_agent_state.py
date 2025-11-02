import uuid
from typing import List, Optional, Any
from langchain_core.messages import BaseMessage
from pydantic import BaseModel, Field

from app.structures.manifests.song_proposal import SongProposalIteration, SongProposal


class BaseRainbowAgentState(BaseModel):

    session_id: str | None = None
    timestamp: str | None = None
    messages: List[BaseMessage] = Field(default_factory=list)
    thread_id: str = f"thread_{uuid.uuid4()}"
    white_proposal: Optional[SongProposalIteration] = None
    song_proposals: Optional[SongProposal] = None
    counter_proposal: Optional[SongProposalIteration] = None
    artifacts: List[Any] = Field(default_factory=list)

    def __init__(self, **data):
        super().__init__(**data)

