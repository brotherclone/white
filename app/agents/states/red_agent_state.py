import uuid
from typing import Optional, List, Any

from pydantic import Field

from app.agents.models.book_artifact import BookArtifact
from app.agents.models.book_data import BookData
from app.agents.states.base_rainbow_agent_state import BaseRainbowAgentState
from app.structures.manifests.song_proposal import SongProposalIteration, SongProposal


class RedAgentState(BaseRainbowAgentState):

    thread_id: str = f"red_thread_{uuid.uuid4()}"
    black_to_white_proposal: Optional[SongProposalIteration] = None
    song_proposals: Optional[SongProposal] = None
    counter_proposal: Optional[SongProposalIteration] = None
    main_generated_book: BookArtifact | None = None
    current_reaction_book: BookData | None = None
    artifacts: List[Any] = Field(default_factory=list)
    should_respond_with_reaction_book: bool = False
    should_create_book: bool = True
    reaction_level: int = 0

    def __init__(self, **data):
        super().__init__(**data)