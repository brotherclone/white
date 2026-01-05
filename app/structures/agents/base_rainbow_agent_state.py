from typing import Any, List, Optional, Annotated
from operator import add

from langchain_core.messages import BaseMessage
from pydantic import BaseModel, Field

from app.structures.manifests.song_proposal import SongProposal, SongProposalIteration


class BaseRainbowAgentState(BaseModel):
    """
    Base state for Rainbow Table agents.

    All fields are Annotated to allow multiple nodes to write concurrently in LangGraph.
    - Single-value fields use `lambda x, y: y or x` (take new value if present, else keep old)
    - List fields use `add` operator (concatenate lists)
    """

    # Scalar fields - use "take last non-None" reducer
    session_id: Annotated[Optional[str], lambda x, y: y or x] = None
    timestamp: Annotated[Optional[str], lambda x, y: y or x] = None
    thread_id: Annotated[Optional[str], lambda x, y: y or x] = Field(
        default=None, description="Unique ID of the thread."
    )
    white_proposal: Annotated[Optional[SongProposalIteration], lambda x, y: y or x] = (
        None
    )
    song_proposals: Annotated[Optional[SongProposal], lambda x, y: y or x] = None
    counter_proposal: Annotated[
        Optional[SongProposalIteration], lambda x, y: y or x
    ] = None
    messages: Annotated[List[BaseMessage], add] = Field(default_factory=list)
    artifacts: Annotated[List[Any], add] = Field(default_factory=list)
    skipped_nodes: Annotated[List[str], add] = Field(default_factory=list)

    def __init__(self, **data):
        super().__init__(**data)
