import uuid
from typing import Optional, List, Dict, Any
from pydantic import Field

from app.agents.models.evp_artifact import EVPArtifact
from app.agents.models.sigil_artifact import SigilArtifact
from app.agents.states.base_rainbow_agent_state import BaseRainbowAgentState
from app.structures.manifests.song_proposal import SongProposal, SongProposalIteration


class BlackAgentState(BaseRainbowAgentState):
    """
    State for Black Agent workflow.

    Fields:
    - white_proposal: The specific iteration Black is responding to
    - song_proposals: Full negotiation history for context
    - counter_proposal: Black's generated response
    - artifacts: Generated sigils, EVPs, etc.
    """

    thread_id: str = f"black_thread_{uuid.uuid4()}"
    white_proposal: Optional[SongProposalIteration] = None
    song_proposals: Optional[SongProposal] = None
    counter_proposal: Optional[SongProposalIteration] = None
    artifacts: List[Any] = Field(default_factory=list)
    evp_artifact: Optional[EVPArtifact] = None
    sigil_artifact: Optional[SigilArtifact] = None
    human_instructions: Optional[str] = ""
    pending_human_tasks: List[Dict[str, Any]] = Field(default_factory=list)
    awaiting_human_action: bool = False

    def __init__(self, **data):
        super().__init__(**data)