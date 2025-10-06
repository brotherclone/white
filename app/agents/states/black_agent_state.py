import uuid
from typing import Optional, List, Dict, Any

from app.agents.models.evp_artifact import EVPArtifact
from app.agents.models.sigil_artifact import SigilArtifact
from app.agents.states.base_rainbow_agent_state import BaseRainbowAgentState
from app.structures.manifests.song_proposal import SongProposal


class BlackAgentState(BaseRainbowAgentState):

    thread_id: str = f"black_thread_{uuid.uuid4()}"
    song_proposal: SongProposal | None = None
    evp_artifact: EVPArtifact | None = None
    sigil_artifact: SigilArtifact | None = None
    human_instructions: Optional[str]
    pending_human_tasks: List[Dict[str, Any]]
    awaiting_human_action: bool

    def __init__(self, **data):
        super().__init__(**data)
