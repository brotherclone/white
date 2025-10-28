from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field, ConfigDict

from app.structures.manifests.song_proposal import SongProposal
from app.agents.enums.white_facet import WhiteFacet

class MainAgentState(BaseModel):

    """
    Main state for White Agent (supervisor) coordinating all rainbow agents.
    """

    thread_id: str
    song_proposals: SongProposal = Field(default_factory=SongProposal)
    artifacts: List[Any] = []
    workflow_paused: bool = False
    pause_reason: Optional[str] = None
    pending_human_action: Optional[Dict[str, Any]] = None
    rebracketing_analysis: Optional[str] = None
    document_synthesis: Optional[str] = None
    ready_for_red: bool = False
    white_facet: WhiteFacet | None = None
    white_facet_metadata: str | Any = None

    """
    Structure when workflow is paused:
    {
        "agent": "black" | "red" | etc.,
        "action": "sigil_charging" | "emotional_review" | etc.,
        "instructions": "Human-readable instructions",
        "pending_tasks": [{"task_id": "...", "task_url": "...", ...}],
        "black_config": {...},  # LangGraph config for resuming sub-agent
        "resume_instructions": "How to resume after completion"
    }
    """
    model_config = ConfigDict(arbitrary_types_allowed=True)
