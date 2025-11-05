from typing import Any, Dict, List, Optional

from pydantic import Field

from app.structures.agents.base_rainbow_agent_state import \
    BaseRainbowAgentState


class BlackAgentState(BaseRainbowAgentState):
    """
    State for Black Agent workflow.

    Fields:
    - white_proposal: The specific iteration Black is responding to
    - song_proposals: Full negotiation history for context
    - counter_proposal: Black's generated response
    - artifacts: Generated sigils, EVPs, etc.
    """

    human_instructions: Optional[str] = ""
    pending_human_tasks: List[Dict[str, Any]] = Field(default_factory=list)
    awaiting_human_action: bool = False
    should_update_proposal_with_evp: bool = False
    should_update_proposal_with_sigil: bool = False

    def __init__(self, **data):
        super().__init__(**data)
