from typing import Any, Dict, List, Optional, Annotated
from operator import add
from pydantic import Field
from app.structures.agents.base_rainbow_agent_state import BaseRainbowAgentState


class BlackAgentState(BaseRainbowAgentState):
    """
    State for Black Agent workflow.

    Fields:
    - white_proposal: The specific iteration Black is responding to
    - song_proposals: Full negotiation history for context
    - counter_proposal: Black's generated response
    - artifacts: Generated sigils, EVPs, etc.
    """

    human_instructions: Annotated[Optional[str], lambda x, y: y or x] = ""
    pending_human_tasks: Annotated[List[Dict[str, Any]], add] = Field(
        default_factory=list
    )
    awaiting_human_action: Annotated[bool, lambda x, y: y if y is not None else x] = (
        False
    )
    should_update_proposal_with_evp: Annotated[
        bool, lambda x, y: y if y is not None else x
    ] = False
    should_update_proposal_with_sigil: Annotated[
        bool, lambda x, y: y if y is not None else x
    ] = False

    def __init__(self, **data):
        super().__init__(**data)
