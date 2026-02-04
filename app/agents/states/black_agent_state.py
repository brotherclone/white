from typing import Annotated
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

    should_update_proposal_with_evp: Annotated[
        bool, lambda x, y: y if y is not None else x
    ] = False

    def __init__(self, **data):
        super().__init__(**data)
