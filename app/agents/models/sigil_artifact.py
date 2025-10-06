from datetime import datetime
from typing import List

from app.agents.enums.sigil_state import SigilState
from app.agents.enums.sigil_type import SigilType
from app.agents.models.base_chain_artifact import ChainArtifact


class SigilArtifact(ChainArtifact):

    """Record of a created sigil for the Black Agent's paranoid tracking"""
    thread_id: str
    wish: str
    statement_of_intent: str
    sigil_type: SigilType
    glyph_description: str
    glyph_components: List[str] | None = None
    activation_state: SigilState
    charging_instructions: str
