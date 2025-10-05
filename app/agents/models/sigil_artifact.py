from pydantic import BaseModel
from typing import Optional, List
from datetime import datetime

from app.agents.enums.gnosis_method import GnosisMethod
from app.agents.enums.sigil_type import SigilType
from app.agents.models.base_chain_artifact import ChainArtifact


class SigilArtifact(ChainArtifact):
    """Record of a created sigil for the Black Agent's paranoid tracking"""
    # sigil_id: str
    # original_intent: str  # Will be encrypted/hashed for "forgetting"
    # creation_timestamp: datetime
    # charging_method: GnosisMethod
    # sigil_type: SigilType
    # glyph_data: str  # Base64 encoded image or text representation - Then shouldn't be byte data?
    # activation_state: str  # "charged", "dormant", "forgotten", "manifested" - ENUMs
    # destruction_method: Optional[str] = None