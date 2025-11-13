from typing import Optional

from pydantic import Field

from app.structures.artifacts.base_artifact import ChainArtifact
from app.structures.enums.symbolic_object_category import SymbolicObjectCategory


class SymbolicObjectArtifact(ChainArtifact):
    """Symbolic object that has emerged from the nostalgia of Rows Bud, the orange agent"""

    thread_id: str = Field(..., description="Thread identifier for the artifact.")
    symbolic_object_category: SymbolicObjectCategory = Field(
        description="""Category of the object: 
    CIRCULAR_TIME - Clocks, calendars, loops, temporal markers (Nash's 182 BPM clock)
    INFORMATION_ARTIFACTS - Newspapers, broadcasts, transmissions, EMORYs, recordings
    LIMINAL_OBJECTS - Doorways, thresholds, portals, Pine Barrens gateways
    PSYCHOGEOGRAPHIC - Maps, coordinates, dimensional markers, location-based objects""",
        default=SymbolicObjectCategory.INFORMATION_ARTIFACTS,
    )
    name: Optional[str] = Field(default=None, description="A name for the object.")
    description: Optional[str] = Field(
        default=None, description="A detailed description of the object."
    )

    def __init__(self, **data):
        super().__init__(**data)
