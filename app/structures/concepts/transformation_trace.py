from typing import List, Dict, Any
from pydantic import BaseModel, ConfigDict, Field


class TransformationTrace(BaseModel):
    """
    Records what boundaries shifted during a rebracketing operation.

    Example: "TIME/SPACE boundary shifted - past events reframed as future potentials"
    """

    agent_name: str
    iteration_id: str
    boundaries_shifted: List[str] = Field(default_factory=list)
    patterns_revealed: List[str] = Field(default_factory=list)
    semantic_resonances: Dict[str, Any] = Field(default_factory=dict)

    model_config = ConfigDict(arbitrary_types_allowed=True)
