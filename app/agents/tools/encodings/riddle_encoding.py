from typing import List, Optional
from pydantic import Field

from app.structures.concepts.infranym_text_encoding import InfranymTextEncoding
from app.structures.enums.infranym_method import InfranymMethod


class RiddleEncoding(InfranymTextEncoding):
    """
    Riddle-specific encoding where the answer is the secret.

    Example:
        Secret: "TEMPORAL"
        Riddle: "I flow but have no river / I bend but never break /
                 The past and future quiver / When I'm at stake..."
    """

    method: InfranymMethod = InfranymMethod.RIDDLE_POEM
    riddle_text: str = Field(..., description="The full riddle as verse")
    clue_lines: List[str] = Field(..., description="Individual clue lines")
    difficulty: str = Field(
        default="medium", description="easy/medium/hard - affects obscurity"
    )
    hint: Optional[str] = Field(
        default=None,
        description="Optional subtle hint (e.g., 'Think about the fourth dimension')",
    )
