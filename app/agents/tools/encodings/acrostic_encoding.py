from typing import List
from pydantic import Field, field_validator

from app.structures.concepts.infranym_text_encoding import InfranymTextEncoding
from app.structures.enums.infranym_method import InfranymMethod


class AcrosticEncoding(InfranymTextEncoding):
    """
    Acrostic-specific encoding where the first letters spell the secret.

    Example:
        Secret: "TEMPORAL"
        Lines:
            "Time bends in spirals..."
            "Echoes of futures untold..."
            "Memory fragments collide..."
            ...
    """

    method: InfranymMethod = InfranymMethod.ACROSTIC_LYRICS
    lines: List[str] = Field(..., description="Each line starting with secret letter")
    reveal_pattern: str = Field(
        default="first_letter",
        description="How to extract secret (first_letter, last_letter, etc.)",
    )

    @field_validator("lines")
    @classmethod
    def validate_acrostic(cls, v, info):
        """Ensure lines match secret word length"""
        secret = info.data.get("secret_word", "")
        if len(v) != len(secret):
            raise ValueError(
                f"Acrostic requires {len(secret)} lines for secret '{secret}', got {len(v)}"
            )
        return v
