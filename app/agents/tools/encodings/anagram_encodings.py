from pydantic import Field, field_validator

from app.structures.concepts.infranym_text_encoding import InfranymTextEncoding
from app.structures.enums.infranym_method import InfranymMethod


class AnagramEncoding(InfranymTextEncoding):
    """
    Anagram encoding where the surface phrase rearranges to secret.

    Example:
        Secret: "TEMPORAL"
        Surface: "MAPLE TOR" (same letters, different meaning)
    """

    method: InfranymMethod = InfranymMethod.ACROSTIC_LYRICS  # Reusing enum value
    surface_phrase: str = Field(..., description="The anagram phrase (visible)")
    letter_bank: str = Field(..., description="Sorted letters used")
    usage_instruction: str = Field(
        default="Repeat as hook/refrain", description="How to emphasize in song"
    )

    @field_validator("surface_phrase")
    @classmethod
    def validate_anagram(cls, v, info):
        """Ensure surface phrase is valid anagram of secret"""
        secret = info.data.get("secret_word", "").upper().replace(" ", "")
        surface = v.upper().replace(" ", "")

        if sorted(secret) != sorted(surface):
            raise ValueError(
                f"Surface '{v}' is not an anagram of '{info.data.get('secret_word')}'"
            )
        return v
