from pydantic import BaseModel, Field


class TimeSignature(BaseModel):
    """
    Represents a musical time signature.
    """

    numerator: int = Field(
        description="Beats per measure (top number)", examples=[4, 3, 6, 7], ge=1, le=16
    )
    denominator: int = Field(
        description="Note value that gets the beat (bottom number, typically 4, 8, or 16)",
        examples=[4, 8, 16],
        ge=1,
        le=32,
    )

    def __init__(self, **data):
        super().__init__(**data)
        if self.denominator not in [1, 2, 4, 8, 16, 32]:
            raise ValueError(
                "Denominator must be one of the following: 1, 2, 4, 8, 16, 32"
            )

    def __str__(self):
        return f"{self.numerator}/{self.denominator}"

    def __eq__(self, other):
        if isinstance(other, TimeSignature):
            return (
                self.numerator == other.numerator
                and self.denominator == other.denominator
            )
        return False

    def __hash__(self):
        return hash((self.numerator, self.denominator))
