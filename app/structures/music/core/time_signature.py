from pydantic import BaseModel

class TimeSignature(BaseModel):

    numerator: int
    denominator: int

    def __init__(self, **data):
        super().__init__(**data)
        if self.denominator not in [1, 2, 4, 8, 16, 32]:
            raise ValueError("Denominator must be one of the following: 1, 2, 4, 8, 16, 32")

    def __str__(self):
        return f"{self.numerator}/{self.denominator}"

    def __eq__(self, other):
        if isinstance(other, TimeSignature):
            return self.numerator == other.numerator and self.denominator == other.denominator
        return False