from enum import Enum


class RainbowColor(Enum):
    Z = "Black"
    R = "Red"
    O = "Orange"
    Y = "Yellow"
    G = "Green"
    B = "Blue"
    I = "Indigo"
    V = "Violet"
    A = "White"

    @classmethod
    def get_key_by_value(cls, value: str) -> str:
        for key, member in cls.__members__.items():
            if member.value == value:
                return key
        raise ValueError(f"No matching key found for value: {value}")
