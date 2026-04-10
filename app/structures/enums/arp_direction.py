from enum import Enum


class ArpDirection(str, Enum):
    """Direction of arpeggio note distribution within a strum pattern."""

    UP = "up"
    DOWN = "down"
