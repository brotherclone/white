from pydantic import BaseModel


class DivergencePoint(BaseModel):
    """Where the alternate timeline splits from actual history."""

    when: str  # "After graduating college in 1997"
    what_changed: str  # "Took the Greyhound to Portland instead of returning to NJ"
    why_plausible: str  # "Had been offered a job at Powell's Books"
