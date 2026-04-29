from enum import Enum


class LastHumanDocumentationType(str, Enum):
    """How The Arbitrary documents this human"""

    DEATH = "death"  # Died during collapse
    DISPLACEMENT = "displacement"  # Forced migration, lost home
    ADAPTATION_FAILURE = "adaptation_failure"  # Couldn't adjust to new conditions
    RESILIENCE = "resilience"  # Survived but fundamentally changed
    WITNESS = "witness"  # Documented the collapse
