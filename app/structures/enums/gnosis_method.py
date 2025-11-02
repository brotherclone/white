from enum import Enum

class GnosisMethod(str, Enum):
    """Methods for achieving gnosis/charging state"""
    EXHAUSTION = "exhaustion"
    ECSTASY = "ecstasy"
    OBSESSION = "obsession"
    SENSORY_OVERLOAD = "sensory_overload"
    MEDITATION = "meditation"
    CHAOS = "chaos"