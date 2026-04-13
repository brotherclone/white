from enum import Enum


class DisruptingEventType(str, Enum):
    STRANGER_ENTERS = "stranger_enters"
    EQUIPMENT_FAILURE = "equipment_failure"
    MEMORY_INTRUSION = "memory_intrusion"
    TEMPORAL_BLEED = "temporal_bleed"
    TRANSMISSION_INTERFERENCE = "transmission_interference"
    IDENTITY_COLLAPSE = "identity_collapse"
