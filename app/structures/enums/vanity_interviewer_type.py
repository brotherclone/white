from enum import Enum

# ToDo: Add tests


class VanityInterviewerType(str, Enum):
    HOSTILE_SKEPTICAL = "hostile_skeptical"
    VANITY_PRESSING_FAN = "vanity_pressing_fan"
    EXPERIMENTAL_PURIST = "experimental_purist"
    EARNEST_BUT_WRONG = "earnest_but_wrong"
