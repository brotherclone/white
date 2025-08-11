from enum import Enum
from pydantic import BaseModel

"""
This is the heart of the Rainbow Table, the Newtonian colors echoing their occult origin in the temporal, spatial, and 
informational dimensions.
"""

class RainbowColorModes(Enum):
    """ Enum representing the modes of rainbow colors. A six dimensional perspective from which they are perceived in
    The Rainbow Table."""

    TIME = "Temporal"
    SPACE = "Objectional"
    INFORMATION = "Ontological"

    def __str__(self):
        return self.value

    def __repr__(self):
        return f"RainbowColorModes.{self.name}"

class RainbowTableTransmigrationalMode(Enum):

    """ The Earthly to Frame and Frame to Earthly transmigrational modes of rainbow colors. Well, how 'bout that?"""

    current_mode: RainbowColorModes
    transitory_mode: RainbowColorModes
    transcendental_mode: RainbowColorModes

    def __init__(self, current_mode: RainbowColorModes, transitory_mode: RainbowColorModes, transcendental_mode: RainbowColorModes):
        self.current_mode = current_mode
        self.transitory_mode = transitory_mode
        self.transcendental_mode = transcendental_mode

    def __str__(self):
        return f"{self.current_mode} -> {self.transitory_mode} -> {self.transcendental_mode}"

    def __repr__(self):
        return f"RainbowTableTransmigrationalMode(current_mode={self.current_mode}, transitory_mode={self.transitory_mode}, transcendental_mode={self.transcendental_mode})"

class RainbowColorTemporalMode(Enum):
    """
    Enum representing the temporal modes of rainbow colors. A fourth dimensional perspective from which they are
    perceived in The Rainbow Table.
    """

    PAST = "Past"
    PRESENT = "Present"
    FUTURE = "Future"

    def __str__(self):
        return self.value

    def __repr__(self):
        return f"RainbowColorTemporalMode.{self.name}"

class RainbowColorObjectionalMode(Enum):

    """ Enum representing the objectional modes of rainbow colors. A third dimensional perspective from which they are
    perceived in The Rainbow Table."""

    THING = "Thing"
    PERSON = "Person"
    PLACE = "Place"

    def __str__(self):
        return self.value

    def __repr__(self):
        return f"RainbowColorObjectionalMode.{self.name}"


class RainbowColorOntologicalMode(Enum):
    """ Enum representing the ontological modes of rainbow colors. A fifth dimensional perspective from which they are
    perceived in The Rainbow Table."""

    KNOWN = "Known"
    IMAGINED = "Imagined"
    FORGOTTEN = "Forgotten"

    def __str__(self):
        return self.value

    def __repr__(self):
        return f"RainbowColorOntologicalMode.{self.name}"



class RainbowTableColor(BaseModel):

    color_name: str
    hex_value: int
    mnemonic_character_value: str
    temporal_mode: RainbowColorTemporalMode | None = None
    objectional_mode: RainbowColorObjectionalMode | None = None
    ontological_mode: list[RainbowColorOntologicalMode] | None = None
    transmigrational_mode: RainbowTableTransmigrationalMode | None = None

    def __init__(self, **data):
        super().__init__(**data)

    def __str__(self):
        return f"{self.color_name} ({self.hex_value:#06x})"

    def __repr__(self):
        return f"RainbowTableColors(color_name={self.color_name}, hex_value={self.hex_value:#06x}, mnemonic_letter_value={self.mnemonic_character_value})"

    def to_dict(self):
        return {
            "color_name": self.color_name,
            "hex_value": self.hex_value,
            "mnemonic_letter_value": self.mnemonic_character_value
        }

    def get_rgba(self) -> tuple[int, int, int, int]:
        r = (self.hex_value >> 16) & 0xFF
        g = (self.hex_value >> 8) & 0xFF
        b = self.hex_value & 0xFF
        return r, g, b, 255