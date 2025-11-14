from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field

"""
This is the heart of the Rainbow Table, the Newtonian colors echoing their occult origin in the temporal, spatial, and 
informational dimensions.
"""


class RainbowColorModes(Enum):
    """Enum representing the modes of rainbow colors. A six-dimensional perspective from which they are perceived in
    The Rainbow Table."""

    TIME = "Temporal"
    SPACE = "Objectional"
    INFORMATION = "Ontological"
    NONE = "None"

    def __str__(self):
        return self.value

    def __repr__(self):
        return f"RainbowColorModes.{self.name}"


class RainbowTableTransmigrationalMode(BaseModel):
    """The Earthly to Frame and Frame to Earthly transmigrational modes of rainbow colors. Well, how 'bout that?"""

    current_mode: RainbowColorModes
    transitory_mode: RainbowColorModes
    transcendental_mode: RainbowColorModes

    def __init__(
        self,
        current_mode: RainbowColorModes,
        transitory_mode: RainbowColorModes,
        transcendental_mode: RainbowColorModes,
    ):
        super().__init__(
            current_mode=current_mode,
            transitory_mode=transitory_mode,
            transcendental_mode=transcendental_mode,
        )

    def __str__(self):
        return f"{self.current_mode} -> {self.transitory_mode} -> {self.transcendental_mode}"

    def __repr__(self):
        return f"RainbowTableTransmigrationalMode(current_mode={self.current_mode}, transitory_mode={self.transitory_mode}, transcendental_mode={self.transcendental_mode})"


class RainbowColorTemporalMode(Enum):
    """
    Enum representing the temporal modes of rainbow colors. A fourth-dimensional perspective from which they are
    perceived in The Rainbow Table.
    """

    PAST = "Past"
    PRESENT = "Present"
    FUTURE = "Future"
    NONE = "None"

    def __str__(self):
        return self.value

    def __repr__(self):
        return f"RainbowColorTemporalMode.{self.name}"


class RainbowColorObjectionalMode(Enum):
    """Enum representing the objectional modes of rainbow colors. A third-dimensional perspective from which they are
    perceived in The Rainbow Table."""

    THING = "Thing"
    PERSON = "Person"
    PLACE = "Place"
    NONE = "None"

    def __str__(self):
        return self.value

    def __repr__(self):
        return f"RainbowColorObjectionalMode.{self.name}"


class RainbowColorOntologicalMode(Enum):
    """Enum representing the ontological modes of rainbow colors. A fifth-dimensional perspective from which they are
    perceived in The Rainbow Table."""

    KNOWN = "Known"
    IMAGINED = "Imagined"
    FORGOTTEN = "Forgotten"

    def __str__(self):
        return self.value

    def __repr__(self):
        return f"RainbowColorOntologicalMode.{self.name}"


class RainbowTableColor(BaseModel):
    """
    Represents a color in the Rainbow Table with its associated properties and modes.
    """

    color_name: str = Field(
        description="Name of the rainbow color (Red, Orange, Yellow, Green, Blue, Indigo, Violet, White, Black)",
        examples=["Indigo", "Red", "Violet", "Black"],
    )
    hex_value: int = Field(
        description="Hexadecimal color value as integer",
        examples=[4915330, 16711680, 65280],
        ge=0,
        le=16777215,  # Max RGB value (0xFFFFFF)
    )
    mnemonic_character_value: str = Field(
        description="Single character mnemonic for the color",
        examples=["I", "R", "V", "B"],
        min_length=1,
        max_length=1,
    )
    temporal_mode: Optional[RainbowColorTemporalMode] = Field(
        default=None,
        description="Temporal positioning: Past, Present, or Future",
        examples=["Future", "Past", "Present"],
    )
    objectional_mode: Optional[RainbowColorObjectionalMode] = Field(
        default=None,
        description="Subject categorization: Thing, Place, or Person",
        examples=["Person", "Place", "Thing"],
    )
    ontological_mode: Optional[list[RainbowColorOntologicalMode]] = Field(
        default=None,
        description="Existential status: Known, Imagined, or Forgotten",
        examples=[["Known"], ["Imagined"], ["Forgotten"], ["Known", "Forgotten"]],
    )
    transmigrational_mode: Optional[RainbowTableTransmigrationalMode] = Field(
        default=None,
        description="Transmigrational modes between Earthly and Frame perspectives",
        examples=[
            {
                "current_mode": "Time",
                "transitory_mode": "Space",
                "transcendental_mode": "Information",
            }
        ],
    )
    file_prefix: Optional[str] = Field(
        default=None,
        description="File prefix associated with the color",
        examples=["01", "02", "03"],
    )

    def __init__(self, **data):
        super().__init__(**data)

    def __str__(self):
        return f"{self.color_name} ({self.hex_value:#06x})"

    def __repr__(self):
        return f"RainbowTableColor(color_name={self.color_name}, hex_value={self.hex_value:#06x}, mnemonic_character_value={self.mnemonic_character_value}, transmigrational_mode={self.transmigrational_mode})"

    def to_dict(self):
        return {
            "color_name": self.color_name,
            "hex_value": self.hex_value,
            "mnemonic_letter_value": self.mnemonic_character_value,
            "temporal_mode": str(self.temporal_mode) if self.temporal_mode else None,
            "objectional_mode": (
                str(self.objectional_mode) if self.objectional_mode else None
            ),
            "ontological_mode": (
                [str(mode) for mode in self.ontological_mode]
                if self.ontological_mode
                else None
            ),
            "transmigrational_mode": (
                {
                    "current_mode": str(self.transmigrational_mode.current_mode),
                    "transitory_mode": str(self.transmigrational_mode.transitory_mode),
                    "transcendental_mode": str(
                        self.transmigrational_mode.transcendental_mode
                    ),
                }
                if self.transmigrational_mode
                else None
            ),
            "file_prefix": self.file_prefix,
        }

    def get_rgba(self) -> tuple[int, int, int, int]:
        r = (self.hex_value >> 16) & 0xFF
        g = (self.hex_value >> 8) & 0xFF
        b = self.hex_value & 0xFF
        return r, g, b, 255


the_rainbow_table_colors = {
    "Z": RainbowTableColor(
        color_name="Black",
        hex_value=0x231F20,
        mnemonic_character_value="Z",
        transmigrational_mode=RainbowTableTransmigrationalMode(
            current_mode=RainbowColorModes.SPACE,
            transitory_mode=RainbowColorModes.TIME,
            transcendental_mode=RainbowColorModes.INFORMATION,
        ),
        file_prefix="01",
    ),
    "R": RainbowTableColor(
        color_name="Red",
        hex_value=0xAE1E36,
        mnemonic_character_value="R",
        temporal_mode=RainbowColorTemporalMode.PAST,
        objectional_mode=RainbowColorObjectionalMode.THING,
        ontological_mode=[RainbowColorOntologicalMode.KNOWN],
        file_prefix="02",
    ),
    "O": RainbowTableColor(
        color_name="Orange",
        hex_value=0xEF7143,
        mnemonic_character_value="O",
        temporal_mode=RainbowColorTemporalMode.PAST,
        objectional_mode=RainbowColorObjectionalMode.THING,
        ontological_mode=[RainbowColorOntologicalMode.IMAGINED],
        file_prefix="03",
    ),
    "Y": RainbowTableColor(
        color_name="Yellow",
        hex_value=0xFFFF00,
        mnemonic_character_value="Y",
        temporal_mode=RainbowColorTemporalMode.FUTURE,
        objectional_mode=RainbowColorObjectionalMode.PLACE,
        ontological_mode=[RainbowColorOntologicalMode.IMAGINED],
        file_prefix="04",
    ),
    "G": RainbowTableColor(
        color_name="Green",
        hex_value=0xABD96D,
        mnemonic_character_value="G",
        temporal_mode=RainbowColorTemporalMode.FUTURE,
        objectional_mode=RainbowColorObjectionalMode.PLACE,
        ontological_mode=[RainbowColorOntologicalMode.FORGOTTEN],
        file_prefix="05",
    ),
    "B": RainbowTableColor(
        color_name="Blue",
        hex_value=0x042A7B,
        mnemonic_character_value="B",
        temporal_mode=RainbowColorTemporalMode.PRESENT,
        objectional_mode=RainbowColorObjectionalMode.PERSON,
        ontological_mode=[RainbowColorOntologicalMode.FORGOTTEN],
        file_prefix="06",
    ),
    "I": RainbowTableColor(
        color_name="Indigo",
        hex_value=0x26294A,
        mnemonic_character_value="I",
        ontological_mode=[
            RainbowColorOntologicalMode.KNOWN,
            RainbowColorOntologicalMode.FORGOTTEN,
        ],
        file_prefix="07",
    ),
    "V": RainbowTableColor(
        color_name="Violet",
        hex_value=0xAD85D6,
        mnemonic_character_value="V",
        temporal_mode=RainbowColorTemporalMode.PRESENT,
        objectional_mode=RainbowColorObjectionalMode.PERSON,
        ontological_mode=[RainbowColorOntologicalMode.KNOWN],
        file_prefix="08",
    ),
    "A": RainbowTableColor(
        color_name="White",
        hex_value=0xF6F6F6,
        mnemonic_character_value="A",
        transmigrational_mode=RainbowTableTransmigrationalMode(
            current_mode=RainbowColorModes.INFORMATION,
            transitory_mode=RainbowColorModes.TIME,
            transcendental_mode=RainbowColorModes.SPACE,
        ),
        file_prefix="09",
    ),
}


def get_rainbow_table_color(color_str: str) -> RainbowTableColor:
    if color_str in the_rainbow_table_colors:
        return the_rainbow_table_colors[color_str]
    else:
        raise ValueError(f"Color {color_str} is not a valid rainbow table color.")
