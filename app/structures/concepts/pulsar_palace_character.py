from typing import Optional

from pydantic import BaseModel, ConfigDict, Field

from app.agents.tools.gaming_tools import no_repeat_roll_dice, roll_dice
from app.structures.artifacts.image_artifact_file import ImageChainArtifactFile
from app.structures.artifacts.pulsar_palace_character_sheet import (
    PulsarPalaceCharacterSheet,
)

PULSAR_PALACE_IMAGE_BASE_PATH = "/Volumes/LucidNonsense/White/app/reference/gaming/img"

PULSAR_PALACE_BACKGROUNDS = [
    {
        "rollId": 1,
        "time": 2121,
        "place": "New York City",
        "image_path": f"{PULSAR_PALACE_IMAGE_BASE_PATH}/character__0014_setting-nyc.png",
    },
    {
        "rollId": 2,
        "time": 1953,
        "place": "Hollywood",
        "image_path": f"{PULSAR_PALACE_IMAGE_BASE_PATH}/character__0015_setting-hollywood.png",
    },
    {
        "rollId": 3,
        "time": 2084,
        "place": "Glasgow",
        "image_path": f"{PULSAR_PALACE_IMAGE_BASE_PATH}/character__0016_setting-glasgow.png",
    },
    {
        "rollId": 4,
        "time": 1973,
        "place": "London",
        "image_path": f"{PULSAR_PALACE_IMAGE_BASE_PATH}/character__0017_setting-london.png",
    },
    {
        "rollId": 5,
        "time": 1865,
        "place": "Paris",
        "image_path": f"{PULSAR_PALACE_IMAGE_BASE_PATH}/character__0018_setting-paris.png",
    },
    {
        "rollId": 6,
        "time": 1949,
        "place": "Berlin",
        "image_path": f"{PULSAR_PALACE_IMAGE_BASE_PATH}/character__0019_setting-berlin.png",
    },
    {
        "rollId": 7,
        "time": 1937,
        "place": "Milan",
        "image_path": f"{PULSAR_PALACE_IMAGE_BASE_PATH}/character__0020_setting-milan.png",
    },
    {
        "rollId": 8,
        "time": 1992,
        "place": "Mexico City",
        "image_path": f"{PULSAR_PALACE_IMAGE_BASE_PATH}/character__0021_setting-mexico-city.png",
    },
    {
        "rollId": 9,
        "time": 1982,
        "place": "Hong Kong",
        "image_path": f"{PULSAR_PALACE_IMAGE_BASE_PATH}/character__0022_setting-hong-kong.png",
    },
    {
        "rollId": 10,
        "time": 1727,
        "place": "Baghdad",
        "image_path": f"{PULSAR_PALACE_IMAGE_BASE_PATH}/character__0023_setting-bahgdad.png",
    },
]
PULSAR_PALACE_DISPOSITIONS = [
    {
        "rollId": 1,
        "disposition": "Angry",
        "image_path": f"{PULSAR_PALACE_IMAGE_BASE_PATH}/character__0000_descriptor-angry.png",
    },
    {
        "rollId": 2,
        "disposition": "Curious",
        "image_path": f"{PULSAR_PALACE_IMAGE_BASE_PATH}/character__0001_descriptor-curious.png",
    },
    {
        "rollId": 3,
        "disposition": "Misguided",
        "image_path": f"{PULSAR_PALACE_IMAGE_BASE_PATH}/character__0002_descriptor-misguided.png",
    },
    {
        "rollId": 4,
        "disposition": "Clumsy",
        "image_path": f"{PULSAR_PALACE_IMAGE_BASE_PATH}/character__0003_descriptor-clumsy.png",
    },
    {
        "rollId": 5,
        "disposition": "Cursed",
        "image_path": f"{PULSAR_PALACE_IMAGE_BASE_PATH}/character__0004_descriptor-cursed.png",
    },
    {
        "rollId": 6,
        "disposition": "Sick",
        "image_path": f"{PULSAR_PALACE_IMAGE_BASE_PATH}/character__0005_descriptor-sick.png",
    },
    {
        "rollId": 7,
        "disposition": "Vengeful",
        "image_path": f"{PULSAR_PALACE_IMAGE_BASE_PATH}/character__0006_descriptor-vengful.png",
    },
    {
        "rollId": 8,
        "disposition": "Crazed",
        "image_path": f"{PULSAR_PALACE_IMAGE_BASE_PATH}/character__0009_descriptor-crazed.png",
    },
]
PULSAR_PALACE_PROFESSIONS = [
    {
        "rollId": 1,
        "profession": "Doctor",
        "image_path": f"{PULSAR_PALACE_IMAGE_BASE_PATH}/character__0010_role-doctor.png",
    },
    {
        "rollId": 2,
        "profession": "Sailor",
        "image_path": f"{PULSAR_PALACE_IMAGE_BASE_PATH}/character__0011_role-sailor.png",
    },
    {
        "rollId": 3,
        "profession": "Breeder",
        "image_path": f"{PULSAR_PALACE_IMAGE_BASE_PATH}/character__0012_role-breeder.png",
    },
    {
        "rollId": 4,
        "profession": "Detective",
        "image_path": f"{PULSAR_PALACE_IMAGE_BASE_PATH}/character__0013_role-detective.png",
    },
    {
        "rollId": 5,
        "profession": "Janitor",
        "image_path": f"{PULSAR_PALACE_IMAGE_BASE_PATH}/character__0024_role-janitor.png",
    },
    {
        "rollId": 6,
        "profession": "Spy",
        "image_path": f"{PULSAR_PALACE_IMAGE_BASE_PATH}/character__0025_role-spy.png",
    },
    {
        "rollId": 7,
        "profession": "Librarian",
        "image_path": f"{PULSAR_PALACE_IMAGE_BASE_PATH}/character__0026_role-librarian.png",
    },
    {
        "rollId": 8,
        "profession": "Inventor",
        "image_path": f"{PULSAR_PALACE_IMAGE_BASE_PATH}/character__0027_role-inventor.png",
    },
    {
        "rollId": 9,
        "profession": "Tax Collector",
        "image_path": f"{PULSAR_PALACE_IMAGE_BASE_PATH}/character__0028_role---tax-collector.png",
    },
    {
        "rollId": 10,
        "profession": "Partisan",
        "image_path": f"{PULSAR_PALACE_IMAGE_BASE_PATH}/character__0029_role---partisan.png",
    },
]


class PulsarPalaceCharacterBackground(BaseModel):

    rollId: int
    time: int = None
    place: str = None
    image_path: str = None

    def __init__(self, **data):
        super().__init__(**data)


class PulsarPalaceCharacterDisposition(BaseModel):

    rollId: int
    disposition: str = None
    image_path: str = None

    def __init__(self, **data):
        super().__init__(**data)


class PulsarPalaceCharacterProfession(BaseModel):

    rollId: int
    profession: str = None
    image_path: str = None

    def __init__(self, **data):
        super().__init__(**data)


class PulsarPalaceCharacter(BaseModel):

    background: PulsarPalaceCharacterBackground = Field(
        default=None,
        description="The background, time and place of origin, of the character.",
    )
    disposition: PulsarPalaceCharacterDisposition = Field(
        default=None, description="The general disposition of the character."
    )
    profession: PulsarPalaceCharacterProfession = Field(
        default=None, description="The profession or role of the character."
    )
    on_max: int = Field(
        default=1,
        description="The positive charge of the character at its maximum value",
        ge=0,
        le=30,
    )
    off_max: int = Field(
        default=1,
        description="The negative charge of the character at its maximum value",
        ge=0,
        le=30,
    )
    on_current: int = Field(
        default=1,
        description="The positive charge of the character at its current value",
        ge=0,
        le=50,
    )
    off_current: int = Field(
        default=1,
        description="The negative charge of the character at its current value",
        ge=0,
        le=50,
    )
    portrait: Optional[ImageChainArtifactFile] = Field(
        default=None, description="Portrait of the character in png format"
    )
    character_sheet: Optional[PulsarPalaceCharacterSheet] = Field(
        default=None, description="Character sheet of the character"
    )
    model_config = ConfigDict(arbitrary_types_allowed=True)

    def __init__(self, **data):
        super().__init__(**data)

    @classmethod
    def create_random(cls):
        bg = PulsarPalaceCharacterBackground(
            **PULSAR_PALACE_BACKGROUNDS[roll_dice([(1, 10)])[0] - 1]
        )
        disp = PulsarPalaceCharacterDisposition(
            **PULSAR_PALACE_DISPOSITIONS[roll_dice([(1, 8)])[0] - 1]
        )
        prof = PulsarPalaceCharacterProfession(
            **PULSAR_PALACE_PROFESSIONS[roll_dice([(1, 10)])[0] - 1]
        )
        on_roll, off_roll = no_repeat_roll_dice([(1, 20)], [(1, 20)])
        return cls(
            background=bg,
            disposition=disp,
            profession=prof,
            on_max=on_roll,
            on_current=on_roll,
            off_max=off_roll,
            off_current=off_roll,
        )

    def create_portrait(self):
        print(self.background.image_path)
        print(self.profession.image_path)
        print(self.disposition.image_path)

    def create_character_sheet(self):
        pass
