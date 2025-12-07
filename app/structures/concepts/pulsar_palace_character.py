import os
from typing import Optional
from dotenv import load_dotenv
from pydantic import BaseModel, ConfigDict, Field

from app.agents.tools.gaming_tools import no_repeat_roll_dice, roll_dice
from app.structures.artifacts.image_artifact_file import ImageChainArtifactFile
from app.structures.artifacts.pulsar_palace_character_sheet import (
    PulsarPalaceCharacterSheet,
)
from app.agents.tools.image_tools import composite_character_portrait

load_dotenv()

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

    rollId: int = Field(description="The roll ID of the background", ge=1, le=10)
    time: int = Field(description="The year of the background", ge=0, le=10000)
    place: str = Field(description="The place of the background")
    image_path: Optional[str] = Field(
        description="The path to the background image", default=None
    )

    def __init__(self, **data):
        super().__init__(**data)


class PulsarPalaceCharacterDisposition(BaseModel):

    rollId: int = Field(description="The roll ID of the disposition", ge=1, le=8)
    disposition: str = Field(description="The disposition of the character")
    image_path: Optional[str] = Field(
        description="The path to the disposition image", default=None
    )

    def __init__(self, **data):
        super().__init__(**data)


class PulsarPalaceCharacterProfession(BaseModel):

    rollId: int = Field(description="The roll ID of the profession", ge=1, le=10)
    profession: str = Field(description="The profession of the character")
    image_path: Optional[str] = Field(
        description="The path to the profession image", default=None
    )

    def __init__(self, **data):
        super().__init__(**data)


class PulsarPalaceCharacter(BaseModel):

    thread_id: str = Field(description="The ID of the thread this character belongs to")
    encounter_id: str = Field(
        description="The ID of the encounter this character belongs to"
    )
    background: Optional[PulsarPalaceCharacterBackground] = Field(
        default=None,
        description="The background, time and place of origin, of the character.",
    )
    disposition: Optional[PulsarPalaceCharacterDisposition] = Field(
        default=None, description="The general disposition of the character."
    )
    profession: Optional[PulsarPalaceCharacterProfession] = Field(
        default=None, description="The profession or role of the character."
    )
    on_max: Optional[int] = Field(
        default=None,
        description="The positive charge of the character at its maximum value",
        ge=0,
        le=30,
    )
    off_max: Optional[int] = Field(
        default=None,
        description="The negative charge of the character at its maximum value",
        ge=0,
        le=30,
    )
    on_current: Optional[int] = Field(
        default=None,
        description="The positive charge of the character at its current value",
        ge=0,
        le=50,
    )
    off_current: Optional[int] = Field(
        default=None,
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
    def create_random(cls, thread_id: str, encounter_id: str):
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
            thread_id=thread_id,
            encounter_id=encounter_id,
            background=bg,
            disposition=disp,
            profession=prof,
            on_max=on_roll,
            on_current=on_roll,
            off_max=off_roll,
            off_current=off_roll,
        )

    def create_portrait(self):
        from PIL import Image

        # Let ImageChainArtifactFile generate the proper filename with UUID
        # First create the artifact to get the filename
        temp_portrait = ImageChainArtifactFile(
            thread_id=self.thread_id,
            base_path=os.getenv("AGENT_WORK_PRODUCT_BASE_PATH"),
            width=300,  # temporary values
            height=300,
        )

        # Use the generated filename for the composite
        output_path = temp_portrait.get_artifact_path()
        png = composite_character_portrait(
            self.background.image_path,
            [self.profession.image_path, self.disposition.image_path],
            output_path,
        )

        # Get actual image dimensions
        with Image.open(png) as img:
            width, height = img.size

        # Create the final portrait artifact with correct dimensions
        self.portrait = ImageChainArtifactFile(
            thread_id=self.thread_id,
            artifact_id=temp_portrait.artifact_id,  # Use same UUID
            base_path=os.getenv("AGENT_WORK_PRODUCT_BASE_PATH"),
            file_path=png,
            width=width,
            height=height,
        )

    def create_character_sheet(self):
        portrait_filename = self.portrait.file_name
        relative_portrait_path = f"../png/{portrait_filename}"

        template = f"""![{self.disposition.disposition} {self.profession.profession}]({relative_portrait_path})
# {self.disposition.disposition} {self.profession.profession}
## from {self.background.time}, {self.background.place}
### ON
{self.on_current} / {self.on_max}
### OFF
{self.off_current} / {self.off_max}
"""
        self.character_sheet = PulsarPalaceCharacterSheet(
            thread_id=self.thread_id,
            sheet_content=template,
            base_path=os.getenv("AGENT_WORK_PRODUCT_BASE_PATH"),
        )
        self.character_sheet.save_file()
