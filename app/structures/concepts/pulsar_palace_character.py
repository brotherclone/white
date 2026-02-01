import os
from pathlib import Path
from typing import Optional, TYPE_CHECKING
from dotenv import load_dotenv
from pydantic import BaseModel, ConfigDict, Field

from app.agents.tools.gaming_tools import no_repeat_roll_dice, roll_dice
from app.agents.tools.image_tools import composite_character_portrait
from app.structures.artifacts.image_artifact_file import ImageChainArtifactFile
from app.structures.artifacts.character_portrait_artifact import (
    CharacterPortraitArtifact,
)
from app.structures.enums.chain_artifact_file_type import ChainArtifactFileType

if TYPE_CHECKING:
    from app.structures.artifacts.pulsar_palace_character_sheet import (
        PulsarPalaceCharacterSheet,
    )

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


class PulsarPalaceCharacterDisposition(BaseModel):

    rollId: int = Field(description="The roll ID of the disposition", ge=1, le=8)
    disposition: str = Field(description="The disposition of the character")
    image_path: Optional[str] = Field(
        description="The path to the disposition image", default=None
    )


class PulsarPalaceCharacterProfession(BaseModel):

    rollId: int = Field(description="The roll ID of the profession", ge=1, le=10)
    profession: str = Field(description="The profession of the character")
    image_path: Optional[str] = Field(
        description="The path to the profession image", default=None
    )


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
    portrait_artifact: Optional[CharacterPortraitArtifact] = Field(
        default=None, description="High-level portrait artifact with metadata"
    )
    # Note: character_sheet creates a circular reference (sheet_content points back to this)
    # Use exclude={"character_sheet"} when serializing to avoid infinite recursion
    character_sheet: Optional["PulsarPalaceCharacterSheet"] = Field(
        default=None, description="Character sheet of the character", exclude=True
    )
    model_config = ConfigDict(arbitrary_types_allowed=True)

    def __init__(self, **data):
        # Rebuild model to resolve forward references if needed
        if not hasattr(self.__class__, "_rebuilt"):
            # Import at runtime to make PulsarPalaceCharacterSheet available for model_rebuild
            from app.structures.artifacts.pulsar_palace_character_sheet import (
                PulsarPalaceCharacterSheet as _PulsarPalaceCharacterSheet,
            )

            self.__class__.model_rebuild(
                _types_namespace={
                    "PulsarPalaceCharacterSheet": _PulsarPalaceCharacterSheet
                }
            )
            self.__class__._rebuilt = True
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

    def to_markdown(self) -> str:
        """Generate markdown representation of the character sheet."""
        if not self.portrait or not self.portrait.file_path:
            raise ValueError(
                "Character must have a portrait before generating markdown"
            )

        portrait_filename = Path(self.portrait.file_path).name
        relative_portrait_path = f"../png/{portrait_filename}"

        return f"""![{self.disposition.disposition} {self.profession.profession}]({relative_portrait_path})
# {self.disposition.disposition} {self.profession.profession}
## from {self.background.time}, {self.background.place}
### ON
{self.on_current} / {self.on_max}
### OFF
{self.off_current} / {self.off_max}
"""

    def create_portrait(self):
        """Create a composite portrait image and portrait artifact for this character."""
        from PIL import Image

        portrait_filename = f"character_portrait_{self.encounter_id}.png"
        output_path = os.path.join(
            os.getenv("AGENT_WORK_PRODUCT_BASE_PATH"),
            self.thread_id,
            ChainArtifactFileType.PNG.value,
            portrait_filename,
        )

        # Composite the background and trait layers
        png = composite_character_portrait(
            self.background.image_path,
            [self.profession.image_path, self.disposition.image_path],
            output_path,
        )

        # Get image dimensions
        with Image.open(png) as img:
            width, height = img.size

        # Create the low-level image artifact
        # Don't pass file_path - it will be computed from base_path + thread_id + file_type
        self.portrait = ImageChainArtifactFile(
            thread_id=self.thread_id,
            file_name=portrait_filename,
            base_path=os.getenv("AGENT_WORK_PRODUCT_BASE_PATH"),
            width=width,
            height=height,
        )

        # Create the high-level portrait artifact with metadata
        character_name = f"{self.disposition.disposition} {self.profession.profession}"
        self.portrait_artifact = CharacterPortraitArtifact(
            thread_id=self.thread_id,
            base_path=os.getenv("AGENT_WORK_PRODUCT_BASE_PATH"),
            character_name=character_name,
            role=self.profession.profession,
            pose=self.disposition.disposition,
            description=f"From {self.background.place} ({self.background.time})",
            image=self.portrait,
        )

    def create_character_sheet(self):
        """Create a markdown character sheet artifact for this character."""
        from app.structures.artifacts.pulsar_palace_character_sheet import (
            PulsarPalaceCharacterSheet,
        )

        base_path = os.getenv("AGENT_WORK_PRODUCT_BASE_PATH", "chain_artifacts")
        self.character_sheet = PulsarPalaceCharacterSheet(
            thread_id=self.thread_id,
            sheet_content=self,
            base_path=base_path,
            image_path=f"{base_path}/img",
        )
        self.character_sheet.save_file()
