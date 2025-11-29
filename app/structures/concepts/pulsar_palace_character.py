from pydantic import BaseModel, ConfigDict

from app.agents.tools.gaming_tools import no_repeat_roll_dice, roll_dice

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


class PulsarPalacePlayer(BaseModel):

    first_name: str | None
    last_name: str | None
    biography: str | None
    attitude: str | None
    initialized: bool = False

    def __init__(self, **data):
        super().__init__(**data)
        if self.first_name and self.last_name and self.biography and self.attitude:
            self.initialized = True

    def get_example(self):
        return f"Write a thumbnail sketch of the person playing as this character. You will be role-playing them role-playing their Pulsar Palace character. Here's an example {self.json_schema_extra['example']}"

    class Config:
        json_schema_extra = {
            "example": {
                "first_name": "Ben",
                "last_name": "Quincy",
                "biography": "Ben is an unemployed writer who's job was taken by AI. This game of Pulsar Palace is something he looks forward to as an escape from living in his parents' basement. Ben is 34 and divorced and loves to read science fiction novels.",
                "attitude": "Ben is taking this game really seriously as its his only chance to socialize. So he's more than willing to explore and try all of the game's mechanics and completely finish it.",
            }
        }


class PulsarPalaceCharacter(BaseModel):

    background: PulsarPalaceCharacterBackground | None
    disposition: PulsarPalaceCharacterDisposition | None
    profession: PulsarPalaceCharacterProfession | None
    on_max: int | None
    off_max: int | None
    on_current: int | None
    off_current: int | None
    player: PulsarPalacePlayer | None = None

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
