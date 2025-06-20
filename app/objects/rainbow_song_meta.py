import datetime
import yaml
import os
from pydantic import BaseModel, ValidationError
from app.objects.rainbow_color import RainbowColor

THE_ARTIST = "The Earthly Frames"

rainbow_table_title_dict ={
    RainbowColor.Z: "The Conjurer's Thread",
    RainbowColor.R: "Light Reading",
    RainbowColor.O: "Ruine",
    RainbowColor.Y: "Pulsar Palace",
    RainbowColor.G: "The Empty Fields",
    RainbowColor.B: "Taped Over",
    RainbowColor.I: "Infranyms",
    RainbowColor.V: "Vanity Pressing",
    RainbowColor.A: "TBD"
}

class RainbowSongLyricModel(BaseModel):
    time_stamp: str | datetime.timedelta | None = None
    lrc: str | datetime.timedelta | None = None
    lyrics: str | None = None
    is_in_range: bool = False

class RainbowSongTrackModel(BaseModel):
    id: int
    description: str
    audio_file: str | None = None
    audio_file_path: str | None = None
    midi_file: str | None = None
    midi_group_file: str | None = None

class RainbowSongStructureModel(BaseModel):
    section_name: str
    section_description: str | None = None
    start_time: str | datetime.timedelta
    end_time: str | datetime.timedelta
    duration: datetime.timedelta | None = None
    sequence: None | int = None

class RainbowMetaDataModel(BaseModel):
    bpm: int
    tempo: str
    key: str
    rainbow_color: str | RainbowColor | None = None
    artist: str | None = None
    album_title: str | None = None
    title: str
    release_date: str | datetime.date
    album_sequence: int
    main_audio_file: str | None = None
    TRT: str | datetime.timedelta
    vocals: bool
    lyrics: bool
    structure: list[RainbowSongStructureModel]
    mood: list[str]
    sounds_like: list[str]
    genres: list[str]
    lrc_file: bool | str = False
    audio_tracks: list[RainbowSongTrackModel] | None =None

class RainbowSongMeta(BaseModel):
    yaml_file_name: str
    base_path: str
    track_materials_path: str
    data: RainbowMetaDataModel | None = None

    def __init__(self, /, **data):
        yaml_path: str
        super().__init__(**data)
        try:
            yaml_path = os.path.join(self.base_path, self.track_materials_path, self.yaml_file_name)
            with open(yaml_path, 'r') as f:
                raw_data = yaml.safe_load(f)
            self.data = RainbowMetaDataModel(**raw_data)
            self.data.artist = THE_ARTIST
            album = RainbowColor.get_key_by_value(self.data.rainbow_color)  # type: ignore
            k: RainbowColor = RainbowColor[album]  # type: ignore
            self.data.album_title = rainbow_table_title_dict[k]
        except FileNotFoundError:
            raise FileNotFoundError(f"YAML file not found")
        except ValidationError as e:
            raise ValueError(f"Invalid YAML structure: {e}")
