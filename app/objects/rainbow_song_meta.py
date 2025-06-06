import datetime
import yaml
import os
from pydantic import BaseModel
from app.objects.rainbow_color import RainbowColor

the_artist = "The Earthly Frames"

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

class MetaDataModel(BaseModel):
    bpm: int
    tempo: str # ToDo: Model better if useful
    key: str # ToDo: Model better if useful
    rainbow_color: RainbowColor
    artist: str
    album_title: str
    song_title: str
    release_date: str
    album_sequence: int
    main_audio_file: str
    TRT: str  # Total Running Time
    vocals: bool
    lyrics: bool

class RainbowSongMeta(BaseModel):
    yaml_file_name: str
    base_path: str
    track_materials_path: str
    data: MetaDataModel | None = None

    def __init__(self, /, **data):
        super().__init__(**data)
        with(open(os.path.join(self.base_path,self.track_materials_path,self.yaml_file_name), 'r')) as f:
            self.data = yaml.safe_load(f)
        self.data['artist'] = the_artist
        album = RainbowColor.get_key_by_value(self.data['rainbow_color'])
        k: RainbowColor = RainbowColor[album] # type: ignore
        self.data['album_title'] = rainbow_table_title_dict[k]

    def get_release_date(self):
        return datetime.datetime.strptime(self.data['release_date'], "%Y-%m-%d").date() if self.data else None

    def get_duration(self) -> datetime.timedelta:
        if self.data and self.data['TRT']:
            return datetime.timedelta(seconds=self.data['TRT'])
        return datetime.timedelta(0)