from datetime import datetime

from pydantic import BaseModel

from app.structures.concepts.rainbow_table_color import (
    RainbowTableColor,
    get_rainbow_table_color,
)
from app.structures.manifests.manifest_song_structure import ManifestSongStructure
from app.structures.manifests.manifest_sounds_like import ManifestSoundsLike
from app.structures.manifests.manifest_track import ManifestTrack
from app.structures.music.core.duration import Duration
from app.structures.music.core.key_signature import KeySignature, get_mode
from app.structures.music.core.notes import get_note
from app.structures.music.core.time_signature import TimeSignature


class Manifest(BaseModel):

    bpm: int
    manifest_id: str
    tempo: str | TimeSignature
    key: str | KeySignature
    rainbow_color: str | RainbowTableColor
    title: str
    release_date: str | datetime
    album_sequence: int
    main_audio_file: str
    TRT: str | Duration
    vocals: bool
    lyrics: bool
    sounds_like: list[ManifestSoundsLike]
    structure: list[ManifestSongStructure]
    mood: list[str]
    genres: list[str]
    lrc_file: str | None = None
    concept: str
    audio_tracks: list[ManifestTrack]

    def __init__(self, **data):
        if "tempo" in data and isinstance(data["tempo"], str):
            try:
                tempo = data["tempo"].split("/")
                data["tempo"] = TimeSignature(
                    numerator=int(tempo[0]), denominator=int(tempo[1])
                )
            except ValueError:
                print("Unable to parse tempo, defaulting to string")
        if "key" in data:
            try:
                if isinstance(data["key"], str):
                    key = data["key"].split(" ")
                    if len(key) != 2:
                        raise ValueError(
                            "Key must be in the format 'Note Mode', e.g., 'C major'"
                        )
                    note = get_note(key[0])
                    mode = get_mode(key[1])
                    data["key"] = KeySignature(note=note, mode=mode)
                elif isinstance(data["key"], dict):
                    note_name = data["key"]["note"]["pitch_name"]
                    mode_name = data["key"]["mode"]["name"]
                    note = get_note(note_name)
                    mode = get_mode(mode_name)
                    data["key"] = KeySignature(note=note, mode=mode)

            except (ValueError, KeyError, TypeError) as e:
                print(f"Unable to parse key: {e}, defaulting to string")
        if "rainbow_color" in data and isinstance(data["rainbow_color"], str):
            try:
                color = data["rainbow_color"]
                if color in ["Z", "R", "O", "Y", "G", "B", "I", "V", "A"]:
                    data["rainbow_color"] = get_rainbow_table_color(color)
                else:
                    raise ValueError(
                        "Rainbow color must be one of the following: Z, R, O, Y, G, B, I, V, A"
                    )
            except ValueError:
                print("Unable to parse rainbow color, defaulting to string")
        if "release_date" in data and isinstance(data["release_date"], str):
            data["release_date"] = datetime.strptime(data["release_date"], "%Y-%m-%d")
        super().__init__(**data)

    def __getitem__(self, key):
        """
        Support manifest['something'] by first checking attributes,
        then falling back to an internal _data mapping if available.
        Raise KeyError for missing keys to match mapping behaviour.
        """
        if hasattr(self, key):
            return getattr(self, key)
        if isinstance(self._data, dict) and key in self._data:
            return self._data[key]
        raise KeyError(key)

    def __contains__(self, key):
        if hasattr(self, key):
            return True
        return isinstance(self._data, dict) and key in self._data

    def get(self, key, default=None):
        try:
            return self[key]
        except KeyError:
            return default
