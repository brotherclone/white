from pydantic import BaseModel

class TrainingSample(BaseModel):
    song_bpm: str | int
    song_key: str
    album_rainbow_color: str
    artist: str
    album_title: str
    song_title: str
    album_realise_date: str
    song_on_album_sequence: int
    song_trt: str
    song_has_vocals: bool
    song_has_lyrics: bool
    song_structure: str
    song_moods: str
    song_sounds_like: str
    song_genres: str
    song_segment_name: str
    song_segment_start_time: str
    song_segment_end_time: str
    song_segment_duration: str
    song_segment_sequence: int | None = None
    song_segment_description: str | None = None
    song_segment_track_name: str | None = None
    song_segment_track_description: str | None = None
    song_segment_track_id: str | int | None = None
    song_segment_main_audio_file_name: str | None = None
    song_segment_track_audio_file_name: str | None = None
    song_segment_main_audio_binary_data: bytes | None = None
    song_segment_track_audio_binary_data: bytes | None = None
    song_segment_lyrics_text: str | None = None
    song_segment_lyrics_lrc: str | None = None
    song_segment_track_midi_data: str | None = None
    song_segment_track_midi_file_name: str | None = None
    song_segment_track_midi_binary_data: bytes | None = None
    song_segment_track_midi_is_group: str | None = None

    def __init__(self, /, **data):
        super().__init__(**data)
