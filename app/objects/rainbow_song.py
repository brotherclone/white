import os
import re

from typing import Any
from pydantic import BaseModel
from pydub import AudioSegment

from app.objects.multimodal_extract import MultimodalExtract, MultimodalExtractModel
from app.objects.rainbow_song_meta import RainbowSongMeta
from app.utils.audio_util import has_significant_audio
from app.utils.time_util import lrc_to_seconds, get_duration
from app.utils.string_util import safe_filename

JUST_NUMBERS_PATTERN = r'^\d+$'
AUDIO_WORKING_DIR = "/Volumes/LucidNonsense/White/working"


class RainbowSong(BaseModel):
    meta_data: RainbowSongMeta
    extracts: list[MultimodalExtract] | None = None

    def __init__(self, /, **data: Any):
        super().__init__(**data)

        if self.extracts is None:
            self.extracts = []
        for song_section in self.meta_data.data.structure:
            s = lrc_to_seconds(song_section.start_time)
            e = lrc_to_seconds(song_section.end_time)
            print(f"Section '{song_section.section_name}': {s} to {e}")

            ext = MultimodalExtractModel(
                start_time=s,
                end_time=e,
                duration=get_duration(s, e),
                section_name=song_section.section_name,
                sequence=song_section.sequence,
                audio_segments=[],
                midi_segments=[],
                lyric_segment=[]
            )
            extract = MultimodalExtract(extract_data=ext)
            self.extract_lyrics(extract)
            self.extracts.append(extract)
        print(f"Extracted {len(self.extracts)} sections from the song '{self.meta_data.data.title}'")
        for extract in self.extracts:
            self.extract_lyrics(extract)
            self.extract_audio(extract)
            self.extract_midi()

    def extract_lyrics(self, an_extract: MultimodalExtract) -> None:
        lrc = os.path.join(self.meta_data.base_path, self.meta_data.track_materials_path, self.meta_data.data.lrc_file)
        try:
            lyrics_in_section = []
            add_to_next_segment_index = 0
            add_to_next_segment = False
            with open(lrc, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line.startswith('[') and ']' in line:
                        if re.match(JUST_NUMBERS_PATTERN, line[1]):
                            lyric_time_stamp = lrc_to_seconds(line)
                            if an_extract.extract_data.end_time >= lyric_time_stamp >= an_extract.extract_data.start_time:
                                add_to_next_segment = True
                                add_to_next_segment_index = len(lyrics_in_section)
                    else:
                        if add_to_next_segment:
                            if add_to_next_segment_index < len(lyrics_in_section):
                                lyrics_in_section[add_to_next_segment_index]['content'] = line
                            else:
                                lyrics_in_section.append({'time': lyric_time_stamp, 'content': line})
        except Exception as e:
            print(f"✗ Failed to extract lyrics: {e}")

    def extract_audio(self, an_extract: MultimodalExtract) -> None:
        """
        Extract audio segments from the song.
        This is a placeholder for actual audio extraction logic.
        """
        for audio_track in self.meta_data.data.audio_tracks or []:
            if audio_track.audio_file:
                audio_file_path = os.path.join(self.meta_data.base_path, self.meta_data.track_materials_path,
                                               audio_track.audio_file)
                try:
                    audio_segment = AudioSegment.from_file(audio_file_path)
                    start_time = an_extract.extract_data.start_time
                    end_time = an_extract.extract_data.end_time
                    segment = audio_segment[
                              start_time.total_seconds() * 1000:end_time.total_seconds() * 1000]  # Convert to milliseconds
                    try:
                        segment = segment.set_frame_rate(44100)  # Ensure the audio segment is at 44100 Hz
                        if has_significant_audio(segment):
                            print(
                                f"Extracted audio segment from '{audio_track.description}' for song '{self.meta_data.data.title}'")
                            file_name = f"{safe_filename(self.meta_data.data.title)}_{safe_filename(an_extract.extract_data.section_name)}_{audio_track.id}.wav"
                            segment.export(os.path.join(AUDIO_WORKING_DIR, file_name), format="wav")

                            #ToDO: add to extract data


                    except Exception as e:
                        print(f"✗ Failed to set frame rate for track '{audio_track.description}': {e}")
                        continue
                    an_extract.extract_data.audio_segments.append(segment)
                except Exception as e:
                    print(f"✗ Failed to extract audio from track '{audio_track.description}': {e}")

    def extract_midi(self) -> None:
        """
        Extract MIDI segments from the song.
        This is a placeholder for actual MIDI extraction logic.
        """
        for audio_track in self.meta_data.data.audio_tracks or []:
            if audio_track.midi_file and audio_track.group is None:
                midi_file_path = os.path.join(self.meta_data.base_path, self.meta_data.track_materials_path,
                                              audio_track.midi_file)
                # print(f"Extracting MIDI from track '{audio_track.description}' {midi_file_path} for song '{self.meta_data.data.title}'")
            # ToDo: Handle grouped MIDI files if necessary

    def create_audio_chunk(self):
        """
        Create audio chunks from the song.
        This is a placeholder for actual audio chunk creation logic.
        """
        # Placeholder for audio chunk creation logic
        print(f"Created audio chunks for song '{self.meta_data.data.title}'")

    def create_midi_chunk(self):
        """
        Create MIDI chunks from the song.
        This is a placeholder for actual MIDI chunk creation logic.
        """
        # Placeholder for MIDI chunk creation logic
        print(f"Created MIDI chunks for song '{self.meta_data.data.title}'")

    def create_temporal_data_frame(self):
        """
        Create a temporal data frame from the song.
        This is a placeholder for actual temporal data frame creation logic.
        """
        # Placeholder for temporal data frame creation logic
        print(f"Created temporal data frame for song '{self.meta_data.data.title}'")

    def create_training_data(self):
        """
        Create training data from the song.
        This is a placeholder for actual training data creation logic.
        """
        # Placeholder for training data creation logic
        print(f"Created training data for song '{self.meta_data.data.title}'")
