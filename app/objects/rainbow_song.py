import os
import re
import mido
from typing import Any
from pydantic import BaseModel
from pydub import AudioSegment

from app.objects.multimodal_extract import MultimodalExtract, MultimodalExtractModel, MultimodalExtractEventModel, \
    ExtractionContentType
from app.objects.rainbow_song_meta import RainbowSongMeta
from app.utils.audio_util import has_significant_audio, get_microseconds_per_beat
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
            ext = MultimodalExtractModel(
                start_time=s,
                end_time=e,
                duration=get_duration(s, e),
                section_name=song_section.section_name,
                sequence=song_section.sequence,
                events=[]
            )
            extract = MultimodalExtract(extract_data=ext)
            self.extracts.append(extract)
        for extract in self.extracts:
            # self.extract_lyrics(extract)
            # self.extract_audio(extract)
            self.extract_midi(extract)

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
            lyric_extract_event = MultimodalExtractEventModel(start_time=an_extract.extract_data.start_time, end_time=an_extract.extract_data.end_time, type=ExtractionContentType.LYRICS, content=lyrics_in_section)
            an_extract.extract_data.events.append(lyric_extract_event)

        except Exception as e:
            print(f"✗ Failed to extract lyrics: {e}")

    def extract_audio(self, an_extract: MultimodalExtract) -> None:
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
                            file_name = f"{safe_filename(self.meta_data.data.title)}_{safe_filename(an_extract.extract_data.section_name)}_{audio_track.id}.wav"
                            segment.export(os.path.join(AUDIO_WORKING_DIR, file_name), format="wav")
                            audio_extract_event = MultimodalExtractEventModel(
                                start_time=an_extract.extract_data.start_time,
                                end_time=an_extract.extract_data.end_time,
                                type=ExtractionContentType.AUDIO,
                                content=file_name
                            )
                            an_extract.extract_data.events.append(audio_extract_event)
                    except Exception as e:
                        print(f"✗ Failed to set frame rate for track '{audio_track.description}': {e}")
                        continue
                except Exception as e:
                    print(f"✗ Failed to extract audio from track '{audio_track.description}': {e}")

    def extract_midi(self, an_extract: MultimodalExtract) -> None:
        """
        Extract MIDI segments from the song.
        This is a placeholder for actual MIDI extraction logic.
        """
        for audio_track in self.meta_data.data.audio_tracks or []:
            if audio_track.midi_file and audio_track.midi_group_file is None:
                midi_file_path = os.path.join(self.meta_data.base_path, self.meta_data.track_materials_path,
                                              audio_track.midi_file)
                try:
                    midi = mido.MidiFile(midi_file_path)
                    ticks_per_beat = midi.ticks_per_beat
                    tempo = midi.tempo if hasattr(midi, 'tempo') else get_microseconds_per_beat(self.meta_data.data.bpm)
                    note_collector = []
                    for t in midi.tracks:
                        mt = 0
                        for msg in t:
                            mt+=  msg.time
                            if msg.type == 'set_tempo':
                                tempo = msg.tempo
                            if msg.type == 'note_on' and msg.velocity > 0:
                                midi_note_start_time = mido.tick2second(mt, ticks_per_beat, tempo)
                                midi_note_end_time = mido.tick2second(mt + msg.time, ticks_per_beat, tempo)
                                start_sec = an_extract.extract_data.start_time.total_seconds()
                                end_sec = an_extract.extract_data.end_time.total_seconds()
                                if start_sec <= midi_note_start_time <= end_sec:
                                    midi_note = {
                                        'note': msg.note,
                                        'velocity': msg.velocity,
                                        'start_time': midi_note_start_time,
                                        'end_time': midi_note_end_time
                                    }
                                    note_collector.append(midi_note)
                    if len(note_collector) > 0:
                        midi_extract_event = MultimodalExtractEventModel(
                        start_time=an_extract.extract_data.start_time,
                        end_time=an_extract.extract_data.end_time,
                        type=ExtractionContentType.MIDI,
                        content=note_collector)
                        an_extract.extract_data.events.append(midi_extract_event)

                except Exception as e:
                   print(f"✗ Failed to load MIDI file '{audio_track.midi_file}': {e}")
            # ToDo: Handle grouped MIDI files if necessary


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
