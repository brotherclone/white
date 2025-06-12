import os
import re
import mido
import pandas as pd
import json

from typing import Any
from pydantic import BaseModel
from pydub import AudioSegment

from app.objects.multimodal_extract import MultimodalExtract, MultimodalExtractModel, MultimodalExtractEventModel, \
    ExtractionContentType
from app.objects.rainbow_song_meta import RainbowSongMeta
from app.utils.audio_util import has_significant_audio, get_microseconds_per_beat
from app.utils.time_util import lrc_to_seconds, get_duration, convert_timedelta
from app.utils.string_util import safe_filename

JUST_NUMBERS_PATTERN = r'^\d+$'
AUDIO_WORKING_DIR = "/Volumes/LucidNonsense/White/working"
TRAINING_DIR = "/Volumes/LucidNonsense/White/training"

class RainbowSong(BaseModel):
    meta_data: RainbowSongMeta
    extracts: list[MultimodalExtract] | None = None
    event_data_frame: Any | None = None
    def __init__(self, /, **data: Any):
        super().__init__(**data)
        # ToDo: Split longer sections into smaller extracts if needed
        if self.extracts is None:
            self.extracts = []
        section_sequence = 0
        for song_section in self.meta_data.data.structure:
            s = lrc_to_seconds(song_section.start_time)
            e = lrc_to_seconds(song_section.end_time)
            ext = MultimodalExtractModel(
                start_time=s,
                end_time=e,
                duration=get_duration(s, e),
                section_name=song_section.section_name,
                sequence=section_sequence,
                events=[]
            )
            section_sequence+= 1
            # ToDo: Add more metadata to the extract
            extract = MultimodalExtract(extract_data=ext)
            self.extracts.append(extract)
        for extract in self.extracts:
            self.extract_lyrics(extract)
            self.extract_audio(extract)
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
                    match = re.match(r'^\[(\d+:\d+(?:\.\d+)?|\d+(?:\.\d+)?)\](.*)', line)
                    if match:
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

            if len(lyrics_in_section) > 0:
                lyric_extract_event = MultimodalExtractEventModel(start_time=an_extract.extract_data.start_time, end_time=an_extract.extract_data.end_time, type=ExtractionContentType.LYRICS, content=lyrics_in_section)
                an_extract.extract_data.events.append(lyric_extract_event)
                lyrics_in_section.sort(key=lambda x: x['time'])
        except Exception as e:
            print(f"✗ Failed to extract lyrics: {e}")


    def extract_audio(self, an_extract: MultimodalExtract) -> None:
        tracks = []
        if self.meta_data.data.main_audio_file:
            tracks.append({
                'audio_file': self.meta_data.data.main_audio_file,
                'id': 'mix',
                'description': 'Stereo Mix',
                'event_type': ExtractionContentType.MIX_AUDIO,
                'file_suffix': ''
            })
        for audio_track in self.meta_data.data.audio_tracks or []:
            if audio_track.audio_file:
                tracks.append({
                    'audio_file': audio_track.audio_file,
                    'id': audio_track.id,
                    'description': audio_track.description,
                    'event_type': ExtractionContentType.TRACK_AUDIO,
                    'file_suffix': f"_{audio_track.id}"
                })

        for track in tracks:
            audio_file_path = os.path.join(
                self.meta_data.base_path,
                self.meta_data.track_materials_path,
                track['audio_file']
            )
            try:
                audio_segment = AudioSegment.from_file(audio_file_path)
                start_ms = an_extract.extract_data.start_time.total_seconds() * 1000
                end_ms = an_extract.extract_data.end_time.total_seconds() * 1000
                segment = audio_segment[start_ms:end_ms]
                try:
                    segment = segment.set_frame_rate(44100)
                    if has_significant_audio(segment):
                        file_name = f"{safe_filename(self.meta_data.data.title)}_{safe_filename(an_extract.extract_data.section_name)}{track['file_suffix']}.wav"
                        segment.export(os.path.join(AUDIO_WORKING_DIR, file_name), format="wav")
                        audio_event = MultimodalExtractEventModel(
                            start_time=an_extract.extract_data.start_time,
                            end_time=an_extract.extract_data.end_time,
                            type=track['event_type'],
                            content={
                                'file_name':file_name,
                                'description': track['description'],
                                'id': track['id'],
                                'source_audio_file': track['audio_file'],
                            }
                        )
                        an_extract.extract_data.events.append(audio_event)
                except Exception as e:
                    print(f"✗ Failed to set frame rate for track '{track['description']}': {e}")
            except Exception as e:
                print(f"✗ Failed to extract audio from track '{track['description']}': {e}")

    def extract_midi(self, an_extract: MultimodalExtract) -> None:
        """
        Extract MIDI segments from the song.
        This is a placeholder for actual MIDI extraction logic.
        """
        for audio_track in self.meta_data.data.audio_tracks or []:
            if audio_track.midi_file and audio_track.midi_group_file is None:
                midi_file_path = os.path.join(self.meta_data.base_path, self.meta_data.track_materials_path,
                                              audio_track.midi_file)
                print(f"Extracting MIDI from: {midi_file_path}")
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
                                        'track': t.name,
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

    def create_dataframe(self) -> None:
        event_holder = []
        for extract in self.extracts:
            for event in extract.extract_data.events:
                event_dict = event.dict() if hasattr(event, "dict") else event.__dict__
                for key in ["start_time", "end_time"]:
                    if isinstance(event_dict.get(key), pd.Timedelta) or hasattr(event_dict.get(key), "total_seconds"):
                        event_dict[key] = event_dict[key].total_seconds()
                if "type" in event_dict and hasattr(event_dict["type"], "value"):
                    event_dict["type"] = event_dict["type"].value
                if "content" in event_dict:
                    content = convert_timedelta(event_dict["content"])
                    if event_dict["type"] in ["track_audio", "mix_audio"]:
                        file_name = content.get("file_name")
                        audio_path = os.path.join(AUDIO_WORKING_DIR, file_name)
                        try:
                            with open(audio_path, "rb") as f:
                                event_dict["audio_bytes"] = f.read()
                        except Exception as e:
                            event_dict["audio_bytes"] = None
                            print (f"✗ Failed to read audio file '{audio_path}': {e}")
                    event_dict["content"] = json.dumps(content)
                event_holder.append(event_dict)
        self.event_data_frame = pd.DataFrame(sorted(event_holder, key=lambda x: x["start_time"]))
        self.event_data_frame.to_parquet(os.path.join(TRAINING_DIR, f"{safe_filename(self.meta_data.data.title)}.parquet"))