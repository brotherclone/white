import datetime
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
from app.objects.rainbow_song_meta import RainbowSongMeta, RainbowSongLyricModel
from app.objects.training_sample import TrainingSample
from app.utils.audio_util import has_significant_audio, get_microseconds_per_beat, audio_to_byes, \
    split_midi_file_by_segment, midi_to_bytes
from app.utils.time_util import lrc_to_seconds, get_duration
from app.utils.string_util import safe_filename,to_str_dict

JUST_NUMBERS_PATTERN = r'^\d+$'
LRC_TIME_STAMP_PATTERN = r'^\[(\d+:\d+(?:\.\d+)?|\d+(?:\.\d+)?)\](.*)'
LRC_META_PATTERN = r'^\[(ti|ar|al):\s*(.*?)\]$'
AUDIO_WORKING_DIR = "/Volumes/LucidNonsense/White/working"
TRAINING_DIR = "/Volumes/LucidNonsense/White/training"
MAXIMUM_EXTRACT_DURATION = datetime.timedelta(seconds=29)


class RainbowSong(BaseModel):
    """
    Represents a song with associated metadata, extracts, and training samples.

    This class handles the segmentation of a song into extracts based on its structure,
    and provides methods to extract lyrics, audio, and MIDI data for each segment.
    It also supports the creation of training samples for machine learning tasks.

    Attributes:
        meta_data (RainbowSongMeta): Metadata for the song.
        extracts (list[MultimodalExtract] | None): List of extracted song segments.
        training_samples (list[TrainingSample] | None): List of generated training samples.
        training_sample_data_frame (Any | None): DataFrame containing training sample data.
    """
    meta_data: RainbowSongMeta
    extracts: list[MultimodalExtract] | None = None
    training_samples: list[TrainingSample] | None = None
    training_sample_data_frame: Any | None = None

    def __init__(self, /, **data: Any):
        super().__init__(**data)
        if self.extracts is None:
            self.extracts = []
        if self.training_samples is None:
            self.training_samples = []
        section_sequence = 0
        for song_section in self.meta_data.data.structure:
            if song_section.duration is not None and song_section.duration > datetime.timedelta(MAXIMUM_EXTRACT_DURATION.total_seconds()):
                section_start_time = lrc_to_seconds(song_section.start_time)
                section_end_time = lrc_to_seconds(song_section.end_time)
                total_seconds = (section_end_time - section_start_time).total_seconds()
                max_seconds = MAXIMUM_EXTRACT_DURATION.total_seconds()
                num_segments = int(total_seconds / max_seconds) + (1 if total_seconds % max_seconds > 0 else 0)
                print(f"Splitting section '{song_section.section_name}' into {num_segments} parts")
                for i in range(num_segments):
                    segment_start_offset = i * max_seconds
                    segment_end_offset = min((i + 1) * max_seconds, total_seconds)
                    segment_start = section_start_time + datetime.timedelta(seconds=segment_start_offset)
                    segment_end = section_start_time + datetime.timedelta(seconds=segment_end_offset)
                    segment_name = f"{song_section.section_name} (part {i + 1}/{num_segments})"
                    segment_description = f"{song_section.section_description} [Part {i + 1}/{num_segments}]"
                    ext = MultimodalExtractModel(
                        start_time=segment_start,
                        end_time=segment_end,
                        duration=get_duration(segment_start, segment_end),
                        section_name=segment_name,
                        section_description=segment_description,
                        sequence=section_sequence,
                        events=[],
                        extract_lrc=None,
                        extract_lyrics=None,
                        midi_group=song_section.midi_group if song_section.midi_group else None
                    )
                    section_sequence += 1
                    extract = MultimodalExtract(extract_data=ext)
                    self.extracts.append(extract)
            else:
                s = lrc_to_seconds(song_section.start_time)
                e = lrc_to_seconds(song_section.end_time)
                ext = MultimodalExtractModel(
                    start_time=s,
                    end_time=e,
                    duration=get_duration(s, e),
                    section_name=song_section.section_name,
                    section_description=song_section.section_description,
                    sequence=section_sequence,
                    events=[],
                    extract_lrc=None,
                    extract_lyrics=None,
                    midi_group=song_section.midi_group if song_section.midi_group else None
                )
                section_sequence += 1
                extract = MultimodalExtract(extract_data=ext)
                self.extracts.append(extract)
        for extract in self.extracts:
            self.extract_lyrics(extract)
            self.extract_audio(extract)
            self.extract_midi(extract)

    def extract_lyrics(self, an_extract: MultimodalExtract) -> None:
        """
        Extract lyrics from the LRC file.
        :param an_extract:
        :return:
        """
        lrc = os.path.join(self.meta_data.base_path, self.meta_data.track_materials_path, self.meta_data.data.lrc_file)
        try:
            lyric_contents: list[RainbowSongLyricModel] = []
            with open(lrc, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    meta_match = re.match(LRC_META_PATTERN, line)
                    if meta_match is None:
                        time_stamp_match = re.match(LRC_TIME_STAMP_PATTERN, line)
                        if time_stamp_match:
                            next_line = next(f, None)
                            date_time_lyric_time_stamp = lrc_to_seconds(line)
                            lyric_content: RainbowSongLyricModel = RainbowSongLyricModel(
                                time_stamp= lrc_to_seconds(line),
                                lrc=None,
                                lyrics=None,
                                is_in_range= (an_extract.extract_data.start_time <= date_time_lyric_time_stamp < an_extract.extract_data.end_time)
                            )
                            if next_line is not None:
                                next_line = next_line.strip()
                                lyric_content.lrc = f"\n{line}\n{next_line}"
                                lyric_content.lyrics = f"{next_line}\n"
                            lyric_contents.append(lyric_content)
                    else:
                        print("Ignoring meta line in lrc:", line)
            segment_lrc_collector = []
            segment_lyric_collector = []
            for lc in lyric_contents:
                if lc.is_in_range:
                    segment_lrc_collector.append(lc.lrc)
                    segment_lyric_collector.append(lc.lyrics)
            an_extract.extract_data.extract_lrc = ''.join(segment_lrc_collector)
            an_extract.extract_data.extract_lyrics = ''.join(segment_lyric_collector)
        except Exception as e:
            print(f"✗ Failed to extract lyrics: {e}")

    def extract_audio(self, an_extract: MultimodalExtract) -> None:
        """
        Extract audio segments from the song.
        :param an_extract:
        :return:
        """
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
                                'file_name': file_name,
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
                    segment_path = split_midi_file_by_segment(tempo,  midi_file_path, an_extract.extract_data.duration.total_seconds())
                    note_collector = []
                    for t in midi.tracks:
                        mt = 0
                        for msg in t:
                            mt += msg.time
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
                            type=ExtractionContentType.MIDI if audio_track.midi_group_file is None else ExtractionContentType.SHARED_MIDI,
                            content={
                                'notes': note_collector,
                                'file_name': segment_path,
                                'bytes': midi_to_bytes(segment_path)
                            })
                        an_extract.extract_data.events.append(midi_extract_event)
                except Exception as e:
                    print(f"✗ Failed to load MIDI file '{audio_track.midi_file}': {e}")

    def create_training_samples(self):
        """
        Create training samples from the extracts.
        :return:
        """
        for extract in self.extracts:
            for event in extract.extract_data.events:
                ts = TrainingSample(
                    song_bpm=str(self.meta_data.data.bpm) if isinstance(self.meta_data.data.bpm, int) else self.meta_data.data.bpm,
                    song_key=self.meta_data.data.key,
                    album_rainbow_color=str(
                        self.meta_data.data.rainbow_color) if self.meta_data.data.rainbow_color else None,
                    artist=self.meta_data.data.artist,
                    album_title=self.meta_data.data.album_title,
                    song_title=self.meta_data.data.title,
                    album_realise_date=self.meta_data.data.release_date,
                    song_on_album_sequence=str(self.meta_data.data.album_sequence) if isinstance(self.meta_data.data.album_sequence, int) else self.meta_data.data.album_sequence,
                    song_trt=str(self.meta_data.data.TRT) if isinstance(self.meta_data.data.TRT, datetime.timedelta) else self.meta_data.data.TRT,
                    song_has_vocals=self.meta_data.data.vocals,
                    song_has_lyrics=self.meta_data.data.lyrics,
                    song_structure=json.dumps([s.model_dump() for s in self.meta_data.data.structure]),
                    song_moods=", ".join([str(m) for m in self.meta_data.data.mood]),
                    song_sounds_like=", ".join([str(l) for l in self.meta_data.data.sounds_like]),
                    song_genres=", ".join([str(g) for g in self.meta_data.data.genres]),
                    song_segment_name=extract.extract_data.section_name,
                    song_segment_start_time=str(extract.extract_data.start_time.total_seconds()),
                    song_segment_end_time=str(extract.extract_data.end_time.total_seconds()),
                    song_segment_duration=str(extract.extract_data.duration.total_seconds()),
                    song_segment_sequence=extract.extract_data.sequence,
                    song_segment_description= extract.extract_data.section_description,
                    song_segment_track_id=None,
                    song_segment_track_name=None,
                    song_segment_track_description=None,
                    song_segment_main_audio_file_name=None,
                    song_segment_track_audio_file_name=None,
                    song_segment_main_audio_binary_data=None,
                    song_segment_track_audio_binary_data=None,
                    song_segment_lyrics_text=extract.extract_data.extract_lyrics,
                    song_segment_lyrics_lrc=extract.extract_data.extract_lrc,
                    song_segment_track_midi_data=None,
                    song_segment_track_midi_file_name=None,
                    song_segment_track_midi_binary_data=None,
                    song_segment_track_midi_is_group=extract.extract_data.midi_group,
                )
                if event.type == ExtractionContentType.MIX_AUDIO:
                    ts.song_segment_main_audio_file_name = event.content.get('file_name')
                    ts.song_segment_main_audio_binary_data = audio_to_byes(event.content.get('file_name'),
                                                                           AUDIO_WORKING_DIR)
                    ts.song_segment_track_description = event.content.get('description')
                    ts.song_segment_track_id = event.content.get('id')
                elif event.type == ExtractionContentType.TRACK_AUDIO:
                    ts.song_segment_track_audio_file_name = event.content.get('file_name')
                    ts.song_segment_track_audio_binary_data = audio_to_byes(event.content.get('file_name'),
                                                                            AUDIO_WORKING_DIR)
                    ts.song_segment_track_description = event.content.get('description')
                    ts.song_segment_track_id = event.content.get('id')
                elif event.type == ExtractionContentType.MIDI or event.type == ExtractionContentType.SHARED_MIDI:
                    ts.song_segment_track_midi_data = json.dumps(event.content['notes'])
                    ts.song_segment_track_midi_file_name = event.content.get('file_name')
                    ts.song_segment_track_midi_binary_data = event.content.get('bytes')
                self.training_samples.append(ts)

        self.training_sample_data_frame = pd.DataFrame([
            to_str_dict(ts.model_dump()) for ts in self.training_samples
        ])
        self.training_sample_data_frame.to_parquet(
            os.path.join(TRAINING_DIR, f"{safe_filename(self.meta_data.data.title)}_training_samples.parquet")
        )
