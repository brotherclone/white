import datetime
import os
import re
import mido
import pandas as pd
import json
import logging
from concurrent.futures import ThreadPoolExecutor
from typing import Any, List, Dict, Optional
from pydantic import BaseModel
from pydub import AudioSegment

from app.objects.multimodal_extract import MultimodalExtract, MultimodalExtractModel, MultimodalExtractEventModel, \
    ExtractionContentType
from app.objects.rainbow_song_meta import RainbowSongMeta, RainbowSongLyricModel
from app.objects.training_sample import TrainingSample
from app.utils.audio_util import has_significant_audio, get_microseconds_per_beat, audio_to_byes, \
    split_midi_file_by_segment, midi_to_bytes
from app.utils.time_util import lrc_to_seconds, get_duration
from app.utils.string_util import safe_filename, to_str_dict
from app.utils.validation import TrainingSampleValidator

# Constants
JUST_NUMBERS_PATTERN = r'^\d+$'
LRC_TIME_STAMP_PATTERN = r'^\[(\d+:\d+(?:\.\d+)?|\d+(?:\.\d+)?)\](.*)'
LRC_META_PATTERN = r'^\[(ti|ar|al):\s*(.*?)\]$'
AUDIO_WORKING_DIR = "/Volumes/LucidNonsense/White/working"
TRAINING_DIR = "/Volumes/LucidNonsense/White/training"
MAXIMUM_EXTRACT_DURATION = datetime.timedelta(seconds=29)

# Configure logging
logger = logging.getLogger(__name__)

class RainbowSong(BaseModel):
    """
    Represents a song with associated metadata, extracts, and training samples.

    This class handles the segmentation of a song into extracts based on its structure,
    and provides methods to extract lyrics, audio, and MIDI data for each segment.
    It also supports the creation of training samples for machine learning tasks.

    Attributes:
        meta_data (RainbowSongMeta): Metadata for the song.
        extracts (list[MultimodalExtract] | None): list of extracted song segments.
        training_samples (list[TrainingSample] | None): list of generated training samples.
        training_sample_data_frame (Any | None): DataFrame containing training sample data.
    """
    meta_data: RainbowSongMeta
    extracts: List[MultimodalExtract] = []
    training_samples: List[TrainingSample] = []
    training_sample_data_frame: Optional[pd.DataFrame] = None

    def __init__(self, /, **data: Any):
        super().__init__(**data)
        # Initialize extracts and segments
        self._segment_song()

    def _segment_song(self) -> None:
        """
        Segments the song into extracts based on its structure.
        Handles breaking longer sections into smaller segments if needed.
        """
        section_sequence = 0
        for song_section in self.meta_data.data.structure:
            if song_section.duration is not None and song_section.duration > MAXIMUM_EXTRACT_DURATION:
                # Split longer sections into smaller segments
                self._segment_long_section(song_section, section_sequence)
                section_sequence += 1
            else:
                # Use section as is
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

        # Process all extracts
        self._process_all_extracts()

    def _segment_long_section(self, song_section, section_sequence) -> None:
        """
        Splits a long section into smaller segments.

        Args:
            song_section: The section to split
            section_sequence: The sequence number of the section
        """
        section_start_time = lrc_to_seconds(song_section.start_time)
        section_end_time = lrc_to_seconds(song_section.end_time)
        total_seconds = (section_end_time - section_start_time).total_seconds()
        max_seconds = MAXIMUM_EXTRACT_DURATION.total_seconds()

        num_segments = int(total_seconds / max_seconds) + (1 if total_seconds % max_seconds > 0 else 0)
        logger.info(f"Splitting section '{song_section.section_name}' into {num_segments} parts")

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
                sequence=section_sequence + i,
                events=[],
                extract_lrc=None,
                extract_lyrics=None,
                midi_group=song_section.midi_group if song_section.midi_group else None
            )
            extract = MultimodalExtract(extract_data=ext)
            self.extracts.append(extract)

    def _process_all_extracts(self) -> None:
        """
        Process all extracts by extracting lyrics, audio, and MIDI data.
        Uses ThreadPoolExecutor to parallelize the processing.
        """
        with ThreadPoolExecutor() as executor:
            # Process each extract concurrently
            executor.map(self._process_extract, self.extracts)

    def _process_extract(self, extract: MultimodalExtract) -> None:
        """
        Process a single extract by extracting lyrics, audio, and MIDI data.

        Args:
            extract: The extract to process
        """
        self.extract_lyrics(extract)
        self.extract_audio(extract)
        self.extract_midi(extract)

    def extract_lyrics(self, an_extract: MultimodalExtract) -> None:
        """
        Extract lyrics from the LRC file for a specific time segment.

        Args:
            an_extract: The extract to process lyrics for
        """
        lrc_path = os.path.join(self.meta_data.base_path, self.meta_data.track_materials_path, self.meta_data.data.lrc_file)
        logger.info(f"Extracting lyrics from: {lrc_path}")

        try:
            lyric_contents: List[RainbowSongLyricModel] = []
            with open(lrc_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()

            i = 0
            while i < len(lines):
                line = lines[i].strip()
                meta_match = re.match(LRC_META_PATTERN, line)

                if meta_match is None:
                    time_stamp_match = re.match(LRC_TIME_STAMP_PATTERN, line)
                    if time_stamp_match:
                        date_time_lyric_time_stamp = lrc_to_seconds(line)
                        next_line = lines[i + 1].strip() if i + 1 < len(lines) else None

                        lyric_content = RainbowSongLyricModel(
                            time_stamp=date_time_lyric_time_stamp,
                            lrc=None,
                            lyrics=None,
                            is_in_range=(
                                an_extract.extract_data.start_time <= date_time_lyric_time_stamp < an_extract.extract_data.end_time
                            )
                        )

                        if next_line is not None:
                            lyric_content.lrc = f"\n{line}\n{next_line}"
                            lyric_content.lyrics = f"{next_line}\n"

                        lyric_contents.append(lyric_content)
                else:
                    logger.debug(f"Ignoring meta line in LRC: {line}")

                i += 1

            # Collect lyrics within the extract's time range
            segment_lrc_collector = []
            segment_lyric_collector = []
            for lc in lyric_contents:
                if lc.is_in_range:
                    segment_lrc_collector.append(lc.lrc)
                    segment_lyric_collector.append(lc.lyrics)

            an_extract.extract_data.extract_lrc = ''.join(segment_lrc_collector)
            an_extract.extract_data.extract_lyrics = ''.join(segment_lyric_collector)

        except Exception as e:
            logger.error(f"Failed to extract lyrics: {e}", exc_info=True)

    def extract_audio(self, an_extract: MultimodalExtract) -> None:
        """
        Extract audio segments from the song.

        Args:
            an_extract: The extract to process audio for

        This method:
        1. Processes the main audio file (stereo mix) if available
        2. Processes individual audio tracks if available
        3. For each track, extracts the segment corresponding to the extract's time range
        4. Saves the extracted audio segments to the working directory
        5. Creates events for each audio segment
        """
        tracks = self._prepare_audio_tracks()

        for track in tracks:
            audio_file_path = os.path.join(
                self.meta_data.base_path,
                self.meta_data.track_materials_path,
                track['audio_file']
            )
            try:
                # Extract the audio segment
                audio_segment = self._extract_audio_segment(audio_file_path, an_extract)
                if audio_segment is None:
                    continue

                # Generate a filename for the segment and export it
                file_name = f"{safe_filename(self.meta_data.data.title)}_{safe_filename(an_extract.extract_data.section_name)}{track['file_suffix']}.wav"
                export_path = os.path.join(AUDIO_WORKING_DIR, file_name)
                audio_segment.export(export_path, format="wav")

                # Create an event for the audio segment
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
                logger.error(f"Failed to extract audio from track '{track['description']}': {e}", exc_info=True)

    def _prepare_audio_tracks(self) -> List[Dict[str, Any]]:
        """
        Prepare the list of audio tracks to process.

        Returns:
            List of track dictionaries containing audio file paths and metadata
        """
        tracks = []

        # Add main audio file if available
        if self.meta_data.data.main_audio_file:
            tracks.append({
                'audio_file': self.meta_data.data.main_audio_file,
                'id': 'mix',
                'description': 'Stereo Mix',
                'event_type': ExtractionContentType.MIX_AUDIO,
                'file_suffix': ''
            })

        # Add individual audio tracks if available
        for audio_track in self.meta_data.data.audio_tracks or []:
            if audio_track.audio_file:
                tracks.append({
                    'audio_file': audio_track.audio_file,
                    'id': audio_track.id,
                    'description': audio_track.description,
                    'event_type': ExtractionContentType.TRACK_AUDIO,
                    'file_suffix': f"_{audio_track.id}"
                })

        return tracks

    def _extract_audio_segment(self, audio_file_path: str, an_extract: MultimodalExtract) -> Optional[AudioSegment]:
        """
        Extract an audio segment from a file based on the extract's time range.

        Args:
            audio_file_path: Path to the audio file
            an_extract: The extract containing the time range to extract

        Returns:
            An AudioSegment object if successful, None otherwise
        """
        try:
            audio_segment = AudioSegment.from_file(audio_file_path)

            # Convert time to milliseconds
            start_ms = an_extract.extract_data.start_time.total_seconds() * 1000
            end_ms = an_extract.extract_data.end_time.total_seconds() * 1000

            # Extract segment
            segment = audio_segment[start_ms:end_ms]

            # Standardize to 44.1kHz
            segment = segment.set_frame_rate(44100)

            # Check if the segment has significant audio
            if not has_significant_audio(segment):
                logger.info(f"Segment from {audio_file_path} contains no significant audio, skipping")
                return None

            return segment

        except Exception as e:
            logger.error(f"Error processing audio file {audio_file_path}: {e}", exc_info=True)
            return None

    def extract_midi(self, an_extract: MultimodalExtract) -> None:
        """
        Extract MIDI segments from the song based on the extract's time range.

        Args:
            an_extract: The extract to process MIDI for

        This method:
        1. Finds all MIDI files associated with audio tracks
        2. For each MIDI file, extracts notes that fall within the extract's time range
        3. Creates events for each extracted MIDI segment
        4. Adds these events to the extract
        """
        for audio_track in self.meta_data.data.audio_tracks or []:
            if not (audio_track.midi_file and audio_track.midi_group is None):
                continue

            midi_file_path = os.path.join(
                self.meta_data.base_path,
                self.meta_data.track_materials_path,
                audio_track.midi_file
            )
            logger.info(f"Extracting MIDI from: {midi_file_path}")

            try:
                # Process the MIDI file
                midi_data = self._extract_midi_segment(midi_file_path, an_extract)

                if not midi_data or not midi_data['notes']:
                    logger.info(f"No relevant MIDI notes found in {midi_file_path} for this time segment")
                    continue

                # Create an event for the MIDI segment
                midi_extract_event = MultimodalExtractEventModel(
                    start_time=an_extract.extract_data.start_time,
                    end_time=an_extract.extract_data.end_time,
                    type=(ExtractionContentType.MIDI
                          if audio_track.midi_group_file is None
                          else ExtractionContentType.SHARED_MIDI),
                    content={
                        'notes': midi_data['notes'],
                        'file_name': midi_data['segment_path'],
                        'bytes': midi_data['bytes']
                    }
                )
                an_extract.extract_data.events.append(midi_extract_event)

            except Exception as e:
                logger.error(f"Failed to load MIDI file '{audio_track.midi_file}': {e}", exc_info=True)

    def _extract_midi_segment(self, midi_file_path: str, an_extract: MultimodalExtract) -> Optional[Dict[str, Any]]:
        """
        Extract MIDI notes from a file that fall within the extract's time range.

        Args:
            midi_file_path: Path to the MIDI file
            an_extract: The extract containing the time range to extract

        Returns:
            Dictionary containing extracted notes, segment path, and binary data, or None if extraction fails
        """
        try:
            midi = mido.MidiFile(midi_file_path)
            ticks_per_beat = midi.ticks_per_beat
            tempo = midi.tempo if hasattr(midi, 'tempo') else get_microseconds_per_beat(self.meta_data.data.bpm)

            # Split the MIDI file into a segment
            segment_path = split_midi_file_by_segment(
                tempo,
                midi_file_path,
                an_extract.extract_data.duration.total_seconds()
            )

            # Extract notes from the MIDI file
            note_collector = []
            start_sec = an_extract.extract_data.start_time.total_seconds()
            end_sec = an_extract.extract_data.end_time.total_seconds()

            # Process each track in the MIDI file
            for track in midi.tracks:
                track_time = 0  # Cumulative time in ticks
                current_tempo = tempo

                for msg in track:
                    # Update track time
                    track_time += msg.time

                    # Update tempo if tempo changes
                    if msg.type == 'set_tempo':
                        current_tempo = msg.tempo

                    # Process note-on events
                    if msg.type == 'note_on' and msg.velocity > 0:
                        midi_note_start_time = mido.tick2second(track_time, ticks_per_beat, current_tempo)
                        midi_note_end_time = mido.tick2second(track_time + msg.time, ticks_per_beat, current_tempo)

                        # Check if the note falls within our time range
                        if start_sec <= midi_note_start_time <= end_sec:
                            midi_note = {
                                'track': track.name,
                                'note': msg.note,
                                'velocity': msg.velocity,
                                'start_time': midi_note_start_time,
                                'end_time': midi_note_end_time
                            }
                            note_collector.append(midi_note)

            # Return collected data
            if note_collector:
                return {
                    'notes': note_collector,
                    'segment_path': segment_path,
                    'bytes': midi_to_bytes(segment_path)
                }
            return None

        except Exception as e:
            logger.error(f"Error processing MIDI file {midi_file_path}: {e}", exc_info=True)
            return None

    def create_training_samples(self):
        """
        Create training samples from the extracts.

        This method:
        1. Iterates through all extracts and their events
        2. Creates a TrainingSample object for each event
        3. Populates the TrainingSample with data based on the event type
        4. Validates the training samples
        5. Saves the training samples to a parquet file if valid

        Returns:
            None, but saves training samples to disk if valid
        """
        for extract in self.extracts:
            for event in extract.extract_data.events:
                # Create a basic training sample with metadata
                ts = self._create_basic_training_sample(extract)

                # Add event-specific data
                self._add_event_data_to_sample(ts, event)

                # Add to training samples collection
                self.training_samples.append(ts)

        # Create a DataFrame from the training samples
        self.training_sample_data_frame = pd.DataFrame([
            to_str_dict(ts.model_dump()) for ts in self.training_samples
        ])

        # Validate the training samples
        validator = TrainingSampleValidator()
        validation_summary = validator.validate_dataframe(self.training_sample_data_frame)
        validator.print_summary()

        # Save the training samples if valid
        output_path = os.path.join(TRAINING_DIR, f"{safe_filename(self.meta_data.data.title)}_training_samples.parquet")
        if validation_summary.samples_with_errors == 0:
            self.training_sample_data_frame.to_parquet(output_path)
            logger.info(f"Training samples saved to {output_path}")
        else:
            logger.warning(f"Training samples have {validation_summary.samples_with_errors} validation errors. Data saved with warning.")
            self.training_sample_data_frame.to_parquet(output_path)

    def _create_basic_training_sample(self, extract: MultimodalExtract) -> TrainingSample:
        """
        Create a basic training sample with metadata.

        Args:
            extract: The extract to create a training sample from

        Returns:
            A TrainingSample object with basic metadata
        """
        return TrainingSample(
            song_bpm=str(self.meta_data.data.bpm) if isinstance(self.meta_data.data.bpm, int) else self.meta_data.data.bpm,
            song_key=self.meta_data.data.key,
            album_rainbow_color=str(self.meta_data.data.rainbow_color) if self.meta_data.data.rainbow_color else None,
            artist=self.meta_data.data.artist,
            album_title=self.meta_data.data.album_title,
            song_title=self.meta_data.data.title,
            album_realise_date=self.meta_data.data.release_date,
            song_on_album_sequence=str(self.meta_data.data.album_sequence) if isinstance(
                self.meta_data.data.album_sequence, int) else self.meta_data.data.album_sequence,
            song_trt=str(self.meta_data.data.TRT) if isinstance(self.meta_data.data.TRT,
                                                                datetime.timedelta) else self.meta_data.data.TRT,
            song_has_vocals=self.meta_data.data.vocals,
            song_has_lyrics=self.meta_data.data.lyrics,
            song_structure=json.dumps([s.model_dump() for s in self.meta_data.data.structure]),
            song_moods=", ".join([str(m) for m in self.meta_data.data.mood]),
            song_sounds_like=json.dumps([s.model_dump() for s in self.meta_data.data.sounds_like]),
            song_genres=", ".join([str(g) for g in self.meta_data.data.genres]),
            song_segment_name=extract.extract_data.section_name,
            song_segment_start_time=str(extract.extract_data.start_time.total_seconds()),
            song_segment_end_time=str(extract.extract_data.end_time.total_seconds()),
            song_segment_duration=str(extract.extract_data.duration.total_seconds()),
            song_segment_sequence=extract.extract_data.sequence,
            song_segment_description=extract.extract_data.section_description,
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
            song_segment_reference_plan_paths=", ".join(
                [str(rp) for rp in self.meta_data.data.reference_plans_paths]),
            song_segment_concept=self.meta_data.data.concept if self.meta_data.data.concept else None
        )

    def _add_event_data_to_sample(self, ts: TrainingSample, event: MultimodalExtractEventModel) -> None:
        """
        Add event-specific data to a training sample.

        Args:
            ts: The training sample to update
            event: The event containing the data to add

        Returns:
            None, updates the training sample in place
        """
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
