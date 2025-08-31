import pandas as pd
import yaml
import re
import numpy as np
import librosa
import mido
from typing import List, Dict, Any, Tuple, Optional
from datetime import timedelta
from pathlib import Path


class MultimodalRebracketerTrainingDataGenerator:
    """
    Enhanced version that captures audio, MIDI, and lyrical data across temporal boundaries
    Perfect for White album boundary fluidity analysis!
    """

    def __init__(self, sample_rate: int = 22050):
        self.segment_types = ['section', 'bar', 'phrase', 'sliding_window']
        self.sample_rate = sample_rate

    def parse_lrc_time(self, time_str: str) -> float:
        """Parse LRC timestamp like [00:28.085] to seconds"""
        try:
            match = re.match(r'\[(\d{2}):(\d{2})\.(\d{3})\]', time_str)
            if match:
                minutes, seconds, milliseconds = map(int, match.groups())
                return minutes * 60 + seconds + milliseconds / 1000
            else:
                print(f"Warning: Timestamp format not recognized: {time_str}")
                return None
        except Exception as e:
            print(f"Error parsing timestamp {time_str}: {e}")
            return None

    def parse_yaml_time(self, time_str: str) -> float:
        """Parse YAML timestamp like '[00:28.086]' to seconds"""
        return self.parse_lrc_time(time_str)

    def load_manifest(self, yaml_path: str) -> Dict[str, Any]:
        """Load the YAML manifest file"""
        with open(yaml_path, 'r') as f:
            return yaml.safe_load(f)

    def load_audio_segment(self, audio_path: str, start_time: float, end_time: float,
                           silence_threshold_db: float = -40.0, min_non_silence_ratio: float = 0.1) -> Dict[str, Any]:
        """Extract audio features for a temporal segment with silence detection"""
        try:
            # Load the full audio file
            y, sr = librosa.load(audio_path, sr=self.sample_rate)

            # Convert time to sample indices
            start_sample = int(start_time * sr)
            end_sample = int(end_time * sr)

            # Extract the segment
            segment = y[start_sample:end_sample]

            if len(segment) == 0:
                return self._empty_audio_features()

            # Analyze silence content
            silence_analysis = self._analyze_silence(segment, sr, silence_threshold_db)

            # If segment is mostly silence, mark it appropriately
            if silence_analysis['non_silence_ratio'] < min_non_silence_ratio:
                features = self._create_silence_features(audio_path, start_time, end_time, segment, silence_analysis)
            else:
                # Process as normal audio segment
                features = self._create_audio_features(audio_path, start_time, end_time, segment, sr, silence_analysis)

            return features

        except Exception as e:
            print(f"Error loading audio segment {start_time:.3f}s-{end_time:.3f}s: {e}")
            return self._empty_audio_features()

    def _analyze_silence(self, segment: np.ndarray, sr: int, threshold_db: float) -> Dict[str, Any]:
        """Analyze silence patterns in an audio segment"""
        if len(segment) == 0:
            return {
                'is_mostly_silence': True,
                'non_silence_ratio': 0.0,
                'silence_gaps': [],
                'non_silence_regions': [],
                'peak_amplitude': 0.0,
                'rms_energy': 0.0
            }

        # Convert to dB
        rms = librosa.feature.rms(y=segment, hop_length=sr // 20)[0]  # 50ms frames
        rms_db = librosa.amplitude_to_db(rms, ref=np.max)

        # Find non-silent frames
        non_silent_frames = rms_db > threshold_db
        non_silence_ratio = np.sum(non_silent_frames) / len(non_silent_frames)

        # Find silence gaps and non-silence regions
        silence_gaps = []
        non_silence_regions = []

        # Convert frame indices back to time
        hop_length = sr // 20
        frame_times = librosa.frames_to_time(np.arange(len(non_silent_frames)), sr=sr, hop_length=hop_length)

        # Find transitions
        transitions = np.diff(non_silent_frames.astype(int))
        silence_starts = np.where(transitions == -1)[0] + 1
        silence_ends = np.where(transitions == 1)[0] + 1
        non_silence_starts = np.where(transitions == 1)[0] + 1
        non_silence_ends = np.where(transitions == -1)[0] + 1

        # Handle edge cases
        if non_silent_frames[0]:
            non_silence_starts = np.concatenate([[0], non_silence_starts])
        if non_silent_frames[-1]:
            non_silence_ends = np.concatenate([non_silence_ends, [len(non_silent_frames)]])

        if not non_silent_frames[0]:
            silence_starts = np.concatenate([[0], silence_starts])
        if not non_silent_frames[-1]:
            silence_ends = np.concatenate([silence_ends, [len(non_silent_frames)]])

        # Convert to time ranges
        for start, end in zip(silence_starts, silence_ends):
            if start < len(frame_times) and end <= len(frame_times):
                silence_gaps.append((frame_times[start], frame_times[min(end - 1, len(frame_times) - 1)]))

        for start, end in zip(non_silence_starts, non_silence_ends):
            if start < len(frame_times) and end <= len(frame_times):
                non_silence_regions.append((frame_times[start], frame_times[min(end - 1, len(frame_times) - 1)]))

        return {
            'is_mostly_silence': non_silence_ratio < 0.1,
            'non_silence_ratio': float(non_silence_ratio),
            'silence_gaps': silence_gaps,
            'non_silence_regions': non_silence_regions,
            'peak_amplitude': float(np.max(np.abs(segment))),
            'rms_energy': float(np.sqrt(np.mean(segment ** 2)))
        }

    def _create_silence_features(self, audio_path: str, start_time: float, end_time: float,
                                 segment: np.ndarray, silence_analysis: Dict) -> Dict[str, Any]:
        """Create features for a mostly-silent segment"""
        return {
            'audio_file_path': audio_path,
            'segment_start_time': start_time,
            'segment_end_time': end_time,
            'duration_samples': len(segment),
            'is_silence': True,
            'silence_analysis': silence_analysis,

            # Minimal features for silence
            'rms_energy': silence_analysis['rms_energy'],
            'spectral_centroid': 0.0,
            'zero_crossing_rate': 0.0,
            'tempo': 0.0,

            # Empty arrays for spectral features
            'mfcc': np.array([]),
            'chroma': np.array([]),
            'spectral_contrast': np.array([]),
            'onset_frames': np.array([]),
            'onset_strength': np.array([]),

            'harmonic_ratio': 0.0,
            'attack_time': 0.0,
            'decay_profile': np.array([]),

            # Silence-specific metrics
            'silence_confidence': 1.0 - silence_analysis['non_silence_ratio']
        }

    def _create_audio_features(self, audio_path: str, start_time: float, end_time: float,
                               segment: np.ndarray, sr: int, silence_analysis: Dict) -> Dict[str, Any]:
        """Create full audio features for non-silent segments"""
        features = {
            # Store path instead of raw data to keep parquet manageable
            'audio_file_path': audio_path,
            'segment_start_time': start_time,
            'segment_end_time': end_time,
            'duration_samples': len(segment),
            'is_silence': False,
            'silence_analysis': silence_analysis,
            'rms_energy': float(librosa.feature.rms(y=segment).mean()),
            'spectral_centroid': float(librosa.feature.spectral_centroid(y=segment, sr=sr).mean()),
            'zero_crossing_rate': float(librosa.feature.zero_crossing_rate(segment).mean()),
            'tempo': float(librosa.tempo(y=segment, sr=sr)[0]) if len(segment) > sr else 0,

            # Spectral features for boundary analysis
            'mfcc': librosa.feature.mfcc(y=segment, sr=sr, n_mfcc=13).T,  # Mel-frequency cepstral coefficients
            'chroma': librosa.feature.chroma_stft(y=segment, sr=sr).T,  # Pitch class profiles
            'spectral_contrast': librosa.feature.spectral_contrast(y=segment, sr=sr).T,

            # Onset detection for rhythm analysis
            'onset_frames': librosa.onset.onset_detect(y=segment, sr=sr),
            'onset_strength': librosa.onset.onset_strength(y=segment, sr=sr),

            # Harmonic/percussive separation
            'harmonic_ratio': self._calculate_harmonic_ratio(segment, sr),

            # Boundary transition indicators
            'attack_time': self._calculate_attack_time(segment, sr),
            'decay_profile': self._calculate_decay_profile(segment, sr)
        }

        return features

    # except Exception as e:
    # print(f"Error loading audio segment {start_time:.3f}s-{end_time:.3f}s: {e}")
    # return self._empty_audio_features()


def _empty_audio_features(self) -> Dict[str, Any]:
    """Return empty audio features structure"""
    return {
        'audio_file_path': None,
        'segment_start_time': 0.0,
        'segment_end_time': 0.0,
        'duration_samples': 0,
        'rms_energy': 0.0,
        'spectral_centroid': 0.0,
        'zero_crossing_rate': 0.0,
        'tempo': 0.0,
        'mfcc': np.array([]),
        'chroma': np.array([]),
        'spectral_contrast': np.array([]),
        'onset_frames': np.array([]),
        'onset_strength': np.array([]),
        'harmonic_ratio': 0.0,
        'attack_time': 0.0,
        'decay_profile': np.array([])
    }


def _calculate_harmonic_ratio(self, segment: np.ndarray, sr: int) -> float:
    """Calculate ratio of harmonic to percussive content"""
    try:
        harmonic, percussive = librosa.effects.hpss(segment)
        harmonic_energy = np.sum(harmonic ** 2)
        percussive_energy = np.sum(percussive ** 2)
        total_energy = harmonic_energy + percussive_energy
        return float(harmonic_energy / total_energy) if total_energy > 0 else 0.0
    except:
        return 0.0


def _calculate_attack_time(self, segment: np.ndarray, sr: int) -> float:
    """Calculate attack time - how quickly the segment starts"""
    if len(segment) < sr // 10:  # Less than 0.1 seconds
        return 0.0

    # Find the time to reach 90% of peak amplitude
    peak_amp = np.max(np.abs(segment))
    if peak_amp == 0:
        return 0.0

    threshold = 0.9 * peak_amp
    attack_samples = np.where(np.abs(segment) >= threshold)[0]

    if len(attack_samples) > 0:
        return float(attack_samples[0] / sr)
    return 0.0


def _calculate_decay_profile(self, segment: np.ndarray, sr: int) -> np.ndarray:
    """Calculate how the segment decays over time"""
    if len(segment) < sr // 10:
        return np.array([])

    # Calculate RMS in overlapping windows
    hop_length = sr // 20  # 50ms windows
    frame_length = sr // 10  # 100ms frames

    try:
        rms_frames = librosa.feature.rms(y=segment,
                                         frame_length=frame_length,
                                         hop_length=hop_length)[0]
        return rms_frames
    except:
        return np.array([])


def load_midi_segment(self, midi_path: str, start_time: float, end_time: float) -> Dict[str, Any]:
    """Extract MIDI events and features for a temporal segment"""
    try:
        mid = mido.MidiFile(midi_path)

        # Convert MIDI time to absolute time
        absolute_time = 0
        tempo = 500000  # Default microseconds per beat (120 BPM)
        ticks_per_beat = mid.ticks_per_beat

        segment_events = []
        active_notes = {}  # Track note_on events without corresponding note_off

        for track in mid.tracks:
            absolute_time = 0

            for msg in track:
                absolute_time += mido.tick2second(msg.time, ticks_per_beat, tempo)

                if msg.type == 'set_tempo':
                    tempo = msg.tempo
                    continue

                # Check if this message falls within our segment
                if start_time <= absolute_time <= end_time:
                    if msg.type in ['note_on', 'note_off', 'control_change', 'program_change']:
                        event_data = {
                            'time': absolute_time,
                            'relative_time': absolute_time - start_time,
                            'type': msg.type,
                            'channel': getattr(msg, 'channel', 0),
                        }

                        if hasattr(msg, 'note'):
                            event_data['note'] = msg.note
                            event_data['velocity'] = getattr(msg, 'velocity', 0)

                        if hasattr(msg, 'control'):
                            event_data['control'] = msg.control
                            event_data['value'] = msg.value

                        if hasattr(msg, 'program'):
                            event_data['program'] = msg.program

                        segment_events.append(event_data)

        # Analyze MIDI features
        features = self._analyze_midi_features(segment_events, start_time, end_time)
        features['midi_file_path'] = midi_path
        features['segment_start_time'] = start_time
        features['segment_end_time'] = end_time
        # Store event summary instead of all raw events
        features['event_summary'] = {
            'total_events': len(segment_events),
            'event_types': list(set(e['type'] for e in segment_events)),
            'channels_used': list(set(e['channel'] for e in segment_events if 'channel' in e))
        }

        return features

    except Exception as e:
        print(f"Error loading MIDI segment {start_time:.3f}s-{end_time:.3f}s: {e}")
        return self._empty_midi_features()


def _analyze_midi_features(self, events: List[Dict], start_time: float, end_time: float) -> Dict[str, Any]:
    """Analyze MIDI events to extract musical features"""
    if not events:
        return self._empty_midi_features()

    note_events = [e for e in events if e['type'] in ['note_on', 'note_off']]
    note_on_events = [e for e in note_events if e['type'] == 'note_on' and e.get('velocity', 0) > 0]

    features = {
        'total_events': len(events),
        'note_events': len(note_events),
        'note_density': len(note_on_events) / (end_time - start_time) if end_time > start_time else 0,

        # Pitch analysis
        'pitch_range': 0,
        'avg_pitch': 0,
        'pitch_variety': 0,

        # Rhythm analysis
        'velocity_range': 0,
        'avg_velocity': 0,
        'velocity_variance': 0,

        # Temporal analysis
        'inter_onset_intervals': [],
        'rhythmic_regularity': 0,

        # Polyphony
        'max_simultaneous_notes': 0,
        'avg_polyphony': 0,
    }

    if note_on_events:
        pitches = [e['note'] for e in note_on_events]
        velocities = [e['velocity'] for e in note_on_events]

        features['pitch_range'] = max(pitches) - min(pitches)
        features['avg_pitch'] = np.mean(pitches)
        features['pitch_variety'] = len(set(pitches))

        features['velocity_range'] = max(velocities) - min(velocities)
        features['avg_velocity'] = np.mean(velocities)
        features['velocity_variance'] = np.var(velocities)

        # Calculate inter-onset intervals
        onset_times = sorted([e['relative_time'] for e in note_on_events])
        if len(onset_times) > 1:
            intervals = np.diff(onset_times)
            features['inter_onset_intervals'] = intervals.tolist()
            features['rhythmic_regularity'] = 1 / (1 + np.var(intervals)) if len(intervals) > 0 else 0

    return features


def _empty_midi_features(self) -> Dict[str, Any]:
    """Return empty MIDI features structure"""
    return {
        'midi_file_path': None,
        'segment_start_time': 0.0,
        'segment_end_time': 0.0,
        'event_summary': {'total_events': 0, 'event_types': [], 'channels_used': []},
        'total_events': 0,
        'note_events': 0,
        'note_density': 0,
        'pitch_range': 0,
        'avg_pitch': 0,
        'pitch_variety': 0,
        'velocity_range': 0,
        'avg_velocity': 0,
        'velocity_variance': 0,
        'inter_onset_intervals': [],
        'rhythmic_regularity': 0,
        'max_simultaneous_notes': 0,
        'avg_polyphony': 0,
    }


def determine_temporal_relationship(self, content_start: float, content_end: float,
                                    segment_start: float, segment_end: float) -> str:
    """Determine how content relates to segment boundaries"""
    if content_start < segment_start and content_end > segment_end:
        return 'spans_across'
    elif content_start < segment_start:
        return 'bleeds_in'
    elif content_end > segment_end:
        return 'bleeds_out'
    else:
        return 'contained'


def generate_multimodal_segments(self, yaml_path: str, lrc_path: str = None,
                                 audio_path: str = None, midi_path: str = None) -> pd.DataFrame:
    """Generate complete multimodal training dataset"""
    print(f"=== MULTIMODAL REBRACKETING TRAINING DATA GENERATION ===")
    print(f"YAML: {yaml_path}")
    print(f"LRC:  {lrc_path}")
    print(f"Audio: {audio_path}")
    print(f"MIDI: {midi_path}")

    manifest = self.load_manifest(yaml_path)
    print(f"✅ Loaded manifest: {manifest.get('title', 'Unknown')}")

    # Load lyrics if provided
    lyrics = []
    if lrc_path and Path(lrc_path).exists():
        lyrics = self.load_lrc(lrc_path)
        print(f"✅ Loaded {len(lyrics)} lyrics")

    all_segments = []

    # Generate section-based segments with multimodal data
    for section in manifest['structure']:
        section_start = self.parse_yaml_time(section['start_time'])
        section_end = self.parse_yaml_time(section['end_time'])

        # Find intersecting lyrics
        intersecting_lyrics = []
        if lyrics:
            for lyric in lyrics:
                lyric_start = lyric['start_time']
                lyric_end = lyric['end_time']

                if not (lyric_end <= section_start or lyric_start >= section_end):
                    temporal_rel = self.determine_temporal_relationship(
                        lyric_start, lyric_end, section_start, section_end
                    )

                    overlap_start = max(lyric_start, section_start)
                    overlap_end = min(lyric_end, section_end)
                    overlap_duration = max(0, overlap_end - overlap_start)
                    lyric_duration = lyric_end - lyric_start
                    confidence = overlap_duration / lyric_duration if lyric_duration > 0 else 0

                    intersecting_lyrics.append({
                        'text': lyric['text'],
                        'start_time': lyric_start,
                        'end_time': lyric_end,
                        'temporal_relationship': temporal_rel,
                        'confidence': confidence,
                        'overlap_seconds': overlap_duration
                    })

        segment = {
            'manifest_id': manifest['manifest_id'],
            'segment_type': 'section',
            'segment_id': f"{manifest['manifest_id']}_{section['section_name'].lower().replace(' ', '_')}",
            'canonical_start': section_start,
            'canonical_end': section_end,
            'duration': section_end - section_start,
            'section_name': section['section_name'],
            'section_description': section['description'],
            'lyrical_content': intersecting_lyrics,

            # Musical context
            'bpm': manifest.get('bpm'),
            'time_signature': manifest.get('tempo'),
            'key': manifest.get('key'),
            'rainbow_color': manifest.get('rainbow_color'),

            # Metadata
            'title': manifest.get('title'),
            'mood_tags': manifest.get('mood', []),
            'concept': manifest.get('concept', ''),
        }

        # Add audio features if audio file provided
        if audio_path and Path(audio_path).exists():
            segment['audio_features'] = self.load_audio_segment(audio_path, section_start, section_end)
            print(f"✅ Added audio features for {section['section_name']}")

        # Add MIDI features if MIDI file provided
        if midi_path and Path(midi_path).exists():
            segment['midi_features'] = self.load_midi_segment(midi_path, section_start, section_end)
            print(f"✅ Added MIDI features for {section['section_name']}")

        # Calculate boundary fluidity metrics across all modalities
        segment['rebracketing_score'] = self._calculate_boundary_fluidity_score(segment)

        all_segments.append(segment)

    df = pd.DataFrame(all_segments)

    # Add computed multimodal metrics
    df['lyric_count'] = df['lyrical_content'].apply(len)
    df['has_temporal_bleeding'] = df['lyrical_content'].apply(
        lambda x: any(l['temporal_relationship'] not in ['contained', 'exact_match'] for l in x)
    )

    if audio_path:
        df['audio_energy'] = df['audio_features'].apply(lambda x: x.get('rms_energy', 0))
        df['spectral_complexity'] = df['audio_features'].apply(lambda x: x.get('spectral_centroid', 0))

    if midi_path:
        df['midi_density'] = df['midi_features'].apply(lambda x: x.get('note_density', 0))
        df['pitch_complexity'] = df['midi_features'].apply(lambda x: x.get('pitch_variety', 0))

    print(f"\n=== MULTIMODAL ANALYSIS ===")
    print(f"Total segments: {len(df)}")
    print(f"Segments with lyrics: {(df['lyric_count'] > 0).sum()}")
    print(f"Segments with temporal bleeding: {df['has_temporal_bleeding'].sum()}")

    if audio_path:
        print(f"Average audio energy: {df['audio_energy'].mean():.4f}")
    if midi_path:
        print(f"Average MIDI note density: {df['midi_density'].mean():.2f} notes/second")

    return df


def _calculate_boundary_fluidity_score(self, segment: Dict[str, Any]) -> float:
    """Calculate how much this segment exhibits boundary fluidity across modalities"""
    score = 0.0

    # Lyrical boundary fluidity
    lyric_bleeding = len([l for l in segment['lyrical_content']
                          if l['temporal_relationship'] not in ['contained', 'exact_match']])
    score += lyric_bleeding * 0.3

    # Audio boundary fluidity (based on attack/decay profiles)
    if 'audio_features' in segment:
        audio = segment['audio_features']
        # Fast attack indicates abrupt boundary
        if audio.get('attack_time', 0) < 0.1:
            score += 0.2
        # Complex decay profile indicates temporal spreading
        if len(audio.get('decay_profile', [])) > 0:
            decay_var = np.var(audio['decay_profile'])
            score += min(decay_var * 0.1, 0.3)

    # MIDI boundary fluidity (based on note overlaps and timing)
    if 'midi_features' in segment:
        midi = segment['midi_features']
        # Irregular timing suggests boundary crossing
        if midi.get('rhythmic_regularity', 1) < 0.5:
            score += 0.2
        # High polyphony suggests complex overlapping
        if midi.get('avg_polyphony', 0) > 2:
            score += 0.2

    return score


def load_lrc(self, lrc_path: str) -> List[Dict[str, Any]]:
    """Parse LRC file into structured lyrical content (from original code)"""
    # [Keeping the original LRC parsing logic unchanged]
    lyrics = []

    try:
        with open(lrc_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        print(f"Loading LRC file: {lrc_path}")
        print(f"Found {len(lines)} lines")

        current_timestamp = None
        current_time = None

        for line_num, line in enumerate(lines):
            line = line.strip()
            if not line:
                continue

            if line.startswith('[') and ']' in line and line.endswith(']'):
                timestamp = line

                if any(timestamp.startswith(f'[{tag}:') for tag in ['ti', 'ar', 'al', 'by', 'offset']):
                    print(f"Skipping metadata: {timestamp}")
                    continue

                try:
                    parsed_time = self.parse_lrc_time(timestamp)
                    if parsed_time is None:
                        print(f"Warning: Could not parse timestamp {timestamp} on line {line_num + 1}")
                        continue

                    current_timestamp = timestamp
                    current_time = parsed_time
                    print(f"Found timestamp: {timestamp} = {parsed_time:.3f}s")

                except Exception as e:
                    print(f"Error parsing timestamp {timestamp}: {e}")
                    continue

            elif line.startswith('[') and ']' in line:
                bracket_end = line.find(']')
                timestamp = line[:bracket_end + 1]
                text = line[bracket_end + 1:].strip()

                if any(timestamp.startswith(f'[{tag}:') for tag in ['ti', 'ar', 'al', 'by', 'offset']):
                    print(f"Skipping metadata: {timestamp}")
                    continue

                try:
                    parsed_time = self.parse_lrc_time(timestamp)
                    if parsed_time is None:
                        print(f"Warning: Could not parse timestamp {timestamp} on line {line_num + 1}")
                        continue
                except Exception as e:
                    print(f"Error parsing timestamp {timestamp}: {e}")
                    continue

                if text:
                    lyrics.append({
                        'text': text,
                        'start_time': parsed_time,
                        'timestamp_raw': timestamp,
                        'line_number': line_num + 1
                    })
                    print(f"Added lyric: '{text}' at {parsed_time:.3f}s")

            else:
                if current_time is not None and line:
                    lyrics.append({
                        'text': line,
                        'start_time': current_time,
                        'timestamp_raw': current_timestamp,
                        'line_number': line_num + 1
                    })
                    print(f"Added lyric: '{line}' at {current_time:.3f}s")

                    current_timestamp = None
                    current_time = None

        print(f"Successfully parsed {len(lyrics)} lyrical entries")

        # Calculate end times
        for i in range(len(lyrics)):
            if i < len(lyrics) - 1:
                lyrics[i]['end_time'] = lyrics[i + 1]['start_time']
            else:
                lyrics[i]['end_time'] = lyrics[i]['start_time'] + 3.0

        return lyrics

    except FileNotFoundError:
        print(f"ERROR: LRC file not found: {lrc_path}")
        return []
    except Exception as e:
        print(f"ERROR loading LRC file {lrc_path}: {e}")
        return []


class MultimodalDataLoader:
    """Helper class to load raw audio/MIDI data from parquet training segments"""

    def __init__(self, sample_rate: int = 22050):
        self.sample_rate = sample_rate

    def load_audio_segment(self, segment_row: pd.Series) -> np.ndarray:
        """Load raw audio for a specific segment from the training data"""
        audio_features = segment_row['audio_features']
        audio_path = audio_features['audio_file_path']
        start_time = audio_features['segment_start_time']
        end_time = audio_features['segment_end_time']

        if not audio_path or not Path(audio_path).exists():
            return np.array([])

        y, sr = librosa.load(audio_path, sr=self.sample_rate)
        start_sample = int(start_time * sr)
        end_sample = int(end_time * sr)

        return y[start_sample:end_sample]

    def load_midi_events(self, segment_row: pd.Series) -> List[Dict]:
        """Load raw MIDI events for a specific segment"""
        midi_features = segment_row['midi_features']
        midi_path = midi_features['midi_file_path']
        start_time = midi_features['segment_start_time']
        end_time = midi_features['segment_end_time']

        if not midi_path or not Path(midi_path).exists():
            return []

        # Re-extract MIDI events for this segment
        generator = MultimodalRebracketerTrainingDataGenerator()
        midi_data = generator.load_midi_segment(midi_path, start_time, end_time)
        return midi_data.get('raw_events', [])


if __name__ == "__main__":
    generator = MultimodalRebracketerTrainingDataGenerator()

    # Generate multimodal training data
    df = generator.generate_multimodal_segments(
        yaml_path='/path/to/manifest.yml',
        lrc_path='/path/to/lyrics.lrc',
        audio_path='/path/to/audio.wav',
        midi_path='/path/to/midi.mid'
    )

    print(f"Generated {len(df)} multimodal training segments")
    print(f"Boundary fluidity scores: {df['rebracketing_score'].describe()}")

    # Save enhanced training data
    df.to_parquet('multimodal_rebracketing_training.parquet', index=False)