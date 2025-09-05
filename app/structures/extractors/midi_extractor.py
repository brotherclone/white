
import mido
import numpy as np

from typing import List, Dict, Any
from app.structures.extractors.base_manifest_extractor import BaseManifestExtractor


class MidiExtractor(BaseManifestExtractor):

    def __init__(self, **data):
        super().__init__(**data)

    def load_midi_segment(self, midi_path: str, start_time: float, end_time: float) -> Dict[str, Any] | None:
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


    @staticmethod
    def _empty_midi_features() -> Dict[str, Any]:
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

    # def load_midi_events(self, segment_row: pd.Series) -> List[Dict]:
    #     """Load raw MIDI events for a specific segment"""
    #     midi_features = segment_row['midi_features']
    #     midi_path = midi_features['midi_file_path']
    #     start_time = midi_features['segment_start_time']
    #     end_time = midi_features['segment_end_time']
    #
    #     if not midi_path or not Path(midi_path).exists():
    #         return []
    #
    #     # Re-extract MIDI events for this segment
    #     generator = MultimodalGenerator()
    #     midi_data = generator.load_midi_segment(midi_path, start_time, end_time)
    #     return midi_data.get('raw_events', [])

if __name__ == "__main__":
    midi_extractor = MidiExtractor(manifest_path="/Volumes/LucidNonsense/White/staged_raw_material/01_01/01_01.yml")
    print(midi_extractor.manifest)